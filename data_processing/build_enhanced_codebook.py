"""
Enhanced Two-Stage CLIP Classification Pipeline for Taskonomy
Stage A: Fast CLIP-B/16 with context padding
Stage B: SAM mask refinement + CLIP-L/14 for fallback cases
"""

import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
from collections import defaultdict
import pickle
import clip
from segment_anything import sam_model_registry, SamPredictor

logging.basicConfig(
    level=logging.INFO,  # Changed back to INFO for cleaner output
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CACHE BUILDING UTILITY
# ============================================================================

def build_instance_cache(processed_dir: Path) -> Dict[int, List[Tuple[Path, int]]]:
    """
    Build cache of instance ID to locations mapping by scanning all JSON files.
    
    Args:
        processed_dir: Directory containing processed Taskonomy JSON files
        
    Returns:
        Dictionary mapping instance_id -> [(json_path, bbox_idx), ...]
    """
    logger.info("Building instance cache by scanning JSON files...")
    
    instance_locations = defaultdict(list)
    json_files = list(processed_dir.glob('**/*.json'))
    
    logger.info(f"Found {len(json_files)} JSON files to scan")
    
    for json_path in tqdm(json_files, desc="Scanning JSON files"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Process each 3D bounding box
            for bbox_idx, bbox_3d in enumerate(data.get('bounding_boxes_3d', [])):
                category = bbox_3d.get('category', '')
                
                # Extract instance ID from category (e.g., "object_18" -> 18)
                if category.startswith('object_'):
                    try:
                        instance_id = int(category.split('_')[1])
                        instance_locations[instance_id].append((json_path, bbox_idx))
                    except (ValueError, IndexError):
                        continue
        
        except Exception as e:
            logger.warning(f"Error processing {json_path}: {e}")
            continue
    
    # Convert defaultdict to regular dict
    instance_locations = dict(instance_locations)
    
    logger.info(f"✅ Found {len(instance_locations)} unique instances across {len(json_files)} files")
    
    # Show some statistics
    location_counts = [len(locs) for locs in instance_locations.values()]
    logger.info(f"   Min locations per instance: {min(location_counts)}")
    logger.info(f"   Max locations per instance: {max(location_counts)}")
    logger.info(f"   Avg locations per instance: {sum(location_counts)/len(location_counts):.1f}")
    
    return instance_locations


# ============================================================================
# CLASS VOCABULARY AND SYNONYM BUCKETS
# ============================================================================

CLASS_VOCAB = [
    # 建筑结构
    "door", "window", "stairs", "column", "beam",
    
    # 家具
    "chair", "sofa", "bench", "stool", "table", "cabinet", "wardrobe",
    "bed", "shelf", "drawer",
    
    # 厨房用品
    "refrigerator", "oven", "stove", "microwave", "dishwasher", "sink",
    
    # 卫浴用品
    "toilet", "bathtub", "shower", "mirror", "towel",
    
    # 电子设备
    "tv", "monitor", "computer", "printer", "speaker", "phone",
    
    # 家电
    "washing machine", "dryer", "air conditioner", "heater", "fan", "vacuum cleaner",
    
    # 照明
    "lamp", "light",
    
    # 装饰品
    "curtain", "rug", "carpet", "picture", "painting", "plant", "vase", "clock",
    
    # 日用品
    "box", "bag", "basket", "trash can", "bottle", "cup", "plate", "book",
    
    # 办公用品
    "whiteboard", "blackboard", "ladder"
]

# Synonym buckets for merging similar concepts
SYNONYM_BUCKETS = {
    "screen_like": ["tv", "monitor", "computer"],
    "floor_covering": ["rug", "carpet"],
    "artwork": ["picture", "painting"],
    "lighting": ["lamp", "light"],
    "writing_surface": ["whiteboard", "blackboard"],
    "seating": ["chair", "bench", "stool"],
    "storage": ["cabinet", "wardrobe", "shelf", "drawer"],
    "washing": ["washing machine", "dryer"],
    "cooling": ["air conditioner", "fan"],
    "heating": ["heater", "oven", "stove"],
}

# Background/null prompts
NULL_PROMPTS = [
    "an empty wall",
    "empty floor with no object", 
    "unrecognizable clutter",
    "background noise",
    "empty space"
]

# Thresholds (tune these on validation data)
TAU_HIGH = 0.015  # Stage A acceptance threshold (adjusted for 70-class softmax)
TAU_MID = 0.010   # Stage B acceptance threshold (adjusted for 70-class softmax)
T_FAST = 0.5     # Temperature for Stage A softmax
T_STRONG = 0.3   # Temperature for Stage B softmax


# ============================================================================
# PROMPT GENERATION
# ============================================================================

def generate_prompts_for_class(class_name: str) -> List[str]:
    """Generate diverse prompts for a class."""
    prompts = [
        f"a photo of a {class_name} in a room",
        f"an indoor scene with a {class_name}",
        f"a household {class_name}",
        f"{class_name} in an interior",
        f"indoor {class_name}"
    ]
    return prompts


# ============================================================================
# CLIP EMBEDDING PRECOMPUTATION
# ============================================================================

class CLIPClassifier:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load CLIP models
        logger.info("Loading CLIP models...")
        self.clip_fast, self.preprocess_fast = clip.load("ViT-B/16", device=device)
        self.clip_strong, self.preprocess_strong = clip.load("ViT-L/14", device=device)
        
        # Load SAM
        logger.info("Loading SAM model...")
        sam_checkpoint = Path(__file__).parent / "models" / "sam_vit_h_4b8939.pth"
        if not sam_checkpoint.exists():
            logger.error(f"SAM model not found at {sam_checkpoint}")
            logger.error("Run setup_enhanced_pipeline.sh first to download the SAM model")
            raise FileNotFoundError("SAM model not found")
        
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # Precompute text embeddings
        self.text_emb_fast = {}
        self.text_emb_strong = {}
        self._precompute_text_embeddings()
        
        # Build synonym mappings
        self.class_to_bucket = {}
        self.bucket_to_classes = {}
        self._build_synonym_mappings()
    
    def _precompute_text_embeddings(self):
        """Precompute and cache text embeddings for all classes."""
        logger.info("Precomputing text embeddings...")
        
        all_classes = CLASS_VOCAB + ["background"]
        
        for class_name in tqdm(all_classes, desc="Text embeddings"):
            if class_name == "background":
                prompts = NULL_PROMPTS
            else:
                prompts = generate_prompts_for_class(class_name)
            
            # Tokenize prompts
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            # Fast CLIP embeddings
            with torch.no_grad():
                fast_embs = self.clip_fast.encode_text(text_tokens)
                fast_embs = F.normalize(fast_embs, dim=-1)
                self.text_emb_fast[class_name] = fast_embs.mean(dim=0)
                
                # Strong CLIP embeddings  
                strong_embs = self.clip_strong.encode_text(text_tokens)
                strong_embs = F.normalize(strong_embs, dim=-1)
                self.text_emb_strong[class_name] = strong_embs.mean(dim=0)
    
    def _build_synonym_mappings(self):
        """Build bidirectional mappings for synonym buckets."""
        # Map each class to its bucket (or itself)
        for class_name in CLASS_VOCAB:
            bucket_name = class_name  # Default: class is its own bucket
            
            for bucket, classes in SYNONYM_BUCKETS.items():
                if class_name in classes:
                    bucket_name = bucket
                    break
            
            self.class_to_bucket[class_name] = bucket_name
            
            if bucket_name not in self.bucket_to_classes:
                self.bucket_to_classes[bucket_name] = []
            if class_name not in self.bucket_to_classes[bucket_name]:
                self.bucket_to_classes[bucket_name].append(class_name)
    
    def crop_with_padding(self, image: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float = 0.15) -> np.ndarray:
        """Crop bbox with context padding."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Calculate padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = int(bbox_w * pad_ratio)
        pad_y = int(bbox_h * pad_ratio)
        
        # Apply padding with bounds checking
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)
        
        return image[y1_pad:y2_pad, x1_pad:x2_pad]
    
    def classify_with_clip(self, patch: np.ndarray, use_strong_model: bool = False) -> Tuple[str, float]:
        """Classify patch with CLIP and return best label + confidence."""
        # Choose model
        if use_strong_model:
            model = self.clip_strong
            preprocess = self.preprocess_strong
            text_embs = self.text_emb_strong
            temperature = T_STRONG
        else:
            model = self.clip_fast
            preprocess = self.preprocess_fast
            text_embs = self.text_emb_fast
            temperature = T_FAST
        
        # Preprocess image
        from PIL import Image
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_pil = Image.fromarray(patch_rgb)  # Convert numpy array to PIL Image
        patch_tensor = preprocess(patch_pil).unsqueeze(0).to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            img_emb = model.encode_image(patch_tensor)
            img_emb = F.normalize(img_emb, dim=-1)
        
        # Compute scores against all classes
        scores = {}
        for class_name in CLASS_VOCAB + ["background"]:
            text_emb = text_embs[class_name]
            score = torch.cosine_similarity(img_emb, text_emb.unsqueeze(0), dim=-1).item()
            scores[class_name] = score
        
        # Apply temperature scaling and softmax
        score_tensor = torch.tensor(list(scores.values())) / temperature
        probs = F.softmax(score_tensor, dim=0)
        
        # Map back to class names
        class_probs = {name: prob.item() for name, prob in zip(scores.keys(), probs)}
        
        # Merge synonym buckets
        bucket_probs = {}
        
        for bucket_name, class_names in self.bucket_to_classes.items():
            # Take max probability within bucket
            bucket_prob = max(class_probs[class_name] for class_name in class_names if class_name in class_probs)
            bucket_probs[bucket_name] = bucket_prob
        
        # Add standalone classes and background
        bucket_probs["background"] = class_probs["background"]
        
        # Find best bucket
        best_bucket = max(bucket_probs.keys(), key=lambda x: bucket_probs[x])
        best_confidence = bucket_probs[best_bucket]
        
        # Debug logging for confidence analysis
        logger.debug(f"Top 3 bucket confidences: {sorted(bucket_probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        # If best is background, return None
        if best_bucket == "background":
            logger.debug(f"Classified as background with confidence {best_confidence:.4f}")
            return None, best_confidence
        
        # Find best individual class within winning bucket
        if best_bucket in self.bucket_to_classes:
            bucket_classes = self.bucket_to_classes[best_bucket]
            best_class = max(bucket_classes, key=lambda x: class_probs[x])
        else:
            best_class = best_bucket
        
        logger.debug(f"Best class: {best_class} with confidence {best_confidence:.4f}")
        return best_class, best_confidence
    
    def get_sam_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Get SAM mask for bbox."""
        self.sam_predictor.set_image(image)
        
        # Convert bbox to SAM format
        input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        return masks[0]  # Return best mask
    
    def apply_mask(self, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to suppress background."""
        # Resize mask to match patch
        if mask.shape[:2] != patch.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (patch.shape[1], patch.shape[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Create masked patch
        masked_patch = patch.copy()
        masked_patch[~mask] = 0  # Zero out background
        
        return masked_patch
    
    def classify_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Two-stage classification pipeline.
        
        Returns:
            {"label": str/None, "confidence": float, "stage": "A"/"B"/"discard"}
        """
        
        # ========== Stage A: Fast CLIP ==========
        patch_raw = self.crop_with_padding(image, bbox, pad_ratio=0.15)
        
        pred_A, conf_A = self.classify_with_clip(patch_raw, use_strong_model=False)
        
        logger.debug(f"Stage A: pred={pred_A}, conf={conf_A:.4f}, threshold={TAU_HIGH}")
        
        if pred_A is not None and conf_A >= TAU_HIGH:
            logger.debug(f"Stage A accepted: {pred_A} with confidence {conf_A:.4f}")
            return {
                "label": pred_A,
                "confidence": conf_A,
                "stage": "A"
            }
        
        # ========== Stage B: SAM + Strong CLIP ==========
        try:
            # Get SAM mask
            mask = self.get_sam_mask(image, bbox)
            
            # Apply mask to suppress background
            patch_masked = self.apply_mask(patch_raw, mask)
            
            # Classify with strong model
            pred_B, conf_B = self.classify_with_clip(patch_masked, use_strong_model=True)
            
            logger.debug(f"Stage B: pred={pred_B}, conf={conf_B:.4f}, threshold={TAU_MID}")
            
            if pred_B is not None and conf_B >= TAU_MID:
                logger.debug(f"Stage B accepted: {pred_B} with confidence {conf_B:.4f}")
                return {
                    "label": pred_B,
                    "confidence": conf_B, 
                    "stage": "B"
                }
        
        except Exception as e:
            logger.debug(f"Stage B failed: {e}")
        
        # ========== Discard ==========
        return {
            "label": None,
            "confidence": 0.0,
            "stage": "discard"
        }


# ============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# ============================================================================

def build_enhanced_codebook(
    representatives: Dict[int, Tuple[Path, int]],
    raw_data_dir: Path,
    device: str = 'cuda'
) -> Dict[int, str]:
    """Build codebook using enhanced two-stage CLIP classification."""
    
    logger.info("Initializing enhanced CLIP classifier...")
    classifier = CLIPClassifier(device=device)
    
    logger.info(f"Processing {len(representatives)} representative instances...")
    
    codebook = {}
    successful = 0
    stage_a_count = 0
    stage_b_count = 0
    discarded = 0
    
    for instance_id, (json_path, bbox_idx) in tqdm(representatives.items(), desc="Enhanced classification"):
        try:
            # Load JSON and get bbox
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Get precise 2D bbox
            if bbox_idx < len(data.get('bounding_boxes_2d', [])):
                bbox_2d_data = data['bounding_boxes_2d'][bbox_idx]
                bbox = (bbox_2d_data['x_min'], bbox_2d_data['y_min'], 
                       bbox_2d_data['x_max'], bbox_2d_data['y_max'])
            else:
                continue
            
            # Load RGB image
            location_name = data['split']
            rgb_filename = data['filename']
            rgb_path = raw_data_dir / 'rgb' / 'taskonomy' / location_name / rgb_filename
            
            if not rgb_path.exists():
                continue
            
            image = cv2.imread(str(rgb_path))
            if image is None:
                continue
            
            # Run two-stage classification
            result = classifier.classify_bbox(image, bbox)
            
            if result['label'] is not None:
                codebook[instance_id] = result['label']
                successful += 1
                
                if result['stage'] == 'A':
                    stage_a_count += 1
                elif result['stage'] == 'B':
                    stage_b_count += 1
                    
                logger.debug(f"Instance {instance_id} → {result['label']} "
                           f"(conf={result['confidence']:.3f}, stage={result['stage']})")
            else:
                discarded += 1
                logger.debug(f"Instance {instance_id} → DISCARDED")
        
        except Exception as e:
            logger.debug(f"Error processing instance {instance_id}: {e}")
            continue
    
    logger.info(f"✅ Enhanced classification complete!")
    logger.info(f"  Successful: {successful}/{len(representatives)}")
    logger.info(f"  Stage A (fast): {stage_a_count}")
    logger.info(f"  Stage B (SAM+strong): {stage_b_count}")  
    logger.info(f"  Discarded: {discarded}")
    
    return codebook


def main():
    """Main enhanced processing."""
    logger.info("="*70)
    logger.info("🚀 Enhanced Two-Stage CLIP Classification Pipeline")
    logger.info("="*70)
    
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'processed_data' / 'taskonomy'
    raw_dir = base_dir / 'raw_data' / 'taskonomy_dataset'
    
    # Build or load instance cache
    cache_path = processed_dir / '.instance_cache.pkl'
    
    if cache_path.exists():
        logger.info(f"Loading instance cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        instance_locations = cached_data['instance_locations']
        logger.info(f"✅ Loaded {len(instance_locations)} instances from cache")
    else:
        logger.info("Cache not found, building from JSON files...")
        instance_locations = build_instance_cache(processed_dir)
        
        # Save cache for future use
        cached_data = {
            'instance_locations': instance_locations,
            'total_files': len(list(processed_dir.glob('**/*.json')))
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        logger.info(f"💾 Saved cache to {cache_path}")
    
    # Select representatives
    representatives = {
        iid: locs[0] 
        for iid, locs in instance_locations.items()
    }
    
    # Build codebook with enhanced pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    codebook = build_enhanced_codebook(representatives, raw_dir, device)
    
    # Save codebook
    codebook_path = processed_dir / 'enhanced_label_codebook.json'
    with open(codebook_path, 'w') as f:
        json.dump(codebook, f, indent=2)
    logger.info(f"💾 Saved enhanced codebook to {codebook_path}")
    
    logger.info(f"\n✅ Enhanced codebook contains {len(codebook)} labeled instances")


if __name__ == '__main__':
    main()