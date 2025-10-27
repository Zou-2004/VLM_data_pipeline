#!/usr/bin/env python3
"""
Enhanced Label Codebook Builder v2 with Hierarchical Classification

This script uses a two-stage hierarchical CLIP + SAM pipeline to generate semantic labels:
- Stage A: CLIP-B/16 quick coarse classification (structure, furniture, appliances, etc.)
- Stage B: SAM + CLIP-L/14 fine-grained classification within coarse category
- Margin-based confidence: Uses gap between top1 and top2 instead of absolute thresholds
- Stage agreement: Both stages must agree on coarse category
- Null/background class: Explicit "not an object" option for each category
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import pickle

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import clip
from tqdm import tqdm
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HIERARCHICAL CATEGORY STRUCTURE
# ============================================================================

# Super-categories (coarse groups)
SUPER_CATEGORIES = {
    "structure": {
        "prompt": "a structural element of a building interior",
        "classes": ["door", "window", "stairs", "column", "beam"]
    },
    "furniture": {
        "prompt": "a piece of household furniture",
        "classes": ["chair", "sofa", "bench", "stool", "table", "cabinet", 
                   "wardrobe", "bed", "shelf", "drawer"]
    },
    "kitchen_bathroom": {
        "prompt": "a bathroom or kitchen fixture",
        "classes": ["refrigerator", "oven", "stove", "microwave", "dishwasher", 
                   "sink", "toilet", "bathtub", "shower", "mirror", "towel"]
    },
    "electronics": {
        "prompt": "a home appliance or electronic device",
        "classes": ["tv", "monitor", "computer", "printer", "speaker", "phone",
                   "washing machine", "dryer", "air conditioner", "heater", 
                   "fan", "vacuum cleaner"]
    },
    "decor": {
        "prompt": "a decorative item in a room",
        "classes": ["lamp", "light", "curtain", "rug", "carpet", "picture", 
                   "painting", "plant", "vase", "clock"]
    },
    "small_items": {
        "prompt": "a small handheld object or container",
        "classes": ["box", "bag", "basket", "trash can", "bottle", "cup", 
                   "plate", "book", "whiteboard", "blackboard", "ladder"]
    }
}

# Null/background prompts (added to each fine-level classification)
NULL_PROMPTS = [
    "an empty wall or background, not an object",
    "unrecognizable clutter or background texture",
    "part of the floor or ceiling, nothing important"
]

# Thresholds
MARGIN_THRESH_COARSE = 0.001   # Margin for super-category selection
MARGIN_THRESH_FINE = 0.0005    # Margin for fine class selection (Stage A)
MARGIN_THRESH_FINE_B = 0.0005  # Margin for fine class selection (Stage B)

# ============================================================================
# ENHANCED CLIP CLASSIFIER v2
# ============================================================================

class EnhancedCLIPClassifierV2:
    """Enhanced CLIP classifier with hierarchical classification and margin-based acceptance."""
    
    def __init__(self, device='cuda'):
        self.device = device
        logger.info("Initializing enhanced CLIP classifier v2...")
        
        # Load CLIP models
        logger.info("Loading CLIP models...")
        self.clip_fast, self.preprocess_fast = clip.load("ViT-B/16", device=device)
        self.clip_strong, self.preprocess_strong = clip.load("ViT-L/14", device=device)
        
        # Load SAM
        logger.info("Loading SAM model...")
        from segment_anything import sam_model_registry, SamPredictor
        sam_checkpoint = Path(__file__).parent / "models" / "sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # Precompute text embeddings
        logger.info("Precomputing text embeddings...")
        self._precompute_embeddings()
        
    def _precompute_embeddings(self):
        """Precompute text embeddings for all categories and classes."""
        
        # Coarse group embeddings (for both models)
        self.group_emb_fast = {}
        self.group_emb_strong = {}
        
        for group_name, group_data in tqdm(SUPER_CATEGORIES.items(), desc="Super-category embeddings"):
            prompt = group_data["prompt"]
            
            # Fast model
            text_tokens = clip.tokenize([f"a photo of {prompt}"]).to(self.device)
            with torch.no_grad():
                emb = self.clip_fast.encode_text(text_tokens)
                self.group_emb_fast[group_name] = F.normalize(emb, dim=-1)[0]
            
            # Strong model
            with torch.no_grad():
                emb = self.clip_strong.encode_text(text_tokens)
                self.group_emb_strong[group_name] = F.normalize(emb, dim=-1)[0]
        
        # Fine class embeddings (per group, for both models)
        self.class_emb_fast = {}
        self.class_emb_strong = {}
        
        for group_name, group_data in tqdm(SUPER_CATEGORIES.items(), desc="Fine class embeddings"):
            classes = group_data["classes"] + ["null"]  # Add null class
            
            self.class_emb_fast[group_name] = {}
            self.class_emb_strong[group_name] = {}
            
            for class_name in classes:
                if class_name == "null":
                    # Use null prompts
                    prompts = NULL_PROMPTS
                else:
                    prompts = [f"a photo of a {class_name}"]
                
                # Fast model
                text_tokens = clip.tokenize(prompts).to(self.device)
                with torch.no_grad():
                    emb = self.clip_fast.encode_text(text_tokens)
                    emb = F.normalize(emb, dim=-1)
                    self.class_emb_fast[group_name][class_name] = emb.mean(dim=0)
                
                # Strong model
                with torch.no_grad():
                    emb = self.clip_strong.encode_text(text_tokens)
                    emb = F.normalize(emb, dim=-1)
                    self.class_emb_strong[group_name][class_name] = emb.mean(dim=0)
    
    def _encode_image(self, patch: np.ndarray, use_strong: bool = False) -> torch.Tensor:
        """Encode image patch to embedding."""
        model = self.clip_strong if use_strong else self.clip_fast
        preprocess = self.preprocess_strong if use_strong else self.preprocess_fast
        
        # Convert to PIL
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_pil = Image.fromarray(patch_rgb)
        patch_tensor = preprocess(patch_pil).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            img_emb = model.encode_image(patch_tensor)
            img_emb = F.normalize(img_emb, dim=-1)
        
        return img_emb[0]
    
    def predict_super_category(self, patch: np.ndarray, use_strong: bool = False) -> Tuple[str, float]:
        """
        Predict super-category (coarse group).
        Returns: (group_name, margin)
        """
        # Get image embedding
        img_emb = self._encode_image(patch, use_strong)
        
        # Get group embeddings
        group_embs = self.group_emb_strong if use_strong else self.group_emb_fast
        
        # Compute similarities
        scores = {}
        for group_name, text_emb in group_embs.items():
            score = torch.cosine_similarity(img_emb.unsqueeze(0), text_emb.unsqueeze(0), dim=-1).item()
            scores[group_name] = score
        
        # Sort by score
        sorted_groups = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-2
        top1_group, top1_score = sorted_groups[0]
        top2_group, top2_score = sorted_groups[1]
        
        margin = top1_score - top2_score
        
        return top1_group, margin
    
    def predict_fine_class(self, patch: np.ndarray, group: str, use_strong: bool = False) -> Tuple[Optional[str], float]:
        """
        Predict fine class within a super-category.
        Returns: (class_name, margin) or (None, 0) if null wins
        """
        # Get image embedding
        img_emb = self._encode_image(patch, use_strong)
        
        # Get class embeddings for this group
        class_embs = self.class_emb_strong[group] if use_strong else self.class_emb_fast[group]
        
        # Compute similarities
        scores = {}
        for class_name, text_emb in class_embs.items():
            score = torch.cosine_similarity(img_emb.unsqueeze(0), text_emb.unsqueeze(0), dim=-1).item()
            scores[class_name] = score
        
        # Sort by score
        sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-2
        top1_class, top1_score = sorted_classes[0]
        top2_class, top2_score = sorted_classes[1]
        
        margin = top1_score - top2_score
        
        # If null wins, return None
        if top1_class == "null":
            return None, margin
        
        return top1_class, margin
    
    def get_sam_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Get SAM mask for bounding box."""
        self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])
        
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        return masks[0]
    
    def classify_bbox_hierarchical(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Classify bbox using hierarchical coarse→fine approach with Stage A/B agreement.
        
        Returns:
            class_name if confident and agreed upon
            None if rejected (low margin, disagreement, or null)
        """
        # Crop with padding
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = int(bbox_w * 0.15)
        pad_y = int(bbox_h * 0.15)
        
        h, w = image.shape[:2]
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)
        
        patch_raw = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # ========== Stage A: Fast CLIP ==========
        # 1. Predict super-category
        group_A, group_margin_A = self.predict_super_category(patch_raw, use_strong=False)
        
        if group_margin_A < MARGIN_THRESH_COARSE:
            return None  # Uncertain about super-category
        
        # 2. Predict fine class within group
        pred_A, margin_A = self.predict_fine_class(patch_raw, group_A, use_strong=False)
        
        if pred_A is None:  # Null class won
            return None
        
        if margin_A < MARGIN_THRESH_FINE:
            # Stage A uncertain, try Stage B
            pass
        else:
            # Stage A confident, but still check Stage B for agreement
            pass
        
        # ========== Stage B: SAM + Strong CLIP ==========
        try:
            # Get SAM mask
            mask = self.get_sam_mask(image, bbox)
            
            # Crop mask to match patch
            mask_crop = mask[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Apply mask (set background to gray)
            patch_masked = patch_raw.copy()
            if mask_crop.shape[:2] == patch_raw.shape[:2]:
                patch_masked[~mask_crop] = 127
            else:
                # Mask shape mismatch, skip masking
                patch_masked = patch_raw
            
        except Exception as e:
            logger.debug(f"SAM failed: {e}, using raw patch")
            patch_masked = patch_raw
        
        # 1. Predict super-category with strong model
        group_B, group_margin_B = self.predict_super_category(patch_masked, use_strong=True)
        
        if group_margin_B < MARGIN_THRESH_COARSE:
            return None  # Uncertain about super-category
        
        # 2. Predict fine class within group
        pred_B, margin_B = self.predict_fine_class(patch_masked, group_B, use_strong=True)
        
        if pred_B is None:  # Null class won
            return None
        
        if margin_B < MARGIN_THRESH_FINE_B:
            return None  # Stage B uncertain
        
        # ========== Final Decision: Require Agreement ==========
        
        # Groups must match
        if group_A != group_B:
            return None  # Disagreement on super-category
        
        # Fine classes must match
        if pred_A != pred_B:
            return None  # Disagreement on fine class
        
        # Both margins must be strong enough
        if margin_A < MARGIN_THRESH_FINE or margin_B < MARGIN_THRESH_FINE_B:
            return None  # Weak margins
        
        # All checks passed!
        return pred_A


def build_enhanced_codebook_v2(representatives: Dict, raw_data_dir: Path, device='cuda') -> Dict[int, str]:
    """
    Build enhanced codebook using hierarchical classification with agreement.
    
    Args:
        representatives: {instance_id: (json_path, bbox_idx)}
        raw_data_dir: Path to raw Taskonomy dataset
        device: 'cuda' or 'cpu'
    
    Returns:
        {instance_id: class_name}
    """
    classifier = EnhancedCLIPClassifierV2(device=device)
    
    codebook = {}
    stats = {"classified": 0, "rejected": 0}
    
    logger.info(f"Processing {len(representatives)} representative instances...")
    
    for instance_id, (json_path, bbox_idx) in tqdm(representatives.items(), desc="Enhanced classification v2"):
        try:
            # Load JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Get 2D bbox
            if bbox_idx >= len(data.get('bounding_boxes_2d', [])):
                stats["rejected"] += 1
                continue
            
            bbox_2d = data['bounding_boxes_2d'][bbox_idx]
            bbox = (bbox_2d['x_min'], bbox_2d['y_min'], bbox_2d['x_max'], bbox_2d['y_max'])
            
            # Load image
            location_name = data['split']
            rgb_filename = data['filename']
            rgb_path = raw_data_dir / 'rgb' / 'taskonomy' / location_name / rgb_filename
            
            if not rgb_path.exists():
                stats["rejected"] += 1
                continue
            
            image = cv2.imread(str(rgb_path))
            if image is None:
                stats["rejected"] += 1
                continue
            
            # Classify with hierarchical approach
            label = classifier.classify_bbox_hierarchical(image, bbox)
            
            if label is not None:
                codebook[instance_id] = label
                stats["classified"] += 1
            else:
                stats["rejected"] += 1
        
        except Exception as e:
            logger.debug(f"Error processing instance {instance_id}: {e}")
            stats["rejected"] += 1
    
    logger.info(f"✅ Enhanced classification v2 complete!")
    logger.info(f"   Classified: {stats['classified']}/{len(representatives)}")
    logger.info(f"   Rejected: {stats['rejected']}")
    
    return codebook


def main():
    """Main function to build enhanced codebook v2."""
    
    logger.info("="*70)
    logger.info("🚀 Enhanced Two-Stage CLIP Classification Pipeline v2")
    logger.info("="*70)
    
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'processed_data' / 'taskonomy'
    raw_data_dir = base_dir / 'raw_data' / 'taskonomy_dataset'
    
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
    
    # Select one representative per instance
    representatives = {iid: locs[0] for iid, locs in instance_locations.items()}
    
    # Build codebook
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    codebook = build_enhanced_codebook_v2(representatives, raw_data_dir, device=device)
    
    # Save codebook
    output_path = processed_dir / 'enhanced_label_codebook_v2.json'
    with open(output_path, 'w') as f:
        json.dump(codebook, f, indent=2)
    
    logger.info(f"💾 Saved enhanced codebook v2 to {output_path}")
    logger.info(f"\n✅ Enhanced codebook v2 contains {len(codebook)} labeled instances")



if __name__ == '__main__':
    main()
