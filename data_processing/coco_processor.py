"""
COCO Dataset Processor
Processes COCO with 2D bounding boxes and optional pseudo-depth.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from PIL import Image
from datetime import datetime
import os
import sys
import cv2
import torch

# Add MoGe to Python path
moge_path = Path(__file__).parent / "MoGe"
if moge_path.exists():
    sys.path.insert(0, str(moge_path))

import utils
from utils import (
    create_unified_json,
    save_json,
    compute_depth_stats,
    convert_bbox_to_9dof
)

# Set cache directories to avoid filling up home directory
cache_base = os.environ.get('PIPELINE_CACHE_DIR', '/tmp')
os.environ['HF_HOME'] = os.path.join(cache_base, 'huggingface')
os.environ['TORCH_HOME'] = os.path.join(cache_base, 'torch')
os.environ['TMPDIR'] = os.path.join(cache_base, 'tmp')

# Create cache directories if they don't exist
for cache_dir in [os.environ['HF_HOME'], os.environ['TORCH_HOME'], os.environ['TMPDIR']]:
    os.makedirs(cache_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COCOProcessor:
    def __init__(self, raw_data_dir: Path, output_dir: Path, depth_model: str = "moge"):
        """
        Initialize COCO processor.
        
        Args:
            raw_data_dir: Path to raw_data/COCO
            output_dir: Path to processed_data/coco
            depth_model: Depth estimation model ("moge" for MoGe-2, None to skip depth)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.depth_model_name = depth_model
        self.depth_model = None
        
        # Find COCO directory (coco-2017)
        self.coco_dir = self.raw_data_dir / "coco-2017"
        if not self.coco_dir.exists():
            self.coco_dir = self.raw_data_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load depth model if specified
        if self.depth_model_name:
            self.load_depth_model()
    
    def load_depth_model(self):
        """Load MoGe-2 depth estimation model"""
        try:
            from moge.model.v2 import MoGeModel
            logger.info("Loading MoGe-2 depth model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
            self.depth_model.eval()
            logger.info(f"MoGe-2 model loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load MoGe-2 model: {e}")
            logger.warning("Depth estimation will be skipped")
            self.depth_model = None
            self.device = None
    
    def estimate_depth(self, image_path: str) -> Optional[np.ndarray]:
        """
        Estimate depth map from RGB image using MoGe-2
        
        Args:
            image_path: Path to RGB image
            
        Returns:
            Depth map as numpy array (H, W) or None if estimation fails
        """
        if self.depth_model is None:
            return None
            
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            
            # Convert to tensor and normalize to [0, 1]
            img_tensor = torch.tensor(img_np, dtype=torch.float32, device=self.device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            
            # Run inference
            with torch.no_grad():
                output = self.depth_model.infer(img_tensor)
            
            # Extract depth map
            depth_map = output["depth"].cpu().numpy()
            
            return depth_map
            
        except Exception as e:
            logger.warning(f"Failed to estimate depth for {image_path}: {e}")
            return None
    
    def convert_2d_to_3d_bbox(self, bbox_2d: Dict, depth_map: np.ndarray, 
                             image_width: int, image_height: int) -> Optional[Dict]:
        """
        Convert 2D bounding box to 3D using depth information.
        
        Args:
            bbox_2d: 2D bounding box dict with x, y, width, height
            depth_map: Depth map (H, W)
            image_width: Image width
            image_height: Image height
            
        Returns:
            3D bounding box dict or None if conversion fails
        """
        try:
            # Get 2D bbox coordinates
            x = bbox_2d["bbox_2d"]["x"]
            y = bbox_2d["bbox_2d"]["y"]
            w = bbox_2d["bbox_2d"]["width"] 
            h = bbox_2d["bbox_2d"]["height"]
            
            # Convert to integer pixel coordinates
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image_width, int(x + w))
            y2 = min(image_height, int(y + h))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract depth values within the bounding box
            depth_roi = depth_map[y1:y2, x1:x2]
            
            # Filter out invalid depth values (typically 0 or very large values)
            valid_depths = depth_roi[(depth_roi > 0.1) & (depth_roi < 100.0)]
            
            if len(valid_depths) < 10:  # Need at least 10 valid depth pixels
                return None
            
            # Use median depth as the object's depth (more robust than mean)
            median_depth = np.median(valid_depths)
            
            # Additional validation - reject if depth seems unreasonable
            if median_depth < 0.5 or median_depth > 50.0:
                return None
            
            # Estimate camera intrinsics (COCO doesn't provide them)
            # Use typical values for consumer cameras with some scaling
            fx = image_width * 0.7  # Rough estimate: focal length ≈ 0.7 * image width
            fy = image_height * 0.7
            cx = image_width / 2.0
            cy = image_height / 2.0
            
            # Convert 2D bbox corners to 3D using camera projection
            # Center of the 2D bbox
            bbox_center_x = x + w / 2.0
            bbox_center_y = y + h / 2.0
            
            # Project to 3D coordinates
            x3d_center = (bbox_center_x - cx) * median_depth / fx
            y3d_center = (bbox_center_y - cy) * median_depth / fy
            z3d_center = median_depth
            
            # Estimate 3D dimensions by projecting bbox corners
            x3d_min = (x - cx) * median_depth / fx
            x3d_max = (x + w - cx) * median_depth / fx
            y3d_min = (y - cy) * median_depth / fy
            y3d_max = (y + h - cy) * median_depth / fy
            
            width_3d = abs(x3d_max - x3d_min)
            height_3d = abs(y3d_max - y3d_min)
            
            # Estimate depth dimension based on object category and size
            # This is a heuristic - use depth variance in the bbox region for better estimation
            depth_variation = np.std(valid_depths)
            depth_3d = max(min(width_3d, height_3d) * 0.8, depth_variation * 2.0)
            
            # Minimum size constraints to avoid tiny objects
            min_size = 0.05  # 5cm minimum
            if width_3d < min_size or height_3d < min_size or depth_3d < min_size:
                return None
            
            # Create 3D bbox dict
            bbox_3d = {
                "category": bbox_2d["category"],
                "category_id": bbox_2d.get("category_id"),
                "center": [float(x3d_center), float(y3d_center), float(z3d_center)],
                "dimensions": [float(width_3d), float(height_3d), float(depth_3d)],
                "rotation": [0.0, 0.0, 0.0],  # No rotation information available
                "confidence": 0.7,  # Moderate confidence for depth-based estimation
                "method": "depth_projection"
            }
            
            # Convert to 9-DoF format
            bbox_9dof = convert_bbox_to_9dof(
                center=bbox_3d["center"],
                dimensions=bbox_3d["dimensions"], 
                rotation=bbox_3d["rotation"],
                rotation_format='euler'
            )
            
            # Add metadata
            bbox_9dof["category"] = bbox_3d["category"]
            bbox_9dof["category_id"] = bbox_3d.get("category_id")
            bbox_9dof["confidence"] = bbox_3d["confidence"]
            bbox_9dof["method"] = bbox_3d["method"]
            
            return bbox_9dof
            
        except Exception as e:
            logger.warning(f"Failed to convert 2D bbox to 3D: {e}")
            return None
    
    def load_coco_annotations(self, split: str = "validation") -> Dict:
        """
        Load COCO annotations from labels.json.
        
        Args:
            split: Dataset split (validation or train)
        
        Returns:
            Dict with images, annotations, and categories
        """
        logger.info(f"Loading COCO annotations for {split}")
        
        # Load labels.json file
        labels_file = self.coco_dir / split / "labels.json"
        if not labels_file.exists():
            logger.warning(f"Labels file not found: {labels_file} - skipping {split} split")
            return {"images": [], "annotations": [], "categories": []}
        
        with open(labels_file, 'r') as f:
            coco_data = json.load(f)
        
        logger.info(f"Loaded {len(coco_data['images'])} images, "
                   f"{len(coco_data['annotations'])} annotations, "
                   f"{len(coco_data['categories'])} categories")
        
        return coco_data
    
    def process_split(self, split: str = "validation"):
        """
        Process a dataset split.
        
        Args:
            split: Dataset split to process
        """
        logger.info(f"Processing COCO {split} split")
        
        # Load annotations
        coco_data = self.load_coco_annotations(split)
        
        if not coco_data["images"]:
            logger.warning(f"No images found for {split} split")
            return
        
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Build image_id to annotations mapping
        img_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Build category mapping
        cat_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        processed_count = 0
        for idx, img_info in enumerate(coco_data["images"]):
            if idx % 10 == 0:
                logger.info(f"  Processing {idx}/{len(coco_data['images'])}")
            
            img_id = img_info["id"]
            filename = img_info["file_name"]
            
            # Load image from data/ subdirectory
            img_path = self.coco_dir / split / "data" / filename
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Load image to get dimensions
            img = Image.open(img_path)
            width, height = img.size
            
            # Get annotations for this image
            anns = img_to_anns.get(img_id, [])
            
            # Build 2D bounding boxes
            bboxes_2d = []
            for ann in anns:
                if "bbox" not in ann:
                    continue
                
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = ann["bbox"]
                category_id = ann["category_id"]
                category_name = cat_map.get(category_id, "unknown")
                
                bbox_2d = {
                    "category": category_name,
                    "category_id": category_id,
                    "bbox_2d": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h)
                    },
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": bool(ann.get("iscrowd", 0))
                }
                bboxes_2d.append(bbox_2d)
            
            # Estimate depth (optional)
            depth_stats = None
            depth_type = "none"
            depth_map = None
            
            if self.depth_model:
                depth_map = self.estimate_depth(img_path)
                if depth_map is not None:
                    depth_stats = compute_depth_stats(depth_map)
                    depth_type = "pseudo"
            
            # Convert 2D bboxes to 3D - skip images without depth or 3D conversions
            bboxes_3d = []
            if depth_map is not None:
                for bbox_2d in bboxes_2d:
                    bbox_3d = self.convert_2d_to_3d_bbox(bbox_2d, depth_map, width, height)
                    if bbox_3d is not None:
                        bboxes_3d.append(bbox_3d)
            
            # Skip images that don't have any 3D bboxes
            if len(bboxes_3d) == 0:
                logger.info(f"  Skipping {filename}: No valid 3D bboxes generated")
                continue
            
            # Create unified JSON
            unified_data = {
                "dataset": "coco",
                "split": split,
                "image_id": str(img_id),
                "filename": filename,
                "rgb_path": f"data/{filename}",
                "depth_path": None,
                "depth_type": depth_type,
                "camera": {
                    "fx": None,  # COCO doesn't have camera intrinsics
                    "fy": None,
                    "cx": None,
                    "cy": None,
                    "image_width": width,
                    "image_height": height,
                    "intrinsics": None,
                    "extrinsics": None
                },
                "depth_stats": depth_stats,
                "bounding_boxes_2d": [],  # Remove 2D annotations - only keep 3D
                "bounding_boxes_3d": bboxes_3d
            }
            
            # Save per-image JSON
            output_file = split_dir / f"{img_id:012d}.json"
            save_json(unified_data, output_file)
            processed_count += 1
        
        logger.info(f"Processed {processed_count} images for {split} split")
    
    def process_all(self):
        """Process all available splits."""
        logger.info("Processing COCO dataset")
        
        total_images = 0
        total_annotations = 0
        total_3d_bboxes = 0
        processed_splits = []
        
        # Check which splits are available and have labels
        available_splits = []
        for split in ["validation", "train"]:
            split_dir = self.coco_dir / split
            labels_file = split_dir / "labels.json"
            if split_dir.exists() and labels_file.exists():
                available_splits.append(split)
            elif split_dir.exists():
                logger.info(f"Split '{split}' directory exists but no labels.json found - skipping")
        
        if not available_splits:
            logger.warning("No valid splits found with labels.json files")
            return
        
        logger.info(f"Found valid splits: {', '.join(available_splits)}")
        
        # Process each available split
        for split in available_splits:
            logger.info(f"\nProcessing {split} split...")
            self.process_split(split)
            
            # Count processed files
            split_output = self.output_dir / split
            if split_output.exists():
                split_files = list(split_output.glob("*.json"))
                split_images = len(split_files)
                split_annotations = 0
                split_3d_bboxes = 0
                
                # Count annotations
                for f in split_files:
                    try:
                        data = json.load(open(f))
                        split_annotations += len(data.get("bounding_boxes_2d", []))  # Keep for backward compatibility stats
                        split_3d_bboxes += len(data.get("bounding_boxes_3d", []))
                    except Exception as e:
                        logger.warning(f"Error reading {f}: {e}")
                
                total_images += split_images
                total_annotations += split_annotations
                total_3d_bboxes += split_3d_bboxes
                processed_splits.append(split)
                
                logger.info(f"  {split}: {split_images} images, {split_3d_bboxes} 3D bboxes")
        
        # Save summary
        summary = {
            "dataset": "coco",
            "processed_splits": processed_splits,
            "total_images": total_images,
            "total_2d_annotations": total_annotations,
            "total_3d_bboxes": total_3d_bboxes,
            "depth_type": "pseudo" if self.depth_model else "none",
            "processing_date": datetime.now().isoformat(),
            "output_directory": str(self.output_dir)
        }
        
        save_json(summary, self.output_dir / "summary.json")
        logger.info(f"\n✅ COCO processing complete!")
        logger.info(f"   Processed splits: {', '.join(processed_splits)}")
        logger.info(f"   Total images: {total_images}")
        logger.info(f"   Total 3D bboxes: {total_3d_bboxes}")
        logger.info(f"   Output: {self.output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        raw_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    else:
        raw_dir = Path(__file__).parent.parent / "raw_data" / "COCO"
        output_dir = Path(__file__).parent.parent / "processed_data" / "coco"
    
    processor = COCOProcessor(raw_dir, output_dir)
    processor.process_all()
