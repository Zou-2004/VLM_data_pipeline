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
    compute_depth_stats
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
            logger.error(f"Labels file not found: {labels_file}")
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
            if self.depth_model:
                depth_map = self.estimate_depth(img_path)
                if depth_map is not None:
                    depth_stats = compute_depth_stats(depth_map)
                    depth_type = "pseudo"
            
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
                "bounding_boxes_2d": bboxes_2d,
                "bounding_boxes_3d": []  # COCO has no 3D boxes
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
        
        # Process validation split
        val_dir = self.coco_dir / "validation"
        if val_dir.exists():
            logger.info("\nProcessing validation split...")
            self.process_split("validation")
            
            # Count processed files
            val_output = self.output_dir / "validation"
            if val_output.exists():
                val_files = list(val_output.glob("*.json"))
                total_images += len(val_files)
                
                # Count annotations
                for f in val_files:
                    data = json.load(open(f))
                    total_annotations += len(data.get("bounding_boxes_2d", []))
        
        # Process train split if available
        train_dir = self.coco_dir / "train"
        if train_dir.exists():
            logger.info("\nProcessing train split...")
            self.process_split("train")
            
            # Count processed files
            train_output = self.output_dir / "train"
            if train_output.exists():
                train_files = list(train_output.glob("*.json"))
                total_images += len(train_files)
                
                # Count annotations
                for f in train_files:
                    data = json.load(open(f))
                    total_annotations += len(data.get("bounding_boxes_2d", []))
        
        # Save summary
        summary = {
            "dataset": "coco",
            "total_images": total_images,
            "total_annotations": total_annotations,
            "depth_type": "pseudo" if self.depth_model else "none",
            "processing_date": datetime.now().isoformat(),
            "output_directory": str(self.output_dir)
        }
        
        save_json(summary, self.output_dir / "summary.json")
        logger.info(f"\n✅ COCO processing complete!")
        logger.info(f"   Total images: {total_images}")
        logger.info(f"   Total annotations: {total_annotations}")
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
