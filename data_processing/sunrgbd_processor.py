"""
SUN RGB-D Dataset Processor
Processes SUN RGB-D scenes with 3D bounding boxes.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from utils import convert_bbox_to_9dof, compute_depth_stats, save_json, load_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_intrinsics(intrinsics_path: Path) -> Dict:
    """Load camera intrinsics from text file."""
    with open(intrinsics_path, 'r') as f:
        line = f.read().strip().split('%')[0].strip()
        values = [float(x) for x in line.split()]
    
    # Intrinsics format: fx 0 cx 0 fy cy 0 0 1
    fx, _, cx, _, fy, cy = values[:6]
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


def parse_3d_bbox(obj: Dict) -> Optional[Dict]:
    """Parse 3D bounding box from annotation object."""
    if 'polygon' not in obj or not obj['polygon']:
        return None
    
    polygon = obj['polygon'][0]
    if not polygon.get('rectangle', False):
        return None  # Only process rectangular bounding boxes
    
    # Get 3D coordinates
    X = polygon.get('X', [])
    Z = polygon.get('Z', [])
    Ymin = polygon.get('Ymin', 0)
    Ymax = polygon.get('Ymax', 0)
    
    if len(X) != 4 or len(Z) != 4:
        return None
    
    # Calculate center and dimensions
    x_center = sum(X) / 4
    z_center = sum(Z) / 4
    y_center = (Ymin + Ymax) / 2
    
    width = max(X) - min(X)
    length = max(Z) - min(Z)
    height = abs(Ymax - Ymin)
    
    # Category name cleaning
    category = obj.get('name', 'unknown')
    # Remove modifiers like :truncated, :occluded
    if ':' in category:
        category = category.split(':')[0]
    
    return {
        'category': category,
        'center': [x_center, y_center, z_center],
        'dimensions': [width, height, length],
        'rotation': [0.0, 0.0, 0.0]  # SUN RGB-D doesn't provide rotation
    }


def process_scene(scene_path: Path, sensor_type: str, dataset_name: str) -> Optional[Dict]:
    """Process a single SUN RGB-D scene."""
    scene_id = scene_path.name
    
    # Check required files exist
    image_path = scene_path / 'image' / f'{scene_id}.jpg'
    depth_path = scene_path / 'depth' / f'{scene_id}.png'
    intrinsics_path = scene_path / 'intrinsics.txt'
    # Use annotation3Dfinal for final annotations
    annotation_path = scene_path / 'annotation3Dfinal' / 'index.json'
    
    if not all([image_path.exists(), depth_path.exists(), 
                intrinsics_path.exists(), annotation_path.exists()]):
        logger.warning(f"Skipping {scene_id}: Missing required files")
        return None
    
    # Load intrinsics
    try:
        intrinsics = load_intrinsics(intrinsics_path)
    except Exception as e:
        logger.warning(f"Failed to load intrinsics for {scene_id}: {e}")
        return None
    
    # Load 3D annotations
    try:
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load annotations for {scene_id}: {e}")
        return None
    
    # Parse 3D bounding boxes
    bboxes_3d = []
    objects = annotation_data.get('objects', [])
    
    for obj in objects:
        bbox = parse_3d_bbox(obj)
        if bbox:
            # Convert to 9-DoF format
            bbox_9dof = convert_bbox_to_9dof(
                center=bbox['center'],
                dimensions=bbox['dimensions'],
                rotation=bbox['rotation'],
                rotation_format='euler'
            )
            # Add category to bbox
            bbox_9dof['category'] = bbox['category']
            bboxes_3d.append(bbox_9dof)
    
    if not bboxes_3d:
        logger.warning(f"No valid 3D bboxes for {scene_id}")
        return None
    
    # Compute depth statistics
    try:
        import cv2
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            # SUN RGB-D depth is in millimeters, convert to meters
            depth_img = depth_img.astype(np.float32) / 1000.0
            depth_stats = compute_depth_stats(depth_img)
        else:
            depth_stats = None
    except Exception as e:
        logger.warning(f"Failed to compute depth stats for {scene_id}: {e}")
        depth_stats = None
    
    # Get image dimensions
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            image_width, image_height = img.size
    except Exception as e:
        logger.warning(f"Failed to get image dimensions for {scene_id}: {e}")
        image_width, image_height = 640, 480  # Default SUN RGB-D resolution
    
    # Create unified JSON
    unified_json = {
        "dataset": "sunrgbd",
        "split": f"{sensor_type}_{dataset_name}",
        "image_id": scene_id,
        "filename": f"{scene_id}.jpg",
        "rgb_path": str(image_path.relative_to(scene_path.parent.parent.parent)),
        "depth_path": str(depth_path.relative_to(scene_path.parent.parent.parent)),
        "depth_type": "depth_png_mm",  # PNG format, millimeters
        "camera": {
            "fx": intrinsics['fx'],
            "fy": intrinsics['fy'],
            "cx": intrinsics['cx'],
            "cy": intrinsics['cy'],
            "image_width": image_width,
            "image_height": image_height,
            "intrinsics": None,
            "extrinsics": None
        },
        "depth_stats": depth_stats,
        "bounding_boxes_2d": [],  # SUN RGB-D doesn't provide 2D boxes
        "bounding_boxes_3d": bboxes_3d
    }
    
    return unified_json


class SUNRGBDProcessor:
    """SUN RGB-D Dataset Processor"""
    
    def __init__(self, raw_data_dir: Path, output_dir: Path, sensor_types: List[str] = None):
        """
        Initialize SUN RGB-D processor.
        
        Args:
            raw_data_dir: Path to raw_data/SUNRGBD
            output_dir: Path to processed_data/sunrgbd
            sensor_types: List of sensor types to process (default: all)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.sensor_types = sensor_types or ['kv1', 'kv2', 'realsense', 'xtion']
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all(self) -> Dict:
        """Process all SUN RGB-D scenes."""
        logger.info("Processing SUN RGB-D dataset...")
        
        total_scenes = 0
        total_bboxes = 0
        processed_scenes = []
        
        # Process each sensor type
        for sensor_type in self.sensor_types:
            sensor_dir = self.raw_data_dir / sensor_type
            if not sensor_dir.exists():
                logger.warning(f"Sensor directory not found: {sensor_dir}")
                continue
            
            logger.info(f"Processing sensor type: {sensor_type}")
            
            # Find all dataset directories
            for dataset_dir in sensor_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                logger.info(f"  Dataset: {dataset_name}")
                
                # Create output subdirectory
                split_output_dir = self.output_dir / f'{sensor_type}_{dataset_name}'
                split_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each scene
                scene_count = 0
                for scene_dir in dataset_dir.iterdir():
                    if not scene_dir.is_dir():
                        continue
                    
                    result = process_scene(scene_dir, sensor_type, dataset_name)
                    if result:
                        # Save to JSON file
                        output_file = split_output_dir / f'{scene_dir.name}.json'
                        save_json(result, output_file)
                        
                        scene_count += 1
                        total_scenes += 1
                        total_bboxes += len(result.get('bounding_boxes_3d', []))
                        processed_scenes.append({
                            'scene_id': scene_dir.name,
                            'sensor_type': sensor_type,
                            'dataset': dataset_name,
                            'num_objects': len(result.get('bounding_boxes_3d', []))
                        })
                        
                        if scene_count % 100 == 0:
                            logger.info(f"    Processed {scene_count} scenes...")
                
                logger.info(f"  Finished {dataset_name}: {scene_count} scenes")
        
        # Create summary
        summary = {
            'dataset': 'sunrgbd',
            'total_scenes': total_scenes,
            'total_3d_bboxes': total_bboxes,
            'sensor_types': self.sensor_types,
            'scenes': processed_scenes
        }
        
        summary_file = self.output_dir / 'summary.json'
        save_json(summary, summary_file)
        logger.info(f"Saved summary to {summary_file}")
        
        logger.info(f"✅ SUN RGB-D processing complete!")
        logger.info(f"   Total scenes: {total_scenes}")
        logger.info(f"   Total 3D bboxes: {total_bboxes}")
        
        return summary


def main():
    """Main entry point."""
    # Use relative paths
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir.parent / 'raw_data' / 'SUNRGBD'
    output_dir = script_dir.parent / 'processed_data' / 'sunrgbd'
    
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        return
    
    # Create processor and run
    processor = SUNRGBDProcessor(raw_data_dir, output_dir)
    summary = processor.process_all()
    
    print(f"\n{'='*60}")
    print(f"SUN RGB-D Processing Summary")
    print(f"{'='*60}")
    print(f"Total scenes processed: {summary['total_scenes']}")
    print(f"Total 3D bounding boxes: {summary['total_3d_bboxes']}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
