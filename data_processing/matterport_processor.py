"""
Matterport3D Dataset Processor (with EmbodiedScan corrections)
Processes Matterport3D with corrected 9-DoF bounding boxes.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from PIL import Image

import utils
from utils import (
    convert_bbox_to_9dof,
    create_unified_json,
    save_json,
    world_to_camera_frame
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatterportProcessor:
    def __init__(self, raw_data_dir: Path, embodiedscan_dir: Path, output_dir: Path):
        """
        Initialize Matterport processor.
        
        Args:
            raw_data_dir: Path to raw_data/v1/scans (Matterport data)
            embodiedscan_dir: Path to raw_data/embodiedscan-v2 (corrected boxes)
            output_dir: Path to processed_data/matterport
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.embodiedscan_dir = Path(embodiedscan_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load EmbodiedScan corrected boxes
        self.corrected_boxes = self.load_embodiedscan_corrections()
    
    def load_embodiedscan_corrections(self) -> Dict:
        """
        Load corrected bounding boxes from EmbodiedScan.
        
        Returns:
            Dict mapping scene_id -> image_path -> boxes
        """
        import pickle
        
        corrections = {}
        
        # Load train and val pickle files
        for split in ['train', 'val']:
            pkl_file = self.embodiedscan_dir / "embodiedscan-v2" / f"embodiedscan_infos_{split}.pkl"
            if not pkl_file.exists():
                logger.warning(f"EmbodiedScan file not found: {pkl_file}")
                continue
            
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # data is a dict with 'data_list'
                for item in data.get('data_list', []):
                    sample_idx = item.get('sample_idx', '')
                    
                    # sample_idx format: "matterport3d/SCENE_NAME/region0" or similar
                    # We only want matterport scenes
                    if not sample_idx.startswith('matterport3d/'):
                        continue
                    
                    # Extract scene name from path (e.g., "matterport3d/1LXtFkjw3qL/region0" -> "1LXtFkjw3qL")
                    parts = sample_idx.split('/')
                    if len(parts) < 2:
                        continue
                    scene_id = parts[1]
                    
                    # Get image paths and instances
                    images = item.get('images', [])
                    instances = item.get('instances', [])
                    
                    # Create a mapping of bbox_id to bbox data
                    bbox_by_id = {}
                    for inst in instances:
                        bbox_3d = inst.get('bbox_3d', [])
                        if len(bbox_3d) == 9:
                            # Format: [cx, cy, cz, dx, dy, dz, rx, ry, rz]
                            bbox_by_id[inst.get('bbox_id')] = {
                                'center': bbox_3d[:3],
                                'dimensions': bbox_3d[3:6],
                                'rotation': bbox_3d[6:9],  # Euler angles in radians
                                'label': inst.get('bbox_label_3d', -1)
                            }
                    
                    # Process each image
                    for img_info in images:
                        img_path = img_info.get('img_path', '')
                        if not img_path:
                            continue
                        
                        # Extract frame_id from path
                        # Path format: "matterport3d/SCENE/matterport_color_images/FRAME_ID.jpg"
                        frame_id = Path(img_path).stem
                        
                        # Get visible instances for this image
                        visible_ids = img_info.get('visible_instance_ids', [])
                        bboxes_3d = []
                        for bbox_id in visible_ids:
                            if bbox_id in bbox_by_id:
                                bboxes_3d.append(bbox_by_id[bbox_id])
                        
                        # Store by scene and frame
                        if scene_id not in corrections:
                            corrections[scene_id] = {}
                        corrections[scene_id][frame_id] = bboxes_3d
                    
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"Loaded corrections for {len(corrections)} scenes")
        return corrections
    
    def process_scene(self, scene_name: str) -> List[Dict]:
        """
        Process all images in a Matterport scene.
        
        Args:
            scene_name: Name of the scene
        
        Returns:
            List of processed image data
        """
        scene_dir = self.raw_data_dir / scene_name
        if not scene_dir.exists():
            logger.warning(f"Scene not found: {scene_dir}")
            return []
        
        processed_data = []
        
        # Matterport has nested structure: scans/SCENE/SCENE/matterport_color_images/
        nested_scene_dir = scene_dir / scene_name
        if nested_scene_dir.exists():
            scene_dir = nested_scene_dir
        
        # Find color images
        color_dir = scene_dir / "matterport_color_images"
        if not color_dir.exists():
            logger.warning(f"Color images not found: {color_dir}")
            return []
        
        # Load camera poses
        pose_dir = scene_dir / "matterport_camera_poses"
        
        for img_path in sorted(color_dir.glob("*.jpg")):
            try:
                frame_id = img_path.stem
                
                # Load camera pose
                pose_path = pose_dir / f"{frame_id}.txt"
                camera_extrinsics = None
                if pose_path.exists():
                    camera_extrinsics = np.loadtxt(pose_path).reshape(4, 4)
                
                # Use default intrinsics (Matterport doesn't always provide them)
                camera_intrinsics = np.array([
                    [1000, 0, 640],
                    [0, 1000, 512],
                    [0, 0, 1]
                ])
                
                # Get corrected boxes from EmbodiedScan
                bboxes_3d = []
                if scene_name in self.corrected_boxes and frame_id in self.corrected_boxes[scene_name]:
                    for box_data in self.corrected_boxes[scene_name][frame_id]:
                        # box_data format: {'center': [x,y,z], 'dimensions': [dx,dy,dz], 'rotation': [rx,ry,rz], 'label': int}
                        center = np.array(box_data['center'])
                        dimensions = np.array(box_data['dimensions'])
                        rotation = box_data['rotation']  # Euler angles in radians
                        
                        # Convert to 9-DoF bbox format
                        bbox_9dof = convert_bbox_to_9dof(
                            center=center,
                            dimensions=dimensions,
                            rotation=rotation,
                            rotation_format="euler"
                        )
                        bbox_9dof['category'] = f"class_{box_data['label']}"
                        bbox_9dof['label_id'] = int(box_data['label'])
                        
                        bboxes_3d.append(bbox_9dof)
                
                # Depth not typically available for Matterport
                depth_map = None
                
                # Create unified JSON in new format
                unified_data = {
                    "dataset": "matterport",
                    "split": "train",
                    "scene_id": scene_name,
                    "frame_id": frame_id,
                    "image_id": f"{scene_name}_{frame_id}",
                    "filename": img_path.name,
                    "rgb_path": str(img_path.relative_to(self.raw_data_dir.parent.parent)),
                    "depth_path": None,
                    "depth_type": "none",
                    "camera": {
                        "fx": float(camera_intrinsics[0, 0]),
                        "fy": float(camera_intrinsics[1, 1]),
                        "cx": float(camera_intrinsics[0, 2]),
                        "cy": float(camera_intrinsics[1, 2]),
                        "image_width": None,
                        "image_height": None,
                        "intrinsics": camera_intrinsics.tolist(),
                        "extrinsics": camera_extrinsics.tolist() if camera_extrinsics is not None else None
                    },
                    "depth_stats": None,
                    "bounding_boxes_2d": [],
                    "bounding_boxes_3d": bboxes_3d
                }
                
                # Save processed JSON
                output_path = self.output_dir / scene_name / f"{frame_id}.json"
                save_json(unified_data, output_path)
                
                processed_data.append(unified_data)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_data)} images from scene {scene_name}")
        return processed_data
    
    def process_all(self) -> Dict[str, List[Dict]]:
        """
        Process all Matterport scenes.
        
        Returns:
            Dict mapping scene names to processed data
        """
        if not self.raw_data_dir.exists():
            logger.error(f"Matterport directory not found: {self.raw_data_dir}")
            return {}
        
        all_data = {}
        scene_dirs = [d for d in self.raw_data_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(scene_dirs)} scenes to process")
        
        for scene_dir in sorted(scene_dirs):
            scene_name = scene_dir.name
            logger.info(f"Processing scene: {scene_name}")
            all_data[scene_name] = self.process_scene(scene_name)
        
        # Save summary
        summary = {
            "dataset": "matterport",
            "total_scenes": len(all_data),
            "total_images": sum(len(v) for v in all_data.values()),
            "scenes": {k: len(v) for k, v in all_data.items()}
        }
        save_json(summary, self.output_dir / "summary.json")
        logger.info(f"Matterport processing complete: {summary['total_images']} images from {summary['total_scenes']} scenes")
        
        return all_data
