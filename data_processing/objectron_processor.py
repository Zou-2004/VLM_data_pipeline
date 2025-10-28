"""
Objectron Dataset Processor
Processes Objectron video dataset with 9-DoF bounding boxes.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

import utils
from utils import (
    quaternion_to_euler,
    convert_bbox_to_9dof,
    create_unified_json,
    save_json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectronProcessor:
    def __init__(self, raw_data_dir: Path, output_dir: Path):
        """
        Initialize Objectron processor.
        
        Args:
            raw_data_dir: Path to raw_data/Objectron
            output_dir: Path to processed_data/objectron
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.annotations_dir = self.raw_data_dir / "annotations"
        self.videos_dir = self.raw_data_dir / "videos"
    
    def parse_pbdata(self, pbdata_path: Path) -> List[Dict]:
        """
        Parse Objectron .pbdata file using protobuf.
        
        Args:
            pbdata_path: Path to .pbdata file
        
        Returns:
            List of frame annotations
        """
        try:
            # Import Objectron protobuf schema
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from objectron.schema import annotation_data_pb2
            
            # Read and parse the protobuf file
            with open(pbdata_path, 'rb') as f:
                sequence = annotation_data_pb2.Sequence()
                sequence.ParseFromString(f.read())
            
            frames = []
            for frame_annotation in sequence.frame_annotations:
                frame_data = {
                    'frame_id': frame_annotation.frame_id,
                    'timestamp': frame_annotation.timestamp,
                    'camera': {
                        'intrinsics': list(frame_annotation.camera.intrinsics) if frame_annotation.HasField('camera') else None,
                        'view_matrix': list(frame_annotation.camera.view_matrix) if frame_annotation.HasField('camera') else None,
                    },
                    'objects': []
                }
                
                # Extract 3D bounding boxes from keypoints
                for obj_ann in frame_annotation.annotations:
                    keypoints_2d = []
                    keypoints_3d = []
                    
                    for kp in obj_ann.keypoints:
                        if kp.HasField('point_2d'):
                            keypoints_2d.append([kp.point_2d.x, kp.point_2d.y])
                        if kp.HasField('point_3d'):
                            keypoints_3d.append([kp.point_3d.x, kp.point_3d.y, kp.point_3d.z])
                    
                    # Objectron requires 9 keypoints (1 center + 8 corners)
                    if len(keypoints_3d) >= 9:
                        obj_data = {
                            'object_id': obj_ann.object_id,
                            'keypoints_2d': keypoints_2d,
                            'keypoints_3d': keypoints_3d,
                            'visibility': obj_ann.visibility  # scalar float, not optional
                        }
                        frame_data['objects'].append(obj_data)
                
                if frame_data['objects']:  # Only add frames with objects
                    frames.append(frame_data)
            
            logger.info(f"Parsed {len(frames)} frames with objects from {pbdata_path.name}")
            return frames
            
        except Exception as e:
            logger.error(f"Error parsing {pbdata_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_video_annotations(self, pbdata_path: Path, category: str) -> List[Dict]:
        """
        Process annotations for a single video.
        
        Args:
            pbdata_path: Path to annotation .pbdata file
            category: Object category
        
        Returns:
            List of processed frame data
        """
        processed_data = []
        
        try:
            # Parse pbdata file
            frame_annotations = self.parse_pbdata(pbdata_path)
            
            if not frame_annotations:
                return []
            
            # Sample frames (e.g., every 10th frame)
            sample_rate = 10
            sampled_frames = frame_annotations[::sample_rate]
            
            video_id = pbdata_path.stem
            
            for frame_idx, frame_data in enumerate(sampled_frames):
                # Extract camera parameters
                camera_intrinsics = None
                camera_extrinsics = None
                
                if frame_data.get('camera'):
                    cam = frame_data['camera']
                    if cam.get('intrinsics') and len(cam['intrinsics']) >= 9:
                        # Reshape 9-element intrinsics to 3x3 matrix
                        intr = cam['intrinsics']
                        camera_intrinsics = np.array([
                            [intr[0], intr[1], intr[2]],
                            [intr[3], intr[4], intr[5]],
                            [intr[6], intr[7], intr[8]]
                        ], dtype=np.float32)
                    
                    if cam.get('view_matrix') and len(cam['view_matrix']) >= 16:
                        # Reshape 16-element view matrix to 4x4
                        view = cam['view_matrix']
                        camera_extrinsics = np.array([
                            [view[0], view[1], view[2], view[3]],
                            [view[4], view[5], view[6], view[7]],
                            [view[8], view[9], view[10], view[11]],
                            [view[12], view[13], view[14], view[15]]
                        ], dtype=np.float32)
                
                # Extract 3D bounding boxes from keypoints
                bboxes_3d = []
                for obj_data in frame_data.get('objects', []):
                    keypoints_3d = np.array(obj_data.get('keypoints_3d', []))
                    
                    if len(keypoints_3d) < 9:
                        continue  # Need 9 keypoints for a complete bbox
                    
                    # In Objectron, keypoint 0 is the center
                    # Keypoints 1-8 define the 8 corners of the bbox
                    center_opengl = keypoints_3d[0]  # Center point in OpenGL convention
                    
                    # Convert from OpenGL to CV convention:
                    # OpenGL: X right, Y up, Z backward (-Z forward)
                    # CV:     X right, Y down, Z forward
                    # Transformation: x_cv = x_gl, y_cv = -y_gl, z_cv = -z_gl
                    center = np.array([
                        center_opengl[0],    # X: keep as is
                        -center_opengl[1],   # Y: flip (up → down)
                        -center_opengl[2]    # Z: flip (backward → forward)
                    ])
                    
                    # Calculate dimensions from corner points
                    # Use distances between opposite corners
                    corners = keypoints_3d[1:9]  # 8 corner points
                    
                    # Width (x): distance between corners 1-2 or 3-4
                    width = np.linalg.norm(corners[1] - corners[0])  # distance along X
                    # Height (y): distance between corners 1-5 or 2-6
                    height = np.linalg.norm(corners[4] - corners[0])  # distance along Y
                    # Depth (z): distance between corners 1-3 or 2-4
                    depth = np.linalg.norm(corners[2] - corners[0])  # distance along Z
                    
                    dimensions = np.array([width, height, depth])
                    
                    # For rotation, we'll use identity (no rotation) since keypoints are already in world space
                    # If rotation is needed, it should be computed from corner orientations
                    rotation = [1, 0, 0, 0]  # quaternion (w, x, y, z) - identity
                    
                    # Convert to 9-DoF bbox
                    bbox_9dof = convert_bbox_to_9dof(
                        center=center,
                        dimensions=dimensions,
                        rotation=rotation,
                        rotation_format="quaternion"
                    )
                    bbox_9dof['category'] = category
                    bbox_9dof['object_id'] = obj_data.get('object_id', f"obj_{len(bboxes_3d)}")
                    
                    bboxes_3d.append(bbox_9dof)
                
                if not bboxes_3d:
                    continue  # Skip frames without 3D boxes
                
                # Create unified JSON
                frame_id = frame_data.get('frame_id', frame_idx)
                unified_data = {
                    "dataset": "objectron",
                    "split": "train",
                    "video_id": video_id,
                    "category": category,
                    "frame_id": str(frame_id),
                    "image_id": f"{video_id}_frame_{frame_idx:04d}",
                    "filename": f"frame_{frame_idx:04d}.jpg",
                    "rgb_path": None,  # No actual images in annotation files
                    "depth_path": None,
                    "depth_type": "none",
                    "camera": {
                        "fx": float(camera_intrinsics[0, 0]) if camera_intrinsics is not None else None,
                        "fy": float(camera_intrinsics[1, 1]) if camera_intrinsics is not None else None,
                        "cx": float(camera_intrinsics[0, 2]) if camera_intrinsics is not None else None,
                        "cy": float(camera_intrinsics[1, 2]) if camera_intrinsics is not None else None,
                        "image_width": None,
                        "image_height": None,
                        "intrinsics": camera_intrinsics.tolist() if camera_intrinsics is not None else None,
                        "extrinsics": camera_extrinsics.tolist() if camera_extrinsics is not None else None
                    },
                    "depth_stats": None,
                    "bounding_boxes_2d": [],
                    "bounding_boxes_3d": bboxes_3d,
                    "timestamp": frame_data.get('timestamp', 0)
                }
                
                # Save processed JSON
                output_dir = self.output_dir / category
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_id}_frame_{frame_idx:04d}.json"
                save_json(unified_data, output_path)
                
                processed_data.append(unified_data)
                
        except Exception as e:
            logger.error(f"Error processing {pbdata_path}: {e}")
            import traceback
            traceback.print_exc()
        
        return processed_data
    
    def process_category(self, category: str) -> List[Dict]:
        """
        Process all videos for a category.
        
        Args:
            category: Object category (e.g., 'bike', 'chair')
        
        Returns:
            List of processed data
        """
        category_dir = self.annotations_dir / category
        if not category_dir.exists():
            logger.warning(f"Category not found: {category_dir}")
            return []
        
        processed_data = []
        pbdata_files = list(category_dir.glob("*.pbdata"))
        
        logger.info(f"Processing {len(pbdata_files)} videos for category: {category}")
        
        for pbdata_path in sorted(pbdata_files):
            logger.info(f"  Processing: {pbdata_path.name}")
            processed_data.extend(self.process_video_annotations(pbdata_path, category))
        
        return processed_data
    
    def process_all(self) -> Dict[str, List[Dict]]:
        """
        Process all Objectron categories.
        
        Returns:
            Dict mapping categories to processed data
        """
        if not self.annotations_dir.exists():
            logger.error(f"Annotations directory not found: {self.annotations_dir}")
            return {}
        
        all_data = {}
        
        # Extract categories from annotation filenames (e.g., "bike_batch-0_10.pbdata" -> "bike")
        annotation_files = list(self.annotations_dir.glob("*.pbdata"))
        categories = set()
        for f in annotation_files:
            # Category is the prefix before the first underscore
            category = f.name.split('_')[0]
            categories.add(category)
        
        categories = sorted(categories)
        logger.info(f"Found {len(categories)} categories: {categories}")
        
        for category in categories:
            logger.info(f"Processing category: {category}")
            category_output_dir = self.output_dir / category
            category_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all annotation files for this category
            category_files = sorted(self.annotations_dir.glob(f"{category}_*.pbdata"))
            logger.info(f"  Found {len(category_files)} annotation files for {category}")
            
            processed_data = []
            for pbdata_path in category_files:
                logger.info(f"  Processing: {pbdata_path.name}")
                processed_data.extend(self.process_video_annotations(pbdata_path, category))
            
            all_data[category] = processed_data
        
        # Save summary
        summary = {
            "dataset": "objectron",
            "total_categories": len(all_data),
            "total_frames": sum(len(v) for v in all_data.values()),
            "categories": {k: len(v) for k, v in all_data.items()}
        }
        save_json(summary, self.output_dir / "summary.json")
        logger.info(f"Objectron processing complete: {summary['total_frames']} frames from {summary['total_categories']} categories")
        
        return all_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Objectron dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to raw Objectron data')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    
    processor = ObjectronProcessor(
        raw_data_dir=Path(args.input),
        output_dir=Path(args.output)
    )
    processor.process_all()
    logger.info("✅ Objectron processing complete!")
