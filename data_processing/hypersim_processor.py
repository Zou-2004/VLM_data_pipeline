"""
Hypersim Dataset Processor
Processes Hypersim photorealistic indoor scenes with 3D bounding boxes.
"""

import json
import numpy as np
import h5py
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from utils import convert_bbox_to_9dof, compute_depth_stats, save_json, load_json, world_to_camera_frame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_camera_metadata(detail_dir: Path) -> Dict:
    """Load camera metadata from CSV file."""
    camera_file = detail_dir / 'metadata_cameras.csv'
    if not camera_file.exists():
        return {}
    
    with open(camera_file, 'r') as f:
        reader = csv.DictReader(f)
        cameras = list(reader)
    
    return {cam['camera_name']: cam for cam in cameras}


def load_camera_poses(detail_dir: Path, camera_name: str, frame_id: int) -> Optional[List[List[float]]]:
    """
    Load camera pose (extrinsics) for a specific frame.
    
    Args:
        detail_dir: Path to scene detail directory
        camera_name: Camera name (e.g., "cam_00")
        frame_id: Frame number
        
    Returns:
        4x4 camera-to-world transformation matrix
    """
    camera_dir = detail_dir / camera_name
    if not camera_dir.exists():
        return None
    
    try:
        # Load camera position and orientation
        positions_file = camera_dir / 'camera_keyframe_positions.hdf5'
        orientations_file = camera_dir / 'camera_keyframe_orientations.hdf5'
        frame_indices_file = camera_dir / 'camera_keyframe_frame_indices.hdf5'
        
        if not all([positions_file.exists(), orientations_file.exists(), frame_indices_file.exists()]):
            return None
        
        with h5py.File(frame_indices_file, 'r') as f:
            keyframe_indices = f['dataset'][:]
        
        with h5py.File(positions_file, 'r') as f:
            keyframe_positions = f['dataset'][:]  # Shape: (N, 3)
        
        with h5py.File(orientations_file, 'r') as f:
            keyframe_orientations = f['dataset'][:]  # Shape: (N, 3, 3)
        
        # Find the keyframe for this frame (interpolate if needed)
        if frame_id in keyframe_indices:
            keyframe_idx = np.where(keyframe_indices == frame_id)[0][0]
            position = keyframe_positions[keyframe_idx]
            orientation = keyframe_orientations[keyframe_idx]
        else:
            # Simple nearest neighbor for now (could do interpolation)
            keyframe_idx = np.argmin(np.abs(keyframe_indices - frame_id))
            position = keyframe_positions[keyframe_idx]
            orientation = keyframe_orientations[keyframe_idx]
        
        # Build 4x4 camera-to-world transformation matrix
        # [R | t]
        # [0 | 1]
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = orientation  # Rotation
        extrinsics[:3, 3] = position      # Translation
        
        return extrinsics.tolist()
    
    except Exception as e:
        logger.warning(f"Failed to load camera pose for {camera_name} frame {frame_id}: {e}")
        return None


def load_object_metadata(detail_dir: Path) -> Dict[int, Dict]:
    """Load object metadata mapping instance IDs to object info."""
    nodes_file = detail_dir / 'metadata_nodes.csv'
    if not nodes_file.exists():
        return {}
    
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        nodes = list(reader)
    
    # Create mapping from node_id to object info
    object_map = {}
    for node in nodes:
        try:
            node_id = int(node['node_id'])
            object_map[node_id] = {
                'name': node.get('object_name', 'unknown'),
                'node_name': node.get('node_name', ''),
                'object_id': node.get('object_id', '')
            }
        except (ValueError, KeyError):
            continue
    
    return object_map


def load_3d_bounding_boxes(mesh_dir: Path) -> Dict[int, Dict]:
    """Load 3D bounding boxes from HDF5 files.
    
    Returns:
        Dictionary mapping instance ID to bbox info (position, extents, orientation)
    """
    bbox_positions_file = mesh_dir / 'metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5'
    bbox_extents_file = mesh_dir / 'metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5'
    bbox_orientations_file = mesh_dir / 'metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5'
    
    if not all([bbox_positions_file.exists(), bbox_extents_file.exists(), bbox_orientations_file.exists()]):
        logger.warning(f"Missing 3D bounding box files in {mesh_dir}")
        return {}
    
    try:
        with h5py.File(bbox_positions_file, 'r') as f:
            positions = f['dataset'][:]  # Shape: (N, 3)
        
        with h5py.File(bbox_extents_file, 'r') as f:
            extents = f['dataset'][:]  # Shape: (N, 3)
        
        with h5py.File(bbox_orientations_file, 'r') as f:
            orientations = f['dataset'][:]  # Shape: (N, 3, 3)
        
        # Create mapping from instance ID to bbox
        bboxes = {}
        for i in range(len(positions)):
            # Skip invalid bboxes (marked with inf)
            if np.any(np.isinf(positions[i])) or np.any(np.isinf(extents[i])):
                continue
            
            bboxes[i] = {
                'position': positions[i].tolist(),  # [x, y, z] center
                'extents': extents[i].tolist(),      # [width, height, depth]
                'orientation': orientations[i].tolist()  # 3x3 rotation matrix
            }
        
        return bboxes
    
    except Exception as e:
        logger.error(f"Failed to load 3D bounding boxes: {e}")
        return {}


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """Convert 3x3 rotation matrix to Euler angles (pitch, yaw, roll)."""
    R = np.array(rotation_matrix)
    
    # Extract Euler angles (ZYX convention)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0
    
    return float(pitch), float(yaw), float(roll)


def get_camera_intrinsics_from_fov(width: int, height: int, fov_x: float = 90.0) -> Dict:
    """
    Estimate camera intrinsics from field of view.
    Hypersim uses a 90-degree horizontal FOV by default.
    
    Args:
        width: Image width
        height: Image height
        fov_x: Horizontal field of view in degrees
        
    Returns:
        Camera intrinsics dictionary
    """
    fov_x_rad = np.deg2rad(fov_x)
    fx = width / (2 * np.tan(fov_x_rad / 2))
    
    # Assume square pixels
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    
    return {
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(cx),
        'cy': float(cy)
    }


def process_frame(
    scene_dir: Path,
    camera_name: str,
    frame_id: int,
    object_map: Dict[int, Dict],
    bboxes_3d: Dict[int, Dict],
    detail_dir: Path,
    meters_per_asset_unit: float = 1.0
) -> Optional[Dict]:
    """Process a single frame from Hypersim dataset."""
    
    # Paths to data files
    final_dir = scene_dir / 'images' / f'scene_{camera_name}_final_hdf5'
    geometry_dir = scene_dir / 'images' / f'scene_{camera_name}_geometry_hdf5'
    
    rgb_file = final_dir / f'frame.{frame_id:04d}.color.hdf5'
    depth_file = geometry_dir / f'frame.{frame_id:04d}.depth_meters.hdf5'
    semantic_instance_file = geometry_dir / f'frame.{frame_id:04d}.semantic_instance.hdf5'
    
    # Check if files exist
    if not rgb_file.exists() or not depth_file.exists():
        return None
    
    try:
        # Load RGB image dimensions
        with h5py.File(rgb_file, 'r') as f:
            rgb_data = f['dataset']
            height, width = rgb_data.shape[:2]
        
        # Load depth data
        with h5py.File(depth_file, 'r') as f:
            depth_data = f['dataset'][:]
        
        # Compute depth statistics
        depth_stats = compute_depth_stats(depth_data)
        
        # Load semantic instances (to know which objects are visible)
        visible_instances = set()
        if semantic_instance_file.exists():
            with h5py.File(semantic_instance_file, 'r') as f:
                instance_data = f['dataset'][:]
                unique_instances = np.unique(instance_data)
                # Filter out background (-1) and store valid instances
                visible_instances = set(int(i) for i in unique_instances if i >= 0)
        
        # Get camera intrinsics (Hypersim uses 90-degree horizontal FOV)
        intrinsics = get_camera_intrinsics_from_fov(width, height, fov_x=90.0)
        
        # Build intrinsics matrix
        intrinsics_matrix = [
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ]
        
        # Load camera extrinsics (pose) - kept for reference but not used for bbox transform
        extrinsics_matrix = load_camera_poses(detail_dir, camera_name, frame_id)
        
        # Process 3D bounding boxes for visible instances
        frame_bboxes_3d = []
        for instance_id in visible_instances:
            if instance_id not in bboxes_3d:
                continue
            
            bbox = bboxes_3d[instance_id]
            obj_info = object_map.get(instance_id, {})
            
            # Get category name
            category = obj_info.get('name', 'unknown')
            if not category or category == '':
                category = 'unknown'
            
            # Convert rotation matrix to Euler angles
            rotation_matrix = np.array(bbox['orientation'])
            pitch, yaw, roll = rotation_matrix_to_euler(rotation_matrix)
            
            # CRITICAL FIX: Hypersim bbox positions are in WORLD SPACE using ASSET UNITS
            # Camera poses are ALSO in asset units!
            # 1. Transform from world-space to camera-space (both in asset units)
            # 2. Convert from OpenGL (Z backward) to VST/CV (Z forward) convention
            # 3. Convert from asset units to meters
            
            center_world_asset_units = np.array(bbox['position'])
            center_world_homogeneous = np.append(center_world_asset_units, 1.0)
            
            # Invert the camera-to-world matrix (in asset units) to get world-to-camera
            camera_to_world = np.array(extrinsics_matrix)
            world_to_camera = np.linalg.inv(camera_to_world)
            
            # Transform world position to camera space (still in asset units)
            center_camera_homogeneous = world_to_camera @ center_world_homogeneous
            center_camera_opengl_asset_units = center_camera_homogeneous[:3]
            
            # Hypersim uses OpenGL convention (Y up, Z backward)
            # Convert to VST/CV convention (Y down, Z forward)
            center_camera_asset_units = np.array([
                center_camera_opengl_asset_units[0],    # X: keep (right)
                -center_camera_opengl_asset_units[1],   # Y: flip (down)
                -center_camera_opengl_asset_units[2]    # Z: flip (forward)
            ])
            
            # Now convert from asset units to meters
            center_camera = center_camera_asset_units * meters_per_asset_unit
            
            # Also scale dimensions from asset units to meters
            dimensions_meters = [d * meters_per_asset_unit for d in bbox['extents']]
            
            # Convert to 9-DoF format in camera space
            # Hypersim: position is center, extents are [width, height, depth]
            bbox_9dof = convert_bbox_to_9dof(
                center=center_camera,  # Now in camera space with correct units and convention
                dimensions=dimensions_meters,  # Now in meters
                rotation=[pitch, yaw, roll],
                rotation_format='euler'
            )
            bbox_9dof['category'] = category
            
            frame_bboxes_3d.append(bbox_9dof)
        
        # Build relative paths
        scene_name = scene_dir.name
        rgb_rel_path = f'{scene_name}/images/scene_{camera_name}_final_hdf5/frame.{frame_id:04d}.color.hdf5'
        depth_rel_path = f'{scene_name}/images/scene_{camera_name}_geometry_hdf5/frame.{frame_id:04d}.depth_meters.hdf5'
        
        # Create unified JSON
        unified_json = {
            "dataset": "hypersim",
            "split": scene_name,
            "image_id": f'{scene_name}_{camera_name}_frame_{frame_id:04d}',
            "filename": f'frame.{frame_id:04d}.color.hdf5',
            "rgb_path": rgb_rel_path,
            "depth_path": depth_rel_path,
            "depth_type": "depth_hdf5_meters",  # HDF5 format, meters
            "camera": {
                "fx": intrinsics['fx'],
                "fy": intrinsics['fy'],
                "cx": intrinsics['cx'],
                "cy": intrinsics['cy'],
                "image_width": width,
                "image_height": height,
                "intrinsics": intrinsics_matrix,
                "extrinsics": extrinsics_matrix
            },
            "depth_stats": depth_stats,
            "bounding_boxes_2d": [],  # Not provided by Hypersim
            "bounding_boxes_3d": frame_bboxes_3d
        }
        
        return unified_json
    
    except Exception as e:
        logger.error(f"Failed to process frame {frame_id} in {scene_dir.name}/{camera_name}: {e}")
        return None


def process_scene(scene_dir: Path) -> Tuple[int, int]:
    """Process all frames in a Hypersim scene.
    
    Returns:
        Tuple of (num_frames_processed, num_bboxes)
    """
    scene_name = scene_dir.name
    logger.info(f"Processing scene: {scene_name}")
    
    detail_dir = scene_dir / '_detail'
    mesh_dir = detail_dir / 'mesh'
    
    if not detail_dir.exists() or not mesh_dir.exists():
        logger.warning(f"Skipping {scene_name}: Missing detail or mesh directory")
        return 0, 0
    
    # Load scene metadata to get asset-to-meter scale factor
    import csv
    scene_metadata_file = detail_dir / 'metadata_scene.csv'
    meters_per_asset_unit = 1.0  # default if not found
    if scene_metadata_file.exists():
        with open(scene_metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['parameter_name'] == 'meters_per_asset_unit':
                    meters_per_asset_unit = float(row['parameter_value'])
                    break
    
    logger.info(f"  Scale: {meters_per_asset_unit:.6f} meters per asset unit")
    
    # Load metadata
    object_map = load_object_metadata(detail_dir)
    bboxes_3d = load_3d_bounding_boxes(mesh_dir)
    
    if not bboxes_3d:
        logger.warning(f"Skipping {scene_name}: No valid 3D bounding boxes")
        return 0, 0
    
    # Find all cameras
    images_dir = scene_dir / 'images'
    if not images_dir.exists():
        logger.warning(f"Skipping {scene_name}: No images directory")
        return 0, 0
    
    # Find camera directories (scene_cam_XX_final_hdf5 pattern)
    camera_dirs = [d for d in images_dir.iterdir() if d.is_dir() and '_final_hdf5' in d.name]
    
    if not camera_dirs:
        logger.warning(f"Skipping {scene_name}: No camera directories found")
        return 0, 0
    
    # Process each camera
    frames_processed = 0
    total_bboxes = 0
    
    for camera_dir in camera_dirs:
        # Extract camera name (e.g., "cam_00" from "scene_cam_00_final_hdf5")
        camera_name = camera_dir.name.split('_final_')[0].replace('scene_', '')
        
        # Find all frame files
        frame_files = sorted(camera_dir.glob('frame.*.color.hdf5'))
        
        logger.info(f"  Camera: {camera_name}, Frames: {len(frame_files)}")
        
        for frame_file in frame_files:
            # Extract frame number
            frame_id = int(frame_file.stem.split('.')[1])
            
            result = process_frame(scene_dir, camera_name, frame_id, object_map, bboxes_3d, detail_dir, meters_per_asset_unit)
            
            if result:
                # Save to output directory
                output_subdir = Path(__file__).parent.parent / 'processed_data' / 'hypersim' / scene_name
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_subdir / f'{camera_name}_frame_{frame_id:04d}.json'
                save_json(result, output_file)
                
                frames_processed += 1
                total_bboxes += len(result.get('bounding_boxes_3d', []))
                
                if frames_processed % 50 == 0:
                    logger.info(f"    Processed {frames_processed} frames...")
    
    logger.info(f"  Finished {scene_name}: {frames_processed} frames, {total_bboxes} bboxes")
    return frames_processed, total_bboxes


class HypersimProcessor:
    """Hypersim Dataset Processor"""
    
    def __init__(self, raw_data_dir: Path, output_dir: Path):
        """
        Initialize Hypersim processor.
        
        Args:
            raw_data_dir: Path to raw_data/Hyperism (note: typo in folder name)
            output_dir: Path to processed_data/hypersim
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all(self) -> Dict:
        """Process all Hypersim scenes."""
        logger.info("Processing Hypersim dataset...")
        
        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory not found: {self.raw_data_dir}")
            return {'error': 'Raw data directory not found'}
        
        total_frames = 0
        total_bboxes = 0
        processed_scenes = []
        
        # Find all scene directories (ai_XXX_XXX pattern)
        scene_dirs = sorted([d for d in self.raw_data_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('ai_')])
        
        logger.info(f"Found {len(scene_dirs)} scenes")
        
        for scene_dir in scene_dirs:
            frames, bboxes = process_scene(scene_dir)
            
            if frames > 0:
                total_frames += frames
                total_bboxes += bboxes
                processed_scenes.append({
                    'scene_name': scene_dir.name,
                    'num_frames': frames,
                    'num_objects': bboxes
                })
        
        # Create summary
        summary = {
            'dataset': 'hypersim',
            'total_scenes': len(processed_scenes),
            'total_frames': total_frames,
            'total_3d_bboxes': total_bboxes,
            'scenes': processed_scenes
        }
        
        summary_file = self.output_dir / 'summary.json'
        save_json(summary, summary_file)
        logger.info(f"Saved summary to {summary_file}")
        
        logger.info(f"✅ Hypersim processing complete!")
        logger.info(f"   Total scenes: {len(processed_scenes)}")
        logger.info(f"   Total frames: {total_frames}")
        logger.info(f"   Total 3D bboxes: {total_bboxes}")
        
        return summary


def main():
    """Main entry point."""
    # Use relative paths
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir.parent / 'raw_data' / 'Hyperism'  # Note: typo in folder name
    output_dir = script_dir.parent / 'processed_data' / 'hypersim'
    
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        logger.info("Please ensure the Hypersim dataset is downloaded to raw_data/Hyperism/")
        return
    
    # Create processor and run
    processor = HypersimProcessor(raw_data_dir, output_dir)
    summary = processor.process_all()
    
    if 'error' not in summary:
        print(f"\n{'='*60}")
        print(f"Hypersim Processing Summary")
        print(f"{'='*60}")
        print(f"Total scenes processed: {summary['total_scenes']}")
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Total 3D bounding boxes: {summary['total_3d_bboxes']}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
