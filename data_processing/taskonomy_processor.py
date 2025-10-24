"""
Taskonomy Dataset Processor
Processes Taskonomy multi-task learning dataset with derived 3D bounding boxes.

Taskonomy provides:
- RGB images
- Depth maps (euclidean)
- Instance segmentation (segment_unsup25d - 2.5D depth-aware)
- Camera parameters (position, rotation, FOV)

We derive 3D bounding boxes from instance masks + depth + camera params.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial.transform import Rotation

from utils import convert_bbox_to_9dof, compute_depth_stats, save_json, load_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Taskonomy semantic classes (from COCO subset)
# Based on: https://github.com/StanfordVL/taskonomy/blob/master/taskbank/assets/web_assets/pseudosemantics/coco_selected_classes.txt
SEMANTIC_CLASSES = [
    "uncertain",      # 0
    "background",     # 1
    "bottle",         # 2
    "chair",          # 3
    "couch",          # 4
    "potted_plant",   # 5
    "bed",            # 6
    "dining_table",   # 7
    "toilet",         # 8
    "tv",             # 9
    "microwave",      # 10
    "oven",           # 11
    "toaster",        # 12
    "sink",           # 13
    "refrigerator",   # 14
    "book",           # 15
    "clock",          # 16
    "vase"            # 17
]


def load_camera_info(point_info_file: Path) -> Optional[Dict]:
    """Load camera parameters from point_info JSON file."""
    if not point_info_file.exists():
        return None
    
    try:
        with open(point_info_file, 'r') as f:
            camera_info = json.load(f)
        return camera_info
    except Exception as e:
        logger.warning(f"Failed to load camera info from {point_info_file}: {e}")
        return None


def build_intrinsics_from_fov(resolution: int, fov_rads: float) -> Dict:
    """Build camera intrinsics from field of view."""
    # Taskonomy uses square images with given resolution
    fx = fy = resolution / (2 * np.tan(fov_rads / 2))
    cx = cy = resolution / 2.0
    
    return {
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(cx),
        'cy': float(cy)
    }


def build_extrinsics_from_camera_info(camera_info: Dict) -> List[List[float]]:
    """
    Build 4x4 camera-to-world extrinsics matrix from Taskonomy camera info.
    
    Args:
        camera_info: Dictionary with camera_location and camera_rotation_final (euler angles)
    
    Returns:
        4x4 transformation matrix
    """
    # Camera position in world coordinates
    position = np.array(camera_info['camera_location'])  # [x, y, z]
    
    # Camera rotation (euler angles in radians)
    rotation_euler = np.array(camera_info['camera_rotation_final'])  # [pitch, yaw, roll]
    
    # Convert Euler angles to rotation matrix
    rotation_matrix = Rotation.from_euler('xyz', rotation_euler).as_matrix()
    
    # Build 4x4 transformation matrix [R | t]
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = position
    
    return extrinsics.tolist()


def depth_to_point_cloud(depth_map: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Convert depth map to 3D point cloud in camera coordinates.
    
    Args:
        depth_map: HxW depth map in meters
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        Nx3 array of 3D points
    """
    h, w = depth_map.shape
    
    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Flatten arrays
    u = u.flatten()
    v = v.flatten()
    depth = depth_map.flatten()
    
    # Filter out invalid depth values
    valid_mask = (depth > 0) & (depth < 100)  # reasonable depth range
    u = u[valid_mask]
    v = v[valid_mask]
    depth = depth[valid_mask]
    
    # Unproject to 3D
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # Stack into Nx3 point cloud
    points_3d = np.stack([x, y, z], axis=1)
    
    return points_3d


def compute_3d_bbox_from_instance_mask(
    instance_mask: np.ndarray,
    depth_map: np.ndarray,
    instance_id: int,
    fx: float, fy: float, cx: float, cy: float
) -> Optional[Dict]:
    """
    Compute 3D bounding box from instance mask and depth map.
    
    Args:
        instance_mask: HxW instance segmentation mask
        depth_map: HxW depth map in meters
        instance_id: ID of the instance to extract
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        Dictionary with 3D bbox center, dimensions, and rotation (identity)
    """
    # Extract pixels belonging to this instance
    instance_pixels = (instance_mask == instance_id)
    
    if not np.any(instance_pixels):
        return None
    
    # Get depth values for this instance
    instance_depth = depth_map[instance_pixels]
    
    # Filter out invalid depth
    valid_depth = instance_depth[(instance_depth > 0) & (instance_depth < 100)]
    
    if len(valid_depth) < 10:  # Need minimum points for reasonable bbox
        return None
    
    # Get pixel coordinates of instance
    v_coords, u_coords = np.where(instance_pixels)
    depths = depth_map[v_coords, u_coords]
    
    # Filter by valid depth
    valid_mask = (depths > 0) & (depths < 100)
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depths = depths[valid_mask]
    
    if len(depths) < 10:
        return None
    
    # Unproject to 3D points
    x = (u_coords - cx) * depths / fx
    y = (v_coords - cy) * depths / fy
    z = depths
    
    # Compute 3D bounding box (axis-aligned in camera coordinates)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    
    # Center and dimensions
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    dimensions = [x_max - x_min, y_max - y_min, z_max - z_min]
    
    # Check if bbox is reasonable (not too small or too large)
    if any(d < 0.05 or d > 50 for d in dimensions):
        return None
    
    return {
        'center': center,
        'dimensions': dimensions,
        'rotation': [0.0, 0.0, 0.0]  # Axis-aligned, no rotation
    }


def get_instance_category(instance_id: int, semantic_class_id: Optional[int] = None) -> str:
    """
    Get category name for instance.
    
    Args:
        instance_id: Instance ID from segment_unsup25d
        semantic_class_id: Optional semantic class from segment_semantic (COCO subset)
    
    Returns:
        Category string like "chair_5" or "object_5" if no semantic label
    """
    if semantic_class_id is not None and 0 <= semantic_class_id < len(SEMANTIC_CLASSES):
        semantic_name = SEMANTIC_CLASSES[semantic_class_id]
        # Skip uncertain and background - treat as object_N
        if semantic_name not in ["uncertain", "background"]:
            return f'{semantic_name}_{instance_id}'
    
    return f'object_{instance_id}'


def process_view(
    location_dir: Path,
    location_name: str,
    point_id: str,
    view_id: int
) -> Optional[Dict]:
    """Process a single view (RGB + depth + instance segmentation)."""
    
    # File paths
    rgb_file = location_dir / 'rgb' / 'taskonomy' / location_name / f'point_{point_id}_view_{view_id}_domain_rgb.png'
    depth_file = location_dir / 'depth_euclidean' / 'taskonomy' / location_name / f'point_{point_id}_view_{view_id}_domain_depth_euclidean.png'
    instance_file = location_dir / 'segment_unsup25d' / 'taskonomy' / location_name / f'point_{point_id}_view_{view_id}_domain_segment_unsup25d.png'
    semantic_file = location_dir / 'segment_semantic' / 'taskonomy' / location_name / f'point_{point_id}_view_{view_id}_domain_segmentsemantic.png'
    point_info_file = location_dir / 'point_info' / 'taskonomy' / location_name / f'point_{point_id}_view_{view_id}_domain_point_info.json'
    
    # Check if required files exist (semantic is optional)
    if not all([rgb_file.exists(), depth_file.exists(), instance_file.exists(), point_info_file.exists()]):
        return None
    
    try:
        # Load camera info
        camera_info = load_camera_info(point_info_file)
        if not camera_info:
            return None
        
        # Get camera intrinsics
        resolution = camera_info.get('resolution', 512)
        fov_rads = camera_info.get('field_of_view_rads', 1.0489)  # ~60 degrees default
        intrinsics = build_intrinsics_from_fov(resolution, fov_rads)
        
        # Build intrinsics matrix
        intrinsics_matrix = [
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ]
        
        # Build extrinsics matrix
        extrinsics_matrix = build_extrinsics_from_camera_info(camera_info)
        
        # Load RGB to get dimensions
        rgb_img = cv2.imread(str(rgb_file))
        if rgb_img is None:
            return None
        height, width = rgb_img.shape[:2]
        
        # Load depth map (Taskonomy depth is in PNG, encoded as 16-bit)
        # The depth is stored in a special format - need to decode
        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            return None
        
        # Taskonomy depth encoding: convert to meters
        # The exact encoding varies, but typically it's scaled
        # For now, normalize to reasonable range (will need adjustment based on actual data)
        depth_map = depth_img.astype(np.float32) / 1000.0  # Adjust scale if needed
        
        # Compute depth statistics
        depth_stats = compute_depth_stats(depth_map)
        
        # Load instance segmentation
        instance_mask = cv2.imread(str(instance_file), cv2.IMREAD_GRAYSCALE)
        if instance_mask is None:
            return None
        
        # Load semantic segmentation (optional - may not be available for all views)
        semantic_mask = None
        semantic_to_instance_map = {}
        if semantic_file.exists():
            semantic_mask = cv2.imread(str(semantic_file), cv2.IMREAD_GRAYSCALE)
            if semantic_mask is not None:
                # Resize semantic mask to match instance mask resolution
                if semantic_mask.shape != instance_mask.shape:
                    semantic_mask = cv2.resize(semantic_mask, 
                                             (instance_mask.shape[1], instance_mask.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
                
                # Map each instance to its dominant semantic class
                unique_instances_temp = np.unique(instance_mask)
                for inst_id in unique_instances_temp:
                    if inst_id > 0:  # Skip background
                        inst_pixels = semantic_mask[instance_mask == inst_id]
                        if len(inst_pixels) > 0:
                            # Get dominant semantic class for this instance
                            semantic_class = np.bincount(inst_pixels).argmax()
                            semantic_to_instance_map[int(inst_id)] = int(semantic_class)
        
        # Get unique instances
        unique_instances = np.unique(instance_mask)
        # Filter out background (typically 0)
        unique_instances = unique_instances[unique_instances > 0]
        
        # Derive 3D bounding boxes for each instance
        bboxes_3d = []
        for instance_id in unique_instances:
            bbox = compute_3d_bbox_from_instance_mask(
                instance_mask, depth_map, int(instance_id),
                intrinsics['fx'], intrinsics['fy'],
                intrinsics['cx'], intrinsics['cy']
            )
            
            if bbox:
                # Convert to 9-DoF format
                bbox_9dof = convert_bbox_to_9dof(
                    center=bbox['center'],
                    dimensions=bbox['dimensions'],
                    rotation=bbox['rotation'],
                    rotation_format='euler'
                )
                
                # Add category (with semantic label if available)
                semantic_class_id = semantic_to_instance_map.get(int(instance_id))
                bbox_9dof['category'] = get_instance_category(int(instance_id), semantic_class_id)
                bboxes_3d.append(bbox_9dof)
        
        # Build relative paths
        rgb_rel_path = f'{location_name}/point_{point_id}_view_{view_id}_domain_rgb.png'
        depth_rel_path = f'{location_name}/point_{point_id}_view_{view_id}_domain_depth_euclidean.png'
        
        # Create unified JSON
        unified_json = {
            "dataset": "taskonomy",
            "split": location_name,
            "image_id": f'{location_name}_point_{point_id}_view_{view_id}',
            "filename": f'point_{point_id}_view_{view_id}_domain_rgb.png',
            "rgb_path": rgb_rel_path,
            "depth_path": depth_rel_path,
            "depth_type": "depth_png_encoded",  # PNG format with special encoding
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
            "bounding_boxes_2d": [],  # Not provided by Taskonomy
            "bounding_boxes_3d": bboxes_3d
        }
        
        return unified_json
    
    except Exception as e:
        logger.error(f"Failed to process {location_name} point {point_id} view {view_id}: {e}")
        return None


def process_location(location_dir: Path, location_name: str) -> Tuple[int, int]:
    """
    Process all views in a Taskonomy location.
    
    Returns:
        Tuple of (num_views_processed, num_bboxes)
    """
    logger.info(f"Processing location: {location_name}")
    
    # Find all RGB files to determine available views
    rgb_dir = location_dir / 'rgb' / 'taskonomy' / location_name
    if not rgb_dir.exists():
        logger.warning(f"Skipping {location_name}: RGB directory not found")
        return 0, 0
    
    # Parse filenames to get point IDs and view IDs
    rgb_files = list(rgb_dir.glob('point_*_view_*_domain_rgb.png'))
    
    if not rgb_files:
        logger.warning(f"Skipping {location_name}: No RGB files found")
        return 0, 0
    
    logger.info(f"  Found {len(rgb_files)} views")
    
    views_processed = 0
    total_bboxes = 0
    
    # Process each view
    for rgb_file in rgb_files:
        # Parse filename: point_X_view_Y_domain_rgb.png
        parts = rgb_file.stem.split('_')
        point_id = parts[1]  # X
        view_id = int(parts[3])  # Y
        
        result = process_view(location_dir, location_name, point_id, view_id)
        
        if result:
            # Save to output directory
            output_subdir = Path(__file__).parent.parent / 'processed_data' / 'taskonomy' / location_name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_subdir / f'point_{point_id}_view_{view_id}.json'
            save_json(result, output_file)
            
            views_processed += 1
            total_bboxes += len(result.get('bounding_boxes_3d', []))
            
            if views_processed % 100 == 0:
                logger.info(f"    Processed {views_processed} views...")
    
    logger.info(f"  Finished {location_name}: {views_processed} views, {total_bboxes} derived bboxes")
    return views_processed, total_bboxes


class TaskonomyProcessor:
    """Taskonomy Dataset Processor"""
    
    def __init__(self, raw_data_dir: Path, output_dir: Path, locations: List[str] = None):
        """
        Initialize Taskonomy processor.
        
        Args:
            raw_data_dir: Path to raw_data/taskonomy_dataset
            output_dir: Path to processed_data/taskonomy
            locations: List of location names to process (default: all)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.locations = locations
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all(self) -> Dict:
        """Process all Taskonomy locations."""
        logger.info("Processing Taskonomy dataset...")
        
        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory not found: {self.raw_data_dir}")
            return {'error': 'Raw data directory not found'}
        
        # Find available locations
        rgb_root = self.raw_data_dir / 'rgb' / 'taskonomy'
        if not rgb_root.exists():
            logger.error(f"RGB directory not found: {rgb_root}")
            return {'error': 'RGB directory not found'}
        
        available_locations = [d.name for d in rgb_root.iterdir() if d.is_dir()]
        
        if self.locations:
            # Process only specified locations
            locations_to_process = [loc for loc in self.locations if loc in available_locations]
        else:
            # Process all available locations
            locations_to_process = available_locations
        
        logger.info(f"Found {len(available_locations)} locations, processing {len(locations_to_process)}")
        
        total_views = 0
        total_bboxes = 0
        processed_locations = []
        
        for location_name in sorted(locations_to_process):
            views, bboxes = process_location(self.raw_data_dir, location_name)
            
            if views > 0:
                total_views += views
                total_bboxes += bboxes
                processed_locations.append({
                    'location_name': location_name,
                    'num_views': views,
                    'num_objects': bboxes
                })
        
        # Create summary
        summary = {
            'dataset': 'taskonomy',
            'total_locations': len(processed_locations),
            'total_views': total_views,
            'total_3d_bboxes': total_bboxes,
            'note': '3D bboxes derived from depth + instance segmentation',
            'locations': processed_locations
        }
        
        summary_file = self.output_dir / 'summary.json'
        save_json(summary, summary_file)
        logger.info(f"Saved summary to {summary_file}")
        
        logger.info(f"✅ Taskonomy processing complete!")
        logger.info(f"   Total locations: {len(processed_locations)}")
        logger.info(f"   Total views: {total_views}")
        logger.info(f"   Total derived 3D bboxes: {total_bboxes}")
        
        return summary


def main():
    """Main entry point."""
    # Use relative paths
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir.parent / 'raw_data' / 'taskonomy_dataset'
    output_dir = script_dir.parent / 'processed_data' / 'taskonomy'
    
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        logger.info("Please ensure the Taskonomy dataset is downloaded to raw_data/taskonomy_dataset/")
        return
    
    # For testing, process just a few locations (you can remove this limit later)
    test_locations = ['ackermanville', 'adairsville', 'adrian']  # Process 3 locations for testing
    
    # Create processor and run
    processor = TaskonomyProcessor(raw_data_dir, output_dir, locations=test_locations)
    summary = processor.process_all()
    
    if 'error' not in summary:
        print(f"\n{'='*60}")
        print(f"Taskonomy Processing Summary")
        print(f"{'='*60}")
        print(f"Total locations processed: {summary['total_locations']}")
        print(f"Total views processed: {summary['total_views']}")
        print(f"Total derived 3D bboxes: {summary['total_3d_bboxes']}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        print(f"Note: 3D bboxes derived from depth maps + instance segmentation")


if __name__ == '__main__':
    main()
