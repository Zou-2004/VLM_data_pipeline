"""
Common utilities for data processing.
Includes coordinate transformation, bbox conversion, and unified JSON output.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial.transform import Rotation


def quaternion_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion (w, x, y, z) to Euler angles (pitch, yaw, roll) in degrees.
    
    Args:
        quat: Quaternion as [w, x, y, z] or [x, y, z, w]
    
    Returns:
        pitch, yaw, roll in degrees
    """
    r = Rotation.from_quat(quat)
    pitch, yaw, roll = r.as_euler('xyz', degrees=True)
    return pitch, yaw, roll


def normalize_angle(angle_deg: float) -> float:
    """
    Normalize angle to [-180, 180] and divide by 180.
    
    Args:
        angle_deg: Angle in degrees
    
    Returns:
        Normalized angle in range [-1, 1]
    """
    # Normalize to [-180, 180]
    angle = angle_deg % 360
    if angle > 180:
        angle -= 360
    # Divide by 180 to get [-1, 1]
    return angle / 180.0


def convert_bbox_to_9dof(
    center: np.ndarray,
    dimensions: np.ndarray,
    rotation: Optional[np.ndarray] = None,
    rotation_format: str = "quaternion"
) -> Dict[str, Any]:
    """
    Convert 3D bounding box to 9-DoF format (x, y, z, xl, yl, zl, pitch, yaw, roll).
    
    Args:
        center: [x, y, z] center position in camera frame
        dimensions: [xl, yl, zl] dimensions
        rotation: Rotation as quaternion [w,x,y,z] or euler angles [pitch,yaw,roll]
        rotation_format: "quaternion", "euler", or "yaw_only"
    
    Returns:
        Dict with 9-DoF parameters
    """
    bbox = {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "xl": float(dimensions[0]),
        "yl": float(dimensions[1]),
        "zl": float(dimensions[2]),
    }
    
    if rotation is None or rotation_format == "axis_aligned":
        # No rotation
        pitch, yaw, roll = 0.0, 0.0, 0.0
    elif rotation_format == "quaternion":
        pitch, yaw, roll = quaternion_to_euler(rotation)
    elif rotation_format == "euler":
        pitch, yaw, roll = rotation[0], rotation[1], rotation[2]
    elif rotation_format == "yaw_only":
        pitch, yaw, roll = 0.0, rotation[0], 0.0
    else:
        raise ValueError(f"Unknown rotation format: {rotation_format}")
    
    # Normalize angles to [-1, 1]
    bbox["pitch"] = normalize_angle(pitch)
    bbox["yaw"] = normalize_angle(yaw)
    bbox["roll"] = normalize_angle(roll)
    
    return bbox


def compute_depth_stats(depth_map: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute statistics for depth map.
    
    Args:
        depth_map: Depth values (H, W)
        valid_mask: Boolean mask of valid pixels
    
    Returns:
        Dict with min, max, median, mean depth and validity info
    """
    if valid_mask is None:
        valid_mask = (depth_map > 0) & np.isfinite(depth_map)
    
    valid_depth = depth_map[valid_mask]
    
    if len(valid_depth) == 0:
        return {
            "present": False,
            "valid_pixels": 0,
            "min": None,
            "max": None,
            "median": None,
            "mean": None
        }
    
    return {
        "present": True,
        "valid_pixels": int(np.sum(valid_mask)),
        "total_pixels": int(depth_map.size),
        "min": float(np.min(valid_depth)),
        "max": float(np.max(valid_depth)),
        "median": float(np.median(valid_depth)),
        "mean": float(np.mean(valid_depth))
    }


def create_unified_json(
    image_path: str,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: Optional[np.ndarray],
    depth_map: Optional[np.ndarray],
    depth_type: str,
    instances_3d: List[Dict[str, Any]],
    dataset_name: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create unified JSON format for processed data.
    
    Args:
        image_path: Relative path to RGB image
        camera_intrinsics: 3x3 intrinsics matrix
        camera_extrinsics: 4x4 extrinsics matrix (world to camera)
        depth_map: Depth array or None
        depth_type: "metric", "pseudo", or "none"
        instances_3d: List of 9-DoF bounding boxes with categories
        dataset_name: Name of source dataset
        additional_info: Any additional metadata
    
    Returns:
        Unified JSON dict
    """
    output = {
        "dataset": dataset_name,
        "image": str(image_path),
        "camera": {
            "intrinsics": camera_intrinsics.tolist() if camera_intrinsics is not None else None,
            "extrinsics": camera_extrinsics.tolist() if camera_extrinsics is not None else None,
        },
        "depth": {
            "type": depth_type,
            "present": depth_map is not None,
        },
        "instances_3d": instances_3d
    }
    
    # Add depth statistics
    if depth_map is not None:
        output["depth"].update(compute_depth_stats(depth_map))
    
    # Add additional info
    if additional_info:
        output.update(additional_info)
    
    return output


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """Save data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(json_path: Path) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def world_to_camera_frame(
    bbox_world: Dict[str, Any],
    world_to_cam: np.ndarray
) -> Dict[str, Any]:
    """
    Transform bounding box from world to camera frame.
    
    Args:
        bbox_world: 9-DoF bbox in world coordinates
        world_to_cam: 4x4 transformation matrix
    
    Returns:
        9-DoF bbox in camera coordinates
    """
    # Extract center and transform
    center_world = np.array([bbox_world["x"], bbox_world["y"], bbox_world["z"], 1.0])
    center_cam = world_to_cam @ center_world
    
    # Create new bbox with transformed center
    bbox_cam = bbox_world.copy()
    bbox_cam["x"] = float(center_cam[0])
    bbox_cam["y"] = float(center_cam[1])
    bbox_cam["z"] = float(center_cam[2])
    
    # Rotation needs to be transformed too (simplified - assumes camera-aligned)
    # For full implementation, would need to transform rotation matrix
    
    return bbox_cam
