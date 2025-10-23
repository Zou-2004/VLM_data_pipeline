"""
3D geometry utilities for spatial calculations
Based on VLM-3R methodology
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def get_bbox_center(bbox: Dict) -> np.ndarray:
    """Get 3D bbox center coordinates"""
    return np.array([bbox['x'], bbox['y'], bbox['z']])


def get_bbox_dimensions(bbox: Dict) -> np.ndarray:
    """Get 3D bbox dimensions (length, width, height)"""
    return np.array([bbox['xl'], bbox['yl'], bbox['zl']])


def get_bbox_rotation(bbox: Dict) -> Tuple[float, float, float]:
    """Get bbox rotation (pitch, yaw, roll)"""
    return (bbox.get('pitch', 0.0), bbox.get('yaw', 0.0), bbox.get('roll', 0.0))


def rotation_matrix_from_angles(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Create rotation matrix from Euler angles
    
    Args:
        pitch, yaw, roll: Rotation angles in radians
        
    Returns:
        3x3 rotation matrix
    """
    # Rotation around X-axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Rotation around Y-axis (yaw)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # Rotation around Z-axis (roll)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def get_bbox_vertices(bbox: Dict) -> np.ndarray:
    """
    Get 8 vertices of oriented 3D bounding box
    
    Args:
        bbox: Bounding box dictionary
        
    Returns:
        8x3 array of vertex coordinates
    """
    center = get_bbox_center(bbox)
    dims = get_bbox_dimensions(bbox)
    pitch, yaw, roll = get_bbox_rotation(bbox)
    
    # Half dimensions
    dx, dy, dz = dims / 2
    
    # Local vertices (before rotation)
    local_vertices = np.array([
        [-dx, -dy, -dz],
        [dx, -dy, -dz],
        [dx, dy, -dz],
        [-dx, dy, -dz],
        [-dx, -dy, dz],
        [dx, -dy, dz],
        [dx, dy, dz],
        [-dx, dy, dz]
    ])
    
    # Rotation matrix
    R = rotation_matrix_from_angles(pitch, yaw, roll)
    
    # Rotate and translate vertices
    rotated_vertices = (R @ local_vertices.T).T + center
    
    return rotated_vertices


def min_distance_between_bboxes(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate minimum Euclidean distance between two 3D bounding boxes
    
    Args:
        bbox1, bbox2: Bounding box dictionaries
        
    Returns:
        Minimum distance in meters
    """
    vertices1 = get_bbox_vertices(bbox1)
    vertices2 = get_bbox_vertices(bbox2)
    
    # Calculate all pairwise distances
    min_dist = float('inf')
    for v1 in vertices1:
        for v2 in vertices2:
            dist = np.linalg.norm(v1 - v2)
            min_dist = min(min_dist, dist)
    
    return min_dist


def get_max_dimension(bbox: Dict) -> float:
    """
    Get maximum dimension of 3D bounding box
    
    Args:
        bbox: Bounding box dictionary
        
    Returns:
        Maximum dimension in meters
    """
    dims = get_bbox_dimensions(bbox)
    return np.max(dims)


def get_camera_position(camera_data: Dict) -> Optional[np.ndarray]:
    """
    Extract camera position from extrinsics matrix
    
    Args:
        camera_data: Camera dictionary with extrinsics
        
    Returns:
        3D camera position or None if not available
    """
    if 'extrinsics' not in camera_data or camera_data['extrinsics'] is None:
        return None
    
    extrinsics = np.array(camera_data['extrinsics'])
    
    # Camera position is in the last column of extrinsics matrix
    # But we need to handle both camera-to-world and world-to-camera formats
    # Typically extrinsics is [R | t] where t is translation
    camera_pos = extrinsics[:3, 3]
    
    return camera_pos


def distance_camera_to_bbox(camera_pos: np.ndarray, bbox: Dict) -> float:
    """
    Calculate minimum distance from camera to 3D bounding box
    
    Args:
        camera_pos: 3D camera position
        bbox: Bounding box dictionary
        
    Returns:
        Minimum distance in meters
    """
    vertices = get_bbox_vertices(bbox)
    
    # Find minimum distance from camera to any vertex
    min_dist = float('inf')
    for vertex in vertices:
        dist = np.linalg.norm(camera_pos - vertex)
        min_dist = min(min_dist, dist)
    
    return min_dist


def transform_bbox_to_camera_frame(bbox: Dict, camera_extrinsics: np.ndarray) -> np.ndarray:
    """
    Transform bounding box vertices to camera coordinate system
    
    Args:
        bbox: Bounding box dictionary
        camera_extrinsics: 4x4 camera extrinsics matrix (camera-to-world)
        
    Returns:
        8x3 array of vertices in camera coordinates
    """
    # Get vertices in world coordinates
    world_vertices = get_bbox_vertices(bbox)
    
    # Convert to homogeneous coordinates
    world_vertices_h = np.hstack([world_vertices, np.ones((8, 1))])
    
    # Invert camera-to-world to get world-to-camera
    camera_extrinsics = np.array(camera_extrinsics)
    world_to_camera = np.linalg.inv(camera_extrinsics)
    
    # Transform vertices
    camera_vertices_h = (world_to_camera @ world_vertices_h.T).T
    
    # Remove homogeneous coordinate
    camera_vertices = camera_vertices_h[:, :3]
    
    return camera_vertices


def get_relative_position_2d(bbox1: Dict, bbox2: Dict, camera_data: Dict) -> Optional[Tuple[str, str, str]]:
    """
    Determine relative position of bbox1 w.r.t. bbox2 in camera frame
    Returns whether bbox1 is Near/Far, Left/Right, Up/Down relative to bbox2
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box  
        camera_data: Camera data with extrinsics
        
    Returns:
        Tuple of (depth_relation, horizontal_relation, vertical_relation)
        Each can be a string value or None
    """
    if 'extrinsics' not in camera_data or camera_data['extrinsics'] is None:
        return None
    
    extrinsics = np.array(camera_data['extrinsics'])
    
    # Transform both bboxes to camera frame
    vertices1 = transform_bbox_to_camera_frame(bbox1, extrinsics)
    vertices2 = transform_bbox_to_camera_frame(bbox2, extrinsics)
    
    # In camera frame: +X is right, +Y is down, +Z is forward (away from camera)
    threshold = 0.1  # meters
    
    # Near/Far: Compare Z coordinates (depth)
    depth_rel = None
    if np.max(vertices1[:, 2]) < np.min(vertices2[:, 2]) - threshold:
        depth_rel = 'Near'
    elif np.min(vertices1[:, 2]) > np.max(vertices2[:, 2]) + threshold:
        depth_rel = 'Far'
    
    # Left/Right: Compare X coordinates
    horizontal_rel = None
    if np.max(vertices1[:, 0]) < np.min(vertices2[:, 0]) - threshold:
        horizontal_rel = 'Left'
    elif np.min(vertices1[:, 0]) > np.max(vertices2[:, 0]) + threshold:
        horizontal_rel = 'Right'
    
    # Up/Down: Compare Y coordinates (+Y is down)
    vertical_rel = None
    if np.max(vertices1[:, 1]) < np.min(vertices2[:, 1]) - threshold:
        vertical_rel = 'Up'
    elif np.min(vertices1[:, 1]) > np.max(vertices2[:, 1]) + threshold:
        vertical_rel = 'Down'
    
    return (depth_rel, horizontal_rel, vertical_rel)
