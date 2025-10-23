"""
3D geometry utilities for spatial calculations
Based on VLM-3R methodology
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


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
    
    try:
        extrinsics = np.array(camera_data['extrinsics'])
        
        # Ensure we have a 4x4 matrix
        if extrinsics.shape != (4, 4):
            return None
        
        # Camera position from extrinsics matrix
        # For camera-to-world transform: camera_pos = extrinsics[:3, 3]
        # For world-to-camera transform: camera_pos = -R.T @ t
        camera_pos = extrinsics[:3, 3]
        
        return camera_pos
    except Exception:
        return None


def distance_camera_to_bbox(camera_pos: np.ndarray, bbox: Dict) -> float:
    """
    Calculate minimum distance from camera to 3D bounding box
    
    Args:
        camera_pos: 3D camera position
        bbox: Bounding box dictionary
        
    Returns:
        Minimum distance in meters
    """
    try:
        vertices = get_bbox_vertices(bbox)
        
        # Find minimum distance from camera to any vertex
        min_dist = float('inf')
        for vertex in vertices:
            dist = np.linalg.norm(camera_pos - vertex)
            min_dist = min(min_dist, dist)
        
        return min_dist
    except Exception:
        # Fallback: use bbox center
        center = get_bbox_center(bbox)
        return np.linalg.norm(camera_pos - center)


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


def get_bbox_2d_params(bbox: Dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract 2D bbox parameters from various formats
    
    Args:
        bbox: 2D bounding box dictionary
        
    Returns:
        Tuple of (x, y, width, height) or None if not found
    """
    # Format 1: Direct x, y, w, h
    if all(k in bbox for k in ['x', 'y', 'w', 'h']):
        return (bbox['x'], bbox['y'], bbox['w'], bbox['h'])
    
    # Format 2: bbox_2d sub-dictionary
    if 'bbox_2d' in bbox:
        bbox_2d = bbox['bbox_2d']
        if all(k in bbox_2d for k in ['x', 'y', 'width', 'height']):
            return (bbox_2d['x'], bbox_2d['y'], bbox_2d['width'], bbox_2d['height'])
    
    # Format 3: COCO format [x, y, width, height] list
    if 'bbox' in bbox and isinstance(bbox['bbox'], list) and len(bbox['bbox']) == 4:
        return tuple(bbox['bbox'])
    
    return None


def get_2d_bbox_center(bbox: Dict) -> Optional[Tuple[float, float]]:
    """
    Get 2D bbox center coordinates
    
    Args:
        bbox: 2D bounding box dictionary
        
    Returns:
        (cx, cy) tuple or None if bbox parameters not found
    """
    params = get_bbox_2d_params(bbox)
    if params is None:
        return None
    
    x, y, w, h = params
    return (x + w/2, y + h/2)


def get_2d_bbox_area(bbox: Dict) -> Optional[float]:
    """
    Get 2D bbox area
    
    Args:
        bbox: 2D bounding box dictionary
        
    Returns:
        Area in pixels or None if bbox parameters not found
    """
    params = get_bbox_2d_params(bbox)
    if params is None:
        # Check if area is directly provided
        if 'area' in bbox:
            return bbox['area']
        return None
    
    x, y, w, h = params
    return w * h


def transform_to_world_coordinates(bbox: Dict, extrinsics: np.ndarray) -> Optional[np.ndarray]:
    """
    Transform 3D bbox from camera coordinates to world coordinates
    
    Args:
        bbox: 3D bounding box in camera coordinates
        extrinsics: 4x4 camera extrinsics matrix (camera-to-world)
        
    Returns:
        8x3 array of vertices in world coordinates or None if failed
    """
    try:
        # Get vertices in camera coordinates
        camera_vertices = get_bbox_vertices(bbox)
        
        # Convert to homogeneous coordinates
        camera_vertices_h = np.hstack([camera_vertices, np.ones((8, 1))])
        
        # Transform to world coordinates
        world_vertices_h = (extrinsics @ camera_vertices_h.T).T
        
        # Remove homogeneous coordinate
        world_vertices = world_vertices_h[:, :3]
        
        return world_vertices
    except Exception:
        return None


def get_camera_orientation(extrinsics: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract camera orientation vectors from extrinsics matrix
    
    Args:
        extrinsics: 4x4 camera extrinsics matrix
        
    Returns:
        Dictionary with 'forward', 'right', 'up' vectors or None if failed
    """
    try:
        if extrinsics.shape != (4, 4):
            return None
        
        # Rotation matrix is top-left 3x3
        R = extrinsics[:3, :3]
        
        # Camera coordinate system (OpenCV convention)
        # X: right, Y: down, Z: forward (into scene)
        right = R[:, 0]     # +X direction in world
        down = R[:, 1]      # +Y direction in world  
        forward = R[:, 2]   # +Z direction in world
        
        return {
            'right': right,
            'down': down,
            'up': -down,        # Up is negative down
            'forward': forward,
            'back': -forward
        }
    except Exception:
        return None


def improved_distance_camera_to_bbox(camera_data: Dict, bbox: Dict) -> Optional[float]:
    """
    Improved camera-to-bbox distance calculation using extrinsics when available
    
    Args:
        camera_data: Camera data dictionary
        bbox: 3D bounding box dictionary
        
    Returns:
        Distance in meters or None if calculation failed
    """
    # Method 1: Use extrinsics for accurate camera position
    camera_pos = get_camera_position(camera_data)
    if camera_pos is not None:
        return distance_camera_to_bbox(camera_pos, bbox)
    
    # Method 2: Fallback to bbox center distance (camera at origin)
    try:
        center = get_bbox_center(bbox)
        # Camera is at origin in camera coordinates
        return np.linalg.norm(center)
    except Exception:
        return None


def enhanced_relative_position(bbox1: Dict, bbox2: Dict, camera_data: Dict) -> Optional[Dict[str, Any]]:
    """
    Enhanced relative position calculation with more detailed spatial relationships
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        camera_data: Camera data with extrinsics
        
    Returns:
        Dictionary with detailed spatial relationships or None if failed
    """
    if 'extrinsics' not in camera_data or camera_data['extrinsics'] is None:
        return None
    
    try:
        extrinsics = np.array(camera_data['extrinsics'])
        
        # Transform both bboxes to camera frame for consistent analysis
        vertices1 = transform_bbox_to_camera_frame(bbox1, extrinsics)
        vertices2 = transform_bbox_to_camera_frame(bbox2, extrinsics)
        
        # Calculate centers and sizes
        center1 = np.mean(vertices1, axis=0)
        center2 = np.mean(vertices2, axis=0)
        
        # Distance calculations
        center_distance = np.linalg.norm(center1 - center2)
        min_distance = min_distance_between_bboxes(bbox1, bbox2)
        
        # Relative position in camera frame
        threshold = 0.1  # meters
        
        # Depth relationship (Z-axis: positive is away from camera)
        depth_diff = center1[2] - center2[2]
        if abs(depth_diff) < threshold:
            depth_rel = "Same depth"
        elif depth_diff < 0:
            depth_rel = "Nearer"
        else:
            depth_rel = "Farther"
        
        # Horizontal relationship (X-axis: positive is right)
        horizontal_diff = center1[0] - center2[0]
        if abs(horizontal_diff) < threshold:
            horizontal_rel = "Same horizontal position"
        elif horizontal_diff < 0:
            horizontal_rel = "Left"
        else:
            horizontal_rel = "Right"
        
        # Vertical relationship (Y-axis: positive is down)
        vertical_diff = center1[1] - center2[1]
        if abs(vertical_diff) < threshold:
            vertical_rel = "Same vertical position"
        elif vertical_diff < 0:
            vertical_rel = "Above"
        else:
            vertical_rel = "Below"
        
        return {
            'depth_relation': depth_rel,
            'horizontal_relation': horizontal_rel,
            'vertical_relation': vertical_rel,
            'center_distance': center_distance,
            'min_distance': min_distance,
            'depth_diff': depth_diff,
            'horizontal_diff': horizontal_diff,
            'vertical_diff': vertical_diff
        }
    except Exception:
        return None
