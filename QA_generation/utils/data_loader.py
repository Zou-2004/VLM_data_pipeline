"""
Data loader utilities for processed datasets
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a single JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_dataset_files(dataset_path: str, pattern: str = "**/*.json", limit: int = None) -> List[Dict[str, Any]]:
    """
    Load all JSON files from a dataset directory
    
    Args:
        dataset_path: Path to dataset directory
        pattern: Glob pattern for finding files
        limit: Maximum number of files to load (None = all files)
        
    Returns:
        List of loaded JSON data
    """
    dataset_path = Path(dataset_path)
    json_files = list(dataset_path.glob(pattern))
    
    # Filter out summary.json files
    json_files = [f for f in json_files if f.name != 'summary.json']
    
    # Limit if specified
    if limit is not None and limit > 0:
        json_files = json_files[:limit]
    
    data = []
    print(f"Loading {len(json_files)} files from {dataset_path.name}...")
    
    for json_file in tqdm(json_files, desc="Loading files"):
        try:
            file_data = load_json_file(str(json_file))
            # Add file path info
            file_data['_source_file'] = str(json_file)
            data.append(file_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return data


def group_by_scene(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group frames by scene_id or video_id for sequence-based tasks
    
    Args:
        data: List of frame data
        
    Returns:
        Dictionary mapping scene/video IDs to frames
    """
    scenes = {}
    
    for item in data:
        # Try different ID fields
        scene_id = item.get('scene_id') or item.get('video_id') or item.get('image_id')
        
        if scene_id not in scenes:
            scenes[scene_id] = []
        scenes[scene_id].append(item)
    
    # Sort frames within each scene if they have frame_id or similar
    for scene_id in scenes:
        frames = scenes[scene_id]
        if len(frames) > 1 and 'frame_id' in frames[0]:
            try:
                frames.sort(key=lambda x: int(x['frame_id']))
            except:
                pass  # If frame_id is not numeric, keep original order
    
    return scenes


def filter_by_bbox_availability(data: List[Dict[str, Any]], bbox_type: str = '3d') -> List[Dict[str, Any]]:
    """
    Filter data by availability of bounding boxes
    
    Args:
        data: List of data items
        bbox_type: '2d' or '3d'
        
    Returns:
        Filtered list with items that have the specified bbox type
    """
    bbox_key = f'bounding_boxes_{bbox_type}'
    filtered = []
    
    for item in data:
        if bbox_key in item and len(item[bbox_key]) > 0:
            filtered.append(item)
    
    return filtered


def get_category_counts(data: List[Dict[str, Any]], bbox_type: str = '3d') -> Dict[str, int]:
    """
    Count occurrences of each object category across dataset
    
    Args:
        data: List of data items
        bbox_type: '2d' or '3d'
        
    Returns:
        Dictionary mapping category names to counts
    """
    bbox_key = f'bounding_boxes_{bbox_type}'
    counts = {}
    
    for item in data:
        if bbox_key in item:
            for bbox in item[bbox_key]:
                category = bbox.get('category', 'unknown')
                counts[category] = counts.get(category, 0) + 1
    
    return counts


def get_frame_category_counts(frame_data: Dict[str, Any], bbox_type: str = '3d') -> Dict[str, int]:
    """
    Count occurrences of each object category in a single frame
    
    Args:
        frame_data: Single frame/image data
        bbox_type: '2d' or '3d'
        
    Returns:
        Dictionary mapping category names to counts in this frame
    """
    bbox_key = f'bounding_boxes_{bbox_type}'
    counts = {}
    
    if bbox_key in frame_data:
        for bbox in frame_data[bbox_key]:
            category = bbox.get('category', 'unknown')
            counts[category] = counts.get(category, 0) + 1
    
    return counts
