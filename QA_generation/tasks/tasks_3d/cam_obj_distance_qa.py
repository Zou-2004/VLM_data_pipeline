"""
Camera-Object Distance QA Generation
Generates questions about distance from camera to objects
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from utils.qa_base import BaseQAGenerator
from utils.geometry import get_camera_position, distance_camera_to_bbox
from config import TEMPLATE_CAM_OBJ_DISTANCE, QA_PARAMS


class CameraObjectDistanceQA(BaseQAGenerator):
    """Generate camera-to-object distance questions"""
    
    def __init__(self, dataset_name: str):
        super().__init__('cam_obj_distance', dataset_name)
        self.params = QA_PARAMS['cam_obj_distance']
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate camera-object distance questions"""
        qa_pairs = []
        
        for item in data:
            if 'bounding_boxes_3d' not in item or 'camera' not in item:
                continue
            
            # Get camera position
            camera_pos = get_camera_position(item['camera'])
            if camera_pos is None:
                continue
            
            # Track categories to avoid duplicates
            asked_categories = set()
            
            for bbox in item['bounding_boxes_3d']:
                category = bbox.get('category', 'unknown')
                
                if category in asked_categories:
                    continue
                
                asked_categories.add(category)
                
                qa_pair = self._generate_distance_question(item, bbox, category, camera_pos)
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_distance_question(self, item: Dict[str, Any], bbox: Dict[str, Any], 
                                   category: str, camera_pos) -> Dict[str, Any]:
        """Generate a single camera-object distance question"""
        
        # Calculate distance
        distance_m = distance_camera_to_bbox(camera_pos, bbox)
        
        # Skip if too close
        if distance_m < self.params['min_distance']:
            return None
        
        # Round to specified decimal places
        distance_rounded = round(distance_m, self.params['decimal_places'])
        
        # Create question
        question = TEMPLATE_CAM_OBJ_DISTANCE.format(category=category)
        
        # Create QA pair (numerical answer)
        qa_pair = self.create_qa_pair(
            question=question,
            answer=distance_rounded,
            answer_type='numerical',
            metadata={
                'source_file': item.get('_source_file', ''),
                'image_id': item.get('image_id', ''),
                'scene_id': item.get('scene_id', ''),
                'frame_id': item.get('frame_id', ''),
                'category': category,
                'distance_meters': distance_rounded,
                'unit': 'meters'
            }
        )
        
        return qa_pair
