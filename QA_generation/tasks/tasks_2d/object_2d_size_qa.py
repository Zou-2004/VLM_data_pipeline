"""
2D Object Size QA Generation for COCO dataset
Generates questions about 2D object dimensions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from utils.qa_base import BaseQAGenerator
from utils.geometry import get_bbox_2d_params, get_2d_bbox_area
from config import QA_PARAMS


class Object2DSizeQA(BaseQAGenerator):
    """Generate 2D object size questions"""
    
    def __init__(self, dataset_name: str):
        super().__init__('obj_2d_size', dataset_name)
        self.params = QA_PARAMS.get('obj_2d_size', {
            'min_area': 100,  # minimum area in pixels
            'decimal_places': 1
        })
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate 2D object size questions"""
        qa_pairs = []
        
        for item in data:
            if 'bounding_boxes_2d' not in item:
                continue
            
            # Skip if no 2D bounding boxes
            if not item['bounding_boxes_2d']:
                continue
            
            # Track categories to avoid duplicates
            asked_categories = set()
            
            for bbox in item['bounding_boxes_2d']:
                category = bbox.get('category', 'unknown')
                
                if category in asked_categories:
                    continue
                
                asked_categories.add(category)
                
                qa_pair = self._generate_size_question(item, bbox, category)
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_size_question(self, item: Dict[str, Any], bbox: Dict[str, Any], 
                               category: str) -> Dict[str, Any]:
        """Generate a single 2D object size question"""
        
        # Get bbox parameters
        bbox_params = get_bbox_2d_params(bbox)
        if bbox_params is None:
            return None
        
        x, y, width, height = bbox_params
        
        # Calculate area
        area = get_2d_bbox_area(bbox)
        if area is None:
            area = width * height
        
        # Skip if too small
        if area < self.params['min_area']:
            return None
        
        # Create different types of size questions
        import random
        question_type = random.choice(['width', 'height', 'area'])
        
        if question_type == 'width':
            question = f"What is the width of the {category} bounding box in pixels?"
            answer = round(width, self.params['decimal_places'])
            unit = 'pixels'
        elif question_type == 'height':
            question = f"What is the height of the {category} bounding box in pixels?"
            answer = round(height, self.params['decimal_places'])
            unit = 'pixels'
        else:  # area
            question = f"What is the area of the {category} bounding box in pixels?"
            answer = round(area, self.params['decimal_places'])
            unit = 'square_pixels'
        
        # Create QA pair
        qa_pair = self.create_qa_pair(
            question=question,
            answer=answer,
            answer_type='numerical',
            metadata={
                'source_file': item.get('_source_file', ''),
                'image_id': item.get('image_id', ''),
                'scene_id': item.get('scene_id', ''),
                'frame_id': item.get('frame_id', ''),
                'category': category,
                'question_type': question_type,
                'bbox_width': width,
                'bbox_height': height,
                'bbox_area': area,
                'unit': unit
            }
        )
        
        return qa_pair