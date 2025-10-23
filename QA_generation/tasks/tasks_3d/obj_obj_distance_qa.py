"""
Object-Object Distance QA Generation
Generates questions about distance between two objects
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from utils.qa_base import BaseQAGenerator
from utils.geometry import min_distance_between_bboxes
from config import TEMPLATE_OBJ_OBJ_DISTANCE, QA_PARAMS


class ObjectObjectDistanceQA(BaseQAGenerator):
    """Generate object-to-object distance questions"""
    
    def __init__(self, dataset_name: str):
        super().__init__('obj_obj_distance', dataset_name)
        self.params = QA_PARAMS['obj_obj_distance']
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate object-object distance questions"""
        qa_pairs = []
        
        for item in data:
            if 'bounding_boxes_3d' not in item:
                continue
            
            bboxes = item['bounding_boxes_3d']
            
            # Need at least 2 objects
            if len(bboxes) < 2:
                continue
            
            # Generate questions for pairs of objects
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    bbox1 = bboxes[i]
                    bbox2 = bboxes[j]
                    
                    qa_pair = self._generate_obj_obj_distance_question(
                        item, bbox1, bbox2
                    )
                    if qa_pair:
                        qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_obj_obj_distance_question(self, item: Dict[str, Any], 
                                           bbox1: Dict[str, Any], 
                                           bbox2: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single object-object distance question"""
        
        category1 = bbox1.get('category', 'unknown')
        category2 = bbox2.get('category', 'unknown')
        
        # Calculate minimum distance between bboxes
        distance_m = min_distance_between_bboxes(bbox1, bbox2)
        
        # Skip if too close or too far
        if distance_m < self.params['min_distance'] or distance_m > self.params['max_distance']:
            return None
        
        # Round to specified decimal places
        distance_rounded = round(distance_m, self.params['decimal_places'])
        
        # Create question - use category_a and category_b to match template
        question = TEMPLATE_OBJ_OBJ_DISTANCE.format(
            category_a=category1,
            category_b=category2
        )
        
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
                'object1_category': category1,
                'object2_category': category2,
                'distance_meters': distance_rounded,
                'unit': 'meters'
            }
        )
        
        return qa_pair
