"""
Object-Object Relative Position QA Generation
Generates questions about relative spatial positions (Near/Far, Left/Right, Up/Down)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
import random
from utils.qa_base import BaseQAGenerator
from utils.geometry import enhanced_relative_position
from utils.class_mapping import parse_class_category
from config import TEMPLATE_OBJ_OBJ_REL_POS, QA_PARAMS


class ObjectObjectRelativePositionQA(BaseQAGenerator):
    """Generate object-object relative position questions"""
    
    def __init__(self, dataset_name: str):
        super().__init__('obj_obj_rel_pos', dataset_name)
        self.params = QA_PARAMS['obj_obj_rel_pos']
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate relative position questions"""
        qa_pairs = []
        
        for item in data:
            if 'bounding_boxes_3d' not in item or 'camera' not in item:
                continue
            
            bboxes = item['bounding_boxes_3d']
            
            # Need at least 2 objects
            if len(bboxes) < 2:
                continue
            
            # Generate questions for pairs
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    bbox1 = bboxes[i]
                    bbox2 = bboxes[j]
                    
                    qa_pair = self._generate_rel_pos_question(
                        item, bbox1, bbox2
                    )
                    if qa_pair:
                        qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_rel_pos_question(self, item: Dict[str, Any], 
                                   bbox1: Dict[str, Any], 
                                   bbox2: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single relative position question"""
        
        category1 = bbox1.get('category', 'unknown')
        category2 = bbox2.get('category', 'unknown')
        
        # Get camera parameters
        camera = item.get('camera', {})
        
        # Get enhanced relative position information
        rel_pos_info = enhanced_relative_position(bbox1, bbox2, camera)
        
        if rel_pos_info is None:
            return None
        
        # Extract relationships
        depth_rel = rel_pos_info.get('depth_relation')
        horizontal_rel = rel_pos_info.get('horizontal_relation')
        vertical_rel = rel_pos_info.get('vertical_relation')
        
        # Randomly choose one aspect to ask about
        aspects = []
        if depth_rel and depth_rel not in ["Same depth"]:
            aspects.append(('depth', depth_rel))
        if horizontal_rel and horizontal_rel not in ["Same horizontal position"]:
            aspects.append(('horizontal', horizontal_rel))
        if vertical_rel and vertical_rel not in ["Same vertical position"]:
            aspects.append(('vertical', vertical_rel))
        
        if not aspects:
            return None
        
        aspect_type, answer = random.choice(aspects)
        
        # Convert class_X format to human-readable names
        readable_category1 = parse_class_category(category1)
        readable_category2 = parse_class_category(category2)
        
        # Create question based on aspect
        if aspect_type == 'depth':
            question = f"Is the {readable_category1} nearer or farther than the {readable_category2} from the camera?"
            # Standardize answer format
            if answer == "Nearer":
                answer = "nearer"
            elif answer == "Farther":
                answer = "farther"
        elif aspect_type == 'horizontal':
            question = f"Is the {readable_category1} to the left or right of the {readable_category2} from the camera's perspective?"
            # Standardize answer format
            if answer == "Left":
                answer = "left"
            elif answer == "Right":
                answer = "right"
        else:  # vertical
            question = f"Is the {readable_category1} above or below the {readable_category2} from the camera's perspective?"
            # Standardize answer format
            if answer == "Above":
                answer = "above"
            elif answer == "Below":
                answer = "below"
        
        # Create QA pair
        qa_pair = self.create_qa_pair(
            question=question,
            answer=answer,
            answer_type='text',
            metadata={
                'source_file': item.get('_source_file', ''),
                'image_id': item.get('image_id', ''),
                'scene_id': item.get('scene_id', ''),
                'frame_id': item.get('frame_id', ''),
                'object1_category': category1,
                'object2_category': category2,
                'object1_readable_category': readable_category1,
                'object2_readable_category': readable_category2,
                'aspect': aspect_type,
                'depth_relation': depth_rel,
                'horizontal_relation': horizontal_rel,
                'vertical_relation': vertical_rel,
                'center_distance': rel_pos_info.get('center_distance'),
                'min_distance': rel_pos_info.get('min_distance'),
                'uses_extrinsics': 'extrinsics' in camera and camera['extrinsics'] is not None
            }
        )
        
        return qa_pair
