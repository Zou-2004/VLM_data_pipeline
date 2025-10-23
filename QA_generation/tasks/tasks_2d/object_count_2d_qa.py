"""
Object Count QA Generation for 2D datasets
Generates questions about object counting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from collections import Counter
from utils.qa_base import BaseQAGenerator
from config import QA_PARAMS


class ObjectCount2DQA(BaseQAGenerator):
    """Generate object counting questions for 2D datasets"""
    
    def __init__(self, dataset_name: str):
        super().__init__('obj_count_2d', dataset_name)
        self.params = QA_PARAMS.get('obj_count_2d', {
            'min_objects': 1,
            'max_objects': 20
        })
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate object counting questions"""
        qa_pairs = []
        
        for item in data:
            # Check for 2D bboxes first, fallback to 3D bboxes for counting
            bboxes_2d = item.get('bounding_boxes_2d', [])
            bboxes_3d = item.get('bounding_boxes_3d', [])
            
            # Skip if no bounding boxes at all
            if not bboxes_2d and not bboxes_3d:
                continue
            
            # Use 2D bboxes if available, otherwise use 3D bboxes for counting
            bboxes = bboxes_2d if bboxes_2d else bboxes_3d
            
            qa_pair = self._generate_count_question(item, bboxes)
            if qa_pair:
                qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_count_question(self, item: Dict[str, Any], bboxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a single counting question"""
        
        # Count objects by category
        category_counts = Counter()
        for bbox in bboxes:
            category = bbox.get('category', 'unknown')
            category_counts[category] += 1
        
        # Total object count
        total_objects = len(bboxes)
        
        # Skip if too few or too many objects
        if total_objects < self.params['min_objects'] or total_objects > self.params['max_objects']:
            return None
        
        # Choose question type
        import random
        question_types = ['total_count']
        
        # Add specific category questions if we have multiple of the same type
        for category, count in category_counts.items():
            if count > 1:
                question_types.append(f'category_count_{category}')
        
        question_type = random.choice(question_types)
        
        if question_type == 'total_count':
            question = "How many objects are visible in this image?"
            answer = total_objects
            metadata_category = 'all_objects'
        else:
            # Category-specific count
            category = question_type.replace('category_count_', '')
            count = category_counts[category]
            question = f"How many {category}s are visible in this image?"
            answer = count
            metadata_category = category
        
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
                'question_type': question_type,
                'target_category': metadata_category,
                'total_objects': total_objects,
                'category_counts': dict(category_counts),
                'unit': 'count'
            }
        )
        
        return qa_pair