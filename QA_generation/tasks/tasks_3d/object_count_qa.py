"""
Object Count QA Generation for 3D datasets
Generates questions about object counting using 3D bounding boxes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from collections import Counter
from utils.qa_base import BaseQAGenerator
from utils.class_mapping import parse_class_category
from config import QA_PARAMS


class ObjectCount3DQA(BaseQAGenerator):
    """Generate object counting questions for 3D datasets"""
    
    def __init__(self, dataset_name: str):
        super().__init__('object_count', dataset_name)
        self.params = QA_PARAMS.get('object_count', {
            'min_objects': 1,
            'max_objects_for_category_specific': 10
        })
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate object counting questions"""
        qa_pairs = []
        
        for item in data:
            # Use 3D bounding boxes for counting
            bboxes_3d = item.get('bounding_boxes_3d', [])
            
            # Skip if no bounding boxes
            if not bboxes_3d:
                continue
            
            qa_pair = self._generate_count_question(item, bboxes_3d)
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
            # Use readable category names
            readable_category = parse_class_category(category)
            category_counts[readable_category] += 1
        
        # Filter out very small objects or unknown categories
        min_objects = self.params.get('min_objects', 1)  # Default to 1 if not specified
        category_counts = {k: v for k, v in category_counts.items() 
                          if v >= min_objects and k != 'unknown'}
        
        if not category_counts:
            return None
        
        # Choose question type: total count or category-specific count
        total_objects = sum(category_counts.values())
        
        if len(category_counts) == 1 or total_objects <= self.params.get('max_objects_for_category_specific', 5):
            # Ask about a specific category
            target_category = max(category_counts.keys(), key=lambda x: category_counts[x])
            question = f"How many {target_category}s are visible in this image?"
            answer = category_counts[target_category]
            question_type = "category_specific"
            target = target_category
        else:
            # Ask about total count
            question = "How many objects are visible in this image?"
            answer = total_objects
            question_type = "total_count"
            target = "all_objects"
        
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
                'target_category': target,
                'total_objects': total_objects,
                'category_counts': dict(category_counts),
                'unit': 'count'
            }
        )
        
        return qa_pair


if __name__ == "__main__":
    # Test the generator
    generator = ObjectCount3DQA("test")
    
    # Test data
    test_data = [{
        'image_id': 'test_001',
        'bounding_boxes_3d': [
            {'category': 'person', 'x': 1, 'y': 1, 'z': 1},
            {'category': 'person', 'x': 2, 'y': 2, 'z': 2},
            {'category': 'chair', 'x': 3, 'y': 3, 'z': 3},
        ]
    }]
    
    qa_pairs = generator.generate_qa(test_data)
    print(f"Generated {len(qa_pairs)} QA pairs")
    if qa_pairs:
        print("Example:")
        print(f"Q: {qa_pairs[0]['question']}")
        print(f"A: {qa_pairs[0]['answer']}")