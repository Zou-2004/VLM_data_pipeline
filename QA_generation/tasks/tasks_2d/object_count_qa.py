"""
Object Count QA Generation
Generates questions about how many instances of each object category exist
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from utils.qa_base import BaseQAGenerator
from utils.data_loader import get_frame_category_counts
from config import TEMPLATE_OBJECT_COUNT, QA_PARAMS


class ObjectCountQA(BaseQAGenerator):
    """Generate object counting questions"""
    
    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name: Name of dataset
        """
        super().__init__('object_count', dataset_name)
        self.params = QA_PARAMS['object_count']
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate object count questions
        
        Strategy:
        - For each image/frame with multiple instances of a category
        - Generate multiple choice question with correct count and distractors
        """
        qa_pairs = []
        
        for item in data:
            # Auto-detect bbox type - prefer 2D if available, fall back to 3D
            bbox_type = '2d'
            if 'bounding_boxes_2d' in item and item['bounding_boxes_2d']:
                bbox_type = '2d'
            elif 'bounding_boxes_3d' in item and item['bounding_boxes_3d']:
                bbox_type = '3d'
            else:
                continue  # Skip if no bboxes
                
            # Get category counts for this frame
            counts = get_frame_category_counts(item, bbox_type)
            
            # Only generate questions for categories with >= min_count instances
            for category, count in counts.items():
                if count >= self.params['min_count']:
                    qa_pair = self._generate_count_question(item, category, count)
                    if qa_pair:
                        qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_count_question(self, item: Dict[str, Any], category: str, correct_count: int) -> Dict[str, Any]:
        """Generate a single count question"""
        
        # Create question
        question = TEMPLATE_OBJECT_COUNT.format(category=category)
        
        # Generate distractor options
        offset_range = self.params['distractor_offset_range']
        options = self.generate_distractor_options(
            correct_count,
            self.params['num_options'],
            offset_range=offset_range
        )
        
        # Ensure all options are positive integers
        options = [max(1, int(round(opt))) for opt in options]
        
        # Format as multiple choice
        mc_data = self.format_multiple_choice(options)
        
        # Create QA pair
        qa_pair = self.create_qa_pair(
            question=question,
            answer=mc_data['answer'],
            answer_type='multiple_choice',
            options=mc_data['options'],
            metadata={
                'source_file': item.get('_source_file', ''),
                'image_id': item.get('image_id', ''),
                'scene_id': item.get('scene_id', ''),
                'category': category,
                'correct_count': correct_count,
                'answer_value': mc_data['answer_value']
            }
        )
        
        return qa_pair
