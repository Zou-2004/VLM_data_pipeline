"""
Object 3D Size QA Generation
Generates questions about the maximum dimension of 3D objects
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from utils.qa_base import BaseQAGenerator
from utils.geometry import get_max_dimension
from config import TEMPLATE_OBJECT_3D_SIZE, QA_PARAMS


class Object3DSizeQA(BaseQAGenerator):
    """Generate 3D object size questions"""
    
    def __init__(self, dataset_name: str):
        super().__init__('object_3d_size', dataset_name)
        self.params = QA_PARAMS['object_3d_size']
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate 3D size questions"""
        qa_pairs = []
        
        for item in data:
            if 'bounding_boxes_3d' not in item:
                continue
            
            # Track categories already asked about in this frame to avoid duplicates
            asked_categories = set()
            
            for bbox in item['bounding_boxes_3d']:
                category = bbox.get('category', 'unknown')
                
                # Only ask once per category per frame
                if category in asked_categories:
                    continue
                
                asked_categories.add(category)
                
                qa_pair = self._generate_size_question(item, bbox, category)
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_size_question(self, item: Dict[str, Any], bbox: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Generate a single size question"""
        
        # Get maximum dimension in meters, convert to centimeters
        max_dim_m = get_max_dimension(bbox)
        max_dim_cm = max_dim_m * 100  # meters to centimeters
        
        # Create question
        question = TEMPLATE_OBJECT_3D_SIZE.format(category=category)
        
        # Generate distractor options
        options = self.generate_distractor_options(
            max_dim_cm,
            self.params['num_options'],
            percent_range=self.params['distractor_percent_range']
        )
        
        # Round to 1 decimal place
        options = [round(opt, 1) for opt in options]
        
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
                'correct_size_cm': round(max_dim_cm, 1),
                'answer_value': mc_data['answer_value'],
                'unit': 'centimeters'
            }
        )
        
        return qa_pair
