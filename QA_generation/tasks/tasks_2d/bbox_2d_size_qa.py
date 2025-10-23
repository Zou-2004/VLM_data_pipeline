"""
2D Bounding Box Size QA Generation (for COCO dataset)
Generates questions about 2D bounding box areas
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any
from utils.qa_base import BaseQAGenerator
from config import TEMPLATE_BBOX_2D_SIZE, QA_PARAMS


class BBox2DSizeQA(BaseQAGenerator):
    """Generate 2D bounding box size questions"""
    
    def __init__(self, dataset_name: str):
        super().__init__('bbox_2d_size', dataset_name)
        self.params = QA_PARAMS['bbox_2d_size']
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate 2D bbox size questions"""
        qa_pairs = []
        
        for item in data:
            if 'bounding_boxes_2d' not in item:
                continue
            
            # Track categories to avoid duplicates
            asked_categories = set()
            
            for bbox in item['bounding_boxes_2d']:
                category = bbox.get('category', 'unknown')
                
                if category in asked_categories:
                    continue
                
                asked_categories.add(category)
                
                qa_pair = self._generate_bbox_size_question(item, bbox, category)
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _generate_bbox_size_question(self, item: Dict[str, Any], 
                                     bbox: Dict[str, Any], 
                                     category: str) -> Dict[str, Any]:
        """Generate a single 2D bbox size question"""
        
        # Get bbox dimensions - handle both formats
        # COCO format: bbox_2d = {x, y, width, height}
        # Other format: bbox = [x, y, width, height]
        bbox_data = bbox.get('bbox_2d', bbox.get('bbox', {}))
        
        if isinstance(bbox_data, dict):
            # Dictionary format (COCO)
            width = bbox_data.get('width', 0)
            height = bbox_data.get('height', 0)
        elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
            # Array format
            x, y, width, height = bbox_data[:4]
        else:
            return None
        
        # Calculate area in pixels
        area = width * height
        
        # Skip if too small
        if area < self.params['min_area_pixels']:
            return None
        
        # Create question
        question = TEMPLATE_BBOX_2D_SIZE.format(category=category)
        
        # Generate distractor options
        options = self.generate_distractor_options(
            area,
            self.params['num_options'],
            percent_range=self.params['distractor_percent_range']
        )
        
        # Round to integers
        options = [int(round(opt)) for opt in options]
        
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
                'category': category,
                'bbox_width': width,
                'bbox_height': height,
                'correct_area_pixels': int(area),
                'answer_value': mc_data['answer_value'],
                'unit': 'pixels'
            }
        )
        
        return qa_pair
