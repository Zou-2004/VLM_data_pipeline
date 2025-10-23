"""
Camera-Object Relative Distance QA Generation
Generates questions about which object is closest/farthest from camera
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any, Tuple
import random
from utils.qa_base import BaseQAGenerator
from utils.geometry import get_camera_position, distance_camera_to_bbox
from config import TEMPLATE_CAM_OBJ_REL_DIST, QA_PARAMS


class CameraObjectRelativeDistanceQA(BaseQAGenerator):
    """Generate which-object-is-closer-to-camera questions"""
    
    def __init__(self, dataset_name: str):
        super().__init__('cam_obj_rel_dist', dataset_name)
        self.params = QA_PARAMS['cam_obj_rel_dist']
    
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate camera-object relative distance questions"""
        qa_pairs = []
        
        for item in data:
            if 'bounding_boxes_3d' not in item or 'camera' not in item:
                continue
            
            # Get camera position
            camera_pos = get_camera_position(item['camera'])
            if camera_pos is None:
                continue
            
            bboxes = item['bounding_boxes_3d']
            
            # Need at least 2 objects for v1/v2, 3 for v3
            if len(bboxes) < 2:
                continue
            
            # Generate different question variants
            qa_pairs.extend(self._generate_v1_questions(item, bboxes, camera_pos))
            qa_pairs.extend(self._generate_v2_questions(item, bboxes, camera_pos))
            
            if len(bboxes) >= 3:
                qa_pairs.extend(self._generate_v3_questions(item, bboxes, camera_pos))
        
        self.add_qa_pairs(qa_pairs)
        return qa_pairs
    
    def _get_distances(self, bboxes: List[Dict], camera_pos) -> List[Tuple[Dict, float]]:
        """Calculate distances from camera to all objects"""
        distances = []
        for bbox in bboxes:
            dist = distance_camera_to_bbox(camera_pos, bbox)
            distances.append((bbox, dist))
        return distances
    
    def _generate_v1_questions(self, item: Dict[str, Any], bboxes: List[Dict], camera_pos) -> List[Dict]:
        """V1: Which object is closest/farthest to camera (2 objects)"""
        qa_pairs = []
        
        # Calculate distances
        distances = self._get_distances(bboxes, camera_pos)
        distances.sort(key=lambda x: x[1])
        
        # Sample pairs
        for _ in range(min(self.params['v1_samples_per_frame'], len(distances) - 1)):
            # Randomly pick 2 objects
            idx1, idx2 = random.sample(range(len(distances)), 2)
            bbox1, dist1 = distances[idx1]
            bbox2, dist2 = distances[idx2]
            
            cat1 = bbox1.get('category', 'unknown')
            cat2 = bbox2.get('category', 'unknown')
            
            # Closest question
            question_close = f"Which object is closest to the camera, {cat1} or {cat2}?"
            answer_close = cat1 if dist1 < dist2 else cat2
            
            qa_pairs.append(self.create_qa_pair(
                question=question_close,
                answer=answer_close,
                answer_type='text',
                metadata={
                    'source_file': item.get('_source_file', ''),
                    'image_id': item.get('image_id', ''),
                    'variant': 'v1_closest',
                    'object1': cat1,
                    'object2': cat2,
                    'distance1': round(dist1, 2),
                    'distance2': round(dist2, 2)
                }
            ))
            
            # Farthest question
            question_far = f"Which object is farthest from the camera, {cat1} or {cat2}?"
            answer_far = cat1 if dist1 > dist2 else cat2
            
            qa_pairs.append(self.create_qa_pair(
                question=question_far,
                answer=answer_far,
                answer_type='text',
                metadata={
                    'source_file': item.get('_source_file', ''),
                    'image_id': item.get('image_id', ''),
                    'variant': 'v1_farthest',
                    'object1': cat1,
                    'object2': cat2,
                    'distance1': round(dist1, 2),
                    'distance2': round(dist2, 2)
                }
            ))
        
        return qa_pairs
    
    def _generate_v2_questions(self, item: Dict[str, Any], bboxes: List[Dict], camera_pos) -> List[Dict]:
        """V2: Which object is closest to camera (3+ objects, multiple choice)"""
        qa_pairs = []
        
        if len(bboxes) < 3:
            return qa_pairs
        
        # Calculate distances
        distances = self._get_distances(bboxes, camera_pos)
        
        for _ in range(min(self.params['v2_samples_per_frame'], 1)):
            # Sample 3-4 objects
            num_options = min(4, len(distances))
            sampled = random.sample(distances, num_options)
            sampled.sort(key=lambda x: x[1])
            
            # Closest object
            closest_bbox = sampled[0][0]
            closest_cat = closest_bbox.get('category', 'unknown')
            
            # Create multiple choice options
            options = [bbox.get('category', 'unknown') for bbox, _ in sampled]
            
            question = "Which object is closest to the camera?"
            mc_data = self.format_multiple_choice(options)
            
            qa_pairs.append(self.create_qa_pair(
                question=question,
                answer=mc_data['answer'],
                answer_type='multiple_choice',
                options=mc_data['options'],
                metadata={
                    'source_file': item.get('_source_file', ''),
                    'image_id': item.get('image_id', ''),
                    'variant': 'v2_multiple_choice',
                    'answer_value': mc_data['answer_value'],
                    'distances': {bbox.get('category'): round(dist, 2) for bbox, dist in sampled}
                }
            ))
        
        return qa_pairs
    
    def _generate_v3_questions(self, item: Dict[str, Any], bboxes: List[Dict], camera_pos) -> List[Dict]:
        """V3: Rank 3 objects by distance from camera"""
        qa_pairs = []
        
        if len(bboxes) < 3:
            return qa_pairs
        
        # Calculate distances
        distances = self._get_distances(bboxes, camera_pos)
        
        for _ in range(min(self.params['v3_samples_per_frame'], 1)):
            # Sample 3 objects
            sampled = random.sample(distances, 3)
            sampled.sort(key=lambda x: x[1])  # Sort by distance
            
            # Get ordered categories
            ordered_cats = [bbox.get('category', 'unknown') for bbox, _ in sampled]
            
            question = f"Rank these three objects by distance from the camera (closest to farthest): {', '.join(ordered_cats)}"
            answer = ', '.join(ordered_cats)
            
            qa_pairs.append(self.create_qa_pair(
                question=question,
                answer=answer,
                answer_type='text',
                metadata={
                    'source_file': item.get('_source_file', ''),
                    'image_id': item.get('image_id', ''),
                    'variant': 'v3_ranking',
                    'ordered_objects': ordered_cats,
                    'distances': {cat: round(dist, 2) for (bbox, dist), cat in zip(sampled, ordered_cats)}
                }
            ))
        
        return qa_pairs
