#!/usr/bin/env python3
"""
Debug script to understand why cam_obj_rel_dist and obj_obj_rel_pos generate 0 questions
"""
import sys
import os
sys.path.append('/mnt/sdd/zcy/VLM_data_pipeline/QA_generation')

import json
from pathlib import Path
from tasks.tasks_3d.cam_obj_rel_dist_qa import CameraObjectRelativeDistanceQA
from tasks.tasks_3d.obj_obj_rel_pos_qa import ObjectObjectRelativePositionQA

def debug_empty_tasks():
    """Debug why certain tasks generate no questions"""
    
    print("🔍 Debugging Empty QA Tasks")
    print("=" * 50)
    
    # Load some COCO data
    data_dir = Path("/mnt/sdd/zcy/VLM_data_pipeline/processed_data/coco/validation")
    sample_files = list(data_dir.glob("*.json"))[:5]
    
    print(f"Loading {len(sample_files)} sample files...")
    
    data_items = []
    for file_path in sample_files:
        with open(file_path, 'r') as f:
            item = json.load(f)
            item['_source_file'] = str(file_path)  # Add source file info
            data_items.append(item)
    
    print(f"Loaded {len(data_items)} data items")
    
    # Test cam_obj_rel_dist
    print("\n🔍 Testing CameraObjectRelativeDistanceQA:")
    cam_rel_generator = CameraObjectRelativeDistanceQA("coco")
    
    for i, item in enumerate(data_items):
        print(f"\n  Sample {i+1}:")
        print(f"    Has bounding_boxes_3d: {'bounding_boxes_3d' in item}")
        print(f"    Has camera: {'camera' in item}")
        
        if 'bounding_boxes_3d' in item:
            bboxes = item['bounding_boxes_3d']
            print(f"    Number of 3D bboxes: {len(bboxes)}")
            
            if 'camera' in item:
                from utils.geometry import get_camera_position
                camera_pos = get_camera_position(item['camera'])
                print(f"    Camera position valid: {camera_pos is not None}")
                
                if camera_pos is not None and len(bboxes) >= 2:
                    print(f"    ✅ Should generate questions (has {len(bboxes)} objects, valid camera)")
                else:
                    print(f"    ❌ Cannot generate questions (camera_pos: {camera_pos is not None}, bboxes: {len(bboxes)})")
    
    # Try generating
    print(f"\n  Attempting to generate QA pairs...")
    try:
        qa_pairs = cam_rel_generator.generate_qa(data_items)
        print(f"  Generated: {len(qa_pairs)} QA pairs")
        if len(qa_pairs) > 0:
            print(f"  Sample question: {qa_pairs[0]['question']}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test obj_obj_rel_pos
    print("\n🔍 Testing ObjectObjectRelativePositionQA:")
    rel_pos_generator = ObjectObjectRelativePositionQA("coco")
    
    try:
        qa_pairs = rel_pos_generator.generate_qa(data_items)
        print(f"  Generated: {len(qa_pairs)} QA pairs")
        if len(qa_pairs) > 0:
            print(f"  Sample question: {qa_pairs[0]['question']}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_empty_tasks()