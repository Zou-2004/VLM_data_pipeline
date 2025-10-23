#!/usr/bin/env python3
"""
Test script to verify QA generation setup
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config imported")
    except Exception as e:
        print(f"✗ config failed: {e}")
        return False
    
    try:
        from utils import data_loader, geometry, qa_base
        print("✓ utils modules imported")
    except Exception as e:
        print(f"✗ utils failed: {e}")
        return False
    
    try:
        from tasks.tasks_2d.object_count_qa import ObjectCountQA
        from tasks.tasks_2d.bbox_2d_size_qa import BBox2DSizeQA
        print("✓ 2D task generators imported")
    except Exception as e:
        print(f"✗ 2D tasks failed: {e}")
        return False
    
    try:
        from tasks.tasks_3d.object_3d_size_qa import Object3DSizeQA
        from tasks.tasks_3d.cam_obj_distance_qa import CameraObjectDistanceQA
        from tasks.tasks_3d.obj_obj_distance_qa import ObjectObjectDistanceQA
        from tasks.tasks_3d.cam_obj_rel_dist_qa import CameraObjectRelativeDistanceQA
        from tasks.tasks_3d.obj_obj_rel_pos_qa import ObjectObjectRelativePositionQA
        print("✓ 3D task generators imported")
    except Exception as e:
        print(f"✗ 3D tasks failed: {e}")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_config():
    """Test configuration"""
    print("Testing configuration...")
    
    from config import DATASETS, QA_PARAMS
    
    print(f"\nConfigured datasets: {list(DATASETS.keys())}")
    for name, config in DATASETS.items():
        print(f"  {name}: {len(config['tasks'])} tasks")
    
    print(f"\nConfigured tasks: {list(QA_PARAMS.keys())}")
    print()


def test_data_loading():
    """Test loading a small sample of data"""
    print("Testing data loading...")
    
    from config import DATASETS
    from utils.data_loader import load_dataset_files
    
    # Try to load COCO data (smallest dataset)
    coco_dir = DATASETS['coco']['data_dir']
    
    if os.path.exists(coco_dir):
        print(f"Loading sample from {coco_dir}...")
        data = load_dataset_files(coco_dir, limit=1)
        
        if data:
            print(f"✓ Loaded {len(data)} sample(s)")
            sample = data[0]
            print(f"  Keys: {list(sample.keys())}")
            if 'bounding_boxes_2d' in sample:
                print(f"  2D bboxes: {len(sample['bounding_boxes_2d'])}")
            if 'bounding_boxes_3d' in sample:
                print(f"  3D bboxes: {len(sample['bounding_boxes_3d'])}")
        else:
            print("✗ No data loaded")
    else:
        print(f"✗ Data directory not found: {coco_dir}")
    
    print()


def main():
    """Run all tests"""
    print("="*60)
    print("QA Generation Setup Test")
    print("="*60 + "\n")
    
    success = test_imports()
    
    if success:
        test_config()
        test_data_loading()
        
        print("="*60)
        print("Setup test complete!")
        print("="*60 + "\n")
        print("To generate QA pairs, run:")
        print("  python generate_qa.py --dataset coco")
        print("  python generate_qa.py --dataset objectron --limit 10")
        print("  python generate_qa.py --all")
        print()
    else:
        print("\n✗ Setup test failed. Please check errors above.\n")


if __name__ == '__main__':
    main()
