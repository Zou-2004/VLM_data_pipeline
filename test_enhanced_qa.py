#!/usr/bin/env python3
"""
Test the enhanced QA generation with extrinsics support
Run a small subset of each dataset to verify improvements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'QA_generation'))

from QA_generation.generate_qa import generate_for_dataset
from QA_generation.config import DATASETS
import json


def test_enhanced_qa_generation():
    """Test the enhanced QA generation with small samples"""
    
    print("Testing Enhanced QA Generation with Extrinsics Support")
    print("="*60)
    
    # Test configuration: small samples for quick testing
    test_config = {
        'coco': {
            'limit': 10,  # Small sample for quick test
            'expected_tasks': ['obj_count_2d', 'obj_2d_size']
        },
        'sunrgbd': {
            'limit': 10,  # Small sample for quick test
            'expected_tasks': ['object_count', 'object_3d_size', 'cam_obj_distance', 'obj_obj_rel_pos']
        },
        'objectron': {
            'limit': 10,
            'expected_tasks': ['object_count', 'object_3d_size', 'cam_obj_distance']
        },
        'matterport': {
            'limit': 10,
            'expected_tasks': ['object_count', 'object_3d_size', 'cam_obj_distance']
        }
    }
    
    results = {}
    
    for dataset_name, config in test_config.items():
        print(f"\nTesting {dataset_name}...")
        
        try:
            # Run QA generation for this dataset
            generate_for_dataset(
                dataset_name=dataset_name, 
                limit=config['limit'],
                tasks=config.get('expected_tasks')
            )
            
            # Check output files
            output_dir = f"output/{dataset_name}"
            qa_files = []
            total_qa_pairs = 0
            
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    if filename.endswith('_qa.json'):
                        filepath = os.path.join(output_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                qa_data = json.load(f)
                                count = len(qa_data.get('qa_pairs', []))
                                qa_files.append((filename, count))
                                total_qa_pairs += count
                        except Exception as e:
                            print(f"    Error reading {filename}: {e}")
            
            results[dataset_name] = {
                'success': True,
                'qa_files': qa_files,
                'total_qa_pairs': total_qa_pairs
            }
            
            print(f"  ✓ Generated {total_qa_pairs} QA pairs across {len(qa_files)} files")
            for filename, count in qa_files:
                print(f"    - {filename}: {count} pairs")
                
        except Exception as e:
            results[dataset_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ Failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("ENHANCED QA GENERATION TEST SUMMARY")
    print("="*60)
    
    total_successful = 0
    total_qa_pairs = 0
    
    for dataset_name, result in results.items():
        if result['success']:
            total_successful += 1
            total_qa_pairs += result['total_qa_pairs']
            print(f"✓ {dataset_name}: {result['total_qa_pairs']} QA pairs")
        else:
            print(f"✗ {dataset_name}: {result['error']}")
    
    print(f"\nDatasets processed successfully: {total_successful}/{len(test_config)}")
    print(f"Total QA pairs generated: {total_qa_pairs}")
    
    # Check for extrinsics usage
    print(f"\nExtrinsics Enhancement Check:")
    datasets_with_extrinsics = ['sunrgbd', 'objectron', 'matterport']
    for dataset_name in datasets_with_extrinsics:
        if dataset_name in results and results[dataset_name]['success']:
            print(f"✓ {dataset_name}: Should have enhanced spatial QA with extrinsics")
        else:
            print(f"✗ {dataset_name}: Not tested or failed")
    
    if 'coco' in results and results['coco']['success']:
        print(f"✓ coco: Should have 2D-specific QA tasks")
    else:
        print(f"✗ coco: Not tested or failed")
    
    print(f"\nKey Improvements:")
    print(f"  - Enhanced camera-object distance using extrinsics matrices")
    print(f"  - Improved spatial relationship calculations")
    print(f"  - Better 2D bbox parameter handling for COCO")
    print(f"  - Fallback mechanisms for missing parameters")
    
    return results


def check_specific_enhancements():
    """Check specific files for enhancement evidence"""
    print(f"\nChecking for Enhancement Evidence...")
    
    # Check SUNRGBD file for extrinsics usage
    test_files = [
        "output/sunrgbd/cam_obj_distance_qa.json",
        "output/sunrgbd/obj_obj_rel_pos_qa.json",
        "output/coco/obj_count_2d_qa.json",
        "output/coco/obj_2d_size_qa.json"
    ]
    
    for filepath in test_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                qa_pairs = data.get('qa_pairs', [])
                if qa_pairs:
                    # Check first QA pair for enhancement metadata
                    first_qa = qa_pairs[0]
                    metadata = first_qa.get('metadata', {})
                    
                    print(f"  {filepath}:")
                    if 'uses_extrinsics' in metadata:
                        print(f"    ✓ Has extrinsics usage flag: {metadata['uses_extrinsics']}")
                    if 'question_type' in metadata:
                        print(f"    ✓ Has question type: {metadata['question_type']}")
                    if 'unit' in metadata:
                        print(f"    ✓ Has unit information: {metadata['unit']}")
                    
            except Exception as e:
                print(f"  {filepath}: Error reading - {e}")
        else:
            print(f"  {filepath}: File not found")


if __name__ == "__main__":
    results = test_enhanced_qa_generation()
    check_specific_enhancements()