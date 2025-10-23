#!/usr/bin/env python3
"""
Comprehensive dataset parameter checker
Checks for missing parameters that could affect QA generation numbers
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
from tqdm import tqdm


def check_single_file(file_path: str) -> Dict[str, Any]:
    """Check parameters in a single file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        checks = {
            'has_camera': 'camera' in data,
            'has_intrinsics': False,
            'has_extrinsics': False,
            'has_2d_bboxes': False,
            'has_3d_bboxes': False,
            'num_2d_bboxes': 0,
            'num_3d_bboxes': 0,
            'has_depth_stats': 'depth_stats' in data,
            'categories_2d': set(),
            'categories_3d': set(),
            'missing_bbox_params': set(),
            'camera_missing_params': set(),
            'file_path': file_path
        }
        
        # Check camera parameters
        if 'camera' in data:
            camera = data['camera']
            
            # Check intrinsics
            intrinsic_fields = ['fx', 'fy', 'cx', 'cy', 'intrinsics']
            if any(field in camera for field in intrinsic_fields):
                checks['has_intrinsics'] = True
            
            # Check extrinsics
            if 'extrinsics' in camera and camera['extrinsics'] is not None:
                checks['has_extrinsics'] = True
            
            # Track missing camera parameters
            expected_camera_params = ['fx', 'fy', 'cx', 'cy', 'image_width', 'image_height']
            for param in expected_camera_params:
                if param not in camera:
                    checks['camera_missing_params'].add(param)
        else:
            checks['camera_missing_params'] = {'camera_data_missing'}
        
        # Check 2D bounding boxes
        if 'bounding_boxes_2d' in data:
            checks['has_2d_bboxes'] = len(data['bounding_boxes_2d']) > 0
            checks['num_2d_bboxes'] = len(data['bounding_boxes_2d'])
            
            for bbox in data['bounding_boxes_2d']:
                if 'category' in bbox:
                    checks['categories_2d'].add(bbox['category'])
                
                # Check for missing bbox parameters
                expected_2d_params = ['x', 'y', 'w', 'h', 'category']
                for param in expected_2d_params:
                    if param not in bbox:
                        checks['missing_bbox_params'].add(f'2d_{param}')
        
        # Check 3D bounding boxes
        if 'bounding_boxes_3d' in data:
            checks['has_3d_bboxes'] = len(data['bounding_boxes_3d']) > 0
            checks['num_3d_bboxes'] = len(data['bounding_boxes_3d'])
            
            for bbox in data['bounding_boxes_3d']:
                if 'category' in bbox:
                    checks['categories_3d'].add(bbox['category'])
                
                # Check for missing bbox parameters
                expected_3d_params = ['x', 'y', 'z', 'xl', 'yl', 'zl', 'category']
                for param in expected_3d_params:
                    if param not in bbox:
                        checks['missing_bbox_params'].add(f'3d_{param}')
        
        return checks
        
    except Exception as e:
        return {
            'error': str(e),
            'file_path': file_path
        }


def check_dataset_directory(dataset_path: str, max_files: int = None) -> Dict[str, Any]:
    """Check all files in a dataset directory"""
    dataset_path = Path(dataset_path)
    json_files = list(dataset_path.glob("**/*.json"))
    
    # Filter out summary files
    json_files = [f for f in json_files if f.name != 'summary.json']
    
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"\nChecking {len(json_files)} files in {dataset_path.name}...")
    
    results = {
        'dataset_name': dataset_path.name,
        'total_files': len(json_files),
        'files_with_camera': 0,
        'files_with_intrinsics': 0,
        'files_with_extrinsics': 0,
        'files_with_2d_bboxes': 0,
        'files_with_3d_bboxes': 0,
        'total_2d_bboxes': 0,
        'total_3d_bboxes': 0,
        'categories_2d': set(),
        'categories_3d': set(),
        'missing_params_count': defaultdict(int),
        'camera_missing_count': defaultdict(int),
        'error_files': [],
        'sample_issues': []
    }
    
    for json_file in tqdm(json_files, desc=f"Checking {dataset_path.name}"):
        file_checks = check_single_file(str(json_file))
        
        if 'error' in file_checks:
            results['error_files'].append({
                'file': str(json_file),
                'error': file_checks['error']
            })
            continue
        
        # Aggregate results
        if file_checks['has_camera']:
            results['files_with_camera'] += 1
        if file_checks['has_intrinsics']:
            results['files_with_intrinsics'] += 1
        if file_checks['has_extrinsics']:
            results['files_with_extrinsics'] += 1
        if file_checks['has_2d_bboxes']:
            results['files_with_2d_bboxes'] += 1
        if file_checks['has_3d_bboxes']:
            results['files_with_3d_bboxes'] += 1
        
        results['total_2d_bboxes'] += file_checks['num_2d_bboxes']
        results['total_3d_bboxes'] += file_checks['num_3d_bboxes']
        
        results['categories_2d'].update(file_checks['categories_2d'])
        results['categories_3d'].update(file_checks['categories_3d'])
        
        # Count missing parameters
        for param in file_checks['missing_bbox_params']:
            results['missing_params_count'][param] += 1
        
        for param in file_checks['camera_missing_params']:
            results['camera_missing_count'][param] += 1
        
        # Collect sample issues for detailed analysis
        if len(results['sample_issues']) < 5:
            if file_checks['missing_bbox_params'] or file_checks['camera_missing_params']:
                results['sample_issues'].append({
                    'file': str(json_file),
                    'missing_bbox_params': list(file_checks['missing_bbox_params']),
                    'missing_camera_params': list(file_checks['camera_missing_params'])
                })
    
    return results


def print_dataset_summary(results: Dict[str, Any]):
    """Print a summary of dataset check results"""
    print(f"\n{'='*60}")
    print(f"DATASET: {results['dataset_name']}")
    print(f"{'='*60}")
    
    print(f"Total files checked: {results['total_files']}")
    print(f"Files with camera data: {results['files_with_camera']} ({results['files_with_camera']/results['total_files']*100:.1f}%)")
    print(f"Files with intrinsics: {results['files_with_intrinsics']} ({results['files_with_intrinsics']/results['total_files']*100:.1f}%)")
    print(f"Files with extrinsics: {results['files_with_extrinsics']} ({results['files_with_extrinsics']/results['total_files']*100:.1f}%)")
    
    print(f"\nBounding Boxes:")
    print(f"  2D: {results['files_with_2d_bboxes']} files, {results['total_2d_bboxes']} total boxes")
    print(f"  3D: {results['files_with_3d_bboxes']} files, {results['total_3d_bboxes']} total boxes")
    
    print(f"\nCategories found:")
    print(f"  2D categories ({len(results['categories_2d'])}): {sorted(results['categories_2d'])}")
    print(f"  3D categories ({len(results['categories_3d'])}): {sorted(results['categories_3d'])}")
    
    if results['missing_params_count']:
        print(f"\nMissing bbox parameters:")
        for param, count in sorted(results['missing_params_count'].items()):
            print(f"  {param}: {count} files ({count/results['total_files']*100:.1f}%)")
    
    if results['camera_missing_count']:
        print(f"\nMissing camera parameters:")
        for param, count in sorted(results['camera_missing_count'].items()):
            print(f"  {param}: {count} files ({count/results['total_files']*100:.1f}%)")
    
    if results['error_files']:
        print(f"\nError files ({len(results['error_files'])}):")
        for error_info in results['error_files'][:5]:  # Show first 5
            print(f"  {error_info['file']}: {error_info['error']}")
        if len(results['error_files']) > 5:
            print(f"  ... and {len(results['error_files']) - 5} more")
    
    if results['sample_issues']:
        print(f"\nSample files with issues:")
        for issue in results['sample_issues']:
            print(f"  {issue['file']}:")
            if issue['missing_bbox_params']:
                print(f"    Missing bbox params: {issue['missing_bbox_params']}")
            if issue['missing_camera_params']:
                print(f"    Missing camera params: {issue['missing_camera_params']}")


def main():
    """Main function to check all datasets"""
    base_path = Path("/mnt/sdd/zcy/VLM_data_pipeline/processed_data")
    
    datasets = ['coco', 'matterport', 'objectron', 'sunrgbd']
    all_results = {}
    
    for dataset_name in datasets:
        dataset_path = base_path / dataset_name
        if dataset_path.exists():
            # Limit files for faster checking (remove limit for full check)
            results = check_dataset_directory(str(dataset_path), max_files=100)
            all_results[dataset_name] = results
            print_dataset_summary(results)
        else:
            print(f"\nDataset {dataset_name} not found at {dataset_path}")
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_files = sum(r['total_files'] for r in all_results.values())
    total_with_extrinsics = sum(r['files_with_extrinsics'] for r in all_results.values())
    total_3d_boxes = sum(r['total_3d_bboxes'] for r in all_results.values())
    
    print(f"Total files across all datasets: {total_files}")
    print(f"Files with extrinsics: {total_with_extrinsics} ({total_with_extrinsics/total_files*100:.1f}%)")
    print(f"Total 3D bounding boxes: {total_3d_boxes}")
    
    print(f"\nDataset-wise extrinsics availability:")
    for dataset_name, results in all_results.items():
        extrinsics_pct = results['files_with_extrinsics'] / results['total_files'] * 100
        print(f"  {dataset_name}: {results['files_with_extrinsics']}/{results['total_files']} ({extrinsics_pct:.1f}%)")
    
    # Identify potential QA generation issues
    print(f"\nPotential QA generation issues:")
    for dataset_name, results in all_results.items():
        issues = []
        
        if results['files_with_extrinsics'] == 0:
            issues.append("No extrinsics data (limits spatial QA)")
        
        if results['files_with_3d_bboxes'] < results['total_files'] * 0.5:
            issues.append("Low 3D bbox coverage")
        
        if 'camera_data_missing' in results['camera_missing_count']:
            issues.append("Missing camera data entirely")
        
        if issues:
            print(f"  {dataset_name}: {', '.join(issues)}")
        else:
            print(f"  {dataset_name}: No major issues detected")


if __name__ == "__main__":
    main()