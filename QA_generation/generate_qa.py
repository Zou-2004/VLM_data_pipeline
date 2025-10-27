#!/usr/bin/env python3
"""
Main QA Generation Script
Generates VLM-3R style QA pairs for processed datasets
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from config import DATASETS, OUTPUT_DIR, CACHE_DIR
from utils.data_loader import load_dataset_files, filter_by_bbox_availability

# Import task generators
from tasks.tasks_2d.object_count_2d_qa import ObjectCount2DQA
from tasks.tasks_2d.object_2d_size_qa import Object2DSizeQA
from tasks.tasks_3d.object_count_qa import ObjectCount3DQA
from tasks.tasks_3d.object_3d_size_qa import Object3DSizeQA
from tasks.tasks_3d.cam_obj_distance_qa import CameraObjectDistanceQA
from tasks.tasks_3d.obj_obj_distance_qa import ObjectObjectDistanceQA
from tasks.tasks_3d.cam_obj_rel_dist_qa import CameraObjectRelativeDistanceQA
from tasks.tasks_3d.obj_obj_rel_pos_qa import ObjectObjectRelativePositionQA


# Task generator mapping
TASK_GENERATORS = {
    'object_count': ObjectCount3DQA,  # 3D object counting
    'obj_count_2d': ObjectCount2DQA,
    'obj_2d_size': Object2DSizeQA,
    'bbox_2d_size': None,  # Deprecated, use obj_2d_size
    'object_3d_size': Object3DSizeQA,
    'cam_obj_distance': CameraObjectDistanceQA,
    'obj_obj_distance': ObjectObjectDistanceQA,
    'cam_obj_rel_dist': CameraObjectRelativeDistanceQA,
    'obj_obj_rel_pos': ObjectObjectRelativePositionQA,
}


def generate_for_dataset(dataset_name: str, tasks: List[str] = None, limit: int = None, base_output_dir: str = None):
    """
    Generate QA pairs for a specific dataset
    
    Args:
        dataset_name: Name of the dataset (objectron, matterport, sunrgbd, hypersim, taskonomy)
        tasks: List of task names to run (None = all supported tasks)
        limit: Maximum number of data files to process (None = all)
        base_output_dir: Base output directory (None = use default OUTPUT_DIR)
    """
    if base_output_dir is None:
        base_output_dir = OUTPUT_DIR
        
    print(f"\n{'='*60}")
    print(f"Generating QA pairs for: {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    # Get dataset configuration
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_config = DATASETS[dataset_name]
    data_dir = dataset_config['data_dir']
    supported_tasks = dataset_config['tasks']
    
    # Determine which tasks to run
    if tasks is None:
        tasks_to_run = supported_tasks
    else:
        # Validate requested tasks
        tasks_to_run = []
        for task in tasks:
            if task in supported_tasks:
                tasks_to_run.append(task)
            else:
                print(f"Warning: Task '{task}' not supported for {dataset_name}, skipping")
        
        if not tasks_to_run:
            print(f"No valid tasks specified for {dataset_name}")
            return
    
    print(f"Tasks to generate: {', '.join(tasks_to_run)}\n")
    
    # Load dataset
    print(f"Loading data from: {data_dir}")
    data = load_dataset_files(data_dir, limit=limit)
    
    if not data:
        print(f"No data found in {data_dir}")
        return
    
    print(f"Loaded {len(data)} data items\n")
    
    # Create output directory
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run each task
    all_qa_pairs = []
    
    for task_name in tqdm(tasks_to_run, desc="Generating QA tasks"):
        print(f"\n  Task: {task_name}")
        
        # Get generator class
        generator_class = TASK_GENERATORS.get(task_name)
        if generator_class is None:
            print(f"    Warning: No generator found for task '{task_name}' (may be deprecated)")
            continue
        
        # Filter data based on task requirements
        if task_name in ['obj_2d_size', 'obj_count_2d']:
            # 2D tasks need 2D bboxes specifically
            filtered_data = filter_by_bbox_availability(data, bbox_type='2d')
        elif task_name == 'object_count':
            # Legacy object count can work with either 2D or 3D bboxes
            # Try 2D first, then fall back to 3D
            filtered_data = filter_by_bbox_availability(data, bbox_type='2d')
            if not filtered_data:
                filtered_data = filter_by_bbox_availability(data, bbox_type='3d')
        else:
            # 3D tasks need 3D bboxes
            filtered_data = filter_by_bbox_availability(data, bbox_type='3d')
        
        if not filtered_data:
            print(f"    No data available with required bbox type")
            continue
        
        # Generate QA pairs
        generator = generator_class(dataset_name)
        qa_pairs = generator.generate_qa(filtered_data)
        
        print(f"    Generated: {len(qa_pairs)} QA pairs")
        
        # Save task-specific output
        task_output_path = os.path.join(output_dir, f"{task_name}.json")
        generator.save_qa_pairs(task_output_path)
        
        # Add to combined output
        all_qa_pairs.extend(qa_pairs)
    
    # Save combined output
    combined_path = os.path.join(output_dir, "all_qa_pairs.json")
    with open(combined_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    # Generate summary
    summary = {
        'dataset': dataset_name,
        'total_qa_pairs': len(all_qa_pairs),
        'tasks': {},
        'output_directory': output_dir
    }
    
    for task_name in tasks_to_run:
        task_qa = [qa for qa in all_qa_pairs if qa.get('task') == task_name]
        summary['tasks'][task_name] = {
            'count': len(task_qa),
            'output_file': f"{task_name}.json"
        }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Summary for {dataset_name.upper()}:")
    print(f"  Total QA pairs generated: {len(all_qa_pairs)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Combined output: {combined_path}")
    print(f"  Summary file: {summary_path}")
    print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate VLM-3R style QA pairs for processed datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all QA pairs for Objectron dataset
  python generate_qa.py --dataset objectron
  
  # Generate specific tasks for Taskonomy
  python generate_qa.py --dataset taskonomy --tasks object_count object_3d_size
  
  # Generate for all datasets
  python generate_qa.py --all
  
  # Process limited number of files for testing
  python generate_qa.py --dataset hypersim --limit 10

Available datasets: objectron, matterport, sunrgbd, hypersim, taskonomy
Available tasks:
  3D tasks: object_count, object_3d_size, cam_obj_distance, obj_obj_distance, 
            cam_obj_rel_dist, obj_obj_rel_pos
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['objectron', 'matterport', 'sunrgbd', 'hypersim', 'taskonomy'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        help='Specific tasks to generate (default: all supported tasks for dataset)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate QA pairs for all datasets'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of data files to process (for testing)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Update output directory if specified
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    
    # Ensure output and cache directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Determine which datasets to process
    if args.all:
        datasets_to_process = list(DATASETS.keys())
    elif args.dataset:
        datasets_to_process = [args.dataset]
    else:
        parser.error("Must specify --dataset or --all")
    
    # Process each dataset
    for dataset_name in datasets_to_process:
        try:
            generate_for_dataset(dataset_name, args.tasks, args.limit, output_dir)
        except Exception as e:
            print(f"\nError processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("QA Generation Complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
