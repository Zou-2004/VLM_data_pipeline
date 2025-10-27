#!/usr/bin/env python3
"""
Create Labeled-Only Taskonomy Dataset
Duplicates Taskonomy JSON files but only includes objects that have semantic labels
Supports both v1 and v2 enhanced codebooks
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_enhanced_codebook(processed_dir: Path, version: Optional[str] = None) -> Tuple[Dict[int, str], str]:
    """
    Load the enhanced codebook with semantic labels.
    Supports both v1 and v2 codebooks.
    
    Args:
        processed_dir: Path to processed data directory
        version: 'v1', 'v2', or None (auto-detect)
    
    Returns:
        (codebook_dict, version_used)
    """
    # Define codebook paths
    codebook_v1_path = processed_dir / 'enhanced_label_codebook.json'
    codebook_v2_path = processed_dir / 'enhanced_label_codebook_v2.json'
    
    # Auto-detect or use specified version
    if version == 'v2' or (version is None and codebook_v2_path.exists()):
        if not codebook_v2_path.exists():
            raise FileNotFoundError(
                f"Enhanced codebook v2 not found at {codebook_v2_path}. "
                "Please run build_enhanced_codebook_v2.py first."
            )
        codebook_path = codebook_v2_path
        version_used = 'v2'
        logger.info(f"Using Enhanced CLIP v2 codebook (hierarchical classification)")
        
    elif version == 'v1' or (version is None and codebook_v1_path.exists()):
        if not codebook_v1_path.exists():
            raise FileNotFoundError(
                f"Enhanced codebook v1 not found at {codebook_v1_path}. "
                "Please run build_enhanced_codebook.py first."
            )
        codebook_path = codebook_v1_path
        version_used = 'v1'
        logger.info(f"Using Enhanced CLIP v1 codebook (threshold-based)")
        
    else:
        raise FileNotFoundError(
            f"No enhanced codebook found. Please run either:\n"
            f"  - build_enhanced_codebook.py (v1: threshold-based)\n"
            f"  - build_enhanced_codebook_v2.py (v2: hierarchical with agreement)"
        )
    
    with open(codebook_path, 'r') as f:
        codebook = json.load(f)
    
    # Convert string keys to int
    codebook_int = {int(k): v for k, v in codebook.items()}
    
    logger.info(f"Loaded codebook with {len(codebook_int)} labeled instances")
    
    return codebook_int, version_used

def create_labeled_dataset(
    input_dir: Path, 
    output_dir: Path, 
    codebook: Dict[int, str],
    version: str = 'v1'
) -> None:
    """Create new dataset with only labeled objects."""
    
    # Get labeled instance IDs
    labeled_instance_ids = set(codebook.keys())
    logger.info(f"Found {len(labeled_instance_ids)} labeled instances")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_dir.glob('**/*.json'))
    logger.info(f"Processing {len(json_files)} JSON files...")
    
    total_processed = 0
    total_bboxes_original = 0
    total_bboxes_labeled = 0
    files_with_labeled_objects = 0
    
    for json_path in tqdm(json_files, desc="Creating labeled dataset"):
        # Load original JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Count original bboxes
        original_bbox_count = len(data.get('bounding_boxes_3d', []))
        total_bboxes_original += original_bbox_count
        
        # Filter bounding boxes to only include labeled ones
        labeled_bboxes_3d = []
        labeled_bboxes_2d = []
        
        for i, bbox_3d in enumerate(data.get('bounding_boxes_3d', [])):
            # Extract instance ID from category field (e.g., "object_18" -> 18)
            category = bbox_3d.get('category', '')
            if category.startswith('object_'):
                try:
                    instance_id = int(category.split('_')[1])
                except (ValueError, IndexError):
                    continue
            else:
                continue
            
            if instance_id in labeled_instance_ids:
                # Update object_id with semantic label
                semantic_label = codebook[instance_id]
                bbox_3d_labeled = bbox_3d.copy()
                bbox_3d_labeled['object_id'] = f"{semantic_label}_{instance_id}"
                bbox_3d_labeled['category'] = semantic_label
                labeled_bboxes_3d.append(bbox_3d_labeled)
                
                # Also update corresponding 2D bbox if it exists
                if i < len(data.get('bounding_boxes_2d', [])):
                    bbox_2d = data['bounding_boxes_2d'][i].copy()
                    bbox_2d['object_id'] = f"{semantic_label}_{instance_id}"
                    bbox_2d['category'] = semantic_label
                    labeled_bboxes_2d.append(bbox_2d)
        
        # Only save files that have labeled objects
        if labeled_bboxes_3d:
            # Update data with labeled bboxes only
            data_labeled = data.copy()
            data_labeled['bounding_boxes_3d'] = labeled_bboxes_3d
            data_labeled['bounding_boxes_2d'] = labeled_bboxes_2d
            
            # Add metadata about labeling
            data_labeled['labeling_info'] = {
                "method": f"enhanced_clip_pipeline_{version}",
                "version": version,
                "original_bbox_count": original_bbox_count,
                "labeled_bbox_count": len(labeled_bboxes_3d),
                "labeling_success_rate": len(labeled_bboxes_3d) / original_bbox_count if original_bbox_count > 0 else 0
            }
            
            # Create output path maintaining directory structure
            relative_path = json_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save labeled JSON
            with open(output_path, 'w') as f:
                json.dump(data_labeled, f, indent=2)
            
            total_bboxes_labeled += len(labeled_bboxes_3d)
            files_with_labeled_objects += 1
        
        total_processed += 1
    
    # Create summary
    summary = {
        "dataset": "taskonomy_labeled",
        "creation_date": "2025-10-27",
        "source_dataset": "taskonomy",
        "labeling_method": f"enhanced_clip_pipeline_{version}",
        "labeling_version": version,
        "statistics": {
            "total_files_processed": total_processed,
            "files_with_labeled_objects": files_with_labeled_objects,
            "original_total_bboxes": total_bboxes_original,
            "labeled_total_bboxes": total_bboxes_labeled,
            "labeling_success_rate": total_bboxes_labeled / total_bboxes_original if total_bboxes_original > 0 else 0,
            "unique_semantic_labels": len(set(codebook.values())),
            "labeled_instance_ids": len(labeled_instance_ids)
        },
        "semantic_labels": sorted(list(set(codebook.values()))),
        "label_distribution": {label: list(codebook.values()).count(label) for label in set(codebook.values())}
    }
    
    # Save summary
    summary_path = output_dir / 'summary_labeled.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Created labeled dataset:")
    logger.info(f"   Input files: {total_processed}")
    logger.info(f"   Output files: {files_with_labeled_objects}")
    logger.info(f"   Original bboxes: {total_bboxes_original}")
    logger.info(f"   Labeled bboxes: {total_bboxes_labeled}")
    logger.info(f"   Success rate: {total_bboxes_labeled/total_bboxes_original*100:.1f}%")
    logger.info(f"   Unique labels: {len(set(codebook.values()))}")
    logger.info(f"   Saved to: {output_dir}")
    logger.info(f"   Summary: {summary_path}")

def main():
    """Main function to create labeled-only Taskonomy dataset."""
    
    logger.info("="*70)
    logger.info("🏷️  Creating Labeled-Only Taskonomy Dataset")
    logger.info("="*70)
    
    base_dir = Path(__file__).parent.parent
    
    # Paths
    input_dir = base_dir / 'processed_data' / 'taskonomy'
    output_dir = base_dir / 'processed_data' / 'taskonomy_labeled'
    
    try:
        # Load enhanced codebook (auto-detects v1 or v2)
        logger.info("Loading enhanced codebook...")
        codebook, version = load_enhanced_codebook(input_dir)
        
        # Create labeled dataset
        create_labeled_dataset(input_dir, output_dir, codebook, version)
        
        logger.info("\n🎉 Labeled dataset creation complete!")
        logger.info(f"   Used: Enhanced CLIP {version}")
        
    except Exception as e:
        logger.error(f"❌ Error creating labeled dataset: {e}")
        raise

if __name__ == '__main__':
    main()