#!/usr/bin/env python3
"""
Visualize Enhanced CLIP Classification Results
Shows cropped bounding boxes with their semantic labels
"""

import json
import pickle
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_data():
    """Load enhanced codebook and instance cache."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / 'processed_data' / 'taskonomy'
    
    # Load enhanced codebook
    codebook_path = processed_dir / 'enhanced_label_codebook.json'
    with open(codebook_path, 'r') as f:
        codebook = json.load(f)
    
    # Load instance cache
    cache_path = processed_dir / '.instance_cache.pkl'
    with open(cache_path, 'rb') as f:
        cached_data = pickle.load(f)
    instance_locations = cached_data['instance_locations']
    
    return codebook, instance_locations

def load_instance_image_and_bbox(instance_id, instance_locations, raw_data_dir):
    """Load image and bbox data for an instance, trying multiple paths."""
    # Convert instance_id to int if it's a string
    if isinstance(instance_id, str):
        instance_id = int(instance_id)
    
    if instance_id not in instance_locations:
        print(f"✗ Instance {instance_id} not found in cache")
        return None, None, None
    
    json_path, bbox_idx = instance_locations[instance_id][0]
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get 2D bbox
    bbox_2d = None
    if bbox_idx < len(data.get('bounding_boxes_2d', [])):
        bbox_2d_data = data['bounding_boxes_2d'][bbox_idx]
        bbox_2d = (bbox_2d_data['x_min'], bbox_2d_data['y_min'], 
                   bbox_2d_data['x_max'], bbox_2d_data['y_max'])
    
    # Try to load image from multiple possible paths
    location_name = data['split']
    rgb_filename = data['filename']
    
    # The correct path based on find results: raw_data/taskonomy_dataset/rgb/taskonomy/location/filename
    possible_paths = [
        raw_data_dir / 'rgb' / 'taskonomy' / location_name / rgb_filename,
        raw_data_dir / location_name / rgb_filename,
        raw_data_dir / rgb_filename,
    ]
    
    image = None
    used_path = None
    for rgb_path in possible_paths:
        try:
            if rgb_path.exists():
                image = cv2.imread(str(rgb_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    used_path = rgb_path
                    print(f"✓ Loaded image from: {rgb_path}")
                    break
        except Exception as e:
            print(f"✗ Failed to load from {rgb_path}: {e}")
            continue
    
    if image is None:
        print(f"✗ Could not find image for instance {instance_id}")
        print(f"  Tried paths:")
        for p in possible_paths:
            print(f"    - {p} (exists: {p.exists()})")
    
    return image, bbox_2d, data

def visualize_cropped_examples(num_examples=16):
    """Visualize cropped bounding boxes with labels."""
    codebook, instance_locations = load_data()
    
    base_dir = Path(__file__).parent.parent
    raw_data_dir = base_dir / 'raw_data' / 'taskonomy_dataset'
    
    # Get some example instance IDs
    classified_ids = list(codebook.keys())[:num_examples]
    
    print(f"📊 Enhanced CLIP Classification: Cropped Bounding Boxes")
    print(f"Total instances: {len(instance_locations)}")
    print(f"Successfully classified: {len(codebook)}")
    print(f"Success rate: {len(codebook)/len(instance_locations)*100:.1f}%")
    print(f"Showing first {num_examples} examples...")
    print()
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Enhanced CLIP Classification: Cropped Bounding Boxes', fontsize=16)
    
    successful_crops = 0
    
    for i, instance_id in enumerate(classified_ids):
        if i >= 16:  # 4x4 grid
            break
            
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        image, bbox, data = load_instance_image_and_bbox(instance_id, instance_locations, raw_data_dir)
        
        if image is not None and bbox is not None:
            try:
                # Crop the bounding box with padding
                x1, y1, x2, y2 = bbox
                
                # Ensure bbox is valid
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Invalid bbox dimensions")
                
                # Add 15% padding
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                pad_x = int(bbox_w * 0.15)
                pad_y = int(bbox_h * 0.15)
                
                # Apply padding with bounds checking
                h, w = image.shape[:2]
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(w, x2 + pad_x)
                y2_pad = min(h, y2 + pad_y)
                
                # Crop the image
                cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    # Display cropped image
                    ax.imshow(cropped)
                    
                    # Get label
                    label = codebook[str(instance_id)]
                    
                    # Title with label
                    ax.set_title(f"ID {instance_id}: {label}", fontsize=10, fontweight='bold')
                    ax.set_xlabel(f"Size: {bbox_w}×{bbox_h}px", fontsize=8)
                    
                    successful_crops += 1
                else:
                    raise ValueError("Empty crop")
                    
            except Exception as e:
                ax.text(0.5, 0.5, f"Crop failed\nID {instance_id}\n{str(e)[:20]}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f"ID {instance_id}: ERROR", color='red')
        else:
            # Show error message
            error_msg = "No image" if image is None else "No bbox"
            ax.text(0.5, 0.5, f"Failed to load\n{error_msg}\nID {instance_id}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f"ID {instance_id}: ERROR", color='red')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    output_file = 'enhanced_clip_cropped_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization to {output_file}")
    print(f"✅ Successfully cropped {successful_crops}/{len(classified_ids)} instances")
    
    return successful_crops

def print_statistics():
    """Print classification statistics."""
    codebook, instance_locations = load_data()
    
    print("\n📈 Classification Statistics:")
    label_counts = {}
    for label in codebook.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Sort by frequency
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 most common labels:")
    for label, count in sorted_labels[:10]:
        print(f"  {label}: {count} instances")
    
    print(f"\nTotal unique labels: {len(label_counts)}")

def print_detailed_examples():
    """Print detailed text examples."""
    codebook, instance_locations = load_data()
    
    print("\n🔍 Detailed Examples:")
    print("=" * 60)
    
    # Show a few examples with their file paths
    for i, (instance_id, label) in enumerate(list(codebook.items())[:5]):
        json_path, bbox_idx = instance_locations[int(instance_id)][0]
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        bbox_2d = data['bounding_boxes_2d'][bbox_idx]
        
        print(f"Instance {instance_id}: {label}")
        print(f"  File: {data['filename']}")
        print(f"  Location: {data['split']}")
        print(f"  Bbox: ({bbox_2d['x_min']}, {bbox_2d['y_min']}) to ({bbox_2d['x_max']}, {bbox_2d['y_max']})")
        print()

if __name__ == '__main__':
    visualize_cropped_examples()
    print_statistics()
    print_detailed_examples()