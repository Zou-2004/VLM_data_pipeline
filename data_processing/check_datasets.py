#!/usr/bin/env python3
"""
Test script to check actual dataset formats
"""

import os
from pathlib import Path

def check_hypersim(raw_data_dir):
    print("\n" + "="*60)
    print("HYPERSIM Dataset Check")
    print("="*60)
    
    hypersim_dir = Path(raw_data_dir) / "Hyperism"
    if not hypersim_dir.exists():
        print("❌ Hypersim directory not found!")
        return
    
    scenes = [d for d in hypersim_dir.iterdir() if d.is_dir() and d.name.startswith("ai_")]
    print(f"✓ Found {len(scenes)} scenes")
    
    if scenes:
        scene = scenes[0]
        print(f"\nChecking scene: {scene.name}")
        print(f"  Structure:")
        for item in sorted(scene.iterdir()):
            if item.is_dir():
                sub_items = list(item.iterdir())[:3]
                print(f"    {item.name}/ ({len(list(item.iterdir()))} items)")
                for sub in sub_items:
                    print(f"      - {sub.name}")
                if len(list(item.iterdir())) > 3:
                    print(f"      ... and {len(list(item.iterdir())) - 3} more")

def check_sunrgbd(raw_data_dir):
    print("\n" + "="*60)
    print("SUN RGB-D Dataset Check")
    print("="*60)
    
    sunrgbd_dir = Path(raw_data_dir) / "SUNRGBD"
    if not sunrgbd_dir.exists():
        print("❌ SUNRGBD directory not found!")
        return
    
    sensors = ["kv1", "kv2", "realsense", "xtion"]
    for sensor in sensors:
        sensor_dir = sunrgbd_dir / sensor
        if sensor_dir.exists():
            subdirs = [d for d in sensor_dir.iterdir() if d.is_dir()]
            print(f"✓ {sensor}: {len(subdirs)} subdirectories")
            
            # Check first subdirectory structure
            if subdirs:
                first_sub = subdirs[0]
                print(f"  Example: {first_sub.name}")
                scenes = list(first_sub.iterdir())[:3]
                for scene in scenes:
                    if scene.is_dir():
                        items = [i.name for i in scene.iterdir()]
                        print(f"    {scene.name}: {items}")

def check_coco(raw_data_dir):
    print("\n" + "="*60)
    print("COCO Dataset Check")
    print("="*60)
    
    coco_dir = Path(raw_data_dir) / "COCO" / "coco-2017"
    if not coco_dir.exists():
        print("❌ COCO directory not found!")
        return
    
    val_dir = coco_dir / "validation"
    if val_dir.exists():
        data_dir = val_dir / "data"
        if data_dir.exists():
            images = list(data_dir.glob("*.jpg"))
            print(f"✓ Validation images: {len(images)}")
            if images:
                print(f"  Example: {images[0].name}")
        
        labels_file = val_dir / "labels.json"
        if labels_file.exists():
            size = labels_file.stat().st_size
            print(f"✓ labels.json exists (size: {size} bytes)")
            if size == 0:
                print("  ⚠️ WARNING: labels.json is empty!")

def check_objectron(raw_data_dir):
    print("\n" + "="*60)
    print("Objectron Dataset Check")
    print("="*60)
    
    objectron_dir = Path(raw_data_dir) / "Objectron"
    if not objectron_dir.exists():
        print("❌ Objectron directory not found!")
        return
    
    ann_dir = objectron_dir / "annotations"
    if ann_dir.exists():
        pbdata_files = list(ann_dir.glob("*.pbdata"))
        print(f"✓ Annotation files: {len(pbdata_files)}")
        
        # Count by category
        categories = {}
        for f in pbdata_files:
            cat = f.name.split("_batch")[0]
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"  Categories:")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count} files")
    
    video_dir = objectron_dir / "videos"
    if video_dir.exists():
        videos = list(video_dir.glob("*"))
        print(f"✓ Video files: {len(videos)}")

def check_embodiedscan(raw_data_dir):
    print("\n" + "="*60)
    print("EmbodiedScan Dataset Check")
    print("="*60)
    
    es_dir = Path(raw_data_dir) / "embodiedscan-v2" / "embodiedscan-v2"
    if not es_dir.exists():
        print("❌ EmbodiedScan directory not found!")
        return
    
    train_file = es_dir / "embodiedscan_train_vg.json"
    val_file = es_dir / "embodiedscan_val_vg.json"
    
    if train_file.exists():
        import json
        with open(train_file) as f:
            data = json.load(f)
        print(f"✓ Training annotations: {len(data)} entries")
        if data:
            print(f"  Example keys: {list(data[0].keys())}")
            print(f"  Example: {data[0]}")
    
    if val_file.exists():
        import json
        with open(val_file) as f:
            data = json.load(f)
        print(f"✓ Validation annotations: {len(data)} entries")

def check_matterport(raw_data_dir):
    print("\n" + "="*60)
    print("Matterport3D Dataset Check")
    print("="*60)
    
    mp_dir = Path(raw_data_dir) / "v1" / "scans"
    if not mp_dir.exists():
        print("❌ Matterport directory not found!")
        return
    
    scans = [d for d in mp_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(scans)} scans")
    
    if scans:
        scan = scans[0]
        print(f"\nExample scan: {scan.name}")
        items = list(scan.iterdir())[:5]
        for item in items:
            print(f"  {item.name}")
        if len(list(scan.iterdir())) > 5:
            print(f"  ... and {len(list(scan.iterdir())) - 5} more")

def main():
    import sys
    
    if len(sys.argv) > 1:
        raw_data_dir = sys.argv[1]
    else:
        raw_data_dir = Path(__file__).parent.parent / "raw_data"
    
    print("Dataset Format Checker")
    print(f"Raw data directory: {raw_data_dir}")
    
    check_hypersim(raw_data_dir)
    check_sunrgbd(raw_data_dir)
    check_coco(raw_data_dir)
    check_objectron(raw_data_dir)
    check_embodiedscan(raw_data_dir)
    check_matterport(raw_data_dir)
    
    print("\n" + "="*60)
    print("Check complete!")
    print("="*60)
    print("\n⚠️  IMPORTANT FINDINGS:")
    print("1. Hypersim uses HDF5 format, not EXR/PFM + JSON")
    print("2. SUN RGB-D has no .mat metadata file, uses folder structure")
    print("3. COCO labels.json is empty, need to use FiftyOne API")
    print("4. Objectron has only 'bike' and 'book' categories")
    print("5. EmbodiedScan has visual grounding annotations, not 3D bboxes")
    print("\nSee DATASET_FORMAT_NOTES.md for detailed information")

if __name__ == "__main__":
    main()
