# Dataset Downloaders

Download all datasets for VLM 3D Data Pipeline.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download all datasets
python download_all.py download

# Check status
python download_all.py check

#After download unzip SUN RGBD and Matterport
# Run from dataset_downloaders directory (auto-detects raw_data)
./unzip_datasets.sh

# Or specify custom raw_data directory
./unzip_datasets.sh /path/to/raw_data

```

## 📦 What Gets Downloaded

- **SUN RGB-D**: RGB-D indoor scenes with 3D annotations (~25GB)
  - Includes: kv1, kv2, realsense, xtion sensor data
  - 3D bounding boxes and depth maps
- **Objectron**: Mobile-captured 3D object detection dataset (~1-5GB)
  - Categories: bike, book (expandable)
  - Protobuf annotations with 9-point 3D bboxes
- **EmbodiedScan-v2**: Corrected Matterport3D annotations (~10GB)
  - Includes: embodiedscan_infos_train.pkl, embodiedscan_infos_val.pkl
  - Corrected 3D bounding boxes for Matterport scenes
- **COCO-2017**: Object detection and segmentation (subset: ~100MB, full: ~25GB)
  - 2D bounding boxes only (depth estimated via MoGe-2)

## 📋 Output Structure

```
raw_data/
├── SUNRGBD/
│   ├── kv1/
│   ├── kv2/
│   ├── realsense/
│   └── xtion/
├── Objectron/
│   ├── annotations/
│   │   ├── bike/
│   │   └── book/
│   └── videos/
│       ├── bike/
│       └── book/
├── embodiedscan-v2/
│   ├── embodiedscan_infos_train.pkl
│   ├── embodiedscan_infos_val.pkl
│   └── ... (spatial annotations)
├── COCO/
│   └── coco-2017/
│       ├── train/
│       ├── validation/
│       └── annotations/
└── v1/
    └── scans/  (Matterport3D scenes)
```

## � COCO-2017 Advanced Usage

```bash
# Download small subset (default, ~100MB)
python coco_downloader.py --subset

# Download full dataset (~25GB)
python coco_downloader.py --full

# Download specific split
python coco_downloader.py --split validation

# Download specific classes
python coco_downloader.py --split train --classes person car chair --max-samples 100

# Download with specific label types
python coco_downloader.py --split validation --label-types detections segmentations

# Visualize downloaded dataset
python -c "import fiftyone as fo; fo.launch_app(fo.load_dataset('coco-2017'))"
```