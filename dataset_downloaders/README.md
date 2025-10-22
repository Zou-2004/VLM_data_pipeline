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

## � What Gets Downloaded

- **SUN RGB-D**: RGB-D indoor scenes with 3D annotations (~25GB)
- **Objectron**: 3D object detection dataset (~1-5GB)
- **EmbodiedScan-v2**: Spatial relationship annotations (~10GB)
- **COCO-2017**: Object detection and segmentation (subset: ~100MB, full: ~25GB)

## 📋 Output Structure

```
raw_data/
├── SUNRGBD/
│   ├── kv1/
│   ├── kv2/
│   ├── realsense/
│   └── xtion/
├── Objectron/
│   ├── bike/
│   ├── book/
│   ├── bottle/
│   └── ... (other categories)
├── embodiedscan-v2/
│   ├── embodiedscan_train_vg.json
│   └── embodiedscan_val_vg.json
└── COCO/
    ├── train/
    ├── validation/
    └── ... (images and annotations)
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