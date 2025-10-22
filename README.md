# VLM 3D Data Pipeline

A comprehensive data processing pipeline for extracting 3D annotations from multiple vision datasets to train Vision Language Models (VLMs) with spatial reasoning capabilities.

## 🎯 Overview

This pipeline processes 4 major 3D vision datasets into a unified JSON format with standardized 9-DoF bounding boxes, camera parameters, and depth information. The output is designed for training VLMs that can understand and reason about 3D spatial relationships.

### Key Features

- **Unified 9-DoF Bounding Box Format**: All 3D annotations converted to (x, y, z, xl, yl, zl, pitch, yaw, roll)
- **VST Camera Frame Convention**: Consistent coordinate system (origin at camera, +X right, +Y down, +Z forward)
- **Multiple Depth Types**: Metric depth, depth maps, and MoGe-2 pseudo-depth
- **4 Diverse Datasets**: ~19,455 images with ~50,000+ 3D bounding boxes
- **Automated Processing**: One-command pipeline for all datasets

## 📊 Supported Datasets

| Dataset | Images | 3D Bboxes | 2D Bboxes | Depth Type | Status |
|---------|--------|-----------|-----------|------------|--------|
| **SUN RGB-D** | 1,449 | 12,315 | - | depth_png_mm | ✅ |
| **Matterport3D** | 4,932 | ~24,462 | - | none | ✅ |
| **Objectron** | 13,024 | ~13,284 | - | none | ✅ |
| **COCO** | 50 | - | 427 | pseudo (MoGe-2) | ✅ |
| **Total** | **~19,455** | **~50,061** | **427** | - | - |

### Dataset Details

- **SUN RGB-D**: Real-world indoor RGB-D scenes from multiple sensors (Kinect v1/v2, RealSense, Xtion)
- **Matterport3D**: High-quality real-world indoor scenes with corrected 3D bboxes from EmbodiedScan
- **Objectron**: Mobile-captured object-centric video frames with protobuf annotations
- **COCO**: Large-scale 2D object detection with MoGe-2 depth estimation

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n data_pipeline python=3.11
conda activate data_pipeline

# Clone repository
git clone https://github.com/yourusername/VLM_data_pipeline.git
cd VLM_data_pipeline

# Install dependencies
cd data_processing
pip install -r requirements.txt

# Clone and install MoGe-2 for COCO depth estimation
git clone https://github.com/microsoft/MoGe.git
cd MoGe
pip install -r requirements.txt
cd ../..
```

### 2. Configure Cache Directories

```bash
# Add to ~/.bashrc to avoid filling up home disk
export HF_HOME=/mnt/sdd/hf_cache
export TORCH_HOME=/mnt/sdd/torch_cache
export TMPDIR=/mnt/sdd/tmp
export PIP_CACHE_DIR=/mnt/sdd/pip_cache
```

### 3. Download Datasets

```bash
cd dataset_downloaders
python download_all.py download

# Unzip SUN RGB-D and Matterport
./unzip_datasets.sh
cd ..
```

### 4. Process All Datasets

```bash
cd data_processing
python process_all.py --raw-data-dir ../raw_data --output-dir ../processed_data
```

## 📁 Project Structure

```
VLM_data_pipeline/
├── README.md                    # This file
├── data_processing/             # Dataset processors
│   ├── README.md                # Detailed processing documentation
│   ├── requirements.txt         # Python dependencies
│   ├── process_all.py           # Master processing script
│   ├── coco_processor.py        # COCO with MoGe-2 depth
│   ├── sunrgbd_processor.py     # SUN RGB-D processor
│   ├── matterport_processor.py  # Matterport3D with EmbodiedScan
│   ├── objectron_processor.py   # Objectron protobuf parser
│   ├── utils.py                 # Common utilities
│   ├── MoGe/                    # MoGe-2 depth estimation model
│   └── objectron/               # Objectron protobuf schemas
├── dataset_downloaders/         # Dataset download scripts
│   ├── README.md                # Download documentation
│   ├── download_all.py          # Master download script
│   ├── coco_downloader.py
│   ├── sunrgbd_downloader.py
│   ├── matterport_downloader.py
│   ├── objectron_downloader.py
│   └── embodiedscan_downloader.py
├── raw_data/                    # Downloaded raw datasets
│   ├── COCO/
│   ├── SUNRGBD/
│   ├── Objectron/
│   ├── embodiedscan-v2/
│   └── v1/ (Matterport3D)
└── processed_data/              # Processed output
    ├── coco/
    ├── sunrgbd/
    ├── matterport/
    └── objectron/
```

## 📝 Output Format

Each processed image generates a JSON file with this structure:

```json
{
  "dataset": "sunrgbd",
  "scene_id": "kv1_NYUdata/NYU0001",
  "image_id": "000001",
  "rgb_path": "images/000001.jpg",
  "depth_path": "depth/000001.png",
  "depth_type": "depth_png_mm",
  "camera": {
    "intrinsics": {
      "fx": 886.81, "fy": 886.81,
      "cx": 512.0, "cy": 384.0
    },
    "extrinsics": {
      "position": [0.0, 0.0, 0.0],
      "rotation": [0.0, 0.0, 0.0]
    },
    "image_width": 1024,
    "image_height": 768
  },
  "depth_stats": {
    "min": 0.5, "max": 10.0,
    "mean": 3.2, "median": 3.0
  },
  "bounding_boxes_3d": [
    {
      "object_id": "chair_01",
      "category": "chair",
      "center": [1.5, 0.5, 3.0],
      "size": [0.6, 0.8, 0.6],
      "rotation": [0.0, 0.17, 0.0]
    }
  ]
}
```

### Coordinate System

All bounding boxes use the **VST camera frame**:
- **Origin**: Camera center
- **+X**: Right
- **+Y**: Down
- **+Z**: Forward (into the scene)

### 9-DoF Bounding Box

- `center`: (x, y, z) in meters, camera frame
- `size`: (xl, yl, zl) dimensions in meters
- `rotation`: (pitch, yaw, roll) normalized to [-1, 1] by dividing by 180°

## 🔧 Advanced Usage

### Process Specific Datasets

```bash
python process_all.py --datasets sunrgbd matterport
```

### Custom Output Directory

```bash
python process_all.py --output-dir /custom/path/processed_data
```

### Individual Dataset Processing

```python
from sunrgbd_processor import SUNRGBDProcessor

processor = SUNRGBDProcessor(
    raw_data_dir="raw_data/SUNRGBD",
    output_dir="processed_data/sunrgbd"
)
processor.process_all()
```

## 📦 Dependencies

### Core Requirements
- Python 3.11+
- PyTorch 2.0+
- NumPy, SciPy, Pillow
- h5py (for HDF5 files)
- protobuf, grpcio-tools (for Objectron)
- pycocotools (for COCO)

### Optional Requirements
- MoGe-2 dependencies (for COCO depth estimation)
  - Install from `data_processing/MoGe/requirements.txt`

See `data_processing/requirements.txt` for complete list.

## 🐛 Troubleshooting

### Disk Space Issues
The pipeline requires significant disk space:
- Raw datasets: ~40-60GB
- Processed data: ~20-30GB
- Model cache (MoGe-2): ~1.3GB

Make sure to configure cache directories on a partition with sufficient space.

### MoGe-2 Model Download
The model will auto-download on first COCO processing run. Ensure:
- Internet connectivity
- ~1.3GB free space in `$HF_HOME`
- Hugging Face hub access

### Objectron Protobuf
Protobuf schemas are pre-compiled in `data_processing/objectron/schema/`. If you encounter issues, recompile:
```bash
cd data_processing/objectron
python -m grpc_tools.protoc -I. --python_out=schema/ *.proto
```

### Memory Issues
For large-scale processing:
- Process datasets one at a time
- Reduce MoGe-2 batch size in `coco_processor.py`
- Limit images per run in individual processors

## 📚 Documentation

- **Processing Guide**: `data_processing/README.md`
- **Download Guide**: `dataset_downloaders/README.md`
- **Dataset Format Notes**: `data_processing/DATASET_FORMAT_NOTES.md`
- **Implementation Status**: `data_processing/IMPLEMENTATION_STATUS.md`

## 📄 License

This project is licensed under the MIT License. See individual dataset licenses for usage restrictions.

## 🙏 Acknowledgments

- **MoGe-2**: [microsoft/MoGe](https://github.com/microsoft/MoGe)
- **EmbodiedScan**: [EmbodiedScan-v2](https://github.com/OpenRobotLab/EmbodiedScan)
- **Datasets**: COCO, SUN RGB-D, Matterport3D, Objectron

## 📧 Contact

For questions or issues, please open a GitHub issue.
