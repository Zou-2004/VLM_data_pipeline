# VLM 3D Data Processing

This module processes multiple 3D vision datasets into a unified JSON format for VLM training.

## Features

- **Unified 9-DoF Bounding Box Format**: All datasets converted to (x, y, z, xl, yl, zl, pitch, yaw, roll)
- **VST Camera Frame Convention**: Origin at camera, +X right, +Y down, +Z forward
- **Angle Normalization**: All angles normalized to [-1, 1] by dividing by 180°
- **Per-Image JSON Output**: Each image gets a metadata JSON file with camera params, depth info, and bboxes
- **Dataset Summary**: Each dataset produces a summary.json with statistics

## Supported Datasets

### 1. SUN RGB-D
- **Input**: .mat metadata files from various sensors (Kinect v1/v2, RealSense, Xtion)
- **Output**: RGB-D images with 9-DoF bounding boxes
- **Key Features**: Real-world indoor scenes, multiple sensor types
- **Status**: ✅ Fully working - 1,449 scenes, 12,315 3D bboxes

### 2. Matterport3D
- **Input**: Matterport scans + EmbodiedScan corrected bounding boxes
- **Output**: RGB images with corrected 9-DoF bounding boxes
- **Key Features**: High-quality real-world scenes, corrected annotations
- **Status**: ✅ Fully working - 4,932 images, ~24,462 3D bboxes

### 3. Objectron
- **Input**: Video sequences with .pbdata protobuf annotations
- **Output**: Sampled video frames with 9-DoF bounding boxes
- **Key Features**: Object-centric views, mobile phone captured
- **Status**: ✅ Fully working - 13,024 frames, ~13,284 3D bboxes

### 4. Hypersim
- **Input**: Hypersim synthetic indoor scenes
- **Output**: RGB-D images with 3D bounding boxes
- **Key Features**: Photorealistic synthetic scenes, perfect depth
- **Status**: ✅ Fully working

### 5. Taskonomy
- **Input**: Taskonomy multi-task dataset with 3D annotations
- **Output**: RGB images with semantic-labeled 3D bounding boxes
- **Key Features**: Real indoor scenes, GroundingDINO semantic labeling
- **Status**: ✅ Fully working - 3,862 views, 90,181 3D bboxes with semantic labels

## Installation

### 1. Create conda environment (recommended)

```bash
conda create -n data_pipeline python=3.11
conda activate data_pipeline
```

### 2. Install core dependencies

```bash
pip install -r requirements.txt
```

### 3. Install GroundingDINO for Taskonomy semantic labeling

```bash
cd data_processing
bash setup_groundingdino.sh
cd ..
```

### 4. Configure cache directories (important for disk space)

```bash
# Optional: Set cache directories to control where large files are stored
export HF_HOME=/path/to/hf_cache
export TORCH_HOME=/path/to/torch_cache
export TMPDIR=/path/to/tmp
export PIP_CACHE_DIR=/path/to/pip_cache
```

Add these to your `~/.bashrc` to make them permanent.

### GroundingDINO Setup for Taskonomy

The Taskonomy processor uses **GroundingDINO** for open-vocabulary semantic labeling of 3D bounding boxes:

1. **Run setup script** (downloads model weights):
   ```bash
   cd data_processing
   bash setup_groundingdino.sh
   ```

2. **Model will be downloaded** (~440MB):
   - Model: `groundingdino_swint_ogc.pth`
   - Saved to: `GroundingDINO/weights/`

3. **Runs on CPU** (CUDA extensions not required)

### MoGe-2 Depth Estimation Setup

The COCO processor uses **MoGe-2** (Metric 3D Geometry Estimation v2) for high-quality pseudo-depth estimation:

1. **Clone MoGe repository** (if not already present):
   ```bash
   cd data_processing
   git clone https://github.com/microsoft/MoGe.git
   ```

2. **Install dependencies**:
   ```bash
   cd MoGe
   pip install -r requirements.txt
   ```

3. **Configure cache directories** (important for disk space management):
   ```bash
   export HF_HOME=/mnt/sdd/hf_cache
   export TORCH_HOME=/mnt/sdd/torch_cache
   export TMPDIR=/mnt/sdd/tmp
   ```

4. **Model will auto-download** from Hugging Face on first run (~1.3GB):
   - Model: `Ruicheng/moge-2-vitl-normal`
   - Cached in: `$HF_HOME/hub/models--Ruicheng--moge-2-vitl-normal/`

## Usage

### Process All Datasets

```bash
cd data_processing
python process_all.py --raw-data-dir ../raw_data --output-dir ../processed_data
```

### Process Specific Datasets

```bash
python process_all.py --datasets sunrgbd matterport objectron hypersim taskonomy
```

### Process Individual Dataset

#### Taskonomy with Semantic Labeling

```python
from taskonomy_processor import TaskonomyProcessor

# Step 1: Process dataset with 3D bboxes
processor = TaskonomyProcessor(
    raw_data_dir="raw_data/taskonomy_dataset",
    output_dir="processed_data/taskonomy"
)
processor.process_all()

# Step 2: Build semantic label codebook using GroundingDINO
# This detects labels for unique instances once, then applies to all occurrences
python build_label_codebook_fast.py
```

**Codebook Approach**:
- Scans all JSONs to find unique unlabeled instances (e.g., `object_5`, `object_12`)
- Detects semantic labels once per unique instance using GroundingDINO
- Applies labels to all occurrences (e.g., `object_5` → `chair_5`)
- **~400x faster** than detecting every instance individually

#### SUN RGB-D

```python
from sunrgbd_processor import SUNRGBDProcessor

processor = SUNRGBDProcessor(
    raw_data_dir="raw_data/SUNRGBD",
    output_dir="processed_data/sunrgbd"
)
processor.process_all()

#### Objectron

```python
from objectron_processor import ObjectronProcessor
```

## Output Format

### Per-Image JSON

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
      "fx": 886.81,
      "fy": 886.81,
      "cx": 512.0,
      "cy": 384.0
    },
    "extrinsics": {
      "position": [0.0, 0.0, 0.0],
      "rotation": [0.0, 0.0, 0.0]
    },
    "image_width": 1024,
    "image_height": 768
  },
  "depth_stats": {
    "min": 0.5,
    "max": 10.0,
    "mean": 3.2,
    "median": 3.0
  },
  "bounding_boxes": [
    {
      "object_id": "chair_01",
      "category": "chair",
      "bbox_9dof": {
        "center": [1.5, 0.5, 3.0],
        "size": [0.6, 0.8, 0.6],
        "rotation": [0.0, 0.17, 0.0]
      }
    }
  ]
}
```

### Summary JSON

Each dataset generates a `summary.json`:

```json
{
  "dataset": "sunrgbd",
  "total_scenes": 1449,
  "total_images": 1449,
  "total_bboxes": 12315,
  "depth_type": "depth_png_mm",
  "processing_date": "2024-01-15T10:30:00",
  "output_directory": "processed_data/sunrgbd"
}
```

## Processing Statistics

| Dataset | Scenes | Images | 3D Bboxes | 2D Bboxes | Depth Type | Semantic Labels |
|---------|--------|--------|-----------|-----------|------------|-----------------|
| SUN RGB-D | 1,449 | 1,449 | 12,315 | - | depth_png_mm | ✓ |
| Matterport3D | 61 | 4,932 | ~24,462 | - | none | ✓ |
| Objectron | 2 categories | 13,024 | ~13,284 | - | none | ✓ |
| Hypersim | - | - | - | - | depth_png_mm | ✓ |
| Taskonomy | 1 location | 3,862 | 90,181 | - | none | ✓ (GroundingDINO) |
| **Total** | **~1,512** | **~23,267** | **~140,242** | **-** | - | - |

## Dataset-Specific Notes

### SUN RGB-D
- **Implementation**: Parses MATLAB .mat files using scipy
- **Annotations**: annotation3Dfinal/index.json with 3D bboxes
- **Depth**: PNG files in millimeters (depth_png_mm format)
- **Rotation**: Converts yaw-only rotations to full 9-DoF (pitch=roll=0)
- **Sensors**: Supports kv1, kv2, realsense, xtion
- **Results**: 1,449 scenes, 12,315 3D bboxes

### Matterport3D
- **Implementation**: Loads EmbodiedScan pickle files for corrected 3D bboxes
- **Structure**: Nested directories (matterport3d/SCENE/regionN)
- **Annotations**: embodiedscan_infos_train.pkl and embodiedscan_infos_val.pkl
- **Mapping**: Uses visible_instance_ids for per-image bbox assignment
- **Format Conversion**: EmbodiedScan [cx,cy,cz,dx,dy,dz,rx,ry,rz] → 9-DoF
- **Results**: 4,932 images from 61 scenes, ~24,462 3D bboxes

### Objectron
- **Implementation**: Parses protobuf .pbdata files using compiled schemas
- **Schema Compilation**: Uses grpc_tools.protoc to generate Python parsers
- **Annotations**: annotation_data_pb2.Sequence with 9 keypoint-based bboxes
- **Conversion**: Computes center and dimensions from keypoints_3d
- **Sampling**: Processes every 10th frame from video sequences
- **Categories**: bike, book (expandable to other categories)
- **Results**: 13,024 frames, ~13,284 3D bboxes

### Taskonomy
- **Implementation**: Processes RGB images with derived 3D bboxes from depth + segmentation
- **Structure**: Point cloud camera locations with multiple views per point
- **Semantic Labeling**: Uses GroundingDINO for open-vocabulary object detection
- **Codebook Approach**: Detects unique instance IDs once, applies to all occurrences
- **Label Format**: Converts `object_N` → `semantic_label_N` (e.g., `chair_5`, `table_12`)
- **Detection**: ~40 indoor object categories (furniture, appliances, structures)
- **Performance**: 253 unique instances detected in ~19 minutes, applied to 90,181 bboxes
- **Results**: 3,862 views from 1 location (ackermanville), 90,181 3D bboxes with semantic labels

### Hypersim
- **Implementation**: Processes synthetic indoor scenes
- **Depth**: Accurate metric depth from rendering
- **Results**: Processing statistics to be added

## Coordinate System

All bounding boxes use the **VST camera frame**:
- **Origin**: Camera center
- **+X**: Right
- **+Y**: Down
- **+Z**: Forward (into the scene)

## Bounding Box Format

**9-DoF representation**:
- `center`: (x, y, z) in meters, camera frame
- `size`: (xl, yl, zl) dimensions in meters
- `rotation`: (pitch, yaw, roll) normalized to [-1, 1]

**Angle normalization**:
```python
normalized_angle = angle_in_degrees / 180.0
# Range: [-1, 1] where -1 = -180°, 0 = 0°, 1 = 180°
```

## Utilities

The `utils.py` module provides common functions:

- `quaternion_to_euler()`: Convert quaternion to Euler angles
- `normalize_angle()`: Normalize angle to [-1, 1]
- `convert_bbox_to_9dof()`: Convert various bbox formats to 9-DoF
- `compute_depth_stats()`: Calculate depth map statistics
- `create_unified_json()`: Generate standardized output JSON
- `world_to_camera_frame()`: Transform world coordinates to camera frame

## Requirements

Core dependencies (see `requirements.txt`):
- `torch>=2.0.0` - PyTorch for deep learning models
- `torchvision>=0.15.0` - Computer vision utilities
- `Pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing (for SUN RGB-D .mat files)
- `h5py>=3.0.0` - HDF5 file handling
- `protobuf>=3.20.0` - Protocol buffers (for Objectron)
- `grpcio-tools>=1.40.0` - Protobuf compilation tools
- `supervision` - GroundingDINO visualization utilities

GroundingDINO specific (installed from `setup_groundingdino.sh`):
- `groundingdino` - Open-vocabulary object detection
- Model weights: `groundingdino_swint_ogc.pth` (~440MB)

MoGe-2 specific (installed from `MoGe/requirements.txt`):
- `huggingface_hub` - Model downloading
- `einops` - Tensor operations
- `timm` - Vision transformer models

## Troubleshooting

### Disk Space Issues
Make sure to set cache directories to avoid filling up home disk:
```bash
export HF_HOME=/mnt/sdd/hf_cache
export TORCH_HOME=/mnt/sdd/torch_cache
export TMPDIR=/mnt/sdd/tmp
export PIP_CACHE_DIR=/mnt/sdd/pip_cache
```

### GroundingDINO Model Issues
The model runs on CPU by default. If you encounter `_C` module errors, this is expected - the CUDA extensions are not compiled. The script automatically falls back to CPU mode.

### Taskonomy Codebook Building
The codebook building process takes ~19 minutes for 253 unique instances. Progress is shown with tqdm progress bars. If interrupted, simply rerun - the cache file will speed up subsequent runs.

### MoGe-2 Model Download
The MoGe-2 model (1.3GB) will be automatically downloaded from Hugging Face on first run of COCO processing. Ensure you have internet connectivity and sufficient disk space.

### Objectron Protobuf Schema
The protobuf schemas are already compiled in `objectron/schema/`. If you need to recompile:
```bash
cd data_processing/objectron
python -m grpc_tools.protoc -I. --python_out=schema/ *.proto
```

### Memory Issues
For large-scale processing, consider:
- Processing datasets one at a time
- Reducing batch sizes in MoGe-2 depth estimation
- Limiting the number of images per run

## License

This processing module is part of the VLM 3D Data Pipeline project.
