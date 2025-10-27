# Dataset Downloaders

Download all datasets for VLM 3D Data Pipeline.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download automated datasets (SUN RGB-D, Objectron, EmbodiedScan, Matterport)
python download_all.py download

# Check status of all datasets
python download_all.py check

# After download, unzip SUN RGB-D and Matterport
# Run from dataset_downloaders directory (auto-detects raw_data)
./unzip_datasets.sh

# Or specify custom raw_data directory
./unzip_datasets.sh /path/to/raw_data

```

## 📦 Individual Dataset Downloads

### Automated Downloads (via download_all.py)

- **SUN RGB-D**: RGB-D indoor scenes with 3D annotations (~25GB)
  - Includes: kv1, kv2, realsense, xtion sensor data
  - 3D bounding boxes and depth maps
- **Objectron**: Mobile-captured 3D object detection dataset (~1-5GB)
  - Categories: bike, book (expandable)
  - Protobuf annotations with 9-point 3D bboxes
- **EmbodiedScan-v2**: Corrected Matterport3D annotations (~10GB)
  - Includes: embodiedscan_infos_train.pkl, embodiedscan_infos_val.pkl
  - Corrected 3D bounding boxes for Matterport scenes
- **Matterport3D**: Indoor scene scans with camera poses
  - RGB images and camera parameters

### Manual Downloads (Large Datasets)

#### Hypersim (~461 scenes, ~11TB total)
Photorealistic synthetic indoor scenes from Apple.

```bash
# Option 1: Download specific scenes (5-10GB per scene)
python hypersim_downloader.py --scenes ai_001_001 ai_001_002 ai_001_003

# Option 2: Download first N scenes for testing
python hypersim_downloader.py --max-scenes 5

# Option 3: Manual download using official script (for full dataset)
# 1. Clone repository
git clone https://github.com/apple/ml-hypersim ../raw_data/ml-hypersim
cd ../raw_data/ml-hypersim

# 2. Download scenes
python code/python/tools/dataset_download_images.py \
  --downloads_dir ../Hyperism \
  --decompress_dir ../Hyperism
```

**Requirements:** `pip install h5py pandas`

#### Taskonomy (34 buildings, >10TB total)
Multi-task learning dataset from Stanford.

```bash
# Option 1: Download specific building and domains (~50-200GB)
python taskonomy_downloader.py --buildings ackermanville \
  --domains rgb depth_euclidean segment_semantic

# Option 2: Download tiny subset for testing
python taskonomy_downloader.py --tiny --buildings ackermanville

# Option 3: Manual download using omnidata-tools (for full dataset)
# 1. Install dependencies
sudo apt-get install aria2  # or: brew install aria2
pip install omnidata-tools

# 2. Download full dataset
omnitools.download all --components taskonomy --subset fullplus \
  --dest ../raw_data/taskonomy_dataset/ \
  --connections_total 40 --agree

# For specific buildings/domains
omnitools.download point_info class_places class_scene --components taskonomy \
  --subset fullplus --dest ../raw_data/taskonomy_dataset/ --only ackermanville \
  --connections_total 40 --agree
```

**Requirements:** `pip install omnidata-tools`  
**Storage:** Plan accordingly - each building is 50-200GB depending on domains selected

## 📊 Dataset Summary

| Dataset | Size | Auto Download | Manual Steps Required |
|---------|------|---------------|----------------------|
| SUN RGB-D | ~25GB | ✅ Yes | Unzip after download |
| Objectron | ~1-5GB | ✅ Yes | None |
| EmbodiedScan-v2 | ~10GB | ✅ Yes | None |
| Matterport3D | Varies | ✅ Yes | Unzip + TOS acceptance |
| **Hypersim** | **~11TB full** | ⚠️ Partial | Use official script for full |
| **Taskonomy** | **>10TB full** | ⚠️ Partial | Use omnidata-tools for full |

## 🔍 Verify Downloads

```bash
# Check status of all datasets
python download_all.py check

# Output example:
# ✅ SUN RGB-D          (4/4 sensors)
# ✅ Objectron          (2 categories)
# ✅ EmbodiedScan-v2    (2 files)
# ✅ Matterport3D       
# ✅ Hypersim           (5 scenes)
# ✅ Taskonomy          (3 domains)
# 📊 Total: 6/7 datasets available
```

## ⚙️ Advanced Usage

### Hypersim Advanced Options

```bash
# List available scenes
python hypersim_downloader.py --help

# Download all scenes (WARNING: ~11TB)
# Use the official ml-hypersim repository method for full download
```

### Taskonomy Advanced Options

```bash
# List available buildings and domains
python taskonomy_downloader.py --list

# Download multiple buildings
python taskonomy_downloader.py --buildings ackermanville albertville \
  --domains rgb depth_euclidean normal

# Available domains: rgb, depth_euclidean, depth_zbuffer, normal, reshading,
#                   segment_semantic, edge_texture, keypoints2d, keypoints3d, etc.
```

## 📝 Notes

- **Hypersim** and **Taskonomy** are very large datasets (>10TB each)
- Use the provided Python scripts for testing/partial downloads
- Use official tools (ml-hypersim scripts, omnidata-tools) for full downloads
- Plan storage carefully before downloading
- Both datasets require post-processing before use in QA generation:
  - Hypersim: Run `hypersim_processor.py`
  - Taskonomy: Run `taskonomy_processor.py` → Enhanced CLIP pipeline → `create_labeled_dataset.py`

