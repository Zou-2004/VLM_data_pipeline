# VLM 3D Data Pipeline

A comprehensive data processing and QA generation pipeline for training Vision Language Models (VLMs) with spatial reasoning capabilities.

## Overview

This pipeline processes 3 major 3D vision datasets into a unified JSON format and generates VLM-3R style spatial reasoning questions. The complete pipeline includes:
- Dataset processing with standardized 3D bounding boxes and camera parameters
- Automated QA generation for spatial reasoning tasks
- Support for 3D spatial understanding

### Key Features

- **Unified Data Format**: Standardized 3D bounding boxes and camera parameters across all datasets
- **QA Generation**: Automated creation of 485K+ spatial reasoning questions in VLM-3R style
- **Multiple Depth Types**: Metric depth, depth maps
- **3 Diverse Datasets**: 25,199 images with 86,288+ 3D bounding boxes
- **Complete Pipeline**: One-command processing and QA generation


### QA Generation Tasks

- **Object Count**: Multiple choice questions about object instances
- **3D Object Size**: Maximum dimension of 3D objects (in centimeters)
- **Distance**: Camera-to-object and object-to-object distances
- **Relative Position**: Near/Far, Left/Right, Up/Down spatial relationships
- **Closest Object**: Which object is nearest to camera

## Quick Start

### 1. Setup Environment

```bash
# (Optional) Configure cache directories for large files
export QA_CACHE_DIR="/path/to/cache/qa_generation"     # Default: ~/qa_generation_cache
export PIPELINE_CACHE_DIR="/path/to/cache/pipeline"    # Default: /tmp

# Create conda environment (or use any Python 3.8+ environment)
conda create -n data_pipeline python=3.11
conda activate data_pipeline

#install download dependincies
cd dataset_downloaders
pip install -r requirements.txt && cd ..

# Install processing dependencies
cd data_processing
pip install -r requirements.txt
```

### 2. Download and Process Datasets

```bash
# Download datasets
cd dataset_downloaders
python download_all.py download
./unzip_datasets.sh
cd ..

# Process all datasets
cd data_processing
python process_all.py --raw-data-dir ../raw_data --output-dir ../processed_data
# Taskonomy with Semantic Labeling
python build_label_codebook_fast.py
```


### 3. Generate QA Pairs

```bash
cd QA_generation
python generate_qa.py --all
```

**💡 For detailed portable setup across different machines, see [QA_generation/PORTABLE_SETUP.md](QA_generation/PORTABLE_SETUP.md)**

## Project Structure

```
VLM_data_pipeline/
├── README.md                    # This file
├── data_processing/             # Dataset processors
│   ├── process_all.py           # Master processing script
│   ├── sunrgbd_processor.py     # SUN RGB-D processor
│   ├── matterport_processor.py  # Matterport3D with EmbodiedScan
│   └── objectron_processor.py   # Objectron protobuf parser
├── dataset_downloaders/         # Dataset download scripts
│   └── download_all.py          # Master download script
├── QA_generation/               # QA generation pipeline
│   ├── generate_qa.py           # Main QA generation script
│   ├── config.py                # Question templates and parameters
│   ├── utils/                   # Data loading and geometry utilities
│   ├── tasks/                   # Task-specific QA generators
│   └── output/                  # Generated QA pairs (485K+ questions)
├── raw_data/                    # Downloaded raw datasets
└── processed_data/              # Processed output (25K+ images)
```

## Output Format

### Processed Data
Each image generates a JSON file with camera parameters, depth information, and 3D bounding boxes in unified format.

### QA Pairs
Generated questions follow VLM-3R style with multiple choice and numerical answers:

```json
{
  "question": "How many chairs are in the image?",
  "answer": "C",
  "answer_type": "multiple_choice",
  "task": "object_count",
  "dataset": "matterport",
  "options": {"A": 2, "B": 5, "C": 3, "D": 1},
  "metadata": {
    "source_file": "17DRP5sb8fy/rgb_00001.json",
    "category": "chair",
    "correct_count": 3
  }
}
```

## Advanced Usage

### Process Specific Datasets
```bash
python process_all.py --datasets sunrgbd matterport
python generate_qa.py --dataset sunrgbd
```

### Generate Specific QA Tasks
```bash
python generate_qa.py --dataset objectron --tasks object_count object_3d_size
```

### Custom Output Directories
```bash
python process_all.py --output-dir /custom/path/processed_data
python generate_qa.py --output-dir /custom/path/qa_output
```

## Dependencies

- Python 3.11+
- PyTorch 2.0+
- NumPy, SciPy, Pillow, tqdm
- h5py (for HDF5 files)
- protobuf, grpcio-tools (for Objectron)

See `data_processing/requirements.txt` and `QA_generation/requirements.txt` for complete lists.

## Troubleshooting

### Disk Space
The pipeline requires significant disk space:
- Raw datasets: ~40-60GB
- Processed data: ~20-30GB
- QA output: ~500MB

Configure cache directories to avoid filling home disk:
```bash
# Alternative: Set environment variables for cache control
export HF_HOME=/path/to/hf_cache
export TORCH_HOME=/path/to/torch_cache
```

### Memory Issues
For large-scale processing:
- Process datasets one at a time
- Use `--limit` parameter for testing

## Documentation

- **Processing Guide**: `data_processing/README.md`
- **QA Generation Guide**: `QA_generation/USAGE_GUIDE.md`
- **Dataset Format Notes**: `data_processing/DATASET_FORMAT_NOTES.md`

## License

This project is licensed under the MIT License. See individual dataset licenses for usage restrictions.

## Acknowledgments

- **EmbodiedScan**: [EmbodiedScan-v2](https://github.com/OpenRobotLab/EmbodiedScan)
- **VLM-3R**: Question format and methodology
- **Datasets**: SUN RGB-D, Matterport3D, Objectron
