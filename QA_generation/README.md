# QA Generation Pipeline

This directory contains scripts to generate VLM-3R style QA pairs from the processed dataset.

## Key Features

### 🎯 Human-Readable Object Names
- **Semantic Class Mapping**: Converts cryptic labels like "class_84" to meaningful names like "hair_dryer"
- **VLM-Friendly Questions**: Questions use natural language object names for better training
- **Backward Compatibility**: Original class IDs preserved in metadata

### 🔧 Enhanced Spatial Reasoning
- **Extrinsics Integration**: Improved camera-object distance calculations using 4x4 transformation matrices
- **World Coordinate System**: Better spatial relationship detection (left/right, near/far, up/down)
- **Fallback Mechanisms**: Graceful handling when extrinsics data unavailable

## Dataset Capabilities

### COCO (2D Only)
- **Available Data**: 2D bounding boxes, RGB images
- **Supported Tasks**:
  - Object Count QA
  - 2D Bounding Box Size QA
  - Category Detection QA

### 3D Datasets (Objectron, Matterport, SUNRGBD)
- **Available Data**: 3D bounding boxes, camera intrinsics/extrinsics, depth (for SUNRGBD)
- **Supported Tasks**:
  - All COCO tasks
  - Object 3D Size QA
  - Object-Object 3D Distance QA  
  - Camera-Object 3D Distance QA (frame-level)
  - Object-Object Relative Position QA (Near/Far, Left/Right, Up/Down) - frame-level
  - Camera-Object Relative Distance QA (frame-level)

## Directory Structure

```
QA_generation/
├── README.md
├── requirements.txt
├── config.py           # Configuration and templates
├── utils/
│   ├── __init__.py
│   ├── data_loader.py  # Load processed JSON files
│   ├── geometry.py     # 3D geometry calculations (enhanced with extrinsics)
│   ├── class_mapping.py # Object class ID to semantic name mapping
│   └── qa_base.py      # Base QA generation class
├── tasks/
│   ├── __init__.py
│   ├── 2d_tasks/
│   │   ├── __init__.py
│   │   ├── object_count_qa.py
│   │   └── bbox_size_qa.py
│   └── 3d_tasks/
│       ├── __init__.py
│       ├── object_3d_size_qa.py      # Enhanced with semantic names
│       ├── obj_obj_distance_qa.py
│       ├── cam_obj_distance_qa.py    # Enhanced with extrinsics support
│       ├── cam_obj_rel_dist_qa.py
│       └── obj_obj_rel_pos_qa.py     # Enhanced with semantic names
├── generate_qa.py      # Main script to generate QA
├── output/             # Generated QA outputs
│   ├── coco/
│   ├── objectron/
│   ├── matterport/
│   └── sunrgbd/
└── analysis_tools/     # Additional analysis scripts
    ├── check_dataset_parameters.py
    ├── analyze_qa_improvements.py
    └── test_enhanced_qa.py
```

## Usage

### Basic Usage
```bash
# Activate environment (use your environment name)
conda activate data_pipeline  # or your environment name

# Optional: Set cache directories
export QA_CACHE_DIR="/path/to/cache"  # Default: ~/qa_generation_cache

# Generate QA for all datasets
python generate_qa.py --all

# Generate QA for specific dataset
python generate_qa.py --dataset coco
python generate_qa.py --dataset objectron
python generate_qa.py --dataset matterport
python generate_qa.py --dataset sunrgbd

# Generate specific task types
python generate_qa.py --dataset sunrgbd --tasks object_count,object_3d_size
```

### Advanced Usage

#### Testing Class Mapping
```bash
# Test the class mapping functionality
python utils/class_mapping.py

# Example output:
# Class 84: hair_dryer
# Class 49: clothes  
# Class 276: special
```

#### Analysis Tools
```bash
# Check dataset parameter coverage
python check_dataset_parameters.py

# Analyze QA generation improvements
python analyze_qa_improvements.py

# Test enhanced QA generation on small sample
python test_enhanced_qa.py
```

### Setup Requirements

**No additional setup needed!** The class mapping is automatically integrated:
- `class_mapping.py` is imported by QA generators
- Works out-of-the-box when you run `generate_qa.py`  
- Compatible across different computers/environments
- No manual configuration required

## Output Format

### Enhanced QA Structure
Each generated QA file follows this structure with enhanced metadata:
```json
{
  "dataset": "dataset_name",
  "task_type": "task_name", 
  "total_questions": 100,
  "generated_date": "2025-10-23",
  "qa_pairs": [
    {
      "id": "unique_id",
      "question": "What is the size of the hair_dryer?",  // Uses readable names!
      "answer": "Answer text or value",
      "options": ["A", "B", "C", "D"],  // For multiple choice
      "answer_type": "numerical|multiple_choice|text",
      "metadata": {
        "source_file": "path/to/source.json",
        "image_id": "...",
        "category": "class_84",                    // Original class ID
        "readable_category": "hair_dryer",        // Human-readable name
        "uses_extrinsics": true,                  // Extrinsics enhancement flag
        "unit": "meters",                         // Measurement units
        ...
      }
    }
  ]
}
```

### Key Improvements in Output
- **Semantic Questions**: "What is the size of the hair_dryer?" instead of "What is the size of the class_84?"
- **Enhanced Metadata**: Both original and readable category names preserved
- **Extrinsics Tracking**: Flag indicates when enhanced spatial calculations used
- **Backward Compatibility**: Original class IDs maintained for reference
