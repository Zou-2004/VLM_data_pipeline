# QA Generation Pipeline

This directory contains scripts to generate VLM-3R style QA pairs from the processed dataset.

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
│   ├── geometry.py     # 3D geometry calculations
│   └── qa_base.py      # Base QA generation class
├── tasks/
│   ├── __init__.py
│   ├── 2d_tasks/
│   │   ├── __init__.py
│   │   ├── object_count_qa.py
│   │   └── bbox_size_qa.py
│   └── 3d_tasks/
│       ├── __init__.py
│       ├── object_3d_size_qa.py
│       ├── obj_obj_distance_qa.py
│       ├── cam_obj_distance_qa.py
│       ├── cam_obj_rel_dist_qa.py
│       └── obj_obj_rel_pos_qa.py
├── generate_qa.py      # Main script to generate QA
└── output/             # Generated QA outputs
    ├── coco/
    ├── objectron/
    ├── matterport/
    └── sunrgbd/
```

## Usage

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

## Output Format

Each generated QA file follows this structure:
```json
{
  "dataset": "dataset_name",
  "task_type": "task_name",
  "total_questions": 100,
  "generated_date": "2025-10-22",
  "qa_pairs": [
    {
      "id": "unique_id",
      "question": "Question text",
      "answer": "Answer text or value",
      "options": ["A", "B", "C", "D"],  // For multiple choice
      "answer_type": "numerical|multiple_choice|text",
      "metadata": {
        "source_file": "path/to/source.json",
        "image_id": "...",
        "categories": ["..."],
        ...
      }
    }
  ]
}
```
