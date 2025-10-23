# QA Generation - Complete Usage Guide

## Quick Start

### 1. Setup Environment

```bash
# Navigate to QA generation directory
cd /mnt/sdd/zcy/VLM_data_pipeline/QA_generation

# Activate conda environment
conda activate /mnt/sdd/conda_env/data_pipeline/

# Verify setup
python test_setup.py
```

### 2. Generate QA Pairs

**Generate for all datasets:**
```bash
python generate_qa.py --all
```

**Generate for specific dataset:**
```bash
python generate_qa.py --dataset coco
python generate_qa.py --dataset objectron
python generate_qa.py --dataset matterport
python generate_qa.py --dataset sunrgbd
```

**Generate specific tasks:**
```bash
# Only count and size for COCO
python generate_qa.py --dataset coco --tasks object_count bbox_2d_size

# Only distance tasks for Objectron
python generate_qa.py --dataset objectron --tasks cam_obj_distance obj_obj_distance
```

**Limit files for testing:**
```bash
# Process only first 10 files
python generate_qa.py --dataset matterport --limit 10
```

## Available Tasks by Dataset

### COCO (2D Only)
- `object_count` - Count objects in image
- `bbox_2d_size` - 2D bounding box area

### Objectron, Matterport, SUNRGBD (3D Datasets)
- `object_count` - Count objects in frame
- `object_3d_size` - Maximum dimension of 3D objects (in cm)
- `cam_obj_distance` - Distance from camera to object (in m)
- `obj_obj_distance` - Distance between two objects (in m)
- `cam_obj_rel_dist` - Which object is closer/farther to camera (3 variants)
- `obj_obj_rel_pos` - Relative position (Near/Far, Left/Right, Up/Down)

## Output Structure

All outputs are saved to `/mnt/sdd/zcy/VLM_data_pipeline/QA_generation/output/`

```
output/
├── coco/
│   ├── all_qa_pairs.json          # All QAs combined
│   ├── summary.json                # Generation statistics
│   ├── object_count.json           # Individual task outputs
│   └── bbox_2d_size.json
├── objectron/
│   ├── all_qa_pairs.json
│   ├── summary.json
│   ├── object_count.json
│   ├── object_3d_size.json
│   ├── cam_obj_distance.json
│   ├── obj_obj_distance.json
│   ├── cam_obj_rel_dist.json
│   └── obj_obj_rel_pos.json
└── (similar for matterport and sunrgbd)
```

## QA Pair Format

### Numerical Answer
```json
{
  "question": "What is the distance from the camera to the chair?",
  "answer": 2.5,
  "answer_type": "numerical",
  "task": "cam_obj_distance",
  "dataset": "objectron",
  "metadata": {
    "source_file": "bike/batch-1-0.json",
    "image_id": "0",
    "category": "chair",
    "distance_meters": 2.5,
    "unit": "meters"
  }
}
```

### Multiple Choice
```json
{
  "question": "How many books are in the image?",
  "answer": "C",
  "answer_type": "multiple_choice",
  "task": "object_count",
  "dataset": "coco",
  "options": {
    "A": 2,
    "B": 5,
    "C": 3,
    "D": 7
  },
  "metadata": {
    "source_file": "validation/000042.json",
    "image_id": "000042",
    "category": "book",
    "correct_count": 3,
    "answer_value": 3
  }
}
```

### Text Answer
```json
{
  "question": "Which object is closest to the camera, chair or table?",
  "answer": "chair",
  "answer_type": "text",
  "task": "cam_obj_rel_dist",
  "dataset": "matterport",
  "metadata": {
    "source_file": "17DRP5sb8fy/rgb_00001.json",
    "image_id": "rgb_00001",
    "variant": "v1_closest",
    "object1": "chair",
    "object2": "table",
    "distance1": 1.8,
    "distance2": 3.2
  }
}
```

## Example Workflows

### Test with small sample
```bash
# Test on 5 files from each dataset
python generate_qa.py --dataset coco --limit 5
python generate_qa.py --dataset objectron --limit 5
python generate_qa.py --dataset matterport --limit 5
python generate_qa.py --dataset sunrgbd --limit 5
```

### Generate specific task types
```bash
# Only counting tasks
python generate_qa.py --dataset coco --tasks object_count
python generate_qa.py --dataset objectron --tasks object_count

# Only distance-related tasks
python generate_qa.py --dataset sunrgbd --tasks cam_obj_distance obj_obj_distance cam_obj_rel_dist

# Only size-related tasks
python generate_qa.py --dataset matterport --tasks bbox_2d_size object_3d_size
```

### Full production run
```bash
# Generate all QA pairs for all datasets
python generate_qa.py --all

# Check the summary files
cat output/coco/summary.json
cat output/objectron/summary.json
cat output/matterport/summary.json
cat output/sunrgbd/summary.json
```

## Customization

### Modify Question Templates
Edit `config.py` to change question templates:
```python
TEMPLATE_OBJECT_COUNT = "How many {category} are in the image?"
TEMPLATE_OBJECT_3D_SIZE = "What is the longest dimension of the {category} in centimeters?"
# etc.
```

### Adjust Parameters
Edit task parameters in `config.py`:
```python
QA_PARAMS = {
    'object_count': {
        'min_count': 2,  # Minimum objects to ask about
        'num_options': 4  # Number of multiple choice options
    },
    'cam_obj_distance': {
        'min_distance': 0.5,  # Minimum distance threshold (meters)
        'decimal_places': 2   # Rounding precision
    },
    # etc.
}
```

### Add Custom Task
1. Create new generator in `tasks/2d_tasks/` or `tasks/3d_tasks/`
2. Inherit from `BaseQAGenerator`
3. Implement `generate_qa()` method
4. Register in `generate_qa.py` TASK_GENERATORS dict
5. Add to dataset configuration in `config.py`

## Troubleshooting

### Import errors during setup
```bash
# Make sure you're in the correct directory
cd /mnt/sdd/zcy/VLM_data_pipeline/QA_generation

# Activate environment
conda activate /mnt/sdd/conda_env/data_pipeline/

# Test imports
python test_setup.py
```

### No data loaded
```bash
# Check that processed data exists
ls -la ../processed_data/coco/
ls -la ../processed_data/objectron/
ls -la ../processed_data/matterport/
ls -la ../processed_data/sunrgbd/
```

### Low QA count
- Check task parameters in `config.py` (e.g., min_count, min_distance thresholds)
- Some tasks require multiple objects in a frame
- Some tasks filter by distance or position criteria

### Out of memory
```bash
# Process datasets one at a time
python generate_qa.py --dataset coco
python generate_qa.py --dataset objectron
# etc.

# Or use limit parameter
python generate_qa.py --dataset sunrgbd --limit 1000
```

## Performance Notes

- **COCO**: ~50 images, fast processing (~few seconds)
- **Objectron**: ~13,000 frames, may take several minutes
- **Matterport**: ~5,000 images, moderate processing time
- **SUNRGBD**: ~7,000 scenes, may take 10-15 minutes

All caches and outputs are stored in `/mnt/sdd` to avoid filling home directory.
