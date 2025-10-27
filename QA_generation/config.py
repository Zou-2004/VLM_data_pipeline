"""
Configuration file for QA generation
Contains templates, parameters, and dataset configurations
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Cache directory (configurable via environment variable)
CACHE_DIR = os.environ.get('QA_CACHE_DIR', os.path.join(os.path.expanduser('~'), 'qa_generation_cache'))

# Dataset configurations
DATASETS = {
    "objectron": {
        "data_dir": os.path.join(PROCESSED_DATA_DIR, "objectron"),
        "has_3d": True,
        "has_camera_poses": True,
        "has_depth": False,
        "tasks": [
            "object_count",
            "object_3d_size",
            "obj_obj_distance",
            "cam_obj_distance",
            "cam_obj_rel_dist",
            "obj_obj_rel_pos",
        ]
    },
    "matterport": {
        "data_dir": os.path.join(PROCESSED_DATA_DIR, "matterport"),
        "has_3d": True,
        "has_camera_poses": True,
        "has_depth": False,
        "tasks": [
            "object_count",
            "object_3d_size",
            "obj_obj_distance",
            "cam_obj_distance",
            "cam_obj_rel_dist",
            "obj_obj_rel_pos",
        ]
    },
    "sunrgbd": {
        "data_dir": os.path.join(PROCESSED_DATA_DIR, "sunrgbd"),
        "has_3d": True,
        "has_camera_poses": True,
        "has_depth": True,
        "tasks": [
            "object_count",
            "object_3d_size",
            "obj_obj_distance",
            "cam_obj_distance",
            "cam_obj_rel_dist",
            "obj_obj_rel_pos",
        ]
    },
    "hypersim": {
        "data_dir": os.path.join(PROCESSED_DATA_DIR, "hypersim"),
        "has_3d": True,
        "has_camera_poses": True,
        "has_depth": True,
        "tasks": [
            "object_count",
            "object_3d_size",
            "obj_obj_distance",
            "cam_obj_distance",
            "cam_obj_rel_dist",
            "obj_obj_rel_pos",
        ]
    },
    "taskonomy": {
        "data_dir": os.path.join(PROCESSED_DATA_DIR, "taskonomy_labeled"),
        "has_3d": True,
        "has_camera_poses": True,
        "has_depth": True,
        "tasks": [
            "object_count",
            "object_3d_size",
            "obj_obj_distance",
            "cam_obj_distance",
            "cam_obj_rel_dist",
            "obj_obj_rel_pos",
        ]
    }
}

# QA Generation Parameters
QA_PARAMS = {
    "object_count": {
        "min_count": 2,  # Only generate questions for categories with at least 2 instances
        "num_options": 4,  # Number of multiple choice options
        "distractor_offset_range": (-3, 3),  # Range for generating wrong answers
    },
    "obj_count_2d": {
        "min_objects": 1,
        "max_objects": 20,
    },
    "obj_2d_size": {
        "min_area": 100,  # minimum area in pixels
        "decimal_places": 1,
    },
    "bbox_2d_size": {
        "num_options": 4,
        "min_area_pixels": 100,  # Skip very small bboxes
        "distractor_percent_range": (0.5, 1.8),
    },
    "object_3d_size": {
        "num_options": 4,
        "distractor_percent_range": (0.4, 1.8),  # 40% to 180% of correct answer
        "unit": "centimeters",
    },
    "obj_obj_distance": {
        "num_options": 4,
        "min_distance": 0.2,  # meters - skip if objects too close
        "max_distance": 20.0,  # meters - skip if objects too far
        "distractor_percent_range": (0.5, 1.5),
        "unit": "meters",
        "decimal_places": 1,
    },
    "cam_obj_distance": {
        "min_distance": 0.1,  # meters
        "decimal_places": 1,
        "unit": "meters",
    },
    "cam_obj_rel_dist": {
        "v1_samples_per_frame": 2,
        "v2_samples_per_frame": 1,
        "v3_samples_per_frame": 1,
        "min_distance_diff": 0.15,  # meters - candidates must be at least this far apart
    },
    "obj_obj_rel_pos": {
        "threshold": 0.1,  # meters - objects must be clearly separated by this threshold
    },
}

# Question Templates (VLM-3R style)

# 2D Tasks
TEMPLATE_OBJECT_COUNT = "How many {category} are there in this image?"

TEMPLATE_BBOX_2D_SIZE = "What is the area (in square pixels) of the bounding box for the {category}?"

# 3D Tasks  
TEMPLATE_OBJECT_3D_SIZE = "What is the length of the longest dimension of the {category} in centimeters?"

TEMPLATE_OBJ_OBJ_DISTANCE = "What is the distance between the {category_a} and the {category_b} in meters?"

TEMPLATE_CAM_OBJ_DISTANCE = "What is the approximate distance (in meters) between the camera and the nearest point of the {category}?"

# Camera-object relative distance (which is closer/farther)
TEMPLATE_CAM_OBJ_REL_DIST = "Which object is closest to the camera?"
TEMPLATE_CAM_OBJ_REL_DIST_V1 = "Measuring from the closest point of each object, which of these objects ({options}) is the closest to the camera?"
TEMPLATE_CAM_OBJ_REL_DIST_V2 = "Measuring from the closest point of each object, which of these objects ({options}) is the closest to the camera?"
TEMPLATE_CAM_OBJ_REL_DIST_V3 = "Which of these two objects ({options}) is closer to the camera?"

# Object-object relative position
TEMPLATE_OBJ_OBJ_REL_POS = "What is the relative position of the {category1} to the {category2}?"
TEMPLATE_OBJ_OBJ_REL_POS_NF = "Relative to the camera, is the {category_a} Near or Far compared to the {category_b}?"
TEMPLATE_OBJ_OBJ_REL_POS_LR = "Relative to the camera, is the {category_a} to the Left or Right of the {category_b}?"
TEMPLATE_OBJ_OBJ_REL_POS_UD = "Relative to the camera, is the {category_a} Up or Down relative to the {category_b}?"

# Answer format templates
MULTIPLE_CHOICE_OPTIONS = ["A", "B", "C", "D"]

# Output file naming
def get_output_path(dataset_name, task_name):
    """Get output file path for a dataset and task"""
    os.makedirs(os.path.join(OUTPUT_DIR, dataset_name), exist_ok=True)
    return os.path.join(OUTPUT_DIR, dataset_name, f"{task_name}_qa.json")
