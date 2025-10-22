# Dataset Format Notes

This document describes the actual data formats found in the downloaded datasets and how the processing code should handle them.

## 1. Hypersim

**Location**: `raw_data/Hyperism/ai_001_001`, `ai_001_002`, etc.

**Structure**:
```
ai_001_001/
  ├── _detail/
  │   ├── cam_00/              # Camera parameters per frame
  │   ├── mesh/                # 3D mesh data
  │   ├── metadata_cameras.csv
  │   ├── metadata_nodes.csv
  │   └── metadata_scene.csv
  └── images/
      ├── scene_cam_00_final_hdf5/      # Final rendered images (HDF5)
      ├── scene_cam_00_final_preview/   # Preview images
      ├── scene_cam_00_geometry_hdf5/   # Geometry data including depth
      └── scene_cam_00_geometry_preview/
```

**Note**: This is NOT the standard ml-hypersim format. No JSON metadata or EXR/PFM depth files are present. The actual format uses HDF5 files for image and geometry data.

**Processing Strategy**:
- Read HDF5 files for images and depth maps
- Parse CSV files for camera parameters
- Extract mesh and bounding box information from HDF5 geometry data

## 2. SUN RGB-D

**Location**: `raw_data/SUNRGBD/kv1/`, `kv2/`, `realsense/`, `xtion/`

**Structure**:
```
SUNRGBD/
  ├── kv1/
  │   ├── NYUdata/
  │   │   ├── NYU0001/
  │   │   │   ├── image/          # RGB image
  │   │   │   ├── depth/          # Depth map
  │   │   │   ├── extrinsics/     # Camera ext rinsics
  │   │   │   └── intrinsics/     # Camera intrinsics
  │   │   ├── NYU0002/
  │   │   └── ...
  │   └── b3dodata/
  ├── kv2/
  ├── realsense/
  └── xtion/
```

**Note**: No `.mat` metadata file is present. The data is organized by sensor type and scene folders. Each scene has separate folders for image, depth, and camera parameters.

**Processing Strategy**:
- Iterate through sensor types (kv1, kv2, realsense, xtion)
- For each scene folder, read:
  - RGB image from `image/` folder
  - Depth map from `depth/` folder
  - Camera parameters from `intrinsics/` and `extrinsics/` folders
- Bounding boxes might need to be extracted from additional annotation files or are not available in this subset

## 3. COCO

**Location**: `raw_data/COCO/coco-2017/validation/`

**Structure**:
```
coco-2017/
  ├── validation/
  │   ├── data/
  │   │   ├── 000000000139.jpg
  │   │   ├── 000000000632.jpg
  │   │   └── ...
  │   └── labels.json  # Empty file
  ├── raw/
  └── info.json
```

**Note**: The labels.json file is empty. This is a FiftyOne dataset structure, and annotations should be loaded using FiftyOne API, not from JSON files.

**Processing Strategy**:
- Use FiftyOne to load the dataset: `fiftyone.load_dataset("coco-2017")`
- Extract annotations programmatically through FiftyOne API
- Images are in `validation/data/` folder
- No depth maps available (pseudo-depth would need to be estimated)

## 4. Objectron

**Location**: `raw_data/Objectron/`

**Structure**:
```
Objectron/
  ├── annotations/
  │   ├── bike_batch-0_10.pbdata
  │   ├── bike_batch-0_13.pbdata
  │   ├── book_batch-15_0.pbdata
  │   └── ...
  └── videos/
      └── ... (video files)
```

**Note**: Only bike and book categories are present in annotations. The `.pbdata` files contain protobuf-encoded frame annotations.

**Processing Strategy**:
- Install `objectron` package to parse `.pbdata` files
- Extract frame-by-frame annotations including:
  - Camera intrinsics and extrinsics
  - 3D bounding box corners
  - Object rotation (quaternions)
- Match annotations with video frames
- Sample frames (e.g., every 10th frame)

## 5. EmbodiedScan (Matterport3D Annotations)

**Location**: `raw_data/embodiedscan-v2/embodiedscan-v2/`

**Structure**:
```
embodiedscan-v2/
  ├── embodiedscan_train_vg.json
  └── embodiedscan_val_vg.json
```

**Note**: These are visual grounding annotations, NOT 3D bounding boxes. The files contain:
- `scan_id`: Scene ID (e.g., "arkitscenes/Training/47332136")
- `target_id`: Target object ID
- `distractor_ids`: List of distractor object IDs
- `text`: Natural language description
- `target`: Object category
- `anchors`: Reference objects
- `anchor_ids`: Reference object IDs
- `tokens_positive`: Token positions in text

**Matterport3D Scans**: `raw_data/v1/scans/`

**Processing Strategy**:
- These are NOT corrected 3D bounding boxes as originally assumed
- These are visual grounding annotations for object localization
- To get actual 3D bounding boxes for Matterport, need different annotation files
- Consider using EmbodiedScan's original 3D bbox annotations if available, or skip Matterport3D processing

## Recommended Fixes

### 1. Hypersim Processor
- Implement HDF5 file reading
- Parse CSV metadata files
- Extract depth from geometry HDF5 files

### 2. SUN RGB-D Processor
- Remove .mat file dependency
- Read directly from folder structure
- Load images, depth maps, and camera parameters from respective folders

### 3. COCO Processor
- Use FiftyOne API to load annotations
- Don't rely on labels.json file

### 4. Objectron Processor
- Implement protobuf parsing with `objectron` package
- Handle only available categories (bike, book)

### 5. Matterport Processor
- Clarify that EmbodiedScan annotations are for visual grounding, not 3D bboxes
- Either find actual 3D bbox annotations or document limitation
- May need to skip or use alternative annotation source
