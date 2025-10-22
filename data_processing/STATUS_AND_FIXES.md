# Data Processing Module - Status and Fixes

## ✅ Import Issues - FIXED

All processors have been updated to use absolute imports instead of relative imports:
- Changed `from .utils import ...` to `from utils import ...`
- This resolves the "cannot be resolved" errors

## 📋 Dataset Format Analysis

### 1. Hypersim ⚠️ **Needs Major Revision**

**Current Implementation**: Assumes EXR/PFM depth files and JSON metadata  
**Actual Format**: HDF5-based structure

**Actual Structure:**
```
ai_001_001/
  ├── _detail/
  │   ├── cam_00/              # Camera data per frame
  │   ├── mesh/                # 3D meshes
  │   ├── metadata_cameras.csv # Camera metadata
  │   ├── metadata_nodes.csv   # Scene graph nodes
  │   └── metadata_scene.csv   # Scene metadata
  └── images/
      ├── scene_cam_00_final_hdf5/      # Final RGB (HDF5)
      ├── scene_cam_00_geometry_hdf5/   # Geometry + Depth (HDF5)
      └── *_preview/                     # Preview images
```

**Required Changes:**
1. Replace EXR/PFM loading with HDF5 file reading
2. Parse CSV files for camera parameters
3. Extract depth from geometry HDF5 files
4. Extract bounding boxes from mesh/scene graph data

### 2. SUN RGB-D ✅ **Works with Current Structure**

**Current Implementation**: Attempts to load .mat metadata file  
**Actual Format**: Individual scene folders with separate files

**Actual Structure:**
```
SUNRGBD/
  ├── kv1/NYUdata/NYU0001/
  │   ├── image/           # RGB image
  │   ├── depth/           # Depth map
  │   ├── extrinsics/      # Camera extrinsics
  │   ├── intrinsics.txt   # Camera intrinsics
  │   ├── annotation3D/    # 3D bounding boxes
  │   └── ...
```

**Good News**: Each scene has complete annotation data!
- `annotation3D/` contains 3D bounding boxes
- `intrinsics.txt` has camera parameters
- `extrinsics/` has camera pose

**Required Changes:**
1. Remove .mat file dependency
2. Iterate through sensor folders (kv1, kv2, realsense, xtion)
3. Read annotations from `annotation3D/` folders
4. Parse `intrinsics.txt` for camera parameters

### 3. COCO ✅ **Data Available**

**Current Implementation**: Tries to use FiftyOne API  
**Actual Format**: Standard COCO JSON format in `labels.json`

**Data Available:**
- ✅ 50 validation images
- ✅ 427 annotations
- ✅ 80 categories
- ✅ Standard COCO bbox format: `[x, y, width, height]`

**Required Changes:**
1. Read directly from `labels.json` file
2. Parse standard COCO format (no FiftyOne needed)
3. Map image IDs to filenames
4. Convert 2D bboxes to format needed

### 4. Objectron ✅ **Ready for Implementation**

**Status**: Only 2 categories available
- ✅ bike: 378 annotation files
- ✅ book: 9 annotation files
- ✅ 776 video files

**Required Changes:**
1. Install `objectron` package
2. Parse `.pbdata` protobuf files
3. Handle only 'bike' and 'book' categories
4. Extract frames from videos or use frame annotations

### 5. EmbodiedScan/Matterport ⚠️ **Different Purpose**

**Current Implementation**: Assumes 3D bounding box corrections  
**Actual Data**: Visual grounding annotations (text-based object localization)

**Data Structure:**
```json
{
  "scan_id": "scannet/scene0191_00",
  "target_id": 5,
  "distractor_ids": [6],
  "text": "the board that is beside the door",
  "target": "board",
  "anchors": ["door"],
  "anchor_ids": [3],
  "tokens_positive": [[4, 9]]
}
```

**This is NOT 3D bounding box data!**

**Options:**
1. Skip Matterport3D processing (no 3D bbox data available)
2. Use EmbodiedScan for different purpose (visual grounding task)
3. Find alternative Matterport3D annotation source

## 🔧 Recommended Action Plan

### Priority 1: Fix SUN RGB-D (Easiest)
- Data is complete and well-structured
- Just need to update file reading logic
- Should take ~1 hour

### Priority 2: Fix COCO (Very Easy)
- Data is already in standard format
- Just parse JSON file directly
- Should take ~30 minutes

### Priority 3: Implement Objectron (Medium)
- Need to install and use objectron package
- Protobuf parsing required
- Should take ~2-3 hours

### Priority 4: Rewrite Hypersim (Complex)
- Requires HDF5 library
- Need to understand scene graph format
- May take ~4-6 hours

### Priority 5: Clarify Matterport3D (Research Needed)
- Current annotations unsuitable for 3D bbox task
- Need to find proper annotation source
- Or skip this dataset entirely

## 📝 Next Steps

1. **Update READMEs** with correct dataset formats
2. **Fix SUN RGB-D processor** to read from folder structure
3. **Fix COCO processor** to use labels.json directly
4. **Test each processor** with actual data
5. **Document limitations** (only 2 Objectron categories, no Matterport 3D bboxes)

## ✅ What's Already Good

- ✅ All utility functions in `utils.py` are correct
- ✅ Coordinate transformation logic is sound
- ✅ 9-DoF bounding box format is well-defined
- ✅ Output JSON structure is appropriate
- ✅ Import errors are fixed
