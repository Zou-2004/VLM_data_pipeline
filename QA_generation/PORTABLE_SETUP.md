# Portable Setup Guide

This guide explains how to configure the QA generation pipeline to work on different machines without hardcoded paths.

## Environment Variables

The pipeline uses environment variables for configurable paths:

### QA Generation Cache
```bash
export QA_CACHE_DIR="/path/to/your/cache"
```
- **Default**: `~/qa_generation_cache`
- **Purpose**: Stores temporary files and caches during QA generation
- **Recommended**: Use a directory with sufficient disk space

### Data Processing Cache
```bash
export PIPELINE_CACHE_DIR="/path/to/your/cache"
```
- **Default**: `/tmp`
- **Purpose**: Stores model downloads (HuggingFace, PyTorch) and temporary files
- **Recommended**: Use a directory with at least 10GB free space

## Setup Examples

### Linux/Mac with Custom Cache Directory
```bash
# Create cache directory
mkdir -p /data/cache

# Set environment variables
export QA_CACHE_DIR="/data/cache/qa_generation"
export PIPELINE_CACHE_DIR="/data/cache/pipeline"

# Run QA generation
cd QA_generation
python generate_qa.py
```

### Linux/Mac with Default Settings
```bash
# Uses ~/qa_generation_cache and /tmp by default
cd QA_generation
python generate_qa.py
```

### Windows
```cmd
REM Create cache directory
mkdir C:\cache

REM Set environment variables
set QA_CACHE_DIR=C:\cache\qa_generation
set PIPELINE_CACHE_DIR=C:\cache\pipeline

REM Run QA generation
cd QA_generation
python generate_qa.py
```

## Conda Environment Setup

The pipeline works with any Python environment that has the required packages. To set up:

```bash
# Create new environment
conda create -n vlm_pipeline python=3.8

# Activate environment
conda activate vlm_pipeline

# Install requirements
pip install -r requirements.txt
pip install -r data_processing/requirements.txt
```

## Storage Requirements

- **QA Generation**: ~1GB for cache files
- **Data Processing**: ~5-10GB for model downloads
- **Output**: ~500MB for generated QA files

## Verification

To verify your setup works:

```bash
cd QA_generation
python test_setup.py
```

This will check:
- ✅ All dependencies installed
- ✅ Cache directories accessible
- ✅ Sample QA generation works
- ✅ No hardcoded paths detected