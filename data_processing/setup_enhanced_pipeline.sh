#!/bin/bash

# Setup Enhanced Classification Pipeline
echo "Setting up Enhanced Two-Stage CLIP Classification Pipeline..."

# Create models directory
mkdir -p models

# Download SAM model if not exists
SAM_MODEL="models/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_MODEL" ]; then
    echo "Downloading SAM ViT-H model..."
    wget -O "$SAM_MODEL" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo "✅ SAM model downloaded"
else
    echo "✅ SAM model already exists"
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r enhanced_requirements.txt

echo "✅ Enhanced classification pipeline setup complete!"
echo ""
echo "Usage:"
echo "  python build_enhanced_codebook.py"