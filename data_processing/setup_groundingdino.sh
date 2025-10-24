#!/bin/bash

# Setup script for GroundingDINO model

set -e

echo "=========================================="
echo "Setting up GroundingDINO for Taskonomy"
echo "=========================================="

# Navigate to GroundingDINO directory
cd /mnt/sdd/zcy/VLM_data_pipeline/GroundingDINO

# Install dependencies first
echo ""
echo "📦 Installing dependencies..."
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118 -q || \
pip install torch torchvision -q

# Install other requirements
echo ""
echo "📦 Installing GroundingDINO requirements..."
pip install transformers addict yapf timm pycocotools opencv-python supervision -q

# Install GroundingDINO (without building CUDA extensions for now - use CPU/pretrained)
echo ""
echo "📦 Installing GroundingDINO..."
# Skip the editable install that requires CUDA compilation
# Instead, just add to PYTHONPATH
export PYTHONPATH="/mnt/sdd/zcy/VLM_data_pipeline/GroundingDINO:$PYTHONPATH"
echo "Added GroundingDINO to PYTHONPATH"

# Create weights directory
echo ""
echo "📁 Creating weights directory..."
mkdir -p weights

# Download model weights
echo ""
echo "⬇️  Downloading GroundingDINO weights..."
cd weights

# Download SwinT-OGC weights (lighter model, faster inference)
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    echo "Downloading groundingdino_swint_ogc.pth..."
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    echo "✅ Downloaded groundingdino_swint_ogc.pth"
else
    echo "✅ groundingdino_swint_ogc.pth already exists"
fi

# Optional: Download SwinB weights (more accurate but slower)
# if [ ! -f "groundingdino_swinb_cogcoor.pth" ]; then
#     echo "Downloading groundingdino_swinb_cogcoor.pth..."
#     wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
#     echo "✅ Downloaded groundingdino_swinb_cogcoor.pth"
# fi

cd ..

echo ""
echo "=========================================="
echo "✅ GroundingDINO setup complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python data_processing/enhance_taskonomy_labels.py"
echo ""
