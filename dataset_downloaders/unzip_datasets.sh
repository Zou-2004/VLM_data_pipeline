#!/bin/bash
#
# Unzip script for SUN RGB-D and Matterport3D datasets
# Usage: ./unzip_datasets.sh [raw_data_directory]
#

# Get the raw_data directory (default: ../raw_data)
RAW_DATA_DIR="${1:-$(dirname "$0")/../raw_data}"

echo "========================================"
echo "Dataset Unzip Script"
echo "========================================"
echo "Raw data directory: $RAW_DATA_DIR"
echo ""

# Function to unzip files in a directory
unzip_directory() {
    local dir="$1"
    local name="$2"
    
    if [ ! -d "$dir" ]; then
        echo "⚠️  $name directory not found: $dir"
        return
    fi
    
    echo "📦 Unzipping $name..."
    local zip_count=$(find "$dir" -name "*.zip" | wc -l)
    
    if [ "$zip_count" -eq 0 ]; then
        echo "   ✅ No zip files found (already extracted or not downloaded)"
        return
    fi
    
    echo "   Found $zip_count zip file(s)"
    
    # Unzip all files
    find "$dir" -name "*.zip" | while read -r zipfile; do
        local dirname=$(dirname "$zipfile")
        local basename=$(basename "$zipfile")
        echo "   Extracting: $basename"
        unzip -q "$zipfile" -d "$dirname" && rm "$zipfile"
    done
    
    echo "   ✅ Done!"
}

# Unzip SUN RGB-D
echo ""
echo "1. SUN RGB-D Dataset"
echo "--------------------"
unzip_directory "$RAW_DATA_DIR/SUNRGBD" "SUN RGB-D"

# Unzip Matterport3D scans
echo ""
echo "2. Matterport3D Dataset"
echo "-----------------------"
MATTERPORT_SCANS="$RAW_DATA_DIR/v1/scans"

if [ -d "$MATTERPORT_SCANS" ]; then
    echo "📦 Unzipping Matterport3D scans..."
    
    # Count total zip files
    total_zips=$(find "$MATTERPORT_SCANS" -name "*.zip" | wc -l)
    
    if [ "$total_zips" -eq 0 ]; then
        echo "   ✅ No zip files found (already extracted or not downloaded)"
    else
        echo "   Found $total_zips zip file(s) across multiple scenes"
        
        # Process each scene directory
        for scene_dir in "$MATTERPORT_SCANS"/*/ ; do
            if [ -d "$scene_dir" ]; then
                scene_name=$(basename "$scene_dir")
                scene_zips=$(find "$scene_dir" -maxdepth 1 -name "*.zip" | wc -l)
                
                if [ "$scene_zips" -gt 0 ]; then
                    echo "   Processing scene: $scene_name ($scene_zips files)"
                    
                    cd "$scene_dir"
                    for zipfile in *.zip; do
                        if [ -f "$zipfile" ]; then
                            echo "     Extracting: $zipfile"
                            unzip -q "$zipfile" && rm "$zipfile"
                        fi
                    done
                    cd - > /dev/null
                fi
            fi
        done
        
        echo "   ✅ Done!"
    fi
else
    echo "⚠️  Matterport3D directory not found: $MATTERPORT_SCANS"
fi

echo ""
echo "========================================"
echo "✅ Unzip Complete!"
echo "========================================"
