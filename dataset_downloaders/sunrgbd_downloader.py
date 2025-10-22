#!/usr/bin/env python3
"""
SUN RGB-D Dataset Downloader
Downloads the complete SUN RGB-D dataset for single-frame QA generation.
"""

import os
import requests
import zipfile
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SunRGBDDownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "SUNRGBD"
        self.url = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
        
    def download(self):
        """Download and extract SUN RGB-D dataset."""
        logger.info("Starting SUN RGB-D dataset download...")
        logger.info(f"Dataset size: ~25GB")
        logger.info(f"Output directory: {self.dataset_dir}")
        
        # Check if already exists
        if self.dataset_dir.exists() and any(self.dataset_dir.iterdir()):
            logger.info("SUN RGB-D dataset already exists. Skipping download.")
            return True
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.output_dir / "SUNRGBD.zip"
        
        try:
            # Download
            logger.info(f"Downloading from: {self.url}")
            self._download_file(self.url, zip_path)
            
            # Extract
            logger.info("Extracting dataset...")
            self._extract_zip(zip_path, self.output_dir)
            
            # Clean up
            logger.info("Cleaning up zip file...")
            zip_path.unlink()
            
            logger.info("✅ SUN RGB-D download completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading SUN RGB-D: {e}")
            return False
    
    def _download_file(self, url, output_path):
        """Download file with progress tracking."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)", end="")
        print()  # New line
    
    def _extract_zip(self, zip_path, extract_to):
        """Extract zip file."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    def verify_download(self):
        """Verify the downloaded dataset structure."""
        logger.info("Verifying SUN RGB-D dataset structure...")
        
        expected_dirs = ["kv1", "kv2", "realsense", "xtion"]
        missing_dirs = []
        
        for dir_name in expected_dirs:
            dir_path = self.dataset_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            logger.warning(f"Missing directories: {missing_dirs}")
            return False
        
        # Count samples
        sample_count = 0
        for subdir in expected_dirs:
            data_dir = self.dataset_dir / subdir
            if (data_dir / "b3dodata").exists():
                sample_count += len(list((data_dir / "b3dodata").glob("img_*")))
            elif (data_dir / "kinect2data").exists():
                sample_count += len(list((data_dir / "kinect2data").glob("*")))
        
        logger.info(f"✅ Found {sample_count} scenes in SUN RGB-D dataset")
        return True

def main():
    parser = argparse.ArgumentParser(description="SUN RGB-D Dataset Downloader")
    parser.add_argument("--output-dir", help="Output directory (default: ../raw_data)")
    args = parser.parse_args()
    
    downloader = SunRGBDDownloader(args.output_dir)
    
    if downloader.download():
        downloader.verify_download()
        print("\n🎉 SUN RGB-D dataset ready for single-frame QA generation!")
        print(f"📁 Location: {downloader.dataset_dir}")
        print("\n📊 Dataset provides:")
        print("  • RGB images (indoor scenes)")
        print("  • 2.5D depth maps")
        print("  • 3D object annotations")
        print("  • Camera intrinsics")
    else:
        print("\n❌ SUN RGB-D download failed. Check logs for details.")

if __name__ == "__main__":
    main()