#!/usr/bin/env python3
"""
EmbodiedScan-v2 Dataset Downloader
Downloads EmbodiedScan-v2 dataset with spatial relationship annotations from Google Drive.
"""

import os
import zipfile
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbodiedScanDownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "embodiedscan-v2"
        # Google Drive file ID from the shared link
        self.gdrive_file_id = "13EZC2wB_aEQFJJDrOWyWBmojOdLCwpDB"
        self.gdrive_url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"
        
    def download(self):
        """Download EmbodiedScan-v2 annotation files from Google Drive."""
        logger.info("Starting EmbodiedScan-v2 dataset download from Google Drive...")
        logger.info(f"Output directory: {self.dataset_dir}")
        
        # Create output directory
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        zip_path = self.dataset_dir / "embodiedscan_infos.zip"
        if self._check_existing_files():
            logger.info("✅ EmbodiedScan-v2 dataset already exists.")
            return True
        
        try:
            # Try to import gdown
            try:
                import gdown
            except ImportError:
                logger.info("Installing gdown for Google Drive downloads...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
                import gdown
            
            # Download from Google Drive
            logger.info(f"Downloading from Google Drive: {self.gdrive_url}")
            logger.info("This may take a few minutes...")
            
            output_file = str(zip_path)
            gdown.download(self.gdrive_url, output_file, quiet=False)
            
            logger.info("✅ Download complete! Extracting files...")
            
            # Extract the zip file
            self._extract_zip(zip_path)
            
            # Clean up zip file
            if zip_path.exists():
                logger.info("Cleaning up zip file...")
                zip_path.unlink()
            
            logger.info("✅ EmbodiedScan-v2 dataset downloaded and extracted successfully!")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error downloading EmbodiedScan-v2: {e}")
            logger.info("\n📥 Manual Download Instructions:")
            logger.info("=" * 60)
            logger.info("1. Download from: https://drive.google.com/file/d/13EZC2wB_aEQFJJDrOWyWBmojOdLCwpDB/view?usp=sharing")
            logger.info(f"2. Extract the zip file to: {self.dataset_dir}")
            logger.info("=" * 60)
            return False
    
    def _check_existing_files(self):
        """Check if dataset files already exist."""
        # Common expected files in EmbodiedScan dataset
        expected_patterns = ["*.json", "*.pkl"]
        for pattern in expected_patterns:
            if list(self.dataset_dir.glob(pattern)):
                return True
        return False
    
    def _extract_zip(self, zip_path):
        """Extract zip file."""
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_dir)
        logger.info("Extraction complete!")
    
    def verify_download(self):
        """Verify the downloaded dataset structure."""
        logger.info("Verifying EmbodiedScan-v2 dataset structure...")
        
        # Check for any JSON or PKL files
        json_files = list(self.dataset_dir.glob("*.json"))
        pkl_files = list(self.dataset_dir.glob("*.pkl"))
        
        if not json_files and not pkl_files:
            logger.warning("No dataset files found!")
            return False
        
        # Count files
        import json
        try:
            total_samples = 0
            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_samples += len(data)
                            logger.info(f"  {json_file.name}: {len(data)} samples")
                        elif isinstance(data, dict):
                            logger.info(f"  {json_file.name}: {len(data)} entries")
                        else:
                            logger.info(f"  {json_file.name}: loaded successfully")
                except Exception as e:
                    logger.warning(f"  Could not read {json_file.name}: {e}")
            
            logger.info(f"✅ Found {len(json_files)} JSON files and {len(pkl_files)} PKL files")
            if total_samples > 0:
                logger.info(f"✅ Total samples: {total_samples}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying dataset files: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download EmbodiedScan-v2 dataset")
    parser.add_argument("--output-dir", help="Output directory for dataset")
    args = parser.parse_args()
    
    downloader = EmbodiedScanDownloader(args.output_dir)
    
    if downloader.download():
        downloader.verify_download()
        print("\n🎉 EmbodiedScan-v2 dataset ready for spatial relationship QA!")
        print(f"📁 Location: {downloader.dataset_dir}")
        print("\n📊 Dataset provides:")
        print("  • Spatial relationship descriptions")
        print("  • Object instance annotations")
        print("  • Natural language scene understanding")
    else:
        print("\n❌ EmbodiedScan-v2 download failed. Check logs for details.")

if __name__ == "__main__":
    main()