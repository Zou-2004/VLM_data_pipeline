#!/usr/bin/env python3
"""
Master Dataset Downloader
Downloads all datasets for VLM 3D Data Pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterDownloader:
    def __init__(self, output_dir=None):
        # Auto-detect output directory relative to script location
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        self.downloaders_dir = Path(__file__).parent
        
        # Dataset configurations
        self.datasets = {
            "sunrgbd": {
                "name": "SUN RGB-D",
                "script": "sunrgbd_downloader.py",
            },
            "objectron": {
                "name": "Objectron",
                "script": "objectron_downloader.py",
            },
            "embodiedscan": {
                "name": "EmbodiedScan-v2",
                "script": "embodiedscan_downloader.py",
            },
            "matterport": {
                "name": "Matterport3D",
                "script": "matterport_downloader.py",
            },
            "hypersim": {
                "name": "Hypersim",
                "script": "hypersim_downloader.py",
            },
            "taskonomy": {
                "name": "Taskonomy",
                "script": "taskonomy_downloader.py",
            }
        }
    
    def download_all(self):
        """Download all datasets."""
        logger.info("Starting download of all datasets...")
        
        success_count = 0
        total_count = 0
        
        for dataset_id, config in self.datasets.items():
            if config.get('manual'):
                continue
            total_count += 1
            if self._download_dataset(dataset_id):
                success_count += 1
        
        print(f"\n✅ Downloaded {success_count}/{total_count} datasets successfully")
        return success_count == total_count
    
    def _download_dataset(self, dataset_id):
        """Download a specific dataset."""
        if dataset_id not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_id}")
            return False
        
        config = self.datasets[dataset_id]
        script_path = self.downloaders_dir / config['script']
        
        if not script_path.exists():
            logger.error(f"Downloader script not found: {script_path}")
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 DOWNLOADING: {config['name']}")
        logger.info(f"{'='*60}")
        
        try:
            # Run the specific downloader
            cmd = [sys.executable, str(script_path), "--output-dir", str(self.output_dir)]
            
            # Add dataset-specific arguments
            if dataset_id == "objectron":
                # Download all categories via HTTP
                cmd.extend(["--categories", "all", "--split", "train", "--workers", "4"])
            elif dataset_id == "matterport":
                # Auto-accept TOS and download minimal file types
                cmd.extend(["--auto-accept", "--type", "matterport_camera_poses", "matterport_color_images"])
            # Hypersim and Taskonomy: No default arguments - user must run manually for full download
            
            result = subprocess.run(cmd, check=True)
            logger.info(f"✅ {config['name']} downloaded successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error downloading {config['name']}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {config['name']}: {e}")
            return False
    
    def check_status(self):
        """Check status of all datasets."""
        print(f"\n{'='*60}")
        print("📋 DATASET STATUS")
        print(f"{'='*60}")
        
        # Check each dataset directory (including commented out ones)
        all_datasets = {
            "sunrgbd": {"name": "SUN RGB-D", "dir": "SUNRGBD"},
            "objectron": {"name": "Objectron", "dir": "Objectron"},
            "embodiedscan": {"name": "EmbodiedScan-v2", "dir": "embodiedscan-v2"},
            "matterport": {"name": "Matterport3D", "dir": "Matterport"},
            "hypersim": {"name": "Hypersim", "dir": "Hyperism"},
            "taskonomy": {"name": "Taskonomy", "dir": "taskonomy_dataset"}
        }
        
        downloaded_count = 0
        
        for dataset_id, info in all_datasets.items():
            dataset_dir = self.output_dir / info["dir"]
            config = info
            
            if dataset_dir.exists() and any(dataset_dir.iterdir()):
                # Quick content check
                if dataset_id == "sunrgbd":
                    subdirs = ["kv1", "kv2", "realsense", "xtion"]
                    existing = [d for d in subdirs if (dataset_dir / d).exists()]
                    print(f"✅ {config['name']:20} ({len(existing)}/4 sensors)")
                
                elif dataset_id == "objectron":
                    categories = list((dataset_dir).glob("*"))
                    print(f"✅ {config['name']:20} ({len(categories)} categories)")
                
                elif dataset_id == "embodiedscan":
                    json_files = list(dataset_dir.glob("*.json"))
                    print(f"✅ {config['name']:20} ({len(json_files)} files)")
                
                elif dataset_id == "hypersim":
                    scene_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('ai_')]
                    print(f"✅ {config['name']:20} ({len(scene_dirs)} scenes)")
                
                elif dataset_id == "taskonomy":
                    domain_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                    print(f"✅ {config['name']:20} ({len(domain_dirs)} domains)")
                    
                else:
                    print(f"✅ {config['name']:20}")
                
                downloaded_count += 1
            else:
                print(f"❌ {config['name']:20} (not found)")
        
        print(f"\n📊 Total: {downloaded_count}/7 datasets available")
        return downloaded_count

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Downloader for VLM 3D Data Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("action", nargs="?", default="download", 
                       choices=["download", "check"],
                       help="Action to perform: download or check")
    parser.add_argument("--output-dir", help="Output directory for datasets (default: ../raw_data)")
    
    args = parser.parse_args()
    
    downloader = MasterDownloader(args.output_dir)
    
    if args.action == "check":
        downloader.check_status()
    else:  # download
        downloader.download_all()
        downloader.check_status()

if __name__ == "__main__":
    main()