#!/usr/bin/env python3
"""
Hypersim Dataset Downloader
Downloads Apple ML-Hypersim dataset for photorealistic synthetic indoor scenes.

Repository: https://github.com/apple/ml-hypersim
Dataset Paper: https://arxiv.org/abs/2011.02523
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HypersimDownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "Hyperism"
        
        # Hypersim download info
        self.github_repo = "https://github.com/apple/ml-hypersim"
        self.download_base = "https://raw.githubusercontent.com/apple/ml-hypersim/main/code/python/tools"
        
        # Scene metadata URL
        self.metadata_url = "https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        missing_deps = []
        
        # Check Python packages
        try:
            import h5py
        except ImportError:
            missing_deps.append("h5py")
        
        try:
            import pandas
        except ImportError:
            missing_deps.append("pandas")
        
        if missing_deps:
            logger.warning(f"⚠️  Missing Python packages: {', '.join(missing_deps)}")
            logger.info("Install with: pip install " + " ".join(missing_deps))
            return False
        
        logger.info("✅ All dependencies found")
        return True
    
    def clone_repository(self):
        """Clone the ml-hypersim repository."""
        repo_dir = self.output_dir / "ml-hypersim"
        
        if repo_dir.exists():
            logger.info(f"✅ ml-hypersim repository already exists at {repo_dir}")
            return repo_dir
        
        logger.info(f"Cloning ml-hypersim repository to {repo_dir}...")
        
        try:
            subprocess.run(
                ["git", "clone", self.github_repo, str(repo_dir)],
                check=True,
                capture_output=True
            )
            logger.info("✅ Repository cloned successfully")
            return repo_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to clone repository: {e.stderr.decode()}")
            return None
    
    def download_metadata(self):
        """Download scene metadata CSV."""
        logger.info("Downloading scene metadata...")
        
        metadata_path = self.dataset_dir / "metadata_images_split_scene_v1.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(self.metadata_url, timeout=30)
            response.raise_for_status()
            
            with open(metadata_path, 'w') as f:
                f.write(response.text)
            
            logger.info(f"✅ Metadata saved to {metadata_path}")
            return metadata_path
        except Exception as e:
            logger.error(f"❌ Failed to download metadata: {e}")
            return None
    
    def download_scenes(self, scenes=None, downloads_dir=None, decompress_dir=None):
        """
        Download Hypersim scenes using the official download script.
        
        Args:
            scenes: List of scene names (e.g., ['ai_001_001', 'ai_001_002'])
                   If None, downloads ALL scenes (full dataset ~11TB)
            downloads_dir: Directory to store downloads (default: dataset_dir)
            decompress_dir: Directory to decompress scenes (default: same as downloads_dir)
        """
        if downloads_dir is None:
            downloads_dir = self.dataset_dir
        if decompress_dir is None:
            decompress_dir = downloads_dir
        
        downloads_dir = Path(downloads_dir)
        decompress_dir = Path(decompress_dir)
        downloads_dir.mkdir(parents=True, exist_ok=True)
        decompress_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*70)
        logger.info("HYPERSIM DATASET DOWNLOAD")
        logger.info("="*70)
        logger.info(f"Downloads directory: {downloads_dir}")
        logger.info(f"Decompress directory: {decompress_dir}")
        
        # Clone repository if needed
        repo_dir = self.clone_repository()
        if not repo_dir:
            return False
        
        # Build download command
        download_script = repo_dir / "code" / "python" / "tools" / "dataset_download_images.py"
        
        if not download_script.exists():
            logger.error(f"❌ Download script not found at {download_script}")
            return False
        
        if scenes is None:
            logger.warning("⚠️  No scenes specified - will download FULL dataset (~11TB)")
            logger.warning("⚠️  This will take a very long time and require significant storage")
            response = input("Continue with full download? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Download cancelled")
                return False
        else:
            logger.info(f"Scenes to download: {scenes}")
            logger.info(f"Total: {len(scenes)} scenes\n")
        
        # Run the official download script
        try:
            cmd = [
                sys.executable,
                str(download_script),
                "--downloads_dir", str(downloads_dir),
                "--decompress_dir", str(decompress_dir)
            ]
            
            # Add scene names if specified
            if scenes:
                cmd.append("--scene_names")
                cmd.extend(scenes)
            
            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info("This may take a long time...")
            
            result = subprocess.run(
                cmd,
                cwd=str(repo_dir),
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Download completed successfully")
                return True
            else:
                logger.error(f"❌ Download failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during download: {e}")
            return False
        
        # Clone repository if needed
        repo_dir = self.clone_repository()
        if not repo_dir:
            return False
        
        # Download script location
        download_script = repo_dir / "code" / "python" / "tools" / "dataset_download_images.py"
        
        if not download_script.exists():
            logger.error(f"❌ Download script not found at {download_script}")
            logger.error(f"   Please ensure ml-hypersim repository is properly cloned")
            return False
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Running Hypersim download script...")
        logger.info(f"{'='*50}")
        logger.info("This will download ALL scenes from the dataset.")
        logger.info("The download is ~77GB compressed, ~460GB uncompressed.")
        logger.info("Press Ctrl+C to cancel if this is not desired.")
        logger.info("")
        
        try:
            # Run the official download script
            cmd = [
                sys.executable,
                str(download_script),
                "--downloads_dir", str(downloads_dir),
                "--decompress_dir", str(decompress_dir)
            ]
            
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info("")
            
            # Run with live output
            process = subprocess.Popen(
                cmd,
                cwd=str(repo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"\n✅ Hypersim download completed successfully")
                return True
            else:
                logger.error(f"\n❌ Download failed with return code {process.returncode}")
                return False
                
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Download interrupted by user")
            return False
        except Exception as e:
            logger.error(f"\n❌ Error during download: {e}")
            return False
    
    def verify_download(self):
        """Verify the downloaded dataset structure."""
        logger.info("Verifying Hypersim dataset structure...")
        
        if not self.dataset_dir.exists():
            logger.warning(f"⚠️  Dataset directory not found: {self.dataset_dir}")
            return False
        
        # Check for scene directories (ai_XXX_XXX format)
        scene_dirs = [d for d in self.dataset_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('ai_')]
        
        if not scene_dirs:
            logger.warning("⚠️  No scene directories found")
            logger.info("   Scenes should be in format: ai_001_001, ai_001_002, etc.")
            return False
        
        logger.info(f"✅ Found {len(scene_dirs)} scene directories")
        
        # Check structure of first scene
        sample_scene = scene_dirs[0]
        expected_subdirs = ['images', '_detail']
        
        found = []
        for subdir in expected_subdirs:
            if (sample_scene / subdir).exists():
                found.append(subdir)
        
        if found:
            logger.info(f"✅ Scene structure verified: {', '.join(found)}")
            logger.info(f"✅ Sample scene: {sample_scene.name}")
            
            # Check for specific file types
            images_dir = sample_scene / 'images'
            if images_dir.exists():
                hdf5_files = list(images_dir.glob("**/*.hdf5"))
                if hdf5_files:
                    logger.info(f"✅ Found {len(hdf5_files)} HDF5 files in sample scene")
            
            return True
        else:
            logger.warning(f"⚠️  Incomplete scene structure in {sample_scene.name}")
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Apple ML-Hypersim dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location (raw_data/Hyperism)
  python hypersim_downloader.py
  
  # Specify custom directories
  python hypersim_downloader.py --downloads-dir /path/to/downloads --decompress-dir /path/to/scenes
  
  # Use existing repository
  python hypersim_downloader.py --output-dir /mnt/data/raw_data

Official command format:
  python code/python/tools/dataset_download_images.py \\
      --downloads_dir /path/to/downloads \\
      --decompress_dir /path/to/evermotion_dataset/scenes

Note: 
  - Full dataset is ~77GB compressed, ~460GB uncompressed
  - Downloads ALL scenes (461 scenes total)
  - Requires ~500GB free space
  - Download time: several hours depending on connection
        """
    )
    
    parser.add_argument("--output-dir", help="Output directory (default: ../raw_data)")
    parser.add_argument("--downloads-dir", help="Directory for downloaded archives (default: output_dir/Hyperism/downloads)")
    parser.add_argument("--decompress-dir", help="Directory to decompress scenes (default: output_dir/Hyperism)")
    
    args = parser.parse_args()
    
    downloader = HypersimDownloader(args.output_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info("HYPERSIM DATASET DOWNLOADER")
    logger.info(f"{'='*70}")
    logger.info(f"Repository: {downloader.github_repo}")
    logger.info(f"Output: {downloader.dataset_dir}")
    logger.info(f"{'='*70}\n")
    
    # Check dependencies
    if not downloader.check_dependencies():
        logger.error("\n❌ Please install missing dependencies first")
        logger.info("Run: pip install h5py pandas")
        return 1
    
    # Download scenes
    if downloader.download_scenes(
        downloads_dir=args.downloads_dir,
        decompress_dir=args.decompress_dir
    ):
        downloader.verify_download()
        print("\n🎉 Hypersim dataset download completed!")
        print(f"📁 Location: {downloader.dataset_dir}")
        print("\n📊 Dataset contains:")
        print("  • Photorealistic synthetic indoor scenes (461 scenes)")
        print("  • RGB images (HDF5 format)")
        print("  • Depth maps (HDF5 format)")
        print("  • Camera parameters")
        print("  • Semantic instance segmentation")
        print("\n⚠️  Next steps:")
        print("  1. Run hypersim_processor.py to convert to standard format")
        print("  2. Then use in QA generation pipeline")
        return 0
    else:
        print("\n❌ Hypersim download failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit(main())
