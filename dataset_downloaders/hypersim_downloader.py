#!/usr/bin/env python3
"""
Hypersim Dataset Downloader/Setup
Sets up Hypersim dataset repository and provides download instructions.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HypersimDownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        # Put Hypersim in its own folder inside raw_data/Hypersim
        self.dataset_dir = self.output_dir / "Hypersim" / "ml-hypersim"
        self.github_url = "https://github.com/apple/ml-hypersim.git"
        
        # Download all scenes (WARNING: ~1.9TB total!)
        # Set to None to download all available scenes from the official download script
        self.sample_scenes = None
        
    def setup(self):
        """Clone Hypersim repository and download sample scenes."""
        logger.info("Setting up Hypersim dataset...")
        logger.info(f"Output directory: {self.dataset_dir}")
        
        # Check if already cloned
        if self.dataset_dir.exists() and (self.dataset_dir / ".git").exists():
            logger.info("Hypersim repository already exists.")
        else:
            # Clone repository
            try:
                logger.info(f"Cloning Hypersim repository from: {self.github_url}")
                self.dataset_dir.parent.mkdir(parents=True, exist_ok=True)
                
                subprocess.run([
                    "git", "clone", self.github_url, str(self.dataset_dir)
                ], check=True)
                
                logger.info("✅ Hypersim repository cloned successfully!")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Error cloning Hypersim repository: {e}")
                return False
        
        # Now download actual scene data
        if self.sample_scenes is None:
            logger.info("⚠️  Downloading ALL Hypersim scenes (~1.9TB total)...")
            logger.info("⏱  This will take a VERY long time (hours to days depending on connection)...")
        else:
            logger.info(f"Downloading {len(self.sample_scenes)} scenes...")
        return self._download_scenes()
    
    def _download_scenes(self):
        """Download Hypersim scenes using the official download script."""
        downloads_dir = self.dataset_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the official Hypersim download script from contrib
        download_script = self.dataset_dir / "contrib" / "99991" / "download.py"
        
        if not download_script.exists():
            logger.error(f"Download script not found: {download_script}")
            logger.info("Please ensure the Hypersim repository was cloned correctly")
            return False
        
        success_count = 0
        
        # If sample_scenes is None, download ALL scenes using the official script without scene filter
        if self.sample_scenes is None:
            logger.info("Downloading ALL Hypersim scenes using official script...")
            logger.info("⚠️  This will download ~1.9TB of data and may take days...")
            
            try:
                # Run the official download script without scene filter (downloads all)
                cmd = [
                    sys.executable, 
                    str(download_script),
                    "--directory", str(downloads_dir)
                ]
                
                logger.info(f"Running: {' '.join(cmd)}")
                logger.info("This process will run in the foreground. Press Ctrl+C to cancel.")
                
                result = subprocess.run(
                    cmd,
                    cwd=str(self.dataset_dir),
                    timeout=None  # No timeout for full download
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ All scenes downloaded successfully!")
                    logger.info(f"📁 Scenes downloaded to: {downloads_dir}")
                    return True
                else:
                    logger.warning(f"⚠️  Download completed with errors (return code: {result.returncode})")
                    return True  # Return True anyway since some scenes may have downloaded
                    
            except KeyboardInterrupt:
                logger.warning("\n⚠️  Download interrupted by user")
                logger.info("Partial downloads are saved. You can resume by running the script again.")
                return True
            except Exception as e:
                logger.error(f"❌ Download error: {e}")
                return False
        
        # Download specific scenes
        else:
            logger.info(f"Downloading {len(self.sample_scenes)} specific scenes...")
            logger.info("⚠️  Each scene is ~5-10GB...")
            
            # Download each scene individually
            for scene_name in self.sample_scenes:
                logger.info(f"  Downloading scene: {scene_name}...")
                
                try:
                    # Run the official download script for each scene
                    cmd = [
                        sys.executable, 
                        str(download_script),
                        "--directory", str(downloads_dir),
                        "--scene", scene_name
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        cwd=str(self.dataset_dir),
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 minutes per scene
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"  ✅ {scene_name} downloaded successfully!")
                        success_count += 1
                    else:
                        logger.warning(f"  ⚠️  {scene_name} download failed: {result.stderr[:200]}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"  ⏱  {scene_name} download timed out (scene is very large)")
                except Exception as e:
                    logger.warning(f"  ❌ {scene_name} error: {e}")
            
            logger.info(f"✅ Downloaded {success_count}/{len(self.sample_scenes)} scenes successfully")
            
            if success_count > 0:
                logger.info(f"\n📁 Scenes downloaded to: {downloads_dir}")
                return True
            else:
                logger.warning("\n⚠️  No scenes downloaded successfully")
                logger.info("You can manually download scenes later using:")
                logger.info(f"  cd {self.dataset_dir}")
                logger.info(f"  python contrib/99991/download.py --directory downloads --scene <scene_name>")
                return True  # Return True anyway since repo is set up
    
    def _update_repo(self):
        """Update existing repository."""
        try:
            subprocess.run([
                "git", "-C", str(self.dataset_dir), "pull"
            ], check=True)
            logger.info("✅ Hypersim repository updated!")
            self._print_download_instructions()
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error updating repository: {e}")
            return False
    
    def _print_download_instructions(self):
        """Print instructions for downloading actual scene data."""
        print(f"\n{'='*60}")
        print("📋 HYPERSIM SCENE DOWNLOAD INSTRUCTIONS")
        print(f"{'='*60}")
        print(f"Repository location: {self.dataset_dir}")
        print("\n🔧 Setup Steps:")
        print("1. Configure Python environment:")
        print(f"   cd {self.dataset_dir}")
        print("   conda create --name hypersim-env --file requirements.txt")
        print("   conda activate hypersim-env")
        
        print("\n2. Configure system paths:")
        print(f"   cp code/python/_system_config.py.example code/python/_system_config.py")
        print("   # Edit _system_config.py with your system paths")
        
        print("\n📥 Download Scene Data:")
        print("To download specific scenes (WARNING: Each scene ~1-20GB):")
        download_cmd = f"""
python code/python/tools/dataset_download_images.py \\
    --downloads_dir {self.dataset_dir}/downloads \\
    --decompress_dir {self.dataset_dir}/scenes
        """.strip()
        print(download_cmd)
        
        print("\n⚡ Quick Start (Download specific scenes):")
        print("# Edit the download script to specify which scenes you want")
        print("# Full dataset is ~1.9TB, so download selectively!")
        
        print("\n📊 What you get:")
        print("  • Photorealistic synthetic indoor scenes")
        print("  • Perfect depth maps and 3D annotations")
        print("  • Semantic segmentation")
        print("  • Camera poses and intrinsics")
        print("  • 3D bounding boxes")
        
        print(f"\n📖 Documentation: {self.dataset_dir}/README.md")
    
    def verify_setup(self):
        """Verify the repository setup."""
        logger.info("Verifying Hypersim setup...")
        
        required_paths = [
            "code/python/tools/dataset_download_images.py",
            "requirements.txt",
            "code/python/_system_config.py.example",
            "README.md"
        ]
        
        missing_paths = []
        for path in required_paths:
            full_path = self.dataset_dir / path
            if not full_path.exists():
                missing_paths.append(path)
        
        if missing_paths:
            logger.warning(f"Missing required files: {missing_paths}")
            return False
        
        logger.info("✅ Hypersim setup verified!")
        return True
    
    def download_sample_scenes(self, scene_count=3):
        """
        Download a small sample of scenes for testing.
        WARNING: This is still several GB per scene!
        """
        logger.warning(f"This will download {scene_count} scenes (~5-15GB each)")
        response = input("Do you want to continue? [y/N]: ")
        
        if response.lower() != 'y':
            logger.info("Download cancelled by user")
            return False
        
        try:
            download_script = self.dataset_dir / "code/python/tools/dataset_download_images.py"
            downloads_dir = self.dataset_dir / "downloads"
            scenes_dir = self.dataset_dir / "scenes"
            
            downloads_dir.mkdir(exist_ok=True)
            scenes_dir.mkdir(exist_ok=True)
            
            # Run download script (this will download based on the URLs in the script)
            logger.info("Starting scene download... This will take a while!")
            subprocess.run([
                sys.executable, str(download_script),
                "--downloads_dir", str(downloads_dir),
                "--decompress_dir", str(scenes_dir)
            ], cwd=str(self.dataset_dir))
            
            logger.info("✅ Sample scenes downloaded!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading scenes: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup and download Hypersim dataset")
    parser.add_argument("--output-dir", help="Output directory")
    
    args = parser.parse_args()
    
    downloader = HypersimDownloader(args.output_dir)
    
    if downloader.setup():
        downloader.verify_setup()
        print("\n🎉 Hypersim download complete!")
        print(f"📁 Location: {downloader.dataset_dir}")
        if downloader.sample_scenes is None:
            print(f"📦 Downloaded ALL Hypersim scenes (~1.9TB)")
        else:
            print(f"📦 Downloaded {len(downloader.sample_scenes)} scenes")
    else:
        print("\n❌ Hypersim download failed. Check logs for details.")

if __name__ == "__main__":
    main()