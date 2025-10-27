#!/usr/bin/env python3
"""
Taskonomy Dataset Downloader
Downloads Stanford Taskonomy dataset using omnidata-tools.

Repository: https://github.com/StanfordVL/taskonomy
Website: http://taskonomy.stanford.edu/
Paper: https://arxiv.org/abs/1804.08328

Download method: Uses omnidata-tools package
Installation: pip install omnidata-tools
Also requires: aria2 (sudo apt-get install aria2)
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskonomyDownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "taskonomy_dataset"
        
        # Taskonomy info
        self.github_repo = "https://github.com/StanfordVL/taskonomy"
        self.website = "http://taskonomy.stanford.edu/"
        
        # Available domains (tasks)
        self.domains = [
            "rgb",                  # RGB images
            "depth_euclidean",      # Euclidean depth
            "depth_zbuffer",        # Z-buffer depth
            "normal",               # Surface normals
            "reshading",            # Reshading
            "principal_curvature",  # Principal curvature
            "edge_texture",         # Texture edges
            "edge_occlusion",       # Occlusion edges
            "keypoints2d",          # 2D keypoints
            "keypoints3d",          # 3D keypoints
            "segment_semantic",     # Semantic segmentation
            "segment_unsup2d",      # 2D unsupervised segmentation
            "segment_unsup25d",     # 2.5D unsupervised segmentation
            "class_object",         # Object classification
            "class_scene",          # Scene classification
            "point_info",           # Point information
            "nonfixated_matches",   # Non-fixated matches
        ]
    
    def check_dependencies(self):
        """Check if required tools are installed."""
        logger.info("Checking dependencies...")
        
        # Check for aria2
        has_aria2 = subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0
        
        if not has_aria2:
            logger.warning("⚠️  aria2 not found. Install with:")
            logger.warning("     sudo apt-get install aria2  (Ubuntu/Debian)")
            logger.warning("     brew install aria2          (macOS)")
            return False
        
        # Check for omnidata-tools
        try:
            result = subprocess.run(
                ["omnitools.download", "--help"],
                capture_output=True,
                timeout=5
            )
            has_omnitools = result.returncode == 0
        except:
            has_omnitools = False
        
        if not has_omnitools:
            logger.warning("⚠️  omnidata-tools not found. Install with:")
            logger.warning("     pip install omnidata-tools")
            return False
        
        logger.info("✅ All dependencies found (aria2, omnidata-tools)")
        return True
    
    def download_dataset(self, subset="fullplus", components="taskonomy", connections=40, agree=False):
        """
        Download Taskonomy dataset using omnidata-tools.
        
        Official command:
        omnitools.download all --components taskonomy --subset fullplus \
            --dest ./taskonomy_dataset/ \
            --connections_total 40 --agree
        
        Args:
            subset: Dataset subset - 'tiny', 'medium', 'full', 'fullplus', or 'debug' (default: fullplus)
            components: Components to download (default: taskonomy)
            connections: Number of parallel connections (default: 40)
            agree: Auto-agree to terms of service (default: False, will prompt)
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*70)
        logger.info("TASKONOMY DATASET DOWNLOAD")
        logger.info("="*70)
        logger.info(f"Subset: {subset}")
        logger.info(f"Components: {components}")
        logger.info(f"Connections: {connections}")
        logger.info(f"Output: {self.dataset_dir}")
        logger.info("="*70)
        
        # Dataset size information
        size_info = {
            "debug": "~100MB (for testing)",
            "tiny": "~5GB (small sample)",
            "medium": "~50GB (moderate subset)",
            "full": "~500GB (complete dataset)",
            "fullplus": "~500GB+ (full with additional data)"
        }
        
        logger.info(f"\n📦 Dataset size: {size_info.get(subset, 'Unknown')}")
        logger.info("⏱️  Download time depends on your connection speed")
        logger.info("💾 Ensure you have sufficient disk space\n")
        
        if not agree:
            logger.info("⚠️  You will be prompted to agree to the Taskonomy Terms of Service")
            logger.info("   Visit: http://taskonomy.stanford.edu/")
            logger.info("")
        
        try:
            # Build omnitools command
            cmd = [
                "omnitools.download",
                "all",
                "--components", components,
                "--subset", subset,
                "--dest", str(self.dataset_dir),
                "--connections_total", str(connections)
            ]
            
            if agree:
                cmd.append("--agree")
            
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info("")
            
            # Run with live output
            process = subprocess.Popen(
                cmd,
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
                logger.info(f"\n✅ Taskonomy download completed successfully")
                return True
            else:
                logger.error(f"\n❌ Download failed with return code {process.returncode}")
                return False
                
        except FileNotFoundError:
            logger.error("\n❌ omnitools.download command not found")
            logger.error("   Please install: pip install omnidata-tools")
            return False
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Download interrupted by user")
            return False
        except Exception as e:
            logger.error(f"\n❌ Error during download: {e}")
            return False
    
    def verify_download(self):
        """Verify the downloaded dataset structure."""
        logger.info("Verifying Taskonomy dataset structure...")
        
        if not self.dataset_dir.exists():
            logger.warning(f"⚠️  Dataset directory not found: {self.dataset_dir}")
            return False
        
        # Check for domain directories (rgb, depth_euclidean, etc.)
        domain_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        
        if not domain_dirs:
            logger.warning("⚠️  No domain directories found")
            logger.info("   Expected: rgb, depth_euclidean, segment_semantic, etc.")
            return False
        
        logger.info(f"✅ Found {len(domain_dirs)} domain directories: {', '.join([d.name for d in domain_dirs[:5]])}...")
        
        # Check for taskonomy subdirectory and buildings
        found_buildings = set()
        for domain_dir in domain_dirs:
            taskonomy_dir = domain_dir / "taskonomy"
            if taskonomy_dir.exists():
                buildings = [b.name for b in taskonomy_dir.iterdir() if b.is_dir()]
                found_buildings.update(buildings)
        
        if found_buildings:
            logger.info(f"✅ Found {len(found_buildings)} buildings: {', '.join(list(found_buildings)[:5])}...")
            
            # Check for actual data files in a sample location
            sample_domain = domain_dirs[0]
            sample_taskonomy = sample_domain / "taskonomy"
            if sample_taskonomy.exists():
                sample_building = next(sample_taskonomy.iterdir(), None)
                if sample_building:
                    files = list(sample_building.glob("*"))
                    logger.info(f"✅ Sample location has {len(files)} files")
            
            return True
        else:
            logger.warning("⚠️  No building data found in taskonomy subdirectories")
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Stanford Taskonomy dataset using omnidata-tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download debug subset (small, for testing)
  python taskonomy_downloader.py --subset debug
  
  # Download tiny subset
  python taskonomy_downloader.py --subset tiny
  
  # Download full dataset (default: fullplus)
  python taskonomy_downloader.py --agree
  
  # Specify output directory
  python taskonomy_downloader.py --output-dir /path/to/data --subset medium

Official command format:
  omnitools.download all --components taskonomy --subset fullplus \\
      --dest ./taskonomy_dataset/ \\
      --connections_total 40 --agree

Prerequisites:
  1. Install aria2: sudo apt-get install aria2
  2. Install omnidata-tools: pip install omnidata-tools

Dataset sizes:
  - debug: ~100MB (for testing)
  - tiny: ~5GB (small sample)
  - medium: ~50GB (moderate subset)
  - full: ~500GB (complete dataset)
  - fullplus: ~500GB+ (full with additional data)

Note: Download time varies based on connection speed and subset size.
        """
    )
    
    parser.add_argument("--output-dir", help="Output directory (default: ../raw_data/taskonomy_dataset)")
    parser.add_argument("--subset", default="fullplus",
                       choices=["debug", "tiny", "medium", "full", "fullplus"],
                       help="Dataset subset to download (default: fullplus)")
    parser.add_argument("--components", default="taskonomy",
                       help="Components to download (default: taskonomy)")
    parser.add_argument("--connections", type=int, default=40,
                       help="Number of parallel download connections (default: 40)")
    parser.add_argument("--agree", action="store_true",
                       help="Auto-agree to terms of service (default: will prompt)")
    
    args = parser.parse_args()
    
    downloader = TaskonomyDownloader(args.output_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info("TASKONOMY DATASET DOWNLOADER")
    logger.info(f"{'='*70}")
    logger.info(f"Repository: {downloader.github_repo}")
    logger.info(f"Website: {downloader.website}")
    logger.info(f"Output: {downloader.dataset_dir}")
    logger.info(f"{'='*70}\n")
    
    # Check dependencies
    if not downloader.check_dependencies():
        logger.error("\n❌ Please install missing dependencies first:")
        logger.info("  1. sudo apt-get install aria2")
        logger.info("  2. pip install omnidata-tools")
        return 1
    
    # Download
    if downloader.download_dataset(
        subset=args.subset,
        components=args.components,
        connections=args.connections,
        agree=args.agree
    ):
        downloader.verify_download()
        print("\n🎉 Taskonomy dataset download completed!")
        print(f"📁 Location: {downloader.dataset_dir}")
        print("\n📊 Dataset contains:")
        print("  • RGB images")
        print("  • Depth maps")
        print("  • Semantic segmentation")
        print("  • Camera parameters")
        print("  • Multi-task annotations")
        print("  • 34 different building locations")
        print("\n⚠️  Next steps:")
        print("  1. Run taskonomy_processor.py to extract 3D bounding boxes")
        print("  2. Run Enhanced CLIP pipeline to add semantic labels:")
        print("     - build_enhanced_codebook_v2.py")
        print("     - create_labeled_dataset.py")
        print("  3. Then use in QA generation pipeline")
        return 0
    else:
        print("\n❌ Taskonomy download failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit(main())
