#!/usr/bin/env python3
"""
Objectron Dataset Downloader
Downloads complete Objectron 3D object detection dataset via HTTP.
"""

import os
import requests
from pathlib import Path
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectronDownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "Objectron"
        self.public_url = "https://storage.googleapis.com/objectron"
        self.categories = ["bike", "book", "bottle", "camera", "cereal_box", "chair", "cup", "laptop", "shoe"]
    
    def download(self, categories=None, split="train", max_workers=4):
        """
        Download Objectron dataset raw files via HTTP.
        
        Args:
            categories: List of categories to download (default: all)
            split: 'train' or 'test' (default: train)
            max_workers: Number of parallel downloads (default: 4)
        """
        logger.info("Starting Objectron dataset download via HTTP...")
        logger.info(f"Split: {split}")
        logger.info(f"Output directory: {self.dataset_dir}")
        
        if not categories:
            categories = self.categories
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"📥 Downloading categories: {categories}")
            return self._download_categories(categories, split, max_workers)
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return False
    
    def _download_file(self, url, output_path):
        """Download a single file with progress bar."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists
            if output_path.exists():
                return True
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _download_categories(self, categories, split="train", max_workers=4):
        """Download specific categories via HTTP."""
        total_success = 0
        total_failed = 0
        
        for category in categories:
            logger.info(f"\n{'='*50}")
            logger.info(f"Downloading category: {category} ({split} split)")
            logger.info(f"{'='*50}")
            
            try:
                # Get index file
                index_url = f"{self.public_url}/v1/index/{category}_annotations_{split}"
                logger.info(f"Fetching index: {index_url}")
                
                response = requests.get(index_url, timeout=10)
                response.raise_for_status()
                
                video_ids = [vid.strip() for vid in response.text.strip().split('\n') if vid.strip()]
                logger.info(f"Found {len(video_ids)} videos in {category}")
                
                # Create download tasks
                download_tasks = []
                for video_id in video_ids:
                    # Video file
                    video_url = f"{self.public_url}/videos/{video_id}/video.MOV"
                    video_path = self.dataset_dir / "videos" / f"{video_id.replace('/', '_')}.MOV"
                    download_tasks.append(('video', video_url, video_path))
                    
                    # Geometry metadata
                    geo_url = f"{self.public_url}/videos/{video_id}/geometry.pbdata"
                    geo_path = self.dataset_dir / "videos" / f"{video_id.replace('/', '_')}_geometry.pbdata"
                    download_tasks.append(('geometry', geo_url, geo_path))
                    
                    # Annotation
                    anno_url = f"{self.public_url}/annotations/{video_id}.pbdata"
                    anno_path = self.dataset_dir / "annotations" / f"{video_id.replace('/', '_')}.pbdata"
                    download_tasks.append(('annotation', anno_url, anno_path))
                
                logger.info(f"Total files to download: {len(download_tasks)}")
                
                # Download in parallel with progress bar
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(self._download_file, url, path): (file_type, path) 
                              for file_type, url, path in download_tasks}
                    
                    with tqdm(total=len(futures), desc=f"{category}", unit="file") as pbar:
                        for future in as_completed(futures):
                            file_type, path = futures[future]
                            try:
                                if future.result():
                                    total_success += 1
                                else:
                                    total_failed += 1
                            except Exception as e:
                                logger.error(f"Error: {e}")
                                total_failed += 1
                            pbar.update(1)
                
                logger.info(f"✅ {category} completed!")
                
            except Exception as e:
                logger.error(f"Error downloading {category}: {e}")
                total_failed += len(video_ids) * 3 if 'video_ids' in locals() else 0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Download Summary:")
        logger.info(f"  Success: {total_success} files")
        logger.info(f"  Failed: {total_failed} files")
        logger.info(f"{'='*50}")
        
        return total_failed == 0
    
    def verify_download(self):
        """Verify the downloaded dataset structure."""
        logger.info("Verifying Objectron dataset structure...")
        
        found_dirs = []
        
        for subdir in ["videos", "annotations", "v1"]:
            path = self.dataset_dir / subdir
            if path.exists():
                found_dirs.append(subdir)
                logger.info(f"✅ Found: {subdir}")
        
        if found_dirs:
            logger.info(f"✅ Objectron dataset verified: {', '.join(found_dirs)}")
            return True
        else:
            logger.warning("⚠️  No Objectron data found")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Objectron dataset via HTTP")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--categories", nargs="+",
                       choices=["bike", "book", "bottle", "camera", "cereal_box", 
                               "chair", "cup", "laptop", "shoe", "all"],
                       default=["all"],
                       help="Categories to download (default: all)")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                       help="Dataset split to download (default: train)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel downloads (default: 4)")
    
    args = parser.parse_args()
    
    downloader = ObjectronDownloader(args.output_dir)
    
    # Handle 'all' categories
    if "all" in args.categories:
        categories = downloader.categories
    else:
        categories = args.categories
    
    logger.info(f"\n{'='*60}")
    logger.info("OBJECTRON DATASET DOWNLOADER (HTTP)")
    logger.info(f"{'='*60}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Output: {downloader.dataset_dir}")
    logger.info(f"{'='*60}\n")
    
    if downloader.download(categories=categories, split=args.split, max_workers=args.workers):
        downloader.verify_download()
        print("\n🎉 Objectron dataset download completed!")
        print(f"📁 Location: {downloader.dataset_dir}")
        print("\n📊 Dataset contains:")
        print("  • Raw videos (.MOV files)")
        print("  • Geometry metadata (.pbdata files)")
        print("  • Annotations (.pbdata files)")
        print("  • 3D bounding boxes")
        return 0
    else:
        print("\n❌ Objectron download failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())