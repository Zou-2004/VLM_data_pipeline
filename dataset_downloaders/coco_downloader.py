#!/usr/bin/env python3
"""
COCO-2017 Dataset Downloader using FiftyOne
Downloads COCO-2017 dataset with detection and segmentation annotations.
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COCODownloader:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "raw_data"
        self.output_dir = Path(output_dir)
        self.coco_dir = self.output_dir / "COCO"
        
        # Set FiftyOne dataset directory to download COCO to our raw_data/COCO folder
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        os.environ['FIFTYONE_DATASET_DIR'] = str(self.coco_dir)
        logger.info(f"COCO dataset directory: {self.coco_dir}")
        
    def download(self, split="train", label_types=None, classes=None, max_samples=None):
        """
        Download COCO-2017 dataset using FiftyOne.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            label_types: List of label types to download (e.g., ['detections', 'segmentations'])
            classes: List of specific classes to download (e.g., ['person', 'car'])
            max_samples: Maximum number of samples to download (None for all)
        """
        try:
            import fiftyone as fo
            import fiftyone.zoo as foz
        except ImportError:
            logger.error("FiftyOne not installed. Please run: pip install fiftyone")
            return False
        
        logger.info(f"{'='*60}")
        logger.info(f"🚀 DOWNLOADING COCO-2017 DATASET")
        logger.info(f"{'='*60}")
        logger.info(f"Split: {split}")
        if label_types:
            logger.info(f"Label types: {label_types}")
        if classes:
            logger.info(f"Classes: {classes}")
        if max_samples:
            logger.info(f"Max samples: {max_samples}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'='*60}")
        
        try:
            # Prepare download arguments
            download_kwargs = {
                "split": split,
            }
            
            if label_types:
                download_kwargs["label_types"] = label_types
            if classes:
                download_kwargs["classes"] = classes
            if max_samples:
                download_kwargs["max_samples"] = max_samples
            
            # Download dataset (FiftyOne manages its own dataset_dir)
            logger.info("Starting download... This may take a while depending on your selection.")
            dataset = foz.load_zoo_dataset("coco-2017", **download_kwargs)
            
            logger.info(f"\n✅ COCO-2017 downloaded successfully!")
            logger.info(f"📊 Dataset info:")
            logger.info(f"   - Samples: {len(dataset)}")
            logger.info(f"   - FiftyOne manages dataset location")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading COCO-2017: {e}")
            return False
    
    def download_full(self):
        """Download full COCO-2017 train and validation sets."""
        logger.info("Downloading FULL COCO-2017 dataset (train + validation)")
        logger.info("⚠️  This will download ~25GB of data!")
        
        success = True
        
        # Download train split
        logger.info("\n1/2 Downloading TRAIN split...")
        if not self.download(split="train"):
            success = False
        
        # Download validation split
        logger.info("\n2/2 Downloading VALIDATION split...")
        if not self.download(split="validation"):
            success = False
        
        return success
    
    def download_subset(self):
        """Download a small subset for testing (50 samples)."""
        logger.info("Downloading COCO-2017 subset (50 validation samples)")
        logger.info("Classes: person, car, chair, bottle, cup")
        
        return self.download(
            split="validation",
            label_types=["detections", "segmentations"],
            classes=["person", "car", "chair", "bottle", "cup"],
            max_samples=50
        )

def main():
    parser = argparse.ArgumentParser(
        description="COCO-2017 Dataset Downloader using FiftyOne",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--output-dir", help="Output directory (default: ../raw_data/COCO)")
    parser.add_argument("--split", default="validation", 
                       choices=["train", "validation", "test"],
                       help="Dataset split to download (default: validation)")
    parser.add_argument("--label-types", nargs="+", 
                       choices=["detections", "segmentations"],
                       help="Label types to download (e.g., detections segmentations)")
    parser.add_argument("--classes", nargs="+",
                       help="Specific classes to download (e.g., person car chair)")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum number of samples to download")
    parser.add_argument("--full", action="store_true",
                       help="Download full COCO-2017 (train + validation, ~25GB)")
    parser.add_argument("--subset", action="store_true",
                       help="Download small subset for testing (50 samples)")
    
    args = parser.parse_args()
    
    downloader = COCODownloader(args.output_dir)
    
    if args.full:
        success = downloader.download_full()
    elif args.subset:
        success = downloader.download_subset()
    else:
        success = downloader.download(
            split=args.split,
            label_types=args.label_types,
            classes=args.classes,
            max_samples=args.max_samples
        )
    
    if success:
        print(f"\n{'='*60}")
        print("🎉 Download complete!")
        print(f"{'='*60}")
        print(f"Dataset location: {downloader.output_dir}")
        print("\n📖 To visualize the dataset:")
        print("   python -c \"import fiftyone as fo; session = fo.launch_app(fo.load_dataset('coco-2017'))\"")
    else:
        print("\n❌ Download failed. Check logs above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
