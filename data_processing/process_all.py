#!/usr/bin/env python3
"""
Master Data Processor
Processes all datasets into unified JSON format.
"""

import argparse
import logging
from pathlib import Path
import sys

from sunrgbd_processor import SUNRGBDProcessor
from matterport_processor import MatterportProcessor
from objectron_processor import ObjectronProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MasterProcessor:
    def __init__(self, raw_data_dir: Path, output_dir: Path):
        """
        Initialize master processor.
        
        Args:
            raw_data_dir: Path to raw_data directory
            output_dir: Path to processed_data directory
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processors = {
            "sunrgbd": lambda: SUNRGBDProcessor(
                self.raw_data_dir / "SUNRGBD",
                self.output_dir / "sunrgbd"
            ),
            "matterport": lambda: MatterportProcessor(
                self.raw_data_dir / "v1" / "scans",
                self.raw_data_dir / "embodiedscan-v2",
                self.output_dir / "matterport"
            ),
            "objectron": lambda: ObjectronProcessor(
                self.raw_data_dir / "Objectron",
                self.output_dir / "objectron"
            )
        }
    
    def process_dataset(self, dataset_name: str) -> bool:
        """
        Process a specific dataset.
        
        Args:
            dataset_name: Name of dataset to process
        
        Returns:
            True if successful
        """
        if dataset_name not in self.processors:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {list(self.processors.keys())}")
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {dataset_name.upper()}")
        logger.info(f"{'='*60}\n")
        
        try:
            processor = self.processors[dataset_name]()
            result = processor.process_all()
            logger.info(f"✅ {dataset_name} processing complete!")
            return True
        except Exception as e:
            logger.error(f"❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all(self, datasets: list = None) -> dict:
        """
        Process all or selected datasets.
        
        Args:
            datasets: List of dataset names to process (None for all)
        
        Returns:
            Dict with success/failure status for each dataset
        """
        if datasets is None:
            datasets = list(self.processors.keys())
        
        logger.info(f"Processing {len(datasets)} datasets: {datasets}")
        
        results = {}
        for dataset_name in datasets:
            results[dataset_name] = self.process_dataset(dataset_name)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        for dataset, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"{status} {dataset}")
        
        logger.info(f"\nTotal: {success_count}/{total_count} datasets processed successfully")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Process VLM 3D datasets into unified JSON format",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw_data",
        help="Path to raw_data directory (default: ../raw_data)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "processed_data",
        help="Path to output directory (default: ../processed_data)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["hypersim", "sunrgbd", "matterport", "objectron", "all"],
        default=["all"],
        help="Datasets to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if "all" in args.datasets:
        datasets = None  # Process all
    else:
        datasets = args.datasets
    
    # Create master processor
    processor = MasterProcessor(args.raw_data_dir, args.output_dir)
    
    # Process datasets
    results = processor.process_all(datasets)
    
    # Exit with error code if any failed
    if not all(results.values()):
        sys.exit(1)
    
    logger.info("\n🎉 All processing complete!")


if __name__ == "__main__":
    main()
