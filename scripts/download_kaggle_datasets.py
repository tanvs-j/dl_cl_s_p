"""
Download and prepare Kaggle EEG datasets for seizure prediction.
"""

import os
import sys
import yaml
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict
import subprocess
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger


class KaggleDatasetDownloader:
    """Download and prepare Kaggle EEG datasets."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize downloader with configuration."""
        self.config = self._load_config(config_path)
        self.kaggle_config = self.config['data']['kaggle']
        self.download_path = Path(self.kaggle_config['download_path'])
        self.download_path.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def check_kaggle_api(self) -> bool:
        """Check if Kaggle API is configured."""
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Kaggle API found: {result.stdout.strip()}")
                return True
            return False
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Install with: pip install kaggle")
            return False
    
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials."""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            logger.warning("Kaggle API credentials not found!")
            logger.info("Please follow these steps:")
            logger.info("1. Go to https://www.kaggle.com/account")
            logger.info("2. Click 'Create New API Token'")
            logger.info("3. Save kaggle.json to ~/.kaggle/")
            logger.info(f"   (Windows: {kaggle_dir})")
            
            # Prompt for manual setup
            username = input("Enter your Kaggle username: ")
            key = input("Enter your Kaggle API key: ")
            
            kaggle_dir.mkdir(exist_ok=True)
            with open(kaggle_json, 'w') as f:
                json.dump({"username": username, "key": key}, f)
            
            # Set permissions (Unix-like systems)
            if os.name != 'nt':
                os.chmod(kaggle_json, 0o600)
            
            logger.success("Kaggle API credentials configured!")
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download a single Kaggle dataset."""
        try:
            dataset_path = self.download_path / dataset_name.replace('/', '_')
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {dataset_name}...")
            
            # Download using Kaggle API
            result = subprocess.run(
                [
                    'kaggle', 'datasets', 'download',
                    '-d', dataset_name,
                    '-p', str(dataset_path),
                    '--unzip'
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.success(f"✓ Downloaded {dataset_name}")
                return True
            else:
                logger.error(f"Failed to download {dataset_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return False
    
    def download_all_datasets(self) -> Dict[str, bool]:
        """Download all configured datasets."""
        results = {}
        
        for dataset_name in self.kaggle_config['datasets']:
            success = self.download_dataset(dataset_name)
            results[dataset_name] = success
        
        return results
    
    def prepare_epileptic_seizure_recognition(self):
        """Prepare the epileptic seizure recognition dataset."""
        dataset_path = self.download_path / "harunshimanto_epileptic-seizure-recognition"
        
        if not dataset_path.exists():
            logger.warning("Dataset not found. Download first.")
            return
        
        logger.info("Preparing Epileptic Seizure Recognition dataset...")
        
        # This dataset contains:
        # - data.csv: 11,500 samples with 179 features each
        # - Columns: 178 EEG values + 1 label (1=seizure, 2-5=non-seizure)
        
        import pandas as pd
        
        data_file = dataset_path / "data.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()[:5]}...")
            
            # Convert labels: 1=seizure, others=non-seizure
            df['seizure'] = (df['y'] == 1).astype(int)
            
            # Save processed data
            processed_path = self.download_path / "processed"
            processed_path.mkdir(exist_ok=True)
            
            df.to_parquet(processed_path / "epileptic_seizure_recognition.parquet")
            logger.success("✓ Dataset prepared and saved")
    
    def prepare_eeg_brainwave_dataset(self):
        """Prepare the EEG brainwave emotion dataset."""
        dataset_path = self.download_path / "birdy654_eeg-brainwave-dataset-feeling-emotions"
        
        if not dataset_path.exists():
            logger.warning("Dataset not found. Download first.")
            return
        
        logger.info("Preparing EEG Brainwave Emotions dataset...")
        
        # This dataset contains emotional states from EEG
        # Can be used for transfer learning or feature validation
        
        import pandas as pd
        
        processed_path = self.download_path / "processed"
        processed_path.mkdir(exist_ok=True)
        
        # Process all CSV files in the dataset
        csv_files = list(dataset_path.glob("*.csv"))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            logger.info(f"{csv_file.name}: {df.shape}")
            
        logger.success("✓ EEG Brainwave dataset prepared")
    
    def generate_dataset_info(self):
        """Generate information about downloaded datasets."""
        info_file = self.download_path / "dataset_info.json"
        
        datasets_info = []
        
        for dataset_dir in self.download_path.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name != "processed":
                info = {
                    "name": dataset_dir.name,
                    "path": str(dataset_dir),
                    "files": [f.name for f in dataset_dir.iterdir()],
                    "size_mb": sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                }
                datasets_info.append(info)
        
        with open(info_file, 'w') as f:
            json.dump(datasets_info, f, indent=2)
        
        logger.info(f"Dataset info saved to {info_file}")
        
        return datasets_info


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Kaggle EEG Dataset Downloader")
    logger.info("=" * 60)
    
    # Initialize downloader
    downloader = KaggleDatasetDownloader()
    
    # Check Kaggle API
    if not downloader.check_kaggle_api():
        logger.error("Please install Kaggle CLI: pip install kaggle")
        return
    
    # Setup credentials if needed
    downloader.setup_kaggle_api()
    
    # Download datasets
    logger.info("\n" + "=" * 60)
    logger.info("Downloading datasets...")
    logger.info("=" * 60 + "\n")
    
    results = downloader.download_all_datasets()
    
    # Show results
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary:")
    logger.info("=" * 60)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status}: {dataset}")
    
    # Prepare datasets
    if any(results.values()):
        logger.info("\n" + "=" * 60)
        logger.info("Preparing datasets...")
        logger.info("=" * 60 + "\n")
        
        downloader.prepare_epileptic_seizure_recognition()
        downloader.prepare_eeg_brainwave_dataset()
        
        # Generate dataset info
        datasets_info = downloader.generate_dataset_info()
        
        logger.info("\n" + "=" * 60)
        logger.info("Datasets Ready:")
        logger.info("=" * 60)
        
        for info in datasets_info:
            logger.info(f"\n{info['name']}:")
            logger.info(f"  Size: {info['size_mb']:.2f} MB")
            logger.info(f"  Files: {len(info['files'])}")
    
    logger.success("\n✓ All done! Datasets are ready for training.")


if __name__ == "__main__":
    main()
