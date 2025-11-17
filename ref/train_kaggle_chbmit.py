"""Train CHB-MIT seizure prediction model using KaggleHub dataset.

This script downloads the Kaggle dataset
    adibadea/chbmitseizuredataset
and reuses the existing CHBMITDataset + CNN training pipeline from train.py

Resulting model weights are saved into data/models/ and the default
trained_seizure_model.pth is updated so that the API / web app can use it.
"""

from __future__ import annotations

import sys
import time
import random
from pathlib import Path
from typing import List, Tuple

import kagglehub
import torch
from torch.utils.data import DataLoader

# Make local modules importable
sys.path.insert(0, str(Path(__file__).parent))

from train import (  # type: ignore
    CHBMITDataset,
    balance_dataset,
    train_model,
    MODEL_DIR,
)


def download_chbmit_via_kaggle() -> Path:
    """Download (or locate cached) CHB-MIT dataset via KaggleHub.

    Returns
    -------
    Path
        Root directory where the dataset was downloaded.
    """
    print("Downloading CHB-MIT dataset via KaggleHub (if not cached)...")
    dataset_path = kagglehub.dataset_download("adibadea/chbmitseizuredataset")
    root = Path(dataset_path)
    print(f"Using dataset at: {root}")
    return root


def find_edf_files(root: Path) -> List[Path]:
    """Recursively find all EDF files under the Kaggle dataset root."""
    edf_files = sorted(root.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files under {root}.")
    return edf_files


def split_train_val(
    edf_files: List[Path], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Split EDF file list into train and validation subsets."""
    if not edf_files:
        raise RuntimeError("No EDF files available for splitting.")

    rng = random.Random(seed)
    files = list(edf_files)
    rng.shuffle(files)

    n_val = max(1, int(len(files) * val_ratio))
    val_files = files[:n_val]
    train_files = files[n_val:]

    print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    return train_files, val_files


def build_datasets(train_files: List[Path], val_files: List[Path]):
    """Create CHBMITDataset instances for train and validation."""
    print("\n🔄 Creating training dataset from Kaggle EDF files...")
    train_dataset = CHBMITDataset(train_files)

    if len(train_dataset) == 0:
        raise RuntimeError("No training epochs were created from the Kaggle dataset.")

    print("\n🔄 Creating validation dataset from Kaggle EDF files...")
    val_dataset = CHBMITDataset(val_files, is_validation=True)

    if len(val_dataset) == 0:
        raise RuntimeError("No validation epochs were created from the Kaggle dataset.")

    return train_dataset, val_dataset


def main() -> None:
    start_time = time.time()

    # 1) Download / locate Kaggle dataset
    dataset_root = download_chbmit_via_kaggle()

    # 2) Collect EDF recordings
    edf_files = find_edf_files(dataset_root)
    if not edf_files:
        raise RuntimeError("No EDF files found in Kaggle dataset.")

    # 3) Train/validation split at file level
    train_files, val_files = split_train_val(edf_files, val_ratio=0.2, seed=42)

    # 4) Build PyTorch datasets
    train_dataset, val_dataset = build_datasets(train_files, val_files)

    # 5) Balance training data (downsample majority class)
    balance_dataset(train_dataset, max_ratio=5.0)

    # 6) DataLoaders
    print("\n📦 Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    # 7) Train model using existing pipeline
    model, history = train_model(
        train_loader, val_loader, model_name="cnn", epochs=30
    )

    # 8) Save model weights (including default path used by web app)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    kaggle_model_path = MODEL_DIR / "trained_seizure_model_chbmit_kaggle.pth"
    torch.save(model.state_dict(), kaggle_model_path)
    print(f"\n💾 Kaggle-trained model saved to: {kaggle_model_path}")

    default_model_path = MODEL_DIR / "trained_seizure_model.pth"
    torch.save(model.state_dict(), default_model_path)
    print(f"💾 Updated default model used by API/web app: {default_model_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 KAGGLE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {elapsed / 60:.2f} minutes")
    print(f"Training epochs: {len(train_dataset)}")
    print(f"Validation epochs: {len(val_dataset)}")

    if history["val_acc"]:
        print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
        print(f"Best sensitivity: {max(history['val_sens']):.2f}%")
        print(f"Best F1 score: {max(history['val_f1']):.2f}%")


if __name__ == "__main__":
    main()
