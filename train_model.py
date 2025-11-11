"""
Train EEGNet1D model for the web app
Simple and fast training on CHB-MIT dataset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import pyedflib
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from models.network import EEGNet1D
from app.preprocess import preprocess_for_model

print("=" * 80)
print("🧠 TRAINING SEIZURE PREDICTION MODEL")
print("=" * 80)

# Configuration
TRAIN_DIR = Path("T:/suezier_p/dataset/training")
VAL_DIR = Path("T:/suezier_p/dataset/validation")
MODEL_SAVE_PATH = Path("models/checkpoints/best.pt")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

# Known seizure annotations from CHB-MIT dataset
SEIZURE_ANNOTATIONS = {
    'chb01_03': [(2996, 3036)],
    'chb01_04': [(1467, 1494)],
    'chb01_15': [(1732, 1772)],
    'chb01_16': [(1015, 1066)],
    'chb01_18': [(1720, 1810)],
    'chb01_21': [(327, 420)],
    'chb01_26': [(1862, 1963)],
}


class EEGDataset(Dataset):
    def __init__(self, edf_dir, max_files=10):
        self.windows = []
        self.labels = []
        
        edf_files = list(edf_dir.glob("*.edf"))
        # Filter out annotation files
        edf_files = [f for f in edf_files if 'seizures' not in f.stem.lower()]
        
        print(f"\n📂 Loading data from {edf_dir.name}...")
        print(f"   Found {len(edf_files)} EDF files")
        
        # Limit files for faster training
        edf_files = edf_files[:max_files]
        print(f"   Processing {len(edf_files)} files...")
        
        for edf_file in tqdm(edf_files, desc="Processing files"):
            try:
                self._load_edf(edf_file)
            except Exception as e:
                print(f"\n   ⚠️ Error loading {edf_file.name}: {e}")
                continue
        
        print(f"\n✓ Loaded {len(self.windows)} windows")
        print(f"   Seizure windows: {sum(self.labels)}")
        print(f"   Normal windows: {len(self.labels) - sum(self.labels)}")
        
        # Add synthetic seizures if needed
        if sum(self.labels) < 50:
            print("\n   Adding synthetic seizure patterns...")
            self._add_synthetic_seizures()
    
    def _load_edf(self, edf_file):
        """Load and process EDF file."""
        # Read EDF
        with pyedflib.EdfReader(str(edf_file)) as f:
            n_channels = f.signals_in_file
            sfreq = f.getSampleFrequency(0)
            
            # Read all channels
            signals = []
            for i in range(min(n_channels, 23)):  # Use up to 23 channels
                try:
                    signal = f.readSignal(i)
                    signals.append(signal)
                except:
                    continue
            
            if len(signals) < 10:
                return  # Skip if too few channels
            
            # Stack into array (channels x samples)
            data = np.array(signals)
            
            # Resample if needed
            if abs(sfreq - 256.0) > 1:
                from scipy import signal as sp_signal
                num_samples = int(data.shape[1] * 256.0 / sfreq)
                data = sp_signal.resample(data, num_samples, axis=1)
                sfreq = 256.0
        
        # Preprocess and window
        try:
            windows = preprocess_for_model(data, sfreq, win_sec=10.0, step_sec=5.0)
        except Exception as e:
            print(f"\n   Preprocessing failed for {edf_file.name}: {e}")
            return
        
        if windows.size == 0:
            return
        
        # Get seizure annotations
        base_name = edf_file.stem
        seizure_times = SEIZURE_ANNOTATIONS.get(base_name, [])
        
        # Label windows
        for i, window in enumerate(windows):
            window_start_time = i * 5.0  # 5 second overlap
            window_end_time = window_start_time + 10.0
            
            # Check if window overlaps with seizure
            is_seizure = False
            for seizure_start, seizure_end in seizure_times:
                if window_start_time < seizure_end and window_end_time > seizure_start:
                    is_seizure = True
                    break
            
            self.windows.append(window)
            self.labels.append(1 if is_seizure else 0)
    
    def _add_synthetic_seizures(self):
        """Add synthetic seizure patterns for better training."""
        if len(self.windows) == 0:
            return
        
        normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
        if len(normal_indices) == 0:
            return
        
        # Add 100 synthetic seizures
        num_synthetic = min(100, len(normal_indices))
        
        import random
        seizure_indices = random.sample(normal_indices, num_synthetic)
        
        for idx in seizure_indices:
            window = self.windows[idx].copy()
            
            # Add seizure-like pattern
            freq = 5 + random.random() * 10  # 5-15 Hz
            t = np.linspace(0, 10.0, window.shape[1])
            
            # Affect multiple channels
            n_affected = random.randint(4, min(12, window.shape[0]))
            affected_channels = random.sample(range(window.shape[0]), n_affected)
            
            for ch in affected_channels:
                amplitude = 50 + random.random() * 100
                seizure_pattern = amplitude * np.sin(2 * np.pi * freq * t)
                seizure_pattern += np.random.randn(len(t)) * 10
                
                # Blend with original
                window[ch] = 0.3 * window[ch] + 0.7 * seizure_pattern
            
            self.windows[idx] = window
            self.labels[idx] = 1
        
        print(f"   Added {num_synthetic} synthetic seizure windows")
        print(f"   New total - Seizure: {sum(self.labels)}, Normal: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.from_numpy(self.windows[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return window, label


def balance_dataset(dataset):
    """Balance the dataset by downsampling normal class."""
    seizure_count = sum(dataset.labels)
    normal_count = len(dataset.labels) - seizure_count
    
    if normal_count > seizure_count * 5:
        print(f"\n⚖️  Balancing dataset...")
        print(f"   Before - Seizure: {seizure_count}, Normal: {normal_count}")
        
        import random
        seizure_indices = [i for i, l in enumerate(dataset.labels) if l == 1]
        normal_indices = [i for i, l in enumerate(dataset.labels) if l == 0]
        
        # Keep all seizures, downsample normals to 5:1 ratio
        target_normal = min(seizure_count * 5, normal_count)
        normal_indices = random.sample(normal_indices, target_normal)
        
        # Combine and shuffle
        keep_indices = seizure_indices + normal_indices
        random.shuffle(keep_indices)
        
        dataset.windows = [dataset.windows[i] for i in keep_indices]
        dataset.labels = [dataset.labels[i] for i in keep_indices]
        
        print(f"   After  - Seizure: {sum(dataset.labels)}, Normal: {len(dataset.labels) - sum(dataset.labels)}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            # Calculate confusion matrix
            for pred, label in zip(predicted, y):
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1
    
    accuracy = 100.0 * correct / total
    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / len(loader), accuracy, sensitivity, specificity, f1 * 100


def main():
    print("\n🔧 Configuration:")
    print(f"   Training dir: {TRAIN_DIR}")
    print(f"   Validation dir: {VAL_DIR}")
    print(f"   Model save: {MODEL_SAVE_PATH}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    # Load datasets
    train_dataset = EEGDataset(TRAIN_DIR, max_files=20)
    val_dataset = EEGDataset(VAL_DIR, max_files=5)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n❌ Failed to load data!")
        return
    
    # Balance training data
    balance_dataset(train_dataset)
    
    # Get number of channels from first sample
    sample_window, _ = train_dataset[0]
    n_channels = sample_window.shape[0]
    print(f"\n📊 Using {n_channels} channels")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    model = EEGNet1D(in_channels=n_channels, num_classes=2, base=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    print(f"\n🎓 Training...")
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_sens, val_spec, val_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_f1)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"        - Sens: {val_sens:.2f}%, Spec: {val_spec:.2f}%, F1: {val_f1:.2f}%")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'state_dict': model.state_dict(),
                'model_kwargs': {'in_channels': n_channels, 'num_classes': 2, 'base': 16}
            }, MODEL_SAVE_PATH)
            print(f"  💾 Saved best model (F1: {best_f1:.2f}%)")
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best F1 Score: {best_f1:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("\n✓ The web app will now use the trained model!")
    print("✓ Restart the app to load the new model")
    print("=" * 80)


if __name__ == "__main__":
    main()
