"""
Comprehensive EEGNet1D Training with Full Seizure Annotations
Trains on CHB-MIT dataset and generates seizure detection report
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
import json

sys.path.insert(0, str(Path(__file__).parent))
from models.network import EEGNet1D
from app.preprocess import preprocess_for_model

print("=" * 80)
print("COMPREHENSIVE MODEL TRAINING")
print("=" * 80)

# Configuration
TRAIN_DIR = Path("T:/suezier_p/dataset/training")
VAL_DIR = Path("T:/suezier_p/dataset/validation") 
MODEL_SAVE_PATH = Path("models/checkpoints/best.pt")
REPORT_PATH = Path("SEIZURE_REPORT.md")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Files WITH seizures (from RECORDS-WITH-SEIZURES)
SEIZURE_FILES = {
    'chb01_03', 'chb01_04', 'chb01_15', 'chb01_16', 'chb01_18', 'chb01_21', 'chb01_26'
}

# Known seizure timings (seconds) from CHB-MIT documentation
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
    def __init__(self, edf_dir, max_files=30, name="Dataset"):
        self.windows = []
        self.labels = []
        self.file_info = []
        self.name = name
        
        edf_files = list(edf_dir.glob("*.edf"))
        edf_files = [f for f in edf_files if 'seizures' not in f.stem.lower()]
        edf_files = sorted(edf_files)[:max_files]
        
        print(f"\nLoading {name}...")
        print(f"   Found {len(edf_files)} EDF files")
        
        for edf_file in tqdm(edf_files, desc=f"Processing {name}"):
            try:
                info = self._load_edf(edf_file)
                if info:
                    self.file_info.append(info)
            except Exception as e:
                print(f"\n   ⚠️ Error loading {edf_file.name}: {e}")
        
        print(f"\n✓ Loaded {len(self.windows)} windows")
        print(f"   Seizure windows: {sum(self.labels)}")
        print(f"   Normal windows: {len(self.labels) - sum(self.labels)}")
        
        if sum(self.labels) < 20 and name == "Training":
            print(f"\n   Adding synthetic seizure patterns...")
            self._add_synthetic_seizures()
    
    def _load_edf(self, edf_file):
        """Load and process EDF file."""
        base_name = edf_file.stem
        has_seizure = base_name in SEIZURE_FILES
        seizure_times = SEIZURE_ANNOTATIONS.get(base_name, [])
        
        # Read EDF
        with pyedflib.EdfReader(str(edf_file)) as f:
            n_channels = f.signals_in_file
            sfreq = f.getSampleFrequency(0)
            nsamples = f.getNSamples()
            duration = nsamples[0] / sfreq if len(nsamples) > 0 else 0
            
            signals = []
            for i in range(min(n_channels, 23)):
                try:
                    signal = f.readSignal(i)
                    signals.append(signal)
                except:
                    continue
            
            if len(signals) < 10:
                return None
            
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
            return None
        
        if windows.size == 0:
            return None
        
        # Track file info
        file_windows_start = len(self.windows)
        seizure_window_count = 0
        
        # Label windows
        for i, window in enumerate(windows):
            window_start_time = i * 5.0
            window_end_time = window_start_time + 10.0
            
            is_seizure = False
            for seizure_start, seizure_end in seizure_times:
                if window_start_time < seizure_end and window_end_time > seizure_start:
                    is_seizure = True
                    seizure_window_count += 1
                    break
            
            self.windows.append(window)
            self.labels.append(1 if is_seizure else 0)
        
        return {
            'filename': edf_file.name,
            'has_seizure': has_seizure,
            'seizure_times': seizure_times,
            'total_windows': len(windows),
            'seizure_windows': seizure_window_count,
            'channels': data.shape[0],
            'duration': duration
        }
    
    def _add_synthetic_seizures(self):
        """Add synthetic seizure patterns."""
        normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
        if len(normal_indices) == 0:
            return
        
        num_synthetic = min(150, len(normal_indices))
        
        import random
        seizure_indices = random.sample(normal_indices, num_synthetic)
        
        for idx in seizure_indices:
            window = self.windows[idx].copy()
            
            freq = 5 + random.random() * 10
            t = np.linspace(0, 10.0, window.shape[1])
            
            n_affected = random.randint(4, min(12, window.shape[0]))
            affected_channels = random.sample(range(window.shape[0]), n_affected)
            
            for ch in affected_channels:
                amplitude = 50 + random.random() * 100
                seizure_pattern = amplitude * np.sin(2 * np.pi * freq * t)
                seizure_pattern += np.random.randn(len(t)) * 10
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
    """Balance dataset."""
    seizure_count = sum(dataset.labels)
    normal_count = len(dataset.labels) - seizure_count
    
    if normal_count > seizure_count * 3:
        print(f"\nBalancing dataset...")
        print(f"   Before - Seizure: {seizure_count}, Normal: {normal_count}")
        
        import random
        seizure_indices = [i for i, l in enumerate(dataset.labels) if l == 1]
        normal_indices = [i for i, l in enumerate(dataset.labels) if l == 0]
        
        target_normal = min(seizure_count * 3, normal_count)
        normal_indices = random.sample(normal_indices, target_normal)
        
        keep_indices = seizure_indices + normal_indices
        random.shuffle(keep_indices)
        
        dataset.windows = [dataset.windows[i] for i in keep_indices]
        dataset.labels = [dataset.labels[i] for i in keep_indices]
        
        print(f"   After  - Seizure: {sum(dataset.labels)}, Normal: {len(dataset.labels) - sum(dataset.labels)}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
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
    total_loss, correct, total = 0, 0, 0
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
            
            for pred, label in zip(predicted, y):
                if pred == 1 and label == 1: tp += 1
                elif pred == 0 and label == 0: tn += 1
                elif pred == 1 and label == 0: fp += 1
                elif pred == 0 and label == 1: fn += 1
    
    accuracy = 100.0 * correct / total
    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / len(loader), accuracy, sensitivity, specificity, f1 * 100


def generate_report(train_dataset, val_dataset, best_metrics):
    """Generate comprehensive seizure detection report."""
    report = f"""# Seizure Prediction Model - Training Report

## Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | {best_metrics['accuracy']:.2f}% |
| **Sensitivity (Recall)** | {best_metrics['sensitivity']:.2f}% |
| **Specificity** | {best_metrics['specificity']:.2f}% |
| **F1 Score** | {best_metrics['f1']:.2f}% |

## Dataset Information

### Training Set
- **Total Windows**: {len(train_dataset)}
- **Seizure Windows**: {sum(train_dataset.labels)}
- **Normal Windows**: {len(train_dataset.labels) - sum(train_dataset.labels)}
- **Files Processed**: {len(train_dataset.file_info)}

### Validation Set
- **Total Windows**: {len(val_dataset)}
- **Seizure Windows**: {sum(val_dataset.labels)}
- **Normal Windows**: {len(val_dataset.labels) - sum(val_dataset.labels)}
- **Files Processed**: {len(val_dataset.file_info)}

## Files with Seizures (CHB-MIT Dataset)

"""
    
    # Add training files info
    report += "### Training Files\n\n"
    report += "| File | Has Seizure | Seizure Times (sec) | Total Windows | Seizure Windows |\n"
    report += "|------|-------------|---------------------|---------------|------------------|\n"
    
    for info in train_dataset.file_info:
        seizure_str = "✅" if info['has_seizure'] else "❌"
        times_str = ", ".join([f"{s}-{e}" for s, e in info['seizure_times']]) if info['seizure_times'] else "N/A"
        report += f"| {info['filename']} | {seizure_str} | {times_str} | {info['total_windows']} | {info['seizure_windows']} |\n"
    
    report += "\n### Validation Files\n\n"
    report += "| File | Has Seizure | Seizure Times (sec) | Total Windows | Seizure Windows |\n"
    report += "|------|-------------|---------------------|---------------|------------------|\n"
    
    for info in val_dataset.file_info:
        seizure_str = "✅" if info['has_seizure'] else "❌"
        times_str = ", ".join([f"{s}-{e}" for s, e in info['seizure_times']]) if info['seizure_times'] else "N/A"
        report += f"| {info['filename']} | {seizure_str} | {times_str} | {info['total_windows']} | {info['seizure_windows']} |\n"
    
    report += f"\n## Status\n\n"
    report += f"- **Model Trained**: ✓\n"
    report += f"- **Model Location**: `{MODEL_SAVE_PATH}`\n"
    report += f"- **Ready for Use**: ✓\n"
    report += f"\n---\n\n"
    report += f"**Generated**: {Path(__file__).name}\n"
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {REPORT_PATH}")


def main():
    print("\nConfiguration:")
    print(f"   Training dir: {TRAIN_DIR}")
    print(f"   Validation dir: {VAL_DIR}")
    print(f"   Model save: {MODEL_SAVE_PATH}")
    print(f"   Batch size: 16")
    print(f"   Epochs: 25")
    
    # Load datasets
    train_dataset = EEGDataset(TRAIN_DIR, max_files=30, name="Training")
    val_dataset = EEGDataset(VAL_DIR, max_files=10, name="Validation")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n❌ Failed to load data!")
        return
    
    # Balance training data
    balance_dataset(train_dataset)
    
    # Get number of channels
    sample_window, _ = train_dataset[0]
    n_channels = sample_window.shape[0]
    print(f"\nUsing {n_channels} channels")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = EEGNet1D(in_channels=n_channels, num_classes=2, base=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    print(f"\nTraining...\n")
    best_f1 = 0
    best_metrics = {}
    
    for epoch in range(25):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_sens, val_spec, val_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_f1)
        
        if epoch % 2 == 0 or epoch == 24:
            print(f"Epoch {epoch+1}/25:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"        - Sens: {val_sens:.2f}%, Spec: {val_spec:.2f}%, F1: {val_f1:.2f}%")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = {
                'accuracy': val_acc,
                'sensitivity': val_sens,
                'specificity': val_spec,
                'f1': val_f1
            }
            torch.save({
                'state_dict': model.state_dict(),
                'model_kwargs': {'in_channels': n_channels, 'num_classes': 2, 'base': 16}
            }, MODEL_SAVE_PATH)
            if epoch % 2 == 0:
                print(f"  💾 Saved best model (F1: {best_f1:.2f}%)")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best F1 Score: {best_f1:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Generate report
    generate_report(train_dataset, val_dataset, best_metrics)
    
    print("\n✓ The web app will now use the trained model!")
    print("✓ Check SEIZURE_REPORT.md for detailed file information")
    print("=" * 80)


if __name__ == "__main__":
    main()
