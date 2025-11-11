"""
Properly Balanced Training for Seizure Prediction
Ensures model learns to distinguish seizures from normal EEG
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

sys.path.insert(0, str(Path(__file__).parent))
from models.network import EEGNet1D
from app.preprocess import preprocess_for_model

print("=" * 80)
print("BALANCED SEIZURE PREDICTION TRAINING")
print("=" * 80)

TRAIN_DIR = Path("T:/suezier_p/dataset/training")
MODEL_SAVE_PATH = Path("models/checkpoints/best.pt")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Known seizure files and times
SEIZURE_FILES = {
    'chb01_03': [(2996, 3036)],
    'chb01_04': [(1467, 1494)],
    'chb01_15': [(1732, 1772)],
    'chb01_16': [(1015, 1066)],
    'chb01_18': [(1720, 1810)],
    'chb01_21': [(327, 420)],
    'chb01_26': [(1862, 1963)],
}

# Files explicitly WITHOUT seizures
NO_SEIZURE_FILES = [
    'chb01_01', 'chb01_02', 'chb01_05', 'chb01_06', 'chb01_07',
    'chb01_08', 'chb01_09', 'chb01_10', 'chb01_11', 'chb01_12'
]


class BalancedEEGDataset(Dataset):
    def __init__(self, edf_dir):
        self.windows = []
        self.labels = []
        self.file_stats = {}
        
        # Load seizure files
        print("\nLoading SEIZURE files...")
        for filename, seizure_times in SEIZURE_FILES.items():
            edf_path = edf_dir / f"{filename}.edf"
            if edf_path.exists():
                print(f"  Processing {filename}.edf (HAS seizure)")
                self._load_file(edf_path, seizure_times, has_seizure=True)
        
        seizure_count = sum(self.labels)
        print(f"\nSeizure windows loaded: {seizure_count}")
        
        # Load equal number of normal files
        print("\nLoading NORMAL files...")
        normal_windows_needed = seizure_count * 2  # 1:2 ratio
        normal_loaded = 0
        
        for filename in NO_SEIZURE_FILES:
            if normal_loaded >= normal_windows_needed:
                break
            edf_path = edf_dir / f"{filename}.edf"
            if edf_path.exists():
                print(f"  Processing {filename}.edf (NO seizure)")
                before = len(self.windows)
                self._load_file(edf_path, [], has_seizure=False)
                after = len(self.windows)
                normal_loaded += (after - before)
        
        print(f"\n=== Dataset Summary ===")
        print(f"Total windows: {len(self.windows)}")
        print(f"Seizure windows: {sum(self.labels)}")
        print(f"Normal windows: {len(self.labels) - sum(self.labels)}")
        print(f"Ratio: {(len(self.labels) - sum(self.labels)) / max(1, sum(self.labels)):.1f}:1 (normal:seizure)")
        
        # Display file statistics
        print(f"\n=== File Statistics ===")
        for fname, stats in self.file_stats.items():
            status = "SEIZURE" if stats['has_seizure'] else "NORMAL"
            print(f"{fname}: {status} - {stats['total_windows']} windows, {stats['seizure_windows']} seizure")
    
    def _load_file(self, edf_path, seizure_times, has_seizure):
        try:
            # Read EDF
            with pyedflib.EdfReader(str(edf_path)) as f:
                n_channels = f.signals_in_file
                sfreq = f.getSampleFrequency(0)
                
                signals = []
                for i in range(min(n_channels, 23)):
                    try:
                        signal = f.readSignal(i)
                        signals.append(signal)
                    except:
                        continue
                
                if len(signals) < 10:
                    return
                
                data = np.array(signals)
                
                # Resample if needed
                if abs(sfreq - 256.0) > 1:
                    from scipy import signal as sp_signal
                    num_samples = int(data.shape[1] * 256.0 / sfreq)
                    data = sp_signal.resample(data, num_samples, axis=1)
                    sfreq = 256.0
            
            # Preprocess
            windows = preprocess_for_model(data, sfreq, win_sec=10.0, step_sec=5.0)
            
            if windows.size == 0:
                return
            
            # Label windows
            seizure_window_count = 0
            for i, window in enumerate(windows):
                window_start = i * 5.0
                window_end = window_start + 10.0
                
                # Check if window overlaps with seizure
                is_seizure = False
                for seizure_start, seizure_end in seizure_times:
                    if window_start < seizure_end and window_end > seizure_start:
                        is_seizure = True
                        seizure_window_count += 1
                        break
                
                self.windows.append(window)
                self.labels.append(1 if is_seizure else 0)
            
            # Store statistics
            self.file_stats[edf_path.name] = {
                'has_seizure': has_seizure,
                'total_windows': len(windows),
                'seizure_windows': seizure_window_count
            }
            
        except Exception as e:
            print(f"    Error: {e}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.from_numpy(self.windows[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return window, label


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
    
    return total_loss / len(loader), accuracy, sensitivity, specificity, f1 * 100, tp, tn, fp, fn


def main():
    print("\nConfiguration:")
    print(f"  Training dir: {TRAIN_DIR}")
    print(f"  Model save: {MODEL_SAVE_PATH}")
    
    # Load dataset
    dataset = BalancedEEGDataset(TRAIN_DIR)
    
    if len(dataset) < 10:
        print("\nERROR: Not enough data loaded!")
        return
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\nTrain/Val Split:")
    print(f"  Training: {len(train_dataset)} windows")
    print(f"  Validation: {len(val_dataset)} windows")
    
    # Get number of channels
    sample_window, _ = dataset[0]
    n_channels = sample_window.shape[0]
    print(f"  Channels: {n_channels}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = EEGNet1D(in_channels=n_channels, num_classes=2, base=16).to(device)
    
    # Use weighted loss to handle any remaining imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training
    print("\nTraining...\n")
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(30):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_sens, val_spec, val_f1, tp, tn, fp, fn = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1}/30:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"        - Sens: {val_sens:.2f}%, Spec: {val_spec:.2f}%, F1: {val_f1:.2f}%")
        print(f"        - TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}")
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'model_kwargs': {'in_channels': n_channels, 'num_classes': 2, 'base': 16}
            }, MODEL_SAVE_PATH)
            print(f"  >> Saved best model (F1: {best_f1:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        print()
    
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best F1 Score: {best_f1:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("\nThe model is now trained to distinguish:")
    print("  - SEIZURE files: chb01_03, 04, 15, 16, 18, 21, 26")
    print("  - NORMAL files: chb01_01, 02, 05, 06, 07, 08, 09, 10, 11, 12")
    print("\nRestart the web app to use the new model!")
    print("=" * 80)


if __name__ == "__main__":
    main()
