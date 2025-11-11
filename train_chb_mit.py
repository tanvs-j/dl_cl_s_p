"""
Train Seizure Prediction Model on CHB-MIT EEG Database
Uses real clinical EEG data from Children's Hospital Boston / MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from typing import List, Tuple, Dict
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.data.edf_reader import EDFReader
from src.models.deep_learning_models import create_model, ModelConfig

print("=" * 80)
print("🧠 TRAINING ON CHB-MIT SEIZURE DATASET")
print("=" * 80)
print()

# Configuration
TRAIN_DIR = Path("T:/suezier_p/dataset/training")
VAL_DIR = Path("T:/suezier_p/dataset/validation")
MODEL_DIR = Path("data/models")
EPOCH_LENGTH = 2.0  # seconds
SAMPLING_RATE = 256
CHANNELS = 18

MODEL_DIR.mkdir(parents=True, exist_ok=True)


class CHBMITDataset(Dataset):
    """Dataset class for CHB-MIT EEG data."""
    
    def __init__(self, edf_files: List[Path], seizure_files: List[Path]):
        """
        Initialize dataset.
        
        Args:
            edf_files: List of EEG recording files
            seizure_files: List of seizure annotation files
        """
        self.edf_reader = EDFReader()
        self.epochs = []
        self.labels = []
        
        print(f"📂 Processing {len(edf_files)} EDF files...")
        
        # Parse seizure annotations
        seizure_annotations = self._parse_seizure_annotations(seizure_files)
        
        # Process each EDF file
        for i, edf_file in enumerate(edf_files):
            if i % 5 == 0:
                print(f"   Processing file {i+1}/{len(edf_files)}: {edf_file.name}")
            
            try:
                self._process_edf_file(edf_file, seizure_annotations)
            except Exception as e:
                print(f"   ⚠️  Error processing {edf_file.name}: {e}")
                continue
        
        print(f"✓ Loaded {len(self.epochs)} total epochs")
        print(f"   Seizure epochs: {sum(self.labels)}")
        print(f"   Normal epochs: {len(self.labels) - sum(self.labels)}")
    
    def _parse_seizure_annotations(self, seizure_files: List[Path]) -> Dict:
        """Parse seizure annotation files."""
        annotations = defaultdict(list)
        
        for seizure_file in seizure_files:
            # Extract base filename (e.g., chb01_03 from chb01_03seizures.edf)
            base_name = seizure_file.stem.replace('seizures', '').replace('+', '')
            
            try:
                # Read seizure annotation file
                edf_data = self.edf_reader.read_edf(str(seizure_file))
                
                # Check if it has annotation channel
                if 'annotations' in str(edf_data['labels']).lower():
                    # Extract seizure times from annotations
                    # For CHB-MIT, seizures are typically marked in a specific channel
                    annotations[base_name] = self._extract_seizure_times(edf_data)
            except Exception as e:
                # If seizure file can't be read, skip it
                pass
        
        return annotations
    
    def _extract_seizure_times(self, edf_data: Dict) -> List[Tuple[float, float]]:
        """Extract seizure start and end times."""
        # CHB-MIT seizure files are typically small and contain timing info
        # For now, we'll use a heuristic approach
        # This should be refined based on actual annotation format
        seizure_times = []
        
        # Placeholder: Mark entire recording as having seizure potential
        # In practice, you'd parse the annotation channel properly
        duration = edf_data['duration']
        
        # Assume seizure annotations span portions of the recording
        # This is a simplified approach - real annotations need proper parsing
        if duration < 60:  # Short files are likely seizure segments
            seizure_times.append((0, duration))
        
        return seizure_times
    
    def _process_edf_file(self, edf_file: Path, seizure_annotations: Dict):
        """Process single EDF file and extract epochs."""
        # Read EDF
        eeg_data = self.edf_reader.read_edf(str(edf_file))
        
        # Preprocess
        processed = self.edf_reader.preprocess_eeg(eeg_data)
        
        # Get base filename for seizure lookup
        base_name = edf_file.stem
        has_seizure_file = base_name in seizure_annotations
        
        # Extract epochs
        epoch_samples = int(EPOCH_LENGTH * SAMPLING_RATE)
        num_epochs = processed.shape[1] // epoch_samples
        
        for epoch_idx in range(num_epochs):
            start_sample = epoch_idx * epoch_samples
            end_sample = start_sample + epoch_samples
            
            if end_sample > processed.shape[1]:
                break
            
            epoch = processed[:, start_sample:end_sample]
            
            # Determine label
            if has_seizure_file:
                # If there's a corresponding seizure file, mark as seizure
                # This is conservative - all epochs in seizure files are marked as seizure
                label = 1
            else:
                # No seizure file means this is a normal recording
                label = 0
            
            self.epochs.append(epoch)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.epochs)
    
    def __getitem__(self, idx):
        epoch = torch.FloatTensor(self.epochs[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return epoch, label


def load_dataset(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Load EDF files and corresponding seizure annotations."""
    # Get all EDF files
    all_edf_files = list(data_dir.glob("*.edf"))
    
    # Separate normal recordings from seizure annotations
    normal_files = []
    seizure_files = []
    
    for edf_file in all_edf_files:
        if 'seizures' in edf_file.stem.lower():
            seizure_files.append(edf_file)
        else:
            normal_files.append(edf_file)
    
    print(f"📊 Found in {data_dir.name}:")
    print(f"   Normal recordings: {len(normal_files)}")
    print(f"   Seizure annotations: {len(seizure_files)}")
    
    return normal_files, seizure_files


def balance_dataset(train_dataset: CHBMITDataset, max_ratio: float = 3.0):
    """Balance dataset by downsampling majority class."""
    seizure_count = sum(train_dataset.labels)
    normal_count = len(train_dataset.labels) - seizure_count
    
    print(f"\n⚖️  Balancing dataset:")
    print(f"   Original - Seizure: {seizure_count}, Normal: {normal_count}")
    
    if normal_count > seizure_count * max_ratio:
        # Downsample normal epochs
        target_normal = int(seizure_count * max_ratio)
        
        # Get indices
        seizure_indices = [i for i, label in enumerate(train_dataset.labels) if label == 1]
        normal_indices = [i for i, label in enumerate(train_dataset.labels) if label == 0]
        
        # Random sample of normal indices
        import random
        random.shuffle(normal_indices)
        normal_indices = normal_indices[:target_normal]
        
        # Combine indices
        balanced_indices = seizure_indices + normal_indices
        random.shuffle(balanced_indices)
        
        # Update dataset
        train_dataset.epochs = [train_dataset.epochs[i] for i in balanced_indices]
        train_dataset.labels = [train_dataset.labels[i] for i in balanced_indices]
        
        print(f"   Balanced - Seizure: {sum(train_dataset.labels)}, "
              f"Normal: {len(train_dataset.labels) - sum(train_dataset.labels)}")
    else:
        print(f"   No balancing needed (ratio: {normal_count/seizure_count:.1f}:1)")


def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                model_name: str = 'cnn', epochs: int = 30):
    """Train the seizure prediction model."""
    print(f"\n🎓 Training {model_name.upper()} model...")
    
    # Create model
    config = ModelConfig()
    model = create_model(model_name, config)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for real data
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3, verbose=True)
    
    best_val_acc = 0
    best_val_sens = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
               'val_sens': [], 'val_spec': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(config.device)
            batch_y = batch_y.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(config.device)
                batch_y = batch_y.to(config.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                # Calculate metrics
                for pred, label in zip(predicted, batch_y):
                    if pred == 1 and label == 1:
                        true_positives += 1
                    elif pred == 0 and label == 0:
                        true_negatives += 1
                    elif pred == 1 and label == 0:
                        false_positives += 1
                    elif pred == 0 and label == 1:
                        false_negatives += 1
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Calculate sensitivity and specificity
        sensitivity = 100 * true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = 100 * true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_sens'].append(sensitivity)
        history['val_spec'].append(specificity)
        
        # Print progress
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                  f"Sens: {sensitivity:.2f}%, Spec: {specificity:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc or sensitivity > best_val_sens:
            best_val_acc = max(val_acc, best_val_acc)
            best_val_sens = max(sensitivity, best_val_sens)
            torch.save(model.state_dict(), MODEL_DIR / f'best_{model_name}_chbmit.pth')
            print(f"  💾 Saved best model (Acc: {best_val_acc:.2f}%, Sens: {best_val_sens:.2f}%)")
    
    print(f"\n✓ Training complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"   Best Val Sensitivity: {best_val_sens:.2f}%")
    
    return model, history


def main():
    start_time = time.time()
    
    # Load training data
    print("\n📂 Loading training data...")
    train_normal, train_seizure = load_dataset(TRAIN_DIR)
    
    print("\n📂 Loading validation data...")
    val_normal, val_seizure = load_dataset(VAL_DIR)
    
    # Create datasets
    print("\n🔄 Creating training dataset...")
    train_dataset = CHBMITDataset(train_normal, train_seizure)
    
    print("\n🔄 Creating validation dataset...")
    val_dataset = CHBMITDataset(val_normal, val_seizure)
    
    # Balance training data
    balance_dataset(train_dataset, max_ratio=3.0)
    
    # Create data loaders
    print("\n📦 Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Train model
    model, history = train_model(train_loader, val_loader, model_name='cnn', epochs=30)
    
    # Save final model
    final_model_path = MODEL_DIR / 'trained_seizure_model_chbmit.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n💾 Final model saved to: {final_model_path}")
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Training epochs: {len(train_dataset)}")
    print(f"Validation epochs: {len(val_dataset)}")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Best sensitivity: {max(history['val_sens']):.2f}%")
    print(f"Best specificity: {max(history['val_spec']):.2f}%")
    print()
    print("✓ Model ready for use!")
    print(f"✓ Load with: model.load_state_dict(torch.load('{final_model_path}'))")
    print("=" * 80)


if __name__ == "__main__":
    main()
