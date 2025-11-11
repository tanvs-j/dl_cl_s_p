"""
Test trained model on seizure and normal files
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.network import EEGNet1D
from app.preprocess import preprocess_for_model
import pyedflib

MODEL_PATH = Path("models/checkpoints/best.pt")
TRAIN_DIR = Path("dataset/training")

# Test files
SEIZURE_FILES = ['chb01_03.edf', 'chb01_16.edf', 'chb01_21.edf']
NORMAL_FILES = ['chb01_01.edf', 'chb01_02.edf', 'chb01_05.edf']

print("=" * 80)
print("MODEL TESTING")
print("=" * 80)

# Load model
print("\nLoading model...")
ckpt = torch.load(MODEL_PATH)
print(f"Model config: {ckpt['model_kwargs']}")

model = EEGNet1D(**ckpt['model_kwargs'])
model.load_state_dict(ckpt['state_dict'])
model.eval()

device = "cpu"
model.to(device)

print(f"Model loaded successfully")
print(f"Channels: {ckpt['model_kwargs']['in_channels']}")

def test_file(filepath, expected_label):
    """Test a single file"""
    print(f"\nTesting: {filepath.name}")
    print(f"Expected: {'SEIZURE' if expected_label == 1 else 'NO SEIZURE'}")
    
    try:
        # Read EDF
        with pyedflib.EdfReader(str(filepath)) as f:
            n_channels = f.signals_in_file
            sfreq = f.getSampleFrequency(0)
            
            signals = []
            for i in range(min(n_channels, 23)):
                try:
                    signal = f.readSignal(i)
                    signals.append(signal)
                except:
                    continue
            
            data = np.array(signals)
            
            # Resample if needed
            if abs(sfreq - 256.0) > 1:
                from scipy import signal as sp_signal
                num_samples = int(data.shape[1] * 256.0 / sfreq)
                data = sp_signal.resample(data, num_samples, axis=1)
                sfreq = 256.0
        
        # Preprocess
        windows = preprocess_for_model(data, sfreq, win_sec=10.0, step_sec=5.0)
        print(f"  Windows shape: {windows.shape}")
        print(f"  Channels: {windows.shape[1]}")
        
        # Predict
        with torch.no_grad():
            x = torch.from_numpy(windows).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        prob_mean = float(probs.mean())
        pred_label = int(prob_mean >= 0.5)
        
        print(f"  Probability: {prob_mean:.3f}")
        print(f"  Predicted: {'SEIZURE' if pred_label == 1 else 'NO SEIZURE'}")
        print(f"  Status: {'✅ CORRECT' if pred_label == expected_label else '❌ WRONG'}")
        
        return pred_label == expected_label
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test seizure files
print("\n" + "=" * 80)
print("TESTING SEIZURE FILES (Should detect SEIZURE)")
print("=" * 80)

seizure_correct = 0
for filename in SEIZURE_FILES:
    filepath = TRAIN_DIR / filename
    if filepath.exists():
        if test_file(filepath, expected_label=1):
            seizure_correct += 1

# Test normal files
print("\n" + "=" * 80)
print("TESTING NORMAL FILES (Should detect NO SEIZURE)")
print("=" * 80)

normal_correct = 0
for filename in NORMAL_FILES:
    filepath = TRAIN_DIR / filename
    if filepath.exists():
        if test_file(filepath, expected_label=0):
            normal_correct += 1

# Summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Seizure files: {seizure_correct}/{len(SEIZURE_FILES)} correct")
print(f"Normal files: {normal_correct}/{len(NORMAL_FILES)} correct")
print(f"Overall: {seizure_correct + normal_correct}/{len(SEIZURE_FILES) + len(NORMAL_FILES)} correct")

if seizure_correct == len(SEIZURE_FILES) and normal_correct == len(NORMAL_FILES):
    print("\n✅ ALL TESTS PASSED! Model is working correctly.")
else:
    print("\n❌ SOME TESTS FAILED! Model needs retraining or debugging.")

print("=" * 80)
