import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess import preprocess_for_model
from inference import InferenceEngine

# Test with synthetic data
sfreq = 256.0
duration = 30  # seconds
n_channels = 19
n_samples = int(duration * sfreq)

# Generate synthetic EEG data
X = np.random.randn(n_channels, n_samples).astype(np.float64)

print(f"Input shape: {X.shape}")
print(f"Sampling frequency: {sfreq} Hz")
print(f"Duration: {duration} seconds")

try:
    # Test preprocessing
    print("\n=== Testing Preprocessing ===")
    windows = preprocess_for_model(X, sfreq)
    print(f"✓ Preprocessing successful!")
    print(f"  Output shape: {windows.shape}")
    print(f"  Number of windows: {windows.shape[0]}")
    print(f"  Channels per window: {windows.shape[1]}")
    print(f"  Samples per window: {windows.shape[2]}")
    
    # Test inference
    print("\n=== Testing Inference ===")
    engine = InferenceEngine()
    pred = engine.predict(windows, sfreq)
    print(f"✓ Inference successful!")
    print(f"  Prediction: {'SEIZURE' if pred['label'] == 1 else 'NO SEIZURE'}")
    print(f"  Probability: {pred['prob']:.3f}")
    print(f"  Model loaded: {pred['model_loaded']}")
    
    if not pred['model_loaded']:
        print("\n  Note: No trained model found, using heuristic fallback.")
        print("  This is expected for first run.")
    
    print("\n=== All Tests Passed! ===")
    
except Exception as e:
    print(f"\n✗ Test failed with error:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
