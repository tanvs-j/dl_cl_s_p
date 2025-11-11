import numpy as np
from preprocess import preprocess_for_model

# Test with synthetic data
sfreq = 256.0
duration = 30  # seconds
n_channels = 19
n_samples = int(duration * sfreq)

# Generate synthetic EEG data
X = np.random.randn(n_channels, n_samples).astype(np.float32)

print(f"Input shape: {X.shape}")
print(f"Sampling frequency: {sfreq} Hz")
print(f"Duration: {duration} seconds")

try:
    windows = preprocess_for_model(X, sfreq)
    print(f"\nPreprocessing successful!")
    print(f"Output shape: {windows.shape}")
    print(f"Number of windows: {windows.shape[0]}")
    print(f"Channels per window: {windows.shape[1]}")
    print(f"Samples per window: {windows.shape[2]}")
except Exception as e:
    print(f"\nPreprocessing failed with error:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
