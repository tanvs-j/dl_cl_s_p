from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
from scipy.signal import iirnotch
import mne

# Basic preprocessing and windowing for 1D CNN

def preprocess_for_model(X: np.ndarray, sfreq: float, band=(0.5, 40.0), notch=50.0, win_sec=10.0, step_sec=5.0) -> np.ndarray:
    """
    X: channels x samples
    Returns: windows as (N, C, T)
    """
    X = np.asarray(X, dtype=np.float64)  # MNE requires float64
    # Bandpass
    Xf = mne.filter.filter_data(X, sfreq, band[0], band[1], verbose=False)
    # Notch
    try:
        Q = 30.0
        b, a = iirnotch(w0=notch/(sfreq/2.0), Q=Q)
        Xf = mne.filter.filter_data(Xf, sfreq, None, None, method='iir', iir_params={'b': b, 'a': a}, verbose=False)
    except Exception:
        pass
    # Standardize per channel
    Xf = (Xf - Xf.mean(axis=1, keepdims=True)) / (Xf.std(axis=1, keepdims=True) + 1e-6)
    # Windowing
    win = int(win_sec * sfreq)
    step = int(step_sec * sfreq)
    if Xf.shape[1] < win:
        return np.empty((0, Xf.shape[0], 0), dtype=np.float32)
    starts = np.arange(0, Xf.shape[1] - win + 1, step)
    windows = np.stack([Xf[:, s:s+win] for s in starts], axis=0)
    # (N, C, T)
    return windows.astype(np.float32)


def spectral_entropy(x: np.ndarray, sfreq: float) -> float:
    psd, freqs = mne.time_frequency.psd_array_welch(x, sfreq=sfreq, fmin=0.5, fmax=40.0, n_fft=1024, verbose=False)
    psd = psd.sum(axis=0)
    p = psd / (psd.sum() + 1e-9)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def line_length(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(x, axis=-1))))


def simple_heuristic_score(windows: np.ndarray, sfreq: float) -> float:
    """Return [0,1] score indicating seizure likelihood from windows.
    Uses line-length and spectral entropy. Looks for ANY window with seizure-like activity."""
    if windows.size == 0:
        return 0.0
    
    scores = []
    for w in windows:  # (C, T)
        ll = line_length(w)
        se = spectral_entropy(w, sfreq)
        
        # Seizures have: high line length (activity) AND low entropy (rhythmic)
        # Normalize line length: typical normal ~0.5-2.0, seizure ~5-15
        ll_score = np.clip(ll / 10.0, 0.0, 1.0)
        
        # Low entropy (rhythmic) is seizure-like
        # Normal entropy ~0.7-0.9, seizure ~0.3-0.6
        se_score = np.clip((0.9 - se) / 0.6, 0.0, 1.0)
        
        # Combined score: both must be high
        window_score = ll_score * 0.7 + se_score * 0.3
        scores.append(window_score)
    
    # Use percentile instead of mean - if top 10% of windows show seizure, flag it
    top_percentile = np.percentile(scores, 90)
    
    # Also check if many windows exceed threshold
    high_score_count = np.sum(np.array(scores) > 0.3)
    high_score_ratio = high_score_count / len(scores)
    
    # Final score: max of (top percentile, high score ratio)
    final_score = max(top_percentile, high_score_ratio)
    
    return float(np.clip(final_score, 0.0, 1.0))
