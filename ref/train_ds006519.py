"""Train a simple classifier on the ds006519 iEEG dataset.

Dataset: BIDS iEEG (negative motor responses) in dataset/ds006519-main

Labeling used here:
- 0 = prestim (baseline epoch before stimulation)
- 1 = poststim (epoch after stimulation)

For each recording, we:
- load the BrainVision file with MNE (.vhdr),
- band-pass filter and notch-filter line noise,
- resample to 256 Hz,
- take sliding windows (2 s, step 1 s) within the 5 s prestim / poststim
  intervals defined in the *_events.tsv files,
- extract features using the existing EEGFeatureExtractor,
- train a binary classifier to separate prestim vs poststim.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Make local src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from src.data.feature_extractor import EEGFeatureExtractor, FeatureConfig

# Optional: use XGBoost if available, otherwise fall back to scikit-learn
try:  # pragma: no cover - optional dependency
    import xgboost as xgb

    HAS_XGB = True
except Exception:  # pragma: no cover - xgboost not installed
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATASET_ROOT = BASE_DIR / "dataset" / "ds006519-main"

WINDOW_SEC = 2.0
STEP_SEC = 1.0
TARGET_SFREQ = 256.0
NUM_CHANNELS = 18  # fixed number of channels to use across all recordings


# ---------------------------------------------------------------------------
# DATA LOADING & FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def find_recordings(root: Path) -> List[Path]:
    """Find all BrainVision header files (.vhdr) in the BIDS iEEG dataset."""
    pattern = "sub-*/ses-*/ieeg/*_task-dcs_ieeg.vhdr"
    files = sorted(root.glob(pattern))
    print(f"Found {len(files)} recordings under {root}.")
    return files


def build_feature_extractor() -> EEGFeatureExtractor:
    """Create an EEGFeatureExtractor configured for ds006519 windows."""
    config = FeatureConfig(
        sampling_rate=int(TARGET_SFREQ),
        epoch_length=WINDOW_SEC,
        num_channels=NUM_CHANNELS,
        window_size=1,
    )
    extractor = EEGFeatureExtractor(config)
    return extractor


def _empty_feature_arrays(extractor: EEGFeatureExtractor) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to create empty X, y arrays with correct feature dimension."""
    feat_dim = extractor.get_feature_dimension(use_temporal=False)
    X_empty = np.empty((0, feat_dim), dtype=float)
    y_empty = np.empty((0,), dtype=int)
    return X_empty, y_empty


def extract_from_recording(
    vhdr_path: Path,
    extractor: EEGFeatureExtractor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract window-level features and labels from one BrainVision recording.

    Returns
    -------
    X_rec : ndarray, shape (n_windows, n_features)
    y_rec : ndarray, shape (n_windows,)
    """
    # Corresponding BIDS events file
    events_path = vhdr_path.with_name(
        vhdr_path.name.replace("_ieeg.vhdr", "_events.tsv")
    )

    if not events_path.exists():
        print(f"  No events TSV for {vhdr_path.name}, skipping.")
        return _empty_feature_arrays(extractor)

    print(f"Processing {vhdr_path.relative_to(DATASET_ROOT)}")

    # Load raw BrainVision iEEG
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Basic preprocessing: band-pass + notch + resample
    raw.filter(l_freq=0.5, h_freq=100.0, fir_design="firwin")
    raw.notch_filter(np.arange(50.0, 251.0, 50.0), filter_length="auto", phase="zero")
    raw.resample(TARGET_SFREQ)

    data = raw.get_data()  # shape: (n_channels, n_samples)
    sfreq = float(raw.info["sfreq"])

    if data.shape[0] < NUM_CHANNELS:
        print(
            f"  Warning: {vhdr_path.name} has only {data.shape[0]} channels (<{NUM_CHANNELS}), skipping."
        )
        return _empty_feature_arrays(extractor)

    # Use a fixed subset of channels to keep feature dimension consistent
    data = data[:NUM_CHANNELS, :]

    # Read BIDS events
    events = pd.read_csv(events_path, sep="\t")

    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _, row in events.iterrows():
        label_str = str(row.get("label", "")).strip().lower()
        if label_str not in {"prestim", "poststim"}:
            # Ignore other event types if present
            continue

        # Map labels: prestim -> 0, poststim -> 1
        y_val = 0 if label_str == "prestim" else 1

        onset = float(row["onset"])  # seconds
        duration = float(row["duration"])  # seconds
        start_sample = int(onset * sfreq)
        end_sample = int((onset + duration) * sfreq)

        # Slide windows completely inside the event interval
        for start in range(start_sample, end_sample - window_samples + 1, step_samples):
            stop = start + window_samples
            epoch = data[:, start:stop]

            # Extract per-epoch features (no temporal stacking)
            feat = extractor.extract_epoch_features(epoch, include_spatial=True)
            X_list.append(feat)
            y_list.append(y_val)

    if not X_list:
        print("  No windows extracted for this recording.")
        return _empty_feature_arrays(extractor)

    X_rec = np.stack(X_list, axis=0)
    y_rec = np.asarray(y_list, dtype=int)

    n_pos = int((y_rec == 1).sum())
    n_neg = int((y_rec == 0).sum())
    print(f"  Extracted {len(y_rec)} windows (prestim={n_neg}, poststim={n_pos})")

    return X_rec, y_rec


def build_dataset(vhdr_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Build full dataset X, y from a list of .vhdr recordings."""
    extractor = build_feature_extractor()

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for idx, vhdr in enumerate(vhdr_files, start=1):
        print(f"[{idx}/{len(vhdr_files)}] {vhdr.name}")
        X_rec, y_rec = extract_from_recording(vhdr, extractor)
        if X_rec.size == 0:
            continue
        X_all.append(X_rec)
        y_all.append(y_rec)

    if not X_all:
        raise RuntimeError("No windows extracted from any recording.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print("\n=== Dataset summary ===")
    print(f"Total windows: {len(y)}")
    print(f"Prestim (0): {int((y == 0).sum())}")
    print(f"Poststim (1): {int((y == 1).sum())}")

    return X, y


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray):
    """Train a binary classifier (prestim vs poststim) and print metrics."""
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need at least two classes (prestim and poststim) to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    print("\n=== Training classifier ===")

    if HAS_XGB:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        evals = [(dtest, "test")]
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        prob_test = bst.predict(dtest)
        model = bst
        print("Using XGBoost classifier.")
    else:
        print("xgboost not installed; using GradientBoostingClassifier instead.")
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_train, y_train)
        prob_test = clf.predict_proba(X_test)[:, 1]
        model = clf

    # Evaluation
    pred = (prob_test > 0.5).astype(int)

    print("\n=== Classification report (test set) ===")
    print(classification_report(y_test, pred, digits=4))

    auc = roc_auc_score(y_test, prob_test)
    print(f"AUC = {auc:.4f}")

    return model


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    print("============================================================")
    print("Training ds006519 prestim/poststim classifier")
    print("Dataset root:", DATASET_ROOT)
    print("============================================================\n")

    vhdr_files = find_recordings(DATASET_ROOT)
    if not vhdr_files:
        raise RuntimeError(f"No BrainVision .vhdr files found under {DATASET_ROOT}")

    X, y = build_dataset(vhdr_files)
    _ = train_model(X, y)


if __name__ == "__main__":
    main()
