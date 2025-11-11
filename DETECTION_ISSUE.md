# Detection Issue - Root Cause Analysis

## Problem
The app shows seizures for ALL files (both normal and seizure files).

## Root Cause
**Relative scoring approach doesn't work** for this dataset because:

1. **Normal EEG has high variability**
   - Movement artifacts
   - State changes (awake vs sleep)
   - Individual differences

2. **Filtering amplifies artifacts**
   - Bandpass filtering (0.5-40Hz) can amplify certain frequency components
   - What looks like "normal" can become "seizure-like" after filtering

3. **Baseline contamination**
   - Using 10th percentile of the SAME file as baseline
   - If file has artifacts, baseline gets inflated
   - Seizure windows don't stand out relative to their own file's baseline

## Test Results

### With Filtering
```
chb01_01 (NORMAL) - Max=0.91, 99th=0.78, Consec=6
chb01_02 (NORMAL) - Max=0.92, 99th=0.85, Consec=5
chb01_03 (SEIZURE) - Max=0.88, 99th=0.53, Consec=5
chb01_04 (SEIZURE) - Max=0.86, 99th=0.66, Consec=3
```
**Result**: Normal files score HIGHER than seizure files!

### Without Filtering (Raw Data)
```
chb01_01 (NORMAL) - Consec=8, Ratio=4.59%
chb01_02 (NORMAL) - Consec=5, Ratio=4.03%
chb01_03 (SEIZURE) - Consec=7, Ratio=1.39%
chb01_04 (SEIZURE) - Consec=9, Ratio=2.36%
```
**Result**: Normal files have MORE consecutive high windows!

## Why Relative Scoring Fails

When comparing a window to its own file's baseline:
- **Normal file**: Quiet periods vs artifact periods = 5x difference
- **Seizure file**: Normal periods vs seizure periods = 3x difference

The artifacts in normal files create LARGER ratios than actual seizures!

## Solutions

### Option 1: Absolute Thresholds (Recommended)
Use known seizure characteristics from literature:
- Amplitude variance > 2e-9 (squared Volts)
- Line length > 1e-5 (Volts/sample)
- Require 3+ consecutive high windows

### Option 2: Cross-File Normalization
- Compute baseline from KNOWN normal files
- Use this as reference for all files
- Requires labeled training set

### Option 3: Deep Learning (Original Approach)
- Train on balanced dataset
- Learn seizure patterns directly
- But: Requires fixing class imbalance issue

## Immediate Fix

The app needs to either:
1. Use absolute thresholds calibrated from your dataset
2. Allow user to provide a "normal baseline" file
3. Fall back to simple heuristic (as in original `preprocess.py`)

## Current Status

The v2.0 app with "relative baseline" approach **does not work** for this dataset.
Recommend reverting to simple heuristic or implementing absolute threshold approach.
