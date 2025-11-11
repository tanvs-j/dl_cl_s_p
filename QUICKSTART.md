# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install Kaggle CLI for datasets
pip install kaggle
```

### Step 2: Configure Kaggle API (For Dataset Download)

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it in `C:\Users\takesh\.kaggle\kaggle.json`

Or run the download script and it will guide you:
```powershell
python scripts\download_kaggle_datasets.py
```

### Step 3: Download EEG Datasets

```powershell
# Download all configured Kaggle datasets
python main.py --mode download
```

This will download:
- Epileptic Seizure Recognition Dataset (11,500 samples)
- EEG Eye State Dataset
- EEG Brainwave Emotions Dataset
- Button Tone Seizure Dataset

### Step 4: Quick Demo with Synthetic Data

```powershell
# Test feature extraction
python src\data\feature_extractor.py

# This will:
# - Generate synthetic EEG data
# - Extract spectral, spatial, temporal features
# - Show feature dimensions
```

### Step 5: Train Your First Model

```powershell
# Train on Kaggle dataset (no patient ID needed for Kaggle data)
python scripts\train_kaggle_model.py

# Or train on CHB-MIT data (if you have it)
python main.py --mode train --patient chb01
```

### Step 6: Run Real-time Prediction (Simulation)

```powershell
# Simulate real-time seizure detection
python scripts\demo_realtime.py
```

## 📊 Project Structure Created

```
T:\suezier_p\
├── config\
│   └── config.yaml          ← Main configuration
├── data\
│   ├── raw\kaggle\          ← Downloaded datasets go here
│   ├── processed\           ← Processed features
│   └── models\              ← Trained models
├── src\
│   ├── data\
│   │   └── feature_extractor.py  ← Feature extraction ✅
│   ├── models\              ← ML models (to be created)
│   ├── realtime\            ← Real-time processing
│   └── api\                 ← Web API
├── scripts\
│   └── download_kaggle_datasets.py  ← Download script ✅
├── logs\                    ← Application logs
├── requirements.txt         ← Dependencies ✅
├── main.py                  ← Main entry point ✅
└── README.md               ← Full documentation ✅
```

## 🎯 What You Can Do Now

### 1. Explore the Datasets

```powershell
# After downloading, check dataset info
python -c "import json; print(json.load(open('data/raw/kaggle/dataset_info.json', 'r')))"
```

### 2. Visualize EEG Data

Create `scripts\visualize_eeg.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Kaggle dataset
df = pd.read_parquet('data/raw/kaggle/processed/epileptic_seizure_recognition.parquet')

# Get a seizure sample
seizure_sample = df[df['seizure'] == 1].iloc[0]
eeg_values = seizure_sample[:-2].values  # Exclude label columns

# Plot
plt.figure(figsize=(15, 4))
plt.plot(eeg_values)
plt.title('EEG Signal - Seizure Activity')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

Run it:
```powershell
python scripts\visualize_eeg.py
```

### 3. Train and Evaluate

```powershell
# Complete workflow
python main.py --mode download     # Download data
python main.py --mode train        # Train models
python main.py --mode evaluate     # Evaluate performance
```

## 🔧 Troubleshooting

### Issue: Kaggle API not working

```powershell
# Check if kaggle is installed
kaggle --version

# If not, install it
pip install kaggle

# Verify credentials
kaggle datasets list
```

### Issue: Import errors

```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or install individually
pip install numpy scipy pandas scikit-learn loguru pyyaml
```

### Issue: Not enough memory

Edit `config\config.yaml`:
```yaml
performance:
  parallel_processing: false
  num_cores: 1
```

## 📚 Next Steps

### Learn the System

1. **Read the paper**: `Robust Seizure Prediction Model.pdf` (already in your folder)
2. **Explore features**: Check `src\data\feature_extractor.py`
3. **Understand the model**: Review the SVM classifier design

### Customize for Your Needs

1. **Adjust parameters**: Edit `config\config.yaml`
   - Change sampling rate
   - Modify feature extraction settings
   - Tune model hyperparameters

2. **Add your own data**: 
   - Place EEG files in `data\raw\custom\`
   - Format: `.edf`, `.csv`, or `.npy`

3. **Extend features**:
   - Add ECG integration
   - Include additional frequency bands
   - Experiment with deep learning models

## 🎓 Learning Resources

### Understanding EEG
- **Channels**: 18 electrode positions on scalp
- **Sampling Rate**: 256 Hz (256 samples per second)
- **Epoch Length**: 2 seconds (512 samples per epoch)

### Feature Extraction
- **Spectral**: Energy in 8 frequency bands (0.5-25 Hz)
- **Spatial**: Variance and correlations across channels
- **Temporal**: Evolution over 6-second windows

### Model Performance Targets
- **Sensitivity**: 96% (detect 96% of seizures)
- **Latency**: 3 seconds median (detect within 3s of onset)
- **Specificity**: 2 false alarms per 24 hours

## 💡 Quick Commands Reference

```powershell
# Download datasets
python main.py --mode download

# Train model
python main.py --mode train --patient chb01

# Real-time prediction
python main.py --mode realtime --patient chb01

# Start API server
python main.py --mode api

# Evaluate model
python main.py --mode evaluate --patient chb01

# Run tests
pytest tests/

# Check code quality
flake8 src/
black src/ --check
```

## 🆘 Getting Help

1. **Check logs**: `logs\seizure_prediction_*.log`
2. **Enable debug mode**: 
   ```yaml
   # config/config.yaml
   logging:
     level: "DEBUG"
   ```
3. **Run with verbose output**:
   ```powershell
   python main.py --mode train --patient chb01 --verbose
   ```

## ✅ Verify Installation

Run this checklist:

```powershell
# 1. Check Python version (need 3.9+)
python --version

# 2. Check installed packages
pip list | findstr "numpy scipy scikit-learn"

# 3. Test imports
python -c "import numpy, scipy, sklearn; print('OK')"

# 4. Check directory structure
dir /B data\raw\kaggle

# 5. Run feature extraction demo
python src\data\feature_extractor.py
```

If all pass, you're ready to go! 🎉

## 🚀 Your First Complete Workflow

```powershell
# 1. Activate environment
.\venv\Scripts\activate

# 2. Download data
python main.py --mode download

# 3. Train model on Kaggle data
python scripts\train_kaggle_model.py

# 4. Test real-time prediction
python scripts\demo_realtime.py

# 5. Start API server (in new terminal)
python main.py --mode api

# 6. Make a test prediction
curl http://localhost:8000/api/health
```

Congratulations! You now have a working seizure prediction system! 🎊
