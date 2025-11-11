# 🚀 Installation & Running Guide

## ⚠️ Python Not Found

Python needs to be installed first. Here's how:

## Step 1: Install Python

### Option A: Official Python (Recommended)
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** or **3.10** (not 3.12 yet for best compatibility)
3. **IMPORTANT**: Check "Add Python to PATH" during installation!
4. Install for all users
5. Verify: Open new PowerShell and type `python --version`

### Option B: Microsoft Store (Easier)
1. Open Microsoft Store
2. Search "Python 3.11"
3. Click "Get" to install
4. Verify: `python --version`

## Step 2: Verify Installation

```powershell
# Open NEW PowerShell window
python --version
# Should show: Python 3.11.x or Python 3.10.x
```

## Step 3: Quick Setup & Run

Once Python is installed, run this:

```powershell
# Navigate to project
cd T:\suezier_p

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install minimal dependencies first
pip install numpy scipy loguru pyyaml

# Test feature extraction (works without datasets!)
python src\data\feature_extractor.py
```

## Step 4: Full Installation

```powershell
# Still in activated venv
pip install -r requirements.txt

# This will take 5-10 minutes
# It installs: PyTorch, scikit-learn, pandas, etc.
```

## Step 5: Run Demos

### Demo 1: Feature Extraction (No data needed)
```powershell
python src\data\feature_extractor.py
```
**Expected output:**
- Creates synthetic EEG data
- Extracts spectral features (8 frequency bands)
- Shows feature dimensions
- Takes ~5 seconds

### Demo 2: Deep Learning Models (No data needed)
```powershell
python src\models\deep_learning_models.py
```
**Expected output:**
- Tests 5 model architectures
- Shows parameter counts
- CNN: ~500K, LSTM: ~350K, etc.
- Takes ~30 seconds

### Demo 3: Continual Learning (No data needed)
```powershell
python src\models\continual_learning.py
```
**Expected output:**
- Simulates online learning
- Shows experience replay
- Drift detection in action
- Takes ~15 seconds

## Step 6: Download Real Data (Optional)

```powershell
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle API
# 1. Go to kaggle.com/account
# 2. Create API token
# 3. Save kaggle.json to C:\Users\takesh\.kaggle\

# Download datasets
python scripts\download_kaggle_datasets.py
```

## Step 7: Train Your First Model (With data)

```powershell
# After downloading data
python main.py --mode train
```

---

## 🎯 Quick Demo Without Installation

If you want to see the code structure first:

```powershell
# Just view the files
notepad src\data\feature_extractor.py
notepad src\models\deep_learning_models.py
notepad config\config.yaml
```

---

## 🐛 Troubleshooting

### Issue: "python not found"
**Solution:** Restart PowerShell after Python installation

### Issue: "Activate.ps1 cannot be loaded"
**Solution:** 
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "pip install fails"
**Solution:** Update pip first:
```powershell
python -m pip install --upgrade pip
```

### Issue: "Out of memory during pip install"
**Solution:** Install packages one by one:
```powershell
pip install torch
pip install numpy scipy
pip install scikit-learn pandas
pip install loguru pyyaml
```

### Issue: "PyTorch installation fails"
**Solution:** Use CPU-only version:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 📊 What Each Demo Does

### Feature Extraction Demo
- Generates 10 seconds of synthetic 18-channel EEG
- Applies 8-band filterbank (0.5-25 Hz)
- Extracts spectral + spatial + temporal features
- Shows: Feature shape is (5, 594) - 5 epochs, 594 features each

### Deep Learning Demo
- Creates dummy input: (4, 18, 512) - batch of 4, 18 channels, 512 samples
- Tests each architecture:
  * CNN1D: Spatial feature extraction
  * LSTM: Temporal with attention
  * Hybrid: CNN + LSTM combined
  * Transformer: Self-attention based
  * ResNet: Residual connections
- Shows parameters and output shapes

### Continual Learning Demo
- Simulates 50 online updates
- Experience replay every 5 updates
- Concept drift detection monitoring
- Shows adaptation statistics

---

## ✅ Verification Checklist

After installation, verify everything works:

```powershell
# 1. Python installed
python --version
# Should show: Python 3.10.x or 3.11.x

# 2. Virtual environment created
dir venv
# Should show: Include, Lib, Scripts folders

# 3. Basic packages installed
python -c "import numpy, scipy, loguru; print('✓ Core packages OK')"

# 4. PyTorch installed (optional, for deep learning)
python -c "import torch; print(f'✓ PyTorch {torch.__version__} OK')"

# 5. Project structure
dir src\data
dir src\models
dir config
# Should show all files

# 6. Run basic test
python src\data\feature_extractor.py
# Should complete without errors
```

---

## 🎓 Learning Path

### Day 1: Setup & Understand
1. Install Python ✓
2. Run feature extraction demo
3. Read `src/data/feature_extractor.py`
4. Understand filterbanks and features

### Day 2: Explore Models
1. Run deep learning demo
2. Compare architectures
3. Read `src/models/deep_learning_models.py`
4. Try modifying hyperparameters

### Day 3: Continual Learning
1. Run continual learning demo
2. Understand experience replay
3. Read `src/models/continual_learning.py`
4. Experiment with drift detection

### Day 4: Real Data
1. Setup Kaggle API
2. Download datasets
3. Train first model
4. Evaluate performance

---

## 🚀 Ready to Start!

Once Python is installed, you can run everything in **under 5 minutes**:

```powershell
# The complete quickstart:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy scipy loguru pyyaml torch scikit-learn pandas
python src\data\feature_extractor.py
python src\models\deep_learning_models.py
python src\models\continual_learning.py
```

**That's it!** You'll have a working seizure prediction system! 🧠⚡

---

**Need Help?** Check:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick reference
- `PROJECT_SUMMARY.md` - Everything created
- Python docs: https://docs.python.org/3/
