# Seizure Prediction Web App - Usage Guide

## ✅ Setup Complete!

The preprocessing issue has been fixed. The app is now fully functional.

## 🚀 Running the App

From the project root directory:

```powershell
cd T:\suezier_p\app
python -m streamlit run app.py
```

The app will open at: **http://localhost:8501**

## 📁 Test Files Available

You can test the app with the existing EDF files in the project:

- **Sample file**: `T:\suezier_p\data\samples\sample_eeg.edf`
- **CHB-MIT dataset files**: `T:\suezier_p\dataset\testing\chb01_*.edf`

## 📤 How to Use

1. **Open the app** in your browser (http://localhost:8501)
2. **Click "Browse files"** to upload EEG files
3. **Supported formats**:
   - `.edf` (European Data Format - EEG recordings)
   - `.eeg` (BrainVision format)
   - `.pdf` (Clinical reports - will be parsed for info)

4. **View results**:
   - Prediction: SEIZURE or NO SEIZURE
   - Probability score (0.0 to 1.0)
   - Model status (trained model or heuristic fallback)

5. **Optional - Improve the model**:
   - Expand "Correct label / improve model"
   - Select the correct label
   - Click "Improve model with this file" to enable online learning

## 🔧 How It Works

### First Run (No Trained Model)
- Uses **heuristic-based detection**
- Analyzes signal features:
  - Line length (signal variability)
  - Spectral entropy (frequency distribution)
- Provides baseline predictions

### After Training (With Model)
- Uses **1D-CNN deep learning model**
- Trained on multiple EEG recordings
- More accurate predictions
- Supports online learning for continuous improvement

## 🎯 Model Training

To train a proper model (optional):

```powershell
# From project root
python train_chb_mit_fixed.py
```

This will:
1. Load CHB-MIT dataset from `dataset/` folder
2. Train a deep learning model
3. Save the model to `models/checkpoints/best.pt`
4. Enable better predictions in the web app

## 📊 Understanding Results

### Probability Scores
- **0.0 - 0.3**: Very unlikely to be seizure
- **0.3 - 0.5**: Low likelihood
- **0.5 - 0.7**: Moderate likelihood (threshold for detection)
- **0.7 - 1.0**: High likelihood of seizure

### Model Status
- **Model not loaded**: Using heuristic fallback (simpler algorithm)
- **Model loaded**: Using trained deep learning model (better accuracy)

## 🐛 Troubleshooting

### "Preprocessing failed" error
✅ **FIXED!** The dtype issue has been resolved.

### App crashes on file upload
- Check file format (must be valid .edf or .eeg)
- Ensure file is not corrupted
- Try with sample files first

### Low accuracy predictions
- This is expected without a trained model
- Train the model using real data for better results
- Use the online learning feature to improve over time

## 💡 Tips

1. **Start with sample files** to test the app
2. **Multiple files**: You can upload several files at once
3. **Batch processing**: The app processes each file independently
4. **Model improvement**: Regular corrections help improve accuracy
5. **Check logs**: Error messages show in the app interface

## 🎓 Technical Details

### Preprocessing Pipeline
1. **Bandpass filter**: 0.5-40 Hz
2. **Notch filter**: 50 Hz (power line noise)
3. **Standardization**: Per-channel normalization
4. **Windowing**: 10-second windows, 5-second overlap

### Model Architecture
- **Type**: 1D Convolutional Neural Network (EEGNet1D)
- **Input**: 19 channels × 2560 samples (10 seconds at 256 Hz)
- **Output**: Binary classification (Seizure/No Seizure)
- **Training**: Continual learning with rehearsal buffer

### Feature Extraction (Heuristic Mode)
- **Line Length**: Measures signal complexity
- **Spectral Entropy**: Analyzes frequency distribution
- **Combined Score**: Weighted average for prediction

## 📝 Example Workflow

```
1. Start app: python -m streamlit run app.py
2. Upload: T:\suezier_p\data\samples\sample_eeg.edf
3. View prediction result
4. If needed, correct the label
5. Click "Improve model" to enable learning
6. Upload more files to build model knowledge
```

## 🔗 Related Files

- **App code**: `app/app.py`
- **Preprocessing**: `app/preprocess.py`
- **Inference**: `app/inference.py`
- **Model**: `models/network.py`
- **Training**: `training/continual.py`

---

**Status**: ✅ Fully Operational

The app is now ready to use! Start it with the command above and begin analyzing EEG data.
