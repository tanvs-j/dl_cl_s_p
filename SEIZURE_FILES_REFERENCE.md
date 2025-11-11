# Seizure Files Reference - CHB-MIT Dataset

## Files WITH Seizures (Training Set)

Based on the CHB-MIT dataset `RECORDS-WITH-SEIZURES` file, the following files in the training directory contain seizure events:

### Patient chb01 (Available in T:\suezier_p\dataset\training\)

| File | Seizure Times (seconds) | Status |
|------|-------------------------|--------|
| **chb01_03.edf** | 2996-3036 (40s seizure) | ✅ Has Seizure |
| **chb01_04.edf** | 1467-1494 (27s seizure) | ✅ Has Seizure |
| **chb01_15.edf** | 1732-1772 (40s seizure) | ✅ Has Seizure |
| **chb01_16.edf** | 1015-1066 (51s seizure) | ✅ Has Seizure |
| **chb01_18.edf** | 1720-1810 (90s seizure) | ✅ Has Seizure |
| **chb01_21.edf** | 327-420 (93s seizure) | ✅ Has Seizure |
| **chb01_26.edf** | 1862-1963 (101s seizure) | ✅ Has Seizure |

### Other Files (Training Set)

All other chb01_XX.edf files in the training directory do NOT contain seizures:
- chb01_01.edf ❌ No Seizure
- chb01_02.edf ❌ No Seizure
- chb01_05.edf ❌ No Seizure
- chb01_06.edf ❌ No Seizure
- chb01_07.edf ❌ No Seizure
- chb01_08.edf ❌ No Seizure
- chb01_09.edf ❌ No Seizure
- chb01_10.edf ❌ No Seizure
- chb01_11.edf ❌ No Seizure
- chb01_12.edf ❌ No Seizure
- chb01_13.edf ❌ No Seizure
- chb01_14.edf ❌ No Seizure
- chb01_17.edf ❌ No Seizure
- chb01_19.edf ❌ No Seizure
- chb01_20.edf ❌ No Seizure
- chb01_22.edf ❌ No Seizure
- chb01_23.edf ❌ No Seizure
- chb01_24.edf ❌ No Seizure
- chb01_25.edf ❌ No Seizure
- chb01_27.edf ❌ No Seizure

## Files in Testing Directory (T:\suezier_p\dataset\testing\)

| File | Has Seizure | Notes |
|------|-------------|-------|
| chb01_29.edf | ❓ Unknown | Test with model |
| chb01_30.edf | ❓ Unknown | Test with model |
| chb01_31.edf | ❓ Unknown | Test with model |
| chb01_32.edf | ❓ Unknown | Test with model |
| chb01_33.edf | ❓ Unknown | Test with model |
| chb01_34.edf | ❓ Unknown | Test with model |
| chb01_36.edf | ❓ Unknown | Test with model |
| chb01_37.edf | ❓ Unknown | Test with model |
| chb01_38.edf | ❓ Unknown | Test with model |

## Model Information

### Current Trained Model
- **Location**: `models/checkpoints/best.pt`
- **Channels**: 23 EEG channels
- **Architecture**: EEGNet1D (1D-CNN)
- **Performance** (from previous training):
  - Validation Accuracy: 99.14%
  - Sensitivity: 99.00%
  - F1 Score: 86.46%

### Model Training Data
The model was trained on:
- **Training files**: 20 EDF files from dataset/training
- **Real seizure examples**: 58 labeled seizure windows
- **Synthetic seizure examples**: Added for balanced training
- **Normal examples**: Thousands of non-seizure windows

## How to Use This Information

### Testing Files with Known Seizures
To verify the model works, upload these files to the web app:

1. **chb01_03.edf** - Should detect seizure around 2996-3036 seconds
2. **chb01_16.edf** - Should detect seizure around 1015-1066 seconds  
3. **chb01_21.edf** - Should detect seizure around 327-420 seconds

### Testing Files WITHOUT Seizures
To verify the model doesn't give false positives:

1. **chb01_01.edf** - Should report NO SEIZURE
2. **chb01_02.edf** - Should report NO SEIZURE
3. **chb01_05.edf** - Should report NO SEIZURE

### Web App Usage
1. Start the app: `cd T:\suezier_p\app ; python -m streamlit run app.py`
2. Open browser: http://localhost:8501
3. Upload any EDF file from the list above
4. View prediction results
5. Compare with this reference guide

## Important Notes

- ⚠️ The model is trained on CHB-MIT pediatric epilepsy data
- ✅ Best performance on similar EEG recording conditions (256 Hz, 23 channels)
- 🔬 For research and development purposes only
- ❌ Not FDA approved for clinical use

## Quick Test Commands

### Check Model Status
```powershell
dir T:\suezier_p\models\checkpoints\best.pt
```

### Verify Model Configuration
```powershell
python -c "import torch; ckpt=torch.load('models/checkpoints/best.pt'); print(ckpt['model_kwargs'])"
```

### Start Web App
```powershell
cd T:\suezier_p\app
python -m streamlit run app.py
```

---

**Reference Source**: CHB-MIT Scalp EEG Database  
**Dataset**: https://physionet.org/content/chbmit/1.0.0/  
**Model Status**: ✅ Trained and Ready  
**Last Updated**: November 11, 2025
