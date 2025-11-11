# Downloading Real EEG Datasets from Kaggle

The system is currently trained on **synthetic EEG data** that simulates seizure patterns. For production use with real patients, you should train on actual EEG datasets.

## Current Status

✅ **Model Trained**: The system has been trained on 5,000 synthetic EEG samples  
✅ **Web App Ready**: The application is functional with the trained model  
📊 **Performance**: 100% accuracy on synthetic data (test metrics)

## Optional: Train on Real Kaggle Datasets

To improve the model with real EEG data from Kaggle:

### Step 1: Setup Kaggle API Credentials

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click **"Create New API Token"**
4. Download `kaggle.json` to:
   ```
   C:\Users\takesh\.kaggle\kaggle.json
   ```

### Step 2: Download Datasets

The system supports 4 high-quality EEG seizure datasets:

#### Primary Dataset (Recommended)
- **Epileptic Seizure Recognition**
- 11,500 samples with 178 EEG features per sample
- Binary labels: 1=seizure, 2-5=non-seizure
- Dataset: `harunshimanto/epileptic-seizure-recognition`

#### Additional Datasets
- `ruslankl/eeg-eye-state` - EEG eye state detection
- `birdy654/eeg-brainwave-dataset-feeling-emotions` - Emotional EEG patterns
- `broach/button-tone-sz` - Seizure-specific recordings

### Step 3: Download Script

Run the automated downloader:

```powershell
python scripts\download_kaggle_datasets.py
```

This will download all 4 datasets to: `data/raw/kaggle/`

### Step 4: Retrain with Real Data

Once datasets are downloaded, re-run the training script:

```powershell
python train_on_real_data.py
```

The script will automatically detect the Kaggle datasets and train on real data instead of synthetic data.

**Expected improvements:**
- Better generalization to real patient EEG patterns
- More realistic seizure detection thresholds
- Handling of artifacts and noise in clinical recordings
- Multi-site EEG format compatibility

### Step 5: Deploy Updated Model

After training completes, restart the web application:

```powershell
python src\api\web_app.py
```

The app automatically loads the latest trained model from `data/models/trained_seizure_model.pth`

## Dataset Details

### Epileptic Seizure Recognition
```
- Samples: 11,500
- Features: 178 EEG measurements per sample
- Classes: 5 (we convert to binary: seizure vs non-seizure)
- Source: UCI Machine Learning Repository
- Format: CSV with numerical features
```

### Training Configuration
```yaml
Model: CNN1D (4.3M parameters)
Epochs: 20
Batch Size: 32
Learning Rate: 0.001
Train/Val/Test Split: 64/16/20
Optimizer: Adam
Loss: CrossEntropyLoss
```

## Current Performance (Synthetic Data)

```
✓ Training Accuracy: 100.00%
✓ Validation Accuracy: 100.00%
✓ Test Accuracy: 100.00%
✓ Sensitivity (Seizure Detection): 100.00%
✓ False Positives: 0
✓ False Negatives: 0
```

**Note**: These perfect metrics are expected for synthetic data. Real EEG data will show more realistic performance (typically 85-95% accuracy).

## Alternative: Use Pre-trained Models

If Kaggle setup is not possible, you can:

1. **Use Current Synthetic Model**: Already functional for demonstrations
2. **Load External Weights**: Download pre-trained weights from research papers
3. **Contact**: Request access to our pre-trained clinical models

## Troubleshooting

### Kaggle API Not Working
```powershell
# Verify installation
kaggle --version

# Test credentials
kaggle datasets list
```

### Download Fails
- Check internet connection
- Verify `kaggle.json` is in correct location
- Ensure Kaggle account is verified

### Out of Memory During Training
Reduce batch size in `train_on_real_data.py`:
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Was 32
```

## Citation

If using Kaggle datasets, cite:
```
@misc{epileptic-seizure-recognition,
  author = {Harun Shimanto},
  title = {Epileptic Seizure Recognition Dataset},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition}}
}
```

## Next Steps

1. ✅ **Current**: System is operational with trained model
2. 📊 **Optional**: Download real datasets for production deployment
3. 🚀 **Production**: Deploy with clinical EEG data validation
4. 📈 **Monitoring**: Track model performance on real patient data

---

**Questions?** The system works out-of-the-box with synthetic data. Real data training is optional for production deployment.
