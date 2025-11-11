"""
Web Application for EEG Report Analysis
Upload PDF EEG reports, extract waveforms, predict seizures, and visualize abnormalities
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import io
import base64
from pathlib import Path
import sys
from typing import List, Dict
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.feature_extractor import EEGFeatureExtractor, FeatureConfig
from src.models.deep_learning_models import create_model, ModelConfig
from src.data.edf_reader import EDFReader

app = FastAPI(title="Seizure Prediction System", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model_config = ModelConfig()
model = create_model('cnn', model_config)

# Load trained weights if available
trained_model_path = Path(__file__).parent.parent.parent / 'data' / 'models' / 'trained_seizure_model.pth'
if trained_model_path.exists():
    try:
        model.load_state_dict(torch.load(trained_model_path, map_location=model_config.device, weights_only=True))
        print(f"✓ Loaded trained model from {trained_model_path}")
    except Exception as e:
        print(f"⚠️  Could not load trained model: {e}")
        print("   Using untrained model")
else:
    print("⚠️  Trained model not found, using untrained model")
    print(f"   Expected at: {trained_model_path}")

model.eval()

# Initialize feature extractor
feature_extractor = EEGFeatureExtractor(FeatureConfig())

# Initialize EDF reader
edf_reader = EDFReader(model_path=str(trained_model_path) if trained_model_path.exists() else None)


def extract_eeg_from_pdf(pdf_bytes: bytes) -> np.ndarray:
    """
    Extract EEG waveforms from PDF report.
    
    Note: This is a placeholder. Real implementation would:
    1. Use PyMuPDF or pdfplumber to read PDF
    2. Extract images containing EEG waveforms
    3. Use image processing to digitize the waveforms
    4. Convert to numerical EEG data
    
    For now, generates synthetic EEG for demonstration.
    """
    # TODO: Implement actual PDF parsing
    # For demonstration, generate synthetic 18-channel EEG
    duration = 30  # seconds
    sampling_rate = 256
    num_channels = 18
    num_samples = duration * sampling_rate
    
    # Generate synthetic EEG with random seizure
    t = np.linspace(0, duration, num_samples)
    eeg_data = np.random.randn(num_channels, num_samples) * 15
    
    # Add brain rhythms
    for ch in range(num_channels):
        eeg_data[ch] += 12 * np.sin(2 * np.pi * 10 * t)  # Alpha
        eeg_data[ch] += 5 * np.sin(2 * np.pi * 18 * t)   # Beta
    
    # Random seizure insertion
    if np.random.rand() > 0.3:  # 70% chance of seizure
        seizure_start = int(num_samples * 0.4)
        seizure_end = int(num_samples * 0.6)
        seizure_channels = min(10, num_channels)
        
        for ch in range(seizure_channels):
            amplitude = 120 * (1 + 0.2 * np.sin(2 * np.pi * 0.3 * t[seizure_start:seizure_end]))
            eeg_data[ch, seizure_start:seizure_end] += amplitude * np.sin(2 * np.pi * 5 * t[seizure_start:seizure_end])
    
    return eeg_data


def analyze_eeg_signal(eeg_data: np.ndarray) -> Dict:
    """Analyze EEG signal and predict seizures."""
    epoch_length = 2  # seconds
    sampling_rate = 256
    epoch_samples = epoch_length * sampling_rate
    num_epochs = eeg_data.shape[1] // epoch_samples
    
    predictions = []
    probabilities = []
    features_list = []
    abnormalities = []
    
    feature_extractor.reset_history()
    
    for epoch_idx in range(num_epochs):
        start_sample = epoch_idx * epoch_samples
        end_sample = start_sample + epoch_samples
        current_time = epoch_idx * epoch_length
        
        eeg_epoch = eeg_data[:, start_sample:end_sample]
        
        # Extract features
        features = feature_extractor.extract_epoch_features(eeg_epoch)
        features_list.append(features.tolist())
        
        # Prepare for model
        eeg_tensor = torch.FloatTensor(eeg_epoch).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(eeg_tensor)
            probs = torch.softmax(output, dim=1)
            seizure_prob = probs[0, 1].item()
            prediction = torch.argmax(output, dim=1).item()
        
        predictions.append(prediction)
        probabilities.append(seizure_prob)
        
        # Detect abnormalities
        avg_amplitude = np.mean(np.abs(eeg_epoch))
        max_amplitude = np.max(np.abs(eeg_epoch))
        
        if max_amplitude > 100 or seizure_prob > 0.7:
            abnormalities.append({
                'time': current_time,
                'type': 'seizure' if prediction == 1 else 'suspicious',
                'probability': seizure_prob,
                'avg_amplitude': float(avg_amplitude),
                'max_amplitude': float(max_amplitude),
                'channels_affected': [int(ch) for ch in range(18) if np.max(np.abs(eeg_epoch[ch])) > 80]
            })
    
    # Overall analysis
    seizure_detected = any(p == 1 for p in predictions)
    avg_seizure_prob = np.mean(probabilities)
    max_seizure_prob = np.max(probabilities)
    
    return {
        'seizure_detected': seizure_detected,
        'confidence': float(max_seizure_prob),
        'average_probability': float(avg_seizure_prob),
        'num_epochs': num_epochs,
        'num_abnormal_epochs': len(abnormalities),
        'predictions': predictions,
        'probabilities': [float(p) for p in probabilities],
        'abnormalities': abnormalities,
        'duration': num_epochs * epoch_length
    }


def prepare_visualization_data(eeg_data: np.ndarray, analysis: Dict) -> Dict:
    """Prepare EEG data for web visualization."""
    # Downsample for web display
    downsample_factor = 4
    eeg_downsampled = eeg_data[:, ::downsample_factor]
    
    # Convert to list for JSON
    channels_data = []
    for ch in range(eeg_data.shape[0]):
        channels_data.append({
            'channel': ch,
            'data': eeg_downsampled[ch].tolist(),
            'label': f'Channel {ch+1}'
        })
    
    return {
        'channels': channels_data,
        'sampling_rate': 256 // downsample_factor,
        'duration': eeg_data.shape[1] / 256,
        'analysis': analysis
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve main web interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Seizure Prediction System</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .upload-section {
                padding: 40px;
                text-align: center;
                border-bottom: 2px solid #f0f0f0;
            }
            .upload-box {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 60px 40px;
                margin: 20px auto;
                max-width: 600px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-box:hover {
                border-color: #764ba2;
                background: #f9f9ff;
            }
            .upload-box.dragover {
                background: #e8e8ff;
                border-color: #764ba2;
            }
            .upload-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 1.1em;
                border-radius: 50px;
                cursor: pointer;
                transition: transform 0.2s;
                margin: 10px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            }
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .results-section {
                padding: 40px;
                display: none;
            }
            .result-card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .result-card h2 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.8em;
            }
            .status-badge {
                display: inline-block;
                padding: 15px 30px;
                border-radius: 50px;
                font-size: 1.3em;
                font-weight: bold;
                margin: 20px 0;
            }
            .status-positive {
                background: #fee;
                color: #c00;
                border: 2px solid #c00;
            }
            .status-negative {
                background: #efe;
                color: #0a0;
                border: 2px solid #0a0;
            }
            .status-warning {
                background: #ffc;
                color: #880;
                border: 2px solid #880;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric {
                background: #f9f9ff;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .metric-value {
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
            }
            .metric-label {
                color: #666;
                margin-top: 10px;
            }
            .abnormality-list {
                margin: 20px 0;
            }
            .abnormality-item {
                background: #fff5f5;
                border-left: 4px solid #c00;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .abnormality-item.warning {
                background: #fffef5;
                border-left-color: #880;
            }
            .plot-container {
                margin: 30px 0;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .loading {
                text-align: center;
                padding: 40px;
                display: none;
            }
            .spinner {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #667eea;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .footer {
                background: #f9f9f9;
                padding: 20px;
                text-align: center;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Seizure Prediction System</h1>
                <p>Upload your EEG report (PDF or EDF) for instant analysis</p>
            </div>

            <div class="upload-section" id="uploadSection">
                <h2>Upload EEG Report</h2>
                <div class="upload-box" id="uploadBox">
                    <div class="upload-icon">📄</div>
                    <h3>Drag & Drop your EEG file here</h3>
                    <p>or</p>
                    <input type="file" id="fileInput" accept=".pdf,.edf" style="display:none">
                    <button class="btn" onclick="document.getElementById('fileInput').click()">
                        Browse Files
                    </button>
                    <p style="margin-top:20px; color:#666;">Accepted formats: PDF, EDF</p>
                </div>
                <button class="btn" id="analyzeBtn" style="display:none" onclick="analyzeReport()">
                    🔬 Analyze Report
                </button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <h3>Analyzing EEG Report...</h3>
                <p>Extracting waveforms and detecting abnormalities</p>
            </div>

            <div class="results-section" id="resultsSection">
                <div class="result-card">
                    <h2>Analysis Results</h2>
                    <div id="statusBadge"></div>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value" id="confidence">-</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="duration">-</div>
                            <div class="metric-label">Duration (s)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="abnormalEpochs">-</div>
                            <div class="metric-label">Abnormal Epochs</div>
                        </div>
                    </div>
                </div>

                <div class="result-card">
                    <h2>Seizure Probability Over Time</h2>
                    <div id="probabilityPlot" class="plot-container"></div>
                </div>

                <div class="result-card">
                    <h2>EEG Waveforms</h2>
                    <div id="eegPlot" class="plot-container"></div>
                </div>

                <div class="result-card" id="abnormalitiesCard" style="display:none">
                    <h2>Detected Abnormalities</h2>
                    <div id="abnormalitiesList" class="abnormality-list"></div>
                </div>

                <button class="btn" onclick="location.reload()">
                    📤 Upload Another Report
                </button>
            </div>

            <div class="footer">
                <p>⚠️ For research and educational purposes only. Not for clinical diagnosis.</p>
                <p>Seizure Prediction System v1.0 | Powered by Deep Learning</p>
            </div>
        </div>

        <script>
            let selectedFile = null;

            // Drag and drop
            const uploadBox = document.getElementById('uploadBox');
            uploadBox.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadBox.classList.add('dragover');
            });
            uploadBox.addEventListener('dragleave', () => {
                uploadBox.classList.remove('dragover');
            });
            uploadBox.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadBox.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) handleFileSelect(files[0]);
            });

            // File input
            document.getElementById('fileInput').addEventListener('change', (e) => {
                if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
            });

            function handleFileSelect(file) {
                const fileName = file.name.toLowerCase();
                if (!fileName.endsWith('.pdf') && !fileName.endsWith('.edf')) {
                    alert('Please select a PDF or EDF file');
                    return;
                }
                selectedFile = file;
                uploadBox.innerHTML = `
                    <div class="upload-icon">✅</div>
                    <h3>${file.name}</h3>
                    <p>Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                `;
                document.getElementById('analyzeBtn').style.display = 'block';
            }

            async function analyzeReport() {
                if (!selectedFile) return;

                document.getElementById('uploadSection').style.display = 'none';
                document.getElementById('loading').style.display = 'block';

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    displayResults(data);
                } catch (error) {
                    alert('Error analyzing report: ' + error.message);
                    location.reload();
                }
            }

            function displayResults(data) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('resultsSection').style.display = 'block';

                const analysis = data.analysis;

                // Status badge
                let statusHTML = '';
                if (analysis.seizure_detected) {
                    statusHTML = '<div class="status-badge status-positive">⚠️ SEIZURE ACTIVITY DETECTED</div>';
                } else if (analysis.average_probability > 0.5) {
                    statusHTML = '<div class="status-badge status-warning">⚠️ SUSPICIOUS ACTIVITY</div>';
                } else {
                    statusHTML = '<div class="status-badge status-negative">✅ NO SEIZURE DETECTED</div>';
                }
                document.getElementById('statusBadge').innerHTML = statusHTML;

                // Metrics
                document.getElementById('confidence').textContent = (analysis.confidence * 100).toFixed(1) + '%';
                document.getElementById('duration').textContent = analysis.duration;
                document.getElementById('abnormalEpochs').textContent = analysis.num_abnormal_epochs;

                // Probability plot
                const times = analysis.probabilities.map((_, i) => i * 2);
                const probabilityTrace = {
                    x: times,
                    y: analysis.probabilities.map(p => p * 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Seizure Probability',
                    line: { color: '#667eea', width: 3 },
                    marker: { size: 8 }
                };

                Plotly.newPlot('probabilityPlot', [probabilityTrace], {
                    title: 'Seizure Probability Over Time',
                    xaxis: { title: 'Time (seconds)' },
                    yaxis: { title: 'Probability (%)' },
                    height: 400
                });

                // EEG waveforms (first 6 channels)
                const eegTraces = data.channels.slice(0, 6).map((ch, i) => ({
                    y: ch.data.map(v => v + i * 100),  // Offset for visibility
                    type: 'scatter',
                    mode: 'lines',
                    name: ch.label,
                    line: { width: 1 }
                }));

                Plotly.newPlot('eegPlot', eegTraces, {
                    title: 'EEG Waveforms (First 6 Channels)',
                    xaxis: { title: 'Samples' },
                    yaxis: { title: 'Amplitude (µV)', showticklabels: false },
                    height: 500
                });

                // Abnormalities
                if (analysis.abnormalities.length > 0) {
                    document.getElementById('abnormalitiesCard').style.display = 'block';
                    let abnormalHTML = '';
                    analysis.abnormalities.forEach(abn => {
                        const className = abn.type === 'seizure' ? 'abnormality-item' : 'abnormality-item warning';
                        abnormalHTML += `
                            <div class="${className}">
                                <strong>${abn.type.toUpperCase()}</strong> at ${abn.time}s
                                <br>Probability: ${(abn.probability * 100).toFixed(1)}%
                                <br>Max Amplitude: ${abn.max_amplitude.toFixed(1)} µV
                                <br>Affected Channels: ${abn.channels_affected.join(', ')}
                            </div>
                        `;
                    });
                    document.getElementById('abnormalitiesList').innerHTML = abnormalHTML;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/analyze")
async def analyze_eeg_report(file: UploadFile = File(...)):
    """Analyze uploaded EEG report (PDF or EDF)."""
    filename_lower = file.filename.lower()
    
    if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.edf')):
        raise HTTPException(status_code=400, detail="Only PDF and EDF files are accepted")
    
    # Read file
    file_bytes = await file.read()
    
    # Handle based on file type
    if filename_lower.endswith('.edf'):
        # Process EDF file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Read and preprocess EDF
            eeg_dict = edf_reader.read_edf(tmp_path)
            eeg_data = edf_reader.preprocess_eeg(eeg_dict)
            
            # Analyze
            analysis = edf_reader.predict_seizures(eeg_data)
            
            # Add file info
            analysis['file_type'] = 'edf'
            analysis['file_name'] = file.filename
            analysis['channels_used'] = eeg_dict['labels'][:18]
            
        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)
    else:
        # Process PDF file
        eeg_data = extract_eeg_from_pdf(file_bytes)
        analysis = analyze_eeg_signal(eeg_data)
        analysis['file_type'] = 'pdf'
        analysis['file_name'] = file.filename
    
    # Prepare visualization data
    viz_data = prepare_visualization_data(eeg_data, analysis)
    
    return JSONResponse(content=viz_data)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "loaded", "version": "1.0"}


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🧠 Starting Seizure Prediction Web Application")
    print("=" * 70)
    print("📡 Server: http://localhost:8000")
    print("📄 Upload EEG PDF reports for instant analysis")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
