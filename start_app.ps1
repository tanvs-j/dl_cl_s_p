#!/usr/bin/env pwsh
# Seizure Prediction Web App Startup Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Seizure Prediction System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (!(Test-Path "app\app.py")) {
    Write-Host "Error: Please run this script from the project root directory" -ForegroundColor Red
    Write-Host "Expected location: T:\suezier_p\" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment if it exists
$venvPath = "venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $venvPath
} else {
    Write-Host "Warning: Virtual environment not found" -ForegroundColor Yellow
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    & $venvPath
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install streamlit numpy scipy mne pyedflib torch pandas matplotlib
}

Write-Host ""
Write-Host "Starting Seizure Prediction Web App..." -ForegroundColor Green
Write-Host ""
Write-Host "Test files available:" -ForegroundColor Cyan
Write-Host "  - data\samples\sample_eeg.edf" -ForegroundColor White
Write-Host "  - dataset\testing\chb01_*.edf" -ForegroundColor White
Write-Host ""
Write-Host "The app will open in your browser at:" -ForegroundColor Cyan
Write-Host "  http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the app" -ForegroundColor Yellow
Write-Host ""

# Change to app directory and run
Set-Location app
python -m streamlit run app.py
