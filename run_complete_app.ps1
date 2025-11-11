#!/usr/bin/env pwsh
# Start the Complete Professional Seizure Prediction App

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  SEIZURE PREDICTION SYSTEM v2.0" -ForegroundColor Cyan
Write-Host "  Professional Edition with Visualization" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Activate venv if exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
}

Write-Host "Features:" -ForegroundColor Yellow
Write-Host "  - Advanced multi-feature detection with L2 regularization" -ForegroundColor White
Write-Host "  - Interactive EEG wave visualization" -ForegroundColor White
Write-Host "  - Power spectral analysis with frequency bands" -ForegroundColor White
Write-Host "  - Detailed statistics and metrics" -ForegroundColor White
Write-Host "  - Professional clean theme" -ForegroundColor White
Write-Host "  - Reduced false positives" -ForegroundColor White
Write-Host ""

Write-Host "Starting application..." -ForegroundColor Green
Write-Host ""

# Change to app directory and run
Set-Location app
python -m streamlit run app_complete.py
