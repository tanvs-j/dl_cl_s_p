# Run the FIXED seizure detection app
Write-Host "Starting FIXED Seizure Detection System v3.0..." -ForegroundColor Green
Write-Host "Using absolute thresholds calibrated from CHB-MIT dataset" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Run the fixed app
cd app
python -m streamlit run app_fixed.py

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
