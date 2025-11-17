# Run the seizure detection app
Write-Host "Starting Seizure Detection System v3.1..." -ForegroundColor Green
Write-Host "Multi-format EEG support: EDF, EEG, CNT, VHDR" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Run the app
cd app
python -m streamlit run app.py

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
