param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$venv = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) { . $venv }
if (-not $NoBrowser) {
  streamlit run app/app.py
} else {
  streamlit run app/app.py --server.headless true
}
