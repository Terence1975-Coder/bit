# PowerShell helper to run the app
# Usage: .\run.ps1

$ErrorActionPreference = "Stop"

if (!(Test-Path .venv)) {
  Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
  python -m venv .venv
}

# Activate venv
$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (!(Test-Path $venvActivate)) {
  Write-Host "Could not find venv activation script." -ForegroundColor Red
  exit 1
}
& $venvActivate

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Starting Streamlit..." -ForegroundColor Green
streamlit run app.py
