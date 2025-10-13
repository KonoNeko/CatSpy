param(
    [switch]$CreateVenv
)

# Ensure script runs from the backend folder where this script lives
Set-Location $PSScriptRoot

# Create a virtual environment if not present or if requested
if ($CreateVenv -or -not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

# Activate the virtual environment for this session
Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
pip install -r .\requirements.txt

Write-Host "Starting uvicorn (FastAPI) on 0.0.0.0:8000..."
python -m uvicorn api:app --host 0.0.0.0 --port 8000
if ($LASTEXITCODE -ne 0) {
    Write-Host "uvicorn failed to start. If you see ImportError related to 'Cache' from transformers, try: pip install --upgrade transformers peft" -ForegroundColor Yellow
}
