param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not available on PATH. Install Python 3.10+ and retry."
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath"
    python -m venv $VenvPath
}

$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Activation script not found at $activateScript"
}

Write-Host "Activating virtual environment"
. $activateScript

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt"
pip install -r requirements.txt

Write-Host "Done. Virtual environment is active in this shell."
