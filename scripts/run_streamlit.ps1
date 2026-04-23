$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
Set-Location $workspace

$venvPython = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found at .venv. Run ./scripts/setup_venv.ps1 first."
}

& $venvPython -m streamlit run streamlit_app.py
