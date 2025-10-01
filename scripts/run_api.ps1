# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt | Out-Null
uvicorn src.api:app --host 127.0.0.1 --port 8000
