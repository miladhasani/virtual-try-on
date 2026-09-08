# Launch the local virtual try-on atelier.
# Usage:  powershell -ExecutionPolicy Bypass -File .\run.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    foreach ($candidate in @("py -3.12", "py -3.11", "py -3.10", "python")) {
        try {
            $ver = Invoke-Expression "$candidate -c `"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')`"" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.(10|11|12)$') {
                return $candidate
            }
        } catch {
            continue
        }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Error "Need Python 3.10–3.12 (3.14 is too new for current PyTorch wheels). Install 3.12 and retry."
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment with $python ..."
    Invoke-Expression "$python -m venv .venv"
}

$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

$torchOk = & $py -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyTorch (CUDA 12.4 wheel)..."
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

& $py -m pip install -r requirements.txt
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_XET = "1"
Write-Host "Starting Atelier (http://127.0.0.1:7860 or the next free port)"
& $py app.py
