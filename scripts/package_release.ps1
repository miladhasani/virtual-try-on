# Build a self-contained Windows release folder for Virtual Try-On.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1 -Zip
#   powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1 -IncludeWeights -Zip
#
# The pack is portable source + launchers. The machine that runs it still needs
# Python 3.10–3.12 and an NVIDIA driver. First launch creates a local venv and
# installs PyTorch + requirements. Pass -IncludeWeights to copy already-downloaded
# SD 1.5 / CatVTON files so the first generate does not hit Hugging Face.

[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$Version = "",
    [switch]$IncludeWeights,
    [switch]$IncludeExamples,
    [switch]$Zip,
    [switch]$SkipInstallers
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot "app.py"))) {
    throw "Could not find app.py next to the repo root. Run this script from the virtual-try-on checkout."
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot "dist"
}

if (-not $Version) {
    $stamp = Get-Date -Format "yyyyMMdd"
    $sha = ""
    try {
        $sha = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
    } catch {
        $sha = ""
    }
    $Version = if ($sha) { "$stamp-$sha" } else { $stamp }
}

$PackName = "Virtual-Try-On-$Version-win"
$PackRoot = Join-Path $OutputDir $PackName
$ZipPath = Join-Path $OutputDir "$PackName.zip"

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message"
}

function Copy-FileToPack($relativePath) {
    $src = Join-Path $RepoRoot $relativePath
    if (-not (Test-Path $src)) {
        return $false
    }
    $dest = Join-Path $PackRoot $relativePath
    $destDir = Split-Path -Parent $dest
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    }
    Copy-Item -LiteralPath $src -Destination $dest -Force
    return $true
}

function Copy-Tree($relativePath, $excludeDirNames) {
    $src = Join-Path $RepoRoot $relativePath
    if (-not (Test-Path $src)) {
        return
    }
    $destRoot = Join-Path $PackRoot $relativePath
    Get-ChildItem -LiteralPath $src -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($src.Length).TrimStart("\", "/")
        $skip = $false
        foreach ($name in $excludeDirNames) {
            if ($rel -like "*${name}\*" -or $rel -like "*/${name}/*") {
                $skip = $true
                break
            }
        }
        if ($skip) { return }
        if ($_.Extension -in @(".pyc", ".pyo")) { return }
        $dest = Join-Path $destRoot $rel
        $destDir = Split-Path -Parent $dest
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
}

if (Test-Path $PackRoot) {
    Write-Step "Removing previous pack $PackRoot"
    Remove-Item -LiteralPath $PackRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PackRoot | Out-Null

Write-Step "Copying application files"
foreach ($file in @(
    "app.py",
    "requirements.txt",
    "README.md",
    "NOTICE",
    "run.ps1"
)) {
    if (-not (Copy-FileToPack $file)) {
        throw "Missing required file: $file"
    }
}

Copy-Tree "src" @("__pycache__")
Copy-FileToPack "scripts\download_weights.ps1" | Out-Null

$examplesCopied = $false
if ($IncludeExamples -or (Test-Path (Join-Path $RepoRoot "assets\examples"))) {
    Copy-Tree "assets\examples" @()
    if (Test-Path (Join-Path $PackRoot "assets\examples")) {
        $examplesCopied = $true
    }
}

$weightsCopied = $false
if ($IncludeWeights) {
    $localWeights = Join-Path $RepoRoot "weights\local"
    if (Test-Path $localWeights) {
        Write-Step "Copying local weights (skipping Hugging Face cache)"
        Copy-Tree "weights\local" @()
        $weightsCopied = $true
    } else {
        Write-Warning "No weights\local folder found. Release will download adapters on first use."
    }
}

Write-Step "Writing launchers"
$startPs1 = @'
# Start Virtual Try-On from this release folder.
# Double-click Start-Virtual-Try-On.cmd or run:
#   powershell -ExecutionPolicy Bypass -File .\Start-Virtual-Try-On.ps1

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
    Write-Host ""
    Write-Host "Virtual Try-On needs Python 3.10, 3.11, or 3.12 (3.14 is too new for current PyTorch wheels)."
    Write-Host "Install 3.12 from https://www.python.org/downloads/ then run this launcher again."
    Write-Host "During setup, enable 'Add python.exe to PATH'."
    if (Test-Path ".\Install-Python.cmd") {
        Write-Host "Or run Install-Python.cmd to open the download page."
    }
    exit 1
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
Write-Host "Starting Virtual Try-On (http://127.0.0.1:7860 or the next free port)"
& $py app.py
'@
Set-Content -LiteralPath (Join-Path $PackRoot "Start-Virtual-Try-On.ps1") -Value $startPs1 -Encoding UTF8

$startCmd = @"
@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Virtual-Try-On.ps1"
if errorlevel 1 pause
"@
Set-Content -LiteralPath (Join-Path $PackRoot "Start-Virtual-Try-On.cmd") -Value $startCmd -Encoding ASCII

$downloadCmd = @"
@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\download_weights.ps1"
if errorlevel 1 pause
"@
Set-Content -LiteralPath (Join-Path $PackRoot "Download-Weights.cmd") -Value $downloadCmd -Encoding ASCII

if (-not $SkipInstallers) {
    $pyCmd = @"
@echo off
echo Virtual Try-On needs Python 3.12 (64-bit). 3.14 is too new for the current PyTorch wheels.
echo Opening the Python download page...
start "" "https://www.python.org/downloads/release/python-31210/"
echo Install it, tick "Add python.exe to PATH", then run Start-Virtual-Try-On.cmd
pause
"@
    Set-Content -LiteralPath (Join-Path $PackRoot "Install-Python.cmd") -Value $pyCmd -Encoding ASCII
}

$lockSrc = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path $lockSrc) {
    Write-Step "Freezing current environment to requirements.lock.txt"
    & $lockSrc -m pip freeze | Out-File -FilePath (Join-Path $PackRoot "requirements.lock.txt") -Encoding ascii
}

$gitSha = ""
try { $gitSha = (git -C $RepoRoot rev-parse HEAD 2>$null) } catch { $gitSha = "" }

$weightNote = if ($weightsCopied) {
    "Local SD 1.5 / CatVTON weights are included under weights\local. FLUX.1-Fill-dev still downloads the first time you pick a FLUX model (Hugging Face login + license)."
} else {
    "Weights are not in this zip. Run Download-Weights.cmd once, or let the app fetch an adapter the first time you select it."
}

$exampleNote = if ($examplesCopied) {
    "Demo person/garment photos are in assets\examples."
} else {
    "Example photos are not bundled. The app can fetch a couple of CatVTON demos on first launch if the network is up."
}

$readMe = @"
Virtual Try-On $Version
================

Self-contained Windows pack for the virtual try-on studio.

What you need on this PC
- Windows 10/11, 64-bit
- NVIDIA GPU + current Game Ready / Studio driver
- Python 3.10, 3.11, or 3.12 (64-bit). 3.14 will not work.
  Run Install-Python.cmd if python is missing.

How to start
1. Unzip this folder anywhere (SSD preferred).
2. Double-click Start-Virtual-Try-On.cmd
   First run creates .venv, installs PyTorch (CUDA 12.4) and requirements.
   That download is a few GB.
3. Open the URL printed in the window (usually http://127.0.0.1:7860).

Weights
$weightNote

Examples
$exampleNote

Licenses
CatVTON / SD 1.5 adapters: CC BY-NC-SA 4.0 (non-commercial).
FLUX.1 [dev]: Black Forest Labs non-commercial license (gated).
See NOTICE.

Build
Version: $Version
Git: $gitSha
Packed: $(Get-Date -Format "yyyy-MM-dd HH:mm")
"@
Set-Content -LiteralPath (Join-Path $PackRoot "RELEASE.txt") -Value $readMe -Encoding UTF8

$manifest = [ordered]@{
    name            = "Virtual Try-On"
    version         = $Version
    git             = $gitSha
    packed_utc      = (Get-Date).ToUniversalTime().ToString("o")
    include_weights = [bool]$weightsCopied
    include_examples= [bool]$examplesCopied
}
($manifest | ConvertTo-Json) | Set-Content -LiteralPath (Join-Path $PackRoot "release.json") -Encoding UTF8

if ($Zip) {
    Write-Step "Creating zip $ZipPath"
    if (Test-Path $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }
    Compress-Archive -Path $PackRoot -DestinationPath $ZipPath -CompressionLevel Optimal
}

Write-Host ""
Write-Host "Release folder: $PackRoot"
if ($Zip) {
    $zipItem = Get-Item $ZipPath
    Write-Host ("Zip:            {0} ({1:N1} MB)" -f $zipItem.FullName, ($zipItem.Length / 1MB))
}
Write-Host "Start with:     $PackRoot\Start-Virtual-Try-On.cmd"
