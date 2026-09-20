# Resume-friendly weight download (avoids Hugging Face xet stalls on Windows).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$unetDir = Join-Path $root "weights\local\inpainting\unet"
$mixDir = Join-Path $root "weights\local\catvton\mix-48k-1024\attention"
$vitonDir = Join-Path $root "weights\local\catvton\vitonhd-16k-512\attention"
$dressDir = Join-Path $root "weights\local\catvton\dresscode-16k-512\attention"
$maskFreeDir = Join-Path $root "weights\local\catvton-maskfree\mix-48k-1024\attention"
$fluxLoraDir = Join-Path $root "weights\local\catvton-flux"
New-Item -ItemType Directory -Force -Path $unetDir, $mixDir, $vitonDir, $dressDir, $maskFreeDir, $fluxLoraDir | Out-Null

function Get-File($url, $dest) {
    Write-Host "Downloading $url"
    & curl.exe -L --retry 8 --retry-all-errors -C - -o $dest $url
}

Get-File "https://huggingface.co/Lykon/dreamshaper-8-inpainting/resolve/main/unet/config.json" (Join-Path $unetDir "config.json")
Get-File "https://huggingface.co/Lykon/dreamshaper-8-inpainting/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors" (Join-Path $unetDir "diffusion_pytorch_model.fp16.safetensors")
Get-File "https://huggingface.co/zhengchong/CatVTON/resolve/main/mix-48k-1024/attention/model.safetensors" (Join-Path $mixDir "model.safetensors")
Get-File "https://huggingface.co/zhengchong/CatVTON/resolve/main/vitonhd-16k-512/attention/model.safetensors" (Join-Path $vitonDir "model.safetensors")
Get-File "https://huggingface.co/zhengchong/CatVTON/resolve/main/dresscode-16k-512/attention/model.safetensors" (Join-Path $dressDir "model.safetensors")
Get-File "https://huggingface.co/zhengchong/CatVTON-MaskFree/resolve/main/mix-48k-1024/attention/model.safetensors" (Join-Path $maskFreeDir "model.safetensors")
Get-File "https://huggingface.co/zhengchong/CatVTON/resolve/main/flux-lora/pytorch_lora_weights.safetensors" (Join-Path $fluxLoraDir "pytorch_lora_weights.safetensors")
Write-Host "Done. FLUX.1-Fill-dev itself downloads the first time you select a FLUX model."
