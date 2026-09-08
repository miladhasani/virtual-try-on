# Atelier — local virtual try-on

A Gradio app that takes a **person photo** and a **garment photo**, then generates a try-on image with pretrained **[CatVTON](https://github.com/Zheng-Chong/CatVTON)** (ICLR 2025). Built for a laptop **RTX 4070** (~8 GB VRAM).

This is inference only. Fine-tuning is intentionally out of scope for v1.

## What you get

- Person + garment upload, cloth type (`upper` / `lower` / `overall`)
- Model dropdown: CatVTON Mix, VITON-HD, DressCode, and Mask-Free (one adapter in VRAM at a time)
- Quality presets: Fast / Balanced / Quality
- Before/after slider, recent-result gallery
- Live GPU VRAM, GPU util, temperature, CPU, and RAM while generating
- Automatic retry on CUDA OOM (drops to Fast)
- Unload button to free VRAM

## Models

All four share the same SD 1.5 inpainting UNet. Switching models only loads a ~198 MB attention adapter, so VRAM stays about the same.

| Dropdown choice | Best for |
|-----------------|----------|
| CatVTON Mix | Everyday tops, pants, and outfits (default) |
| CatVTON VITON-HD | Upper-body garments |
| CatVTON DressCode | Dresses and fuller looks |
| CatVTON Mask-Free | Photos where the auto clothing mask is messy |

## Hardware

| Preset    | Size      | Steps | Notes                                      |
|-----------|-----------|-------|--------------------------------------------|
| Fast      | 384×512   | 20    | Safest on 8 GB                             |
| Balanced  | 576×768   | 30    | Default                                    |
| Quality   | 768×1024  | 50    | May OOM if the GPU is also driving a busy display |

Use **CUDA-enabled NVIDIA drivers**. Close Chrome hardware acceleration / other GPU apps before Quality. First launch downloads several GB of checkpoints.

If native Windows install fails (rare parser / CUDA issues), run the same app in **WSL2 + CUDA**.

## Setup (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Or manually:

Use **Python 3.10–3.12** (3.14 is too new for current PyTorch wheels). `run.ps1` prefers 3.12 automatically.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

If Hugging Face xet stalls on Windows, download weights with curl (resumable):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_weights.ps1
```

If Hugging Face downloads stall (common on Windows with XET), run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_weights.ps1
```

That script also fetches the extra CatVTON attention adapters (~198 MB each). The app can download a missing adapter the first time you select it.

The SD 1.5 inpainting UNet is often **gated** on Hugging Face. If download fails:

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Run `huggingface-cli login`
3. Accept the license on one of: `runwayml/stable-diffusion-inpainting`, `sd-legacy/stable-diffusion-inpainting`

You can also point at a local or public checkpoint:

```powershell
$env:CATVTON_BASE_MODEL = "path-or-hf-id"
```

## CLI

```powershell
python -m src.pipeline.tryon --info
python -m src.pipeline.tryon --person photo.jpg --garment shirt.jpg --preset Fast --model catvton-mix
```

## License

CatVTON code and weights are **CC BY-NC-SA 4.0** — non-commercial research only. See [NOTICE](NOTICE). Commercial production needs a different model or a license from the authors.

Cloth parsing uses [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes) instead of CatVTON’s DensePose + SCHP stack, so this repo stays installable on Windows without Detectron2.

## Fine-tuning (later)

CatVTON can be fine-tuned on VITON-HD or DressCode. That needs a large paired dataset and hours of GPU time. Use the pretrained mix checkpoint until the app is solid.
