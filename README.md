# Atelier — local virtual try-on

A Gradio app that takes a **person photo** and a **garment photo**, then generates a try-on image with pretrained **[CatVTON](https://github.com/Zheng-Chong/CatVTON)** (ICLR 2025). SD 1.5 adapters fit a laptop **RTX 4070** (~8 GB VRAM). FLUX checkpoints need about **24 GB**.

This is inference only. Fine-tuning is intentionally out of scope for v1.

## What you get

- Person + garment upload, cloth type (`upper` / `lower` / `overall`)
- Model dropdown: four SD 1.5 CatVTON adapters plus three FLUX checkpoints (24 GB+)
- Quality presets: Fast / Balanced / Quality / Studio
- Tabbed stage: result, before/after slider, and a session gallery
- Run metadata after every render (model, preset, canvas, steps, elapsed)
- Live VRAM gauge plus GPU util, temperature, CPU, and RAM while generating
- Automatic retry on CUDA OOM (drops to Fast)
- Unload button to free VRAM

## Models

Selecting a checkpoint shows a comparison card (best-for, training data, VRAM, mask behavior). Dropdown labels also include the VRAM class.

**SD 1.5 (about 8 GB).** These four share the same inpainting UNet. Switching only loads a ~198 MB attention adapter. **No Hugging Face login.** Weights are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial). Official SD 1.5 inpainting is often gated; this app prefers a public mirror first.

| Dropdown choice | Best for | Access |
|-----------------|----------|--------|
| Mix — everyday default | Tops, pants, and outfits. Start here. | Open · no login · CC BY-NC-SA 4.0 |
| VITON-HD — shirts & jackets | Upper-body garments only. | Open · no login · CC BY-NC-SA 4.0 |
| DressCode — dresses & variety | Dresses, bottoms, and fuller looks. | Open · no login · CC BY-NC-SA 4.0 |
| Mask-Free — imperfect photos | Casual shots where auto-masking is messy. | Open · no login · CC BY-NC-SA 4.0 |

**FLUX (about 24 GB).** Sharper than SD 1.5. The CatVTON / community adapters are public, but first use downloads the gated [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) backbone. Create a Hugging Face account, accept the FLUX.1 [dev] non-commercial license, and run `huggingface-cli login`. Optional: `$env:CATVTON_FLUX_FILL = "path-or-hf-id"`.

| Dropdown choice | Best for | Access |
|-----------------|----------|--------|
| FLUX — highest fidelity | Author CatVTON LoRA on FLUX fill. | HF login + FLUX license |
| FLUX Alpha — logos & fabric | Community VITON-HD LoRA; better small prints. | HF login + FLUX license |
| FLUX Beta — mixed garments | Full FLUX fill fine-tune; widest garment mix. | HF login + FLUX license |

## Hardware

| Preset    | Size       | Steps | Notes                                         |
|-----------|------------|-------|-----------------------------------------------|
| Tiny      | 256×384    | 16    | Smallest canvas; use with FLUX offload     |
| Fast      | 384×512    | 20    | Safest SD 1.5 on 8 GB                      |
| Balanced  | 576×768    | 30    | Default                                    |
| Quality   | 768×1024   | 50    | May OOM if the GPU is also driving a display |
| Studio    | 1024×1280  | 36    | For 16 GB+ cards                           |

**Low-VRAM FLUX:** open **Advanced controls**. Set **VRAM mode** to Offload, 4-bit (needs `bitsandbytes`), or Sequential, and drop to Tiny/Fast. Generation is slower because layers swap to RAM. Auto does this for you when the card is smaller than 24 GB.

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

CatVTON code and SD 1.5 / Mask-Free weights are **CC BY-NC-SA 4.0** — non-commercial research only. FLUX checkpoints also need the Black Forest Labs **FLUX.1 [dev] non-commercial license** (gated download). See [NOTICE](NOTICE). Commercial production needs a different model or a license from the authors.

Cloth parsing uses [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes) instead of CatVTON’s DensePose + SCHP stack, so this repo stays installable on Windows without Detectron2.

## Fine-tuning (later)

CatVTON can be fine-tuned on VITON-HD or DressCode. That needs a large paired dataset and hours of GPU time. Use the pretrained mix checkpoint until the app is solid.
