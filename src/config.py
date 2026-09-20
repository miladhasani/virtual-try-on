"""Shared paths, model IDs, and quality presets."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
ASSETS_DIR = ROOT / "assets"
EXAMPLES_DIR = ASSETS_DIR / "examples"
THEME_CSS = Path(__file__).resolve().parent / "ui" / "theme.css"

LOCAL_INPAINT = WEIGHTS_DIR / "local" / "inpainting"
LOCAL_ATTN = WEIGHTS_DIR / "local" / "catvton"


ATTN_MIN_BYTES = 180_000_000


def _complete_file(path: Path, min_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


ATTN_REPO = os.environ.get(
    "CATVTON_ATTN_REPO",
    str(LOCAL_ATTN)
    if _complete_file(LOCAL_ATTN / "mix-48k-1024" / "attention" / "model.safetensors", ATTN_MIN_BYTES)
    else "zhengchong/CatVTON",
)
ATTN_VERSION = os.environ.get("CATVTON_ATTN_VERSION", "mix")
VAE_REPO = os.environ.get("CATVTON_VAE_REPO", "stabilityai/sd-vae-ft-mse")
PARSER_REPO = os.environ.get("CATVTON_PARSER_REPO", "mattmdjaga/segformer_b2_clothes")

# Inpainting UNet hosts. Prefer a finished local copy, then public checkpoints.
# Several official IDs are gated and hang without a Hugging Face token.
BASE_MODEL_CANDIDATES = [
    candidate
    for candidate in (
        os.environ.get("CATVTON_BASE_MODEL"),
        str(LOCAL_INPAINT)
        if _complete_file(LOCAL_INPAINT / "unet" / "diffusion_pytorch_model.fp16.safetensors", 1_500_000_000)
        else None,
        "Lykon/dreamshaper-8-inpainting",
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "runwayml/stable-diffusion-inpainting",
        "sd-legacy/stable-diffusion-inpainting",
        "botp/stable-diffusion-v1-5-inpainting",
        "benjamin-paine/stable-diffusion-v1-5-inpainting",
    )
    if candidate
]

QUALITY_PRESETS = {
    "Tiny": {"width": 256, "height": 384, "steps": 16},
    "Fast": {"width": 384, "height": 512, "steps": 20},
    "Balanced": {"width": 576, "height": 768, "steps": 30},
    "Quality": {"width": 768, "height": 1024, "steps": 50},
    "Studio": {"width": 1024, "height": 1280, "steps": 36},
}

VRAM_PROFILES = {
    "Auto": "Pick from this GPU",
    "GPU": "All on the card · fastest",
    "Offload": "Swap layers to RAM · slower",
    "Sequential": "Lowest VRAM · much slower",
    "4-bit": "Quantized FLUX · 8–12 GB",
}
DEFAULT_VRAM_PROFILE = "Auto"
VRAM_PROFILE_FALLBACK = {
    "GPU": "Offload",
    "Offload": "4-bit",
    "4-bit": "Sequential",
}


def snap_canvas(width: int, height: int) -> tuple[int, int]:
    def snap(value: int, floor: int) -> int:
        value = int(round(int(value) / 64) * 64)
        return max(floor, value)

    return snap(width, 256), snap(height, 320)

DEFAULT_PRESET = "Balanced"
DEFAULT_CLOTH_TYPE = "upper"
DEFAULT_GUIDANCE = 2.5
DEFAULT_SEED = 42
DEFAULT_MODEL = "catvton-mix"

CLOTH_TYPES = ["upper", "lower", "overall"]

# SD 1.5 CatVTON adapters share one UNet. FLUX checkpoints need a ~24 GB card.
TRYON_MODELS = {
    "catvton-mix": {
        "id": "catvton-mix",
        "label": "Mix — everyday default",
        "tagline": "Recommended on 8 GB",
        "info": (
            "General CatVTON adapter trained on mixed VITON-HD and DressCode data at 1024px. "
            "Use this first for tops, pants, and full outfits. Same ~8 GB VRAM as the other SD 1.5 adapters."
        ),
        "best_for": "Everyday tops, pants, and outfits",
        "trained_on": "Mixed 48k pairs at 1024px",
        "mask_note": "Needs a clean clothing mask",
        "vram_gb": 8,
        "vram_label": "8 GB",
        "weight_label": "mix-48k-1024",
        "foot": "SD 1.5 inpainting · swaps a 198 MB attention adapter",
        "backend": "sd15",
        "default_guidance": 2.5,
        "hf_repo": "zhengchong/CatVTON",
        "local_dir": "catvton",
        "attn_version": "mix",
        "attn_subfolder": "mix-48k-1024",
        "access": "open",
        "access_short": "no login",
        "access_label": "Open download · no login",
        "license_label": "CC BY-NC-SA 4.0",
        "access_note": (
            "Public CatVTON adapter — no Hugging Face login. "
            "Non-commercial only. Official SD 1.5 inpainting is gated; this app uses a public mirror first."
        ),
    },
    "catvton-vitonhd": {
        "id": "catvton-vitonhd",
        "label": "VITON-HD — shirts & jackets",
        "tagline": "Upper-body specialist",
        "info": (
            "Trained only on VITON-HD, so it is strongest on shirts, tees, jackets, and other tops. "
            "Weaker on dresses, pants, and full looks than Mix or DressCode. Same 8 GB UNet as Mix."
        ),
        "best_for": "Upper-body garments",
        "trained_on": "VITON-HD 16k pairs at 512px",
        "mask_note": "Needs a clean clothing mask",
        "vram_gb": 8,
        "vram_label": "8 GB",
        "weight_label": "vitonhd-16k-512",
        "foot": "SD 1.5 inpainting · swaps a 198 MB attention adapter",
        "backend": "sd15",
        "default_guidance": 2.5,
        "hf_repo": "zhengchong/CatVTON",
        "local_dir": "catvton",
        "attn_version": "vitonhd",
        "attn_subfolder": "vitonhd-16k-512",
        "access": "open",
        "access_short": "no login",
        "access_label": "Open download · no login",
        "license_label": "CC BY-NC-SA 4.0",
        "access_note": (
            "Public CatVTON adapter — no Hugging Face login. "
            "Non-commercial only. Official SD 1.5 inpainting is gated; this app uses a public mirror first."
        ),
    },
    "catvton-dresscode": {
        "id": "catvton-dresscode",
        "label": "DressCode — dresses & variety",
        "tagline": "Dresses and fuller looks",
        "info": (
            "Trained on DressCode, which includes dresses, lower-body, and more garment variety than VITON-HD. "
            "Pick this for dresses and full looks. Same 8 GB UNet as Mix."
        ),
        "best_for": "Dresses, bottoms, and full looks",
        "trained_on": "DressCode 16k pairs at 512px",
        "mask_note": "Needs a clean clothing mask",
        "vram_gb": 8,
        "vram_label": "8 GB",
        "weight_label": "dresscode-16k-512",
        "foot": "SD 1.5 inpainting · swaps a 198 MB attention adapter",
        "backend": "sd15",
        "default_guidance": 2.5,
        "hf_repo": "zhengchong/CatVTON",
        "local_dir": "catvton",
        "attn_version": "dresscode",
        "attn_subfolder": "dresscode-16k-512",
        "access": "open",
        "access_short": "no login",
        "access_label": "Open download · no login",
        "license_label": "CC BY-NC-SA 4.0",
        "access_note": (
            "Public CatVTON adapter — no Hugging Face login. "
            "Non-commercial only. Official SD 1.5 inpainting is gated; this app uses a public mirror first."
        ),
    },
    "catvton-maskfree": {
        "id": "catvton-maskfree",
        "label": "Mask-Free — imperfect photos",
        "tagline": "Forgiving when the mask is messy",
        "info": (
            "Mask-free Mix variant. More tolerant when auto-masking misses straps, prints, or busy poses. "
            "Still SD 1.5, still ~8 GB — not a quality upgrade over Mix on clean photos."
        ),
        "best_for": "Casual photos with a messy auto-mask",
        "trained_on": "Mask-free Mix 48k at 1024px",
        "mask_note": "Tolerates an imperfect mask",
        "vram_gb": 8,
        "vram_label": "8 GB",
        "weight_label": "maskfree/mix-48k-1024",
        "foot": "SD 1.5 inpainting · swaps a 198 MB attention adapter",
        "backend": "sd15",
        "default_guidance": 2.5,
        "hf_repo": "zhengchong/CatVTON-MaskFree",
        "local_dir": "catvton-maskfree",
        "attn_version": "mix",
        "attn_subfolder": "mix-48k-1024",
        "access": "open",
        "access_short": "no login",
        "access_label": "Open download · no login",
        "license_label": "CC BY-NC-SA 4.0",
        "access_note": (
            "Public Mask-Free adapter — no Hugging Face login. "
            "Non-commercial only. Official SD 1.5 inpainting is gated; this app uses a public mirror first."
        ),
    },
    "catvton-flux": {
        "id": "catvton-flux",
        "label": "FLUX — highest fidelity",
        "tagline": "24 GB GPU, or 8 GB with Offload / 4-bit",
        "info": (
            "Official CatVTON LoRA on FLUX.1-Fill-dev. Much sharper fabric and lighting than SD 1.5, "
            "but the FLUX fill backbone is large. On 8–12 GB use Tiny/Fast plus Offload, Sequential, or 4-bit."
        ),
        "best_for": "High-quality try-on on a 24 GB+ card",
        "trained_on": "Author FLUX.1-Fill LoRA (37 MB)",
        "mask_note": "Uses the clothing mask plus FLUX fill",
        "vram_gb": 24,
        "vram_label": "24 GB+",
        "weight_label": "flux-lora",
        "foot": "FLUX.1-Fill-dev · Offload / 4-bit / Sequential for low VRAM",
        "backend": "flux-lora",
        "default_guidance": 30.0,
        "concat": "person_first",
        "hf_repo": "zhengchong/CatVTON",
        "local_dir": "catvton-flux",
        "lora_hf_path": "flux-lora/pytorch_lora_weights.safetensors",
        "lora_filename": "pytorch_lora_weights.safetensors",
        "access": "gated",
        "access_short": "HF login",
        "access_label": "HF login + FLUX license",
        "license_label": "FLUX.1 [dev] NC",
        "access_note": (
            "The CatVTON LoRA is public. The FLUX.1-Fill-dev backbone is gated — "
            "accept the license on Hugging Face and run huggingface-cli login."
        ),
    },
    "catvton-flux-alpha": {
        "id": "catvton-flux-alpha",
        "label": "FLUX Alpha — logos & fabric",
        "tagline": "24 GB GPU, or 8 GB with Offload / 4-bit",
        "info": (
            "Community FLUX LoRA trained on VITON-HD. Often keeps small prints, logos, and fabric texture "
            "better than the author LoRA. Tops-focused. Same FLUX fill backbone; use Offload/4-bit on smaller GPUs."
        ),
        "best_for": "Upper-body detail, logos, and texture",
        "trained_on": "Community VITON-HD FLUX LoRA",
        "mask_note": "Uses the clothing mask plus FLUX fill",
        "vram_gb": 24,
        "vram_label": "24 GB+",
        "weight_label": "catvton-flux-lora-alpha",
        "foot": "FLUX.1-Fill-dev · Offload / 4-bit / Sequential for low VRAM",
        "backend": "flux-lora",
        "default_guidance": 30.0,
        "concat": "garment_first",
        "hf_repo": "xiaozaa/catvton-flux-lora-alpha",
        "local_dir": "catvton-flux-alpha",
        "lora_hf_path": "pytorch_lora_weights.safetensors",
        "lora_filename": "pytorch_lora_weights.safetensors",
        "access": "gated",
        "access_short": "HF login",
        "access_label": "HF login + FLUX license",
        "license_label": "FLUX.1 [dev] NC",
        "access_note": (
            "The community LoRA is public. The FLUX.1-Fill-dev backbone is gated — "
            "accept the license on Hugging Face and run huggingface-cli login."
        ),
    },
    "catvton-flux-beta": {
        "id": "catvton-flux-beta",
        "label": "FLUX Beta — mixed garments",
        "tagline": "24 GB GPU, or 8 GB with Offload / 4-bit",
        "info": (
            "Community full FLUX fill fine-tune, not just a LoRA. Handles a wider mix of garments than Alpha. "
            "Heaviest option. Same 24 GB class on GPU; Offload or 4-bit can run it slowly on 8–12 GB."
        ),
        "best_for": "Mixed garments when VRAM is plentiful",
        "trained_on": "Community FLUX fill fine-tune",
        "mask_note": "Uses the clothing mask plus FLUX fill",
        "vram_gb": 24,
        "vram_label": "24 GB+",
        "weight_label": "catvton-flux-beta",
        "foot": "Full FLUX transformer · Offload / 4-bit / Sequential for low VRAM",
        "backend": "flux-full",
        "default_guidance": 30.0,
        "concat": "garment_first",
        "hf_repo": "xiaozaa/catvton-flux-beta",
        "local_dir": "catvton-flux-beta",
        "transformer_repo": "xiaozaa/catvton-flux-beta",
        "access": "gated",
        "access_short": "HF login",
        "access_label": "HF login + FLUX license",
        "license_label": "FLUX.1 [dev] NC",
        "access_note": (
            "The community transformer is public. FLUX.1-Fill-dev / FLUX.1-dev components are gated — "
            "accept the license on Hugging Face and run huggingface-cli login."
        ),
    },
}


def _choice_label(spec: dict) -> str:
    return f"{spec['label']} · {spec['vram_label']} · {spec['access_short']}"


MODEL_CHOICES = [(_choice_label(spec), spec["id"]) for spec in TRYON_MODELS.values()]


def get_model(model_id: str) -> dict:
    if model_id not in TRYON_MODELS:
        known = ", ".join(TRYON_MODELS)
        raise ValueError(f"Unknown model '{model_id}'. Choose one of: {known}")
    return TRYON_MODELS[model_id]


def is_flux(spec: dict) -> bool:
    return str(spec.get("backend", "sd15")).startswith("flux")


def attn_local_root(spec: dict) -> Path:
    return WEIGHTS_DIR / "local" / spec["local_dir"]


def attn_weight_path(spec: dict) -> Path:
    return attn_local_root(spec) / spec["attn_subfolder"] / "attention" / "model.safetensors"


def _download_attn(spec: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/{spec['hf_repo']}/resolve/main/"
        f"{spec['attn_subfolder']}/attention/model.safetensors"
    )
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        subprocess.check_call(
            [curl, "-L", "--retry", "8", "--retry-all-errors", "-C", "-", "-o", str(dest), url]
        )
        return
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=spec["hf_repo"],
        filename=f"{spec['attn_subfolder']}/attention/model.safetensors",
        local_dir=str(attn_local_root(spec)),
    )


def ensure_attn_ckpt(model_id: str, progress: Optional[Callable[[str], None]] = None) -> tuple[str, str]:
    """Return (attn_ckpt_dir_or_repo, attn_version), downloading the adapter if needed."""
    spec = get_model(model_id)
    dest = attn_weight_path(spec)
    if _complete_file(dest, ATTN_MIN_BYTES):
        return str(attn_local_root(spec)), spec["attn_version"]

    if progress:
        progress(f"Downloading {spec['label']} (~198 MB)")
    try:
        _download_attn(spec, dest)
    except Exception:
        if progress:
            progress("Local download failed · falling back to Hugging Face cache")
        return spec["hf_repo"], spec["attn_version"]

    if _complete_file(dest, ATTN_MIN_BYTES):
        return str(attn_local_root(spec)), spec["attn_version"]
    return spec["hf_repo"], spec["attn_version"]


FLUX_LORA_MIN_BYTES = 20_000_000


def flux_lora_path(spec: dict) -> Path:
    return WEIGHTS_DIR / "local" / spec["local_dir"] / spec.get(
        "lora_filename", "pytorch_lora_weights.safetensors"
    )


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        subprocess.check_call(
            [curl, "-L", "--retry", "8", "--retry-all-errors", "-C", "-", "-o", str(dest), url]
        )
        return
    from huggingface_hub import hf_hub_download

    repo_id, _, filename = url.split("/resolve/main/")
    repo_id = repo_id.rsplit("huggingface.co/", 1)[-1]
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent))


def ensure_flux_lora(spec: dict, progress: Optional[Callable[[str], None]] = None) -> Path | str:
    """Return a local LoRA file path, or the Hugging Face repo id as fallback."""
    dest = flux_lora_path(spec)
    if _complete_file(dest, FLUX_LORA_MIN_BYTES):
        return dest
    if progress:
        progress(f"Downloading {spec['label']} LoRA")
    url = f"https://huggingface.co/{spec['hf_repo']}/resolve/main/{spec['lora_hf_path']}"
    try:
        _download_file(url, dest)
    except Exception:
        if progress:
            progress("Local download failed · falling back to Hugging Face cache")
        return spec["hf_repo"]
    if _complete_file(dest, FLUX_LORA_MIN_BYTES):
        return dest
    return spec["hf_repo"]


HF_CACHE = Path(os.environ.get("HF_HOME", WEIGHTS_DIR / "hf"))

# hf-xet often stalls on Windows; force the classic HTTP downloader.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def ensure_dirs() -> None:
    for path in (WEIGHTS_DIR, OUTPUTS_DIR, EXAMPLES_DIR / "person", EXAMPLES_DIR / "garment", HF_CACHE):
        path.mkdir(parents=True, exist_ok=True)
