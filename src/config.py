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
    "Fast": {"width": 384, "height": 512, "steps": 20},
    "Balanced": {"width": 576, "height": 768, "steps": 30},
    "Quality": {"width": 768, "height": 1024, "steps": 50},
}

DEFAULT_PRESET = "Balanced"
DEFAULT_CLOTH_TYPE = "upper"
DEFAULT_GUIDANCE = 2.5
DEFAULT_SEED = 42
DEFAULT_MODEL = "catvton-mix"

CLOTH_TYPES = ["upper", "lower", "overall"]

# Shared SD inpainting UNet; only the ~198 MB attention adapter changes per model.
TRYON_MODELS = {
    "catvton-mix": {
        "id": "catvton-mix",
        "label": "CatVTON Mix — recommended",
        "info": "General try-on trained on mixed data. Best default for tops, pants, and outfits.",
        "hf_repo": "zhengchong/CatVTON",
        "local_dir": "catvton",
        "attn_version": "mix",
        "attn_subfolder": "mix-48k-1024",
    },
    "catvton-vitonhd": {
        "id": "catvton-vitonhd",
        "label": "CatVTON VITON-HD — tops",
        "info": "Trained on VITON-HD. Strongest on upper-body garments (shirts, jackets, tees).",
        "hf_repo": "zhengchong/CatVTON",
        "local_dir": "catvton",
        "attn_version": "vitonhd",
        "attn_subfolder": "vitonhd-16k-512",
    },
    "catvton-dresscode": {
        "id": "catvton-dresscode",
        "label": "CatVTON DressCode — dresses",
        "info": "Trained on DressCode. Better for dresses, full looks, and more garment variety.",
        "hf_repo": "zhengchong/CatVTON",
        "local_dir": "catvton",
        "attn_version": "dresscode",
        "attn_subfolder": "dresscode-16k-512",
    },
    "catvton-maskfree": {
        "id": "catvton-maskfree",
        "label": "CatVTON Mask-Free — messy photos",
        "info": "Mask-free Mix checkpoint. More forgiving when the auto clothing mask is imperfect.",
        "hf_repo": "zhengchong/CatVTON-MaskFree",
        "local_dir": "catvton-maskfree",
        "attn_version": "mix",
        "attn_subfolder": "mix-48k-1024",
    },
}

MODEL_CHOICES = [(spec["label"], spec["id"]) for spec in TRYON_MODELS.values()]


def get_model(model_id: str) -> dict:
    if model_id not in TRYON_MODELS:
        known = ", ".join(TRYON_MODELS)
        raise ValueError(f"Unknown model '{model_id}'. Choose one of: {known}")
    return TRYON_MODELS[model_id]


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


HF_CACHE = Path(os.environ.get("HF_HOME", WEIGHTS_DIR / "hf"))

# hf-xet often stalls on Windows; force the classic HTTP downloader.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def ensure_dirs() -> None:
    for path in (WEIGHTS_DIR, OUTPUTS_DIR, EXAMPLES_DIR / "person", EXAMPLES_DIR / "garment", HF_CACHE):
        path.mkdir(parents=True, exist_ok=True)
