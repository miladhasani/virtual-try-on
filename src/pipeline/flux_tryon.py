"""FLUX.1-Fill virtual try-on (CatVTON-FLUX LoRA and community fine-tunes).

Needs about 24 GB of VRAM. First load downloads the gated FLUX fill backbone.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from PIL import Image

from src.config import HF_CACHE, ensure_flux_lora, is_flux
from src.pipeline.preprocess import resize_and_crop, resize_and_padding

FLUX_PROMPT = (
    "The pair of images highlights a clothing and its styling on a model, "
    "high resolution, 4K, 8K; [IMAGE1] Detailed product shot of a clothing "
    "[IMAGE2] The same cloth is worn by a model in a lifestyle setting."
)


def _hconcat(left: Image.Image, right: Image.Image) -> Image.Image:
    width, height = left.size
    right = right.resize((width, height), Image.LANCZOS)
    canvas = Image.new(left.mode, (width * 2, height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    return canvas


def _bitsandbytes_config():
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError:
        from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_transformer(repo_id: str, dtype: torch.dtype, quantize: bool):
    from diffusers import FluxTransformer2DModel

    kwargs = {"torch_dtype": dtype, "cache_dir": str(HF_CACHE)}
    if quantize:
        kwargs["quantization_config"] = _bitsandbytes_config()
    print(f"Loading FLUX transformer from {repo_id} ({'4-bit' if quantize else dtype}) ...", flush=True)
    return FluxTransformer2DModel.from_pretrained(repo_id, **kwargs)


def _load_fill_pipeline(dtype: torch.dtype, transformer=None):
    from diffusers import FluxFillPipeline, FluxTransformer2DModel

    cache = str(HF_CACHE)
    last_error: Exception | None = None
    repos = [
        os.environ.get("CATVTON_FLUX_FILL"),
        "black-forest-labs/FLUX.1-Fill-dev",
    ]
    for repo in repos:
        if not repo:
            continue
        try:
            print(f"Loading FLUX fill from {repo} ...", flush=True)
            kwargs = {"torch_dtype": dtype, "cache_dir": cache}
            if transformer is not None:
                kwargs["transformer"] = transformer
            pipe = FluxFillPipeline.from_pretrained(repo, **kwargs)
            print(f"Loaded FLUX fill from {repo}", flush=True)
            return pipe
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Skipping {repo}: {exc}", flush=True)

    try:
        transformer = transformer or FluxTransformer2DModel.from_pretrained(
            "xiaozaa/flux1-fill-dev-diffusers",
            torch_dtype=dtype,
            cache_dir=cache,
        )
        print("Loading FLUX.1-dev components around the fill transformer ...", flush=True)
        return FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=cache,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not load FLUX.1-Fill-dev. Accept the FLUX license on Hugging Face, "
            "run `huggingface-cli login`, or set CATVTON_FLUX_FILL to a local checkpoint.\n"
            f"Last error: {last_error or exc}"
        ) from exc


class CatVTONFluxPipeline:
    """Wrapper around diffusers FluxFillPipeline with CatVTON side-by-side conditioning."""

    def __init__(
        self,
        pipe,
        spec: dict,
        device: torch.device,
        dtype: torch.dtype,
        offloaded: bool,
        vram_profile: str = "GPU",
    ) -> None:
        self.pipe = pipe
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.offloaded = offloaded
        self.vram_profile = vram_profile
        self.max_sequence_length = 256 if vram_profile in {"Sequential", "4-bit"} else 512

    @classmethod
    def from_spec(
        cls,
        spec: dict,
        device: torch.device,
        dtype: torch.dtype,
        progress: Optional[Callable[[str], None]] = None,
        vram_profile: str = "GPU",
    ) -> "CatVTONFluxPipeline":
        def note(text: str) -> None:
            if progress:
                progress(text)

        weight_dtype = torch.bfloat16 if device.type == "cuda" else dtype
        quantize = vram_profile == "4-bit"
        transformer = None
        fill_transformer_repo = os.environ.get("CATVTON_FLUX_FILL_TRANSFORMER", "xiaozaa/flux1-fill-dev-diffusers")

        if spec.get("backend") == "flux-full":
            note(f"Loading FLUX transformer · {spec['label']}")
            try:
                transformer = _load_transformer(spec["transformer_repo"], weight_dtype, quantize)
            except Exception as exc:
                if quantize:
                    note("4-bit load failed · falling back to Sequential offload")
                    vram_profile = "Sequential"
                    quantize = False
                    transformer = _load_transformer(spec["transformer_repo"], weight_dtype, False)
                else:
                    raise RuntimeError(f"Could not load FLUX transformer: {exc}") from exc
        elif quantize:
            note("Loading 4-bit FLUX fill transformer")
            try:
                transformer = _load_transformer(fill_transformer_repo, weight_dtype, True)
            except Exception as exc:
                note(f"4-bit unavailable ({exc}) · Sequential offload instead")
                vram_profile = "Sequential"
                quantize = False
                transformer = None

        note("Loading FLUX.1-Fill-dev · several GB on first use")
        pipe = _load_fill_pipeline(weight_dtype, transformer=transformer)
        pipe.vae.enable_slicing()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

        offloaded = vram_profile != "GPU"
        if device.type != "cuda":
            pipe.to(device)
            offloaded = False
            vram_profile = "GPU"
        elif vram_profile == "GPU":
            try:
                pipe.to(device)
                offloaded = False
            except torch.cuda.OutOfMemoryError:
                note("GPU full · enabling CPU offload")
                torch.cuda.empty_cache()
                pipe.enable_model_cpu_offload()
                offloaded = True
                vram_profile = "Offload"
        elif vram_profile == "Sequential":
            note("Sequential CPU offload · lowest VRAM, much slower")
            pipe.enable_sequential_cpu_offload()
        else:
            note("Model CPU offload · slower, less VRAM")
            pipe.enable_model_cpu_offload()

        wrapper = cls(pipe, spec, device, weight_dtype, offloaded, vram_profile)
        if spec.get("backend") == "flux-lora":
            wrapper._load_lora(spec, progress=progress)
        return wrapper

    def _load_lora(self, spec: dict, progress: Optional[Callable[[str], None]] = None) -> None:
        source = ensure_flux_lora(spec, progress=progress)
        if isinstance(source, str):
            self.pipe.load_lora_weights(source, weight_name=spec.get("lora_hf_path"))
            return
        self.pipe.load_lora_weights(str(source.parent), weight_name=source.name)

    def switch_spec(self, spec: dict, progress: Optional[Callable[[str], None]] = None) -> None:
        if not is_flux(spec):
            raise ValueError("CatVTONFluxPipeline can only switch among FLUX checkpoints")
        if spec.get("backend") != self.spec.get("backend") or spec.get("backend") == "flux-full":
            raise RuntimeError("Rebuild the FLUX pipeline when switching to a full transformer")
        if hasattr(self.pipe, "unload_lora_weights"):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
        self._load_lora(spec, progress=progress)
        self.spec = spec

    def __call__(
        self,
        image: Image.Image,
        condition_image: Image.Image,
        mask: Image.Image,
        num_inference_steps: int = 30,
        guidance_scale: float = 30.0,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Image.Image:
        del eta
        person = resize_and_crop(image, (width, height))
        garment = resize_and_padding(condition_image, (width, height))
        mask_img = resize_and_crop(mask.convert("RGB"), (width, height))
        blank = Image.new("RGB", (width, height), (0, 0, 0))

        if self.spec.get("concat") == "person_first":
            inpaint = _hconcat(person, garment)
            extended_mask = _hconcat(mask_img, blank)
            crop_box = (0, 0, width, height)
        else:
            inpaint = _hconcat(garment, person)
            extended_mask = _hconcat(blank, mask_img)
            crop_box = (width, 0, width * 2, height)

        if cancel_check is not None:
            cancel_check()

        def on_step(pipe, step_index, timestep, callback_kwargs):
            del pipe, timestep
            if cancel_check is not None:
                cancel_check()
            if progress_callback is not None:
                progress_callback(int(step_index) + 1, num_inference_steps)
            return callback_kwargs

        result = self.pipe(
            prompt=FLUX_PROMPT,
            image=inpaint,
            mask_image=extended_mask,
            height=height,
            width=width * 2,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=self.max_sequence_length,
            callback_on_step_end=on_step,
        ).images[0]
        return result.crop(crop_box)
