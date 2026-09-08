"""High-level try-on service: lazy load, generate, unload, OOM fallback."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from src.config import (
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL,
    DEFAULT_PRESET,
    DEFAULT_SEED,
    OUTPUTS_DIR,
    QUALITY_PRESETS,
    TRYON_MODELS,
    ensure_attn_ckpt,
    ensure_dirs,
    get_model,
)
from src.monitor.resources import JobState
from src.pipeline.masker import ClothMasker, vis_mask
from src.pipeline.preprocess import blur_mask, resize_and_crop, to_rgb


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def detect_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    # Ada / Ampere: bf16 is preferred; fall back to fp16 if the driver rejects it.
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


@dataclass
class TryOnResult:
    result: Image.Image
    person: Image.Image
    garment: Image.Image
    mask: Image.Image
    masked_person: Image.Image
    elapsed_s: float
    preset: str
    width: int
    height: int
    steps: int
    path: Path
    model_id: str
    model_label: str


class TryOnService:
    def __init__(self, job: Optional[JobState] = None) -> None:
        ensure_dirs()
        self.job = job or JobState()
        self.device = detect_device()
        self.dtype = detect_dtype(self.device)
        self.pipeline = None
        self.masker: Optional[ClothMasker] = None
        self.loaded_model_id: Optional[str] = None
        self.history: list[TryOnResult] = []

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def _set_phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self.job.phase = phase
        self.job.step = step
        self.job.total = total
        self.job.message = phase

    def load(
        self,
        model_id: str = DEFAULT_MODEL,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        spec = get_model(model_id)

        def note(text: str) -> None:
            self._set_phase(text)
            if progress:
                progress(text)

        if self.pipeline is not None and self.masker is not None and self.loaded_model_id == model_id:
            return

        from src.pipeline.catvton import CatVTONPipeline

        if self.masker is None:
            note("Loading model · cloth parser")
            # Parser stays on CPU so the 8 GB laptop GPU is reserved for diffusion.
            self.masker = ClothMasker(device="cpu")

        attn_ckpt, attn_version = ensure_attn_ckpt(model_id, progress=note)

        if self.pipeline is None:
            note(f"Loading model · {spec['label']}")
            if self.device.type == "cpu":
                note("No CUDA GPU — generation will be very slow")
            self.pipeline = CatVTONPipeline(
                attn_ckpt=attn_ckpt,
                attn_ckpt_version=attn_version,
                weight_dtype=self.dtype,
                device=self.device,
                use_tf32=self.device.type == "cuda",
            )
        elif self.loaded_model_id != model_id:
            note(f"Switching model · {spec['label']}")
            self.pipeline.reload_attn(attn_ckpt, attn_version)

        self.loaded_model_id = model_id
        if self.job.started_at is None:
            note("Idle")

    def unload(self) -> None:
        self.pipeline = None
        self.masker = None
        self.loaded_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self._set_phase("Idle · models unloaded")
        self.job.started_at = None

    def generate(
        self,
        person: Image.Image | str,
        garment: Image.Image | str,
        cloth_type: str = "upper",
        preset: str = DEFAULT_PRESET,
        model_id: str = DEFAULT_MODEL,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = DEFAULT_GUIDANCE,
        seed: int = DEFAULT_SEED,
        allow_oom_fallback: bool = True,
    ) -> TryOnResult:
        if preset not in QUALITY_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        spec = QUALITY_PRESETS[preset]
        width, height = spec["width"], spec["height"]
        steps = num_inference_steps or spec["steps"]

        self.job.started_at = time.time()
        try:
            return self._generate_once(
                person,
                garment,
                cloth_type,
                preset,
                model_id,
                width,
                height,
                steps,
                guidance_scale,
                seed,
            )
        except torch.cuda.OutOfMemoryError:
            if not allow_oom_fallback or preset == "Fast":
                self._set_phase("Error · out of GPU memory")
                self.job.started_at = None
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._set_phase("OOM · retrying with Fast preset")
            return self.generate(
                person,
                garment,
                cloth_type=cloth_type,
                preset="Fast",
                model_id=model_id,
                guidance_scale=guidance_scale,
                seed=seed,
                allow_oom_fallback=False,
            )
        except Exception:
            self._set_phase("Error")
            self.job.started_at = None
            raise

    def _generate_once(
        self,
        person,
        garment,
        cloth_type: str,
        preset: str,
        model_id: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> TryOnResult:
        self.load(model_id)
        person_img = to_rgb(person)
        garment_img = to_rgb(garment)
        person_resized = resize_and_crop(person_img, (width, height))

        self._set_phase("Masking")
        assert self.masker is not None
        mask = self.masker(person_resized, cloth_type)
        mask = blur_mask(mask, blur_factor=9)
        masked_person = vis_mask(person_resized, mask)

        self._set_phase("Generating", 0, steps)
        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        assert self.pipeline is not None

        def on_step(step: int, total: int) -> None:
            self._set_phase(f"Generating ({step}/{total})", step, total)

        result = self.pipeline(
            image=person_resized,
            condition_image=garment_img,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            progress_callback=on_step,
        )

        elapsed = time.time() - (self.job.started_at or time.time())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUTS_DIR / f"tryon_{stamp}.png"
        result.save(out_path)

        payload = TryOnResult(
            result=result,
            person=person_resized,
            garment=garment_img,
            mask=mask,
            masked_person=masked_person,
            elapsed_s=elapsed,
            preset=preset,
            width=width,
            height=height,
            steps=steps,
            path=out_path,
            model_id=model_id,
            model_label=get_model(model_id)["label"],
        )
        self.history.insert(0, payload)
        self.history = self.history[:12]
        self._set_phase(f"Done · {elapsed:.1f}s")
        self.job.started_at = None
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="CatVTON virtual try-on CLI")
    parser.add_argument("--person", type=str, help="Person image path")
    parser.add_argument("--garment", type=str, help="Garment image path")
    parser.add_argument("--cloth-type", default="upper", choices=["upper", "lower", "overall"])
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=list(QUALITY_PRESETS))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(TRYON_MODELS))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--info", action="store_true", help="Print device info and exit")
    args = parser.parse_args()

    device = detect_device()
    dtype = detect_dtype(device)
    print(f"device={device} dtype={dtype}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"vram={props.total_memory / 1024**3:.1f} GB")
    if args.info:
        return
    if not args.person or not args.garment:
        parser.error("--person and --garment are required unless --info is set")

    service = TryOnService()
    result = service.generate(
        args.person,
        args.garment,
        cloth_type=args.cloth_type,
        preset=args.preset,
        model_id=args.model,
    )
    dest = Path(args.out) if args.out else result.path
    result.result.save(dest)
    print(f"saved {dest} in {result.elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
