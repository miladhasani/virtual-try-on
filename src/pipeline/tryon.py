"""High-level try-on service: lazy load, generate, unload, OOM fallback."""

from __future__ import annotations

import argparse
import gc
import threading
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
    DEFAULT_VRAM_PROFILE,
    OUTPUTS_DIR,
    QUALITY_PRESETS,
    TRYON_MODELS,
    VRAM_PROFILE_FALLBACK,
    ensure_attn_ckpt,
    ensure_dirs,
    get_model,
    is_cloud,
    is_flux,
    is_gemini,
    is_hf_space,
    is_p2p,
    save_secrets,
    snap_canvas,
)
from src.monitor.resources import JobState
from src.pipeline.cancel import GenerationCancelled
from src.pipeline.masker import ClothMasker, PROTECT_PHASE, composite_tryon, vis_mask
from src.pipeline.preprocess import blur_mask, resize_and_crop, to_rgb


def format_user_error(error: str | BaseException) -> str:
    """Turn a backend exception into a short message the studio can show."""
    if isinstance(error, BaseException):
        text = str(error).strip() or error.__class__.__name__
    else:
        text = str(error).strip()
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return (
            "This GPU ran out of VRAM. Try Tiny or Fast, set VRAM mode to "
            "Offload / 4-bit / Sequential, or run Mix / Hugging Face instead."
        )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1] if len(lines[-1]) >= 12 else " ".join(lines)
    text = " ".join(text.split())
    if len(text) > 600:
        text = text[:597] + "…"
    return text or "Something went wrong while generating."


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


def gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def resolve_vram_profile(model_id: str, requested: str = DEFAULT_VRAM_PROFILE) -> str:
    if requested and requested != "Auto":
        return requested
    spec = get_model(model_id)
    vram = gpu_vram_gb()
    need = float(spec.get("vram_gb", 8))
    if vram <= 0:
        return "Sequential"
    if vram >= need:
        return "GPU"
    if is_flux(spec):
        return "Offload" if vram >= 10 else "4-bit"
    return "Offload" if vram < 7.5 else "GPU"


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
    vram_profile: str = "GPU"


class TryOnService:
    def __init__(self, job: Optional[JobState] = None) -> None:
        ensure_dirs()
        self.job = job or JobState()
        self.device = detect_device()
        self.dtype = detect_dtype(self.device)
        self.pipeline = None
        self.masker: Optional[ClothMasker] = None
        self.loaded_model_id: Optional[str] = None
        self.loaded_vram_profile: Optional[str] = None
        self.history: list[TryOnResult] = []
        self._cancel = threading.Event()

    def request_stop(self) -> bool:
        """Ask the in-flight job to exit at the next diffusion step."""
        running = self.job.started_at is not None
        self._cancel.set()
        if running:
            self._set_phase("Stopping")
        return running

    def _raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise GenerationCancelled("Generation stopped.")

    def _finish_cancelled(self) -> None:
        elapsed = time.time() - (self.job.started_at or time.time())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.job.error = None
        self._set_phase(f"Stopped · {elapsed:.1f}s")
        self.job.started_at = None
        self.job.step = 0
        self.job.total = 0

    def fail(self, error: str | BaseException) -> None:
        """Record a user-visible failure and mark the job idle."""
        self.job.error = format_user_error(error)
        self.job.started_at = None
        self._set_phase("Error", step=0, total=0)

    def _drop_pipeline(self) -> None:
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def _set_phase(self, phase: str, step: Optional[int] = None, total: Optional[int] = None) -> None:
        self.job.phase = phase
        self.job.message = phase
        if step is not None:
            self.job.step = step
        if total is not None:
            self.job.total = total

    def begin_job(self, phase: str = "Starting") -> None:
        self._cancel.clear()
        self.job.error = None
        if self.job.started_at is None:
            self.job.started_at = time.time()
        self._set_phase(phase, step=0, total=0)

    def is_busy(self) -> bool:
        return self.job.started_at is not None

    def load(
        self,
        model_id: str = DEFAULT_MODEL,
        progress: Optional[Callable[[str], None]] = None,
        vram_profile: str = DEFAULT_VRAM_PROFILE,
    ) -> None:
        spec = get_model(model_id)
        if is_cloud(spec):
            return
        resolved = resolve_vram_profile(model_id, vram_profile)

        def note(text: str) -> None:
            self._raise_if_cancelled()
            self._set_phase(text, step=0, total=0)
            if progress:
                progress(text)

        if self.masker is None:
            note("Loading model · cloth parser")
            # Parser stays on CPU so the laptop GPU is reserved for diffusion.
            self.masker = ClothMasker(device="cpu")

        same_model = self.pipeline is not None and self.loaded_model_id == model_id
        if same_model and self.loaded_vram_profile == resolved:
            return

        current = get_model(self.loaded_model_id) if self.loaded_model_id else None
        want_flux = is_flux(spec)
        have_flux = bool(current and is_flux(current))
        want_p2p = is_p2p(spec)
        have_p2p = bool(current and is_p2p(current))
        profile_changed = self.loaded_vram_profile != resolved
        family_changed = have_flux != want_flux or have_p2p != want_p2p

        if want_flux:
            rebuild = (
                self.pipeline is None
                or not have_flux
                or profile_changed
                or spec.get("backend") == "flux-full"
                or (current or {}).get("backend") == "flux-full"
            )
            if rebuild:
                if self.pipeline is not None:
                    note("Unloading previous checkpoint")
                    self._drop_pipeline()
                from src.pipeline.flux_tryon import CatVTONFluxPipeline

                note(f"Loading model · {spec['label']} · {resolved}")
                if self.device.type == "cpu":
                    note("No CUDA GPU — FLUX will be extremely slow")
                self.pipeline = CatVTONFluxPipeline.from_spec(
                    spec, self.device, self.dtype, progress=note, vram_profile=resolved
                )
                resolved = getattr(self.pipeline, "vram_profile", resolved)
            else:
                note(f"Switching FLUX LoRA · {spec['label']}")
                self.pipeline.switch_spec(spec, progress=note)
        else:
            if self.pipeline is not None and (family_changed or profile_changed):
                note("Unloading previous checkpoint")
                self._drop_pipeline()

            attn_ckpt, attn_version = ensure_attn_ckpt(model_id, progress=note)
            if self.pipeline is None:
                note(f"Loading model · {spec['label']}")
                if self.device.type == "cpu":
                    note("No CUDA GPU — generation will be very slow")
                if want_p2p:
                    from src.pipeline.catvton import CatVTONPix2PixPipeline

                    self.pipeline = CatVTONPix2PixPipeline(
                        attn_ckpt=attn_ckpt,
                        attn_ckpt_version=attn_version,
                        weight_dtype=self.dtype,
                        device=self.device,
                        use_tf32=self.device.type == "cuda",
                    )
                else:
                    from src.pipeline.catvton import CatVTONPipeline

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
            self.pipeline.apply_vram_profile(resolved)

        self.loaded_model_id = model_id
        self.loaded_vram_profile = resolved
        if self.job.started_at is None:
            note("Idle")

    def unload(self) -> None:
        self._drop_pipeline()
        self.masker = None
        self.loaded_model_id = None
        self.loaded_vram_profile = None
        self.job.error = None
        self._set_phase("Idle · models unloaded", step=0, total=0)
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
        width: Optional[int] = None,
        height: Optional[int] = None,
        vram_profile: str = DEFAULT_VRAM_PROFILE,
        allow_oom_fallback: bool = True,
        gemini_api_key: str = "",
        hf_token: str = "",
        _root: bool = True,
    ) -> TryOnResult:
        if preset not in QUALITY_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        spec = QUALITY_PRESETS[preset]
        width, height = snap_canvas(width or spec["width"], height or spec["height"])
        steps = num_inference_steps or spec["steps"]
        resolved = resolve_vram_profile(model_id, vram_profile)
        if gemini_api_key or hf_token:
            save_secrets(gemini_api_key=gemini_api_key, hf_token=hf_token)

        if _root:
            self.begin_job("Starting")
        try:
            self._raise_if_cancelled()
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
                resolved,
                gemini_api_key=gemini_api_key,
                hf_token=hf_token,
            )
        except torch.cuda.OutOfMemoryError:
            nxt_profile = VRAM_PROFILE_FALLBACK.get(resolved)
            nxt_preset = {"Studio": "Quality", "Quality": "Balanced", "Balanced": "Fast", "Fast": "Tiny"}.get(preset)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if allow_oom_fallback and nxt_profile:
                self._set_phase(f"OOM · retrying with {nxt_profile} VRAM mode")
                self._drop_pipeline()
                self.loaded_model_id = None
                self.loaded_vram_profile = None
                return self.generate(
                    person,
                    garment,
                    cloth_type=cloth_type,
                    preset=preset,
                    model_id=model_id,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    width=width,
                    height=height,
                    vram_profile=nxt_profile,
                    allow_oom_fallback=True,
                    gemini_api_key=gemini_api_key,
                    hf_token=hf_token,
                    _root=False,
                )
            if allow_oom_fallback and nxt_preset:
                smaller = QUALITY_PRESETS[nxt_preset]
                self._set_phase(f"OOM · shrinking canvas to {nxt_preset}")
                return self.generate(
                    person,
                    garment,
                    cloth_type=cloth_type,
                    preset=nxt_preset,
                    model_id=model_id,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    width=min(width, smaller["width"]),
                    height=min(height, smaller["height"]),
                    vram_profile=resolved,
                    allow_oom_fallback=nxt_preset != "Tiny",
                    gemini_api_key=gemini_api_key,
                    hf_token=hf_token,
                    _root=False,
                )
            spec = get_model(model_id)
            if is_flux(spec):
                self.fail(
                    "This FLUX run ran out of VRAM. Try Tiny or Fast, set VRAM mode to "
                    "Offload / 4-bit / Sequential, or switch to Mix on this GPU."
                )
            else:
                self.fail(
                    "This GPU ran out of VRAM. Try Tiny or Fast, set VRAM mode to "
                    "Offload / 4-bit / Sequential, or run Mix / Hugging Face instead."
                )
            raise RuntimeError(self.job.error) from None
        except GenerationCancelled:
            self._finish_cancelled()
            raise
        except Exception as exc:
            self.fail(exc)
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
        vram_profile: str,
        gemini_api_key: str = "",
        hf_token: str = "",
    ) -> TryOnResult:
        person_img = to_rgb(person)
        garment_img = to_rgb(garment)
        person_resized = resize_and_crop(person_img, (width, height))
        model_spec = get_model(model_id)

        if is_cloud(model_spec):
            return self._generate_cloud(
                person_resized,
                garment_img,
                cloth_type,
                preset,
                model_id,
                width,
                height,
                steps,
                guidance_scale,
                seed,
                gemini_api_key=gemini_api_key,
                hf_token=hf_token,
            )

        self.load(model_id, vram_profile=vram_profile)
        self._raise_if_cancelled()
        parse = None
        if is_p2p(model_spec):
            self._set_phase("Preparing", step=0, total=0)
            self._raise_if_cancelled()
            if self.masker is None:
                self.masker = ClothMasker(device="cpu")
            parse = self.masker.parse(person_resized)
            mask = Image.new("L", person_resized.size, 0)
            masked_person = person_resized
        else:
            self._set_phase("Masking", step=0, total=0)
            self._raise_if_cancelled()
            assert self.masker is not None
            mask = self.masker(person_resized, cloth_type)
            mask = blur_mask(mask, blur_factor=9)
            masked_person = vis_mask(person_resized, mask)

        self._set_phase("Generating", step=0, total=steps)
        self._raise_if_cancelled()
        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        assert self.pipeline is not None

        def on_step(step: int, total: int) -> None:
            self._raise_if_cancelled()
            self._set_phase("Generating", step=step, total=total)

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
            cancel_check=self._raise_if_cancelled,
        )
        if is_p2p(model_spec) and parse is not None:
            self._set_phase(PROTECT_PHASE.get(cloth_type, PROTECT_PHASE["upper"]), step=steps, total=steps)
            self._raise_if_cancelled()
            result, mask = composite_tryon(person_resized, result, parse, cloth_type)
            masked_person = vis_mask(person_resized, mask)
        return self._pack_result(
            result,
            person_resized,
            garment_img,
            mask,
            masked_person,
            preset,
            width,
            height,
            steps,
            model_id,
            self.loaded_vram_profile or vram_profile,
        )

    def _generate_cloud(
        self,
        person_resized: Image.Image,
        garment_img: Image.Image,
        cloth_type: str,
        preset: str,
        model_id: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        gemini_api_key: str = "",
        hf_token: str = "",
    ) -> TryOnResult:
        from src.pipeline.cloud import gemini_tryon, hf_space_tryon

        spec = get_model(model_id)
        mask = Image.new("L", person_resized.size, 0)
        masked_person = person_resized

        def note(text: str) -> None:
            self._raise_if_cancelled()
            self._set_phase(text, step=0, total=0)

        note(f"Calling {spec['label']}")
        if is_gemini(spec):
            result = gemini_tryon(
                person_resized,
                garment_img,
                cloth_type=cloth_type,
                width=width,
                height=height,
                seed=seed,
                api_key=gemini_api_key,
                models=spec.get("gemini_models"),
                cancel_check=self._raise_if_cancelled,
                progress=note,
            )
        elif is_hf_space(spec):
            result = hf_space_tryon(
                person_resized,
                garment_img,
                cloth_type=cloth_type,
                steps=steps,
                guidance=guidance_scale,
                seed=seed,
                hf_token=hf_token,
                cancel_check=self._raise_if_cancelled,
                progress=note,
            )
        else:
            raise RuntimeError(f"Unknown cloud backend: {spec.get('backend')}")
        return self._pack_result(
            result,
            person_resized,
            garment_img,
            mask,
            masked_person,
            preset,
            width,
            height,
            steps,
            model_id,
            "Cloud",
        )

    def _pack_result(
        self,
        result: Image.Image,
        person_resized: Image.Image,
        garment_img: Image.Image,
        mask: Image.Image,
        masked_person: Image.Image,
        preset: str,
        width: int,
        height: int,
        steps: int,
        model_id: str,
        vram_profile: str,
    ) -> TryOnResult:
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
            vram_profile=vram_profile,
        )
        self.history.insert(0, payload)
        self.history = self.history[:12]
        self.job.error = None
        self._set_phase(f"Done · {elapsed:.1f}s", step=max(steps, 1), total=max(steps, 1))
        self.job.started_at = None
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="CatVTON virtual try-on CLI")
    parser.add_argument("--person", type=str, help="Person image path")
    parser.add_argument("--garment", type=str, help="Garment image path")
    parser.add_argument("--cloth-type", default="upper", choices=["upper", "lower", "overall"])
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=list(QUALITY_PRESETS))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(TRYON_MODELS))
    parser.add_argument("--vram", default=DEFAULT_VRAM_PROFILE, choices=["Auto", "GPU", "Offload", "Sequential", "4-bit"])
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
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
        vram_profile=args.vram,
        width=args.width,
        height=args.height,
    )
    dest = Path(args.out) if args.out else result.path
    result.result.save(dest)
    print(f"saved {dest} in {result.elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
