"""CatVTON inference pipeline, adapted from Zheng-Chong/CatVTON.

Original: https://github.com/Zheng-Chong/CatVTON
License: CC BY-NC-SA 4.0
"""

from __future__ import annotations

import inspect
import os
from typing import Callable, Optional, Union

import numpy as np
import PIL.Image
import torch
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from tqdm import tqdm

from src.config import (
    ATTN_REPO,
    ATTN_VERSION,
    BASE_MODEL_CANDIDATES,
    HF_CACHE,
    P2P_BASE_CANDIDATES,
    VAE_REPO,
)
from src.pipeline.attn_processor import SkipAttnProcessor
from src.pipeline.catvton_utils import get_trainable_module, init_adapter
from src.pipeline.preprocess import (
    compute_vae_encodings,
    numpy_to_pil,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)

ATTN_SUBFOLDERS = {
    "mix": "mix-48k-1024",
    "vitonhd": "vitonhd-16k-512",
    "dresscode": "dresscode-16k-512",
}


def _sd15_ddim_scheduler() -> DDIMScheduler:
    """Build a standard SD 1.5 DDIM scheduler without downloading gated repos."""
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
    )


def _load_inpainting_unet(weight_dtype: torch.dtype, device: torch.device) -> UNet2DConditionModel:
    last_error: Exception | None = None
    for repo_id in BASE_MODEL_CANDIDATES:
        try:
            print(f"Loading inpainting UNet from {repo_id} ...", flush=True)
            load_kwargs = {"subfolder": "unet", "torch_dtype": weight_dtype, "cache_dir": str(HF_CACHE)}
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    repo_id, variant="fp16", use_safetensors=True, **load_kwargs
                )
            except Exception:
                unet = UNet2DConditionModel.from_pretrained(repo_id, **load_kwargs)
            print(f"Loaded inpainting UNet from {repo_id}", flush=True)
            return unet.to(device)
        except Exception as exc:  # noqa: BLE001 — try the next public/gated host
            last_error = exc
            print(f"Skipping {repo_id}: {exc}", flush=True)
            continue
    raise RuntimeError(
        "Could not download the SD 1.5 inpainting UNet. "
        "Accept the license on Hugging Face and run `huggingface-cli login`, "
        "or set CATVTON_BASE_MODEL to a local/public inpainting checkpoint.\n"
        f"Tried: {', '.join(BASE_MODEL_CANDIDATES)}\n"
        f"Last error: {last_error}"
    )


def _from_pretrained_unet(repo_id: str, weight_dtype: torch.dtype) -> UNet2DConditionModel:
    load_kwargs = {"subfolder": "unet", "torch_dtype": weight_dtype, "cache_dir": str(HF_CACHE)}
    try:
        return UNet2DConditionModel.from_pretrained(
            repo_id, variant="fp16", use_safetensors=True, **load_kwargs
        )
    except Exception:
        try:
            return UNet2DConditionModel.from_pretrained(repo_id, **load_kwargs)
        except Exception:
            return UNet2DConditionModel.from_pretrained(
                repo_id, torch_dtype=weight_dtype, cache_dir=str(HF_CACHE)
            )


def _load_pix2pix_unet(weight_dtype: torch.dtype, device: torch.device) -> UNet2DConditionModel:
    last_error: Exception | None = None
    for repo_id in P2P_BASE_CANDIDATES:
        try:
            print(f"Loading InstructPix2Pix UNet from {repo_id} ...", flush=True)
            unet = _from_pretrained_unet(repo_id, weight_dtype)
            in_channels = int(getattr(unet.config, "in_channels", 0) or 0)
            if in_channels != 8:
                raise RuntimeError(
                    f"{repo_id} UNet has in_channels={in_channels}, expected 8 for InstructPix2Pix"
                )
            print(f"Loaded InstructPix2Pix UNet from {repo_id}", flush=True)
            return unet.to(device)
        except Exception as exc:  # noqa: BLE001 — try the next host
            last_error = exc
            print(f"Skipping {repo_id}: {exc}", flush=True)
            continue
    raise RuntimeError(
        "Could not download the InstructPix2Pix UNet needed by Mask-Free CatVTON. "
        "Set CATVTON_P2P_MODEL to timbrooks/instruct-pix2pix or a local copy.\n"
        f"Tried: {', '.join(P2P_BASE_CANDIDATES)}\n"
        f"Last error: {last_error}"
    )


class CatVTONPipeline:
    def __init__(
        self,
        attn_ckpt: str = ATTN_REPO,
        attn_ckpt_version: str = ATTN_VERSION,
        weight_dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        use_tf32: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype

        self.noise_scheduler = _sd15_ddim_scheduler()
        self.vae = AutoencoderKL.from_pretrained(
            VAE_REPO, torch_dtype=weight_dtype, cache_dir=str(HF_CACHE)
        )
        self.vae.enable_slicing()
        self.unet = _load_inpainting_unet(weight_dtype, self.device)
        self._attach_attn(attn_ckpt, attn_ckpt_version)

        if use_tf32 and self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def _attach_attn(self, attn_ckpt: str, attn_ckpt_version: str) -> None:
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self._load_attn_checkpoint(attn_ckpt, attn_ckpt_version)
        self.unet.eval()
        self.vae.eval()
        self.vram_profile = "GPU"
        self.apply_vram_profile("GPU")

    def apply_vram_profile(self, profile: str) -> None:
        """Keep the UNet on GPU; park the VAE in RAM when saving VRAM."""
        if profile == "4-bit":
            profile = "Offload"
        self.vram_profile = profile
        if self.device.type != "cuda" or profile == "GPU":
            self.vae.to(self.device)
            self.unet.to(self.device)
            return
        self.vae.to("cpu")
        self.unet.to(self.device)
        try:
            self.unet.set_attention_slice("auto")
        except Exception:
            pass
        if profile == "Sequential":
            try:
                self.unet.enable_forward_chunking(chunk_size=1, dim=0)
            except Exception:
                pass

    def _load_attn_checkpoint(self, attn_ckpt: str, version: str) -> None:
        sub_folder = ATTN_SUBFOLDERS[version]
        if os.path.isdir(attn_ckpt):
            repo_path = attn_ckpt
        else:
            repo_path = snapshot_download(
                repo_id=attn_ckpt,
                cache_dir=str(HF_CACHE),
                allow_patterns=[f"{sub_folder}/attention/**"],
            )
        load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, "attention"))
        self.attn_ckpt = repo_path
        self.attn_version = version

    def reload_attn(self, attn_ckpt: str, version: str) -> None:
        """Swap CatVTON attention weights without reloading the UNet / VAE."""
        self._load_attn_checkpoint(attn_ckpt, version)

    def to(self, device: Union[str, torch.device]) -> "CatVTONPipeline":
        self.device = torch.device(device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        return self

    def cpu(self) -> "CatVTONPipeline":
        return self.to("cpu")

    @staticmethod
    def _prepare_extra_step_kwargs(scheduler, generator, eta: float) -> dict:
        extra_step_kwargs = {}
        params = inspect.signature(scheduler.step).parameters
        if "eta" in params:
            extra_step_kwargs["eta"] = eta
        if "generator" in params:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image,
        condition_image: PIL.Image.Image,
        mask: PIL.Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> PIL.Image.Image:
        concat_dim = -2
        if cancel_check is not None:
            cancel_check()
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))

        image_t = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_t = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask_t = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        masked_image = image_t * (mask_t < 0.5)

        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_t, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask_t, size=masked_latent.shape[-2:], mode="nearest")
        del image_t, mask_t, condition_t, masked_image

        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            masked_latent_concat = torch.cat(
                [torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim), masked_latent_concat]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        extra_step_kwargs = self._prepare_extra_step_kwargs(self.noise_scheduler, generator, eta)
        for i, t in enumerate(tqdm(timesteps, desc="CatVTON")):
            if cancel_check is not None:
                cancel_check()
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            inpainting_input = torch.cat(
                [latent_model_input, mask_latent_concat, masked_latent_concat], dim=1
            )
            noise_pred = self.unet(
                inpainting_input,
                t.to(self.device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if progress_callback is not None:
                progress_callback(i + 1, num_inference_steps)

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        vae_device = next(self.vae.parameters()).device
        decoded = self.vae.decode(latents.to(vae_device, dtype=self.weight_dtype)).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        image_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        return numpy_to_pil(image_np)[0]


class CatVTONPix2PixPipeline(CatVTONPipeline):
    """Mask-Free CatVTON: InstructPix2Pix UNet, no clothing mask.

    Person and garment latents are concatenated on the width axis. The UNet
    takes 8 channels (noisy latents + image condition), not the 9-channel
    inpainting layout. Loading Mask-Free attention onto an inpaint UNet
    strips clothing instead of putting the garment on.
    """

    def __init__(
        self,
        attn_ckpt: str = ATTN_REPO,
        attn_ckpt_version: str = ATTN_VERSION,
        weight_dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        use_tf32: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype
        self.noise_scheduler = _sd15_ddim_scheduler()
        self.vae = AutoencoderKL.from_pretrained(
            VAE_REPO, torch_dtype=weight_dtype, cache_dir=str(HF_CACHE)
        )
        self.vae.enable_slicing()
        self.unet = _load_pix2pix_unet(weight_dtype, self.device)
        self._attach_attn(attn_ckpt, attn_ckpt_version)

        if use_tf32 and self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image,
        condition_image: PIL.Image.Image,
        mask: Optional[PIL.Image.Image] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> PIL.Image.Image:
        concat_dim = -1
        if cancel_check is not None:
            cancel_check()
        image = resize_and_crop(image, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))

        image_t = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_t = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)

        image_latent = compute_vae_encodings(image_t, self.vae).to(self.device, dtype=self.weight_dtype)
        condition_latent = compute_vae_encodings(condition_t, self.vae).to(
            self.device, dtype=self.weight_dtype
        )
        del image_t, condition_t

        condition_latent_concat = torch.cat([image_latent, condition_latent], dim=concat_dim)
        latents = randn_tensor(
            condition_latent_concat.shape,
            generator=generator,
            device=self.device,
            dtype=self.weight_dtype,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            condition_latent_concat = torch.cat(
                [
                    torch.cat([image_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    condition_latent_concat,
                ]
            )

        extra_step_kwargs = self._prepare_extra_step_kwargs(self.noise_scheduler, generator, eta)
        for i, t in enumerate(tqdm(timesteps, desc="CatVTON-P2P")):
            if cancel_check is not None:
                cancel_check()
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            p2p_input = torch.cat([latent_model_input, condition_latent_concat], dim=1)
            noise_pred = self.unet(
                p2p_input,
                t.to(self.device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if progress_callback is not None:
                progress_callback(i + 1, num_inference_steps)

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        vae_device = next(self.vae.parameters()).device
        decoded = self.vae.decode(latents.to(vae_device, dtype=self.weight_dtype)).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        image_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        return numpy_to_pil(image_np)[0]
