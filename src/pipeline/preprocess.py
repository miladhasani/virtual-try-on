"""Image tensor helpers adapted from CatVTON utils.

Original: https://github.com/Zheng-Chong/CatVTON
License: CC BY-NC-SA 4.0
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import PIL.Image
import torch
from PIL import Image


def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
    return model_input


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.to(dtype=torch.float32)

    if isinstance(image, (PIL.Image.Image, np.ndarray)):
        image = [image]
    if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
    elif isinstance(image, list) and isinstance(image[0], np.ndarray):
        image = np.concatenate([i[None, :] for i in image], axis=0)
    image = image.transpose(0, 3, 1, 2)
    return torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0


def prepare_mask_image(mask_image):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        return mask_image

    if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
        mask_image = [mask_image]
    if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
        mask_image = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0)
        mask_image = mask_image.astype(np.float32) / 255.0
    elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
        mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

    mask_image[mask_image < 0.5] = 0
    mask_image[mask_image >= 0.5] = 1
    return torch.from_numpy(mask_image)


def numpy_to_pil(images) -> List[Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        return [Image.fromarray(image.squeeze(), mode="L") for image in images]
    return [Image.fromarray(image) for image in images]


def resize_and_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    image = image.crop(((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2))
    return image.resize(size, Image.LANCZOS)


def resize_and_padding(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding


def to_rgb(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    return image.convert("RGB")


def blur_mask(mask: Image.Image, blur_factor: int = 9) -> Image.Image:
    import cv2

    arr = np.array(mask.convert("L"))
    k = blur_factor if blur_factor % 2 == 1 else blur_factor + 1
    arr = cv2.GaussianBlur(arr, (k, k), 0)
    arr[arr < 25] = 0
    arr[arr >= 25] = 255
    return Image.fromarray(arr)
