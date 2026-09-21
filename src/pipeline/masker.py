"""Windows-friendly cloth-agnostic masker using SegFormer clothes parsing.

Mirrors CatVTON AutoMasker part selection (upper / lower / overall) without
DensePose or Detectron2. Parser: mattmdjaga/segformer_b2_clothes (ATR labels).
"""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from src.config import HF_CACHE, PARSER_REPO

# ATR-style labels used by segformer_b2_clothes
LABELS = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17,
}

MASK_PARTS = {
    "upper": ["Upper-clothes", "Dress", "Belt", "Left-arm", "Right-arm"],
    "lower": ["Pants", "Skirt", "Dress", "Left-leg", "Right-leg"],
    "overall": [
        "Upper-clothes",
        "Dress",
        "Pants",
        "Skirt",
        "Belt",
        "Left-arm",
        "Right-arm",
        "Left-leg",
        "Right-leg",
    ],
}

PROTECT_PARTS = {
    "upper": ["Face", "Hair", "Hat", "Left-leg", "Right-leg", "Pants", "Skirt", "Left-shoe", "Right-shoe"],
    "lower": ["Face", "Hair", "Hat", "Left-arm", "Right-arm", "Upper-clothes", "Left-shoe", "Right-shoe"],
    "overall": ["Face", "Hair", "Hat", "Left-shoe", "Right-shoe", "Bag", "Sunglasses"],
}


def vis_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image_np = np.array(image).astype(np.uint8)
    mask_np = np.array(mask.convert("L")).astype(np.uint8)
    mask_np = (mask_np > 127).astype(np.float32)[..., None]
    return Image.fromarray((image_np * (1 - mask_np)).astype(np.uint8))


PROTECT_PHASE = {
    "upper": "Keeping trousers, face, and hair",
    "lower": "Keeping the top, face, and hair",
    "overall": "Keeping face and hair",
}


def composite_tryon(
    original: Image.Image,
    generated: Image.Image,
    parse: np.ndarray,
    cloth_type: str,
) -> tuple[Image.Image, Image.Image]:
    """Keep original pixels on regions that this garment type must not change.

    Mask-Free Pix2Pix has no inpaint mask, so it can rewrite trousers when you
    only asked for a top. Blend those protected parts back from the person photo.
    """
    if cloth_type not in PROTECT_PARTS:
        cloth_type = "upper"

    original = original.convert("RGB")
    generated = generated.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    orig = np.array(original, dtype=np.float32)
    gen = np.array(generated, dtype=np.float32)
    h, w = orig.shape[:2]
    if parse.shape[:2] != (h, w):
        parse = cv2.resize(parse, (w, h), interpolation=cv2.INTER_NEAREST)

    keep = _part_mask(parse, PROTECT_PARTS[cloth_type]).astype(np.uint8)
    if cloth_type == "upper":
        keep = np.logical_or(keep, _lower_person_band(parse)).astype(np.uint8)

    dilate = max(max(w, h) // 90, 7)
    if dilate % 2 == 0:
        dilate += 1
    kernel = np.ones((dilate, dilate), np.uint8)
    keep = cv2.dilate(keep, kernel, iterations=1)

    blur_k = max(max(w, h) // 28, 15)
    if blur_k % 2 == 0:
        blur_k += 1
    keep_f = cv2.GaussianBlur(keep.astype(np.float32), (blur_k, blur_k), 0)
    keep_f = np.clip(keep_f, 0.0, 1.0)[..., None]

    blended = gen * (1.0 - keep_f) + orig * keep_f
    change = ((1.0 - keep_f[..., 0]) * 255.0).astype(np.uint8)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), Image.fromarray(change)


def _lower_person_band(parse: np.ndarray) -> np.ndarray:
    """Extra trousers lock when the parser is messy on the lower body."""
    body = parse != LABELS["Background"]
    ys, xs = np.where(body)
    if ys.size == 0:
        return np.zeros_like(parse, dtype=np.uint8)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    split = int(y0 + 0.55 * (y1 - y0))
    band = np.zeros_like(parse, dtype=np.uint8)
    band[split : y1 + 1, x0 : x1 + 1] = 1
    top_ok = _part_mask(
        parse, ["Upper-clothes", "Dress", "Left-arm", "Right-arm", "Face", "Hair", "Scarf"]
    )
    return np.logical_and(band, np.logical_not(top_ok)).astype(np.uint8)


def _part_mask(parse: np.ndarray, parts: list[str]) -> np.ndarray:
    mask = np.zeros_like(parse, dtype=np.uint8)
    for part in parts:
        if part not in LABELS:
            continue
        mask = np.logical_or(mask, parse == LABELS[part]).astype(np.uint8)
    return mask


def _hull_mask(mask_area: np.ndarray) -> np.ndarray:
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(mask_area)
    for contour in contours:
        if len(contour) < 3:
            continue
        filled = cv2.fillPoly(np.zeros_like(mask_area), [cv2.convexHull(contour)], 255)
        hull = np.logical_or(hull, filled).astype(np.uint8) * 255
    return hull


def build_agnostic_mask(parse: np.ndarray, cloth_type: str) -> Image.Image:
    if cloth_type not in MASK_PARTS:
        raise ValueError(f"cloth_type must be one of {list(MASK_PARTS)}, got {cloth_type}")

    h, w = parse.shape
    dilate = max(w, h) // 250
    dilate = dilate if dilate % 2 == 1 else dilate + 1
    dilate = max(dilate, 3)
    kernel = np.ones((dilate, dilate), np.uint8)
    blur_k = max(w, h) // 25
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
    blur_k = max(blur_k, 5)

    region = _part_mask(parse, MASK_PARTS[cloth_type])
    protect = _part_mask(parse, PROTECT_PARTS[cloth_type])
    background = parse == LABELS["Background"]

    region = cv2.dilate(region, kernel, iterations=2)
    region = _hull_mask(region * 255) // 255
    region = np.logical_and(region, np.logical_not(protect)).astype(np.uint8)
    region = np.logical_and(region, np.logical_not(background)).astype(np.uint8)

    blurred = cv2.GaussianBlur(region * 255, (blur_k, blur_k), 0)
    blurred[blurred < 25] = 0
    blurred[blurred >= 25] = 1
    blurred = cv2.dilate(blurred, kernel, iterations=1)
    blurred = np.logical_and(blurred, np.logical_not(protect)).astype(np.uint8)
    return Image.fromarray(blurred * 255)


class ClothMasker:
    def __init__(self, device: Union[str, torch.device] = "cpu") -> None:
        self.device = torch.device(device)
        self.processor = SegformerImageProcessor.from_pretrained(PARSER_REPO, cache_dir=str(HF_CACHE))
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            PARSER_REPO, cache_dir=str(HF_CACHE)
        )
        self.model.to(self.device)
        self.model.eval()

    def to(self, device: Union[str, torch.device]) -> "ClothMasker":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def parse(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def __call__(self, image: Image.Image, cloth_type: str = "upper") -> Image.Image:
        parse = self.parse(image)
        return build_agnostic_mask(parse, cloth_type)
