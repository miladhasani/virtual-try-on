"""Cloud try-on backends: Google Gemini and Hugging Face Spaces."""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin

import httpx
from PIL import Image

from src.config import load_secrets
from src.pipeline.cancel import GenerationCancelled
from src.pipeline.preprocess import to_rgb

CancelFn = Optional[Callable[[], None]]
NoteFn = Optional[Callable[[str], None]]

CLOTH_WORDS = {
    "upper": "upper-body garment (shirt, top, jacket, or sweater)",
    "lower": "lower-body garment (pants, skirt, or shorts)",
    "overall": "full outfit or dress",
}

HF_SPACE_CANDIDATES = (
    os.environ.get("CATVTON_HF_SPACE"),
    # Prefer the Space that asks ZeroGPU for less time so anonymous quota can fit.
    "tryitonvirtual/virtual-tryon",
    "Nandha2017/virtual-tryon",
    "zhengchong/CatVTON",
)

GEMINI_MODELS = (
    os.environ.get("CATVTON_GEMINI_MODEL"),
    "gemini-3.1-flash-image",
    "gemini-2.5-flash-image",
)


def _note(note: NoteFn, text: str) -> None:
    if note:
        note(text)


def _check(cancel: CancelFn) -> None:
    if cancel:
        cancel()


def _aspect_ratio(width: int, height: int) -> str:
    ratio = max(width, 1) / max(height, 1)
    choices = {
        "1:1": 1.0,
        "4:5": 0.8,
        "3:4": 0.75,
        "2:3": 2 / 3,
        "9:16": 9 / 16,
        "5:4": 1.25,
        "4:3": 4 / 3,
        "3:2": 1.5,
        "16:9": 16 / 9,
    }
    return min(choices, key=lambda key: abs(choices[key] - ratio))


def _gemini_size(width: int, height: int) -> str:
    long_edge = max(width, height)
    if long_edge >= 1400:
        return "2K"
    if long_edge <= 400:
        return "512"
    return "1K"


def _as_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return to_rgb(value)
    if isinstance(value, (str, Path)):
        return to_rgb(Image.open(value))
    if isinstance(value, dict):
        path = value.get("path") or value.get("url") or (value.get("image") or {}).get("path")
        if path:
            if str(path).startswith("http"):
                with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                    response = client.get(path)
                    response.raise_for_status()
                return to_rgb(Image.open(io.BytesIO(response.content)))
            return to_rgb(Image.open(path))
    raise RuntimeError("Cloud backend returned an image we could not read.")


def gemini_tryon(
    person: Image.Image,
    garment: Image.Image,
    *,
    cloth_type: str = "upper",
    width: int = 768,
    height: int = 1024,
    seed: int = -1,
    api_key: str = "",
    models: tuple[str, ...] | list[str] | None = None,
    cancel_check: CancelFn = None,
    progress: NoteFn = None,
) -> Image.Image:
    key = (api_key or load_secrets().get("gemini_api_key") or "").strip()
    if not key:
        raise RuntimeError(
            "Add a Google Gemini API key from https://aistudio.google.com/apikey "
            "and paste it in the Gemini key field."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai to use Gemini: pip install google-genai") from exc

    person = to_rgb(person)
    garment = to_rgb(garment)
    prompt = (
        "Virtual try-on photograph. Image 1 is the person — keep their face, hair, skin, "
        "body shape, pose, camera angle, and background. "
        f"Image 2 is the {CLOTH_WORDS.get(cloth_type, 'garment')} they should wear. "
        "Replace only the relevant clothing so the person is wearing the garment from image 2, "
        "matching its color, pattern, texture, logos, and cut. "
        "Photorealistic single photo. No collage, no split view, no extra people, no text."
    )
    aspect = _aspect_ratio(width, height)
    size = _gemini_size(width, height)
    wanted = [item for item in (models or GEMINI_MODELS) if item]
    seen: set[str] = set()
    unique_models = []
    for name in wanted:
        if name not in seen:
            unique_models.append(name)
            seen.add(name)

    client = genai.Client(api_key=key)
    last_error: Exception | None = None
    for model_name in unique_models:
        _check(cancel_check)
        _note(progress, f"Gemini · {model_name}")
        for with_size in (True, False):
            _check(cancel_check)
            image_kwargs: dict[str, str] = {"aspect_ratio": aspect}
            if with_size:
                image_kwargs["image_size"] = size
            config_kwargs: dict = {
                "response_modalities": ["TEXT", "IMAGE"],
                "image_config": types.ImageConfig(**image_kwargs),
            }
            if seed is not None and int(seed) >= 0:
                config_kwargs["seed"] = int(seed)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, person, garment],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                last_error = exc
                if with_size and ("image_size" in message or "unknown" in message):
                    continue
                if "not found" in message or "not_found" in message or "404" in message:
                    break
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            _check(cancel_check)
            parts = getattr(response, "parts", None) or []
            for part in parts:
                image = part.as_image() if hasattr(part, "as_image") else None
                if image is not None:
                    return to_rgb(image)
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    blob = inline.data
                    if isinstance(blob, str):
                        import base64

                        blob = base64.b64decode(blob)
                    return to_rgb(Image.open(io.BytesIO(blob)))
            last_error = RuntimeError("Gemini returned no image. Try another seed or garment photo.")
            break
    raise RuntimeError(str(last_error) if last_error else "Gemini did not return an image.")


def _space_url(space_id: str, token: str | None) -> str:
    from huggingface_hub import HfApi

    runtime = HfApi(token=token).get_space_runtime(space_id)
    domains = ((runtime.raw or {}).get("domains") or [])
    if domains:
        domain = domains[0].get("domain")
        if domain:
            return f"https://{domain}"
    slug = space_id.replace("/", "-").lower()
    return f"https://{slug}.hf.space"


def _pick_hf_space(token: str | None, progress: NoteFn) -> tuple[str, str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    last_error = "No Hugging Face Space is reachable."
    for space_id in HF_SPACE_CANDIDATES:
        if not space_id:
            continue
        try:
            runtime = api.get_space_runtime(space_id)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{space_id}: {exc}"
            continue
        stage = str(getattr(runtime, "stage", "") or "")
        if stage in {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "STOPPED", "PAUSED"}:
            last_error = f"{space_id} is {stage}"
            _note(progress, f"Skipping {space_id} ({stage})")
            continue
        url = _space_url(space_id, token)
        _note(progress, f"Hugging Face · {space_id}")
        return space_id, url
    raise RuntimeError(
        "No live CatVTON Hugging Face Space is available. "
        f"Last check: {last_error}. Set CATVTON_HF_SPACE to a running Space id."
    )


def _auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _upload_image(
    client: httpx.Client, host: str, image: Image.Image, name: str, session: str
) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    response = client.post(
        urljoin(host.rstrip("/") + "/", "gradio_api/upload"),
        params={"upload_id": session},
        files={"files": (name, buffer, "image/jpeg")},
        timeout=120.0,
    )
    response.raise_for_status()
    payload = response.json()
    path = payload[0] if isinstance(payload, list) else payload
    return {
        "path": path,
        "url": f"{host.rstrip('/')}/gradio_api/file={path}",
        "orig_name": name,
        "meta": {"_type": "gradio.FileData"},
    }


def _space_fn_index(client: httpx.Client, host: str) -> int:
    response = client.get(urljoin(host.rstrip("/") + "/", "config"), timeout=30.0)
    response.raise_for_status()
    for dep in response.json().get("dependencies") or []:
        if isinstance(dep, dict) and dep.get("api_name") == "run_tryon":
            return int(dep.get("id", 1))
    return 1


def _parse_sse_image(host: str, payload) -> Image.Image:
    if isinstance(payload, list) and payload:
        return _parse_sse_image(host, payload[0])
    if isinstance(payload, dict):
        image = payload.get("image") if isinstance(payload.get("image"), dict) else payload
        path = image.get("url") or image.get("path") if isinstance(image, dict) else None
        if path:
            if str(path).startswith("http"):
                return _as_pil({"path": path})
            url = urljoin(host.rstrip("/") + "/", str(path).lstrip("/"))
            return _as_pil({"path": url})
        nested = payload.get("path") or payload.get("url")
        if nested:
            return _as_pil({"path": nested})
    if isinstance(payload, str) and payload:
        path = payload if payload.startswith("http") else urljoin(host.rstrip("/") + "/", payload.lstrip("/"))
        return _as_pil({"path": path})
    raise RuntimeError("Hugging Face Space did not return a result image.")


def hf_space_tryon(
    person: Image.Image,
    garment: Image.Image,
    *,
    cloth_type: str = "upper",
    steps: int = 30,
    guidance: float = 2.5,
    seed: int = 42,
    hf_token: str = "",
    cancel_check: CancelFn = None,
    progress: NoteFn = None,
) -> Image.Image:
    from huggingface_hub import get_token

    token = (hf_token or load_secrets().get("hf_token") or get_token() or "").strip() or None
    space_id, host = _pick_hf_space(token, progress)
    person = to_rgb(person)
    garment = to_rgb(garment)
    steps = max(10, min(50, int(steps)))
    guidance = max(1.0, min(10.0, float(guidance)))
    cloth = cloth_type if cloth_type in CLOTH_WORDS else "upper"
    session = uuid.uuid4().hex

    headers = _auth_headers(token)
    timeout = httpx.Timeout(30.0, read=420.0)
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        _check(cancel_check)
        _note(progress, "Uploading photos to Hugging Face")
        fn_index = _space_fn_index(client, host)
        person_ref = _upload_image(client, host, person, "person.jpg", session)
        garment_ref = _upload_image(client, host, garment, "garment.jpg", session)
        _check(cancel_check)
        _note(progress, f"Queued on {space_id}")
        join = client.post(
            urljoin(host.rstrip("/") + "/", "gradio_api/queue/join"),
            json={
                "data": [person_ref, garment_ref, cloth, steps, guidance, int(seed)],
                "fn_index": fn_index,
                "session_hash": session,
            },
        )
        join.raise_for_status()
        if not (join.json() or {}).get("event_id"):
            raise RuntimeError(f"Hugging Face Space {space_id} did not accept the job.")
        started = time.time()
        with client.stream(
            "GET",
            urljoin(host.rstrip("/") + "/", "gradio_api/queue/data"),
            params={"session_hash": session},
        ) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                _check(cancel_check)
                if time.time() - started > 400:
                    raise RuntimeError("Hugging Face Space timed out. Try again, or use Mix locally.")
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg = str(payload.get("msg") or "")
                if msg == "estimation":
                    eta = payload.get("rank_eta")
                    rank = payload.get("rank")
                    extra = f" · place {rank + 1}" if isinstance(rank, int) else ""
                    if eta:
                        extra += f" · ETA {int(eta)}s"
                    _note(progress, f"Hugging Face queue{extra}")
                elif msg == "process_starts":
                    _note(progress, f"Generating on {space_id}")
                elif msg == "process_completed":
                    output = payload.get("output") or {}
                    if not payload.get("success", True):
                        error = output.get("error") or output.get("title") or "Space job failed"
                        if "quota" in str(error).lower() or "zerogpu" in str(error).lower():
                            raise RuntimeError(
                                f"{error} Paste a Hugging Face token (huggingface.co/settings/tokens) "
                                "for more ZeroGPU quota, or run Mix locally."
                            )
                        raise RuntimeError(str(error))
                    data = output.get("data")
                    return _parse_sse_image(host, data)
        raise RuntimeError(f"Hugging Face Space {space_id} finished without an image.")
