"""Download a few CatVTON demo images for the Gradio gallery."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from src.config import EXAMPLES_DIR, ensure_dirs

_BASE = "https://raw.githubusercontent.com/Zheng-Chong/CatVTON/edited/resource/demo/example"

EXAMPLE_FILES = {
    "person/woman.jpg": f"{_BASE}/person/women/049713_0.jpg",
    "person/man.png": f"{_BASE}/person/men/model_5.png",
    "garment/top_a.jpg": f"{_BASE}/condition/upper/21514384_52353349_1000.jpg",
    "garment/top_b.jpg": f"{_BASE}/condition/upper/23255574_53383833_1000.jpg",
}


def ensure_examples() -> list[list[str]]:
    ensure_dirs()
    for rel, url in EXAMPLE_FILES.items():
        dest = EXAMPLES_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            with urlopen(url, timeout=30) as response, dest.open("wb") as handle:
                handle.write(response.read())
        except Exception:
            continue

    pairs: list[list[str]] = []
    woman = EXAMPLES_DIR / "person" / "woman.jpg"
    man = EXAMPLES_DIR / "person" / "man.png"
    top_a = EXAMPLES_DIR / "garment" / "top_a.jpg"
    top_b = EXAMPLES_DIR / "garment" / "top_b.jpg"
    if woman.exists() and top_a.exists():
        pairs.append([str(woman), str(top_a), "upper"])
    if man.exists() and top_b.exists():
        pairs.append([str(man), str(top_b), "upper"])
    return pairs


def list_existing_pairs() -> list[list[str]]:
    pairs: list[list[str]] = []
    for person in sorted((EXAMPLES_DIR / "person").glob("*")):
        for garment in sorted((EXAMPLES_DIR / "garment").glob("*")):
            pairs.append([str(person), str(garment), "upper"])
            break
    return pairs
