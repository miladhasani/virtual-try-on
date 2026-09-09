"""Entry point for the local CatVTON virtual try-on studio."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs
from src.ui.layout import build_ui
from src.ui.theme import build_theme, load_css


def free_port(preferred: int = 7860, attempts: int = 8) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def main() -> None:
    ensure_dirs()
    build_ui().queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=free_port(),
        show_error=True,
        inbrowser=False,
        theme=build_theme(),
        css=load_css(),
        allowed_paths=[str(ROOT / "assets"), str(ROOT / "outputs")],
    )


if __name__ == "__main__":
    main()
