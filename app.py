"""Gradio atelier UI for local CatVTON virtual try-on."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from PIL import Image

from src.config import (
    CLOTH_TYPES,
    DEFAULT_CLOTH_TYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL,
    DEFAULT_PRESET,
    DEFAULT_SEED,
    MODEL_CHOICES,
    QUALITY_PRESETS,
    THEME_CSS,
    ensure_dirs,
    get_model,
)
from src.monitor.resources import JobState, render_monitor_html, sample_resources
from src.pipeline.tryon import TryOnService
from src.ui.examples import ensure_examples

ensure_dirs()
JOB = JobState()
SERVICE = TryOnService(job=JOB)
EXAMPLE_PAIRS = ensure_examples()


def _status_chip() -> str:
    return f'<div id="status-chip">{JOB.phase}</div>'


def _monitor() -> str:
    return render_monitor_html(sample_resources(), JOB)


def _theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.zinc,
    ).set(
        body_background_fill="#0b0b0d",
        body_text_color="#f4efe6",
        block_background_fill="#141417",
        block_border_color="#2a2722",
        block_label_text_color="#d4b483",
        button_primary_background_fill="#d4b483",
        button_primary_text_color="#1a140c",
    )


def _css() -> str:
    return THEME_CSS.read_text(encoding="utf-8") if THEME_CSS.exists() else ""


def _model_info(model_id: str) -> str:
    spec = get_model(model_id)
    return f'<p id="model-info">{spec["info"]}</p>'


def generate(
    person,
    garment,
    cloth_type,
    preset,
    model_id,
    steps,
    guidance,
    seed,
):
    if person is None or garment is None:
        raise gr.Error("Upload both a person photo and a garment photo.")
    person_img = person if isinstance(person, Image.Image) else Image.open(person)
    garment_img = garment if isinstance(garment, Image.Image) else Image.open(garment)

    try:
        result = SERVICE.generate(
            person_img,
            garment_img,
            cloth_type=cloth_type,
            preset=preset,
            model_id=model_id,
            num_inference_steps=int(steps) if steps else None,
            guidance_scale=float(guidance),
            seed=int(seed),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    gallery = [item.result for item in SERVICE.history]
    return (
        result.result,
        (result.person, result.result),
        gallery,
        _status_chip(),
        _monitor(),
    )


def unload_models():
    SERVICE.unload()
    return _status_chip(), _monitor()


def sync_steps(preset: str):
    return QUALITY_PRESETS[preset]["steps"]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Atelier — Virtual Try-On") as demo:
        gr.HTML(
            """
            <div id="atelier-header">
              <p id="atelier-kicker">Virtual atelier</p>
              <h1 id="atelier-title">See the garment<br>on you.</h1>
              <p id="atelier-lede">
                Upload a full-body or half-body photo and a clothing image.
                Pick a CatVTON checkpoint, then generate on your local GPU.
              </p>
            </div>
            """
        )
        status = gr.HTML(_status_chip())

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    person = gr.Image(label="Person", type="pil", height=420)
                    garment = gr.Image(label="Garment", type="pil", height=420)
                cloth_type = gr.Radio(
                    CLOTH_TYPES, value=DEFAULT_CLOTH_TYPE, label="Cloth type"
                )
                model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Model",
                    info="All four share the same UNet. Switching only swaps the ~198 MB attention adapter.",
                )
                model_blurb = gr.HTML(_model_info(DEFAULT_MODEL))
                preset = gr.Radio(
                    list(QUALITY_PRESETS.keys()),
                    value=DEFAULT_PRESET,
                    label="Quality preset",
                    info="Fast is safest on 8 GB laptops. Quality may run out of VRAM.",
                )
                with gr.Accordion("Advanced", open=False):
                    steps = gr.Slider(10, 80, value=QUALITY_PRESETS[DEFAULT_PRESET]["steps"], step=1, label="Steps")
                    guidance = gr.Slider(1.0, 7.5, value=DEFAULT_GUIDANCE, step=0.5, label="CFG")
                    seed = gr.Slider(-1, 10_000, value=DEFAULT_SEED, step=1, label="Seed (−1 = random)")
                with gr.Row():
                    generate_btn = gr.Button("Generate try-on", variant="primary", elem_id="generate-btn")
                    unload_btn = gr.Button("Unload models", elem_id="unload-btn")

            with gr.Column(scale=5):
                result = gr.Image(label="Result", type="pil", height=520)
                comparison = gr.ImageSlider(label="Before / after", type="pil")
                gallery = gr.Gallery(label="Recent results", columns=4, height=180, object_fit="cover")

            with gr.Column(scale=3):
                monitor = gr.HTML(_monitor())
                gr.Markdown(
                    """
<div id="license-note">

All listed CatVTON checkpoints are **CC BY-NC-SA 4.0** (non-commercial).
The first time you pick a model it may download ~198 MB of attention weights.
Close other GPU apps before **Quality**.

</div>
                    """
                )

        if EXAMPLE_PAIRS:
            gr.Examples(
                examples=EXAMPLE_PAIRS,
                inputs=[person, garment, cloth_type],
                label="Examples",
            )

        preset.change(sync_steps, inputs=preset, outputs=steps)
        model.change(_model_info, inputs=model, outputs=model_blurb)
        generate_btn.click(
            generate,
            inputs=[person, garment, cloth_type, preset, model, steps, guidance, seed],
            outputs=[result, comparison, gallery, status, monitor],
        )
        unload_btn.click(unload_models, outputs=[status, monitor])
        timer = gr.Timer(1.0)
        timer.tick(_monitor, outputs=monitor, queue=False)
        timer.tick(_status_chip, outputs=status, queue=False)

    return demo


def _free_port(preferred: int = 7860) -> int:
    import socket

    for port in range(preferred, preferred + 8):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=_free_port(),
        show_error=True,
        inbrowser=False,
        theme=_theme(),
        css=_css(),
        allowed_paths=[str(ROOT / "assets"), str(ROOT / "outputs")],
    )
