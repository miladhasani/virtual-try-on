"""Gradio layout for the virtual try-on studio."""

from __future__ import annotations

from typing import Optional

import gradio as gr

from src.config import (
    CLOTH_TYPES,
    DEFAULT_CLOTH_TYPE,
    DEFAULT_GUIDANCE,
    DEFAULT_MODEL,
    DEFAULT_PRESET,
    DEFAULT_SEED,
    MODEL_CHOICES,
    QUALITY_PRESETS,
)
from src.monitor.resources import JobState
from src.pipeline.tryon import TryOnService
from src.ui import fragments
from src.ui.examples import ensure_examples
from src.ui.handlers import StudioHandlers

CLOTH_LABELS = {"upper": "Top", "lower": "Bottom", "overall": "Full look"}


def _preset_hint() -> str:
    spec = QUALITY_PRESETS[DEFAULT_PRESET]
    return f"{spec['width']}×{spec['height']} · {spec['steps']} steps by default"


def build_ui(
    service: Optional[TryOnService] = None,
    job: Optional[JobState] = None,
) -> gr.Blocks:
    job = job or JobState()
    service = service or TryOnService(job=job)
    handlers = StudioHandlers(service, job)
    example_pairs = ensure_examples()

    with gr.Blocks(title="Atelier — Virtual Try-On", fill_width=True) as demo:
        gr.HTML(fragments.hero())
        status = gr.HTML(handlers.status())

        with gr.Row(equal_height=False, elem_id="stage"):
            with gr.Column(scale=5, min_width=360, elem_classes=["panel", "panel-desk"]):
                gr.HTML(fragments.section("01", "Photos", "portrait + garment"))
                with gr.Row(elem_classes=["drop-row"]):
                    person = gr.Image(
                        label="Person",
                        type="pil",
                        height=286,
                        elem_classes=["drop"],
                        sources=["upload", "clipboard"],
                    )
                    garment = gr.Image(
                        label="Garment",
                        type="pil",
                        height=286,
                        elem_classes=["drop"],
                        sources=["upload", "clipboard"],
                    )
                cloth_type = gr.Radio(
                    choices=[(CLOTH_LABELS[key], key) for key in CLOTH_TYPES],
                    value=DEFAULT_CLOTH_TYPE,
                    label="Garment type",
                    elem_classes=["segmented"],
                )

                gr.HTML(fragments.section("02", "Checkpoint", "one adapter in VRAM"))
                model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Model",
                    filterable=False,
                    elem_id="model-pick",
                )
                model_blurb = gr.HTML(handlers.describe_model(DEFAULT_MODEL))

                gr.HTML(fragments.section("03", "Render", _preset_hint()))
                preset = gr.Radio(
                    choices=list(QUALITY_PRESETS),
                    value=DEFAULT_PRESET,
                    label="Quality preset",
                    info="Fast is safest on 8 GB. Quality can exhaust VRAM.",
                    elem_classes=["segmented"],
                )
                with gr.Accordion("Advanced controls", open=False, elem_id="advanced"):
                    steps = gr.Slider(
                        10,
                        80,
                        value=QUALITY_PRESETS[DEFAULT_PRESET]["steps"],
                        step=1,
                        label="Diffusion steps",
                    )
                    guidance = gr.Slider(
                        1.0, 7.5, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance (CFG)"
                    )
                    seed = gr.Slider(
                        -1, 10_000, value=DEFAULT_SEED, step=1, label="Seed (−1 = random)"
                    )

                with gr.Row(elem_classes=["actions"]):
                    generate_btn = gr.Button(
                        "Generate try-on", variant="primary", elem_id="generate-btn", scale=3
                    )
                    unload_btn = gr.Button("Free VRAM", elem_id="unload-btn", scale=1)

            with gr.Column(scale=7, min_width=420, elem_classes=["panel", "panel-stage"]):
                with gr.Tabs(elem_id="stage-tabs"):
                    with gr.Tab("Result"):
                        result = gr.Image(
                            label="", show_label=False, type="pil", height=560, elem_classes=["canvas"]
                        )
                    with gr.Tab("Before / after"):
                        comparison = gr.ImageSlider(
                            label="", show_label=False, type="pil", height=560, elem_classes=["canvas"]
                        )
                    with gr.Tab("This session"):
                        gallery = gr.Gallery(
                            label="",
                            show_label=False,
                            columns=3,
                            height=560,
                            object_fit="cover",
                            elem_classes=["canvas"],
                        )
                meta = gr.HTML(fragments.run_meta())

            with gr.Column(scale=3, min_width=250, elem_classes=["panel", "panel-side"]):
                monitor = gr.HTML(handlers.monitor())
                gr.HTML(fragments.license_note())

        if example_pairs:
            with gr.Row(elem_id="examples-row"):
                gr.Examples(
                    examples=example_pairs,
                    inputs=[person, garment, cloth_type],
                    label="Start from an example",
                )

        preset.change(handlers.sync_steps, inputs=preset, outputs=steps)
        model.change(handlers.describe_model, inputs=model, outputs=model_blurb)
        generate_btn.click(
            handlers.generate,
            inputs=[person, garment, cloth_type, preset, model, steps, guidance, seed],
            outputs=[result, comparison, gallery, meta, status, monitor],
        )
        unload_btn.click(handlers.unload, outputs=[status, monitor])

        timer = gr.Timer(1.0)
        timer.tick(handlers.monitor, outputs=monitor, queue=False)
        timer.tick(handlers.status, outputs=status, queue=False)

    return demo
