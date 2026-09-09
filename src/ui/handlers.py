"""Event handlers bridging Gradio widgets to the try-on service."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from src.config import QUALITY_PRESETS
from src.monitor.resources import JobState, sample_resources
from src.pipeline.tryon import TryOnService
from src.ui import fragments


def _as_image(value) -> Image.Image:
    return value if isinstance(value, Image.Image) else Image.open(value)


class StudioHandlers:
    """Stateful callbacks shared by every control in the layout."""

    def __init__(self, service: TryOnService, job: JobState) -> None:
        self.service = service
        self.job = job

    def status(self) -> str:
        return fragments.status_bar(self.job)

    def monitor(self) -> str:
        return fragments.monitor(sample_resources(), self.job)

    def describe_model(self, model_id: str) -> str:
        return fragments.model_card(model_id)

    def sync_steps(self, preset: str) -> int:
        return QUALITY_PRESETS[preset]["steps"]

    def generate(
        self,
        person,
        garment,
        cloth_type: str,
        preset: str,
        model_id: str,
        steps,
        guidance,
        seed,
    ):
        if person is None or garment is None:
            raise gr.Error("Upload both a person photo and a garment photo.")

        try:
            result = self.service.generate(
                _as_image(person),
                _as_image(garment),
                cloth_type=cloth_type,
                preset=preset,
                model_id=model_id,
                num_inference_steps=int(steps) if steps else None,
                guidance_scale=float(guidance),
                seed=int(seed),
            )
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        return (
            result.result,
            (result.person, result.result),
            [item.result for item in self.service.history],
            fragments.run_meta(result),
            self.status(),
            self.monitor(),
        )

    def unload(self):
        self.service.unload()
        return self.status(), self.monitor()
