"""Event handlers bridging Gradio widgets to the try-on service."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from src.config import QUALITY_PRESETS, get_model, is_gemini, is_hf_space, preset_tooltip
from src.monitor.resources import JobState, sample_resources
from src.pipeline.cancel import GenerationCancelled
from src.pipeline.tryon import TryOnService
from src.ui import fragments

GENERATE_IDLE = "Generate try-on"
GENERATE_BUSY = "Generating…"


def _as_image(value) -> Image.Image:
    return value if isinstance(value, Image.Image) else Image.open(value)


class StudioHandlers:
    """Stateful callbacks shared by every control in the layout."""

    def __init__(self, service: TryOnService, job: JobState) -> None:
        self.service = service
        self.job = job
        self._last_busy: bool | None = None
        self._armed = False

    def status(self) -> str:
        return fragments.status_bar(self.job)

    def alert(self) -> str:
        return fragments.error_banner(self.job)

    def monitor(self) -> str:
        return fragments.monitor(sample_resources(), self.job)

    def action_updates(self):
        return self._button_state(force=True)

    def _button_state(self, force: bool = True):
        busy = self.service.is_busy()
        if not force and busy == self._last_busy:
            return gr.update(), gr.update(), gr.update()
        self._last_busy = busy
        return (
            gr.update(interactive=not busy, value=GENERATE_BUSY if busy else GENERATE_IDLE),
            gr.update(interactive=busy),
            gr.update(interactive=not busy),
        )

    def tick(self):
        return self.status(), self.alert(), self.monitor(), *self._button_state(force=False)

    def describe_model(self, model_id: str) -> str:
        return fragments.model_card(model_id)

    def sync_model(self, model_id: str):
        spec = get_model(model_id)
        show_gemini = is_gemini(spec)
        show_hf = is_hf_space(spec) or spec.get("access") == "gated"
        return (
            fragments.model_card(model_id),
            float(spec.get("default_guidance", 2.5)),
            gr.update(visible=show_gemini or show_hf),
            gr.update(visible=show_gemini),
            gr.update(visible=show_hf),
        )

    def sync_preset(self, preset: str):
        spec = QUALITY_PRESETS[preset]
        return (
            spec["steps"],
            spec["width"],
            spec["height"],
            fragments.section("03", "Render", preset_tooltip(preset)),
        )

    def _failed_generate(self):
        return (
            gr.update(),
            gr.update(),
            [item.result for item in self.service.history],
            fragments.run_meta(),
            self.status(),
            self.monitor(),
            self.alert(),
        )

    def arm_generate(self, person, garment):
        self._armed = False
        if person is None or garment is None:
            self.service.fail("Upload both a person photo and a garment photo.")
            return *self.action_updates(), self.status(), self.alert()
        if self.service.is_busy():
            return *self.action_updates(), self.status(), self.alert()
        self._armed = True
        self.service.begin_job("Starting")
        return *self.action_updates(), self.status(), self.alert()

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
        width,
        height,
        vram_profile,
        gemini_api_key,
        hf_token,
    ):
        if not self._armed:
            if person is None or garment is None:
                if not self.job.error:
                    self.service.fail("Upload both a person photo and a garment photo.")
            elif not self.service.is_busy() and not self.job.error:
                self.service.fail("Could not start this generation.")
            return self._failed_generate()
        self._armed = False

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
                width=int(width) if width else None,
                height=int(height) if height else None,
                vram_profile=str(vram_profile or "Auto"),
                gemini_api_key=str(gemini_api_key or ""),
                hf_token=str(hf_token or ""),
            )
        except GenerationCancelled:
            return (
                gr.update(),
                gr.update(),
                [item.result for item in self.service.history],
                fragments.run_meta(),
                self.status(),
                self.monitor(),
                self.alert(),
            )
        except Exception as exc:
            self.service.fail(exc)
            return self._failed_generate()

        return (
            result.result,
            (result.person, result.result),
            [item.result for item in self.service.history],
            fragments.run_meta(result),
            self.status(),
            self.monitor(),
            self.alert(),
        )

    def stop(self):
        self.service.request_stop()
        return self.status(), self.alert(), self.monitor(), *self.action_updates()

    def unload(self):
        self.service.request_stop()
        self.service.unload()
        return self.status(), self.alert(), self.monitor(), *self.action_updates()
