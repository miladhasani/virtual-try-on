"""Gradio theme tokens and stylesheet loading."""

from __future__ import annotations

import json

import gradio as gr

from src.config import QUALITY_PRESETS, THEME_CSS, preset_tooltip

INK = "#f7f2e9"
MUTE = "#9b9385"
GOLD = "#d8b478"
GOLD_SOFT = "#f2dcae"
BASE = "#08080a"
CARD = "#131317"
LINE = "#2b2822"


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.zinc,
    ).set(
        body_background_fill=BASE,
        body_text_color=INK,
        body_text_color_subdued=MUTE,
        block_background_fill="transparent",
        block_border_color=LINE,
        block_label_background_fill="transparent",
        block_label_text_color=GOLD,
        block_title_text_color=GOLD,
        input_background_fill=CARD,
        border_color_primary=LINE,
        button_primary_background_fill=GOLD,
        button_primary_text_color="#1a1409",
        button_secondary_background_fill="transparent",
        button_secondary_text_color=MUTE,
    )


def _preset_tooltip_css() -> str:
    rules = []
    for name in QUALITY_PRESETS:
        tip = preset_tooltip(name).replace("\\", "\\\\").replace('"', '\\"')
        rules.append(
            f'#quality-preset label:has(input[value="{name}"])::after {{ content: "{tip}"; }}'
        )
    rules.append("#quality-preset label[data-tip]::after { content: attr(data-tip); }")
    return "\n".join(rules)


def preset_tooltip_js() -> str:
    tips = {name: preset_tooltip(name) for name in QUALITY_PRESETS}
    payload = json.dumps(tips, ensure_ascii=False)
    return f"""
(() => {{
  const tips = {payload};
  const apply = () => {{
    const root = document.getElementById("quality-preset");
    if (!root) return;
    root.querySelectorAll("label").forEach((label) => {{
      const text = (label.textContent || "").replace(/\\s+/g, " ").trim();
      const input = label.querySelector("input");
      const key = tips[text] ? text : (input && input.value);
      const tip = key && tips[key];
      if (tip) label.setAttribute("data-tip", tip);
    }});
  }};
  apply();
  new MutationObserver(apply).observe(document.documentElement, {{ childList: true, subtree: true }});
}})();
"""


def load_css() -> str:
    base = THEME_CSS.read_text(encoding="utf-8") if THEME_CSS.exists() else ""
    return f"{base}\n{_preset_tooltip_css()}\n"
