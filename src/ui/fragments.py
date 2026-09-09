"""HTML fragments rendered inside the Gradio layout."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, Optional

from src.config import TRYON_MODELS, get_model
from src.monitor.resources import JobState

if TYPE_CHECKING:
    from src.pipeline.tryon import TryOnResult


def hero() -> str:
    return """
    <header class="hero">
      <div class="hero-aura"></div>
      <div class="hero-body">
        <span class="hero-kicker"><i></i>Virtual atelier</span>
        <h1 class="hero-title">See the garment <em>on you</em>.</h1>
        <p class="hero-lede">
          Drop in a portrait and a clothing shot. A CatVTON checkpoint fits the
          piece to the body, entirely on your own GPU — nothing leaves this machine.
        </p>
        <div class="hero-tags">
          <span class="tag">CatVTON</span>
          <span class="tag">Stable Diffusion 1.5</span>
          <span class="tag">Local inference</span>
          <span class="tag gold">4 checkpoints</span>
        </div>
      </div>
    </header>
    """


def section(number: str, title: str, hint: str = "") -> str:
    hint_html = f'<span class="sec-hint">{escape(hint)}</span>' if hint else ""
    return (
        f'<div class="sec"><span class="sec-num">{escape(number)}</span>'
        f'<span class="sec-title">{escape(title)}</span>{hint_html}</div>'
    )


def status_bar(job: JobState) -> str:
    busy = job.started_at is not None
    percent = (job.step / job.total * 100.0) if job.total else 0.0
    return f"""
    <div class="statusbar {'busy' if busy else 'calm'}">
      <span class="status-dot"></span>
      <span class="status-phase">{escape(job.phase)}</span>
      <span class="status-timer">{job.elapsed_label()}</span>
      <span class="status-track"><i style="width:{percent:.1f}%"></i></span>
    </div>
    """


def model_card(model_id: str) -> str:
    spec = get_model(model_id)
    others = len(TRYON_MODELS) - 1
    return f"""
    <div class="model-card">
      <p class="model-info">{escape(spec['info'])}</p>
      <p class="model-foot">
        <span class="mono">{escape(spec['attn_subfolder'])}</span>
        <span>·</span>
        <span>{others} other checkpoints share this UNet</span>
      </p>
    </div>
    """


def run_meta(result: Optional["TryOnResult"] = None) -> str:
    if result is None:
        return (
            '<div class="meta meta-empty">Nothing rendered yet. '
            "Choose a checkpoint, then press generate.</div>"
        )
    chips = [
        ("Model", result.model_label.split(" — ")[0]),
        ("Preset", result.preset),
        ("Canvas", f"{result.width}×{result.height}"),
        ("Steps", str(result.steps)),
        ("Elapsed", f"{result.elapsed_s:.1f}s"),
    ]
    body = "".join(
        f'<span class="chip"><b>{escape(label)}</b>{escape(value)}</span>'
        for label, value in chips
    )
    return f'<div class="meta">{body}</div>'


def license_note() -> str:
    return """
    <div class="note">
      <p><b>Non-commercial.</b> Every CatVTON checkpoint here is CC BY-NC-SA 4.0.</p>
      <p>Selecting a checkpoint for the first time pulls ~198 MB of attention weights.</p>
      <p>Close other GPU apps before running the <b>Quality</b> preset.</p>
    </div>
    """


def _clamp(percent: float) -> float:
    return max(0.0, min(100.0, float(percent)))


def _meter(percent: float, tone: str) -> str:
    return (
        f'<span class="meter"><i class="{tone}" '
        f'style="width:{_clamp(percent):.1f}%"></i></span>'
    )


def _stat(label: str, value: str, percent: float, tone: str) -> str:
    return f"""
      <div class="stat">
        <div class="stat-head"><span>{escape(label)}</span><span>{escape(value)}</span></div>
        {_meter(percent, tone)}
      </div>
    """


def _gauge(percent: float, primary: str, secondary: str) -> str:
    return f"""
      <div class="gauge" style="--pct:{_clamp(percent):.1f}">
        <div class="gauge-core">
          <span class="gauge-primary">{escape(primary)}</span>
          <span class="gauge-secondary">{escape(secondary)}</span>
        </div>
      </div>
    """


def monitor(stats: dict[str, Any], job: JobState) -> str:
    busy = job.started_at is not None
    if stats.get("gpu_available"):
        gauge = _gauge(
            stats["vram_percent"],
            f"{stats['vram_used_gb']:.1f}",
            f"of {stats['vram_total_gb']:.1f} GB",
        )
        gpu_stat = _stat("GPU load", f"{stats['gpu_util']:.0f}%", stats["gpu_util"], "rose")
        temp = stats.get("gpu_temp")
        device = escape(str(stats["gpu_name"]))
        badge = f'<span class="temp">{temp}°C</span>' if temp is not None else ""
    else:
        gauge = _gauge(0, "—", "no GPU data")
        gpu_stat = '<div class="stat-warn">GPU telemetry unavailable — CPU and RAM only</div>'
        device = escape(str(stats["gpu_name"]))
        badge = ""

    return f"""
    <div class="monitor">
      <div class="mon-head">
        <span class="mon-kicker">Live resources</span>
        <span class="mon-pulse {'on' if busy else ''}"></span>
      </div>
      <div class="mon-device">{device}{badge}</div>
      {gauge}
      <div class="mon-vram-label">VRAM in use</div>
      <div class="stats">
        {gpu_stat}
        {_stat("CPU", f"{stats['cpu_percent']:.0f}%", stats["cpu_percent"], "steel")}
        {_stat(
            "RAM",
            f"{stats['ram_used_gb']:.1f} / {stats['ram_total_gb']:.1f} GB",
            stats["ram_percent"],
            "steel",
        )}
      </div>
    </div>
    """
