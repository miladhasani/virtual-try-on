"""HTML fragments rendered inside the Gradio layout."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, Optional

from src.config import get_model
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
          Drop in a portrait and a clothing shot. A CatVTON or FLUX checkpoint
          fits the piece to the body on the GPU you choose — your machine, or
          Hugging Face when you need external hardware.
        </p>
        <div class="hero-tags">
          <span class="tag">CatVTON</span>
          <span class="tag">SD 1.5 · FLUX</span>
          <span class="tag">Local or Hugging Face</span>
          <span class="tag gold">7 checkpoints</span>
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
    gated = spec.get("access") == "gated"
    pill = (
        f'<span class="access-pill {"gated" if gated else "open"}">'
        f'{escape(spec["access_label"])}</span>'
    )
    facts = [
        ("Best for", spec["best_for"]),
        ("Trained on", spec["trained_on"]),
        ("VRAM", spec["vram_label"]),
        ("Access", spec["access_label"]),
        ("License", spec["license_label"]),
        ("Mask", spec["mask_note"]),
    ]
    chips = "".join(
        f'<span class="chip"><b>{escape(label)}</b>{escape(value)}</span>'
        for label, value in facts
    )
    return f"""
    <div class="model-card">
      <div class="model-head">
        <span class="model-kicker">{escape(spec["tagline"])}</span>
        {pill}
      </div>
      <p class="model-info">{escape(spec["info"])}</p>
      <p class="model-access">{escape(spec["access_note"])}</p>
      <div class="model-facts">{chips}</div>
      <p class="model-foot">
        <span class="mono">{escape(spec["weight_label"])}</span>
        <span>·</span>
        <span>{escape(spec["foot"])}</span>
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
        ("VRAM", result.vram_profile),
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
      <p><b>Where it runs.</b> Use your own GPU, or Hugging Face when you need external hardware.</p>
      <p><b>No login.</b> Mix, VITON-HD, DressCode, and Mask-Free download in the open. License is CC BY-NC-SA 4.0 (personal / research, not commercial).</p>
      <p><b>HF login + license.</b> FLUX, FLUX Alpha, and FLUX Beta need a Hugging Face account. Accept <span class="mono">FLUX.1-Fill-dev</span> (and sometimes <span class="mono">FLUX.1-dev</span>), then run <span class="mono">huggingface-cli login</span>.</p>
      <p>The CatVTON / community FLUX adapters themselves are public. The Black Forest Labs backbone is gated and under the FLUX.1 [dev] non-commercial license.</p>
      <p><b>Studio</b> and GPU-mode FLUX are heavy on a small local card. On 8 GB open Advanced and use Tiny/Fast with Offload, Sequential, or 4-bit — or run them on Hugging Face.</p>
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
    if stats.get("gpu_available") and stats.get("vram_total_gb"):
        gauge = _gauge(
            stats["vram_percent"],
            f"{stats['vram_percent']:.0f}%",
            f"{stats['vram_used_gb']:.1f} / {stats['vram_total_gb']:.1f} GB",
        )
        vram_stat = _stat(
            "VRAM",
            f"{stats['vram_used_gb']:.1f} / {stats['vram_total_gb']:.1f} GB",
            stats["vram_percent"],
            "gold",
        )
        gpu_stat = _stat("GPU load", f"{stats['gpu_util']:.0f}%", stats["gpu_util"], "rose")
        temp = stats.get("gpu_temp")
        device = escape(str(stats["gpu_name"]))
        badge = f'<span class="temp">{temp}°C</span>' if temp is not None else ""
        app_gb = float(stats.get("vram_app_reserved_gb") or stats.get("vram_app_gb") or 0.0)
        app_stat = ""
        if app_gb > 0:
            app_stat = _stat(
                "This app",
                f"{app_gb:.1f} GB reserved",
                stats.get("vram_app_percent") or 0.0,
                "gold",
            )
    else:
        gauge = _gauge(0, "—", "no GPU data")
        vram_stat = _stat("VRAM", "unavailable", 0, "gold")
        gpu_stat = '<div class="stat-warn">GPU telemetry unavailable — CPU and RAM only</div>'
        app_stat = ""
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
        {vram_stat}
        {app_stat}
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
