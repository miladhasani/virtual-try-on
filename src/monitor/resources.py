"""Live CPU / RAM / GPU sampling for the Gradio dashboard."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import psutil

try:
    import pynvml

    _NVML_OK = True
except Exception:  # noqa: BLE001
    pynvml = None  # type: ignore
    _NVML_OK = False

_NVML_READY = False


def _init_nvml() -> bool:
    global _NVML_READY
    if not _NVML_OK:
        return False
    if _NVML_READY:
        return True
    try:
        pynvml.nvmlInit()
        _NVML_READY = True
        return True
    except Exception:  # noqa: BLE001
        _NVML_READY = False
        return False


@dataclass
class JobState:
    phase: str = "Idle"
    step: int = 0
    total: int = 0
    started_at: Optional[float] = None
    message: str = "Ready"

    def elapsed_label(self) -> str:
        if self.started_at is None:
            return "—"
        seconds = max(0, int(time.time() - self.started_at))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"


def sample_resources(gpu_index: int = 0) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    stats: dict[str, Any] = {
        "cpu_percent": float(psutil.cpu_percent(interval=None)),
        "ram_used_gb": vm.used / 1024**3,
        "ram_total_gb": vm.total / 1024**3,
        "ram_percent": float(vm.percent),
        "gpu_available": False,
        "gpu_name": "GPU stats unavailable",
        "gpu_util": 0.0,
        "gpu_temp": None,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "vram_percent": 0.0,
    }

    if not _init_nvml():
        return stats

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except Exception:  # noqa: BLE001
            util = 0
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:  # noqa: BLE001
            temp = None
        stats.update(
            {
                "gpu_available": True,
                "gpu_name": name,
                "gpu_util": float(util),
                "gpu_temp": temp,
                "vram_used_gb": mem.used / 1024**3,
                "vram_total_gb": mem.total / 1024**3,
                "vram_percent": (mem.used / mem.total * 100.0) if mem.total else 0.0,
            }
        )
    except Exception:  # noqa: BLE001
        stats["gpu_name"] = "GPU stats unavailable"
    return stats


def _bar(percent: float, kind: str = "gold") -> str:
    pct = max(0.0, min(100.0, percent))
    return (
        f'<div class="mon-bar"><span class="mon-bar-fill {kind}" '
        f'style="width:{pct:.1f}%"></span></div>'
    )


def render_monitor_html(stats: dict[str, Any], job: JobState) -> str:
    gpu_meta = stats["gpu_name"]
    if stats.get("gpu_temp") is not None:
        gpu_meta = f"{gpu_meta}  ·  {stats['gpu_temp']}°C"
    if not stats.get("gpu_available"):
        gpu_note = '<div class="mon-warn">GPU stats unavailable — CPU / RAM only</div>'
        vram_label = "—"
        util_label = "—"
        vram_bar = _bar(0)
        util_bar = _bar(0)
    else:
        gpu_note = ""
        vram_label = f"{stats['vram_used_gb']:.1f} / {stats['vram_total_gb']:.1f} GB"
        util_label = f"{stats['gpu_util']:.0f}%"
        vram_bar = _bar(stats["vram_percent"], "gold")
        util_bar = _bar(stats["gpu_util"], "rose")

    busy = job.started_at is not None
    pulse = "on" if busy else ""
    phase = job.phase
    return f"""
    <div class="monitor">
      <div class="mon-head">
        <span class="mon-pulse {pulse}"></span>
        <span class="mon-kicker">Live resources</span>
        <span class="mon-elapsed">{job.elapsed_label()}</span>
      </div>
      <div class="mon-phase">{phase}</div>
      <div class="mon-gpu">{gpu_meta}</div>
      {gpu_note}
      <div class="mon-row">
        <span>VRAM</span><span>{vram_label}</span>
      </div>
      {vram_bar}
      <div class="mon-row">
        <span>GPU</span><span>{util_label}</span>
      </div>
      {util_bar}
      <div class="mon-row">
        <span>CPU</span><span>{stats['cpu_percent']:.0f}%</span>
      </div>
      {_bar(stats['cpu_percent'], "steel")}
      <div class="mon-row">
        <span>RAM</span>
        <span>{stats['ram_used_gb']:.1f} / {stats['ram_total_gb']:.1f} GB</span>
      </div>
      {_bar(stats['ram_percent'], "steel")}
    </div>
    """


# Warm up CPU percent so the first sample is not 0.
psutil.cpu_percent(interval=None)
