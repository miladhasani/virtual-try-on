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


# Warm up CPU percent so the first sample is not 0.
psutil.cpu_percent(interval=None)
