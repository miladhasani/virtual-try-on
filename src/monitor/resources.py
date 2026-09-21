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
    error: Optional[str] = None

    def elapsed_label(self) -> str:
        if self.started_at is None:
            return "—"
        seconds = max(0, int(time.time() - self.started_at))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(100.0, self.step / self.total * 100.0))

    def eta_label(self) -> str:
        if self.started_at is None or self.step <= 0 or self.total <= self.step:
            return ""
        elapsed = max(0.0, time.time() - self.started_at)
        remaining = elapsed / self.step * (self.total - self.step)
        seconds = max(0, int(remaining))
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
        "vram_app_gb": 0.0,
        "vram_app_reserved_gb": 0.0,
        "vram_app_percent": 0.0,
    }

    _fill_torch_vram(stats)

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
        if not stats["gpu_available"]:
            stats["gpu_name"] = "GPU stats unavailable"
    return stats


def _fill_torch_vram(stats: dict[str, Any]) -> None:
    """Board totals plus this process, using PyTorch when CUDA is up."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if not torch.cuda.is_available():
        return
    try:
        props = torch.cuda.get_device_properties(0)
        total = float(props.total_memory)
        allocated = float(torch.cuda.memory_allocated(0))
        reserved = float(torch.cuda.memory_reserved(0))
        stats["gpu_name"] = stats["gpu_name"] if stats.get("gpu_available") else props.name
        stats["vram_app_gb"] = allocated / 1024**3
        stats["vram_app_reserved_gb"] = reserved / 1024**3
        stats["vram_app_percent"] = (reserved / total * 100.0) if total else 0.0
        if not stats.get("gpu_available"):
            # NVML missing: still show board totals from CUDA.
            stats["gpu_available"] = True
            stats["vram_total_gb"] = total / 1024**3
            stats["vram_used_gb"] = reserved / 1024**3
            stats["vram_percent"] = stats["vram_app_percent"]
    except Exception:  # noqa: BLE001
        return


# Warm up CPU percent so the first sample is not 0.
psutil.cpu_percent(interval=None)
