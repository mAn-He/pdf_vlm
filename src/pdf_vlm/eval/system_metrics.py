"""System resource metrics (RSS / VRAM)."""

from __future__ import annotations

import os
from typing import Any

import psutil


def get_rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def get_vram_mb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / (1024 * 1024)
        pynvml.nvmlShutdown()
        return used
    except Exception:
        return None


class SystemMonitor:
    def __init__(self, track_rss: bool = True, track_vram: bool = True):
        self.track_rss = track_rss
        self.track_vram = track_vram
        self.peak_rss_mb = 0.0
        self.peak_vram_mb: float | None = None

    def start(self) -> None:
        self.peak_rss_mb = get_rss_mb() if self.track_rss else 0.0
        self.peak_vram_mb = get_vram_mb() if self.track_vram else None
        self.sample()

    def sample(self) -> None:
        if self.track_rss:
            self.peak_rss_mb = max(self.peak_rss_mb, get_rss_mb())
        if self.track_vram:
            v = get_vram_mb()
            if v is not None:
                self.peak_vram_mb = max(self.peak_vram_mb or 0.0, v)

    def stop(self) -> dict[str, Any]:
        self.sample()
        return {
            "peak_rss_mb": round(self.peak_rss_mb, 2) if self.track_rss else None,
            "peak_vram_mb": round(self.peak_vram_mb, 2) if self.peak_vram_mb is not None else None,
        }
