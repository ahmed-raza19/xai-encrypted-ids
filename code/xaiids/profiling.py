"""Cost-measurement utilities (Phase 2, E1): single-core pinning and peak-RSS sampling."""
from __future__ import annotations

import os
import threading
import time

import psutil


def pin_single_core(cpu: int | None = None):
    """Pin the current process to one logical CPU and limit native thread pools to one, so
    every timing is a per-core cost that does not depend on idle cores. The core can be
    overridden with the XAI_CORE environment variable (default: logical CPU 2, a P-core)."""
    if cpu is None:
        cpu = int(os.environ.get("XAI_CORE", "2"))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    p = psutil.Process()
    try:
        p.cpu_affinity([cpu])
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    return p


class PeakRSS:
    """Samples the resident set size of this process every ``interval`` seconds."""

    def __init__(self, interval=0.002):
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        rss = self.proc.memory_info().rss
        self.peak = max(self.peak, rss)

    @property
    def increment_mb(self):
        return (self.peak - self.base) / 2**20
