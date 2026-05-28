"""Tiny inference-runtime helpers: device selection and a benchmark timer."""

import time
from contextlib import contextmanager

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Benchmarker:
    """Tiny inference-time timer with an opt-in context manager.

    The diffusion pipeline accepts an instance to record per-stage wall
    clock timings; passing ``Benchmarker(enabled=False)`` (the default)
    turns every ``benchmark()`` block into a no-op.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: dict[str, list[float]] = {}

    @contextmanager
    def benchmark(self, name: str):
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timings.setdefault(name, []).append(elapsed)
