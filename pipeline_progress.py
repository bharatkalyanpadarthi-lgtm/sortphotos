"""Throttled progress for synchronous pipeline work, including skipped items."""

from __future__ import annotations

import time


def terminal_progress(message: str) -> None:
    print(message, flush=True)


class StageProgress:
    def __init__(self, label, total, progress=terminal_progress, *, interval=5.0):
        self.label = label
        self.total = total
        self.progress = progress
        self.interval = interval
        self.started = time.monotonic()
        self.last_update = self.started
        self.update(0, force=True)

    def update(self, completed, detail="", *, force=False):
        now = time.monotonic()
        if self.progress is None or not (force or completed == self.total
                                        or now - self.last_update >= self.interval):
            return
        elapsed = int(now - self.started)
        suffix = f"; {detail}" if detail else ""
        self.progress(f"{self.label}: {completed}/{self.total} "
                      f"({elapsed // 60}m {elapsed % 60:02d}s){suffix}")
        self.last_update = now

    def items(self, values, detail=None):
        for number, value in enumerate(values, 1):
            yield value
            self.update(number, detail() if detail else "")
