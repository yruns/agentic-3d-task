"""Bound request admission without per-client maps or unbounded wait queues."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from codex_adapter.config import ServerSettings
from codex_adapter.domain.errors import AdapterError


@dataclass(frozen=True)
class _Window:
    start: float
    requests: int = 0
    active: int = 0


class Admission:
    """Single event-loop admission counter; callers release exactly once."""

    def __init__(
        self, settings: ServerSettings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.window = _Window(clock())

    def acquire(self) -> None:
        now = self.clock()
        window = self.window
        if now - window.start >= 60.0:
            window = replace(window, start=now, requests=0)
        if window.requests >= self.settings.requests_per_minute:
            raise AdapterError(
                "rate_limit_exceeded",
                "Adapter admission rate exceeded",
                status_code=429,
            )
        if window.active >= self.settings.max_concurrent_requests:
            raise AdapterError(
                "adapter_busy", "Adapter concurrency limit reached", status_code=503
            )
        self.window = replace(
            window, requests=window.requests + 1, active=window.active + 1
        )

    def release(self) -> None:
        if self.window.active <= 0:
            raise RuntimeError("Admission released without an active request")
        self.window = replace(self.window, active=self.window.active - 1)
