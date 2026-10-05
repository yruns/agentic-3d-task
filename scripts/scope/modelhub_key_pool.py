"""Schedule independent requests across credentials without exposing secrets."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Condition
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class CredentialLimits(BaseModel):
    """Per-credential dispatch budget and concurrent HTTP request capacity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    qpm_per_key: int = Field(default=500, ge=1, le=500)
    max_inflight_per_key: int = Field(default=64, ge=1, le=512)


@dataclass(frozen=True)
class CredentialLease:
    slot: int
    api_key: SecretStr
    request_id: str


@dataclass
class _CredentialState:
    api_key: SecretStr
    dispatch_times: deque[float] = field(default_factory=deque)
    active_requests: set[str] = field(default_factory=set)
    cooldown_until: float = 0.0
    consecutive_throttles: int = 0


class CredentialPool:
    """Thread-safe rolling budgets; the lock is released during network IO.

    Retry ownership stays with the caller. A 429 temporarily removes its key
    from scheduling; other keys remain available for independent requests.
    """

    def __init__(
        self,
        keys: tuple[SecretStr, ...],
        limits: CredentialLimits,
        *,
        clock: Callable[[], float],
        condition: Condition,
    ) -> None:
        values = tuple(key.get_secret_value() for key in keys)
        if not values or any(not value for value in values):
            raise ValueError("credential_pool_empty")
        if len(set(values)) != len(values):
            raise ValueError("credential_pool_duplicates")
        self._states = tuple(_CredentialState(key) for key in keys)
        self._limits = limits
        self._clock = clock
        self._condition = condition
        self._next_slot = 0

    def acquire(self) -> CredentialLease:
        """Reserve a rate-budget entry and return an exclusive request lease."""
        with self._condition:
            while True:
                now = self._clock()
                ready: list[int] = []
                delays: list[float] = []
                for slot, state in enumerate(self._states):
                    while state.dispatch_times and state.dispatch_times[0] <= now - 60:
                        state.dispatch_times.popleft()
                    delay = max(0.0, state.cooldown_until - now)
                    if len(state.dispatch_times) >= self._limits.qpm_per_key:
                        delay = max(delay, state.dispatch_times[0] + 60 - now)
                    if delay > 0:
                        delays.append(delay)
                        continue
                    if len(state.active_requests) < self._limits.max_inflight_per_key:
                        ready.append(slot)
                if ready:
                    slot = min(
                        ready,
                        key=lambda index: (
                            len(self._states[index].active_requests),
                            (index - self._next_slot) % len(self._states),
                        ),
                    )
                    state = self._states[slot]
                    request_id = uuid4().hex
                    state.active_requests.add(request_id)
                    state.dispatch_times.append(now)
                    self._next_slot = (slot + 1) % len(self._states)
                    return CredentialLease(slot, state.api_key, request_id)
                self._condition.wait(timeout=min(delays) if delays else None)

    def release(
        self,
        lease: CredentialLease,
        *,
        status: int,
        retry_after_seconds: float = 0.0,
    ) -> None:
        """Release capacity and honor rate-limit cooldowns without fake retries."""
        if not 100 <= status <= 599 or not 0 <= retry_after_seconds <= 600:
            raise ValueError("credential_release_status_or_retry_after_invalid")
        with self._condition:
            if not 0 <= lease.slot < len(self._states):
                raise ValueError("credential_lease_not_active")
            state = self._states[lease.slot]
            if lease.request_id not in state.active_requests:
                raise ValueError("credential_lease_not_active")
            state.active_requests.remove(lease.request_id)
            if status == 429:
                state.consecutive_throttles += 1
                delay = max(
                    retry_after_seconds,
                    min(60.0, 2.0 ** min(state.consecutive_throttles, 6)),
                )
                state.cooldown_until = max(state.cooldown_until, self._clock() + delay)
            elif status == 200:
                state.consecutive_throttles = 0
            self._condition.notify_all()
