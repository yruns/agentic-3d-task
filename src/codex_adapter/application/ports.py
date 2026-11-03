"""Application ports for durable state and upstream HTTP exchange."""

from __future__ import annotations

from typing import Protocol

import httpx

from codex_adapter.domain.models import RegisteredModel
from codex_adapter.domain.state import (
    Binding,
    BoundRoute,
    Completion,
    Replay,
    RequestLease,
    SavedTurn,
)


class UpstreamTransport(Protocol):
    """Open and consume one bound upstream exchange without retrying it."""

    async def open(
        self,
        route: BoundRoute,
        payload: bytes,
        *,
        request_id: str,
        compact: bool = False,
    ) -> httpx.Response: ...

    async def read(self, response: httpx.Response) -> bytes: ...

    async def close(self, response: httpx.Response) -> None: ...


class StateStore(Protocol):
    """Own durable binding/state lifecycle and serialize one session's writes."""

    def initialize(self) -> None: ...

    def ready(self) -> None: ...

    def close(self) -> None: ...

    def bind(
        self, session_id: str, model: RegisteredModel, *, principal: str = "local"
    ) -> BoundRoute: ...

    def turns(self, binding: Binding) -> tuple[SavedTurn, ...]: ...

    def reserve(
        self,
        binding: Binding,
        request_digest: str,
        *,
        previous_carrier: str | None,
        idempotency_key: str | None = None,
    ) -> RequestLease | Replay: ...

    def commit(self, lease: RequestLease, completion: Completion) -> None: ...

    def reject(self, lease: RequestLease) -> None: ...

    def indeterminate(self, lease: RequestLease) -> None: ...
