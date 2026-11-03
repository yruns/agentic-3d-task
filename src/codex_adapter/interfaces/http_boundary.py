"""Explicit authentication, bounded bodies, and safe request metadata."""

from __future__ import annotations

import hmac

from fastapi import Request

from codex_adapter.application.requests import (
    validate_idempotency_key,
    validate_session_id,
)
from codex_adapter.config import AdapterSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import decode_object

_FORBIDDEN_HEADERS = frozenset(
    {"x-codex-upstream-ak", "x-codex-upstream-url", "x-upstream-url", "x-upstream-ak"}
)


def authenticate(request: Request, settings: AdapterSettings) -> None:
    """Authenticate the adapter only; caller headers never authenticate upstream."""
    expected = settings.auth_token
    if expected is not None:
        candidate = request.headers.get("authorization", "")
        if not hmac.compare_digest(
            candidate.encode("utf-8"), f"Bearer {expected}".encode()
        ):
            raise AdapterError(
                "unauthorized", "Adapter authentication required", status_code=401
            )
    if any(name in request.headers for name in _FORBIDDEN_HEADERS):
        raise AdapterError(
            "forbidden_route_override",
            "Upstream credentials and endpoints are server-owned",
        )


def request_metadata(request: Request) -> tuple[str, str | None]:
    session_id = validate_session_id(request.headers.get("x-codex-session-id"))
    extra = request.headers.get("extra")
    if extra is not None:
        document = decode_object(extra.encode("utf-8"))
        if document.get("session_id", session_id) != session_id:
            raise AdapterError(
                "session_mismatch", "Affinity metadata must match X-Codex-Session-ID"
            )
    return session_id, validate_idempotency_key(request.headers.get("idempotency-key"))


async def read_body(request: Request, limit: int) -> bytes:
    """Bound accumulated body bytes, independent of untrusted Content-Length."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise AdapterError(
                "request_too_large",
                "Request exceeds configured byte limit",
                status_code=413,
            )
        chunks.append(chunk)
    return b"".join(chunks)
