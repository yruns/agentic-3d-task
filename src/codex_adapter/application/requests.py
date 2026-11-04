"""Validate the routing envelope without rewriting native Responses JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from codex_adapter.domain.errors import AdapterError

_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}$")
_IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_JSON_NUMBER = object()


@dataclass(frozen=True)
class RequestEnvelope:
    model: str
    stream: bool
    raw: bytes = field(repr=False)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("Duplicate JSON field")
    return dict(pairs)


def _reject_constant(value: str) -> None:
    raise ValueError("Invalid JSON numeric constant")


def _ignore_number(value: str) -> object:
    # Only routing fields are inspected; native numeric lexemes stay in raw.
    return _JSON_NUMBER


def parse_envelope(raw: bytes) -> RequestEnvelope:
    """Inspect only routing fields, preserving arbitrary JSON number precision."""
    try:
        parsed: object = json.loads(
            raw,
            parse_int=_ignore_number,
            parse_float=_ignore_number,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (ValueError, UnicodeError, RecursionError):
        pass
    else:
        return _request_envelope(parsed, raw)
    raise AdapterError("invalid_json", "Expected unambiguous JSON object") from None


def _request_envelope(parsed: object, raw: bytes) -> RequestEnvelope:
    if not isinstance(parsed, dict):
        raise AdapterError("invalid_request", "Request body must be an object")
    model = parsed.get("model")
    if not isinstance(model, str) or not 0 < len(model) <= 128:
        raise AdapterError(
            "invalid_model", "model must be a nonempty registered ID", param="model"
        )
    stream = parsed.get("stream", False)
    if not isinstance(stream, bool):
        raise AdapterError("invalid_stream", "stream must be a boolean", param="stream")
    return RequestEnvelope(model=model, stream=stream, raw=raw)


def validate_session_id(value: str | None) -> str:
    """Require an explicit, unique caller-owned SDK session identifier."""
    if value is None or not _SESSION_PATTERN.fullmatch(value):
        raise AdapterError(
            "invalid_session",
            "X-Codex-Session-ID must contain 8-128 safe identifier characters",
            param="X-Codex-Session-ID",
        )
    return value


def validate_idempotency_key(value: str | None) -> str | None:
    if value is not None and not _IDEMPOTENCY_PATTERN.fullmatch(value):
        raise AdapterError(
            "invalid_idempotency_key",
            "Idempotency-Key must contain 1-128 safe identifier characters",
            param="Idempotency-Key",
        )
    return value
