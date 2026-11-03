"""Strict, non-disclosing boundary helpers for the Gemini Chat codec."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import (
    JsonObject,
    JsonValue,
    decode_object,
    encode_json,
)


def invalid(field: str, *, upstream: bool = False) -> AdapterError:
    """Describe an invalid field without echoing its potentially sensitive value."""
    return AdapterError(
        "upstream_invalid_response" if upstream else "invalid_request",
        f"Invalid {field}",
        status_code=502 if upstream else 400,
        param=None if upstream else field,
    )


def unsupported(field: str) -> AdapterError:
    return AdapterError("unsupported_parameter", f"Unsupported {field}", param=field)


def conflict() -> AdapterError:
    return AdapterError(
        "state_conflict", "Conversation state does not match", status_code=409
    )


def snapshot(value: JsonObject, *, upstream: bool = False) -> JsonObject:
    """Copy and validate JSON without exposing malformed values in errors."""
    try:
        return decode_object(encode_json(value))
    except (AdapterError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise invalid(
            "upstream response" if upstream else "request", upstream=upstream
        ) from exc


def saved_object(value: bytes) -> JsonObject:
    try:
        return decode_object(value)
    except (AdapterError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise conflict() from exc


def object_value(value: JsonValue, field: str, *, upstream: bool = False) -> JsonObject:
    if not isinstance(value, dict):
        raise invalid(field, upstream=upstream)
    return value


def array_value(
    value: JsonValue, field: str, *, upstream: bool = False
) -> list[JsonValue]:
    if not isinstance(value, list):
        raise invalid(field, upstream=upstream)
    return value


def string_value(
    value: JsonValue, field: str, *, upstream: bool = False, empty: bool = False
) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise invalid(field, upstream=upstream)
    return value


def boolean_value(value: JsonValue, field: str) -> bool:
    if not isinstance(value, bool):
        raise invalid(field)
    return value


def number_value(
    value: JsonValue, field: str, *, low: float, high: float
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid(field)
    if not math.isfinite(value) or not low <= value <= high:
        raise invalid(field)
    return value


def reject_extra(value: JsonObject, allowed: Collection[str], field: str) -> None:
    if set(value) - set(allowed):
        raise unsupported(field)


def json_digest(value: JsonValue) -> str:
    return hashlib.sha256(encode_json(value)).hexdigest()


def messages_digest(messages: tuple[bytes, ...]) -> str:
    """Hash the canonical JSON array, including all opaque native fields."""
    return hashlib.sha256(b"[" + b",".join(messages) + b"]").hexdigest()
