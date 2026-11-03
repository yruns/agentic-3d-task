"""Strict JSON boundaries and deterministic internal snapshots."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from decimal import Decimal

from pydantic import JsonValue, TypeAdapter, ValidationError

from .errors import AdapterError

JsonObject = dict[str, JsonValue]
_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def _unique_pairs(pairs: Sequence[tuple[str, JsonValue]]) -> JsonObject:
    keys = tuple(key for key, _ in pairs)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate JSON field")
    return dict(pairs)


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _parse_float(value: str) -> float:
    """Accept floats only when JSON encoding preserves their decimal value."""
    number = float(value)
    if not math.isfinite(number) or Decimal(value) != Decimal(str(number)):
        raise ValueError("JSON number cannot be represented without loss")
    return number


def _validate_unicode(value: JsonValue) -> None:
    """Reject lone surrogate code points, including in nested object keys."""
    if isinstance(value, str):
        value.encode("utf-8")
    elif isinstance(value, dict):
        for key, item in value.items():
            key.encode("utf-8")
            _validate_unicode(item)
    elif isinstance(value, list):
        for item in value:
            _validate_unicode(item)


def decode_object(payload: bytes) -> JsonObject:
    """Decode an unambiguous object without changing JSON numeric values."""
    try:
        boundary = json.loads(
            payload,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
        value = _JSON_VALUE.validate_python(boundary)
        _validate_unicode(value)
    except (ValueError, UnicodeError, ArithmeticError, ValidationError, RecursionError):
        pass
    else:
        if not isinstance(value, dict):
            raise AdapterError("invalid_json", "Expected a JSON object")
        return value
    # Raise outside the handler: validation exceptions can embed request content.
    raise AdapterError("invalid_json", "Request must contain unambiguous, finite JSON")


def encode_json(payload: JsonValue) -> bytes:
    """Encode internal snapshots; native Responses wire bodies bypass this."""
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        pass
    # Discard the original exception, which may contain values from the payload.
    raise AdapterError("invalid_json", "Value cannot be encoded as finite UTF-8 JSON")
