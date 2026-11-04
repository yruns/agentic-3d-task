"""Normalize only the observed ModelHub Gemini token accounting contract."""

from __future__ import annotations

from codex_adapter.domain.json_codec import JsonObject, JsonValue, encode_json

from .gemini_common import invalid, object_value


def _count(value: JsonValue, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise invalid(field, upstream=True)
    return value


def _optional_count(container: JsonObject, field: str) -> int | None:
    if field not in container or container[field] is None:
        return None
    return _count(container[field], f"usage.{field}")


def normalize_usage(value: JsonValue) -> tuple[JsonObject | None, bytes | None]:
    """Keep unreported cached and reasoning counts unknown, never zero-filled."""
    if value is None:
        return None, None
    usage = object_value(value, "usage", upstream=True)
    prompt = _count(usage.get("prompt_tokens"), "usage.prompt_tokens")
    completion = _count(usage.get("completion_tokens"), "usage.completion_tokens")
    total = _count(usage.get("total_tokens"), "usage.total_tokens")
    reasoning = _optional_count(usage, "reasoning_tokens")
    if total < prompt + completion or (
        reasoning is not None and total != prompt + completion + reasoning
    ):
        raise invalid("usage accounting", upstream=True)
    result: JsonObject = {
        "input_tokens": prompt,
        "output_tokens": total - prompt,
        "total_tokens": total,
    }
    details_value = usage.get("prompt_tokens_details")
    if details_value is not None:
        details = object_value(
            details_value, "usage.prompt_tokens_details", upstream=True
        )
        cached = _optional_count(details, "cached_tokens")
        if cached is not None:
            if cached > prompt:
                raise invalid("usage.cached_tokens", upstream=True)
            result = {**result, "input_tokens_details": {"cached_tokens": cached}}
    if reasoning is not None:
        result = {**result, "output_tokens_details": {"reasoning_tokens": reasoning}}
    return result, encode_json(usage)
