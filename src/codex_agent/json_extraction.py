"""Extract a single JSON object from (untrusted) Codex response text.

Codex SDK responses are treated as external input: even with an output schema
the model may wrap JSON in markdown fences or prepend prose. This module is the
boundary that turns that text into a typed ``dict`` (or raises).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import CodexResponseError

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object found in ``text``.

    Handles three shapes, in order:
    1. A bare JSON object.
    2. A JSON object inside a ```` ```json ```` / ```` ``` ```` fence.
    3. A JSON object embedded after leading prose (decoded greedily from the
       first ``{``).

    Raises:
        CodexResponseError: If no JSON object can be parsed, or the parsed value
            is not a JSON object.
    """
    stripped = text.strip()
    if not stripped:
        raise CodexResponseError("Codex response is empty; expected a JSON object")

    fenced = _FENCED_JSON_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1)

    parsed = _loads_lenient(stripped, original=text)
    if not isinstance(parsed, dict):
        raise CodexResponseError(
            f"Codex response must be a JSON object, got {type(parsed).__name__}: "
            f"{_preview(text)}"
        )
    return parsed


def _loads_lenient(candidate: str, *, original: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        start = candidate.find("{")
        if start < 0:
            raise CodexResponseError(
                f"Codex response contains no JSON object: {_preview(original)}"
            ) from exc
        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as inner:
            raise CodexResponseError(
                f"Codex response is not valid JSON: {_preview(original)}"
            ) from inner
        return parsed


def _preview(text: str, *, max_chars: int = 300) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


__all__ = ["extract_json_object"]
