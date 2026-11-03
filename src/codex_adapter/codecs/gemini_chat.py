"""Pure Responses-to-ModelHub Gemini Chat codec public API."""

from __future__ import annotations

from codex_adapter.domain.json_codec import JsonObject, encode_json
from codex_adapter.domain.models import RegisteredModel
from codex_adapter.domain.state import PreparedGemini, SavedTurn

from .gemini_common import json_digest, saved_object, snapshot
from .gemini_options import translate_options
from .gemini_replay import last_carrier as _last_carrier
from .gemini_replay import reconstruct_messages
from .gemini_response import convert_response
from .gemini_sse import response_events

__all__ = ["last_carrier", "prepare_request", "convert_response", "response_events"]


def last_carrier(body: JsonObject) -> str | None:
    """Locate the most recent carrier; authentication belongs to the state store."""
    return _last_carrier(snapshot(body))


def prepare_request(
    body: JsonObject, model: RegisteredModel, turns: tuple[SavedTurn, ...]
) -> PreparedGemini:
    """Translate a full, verified conversation into one buffered Chat request."""
    checked = snapshot(body)
    options, names = translate_options(checked, model)
    # Tool definitions determine name reconstruction; generation controls and output
    # schema apply to the new turn, not to the committed native message prefix.
    options_digest = json_digest({"model": model.id, "tools": options.get("tools", [])})
    messages = reconstruct_messages(checked, turns, options_digest, model.capabilities)
    payload: JsonObject = {
        **options,
        "messages": [saved_object(message) for message in messages],
    }
    return PreparedGemini(
        payload=encode_json(payload),
        messages=messages,
        options_digest=options_digest,
        tool_names=names,
    )
