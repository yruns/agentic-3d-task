"""Project a complete ModelHub Gemini Chat result into Responses objects."""

from __future__ import annotations

import hashlib
import math

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import (
    JsonObject,
    JsonValue,
    decode_object,
    encode_json,
)
from codex_adapter.domain.models import RegisteredModel
from codex_adapter.domain.state import (
    Completion,
    PreparedGemini,
    RequestLease,
    SavedTurn,
    ToolName,
)

from .gemini_common import (
    array_value,
    invalid,
    messages_digest,
    object_value,
    snapshot,
    string_value,
)
from .gemini_usage import normalize_usage


def _id(lease: RequestLease, kind: str, index: int = 0) -> str:
    source = f"{lease.response_id}\0{lease.sequence}\0{kind}\0{index}".encode()
    return f"{kind}_{hashlib.sha256(source).hexdigest()[:24]}"


def _validate_lease(lease: RequestLease, model: RegisteredModel) -> None:
    binding = lease.binding
    if (
        model.protocol != "gemini_chat"
        or binding.model_id != model.id
        or binding.protocol != model.protocol
        or binding.codec_version != model.codec_version
        or binding.endpoint != model.endpoint
        or not lease.carrier
        or not lease.response_id
        or not math.isfinite(lease.created_at)
        or lease.created_at < 0
    ):
        raise invalid("request lease")


def _message_and_finish(native: JsonObject) -> tuple[JsonObject, str]:
    choices = array_value(native.get("choices"), "choices", upstream=True)
    if len(choices) != 1:
        raise invalid("choices", upstream=True)
    choice = object_value(choices[0], "choice", upstream=True)
    if "index" in choice and (
        isinstance(choice["index"], bool) or choice["index"] != 0
    ):
        raise invalid("choice index", upstream=True)
    finish = choice.get("finish_reason")
    if finish not in ("stop", "tool_calls", "length"):
        raise invalid("finish_reason", upstream=True)
    message = object_value(choice.get("message"), "assistant message", upstream=True)
    if message.get("role") != "assistant":
        raise invalid("assistant role", upstream=True)
    return message, finish


def _tool_calls(message: JsonObject) -> tuple[JsonObject, ...]:
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return ()
    return tuple(
        object_value(call, "tool call", upstream=True)
        for call in array_value(raw_calls, "tool_calls", upstream=True)
    )


def _native_ids(calls: tuple[JsonObject, ...]) -> tuple[str, ...]:
    ids = tuple(
        string_value(call.get("id"), "tool call id", upstream=True, empty=True)
        for call in calls
    )
    if len(set(ids)) != len(ids) or any(value and not value.strip() for value in ids):
        raise invalid("ambiguous tool call ids", upstream=True)
    return ids


def _wire_id(native_id: str, native_ids: tuple[str, ...], lease: RequestLease) -> str:
    if native_id:
        return native_id
    candidates = (_id(lease, "call", index) for index in range(len(native_ids) + 1))
    return next(candidate for candidate in candidates if candidate not in native_ids)


def _function_arguments(value: JsonValue) -> str:
    arguments = string_value(value, "function arguments", upstream=True, empty=True)
    try:
        decode_object(arguments.encode("utf-8"))
    except (AdapterError, UnicodeError, ValueError, TypeError) as exc:
        raise invalid("function arguments", upstream=True) from exc
    return arguments


def _projection_for_call(
    call: JsonObject,
    native_ids: tuple[str, ...],
    names: tuple[ToolName, ...],
    lease: RequestLease,
    index: int,
) -> tuple[JsonObject, tuple[str, str]]:
    if call.get("type") != "function":
        raise invalid("tool call type", upstream=True)
    function = object_value(call.get("function"), "tool call function", upstream=True)
    upstream_name = string_value(function.get("name"), "function name", upstream=True)
    matches = tuple(name for name in names if name.upstream_name == upstream_name)
    if len(matches) != 1:
        raise invalid("unregistered function name", upstream=True)
    arguments = _function_arguments(function.get("arguments"))
    native_id = native_ids[index]
    wire_id = _wire_id(native_id, native_ids, lease)
    projection: JsonObject = {
        "id": _id(lease, "fc", index),
        "type": "function_call",
        "status": "completed",
        "call_id": wire_id,
        "name": matches[0].name,
        "arguments": arguments,
    }
    if matches[0].namespace is not None:
        projection = {**projection, "namespace": matches[0].namespace}
    return projection, (wire_id, native_id)


def _text_projection(
    message: JsonObject, lease: RequestLease, finish: str
) -> tuple[JsonObject, ...]:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise invalid("assistant content", upstream=True)
    if not content:
        return ()
    return (
        {
            "id": _id(lease, "msg"),
            "type": "message",
            "role": "assistant",
            "status": "incomplete" if finish == "length" else "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        },
    )


def _projections(
    message: JsonObject, finish: str, prepared: PreparedGemini, lease: RequestLease
) -> tuple[tuple[JsonObject, ...], tuple[tuple[str, str], ...]]:
    calls = _tool_calls(message)
    if (finish == "tool_calls") != bool(calls) or (finish == "length" and calls):
        raise invalid("finish reason and tool calls", upstream=True)
    ids = _native_ids(calls)
    projected = tuple(
        _projection_for_call(call, ids, prepared.tool_names, lease, index)
        for index, call in enumerate(calls)
    )
    items = (
        *_text_projection(message, lease, finish),
        *(item for item, _ in projected),
    )
    if not items:
        raise invalid("empty assistant output", upstream=True)
    return items, tuple(pair for _, pair in projected)


def convert_response(
    native: JsonObject,
    prepared: PreparedGemini,
    lease: RequestLease,
    model: RegisteredModel,
) -> Completion:
    """Build a lossless native snapshot and a client-visible Responses projection."""
    _validate_lease(lease, model)
    checked = snapshot(native, upstream=True)
    message, finish = _message_and_finish(checked)
    projections, call_ids = _projections(message, finish, prepared, lease)
    usage, raw_usage = normalize_usage(checked.get("usage"))
    reasoning: JsonObject = {
        "id": _id(lease, "rs"),
        "type": "reasoning",
        "summary": [],
        "encrypted_content": lease.carrier,
    }
    response: JsonObject = {
        "id": lease.response_id,
        "object": "response",
        "created_at": int(lease.created_at),
        "status": "incomplete" if finish == "length" else "completed",
        "model": model.id,
        "output": [reasoning, *projections],
        "usage": usage,
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if finish == "length" else None
        ),
    }
    native_bytes = encode_json(message)
    messages = (*prepared.messages, native_bytes)
    turn = SavedTurn(
        carrier=lease.carrier,
        native_message=native_bytes,
        projections=tuple(encode_json(item) for item in projections),
        call_ids=call_ids,
        prefix_count=len(messages),
        prefix_digest=messages_digest(messages),
        options_digest=prepared.options_digest,
    )
    return Completion(response=encode_json(response), turn=turn, raw_usage=raw_usage)
