"""Rebuild complete native Chat history from authenticated saved turns."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from codex_adapter.domain.json_codec import JsonObject, JsonValue, encode_json
from codex_adapter.domain.models import ModelCapabilities
from codex_adapter.domain.state import SavedTurn

from .gemini_common import (
    array_value,
    conflict,
    invalid,
    messages_digest,
    object_value,
    reject_extra,
    saved_object,
    string_value,
    unsupported,
)
from .gemini_content import message_content

_TOOL_IMAGE_PLACEHOLDER = "Image attached in the following user message."


@dataclass(frozen=True)
class _Context:
    items: tuple[JsonObject, ...]
    turns: tuple[SavedTurn, ...]
    options_digest: str
    capabilities: ModelCapabilities


@dataclass(frozen=True)
class _Cursor:
    messages: tuple[bytes, ...]
    objects: tuple[JsonValue, ...] = ()
    index: int = 0
    restored: int = 0
    pending: tuple[tuple[str, str], ...] = ()
    deferred_images: tuple[JsonValue, ...] = ()


def _items(body: JsonObject) -> tuple[JsonObject, ...]:
    value = body.get("input")
    if isinstance(value, str):
        return ({"role": "user", "content": value},)
    return tuple(
        object_value(item, "input item") for item in array_value(value, "input")
    )


def last_carrier(body: JsonObject) -> str | None:
    """Find the final opaque handle without attempting to authenticate it."""
    result: str | None = None
    for item in _items(body):
        if item.get("type") == "reasoning":
            result = string_value(
                item.get("encrypted_content"), "reasoning.encrypted_content"
            )
    return result


def _reasoning_carrier(item: JsonObject) -> str:
    reject_extra(
        item,
        {"type", "id", "summary", "encrypted_content", "status", "content"},
        "reasoning item field",
    )
    if "id" in item:
        string_value(item["id"], "reasoning.id")
    # The pinned Codex core adds an empty optional content field on replay.
    # Nonempty reasoning content is not a projection of our opaque reference.
    if item.get("content") not in (None, []):
        raise unsupported("reasoning.content")
    if item.get("summary", []) != [] or item.get("status", "completed") != "completed":
        raise unsupported("reasoning item")
    return string_value(item.get("encrypted_content"), "reasoning.encrypted_content")


def _saved_calls(turn: SavedTurn, native: JsonObject) -> tuple[tuple[str, str], ...]:
    projections = tuple(saved_object(value) for value in turn.projections)
    wire_ids = tuple(
        item.get("call_id")
        for item in projections
        if item.get("type") == "function_call"
    )
    native_calls = native.get("tool_calls")
    if native_calls is None:
        native_calls = []
    if not isinstance(native_calls, list) or any(
        not isinstance(call, dict) for call in native_calls
    ):
        raise conflict()
    native_ids = tuple(
        call.get("id") for call in native_calls if isinstance(call, dict)
    )
    pairs = turn.call_ids
    if any(
        not isinstance(wire, str) or not wire or not isinstance(native_id, str)
        for wire, native_id in pairs
    ):
        raise conflict()
    if wire_ids != tuple(wire for wire, _ in pairs) or native_ids != tuple(
        native_id for _, native_id in pairs
    ):
        raise conflict()
    if len(set(wire_ids)) != len(wire_ids) or len(set(native_ids)) != len(native_ids):
        raise conflict()
    return pairs


@lru_cache(maxsize=4096)
def _validated_turn_data(
    native_message: bytes,
    projections: tuple[bytes, ...],
    call_ids: tuple[tuple[str, str], ...],
) -> tuple[JsonObject, tuple[tuple[str, str], ...]]:
    native = saved_object(native_message)
    if native.get("role") != "assistant" or encode_json(native) != native_message:
        raise conflict()
    turn = SavedTurn(
        carrier="cached",
        native_message=native_message,
        projections=projections,
        call_ids=call_ids,
        prefix_count=0,
        prefix_digest="cached",
        options_digest="cached",
    )
    return native, _saved_calls(turn, native)


def _restore(context: _Context, cursor: _Cursor) -> _Cursor:
    if cursor.pending or cursor.restored >= len(context.turns):
        raise conflict()
    item = context.items[cursor.index]
    turn = context.turns[cursor.restored]
    if (
        _reasoning_carrier(item) != turn.carrier
        or turn.options_digest != context.options_digest
    ):
        raise conflict()
    native, saved_calls = _validated_turn_data(
        turn.native_message, turn.projections, turn.call_ids
    )
    messages = (*cursor.messages, turn.native_message)
    if (
        len(messages) != turn.prefix_count
        or messages_digest(messages) != turn.prefix_digest
    ):
        raise conflict()
    first = cursor.index + 1
    actual = context.items[first : first + len(turn.projections)]
    if len(actual) != len(turn.projections):
        raise conflict()
    if tuple(encode_json(item) for item in actual) != turn.projections:
        raise conflict()
    return _Cursor(
        messages,
        (*cursor.objects, native),
        first + len(actual),
        cursor.restored + 1,
        saved_calls,
    )


def _optional_item_fields(item: JsonObject) -> None:
    if "id" in item:
        string_value(item["id"], "input.id")
    if item.get("status", "completed") != "completed":
        raise unsupported("input.status")


def _split_tool_content(
    content: JsonValue,
) -> tuple[str, tuple[JsonValue, ...]]:
    if isinstance(content, str):
        return content, ()
    if not isinstance(content, list):
        raise invalid("tool output content")
    texts: list[str] = []
    images: list[JsonValue] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        else:
            images.append(part)
    tool_text = "\n".join(texts) if texts else _TOOL_IMAGE_PLACEHOLDER
    return tool_text, tuple(images)


def _tool_output(context: _Context, cursor: _Cursor, item: JsonObject) -> _Cursor:
    reject_extra(
        item, {"type", "call_id", "output", "id", "status"}, "tool output field"
    )
    _optional_item_fields(item)
    call_id = string_value(item.get("call_id"), "function_call_output.call_id")
    matches = tuple(pair for pair in cursor.pending if pair[0] == call_id)
    if len(matches) != 1:
        raise conflict()
    # Codex view_image returns image parts inside the function result. ModelHub
    # Gemini Chat treats role="tool" content as plain text inside functionResponse
    # (turning base64 images into hundreds of thousands of text tokens), whereas
    # role="user" image_url parts are parsed as native multimodal images. Emit a
    # text receipt in the tool message and attach the image parts in a following
    # user message once all pending tool results for the turn are collected.
    content = message_content(
        item.get("output"), context.capabilities, allow_images=True
    )
    tool_text, images = _split_tool_content(content)
    message: JsonObject = {
        "role": "tool",
        "tool_call_id": matches[0][1],
        "content": tool_text,
    }
    messages = (*cursor.messages, encode_json(message))
    objects = (*cursor.objects, message)
    deferred = (*cursor.deferred_images, *images)
    remaining = tuple(pair for pair in cursor.pending if pair[0] != call_id)
    if not remaining and deferred:
        user_image_msg: JsonObject = {
            "role": "user",
            "content": list(deferred),
        }
        messages = (*messages, encode_json(user_image_msg))
        objects = (*objects, user_image_msg)
        deferred = ()
    return _Cursor(
        messages,
        objects,
        cursor.index + 1,
        cursor.restored,
        remaining,
        deferred,
    )


def _normal_message(context: _Context, cursor: _Cursor, item: JsonObject) -> _Cursor:
    if cursor.pending:
        raise conflict()
    reject_extra(item, {"type", "role", "content", "id", "status"}, "message field")
    _optional_item_fields(item)
    if item.get("type", "message") != "message":
        raise unsupported("input item type")
    role = item.get("role")
    if role not in ("user", "system", "developer"):
        raise unsupported("input role or assistant without state")
    content = message_content(
        item.get("content"), context.capabilities, allow_images=role == "user"
    )
    message: JsonObject = {
        "role": "system" if role == "developer" else role,
        "content": content,
    }
    return _Cursor(
        (*cursor.messages, encode_json(message)),
        (*cursor.objects, message),
        cursor.index + 1,
        cursor.restored,
    )


def reconstruct_messages(
    body: JsonObject,
    turns: tuple[SavedTurn, ...],
    options_digest: str,
    capabilities: ModelCapabilities,
) -> tuple[tuple[bytes, ...], list[JsonValue]]:
    """Require the entire ordered chain, exact projections, and committed prefixes."""
    instructions = body.get("instructions")
    messages: tuple[bytes, ...] = ()
    objects: tuple[JsonValue, ...] = ()
    if instructions is not None:
        text = string_value(instructions, "instructions", empty=True)
        if text:
            system_msg: JsonObject = {"role": "system", "content": text}
            messages = (encode_json(system_msg),)
            objects = (system_msg,)
    context = _Context(_items(body), turns, options_digest, capabilities)
    cursor = _Cursor(messages, objects)
    while cursor.index < len(context.items):
        item = context.items[cursor.index]
        kind = item.get("type", "message")
        if kind == "reasoning":
            cursor = _restore(context, cursor)
        elif kind == "function_call_output":
            cursor = _tool_output(context, cursor, item)
        else:
            cursor = _normal_message(context, cursor, item)
    if cursor.restored != len(turns) or cursor.pending:
        raise conflict()
    if not cursor.messages:
        raise invalid("input")
    return cursor.messages, list(cursor.objects)
