"""Encode buffered Responses objects as complete, ordered SSE lifecycles."""

from __future__ import annotations

from codex_adapter.domain.json_codec import JsonObject, decode_object, encode_json

from .gemini_common import array_value, invalid, object_value, string_value


def _event(kind: str, **fields: object) -> JsonObject:
    return {"type": kind, **fields}  # type: ignore[dict-item]


def _message_events(item: JsonObject, output_index: int) -> tuple[JsonObject, ...]:
    item_id = string_value(item.get("id"), "output item id")
    parts = array_value(item.get("content"), "output content")
    events: tuple[JsonObject, ...] = ()
    for content_index, raw_part in enumerate(parts):
        part = object_value(raw_part, "output content part")
        if part.get("type") != "output_text":
            raise invalid("output content part")
        text = string_value(part.get("text"), "output text", empty=True)
        shared = {
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
        }
        events = (
            *events,
            _event("response.content_part.added", **shared, part={**part, "text": ""}),
        )
        if text:
            events = (
                *events,
                _event("response.output_text.delta", **shared, delta=text),
            )
        events = (
            *events,
            _event("response.output_text.done", **shared, text=text),
            _event("response.content_part.done", **shared, part=part),
        )
    return events


def _function_events(item: JsonObject, output_index: int) -> tuple[JsonObject, ...]:
    item_id = string_value(item.get("id"), "output item id")
    arguments = string_value(item.get("arguments"), "function arguments", empty=True)
    shared = {"item_id": item_id, "output_index": output_index}
    events: tuple[JsonObject, ...] = ()
    if arguments:
        events = (
            _event("response.function_call_arguments.delta", **shared, delta=arguments),
        )
    return (
        *events,
        _event("response.function_call_arguments.done", **shared, arguments=arguments),
    )


def _item_events(item: JsonObject, output_index: int) -> tuple[JsonObject, ...]:
    kind = item.get("type")
    string_value(item.get("id"), "output item id")
    added = {**item, "status": "in_progress"}
    details: tuple[JsonObject, ...] = ()
    if kind == "function_call":
        added = {**added, "arguments": ""}
        details = _function_events(item, output_index)
    elif kind == "message":
        added = {**added, "content": []}
        details = _message_events(item, output_index)
    elif kind != "reasoning":
        raise invalid("output item type")
    return (
        _event("response.output_item.added", output_index=output_index, item=added),
        *details,
        _event("response.output_item.done", output_index=output_index, item=item),
    )


def _serialize_event(event: JsonObject, number: int) -> bytes:
    kind = string_value(event.get("type"), "event type")
    encoded = encode_json({**event, "sequence_number": number})
    return b"event: " + kind.encode("utf-8") + b"\ndata: " + encoded + b"\n\n"


def response_events(response: bytes) -> tuple[bytes, ...]:
    """Render stable SSE frames from one already-complete response snapshot."""
    payload = decode_object(response)
    status = payload.get("status")
    if status not in ("completed", "incomplete"):
        raise invalid("response status")
    outputs = array_value(payload.get("output"), "response output")
    initial: JsonObject = {
        **payload,
        "status": "in_progress",
        "output": [],
        "usage": None,
        "incomplete_details": None,
    }
    events: tuple[JsonObject, ...] = (
        _event("response.created", response=initial),
        _event("response.in_progress", response=initial),
    )
    for index, value in enumerate(outputs):
        events = (*events, *_item_events(object_value(value, "output item"), index))
    events = (*events, _event(f"response.{status}", response=payload))
    return tuple(_serialize_event(event, number) for number, event in enumerate(events))
