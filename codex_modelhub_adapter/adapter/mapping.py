from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, AsyncIterator


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SanitizedStateStats:
    removed_encrypted_content_count: int
    dropped_empty_reasoning_count: int
    removed_previous_response_id: bool


def build_modelhub_payload(request: JsonObject) -> JsonObject:
    """Convert a Responses API request into the legacy ModelHub crawl payload."""
    payload: JsonObject = {
        "stream": False,
        "model": request.get("model"),
        "messages": _responses_input_to_modelhub_messages(request),
    }

    max_tokens = request.get("max_output_tokens") or request.get("max_tokens")
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    for key in ("temperature", "top_p"):
        if key in request and request[key] is not None:
            payload[key] = request[key]

    tools = _responses_tools_to_chat_tools(request.get("tools"))
    if tools:
        payload["tools"] = tools

    tool_choice = _responses_tool_choice_to_chat_tool_choice(request.get("tool_choice"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return payload


def normalize_responses_body(
    raw_body: Any,
    *,
    max_output_tokens: int = 65536,
    ensure_store: bool = True,
) -> JsonObject:
    body = dict(raw_body) if isinstance(raw_body, dict) else {}
    current = body.get("max_output_tokens")
    if not isinstance(current, int) or current < max_output_tokens:
        body["max_output_tokens"] = max_output_tokens
    if ensure_store and "store" not in body:
        body["store"] = True
    return body


def build_chat_completions_body(
    raw_body: Any,
    *,
    max_output_tokens: int = 65536,
    token_limit: int = 820000,
    chars_per_token: float = 2.8,
) -> JsonObject:
    """Adapt Responses requests to the AIDP Chat Completions crawl contract."""
    responses_body = normalize_responses_body(
        raw_body,
        max_output_tokens=max_output_tokens,
        ensure_store=False,
    )
    messages = _trim_chat_messages_for_context(
        _responses_input_to_chat_messages(responses_body),
        token_limit=token_limit,
        chars_per_token=chars_per_token,
    )
    chat_body: JsonObject = {
        "model": responses_body.get("model"),
        "messages": messages,
        "stream": bool(responses_body.get("stream")),
        "max_tokens": responses_body.get("max_output_tokens"),
    }
    # Chat Completions only returns token usage (incl. prompt-cache hits) on a
    # streamed call when include_usage is requested. Without this, streamed
    # responses carry no usage at all and prompt-cache hits are invisible.
    if chat_body["stream"]:
        chat_body["stream_options"] = {"include_usage": True}

    tools = _responses_tools_to_chat_tools(responses_body.get("tools"))
    if tools:
        chat_body["tools"] = tools
        if "tool_choice" in responses_body:
            chat_body["tool_choice"] = _responses_tool_choice_to_chat_tool_choice(
                responses_body.get("tool_choice")
            )
        if "parallel_tool_calls" in responses_body:
            chat_body["parallel_tool_calls"] = responses_body["parallel_tool_calls"]

    for source_key, target_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("store", "store"),
    ):
        if source_key in responses_body:
            chat_body[target_key] = responses_body[source_key]

    response_format = _responses_text_format_to_chat_response_format(
        responses_body.get("text")
    )
    if response_format is not None:
        chat_body["response_format"] = response_format

    reasoning = responses_body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            chat_body["reasoning_effort"] = effort

    return {key: value for key, value in chat_body.items() if value is not None}


def _responses_text_format_to_chat_response_format(text_field: Any) -> JsonObject | None:
    """Map a Responses ``text.format`` block to a Chat ``response_format``.

    Codex sends a turn ``output_schema`` as
    ``text: {"format": {"type": "json_schema", "name", "strict", "schema"}}``.
    The Chat Completions contract expects
    ``response_format: {"type": "json_schema", "json_schema": {...}}``. Only the
    ``json_schema`` form is forwarded; ``text``/other formats are ignored so the
    model is not constrained when no schema was requested.
    """
    if not isinstance(text_field, dict):
        return None
    fmt = text_field.get("format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return None
    schema = fmt.get("schema")
    if not isinstance(schema, dict):
        return None
    json_schema: JsonObject = {
        "name": str(fmt.get("name") or "codex_output_schema"),
        "schema": schema,
    }
    if "strict" in fmt:
        json_schema["strict"] = bool(fmt.get("strict"))
    return {"type": "json_schema", "json_schema": json_schema}


def build_responses_payload(modelhub_response: JsonObject, model: str | None = None) -> JsonObject:
    """Convert a ModelHub or Chat Completions response into Responses format."""
    choice = _first_choice(modelhub_response)
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    output = _chat_message_to_response_output(message)
    if not output:
        text = extract_text(modelhub_response)
        output = [
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]

    response: JsonObject = {
        "id": _response_id(modelhub_response),
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model or modelhub_response.get("model"),
        "output": output,
        "output_text": _collect_output_text(output),
        "usage": _usage(modelhub_response),
    }
    return response


def build_compaction_response(raw_body: Any, *, max_chars: int = 60000) -> JsonObject:
    body = normalize_responses_body(raw_body, ensure_store=False)
    summary = _summarize_responses_input_for_compaction(body.get("input"), max_chars=max_chars)
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response.compaction",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model"),
        "output": [
            {
                "id": message_id,
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": summary}],
            }
        ],
        "metadata": {
            "codex_modelhub_adapter_compaction_source": "local_proxy_compaction",
            "codex_modelhub_adapter_compaction_format": "lossy_text_checkpoint",
        },
        "usage": None,
    }


def chat_completion_to_response(payload: JsonObject) -> JsonObject:
    return build_responses_payload(payload, payload.get("model"))


def extract_text(modelhub_response: JsonObject) -> str:
    """Extract text from common ModelHub and Chat Completions response shapes."""
    for key in ("output_text", "text", "response", "content"):
        value = modelhub_response.get(key)
        if isinstance(value, str):
            return value

    data = modelhub_response.get("data")
    if isinstance(data, dict):
        for key in ("output", "text", "response", "content"):
            value = data.get(key)
            if isinstance(value, str):
                return value

    choices = modelhub_response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") or first.get("delta") or {}
            if isinstance(message, dict):
                return _content_to_text(message.get("content"))
            return _content_to_text(first.get("text"))

    return ""


def extract_tool_calls(modelhub_response: JsonObject) -> list[JsonObject]:
    choices = modelhub_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, dict):
        return []
    message = first.get("message") or first.get("delta") or {}
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        return [call for call in tool_calls if isinstance(call, dict)]
    return []


def iter_sse_events(response: JsonObject) -> Iterable[str]:
    """Emit a minimal Responses SSE stream from a completed response object."""
    output = response.get("output") if isinstance(response.get("output"), list) else []
    item = output[0] if output else {}
    text = response.get("output_text") or ""

    yield _sse(
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
            "sequence_number": 0,
        }
    )

    if item.get("type") == "message":
        yield _sse(
            {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "sequence_number": 1,
            }
        )
        yield _sse(
            {
                "type": "response.output_text.done",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "text": text,
                "sequence_number": 2,
            }
        )

    yield _sse(
        {
            "type": "response.completed",
            "response": response,
            "sequence_number": 3,
        }
    )
    yield "data: [DONE]\n\n"


async def iter_chat_sse_as_responses(
    chunks: AsyncIterator[str],
    *,
    logid: str = "",
    key_alias: str = "",
) -> AsyncIterator[bytes]:
    """Translate Chat Completions SSE chunks into the Responses SSE lifecycle."""
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    sequence = 1
    text_item_id = f"msg_{uuid.uuid4().hex[:24]}"
    text_started = False
    text = ""
    tool_calls: dict[int, JsonObject] = {}
    captured_usage: JsonObject | None = None

    def event(payload: JsonObject) -> bytes:
        return (
            f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        ).encode("utf-8")

    yield event(
        {
            "type": "response.created",
            "sequence_number": sequence,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created,
                "status": "in_progress",
            },
        }
    )
    sequence += 1

    async for data in _iter_sse_data(chunks):
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("usage"), dict):
            captured_usage = payload["usage"]
        choice = _first_choice(payload)
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        content_delta = delta.get("content")
        if isinstance(content_delta, str) and content_delta:
            if not text_started:
                text_started = True
                yield event(
                    {
                        "type": "response.output_item.added",
                        "sequence_number": sequence,
                        "output_index": 0,
                        "item": {
                            "id": text_item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    }
                )
                sequence += 1
                yield event(
                    {
                        "type": "response.content_part.added",
                        "sequence_number": sequence,
                        "item_id": text_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
                sequence += 1
            text += content_delta
            yield event(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": text_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content_delta,
                }
            )
            sequence += 1

        for call_delta in delta.get("tool_calls") or []:
            if not isinstance(call_delta, dict):
                continue
            index = int(call_delta.get("index") or 0)
            generated_call_id = f"call_{uuid.uuid4().hex[:24]}"
            current = tool_calls.setdefault(
                index,
                {
                    "id": call_delta.get("id") or generated_call_id,
                    "type": "function_call",
                    "call_id": call_delta.get("id") or generated_call_id,
                    "name": "",
                    "arguments": "",
                    "status": "in_progress",
                },
            )
            if call_delta.get("id"):
                current["id"] = call_delta["id"]
                current["call_id"] = call_delta["id"]
            function_delta = (
                call_delta.get("function")
                if isinstance(call_delta.get("function"), dict)
                else {}
            )
            if function_delta.get("name"):
                current["name"] = function_delta["name"]
            arguments_delta = function_delta.get("arguments")
            if not current.get("_added"):
                current["_added"] = True
                yield event(
                    {
                        "type": "response.output_item.added",
                        "sequence_number": sequence,
                        "output_index": index,
                        "item": {
                            key: value
                            for key, value in current.items()
                            if not key.startswith("_")
                        },
                    }
                )
                sequence += 1
            if isinstance(arguments_delta, str) and arguments_delta:
                current["arguments"] += arguments_delta
                yield event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": sequence,
                        "item_id": current["id"],
                        "output_index": index,
                        "delta": arguments_delta,
                    }
                )
                sequence += 1

    if text_started:
        yield event(
            {
                "type": "response.output_text.done",
                "sequence_number": sequence,
                "item_id": text_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            }
        )
        sequence += 1
        yield event(
            {
                "type": "response.content_part.done",
                "sequence_number": sequence,
                "item_id": text_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            }
        )
        sequence += 1
        yield event(
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": 0,
                "item": {
                    "id": text_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                },
            }
        )
        sequence += 1

    output: list[JsonObject] = []
    if text_started:
        output.append(
            {
                "id": text_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for index in sorted(tool_calls):
        item = {
            key: value
            for key, value in tool_calls[index].items()
            if not key.startswith("_")
        }
        item["status"] = "completed"
        output.append(item)
        yield event(
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": sequence,
                "item_id": item["id"],
                "output_index": index,
                "arguments": item.get("arguments", ""),
                "name": item.get("name", ""),
            }
        )
        sequence += 1
        yield event(
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": index,
                "item": item,
            }
        )
        sequence += 1

    completed_response: JsonObject = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "output": output,
        "output_text": text,
    }
    if captured_usage is not None:
        completed_response["usage"] = _usage({"usage": captured_usage})
        log_cache_hit(captured_usage, logid=logid, key_alias=key_alias)
    yield event(
        {
            "type": "response.completed",
            "sequence_number": sequence,
            "response": completed_response,
        }
    )
    yield b"data: [DONE]\n\n"


def sanitize_encrypted_state(raw_body: JsonObject) -> tuple[JsonObject, SanitizedStateStats]:
    sanitized, stats = _sanitize_encrypted_state_value(raw_body, top_level=True)
    if not isinstance(sanitized, dict):
        sanitized = {}
    return sanitized, SanitizedStateStats(
        removed_encrypted_content_count=int(stats["removed_encrypted_content_count"]),
        dropped_empty_reasoning_count=int(stats["dropped_empty_reasoning_count"]),
        removed_previous_response_id=bool(stats["removed_previous_response_id"]),
    )


def is_context_length_exceeded(body: bytes) -> bool:
    if not body:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return b"context_length_exceeded" in body
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        values = [error.get("code"), error.get("message"), error.get("type")]
        return any(
            "context_length_exceeded" in str(value)
            for value in values
            if value is not None
        )
    return "context_length_exceeded" in json.dumps(payload, ensure_ascii=False)


def is_invalid_encrypted_content(body: bytes) -> bool:
    if not body:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return b"invalid_encrypted_content" in body or b"Encrypted content" in body
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        values = [error.get("code"), error.get("message"), error.get("type")]
        return any(
            "invalid_encrypted_content" in str(value)
            for value in values
            if value is not None
        )
    return "invalid_encrypted_content" in json.dumps(payload, ensure_ascii=False)


def _responses_input_to_modelhub_messages(request: JsonObject) -> list[JsonObject]:
    messages: list[JsonObject] = []
    instructions = request.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": instructions}],
            }
        )

    input_value = request.get("input", "")
    if isinstance(input_value, str):
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": input_value}],
            }
        )
        return messages

    if isinstance(input_value, list):
        for item in input_value:
            message = _input_item_to_modelhub_message(item)
            if message is not None:
                messages.append(message)

    return messages


def _input_item_to_modelhub_message(item: Any) -> JsonObject | None:
    if isinstance(item, str):
        return {"role": "user", "content": [{"type": "text", "text": item}]}

    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    role = item.get("role") or ("assistant" if item_type == "message" else "user")

    if item_type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": item.get("call_id") or item.get("id"),
            "content": _content_to_text(item.get("output")),
        }

    content = item.get("content")
    return {
        "role": role,
        "content": _content_to_modelhub_parts(content),
    }


def _content_to_modelhub_parts(content: Any) -> list[JsonObject]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return []

    parts: list[JsonObject] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            parts.append({"type": "text", "text": part.get("text", "")})
        elif part_type in {"input_image", "image_url"}:
            parts.append(part)
    return parts


def _responses_input_to_chat_messages(body: JsonObject) -> list[JsonObject]:
    messages: list[JsonObject] = []
    pending_tool_calls: list[JsonObject] = []
    pending_tool_outputs: list[JsonObject] = []
    pending_tool_images: list[JsonObject] = []

    def flush_pending_tools() -> None:
        nonlocal pending_tool_calls, pending_tool_outputs, pending_tool_images
        if not pending_tool_calls and not pending_tool_outputs and not pending_tool_images:
            return

        output_call_ids = {
            str(output.get("tool_call_id") or "")
            for output in pending_tool_outputs
        }
        paired_tool_calls = [
            call
            for call in pending_tool_calls
            if str(call.get("id") or "") in output_call_ids
        ]
        paired_call_ids = {str(call.get("id") or "") for call in paired_tool_calls}
        if paired_tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": paired_tool_calls,
                }
            )
            for output in pending_tool_outputs:
                if output.get("tool_call_id") in paired_call_ids:
                    messages.append(output)
                else:
                    messages.append(_orphan_tool_output_to_user_message(output))
        else:
            for output in pending_tool_outputs:
                messages.append(_orphan_tool_output_to_user_message(output))

        # Chat Completions `tool` messages cannot carry images, so surface any
        # image parts a tool returned (e.g. Codex `view_image`) as a follow-up
        # user message; otherwise the model never sees the pixels.
        if pending_tool_images:
            messages.append({"role": "user", "content": list(pending_tool_images)})

        pending_tool_calls = []
        pending_tool_outputs = []
        pending_tool_images = []

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    elif isinstance(instructions, list):
        text = _content_to_text(instructions)
        if text:
            messages.append({"role": "system", "content": text})

    input_value = body.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages
    if not isinstance(input_value, list):
        return messages

    for item in input_value:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call_output":
            output = item.get("output")
            text_part = (
                _content_to_text(output) if isinstance(output, list) else str(output or "")
            )
            image_parts = _extract_chat_image_parts(output)
            if image_parts and not text_part:
                text_part = "(tool returned an image; shown in the next user message)"
            pending_tool_outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": text_part,
                }
            )
            pending_tool_images.extend(image_parts)
            continue
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}"
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
            continue
        role = item.get("role")
        if role in {"user", "assistant", "system", "developer"}:
            flush_pending_tools()
            messages.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": _content_to_chat_content(item.get("content")),
                }
            )
    flush_pending_tools()
    return messages


def _trim_chat_messages_for_context(
    messages: list[JsonObject],
    *,
    token_limit: int,
    chars_per_token: float,
) -> list[JsonObject]:
    if token_limit <= 0 or _estimate_chat_tokens(messages, chars_per_token=chars_per_token) <= token_limit:
        return messages

    system_messages = [message for message in messages if message.get("role") == "system"]
    conversation = [message for message in messages if message.get("role") != "system"]
    blocks = _split_chat_messages_into_safe_blocks(conversation)
    kept_blocks = list(blocks)
    dropped_count = 0

    while kept_blocks:
        candidate = _flatten_chat_blocks(kept_blocks)
        compact_notice = _build_chat_context_compaction_notice(dropped_count + 1)
        trimmed = [*system_messages, compact_notice, *candidate]
        if _estimate_chat_tokens(trimmed, chars_per_token=chars_per_token) <= token_limit:
            return trimmed
        kept_blocks.pop(0)
        dropped_count += 1

    newest_block = blocks[-1] if blocks else []
    return [
        *system_messages,
        _build_chat_context_compaction_notice(dropped_count or len(blocks)),
        *newest_block,
    ]


def _estimate_chat_tokens(messages: list[JsonObject], *, chars_per_token: float) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / max(chars_per_token, 0.1)))


def _split_chat_messages_into_safe_blocks(messages: list[JsonObject]) -> list[list[JsonObject]]:
    blocks: list[list[JsonObject]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            block = [message]
            call_ids = {
                str(call.get("id") or "")
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and str(call.get("id") or "")
            }
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                block.append(messages[index])
                if not call_ids or len(block) - 1 >= len(call_ids):
                    index += 1
                    break
                index += 1
            blocks.append(block)
            continue
        if role == "tool":
            blocks.append(
                [
                    _orphan_tool_output_to_user_message(
                        {
                            "tool_call_id": str(message.get("tool_call_id") or ""),
                            "content": str(message.get("content") or ""),
                        }
                    )
                ]
            )
        else:
            blocks.append([message])
        index += 1
    return blocks


def _flatten_chat_blocks(blocks: list[list[JsonObject]]) -> list[JsonObject]:
    return [message for block in blocks for message in block]


def _build_chat_context_compaction_notice(dropped_blocks: int) -> JsonObject:
    return {
        "role": "system",
        "content": (
            "Codex ModelHub adapter compacted older history for GPT-5.5 Chat upstream. "
            f"Dropped blocks: {dropped_blocks}. Continue from the remaining recent messages."
        ),
    }


def _orphan_tool_output_to_user_message(output: JsonObject) -> JsonObject:
    return {
        "role": "user",
        "content": f"Tool output {output.get('tool_call_id') or 'unknown'}:\n{output.get('content') or ''}",
    }


def _content_to_chat_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[JsonObject] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            parts.append({"type": "text", "text": part.get("text") or ""})
        elif part_type in {"input_image", "image_url"} and part.get("image_url"):
            parts.append({"type": "image_url", "image_url": part["image_url"]})
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0].get("text") or ""
    return parts


def _extract_chat_image_parts(output: Any) -> list[JsonObject]:
    """Pull image parts out of a tool output (Responses ``input_image`` form).

    Chat Completions ``tool`` role messages cannot carry images, so callers
    surface these as a separate user message instead.
    """
    if not isinstance(output, list):
        return []
    parts: list[JsonObject] = []
    for part in output:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_image", "image_url"} and part.get("image_url"):
            parts.append({"type": "image_url", "image_url": part["image_url"]})
    return parts


def _content_to_text(content: Any) -> str:
    converted = _content_to_chat_content(content)
    if isinstance(converted, str):
        return converted
    if isinstance(converted, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in converted
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _summarize_responses_input_for_compaction(input_items: Any, *, max_chars: int) -> str:
    lines = [
        "CONTEXT CHECKPOINT SUMMARY",
        "Older Responses API input was compacted by the Codex ModelHub adapter.",
        "This is a lossy text checkpoint for AIDP Codex proxy compatibility.",
        "",
    ]
    if isinstance(input_items, str):
        lines.append(input_items)
    elif isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            role = str(item.get("role") or "")
            if item_type == "function_call":
                lines.append(
                    f"[function_call] {item.get('name') or ''} "
                    f"call_id={item.get('call_id') or item.get('id') or ''} "
                    f"arguments={_truncate_text(str(item.get('arguments') or ''), 2000)}"
                )
            elif item_type == "function_call_output":
                lines.append(
                    f"[function_call_output] call_id={item.get('call_id') or item.get('id') or ''}\n"
                    f"{_truncate_text(str(item.get('output') or ''), 4000)}"
                )
            elif role:
                lines.append(f"[{role}] {_truncate_text(_content_to_text(item.get('content')), 6000)}")
    summary = "\n".join(line for line in lines if line is not None)
    return _truncate_text(summary, max_chars)


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."


def _responses_tools_to_chat_tools(tools: Any) -> list[JsonObject]:
    if not isinstance(tools, list):
        return []

    converted: list[JsonObject] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if tool.get("type") not in {None, "function"} and not isinstance(tool.get("function"), dict):
            continue

        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters") or function.get("input_schema") or {},
                },
            }
        )

    return converted


def _responses_tool_choice_to_chat_tool_choice(tool_choice: Any) -> Any:
    if tool_choice in (None, "auto", "none", "required"):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice

    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name"):
        return {"type": "function", "function": {"name": function["name"]}}

    name = tool_choice.get("name")
    if isinstance(name, str) and name:
        return {"type": "function", "function": {"name": name}}

    return "auto"


def _chat_message_to_response_output(message: JsonObject) -> list[JsonObject]:
    output: list[JsonObject] = []
    content = message.get("content")
    text = _content_to_text(content)
    if text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = (
            tool_call.get("function")
            if isinstance(tool_call.get("function"), dict)
            else {}
        )
        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        output.append(
            {
                "id": call_id,
                "type": "function_call",
                "call_id": call_id,
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
        )
    return output


def _collect_output_text(output: list[JsonObject]) -> str:
    texts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                texts.append(str(part.get("text") or ""))
    return "".join(texts)


def _usage(modelhub_response: JsonObject) -> JsonObject:
    usage = modelhub_response.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    result: JsonObject = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    cached = cached_tokens(usage)
    if cached is not None:
        # Surface prompt-cache hits in the Responses-format usage so the Codex
        # SDK (TokenUsageBreakdown.cached_input_tokens) can read them.
        result["input_tokens_details"] = {"cached_tokens": cached}
    out_details = usage.get("completion_tokens_details") or usage.get(
        "output_tokens_details"
    )
    if isinstance(out_details, dict) and out_details.get("reasoning_tokens") is not None:
        # Surface upstream reasoning tokens so the Codex SDK
        # (TokenUsageBreakdown.reasoning_output_tokens) can read them.
        result["output_tokens_details"] = {
            "reasoning_tokens": out_details.get("reasoning_tokens")
        }
    return result


def cached_tokens(usage: JsonObject) -> int | None:
    """Return cached prompt tokens from a Chat/ModelHub usage object, if present."""
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        try:
            return int(details["cached_tokens"])
        except (TypeError, ValueError):
            return None
    return None


def log_cache_hit(usage: JsonObject | None, *, logid: str = "", key_alias: str = "") -> None:
    """Emit a one-line prompt-cache report when AIDP_LOG_PROMPT_CACHE is set."""
    if not os.getenv("AIDP_LOG_PROMPT_CACHE"):
        return
    if not isinstance(usage, dict):
        return
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    cached = cached_tokens(usage) or 0
    ratio = (cached / prompt * 100) if prompt else 0.0
    print(
        f"[prompt-cache] alias={key_alias or '-'} logid={logid or '-'} "
        f"prompt_tokens={prompt} cached_tokens={cached} ratio={ratio:.1f}%",
        flush=True,
    )


def _response_id(modelhub_response: JsonObject) -> str:
    value = modelhub_response.get("id")
    if isinstance(value, str) and value.startswith("resp_"):
        return value
    return f"resp_{uuid.uuid4().hex}"


def _first_choice(payload: JsonObject) -> JsonObject:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


async def _iter_sse_data(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in raw_event.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                yield "\n".join(data_lines)
    if buffer.strip():
        data_lines = [
            line[5:].strip()
            for line in buffer.splitlines()
            if line.startswith("data:")
        ]
        if data_lines:
            yield "\n".join(data_lines)


class _DropSanitizedItem:
    pass


_DROP_SANITIZED_ITEM = _DropSanitizedItem()


def _sanitize_encrypted_state_value(value: Any, *, top_level: bool = False) -> tuple[Any, JsonObject]:
    stats: JsonObject = {
        "removed_encrypted_content_count": 0,
        "dropped_empty_reasoning_count": 0,
        "removed_previous_response_id": False,
    }

    def merge(child_stats: JsonObject) -> None:
        stats["removed_encrypted_content_count"] += int(
            child_stats.get("removed_encrypted_content_count") or 0
        )
        stats["dropped_empty_reasoning_count"] += int(
            child_stats.get("dropped_empty_reasoning_count") or 0
        )
        stats["removed_previous_response_id"] = bool(
            stats["removed_previous_response_id"]
            or child_stats.get("removed_previous_response_id")
        )

    if isinstance(value, list):
        sanitized_list: list[Any] = []
        for item in value:
            sanitized_item, child_stats = _sanitize_encrypted_state_value(item)
            merge(child_stats)
            if sanitized_item is _DROP_SANITIZED_ITEM:
                continue
            sanitized_list.append(sanitized_item)
        return sanitized_list, stats

    if not isinstance(value, dict):
        return value, stats

    sanitized_dict: JsonObject = {}
    for key, item_value in value.items():
        if key == "encrypted_content":
            stats["removed_encrypted_content_count"] += 1
            continue
        if top_level and key == "previous_response_id":
            stats["removed_previous_response_id"] = True
            continue
        sanitized_item, child_stats = _sanitize_encrypted_state_value(item_value)
        merge(child_stats)
        if sanitized_item is _DROP_SANITIZED_ITEM:
            continue
        sanitized_dict[key] = sanitized_item

    if _is_empty_reasoning_item(sanitized_dict):
        stats["dropped_empty_reasoning_count"] += 1
        return _DROP_SANITIZED_ITEM, stats
    return sanitized_dict, stats


def _is_empty_reasoning_item(item: JsonObject) -> bool:
    if item.get("type") != "reasoning":
        return False
    readable_keys = set(item) - {"id", "type", "status"}
    if not readable_keys:
        return True
    for key in readable_keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return False
    return True


def _sse(payload: JsonObject) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
