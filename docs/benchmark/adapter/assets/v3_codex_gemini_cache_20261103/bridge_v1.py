"""Isolated Responses-to-Gemini probe; provider state lives only in this process.

The a3d carrier is an authenticated reference, not OpenAI encrypted reasoning.
No answer cache, credential failover, retry, or reasoning substitution is used.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODEL = "gemini-3.8-flash"
UPSTREAM = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
CARRIER_PREFIX = "a3d-gemini-state-v1:"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
Json = dict[str, Any]


class ProbeError(Exception):
    def __init__(self, code: str, *, status: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _digest(value: Any) -> Json:
    encoded = _json_bytes(value)
    return {
        "redacted": True,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _redact(value: Any, *, key: str = "", secret: str = "") -> Any:
    lowered = key.lower()
    if any(
        word in lowered
        for word in (
            "signature",
            "encrypted",
            "reasoning_content",
            "authorization",
            "api_key",
            "upstream_ak",
        )
    ):
        return _digest(value)
    if lowered in {"data", "blob", "opaque", "token", "access_token", "key"}:
        return _digest(value)
    if isinstance(value, dict):
        return {
            name: _redact(item, key=name, secret=secret) for name, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret=secret) for item in value]
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED_AK]") if secret else value
    return value


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if (
                not isinstance(part, dict)
                or part.get("type") not in {"input_text", "output_text", "text"}
                or not isinstance(part.get("text"), str)
            ):
                raise ProbeError("unsupported_nontext_content")
            pieces.append(part["text"])
        return "".join(pieces)
    raise ProbeError("content_must_be_text")


def _projection(item: Json) -> tuple[Any, ...]:
    kind = item.get("type", "message")
    if kind == "message" and item.get("role") == "assistant":
        return ("message", _text(item.get("content")))
    if kind == "function_call" and not item.get("namespace"):
        return (
            "function_call",
            item.get("call_id"),
            item.get("name"),
            item.get("arguments"),
        )
    raise ProbeError("invalid_state_projection")


@dataclass(frozen=True)
class StoredTurn:
    session_id: str
    native_message: Json
    projections: list[Json]
    call_ids: dict[str, str]


@dataclass
class Session:
    key_fingerprint: str
    model: str
    carriers: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ProbeBridge:
    def __init__(
        self, *, transport: httpx.AsyncBaseTransport | None, trace_dir: Path | None
    ) -> None:
        self.transport = transport
        self.trace_dir = trace_dir
        self.sessions: dict[str, Session] = {}
        self.turns: dict[str, StoredTurn] = {}
        self.mac_key = secrets.token_bytes(32)

    def bind(self, session_id: str, key: str, model: Any) -> Session:
        if model != MODEL:
            raise ProbeError("only_gemini_3_8_flash_is_supported")
        if not session_id or len(session_id) > 256 or not key:
            raise ProbeError("session_id_and_upstream_ak_headers_required")
        fingerprint = hmac.new(self.mac_key, key.encode(), hashlib.sha256).hexdigest()
        session = self.sessions.get(session_id)
        if session is None:
            session = Session(fingerprint, MODEL)
            self.sessions[session_id] = session
        elif session.model != model or not hmac.compare_digest(
            session.key_fingerprint, fingerprint
        ):
            raise ProbeError("immutable_session_binding_mismatch", status=409)
        return session

    def _new_carrier(self, session_id: str) -> str:
        nonce = secrets.token_urlsafe(24)
        mac = hmac.new(
            self.mac_key, f"{session_id}:{MODEL}:{nonce}".encode(), hashlib.sha256
        ).hexdigest()
        return f"{CARRIER_PREFIX}{nonce}:{mac}"

    def _restore(self, carrier: Any, session_id: str) -> StoredTurn:
        if not isinstance(carrier, str) or not carrier.startswith(CARRIER_PREFIX):
            raise ProbeError("missing_or_foreign_state_carrier")
        fields = carrier[len(CARRIER_PREFIX) :].split(":")
        if len(fields) != 2:
            raise ProbeError("malformed_state_carrier")
        nonce, supplied_mac = fields
        if len(supplied_mac) != 64 or any(
            char not in "0123456789abcdef" for char in supplied_mac
        ):
            raise ProbeError("malformed_state_carrier")
        expected = hmac.new(
            self.mac_key, f"{session_id}:{MODEL}:{nonce}".encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(supplied_mac, expected):
            raise ProbeError("state_carrier_authentication_failed")
        turn = self.turns.get(carrier)
        if turn is None or turn.session_id != session_id:
            raise ProbeError("state_carrier_not_available", status=409)
        return turn

    def _messages(
        self, body: Json, session_id: str, session: Session
    ) -> tuple[list[Json], list[Json]]:
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [{"role": "user", "content": inputs}]
        if not isinstance(inputs, list):
            raise ProbeError("input_must_be_text_or_array")
        messages: list[Json] = []
        instructions = body.get("instructions")
        if instructions is not None:
            if not isinstance(instructions, str):
                raise ProbeError("instructions_must_be_string")
            if instructions:
                messages.append({"role": "system", "content": instructions})
        seen: list[str] = []
        restored: list[Json] = []
        pending_calls: dict[str, str] = {}
        index = 0
        while index < len(inputs):
            item = inputs[index]
            if not isinstance(item, dict):
                raise ProbeError("input_items_must_be_objects")
            kind = item.get("type", "message")
            if kind == "reasoning":
                if pending_calls:
                    raise ProbeError("tool_results_missing_before_next_assistant")
                carrier = item.get("encrypted_content")
                if not isinstance(carrier, str):
                    raise ProbeError("missing_or_foreign_state_carrier")
                turn = self._restore(carrier, session_id)
                if carrier in seen:
                    raise ProbeError("duplicate_state_carrier")
                seen.append(carrier)
                messages.append(copy.deepcopy(turn.native_message))
                restored.append(_digest(turn.native_message))
                for offset, expected in enumerate(turn.projections, start=1):
                    position = index + offset
                    if (
                        position >= len(inputs)
                        or not isinstance(inputs[position], dict)
                        or _projection(inputs[position]) != _projection(expected)
                    ):
                        raise ProbeError("state_projection_missing_or_modified")
                pending_calls.update(turn.call_ids)
                index += len(turn.projections) + 1
                continue
            if kind == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or call_id not in pending_calls:
                    raise ProbeError("unpaired_or_duplicate_tool_output")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": pending_calls.pop(call_id),
                        "content": _text(item.get("output")),
                    }
                )
            elif kind == "message" and item.get("role") in {
                "user",
                "system",
                "developer",
            }:
                if pending_calls:
                    raise ProbeError("tool_results_missing_before_message")
                messages.append(
                    {"role": item["role"], "content": _text(item.get("content"))}
                )
            else:
                raise ProbeError("unsupported_input_or_assistant_without_carrier")
            index += 1
        if pending_calls:
            raise ProbeError("tool_results_missing")
        if seen != session.carriers:
            raise ProbeError("prior_state_missing_or_reordered", status=409)
        if not messages:
            raise ProbeError("messages_empty")
        return messages, restored

    def prepare(
        self, body: Json, session_id: str, session: Session
    ) -> tuple[Json, list[Json]]:
        if body.get("service_tier") is not None:
            raise ProbeError("service_tier_not_supported_by_probe")
        if not isinstance(body.get("stream", False), bool):
            raise ProbeError("stream_must_be_boolean")
        if (
            body.get("previous_response_id") is not None
            or body.get("conversation") is not None
        ):
            raise ProbeError("stateful_responses_ids_unsupported_use_full_input")
        if body.get("truncation", "disabled") != "disabled":
            raise ProbeError("truncation_unsupported")
        if body.get("include") not in (None, [], ["reasoning.encrypted_content"]):
            raise ProbeError("unsupported_include")
        messages, restored = self._messages(body, session_id, session)
        upstream: Json = {"model": MODEL, "messages": messages, "stream": False}
        tools = body.get("tools", [])
        if not isinstance(tools, list):
            raise ProbeError("tools_must_be_array")
        converted = []
        for tool in tools:
            if (
                not isinstance(tool, dict)
                or tool.get("type") != "function"
                or tool.get("namespace")
            ):
                raise ProbeError("only_plain_function_tools_supported")
            if (
                not isinstance(tool.get("name"), str)
                or not tool["name"]
                or not isinstance(tool.get("parameters"), dict)
            ):
                raise ProbeError("invalid_function_tool")
            function = {
                key: copy.deepcopy(tool[key])
                for key in ("name", "description", "parameters", "strict")
                if key in tool
            }
            converted.append({"type": "function", "function": function})
        if converted:
            upstream["tools"] = converted
        choice = body.get("tool_choice")
        if choice is not None:
            if isinstance(choice, str) and choice in {"auto", "required", "none"}:
                upstream["tool_choice"] = choice
            elif (
                isinstance(choice, dict)
                and choice.get("type") == "function"
                and isinstance(choice.get("name"), str)
            ):
                upstream["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice["name"]},
                }
            else:
                raise ProbeError("unsupported_tool_choice")
        for key in ("temperature", "top_p", "parallel_tool_calls"):
            if key in body:
                upstream[key] = body[key]
        max_output_tokens = body.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ProbeError("max_output_tokens_must_be_positive_integer")
        upstream["max_tokens"] = max_output_tokens
        reasoning = body.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict) or set(reasoning) - {
                "effort",
                "summary",
            }:
                raise ProbeError("unsupported_reasoning_options")
            if reasoning.get("summary") not in (None, "none"):
                raise ProbeError("reasoning_summary_not_supported_by_probe")
            if reasoning.get("effort") is not None:
                upstream["reasoning_effort"] = reasoning["effort"]
        text = body.get("text")
        if text not in (None, {}, {"format": {"type": "text"}}):
            raise ProbeError("structured_output_not_supported_by_probe")
        allowed = {
            "model",
            "input",
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "text",
            "include",
            "store",
            "stream",
            "metadata",
            "client_metadata",
            "prompt_cache_key",
            "service_tier",
            "max_output_tokens",
            "temperature",
            "top_p",
            "user",
            "truncation",
            "previous_response_id",
            "conversation",
        }
        if set(body) - allowed:
            raise ProbeError("unsupported_top_level_fields")
        return upstream, restored

    def convert(self, native: Json, session_id: str, session: Session) -> Json:
        choices = native.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            raise ProbeError("upstream_requires_exactly_one_choice", status=502)
        choice = choices[0]
        if choice.get("finish_reason") not in {"stop", "tool_calls"}:
            raise ProbeError("upstream_incomplete_or_unknown_finish", status=502)
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ProbeError("upstream_invalid_assistant", status=502)
        carrier = self._new_carrier(session_id)
        response_id = f"resp_{secrets.token_hex(16)}"
        projections: list[Json] = []
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProbeError("upstream_nontext_output_unsupported", status=502)
        if content:
            projections.append(
                {
                    "id": f"msg_{secrets.token_hex(12)}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": content, "annotations": []}
                    ],
                }
            )
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ProbeError("upstream_tool_calls_must_be_array", status=502)
        call_ids: dict[str, str] = {}
        native_ids: set[str] = set()
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or call.get("type") != "function":
                raise ProbeError("upstream_tool_type_unsupported", status=502)
            function = call.get("function")
            native_id = call.get("id")
            if (
                not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)
                or not isinstance(native_id, str)
            ):
                raise ProbeError("upstream_tool_fields_invalid", status=502)
            if native_id in native_ids or (len(calls) > 1 and not native_id):
                raise ProbeError("ambiguous_upstream_tool_ids", status=502)
            native_ids.add(native_id)
            wire_id = native_id or f"call_{response_id[5:]}_{index}"
            call_ids[wire_id] = native_id
            projections.append(
                {
                    "id": f"fc_{secrets.token_hex(12)}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": wire_id,
                    "name": function["name"],
                    "arguments": function["arguments"],
                }
            )
        if not projections:
            raise ProbeError("upstream_has_no_text_or_tool_output", status=502)
        usage = convert_usage(native.get("usage"))
        output = [
            {
                "id": f"rs_{secrets.token_hex(12)}",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": carrier,
            },
            *projections,
        ]
        response = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": MODEL,
            "output": output,
            "usage": usage,
            "error": None,
            "incomplete_details": None,
        }
        self.turns[carrier] = StoredTurn(
            session_id, copy.deepcopy(message), copy.deepcopy(projections), call_ids
        )
        session.carriers.append(carrier)
        return response

    def trace(self, record: Json, key: str) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with (self.trace_dir / "bridge_trace.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(_redact(record, secret=key), ensure_ascii=False) + "\n"
            )


def convert_usage(usage: Any) -> Json:
    if not isinstance(usage, dict):
        raise ProbeError("upstream_usage_missing", status=502)
    counts = {}
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProbeError("upstream_usage_invalid", status=502)
        counts[field_name] = value
    reasoning = usage.get("reasoning_tokens", 0)
    prompt_details = usage.get("prompt_tokens_details", {})
    if not isinstance(prompt_details, dict):
        raise ProbeError("upstream_usage_details_invalid", status=502)
    cached = prompt_details.get("cached_tokens", 0)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (reasoning, cached)
    ):
        raise ProbeError("upstream_usage_details_invalid", status=502)
    if (
        counts["total_tokens"]
        != counts["prompt_tokens"] + counts["completion_tokens"] + reasoning
    ):
        raise ProbeError("upstream_usage_accounting_mismatch", status=502)
    return {
        "input_tokens": counts["prompt_tokens"],
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": counts["completion_tokens"] + reasoning,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": counts["total_tokens"],
    }


def response_events(response: Json) -> list[Json]:
    events: list[Json] = []

    def emit(kind: str, **fields: Any) -> None:
        events.append({"type": kind, "sequence_number": len(events), **fields})

    started = {**response, "status": "in_progress", "output": [], "usage": None}
    emit("response.created", response=started)
    emit("response.in_progress", response=started)
    for output_index, item in enumerate(response["output"]):
        kind = item["type"]
        added = {**item, "status": "in_progress"}
        if kind == "function_call":
            added["arguments"] = ""
        elif kind == "message":
            added["content"] = []
        emit("response.output_item.added", output_index=output_index, item=added)
        shared = {"item_id": item["id"], "output_index": output_index}
        if kind == "function_call":
            for start in range(0, len(item["arguments"]), 128):
                emit(
                    "response.function_call_arguments.delta",
                    **shared,
                    delta=item["arguments"][start : start + 128],
                )
            emit(
                "response.function_call_arguments.done",
                **shared,
                arguments=item["arguments"],
            )
        elif kind == "message":
            part = item["content"][0]
            emit(
                "response.content_part.added",
                **shared,
                content_index=0,
                part={**part, "text": ""},
            )
            for start in range(0, len(part["text"]), 128):
                emit(
                    "response.output_text.delta",
                    **shared,
                    content_index=0,
                    delta=part["text"][start : start + 128],
                )
            emit(
                "response.output_text.done",
                **shared,
                content_index=0,
                text=part["text"],
            )
            emit("response.content_part.done", **shared, content_index=0, part=part)
        emit("response.output_item.done", output_index=output_index, item=item)
    emit("response.completed", response=response)
    return events


def create_app(
    *, transport: httpx.AsyncBaseTransport | None = None, trace_dir: Path | None = None
) -> FastAPI:
    """Build a probe app. MockTransport is supported; upstream URL is immutable."""
    configured_dir = os.getenv("PROBE_TRACE_DIR")
    bridge = ProbeBridge(
        transport=transport,
        trace_dir=trace_dir or (Path(configured_dir) if configured_dir else None),
    )
    application = FastAPI(title="Isolated Gemini Responses bridge probe")
    application.state.bridge = bridge

    @application.get("/health")
    async def health() -> Json:
        return {
            "status": "healthy",
            "model": MODEL,
            "state_storage": "process_memory",
            "upstream_verified": False,
        }

    @application.post("/v1/responses")
    async def respond(request: Request):
        key = request.headers.get("X-Codex-Upstream-AK", "")
        session_id = request.headers.get("X-Codex-Session-ID", "")
        request_id = secrets.token_hex(16)
        started = time.monotonic()
        record: Json = {
            "request_id": request_id,
            "session_id": session_id,
            "model": MODEL,
        }
        try:
            try:
                body = await request.json()
            except (ValueError, UnicodeDecodeError) as exc:
                raise ProbeError("invalid_json") from exc
            if not isinstance(body, dict):
                raise ProbeError("body_must_be_object")
            record["ingress"] = body
            session = bridge.bind(session_id, key, body.get("model"))
            async with session.lock:
                upstream, restored = bridge.prepare(body, session_id, session)
                record.update(upstream_request=upstream, restored_states=restored)
                record["client_metadata_policy"] = {
                    "client_metadata": "trace_only_not_model_input",
                    "prompt_cache_key": "trace_only_extra_session_id_controls_route",
                }
                record["resolved_config"] = {
                    "max_output_tokens": upstream["max_tokens"],
                    "max_output_tokens_source": (
                        "request"
                        if "max_output_tokens" in body
                        else "probe_fixed_default"
                    ),
                    "default_max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                    "upstream_stream": False,
                    "request_timeout_seconds": 120,
                }
                async with httpx.AsyncClient(
                    transport=bridge.transport, timeout=120.0, follow_redirects=False
                ) as client:
                    result = await client.post(
                        UPSTREAM,
                        params={"ak": key},
                        headers={
                            "extra": json.dumps({"session_id": session_id}),
                            "X-TT-LOGID": request_id,
                        },
                        json=upstream,
                    )
                record["upstream_status"] = result.status_code
                if not 200 <= result.status_code < 300:
                    record["upstream_error_body"] = _digest(result.text)
                    raise ProbeError("upstream_http_error", status=502)
                try:
                    native = result.json()
                except ValueError as exc:
                    raise ProbeError("upstream_invalid_json", status=502) from exc
                if not isinstance(native, dict):
                    raise ProbeError("upstream_response_must_be_object", status=502)
                record["upstream_response"] = native
                response = bridge.convert(native, session_id, session)
                record.update(
                    downstream_response=response,
                    saved_state=_digest(native["choices"][0]["message"]),
                    raw_usage=native.get("usage"),
                    elapsed_seconds=time.monotonic() - started,
                )
                bridge.trace(record, key)
            if body.get("stream", False):

                async def stream():
                    for event in response_events(response):
                        yield b"event: " + event[
                            "type"
                        ].encode() + b"\ndata: " + _json_bytes(event) + b"\n\n"
                        await asyncio.sleep(0)

                return StreamingResponse(stream(), media_type="text/event-stream")
            return JSONResponse(response)
        except httpx.HTTPError as exc:
            # Never stringify an HTTP exception: its URL contains the upstream AK.
            record.update(
                error="upstream_transport_error",
                error_type=type(exc).__name__,
                elapsed_seconds=time.monotonic() - started,
            )
            bridge.trace(record, key)
            return JSONResponse(
                {
                    "error": {
                        "type": "probe_error",
                        "code": "upstream_transport_error",
                        "message": "upstream_transport_error",
                    }
                },
                status_code=502,
            )
        except ProbeError as exc:
            record.update(error=exc.code, elapsed_seconds=time.monotonic() - started)
            bridge.trace(record, key)
            return JSONResponse(
                {
                    "error": {
                        "type": "probe_error",
                        "code": exc.code,
                        "message": exc.code,
                    }
                },
                status_code=exc.status,
            )

    return application


app = create_app()
