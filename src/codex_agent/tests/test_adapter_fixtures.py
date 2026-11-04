"""Reusable loopback-only adapter contract and real-SDK response fixtures."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote


@dataclass(frozen=True)
class AdapterRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    body: Mapping[str, object]

    def header(self, name: str) -> str:
        return next(
            (
                value
                for key, value in self.headers.items()
                if key.lower() == name.lower()
            ),
            "",
        )


@dataclass(frozen=True)
class AdapterReply:
    body: bytes
    status: int = 200
    content_type: str = "application/json"


@dataclass(frozen=True)
class AdapterTestServer:
    server: ThreadingHTTPServer
    requests: list[AdapterRequest]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"


def adapter_profile(
    *, model: str = "gemini-test", protocol: str = "gemini_chat", **capabilities: object
) -> dict[str, object]:
    is_responses = protocol == "responses"
    return {
        "model": model,
        "protocol": protocol,
        "codec_version": f"{protocol}:test-v1",
        "capabilities": {
            "structured_output": True,
            "images": True,
            "function_tools": True,
            "namespace_tools": is_responses,
            "parallel_tools": True,
            "reasoning_summary": is_responses,
            "reasoning_efforts": (
                ["minimal", "low", "medium", "high"] if is_responses else []
            ),
            "default_reasoning_effort": "medium" if is_responses else None,
            "context_window": 1000000,
            "compact": is_responses,
            "streaming": "native" if is_responses else "buffered_sse",
            **capabilities,
        },
    }


def _json_reply(payload: object, *, status: int = 200) -> AdapterReply:
    return AdapterReply(json.dumps(payload).encode(), status=status)


def _handler(
    dispatch: Callable[[AdapterRequest], AdapterReply], requests: list[AdapterRequest]
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            decoded = json.loads(raw) if raw else {}
            if not isinstance(decoded, dict):
                raise ValueError("Expected an object request body")
            call = AdapterRequest(method, self.path, dict(self.headers), decoded)
            requests.append(call)
            reply = dispatch(call)
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.content_type)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            try:
                self.wfile.write(reply.body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@contextmanager
def adapter_test_server(
    profile: Mapping[str, object],
    *,
    health: object | None = None,
    respond: Callable[[AdapterRequest], AdapterReply] | None = None,
) -> Iterator[AdapterTestServer]:
    """Expose only fake loopback endpoints, never an upstream model service."""
    requests: list[AdapterRequest] = []
    ready = (
        health
        if health is not None
        else {
            "status": "ready",
            "service": "codex-adapter",
            "version": "1.0.0",
            "schema_version": 1,
            "models": [profile["model"]],
        }
    )
    capability_path = f"/v1/models/{quote(str(profile['model']), safe='')}/capabilities"

    def dispatch(call: AdapterRequest) -> AdapterReply:
        if call.method == "GET" and call.path == "/health":
            return _json_reply(ready)
        if call.method == "GET" and call.path == capability_path:
            return _json_reply(dict(profile))
        if respond is not None:
            return respond(call)
        return _json_reply({"error": "unexpected fake request"}, status=404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(dispatch, requests))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield AdapterTestServer(server, requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def sdk_response_events(
    output: Sequence[Mapping[str, object]], *, response_id: str, input_tokens: int = 100
) -> AdapterReply:
    """Encode a complete Responses SSE stream understood by pinned Codex core."""
    response = {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "model": "fake-model",
        "status": "completed",
        "output": list(output),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": 10,
            "total_tokens": input_tokens + 10,
        },
    }
    events: list[dict[str, object]] = [
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        }
    ]
    for index, item in enumerate(output):
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": dict(item),
                },
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": dict(item),
                },
            ]
        )
    events.append({"type": "response.completed", "response": response})
    body = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()
    return AdapterReply(body, content_type="text/event-stream")


def sdk_message(text: str) -> dict[str, object]:
    return {
        "type": "message",
        "id": "msg-fake",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
