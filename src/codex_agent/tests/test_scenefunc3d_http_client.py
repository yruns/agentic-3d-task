"""Tests for local SceneFunc3D JSON HTTP client/server helpers."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypeAlias

import pytest
from pydantic import BaseModel, ConfigDict

from codex_agent.scenefunc3d.backends.http_client import post_json
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.tools.models import ToolInputError

JsonObject: TypeAlias = dict[str, object]


class _EchoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    message: str


class _InvalidJsonHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        body = b'{"ok": '
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress noisy access logs in tests."""


def test_post_json_round_trip() -> None:
    server = _start_json_server(
        {
            "/echo": lambda payload: {
                "ok": True,
                "message": str(payload["message"]),
            }
        }
    )
    try:
        response = post_json(
            f"http://127.0.0.1:{server.server_port}/echo",
            payload={"message": "hello"},
            response_model=_EchoResponse,
            timeout_seconds=2.0,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.ok is True
    assert response.message == "hello"


def test_json_server_get_health_round_trip() -> None:
    server = _start_json_server(
        {
            "/health": lambda payload: {
                "ok": True,
                "message": "healthy",
            }
        }
    )
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/health", timeout=2.0
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert payload == {"ok": True, "message": "healthy"}


def test_json_server_rejects_non_object_post_body() -> None:
    server = _start_json_server({"/echo": lambda payload: payload})
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/echo",
        data=b'["not", "an", "object"]',
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=2.0)
        response_body = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 500
    assert response_body == {"error": "request body must be a JSON object"}


def test_json_server_route_value_error_does_not_echo_secret() -> None:
    secret_message = "token=secret-token-value"
    server = _start_json_server(
        {"/fail": lambda payload: _raise_value_error(secret_message)}
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/fail",
        data=b'{"message": "hello"}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=2.0)
        response_body = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 500
    assert response_body == {"error": "route_exception"}
    assert secret_message not in json.dumps(response_body)


def test_post_json_http_500_is_recoverable_without_body_echo() -> None:
    secret_body = "backend-token-secret"
    server = _start_json_server(
        {"/fail": lambda payload: _raise_runtime_error(secret_body)}
    )
    try:
        with pytest.raises(ToolInputError) as exc_info:
            post_json(
                f"http://127.0.0.1:{server.server_port}/fail",
                payload={"message": "hello"},
                response_model=_EchoResponse,
                timeout_seconds=2.0,
            )
    finally:
        server.shutdown()
        server.server_close()

    message = str(exc_info.value)
    assert "HTTP 500" in message
    assert "http_error" in message
    assert secret_body not in message


def test_post_json_schema_mismatch_is_recoverable() -> None:
    server = _start_json_server({"/bad": lambda payload: {"ok": "yes", "extra": 1}})
    try:
        with pytest.raises(ToolInputError, match="invalid_response_schema"):
            post_json(
                f"http://127.0.0.1:{server.server_port}/bad",
                payload={"message": "hello"},
                response_model=_EchoResponse,
                timeout_seconds=2.0,
            )
    finally:
        server.shutdown()
        server.server_close()


def test_post_json_invalid_json_is_recoverable() -> None:
    server = _start_server(_InvalidJsonHandler)
    try:
        with pytest.raises(ToolInputError, match="invalid_response_json"):
            post_json(
                f"http://127.0.0.1:{server.server_port}/invalid",
                payload={"message": "hello"},
                response_model=_EchoResponse,
                timeout_seconds=2.0,
            )
    finally:
        server.shutdown()
        server.server_close()


def test_post_json_connection_failure_is_recoverable_and_sanitized() -> None:
    server = _start_server(_InvalidJsonHandler)
    server_port = server.server_port
    server.shutdown()
    server.server_close()
    unsafe_url = (
        f"http://user:secret-password@127.0.0.1:{server_port}"
        "/echo?token=secret-token-value"
    )

    with pytest.raises(ToolInputError) as exc_info:
        post_json(
            unsafe_url,
            payload={"message": "hello"},
            response_model=_EchoResponse,
            timeout_seconds=0.01,
        )

    message = str(exc_info.value)
    assert "sidecar_request_failed" in message or "sidecar_request_timeout" in message
    assert "127.0.0.1" in message
    assert "secret-password" not in message
    assert "secret-token-value" not in message
    assert "?token=" not in message


def _raise_runtime_error(message: str) -> JsonObject:
    raise RuntimeError(message)


def _raise_value_error(message: str) -> JsonObject:
    raise ValueError(message)


def _start_json_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    return _start_server(make_json_handler(routes))


def _start_server(handler_type: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
