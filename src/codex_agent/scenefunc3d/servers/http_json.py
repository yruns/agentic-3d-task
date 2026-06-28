"""Small JSON HTTP server helpers for local SceneFunc3D sidecars."""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]
JsonRoute: TypeAlias = Callable[[JsonObject], JsonObject]


def make_json_handler(routes: dict[str, JsonRoute]) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class for fixed local JSON POST routes."""

    class JsonHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            if self.path != "/health":
                self._write_json(404, {"error": "unknown_route"})
                return
            route = routes.get("/health")
            if route is None:
                self._write_json(404, {"error": "unknown_route"})
                return
            try:
                response_payload = route({})
                self._write_json(200, response_payload)
            except Exception:  # noqa: BLE001 - HTTP route boundary.
                self._write_json(500, {"error": "route_exception"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            route = routes.get(self.path)
            if route is None:
                self._write_json(404, {"error": "unknown_route"})
                return
            try:
                request_payload = self._read_json_object()
            except ValueError as exc:
                self._write_json(500, {"error": str(exc)})
                return
            try:
                response_payload = route(request_payload)
                self._write_json(200, response_payload)
            except Exception:  # noqa: BLE001 - HTTP route boundary.
                self._write_json(500, {"error": "route_exception"})

        def log_message(self, format: str, *args: object) -> None:
            """Suppress default access logs for sidecars and tests."""

        def _read_json_object(self) -> JsonObject:
            length_header = self.headers.get("Content-Length", "0")
            try:
                content_length = int(length_header)
            except ValueError as exc:
                raise ValueError("Content-Length must be an integer") from exc
            raw_body = self.rfile.read(content_length)
            try:
                decoded_payload: object = json.loads(raw_body.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("request body must be valid UTF-8 JSON") from exc
            except json.JSONDecodeError as exc:
                raise ValueError("request body must be valid JSON") from exc
            if not isinstance(decoded_payload, dict):
                raise ValueError("request body must be a JSON object")

            json_payload: JsonObject = {}
            for key, value in decoded_payload.items():
                if not isinstance(key, str):
                    raise ValueError("request body JSON object keys must be strings")
                json_payload[key] = value
            return json_payload

        def _write_json(self, status_code: int, payload: JsonObject) -> None:
            encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded_payload)))
            self.end_headers()
            self.wfile.write(encoded_payload)

    return JsonHandler


__all__ = ["JsonObject", "JsonRoute", "make_json_handler"]
