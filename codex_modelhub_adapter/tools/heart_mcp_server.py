from __future__ import annotations

import json
import sys
from typing import Any


HEART_OUTPUT = """  **   **
 **** ****
***********
 *********
  *******
   *****
    ***
     *"""


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("utf-8").strip()
        if not line:
            break
        key, _, value = line.partition(":")
        headers[key.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def _write_message(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    message_id = message.get("id")

    if method == "initialize":
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        return _result(
            message_id,
            {
                "protocolVersion": params.get("protocolVersion") or "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "heart-template-mcp", "version": "0.1.0"},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return _result(
            message_id,
            {
                "tools": [
                    {
                        "name": "heart_template",
                        "description": "Return the exact ASCII heart template for Codex SDK validation.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "Optional trace label to echo in the response.",
                                }
                            },
                            "additionalProperties": False,
                        },
                    }
                ]
            },
        )

    if method == "tools/call":
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name != "heart_template":
            return _error(message_id, -32601, f"unknown tool: {name}")
        label = arguments.get("label") or "codex-sdk-skill-tool-test"
        return _result(
            message_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": f"TRACE_LABEL={label}\nHEART_OUTPUT_START\n{HEART_OUTPUT}\nHEART_OUTPUT_END",
                    }
                ],
                "isError": False,
            },
        )

    if message_id is None:
        return None
    return _error(message_id, -32601, f"unknown method: {method}")


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break
        response = _handle(message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    main()
