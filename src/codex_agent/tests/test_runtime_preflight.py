from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from codex_agent.cli.runtime_preflight import preflight_codex_runtime
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError


def test_preflight_checks_local_provider_health(tmp_path: Path) -> None:
    health_server = _start_health_server(status_code=200, body=b'{"status":"ok"}')
    codex_home = _write_codex_home(
        tmp_path,
        base_url=f"http://127.0.0.1:{health_server.server.server_port}/v1",
    )
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")
    try:
        result = preflight_codex_runtime(config, timeout_seconds=1.0)
    finally:
        health_server.close()

    assert result.status == "checked"
    assert result.model_provider == "modelhub_adapter"
    assert (
        result.health_url
        == f"http://127.0.0.1:{health_server.server.server_port}/health"
    )
    assert health_server.requested_paths == ["/health"]


def test_preflight_fails_before_codex_turn_when_local_provider_down(
    tmp_path: Path,
) -> None:
    health_server = _start_health_server(
        status_code=503, body=b'{"status":"unavailable"}'
    )
    codex_home = _write_codex_home(
        tmp_path,
        base_url=f"http://127.0.0.1:{health_server.server.server_port}/v1",
    )
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")

    try:
        with pytest.raises(
            CodexConfigError, match="Codex model provider is unavailable"
        ):
            preflight_codex_runtime(config, timeout_seconds=0.2)
    finally:
        health_server.close()


def test_preflight_skips_remote_provider_without_network_probe(tmp_path: Path) -> None:
    codex_home = _write_codex_home(
        tmp_path,
        base_url="https://example.invalid/api/modelhub/online/responses",
    )
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")

    result = preflight_codex_runtime(config, timeout_seconds=0.2)

    assert result.status == "skipped_remote"
    assert result.base_url.startswith("https://")


def test_preflight_reports_missing_provider_section(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model_provider = "modelhub_adapter"\n',
        encoding="utf-8",
    )
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")

    with pytest.raises(CodexConfigError, match="model provider config is missing"):
        preflight_codex_runtime(config, timeout_seconds=0.2)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:bad/v1",
        "http://[::1/v1",
    ],
)
def test_preflight_wraps_malformed_provider_urls(
    tmp_path: Path,
    base_url: str,
) -> None:
    codex_home = _write_codex_home(tmp_path, base_url=base_url)
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")

    with pytest.raises(CodexConfigError, match="model provider base_url is invalid"):
        preflight_codex_runtime(config, timeout_seconds=0.2)


def _write_codex_home(tmp_path: Path, *, base_url: str) -> Path:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                'model = "gpt-5.4-2026-03-05"',
                'model_provider = "modelhub_adapter"',
                "",
                "[model_providers.modelhub_adapter]",
                'name = "ModelHub local adapter"',
                f"base_url = {json.dumps(base_url)}",
                'wire_api = "responses"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return codex_home


@dataclass(frozen=True)
class _HealthServer:
    server: ThreadingHTTPServer
    requested_paths: list[str]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _start_health_server(*, status_code: int, body: bytes) -> _HealthServer:
    requested_paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requested_paths.append(self.path)
            response_status = status_code if self.path == "/health" else 404
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return _HealthServer(server=server, requested_paths=requested_paths)
