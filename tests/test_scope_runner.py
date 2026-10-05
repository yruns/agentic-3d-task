"""Regression checks for the SCOPE transport boundary."""

from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion

from scripts.scope import run_scope as runner
from scripts.scope.run_scope import (
    FailClosedCompletions,
    RunSettings,
    ScopeClient,
)


@dataclass
class _ApiModule:
    client: OpenAI | ScopeClient


@dataclass(frozen=True)
class _ApiImports:
    modules: dict[str, _ApiModule]

    def import_module(self, name: str) -> _ApiModule:
        return self.modules[name]


def _execute_runner_with_mock_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[httpx.Request], httpx.Response],
) -> ChatCompletion:
    """Exercise the production client's actual timeout/retry configuration."""
    configured: list[tuple[int, float]] = []

    class MockTransportClient(OpenAI):
        def __init__(
            self,
            *,
            base_url: str,
            api_key: str,
            max_retries: int,
            timeout: float,
            default_headers: Mapping[str, str],
            http_client: httpx.Client,
        ) -> None:
            configured.append((max_retries, timeout))
            super().__init__(
                base_url=base_url,
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
                http_client=http_client,
                default_headers=default_headers,
            )

    modules = {
        name: _ApiModule(
            MockTransportClient(
                base_url="http://test-bridge.invalid/v1",
                api_key="test-token",
                max_retries=0,
                timeout=1.0,
                default_headers={},
                http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            )
        )
        for name in (
            "src.eval_utils_gpt_goatbench",
            "src.potential_estimation_gpt_goal",
        )
    }
    completions: list[ChatCompletion] = []

    def execute(path_name: str, *, run_name: str) -> dict[str, object]:
        assert run_name == "__main__"
        client = modules["src.eval_utils_gpt_goatbench"].client
        assert isinstance(client, ScopeClient)
        completions.append(
            client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )
        )
        return {}

    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    (upstream_root / "run_goatbench_evaluation.py").write_text("# mocked entrypoint\n")
    cfg_file = tmp_path / "evaluation.yaml"
    cfg_file.write_text("seed: 3407\n")
    bridge_config = tmp_path / "test_bridge.json"
    bridge_config.write_text(
        json.dumps(
            {
                "endpoint": "http://test-bridge.invalid/v1",
                "bridge_token": "test-token",
                "expected_model": "gemini-3.8-flash",
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(runner, "OpenAI", MockTransportClient)

    def mock_http_transport() -> httpx.MockTransport:
        return httpx.MockTransport(respond)

    monkeypatch.setattr(httpx, "HTTPTransport", mock_http_transport)
    monkeypatch.setattr(runner, "importlib", _ApiImports(modules))
    monkeypatch.setattr(runpy, "run_path", execute)
    settings = RunSettings(upstream_root, cfg_file, bridge_config, 1, 0.0, 1.0)
    runner.run_upstream(settings)
    assert configured[-1] == (2, 240.0)
    assert len(completions) == 1
    return completions[0]


def test_api_failure_exits_instead_of_entering_upstream_fallback() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": {"message": "failed"}})

    with OpenAI(
        api_key="test-token",
        base_url="http://localhost/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        completions = FailClosedCompletions(
            client.chat.completions, expected_model="gemini-3.8-flash"
        )
        with pytest.raises(SystemExit, match="HTTP 502"):
            completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_completion_exits(content: str | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 1,
                "object": "chat.completion",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    with OpenAI(
        api_key="test-token",
        base_url="http://localhost/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        completions = FailClosedCompletions(
            client.chat.completions, expected_model="gemini-3.8-flash"
        )
        with pytest.raises(SystemExit, match="empty assistant content"):
            completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )


def test_runner_recovers_from_one_transient_gateway_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                504,
                json={"error": {"message": "temporary gateway timeout"}},
                headers={"retry-after-ms": "1"},
            )
        return httpx.Response(
            200,
            json={
                "id": "valid-retry",
                "created": 1,
                "object": "chat.completion",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "frontier 0"},
                    }
                ],
            },
        )

    completion = _execute_runner_with_mock_transport(tmp_path, monkeypatch, respond)
    assert len(requests) == 2
    assert completion.id == "valid-retry"
    assert completion.choices[0].message.content == "frontier 0"


def test_runner_still_exits_after_two_failed_transport_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            504,
            json={"error": {"message": "persistent gateway timeout"}},
            headers={"retry-after-ms": "1"},
        )

    with pytest.raises(SystemExit, match="SCOPE model transport failed: HTTP 504"):
        _execute_runner_with_mock_transport(tmp_path, monkeypatch, respond)
    assert len(requests) == 3
