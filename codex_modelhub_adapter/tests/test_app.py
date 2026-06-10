from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from adapter.app import app


class _FakeUpstreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes = b'{"ok": true}',
        content_type: str = "application/json",
    ) -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self._body = body

    async def aread(self) -> bytes:
        return self._body

    async def aiter_raw(self):
        yield self._body

    async def aiter_text(self):
        yield self._body.decode("utf-8")


class _FakeStreamContext:
    def __init__(self, response: _FakeUpstreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _FakeUpstreamResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeAsyncClient:
    calls: list[dict[str, Any]] = []
    response: _FakeUpstreamResponse | list[_FakeUpstreamResponse] = _FakeUpstreamResponse()

    def __init__(self, *args, **kwargs) -> None:
        self.init_kwargs = kwargs

    def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json})
        response = self.response.pop(0) if isinstance(self.response, list) else self.response
        return _FakeStreamContext(response)

    async def aclose(self) -> None:
        return None


@contextmanager
def _env(values: dict[str, str], *, remove: tuple[str, ...] = ()):
    old_values = {key: os.environ.get(key) for key in (*values, *remove)}
    try:
        for key in remove:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class AppTest(unittest.TestCase):
    def test_responses_route_uses_office_chat_adapter(self):
        _FakeAsyncClient.calls = []
        _FakeAsyncClient.response = _FakeUpstreamResponse(
            body=json.dumps(
                {
                    "model": "gpt-5.5-2026-04-24",
                    "choices": [{"message": {"role": "assistant", "content": "2"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                }
            ).encode()
        )

        with _env(
            {
                "AIDP_GPT_AK": "ak-1",
                "AIDP_CODEX_PROXY_UPSTREAM_ENV": "office",
                "AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS": "gpt-5.5*",
            },
            remove=("MODELHUB_AK", "MODELHUB_URL"),
        ):
            with patch("adapter.app.httpx.AsyncClient", _FakeAsyncClient):
                response = TestClient(app).post(
                    "/v1/responses",
                    json={"model": "gpt-5.5-2026-04-24", "input": "What is 1+1?", "stream": False},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["output_text"], "2")
        self.assertEqual(len(_FakeAsyncClient.calls), 1)
        call = _FakeAsyncClient.calls[0]
        self.assertEqual(
            call["url"],
            "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl?ak=ak-1",
        )
        self.assertEqual(call["json"]["messages"], [{"role": "user", "content": "What is 1+1?"}])
        self.assertEqual(call["json"]["max_tokens"], 65536)

    def test_compact_route_returns_local_response_compaction(self):
        response = TestClient(app).post(
            "/v1/responses/compact",
            json={
                "model": "gpt-5.5-2026-04-24",
                "input": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "response.compaction")
        self.assertEqual(body["output"][0]["content"][0]["type"], "input_text")

    def test_health_reports_toml_upstreams_without_exposing_ak(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "modelhub_upstreams.toml"
            toml_path.write_text(
                """
[[upstreams]]
url = "https://modelhub-a.example.test/api/modelhub/online"
model_name = "gpt-5.5-2026-04-24"
ak = "ak-secret-from-toml"
weight = 2
""".strip(),
                encoding="utf-8",
            )

            with _env(
                {"AIDP_MODELHUB_UPSTREAMS_TOML": str(toml_path)},
                remove=(
                    "AIDP_GPT_AK",
                    "AIDP_MODELHUB_AK",
                    "MODELHUB_AK",
                    "CASE_REVIEW_LLM_AK",
                    "AIDP_MODELHUB_AK_POOL",
                ),
            ):
                response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertTrue(payload["config"]["has_upstream_ak"])
        self.assertEqual(payload["config"]["modelhub_toml_upstream_count"], 1)
        self.assertEqual(payload["config"]["modelhub_toml_models"], ["gpt-5.5-2026-04-24"])
        self.assertNotIn("ak-secret-from-toml", json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
