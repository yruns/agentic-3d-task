"""The public gateway enforces routing before touching an upstream."""

import asyncio
import gzip
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from codex_adapter.config import AdapterSettings, ServerSettings
from codex_adapter.domain.models import (
    Credential,
    ModelCapabilities,
    RegisteredModel,
    Registry,
)
from codex_adapter.domain.state import StateKey, StateSettings
from codex_adapter.interfaces.app import create_app


def settings(tmp_path: Path) -> AdapterSettings:
    credential = Credential("fake-key", "v1", "never-log-test-secret")
    gpt = RegisteredModel(
        "gpt-test",
        "responses",
        "https://upstream.example/responses",
        (credential,),
        ModelCapabilities(),
    )
    gemini = RegisteredModel(
        "gemini-test",
        "gemini_chat",
        "https://upstream.example/chat",
        (credential,),
        ModelCapabilities(
            reasoning_efforts=(), default_reasoning_effort=None, reasoning_summary=False
        ),
    )
    return AdapterSettings(
        Registry((gpt, gemini)),
        StateSettings(tmp_path / "state.sqlite", (StateKey("v1", b"k" * 32),), "v1"),
        ServerSettings(),
    )


def client_for(
    tmp_path: Path,
    handler: httpx.MockTransport,
    *,
    auth_token: str | None = None,
    server: ServerSettings | None = None,
) -> TestClient:
    defaults = settings(tmp_path)
    configured = replace(
        defaults, auth_token=auth_token, server=server or defaults.server
    )
    return TestClient(
        create_app(configured, upstream_transport=handler, initialize_state=True)
    )


def test_missing_session_and_unknown_model_never_call_upstream(tmp_path: Path) -> None:
    calls: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    with client_for(tmp_path, httpx.MockTransport(upstream)) as client:
        missing = client.post(
            "/v1/responses", json={"model": "gpt-test", "input": "hi"}
        )
        unknown = client.post(
            "/v1/responses",
            json={"model": "not-registered", "input": "hi"},
            headers={"X-Codex-Session-ID": "session-1234"},
        )
    assert missing.status_code == 400
    assert unknown.status_code == 404
    assert calls == []


def test_native_json_and_sse_payload_are_opaque(tmp_path: Path) -> None:
    sent: list[bytes] = []
    raw = b'{ "model":"gpt-test","stream":true,"input":[{"type":"reasoning","encrypted_content":"opaque+/="}],"future":1e400}'
    sse = b'event: response.created\ndata: {"type":"response.created","future":"opaque"}\n\n'

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"}
        )

    with client_for(tmp_path, httpx.MockTransport(upstream)) as client:
        response = client.post(
            "/v1/responses",
            content=raw,
            headers={"X-Codex-Session-ID": "session-native-123"},
        )
    assert response.status_code == 200
    assert response.content == sse
    assert sent == [raw]


def test_health_and_models_never_expose_upstream_identity(tmp_path: Path) -> None:
    with client_for(
        tmp_path, httpx.MockTransport(lambda request: httpx.Response(500))
    ) as client:
        health = client.get("/health")
        model = client.get("/v1/models/gemini-test/capabilities")
    assert health.json()["status"] == "ready"
    assert health.json()["models"] == ["gpt-test", "gemini-test"]
    assert model.json()["capabilities"]["compact"] is False
    assert "never-log-test-secret" not in health.text + model.text
    assert "upstream.example" not in health.text + model.text


def test_gemini_compact_is_explicitly_rejected(tmp_path: Path) -> None:
    with client_for(
        tmp_path,
        httpx.MockTransport(lambda request: pytest.fail("No upstream call allowed")),
    ) as client:
        response = client.post(
            "/v1/responses/compact",
            json={"model": "gemini-test", "input": []},
            headers={"X-Codex-Session-ID": "session-compact-123"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_capability"


def test_authentication_and_body_limits(tmp_path: Path) -> None:
    with client_for(
        tmp_path,
        httpx.MockTransport(lambda request: pytest.fail("No upstream call allowed")),
        auth_token="local-auth-token",
        server=ServerSettings(max_request_bytes=8),
    ) as client:
        denied = client.post("/v1/responses", json={"model": "gpt-test"})
        too_big = client.post(
            "/v1/responses",
            content=b"123456789",
            headers={"Authorization": "Bearer local-auth-token"},
        )
    assert denied.status_code == 401
    assert too_big.status_code == 413


def test_health_and_models_share_rate_limit(tmp_path: Path) -> None:
    with client_for(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(500)),
        server=ServerSettings(requests_per_minute=2),
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
        assert client.get("/v1/models/gemini-test/capabilities").status_code == 429


class RawStream(httpx.AsyncByteStream):
    def __init__(self, value: bytes) -> None:
        self.value = value

    async def __aiter__(self):
        yield self.value

    async def aclose(self) -> None:
        pass


def test_native_gzip_stream_preserves_wire_bytes_and_headers(tmp_path: Path) -> None:
    raw = gzip.compress(
        b'event: response.created\ndata: {"type":"response.created"}\n\n'
    )
    handler = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            stream=RawStream(raw),
            headers={"content-encoding": "gzip", "content-type": "text/event-stream"},
        )
    )
    with client_for(tmp_path, handler) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "gpt-test", "stream": True},
            headers={"X-Codex-Session-ID": "session-gzip-123"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-encoding"] == "gzip"
            assert b"".join(response.iter_raw()) == raw


@pytest.mark.asyncio
async def test_cancelled_native_request_cancels_inflight_upstream(
    tmp_path: Path,
) -> None:
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200, content=b"{}")

    configured = replace(
        settings(tmp_path), server=ServerSettings(upstream_timeout_seconds=1)
    )
    app = create_app(
        configured,
        upstream_transport=httpx.MockTransport(handler),
        initialize_state=True,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://adapter"
        ) as client:
            task = asyncio.create_task(
                client.post(
                    "/v1/responses",
                    json={"model": "gpt-test"},
                    headers={"X-Codex-Session-ID": "session-native-123"},
                )
            )
            try:
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 0.15)
                assert cancelled.is_set()
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)


async def _wait_until_healthy(client: httpx.AsyncClient) -> None:
    for _ in range(20):
        if (await client.get("/health")).status_code == 200:
            return
        await asyncio.sleep(0.01)
    pytest.fail("Admission was not released after background completion")


@pytest.mark.asyncio
async def test_cancelled_gemini_keeps_admission_until_background_finishes(
    tmp_path: Path,
) -> None:
    from codex_adapter.application.gateway import BufferedReply

    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class BlockingGateway:
        async def forward(self, call: object) -> BufferedReply:
            started.set()
            await release.wait()
            finished.set()
            return BufferedReply(b"{}")

    configured = replace(
        settings(tmp_path), server=ServerSettings(max_concurrent_requests=1)
    )
    app = create_app(
        configured,
        upstream_transport=httpx.MockTransport(lambda req: httpx.Response(500)),
        initialize_state=True,
    )
    async with app.router.lifespan_context(app):
        app.state.gateway = BlockingGateway()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://adapter"
        ) as client:
            task = asyncio.create_task(
                client.post(
                    "/v1/responses",
                    json={"model": "gemini-test"},
                    headers={"X-Codex-Session-ID": "session-gemini-123"},
                )
            )
            try:
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 0.15)
                assert (await client.get("/health")).status_code == 503
                release.set()
                await asyncio.wait_for(finished.wait(), 1)
                await _wait_until_healthy(client)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
