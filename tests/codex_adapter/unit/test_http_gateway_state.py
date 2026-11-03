"""Gateway lifecycle contracts with isolated codec and state test doubles."""

import logging
import sys
from pathlib import Path
from traceback import format_exception
from types import ModuleType
from typing import Any

import httpx
import pytest

from codex_adapter.application.gateway import BufferedReply, Gateway, GatewayRequest
from codex_adapter.application.requests import RequestEnvelope
from codex_adapter.config import AdapterSettings, ServerSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.models import (
    Credential,
    ModelCapabilities,
    RegisteredModel,
    Registry,
)
from codex_adapter.domain.state import (
    Binding,
    BoundRoute,
    Completion,
    PreparedGemini,
    Replay,
    RequestLease,
    SavedTurn,
    StateKey,
    StateSettings,
)


def model() -> RegisteredModel:
    return RegisteredModel(
        "gemini-test",
        "gemini_chat",
        "https://example.invalid/chat",
        (Credential("key", "v1", "test-secret"),),
        ModelCapabilities(reasoning_efforts=(), default_reasoning_effort=None),
    )


def binding() -> Binding:
    return Binding(
        "binding",
        "local",
        "session-1234",
        "gemini-test",
        "gemini_chat",
        "gemini_chat:1",
        "https://example.invalid/chat",
        "key",
        "v1",
        "fingerprint",
        "key",
        1.0,
        9999999999.0,
    )


def lease() -> RequestLease:
    return RequestLease(
        binding(), "request", "digest", "idem", "carrier", "response", 1, None, 42.0
    )


def completion(raw_usage: bytes | None = None) -> Completion:
    turn = SavedTurn("carrier", b"{}", (), (), 0, "prefix", "options")
    return Completion(b'{"id":"response"}', turn, raw_usage)


def config() -> AdapterSettings:
    return AdapterSettings(
        Registry((model(),)),
        StateSettings(Path("/unused"), (StateKey("key", b"x" * 32),), "key"),
        ServerSettings(),
    )


def request() -> GatewayRequest:
    raw = b'{"model":"gemini-test","input":[]}'
    return GatewayRequest(
        RequestEnvelope("gemini-test", False, raw),
        "session-1234",
        "request",
        idempotency_key="idem",
    )


class RecordingStore:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.replay: Replay | None = None
        self.commit_error: Exception | None = None

    def initialize(self) -> None:
        pass

    def ready(self) -> None:
        pass

    def close(self) -> None:
        pass

    def bind(
        self, session_id: str, model: RegisteredModel, *, principal: str = "local"
    ) -> BoundRoute:
        self.events.append("bind")
        return BoundRoute(binding(), model, model.credentials[0])

    def reserve(
        self,
        binding: Binding,
        request_digest: str,
        *,
        previous_carrier: str | None,
        idempotency_key: str | None = None,
    ) -> RequestLease | Replay:
        self.events.append("reserve")
        return self.replay or lease()

    def turns(self, binding: Binding) -> tuple[SavedTurn, ...]:
        self.events.append("turns")
        return ()

    def reject(self, lease: RequestLease) -> None:
        self.events.append("reject")

    def indeterminate(self, lease: RequestLease) -> None:
        self.events.append("indeterminate")

    def commit(self, lease: RequestLease, completion: Completion) -> None:
        self.events.append("commit")
        if self.commit_error is not None:
            raise self.commit_error


class FakeTransport:
    def __init__(self, status: int = 200, body: bytes = b"{}") -> None:
        self.status = status
        self.body = body
        self.events: list[str] = []
        self.open_error: Exception | None = None
        self.read_error: Exception | None = None

    async def open(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.events.append("open")
        if self.open_error is not None:
            raise self.open_error
        return httpx.Response(self.status)

    async def read(self, response: httpx.Response) -> bytes:
        self.events.append("read")
        if self.read_error is not None:
            raise self.read_error
        return self.body

    async def read_decoded(self, response: httpx.Response) -> bytes:
        return await self.read(response)

    async def close(self, response: httpx.Response) -> None:
        self.events.append("close")
        await response.aclose()


def codec(
    monkeypatch: pytest.MonkeyPatch, result: Completion | None = None
) -> ModuleType:
    module = ModuleType("codex_adapter.codecs.gemini_chat")
    monkeypatch.setattr(module, "last_carrier", lambda body: None, raising=False)
    monkeypatch.setattr(
        module,
        "prepare_request",
        lambda body, selected, turns: PreparedGemini(b"{}", (), "options"),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "convert_response",
        lambda body, prepared, reserved, selected: result or completion(),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.asyncio
async def test_validation_after_reservation_rejects_without_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = codec(monkeypatch)

    def invalid(*args: Any) -> None:
        raise AdapterError("invalid_request", "Invalid request")

    monkeypatch.setattr(module, "prepare_request", invalid)
    store, transport = RecordingStore(), FakeTransport()
    with pytest.raises(AdapterError, match="Invalid request"):
        await Gateway(config(), store, transport).forward(request())
    assert store.events == ["bind", "reserve", "turns", "reject"]
    assert transport.events == []


@pytest.mark.asyncio
async def test_programming_error_after_reservation_is_rejected_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = codec(monkeypatch)

    def invalid(*args: Any) -> None:
        raise RuntimeError("programming error")

    monkeypatch.setattr(module, "prepare_request", invalid)
    store, transport = RecordingStore(), FakeTransport()
    with pytest.raises(RuntimeError, match="programming error"):
        await Gateway(config(), store, transport).forward(request())
    assert store.events == ["bind", "reserve", "turns", "reject"]
    assert transport.events == []


@pytest.mark.asyncio
async def test_explicit_http_rejection_does_not_read_body_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec(monkeypatch)
    store, transport = RecordingStore(), FakeTransport(status=429)
    transport.read_error = RuntimeError("body must not be read")
    with pytest.raises(AdapterError) as raised:
        await Gateway(config(), store, transport).forward(request())
    assert raised.value.code == "upstream_rejected"
    assert raised.value.status_code == 429
    assert store.events == ["bind", "reserve", "turns", "reject"]
    assert transport.events == ["open", "close"]


@pytest.mark.asyncio
async def test_invalid_upstream_json_is_safe_502_and_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec(monkeypatch)
    store, transport = RecordingStore(), FakeTransport(
        body=b'{"private":"sensitive-marker",'
    )
    with pytest.raises(AdapterError) as raised:
        await Gateway(config(), store, transport).forward(request())
    assert raised.value.status_code == 502
    assert raised.value.code == "invalid_upstream_response"
    assert raised.value.__context__ is None
    assert "sensitive-marker" not in "".join(format_exception(raised.value))
    assert store.events == ["bind", "reserve", "turns", "indeterminate"]


@pytest.mark.asyncio
async def test_codec_upstream_error_is_safe_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = codec(monkeypatch)

    def invalid(*args: Any) -> None:
        raise AdapterError("invalid_json", "sensitive-marker")

    monkeypatch.setattr(module, "convert_response", invalid)
    store, transport = RecordingStore(), FakeTransport()
    with pytest.raises(AdapterError) as raised:
        await Gateway(config(), store, transport).forward(request())
    assert raised.value.status_code == 502
    assert raised.value.code == "invalid_upstream_response"
    assert "sensitive-marker" not in str(raised.value)
    assert store.events[-1] == "indeterminate"


@pytest.mark.asyncio
async def test_network_failure_marks_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec(monkeypatch)
    store, transport = RecordingStore(), FakeTransport()
    transport.open_error = AdapterError(
        "upstream_timeout", "timed out", status_code=504
    )
    with pytest.raises(AdapterError) as raised:
        await Gateway(config(), store, transport).forward(request())
    assert raised.value.code == "upstream_timeout"
    assert store.events == ["bind", "reserve", "turns", "indeterminate"]


@pytest.mark.asyncio
async def test_success_is_committed_before_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec(monkeypatch)
    store, transport = RecordingStore(), FakeTransport()
    result = await Gateway(config(), store, transport).forward(request())
    assert isinstance(result, BufferedReply) and result.body == b'{"id":"response"}'
    assert store.events == ["bind", "reserve", "turns", "commit"]


@pytest.mark.asyncio
async def test_replay_never_recontacts_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec(monkeypatch)
    store, transport = RecordingStore(), FakeTransport()
    store.replay = Replay(b'{"id":"old"}', None)
    result = await Gateway(config(), store, transport).forward(request())
    assert isinstance(result, BufferedReply) and result.replayed
    assert result.body == b'{"id":"old"}'
    assert store.events == ["bind", "reserve"] and transport.events == []


@pytest.mark.asyncio
async def test_commit_failure_is_indeterminate_and_not_faked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec(monkeypatch)
    store, transport = RecordingStore(), FakeTransport()
    store.commit_error = RuntimeError("commit error")
    with pytest.raises(RuntimeError, match="commit error"):
        await Gateway(config(), store, transport).forward(request())
    assert store.events == ["bind", "reserve", "turns", "commit", "indeterminate"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_usage",
    [
        b"not-json",
        b'{"prompt_tokens":"sensitive-marker","prompt_tokens_details":{"cached_tokens":"sensitive-marker"}}',
    ],
)
async def test_observability_never_changes_committed_result_or_logs_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw_usage: bytes
) -> None:
    codec(monkeypatch, completion(raw_usage))
    store, transport = RecordingStore(), FakeTransport()
    with caplog.at_level(logging.INFO):
        result = await Gateway(config(), store, transport).forward(request())
    assert isinstance(result, BufferedReply)
    assert store.events[-1] == "commit"
    assert "sensitive-marker" not in caplog.text
