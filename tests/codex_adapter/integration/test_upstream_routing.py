"""Independent upstream selection, durable bindings, and old-record compatibility."""

import sqlite3
from dataclasses import replace
from pathlib import Path
from traceback import format_exception
from typing import Literal

import httpx
import pytest

from codex_adapter.codecs.gemini_response import convert_response
from codex_adapter.config import ServerSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject
from codex_adapter.domain.models import (
    ModelCapabilities,
    RegisteredModel,
    Upstream,
    validate_upstream_url,
)
from codex_adapter.domain.state import (
    Binding,
    PreparedGemini,
    RequestLease,
    StateKey,
    StateSettings,
)
from codex_adapter.infrastructure.http_transport import HTTPTransport
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore

# Genuine authenticated schema-v2 row produced by the pre-upstream HEAD, using
# only the synthetic legacy key and secret defined by this test.
_LEGACY_BINDING_ROW = (
    "bind_80290cba363510ba45d310e510619eb7d9da3eb42be67b7f481d3abfcffd6ade",
    "local",
    "legacy-session",
    "gpt-legacy",
    "responses",
    "responses:1",
    "https://legacy.example/v1/responses",
    "legacy-upstream",
    "v7",
    "1aff390c38bdfc184ee9005b591ec3bf52ccdc47ab360436a0d4792be5bf220d",
    "legacy-state-key",
    100.0,
    86500.0,
    bytes.fromhex("cf629525bd78c6e25a504880b9a8d2e8857891331e66ab23b91672e9d7c99933"),
    bytes.fromhex(
        "4341533100106c65676163792d73746174652d6b6579cf7d00e75dcd34ad31e11b98b44b4283b6b50d63a9361c0d2bd2351cad1d26ad2c3276b8d27ab379571ec81ca278e1d7aac04a58ded4daae1dff159f4f6b343f56b37e2bdeaa215631b3604351216e9aa8e4ca6bf1cfc55cd1b65438430b556b25455bbe59f6d5987f"
    ),
)


def upstream_model(second_url: str, second_secret: str) -> RegisteredModel:
    return RegisteredModel(
        "gpt-upstreams",
        "responses",
        (
            Upstream("first", "v1", "https://first.example/v1", "fake-first", 2),
            Upstream("second", "v2", second_url, second_secret, 1),
        ),
        ModelCapabilities(),
    )


def settings(path: Path) -> StateSettings:
    return StateSettings(path, (StateKey("state", b"k" * 32),), "state")


def legacy_model() -> RegisteredModel:
    return RegisteredModel(
        "gpt-legacy",
        "responses",
        (
            Upstream(
                "legacy-upstream",
                "v7",
                "https://legacy.example/v1/responses",
                "legacy-fake-ak",
            ),
        ),
        ModelCapabilities(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_url,second_secret",
    [
        ("https://second.example/v1", "fake-second"),
        ("https://first.example/v1", "fake-second"),
        ("https://second.example/v1", "fake-first"),
    ],
)
@pytest.mark.parametrize("slot,index", [(0, 0), (1, 0), (2, 1)])
async def test_weighted_slots_and_restart_send_the_selected_url_and_secret(
    tmp_path: Path, second_url: str, second_secret: str, slot: int, index: int
) -> None:
    model = upstream_model(second_url, second_secret)
    draws: list[int] = []

    def draw(total: int) -> int:
        draws.append(total)
        return slot

    config = settings(tmp_path / "state.sqlite")
    store = SQLiteStateStore(config, clock=lambda: 100.0, draw=draw)
    store.initialize()
    first = store.bind("session", model)
    store.close()
    reopened = SQLiteStateStore(config, clock=lambda: 101.0, draw=draw)
    observed: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(
            (str(request.url.copy_with(query=None)), request.url.params["ak"])
        )
        return httpx.Response(200, content=b"{}")

    try:
        reopened.ready()
        rebound = reopened.bind("session", model)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HTTPTransport(client, ServerSettings())
            for route in (first, rebound):
                response = await transport.open(route, b"{}", request_id="request")
                await transport.read(response)
        selected = model.upstreams[index]
        assert first.upstream == rebound.upstream == selected
        assert first.binding == rebound.binding
        assert observed == [(selected.url, selected.secret)] * 2
        assert draws == [3]
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["url", "secret", "version", "reorder", "add"])
def test_unselected_changes_do_not_break_an_existing_binding(
    tmp_path: Path, change: str
) -> None:
    model = upstream_model("https://second.example/v1", "fake-second")
    store = SQLiteStateStore(
        settings(tmp_path / "state.sqlite"), clock=lambda: 100.0, draw=lambda total: 0
    )
    try:
        store.initialize()
        original = store.bind("session", model)
        first, second = model.upstreams
        changes = {
            "url": (first, replace(second, url="https://different.example/v1")),
            "secret": (first, replace(second, secret="fake-replaced")),
            "version": (first, replace(second, version="v9")),
            "reorder": (replace(second, weight=100), replace(first, weight=0)),
            "add": (
                first,
                second,
                Upstream("third", "v1", "https://third.example/v1", "fake-third"),
            ),
        }
        rebound = store.bind("session", replace(model, upstreams=changes[change]))
        assert rebound.binding == original.binding
        assert (rebound.upstream.url, rebound.upstream.secret) == (
            first.url,
            first.secret,
        )
    finally:
        store.close()


@pytest.mark.parametrize("change", ["url", "secret", "version", "revoked", "deleted"])
def test_selected_changes_fail_closed(tmp_path: Path, change: str) -> None:
    model = upstream_model("https://second.example/v1", "fake-second")
    store = SQLiteStateStore(
        settings(tmp_path / "state.sqlite"), clock=lambda: 100.0, draw=lambda total: 0
    )
    try:
        store.initialize()
        store.bind("session", model)
        first, second = model.upstreams
        changes = {
            "url": (replace(first, url="https://different.example/v1"), second),
            "secret": (replace(first, secret="fake-replaced"), second),
            "version": (replace(first, version="v9"), second),
            "revoked": (replace(first, status="revoked"), second),
            "deleted": (second,),
        }
        with pytest.raises(AdapterError) as raised:
            store.bind("session", replace(model, upstreams=changes[change]))
        assert raised.value.code == "state_conflict"
        assert raised.value.status_code == 409
        assert "fake-" not in str(raised.value)
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["id", "version", "url", "upstream", "secret"])
async def test_transport_rejects_mismatched_binding_and_upstream_before_io(
    tmp_path: Path, mismatch: str
) -> None:
    model = upstream_model("https://second.example/v1", "fake-second")
    store = SQLiteStateStore(
        settings(tmp_path / "state.sqlite"), clock=lambda: 100.0, draw=lambda total: 0
    )
    try:
        store.initialize()
        route = store.bind("session", model)
        binding = route.binding
        mismatches = {
            "id": replace(route, binding=replace(binding, credential_id="second")),
            "version": replace(
                route, binding=replace(binding, credential_version="v2")
            ),
            "url": replace(
                route, binding=replace(binding, endpoint=model.upstreams[1].url)
            ),
            "upstream": replace(route, upstream=model.upstreams[1]),
            "secret": replace(
                route, upstream=replace(route.upstream, secret="fake-other")
            ),
        }
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail("unexpected IO"))
        ) as client:
            with pytest.raises(AdapterError) as raised:
                await HTTPTransport(client, ServerSettings()).open(
                    mismatches[mismatch], b"{}", request_id="request"
                )
        assert raised.value.code == "state_conflict"
        assert "fake-" not in str(raised.value)
        assert "example" not in str(raised.value)
    finally:
        store.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example/v1",
        "https://user:password@example.com/v1",
        "https://@example.com/v1",
        "https://example.com/v1?token=secret",
        "https://example.com/v1#private",
        "https://example.com:invalid/v1",
        "https://[invalid/v1",
        "https://example.com\\@evil.example/v1",
        "https://example.com/\nprivate",
        "https://example.com/\ud800private",
        "https://\ud800.invalid/v1",
    ],
)
def test_invalid_urls_fail_at_construction_without_echoing_input(url: str) -> None:
    secret = "fake-secret"
    with pytest.raises(AdapterError) as raised:
        Upstream("id", "v1", url, secret)
    assert raised.value.code == "invalid_config"
    traceback = "".join(format_exception(raised.value))
    assert url not in traceback
    assert secret not in traceback
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "https://example.com/v1?",  # Empty legacy delimiters carry no query data.
        "https://example.com/v1#",
        "http://localhost:8080/v1",
        "http://127.0.0.2/v1",
        "http://[::1]/v1",
    ],
)
def test_valid_upstream_urls_are_preserved(url: str) -> None:
    assert validate_upstream_url(url) == url
    assert Upstream("id", "v1", url, "fake-secret").url == url


def test_authenticated_schema_two_binding_from_old_release_remains_readable(
    tmp_path: Path,
) -> None:
    config = StateSettings(
        tmp_path / "legacy.sqlite",
        (StateKey("legacy-state-key", b"k" * 32),),
        "legacy-state-key",
    )
    initialized = SQLiteStateStore(config, clock=lambda: 100.0)
    initialized.initialize()
    initialized.close()
    with sqlite3.connect(config.database) as db:
        db.execute(
            "INSERT INTO bindings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _LEGACY_BINDING_ROW,
        )
        schema = db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
        metadata = db.execute("SELECT * FROM metadata").fetchall()
    legacy = legacy_model()
    reopened = SQLiteStateStore(
        config,
        clock=lambda: 101.0,
        draw=lambda total: pytest.fail("rebound must not draw"),
    )
    try:
        reopened.ready()
        bound = reopened.bind("legacy-session", legacy)
        assert bound.upstream == legacy.upstreams[0]
        assert reopened.turns(bound.binding) == ()
    finally:
        reopened.close()
    with sqlite3.connect(config.database) as db:
        assert db.execute("SELECT schema_version FROM metadata").fetchone() == (2,)
        assert db.execute("SELECT * FROM bindings").fetchone() == _LEGACY_BINDING_ROW
        assert db.execute("SELECT * FROM metadata").fetchall() == metadata
        assert (
            db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
            == schema
        )


@pytest.mark.parametrize("change", ["id", "version", "url"])
def test_gemini_lease_must_identify_its_specific_upstream(
    change: Literal["id", "version", "url"],
) -> None:
    upstream = Upstream("one", "v1", "https://first.example/chat", "fake-first")
    second = Upstream("two", "v1", "https://second.example/chat", "fake-second")
    model = RegisteredModel(
        "gemini-test", "gemini_chat", (upstream, second), ModelCapabilities()
    )
    binding = Binding(
        "id",
        "local",
        "session",
        model.id,
        model.protocol,
        model.codec_version,
        upstream.url,
        upstream.id,
        upstream.version,
        "fingerprint",
        "key",
        1.0,
        10.0,
    )
    lease = RequestLease(
        binding, "request", "digest", "idem", "carrier", "response", 1, None, 2.0
    )
    prepared = PreparedGemini(b"{}", (), "digest")
    native: JsonObject = {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ]
    }
    assert convert_response(native, prepared, lease, model).response
    changes = {
        "id": replace(binding, credential_id="two"),
        "version": replace(binding, credential_version="v2"),
        "url": replace(binding, endpoint=second.url),
    }
    with pytest.raises(AdapterError):
        convert_response(
            native, prepared, replace(lease, binding=changes[change]), model
        )
