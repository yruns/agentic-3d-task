"""Native wire transport is opaque and does not retry on another account."""

import asyncio
import gzip
import json
import logging
import time
from dataclasses import replace
from traceback import format_exception

import httpx
import pytest

from codex_adapter.config import ServerSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.models import ModelCapabilities, RegisteredModel, Upstream
from codex_adapter.domain.state import Binding, BoundRoute
from codex_adapter.infrastructure.http_transport import HTTPTransport


def route() -> BoundRoute:
    credential = Upstream(
        "test-key", "v1", "https://upstream.example/responses", "test-never-log-this", 1
    )
    model = RegisteredModel(
        "gpt-test",
        "responses",
        (credential,),
        ModelCapabilities(),
    )
    binding = Binding(
        "binding-1",
        "local",
        "session-unique-1",
        model.id,
        model.protocol,
        model.codec_version,
        credential.url,
        credential.id,
        credential.version,
        "fingerprint",
        1.0,
        9999999999.0,
    )
    return BoundRoute(binding, model, credential)


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["?", "#"])
async def test_compact_uses_path_before_legacy_empty_delimiters(suffix: str) -> None:
    original = route()
    upstream = replace(original.upstream, url=original.upstream.url + suffix)
    candidate = replace(
        original,
        upstream=upstream,
        model=replace(original.model, upstreams=(upstream,)),
        binding=replace(original.binding, endpoint=upstream.url),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/responses/compact"
        assert request.url.params["ak"] == upstream.secret
        return httpx.Response(200, json={"output": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HTTPTransport(client, ServerSettings())
        response = await transport.open(
            candidate, b"{}", request_id="legacy-compact", compact=True
        )
        assert await transport.read(response) == b'{"output":[]}'


@pytest.mark.asyncio
async def test_native_bytes_and_opaque_fields_are_not_reencoded() -> None:
    payload = b'{ "model":"gpt-test", "input":[{"type":"reasoning","encrypted_content":"opaque+/="}],"future":{"x":1e400}}'
    seen: list[httpx.Request] = []
    response_bytes = (
        b'{"output":[{"type":"reasoning","encrypted_content":"unchanged"}]}'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, content=response_bytes, headers={"content-type": "application/json"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HTTPTransport(client, ServerSettings())
        upstream = await transport.open(route(), payload, request_id="request-1")
        assert await transport.read(upstream) == response_bytes
    assert seen[0].content == payload
    assert seen[0].url.params["ak"] == route().upstream.secret
    assert (
        json.loads(seen[0].headers["extra"])["session_id"] == route().binding.session_id
    )


@pytest.mark.asyncio
async def test_compact_preserves_key_and_429_is_not_retried() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url.path))
        return httpx.Response(
            429,
            content=b'{"error":{"code":"rate_limit"}}',
            headers={"retry-after": "2"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HTTPTransport(client, ServerSettings())
        upstream = await transport.open(
            route(), b'{"model":"gpt-test"}', request_id="request-2", compact=True
        )
        assert upstream.status_code == 429
        assert transport.response_headers(upstream)["retry-after"] == "2"
        await transport.read(upstream)
    assert calls == ["/responses/compact"]


@pytest.mark.asyncio
async def test_transport_errors_do_not_expose_secret_or_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failure at {request.url}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AdapterError) as raised:
            await HTTPTransport(client, ServerSettings()).open(
                route(), b"{}", request_id="request-3"
            )
    assert "test-never-log-this" not in str(raised.value)
    assert "upstream.example" not in str(raised.value)
    assert raised.value.__suppress_context__


@pytest.mark.asyncio
async def test_response_limit_closes_resource() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"oversized")
        )
    ) as client:
        transport = HTTPTransport(client, ServerSettings(max_response_bytes=2))
        upstream = await transport.open(route(), b"{}", request_id="request-4")
        with pytest.raises(AdapterError, match="limit"):
            await transport.read(upstream)
        assert upstream.is_closed


class TimedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...], delay: float) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False
        self.started = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_overall_deadline_includes_open_and_read() -> None:
    stream = TimedStream((b"response",), delay=0.04)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.04)
        return httpx.Response(200, stream=stream)

    settings = ServerSettings(upstream_timeout_seconds=0.06)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HTTPTransport(client, settings)
        response = await transport.open(route(), b"{}", request_id="request")
        with pytest.raises(AdapterError) as raised:
            await transport.read(response)
        assert raised.value.code == "upstream_timeout"
        assert raised.value.status_code == 504
        assert response.is_closed and stream.closed


@pytest.mark.asyncio
async def test_trickling_stream_has_absolute_deadline() -> None:
    stream = TimedStream((b"chunk",) * 20, delay=0.02)
    settings = ServerSettings(upstream_timeout_seconds=0.07)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream)
        )
    ) as client:
        transport = HTTPTransport(client, settings)
        response = await transport.open(route(), b"{}", request_id="request")
        with pytest.raises(AdapterError) as raised:
            await transport.read(response)
        assert raised.value.code == "upstream_timeout"
        assert response.is_closed and stream.closed


@pytest.mark.asyncio
async def test_connection_open_has_absolute_deadline() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=b"{}")

    settings = ServerSettings(upstream_timeout_seconds=0.04)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AdapterError) as raised:
            await HTTPTransport(client, settings).open(
                route(), b"{}", request_id="request"
            )
        assert raised.value.code == "upstream_timeout"
        assert raised.value.status_code == 504
        assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_deadline_expires_before_body_consumption() -> None:
    stream = TimedStream((b"body",), delay=0)
    settings = ServerSettings(upstream_timeout_seconds=0.03)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream)
        )
    ) as client:
        transport = HTTPTransport(client, settings)
        response = await transport.open(route(), b"{}", request_id="request")
        await asyncio.sleep(0.05)
        assert response.is_closed and stream.closed
        with pytest.raises(AdapterError) as raised:
            await transport.read(response)
        assert raised.value.code == "upstream_timeout"
        assert stream.closed


@pytest.mark.asyncio
async def test_cancelled_response_read_closes_resource() -> None:
    stream = TimedStream((b"body",), delay=10)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream)
        )
    ) as client:
        transport = HTTPTransport(client, ServerSettings())
        response = await transport.open(route(), b"{}", request_id="request")
        task = asyncio.create_task(transport.read(response))
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.is_closed and stream.closed


@pytest.mark.asyncio
async def test_gzip_response_is_forwarded_raw_with_encoding_header() -> None:
    raw = gzip.compress(b'event: data\ndata: {"test":true}\n\n')
    stream = TimedStream((raw,), delay=0)
    headers = {
        "content-encoding": "gzip",
        "content-type": "text/event-stream",
        "content-length": str(len(raw)),
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream, headers=headers)
        )
    ) as client:
        transport = HTTPTransport(client, ServerSettings())
        response = await transport.open(route(), b"{}", request_id="request")
        forwarded_headers = transport.response_headers(response)
        assert await transport.read(response) == raw
        assert forwarded_headers["content-encoding"] == "gzip"
        assert "content-length" not in forwarded_headers


@pytest.mark.asyncio
async def test_predecoded_compressed_response_is_not_mislabeled_as_raw() -> None:
    raw = gzip.compress(b"hello")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=raw, headers={"content-encoding": "gzip"}
            )
        )
    ) as client:
        transport = HTTPTransport(client, ServerSettings())
        response = await transport.open(route(), b"{}", request_id="request")
        with pytest.raises(AdapterError) as raised:
            await transport.read(response)
        assert raised.value.status_code == 502
        assert response.is_closed


@pytest.mark.asyncio
async def test_transport_exception_has_no_sensitive_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failure at {request.url}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AdapterError) as raised:
            await HTTPTransport(client, ServerSettings()).open(
                route(), b"{}", request_id="request"
            )
    assert raised.value.__context__ is None
    assert "test-never-log-this" not in "".join(format_exception(raised.value))


@pytest.mark.asyncio
async def test_open_detects_deadline_overrun_even_after_blocking_handler() -> None:
    stream = TimedStream((b"response",), delay=0)

    async def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.04)
        return httpx.Response(200, stream=stream)

    settings = ServerSettings(upstream_timeout_seconds=0.02)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AdapterError) as raised:
            await HTTPTransport(client, settings).open(
                route(), b"{}", request_id="request"
            )
        assert raised.value.code == "upstream_timeout"
        assert stream.closed


@pytest.mark.asyncio
async def test_sensitive_network_logs_are_scoped_to_adapter_requests(
    caplog: pytest.LogCaptureFixture,
) -> None:
    network_logger = logging.getLogger("httpx")

    async def handler(request: httpx.Request) -> httpx.Response:
        network_logger.info("upstream request %s", request.url)
        return httpx.Response(200, content=b"{}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HTTPTransport(client, ServerSettings())
        with caplog.at_level(logging.INFO, logger="httpx"):
            network_logger.info("unrelated client activity")
            response = await transport.open(route(), b"{}", request_id="request")
            await transport.read(response)
    assert "unrelated client activity" in caplog.text
    assert "test-never-log-this" not in caplog.text
    assert "upstream.example" not in caplog.text


@pytest.mark.asyncio
async def test_uncooperative_response_close_is_bounded() -> None:
    class HangingClose(TimedStream):
        async def aclose(self) -> None:
            await asyncio.Event().wait()

    stream = HangingClose((b"body",), delay=0)
    settings = ServerSettings(upstream_timeout_seconds=0.03)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream)
        )
    ) as client:
        transport = HTTPTransport(client, settings)
        response = await transport.open(route(), b"{}", request_id="request")
        with pytest.raises(AdapterError):
            await asyncio.wait_for(transport.close(response), 0.2)
