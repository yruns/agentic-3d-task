"""Bounded HTTP transport; protocol payloads and streams remain opaque."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextvars import ContextVar
from urllib.parse import urlsplit

import httpx

from codex_adapter.config import ServerSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import BoundRoute

logger = logging.getLogger(__name__)
_SILENCE_NETWORK_LOGS: ContextVar[bool] = ContextVar(
    "adapter_network_logs", default=False
)
_DEADLINE_EXTENSION = "codex_adapter_deadline"
_MAX_CLOSE_SECONDS = 5.0
_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "content-encoding",
        "retry-after",
        "x-request-id",
        "x-tt-logid",
        "openai-processing-ms",
    }
)


class _SecretFilter(logging.Filter):
    """Suppress raw HTTP diagnostics only while this adapter is doing IO."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _SILENCE_NETWORK_LOGS.get()


def silence_network_diagnostics() -> None:
    """Prevent credential-bearing adapter URLs from entering network logs."""
    for name in (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
    ):
        target = logging.getLogger(name)
        if not any(isinstance(item, _SecretFilter) for item in target.filters):
            target.addFilter(_SecretFilter())


def _timeout_error() -> AdapterError:
    return AdapterError(
        "upstream_timeout",
        "Upstream request timed out; outcome may be indeterminate",
        status_code=504,
    )


def _stream_error() -> AdapterError:
    return AdapterError(
        "upstream_stream_interrupted",
        "Upstream stream interrupted; no automatic retry",
        status_code=502,
    )


class HTTPTransport:
    """Reuse an injected client; all response lifetimes have one total deadline."""

    def __init__(self, client: httpx.AsyncClient, settings: ServerSettings) -> None:
        self.client = client
        self.settings = settings
        self._close_timeout = min(
            settings.connect_timeout_seconds,
            settings.upstream_timeout_seconds,
            _MAX_CLOSE_SECONDS,
        )
        self._watchdogs: dict[httpx.Response, asyncio.Task[None]] = {}
        silence_network_diagnostics()

    async def open(
        self,
        route: BoundRoute,
        payload: bytes,
        *,
        request_id: str,
        compact: bool = False,
    ) -> httpx.Response:
        """Send once with the already bound AK; never follow redirects or fail over."""
        deadline = (
            asyncio.get_running_loop().time() + self.settings.upstream_timeout_seconds
        )
        token = _SILENCE_NETWORK_LOGS.set(True)
        error: AdapterError | None = None
        try:
            request = self._build_request(route, payload, request_id, compact)
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            response = await asyncio.wait_for(
                self.client.send(request, stream=True, follow_redirects=False),
                timeout=remaining,
            )
        except (httpx.TimeoutException, asyncio.TimeoutError):
            error = _timeout_error()
        except (httpx.HTTPError, httpx.InvalidURL):
            error = AdapterError(
                "upstream_unavailable",
                "Unable to reach upstream; outcome may be indeterminate",
                status_code=502,
            )
        finally:
            _SILENCE_NETWORK_LOGS.reset(token)
        if error is not None:
            raise error from None
        if asyncio.get_running_loop().time() >= deadline:
            try:
                await _close_response(response, self._close_timeout)
            except AdapterError:
                logger.error("Unable to close an expired upstream response")
            raise _timeout_error() from None
        response.extensions = {**response.extensions, _DEADLINE_EXTENSION: deadline}
        self._watch(response, deadline)
        return response

    def _build_request(
        self, route: BoundRoute, payload: bytes, request_id: str, compact: bool
    ) -> httpx.Request:
        binding, upstream = route.binding, route.upstream
        if (binding.credential_id, binding.credential_version, binding.endpoint) != (
            upstream.id,
            upstream.version,
            upstream.url,
        ) or upstream not in route.model.upstreams:
            raise AdapterError(
                "state_conflict",
                "Request does not match the durable session state",
                status_code=409,
            )
        endpoint = binding.endpoint
        if compact:
            parts = urlsplit(endpoint)
            endpoint = parts._replace(path=parts.path.rstrip("/") + "/compact").geturl()
        headers = {
            "content-type": "application/json",
            "accept-encoding": "identity",
            "X-TT-LOGID": request_id,
            "extra": json.dumps(
                {"session_id": route.binding.session_id}, separators=(",", ":")
            ),
        }
        timeout = httpx.Timeout(
            self.settings.upstream_timeout_seconds,
            connect=self.settings.connect_timeout_seconds,
        )
        return self.client.build_request(
            "POST",
            endpoint,
            params={"ak": route.upstream.secret},
            content=payload,
            headers=headers,
            timeout=timeout,
        )

    def _watch(self, response: httpx.Response, deadline: float) -> None:
        task = asyncio.create_task(self._expire(response, deadline))
        self._watchdogs = {**self._watchdogs, response: task}
        task.add_done_callback(lambda done: self._forget(response, done))

    async def _expire(self, response: httpx.Response, deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))
        await _close_response(response, self._close_timeout)

    def _forget(self, response: httpx.Response, task: asyncio.Task[None]) -> None:
        if self._watchdogs.get(response) is task:
            self._watchdogs = {
                key: value
                for key, value in self._watchdogs.items()
                if key is not response
            }
        if not task.cancelled() and task.exception() is not None:
            logger.error("Upstream deadline cleanup failed")

    @staticmethod
    def response_headers(response: httpx.Response) -> dict[str, str]:
        """Copy a deliberate allowlist, never hop-by-hop or stale length headers."""
        return {
            key: value
            for key, value in response.headers.items()
            if key.lower() in _RESPONSE_HEADERS
        }

    async def chunks(self, response: httpx.Response) -> AsyncGenerator[bytes, None]:
        """Yield raw bytes, enforcing the total deadline and size limit."""
        deadline = response.extensions.get(_DEADLINE_EXTENSION)
        if not isinstance(deadline, float):
            await self.close(response)
            raise AdapterError(
                "invalid_upstream_response",
                "Upstream deadline is missing",
                status_code=502,
            )
        error: AdapterError | None = None
        total = 0
        try:
            iterator = _response_chunks(response)
            while True:
                try:
                    chunk = await _next_chunk(iterator, deadline)
                except StopAsyncIteration:
                    break
                total += len(chunk)
                if total > self.settings.max_response_bytes:
                    raise AdapterError(
                        "upstream_response_too_large",
                        "Upstream response exceeds configured limit",
                        status_code=502,
                    )
                yield chunk
        except (httpx.TimeoutException, asyncio.TimeoutError):
            error = _timeout_error()
        except httpx.HTTPError:
            error = (
                _timeout_error()
                if asyncio.get_running_loop().time() >= deadline
                else _stream_error()
            )
        finally:
            await self.close(response)
        if error is not None:
            raise error from None

    async def read(self, response: httpx.Response) -> bytes:
        """Buffer a bounded complete response, for Gemini or native JSON."""
        return b"".join([chunk async for chunk in self.chunks(response)])

    async def close(self, response: httpx.Response) -> None:
        """Stop its deadline watcher and close the upstream response."""
        watcher = self._watchdogs.get(response)
        self._watchdogs = {
            key: value for key, value in self._watchdogs.items() if key is not response
        }
        try:
            if watcher is not None:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
        finally:
            await _close_response(response, self._close_timeout)

    async def aclose(self) -> None:
        """Release any responses still owned by the transport at shutdown."""
        responses = tuple(self._watchdogs)
        outcomes = await asyncio.gather(
            *(self.close(response) for response in responses), return_exceptions=True
        )
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            raise AdapterError(
                "upstream_cleanup_failed",
                "Unable to close all upstream responses",
                status_code=502,
            )


async def _next_chunk(iterator: AsyncIterator[bytes], deadline: float) -> bytes:
    token = _SILENCE_NETWORK_LOGS.set(True)
    try:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        chunk = await asyncio.wait_for(anext(iterator), timeout=remaining)
        if asyncio.get_running_loop().time() >= deadline:
            raise asyncio.TimeoutError
        return chunk
    finally:
        _SILENCE_NETWORK_LOGS.reset(token)


async def _close_response(response: httpx.Response, timeout: float) -> None:
    token = _SILENCE_NETWORK_LOGS.set(True)
    try:
        try:
            await asyncio.wait_for(response.aclose(), timeout=timeout)
        except (httpx.HTTPError, OSError, asyncio.TimeoutError):
            pass
        else:
            return
    finally:
        _SILENCE_NETWORK_LOGS.reset(token)
    raise _stream_error() from None


async def _response_chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    if response.is_stream_consumed:
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise AdapterError(
                "invalid_upstream_response",
                "Cannot forward an already decoded encoded response",
                status_code=502,
            )
        yield response.content
        return
    async for chunk in response.aiter_raw():
        yield chunk
