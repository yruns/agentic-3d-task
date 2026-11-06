"""Route once, preserve native Responses, or execute a durable Gemini turn."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import httpx

from codex_adapter.application.ports import StateStore, UpstreamTransport
from codex_adapter.application.requests import RequestEnvelope
from codex_adapter.config import AdapterSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import decode_object, encode_json
from codex_adapter.domain.models import RegisteredModel
from codex_adapter.domain.state import (
    BoundRoute,
    Completion,
    PreparedGemini,
    Replay,
    RequestLease,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_MAX_TOKEN_COUNT = (1 << 63) - 1
_RETRYABLE_UPSTREAM_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_UPSTREAM_CODES = frozenset(
    {
        "upstream_unavailable",
        "upstream_timeout",
        "upstream_stream_interrupted",
        "invalid_upstream_response",
    }
)


def _is_retryable_upstream_error(exc: BaseException) -> bool:
    return isinstance(exc, AdapterError) and exc.code in _RETRYABLE_UPSTREAM_CODES


@dataclass(frozen=True)
class BufferedReply:
    body: bytes = field(repr=False)
    status_code: int = 200
    media_type: str = "application/json"
    replayed: bool = False


@dataclass(frozen=True)
class NativeReply:
    response: httpx.Response = field(repr=False)


@dataclass(frozen=True)
class GatewayRequest:
    envelope: RequestEnvelope
    session_id: str = field(repr=False)
    request_id: str
    principal: str = "local"
    idempotency_key: str | None = field(default=None, repr=False)
    compact: bool = False


class Gateway:
    """Coordinate IO through injected transport/store; tools remain in Codex."""

    def __init__(
        self, settings: AdapterSettings, store: StateStore, transport: UpstreamTransport
    ) -> None:
        self.settings = settings
        self.store = store
        self.transport = transport
        self._store_tasks: set[asyncio.Task[object]] = set()
        self._inflight_gemini: dict[
            str, tuple[str, str, str | None, asyncio.Task[BufferedReply]]
        ] = {}
        self._abandoned_requests: set[str] = set()

    def abandon_request(self, request_id: str) -> None:
        """Mark a request whose HTTP client disconnected before response delivery."""
        self._abandoned_requests.add(request_id)

    async def _store_call(
        self, function: Callable[..., _T], *args: object, **kwargs: object
    ) -> _T:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._store_tasks.add(task)
        task.add_done_callback(self._store_finished)
        return await asyncio.shield(task)

    def _store_finished(self, task: asyncio.Task[object]) -> None:
        self._store_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def drain_store(self, timeout: float) -> bool:
        """Wait for database threads without claiming cancelled threads stopped."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self._store_tasks:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            _, pending = await asyncio.wait(tuple(self._store_tasks), timeout=remaining)
            if pending:
                return False
            self._store_tasks = {task for task in self._store_tasks if not task.done()}
        return True

    async def forward(self, request: GatewayRequest) -> BufferedReply | NativeReply:
        model = self.settings.registry.get(request.envelope.model)
        if request.compact and model.protocol != "responses":
            raise AdapterError(
                "unsupported_capability",
                "This model does not support compact",
                param="compact",
            )
        route = await self._store_call(
            self.store.bind, request.session_id, model, principal=request.principal
        )
        if model.protocol == "responses":
            response = await self.transport.open(
                route,
                request.envelope.raw,
                request_id=request.request_id,
                compact=request.compact,
            )
            return NativeReply(response)
        return await self._gemini(request, route)

    async def _gemini(
        self, request: GatewayRequest, route: BoundRoute
    ) -> BufferedReply:
        from codex_adapter.codecs.gemini_chat import last_carrier, prepare_request

        body = decode_object(request.envelope.raw)
        digest = hashlib.sha256(encode_json(body)).hexdigest()
        previous = last_carrier(body)
        existing = self._inflight_gemini.get(route.binding.id)
        if existing is not None:
            prev_req_id, prev_digest, prev_previous, prev_task = existing
            if not prev_task.done():
                if (
                    prev_digest == digest
                    and prev_previous == previous
                    and request.idempotency_key is None
                ):
                    self._abandoned_requests.discard(prev_req_id)
                    try:
                        reply = await asyncio.shield(prev_task)
                        return BufferedReply(
                            reply.body,
                            status_code=reply.status_code,
                            media_type=reply.media_type,
                            replayed=True,
                        )
                    except BaseException:
                        pass
                else:
                    self._abandoned_requests.add(prev_req_id)
                    prev_task.cancel()
                    await asyncio.gather(prev_task, return_exceptions=True)

        current_task = asyncio.current_task()
        if current_task is not None:
            self._inflight_gemini[route.binding.id] = (
                request.request_id,
                digest,
                previous,
                current_task,
            )
        try:
            reserved = await self._store_call(
                self.store.reserve,
                route.binding,
                digest,
                previous_carrier=previous,
                idempotency_key=request.idempotency_key,
            )
            if isinstance(reserved, Replay):
                return BufferedReply(reserved.response, replayed=True)
            try:
                turns = await self._store_call(self.store.turns, route.binding)
                prepared = prepare_request(body, route.model, turns)
            except BaseException:
                await self._cleanup_lease(self.store.reject, reserved)
                raise
            return await self._generate(request, route, reserved, prepared)
        finally:
            self._abandoned_requests.discard(request.request_id)
            active = self._inflight_gemini.get(route.binding.id)
            if active is not None and active[0] == request.request_id:
                self._inflight_gemini.pop(route.binding.id, None)

    async def _generate(
        self,
        request: GatewayRequest,
        route: BoundRoute,
        lease: RequestLease,
        prepared: PreparedGemini,
    ) -> BufferedReply:
        max_retries = self.settings.server.upstream_max_retries
        backoff = self.settings.server.upstream_retry_backoff_seconds
        completion: Completion | None = None
        for attempt in range(max_retries + 1):
            try:
                upstream = await self.transport.open(
                    route, prepared.payload, request_id=request.request_id
                )
            except BaseException as exc:
                if (
                    attempt < max_retries
                    and request.request_id not in self._abandoned_requests
                    and _is_retryable_upstream_error(exc)
                ):
                    if backoff > 0:
                        await asyncio.sleep(backoff * (attempt + 1))
                    continue
                await self._cleanup_lease(self.store.reject, lease)
                raise
            if upstream.status_code >= 300:
                if (
                    attempt < max_retries
                    and request.request_id not in self._abandoned_requests
                    and upstream.status_code in _RETRYABLE_UPSTREAM_STATUSES
                ):
                    try:
                        await self.transport.close(upstream)
                    except BaseException:
                        pass
                    if backoff > 0:
                        await asyncio.sleep(backoff * (attempt + 1))
                    continue
                await self._reject_upstream(upstream, lease)
            try:
                raw = await self.transport.read(upstream)
                completion = _convert_upstream(raw, prepared, lease, route.model)
                break
            except BaseException as exc:
                if (
                    attempt < max_retries
                    and request.request_id not in self._abandoned_requests
                    and _is_retryable_upstream_error(exc)
                ):
                    if backoff > 0:
                        await asyncio.sleep(backoff * (attempt + 1))
                    continue
                await self._cleanup_lease(self.store.reject, lease)
                raise
        assert completion is not None
        if request.request_id in self._abandoned_requests:
            await self._cleanup_lease(self.store.reject, lease)
            raise AdapterError(
                "client_disconnected",
                "Client disconnected before response delivery; binding is unchanged",
                status_code=499,
            )
        try:
            await self._store_call(self.store.commit, lease, completion)
        except BaseException:
            await self._cleanup_lease(self.store.indeterminate, lease)
            raise
        _observe(request, completion)
        return BufferedReply(completion.response)

    async def _reject_upstream(
        self, upstream: httpx.Response, lease: RequestLease
    ) -> None:
        try:
            await self.transport.close(upstream)
        finally:
            await self._store_call(self.store.reject, lease)
        raise AdapterError(
            "upstream_rejected",
            "Upstream rejected the request; binding is unchanged",
            status_code=upstream.status_code if upstream.status_code >= 400 else 502,
        )

    async def _cleanup_lease(
        self, operation: Callable[[RequestLease], None], lease: RequestLease
    ) -> None:
        try:
            await self._store_call(operation, lease)
        except Exception:
            # Preserve the original failure, including cancellation; state remains fail-closed.
            logger.error("Unable to finalize a request lease")


def _convert_upstream(
    raw: bytes, prepared: PreparedGemini, lease: RequestLease, model: RegisteredModel
) -> Completion:
    from codex_adapter.codecs.gemini_chat import convert_response

    try:
        return convert_response(decode_object(raw), prepared, lease, model)
    except AdapterError:
        pass
    raise AdapterError(
        "invalid_upstream_response",
        "Upstream returned an invalid response",
        status_code=502,
    ) from None


def _token_count(value: object) -> int | None:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_TOKEN_COUNT
    ):
        return value
    return None


def _request_id(value: str) -> str:
    try:
        return uuid.UUID(value).hex
    except (ValueError, AttributeError):
        return "unknown"


def _observe(request: GatewayRequest, completion: Completion) -> None:
    try:
        usage = decode_object(completion.raw_usage) if completion.raw_usage else {}
    except AdapterError:
        usage = {}
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    logger.info(
        "request_completed request_id=%s protocol=gemini_chat input_tokens=%s cached_tokens=%s",
        _request_id(request.request_id),
        _token_count(usage.get("prompt_tokens")),
        _token_count(cached),
    )
