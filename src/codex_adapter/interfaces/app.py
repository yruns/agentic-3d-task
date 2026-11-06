"""Small HTTP boundary; routing, codecs, persistence and IO remain separate."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager
from typing import cast

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from codex_adapter import __version__
from codex_adapter.application.catalog import describe_model
from codex_adapter.application.gateway import (
    BufferedReply,
    Gateway,
    GatewayRequest,
    NativeReply,
)
from codex_adapter.application.requests import parse_envelope
from codex_adapter.config import AdapterSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.infrastructure.http_transport import HTTPTransport
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore
from codex_adapter.interfaces.admission import Admission
from codex_adapter.interfaces.http_boundary import (
    authenticate,
    read_body,
    request_metadata,
)

logger = logging.getLogger(__name__)
_STATE_DRAIN_SECONDS = 5.0
_Reply = BufferedReply | NativeReply


class _PendingRequests:
    """Bounded by admission; retain cancelled clients' operations until settled."""

    def __init__(self, transport: HTTPTransport) -> None:
        self.transport = transport
        self._operations: set[asyncio.Task[_Reply]] = set()
        self._cleanup: set[asyncio.Task[None]] = set()
        self._closing = False

    def start(self, gateway: Gateway, request: GatewayRequest) -> asyncio.Task[_Reply]:
        if self._closing:
            raise AdapterError(
                "adapter_stopping", "Adapter is shutting down", status_code=503
            )
        task = asyncio.create_task(gateway.forward(request))
        self._operations.add(task)
        task.add_done_callback(self._operation_done)
        return task

    def _operation_done(self, task: asyncio.Task[_Reply]) -> None:
        self._operations.discard(task)
        if not task.cancelled():
            task.exception()

    def abandon(
        self, task: asyncio.Task[_Reply], admission: Admission, *, cancel: bool
    ) -> None:
        if cancel:
            task.cancel()
        cleanup = asyncio.create_task(self._dispose(task))
        self._cleanup.add(cleanup)
        cleanup.add_done_callback(lambda done: self._cleanup_done(done, admission))

    async def _dispose(self, task: asyncio.Task[_Reply]) -> None:
        try:
            reply = await asyncio.shield(task)
            if isinstance(reply, NativeReply):
                await self.transport.close(reply.response)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("Abandoned adapter request failed or could not be closed")

    def _cleanup_done(self, task: asyncio.Task[None], admission: Admission) -> None:
        self._cleanup.discard(task)
        admission.release()
        if not task.cancelled() and task.exception() is not None:
            logger.error("Abandoned request cleanup failed")

    async def shutdown(self, timeout: float) -> None:
        self._closing = True
        if await self._drain(timeout):
            return
        logger.warning("Adapter requests exceeded the shutdown drain deadline")
        for task in tuple(self._operations | self._cleanup):
            task.cancel()
        if not await self._drain(_STATE_DRAIN_SECONDS):
            logger.error("Some adapter requests did not stop after cancellation")

    async def _drain(self, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while self._operations or self._cleanup:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait(
                tuple(self._operations | self._cleanup),
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return False
        return True


def create_app(
    settings: AdapterSettings,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    initialize_state: bool = False,
) -> FastAPI:
    """Build an explicitly configured application, without import-time clients."""
    store = SQLiteStateStore(settings.state)
    admission = Admission(settings.server)
    app = FastAPI(
        title="Codex Adapter",
        version=__version__,
        lifespan=_lifespan(settings, store, upstream_transport, initialize_state),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    _install_error_handler(app)
    _install_metadata_routes(app, settings, admission, store)
    _install_response_routes(app, settings, admission)
    return app


def _lifespan(
    settings: AdapterSettings,
    store: SQLiteStateStore,
    upstream_transport: httpx.AsyncBaseTransport | None,
    initialize_state: bool,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if initialize_state:
            await asyncio.to_thread(store.initialize)
        await asyncio.to_thread(store.ready)
        async with httpx.AsyncClient(
            transport=upstream_transport,
            timeout=settings.server.upstream_timeout_seconds,
        ) as client:
            transport = HTTPTransport(client, settings.server)
            gateway = Gateway(settings, store, transport)
            pending = _PendingRequests(transport)
            app.state.gateway = gateway
            app.state.transport = transport
            app.state.pending_requests = pending
            try:
                yield
            finally:
                await _shutdown(settings, pending, gateway, transport, store)

    return lifespan


async def _shutdown(
    settings: AdapterSettings,
    pending: _PendingRequests,
    gateway: Gateway,
    transport: HTTPTransport,
    store: SQLiteStateStore,
) -> None:
    try:
        await pending.shutdown(
            settings.server.upstream_timeout_seconds + _STATE_DRAIN_SECONDS
        )
        if not await gateway.drain_store(_STATE_DRAIN_SECONDS):
            logger.error("Database worker threads remain active during shutdown")
        await transport.aclose()
    finally:
        await asyncio.to_thread(store.close)


def _install_error_handler(app: FastAPI) -> None:
    @app.exception_handler(AdapterError)
    async def adapter_error(request: Request, error: AdapterError) -> JSONResponse:
        return JSONResponse(error.payload(), status_code=error.status_code)


def _install_metadata_routes(
    app: FastAPI,
    settings: AdapterSettings,
    admission: Admission,
    store: SQLiteStateStore,
) -> None:
    _install_health_route(app, settings, admission, store)
    _install_model_routes(app, settings, admission)


def _install_health_route(
    app: FastAPI,
    settings: AdapterSettings,
    admission: Admission,
    store: SQLiteStateStore,
) -> None:
    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        authenticate(request, settings)
        admission.acquire()
        try:
            await asyncio.to_thread(store.ready)
            return {
                "status": "ready",
                "service": "codex-adapter",
                "version": __version__,
                "schema_version": 1,
                "models": [model.id for model in settings.registry.models],
            }
        finally:
            admission.release()


def _install_model_routes(
    app: FastAPI, settings: AdapterSettings, admission: Admission
) -> None:
    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, object]:
        authenticate(request, settings)
        admission.acquire()
        try:
            return {
                "object": "list",
                "data": [
                    {"id": model.id, "object": "model", "owned_by": "codex-adapter"}
                    for model in settings.registry.models
                ],
            }
        finally:
            admission.release()

    @app.get("/v1/models/{model_id:path}/capabilities")
    async def capabilities(request: Request, model_id: str) -> dict[str, object]:
        authenticate(request, settings)
        admission.acquire()
        try:
            return dict(describe_model(settings.registry.get(model_id)))
        finally:
            admission.release()


def _install_response_routes(
    app: FastAPI, settings: AdapterSettings, admission: Admission
) -> None:
    @app.post("/v1/responses")
    @app.post("/v1/responses/compact")
    async def responses(request: Request) -> Response:
        return await _respond(request, settings, admission)


async def _make_request(request: Request, settings: AdapterSettings) -> GatewayRequest:
    raw = await read_body(request, settings.server.max_request_bytes)
    envelope = parse_envelope(raw)
    session_id, idempotency_key = request_metadata(request)
    return GatewayRequest(
        envelope,
        session_id,
        uuid.uuid4().hex,
        idempotency_key=idempotency_key,
        compact=request.url.path.endswith("/compact"),
    )


async def _respond(
    request: Request, settings: AdapterSettings, admission: Admission
) -> Response:
    authenticate(request, settings)
    admission.acquire()
    transferred = False
    try:
        call = await _make_request(request, settings)
        model = settings.registry.get(call.envelope.model)
        gateway = cast(Gateway, request.app.state.gateway)
        pending = cast(_PendingRequests, request.app.state.pending_requests)
        operation = pending.start(gateway, call)
        try:
            reply = (
                await asyncio.shield(operation)
                if model.protocol == "gemini_chat"
                else await operation
            )
        except asyncio.CancelledError:
            abandon = getattr(gateway, "abandon_request", None)
            if callable(abandon):
                abandon(call.request_id)
            pending.abandon(operation, admission, cancel=model.protocol == "responses")
            transferred = True
            raise
        response = await _build_response(
            request, reply, call.envelope.stream, admission
        )
        transferred = call.envelope.stream and _status(reply) < 300
        return response
    finally:
        if not transferred:
            admission.release()


def _status(reply: _Reply) -> int:
    return (
        reply.response.status_code
        if isinstance(reply, NativeReply)
        else reply.status_code
    )


def _one_shot_release(admission: Admission) -> Callable[[], None]:
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            admission.release()

    return release


async def _build_response(
    request: Request,
    reply: _Reply,
    stream: bool,
    admission: Admission,
) -> Response:
    if isinstance(reply, NativeReply):
        transport = cast(HTTPTransport, request.app.state.transport)
        headers = transport.response_headers(reply.response)
        if stream and reply.response.status_code < 300:
            release = _one_shot_release(admission)
            return StreamingResponse(
                _native_stream(transport, reply.response, release),
                status_code=reply.response.status_code,
                headers=headers,
                background=BackgroundTask(release),
            )
        return Response(
            await transport.read(reply.response),
            status_code=reply.response.status_code,
            headers=headers,
        )
    headers = {"X-Codex-Adapter-Replay": "true" if reply.replayed else "false"}
    if stream and reply.status_code < 300:
        from codex_adapter.codecs.gemini_chat import response_events

        release = _one_shot_release(admission)
        return StreamingResponse(
            _buffered_events(response_events(reply.body), release),
            media_type="text/event-stream",
            headers=headers,
            background=BackgroundTask(release),
        )
    return Response(
        reply.body,
        status_code=reply.status_code,
        media_type=reply.media_type,
        headers=headers,
    )


async def _native_stream(
    transport: HTTPTransport,
    response: httpx.Response,
    release: Callable[[], None],
) -> AsyncIterator[bytes]:
    try:
        async with aclosing(transport.chunks(response)) as chunks:
            async for chunk in chunks:
                yield chunk
    finally:
        release()


async def _buffered_events(
    events: tuple[bytes, ...], release: Callable[[], None]
) -> AsyncIterator[bytes]:
    try:
        for event in events:
            yield event
    finally:
        release()
