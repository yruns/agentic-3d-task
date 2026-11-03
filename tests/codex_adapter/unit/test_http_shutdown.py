"""Cancellation and shutdown retain ownership of in-flight resources."""

import asyncio
import threading

import httpx
import pytest

from codex_adapter.application.gateway import Gateway, NativeReply
from codex_adapter.config import ServerSettings
from codex_adapter.interfaces import app as app_module
from codex_adapter.interfaces.admission import Admission


@pytest.mark.asyncio
async def test_cancelled_sql_await_is_tracked_until_thread_finishes() -> None:
    started, release = threading.Event(), threading.Event()

    def database_work() -> str:
        started.set()
        assert release.wait(2)
        return "done"

    gateway = Gateway(None, None, None)  # type: ignore[arg-type]
    task = asyncio.create_task(gateway._store_call(database_work))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not await gateway.drain_store(0.01)
    finally:
        release.set()
    assert await gateway.drain_store(1)


@pytest.mark.asyncio
async def test_shutdown_cancels_abandoned_operations_and_releases_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    class BlockingGateway:
        async def forward(self, call: object) -> NativeReply:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return NativeReply(httpx.Response(200))

    class Transport:
        async def close(self, response: httpx.Response) -> None:
            await response.aclose()

    monkeypatch.setattr(app_module, "_STATE_DRAIN_SECONDS", 0.1)
    pending = app_module._PendingRequests(Transport())  # type: ignore[arg-type]
    admission = Admission(ServerSettings(max_concurrent_requests=1))
    admission.acquire()
    operation = pending.start(BlockingGateway(), None)  # type: ignore[arg-type]
    await asyncio.wait_for(started.wait(), 1)
    pending.abandon(operation, admission, cancel=False)
    await asyncio.sleep(0)
    await pending.shutdown(0.01)
    assert cancelled.is_set()
    assert operation.cancelled()
    admission.acquire()
    admission.release()


@pytest.mark.asyncio
async def test_abandoned_completed_native_reply_is_closed() -> None:
    closed: list[bool] = []

    class CompletedGateway:
        async def forward(self, call: object) -> NativeReply:
            return NativeReply(httpx.Response(200))

    class Transport:
        async def close(self, response: httpx.Response) -> None:
            closed.append(True)
            await response.aclose()

    pending = app_module._PendingRequests(Transport())  # type: ignore[arg-type]
    admission = Admission(ServerSettings(max_concurrent_requests=1))
    admission.acquire()
    operation = pending.start(CompletedGateway(), None)  # type: ignore[arg-type]
    await operation
    pending.abandon(operation, admission, cancel=False)
    await pending.shutdown(1)
    assert closed == [True]
    admission.acquire()
    admission.release()


@pytest.mark.asyncio
async def test_shutdown_waits_for_cleanup_added_during_drain() -> None:
    operation_release, close_started, close_release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    class DelayedGateway:
        async def forward(self, call: object) -> NativeReply:
            await operation_release.wait()
            return NativeReply(httpx.Response(200))

    class DelayedTransport:
        async def close(self, response: httpx.Response) -> None:
            close_started.set()
            await close_release.wait()
            await response.aclose()

    pending = app_module._PendingRequests(DelayedTransport())  # type: ignore[arg-type]
    admission = Admission(ServerSettings(max_concurrent_requests=1))
    admission.acquire()
    operation = pending.start(DelayedGateway(), None)  # type: ignore[arg-type]
    shutdown = asyncio.create_task(pending.shutdown(1))
    try:
        await asyncio.sleep(0)
        pending.abandon(operation, admission, cancel=False)
        operation_release.set()
        await asyncio.wait_for(close_started.wait(), 1)
        await asyncio.sleep(0.02)
        assert not shutdown.done()
    finally:
        operation_release.set()
        close_release.set()
        await shutdown
    admission.acquire()
    admission.release()
