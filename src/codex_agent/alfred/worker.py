"""Bounded JSONL RPC for explicitly configured legacy model/simulator workers."""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel, ConfigDict

_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class WorkerError(RuntimeError):
    """An external worker failed, disconnected, or exceeded its deadline."""


class _RpcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    result: dict[str, object] | None = None
    error_type: str | None = None
    message: str | None = None


class JsonWorker:
    """Own a sequential JSONL worker and its child process group.

    Responses are ``{ok: true, result: {...}}`` or
    ``{ok: false, error_type: ..., message: ...}``. All diagnostics must go to
    stderr. Failure closes the group; the worker is never silently restarted.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: float,
    ) -> None:
        if not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("command must contain non-empty strings")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.log_path = log_path
        self.timeout_seconds = timeout_seconds
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = log_path.open("ab")
        self._closed = False
        self._buffer = bytearray()
        try:
            self._process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                start_new_session=True,
                bufsize=0,
            )
        except OSError:
            self._stderr.close()
            raise

    @property
    def is_closed(self) -> bool:
        """Whether this worker and its owned process group have been stopped."""
        return self._closed

    def __enter__(self) -> JsonWorker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        """Send one request and wait at most the configured RPC time budget."""
        if self._closed:
            raise WorkerError("worker is closed")
        if self._process.stdin is None or self._process.stdout is None:
            raise WorkerError("worker RPC pipes were not created")
        try:
            self._process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            self._process.stdin.flush()
            response = _RpcResponse.model_validate_json(self._read_response())
        except (OSError, ValueError, WorkerError) as exc:
            self.close()
            raise WorkerError(f"worker RPC failed: {exc}; see {self.log_path}") from exc
        if not response.ok:
            self.close()
            raise WorkerError(
                f"worker {response.error_type}: {response.message}; see {self.log_path}"
            )
        if response.result is None:
            self.close()
            raise WorkerError("worker returned no RPC result")
        return response.result

    def close(self) -> None:
        """Stop only this worker and descendants, including on deadline failure."""
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(b'{"command":"close"}\n')
                self._process.stdin.flush()
                self._process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass  # Failed graceful shutdown is followed by group termination.
        self._signal_group(signal.SIGTERM)
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._signal_group(signal.SIGKILL)
            self._process.wait(timeout=2.0)
        # The worker can exit while Unity/model children are still running.
        self._signal_group(signal.SIGKILL)
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                stream.close()
        self._stderr.close()

    def _signal_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self._process.pid, sig)
        except ProcessLookupError:
            pass  # The group already exited, which is the requested state.

    def _read_response(self) -> bytes:
        assert self._process.stdout is not None
        deadline = time.monotonic() + self.timeout_seconds
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(timeout=remaining):
                    raise WorkerError(f"worker timed out after {self.timeout_seconds}s")
                chunk = os.read(self._process.stdout.fileno(), 65536)
                if not chunk:
                    raise WorkerError("worker exited before replying")
                self._buffer.extend(chunk)
                if len(self._buffer) > _MAX_RESPONSE_BYTES:
                    raise WorkerError("worker response exceeds the size limit")
        line, _, tail = self._buffer.partition(b"\n")
        self._buffer = bytearray(tail)
        return bytes(line)
