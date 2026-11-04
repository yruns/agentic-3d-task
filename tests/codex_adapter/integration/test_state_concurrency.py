"""Process ownership, abrupt owner loss, and close-versus-thread coordination."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from test_state_support import completion, digest, registered_model, state_settings
from test_state_support import store as store

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import RequestLease
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore


def test_close_gates_queued_operations_and_drains_the_active_transaction(
    tmp_path: Path,
) -> None:
    entered, release = Event(), Event()

    def draw(total: int) -> int:
        entered.set()
        assert release.wait(5.0)
        return 0

    instance = SQLiteStateStore(
        state_settings(tmp_path), clock=lambda: 100.0, draw=draw
    )
    instance.initialize()
    with ThreadPoolExecutor(max_workers=3) as pool:
        active = pool.submit(instance.bind, "active", registered_model())
        assert entered.wait(5.0)
        closing = pool.submit(instance.close)
        try:
            assert instance._database._closing.wait(5.0)
            queued = pool.submit(instance.bind, "queued", registered_model())
        finally:
            release.set()
        assert active.result(timeout=5.0).binding.session_id == "active"
        closing.result(timeout=5.0)
        with pytest.raises(AdapterError) as raised:
            queued.result(timeout=5.0)
        assert raised.value.status_code == 503
    instance.close()


def worker(directory: Path, operation: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[3]
    script = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from test_state_support import state_settings, registered_model, digest
from codex_adapter.domain.errors import AdapterError
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore
store = SQLiteStateStore(state_settings(Path(sys.argv[3])), clock=lambda: 100.0)
try:
    store.ready()
except AdapterError as error:
    raise SystemExit(2 if error.status_code == 503 else 3)
if sys.argv[4] == 'crash':
    binding = store.bind('session', registered_model()).binding
    store.reserve(binding, digest('interrupted'), previous_carrier=None)
    os._exit(0)
store.close()
"""
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(root / "src"),
            str(Path(__file__).parent),
            str(directory),
            operation,
        ],
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )


def test_owner_lock_is_cross_process_and_released_after_close(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    assert worker(tmp_path, "open").returncode == 2
    store.close()
    reopened = worker(tmp_path, "open")
    assert reopened.returncode == 0, reopened.stderr


def test_abrupt_process_death_releases_ownership_and_quarantines_inflight(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    binding = store.bind("session", registered_model()).binding
    store.close()
    interrupted = worker(tmp_path, "crash")
    assert interrupted.returncode == 0, interrupted.stderr
    reopened = SQLiteStateStore(state_settings(tmp_path), clock=lambda: 110.0)
    try:
        reopened.ready()
        assert reopened.turns(binding) == ()
        with pytest.raises(AdapterError) as raised:
            reopened.reserve(binding, digest("interrupted"), previous_carrier=None)
        assert raised.value.status_code == 409
    finally:
        reopened.close()


def test_wal_readers_do_not_block_on_active_writer_transaction(
    tmp_path: Path,
) -> None:
    writer_entered, writer_release = Event(), Event()
    block_writer = False

    def draw(total: int) -> int:
        if block_writer:
            writer_entered.set()
            assert writer_release.wait(5.0)
        return 0

    instance = SQLiteStateStore(
        state_settings(tmp_path), clock=lambda: 100.0, draw=draw
    )
    instance.initialize()
    try:
        existing = instance.bind("existing-session", registered_model()).binding
        lease = instance.reserve(existing, digest("first"), previous_carrier=None)
        assert isinstance(lease, RequestLease)
        expected = completion(lease)
        instance.commit(lease, expected)

        block_writer = True
        with ThreadPoolExecutor(max_workers=10) as pool:
            writer_future = pool.submit(
                instance.bind, "slow-writer-session", registered_model()
            )
            assert writer_entered.wait(5.0)
            try:
                read_binds = [
                    pool.submit(instance.bind, "existing-session", registered_model())
                    for _ in range(4)
                ]
                read_turns = [pool.submit(instance.turns, existing) for _ in range(4)]
                for bind_future in read_binds:
                    assert bind_future.result(timeout=2.0).binding == existing
                for turn_future in read_turns:
                    assert turn_future.result(timeout=2.0) == (expected.turn,)
            finally:
                writer_release.set()
            assert (
                writer_future.result(timeout=5.0).binding.session_id
                == "slow-writer-session"
            )
    finally:
        instance.close()
