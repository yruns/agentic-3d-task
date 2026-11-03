"""Malformed persisted evidence is never treated as recoverable empty state."""

import sqlite3
from pathlib import Path

import pytest
from test_state_support import completion, digest, registered_model, state_settings
from test_state_support import store as store

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import RequestLease
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore


def committed_turn(store: SQLiteStateStore, session: str = "session") -> RequestLease:
    binding = store.bind(session, registered_model()).binding
    lease = store.reserve(binding, digest("request"), previous_carrier=None)
    assert isinstance(lease, RequestLease)
    store.commit(lease, completion(lease))
    return lease


def test_turns_and_requests_persist_plaintext_json_without_upstream_ak(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    committed_turn(store)
    store.close()
    content = state_settings(tmp_path).database.read_bytes()
    assert b"test-ak-first" not in content
    assert b"private-answer" in content


def test_missing_turn_segment_fails_instead_of_restoring_partial_history(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    first = committed_turn(store)
    second = store.reserve(
        first.binding, digest("second"), previous_carrier=first.carrier
    )
    assert isinstance(second, RequestLease)
    store.commit(second, completion(second))
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        database.execute(
            "DELETE FROM turns WHERE binding_id = ? AND sequence = ?",
            (first.binding.id, 1),
        )
    with pytest.raises(AdapterError) as raised:
        store.turns(first.binding)
    assert raised.value.code == "state_corrupt"
    with pytest.raises(AdapterError):
        store.reserve(first.binding, digest("third"), previous_carrier=second.carrier)


def test_deleted_tail_is_detected_by_authenticated_head(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    lease = committed_turn(store)
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        database.execute("DELETE FROM turns WHERE binding_id = ?", (lease.binding.id,))
    with pytest.raises(AdapterError) as raised:
        store.turns(lease.binding)
    assert raised.value.code == "state_corrupt"


@pytest.mark.parametrize("table", ["turns", "requests"])
def test_bad_ciphertext_fails_closed_with_a_safe_error(
    store: SQLiteStateStore, tmp_path: Path, table: str
) -> None:
    lease = committed_turn(store)
    statements = {
        "turns": "UPDATE turns SET payload = ? WHERE binding_id = ?",
        "requests": "UPDATE requests SET payload = ? WHERE binding_id = ?",
    }
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        database.execute(
            statements[table], (b"private-sentinel-corrupt", lease.binding.id)
        )
    with pytest.raises(AdapterError) as raised:
        if table == "turns":
            store.turns(lease.binding)
        else:
            store.reserve(lease.binding, lease.request_digest, previous_carrier=None)
    assert raised.value.code == "state_corrupt"
    assert "private-sentinel" not in str(raised.value)
    assert "UPDATE" not in str(raised.value)


def test_ciphertext_cannot_be_swapped_between_sessions(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    first = committed_turn(store, "first")
    second = committed_turn(store, "second")
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        row = database.execute(
            "SELECT payload FROM turns WHERE binding_id = ?", (first.binding.id,)
        ).fetchone()
        assert row is not None
        database.execute(
            "UPDATE turns SET payload = ? WHERE binding_id = ?",
            (row[0], second.binding.id),
        )
    with pytest.raises(AdapterError) as raised:
        store.turns(second.binding)
    assert raised.value.code == "state_corrupt"
    assert len(store.turns(first.binding)) == 1


def test_corrupt_head_is_not_silently_reset(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    lease = committed_turn(store)
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        database.execute(
            "UPDATE bindings SET head_state = ? WHERE id = ?",
            (b"corrupt", lease.binding.id),
        )
    with pytest.raises(AdapterError) as raised:
        store.turns(lease.binding)
    assert raised.value.code == "state_corrupt"


def test_sqlite_failures_do_not_expose_sql_or_payloads(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    binding = store.bind("session", registered_model()).binding
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        database.execute("DROP TABLE turns")
    with pytest.raises(AdapterError) as raised:
        store.turns(binding)
    assert raised.value.status_code == 503
    assert "SELECT" not in str(raised.value)
    assert "turns" not in str(raised.value)


def test_schema_corruption_is_reported_and_does_not_claim_readiness(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    store.close()
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        database.execute("ALTER TABLE turns ADD COLUMN unexpected TEXT")
    reopened = SQLiteStateStore(state_settings(tmp_path))
    try:
        with pytest.raises(AdapterError):
            reopened.ready()
    finally:
        reopened.close()
