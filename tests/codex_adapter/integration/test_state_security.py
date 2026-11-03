"""Precise local-state tampering regressions; not a rollback-proof database claim."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from test_state_support import digest, registered_model, state_settings
from test_state_support import store as store

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import RequestLease, StateKey
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore


@pytest.mark.parametrize("field", ["session_id", "principal"])
def test_mutated_session_lookup_cannot_silently_create_a_new_binding(
    store: SQLiteStateStore, tmp_path: Path, field: str
) -> None:
    original = store.bind("original-session", registered_model()).binding
    statements = {
        "session_id": "UPDATE bindings SET session_id = ? WHERE id = ?",
        "principal": "UPDATE bindings SET principal = ? WHERE id = ?",
    }
    with sqlite3.connect(state_settings(tmp_path).database) as connection:
        connection.execute(statements[field], ("mutated-identity", original.id))
    with pytest.raises(AdapterError) as raised:
        store.bind("original-session", registered_model())
    assert raised.value.code == "state_corrupt"
    with sqlite3.connect(state_settings(tmp_path).database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM bindings").fetchone() == (1,)


def test_session_lookup_authentication_survives_active_key_rotation(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    original = store.bind("original-session", registered_model()).binding
    store.close()
    settings = state_settings(tmp_path)
    rotated = replace(
        settings,
        keys=(*settings.keys, StateKey("rotated", b"r" * 32)),
        active_key_id="rotated",
    )
    reopened = SQLiteStateStore(rotated, clock=lambda: 110.0)
    try:
        reopened.ready()
        assert reopened.bind("original-session", registered_model()).binding == original
        with sqlite3.connect(settings.database) as connection:
            connection.execute(
                "UPDATE bindings SET session_id = ? WHERE id = ?",
                ("tampered", original.id),
            )
        with pytest.raises(AdapterError) as raised:
            reopened.bind("original-session", registered_model())
        assert raised.value.code == "state_corrupt"
    finally:
        reopened.close()


def test_legacy_schema_is_rejected_without_automatic_reinitialization(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    store.close()
    database = state_settings(tmp_path).database
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE metadata SET schema_version = ?", (1,))
    reopened = SQLiteStateStore(state_settings(tmp_path))
    try:
        with pytest.raises(AdapterError):
            reopened.ready()
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT schema_version FROM metadata"
            ).fetchone() == (1,)
    finally:
        reopened.close()


def remove_request(directory: Path, lease: RequestLease) -> None:
    with sqlite3.connect(state_settings(directory).database) as connection:
        connection.execute(
            "DELETE FROM idempotency_keys WHERE request_id = ?", (lease.request_id,)
        )
        connection.execute(
            "DELETE FROM requests WHERE request_id = ?", (lease.request_id,)
        )


@pytest.mark.parametrize("indeterminate", [False, True])
def test_deleting_unfinished_journal_does_not_release_the_session(
    store: SQLiteStateStore, tmp_path: Path, indeterminate: bool
) -> None:
    binding = store.bind("session", registered_model()).binding
    lease = store.reserve(binding, digest("first"), previous_carrier=None)
    assert isinstance(lease, RequestLease)
    if indeterminate:
        store.indeterminate(lease)
    remove_request(tmp_path, lease)
    with pytest.raises(AdapterError) as raised:
        store.reserve(binding, digest("different"), previous_carrier=None)
    assert raised.value.code == "state_corrupt"


def test_deleted_unfinished_journal_stays_fail_closed_after_restart(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    binding = store.bind("session", registered_model()).binding
    lease = store.reserve(binding, digest("first"), previous_carrier=None)
    assert isinstance(lease, RequestLease)
    store.close()
    remove_request(tmp_path, lease)
    reopened = SQLiteStateStore(state_settings(tmp_path), clock=lambda: 110.0)
    try:
        reopened.ready()
        with pytest.raises(AdapterError) as raised:
            reopened.reserve(binding, digest("different"), previous_carrier=None)
        assert raised.value.code == "state_corrupt"
    finally:
        reopened.close()
