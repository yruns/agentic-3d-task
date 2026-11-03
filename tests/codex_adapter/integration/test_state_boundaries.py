"""Boundary regressions found during the independent persistence self-review."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from test_state_support import completion, digest, registered_model, state_settings
from test_state_support import store as store

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import Replay, RequestLease
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore


@pytest.mark.parametrize("field", ["session_id", "principal"])
def test_invalid_utf8_identity_is_a_sanitized_validation_error(
    store: SQLiteStateStore, field: str
) -> None:
    values = {
        "session_id": "session",
        "principal": "local",
        field: "private-\ud800-input",
    }
    with pytest.raises(AdapterError) as raised:
        store.bind(
            values["session_id"], registered_model(), principal=values["principal"]
        )
    assert raised.value.status_code == 400
    assert "private" not in str(raised.value)


@pytest.mark.parametrize("field", ["digest", "idempotency", "previous"])
def test_invalid_utf8_request_identity_never_leaks_a_raw_encoding_error(
    store: SQLiteStateStore, field: str
) -> None:
    binding = store.bind("session", registered_model()).binding
    with pytest.raises(AdapterError):
        store.reserve(
            binding,
            "private-\ud800-body" if field == "digest" else digest("request"),
            previous_carrier="private-\ud800-carrier" if field == "previous" else None,
            idempotency_key="private-\ud800-key" if field == "idempotency" else None,
        )


@pytest.mark.parametrize("clock_value", [-1.0, float("nan"), float("inf")])
def test_nonfinite_or_negative_clock_fails_closed(
    tmp_path: Path, clock_value: float
) -> None:
    instance = SQLiteStateStore(state_settings(tmp_path), clock=lambda: clock_value)
    try:
        instance.initialize()
        with pytest.raises(AdapterError) as raised:
            instance.bind("session", registered_model())
        assert raised.value.code == "invalid_clock"
        assert raised.value.status_code == 500
    finally:
        instance.close()


def test_invalid_utf8_credential_secret_fails_without_echoing_the_secret(
    store: SQLiteStateStore,
) -> None:
    model = registered_model()
    invalid = replace(model.upstreams[0], secret="private-ak-\ud800")
    with pytest.raises(AdapterError) as raised:
        store.bind("session", replace(model, upstreams=(invalid,)))
    assert "private-ak" not in str(raised.value)


@pytest.mark.parametrize("kind", ["trigger", "index", "view"])
def test_ready_rejects_unrecognized_schema_objects(
    store: SQLiteStateStore, tmp_path: Path, kind: str
) -> None:
    store.close()
    statements = {
        "trigger": "CREATE TRIGGER unexpected AFTER INSERT ON turns BEGIN SELECT 1; END",
        "index": "CREATE INDEX unexpected ON turns (carrier)",
        "view": "CREATE VIEW unexpected AS SELECT * FROM turns",
    }
    with sqlite3.connect(state_settings(tmp_path).database) as database:
        database.execute(statements[kind])
    reopened = SQLiteStateStore(state_settings(tmp_path))
    try:
        with pytest.raises(AdapterError):
            reopened.ready()
    finally:
        reopened.close()


def test_known_committed_replay_survives_a_reduced_byte_quota(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    binding = store.bind("session", registered_model()).binding
    lease = store.reserve(binding, digest("request"), previous_carrier=None)
    assert isinstance(lease, RequestLease)
    result = completion(lease)
    store.commit(lease, result)
    store.close()
    reduced = SQLiteStateStore(
        replace(state_settings(tmp_path), max_state_bytes=1), clock=lambda: 110.0
    )
    try:
        reduced.ready()
        replay = reduced.reserve(binding, lease.request_digest, previous_carrier=None)
        assert replay == Replay(result.response, result.raw_usage)
        with pytest.raises(AdapterError) as raised:
            reduced.reserve(binding, digest("next"), previous_carrier=lease.carrier)
        assert raised.value.status_code == 413
    finally:
        reduced.close()


def test_tombstones_continue_to_count_against_the_hard_session_cap(
    tmp_path: Path,
) -> None:
    settings = replace(state_settings(tmp_path), max_sessions=1, session_ttl_seconds=1)
    first = SQLiteStateStore(settings, clock=lambda: 100.0)
    first.initialize()
    first.bind("expired", registered_model())
    first.close()
    restarted = SQLiteStateStore(settings, clock=lambda: 101.0)
    try:
        restarted.ready()
        with pytest.raises(AdapterError) as raised:
            restarted.bind("expired", registered_model())
        assert raised.value.status_code == 410
        with pytest.raises(AdapterError) as raised:
            restarted.bind("new", registered_model())
        assert raised.value.status_code == 429
    finally:
        restarted.close()


def test_empty_native_tool_id_and_opaque_native_fields_round_trip(
    store: SQLiteStateStore,
) -> None:
    binding = store.bind("session", registered_model()).binding
    lease = store.reserve(binding, digest("tool-request"), previous_carrier=None)
    assert isinstance(lease, RequestLease)
    original = completion(lease)
    native = b'{"role":"assistant","signature":"opaque","tool_calls":[{"id":"","signature":"tool-opaque"}],"multimodal_contents":[{"signature":"part-opaque"}]}'
    result = replace(
        original,
        turn=replace(
            original.turn, native_message=native, call_ids=(("projected-call-id", ""),)
        ),
    )
    store.commit(lease, result)
    assert store.turns(binding) == (result.turn,)
    assert store.reserve(
        binding, lease.request_digest, previous_carrier=None
    ) == Replay(result.response, result.raw_usage)
