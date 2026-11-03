"""SQLite lifecycle, durable weighted routing, and fail-closed configuration."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from test_state_support import completion, digest, registered_model, state_settings
from test_state_support import store as store

from codex_adapter.application.ports import StateStore
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.models import Upstream, UpstreamStatus
from codex_adapter.domain.state import StateKey
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore


def test_store_structurally_implements_the_persistence_port(
    store: SQLiteStateStore,
) -> None:
    persistence: StateStore = store
    assert (
        persistence.bind("session", registered_model()).binding.session_id == "session"
    )


def test_ready_requires_explicit_initialization_and_never_creates_a_database(
    tmp_path: Path,
) -> None:
    configured = state_settings(tmp_path)
    instance = SQLiteStateStore(configured)
    with pytest.raises(AdapterError) as raised:
        instance.ready()
    assert raised.value.status_code == 503
    assert not configured.database.exists()
    instance.close()


def test_initialize_does_not_overwrite_an_existing_file(tmp_path: Path) -> None:
    configured = state_settings(tmp_path)
    configured.database.write_bytes(b"existing-content")
    instance = SQLiteStateStore(configured)
    with pytest.raises(AdapterError):
        instance.initialize()
    assert configured.database.read_bytes() == b"existing-content"
    instance.close()


def test_initialize_does_not_overwrite_an_existing_database(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    original = store.bind("session", registered_model()).binding
    with pytest.raises(AdapterError):
        store.initialize()
    assert store.bind("session", registered_model()).binding == original
    store.close()
    reopened = SQLiteStateStore(state_settings(tmp_path), clock=lambda: 100.0)
    try:
        reopened.ready()
        assert reopened.bind("session", registered_model()).binding == original
    finally:
        reopened.close()


def test_a_second_store_is_rejected_until_the_owner_closes(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    second = SQLiteStateStore(state_settings(tmp_path))
    try:
        with pytest.raises(AdapterError) as raised:
            second.ready()
        assert raised.value.status_code == 503
        store.close()
        second.ready()
    finally:
        second.close()


def test_close_is_idempotent_and_operations_require_a_ready_store(
    store: SQLiteStateStore,
) -> None:
    store.close()
    store.close()
    with pytest.raises(AdapterError):
        store.bind("session", registered_model())


def test_ready_rejects_incompatible_schema_without_reinitializing(
    tmp_path: Path,
) -> None:
    configured = state_settings(tmp_path)
    with sqlite3.connect(configured.database) as database:
        database.execute("CREATE TABLE unrelated (id TEXT)")
    instance = SQLiteStateStore(configured)
    try:
        with pytest.raises(AdapterError):
            instance.ready()
        with sqlite3.connect(configured.database) as database:
            names = database.execute(
                "SELECT name FROM sqlite_master WHERE type = ?", ("table",)
            ).fetchall()
        assert names == [("unrelated",)]
    finally:
        instance.close()


def test_binding_is_sticky_across_restarts(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    bound = store.bind("session", registered_model())
    store.close()
    restarted = SQLiteStateStore(
        state_settings(tmp_path), clock=lambda: 110.0, draw=lambda total: total - 1
    )
    try:
        restarted.ready()
        rebound = restarted.bind("session", registered_model())
        assert rebound.binding == bound.binding
        assert rebound.upstream == bound.upstream
    finally:
        restarted.close()


def test_concurrent_first_bind_draws_once_and_returns_a_unique_binding(
    tmp_path: Path,
) -> None:
    draws: tuple[int, ...] = ()

    def draw(total: int) -> int:
        nonlocal draws
        draws = (*draws, total)
        return total - 1

    instance = SQLiteStateStore(
        state_settings(tmp_path), clock=lambda: 100.0, draw=draw
    )
    try:
        instance.initialize()
        with ThreadPoolExecutor(max_workers=12) as pool:
            routes = tuple(
                pool.map(
                    lambda _: instance.bind("same-session", registered_model()),
                    range(24),
                )
            )
        assert len({route.binding.id for route in routes}) == 1
        assert {route.upstream.id for route in routes} == {"credential-b"}
        assert draws == (4,)
    finally:
        instance.close()


def test_weight_changes_and_new_credentials_do_not_rebalance_existing_bindings(
    store: SQLiteStateStore,
) -> None:
    model = registered_model()
    original = store.bind("old-session", model)
    revised = replace(
        model,
        upstreams=(
            replace(model.upstreams[0], weight=0),
            model.upstreams[1],
            Upstream(
                "new",
                "v1",
                "https://upstream.example/chat/completions",
                "new-test-ak",
                50,
            ),
        ),
    )
    assert store.bind("old-session", revised).binding == original.binding
    assert store.bind("new-session", revised).upstream.id == "credential-b"


@pytest.mark.parametrize("status,weight", [("draining", 3), ("active", 0)])
def test_old_bindings_allow_draining_and_zero_weight_credentials(
    store: SQLiteStateStore, status: UpstreamStatus, weight: int
) -> None:
    model = registered_model()
    original = store.bind("session", model)
    changed = replace(model.upstreams[0], status=status, weight=weight)
    revised = replace(model, upstreams=(changed, model.upstreams[1]))
    assert store.bind("session", revised).binding == original.binding
    assert store.bind("another-session", revised).upstream.id == "credential-b"


@pytest.mark.parametrize(
    "change",
    ["deleted", "revoked", "secret", "version", "model", "endpoint", "protocol"],
)
def test_changed_route_or_credential_reference_fails_closed(
    store: SQLiteStateStore, change: str
) -> None:
    model = registered_model()
    store.bind("session", model)
    changes = {
        "deleted": replace(model, upstreams=(model.upstreams[1],)),
        "revoked": replace(
            model,
            upstreams=(
                replace(model.upstreams[0], status="revoked"),
                model.upstreams[1],
            ),
        ),
        "secret": replace(
            model,
            upstreams=(
                replace(model.upstreams[0], secret="replacement-ak"),
                model.upstreams[1],
            ),
        ),
        "version": replace(
            model,
            upstreams=(
                replace(model.upstreams[0], version="v2"),
                model.upstreams[1],
            ),
        ),
        "model": replace(model, id="other-model"),
        "endpoint": replace(
            model,
            upstreams=(
                replace(
                    model.upstreams[0], url="https://different.example/chat/completions"
                ),
                model.upstreams[1],
            ),
        ),
        "protocol": replace(model, protocol="responses"),
    }
    with pytest.raises(AdapterError) as raised:
        store.bind("session", changes[change])
    assert raised.value.status_code == 409
    assert "replacement-ak" not in str(raised.value)


def test_principals_have_distinct_session_namespaces(store: SQLiteStateStore) -> None:
    first = store.bind("same", registered_model(), principal="user-a").binding
    second = store.bind("same", registered_model(), principal="user-b").binding
    assert first.id != second.id
    assert first.principal == "user-a"
    assert second.principal == "user-b"


def test_expired_session_is_a_gone_tombstone_not_a_new_binding(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    original = store.bind("session", registered_model()).binding
    store.close()
    expired = SQLiteStateStore(
        state_settings(tmp_path), clock=lambda: original.expires_at
    )
    try:
        expired.ready()
        for operation in (
            lambda: expired.bind("session", registered_model()),
            lambda: expired.turns(original),
            lambda: expired.reserve(original, digest("request"), previous_carrier=None),
        ):
            with pytest.raises(AdapterError) as raised:
                operation()
            assert raised.value.status_code == 410
        assert expired.bind("new-session", registered_model()).binding.id != original.id
    finally:
        expired.close()


def test_max_sessions_is_a_hard_cap_without_silent_eviction(tmp_path: Path) -> None:
    configured = replace(state_settings(tmp_path), max_sessions=1)
    instance = SQLiteStateStore(configured, clock=lambda: 100.0)
    try:
        instance.initialize()
        original = instance.bind("first", registered_model()).binding
        with pytest.raises(AdapterError) as raised:
            instance.bind("second", registered_model())
        assert raised.value.status_code == 429
        assert instance.bind("first", registered_model()).binding == original
    finally:
        instance.close()


def test_missing_or_replaced_state_key_cannot_open_existing_database(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    store.bind("session", registered_model())
    store.close()
    original = state_settings(tmp_path)
    for keys, active in (
        ((StateKey("replacement", b"r" * 32),), "replacement"),
        ((StateKey("key-1", b"x" * 32),), "key-1"),
    ):
        instance = SQLiteStateStore(replace(original, keys=keys, active_key_id=active))
        try:
            with pytest.raises(AdapterError) as raised:
                instance.ready()
            assert raised.value.status_code == 503
        finally:
            instance.close()


def test_old_key_remains_readable_while_new_state_uses_rotated_key(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    route = store.bind("session", registered_model())
    lease = store.reserve(route.binding, digest("first"), previous_carrier=None)
    assert not isinstance(lease, tuple)
    from codex_adapter.domain.state import RequestLease

    assert isinstance(lease, RequestLease)
    first = completion(lease)
    store.commit(lease, first)
    store.close()
    original = state_settings(tmp_path)
    rotated = replace(
        original,
        keys=(*original.keys, StateKey("key-2", b"n" * 32)),
        active_key_id="key-2",
    )
    instance = SQLiteStateStore(rotated, clock=lambda: 110.0)
    try:
        instance.ready()
        assert instance.turns(route.binding) == (first.turn,)
        new_route = instance.bind("new-session", registered_model())
        assert new_route.binding.key_id == "key-2"
        second = instance.reserve(
            route.binding, digest("second"), previous_carrier=lease.carrier
        )
        assert isinstance(second, RequestLease)
        instance.commit(second, completion(second))
        assert len(instance.turns(route.binding)) == 2
    finally:
        instance.close()


@pytest.mark.parametrize(
    "session_id,principal", [("", "local"), ("valid", ""), ("x" * 1025, "local")]
)
def test_untrusted_session_identity_is_validated(
    store: SQLiteStateStore, session_id: str, principal: str
) -> None:
    with pytest.raises(AdapterError):
        store.bind(session_id, registered_model(), principal=principal)


def test_sql_metacharacters_are_data_not_executable_statements(
    store: SQLiteStateStore,
) -> None:
    session_id = "'; DROP TABLE bindings; --"
    binding = store.bind(session_id, registered_model()).binding
    assert store.bind(session_id, registered_model()).binding == binding
    assert store.bind("other", registered_model()).binding.session_id == "other"


def test_forged_or_cross_store_binding_is_rejected(store: SQLiteStateStore) -> None:
    binding = store.bind("session", registered_model()).binding
    for forged in (replace(binding, principal="other"), replace(binding, id="unknown")):
        with pytest.raises(AdapterError):
            store.turns(forged)


def test_binding_record_does_not_contain_plaintext_credentials(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    route = store.bind("session", registered_model())
    store.close()
    contents = state_settings(tmp_path).database.read_bytes()
    assert route.upstream.secret.encode() not in contents
    assert route.binding.credential_fingerprint != route.upstream.secret
    assert len(route.binding.credential_fingerprint) == 64
