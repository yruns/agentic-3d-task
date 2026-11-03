"""Durable reservation, atomic publication, replay, and crash isolation."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from test_state_support import completion, digest, registered_model, state_settings
from test_state_support import store as store

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import Replay, RequestLease
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore


def reserve(
    store: SQLiteStateStore,
    text: str = "request",
    *,
    previous: str | None = None,
    idem: str | None = None,
) -> RequestLease:
    binding = store.bind("session", registered_model()).binding
    lease = store.reserve(
        binding, digest(text), previous_carrier=previous, idempotency_key=idem
    )
    assert isinstance(lease, RequestLease)
    return lease


def test_reserve_includes_all_response_identity_before_upstream_execution(
    store: SQLiteStateStore,
) -> None:
    lease = reserve(store)
    assert lease.request_id
    assert lease.response_id.startswith("resp_")
    assert lease.carrier.startswith("codex_adapter.state.v2.")
    assert lease.sequence == 1
    assert lease.previous_carrier is None
    assert lease.created_at == 100.0
    assert lease.idempotency_key
    assert store.turns(lease.binding) == ()


def test_commit_atomically_persists_turn_and_exact_journal_response(
    store: SQLiteStateStore,
) -> None:
    lease = reserve(store)
    result = completion(lease, response=b'{ "exact_raw_response": true }\n')
    store.commit(lease, result)
    assert store.turns(lease.binding) == (result.turn,)
    replay = store.reserve(lease.binding, lease.request_digest, previous_carrier=None)
    assert replay == Replay(result.response, result.raw_usage)


def test_committed_response_replays_after_process_restart(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    lease = reserve(store, idem="explicit-key")
    result = completion(lease)
    store.commit(lease, result)
    store.close()
    restarted = SQLiteStateStore(state_settings(tmp_path), clock=lambda: 110.0)
    try:
        restarted.ready()
        binding = restarted.bind("session", registered_model()).binding
        replay = restarted.reserve(
            binding,
            lease.request_digest,
            previous_carrier=None,
            idempotency_key="explicit-key",
        )
        assert replay == Replay(result.response, result.raw_usage)
        assert restarted.turns(binding) == (result.turn,)
    finally:
        restarted.close()


def test_same_predecessor_and_digest_replays_even_with_another_explicit_key(
    store: SQLiteStateStore,
) -> None:
    lease = reserve(store, idem="first-idempotency")
    result = completion(lease)
    store.commit(lease, result)
    replay = store.reserve(
        lease.binding,
        lease.request_digest,
        previous_carrier=None,
        idempotency_key="second-idempotency",
    )
    assert replay == Replay(result.response, result.raw_usage)
    with pytest.raises(AdapterError) as raised:
        store.reserve(
            lease.binding,
            digest("different"),
            previous_carrier=None,
            idempotency_key="second-idempotency",
        )
    assert raised.value.status_code == 409


def test_older_committed_request_replays_after_later_turns(
    store: SQLiteStateStore,
) -> None:
    first = reserve(store, "first")
    first_result = completion(first)
    store.commit(first, first_result)
    second = reserve(store, "second", previous=first.carrier)
    store.commit(second, completion(second))
    assert store.reserve(
        first.binding, first.request_digest, previous_carrier=None
    ) == Replay(first_result.response, first_result.raw_usage)
    assert tuple(turn.carrier for turn in store.turns(first.binding)) == (
        first.carrier,
        second.carrier,
    )


@pytest.mark.parametrize(
    "first_status", ["inflight", "committed", "rejected", "indeterminate"]
)
def test_explicit_idempotency_key_never_accepts_a_different_body(
    store: SQLiteStateStore, first_status: str
) -> None:
    lease = reserve(store, "first", idem="fixed-key")
    if first_status == "committed":
        store.commit(lease, completion(lease))
    elif first_status == "rejected":
        store.reject(lease)
    elif first_status == "indeterminate":
        store.indeterminate(lease)
    with pytest.raises(AdapterError) as raised:
        store.reserve(
            lease.binding,
            digest("second"),
            previous_carrier=None,
            idempotency_key="fixed-key",
        )
    assert raised.value.status_code == 409


def test_concurrent_reservations_allow_only_one_inflight_request(
    store: SQLiteStateStore,
) -> None:
    binding = store.bind("session", registered_model()).binding

    def attempt(number: int) -> RequestLease | AdapterError:
        try:
            result = store.reserve(binding, digest(str(number)), previous_carrier=None)
            assert isinstance(result, RequestLease)
            return result
        except AdapterError as error:
            return error

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(attempt, range(16)))
    assert sum(isinstance(result, RequestLease) for result in results) == 1
    assert all(
        isinstance(result, RequestLease) or result.status_code == 409
        for result in results
    )


def test_same_inflight_request_is_not_blindly_resent(store: SQLiteStateStore) -> None:
    lease = reserve(store)
    with pytest.raises(AdapterError) as raised:
        store.reserve(lease.binding, lease.request_digest, previous_carrier=None)
    assert raised.value.status_code == 409


def test_reject_releases_known_failure_and_preserves_idempotency_identity(
    store: SQLiteStateStore,
) -> None:
    first = reserve(store, idem="safe-retry")
    store.reject(first)
    retried = store.reserve(
        first.binding,
        first.request_digest,
        previous_carrier=None,
        idempotency_key="safe-retry",
    )
    assert isinstance(retried, RequestLease)
    store.commit(retried, completion(retried))
    assert len(store.turns(first.binding)) == 1


def test_rejected_request_does_not_block_a_different_request(
    store: SQLiteStateStore,
) -> None:
    first = reserve(store, "first")
    store.reject(first)
    second = reserve(store, "second")
    assert second.sequence == 1
    assert second.request_id != first.request_id


def test_indeterminate_result_quarantines_the_entire_session(
    store: SQLiteStateStore,
) -> None:
    lease = reserve(store)
    store.indeterminate(lease)
    for body in (lease.request_digest, digest("another")):
        with pytest.raises(AdapterError) as raised:
            store.reserve(lease.binding, body, previous_carrier=None)
        assert raised.value.status_code == 409
    with pytest.raises(AdapterError):
        store.reject(lease)
    with pytest.raises(AdapterError):
        store.commit(lease, completion(lease))


def test_startup_quarantines_unfinished_requests_without_damaging_commits(
    store: SQLiteStateStore, tmp_path: Path
) -> None:
    committed = reserve(store, "committed")
    result = completion(committed)
    store.commit(committed, result)
    inflight = reserve(store, "interrupted", previous=committed.carrier)
    store.close()
    restarted = SQLiteStateStore(state_settings(tmp_path), clock=lambda: 110.0)
    try:
        restarted.ready()
        assert restarted.turns(committed.binding) == (result.turn,)
        assert restarted.reserve(
            committed.binding, committed.request_digest, previous_carrier=None
        ) == Replay(result.response, result.raw_usage)
        with pytest.raises(AdapterError) as raised:
            restarted.reserve(
                inflight.binding,
                inflight.request_digest,
                previous_carrier=committed.carrier,
            )
        assert raised.value.status_code == 409
        with pytest.raises(AdapterError):
            restarted.reject(inflight)
    finally:
        restarted.close()


def test_ready_called_again_does_not_quarantine_this_instances_active_lease(
    store: SQLiteStateStore,
) -> None:
    lease = reserve(store)
    store.ready()
    store.commit(lease, completion(lease))
    assert len(store.turns(lease.binding)) == 1


def test_stale_or_cross_session_previous_carrier_is_rejected(
    store: SQLiteStateStore,
) -> None:
    first = reserve(store, "first")
    store.commit(first, completion(first))
    second_binding = store.bind("other-session", registered_model()).binding
    for binding, previous in ((first.binding, None), (second_binding, first.carrier)):
        with pytest.raises(AdapterError) as raised:
            store.reserve(binding, digest("new"), previous_carrier=previous)
        assert raised.value.status_code == 409


def test_rejected_stale_request_cannot_overwrite_a_new_head(
    store: SQLiteStateStore,
) -> None:
    rejected = reserve(store, "rejected", idem="old")
    store.reject(rejected)
    advanced = reserve(store, "advanced")
    store.commit(advanced, completion(advanced))
    with pytest.raises(AdapterError) as raised:
        store.reserve(
            rejected.binding,
            rejected.request_digest,
            previous_carrier=None,
            idempotency_key="old",
        )
    assert raised.value.status_code == 409


def test_forged_lease_or_wrong_completion_carrier_is_rejected(
    store: SQLiteStateStore,
) -> None:
    lease = reserve(store)
    original = completion(lease)
    for forged in (
        replace(lease, request_digest=digest("forged")),
        replace(lease, sequence=99),
    ):
        with pytest.raises(AdapterError):
            store.commit(forged, original)
    wrong = replace(original, turn=replace(original.turn, carrier="other"))
    with pytest.raises(AdapterError):
        store.commit(lease, wrong)
    assert store.turns(lease.binding) == ()
    store.commit(lease, original)
    assert store.turns(lease.binding) == (original.turn,)


def test_commit_reject_and_indeterminate_do_not_rewrite_committed_results(
    store: SQLiteStateStore,
) -> None:
    lease = reserve(store)
    result = completion(lease)
    store.commit(lease, result)
    actions: tuple[Callable[[], None], ...] = (
        lambda: store.commit(lease, completion(lease, response=b"changed")),
        lambda: store.reject(lease),
        lambda: store.indeterminate(lease),
    )
    for action in actions:
        with pytest.raises(AdapterError):
            action()
    assert store.reserve(
        lease.binding, lease.request_digest, previous_carrier=None
    ) == Replay(result.response, result.raw_usage)


def test_turn_quota_blocks_new_requests_but_not_committed_replays(
    tmp_path: Path,
) -> None:
    configured = replace(state_settings(tmp_path), max_turns=1)
    instance = SQLiteStateStore(configured, clock=lambda: 100.0)
    try:
        instance.initialize()
        lease = reserve(instance)
        result = completion(lease)
        instance.commit(lease, result)
        with pytest.raises(AdapterError) as raised:
            reserve(instance, "second", previous=lease.carrier)
        assert raised.value.status_code == 413
        assert instance.reserve(
            lease.binding, lease.request_digest, previous_carrier=None
        ) == Replay(result.response, result.raw_usage)
    finally:
        instance.close()


def test_state_byte_quota_rolls_back_partial_commit(tmp_path: Path) -> None:
    configured = replace(state_settings(tmp_path), max_state_bytes=8192)
    instance = SQLiteStateStore(configured, clock=lambda: 100.0)
    try:
        instance.initialize()
        lease = reserve(instance)
        with pytest.raises(AdapterError) as raised:
            instance.commit(lease, completion(lease, native_message=b"x" * 8192))
        assert raised.value.status_code == 413
        assert instance.turns(lease.binding) == ()
        instance.reject(lease)
        retried = reserve(instance)
        instance.commit(retried, completion(retried))
        assert len(instance.turns(lease.binding)) == 1
    finally:
        instance.close()


def test_optional_raw_usage_round_trips_as_none(store: SQLiteStateStore) -> None:
    lease = reserve(store)
    result = completion(lease, raw_usage=None)
    store.commit(lease, result)
    assert store.reserve(
        lease.binding, lease.request_digest, previous_carrier=None
    ) == Replay(result.response, None)


@pytest.mark.parametrize(
    "request_digest,idem", [("", None), ("valid", ""), ("valid", "x" * 1025)]
)
def test_request_identity_is_validated_before_reservation(
    store: SQLiteStateStore, request_digest: str, idem: str | None
) -> None:
    binding = store.bind("session", registered_model()).binding
    with pytest.raises(AdapterError):
        store.reserve(
            binding, request_digest, previous_carrier=None, idempotency_key=idem
        )
