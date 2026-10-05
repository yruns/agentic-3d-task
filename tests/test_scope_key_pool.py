"""Exercise concurrent credential routing and rolling rate limits."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Condition, Event

import pytest
from pydantic import SecretStr

from scripts.scope.modelhub_key_pool import CredentialLimits, CredentialPool


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_pool(
    *, qpm: int = 500, inflight: int = 64
) -> tuple[CredentialPool, ManualClock, Condition]:
    clock = ManualClock()
    condition = Condition()
    pool = CredentialPool(
        (SecretStr("test-a"), SecretStr("test-b"), SecretStr("test-c")),
        CredentialLimits(qpm_per_key=qpm, max_inflight_per_key=inflight),
        clock=clock,
        condition=condition,
    )
    return pool, clock, condition


def test_uses_all_credentials_without_serializing_active_requests() -> None:
    pool, _, _ = make_pool()
    barrier = Barrier(12)

    def request() -> int:
        lease = pool.acquire()
        barrier.wait(timeout=5)
        pool.release(lease, status=200)
        return lease.slot

    with ThreadPoolExecutor(max_workers=12) as executor:
        slots = list(executor.map(lambda _: request(), range(12)))
    assert {slot: slots.count(slot) for slot in set(slots)} == {0: 4, 1: 4, 2: 4}


def test_rate_budget_counts_dispatches_even_after_requests_finish() -> None:
    pool, clock, condition = make_pool(qpm=1)
    for _ in range(3):
        pool.release(pool.acquire(), status=200)
    started = Event()

    def next_request() -> int:
        started.set()
        lease = pool.acquire()
        pool.release(lease, status=200)
        return lease.slot

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(next_request)
        assert started.wait(2)
        assert not future.done()
        with condition:
            clock.now = 60.0
            condition.notify_all()
        assert future.result(timeout=2) in {0, 1, 2}


def test_limits_inflight_and_avoids_a_throttled_key() -> None:
    pool, _, _ = make_pool(inflight=1)
    leases = [pool.acquire() for _ in range(3)]
    pool.release(leases[0], status=429, retry_after_seconds=30)
    pool.release(leases[1], status=200)
    next_lease = pool.acquire()
    assert next_lease.slot == leases[1].slot
    pool.release(next_lease, status=200)
    pool.release(leases[2], status=200)


def test_rejects_empty_or_duplicate_credentials_without_revealing_them() -> None:
    for keys in ((), (SecretStr("private"), SecretStr("private"))):
        with pytest.raises(ValueError) as error:
            CredentialPool(
                keys, CredentialLimits(), clock=ManualClock(), condition=Condition()
            )
        assert "private" not in str(error.value)


def test_released_lease_cannot_release_a_different_active_request() -> None:
    pool, _, _ = make_pool()
    lease = pool.acquire()
    pool.release(lease, status=200)
    with pytest.raises(ValueError, match="lease_not_active"):
        pool.release(lease, status=200)
