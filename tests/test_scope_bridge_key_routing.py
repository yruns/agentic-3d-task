"""Verify pooled forwarding uses real responses and preserves request content."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import httpx
import pytest
from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from scripts.scope.modelhub_bridge import (
    BridgeConfig,
    BridgeService,
    ChatCompletionRequest,
    ChatMessage,
    LedgerRecord,
)
from scripts.scope.modelhub_key_pool import CredentialLimits


class LockedLedger:
    def __init__(self) -> None:
        self.records: list[LedgerRecord] = []
        self.lock = Lock()

    def record(self, record: LedgerRecord) -> None:
        with self.lock:
            self.records.append(record)


def pool_config() -> BridgeConfig:
    return BridgeConfig(
        endpoint="https://provider.invalid/completions",
        api_key=SecretStr("test-a"),
        additional_api_keys=(SecretStr("test-b"), SecretStr("test-c")),
        bridge_token=SecretStr("test-token"),
        credential_limits=CredentialLimits(max_inflight_per_key=4),
    )


def response_body() -> dict[str, JsonValue]:
    return {
        "id": "test-completion",
        "created": 1,
        "model": "gemini-3.8-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "0.7"},
            }
        ],
    }


def test_twelve_dispatches_overlap_use_all_keys_and_retain_payloads() -> None:
    barrier = Barrier(12)
    ledger = LockedLedger()
    received: list[tuple[str, dict[str, JsonValue]]] = []
    lock = Lock()

    def upstream(request: httpx.Request) -> httpx.Response:
        payload = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        with lock:
            received.append((request.url.params["ak"], payload))
        barrier.wait(timeout=5)
        return httpx.Response(200, json=response_body())

    service = BridgeService(
        pool_config(),
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )

    def forward(index: int) -> int:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gpt-4o-2024-11-20",
                    "messages": [{"role": "user", "content": f"candidate {index}"}],
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 4096,
                }
            )
        )
        assert (
            TypeAdapter(dict[str, JsonValue]).validate_json(response.body)
            == response_body()
        )
        return response.status_code

    try:
        with ThreadPoolExecutor(max_workers=12) as executor:
            assert list(executor.map(forward, range(12))) == [200] * 12
    finally:
        service.close()
    assert {
        key: sum(key == actual for actual, _ in received)
        for key in ("test-a", "test-b", "test-c")
    } == {"test-a": 4, "test-b": 4, "test-c": 4}
    assert {
        TypeAdapter(list[ChatMessage]).validate_python(payload["messages"])[0].content
        for _, payload in received
    } == {f"candidate {index}" for index in range(12)}
    for _, payload in received:
        assert payload["thinking"] == {"budget_tokens": 2000, "include_thoughts": True}
        assert (payload["temperature"], payload["top_p"], payload["max_tokens"]) == (
            0.7,
            0.95,
            4096,
        )
    assert len(ledger.records) == 12
    assert {record.credential_slot for record in ledger.records} == {0, 1, 2}
    assert all(
        record.started_at_unix is not None and record.recorded_at_unix is not None
        for record in ledger.records
    )
    assert all(record.queued_ms is not None for record in ledger.records)
    assert all("test-a" not in record.model_dump_json() for record in ledger.records)


def test_throttled_key_is_not_selected_for_next_sdk_retry() -> None:
    seen: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        key = request.url.params["ak"]
        seen.append(key)
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "30"}, text="private")
        return httpx.Response(200, json=response_body())

    ledger = LockedLedger()
    service = BridgeService(
        pool_config(),
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    request = ChatCompletionRequest.model_validate(
        {
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    try:
        assert service.forward(request).status_code == 429
        assert service.forward(request).status_code == 200
    finally:
        service.close()
    assert seen == ["test-a", "test-b"]
    assert [record.status for record in ledger.records] == [429, 200]


def test_duplicate_private_keys_fail_configuration_validation() -> None:
    with pytest.raises(ValidationError, match="duplicate_upstream_credentials"):
        BridgeConfig(
            endpoint="https://provider.invalid/completions",
            api_key=SecretStr("private-key"),
            additional_api_keys=(SecretStr("private-key"),),
            bridge_token=SecretStr("private-token"),
        )
