"""Boundary and failure tests for the local SCOPE ModelHub bridge."""

from collections.abc import Iterator
from http.client import HTTPResponse
from pathlib import Path
from socket import create_connection
from threading import Thread
from uuid import UUID

import httpx
import pytest
from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from scripts.scope.modelhub_bridge import (
    MAX_REQUEST_BYTES,
    BridgeConfig,
    BridgeServer,
    BridgeService,
    ChatCompletionRequest,
    JsonlLedger,
    LedgerRecord,
    ThinkingConfig,
    main,
)


class MemoryLedger:
    """Collect metadata without filesystem or credential access."""

    def __init__(self) -> None:
        self.records: list[LedgerRecord] = []

    def record(self, record: LedgerRecord) -> None:
        self.records.append(record)


def decode_object(payload: bytes) -> dict[str, JsonValue]:
    return TypeAdapter(dict[str, JsonValue]).validate_json(payload)


@pytest.fixture
def config() -> BridgeConfig:
    return BridgeConfig(
        endpoint="https://provider.invalid/v1/chat/completions?region=test",
        api_key=SecretStr("unit-test-key&?#"),
        bridge_token=SecretStr("unit-test-bridge-token"),
    )


@pytest.fixture
def completion() -> dict[str, JsonValue]:
    return {
        "id": "unit-completion",
        "created": 12345,
        "object": "chat.completion",
        "model": "gemini-3.8-flash-provider-snapshot",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "frontier 0",
                    "reasoning_content": "provider thought retained separately",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
        "provider_extension": {"unchanged": True},
    }


@pytest.fixture
def request_payload() -> dict[str, JsonValue]:
    return {
        "model": "gpt-4o-2024-11-20",
        "messages": [
            {"role": "system", "content": "Choose a frontier."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Private unit-test prompt."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aGVsbG8=",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": True,
    }


def test_forward_preserves_multimodal_parameters_and_provider_response(
    config: BridgeConfig,
    completion: dict[str, JsonValue],
    request_payload: dict[str, JsonValue],
) -> None:
    captured: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=completion)

    ledger = MemoryLedger()
    client = httpx.Client(transport=httpx.MockTransport(upstream))
    service = BridgeService(config, client=client, ledger=ledger)
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(request_payload)
        )
        assert response.status_code == 200
        assert decode_object(response.body) == completion
        assert len(captured) == 1
        upstream_request = captured[0]
        sent = decode_object(upstream_request.content)
        assert sent["model"] == "gemini-3.8-flash"
        assert sent["thinking"] == {"budget_tokens": 2000, "include_thoughts": True}
        assert sent["stream"] is False
        for key in (
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
        ):
            assert sent[key] == request_payload[key]
        assert upstream_request.url.params["ak"] == "unit-test-key&?#"
        assert upstream_request.url.params["region"] == "test"
        assert UUID(hex=upstream_request.headers["X-TT-LOGID"]).version == 4
        record = ledger.records[0]
        assert record.model == completion["model"]
        assert record.status == 200
        assert record.tokens.prompt_tokens == 100
        assert record.tokens.completion_tokens == 25
        assert record.tokens.reasoning_tokens == 10
        assert record.thinking_budget_tokens == 2000
        assert record.elapsed_ms >= 0
    finally:
        service.close()
    assert client.is_closed


@pytest.mark.parametrize("model", ["gpt-4o-2024-11-20", "gemini-3.8-flash"])
def test_only_source_and_target_models_are_accepted(model: str) -> None:
    request = ChatCompletionRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": "hello"}]}
    )
    assert request.model == model


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
        {"model": "gemini-3.8-flash", "messages": []},
        {
            "model": "gemini-3.8-flash",
            "messages": [{"role": "tool", "content": "hello"}],
        },
        {
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": [{"type": "audio"}]}],
        },
        {
            "model": "gemini-3.8-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://[invalid"}}
                    ],
                }
            ],
        },
        {
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 4,
        },
    ],
)
def test_invalid_external_request_is_rejected(payload: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


@pytest.mark.parametrize("upstream_status", [400, 401, 403, 404, 429, 500])
def test_upstream_errors_are_sanitized_and_recorded(
    config: BridgeConfig, upstream_status: int
) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            upstream_status,
            text=f"secret body {request.url} unit-test-bridge-token",
        )

    ledger = MemoryLedger()
    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gpt-4o-2024-11-20",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        )
    finally:
        service.close()
    assert response.status_code == (429 if upstream_status == 429 else 502)
    assert "error" in decode_object(response.body)
    assert b"provider.invalid" not in response.body
    assert b"unit-test" not in response.body
    assert ledger.records[0].status == response.status_code


def test_timeout_has_no_fake_completion(config: BridgeConfig) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"private URL {request.url}", request=request)

    ledger = MemoryLedger()
    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gemini-3.8-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        )
    finally:
        service.close()
    assert response.status_code == 504
    assert "choices" not in decode_object(response.body)
    assert b"private" not in response.body
    assert ledger.records[0].status == 504


@pytest.mark.parametrize(
    "upstream_payload",
    [
        b"not JSON",
        b'{"error":{"message":"secret upstream error"}}',
        b'{"model":"gemini-3.8-flash","choices":[]}',
        b'{"model":"gemini-3.8-flash","choices":[{"message":{"role":"assistant","reasoning_content":"thought only"}}]}',
        b'{"model":"gemini-3.8-flash","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":" "}}]}',
    ],
)
def test_invalid_upstream_payload_cannot_become_success(
    config: BridgeConfig, upstream_payload: bytes
) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=upstream_payload)

    ledger = MemoryLedger()
    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gemini-3.8-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        )
    finally:
        service.close()
    assert response.status_code == 502
    assert ledger.records[0].status == 502


def test_ledger_contains_only_call_metadata(tmp_path: Path) -> None:
    ledger_path = tmp_path / "calls.jsonl"
    ledger = JsonlLedger(ledger_path)
    record = LedgerRecord.model_validate(
        {
            "model": "gemini-3.8-flash",
            "elapsed_ms": 12.5,
            "status": 200,
            "thinking_budget_tokens": 2000,
            "tokens": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125,
            },
        }
    )
    ledger.record(record)
    payload = decode_object(ledger_path.read_bytes())
    assert set(payload) == {
        "model",
        "elapsed_ms",
        "status",
        "tokens",
        "thinking_budget_tokens",
        "response_id_synthesized",
        "metadata_synthesized",
    }
    assert ledger_path.stat().st_mode & 0o777 == 0o600


@pytest.fixture
def running_bridge(
    config: BridgeConfig, completion: dict[str, JsonValue]
) -> Iterator[tuple[str, MemoryLedger, list[httpx.Request]]]:
    captured: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=completion)

    ledger = MemoryLedger()
    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    server = BridgeServer(("127.0.0.1", 0), service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", ledger, captured
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_http_health_auth_validation_and_completion(
    running_bridge: tuple[str, MemoryLedger, list[httpx.Request]],
    request_payload: dict[str, JsonValue],
    completion: dict[str, JsonValue],
) -> None:
    url, ledger, captured = running_bridge
    headers = {"Authorization": "Bearer unit-test-bridge-token"}
    with httpx.Client(base_url=url, trust_env=False) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert "unit-test" not in health.text
        assert "provider.invalid" not in health.text
        assert not captured
        unauthorized = client.post("/v1/chat/completions", json=request_payload)
        assert unauthorized.status_code == 401
        assert not captured
        invalid = client.post(
            "/v1/chat/completions", headers=headers, content=b"bad JSON"
        )
        assert invalid.status_code == 400
        assert not captured
        parsed_url = httpx.URL(url)
        assert parsed_url.port is not None
        with create_connection(
            (parsed_url.host, parsed_url.port), timeout=5
        ) as connection:
            connection.sendall(
                (
                    "POST /v1/chat/completions HTTP/1.1\r\n"
                    "Host: localhost\r\n"
                    "Authorization: Bearer unit-test-bridge-token\r\n"
                    f"Content-Length: {MAX_REQUEST_BYTES + 1}\r\n\r\n"
                ).encode("ascii")
            )
            oversized = HTTPResponse(connection)
            oversized.begin()
            assert oversized.status == 413
            oversized.read()
        assert not captured
        response = client.post(
            "/v1/chat/completions", headers=headers, json=request_payload
        )
        assert response.status_code == 200
        assert decode_object(response.content) == completion
    assert len(captured) == 1
    assert ledger.records[-1].status == 200


def test_bad_private_config_does_not_print_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "private.json"
    config_path.write_text('{"api_key":"private-unit-secret","endpoint":"bad-url"}')
    assert main(["--config", str(config_path), "--port", "0"]) == 2
    output = capsys.readouterr()
    assert "private-unit-secret" not in output.err
    assert "bad-url" not in output.err


def test_private_config_supports_explicit_ledger_path(tmp_path: Path) -> None:
    config = BridgeConfig.model_validate(
        {
            "endpoint": "https://provider.invalid/v1/chat/completions",
            "api_key": "unit-test-key",
            "bridge_token": "unit-test-token",
            "model": "gemini-3.8-flash",
            "thinking": {"budget_tokens": 2000, "include_thoughts": True},
            "ledger_path": str(tmp_path / "calls.jsonl"),
        }
    )
    assert config.ledger_path == tmp_path / "calls.jsonl"


def test_missing_standard_metadata_is_synthesized_without_altering_provider_fields(
    config: BridgeConfig, completion: dict[str, JsonValue]
) -> None:
    provider_payload = {
        key: value for key, value in completion.items() if key not in {"id", "created"}
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=provider_payload)

    ledger = MemoryLedger()
    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gemini-3.8-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        )
    finally:
        service.close()
    actual = decode_object(response.body)
    assert response.status_code == 200
    assert isinstance(actual["id"], str) and actual["id"].startswith("chatcmpl-")
    assert isinstance(actual["created"], int)
    assert {
        key: value for key, value in actual.items() if key not in {"id", "created"}
    } == provider_payload
    assert ledger.records[0].response_id_synthesized is True
    assert ledger.records[0].metadata_synthesized is True


def test_missing_choice_index_is_synthesized(config: BridgeConfig) -> None:
    provider_payload: dict[str, JsonValue] = {
        "object": "chat.completion",
        "model": "gemini-3.8-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "frontier 0",
                    "reasoning_content": "separate thought",
                },
            }
        ],
        "usage": {"prompt_tokens": 42},
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=provider_payload)

    ledger = MemoryLedger()
    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gemini-3.8-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        )
    finally:
        service.close()
    actual = decode_object(response.body)
    choices = actual["choices"]
    assert isinstance(choices, list) and isinstance(choices[0], dict)
    assert choices[0]["index"] == 0
    assert choices[0]["message"] == {
        "role": "assistant",
        "content": "frontier 0",
        "reasoning_content": "separate thought",
    }
    assert actual["usage"] == provider_payload["usage"]
    assert ledger.records[0].metadata_synthesized is True


def test_http_library_logs_do_not_expose_query_credentials(
    config: BridgeConfig,
    completion: dict[str, JsonValue],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion)

    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=MemoryLedger(),
    )
    try:
        with caplog.at_level("DEBUG"):
            service.forward(
                ChatCompletionRequest.model_validate(
                    {
                        "model": "gemini-3.8-flash",
                        "messages": [
                            {"role": "user", "content": "private-unit-prompt"}
                        ],
                    }
                )
            )
    finally:
        service.close()
    assert "unit-test-key" not in caplog.text
    assert "private-unit-prompt" not in caplog.text
    assert "ak=" not in caplog.text


def test_ledger_write_failure_cannot_return_completion(
    config: BridgeConfig, completion: dict[str, JsonValue]
) -> None:
    class FailingLedger:
        def record(self, record: LedgerRecord) -> None:
            raise OSError("private-unit-ledger-path")

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion)

    service = BridgeService(
        config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=FailingLedger(),
    )
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gemini-3.8-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        )
    finally:
        service.close()
    assert response.status_code == 500
    assert b"private-unit" not in response.body
    assert "choices" not in decode_object(response.body)


def test_provider_thinking_and_cached_tokens_are_recorded_without_changing_usage(
    config: BridgeConfig, completion: dict[str, JsonValue]
) -> None:
    usage: dict[str, JsonValue] = {
        "prompt_tokens": 1102,
        "completion_tokens": 18,
        "total_tokens": 1182,
        "reasoning_tokens": 62,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    completion["usage"] = usage
    custom_config = config.model_copy(
        update={"thinking": ThinkingConfig(budget_tokens=3000)}
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion)

    ledger = MemoryLedger()
    service = BridgeService(
        custom_config,
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        ledger=ledger,
    )
    try:
        response = service.forward(
            ChatCompletionRequest.model_validate(
                {
                    "model": "gemini-3.8-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        )
    finally:
        service.close()
    assert decode_object(response.body)["usage"] == usage
    record = ledger.records[0]
    assert record.tokens.reasoning_tokens == 62
    assert record.tokens.cached_prompt_tokens == 0
    assert record.tokens.total_tokens == 1182
    assert record.thinking_budget_tokens == 3000
