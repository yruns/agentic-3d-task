"""Exercise the empty-candidate contract across both SCOPE HTTP boundaries."""

from pathlib import Path
from threading import Thread
from typing import cast

import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import JsonValue, SecretStr, TypeAdapter

from scripts.scope.modelhub_bridge import (
    BridgeConfig,
    BridgeServer,
    BridgeService,
    ChatCompletionRequest,
    LedgerRecord,
)
from scripts.scope.prefilter_response_contract import EMPTY_PREFILTER_HEADER
from scripts.scope.run_scope import FailClosedCompletions


class _Ledger:
    def __init__(self) -> None:
        self.records: list[LedgerRecord] = []

    def record(self, record: LedgerRecord) -> None:
        self.records.append(record)


def _messages() -> list[ChatCompletionMessageParam]:
    path = Path(__file__).parent / "fixtures/scope_empty_object_prefilter_messages.json"
    messages = TypeAdapter(list[dict[str, JsonValue]]).validate_json(path.read_bytes())
    ChatCompletionRequest.model_validate(
        {"model": "gpt-4o-2024-11-20", "messages": messages}
    )
    return cast(list[ChatCompletionMessageParam], messages)


def _completion(content: str | None, finish_reason: str) -> dict[str, JsonValue]:
    return {
        "id": "empty-prefilter",
        "created": 1,
        "object": "chat.completion",
        "model": "gemini-3.8-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100},
    }


@pytest.mark.parametrize("content", ["", "   \n"])
def test_completed_empty_class_response_preserves_raw_answer_and_model_call(
    content: str,
) -> None:
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_completion(content, "stop"))

    ledger = _Ledger()
    config = BridgeConfig(
        endpoint="https://provider.invalid/chat",
        api_key=SecretStr("test-key"),
        bridge_token=SecretStr("test-bridge"),
    )
    with httpx.Client(transport=httpx.MockTransport(upstream)) as provider:
        service = BridgeService(config, client=provider, ledger=ledger)

        def bridge(request: httpx.Request) -> httpx.Response:
            response = service.forward(
                ChatCompletionRequest.model_validate_json(request.content),
                supports_empty_object_prefilter=(
                    request.headers.get(EMPTY_PREFILTER_HEADER) == "1"
                ),
            )
            return httpx.Response(response.status_code, content=response.body)

        with OpenAI(
            api_key="test-bridge",
            base_url="http://bridge.invalid/v1",
            max_retries=0,
            default_headers={EMPTY_PREFILTER_HEADER: "1"},
            http_client=httpx.Client(transport=httpx.MockTransport(bridge)),
        ) as client:
            result = FailClosedCompletions(
                client.chat.completions, expected_model="gemini-3.8-flash"
            ).create(model="gpt-4o-2024-11-20", messages=_messages())

    assert len(calls) == 1
    assert result.choices[0].message.content == content
    assert result.choices[0].finish_reason == "stop"
    assert result.usage is not None and result.usage.completion_tokens == 0
    assert len(ledger.records) == 1 and ledger.records[0].status == 200
    assert ledger.records[0].tokens.completion_tokens == 0


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [(None, "stop"), ("", "length"), ("", "content_filter")],
)
def test_missing_or_incomplete_empty_class_response_still_fails_closed(
    content: str | None, finish_reason: str
) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(content, finish_reason))

    ledger = _Ledger()
    config = BridgeConfig(
        endpoint="https://provider.invalid/chat",
        api_key=SecretStr("test-key"),
        bridge_token=SecretStr("test-bridge"),
    )
    with httpx.Client(transport=httpx.MockTransport(upstream)) as provider:
        response = BridgeService(config, client=provider, ledger=ledger).forward(
            ChatCompletionRequest.model_validate(
                {"model": "gpt-4o-2024-11-20", "messages": _messages()}
            ),
            supports_empty_object_prefilter=True,
        )

    assert response.status_code == 502
    assert ledger.records[0].status == 502


def test_legacy_client_empty_answer_remains_rejected() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("", "stop"))

    ledger = _Ledger()
    config = BridgeConfig(
        endpoint="https://provider.invalid/chat",
        api_key=SecretStr("test-key"),
        bridge_token=SecretStr("test-bridge"),
    )
    with httpx.Client(transport=httpx.MockTransport(upstream)) as provider:
        response = BridgeService(config, client=provider, ledger=ledger).forward(
            ChatCompletionRequest.model_validate(
                {"model": "gpt-4o-2024-11-20", "messages": _messages()}
            )
        )
    assert response.status_code == 502


@pytest.mark.parametrize(
    ("capability", "expected_status"), [("1", 200), ("", 502), ("true", 502)]
)
def test_http_handler_requires_explicit_empty_prefilter_capability(
    capability: str, expected_status: int
) -> None:
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_completion("", "stop"))

    config = BridgeConfig(
        endpoint="https://provider.invalid/chat",
        api_key=SecretStr("test-key"),
        bridge_token=SecretStr("test-bridge"),
    )
    with httpx.Client(transport=httpx.MockTransport(upstream)) as provider:
        server = BridgeServer(
            ("127.0.0.1", 0), BridgeService(config, client=provider, ledger=_Ledger())
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = httpx.post(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                headers={
                    "Authorization": "Bearer test-bridge",
                    EMPTY_PREFILTER_HEADER: capability,
                },
                json={"model": "gpt-4o-2024-11-20", "messages": _messages()},
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        assert not thread.is_alive()
    assert len(calls) == 1
    assert response.status_code == expected_status
