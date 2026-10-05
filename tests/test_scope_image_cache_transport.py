"""Lossless cache references preserve provider inputs and model concurrency."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock, Thread

import httpx
from pydantic import JsonValue, SecretStr, TypeAdapter

from scripts.scope.image_cache_transport import ImageCachingTransport
from scripts.scope.modelhub_bridge import (
    BridgeConfig,
    BridgeServer,
    BridgeService,
    ChatCompletionRequest,
    LedgerRecord,
)
from scripts.scope.modelhub_image_cache import ImageCacheStore


class MemoryLedger:
    def record(self, record: LedgerRecord) -> None:
        return


def payload() -> dict[str, JsonValue]:
    return {
        "model": "gpt-4o-2024-11-20",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Choose candidate 0."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aGVsbG8=",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
    }


def test_shared_upload_once_and_provider_calls_overlap_with_exact_inputs(
    tmp_path: Path,
) -> None:
    uploads = 0
    barrier = Barrier(6)
    lock = Lock()
    store = ImageCacheStore(tmp_path)
    captured: list[dict[str, JsonValue]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        decoded = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        with lock:
            captured.append(decoded)
        barrier.wait(timeout=5)
        return httpx.Response(
            200,
            json={
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "0.7"},
                    }
                ],
            },
        )

    service = BridgeService(
        BridgeConfig(
            endpoint="https://provider.invalid/crawl",
            api_key=SecretStr("test-key"),
            bridge_token=SecretStr("test-token"),
        ),
        client=httpx.Client(transport=httpx.MockTransport(provider)),
        ledger=MemoryLedger(),
        image_cache=store,
    )

    def delegate(request: httpx.Request) -> httpx.Response:
        nonlocal uploads
        assert request.headers["Authorization"] == "Bearer test-token"
        assert int(request.headers["Content-Length"]) == len(request.content)
        if request.method == "PUT":
            with lock:
                uploads += 1
            uri = "scope-cache://" + request.url.path.rsplit("/", 1)[1]
            store.put(uri, request.content.decode())
            return httpx.Response(
                200, json={"uri": uri, "stored_bytes": len(request.content)}
            )
        completion = ChatCompletionRequest.model_validate_json(request.content)
        response = service.forward(completion)
        return httpx.Response(response.status_code, content=response.body)

    try:
        with httpx.Client(
            transport=ImageCachingTransport(httpx.MockTransport(delegate))
        ) as client:

            def request(index: int) -> int:
                return client.post(
                    "http://bridge.invalid/v1/chat/completions",
                    headers={"Authorization": "Bearer test-token"},
                    json=payload(),
                ).status_code

            with ThreadPoolExecutor(max_workers=6) as executor:
                assert list(executor.map(request, range(6))) == [200] * 6
    finally:
        service.close()
    assert uploads == 1 and len(captured) == 6
    for body in captured:
        for field in ("messages", "temperature", "top_p", "max_tokens"):
            assert body[field] == payload()[field]
        assert body["model"] == "gemini-3.8-flash"
        assert body["thinking"] == {"budget_tokens": 2000, "include_thoughts": True}


def test_upload_failure_is_not_a_completion() -> None:
    methods: list[str] = []

    def delegate(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(503, json={"error": {"message": "upload failed"}})

    with httpx.Client(
        transport=ImageCachingTransport(httpx.MockTransport(delegate))
    ) as client:
        response = client.post(
            "http://bridge.invalid/v1/chat/completions", json=payload()
        )
    assert response.status_code == 503 and methods == ["PUT"]


def test_non_image_and_external_urls_are_not_uploaded() -> None:
    request_payload = {
        "model": "gpt-4o-2024-11-20",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://images.invalid/image.png"},
                    }
                ],
            }
        ],
    }
    methods: list[str] = []

    def delegate(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        assert (
            TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
            == request_payload
        )
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(
        transport=ImageCachingTransport(httpx.MockTransport(delegate))
    ) as client:
        assert (
            client.post(
                "http://bridge.invalid/v1/chat/completions", json=request_payload
            ).status_code
            == 200
        )
    assert methods == ["POST"]


def test_real_http_upload_ack_and_cache_reconstruction(tmp_path: Path) -> None:
    captured: list[dict[str, JsonValue]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        captured.append(
            TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        )
        return httpx.Response(
            200,
            json={
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "0.7"},
                    }
                ],
            },
        )

    service = BridgeService(
        BridgeConfig(
            endpoint="https://provider.invalid/crawl",
            api_key=SecretStr("test-key"),
            bridge_token=SecretStr("test-token"),
        ),
        client=httpx.Client(transport=httpx.MockTransport(provider)),
        ledger=MemoryLedger(),
        image_cache=ImageCacheStore(tmp_path),
    )
    server = BridgeServer(("127.0.0.1", 0), service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(
            transport=ImageCachingTransport(httpx.HTTPTransport()), timeout=5
        ) as client:
            url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            for _ in range(2):
                assert (
                    client.post(
                        url,
                        headers={"Authorization": "Bearer test-token"},
                        json=payload(),
                    ).status_code
                    == 200
                )
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(tuple(tmp_path.iterdir())) == 1
    assert len(captured) == 2 and all(
        body["messages"] == payload()["messages"] for body in captured
    )
