"""Regression checks for the SCOPE transport boundary."""

from __future__ import annotations

import httpx
import pytest
from openai import OpenAI

from scripts.scope.run_scope import FailClosedCompletions


def test_api_failure_exits_instead_of_entering_upstream_fallback() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": {"message": "failed"}})

    with OpenAI(
        api_key="test-token",
        base_url="http://localhost/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        completions = FailClosedCompletions(
            client.chat.completions, expected_model="gemini-3.8-flash"
        )
        with pytest.raises(SystemExit, match="HTTP 502"):
            completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_completion_exits(content: str | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 1,
                "object": "chat.completion",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    with OpenAI(
        api_key="test-token",
        base_url="http://localhost/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        completions = FailClosedCompletions(
            client.chat.completions, expected_model="gemini-3.8-flash"
        )
        with pytest.raises(SystemExit, match="empty assistant content"):
            completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )
