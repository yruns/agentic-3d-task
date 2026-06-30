import asyncio
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from adapter.mapping import (
    build_chat_completions_body,
    build_compaction_response,
    build_modelhub_payload,
    build_responses_payload,
    extract_text,
    iter_chat_sse_as_responses,
    iter_sse_events,
    sanitize_encrypted_state,
)
from adapter.proxy import (
    AdapterSettings,
    ModelHubUpstream,
    build_upstream_request,
    resolve_modelhub_upstream,
)


@contextmanager
def _env(values: dict[str, str], *, remove: tuple[str, ...] = ()):
    old_values = {key: os.environ.get(key) for key in (*values, *remove)}
    try:
        for key in remove:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class MappingTest(unittest.TestCase):
    def test_build_modelhub_payload_maps_responses_input_to_modelhub_messages(self):
        request = {
            "model": "gpt-5.5-2026-04-24",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is 1+1?"},
                        {"type": "text", "text": "Return only the number."},
                    ],
                }
            ],
            "instructions": "You are concise.",
            "max_output_tokens": 500,
            "stream": True,
        }

        payload = build_modelhub_payload(request)

        self.assertEqual(
            payload,
            {
                "stream": False,
                "model": "gpt-5.5-2026-04-24",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are concise."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 1+1?"},
                            {"type": "text", "text": "Return only the number."},
                        ],
                    },
                ],
            },
        )

    def test_build_modelhub_payload_maps_responses_tools_to_chat_tools(self):
        request = {
            "model": "gpt-5.5-2026-04-24",
            "input": "Use the tool.",
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                        },
                        "required": ["cmd"],
                    },
                }
            ],
        }

        payload = build_modelhub_payload(request)

        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "Run a shell command.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cmd": {"type": "string"},
                            },
                            "required": ["cmd"],
                        },
                    },
                }
            ],
        )

    def test_build_responses_payload_maps_chat_completion_text(self):
        modelhub_response = {
            "id": "chatcmpl_test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "2",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        response = build_responses_payload(modelhub_response, "gpt-5.5-2026-04-24")

        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["model"], "gpt-5.5-2026-04-24")
        self.assertEqual(response["output_text"], "2")
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["content"][0]["type"], "output_text")
        self.assertEqual(response["output"][0]["content"][0]["text"], "2")
        self.assertEqual(
            response["usage"],
            {
                "input_tokens": 10,
                "output_tokens": 1,
                "total_tokens": 11,
            },
        )

    def test_extract_text_supports_common_modelhub_shapes(self):
        self.assertEqual(
            extract_text({"output_text": "from output_text"}), "from output_text"
        )
        self.assertEqual(extract_text({"text": "from text"}), "from text")
        self.assertEqual(
            extract_text({"data": {"output": "from data output"}}), "from data output"
        )
        self.assertEqual(
            extract_text(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [{"type": "text", "text": "from list"}]
                            }
                        }
                    ]
                }
            ),
            "from list",
        )

    def test_iter_sse_events_emits_responses_stream_events(self):
        response = build_responses_payload(
            {"choices": [{"message": {"content": "hello"}}]},
            "gpt-5.5-2026-04-24",
        )

        events = list(iter_sse_events(response))

        self.assertEqual(events[-1], "data: [DONE]\n\n")
        payloads = [
            json.loads(event.removeprefix("data: ").strip())
            for event in events
            if event != "data: [DONE]\n\n"
        ]
        self.assertEqual(
            [payload["type"] for payload in payloads],
            [
                "response.created",
                "response.output_text.delta",
                "response.output_text.done",
                "response.completed",
            ],
        )
        self.assertEqual(payloads[1]["delta"], "hello")
        self.assertEqual(payloads[3]["response"]["output_text"], "hello")

    def test_build_chat_completions_body_preserves_tool_pairs_and_drops_unpaired_calls(
        self,
    ):
        request = {
            "model": "gpt-5.5-2026-04-24",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "start"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_keep",
                    "name": "shell",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_keep",
                    "output": "/tmp/project",
                },
                {
                    "type": "function_call",
                    "call_id": "call_drop",
                    "name": "apply_patch",
                    "arguments": '{"patch":"..."}',
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ],
            "max_output_tokens": 64,
        }

        payload = build_chat_completions_body(request, max_output_tokens=65536)

        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_keep",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_keep",
                    "content": "/tmp/project",
                },
                {"role": "user", "content": "continue"},
            ],
        )
        self.assertEqual(payload["max_tokens"], 65536)
        self.assertNotIn("call_drop", json.dumps(payload, ensure_ascii=False))

    def test_iter_chat_sse_as_responses_emits_complete_lifecycle(self):
        async def chunks():
            yield 'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
            yield "data: [DONE]\n\n"

        async def collect():
            return [
                chunk.decode("utf-8")
                async for chunk in iter_chat_sse_as_responses(chunks())
            ]

        events = asyncio.run(collect())
        payloads = [
            json.loads(event.removeprefix("data: ").strip())
            for event in events
            if event != "data: [DONE]\n\n"
        ]

        self.assertEqual(
            [payload["type"] for payload in payloads],
            [
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(payloads[-1]["response"]["output_text"], "hello")

    def test_sanitize_encrypted_state_removes_opaque_codex_state(self):
        request = {
            "model": "gpt-5.4-2026-03-05",
            "previous_response_id": "resp_old",
            "input": [
                {"type": "message", "role": "user", "content": "continue"},
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "gAAA",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "visible"}],
                    "encrypted_content": "gBBB",
                },
            ],
        }

        sanitized, stats = sanitize_encrypted_state(request)

        self.assertNotIn("previous_response_id", sanitized)
        self.assertNotIn("encrypted_content", json.dumps(sanitized, ensure_ascii=False))
        self.assertEqual(stats.removed_encrypted_content_count, 2)
        self.assertEqual(stats.dropped_empty_reasoning_count, 1)
        self.assertTrue(stats.removed_previous_response_id)
        self.assertEqual(len(sanitized["input"]), 2)

    def test_build_upstream_request_uses_office_chat_adapter_and_extra_header(self):
        settings = AdapterSettings(
            upstream_env="office",
            upstream_api="chat_completions",
            chat_completions_models=("gpt-5.5*",),
            modelhub_ak="ak-1",
        )

        upstream = build_upstream_request(
            {"model": "gpt-5.5-2026-04-24", "input": "hello", "max_output_tokens": 10},
            settings=settings,
            upstream_extra={"session_id": "codex-demo", "source": "local"},
        )

        self.assertEqual(
            upstream.url,
            "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl?ak=ak-1",
        )
        self.assertEqual(upstream.upstream_api, "chat_completions")
        self.assertEqual(
            upstream.body["messages"], [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(upstream.body["max_tokens"], 65536)
        self.assertEqual(
            json.loads(upstream.headers["extra"]),
            {"session_id": "codex-demo", "source": "local"},
        )

    def test_build_upstream_request_routes_gpt54_to_responses_by_default(self):
        request = {
            "model": "gpt-5.4-2026-03-05",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "opaque-state",
                },
            ],
            "previous_response_id": "resp_previous",
            "max_output_tokens": 123,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        upstream = build_upstream_request(
            request,
            settings=AdapterSettings(modelhub_ak="ak-1"),
        )

        self.assertEqual(
            upstream.url,
            "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/responses?ak=ak-1",
        )
        self.assertEqual(upstream.upstream_api, "responses")
        self.assertEqual(upstream.body, request)

    def test_build_upstream_request_uses_sticky_key_pool(self):
        settings = AdapterSettings(
            upstream_env="office",
            upstream_api="auto",
            chat_completions_models=("gpt-5.5*",),
            modelhub_ak_pool=(("k1", "ak-1"), ("k2", "ak-2"), ("k3", "ak-3")),
        )
        request = {"model": "gpt-5.5-2026-04-24", "input": "hello"}
        extra = {"session_id": "codex-sticky-session"}

        first = build_upstream_request(request, settings=settings, upstream_extra=extra)
        second = build_upstream_request(
            request, settings=settings, upstream_extra=extra
        )

        self.assertEqual(first.url, second.url)
        self.assertEqual(first.upstream_key_alias, second.upstream_key_alias)
        self.assertIn(first.upstream_key_alias, {"k1", "k2", "k3"})

    def test_build_upstream_request_uses_toml_weighted_upstream_for_matching_model(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "modelhub_upstreams.toml"
            toml_path.write_text(
                """
[[upstreams]]
url = "https://modelhub-a.example.test/api/modelhub/online"
model_name = "gpt-5.5-2026-04-24"
ak = "ak-from-toml"
weight = 3
""".strip(),
                encoding="utf-8",
            )

            with _env(
                {"AIDP_MODELHUB_UPSTREAMS_TOML": str(toml_path)},
                remove=(
                    "AIDP_GPT_AK",
                    "AIDP_MODELHUB_AK",
                    "MODELHUB_AK",
                    "CASE_REVIEW_LLM_AK",
                    "AIDP_MODELHUB_AK_POOL",
                ),
            ):
                settings = AdapterSettings.from_env()

        upstream = build_upstream_request(
            {"model": "gpt-5.5-2026-04-24", "input": "hello"},
            settings=settings,
            upstream_extra={"session_id": "codex-demo"},
        )

        self.assertEqual(
            upstream.url,
            "https://modelhub-a.example.test/api/modelhub/online/responses?ak=ak-from-toml",
        )
        self.assertEqual(upstream.upstream_api, "responses")
        self.assertEqual(upstream.upstream_key_alias, "u1")
        self.assertEqual(upstream.upstream_key_selection, "toml_weighted_extra_hash")

    def test_toml_weighted_upstreams_use_chat_run_id_before_session_id(self):
        settings = AdapterSettings(
            upstream_api="chat_completions",
            modelhub_upstreams=(
                ModelHubUpstream(
                    alias="a",
                    url="https://modelhub-a.example.test/api/modelhub/online",
                    model_name="gpt-5.4-2026-03-05",
                    ak="ak-a",
                    weight=1,
                ),
                ModelHubUpstream(
                    alias="b",
                    url="https://modelhub-b.example.test/api/modelhub/online",
                    model_name="gpt-5.4-2026-03-05",
                    ak="ak-b",
                    weight=1,
                ),
            ),
        )

        aliases = {
            resolve_modelhub_upstream(
                settings,
                {"model": "gpt-5.4-2026-03-05"},
                {
                    "session_id": "fixed-prefix-cache-session",
                    "chat_run_id": f"turn-{index}",
                },
            ).alias
            for index in range(200)
        }

        self.assertEqual(aliases, {"a", "b"})

    def test_toml_upstreams_fail_closed_when_no_model_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "modelhub_upstreams.toml"
            toml_path.write_text(
                """
[[upstreams]]
url = "https://modelhub-a.example.test/api/modelhub/online"
model_name = "gpt-5.5-2026-04-24"
ak = "ak-from-toml"
weight = 1
""".strip(),
                encoding="utf-8",
            )

            with _env(
                {"AIDP_MODELHUB_UPSTREAMS_TOML": str(toml_path)},
                remove=(
                    "AIDP_GPT_AK",
                    "AIDP_MODELHUB_AK",
                    "MODELHUB_AK",
                    "CASE_REVIEW_LLM_AK",
                    "AIDP_MODELHUB_AK_POOL",
                ),
            ):
                settings = AdapterSettings.from_env()

        with self.assertRaisesRegex(
            RuntimeError, "No ModelHub TOML upstream matches model 'gpt-5.4-2026-03-05'"
        ):
            build_upstream_request(
                {"model": "gpt-5.4-2026-03-05", "input": "hello"},
                settings=settings,
                upstream_extra={"session_id": "codex-demo"},
            )

    def test_build_compaction_response_returns_local_checkpoint(self):
        payload = build_compaction_response(
            {
                "model": "gpt-5.5-2026-04-24",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello" * 100}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_demo",
                        "name": "search",
                        "arguments": '{"q":"demo"}',
                    },
                ],
            },
            max_chars=180,
        )

        self.assertEqual(payload["object"], "response.compaction")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output"][0]["type"], "message")
        text = payload["output"][0]["content"][0]["text"]
        self.assertTrue(text.startswith("CONTEXT CHECKPOINT SUMMARY"))
        self.assertIn("AIDP Codex proxy compatibility", text)
        self.assertLessEqual(len(text), 180)


if __name__ == "__main__":
    unittest.main()
