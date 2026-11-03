"""Run a bounded, credential-safe real Codex dynamic-tool integration probe.

The caller supplies an external process timeout (recommended: 240 seconds).
No model request is made on import. Credentials are passed only in child env.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import re
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import openai_codex
from openai_codex import CodexConfig, Thread
from openai_codex._run import _collect_turn_result
from openai_codex.client import CodexClient
from openai_codex.models import Notification
from pydantic import BaseModel

from probe_support import credential, digest, save, state_fields

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemini-3.8-flash"
PROVIDER = "gemini_probe_adapter"
BASE_INSTRUCTIONS = (
    "Perform the user's protocol test using only lookup_record. "
    "Call requested records sequentially and wait for each result. "
    "Preserve context markers and receipts exactly. Return the requested JSON."
)
TOOL = {
    "type": "function",
    "name": "lookup_record",
    "description": "Return a record's value and verification receipt.",
    "inputSchema": {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["alpha", "beta"]}},
        "required": ["name"],
        "additionalProperties": False,
    },
    "deferLoading": False,
}


def json_value(value: Any) -> Any:
    """Convert SDK boundary models into JSON without stringifying exceptions."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: json_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported SDK boundary value")


def provenance() -> dict[str, Any]:
    """Identify clean tracked code plus actual probe and SDK source bytes."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("Tracked execution worktree must be clean")
    sdk_root = Path(openai_codex.__file__).resolve().parent
    local_sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    sdk_sources = sorted(sdk_root.rglob("*.py"))
    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "tracked_drift": False,
        "worktree": str(ROOT),
        "sdk_version": importlib.metadata.version("openai-codex"),
        "local_python_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in local_sources
        },
        "sdk_python_sha256": {
            str(p.relative_to(sdk_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sdk_sources
        },
    }


def model_catalog() -> dict[str, Any]:
    """Supply exact custom-model metadata without inherited GPT instructions."""
    return {
        "models": [
            {
                "slug": MODEL,
                "display_name": "Gemini protocol probe",
                "description": None,
                "supported_reasoning_levels": [],
                "shell_type": "disabled",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 1,
                "upgrade": None,
                "model_messages": {"instructions_template": BASE_INSTRUCTIONS},
                "include_skills_usage_instructions": False,
                "include_plugin_usage_instructions": False,
                "include_apps_usage_instructions": False,
                "supports_reasoning_summary_parameter": False,
                "default_reasoning_summary": "none",
                "support_verbosity": False,
                "default_verbosity": None,
                "apply_patch_tool_type": None,
                "truncation_policy": {"mode": "bytes", "limit": 10000},
                "supports_image_detail_original": False,
                "context_window": None,
                "max_context_window": None,
                "auto_compact_token_limit": None,
                "effective_context_window_percent": 95,
                "experimental_supported_tools": [],
                "node_repl_disabled": True,
                "tool_mode": "direct",
                "multi_agent_version": "disabled",
            }
        ]
    }


def config_toml(adapter_url: str, catalog_path: Path) -> str:
    """Build non-secret provider/config text; headers refer only to env names."""
    disabled_features = (
        "shell_tool",
        "view_image",
        "sleep_tool",
        "unified_exec",
        "stable_environment_tools",
        "deferred_executor",
        "code_mode",
        "js_repl",
        "multi_agent",
        "multi_agent_v2",
        "multi_agent_v2_dynamic_tools",
        "agent_message_board",
        "apps",
        "plugins",
        "hooks",
        "plugin_hooks",
        "goals",
        "tool_search",
        "tool_suggest",
        "skill_search",
        "context_management",
        "async_compaction",
        "responses_websockets",
        "responses_websockets_v2",
        "external_agent_memory_import",
    )
    text = (
        f"model = {json.dumps(MODEL)}\nmodel_provider = {json.dumps(PROVIDER)}\n"
        f"model_catalog_json = {json.dumps(str(catalog_path))}\n"
        'project_doc_max_bytes = 0\nweb_search = "disabled"\n'
        'approval_policy = "never"\nsandbox_mode = "read-only"\n'
        "model_post_turn_compact_threshold_percent = 0\n"
        "[skills]\ninclude_instructions = false\n"
        "[skills.bundled]\nenabled = false\n"
        "[cloud.skills]\nenabled = false\n"
        "[orchestrator.mcp]\nenabled = false\n"
        "[tools.update_plan]\nenabled = false\n"
        "[tools.experimental_request_user_input]\nenabled = false\n"
        "[features]\n"
    )
    text += "".join(f"{feature} = false\n" for feature in disabled_features)
    text += (
        f"[model_providers.{PROVIDER}]\n"
        'name = "Isolated Gemini adapter"\n'
        f"base_url = {json.dumps(adapter_url)}\n"
        'wire_api = "responses"\nrequires_openai_auth = false\n'
        "supports_websockets = false\nrequest_max_retries = 0\nstream_max_retries = 0\n"
        f"[model_providers.{PROVIDER}.env_http_headers]\n"
        'X-Codex-Upstream-AK = "PROBE_UPSTREAM_AK"\n'
        'X-Codex-Session-ID = "PROBE_SESSION_ID"\n'
    )
    return text


def parse_answer(text: str | None) -> tuple[Any, str]:
    """Accept plain JSON or one complete JSON fence, without repairing content."""
    if text is None:
        return None, "missing"
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", stripped, re.DOTALL)
    if match:
        stripped = match.group(1)
    try:
        return json.loads(stripped), "markdown_json" if match else "plain"
    except json.JSONDecodeError:
        return None, "invalid_json"


def run(args: argparse.Namespace, ak: str) -> int:
    """Execute two turns of the same real Codex thread and retain safe evidence."""
    parsed_url = urlparse(args.adapter_url)
    if (
        parsed_url.scheme != "http"
        or parsed_url.hostname not in {"127.0.0.1", "localhost"}
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
    ):
        raise ValueError(
            "Adapter URL must be a local HTTP URL without credentials/query"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    (output / "events").mkdir()
    session_id = "gemini_codex_sdk_20261103_" + uuid.uuid4().hex[:12]
    receipts = {name: "receipt-" + uuid.uuid4().hex[:12] for name in ("alpha", "beta")}
    expected = {
        "marker": "context-" + uuid.uuid4().hex[:12],
        "sum": 348,
        "alpha_receipt": receipts["alpha"],
        "beta_receipt": receipts["beta"],
    }
    summary: dict[str, Any] = {
        **provenance(),
        "run_id": session_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "model": MODEL,
        "adapter_url": args.adapter_url,
        "credential_source": "Current chat user credential; memory/env only",
        "same_credential_all_requests": True,
        "scope": "Real Codex SDK/app-server -> isolated Responses adapter -> ModelHub Gemini",
        "result": "running",
        "expected": expected,
        "turns": [],
        "tool_executions": [],
        "compact_disabled": True,
        "native_tools_disabled": True,
        "dynamic_tools": ["lookup_record"],
        "effort": "omitted: provider default",
    }
    save(output / "summary.json", summary, ak)
    current_turn = 0
    event_counter = 0
    expected_order = ["alpha", "beta", "alpha"]

    def server_request(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if method != "item/tool/call" or not isinstance(params, dict):
            raise RuntimeError("Unexpected SDK server request")
        arguments = params.get("arguments")
        if (
            params.get("tool") != "lookup_record"
            or params.get("namespace") is not None
            or not isinstance(arguments, dict)
            or set(arguments) != {"name"}
            or arguments["name"] not in receipts
        ):
            raise ValueError("Invalid dynamic tool call")
        call_index = len(summary["tool_executions"])
        if (
            call_index >= len(expected_order)
            or arguments["name"] != expected_order[call_index]
        ):
            raise ValueError("Unexpected tool order/count")
        if (current_turn == 1 and call_index >= 2) or (
            current_turn == 2 and call_index < 2
        ):
            raise ValueError("Unexpected tool turn")
        name = arguments["name"]
        result = {
            "name": name,
            "value": 137 if name == "alpha" else 211,
            "receipt": receipts[name],
        }
        summary["tool_executions"].append(
            {
                "turn_number": current_turn,
                "thread_id": params.get("threadId"),
                "turn_id": params.get("turnId"),
                "call_id": params.get("callId"),
                "tool": params["tool"],
                "arguments": arguments,
                "result": result,
            }
        )
        save(output / "tool_executions.json", summary["tool_executions"], ak)
        return {
            "success": True,
            "contentItems": [{"type": "inputText", "text": json.dumps(result)}],
        }

    def record_events(stream: Iterator[Notification]) -> Iterator[Notification]:
        nonlocal event_counter
        for event in stream:
            event_counter += 1
            record = {"turn_number": current_turn, **json_value(event)}
            record["state_fields"] = state_fields(record)
            save(output / "events" / f"{event_counter:05d}.json", record, ak)
            if "compact" in event.method.lower():
                raise RuntimeError("Unexpected compaction event")
            yield event

    prompt = (
        f"The original marker is {expected['marker']}. First call lookup_record for alpha, "
        "wait for its result, then call lookup_record for beta and wait for its result. "
        "Do not call both together. Return only a JSON object with marker (the original "
        "marker), sum (sum of the two returned values), alpha_receipt and beta_receipt "
        "(exact tool receipts). Do not invent receipts. Use no other tools."
    )
    if args.prompt_file:
        extra = args.prompt_file.read_text()
        summary["extra_prompt_sha256"] = hashlib.sha256(extra.encode()).hexdigest()
        prompt += (
            "\nAdditional task instructions (keep the required tool order and final schema):\n"
            + extra
        )
    followup = (
        "Verify by calling lookup_record for alpha exactly once more, wait for its result, "
        "then repeat the SAME JSON using the ORIGINAL marker, sum, alpha_receipt and "
        "beta_receipt from our conversation. Use no other tools."
    )
    save(output / "prompts.json", {"initial": prompt, "followup": followup}, ak)
    started_at = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="gemini-sdk-probe-") as temporary:
            root = Path(temporary)
            home, codex_home, cwd = (
                root / name for name in ("home", "codex_home", "cwd")
            )
            for path in (home, codex_home, cwd):
                path.mkdir(mode=0o700)
            catalog_path = root / "models.json"
            catalog_path.write_text(json.dumps(model_catalog()))
            config_text = config_toml(args.adapter_url, catalog_path)
            (codex_home / "config.toml").write_text(config_text)
            save(output / "model_catalog.json", model_catalog(), ak)
            save(output / "runtime_config.json", {"toml": config_text}, ak)
            config = CodexConfig(
                cwd=str(cwd),
                experimental_api=True,
                env={
                    "CODEX_HOME": str(codex_home),
                    "HOME": str(home),
                    "RUST_LOG": "warn",
                    "PROBE_UPSTREAM_AK": ak,
                    "PROBE_SESSION_ID": session_id,
                },
            )
            with CodexClient(config, approval_handler=server_request) as client:
                summary["cli_version"] = client._runtime_version
                started = client.thread_start(
                    {
                        "model": MODEL,
                        "modelProvider": PROVIDER,
                        "cwd": str(cwd),
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                        "baseInstructions": BASE_INSTRUCTIONS,
                        "environments": [],
                        "dynamicTools": [TOOL],
                        "allowProviderModelFallback": False,
                    }
                )
                summary["thread_id"] = started.thread.id
                save(output / "thread_start.json", json_value(started), ak)
                thread = Thread(client, started.thread.id)
                for current_turn, turn_prompt in enumerate((prompt, followup), start=1):
                    handle = thread.turn(turn_prompt)
                    stream = handle.stream()
                    try:
                        result = _collect_turn_result(
                            record_events(stream), turn_id=handle.id
                        )
                    finally:
                        stream.close()
                    result_payload = json_value(result)
                    answer, answer_format = parse_answer(result.final_response)
                    result_payload.update(
                        {
                            "answer": answer,
                            "answer_format": answer_format,
                            "answer_matches_expected": answer == expected,
                        }
                    )
                    save(output / f"turn_{current_turn}.json", result_payload, ak)
                    summary["turns"].append(result_payload)
                    save(output / "summary.json", summary, ak)
                    if answer != expected:
                        raise ValueError(
                            "Final answer did not match expected marker/receipts/sum"
                        )
            summary["tool_order_correct"] = [
                x["arguments"]["name"] for x in summary["tool_executions"]
            ] == expected_order
            if not summary["tool_order_correct"]:
                raise ValueError("Tool execution order/count mismatch")
            summary["result"] = "passed"
    except Exception as exc:
        summary["result"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error_details_suppressed"] = True
    finally:
        summary["elapsed_s"] = round(time.monotonic() - started_at, 3)
        summary["event_count"] = event_counter
        save(output / "summary.json", summary, ak)
    print(
        json.dumps(
            {
                "result": summary["result"],
                "run_id": session_id,
                "turns": len(summary["turns"]),
                "tools": len(summary["tool_executions"]),
            }
        ),
        flush=True,
    )
    return 0 if summary["result"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-url", default="http://127.0.0.1:8791/v1")
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    try:
        return run(args, credential(args.credential_transcript))
    except Exception as exc:
        print(
            json.dumps(
                {"fatal_error_type": type(exc).__name__, "details_suppressed": True}
            ),
            flush=True,
        )
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
