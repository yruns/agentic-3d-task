"""Real Codex SDK seed-reasoning and eight acknowledgement cache probe.

Invoke under an external 480-second timeout. Importing this module does not
start a client or issue requests. A passed run means nine completed SDK turns,
not proof of a cache hit or a correct puzzle answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cache_probe import PUZZLE
from openai_codex import CodexConfig, Thread
from openai_codex._run import _collect_turn_result
from openai_codex.client import CodexClient
from openai_codex.models import Notification
from probe_support import credential, save, state_fields
from sdk_driver import (
    MODEL,
    PROVIDER,
    ROOT,
    config_toml,
    json_value,
    model_catalog,
    provenance,
)

BASE_INSTRUCTIONS = (
    "Follow the user task exactly. Work internally; do not print reasoning."
)
ACKNOWLEDGEMENT = (
    "Acknowledge the preceding solution with OK only; do not redo the calculation."
)
REPEATS = 8
DELAY_SECONDS = 2.0


def run(args: argparse.Namespace, ak: str) -> int:
    """Use one fixed provider/header binding and a single persistent SDK thread."""
    parsed_url = urlparse(args.adapter_url)
    second_puzzle = (
        "Solve a different base-10 alphametic, DONALD + GERALD = ROBERT. "
        "Different letters are distinct digits and leading letters are nonzero. "
        "Verify all column constraints internally without tools or code. "
        "Output only a valid digit mapping and arithmetic check, no reasoning text."
    )
    prompts = [("seed", PUZZLE)]
    if args.second_seed:
        prompts.append(("second_seed", second_puzzle))
    if args.third_seed:
        prompts.append(
            (
                "third_seed",
                "Solve the base-10 alphametic SATURN + URANUS = PLANETS. All distinct letters have distinct digits, with no leading zero. Verify the column carries carefully internally without tools or code, then output only one valid digit mapping and the arithmetic check. Do not output reasoning text.",
            )
        )
    prompts.extend(("acknowledgement", ACKNOWLEDGEMENT) for _ in range(REPEATS))
    if (
        parsed_url.scheme != "http"
        or parsed_url.hostname not in {"127.0.0.1", "localhost"}
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
    ):
        raise ValueError("Adapter URL must be local HTTP without credentials/query")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    (output / "events").mkdir()
    session_id = "gemini_sdk_cache_20261103_" + uuid.uuid4().hex[:12]
    summary: dict[str, Any] = {
        "run_id": session_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "model": MODEL,
        "adapter_url": args.adapter_url,
        "same_credential_all_requests": True,
        "credential_source": "Current chat user credential; memory/env only",
        "scope": "Real Codex SDK -> isolated Responses adapter -> ModelHub Gemini",
        "case": "short_puzzle_reasoning_then_eight_same_thread_acknowledgements",
        "result": "running",
        "pass_definition": "All planned seed and acknowledgement turns completed with nonempty final responses",
        "second_seed": args.second_seed,
        "third_seed": args.third_seed,
        "planned_turn_count": len(prompts),
        "puzzle_accuracy_evaluated": False,
        "cache_hit_required_for_pass": False,
        "dynamic_tools": [],
        "native_tools_disabled": True,
        "repeats": REPEATS,
        "repeat_delay_seconds": DELAY_SECONDS,
        "seed_prompt_sha256": hashlib.sha256(PUZZLE.encode()).hexdigest(),
        "turns": [],
        "sdk_opaque_state_fields": [],
    }
    save(output / "summary.json", summary, ak)
    event_counter = 0
    current_turn = 0
    client: CodexClient | None = None
    started_at = time.monotonic()

    def server_request(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        save(
            output / "unexpected_server_request.json",
            {"method": method, "params": params, "turn_number": current_turn},
            ak,
        )
        raise RuntimeError("No SDK server requests or tools are expected in this probe")

    def record_events(stream: Iterator[Notification]) -> Iterator[Notification]:
        nonlocal event_counter
        for event in stream:
            event_counter += 1
            record = {"turn_number": current_turn, **json_value(event)}
            fields = state_fields(record)
            record["state_fields"] = fields
            opaque = [field for field in fields if field.get("opaque_name")]
            if opaque:
                summary["sdk_opaque_state_fields"].append(
                    {
                        "turn_number": current_turn,
                        "event_index": event_counter,
                        "method": event.method,
                        "fields": opaque,
                    }
                )
            save(output / "events" / f"{event_counter:05d}.json", record, ak)
            if "compact" in event.method.lower():
                raise RuntimeError("Unexpected compaction event")
            yield event

    try:
        summary.update(provenance())
        save(output / "summary.json", summary, ak)
        save(
            output / "prompts.json",
            {
                "base_instructions": BASE_INSTRUCTIONS,
                "seed": PUZZLE,
                "acknowledgement": ACKNOWLEDGEMENT,
            },
            ak,
        )
        runtime_root = ROOT / "tmp"
        runtime_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="gemini-sdk-cache-", dir=runtime_root
        ) as temporary:
            root = Path(temporary)
            home, codex_home, cwd = (
                root / name for name in ("home", "codex_home", "cwd")
            )
            for path in (home, codex_home, cwd):
                path.mkdir(mode=0o700)
            catalog = model_catalog()
            catalog["models"][0]["model_messages"] = {
                "instructions_template": BASE_INSTRUCTIONS
            }
            catalog_path = root / "models.json"
            catalog_path.write_text(json.dumps(catalog))
            config_text = config_toml(args.adapter_url, catalog_path)
            (codex_home / "config.toml").write_text(config_text)
            save(output / "model_catalog.json", catalog, ak)
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
                initialized = client.initialize()
                summary["cli_version"] = client._runtime_version
                save(output / "initialize.json", json_value(initialized), ak)
                started = client.thread_start(
                    {
                        "model": MODEL,
                        "modelProvider": PROVIDER,
                        "cwd": str(cwd),
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                        "baseInstructions": BASE_INSTRUCTIONS,
                        "environments": [],
                        "dynamicTools": [],
                        "allowProviderModelFallback": False,
                        "experimentalRawEvents": True,
                    }
                )
                summary["thread_id"] = started.thread.id
                save(output / "thread_start.json", json_value(started), ak)
                thread = Thread(client, started.thread.id)
                for current_turn, (turn_kind, turn_prompt) in enumerate(
                    prompts, start=1
                ):
                    if current_turn > 1:
                        time.sleep(DELAY_SECONDS)
                    turn_started_at = time.monotonic()
                    handle = thread.turn(turn_prompt)
                    stream = handle.stream()
                    try:
                        result = _collect_turn_result(
                            record_events(stream), turn_id=handle.id
                        )
                    finally:
                        stream.close()
                    payload = json_value(result)
                    final = result.final_response
                    completed = payload["status"] == "completed" and bool(
                        final and final.strip()
                    )
                    payload.update(
                        {
                            "turn_number": current_turn,
                            "kind": turn_kind,
                            "completed_with_answer": completed,
                            "acknowledgement_is_ok": (
                                final.strip() == "OK"
                                if final is not None and turn_kind == "acknowledgement"
                                else None
                            ),
                            "elapsed_s": round(time.monotonic() - turn_started_at, 3),
                        }
                    )
                    save(output / f"turn_{current_turn}.json", payload, ak)
                    summary["turns"].append(payload)
                    summary["event_count"] = event_counter
                    save(output / "summary.json", summary, ak)
                    print(
                        json.dumps(
                            {
                                "run_id": session_id,
                                "turn": current_turn,
                                "status": payload["status"],
                                "acknowledgement_is_ok": payload[
                                    "acknowledgement_is_ok"
                                ],
                            }
                        ),
                        flush=True,
                    )
                    if not completed:
                        raise RuntimeError(
                            "SDK turn did not complete with a final answer"
                        )
            summary["all_acknowledgements_ok"] = all(
                turn["acknowledgement_is_ok"]
                for turn in summary["turns"]
                if turn["kind"] == "acknowledgement"
            )
            summary["result"] = "passed"
    except Exception as exc:
        summary["result"] = "failed"
        summary["error_type"] = type(exc).__name__
        save(
            output / "failure_detail.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "turn_number": current_turn,
                "stderr_tail": list(getattr(client, "_stderr_lines", []))[-40:],
            },
            ak,
        )
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
                "all_acknowledgements_ok": summary.get("all_acknowledgements_ok"),
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
    parser.add_argument("--second-seed", action="store_true")
    parser.add_argument("--third-seed", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
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
