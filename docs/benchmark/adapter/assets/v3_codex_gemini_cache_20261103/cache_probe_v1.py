"""Direct ModelHub cache controls, kept separate from the real Codex chain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from probe_support import credential, digest, save, state_fields

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
MODEL = "gemini-3.8-flash"
PUZZLE = (
    "Solve the base-10 alphametic CROSS + ROADS = DANGER. Different letters are "
    "different digits; leading letters are nonzero. Do not assume a memorized "
    "answer: check column constraints and verify the solution carefully internally. "
    "Use no tools or code. Output only one valid digit mapping and its arithmetic "
    "check, with no reasoning text."
)


def remove_signatures(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: remove_signatures(v) for k, v in value.items() if k != "signature"}
    if isinstance(value, list):
        return [remove_signatures(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-transcript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--case", choices=["visible_prefix", "reasoning"], required=True
    )
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    ak = credential(args.credential_transcript)
    args.output.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT):
        raise RuntimeError("Execution worktree must have no tracked drift")
    sid = "gemini_cache_20261103_" + uuid.uuid4().hex[:12]
    records: list[dict[str, Any]] = []
    summary = {
        "run_id": sid,
        "case": args.case,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "session_id": sid,
        "same_ak": True,
        "result": "running",
        "requests": records,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_drift": False,
        "worktree": str(ROOT),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("probe_support.py")]
        },
        "scope": "Direct HTTP cache controls; this is not the Codex SDK integration run",
    }

    with httpx.Client(
        timeout=httpx.Timeout(180.0, connect=20.0), follow_redirects=False
    ) as client:

        def request(label: str, body: dict[str, Any]) -> dict[str, Any] | None:
            index = len(records) + 1
            logid = f"{sid}_{index}"
            headers = {
                "Content-Type": "application/json",
                "X-TT-LOGID": logid,
                "extra": json.dumps(
                    {
                        "session_id": sid,
                        "chat_run_id": sid,
                        "source": "cache_diagnostic",
                    }
                ),
            }
            save(
                args.output / f"request_{index}.json",
                {"label": label, "headers": headers, "body": body},
                ak,
            )
            item: dict[str, Any] = {
                "index": index,
                "label": label,
                "logid": logid,
                "body_sha256": digest(body),
                "input_state_fields": state_fields(body),
            }
            started = time.monotonic()
            try:
                response = client.post(
                    ENDPOINT, params={"ak": ak}, headers=headers, json=body
                )
                item["http_status"] = response.status_code
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"non_json_body": response.text[:2000]}
                save(args.output / f"response_{index}.json", payload, ak)
                item["usage"] = payload.get("usage")
                item["error"] = payload.get("error")
                item["response_state_fields"] = state_fields(payload)
                choices = payload.get("choices", [])
                item["finish_reason"] = (
                    choices[0].get("finish_reason") if choices else None
                )
                item["response_model"] = payload.get("model")
            except httpx.HTTPError as exc:
                item["error_type"] = type(exc).__name__
                payload = None
            item["elapsed_s"] = round(time.monotonic() - started, 3)
            records.append(item)
            save(args.output / "summary.json", summary, ak)
            print(
                json.dumps(
                    {
                        "label": label,
                        "status": item.get("http_status"),
                        "elapsed_s": item["elapsed_s"],
                        "usage": item.get("usage"),
                        "finish": item.get("finish_reason"),
                    }
                ),
                flush=True,
            )
            return payload if item.get("http_status") == 200 else None

        if args.case == "visible_prefix":
            prefix = "Fixed reference data. Do not analyze or repeat it.\n" + "\n".join(
                f"Record {i:04d}: alpha={i * 17 + 3}; beta={i * 31 + 9}; label=cedar meadow silver river; checksum={i * 47 + 11}."
                for i in range(240)
            )
            body = {
                "model": MODEL,
                "stream": False,
                "max_tokens": 256,
                "messages": [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": "Reply READY only."},
                ],
            }
            for repeat in range(3):
                if request(f"visible_prefix_repeat_{repeat + 1}", body) is None:
                    break
        else:
            body = {
                "model": MODEL,
                "stream": False,
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": PUZZLE}],
            }
            first = request("short_prompt_reasoning_seed", body)
            if first and records[-1]["finish_reason"] == "stop":
                message = first["choices"][0]["message"]
                replay = {
                    "model": MODEL,
                    "stream": False,
                    "max_tokens": 128,
                    "messages": [
                        *copy.deepcopy(body["messages"]),
                        copy.deepcopy(message),
                        {
                            "role": "user",
                            "content": "Acknowledge the previous solution with OK only. Do not redo the calculation.",
                        },
                    ],
                }
                summary["seed_message_sha256"] = digest(message)
                for repeat in range(3):
                    if request(f"full_state_repeat_{repeat + 1}", replay) is None:
                        break
                stripped = remove_signatures(replay)
                request("without_signatures_control", stripped)
                # No signature bytes enter text in this independent visible-only
                # accounting control. It is not treated as stateful continuation.
                visible = {
                    "model": MODEL,
                    "stream": False,
                    "max_tokens": 128,
                    "messages": [
                        {"role": "user", "content": PUZZLE},
                        {"role": "assistant", "content": message.get("content") or ""},
                        copy.deepcopy(replay["messages"][-1]),
                    ],
                }
                request("visible_history_only_accounting", visible)
    summary["result"] = (
        "completed"
        if records
        and all(
            r.get("http_status") == 200
            for r in records
            if "control" not in r["label"] and "accounting" not in r["label"]
        )
        else "partial_or_failed"
    )
    save(args.output / "summary.json", summary, ak)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {"fatal_error_type": type(exc).__name__, "details_suppressed": True}
            ),
            flush=True,
        )
        raise SystemExit(1) from None
