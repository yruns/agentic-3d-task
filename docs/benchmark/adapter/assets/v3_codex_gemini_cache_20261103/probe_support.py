"""Small, credential-safe ModelHub tool/state diagnostic (no SDK or adapter)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
MODEL = "gemini-3.8-flash"
IMAGE_URL = "https://static.wixstatic.com/media/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg/v1/fill/w_585,h_405,al_c,q_80,enc_auto/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg"
ENDPOINTS = {
    "legacy_chat": ENDPOINT,
    "unified_chat": "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/unified/v1/chat/completions",
    "responses": "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/unified/v1/responses",
}
STATE = re.compile(r"reasoning|thinking|signature|encrypted|opaque|continuation", re.I)
OPAQUE = re.compile(r"signature|encrypted|opaque", re.I)


def digest(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def credential(transcript: Path) -> str:
    """Extract only the key in this chat's user-supplied curl, never print it."""
    matches: set[str] = set()
    with transcript.open() as handle:
        for line in handle:
            entry = json.loads(line)
            payload = entry.get("payload", {})
            if entry.get("type") != "response_item" or payload.get("role") != "user":
                continue
            for block in payload.get("content", []):
                text = block.get("text", "")
                if MODEL not in text or "/v2/crawl?ak=" not in text:
                    continue
                matches.update(re.findall(r"/v2/crawl\?ak=([^\s'\"&<>]+)", text))
    if len(matches) != 1:
        raise ValueError("Expected exactly one user-provided credential")
    return matches.pop()


def field_paths(value: Any, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            paths.append(child_path)
            paths.extend(field_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(field_paths(child, f"{path}[{index}]"))
    return paths


def state_fields(value: Any, path: str = "$") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            loc = f"{path}.{key}"
            if STATE.search(key):
                found.append(
                    {
                        "path": loc,
                        "type": type(child).__name__,
                        "nonempty": child not in (None, "", [], {}),
                        "sha256": digest(child),
                        "length": (
                            len(child) if isinstance(child, (str, list, dict)) else None
                        ),
                        "opaque_name": bool(OPAQUE.search(key)),
                    }
                )
            found.extend(state_fields(child, loc))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(state_fields(child, f"{path}[{i}]"))
    return found


def scrub(value: Any, ak: str) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if STATE.search(key) and isinstance(child, str) and child:
                result[key] = {
                    "redacted_state": True,
                    "length": len(child),
                    "sha256": digest(child),
                }
            else:
                result[key] = scrub(child, ak)
        return result
    if isinstance(value, list):
        return [scrub(child, ak) for child in value]
    if isinstance(value, str):
        if "__thought__" in value:
            prefix, state = value.split("__thought__", 1)
            value = (
                prefix
                + "__thought__<redacted:"
                + str(len(state))
                + ":"
                + digest(state)
                + ">"
            )
        value = value.replace(ak, "<REDACTED_AK>").replace(
            quote(ak, safe=""), "<REDACTED_AK>"
        )
        return re.sub(r"(?i)([?&]ak=)[^\s\"'<>]+", r"\1<REDACTED_AK>", value)
    return value


def save(path: Path, value: Any, ak: str) -> None:
    rendered = json.dumps(scrub(value, ak), ensure_ascii=False, indent=2)
    if ak in rendered or quote(ak, safe="") in rendered:
        raise ValueError("Credential redaction failed")
    path.write_text(rendered + "\n")


def execute_tool(call: dict[str, Any], receipts: dict[str, str]) -> dict[str, Any]:
    function = call["function"]
    if function["name"] != "lookup_record":
        raise ValueError("Unexpected tool requested")
    args = json.loads(function["arguments"])
    name = args["name"]
    if name not in receipts or set(args) != {"name"}:
        raise ValueError("Invalid tool arguments")
    values = {"alpha": 137, "beta": 211}
    return {"name": name, "value": values[name], "receipt": receipts[name]}


def opaque_locations(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    found: list[tuple[Any, ...]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            loc = path + (key,)
            if (
                isinstance(child, str)
                and child
                and (
                    OPAQUE.search(key)
                    or (key == "data" and value.get("type") == "redacted_thinking")
                    or (
                        key in {"id", "call_id", "tool_call_id"}
                        and "__thought__" in child
                    )
                )
            ):
                found.append(loc)
            else:
                found.extend(opaque_locations(child, loc))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(opaque_locations(child, path + (index,)))
    elif (
        isinstance(value, str)
        and value
        and path
        and any(OPAQUE.search(str(k)) for k in path[-2:])
    ):
        found.append(path)
    return found


def read_path(value: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


def run_state_controls(
    client: httpx.Client,
    endpoint: str,
    body: dict[str, Any],
    headers: dict[str, str],
    output: Path,
    ak: str,
) -> list[dict[str, Any]]:
    """Deliberate negative controls; never used as production recovery."""
    records = []
    for mode in (
        "corrupt_message_only",
        "corrupt_tool_only",
        "corrupt_all_copies",
        "remove_all",
    ):
        altered = copy.deepcopy(body)
        locations = opaque_locations(altered)
        if not locations:
            continue
        if mode == "corrupt_message_only":
            selected = [
                p
                for p in locations
                if len(p) == 3 and p[0] == "messages" and p[-1] == "signature"
            ][:1]
        elif mode == "corrupt_tool_only":
            selected = [
                p for p in locations if "tool_calls" in p and p[-1] == "signature"
            ][:1]
        else:
            selected = list(reversed(locations))
        if not selected:
            records.append({"mode": mode, "skipped": "target_field_not_observed"})
            continue
        touched = []
        for path in selected:
            parent = read_path(altered, path[:-1])
            state = parent[path[-1]]
            touched.append(
                {"path": list(path), "length": len(state), "sha256": digest(state)}
            )
            if "__thought__" in state:
                return [
                    {
                        "mode": mode,
                        "skipped": "embedded_tool_id_signature_requires_paired_mutation",
                    }
                ]
            if mode.startswith("corrupt"):
                pos = len(state) // 2
                parent[path[-1]] = (
                    state[:pos] + ("A" if state[pos] != "A" else "B") + state[pos + 1 :]
                )
            else:
                del parent[path[-1]]
        control_headers = dict(headers)
        control_headers["X-TT-LOGID"] += "_" + mode
        save(
            output / f"control_{mode}_request.json",
            {"headers": control_headers, "body": altered},
            ak,
        )
        started = time.monotonic()
        try:
            response = client.post(
                endpoint, params={"ak": ak}, headers=control_headers, json=altered
            )
            try:
                payload = response.json()
            except ValueError:
                payload = {"non_json_body": response.text[:8000]}
            save(output / f"control_{mode}_response.json", payload, ak)
            record = {
                "mode": mode,
                "http_status": response.status_code,
                "elapsed_s": round(time.monotonic() - started, 3),
                "error": payload.get("error") if isinstance(payload, dict) else None,
                "usage": payload.get("usage") if isinstance(payload, dict) else None,
                "request_state_fields": state_fields(altered),
                "response_state_fields": state_fields(payload),
                "touched_state_fields": touched,
            }
        except httpx.HTTPError as exc:
            record = {"mode": mode, "error_type": type(exc).__name__}
        records.append(record)
        print(
            json.dumps(
                {
                    "control": mode,
                    "http_status": record.get("http_status"),
                    "error_type": record.get("error_type"),
                }
            ),
            flush=True,
        )
    return records


def run(args: argparse.Namespace, ak: str) -> int:
    logging.disable(logging.CRITICAL)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    tracked_drift = subprocess.check_output(
        ["git", "diff", "HEAD", "--"], cwd=ROOT, text=True
    )
    if tracked_drift:
        raise ValueError("Tracked worktree must be clean at probe launch")
    run_id = "gemini_tool_state_20261103_" + uuid.uuid4().hex[:10]
    marker = "context-" + uuid.uuid4().hex[:12]
    receipts = {name: "receipt-" + uuid.uuid4().hex[:12] for name in ("alpha", "beta")}
    endpoint = ENDPOINTS[args.protocol]
    is_responses = args.protocol == "responses"
    prior_outputs: list[dict[str, Any]] = []
    followed_up = False
    control_candidate: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are testing a tool protocol. Always use lookup_record to obtain the requested records. First call alpha, wait for its result, then call beta, wait for its result, then return only JSON with marker, sum, alpha_receipt, beta_receipt. Never invent tool results. Do not call both tools in the same response. Preserve the user's marker exactly.",
        },
        {
            "role": "user",
            "content": f"Fetch records alpha and beta, add their values, and report their receipts. The marker is {marker}.",
        },
    ]
    if args.image:
        messages[-1]["content"] = [
            {"type": "text", "text": messages[-1]["content"]},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_record",
                "description": "Look up one record and return its numeric value and unique receipt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": ["alpha", "beta"]}
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_drift": False,
        "worktree": str(ROOT),
        "executed_probe_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "endpoint_without_credentials": endpoint,
        "protocol": args.protocol,
        "model": MODEL,
        "stream": False,
        "input_modalities": ["text", "image_url"] if args.image else ["text"],
        "credential_source": "Explicit user curl from this chat; loaded in memory only",
        "same_credential_all_requests": True,
        "session_id": run_id,
        "effort": args.effort or "omitted: provider default",
        "max_tokens": 4096,
        "requests": [],
        "tool_executions": [],
        "result": "running",
        "expected": {
            "marker": marker,
            "sum": 348,
            "alpha_receipt": receipts["alpha"],
            "beta_receipt": receipts["beta"],
        },
        "scope": "Direct HTTP protocol smoke; no Codex SDK, adapter, dataset or benchmark scoring",
    }
    save(output / "summary.json", summary, ak)
    with httpx.Client(
        timeout=httpx.Timeout(90.0, connect=20.0), follow_redirects=False
    ) as client:
        for index in range(1, 7):
            body: dict[str, Any] = {
                "model": MODEL,
                "stream": False,
                "max_tokens": 4096,
                "messages": copy.deepcopy(messages),
                "tools": tools,
            }
            if args.effort:
                body["reasoning_effort"] = args.effort
            if is_responses:
                body = {
                    "model": MODEL,
                    "stream": False,
                    "max_output_tokens": 4096,
                    "store": False,
                    "include": ["reasoning.encrypted_content"],
                    "input": copy.deepcopy(messages),
                    "tools": [
                        {"type": "function", **tool["function"]} for tool in tools
                    ],
                }
                if args.effort:
                    body["reasoning"] = {"effort": args.effort}
            if control_candidate is None and opaque_locations(body):
                control_candidate = copy.deepcopy(body)
                summary["control_base_request_index"] = index
            request_id = f"{run_id}_{index}"
            headers = {
                "Content-Type": "application/json",
                "X-TT-LOGID": request_id,
                "extra": json.dumps(
                    {
                        "session_id": run_id,
                        "chat_run_id": run_id,
                        "source": "codex_adapter_protocol_probe",
                    }
                ),
            }
            save(
                output / f"request_{index}.json", {"headers": headers, "body": body}, ak
            )
            start = time.monotonic()
            item: dict[str, Any] = {
                "index": index,
                "logid": request_id,
                "request_state_fields": state_fields(body),
                "assistant_replay_hashes": [
                    digest(m) for m in messages if m.get("role") == "assistant"
                ],
                "output_replay_hashes": [digest(m) for m in prior_outputs],
                "outputs_present_in_request": all(
                    m in body.get("input", body.get("messages", []))
                    for m in prior_outputs
                ),
            }
            try:
                response = client.post(
                    endpoint, params={"ak": ak}, headers=headers, json=body
                )
            except httpx.HTTPError as exc:
                item.update(
                    error_type=type(exc).__name__,
                    elapsed_s=round(time.monotonic() - start, 3),
                )
                summary["requests"].append(item)
                summary["result"] = "transport_failure"
                save(output / "summary.json", summary, ak)
                print(
                    json.dumps(
                        {
                            "round": index,
                            "result": summary["result"],
                            "error_type": type(exc).__name__,
                        }
                    ),
                    flush=True,
                )
                return 2
            item.update(
                http_status=response.status_code,
                elapsed_s=round(time.monotonic() - start, 3),
            )
            try:
                payload = response.json()
            except ValueError:
                payload = {"non_json_body": response.text[:8000]}
            save(output / f"response_{index}.json", payload, ak)
            item["response_fields"] = field_paths(payload)
            item["response_state_fields"] = state_fields(payload)
            if isinstance(payload, dict):
                item["usage"] = payload.get("usage")
                item["response_model"] = payload.get("model")
            summary["requests"].append(item)
            if response.status_code != 200:
                summary["result"] = "upstream_http_failure"
                save(output / "summary.json", summary, ak)
                print(
                    json.dumps(
                        {
                            "round": index,
                            "http_status": response.status_code,
                            "result": summary["result"],
                        }
                    ),
                    flush=True,
                )
                return 3
            if is_responses:
                output_items = payload.get("output", [])
                if not isinstance(output_items, list) or not output_items:
                    summary["result"] = "unexpected_response_shape"
                    save(output / "summary.json", summary, ak)
                    return 4
                calls = [
                    {
                        "id": v["call_id"],
                        "function": {"name": v["name"], "arguments": v["arguments"]},
                    }
                    for v in output_items
                    if v.get("type") == "function_call"
                ]
                content = "".join(
                    part.get("text", "")
                    for v in output_items
                    if v.get("type") == "message"
                    for part in v.get("content", [])
                    if part.get("type") == "output_text"
                )
                message = {"role": "assistant", "content": content, "tool_calls": calls}
                item["finish_reason"] = payload.get("status")
            else:
                choices = (
                    payload.get("choices", []) if isinstance(payload, dict) else []
                )
                if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                    summary["result"] = "unexpected_response_shape"
                    save(output / "summary.json", summary, ak)
                    return 4
                message = choices[0]["message"]
                output_items = [message]
                calls = message.get("tool_calls") or []
                item["finish_reason"] = choices[0].get("finish_reason")
            item["assistant_message_sha256"] = digest(message)
            item["assistant_message_fields"] = list(message)
            item["output_item_hashes"] = [digest(v) for v in output_items]
            messages.extend(copy.deepcopy(output_items))
            prior_outputs.extend(copy.deepcopy(output_items))
            print(
                json.dumps(
                    {
                        "round": index,
                        "http_status": response.status_code,
                        "finish_reason": item["finish_reason"],
                        "message_fields": list(message),
                        "state_field_paths": [
                            v["path"] for v in item["response_state_fields"]
                        ],
                        "tool_calls": len(calls),
                        "elapsed_s": item["elapsed_s"],
                    }
                ),
                flush=True,
            )
            if item["finish_reason"] in {"length", "incomplete", "failed"}:
                summary["result"] = "truncated_response"
                save(output / "summary.json", summary, ak)
                return 5
            if calls:
                for call in calls:
                    result = execute_tool(call, receipts)
                    summary["tool_executions"].append(
                        {"round": index, "id": call["id"], "result": result}
                    )
                    tool_output = {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result),
                    }
                    if is_responses:
                        tool_output = {
                            "type": "function_call_output",
                            "call_id": call["id"],
                            "output": json.dumps(result),
                        }
                    messages.append(tool_output)
                save(output / "summary.json", summary, ak)
                continue
            content = message.get("content") or ""
            json_content = content.strip()
            fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", json_content)
            if fenced:
                json_content = fenced.group(1)
            summary["final_answer_format"] = "markdown_json" if fenced else "plain"
            try:
                final = json.loads(json_content)
            except (ValueError, TypeError):
                final = None
            summary["final_answer"] = final if final is not None else content
            summary["answer_matches_expected"] = final == summary["expected"]
            summary["tool_order_correct"] = [
                r["result"]["name"] for r in summary["tool_executions"]
            ] == (["alpha", "beta", "alpha"] if followed_up else ["alpha", "beta"])
            summary["result"] = (
                "passed"
                if summary["answer_matches_expected"] and summary["tool_order_correct"]
                else "answer_or_tool_order_mismatch"
            )
            summary["opaque_state_observed"] = any(
                s["opaque_name"] and s["nonempty"]
                for r in summary["requests"]
                for s in r.get("response_state_fields", [])
            )
            summary["replay_hashes_match"] = all(
                current["outputs_present_in_request"]
                and current["output_replay_hashes"]
                == [
                    h
                    for old in summary["requests"][:i]
                    for h in old["output_item_hashes"]
                ]
                for i, current in enumerate(summary["requests"])
            )
            if args.followup and not followed_up and summary["result"] == "passed":
                summary["initial_chain_passed"] = True
                summary["result"] = "running"
                followed_up = True
                messages.append(
                    {
                        "role": "user",
                        "content": "Now verify by calling lookup_record for alpha once more. After the tool returns, repeat the same final JSON with the ORIGINAL marker, sum, alpha_receipt and beta_receipt from our conversation. Do not invent receipts.",
                    }
                )
                save(output / "summary.json", summary, ak)
                continue
            if args.controls and control_candidate is not None:
                summary["controls"] = run_state_controls(
                    client, endpoint, control_candidate, headers, output, ak
                )
            save(output / "summary.json", summary, ak)
            print(
                json.dumps(
                    {
                        "result": summary["result"],
                        "opaque_state_observed": summary["opaque_state_observed"],
                        "replay_hashes_match": summary["replay_hashes_match"],
                    }
                ),
                flush=True,
            )
            return 0 if summary["result"] == "passed" else 6
    summary["result"] = "request_budget_exhausted"
    save(output / "summary.json", summary, ak)
    return 7


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-transcript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--effort", choices=["medium", "none"])
    parser.add_argument("--protocol", choices=list(ENDPOINTS), default="legacy_chat")
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--image", action="store_true")
    args = parser.parse_args()
    try:
        ak = credential(args.credential_transcript)
        return run(args, ak)
    except Exception as exc:
        # Exception strings/tracebacks may contain HTTP URLs with credentials.
        print(
            json.dumps(
                {"fatal_error_type": type(exc).__name__, "details_suppressed": True}
            ),
            flush=True,
        )
        return 10


if __name__ == "__main__":
    sys.exit(main())
