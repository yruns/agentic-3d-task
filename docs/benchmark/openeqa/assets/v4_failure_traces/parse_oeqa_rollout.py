"""Parse a Codex OpenEQA rollout JSONL into an ordered tool-call trace.

Reads one ``rollout-*.jsonl`` (the per-turn Codex session log preserved when
``CODEX_AGENT_KEEP_RUN_HOME=1``) and reconstructs, in order, every tool action
the agent took: shell tool invocations (``python -m codex_agent.openeqa.tools
<tool> <json>``), ``view_image`` calls, and their (truncated) outputs. This is
how we answer "why did N tool rounds still not find a useful frame?".
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

OEQA_RE = re.compile(r"codex_agent\.openeqa\.tools\s+([a-z_]+)")
MD_READ_RE = re.compile(r"\b(sed|cat|head|tail|less|more|grep|rg)\b.*\.md\b")


def _classify(cmd: str) -> str:
    m = OEQA_RE.search(cmd)
    if m:
        return "oeqa:" + m.group(1)
    if MD_READ_RE.search(cmd) or "SKILL.md" in cmd:
        return "md_read"
    first = cmd.strip().split()[0] if cmd.strip() else "(empty)"
    return "other:" + first


def _short(text: str, n: int = 220) -> str:
    text = str(text)
    # collapse base64 image payloads so the trace stays readable
    text = re.sub(r"data:image/[^\"']+", "<image-bytes>", text)
    text = re.sub(r"[A-Za-z0-9+/]{120,}={0,2}", "<b64>", text)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _kf_result(out: str) -> str:
    """Pull frame ids + note out of a keyframe_selector JSON output blob."""
    ids = re.findall(r'"frame_id"\s*:\s*(\d+)', out)
    note = re.search(r'"note"\s*:\s*"([^"]*)"', out)
    hyp = re.search(r'"hypothesis_summary"\s*:\s*"([^"]*)"', out)
    parts = []
    if hyp:
        parts.append("hyp=" + hyp.group(1))
    parts.append("frames=" + (",".join(ids) if ids else "NONE"))
    if note and note.group(1):
        parts.append("note=" + note.group(1))
    return " | ".join(parts)


def _tool_arg(cmd: str) -> str:
    """Pull the trailing JSON arg blob out of an openeqa tools command."""
    m = re.search(r"openeqa\.tools\s+[a-z_]+\s+(.*)$", cmd)
    return _short(m.group(1), 160) if m else ""


def parse(path: Path) -> dict[str, object]:
    calls: list[dict[str, str]] = []  # ordered tool calls, each with aggregated output
    query = ""
    aborted = False
    final_text = ""
    current: dict[str, str] | None = None  # the call currently receiving output

    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        pl = o.get("payload") or {}
        t = o.get("type")
        pt = pl.get("type") if isinstance(pl, dict) else None

        if t == "event_msg" and pt == "user_message" and not query:
            msg = pl.get("message") or ""
            qm = re.search(r"Question:\s*(.+)", msg)
            query = qm.group(1).strip() if qm else _short(msg, 120)
        if t == "event_msg" and pt == "turn_aborted":
            aborted = True

        if t == "response_item" and pt == "function_call":
            name = pl.get("name")
            # write_stdin is the model polling a still-running PTY exec; its output
            # chunks belong to the preceding exec call. Don't start a new action
            # (and don't count it) — just let output keep flowing to `current`.
            if name in ("write_stdin", "read_stdout", "kill_command"):
                if current is not None:
                    current["polls"] = str(int(current.get("polls", "0")) + 1)
                continue
            if name == "view_image":
                try:
                    a = json.loads(pl.get("arguments") or "{}")
                    detail = _short(a.get("path", ""), 90)
                except Exception:
                    detail = ""
                current = {"kind": "view_image", "detail": detail, "_out": ""}
            else:
                try:
                    a = json.loads(pl.get("arguments") or "{}")
                    cmd = a.get("cmd") or a.get("command") or ""
                    if isinstance(cmd, list):
                        cmd = " ".join(str(x) for x in cmd)
                except Exception:
                    cmd = pl.get("arguments") or ""
                kind = _classify(cmd)
                detail = _tool_arg(cmd) if kind.startswith("oeqa:") else _short(cmd, 120)
                current = {"kind": kind, "detail": detail, "_out": ""}
            calls.append(current)

        # exec output streams as several function_call_output chunks (each a new
        # call_id) right after the call; attribute them all to the current call.
        if t == "response_item" and pt == "function_call_output" and current is not None:
            out = pl.get("output")
            if isinstance(out, dict):
                out = out.get("content") or out.get("output") or json.dumps(out)
            current["_out"] += str(out or "")

        if t == "response_item" and pt == "message" and pl.get("role") == "assistant":
            content = pl.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        final_text = _short(c["text"], 400)
            elif isinstance(content, str):
                final_text = _short(content, 400)

    # summarize aggregated output per call
    for call in calls:
        raw = call.pop("_out", "")
        if call["kind"] == "oeqa:keyframe_selector":
            call["out"] = _kf_result(raw)
        elif call["kind"].startswith("oeqa:"):
            tail = raw[-600:]
            call["out"] = _short(tail, 240)
        else:
            call["out"] = ""

    cats = Counter(c["kind"] for c in calls)
    total_polls = sum(int(c.get("polls", "0")) for c in calls)
    return {
        "query": query,
        "aborted": aborted,
        "n_counted_actions": len(calls),  # exec + view_image (what the guard counts)
        "n_polls": total_polls,  # write_stdin waits (NOT counted, but add latency)
        "category_counts": dict(cats),
        "calls": calls,
        "final_text": final_text,
    }


def main() -> None:
    path = Path(sys.argv[1])
    r = parse(path)
    print("ABORTED (hit 24-cap):", r["aborted"])
    print("COUNTED ACTIONS (exec+view_image):", r["n_counted_actions"], "| write_stdin polls:", r["n_polls"])
    print("MIX:", json.dumps(r["category_counts"], ensure_ascii=False))
    print("\n--- ordered trace (counted actions only; polls folded into preceding exec) ---")
    for i, c in enumerate(r["calls"], 1):
        polls = c.get("polls", "0")
        suffix = f"  [+{polls} polls]" if polls != "0" else ""
        print(f"{i:2d}. {c['kind']:24s} {c['detail']}{suffix}")
        if c.get("out"):
            print(f"       -> {c['out']}")
    print("\nFINAL:", r["final_text"])


if __name__ == "__main__":
    main()
