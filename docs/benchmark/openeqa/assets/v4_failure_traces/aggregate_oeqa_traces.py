"""Aggregate every captured OpenEQA failure rollout into one comparison table.

Reads each ``tmp/v4_trace/homes/<qid>/.../rollout-*.jsonl`` and reports, per
question, the metrics that explain "many tool calls, still no useful frame":
counted actions vs the 24 cap, whether it was interrupted, how many distinct
keyframe_selector queries it tried, and crucially how many DISTINCT frames those
queries actually surfaced (retrieval-pool collapse shows up as distinct << total).
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

HOMES = Path("tmp/v4_trace/homes")
PER_SAMPLE = Path("tmp/openeqa_eval_v4_noframes_full1079_20260612/per_sample")
# args embed JSON as an escaped string: ...keyframe_selector ... '{\"query\":\"...\"}'
KF_QUERY_RE = re.compile(r'keyframe_selector .*?\\?"query\\?":\s*\\?"([^"\\]+)')
FRAME_RE = re.compile(r'"frame_id"\s*:\s*(\d+)')


def _gt_pred(qid: str) -> tuple[str, str, int, str]:
    matches = glob.glob(str(PER_SAMPLE / f"{qid}_*.json"))
    if not matches:
        return "?", "?", 0, "?"
    d = json.loads(Path(matches[0]).read_text())
    return (
        str(d.get("gt_answer", "?")),
        str(d.get("prediction", "?")),
        int(d.get("judge_score") or 0),
        str(d.get("category", "?")),
    )


def analyze(rollout: Path) -> dict[str, object]:
    text = rollout.read_text(encoding="utf-8")
    n_exec = n_view_image = n_kf = n_polls = 0
    aborted = False
    kf_queries: list[str] = []
    frames_all: list[int] = []
    cur_is_exec = False

    for ln in text.splitlines():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        pl = o.get("payload") or {}
        pt = pl.get("type") if isinstance(pl, dict) else None
        if pt == "turn_aborted":
            aborted = True
        if o.get("type") == "response_item" and pt == "function_call":
            name = pl.get("name")
            args = str(pl.get("arguments") or "")
            if name in ("write_stdin", "read_stdout", "kill_command"):
                n_polls += 1
                continue
            if name == "view_image":
                n_view_image += 1
                cur_is_exec = False
                continue
            n_exec += 1
            cur_is_exec = True
            if "keyframe_selector" in args:
                n_kf += 1
                m = KF_QUERY_RE.search(args)
                if m:
                    kf_queries.append(m.group(1))
        if (
            o.get("type") == "response_item"
            and pt == "function_call_output"
            and cur_is_exec
        ):
            out = pl.get("output")
            if isinstance(out, str) and '"frame_id"' in out:
                frames_all.extend(int(x) for x in FRAME_RE.findall(out))

    return {
        "counted": n_exec + n_view_image,
        "n_exec": n_exec,
        "n_view_image": n_view_image,
        "n_kf": n_kf,
        "n_polls": n_polls,
        "aborted": aborted,
        "kf_queries": kf_queries,
        "distinct_queries": len(set(kf_queries)),
        "frames_total": len(frames_all),
        "frames_distinct": len(set(frames_all)),
    }


def main() -> None:
    qids = sorted(p.name for p in HOMES.iterdir() if p.is_dir())
    print(f"captured rollouts: {len(qids)}\n")
    hdr = (
        f"{'qid':10s} {'cat':22s} {'cnt':>4s} {'abrt':>4s} {'kf':>3s} "
        f"{'uqry':>4s} {'frmA':>4s} {'frmD':>4s} {'poll':>4s} score gt/pred"
    )
    print(hdr)
    print("-" * len(hdr))
    for qid in qids:
        rfs = glob.glob(str(HOMES / qid / "**/rollout-*.jsonl"), recursive=True)
        if not rfs:
            continue
        a = analyze(Path(rfs[0]))
        gt, pred, score, cat = _gt_pred(qid)
        print(
            f"{qid[:8]:10s} {cat[:22]:22s} {a['counted']:>4d} "
            f"{'YES' if a['aborted'] else '-':>4s} {a['n_kf']:>3d} "
            f"{a['distinct_queries']:>4d} {a['frames_total']:>4d} "
            f"{a['frames_distinct']:>4d} {a['n_polls']:>4d} {score:>5d} "
            f"{gt[:18]} / {pred[:24]}"
        )
        if a["kf_queries"]:
            print(f"           queries: {a['kf_queries']}")


if __name__ == "__main__":
    main()
