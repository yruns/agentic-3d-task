"""Trace analysis for the halted NR3D v2 tool run.

Parses the surviving Codex session rollouts (`.codex-home/runs/*/sessions/.../
rollout-*.jsonl`) to answer: on View-Dep cases, did the agent actually use the
spatial / co-visible tools, or did it burn the turn re-reading SKILL.md?

These rollouts are the run homes that were *in flight* when the run was killed
(completed turns delete their run home), so this set is biased toward the slow
tail — exactly the cases that caused the throughput collapse.
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path("/Users/bytedance/project/agentic-3d-task")
DATA_ROOT = Path("/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet")
PACK = "pack_nr3d_v9_catalog_first"
FOLD = REPO / "tmp/nr3d_case600/sample_ids.json"

NR3D_RE = re.compile(r"codex_agent\.nr3d\.tools\s+([a-z_]+)")
QUERY_RE = re.compile(r"query:\s*(.+)")
MD_READ_RE = re.compile(r"\b(sed|cat|head|tail|less|more|grep|rg)\b.*\.md\b")


def build_query_to_tier() -> dict[str, dict[str, object]]:
    fold = json.loads(FOLD.read_text())
    tier_by_id = {r["sample_id"]: r for r in fold}
    out: dict[str, dict[str, object]] = {}
    for sid, row in tier_by_id.items():
        scene = row["scene_id"]
        safe = sid.replace("/", "__").replace("::", "__")
        art = DATA_ROOT / scene / PACK / "samples" / f"{safe}.json"
        if not art.exists():
            continue
        try:
            q = json.loads(art.read_text()).get("query", "")
        except Exception:
            continue
        if q:
            out[q.strip()] = {
                "sample_id": sid,
                "is_view_dep": bool(row.get("is_view_dep")),
                "is_easy": bool(row.get("is_easy")),
            }
    return out


def classify_cmd(cmd: str) -> str:
    m = NR3D_RE.search(cmd)
    if m:
        return "nr3d:" + m.group(1)
    if MD_READ_RE.search(cmd) or "SKILL.md" in cmd or "AGENTS.md" in cmd:
        return "md_read"
    first = cmd.strip().split()[0] if cmd.strip() else "(empty)"
    return "other:" + first


def parse_rollout(path: Path) -> dict[str, object]:
    query = ""
    cmds: list[str] = []
    n_view_image = 0
    aborted = False
    final_msg = False
    for ln in path.read_text().splitlines():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        pl = o.get("payload") or {}
        t = o.get("type")
        pt = pl.get("type") if isinstance(pl, dict) else None
        if t == "event_msg" and pt == "user_message" and not query:
            msg = pl.get("message") or ""
            qm = QUERY_RE.search(msg)
            if qm:
                query = qm.group(1).strip()
        if t == "event_msg" and pt == "turn_aborted":
            aborted = True
        if t == "response_item" and pt == "function_call":
            name = pl.get("name")
            if name == "view_image":
                n_view_image += 1
            elif name in ("exec_command", "shell", "local_shell"):
                try:
                    a = json.loads(pl.get("arguments") or "{}")
                    cmd = a.get("cmd") or a.get("command") or ""
                    if isinstance(cmd, list):
                        cmd = " ".join(str(x) for x in cmd)
                except Exception:
                    cmd = pl.get("arguments") or ""
                cmds.append(cmd)
        if t == "response_item" and pt == "message" and pl.get("role") == "assistant":
            final_msg = True
    cats = Counter(classify_cmd(c) for c in cmds)
    nr3d = {k.split(":", 1)[1]: v for k, v in cats.items() if k.startswith("nr3d:")}
    n_md = cats.get("md_read", 0)
    n_nr3d = sum(nr3d.values())
    n_other = sum(v for k, v in cats.items() if k.startswith("other:"))
    used_spatial = nr3d.get("compare_proposals_spatial", 0) + nr3d.get(
        "compare_candidates_to_anchors", 0
    )
    covis = sum(
        1
        for c in cmds
        if "select_by_proposal" in c and '"require_all":true' in c.replace(" ", "")
    )
    return {
        "query": query,
        "n_exec": len(cmds),
        "n_nr3d": n_nr3d,
        "n_md_read": n_md,
        "n_other": n_other,
        "n_view_image": n_view_image,
        "nr3d_breakdown": nr3d,
        "used_spatial": used_spatial > 0,
        "used_covisible": covis > 0,
        "used_any_nr3d": n_nr3d > 0,
        "aborted": aborted,
        "final_msg": final_msg,
    }


def main() -> None:
    q2t = build_query_to_tier()
    rollouts = sorted(glob.glob(str(REPO / ".codex-home/runs/*/sessions/**/rollout-*.jsonl"), recursive=True))
    rows = []
    for rf in rollouts:
        r = parse_rollout(Path(rf))
        tier = q2t.get(r["query"])
        r["is_view_dep"] = tier["is_view_dep"] if tier else None
        r["sample_id"] = tier["sample_id"] if tier else None
        rows.append(r)

    n = len(rows)
    mapped = [r for r in rows if r["is_view_dep"] is not None]
    print(f"=== {n} rollouts ({len(mapped)} mapped to a fold tier) ===\n")

    # headline: SKILL.md loop
    loop = [r for r in rows if r["n_md_read"] >= 5]
    md_total = sum(r["n_md_read"] for r in rows)
    exec_total = sum(r["n_exec"] for r in rows)
    nr3d_total = sum(r["n_nr3d"] for r in rows)
    print("--- exec command mix (all 57 surviving = slow/in-flight tail) ---")
    print(f"total exec commands : {exec_total}")
    print(f"  NR3D tool calls   : {nr3d_total} ({100*nr3d_total/max(exec_total,1):.0f}%)")
    print(f"  SKILL/.md reads   : {md_total} ({100*md_total/max(exec_total,1):.0f}%)")
    print(f"rollouts with >=5 .md reads (loop-dominated): {len(loop)}/{n}")
    print(f"aborted (killed mid-turn): {sum(1 for r in rows if r['aborted'])}/{n}")
    import statistics as st
    print(f"n_exec per rollout: median={st.median([r['n_exec'] for r in rows]):.0f} "
          f"max={max(r['n_exec'] for r in rows)}")
    print(f"n_md_read per rollout: median={st.median([r['n_md_read'] for r in rows]):.0f} "
          f"max={max(r['n_md_read'] for r in rows)}")

    def rate(rs, key):
        return f"{sum(1 for r in rs if r[key])}/{len(rs)}" if rs else "0/0"

    print("\n--- tool usage (mapped rollouts) ---")
    for label, subset in [
        ("ALL mapped", mapped),
        ("View-Dep", [r for r in mapped if r["is_view_dep"]]),
        ("View-Indep", [r for r in mapped if not r["is_view_dep"]]),
    ]:
        print(f"\n[{label}] n={len(subset)}")
        print(f"  used any NR3D tool : {rate(subset,'used_any_nr3d')}")
        print(f"  used spatial tool  : {rate(subset,'used_spatial')}")
        print(f"  used co-visible    : {rate(subset,'used_covisible')}")
        print(f"  used view_image    : {rate(subset,'n_view_image')}")
        if subset:
            print(f"  median NR3D calls  : {st.median([r['n_nr3d'] for r in subset]):.0f}")
            print(f"  median .md reads   : {st.median([r['n_md_read'] for r in subset]):.0f}")

    # aggregate nr3d tool histogram
    agg = Counter()
    for r in rows:
        for k, v in r["nr3d_breakdown"].items():
            agg[k] += v
    print("\n--- NR3D tool call histogram (all 57) ---")
    for k, v in agg.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
