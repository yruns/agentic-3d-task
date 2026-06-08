# NR3D v1 — `CodexAgentRuntime` visual-grounding smoke (10-case)

First end-to-end run of the migrated **Codex Agent SDK runtime**
(`src/codex_agent/`) on NR3D visual grounding. This is the "make 10 cases run"
milestone, **not** a leaderboard number — it validates that the migrated runtime
selects a proposal, parses the structured answer, and scores oriented 3D IoU
end to end against the live ModelHub Codex backend.

This is a different track from
[`v1_keyframe_targetid_coverage_strat600`](v1_keyframe_targetid_coverage_strat600_20260608.md):
that doc measures Stage-1 keyframe **coverage** (`hit@K`); this doc measures the
agent's **VG accuracy** (IoU / Acc@τ).

## What was migrated

From `3DVLMReasoning` (`src/agents/runtime/codex_sdk_agent.py`,
`stage2_codex_agent.py`, `mcp/nr3d_tools_*`, `evaluation/scripts/run_nr3d_vg_side_by_side.py`,
`packs/vg_embodiedscan/*`, `benchmarks/embodiedscan_eval.py`) into a clean,
task-agnostic package `src/codex_agent/`:

- `CodexAgentRuntime` — the Stage-2 naming is dropped; the runtime is generic and
  drives one Codex turn (isolated `CODEX_HOME`, ModelHub prefix-cache headers,
  finalization re-ask). Tasks plug in via the `CodexTask` protocol, so the
  runtime is not NR3D-specific.
- `codex_agent.nr3d` — the first task family: proposal-pool loader, sample/scene
  loaders, oriented 3D IoU, and the catalog-first grounding task.
- `codex_agent.evaluation.nr3d_runner` — fold runner with per-sample
  checkpointing + IoU metrics.

Deliberately **not** migrated for this milestone: the DeepAgents runtime, the
`pack_v1` chassis, and the MCP/CLI evidence tools. This run is therefore
**catalog-first prompt-only** (proposal catalog + BEV image, no in-turn frame
inspection / spatial tools). Those evidence tools are the documented next phase.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch / commit | `agentic-3d-task` / `master` / `f1fe9ad` |
| Working tree | new package `src/codex_agent/` **uncommitted** at run time (migration in progress) |
| Fold | first 10 of canonical `v9_3_strat600` (`tmp/nr3d_case10/sample_ids.json`) |
| Fold size | 10 (Easy 6 / Hard 4 ; View-Dep 5 / View-Indep 5) |
| Data root | `3DVLMReasoning/data/nr3d/scannet` (shared prepared packs) |
| Pack | `pack_nr3d_v9_catalog_first` |
| Proposal pool | `source = gt` → selecting the right proposal yields IoU ≈ 1.0 |
| Backend | `gpt-5.4-2026-03-05` via Codex `model_provider = modelhub_adapter` (`http://127.0.0.1:8787/v1`, `wire_api = responses`) |
| Codex SDK | `openai-codex==0.1.0b2` (+ `openai-codex-cli-bin==0.132.0`) |
| Sandbox | `read_only` (prompt-only, no filesystem writes) |
| Skill | `.agents/skills/nr3d-codex-sdk/SKILL.md` (catalog-first, CLI-tool refs removed) |
| Prefix cache | on; `session_id = codex_agent`, per-turn `chat_run_id` |
| Workers / retries | 5 / 2 |
| Judge | none — deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-08 |

## Headline

| Metric | Value |
|---|---:|
| n | 10 |
| mean IoU | **0.703** |
| Acc@0.25 | **0.70** |
| Acc@0.50 | **0.70** |
| hard errors | **0** |

7 / 10 correct; 0 pipeline failures. The metric is real VG accuracy: because the
pool is `source = gt`, a correct proposal pick scores IoU ≈ 1.0 and a wrong pick
scores ≈ 0.

## Per-sample

| sample_id | tier | sel | IoU | ms | query (truncated) |
|---|---|---:|---:|---:|---|
| scene0207_00::20 | hard·vdep | 20 | 1.00 | 8944 | The closet doors across from the bed. |
| scene0221_00::47 | easy·vindep | 6 | 0.00 | 8027 | The correct table is square and close to the door. |
| scene0246_00::5 | easy·vindep | 5 | 1.00 | 16456 | The lamp on the square bedside table. |
| scene0426_00::22 | hard·vindep | 22 | 1.00 | 28170 | the dark colored pillow in the corner. |
| scene0462_00::6 | easy·vdep | 6 | 1.00 | 8655 | …the door next to the big grey box… |
| scene0500_00::25 | hard·vdep | 25 | 1.00 | 11461 | The middle rectangular window on the wall left of … |
| scene0591_00::14 | easy·vindep | 16 | 0.00 | 9205 | The darker pillow, laying against the armrest… |
| scene0608_00::9 | easy·vdep | 9 | 1.00 | 8465 | Looking for a black ottoman, right in front of … |
| scene0653_00::18 | hard·vdep | 18 | 1.00 | 16769 | The desk on the right side of the room closest to … |
| scene0699_00::26 | easy·vindep | 25 | 0.03 | 15674 | the pillow on the side near the kitchen table. |

The 3 misses (`scene0221`, `scene0591`, `scene0699`) are same-category
disambiguation under spatial language ("square table near the door", "darker
pillow against the armrest", "pillow near the kitchen table") — exactly the
cases the not-yet-migrated evidence tools (per-candidate `mark_frame_with_bbox`,
`compare_proposals_spatial`) are designed to resolve.

## Bug found + fixed during the run

The first 10-case run had **1 hard failure** on `scene0500_00::25`:

```
[Errno 66] Directory not empty:
'.codex-home/runs/<uuid>/.tmp/plugins-clone-XXXX/plugins'
```

Root cause: `CodexAgentRuntime._cleanup_run_home` called `shutil.rmtree` on the
per-turn `CODEX_HOME` while the Codex app-server was still finalizing its plugin
clone in `.tmp/`. Removing the scratch home is housekeeping and must never fail
an otherwise-successful turn. Fix: `_remove_tree_best_effort` retries the rmtree
a few times and then logs a warning instead of raising (unit-tested in
`test_runtime.py::test_remove_tree_best_effort_*`). Re-running the case resumed
from checkpoint and scored IoU = 1.0; the fold went 9/10 → 10/10 completed.

## Rerun — gpt-5.4 AK pool sourced from `configs/llm.toml` (same day)

Re-ran the identical 10-case fold/code after pointing the Codex `modelhub_adapter`
upstream pool at `configs/llm.toml → [models."gpt-5.4-2026-03-05"]`: the **same 3
AKs** already in use, with weights re-synced to each key's real QPM quota
(`200 / 20 / 100`, was `5 / 1 / 5`). Adapter restarted, health-verified.

| Metric | first run | rerun (configs/llm.toml pool) |
|---|---:|---:|
| mean IoU | 0.703 | **0.800** |
| Acc@0.50 | 0.70 | **0.80** |
| completed / errors | 10 / 0 | **10 / 0** |

Adapter access log: all 10 `POST /v1/responses` → `200 OK`, no 429/503. The
+0.10 IoU swing is within n=10 non-determinism (per-sample picks differ on the
ambiguous cases run-to-run), not a code change — the only delta is AK weights
across the identical key set. Artifacts: `tmp/nr3d_case10/run_gpt54/`,
`assets/codex_runtime_grounding_smoke10_gpt54_llmtoml_20260608.json`. **Conclusion:
the multi-AK gpt-5.4 config from `configs/llm.toml` runs the fold end-to-end with
zero errors.**

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_case10/sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_case10/run10 \
  --workers 5 --sample-retries 2
```

Requires the ModelHub adapter running on `127.0.0.1:8787` (see
`CLAUDE.md §Codex SDK ModelHub Adapter`) and `.codex-home/config.toml` pointing
at it.

## Raw artifacts

- Per-sample checkpoints: `tmp/nr3d_case10/run10/per_sample/pack_nr3d_v9_catalog_first/`
- Run summary: `tmp/nr3d_case10/run10/summary.json`
- Console log: `tmp/nr3d_case10/run10.log`, `tmp/nr3d_case10/run10_resume.log`
- Trimmed durable summary: `assets/codex_runtime_grounding_smoke10_20260608.json`

## Caveats

- **10-case smoke, not a leaderboard.** Variance at n=10 is large; do not compare
  these numbers to strat600/full-set runs. The point was end-to-end execution.
- **Prompt-only (no evidence tools).** Accuracy here is a floor; the source
  pipeline's CLI/MCP tools (frame inspection, spatial comparison) are the next
  migration phase.
- `source = gt` pool makes VG a selection problem (correct id → IoU ≈ 1.0).
- `tmp/` is gitignored; the durable record is this doc + the trimmed asset.
