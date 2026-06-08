# NR3D v1 — `CodexAgentRuntime` visual grounding, full strat600

First full **canonical strat600** evaluation of the migrated Codex Agent SDK
runtime (`src/codex_agent/`). Same pipeline as the
[10-case smoke](v1_codex_runtime_grounding_smoke_20260608.md), now on all 600
samples: catalog-first **prompt-only** (proposal catalog + BEV image, no in-turn
evidence tools), gpt-5.4 via the ModelHub adapter with the multi-AK pool sourced
from `configs/llm.toml`.

## Headline (strat600 leaderboard)

| Slice | n | Acc@0.25 | Acc@0.50 | mean IoU |
|---|---:|---:|---:|---:|
| **Overall** | 600 | **63.83%** | 63.83% | 0.643 |
| Easy | 290 | 73.10% | 73.10% | 0.734 |
| Hard | 310 | 55.16% | 55.16% | 0.558 |
| View-Dep | 211 | 51.66% | 51.66% | 0.523 |
| View-Indep | 389 | 70.44% | 70.44% | 0.709 |

`Acc@0.25 == Acc@0.50` because the pool is `source = gt`: a correct proposal pick
scores IoU ≈ 1.0, a wrong/declined pick ≈ 0, so the IoU metric collapses to NR3D
**selection accuracy** (directly comparable to ReferIt3D/NR3D classification
accuracy). 63.83% overall is the **prompt-only floor** — no frame inspection or
spatial tools yet; the expected View-Dep < View-Indep gap (51.7% vs 70.4%)
is exactly what those not-yet-migrated tools target.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch / commit | `agentic-3d-task` / `master` / `f1fe9ad` |
| Working tree | `src/codex_agent/` **uncommitted** at run time (migration in progress; no worktree drift) |
| Fold | canonical `v9_3_strat600` (`tmp/nr3d_case600/sample_ids.json`, copied from `3DVLMReasoning/tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`) |
| Fold size | 600 (Easy 290 / Hard 310 ; View-Dep 211 / View-Indep 389) |
| Data root | `3DVLMReasoning/data/nr3d/scannet` (119 scenes, shared packs) |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via Codex `modelhub_adapter` (`127.0.0.1:8787`, `wire_api=responses`, chat-completions routing for `gpt-5.4*`) |
| AK pool | synced from `configs/llm.toml [models."gpt-5.4-2026-03-05"]` — 3 AKs, weights 200/20/100 (≈320 QPM) |
| Sandbox | `read_only` (prompt-only) |
| Skill | `.agents/skills/nr3d-codex-sdk/SKILL.md` |
| Concurrency | **50 workers**, `sample_retries = 3` |
| Judge | none — deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-08 |

## Run health

- **0 infrastructure errors / 600.** The 2 `status=failed` rows
  (`scene0578_00::18`, `scene0353_00::5`) are the model returning
  `proposal_id = -1` (target declared absent), not pipeline failures.
- Adapter HTTP: **632× `200 OK`**, 1×500 / 1×502 / 2×503 (0.6% transient, all
  recovered by retries). No `429` — never rate-limited.
- Wall clock: **15.8 min** (22:56:43 → 23:12:29).
- Per-turn latency: median **64.6 s**, p90 **144.7 s**, max 421 s — gpt-5.4 is a
  slow multimodal reasoner; this dominates throughput.
- Requests: ~636 for 600 samples ≈ 1.06 req/case (low finalization re-ask rate).

## Concurrency note ("max concurrency")

The run was **not** QPM-bound: ~636 requests over 15.8 min ≈ **40 req/min**, only
~12% of the gpt-5.4 pool's ~320 QPM. The bottleneck is per-turn upstream latency
(median ~65 s) × worker count, so throughput is `workers / latency`. With 8×
unused QPM headroom, concurrency can safely go well above 50 (≈100–150 workers
would roughly halve wall time); the practical cap then becomes local Codex
app-server subprocess load (one process + isolated `CODEX_HOME` per turn) and the
upstream's tolerance for concurrent multimodal requests, not the AK quota. 50 was
chosen as a conservative-but-high first full run; it completed clean.

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=src CODEX_AGENT_MODEL=gpt-5.4-2026-03-05 \
  python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_case600/sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_case600/run1 \
  --model gpt-5.4-2026-03-05 --workers 50 --sample-retries 3
```

Requires the ModelHub adapter on `127.0.0.1:8787` with the gpt-5.4 pool
(`.modelhub_upstreams.toml` synced from `configs/llm.toml`).

## Raw artifacts

- Per-sample checkpoints: `tmp/nr3d_case600/run1/per_sample/pack_nr3d_v9_catalog_first/` (600)
- Run summary: `tmp/nr3d_case600/run1/summary.json`
- Console log: `tmp/nr3d_case600/run_case600.log`
- Durable leaderboard + per-tier asset: `assets/codex_runtime_grounding_strat600_gpt54_20260608.json`

## Caveats

- **Prompt-only floor.** No MCP/CLI evidence tools (frame inspection,
  `compare_proposals_spatial`, `mark_frame_with_bbox`). Those are the next
  migration phase and target precisely the Hard / View-Dep gap.
- `source = gt` pool → VG is a selection task (Acc == IoU@τ).
- Single run; strat600 90% variance band ≈ ±2.3 pp Overall — treat sub-band
  deltas as noise until confirmed on the full 7805.
- `tmp/` is gitignored; durable record = this doc + the asset.
