# OpenEQA benchmark archive (agentic-3d-task)

Process archive for OpenEQA evaluations in `agentic-3d-task`. Mirrors the
`docs/benchmark/nr3d/` layout.

OpenEQA is open-ended embodied question answering over a 3D scene. Our task
family (`codex_agent.openeqa`) answers each question from a small set of
uniformly-sampled first-person frames and scores the answer with the **official
1–5 `mmbench` LLM-as-judge**, mapped to the headline metric:

```
MNAS = mean over questions of  100 * (clip(judge_score, 1, 5) - 1) / 4
```

(MNAS = Mean Normalized Accuracy Score; identical to the upstream FAIR
`evaluate-predictions.py` and to the 3DVLMReasoning `openeqa_official_eval`.)

## Track

- **Codex agent QA** (`codex_agent.CodexAgentRuntime` + `OpenEqaQuestionAnsweringTask`):
  does the agent's answer match the reference answer? Metric: MNAS (per-category
  MNAS as breakdown). Judge: `gemini-2.5-pro` via `configs/llm.toml` (note: the
  OpenEQA paper baseline uses GPT-4 — judge mismatch caveat applies, see below).
  From v2 the agent is **evidence-seeking**: it can call CLI tools
  (`keyframe_selector`, `view_frame`, `view_bev`, `list_objects`) to fetch more
  visual evidence than the attached frame sample.

## Version timeline

| Version | Date | Scope | Headline |
|---|---|---|---|
| [v1_codex_sdk_migration_smoke10_20260611](v1_codex_sdk_migration_smoke10_20260611.md) | 2026-06-11 | OpenEQA QA logic migrated from 3DVLMReasoning into the `codex_agent` task framework (question loader + frame sampler + QA task + official MNAS judge + runner + CLI). Smoke run over a **category-stratified 10-question fold**, 8 uniform frames, prompt-only (read_only sandbox), gpt-5.4 via ModelHub adapter, judge `gemini-2.5-pro`, 5 workers — **COMPLETE 10/10** | **MNAS = 75.0** (completion 10/10, 0 errors, ~59 s). Per-category (n≤2 each): attr 100 / obj-rec 100 / obj-loc 100 / obj-state 75 / world 75 / func 50 / spatial 0. Pipeline runs end-to-end; numbers are a smoke (tiny fold, high variance). |
| [v2_tools_bev_smoke10_20260612](v2_tools_bev_smoke10_20260612.md) | 2026-06-12 | Evidence-seeking **tools + mesh-free BEV**: adds `keyframe_selector` / `view_frame` / `view_bev` / `list_objects` CLI tools, an inlined tool playbook + skill, and a mesh-free schematic BEV usable in both the keyframe selector and the agent. Same fold/frames/judge as v1; `--tools` (workspace_write + network, loop caps), 5 workers — **COMPLETE 10/10** | **MNAS = 67.5** (completion 10/10, 0 errors, ~105 s). Tool usage confirmed (keyframe_selector + view_bev + view_image). 67.5 vs v1 75.0 is **within n=10 smoke noise** (one spatial miss = 10 pp), not a regression — needs the canonical fold for a real A/B. |
| [v3_tools_full1079_20260612](v3_tools_full1079_20260612.md) | 2026-06-12 | **First full-set run.** Entire ScanNet split — **1,079 questions / 89 scenes** — with the v2 tools+BEV pipeline (`--tools`, workspace_write + network, loop caps), 8 uniform frames, judge `gemini-2.5-pro`, gpt-5.4 answerer, 30 workers (+6-q recovery at 4 workers). Adds the SQLite ingester + `runs.sqlite` — **COMPLETE 1079/1079** | **MNAS = 73.96** (completion 100 %, cache-hit 0.94, ~41 min). Per-category: obj-state 86.9 / attr 82.7 / obj-loc 76.9 / world 70.7 / obj-rec 70.5 / func 69.3 / spatial 58.6. |

## Leaderboard (our pipeline)

| Version | Fold | n | MNAS | Completion | Mode | Judge |
|---|---|---:|---:|---:|---|---|
| v1 (migration smoke) | category-stratified smoke10 | 10 | **75.0** | 10/10 | prompt-only | gemini-2.5-pro |
| v2 (tools + BEV smoke) | category-stratified smoke10 | 10 | **67.5** | 10/10 | tools + BEV | gemini-2.5-pro |
| **v3 (tools + BEV, full ScanNet)** | **full ScanNet split** | **1079** | **73.96** | **1079/1079** | **tools + BEV** | **gemini-2.5-pro** |

v3 is the first decision-grade number: the **full ScanNet split** (all 89 scenes,
all 1,079 questions), 100 % completion. v1/v2 remain n=10 smokes and are not
comparable to v3. There is still **no full-set prompt-only baseline**, so v3 is
not yet a tools-vs-prompt-only A/B — see the v3 doc's next steps.

## Caveats (read before citing any number)

- **Smoke folds (v1/v2, n=10).** Per-category cells are n≤2 — sanity signals,
  not statistics. Do not compare v1/v2 across versions or against v3. v3 is the
  full ScanNet split (n=1079) and is the figure to cite. A **canonical
  stratified pilot fold** (per the §canonical pilot fold rule in `CLAUDE.md`)
  should be salt-locked against v3 before the next round of ablations.
- **Judge mismatch.** We judge with `gemini-2.5-pro`; the OpenEQA paper uses
  GPT-4. MNAS is judge-sensitive, so our absolute MNAS is **not** directly
  comparable to published OpenEQA numbers.
- **Frame selection.** v1 uses uniform first-person frame sampling only. v2 still
  attaches the same uniform sample as the starting point but lets the agent fetch
  more evidence on demand (`keyframe_selector` / `view_frame` / `view_bev`). A
  uniform-vs-tools A/B needs the canonical fold, not this smoke.
- **BEV is a schematic.** OpenEQA ScanNet clips have no `mesh.ply`, so `view_bev`
  draws a top-down floor plan from ConceptGraph object footprints + the camera
  trajectory — not a textured mesh render like NR3D's BEV.
- **Tools need a running judge/LLM + ModelHub adapter** (`keyframe_selector`
  parses the query with the LLM) and `workspace_write` + network sandbox.
- **Data is local-only.** `data/open-eqa-v0.json` + `data/OpenEQA/scannet/` are
  gitignored symlinks to the prepared assets; they never arrive via `git clone`.

## Assets

- `assets/smoke10_sample_ids_20260611.json` — the durable 10-question fold (shared by v1 and v2).
- `assets/smoke10_summary_20260611.json` — the v1 (prompt-only) run summary.
- `assets/smoke10_predictions_20260611.json` — the v1 prediction file.
- `assets/tools_smoke10_summary_20260612.json` — the v2 (tools + BEV) run summary.
- `assets/tools_smoke10_predictions_20260612.json` — the v2 prediction file.
- `assets/v3_tools_full1079_summary_20260612.json` — the v3 full-set aggregate summary.
- `assets/v3_tools_full1079_predictions_20260612.json` — the v3 full-set prediction file (1,079 answers).
- `runs.sqlite` — per-benchmark SQLite DB (`runs` / `samples` / `tool_calls` / `llm_calls`); v3 fully ingested. Ingester: `scripts/ingest_openeqa_run.py`.
