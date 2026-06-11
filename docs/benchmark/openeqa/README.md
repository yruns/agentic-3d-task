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

## Leaderboard (our pipeline)

| Version | Fold | n | MNAS | Completion | Mode | Judge |
|---|---|---:|---:|---:|---|---|
| v1 (migration smoke) | category-stratified smoke10 | 10 | **75.0** | 10/10 | prompt-only | gemini-2.5-pro |
| v2 (tools + BEV smoke) | category-stratified smoke10 | 10 | **67.5** | 10/10 | tools + BEV | gemini-2.5-pro |

Full-set OpenEQA numbers are **not yet run**; both rows are smokes on the same
n=10 fold. The v1↔v2 MNAS gap is inside the ±10–20 pp variance of a 10-question
fold — do **not** read it as a tools-vs-prompt-only effect. Design the canonical
stratified fold (per `CLAUDE.md`) before the first decision-grade A/B.

## Caveats (read before citing any number)

- **Smoke fold (n=10).** Per-category cells are n≤2 — they are sanity signals,
  not statistics. Do not compare across versions on this fold; design a
  canonical stratified fold before the second run (see the §canonical pilot fold
  rule in `CLAUDE.md`).
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
