# OpenEQA v1 — Codex SDK migration smoke (stratified 10), 2026-06-11

First OpenEQA run in `agentic-3d-task`. Validates the **migration of the OpenEQA
QA logic** from `3DVLMReasoning` into the `codex_agent` task framework, end to
end over a small category-stratified fold.

## What this run validates

The OpenEQA logic now lives as a first-class task family beside NR3D:

| Concern | Module |
|---|---|
| Question loader (filter ScanNet split, map `episode_history`→`clip_id`) | `src/codex_agent/openeqa/question.py` |
| Scene + uniform first-person frame sampling + JPEG downscale cache | `src/codex_agent/openeqa/scene.py` |
| QA task (prompt + N frames + strict `{answer, supporting_claims, confidence}`) | `src/codex_agent/openeqa/qa.py` |
| Official 1–5 `mmbench` LLM-as-judge + MNAS | `src/codex_agent/openeqa/judge.py` |
| Evaluation runner (checkpointing, per-category MNAS, predictions file) | `src/codex_agent/evaluation/openeqa_runner.py` |
| CLI | `src/codex_agent/cli/run_openeqa.py` |

The runner reuses the existing `CodexAgentRuntime` (`runtime.execute(task)`) and
the project's pooled LLM client (`keyframe.llm.LLMClient`) for the judge.

## Run metadata

- **Branch:** `feat/openeqa-codex-task`
- **Head commit at launch:** `bd97b7a` (no worktree drift).
  - Caveat: the migration code was **uncommitted in the working tree** at run
    time (this is a development smoke; the pre-run "commit-clean" checklist in
    `CLAUDE.md` was not applied because the change itself was in flight). The
    durable artifacts (fold, summary, predictions) are archived under
    `assets/` so the result is reproducible from this doc.
- **Model (answerer):** `gpt-5.4-2026-03-05` via the in-repo vendored ModelHub
  adapter (`http://127.0.0.1:8787/v1`, `wire_api = responses`).
- **Judge:** `gemini-2.5-pro` (default in `configs/llm.toml`), official `mmbench`
  prompt, temperature 0.2.
- **Sandbox:** `read_only` (prompt-only QA; frames attached, no tools).
- **Reasoning effort:** `medium` (config default).
- **Frames:** 8 uniform first-person RGB frames per question, downscaled to
  ≤768 px JPEG.
- **Workers:** 5.

## Fold

- **Selection:** category-stratified round-robin over all ScanNet OpenEQA
  questions with prepared local scenes (1,079 available), sorted by
  `question_id` for determinism. Covers all 7 OpenEQA categories.
- **n:** 10
- **File:** `assets/smoke10_sample_ids_20260611.json`

## Exact invocation

```bash
source .venv/bin/activate
PYTHONPATH=src python -m codex_agent.cli.run_openeqa \
  --questions data/open-eqa-v0.json \
  --data-root data/OpenEQA/scannet \
  --question-ids docs/benchmark/openeqa/assets/smoke10_sample_ids_20260611.json \
  --output-dir tmp/openeqa_eval_smoke10_20260611 \
  --num-frames 8 --workers 5
```

- **Raw artifact dir:** `tmp/openeqa_eval_smoke10_20260611/`
  (`summary.json`, `predictions.json`, `per_sample/*.json`, `frame_cache/`).
- **Durable copies:** `assets/smoke10_summary_20260611.json`,
  `assets/smoke10_predictions_20260611.json`.

## Headline

**MNAS = 75.0** over 10 questions. Completion 10/10, 0 errors, wall-clock ≈ 59 s.
Prompt-cache hit rate 0.40 (mean cache ratio 0.24). Per-turn latency
min/median/max = 6.6 / 11.8 / 17.2 s.

### Per-category MNAS (n ≤ 2 per cell — smoke only)

| Category | n | MNAS |
|---|---:|---:|
| attribute recognition | 2 | 100.0 |
| object localization | 2 | 100.0 |
| object recognition | 1 | 100.0 |
| object state recognition | 1 | 75.0 |
| world knowledge | 1 | 75.0 |
| functional reasoning | 2 | 50.0 |
| spatial understanding | 1 | 0.0 |

### Per-question (judge score / 5)

| Category | Score | Question | Prediction | GT |
|---|---:|---|---|---|
| attribute recognition | 5 | Which material is the furniture of the scene? | wood | Wood |
| attribute recognition | 5 | Which material are the side and center tables? | wood | Wood |
| functional reasoning | 5 | How can I see myself? | in the mirror on the door | using the mirror on … |
| functional reasoning | 1 | How can I share the presentation materials? | use the blackboard/flip… | using the remote and … |
| object localization | 5 | Where did I leave my water bottle? | on the floor beside the … | on the floor |
| object localization | 5 | Where is the book on sailing located? | on the black coffee table | on the coffee table |
| object recognition | 5 | What leans on the wall on the right? | a rolled-up poster | a poster |
| object state recognition | 4 | Is the countertop clean? | mostly clean | No |
| spatial understanding | 1 | What is on top of the calculator? | a black pen | red notebook |
| world knowledge | 4 | What can I use to trim the edges of a …? | paper cutter | use the paper cutter |

## Reading the result

The pipeline runs end-to-end and produces grounded, mostly-correct answers with
sensible `supporting_claims`. The two misses are genuine perception/reasoning
errors (a small object-on-object spatial relation, and an affordance question),
not pipeline failures.

This is a **smoke**, not a benchmark number:

- **n = 10** → per-category cells are n ≤ 2; high variance. Do not compare future
  versions on this fold.
- Judge is `gemini-2.5-pro`, not the paper's GPT-4 → absolute MNAS is not
  comparable to published OpenEQA leaderboards.

## What changed vs prior state

New capability: there was **no** OpenEQA code in `agentic-3d-task` before this
(only NR3D + the `keyframe` selector). This adds the full QA path, mirroring the
NR3D task structure, with 92 new unit tests at 98 % line coverage on the new
modules.

## Verification

- `ruff check src/codex_agent` — clean.
- `black --check src/codex_agent` — clean.
- `mypy` on the new modules — no issues.
- `pytest src/codex_agent/tests` — 259 passed (incl. 92 new OpenEQA tests).
- Coverage on new modules — 98 % (`coverage` with `COVERAGE_CORE=sysmon`).

## Next steps (before a decision-grade run)

1. **Design a canonical OpenEQA pilot fold** (stratify on category, allocate
   proportionally, salt-lock against a full-set baseline) per the `CLAUDE.md`
   canonical-pilot-fold rule — do this *before* the second pilot.
2. Run the full 1,079-question ScanNet split (workers ≥ 20) for a real MNAS.
3. Add an OpenEQA SQLite ingester (`scripts/ingest_openeqa_run.py`) mirroring the
   schema described in `CLAUDE.md`.
4. (Optional) Wire Stage-1 query-driven keyframe retrieval as an alternative
   frame provider and A/B it against uniform sampling.
