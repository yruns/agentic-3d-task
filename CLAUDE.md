# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management (Platform-Dependent)

This project uses **different Python environments depending on the platform**:

### Linux (Ubuntu) — Conda `conceptgraph`

On Linux, use the **conda `conceptgraph` environment** for all work. This env has torch, SAM, open_clip, Florence-2, hydra, and all pipeline dependencies.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
python ...  # resolves to conda env's python

# Install new packages
conda install <package>
pip install <package>  # within conda env
```

**The project `.venv` has been deleted on Linux.** Do NOT recreate it — it shadowed conda's python and caused silent failures (see `docs/0325.md`).

### macOS (Darwin) — uv / .venv

On macOS, use **`uv`** for Python package management with `.venv`:

```bash
# Install dependencies (creates .venv automatically)
uv pip install -e ".[dev]"

# Install with all optional dependencies
uv pip install -e ".[dev,full,agents]"

# Add new dependency
uv pip install <package>

# Sync from pyproject.toml
uv pip sync
```

### How to detect platform in scripts

```bash
if [[ "$(uname -s)" == "Linux" ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
else
    # macOS — uv / .venv
    source .venv/bin/activate 2>/dev/null || uv venv && source .venv/bin/activate
fi
```

Note: The `agents` optional dependency group needs Python 3.11+ (`deepagents` is on public PyPI but only ships wheels for >=3.11). Since the Linux conda `conceptgraph` env is Python 3.10, agents work uses a separate venv: `uv venv .venv-agents --python 3.11 && uv pip install -e ".[dev,agents]"`. Activate `.venv-agents` for agents/Stage-2 tests; keep `conceptgraph` for feasibility / perception code.

## Development Commands

```bash
# Run all tests
pytest src/ -v

# Run Stage 1 tests only
pytest src/query_scene/tests/ -v

# Run Stage 2 tests only
pytest src/agents/tests/ -v

# Run benchmark tests only
pytest src/benchmarks/tests/ -v

# Run single test file
pytest src/agents/tests/test_stage2_deep_agent.py -v

# Run single test function
pytest src/agents/tests/test_stage2_deep_agent.py::test_function_name -v

# Linting
ruff check src/

# Formatting
black src/

# Type checking
mypy src/
```

## Architecture Overview

This is a **two-stage framework for 3D scene understanding with evidence-seeking VLM agents**:

### Stage 1: Query-Driven Keyframe Retrieval (`src/query_scene/`)

Handles task-conditioned evidence retrieval from 3D scene graphs:

- **Query parsing**: `query_parser.py` - Parses natural language queries into structured `HypothesisOutputV1` using LLM
- **Query execution**: `query_executor.py` - Executes parsed queries against scene indices
- **Keyframe selection**: `keyframe_selector.py` - Main entry point (`select_keyframes_v2()`) for selecting task-relevant keyframes
- **Scene indices**: `index_builder.py` - Multi-granularity CLIP indexing (region → object → point), visibility index, spatial index
- **Spatial relations**: `spatial_relations.py` - Geometric relation checking between objects

The Stage 1 output is treated as **visual evidence entry points** (high recall, not necessarily high precision), not final answers.

### Stage 2: VLM Agentic Reasoning (`src/agents/`)

A ReAct-style VLM agent built on **LangChain v1 + DeepAgents** that reasons over retrieved evidence:

- **Agent core**: `stage2_deep_agent.py` - `Stage2DeepResearchAgent` class
- **Data models**: `models.py` - Pydantic schemas (`Stage2TaskSpec`, `Stage2EvidenceBundle`, `Stage2AgentResult`, etc.)
- **Adapters**: `adapters.py` - Bridge Stage 1 output to Stage 2 input (`build_stage2_evidence_bundle()`)
- **Benchmark adapters**: `benchmark_adapters.py` - Unified interface for OpenEQA, ScanNet, Replica, etc.
- **Tools**: `tools/` - Agent tools like `request_crops.py`, `hypothesis_repair.py`
- **Tracing**: `trace.py`, `trace_server.py` - Execution trace recording and HTML rendering

### Key Design Principles

1. **Hypothesis as soft prior**: Stage 2 treats Stage 1 hypotheses as soft priors to verify/correct, not ground truth
2. **Evidence-seeking**: Agent actively decides what additional evidence to request (more views, crops, BEV)
3. **Unified task interface**: Single agent handles QA, visual grounding, navigation planning, manipulation
4. **Budget-aware reasoning**: Agent operates within fixed token/image budgets

### Benchmark Loaders (`src/benchmarks/`)

- `openeqa_loader.py` - OpenEQA dataset
- `sqa3d_loader.py` - SQA3D dataset
- `scanrefer_loader.py` - ScanRefer dataset

## Current Local Checkout Reality

These notes reflect the local repository state verified on **2026-03-23**.

### Data under `data/`

The local checkout currently has:

- `data/OpenEQA/scannet/`
- `89` valid prepared scene directories
- about `37G` of prepared OpenEQA ScanNet scene assets

Each scene is already packaged into a ConceptGraph-style prepared layout under:

- `data/OpenEQA/scannet/<clip_id>/conceptgraph/`

with assets such as:

- `*-rgb.png`, `*-depth.png`, pose `*.txt`
- `intrinsic_*.txt`, `extrinsic_*.txt`, `traj.txt`
- `mesh.ply`
- `indices/`
- `gsa_detections_ram_withbg_allclasses/`
- `gsa_vis_ram_withbg_allclasses/`
- `pcd_saves/`
- `scene_info.json`

Important:

- this is **not** the official OpenEQA benchmark repo layout
- there is currently **no** `data/benchmark/` or `data/benchmarks/`
- there is currently **no** local `data/open-eqa-v0.json`

Implication:

- use `data/OpenEQA/scannet/*/conceptgraph` for prepared-scene / full-pipeline work
- do **not** assume `src/benchmarks/openeqa_loader.py` can load the local `data/OpenEQA/` tree as-is

### Dataset interface caveat

The standard `ScanNetAdapter` expects raw ScanNet-style scene folders such as:

- `sceneXXXX_XX/color/`
- `sceneXXXX_XX/depth/`
- `sceneXXXX_XX/pose/`
- `sceneXXXX_XX/intrinsic/`

It does **not** directly consume the prepared `conceptgraph/` scene packages under `data/OpenEQA/scannet/`.

### Scene metadata provenance caveat

`conceptgraph/scene_info.json` files may contain old absolute source paths from the machine that originally generated the assets (for example `/home/ysh/...`).

Treat those fields as provenance only, not as runnable local paths.

## Repository Transition Notes

The repository is functionally migrated, but it still carries compatibility layers and historical terminology.

### Canonical modules to prefer

- Stage 1 selector implementation: `src/query_scene/keyframe_selector.py`
- Stage 1 -> Stage 2 bridge: `src/agents/stage1_adapters.py`

### Compatibility areas still present

- `src/query_scene/retrieval/__init__.py` lazily re-exports selector symbols
- `src/query_scene/retrieval/keyframe_selector.py` still exists as a legacy duplicate
- `src/agents/adapters.py` is a backward-compatible export shim
- `src/agents/adapters/` and `src/agents/adapters_pkg/` are benchmark-adapter abstractions, not the canonical Stage 1 -> Stage 2 bridge

### Tests and packaging

Test layout is split across:

- `src/**/tests`
- `tests/`

But `pyproject.toml` currently uses:

- `testpaths = ["src"]`

So default pytest discovery does not automatically include every root-level test module.

Wheel packaging currently includes:

- `src/query_scene`
- `src/agents`
- `src/benchmarks`
- `src/utils`

and currently omits:

- `src/dataset`
- `src/config`
- `src/evaluation`

Do not assume the built wheel mirrors the full source tree.

### Documentation interpretation rule

When docs mention:

- `conceptgraph/*`
- `data/benchmark/*`
- `data/benchmarks/*`

verify whether the note is historical migration context or current local truth before acting on it.

## Stage 2 Backend Configuration

Default VLM backend is `gpt-5.2-2025-12-11` via Azure-compatible endpoint. Configuration is in `Stage2DeepAgentConfig`:

- Uses single-key `AzureChatOpenAI` client (not connection pooling)
- Base URL: internal GenAI endpoint
- Session ID required in `extra_body` for prompt caching
- Gemini available as override but not default (unstable FC with DeepAgents)

### Codex SDK ModelHub Adapter AKs

For Codex Agent SDK NR3D runs, the local ModelHub adapter uses a private,
gitignored weighted TOML upstream file:

```text
/Users/bytedance/aispace/codex_modelhub_adapter/.modelhub_upstreams.toml
```

The real AKs are in that file. Do not copy them into tracked docs, benchmark
records, git commits, or terminal logs. The current private pool is
`gpt-5.4-2026-03-05` on the office endpoint
`https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online`, with weights
`5:1:5` across aliases `gpt54_a`, `gpt54_b`, and `gpt54_c`.
Codex SDK keeps `extra.session_id` stable for prefix caching and sends a
per-turn `extra.chat_run_id` so the adapter can distribute concurrent requests
across the weighted AK pool.

Start the adapter with:

```bash
cd /Users/bytedance/aispace/codex_modelhub_adapter
export AIDP_MODELHUB_UPSTREAMS_TOML=/Users/bytedance/aispace/codex_modelhub_adapter/.modelhub_upstreams.toml
uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787
```

Durable config/smoke record:
`docs/benchmark/nr3d/codex_sdk_modelhub_adapter_config_20260607.md`.

## JSON Schemas

Output schemas are in `schema/` directory:
- `hypothesis_output_v1.json` - Schema for Stage 1 query parsing output

## Benchmark Process Documentation (MANDATORY)

**Every benchmark evaluation MUST leave a permanent process record under `docs/benchmark/<name>/`. Process docs in `tmp/`, commit messages, ad-hoc handoff files, or `outputs/` directories are not durable — they get garbage-collected, lost in branch deletions, or buried in commit history.**

### Layout

```
docs/benchmark/
├── README.md                    ← index across benchmarks (already exists)
└── <benchmark>/
    ├── README.md                ← per-benchmark version timeline + summary leaderboard
    ├── leaderboard.md           ← extended public leaderboard with paper references
    ├── <vN>_<short_tag>_<YYYYMMDD>.md  ← one file per evaluated version of OUR pipeline
    └── *.html                   ← optional dashboards / case studies (kept verbatim)
```

The `<vN>` is the internal version of *our* pipeline, not the benchmark version. The `<short_tag>` is one or two words capturing the change (e.g. `trajectory_aware`, `enrichment`, `callbacks`). The `<YYYYMMDD>` is the date the run was harvested.

### When to write a benchmark doc

Write a new `<vN>_<tag>_<YYYYMMDD>.md` whenever:

- A new evaluation run completes against any benchmark — even ablations, partial folds, and negative results.
- A reproduction or rerun of an earlier version produces materially different numbers.
- A judge / model / fold / prompt change retro-affects an existing version's numbers.

Update the per-benchmark `README.md` and `leaderboard.md` at the same time so the index never lags behind the evidence.

### Pre-run checklist (MANDATORY)

Before launching ANY benchmark run — including reruns, ablations, A/B variants, or reproductions of older results — you MUST:

1. **Commit pending changes.** `git status` must report no modified tracked files at the moment the eval starts. Commit:
   - Code changes (toggles, prompts, playbook variants)
   - SQLite DB updates from prior runs
   - Doc / metric edits from the previous iteration

2. **Capture both commit SHAs**:
   - **Head commit** — where the branch is at launch (`git rev-parse HEAD`).
   - **Run-time code commit** — where the code being executed lives. Equal to head commit unless you're running from a `git worktree` at an older commit (e.g. to reproduce a historical run); in that case capture `git -C <worktree> rev-parse HEAD` separately.

3. **Record both SHAs in the version doc**, in this format:

   ```
   Branch:                 feat/v9-1-selectors-return-images
   Head commit at launch:  c2c52d0
   Run-time code commit:   d5f40ba   (worktree at /var/folders/.../v91fix-worktree)
   ```

   When head == run-time, list once and note "no worktree drift".

4. **Tag reproductions explicitly.** Prefix `run-id` with `REPRO_<YYYYMMDD>_<source-run-id>` so SQLite history shows the lineage. Example: `v9_1_fix_random100_REPRO_20260516`.

This makes every result reproducible from a commit + sample-ids fold + leaderboard-metrics JSON — and lets future agents bisect regressions across iterations without guessing what code was actually live at each run.

### Canonical pilot folds (MANDATORY)

**For any quick / small-batch validation run, you MUST use the canonical
pilot fold for that benchmark.** Do NOT spin up an ad-hoc `head -N`, a
fresh `random.sample()`, or a smoke fold of convenience. Ad-hoc subsets
make iteration-vs-iteration comparison impossible: variance across two
arbitrary 100-question samples is large enough (±5–10 pp on tier metrics)
to flip wins/losses purely by sampling noise.

Canonical pilot folds are designed once, salt-locked to track the full
benchmark within a known band, and documented under the per-benchmark
process archive. Use them by default for:

- pilot ablations (toggles, prompt variants, playbook variants)
- pre-merge smoke runs of new code paths
- regression scans before launching the full set
- A/B comparisons that don't need full-set statistical power

| Benchmark | Canonical pilot fold | n | Tracks full-set within | Design doc |
|---|---|---:|---|---|
| **NR3D** | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` (durable copy: `docs/benchmark/nr3d/assets/v9_3_strat600_sample_ids_20260517.json`) | **600** | **±0.19 pp** salt-locked on v9.1_fix; **±2.3 pp 90 % band** on Overall in bootstrap | [`docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md`](docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md) |

NR3D-specific notes:

- **Stratified on `(is_easy × is_view_dep)`** with proportional largest-remainder allocation, so all 5 leaderboard columns (Overall / Easy / Hard / View-Dep / View-Indep) are calibrated jointly.
- **The old random100 fold** (`tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`) is preserved for **historical backward-compat only**. Do NOT use it for any new decision-grade A/B. Random100's per-tier ±5–10 pp noise was a major source of the v9.1 → v9.2 misread earlier — strat600 fixes that.
- **Run it like this:**
  ```bash
  PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
      --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
      --data-root data/nr3d/scannet \
      --pack-name pack_nr3d_v9_catalog_first \
      --output-dir tmp/nr3d_eval_<run_id>/ \
      --workers 20
  PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
      --side-by-side tmp/nr3d_eval_<run_id>/side_by_side.json \
      --nr3d-data-root data/nr3d --phase8-data-root data/nr3d/scannet \
      --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
      --output tmp/nr3d_eval_<run_id>/leaderboard_strat600.json
  ```
- **Variance budget when comparing two strat600 runs**: Overall ±2.3 pp 90 %, Easy/V-Indep ±2.7–2.9 pp, Hard ±3.5 pp, V-Dep ±4.5 pp (n=119). Deltas smaller than these bands should be confirmed on the FULL 7805 before being claimed as a real change.

**When a benchmark does not yet have a canonical pilot fold**, the rule is to design one *before* the second pilot run, not after the fifth. The bar is: stratify on every leaderboard tier column, allocate proportionally, salt-search against the strongest current full-set baseline. See `scripts/build_nr3d_strat600_fold.py` + `scripts/search_nr3d_strat600_salt.py` for the reference implementation pattern.

### Required content per version doc

- **Branch + tip commit** at run time. Plus run-time code commit if different (worktree case).
- **Exact CLI invocation** or path to the launcher script (e.g. `scripts/run_v15_eval_matrix.sh`).
- **Raw artifact directory** under `tmp/...` so numbers can be re-derived from JSON.
- **Judge model name** for LLM-as-judge benchmarks (e.g. `gemini-2.5-pro`).
- **Fold size and selection mechanism** (e.g. `--force-selection /path/to/frozen.json` plus the frozen file's existence in `tmp/<benchmark>_artifacts/`).
- **What changed** vs the previous version (code-level deltas with commit hashes when possible).
- **Headline metric** (e.g. MNAS) plus per-category breakdown when applicable.
- **Cross-version comparison** on the matched column / fold (e.g. v14 vs v15 stage2 MNAS on the same frozen 1050Q set).
- **Caveats** — partial judging, fold mismatches, judge differences, anything that makes the number not directly comparable to a prior baseline or a published number.

### Mandatory: SQLite ingestion of per-run logs

Every benchmark run MUST be ingested into the per-benchmark SQLite database immediately after evaluation completes. Without this, ad-hoc per-question regression analysis (`Δ = chassis vs v15` per qid, per category, with tool traces) is impossible — and that analysis is what tells you *why* the headline metric moved.

For OpenEQA, the canonical ingester is `scripts/ingest_openeqa_run.py` and the canonical DB is `docs/benchmark/openeqa/runs.sqlite`:

```bash
python scripts/ingest_openeqa_run.py \
    --output-dir tmp/openeqa_eval_<run>/ \
    --run-id <run> \
    --branch <branch> \
    --commit <short_sha> \
    --judge-model gemini-2.5-pro \
    --notes "<one-line summary of what's different>" \
    --db docs/benchmark/openeqa/runs.sqlite
```

Tables: `runs`, `samples` (one row per qid with question + GT + chassis answer + stage2 score + tool count), `tool_calls` (one row per tool invocation with input/response), `llm_calls` (one row per chat-completion with prompt/cached/completion tokens). Add a query in the version doc that reproduces the headline number from the DB.

When adding a new benchmark, write a parallel `scripts/ingest_<benchmark>_run.py` mirroring this schema. Reuse the `runs` / `tool_calls` / `llm_calls` columns; only `samples` should diverge to fit the benchmark's own per-question shape.

### Per-LLM-call durability (going forward)

The `token_usage.jsonl` produced by `scripts/run_openeqa_with_token_log.py` records prompt/cached/completion tokens per chat-completion response by monkey-patching `BaseChatOpenAI._generate`. It is process-global (not yet keyed by question_id). Future iterations of the pilot SHOULD use loguru contextvars (or langchain callbacks with `run_id` tagging) to associate each LLM call with the active question_id, so the ingester can populate `llm_calls.question_id` properly. Until that lands, treat `llm_calls` aggregates as run-level only.

Console log per-question slicing has the same gap: the tee'd `/tmp/<run>.log` is one large append-only stream. Loguru `logger.bind(question_id=...)` plus a per-qid sink would close this gap.

### When adding a new benchmark

Mirror the OpenEQA layout exactly. Create:

1. `docs/benchmark/<new_benchmark>/README.md` — version timeline.
2. `docs/benchmark/<new_benchmark>/leaderboard.md` — extended public leaderboard if a public leaderboard exists.
3. `docs/benchmark/<new_benchmark>/<v1>_<tag>_<date>.md` — first evaluation doc.

Then add a row to `docs/benchmark/README.md §Active Benchmarks`.

### Anti-patterns

- Putting evaluation results only in commit messages.
- Putting per-run summaries under `tmp/` or `outputs/` and assuming they will survive.
- Writing a single growing log file instead of dated per-version files.
- Updating numbers in an old version doc instead of writing a new one.
- Leaving the per-benchmark README's leaderboard / version-timeline stale after a new run completes.

The point is to make process documentation immutable, dated, and tracked, so any future agent or paper-writer can audit exactly what was run, on what code, on what fold, and what the judge said — without re-running anything.

## Long-Running Tasks: tmux (MANDATORY)

**All long-running commands (training, evaluation, batch processing, large test suites, data preparation) MUST be executed inside tmux sessions.** This prevents task loss from SSH disconnection, terminal closure, or Bash tool timeout (120s default, 600s max).

### Why tmux is required

- Bash tool has a hard timeout limit — long tasks will be killed
- Network interruptions lose running work without tmux
- tmux enables parallel task execution across multiple windows/panes

### Basic tmux workflow

```bash
# Create a named session for a long task
tmux new-session -d -s eval "cd /Users/bytedance/project/3DVLMReasoning && python -m src.evaluation.batch_eval --scenes all"

# Create multiple parallel sessions
tmux new-session -d -s eval-stage1 "pytest src/query_scene/tests/ -v 2>&1 | tee /tmp/stage1.log"
tmux new-session -d -s eval-stage2 "pytest src/agents/tests/ -v 2>&1 | tee /tmp/stage2.log"
```

### Sending commands to an existing tmux session

```bash
# Send a command to a running session
tmux send-keys -t eval "echo 'hello'" Enter

# Send Ctrl+C to cancel a running process
tmux send-keys -t eval C-c
```

### Capturing output from a tmux session

```bash
# Capture the last N lines of visible output from a pane
tmux capture-pane -t eval -p -S -50

# Capture entire scrollback buffer to a file
tmux capture-pane -t eval -p -S - > /tmp/eval_output.txt

# Then read the captured output
cat /tmp/eval_output.txt
```

### Checking session status

```bash
# List all active tmux sessions
tmux list-sessions

# Check if a specific session is still running (non-zero exit = not found)
tmux has-session -t eval 2>/dev/null && echo "running" || echo "finished/not found"

# Peek at the last few lines of output without attaching
tmux capture-pane -t eval -p -S -10
```

### Parallel task pattern

For independent tasks that can run concurrently:

```bash
# Launch parallel evaluation jobs
tmux new-session -d -s job-openeqa "cd /Users/bytedance/project/3DVLMReasoning && python run_openeqa.py 2>&1 | tee /tmp/openeqa.log"
tmux new-session -d -s job-sqa3d  "cd /Users/bytedance/project/3DVLMReasoning && python run_sqa3d.py 2>&1 | tee /tmp/sqa3d.log"
tmux new-session -d -s job-scanref "cd /Users/bytedance/project/3DVLMReasoning && python run_scanrefer.py 2>&1 | tee /tmp/scanrefer.log"

# Monitor all jobs
tmux list-sessions
for s in job-openeqa job-sqa3d job-scanref; do
  echo "=== $s ===" && tmux capture-pane -t "$s" -p -S -5
done
```

### Cleanup

```bash
# Kill a specific session when done
tmux kill-session -t eval

# Kill all sessions
tmux kill-server
```

## Automation Scripts

### auto-claude.sh

Migration automation script that invokes Claude for multi-phase tasks. **Must use `ttadk` CLI** (ByteDance internal):

```bash
# Run full migration
./auto-claude.sh

# Run specific phase (1-6)
./auto-claude.sh --phase 2

# Preview without changes
./auto-claude.sh --dry-run

# Check migration status
./auto-claude.sh --status
```

The script uses `ttadk code` with piped prompts:
```bash
echo "$prompt" | ttadk code --model "claude-opus-4-5" \
    -a "--dangerously-skip-permissions --print --output-format stream-json"
```

Note: The standard `claude` CLI fallback does not work in this environment.

## Strict No-Fallback Rule (MANDATORY)

**Every pipeline step MUST succeed exactly as designed. Silent fallbacks are strictly prohibited.**

- If a dependency fails to import (e.g., SAM, torch), the script MUST raise an error and stop — never silently degrade (e.g., never fall back to bbox-rectangle masks when SAM fails).
- If a required file, checkpoint, or data path is missing, fail immediately with a clear error message.
- `try/except` blocks that swallow `ImportError` and substitute `None` are forbidden for critical dependencies.
- Before running any pipeline step, verify that all prerequisites are met (correct python, GPU available, models loadable). If verification fails, fix the root cause before proceeding.
- This applies to all code in `conceptgraph/`, `scripts/`, and `src/` — no exceptions.

### Python Environment on Linux

On Linux, the `.venv` has been deleted. After `conda activate conceptgraph`, bare `python` correctly resolves to conda's python with all dependencies (torch, SAM, open_clip, Florence-2). See the **Package Management** section above for details.

### GPU Usage on Linux (MANDATORY)

**GPU 1 is broken on the Linux server.** Do NOT use `CUDA_VISIBLE_DEVICES=1`. It causes CUDA initialization failures that crash the entire process.

When running GPU tasks, use any of GPUs 0, 2, 3, 4, 5, 6, 7:

```bash
# Single GPU
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU parallel (assign one scene per GPU, skip GPU 1)
for scene_gpu in "scene_a:0" "scene_b:2" "scene_c:3" "scene_d:4"; do
    ...
done
```

Always check GPU availability before launching: `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader`
