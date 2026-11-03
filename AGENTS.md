# Repository Instructions

`AGENTS.md` is the single source of agent instructions. `CLAUDE.md` must remain a
relative symlink to `AGENTS.md`; edit this file only.

## Current code

- `src/keyframe/`: prepared-scene query parsing, object grounding, keyframe/BEV
  retrieval. Entry: `KeyframeSelector` in `keyframe_selector.py`.
- `src/codex_agent/`: Codex SDK runtime, NR3D/OpenEQA/SceneFunc3D tasks, tools,
  evaluation, and `cli/run_*.py` entry points.
- `codex_modelhub_adapter/`: local proxy to remote LLMs, not local LLM inference.
- Trust current source and `pyproject.toml` over archived module names and paths.

## Environment and assets

On Linux, use the existing `agentic3d` conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
scripts/link_agentic3d_assets.sh
```

- Shared models/sources: `/home/ysh/Models/agentic3d`, linked through ignored
  `external/models` and `external/third_party`. Do not vendor weights into Git.
- Do not recreate Linux `.venv` / `.venv-agents` or modify the legacy
  `conceptgraph` environment for current work. macOS uses `uv` / `.venv`.
- **Never use physical GPU 1.** Check GPU availability with `nvidia-smi` before
  launching GPU jobs; the environment defaults to physical GPU 0.
- Run downloads, installs, model services, evaluations, data preparation, and
  large test suites inside **tmux**. Do not stop unrelated sessions/processes.
- Exact versions, SDK/CLI compatibility, and service commands:
  [environment guide](docs/agentic3d_environment.md).

## Data on this machine

Verified on this Linux host on **2026-11-03**; check existence before reuse on
another machine. `/home/ysh/Datasets` points to `/data1/ysh/Datasets`.
Use the imported **processed_v1** packages for current evaluations, not the older
raw/processed copies below. `source scripts/datasets/processed_v1.sh` exports
validated paths for explicit CLI arguments (it does not start an evaluation).
See [dataset locations](docs/dataset_locations.md) for commands and verification.

| Data | Local path |
|---|---|
| NR3D current data-root + proposal packs | `/home/ysh/Datasets/nr3d/processed_v1/scannet/` |
| NR3D current full/pilot IDs | `/home/ysh/Datasets/nr3d/processed_v1/folds/` (`full_test_sample_ids.json`: 8584; `strat600_sample_ids.json`: 600) |
| OpenEQA current data-root | `/home/ysh/Datasets/open-eqa/processed_v1/scannet/` |
| OpenEQA current questions / pilot IDs | `/home/ysh/Datasets/open-eqa/processed_v1/open-eqa-v0.json` / `folds/` (1079 ScanNet Qs; pilot300) |
| NR3D original annotations/splits | `/home/ysh/Datasets/nr3d/raw/` (`nr3d.csv`, `train_scans.txt`, `test_scans.txt`) |
| NR3D extracted frames + ConceptGraph | `/home/ysh/Datasets/nr3d/scannet/<scene>/` (`raw/` contains RGB/depth/poses) |
| NR3D instance/segmentation metadata | `/home/ysh/Datasets/nr3d/scannet_aux/` |
| ScanNet source `.sens` and meshes | `/home/ysh/Datasets/ScanNet/scans/` |
| OpenEQA official questions | `/home/ysh/Datasets/open-eqa/data/open-eqa-v0.json` |
| OpenEQA episode frames | `/home/ysh/Datasets/open-eqa/data/frames/` (`scannet-v0/`, `hm3d-v0/`) |
| OpenEQA prepared ConceptGraph scenes | `/home/ysh/Datasets/OpenEQA/scannet/<clip>/conceptgraph/` |

The current checkout has **no `data/` directory**. Uppercase `OpenEQA` holds
prepared outputs; lowercase `open-eqa` holds the official dataset checkout.
The **old** `nr3d/scannet` lacks `pack_nr3d_v9_catalog_first`; the imported
`nr3d/processed_v1/scannet` includes it. The 8584 full fold is not the historical
7805-sample filtered fold. OpenEQA has 926 nonfinite poses across 13 clips;
see the dataset guide before pose-dependent tools (source data was not patched).
After import, the data disk has only ~40 GB free; check capacity before new
outputs/caches. No `SceneFuncVal-CG` package was found in these roots; do not
reuse other-machine NAS paths without verification.

## Runtime prerequisites

- Before Codex agent runs, configure/start the adapter and verify
  `curl -fsS http://127.0.0.1:8787/health`. SDK startup alone does not verify
  upstream credentials. Runtime settings: `src/codex_agent/config.py` and
  `.codex-home/config.toml`.
- Query parsing/judging reads `configs/llm.toml` (`KEYFRAME_LLM_CONFIG` override).
  SceneFunc3D requires a backend TOML and running MolmoPoint/SAM2 sidecars; use
  `scripts/scenefunc3d/serve_local_agentic3d.sh` for the combined local service.
- Datasets are not bundled. Verify input paths/formats; old absolute paths in
  scene metadata are provenance, not runnable defaults. SceneFunc3D annotation
  point IDs refer to **`raw/mesh.ply`**, not the ConceptGraph mesh.
- Never log or commit API keys, private `.env`/upstream TOMLs, or credentials.

## Changes and verification

- Follow the [Python quality guide](docs/python_code_agent_quality_guide.md),
  including its review checklist. Read relevant code/tests first, keep changes
  focused, use typed/validated boundaries, and add regression tests for fixes.
- Keep heavy/optional model imports lazy. Required dependency, checkpoint, or
  data failures must be explicit; no silent backend substitution or fake evidence.
- Use **Black**, not `ruff format`. For Python changes, run the relevant tests
  and these gates in the activated environment (full suites in tmux):

```bash
ruff check src/
black --check src/
mypy src/
pytest src/ -q
(cd codex_modelhub_adapter && python -m pytest tests/ -q)
```

- `pytest src/` does not include adapter tests. For docs-only changes, validate
  links/content; run `mkdocs build` when changing the documentation site.
  Keep `docs/` indexes and `mkdocs.yml` navigation aligned when adding pages.
- Report changes, actual verification results, and anything not tested.
  Commit/push only when the user authorizes it.

## Benchmark discipline

- Before any evaluation, require a clean tracked working tree and record branch,
  HEAD SHA, and executed-code SHA (including worktree drift). Obtain commit
  authorization if needed; do not run on undocumented dirty code.
- Use frozen, matched folds from the benchmark records (NR3D: canonical
  strat600); verify the sample-ID file exists. Do not invent ad-hoc comparison
  subsets. Label reproductions with `REPRO_<YYYYMMDD>_<source-run-id>`.
- Every run, including partial/negative results, needs a new dated version record
  under `docs/benchmark/<name>/`: exact command/config, code/fold/model/judge,
  artifact locations, metrics, comparison, and caveats. Do not overwrite history.
- Preserve per-sample/tool logs and ingest runs into the benchmark SQLite DB.
  OpenEQA uses `scripts/ingest_openeqa_run.py` and
  `docs/benchmark/openeqa/runs.sqlite`; verify/provide ingestion for other tracks.
- Update the benchmark timeline, existing leaderboard, and site navigation.
  Process records must not live only in `tmp/`, outputs, or commit messages.
  See the [benchmark archive](docs/benchmark/index.md).
