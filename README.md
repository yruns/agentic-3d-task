# 3DVLMReasoning / Agentic 3D Task

Evidence-seeking agents for 3D scene understanding.

This repository currently packages two Python modules:

- `keyframe`: query-driven keyframe retrieval over prepared ConceptGraph-style
  3D scenes.
- `codex_agent`: a task-agnostic Codex Agent SDK runtime plus NR3D and OpenEQA
  task families.

Older documents may still mention `query_scene` and `agents`; those names are
historical. The current wheel packages are `src/keyframe` and
`src/codex_agent`.

## What This Repo Does

The project studies a two-stage 3D reasoning pattern:

1. **Evidence retrieval**: parse a natural-language query, ground likely target
   and anchor objects in a prepared 3D scene, and select first-person frames or
   BEV views that are useful evidence.
2. **Agentic reasoning**: run a Codex/VLM agent that can inspect retrieved
   evidence, call scene tools, and return structured answers for benchmark
   tasks.

The two production benchmark tracks are:

- **NR3D visual grounding**: select one object proposal for a referring
  expression and score it with oriented 3D IoU / Acc@tau.
- **OpenEQA question answering**: answer open-ended questions about prepared
  ScanNet clips and score with the official 1-5 `mmbench` judge mapped to MNAS.

## Repository Layout

```text
src/
├── keyframe/                 # Query parsing, spatial grounding, keyframe/BEV selection
│   ├── keyframe_selector.py  # Production entry point: KeyframeSelector
│   ├── query_executor.py     # Executes structured grounding queries
│   ├── lightweight_conceptgraph.py
│   ├── bev/                  # Schematic / mesh-aware BEV rendering helpers
│   ├── llm/                  # LLM config/client for query parsing
│   ├── models/               # Typed scene, hypothesis, and result models
│   ├── parsing/
│   ├── spatial/
│   └── tests/
└── codex_agent/              # Codex Agent SDK runtime and benchmark tasks
    ├── runtime.py            # Task-agnostic Codex turn driver
    ├── config.py             # Central CODEX_AGENT_* configuration
    ├── tasks/                # CodexTask protocol
    ├── nr3d/                 # NR3D grounding task, proposals, tools, geometry
    ├── openeqa/              # OpenEQA QA task, tools, judge, scene loading
    ├── evaluation/           # Batch runners, checkpointing, metrics
    ├── cli/                  # python -m codex_agent.cli.run_*
    └── tests/

src/codex_adapter/           # Stateful Codex Responses gateway and provider codecs
configs/llm.example.toml      # Placeholder LLM config; real configs are gitignored
docs/                         # Markdown source for the MkDocs Material site
docs/benchmark/               # Durable NR3D / OpenEQA run records and assets
scripts/                      # Utility scripts such as OpenEQA run ingestion
```

## Installation

On macOS, use the project `.venv` managed by `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Install optional groups only when you need them:

```bash
# Vision/BEV/keyframe work with prepared ConceptGraph scenes
uv pip install -e ".[dev,vision]"

# CLIP semantic fallback
uv pip install -e ".[dev,vision,clip]"

# Codex Agent SDK runtime and benchmark agents
uv pip install -e ".[dev,vision,codex]"
```

On Linux, use the `agentic3d` conda environment and the shared models/sources
under `/home/ysh/Models/agentic3d`. See
[`docs/agentic3d_environment.md`](docs/agentic3d_environment.md) for activation,
asset links, pinned versions, and native model-service commands. The older
`conceptgraph` environment is preserved. Do not recreate a project `.venv` on Linux.

## Quick Start: Keyframe Retrieval

`KeyframeSelector` is the main entry point for Stage-1 evidence retrieval.
It expects a prepared scene directory with ConceptGraph assets such as
`pcd_saves/`, RGB/depth frames, camera trajectory, and visibility indices.

```python
from keyframe import KeyframeSelector

selector = KeyframeSelector.from_scene_path("/path/to/scene/conceptgraph")
result = selector.select_keyframes_v2(
    "the pillow on the sofa nearest the door",
    k=3,
)

print(result.keyframe_indices)
print(result.keyframe_paths)
print(result.summary())
```

The selector:

1. loads scene objects, camera poses, frames, and object-view visibility;
2. parses the query into ranked hypotheses with an LLM;
3. executes those hypotheses with category, attribute, and spatial constraints;
4. selects frames that jointly cover the target and relevant anchors.

`torch` and `open_clip` are optional. If they are not installed, the selector
falls back to string and multi-label category matching.

## Quick Start: NR3D Grounding

NR3D runs ask the Codex agent to select one proposal id from a prepared scene
proposal pool. Ground-truth target ids, categories, and boxes are not included
in the prompt.

On the configured Linux host, use the imported `processed_v1` canonical pilot
(see [dataset locations](docs/dataset_locations.md) for prerequisites and full folds).
Run evaluations in tmux after recording the clean code revision.

```bash
source scripts/datasets/processed_v1.sh
python -m codex_agent.cli.run_nr3d \
  --sample-ids "$NR3D_PILOT_SAMPLE_IDS" \
  --data-root "$NR3D_DATA_ROOT" \
  --pack-name "$NR3D_PACK_NAME" \
  --output-dir tmp/nr3d_run \
  --workers 1 \
  --tools \
  --reasoning-summary auto
```

The runner checkpoints each sample under `--output-dir`, resumes completed
samples, and writes `summary.json` with mean IoU, Acc@0.25, Acc@0.50, and
prompt-cache metadata.

## Quick Start: OpenEQA QA

OpenEQA runs are tool-based. No images are attached to the initial turn; the
agent must fetch all visual evidence through scene tools such as
`keyframe_selector`, `view_frame`, `view_crop`, `view_bev`, and `list_objects`.

```bash
source scripts/datasets/processed_v1.sh
python -m codex_agent.cli.run_openeqa \
  --questions "$OPENEQA_QUESTIONS" \
  --data-root "$OPENEQA_DATA_ROOT" \
  --question-ids "$OPENEQA_PILOT_QUESTION_IDS" \
  --output-dir tmp/openeqa_run \
  --workers 1
```

Add `--no-judge` when you only want predictions and do not want to call the
LLM-as-judge. With judging enabled, the runner writes checkpoints,
`summary.json`, and `predictions.json`.

## Codex Multi-Protocol Adapter

The active gateway is `src/codex_adapter/`, using the root project's `adapter`
extra and lock file. It supports native GPT Responses and a dedicated Gemini
Chat codec with authenticated, encrypted persistent state. Codex still sees a
Responses provider and executes the tools itself.

Models must be registered exactly. Each model contains multiple upstreams, each
with its own URL, direct AK, and weight. New sessions select one complete
upstream, then retain that URL/AK binding across turns and restarts.
Unknown models fail without fallback; rate limits never trigger account
failover. Stable affinity and intact history preserve cache eligibility, not a
guarantee that the upstream will report a cache hit.

Copy `configs/codex_adapter.example.toml` to the ignored private configuration,
fill each upstream's `ak` directly, and protect the file with owner-only
permissions. Upstream AKs are not read from environment variables. Run
installation and service commands inside tmux, using the existing environment.

```bash
python -m codex_adapter check --config configs/codex_adapter.toml
```

```bash
python -m codex_adapter init-state --config configs/codex_adapter.toml
```

```bash
python -m codex_adapter serve --config configs/codex_adapter.toml
```

For a fresh Codex home:

```bash
scripts/codex_agent/bootstrap_codex_home.sh --model gpt-5.4-2026-03-05
```

See the [adapter operations and capability guide](docs/codex_agent/adapter.md)
for installation, model registration, authentication, state recovery, Gemini's
buffered SSE and unsupported compact, and local verification. Gemini schema/image
capability declarations require upstream verification before task use. The old
`codex_modelhub_adapter/` implementation is retired; archived benchmark commands
are not current launch instructions. Never commit or log AKs or opaque state.

## Benchmarks

Benchmark records live under `docs/benchmark/` and are the durable source of
run history.

### NR3D

The NR3D archive tracks both:

- Stage-1 keyframe target-id coverage (`hit@K`);
- Codex agent visual grounding accuracy (oriented 3D IoU / Acc@tau).

On the canonical `strat600` fold, the documented trajectory is:

- prompt-only catalog baseline: Acc@0.25 = 63.83%;
- tools with loop/network fixes: Acc@0.25 = 78.33%;
- reasoning-summary `/responses` path: Acc@0.25 = 85.33%.

See `docs/benchmark/nr3d/README.md` for the full timeline and caveats.

### OpenEQA

The current OpenEQA production baseline is the no-default-frames pipeline:
the agent starts with zero images and fetches all evidence through tools.

On the documented full ScanNet split:

- v3 with 8 uniform seed frames + tools: MNAS = 73.96;
- v4 no seed frames, tools only: MNAS = 64.09.

The v4 number is the baseline for the current architecture. See
`docs/benchmark/openeqa/README.md` for judge caveats and failure analysis.

## Documentation

Project documentation is Markdown-first under `docs/` and rendered as a
MkDocs Material site for human browsing.

```bash
mkdocs serve
mkdocs build
```

The important entry points are:

- `docs/index.md`: documentation home;
- `docs/python_code_agent_quality_guide.md`: mandatory Python coding standard;
- `docs/codex_agent/index.md`: Codex Agent design and investigation notes;
- `docs/benchmark/index.md`: benchmark archive overview;
- `AGENTS.md` and `CLAUDE.md`: operational rules for coding agents.

Top-level `README.md`, `AGENTS.md`, and `CLAUDE.md` are intentionally not part
of the MkDocs navigation.

## Development Checks

For Python code changes, follow `AGENTS.md` and run the project gate:

```bash
source .venv/bin/activate
ruff check src/
black src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

For docs-only changes, run a narrower check that matches the touched files,
for example:

```bash
mkdocs build
```

If a required command cannot run in the current environment, report that
explicitly with the remaining risk.

## Data Notes

On the current Linux host, use the shared `/home/ysh/Datasets` root instead of
another project's checkout. See [Local Dataset Locations](docs/dataset_locations.md)
for current `nr3d/processed_v1` and `open-eqa/processed_v1` packages, full/pilot
folds, and original NR3D/OpenEQA/ScanNet data. Use
`source scripts/datasets/processed_v1.sh` before the examples above; the generic
CLIs still require explicit data/fold arguments.

Large benchmark assets are local-only and not bundled. A fresh clone has no
`data/` directory; do not silently use old project-relative data paths or mix
original annotations with a different processed package's sample IDs.

Prepared scenes may contain provenance paths from the machine that generated
them. Treat those paths as metadata, not as runnable local paths.

## License

MIT License
