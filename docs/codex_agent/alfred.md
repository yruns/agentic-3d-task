# ALFRED: online OpenEQA scene mapping and execution

ALFRED uses our OpenEQA object–view mapping method incrementally. Each new RGB
observation supplies fixed-vocabulary detections, instance masks, CLIP features
and predicted metric depth. An episode-local memory associates and fuses objects,
retains original views, updates the exploration map and serves the next decision.
This iteration implements incremental mapping and bounded typed actions; generated
local Python policy programs remain future work.

The implementation replaces the earlier FILM perception and RGB-histogram mapper.
Historical code/configuration and 50 official scores remain archived in
[v8](../benchmark/alfred/v8_pilot50_complete_20261105.md). The new experiment uses
[frozen pilot15](../benchmark/alfred/folds/pilot15.json): seven seen and eight unseen,
selected deterministically from the original pilot50 without reading scores.
See [v9 setup and protocol](../benchmark/alfred/v9_online_mapping_20261105.md) for
actual validation status; old scores are not results of this implementation.

## Inputs and information boundary

The actor receives the high-level language instruction, current RGB, predicted
scene geometry, historical RGB evidence, gridded BEV, yaw/pitch proprioception,
successful-action odometry and action-success feedback. H+L is an explicit option.
The pose setting is unchanged: successful MoveAhead advances 0.25 m, the standing
camera is 1.576 m high, and calibrated vertical FOV is 60 degrees.

Simulator object identities, world positions, reachable-position lists, true
masks/depth, expert plans, task types, PDDL parameters and terminal goal conditions
remain behind the evaluator boundary. Action error strings redact coordinate-bearing
object IDs but can contain native category/GameObject names. A separate train-only
depth calibration process reads true depth offline; it is never an actor input.
This is a declared validation research setting, not a strict RGB-only submission.

The actor uses native Responses through the registered adapter, with an empty
host-tool catalog. Typed decisions are the only path to retrieval and actions.
GPT requests never fall back to Chat Completions. The existing official simulator
and scorer stay pinned to AI2-THOR 2.1.0, build 201909061227.

## Perception and incremental memory

The fixed vocabulary contains ALFRED's 108 global object categories plus Faucet
and five sliced forms. Human-readable phrases are deterministic; no current-scene
metadata or task target prunes the list. Since categories are provided, RAM's
automatic category proposal is omitted. GroundingDINO SwinT-OGC detects boxes,
SAM1 ViT-H chooses the highest-quality multimask candidate for each box, and
OpenCLIP ViT-H-14 / laion2b_s32b_b79k encodes the original detector crop with 20-pixel
padding and the category phrase. Full-frame CLIP features are retained as well.

Our mapper combines geometric point overlap and CLIP cosine similarity, performs
voxel reduction and DBSCAN, and periodically merges duplicate objects. Online
extensions preserve stable IDs and aliases, distinguish provisional objects,
and start new geometry epochs after affected interactions. Original RGB, predicted
depth, intrinsics, poses, detections and feature arrays are retained. Object/view
and grid/view associations support retrieval even when a requested target was
not detected. Successful actions are recorded as action-derived state evidence,
not mislabeled as visual confirmation. Interactions use the effective target mask
from the current observation, with its saved path and hash in the action log.

The navigation map uses a 5 cm grid; semantic BEV regions are 50 cm. Known free,
occupied/collision and unknown space are separate from visual observation coverage.
The agent body and a successful traversal provide free-space evidence without
claiming that surrounding regions have been visually inspected. FMM plans on
observed space with explicit body clearance. Explore selects a reachable frontier,
keeps it across a bounded action chunk, reobserves every step and uses bounded
camera scans when travel cannot expose more space. Navigation, interaction and
exploration report failures explicitly.

| Module | Responsibility |
|---|---|
| `dataset`, `settings` | Frozen fold and strict public scene configuration |
| `perception`, `perception_worker`, `vision_models`, `depth_models` | Local detector, masks, CLIP and meter-scale depth |
| `geometry`, `object_mapping`, `object_store`, `memory` | Registered observations, object fusion and indexed evidence |
| `spatial_map`, `navigation`, `exploration`, `rendering` | Coverage, traversability, FMM, frontier selection and BEV |
| `policy`, `execution`, `actor_runtime` | Typed decisions, action budgets and isolated native Responses turns |
| `simulator`, `simulator_worker` | Official scene lifecycle, mask dispatch and terminal scoring |
| `calibration_capture`, `depth_calibration`, `depth_calibration_runner` | Private train observations and independent raw metric depth comparison |
| `metrics`, `results`, `provenance` | Official aggregation, immutable attempts and code/asset provenance |

## Local model setup

Use the existing `agentic3d` environment for the runner and perception. The older
`/home/ysh/codebase/FILM/.venv/bin/python` is used only for the existing official
simulator worker. It supplies no FILM perception or navigation code. Checkpoint
and source locations are explicit in `configs/alfred/scene.toml`; inference does
not download assets. Weights stay outside Git under `/home/ysh/Models/agentic3d`.
Pinned source revisions, model revisions and hashes are in the v9 record.

GroundingDINO's native CUDA extension needs the recorded
`scripts/alfred/patches/groundingdino_torch28_scalar_type.patch` for Torch 2.8.
Build with the environment's CUDA 12.6, GCC 11, explicit compute capability 8.9,
and NVIDIA package include directories. The patch changes the dispatch scalar
type API, not model computation. Install source packages without their full
training/export dependency stacks: those stacks conflict with the shared NumPy
version. Do not upgrade Torch/NumPy/Transformers or modify the legacy environment.

The current run uses physical GPU 4, explicitly configured and resolved to its
GPU UUID before process launch (logical `cuda:0`). Models move between CPU and GPU by stage;
a shared file lock serializes inference across ALFRED workers. Inspect GPU
availability before launching. Physical GPU 1 is forbidden. The current display
`:5` renders on GPU 6; its authentication is
`/home/ysh/codebase/agentic-3d-task/tmp/alfred_setup/Xauthority`.

<a id="run-the-frozen-validation-fold"></a>

## Frozen pilot15 invocation

Activate the environment, verify adapter health and one native upstream Responses
turn, then run inside tmux. All executable source and configuration must be
committed before evaluation; dirty tracked code and untracked source are rejected.
Use a unique run ID/output directory for every invocation, including retries.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
curl -fsS http://127.0.0.1:8787/health
NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}" \
no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}" \
python -m codex_agent.cli.run_alfred \
  --data-root /home/ysh/Datasets/FILM/alfred/data/json_2.1.0 \
  --splits /home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json \
  --fold docs/benchmark/alfred/folds/pilot15.json \
  --output-dir outputs/alfred/pilot15_<date>_<attempt> \
  --simulator-python /home/ysh/codebase/FILM/.venv/bin/python \
  --alfred-root /home/ysh/Datasets/FILM/alfred \
  --scene-config configs/alfred/scene.toml --x-display 5 \
  --model gpt-5.4-2026-03-05 --reasoning-effort medium \
  --max-actions 1000 --max-failures 10 --max-decisions 60 \
  --preflight-only
```

`--preflight-only` checks paths, fold and service readiness without starting
inference. Remove it to run. A diagnostic may use `--limit 1` on the same frozen
fold; it must have a separate record. Sharding selects `manifest[index::num_shards]`.
The runner archives every completed/failed attempt into the benchmark SQLite DB.
Keep per-episode observations, predictions, actions, decisions, memory and official
terminal snapshots. Report scored/requested coverage and infrastructure errors
beside SR, GC, PLWSR, PLWGC, actions, model calls, tokens and timings. Do not count
unscored infrastructure failures as synthetic zero-goal task failures.

The adapter's existing SOCKS forward on 10800 must be healthy; its local health
endpoint alone does not prove upstream connectivity. Keep loopback proxy exclusions
local to the new run and do not restart unrelated services.
