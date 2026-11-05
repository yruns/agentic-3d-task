# v9: online OpenEQA mapping rewrite and frozen pilot15

Date: 2026-11-05. Status: implementation and dependency validation in progress;
**no v9 task scores or measured depth winner yet**. This is a new method/configuration,
not a continuation of the v8 score cohort. The user approved replacing the mapper
and reduced the requested validation from 50 to 15 tasks.

## Frozen protocol

- Branch: `feat/alfred-benchmark`; previous completed report HEAD:
  `7e93524590aab824a5835d45a08e0c8141bcf646`.
- New fold: `folds/pilot15.json`, SHA-256
  `2fa0520b130ec9748762b31b7dc1c3aeeb072edbb5bb7e9393ea4c406da1f34a`.
- Parent: unchanged `folds/pilot50.json`, SHA-256
  `1d22afc51da128395a6d65a250d9dc603c8f50f0252942ca28b335568f646d4e`.
- Selection: seed 0; within each split and task type choose the smallest
  SHA-256 of `seed:episode_id`, then the smallest remaining unseen identity.
  Preserve parent order. Seven seen + eight unseen; all seven task types in both.
  Scores were not read by selection. Membership, trajectory hashes, parent hash
  and deterministic derivation are checked when loading the fold.
- Input: H, predicted metric depth/masks, action odometry and yaw/pitch feedback.
  No scene GT categories, geometry, object IDs or reachable-position queries.
- Policy: native Responses `gpt-5.4-2026-03-05`, medium, tool-free actor catalog.
  Budgets: 1,000 attempted actions, 10 failed actions, 60 policy decisions,
  navigation chunk 12 and at most three retrieved history images.
- Simulator and official scoring unchanged. Seen/unseen reported separately.
  Compare to v8 only on the same 15 identities, and label the result a bundled
  perception/mapping/navigation change rather than an isolated component gain.

## Method and scope

The new pipeline uses our OpenEQA scene mapping method: global category phrases,
GroundingDINO, SAM1, OpenCLIP crop/text features, geometric overlap association,
weighted feature/point fusion, voxel reduction, DBSCAN and periodic duplicate
merges. RAM automatic tagging is unnecessary when the full vocabulary is given.
OpenEQA filter defaults remain confidence 0.25, mask area 25 pixels and maximum
box area 0.5 of the image. The ALFRED configuration predeclares maximum box area
1.0 because interaction requires close-up receptacle views; this is an explicit
parameter change, not a new default for the existing benchmarks.
The fixed 114-category vocabulary is the official global 108 plus Faucet and five
sliced forms. Category aliases are explicit, not inferred from current-scene GT.
The original offline producer used scene-independent image observations; its
`class_set=scene` option must never be used for ALFRED.

The [operator consistency audit](v9_assets/mapping_parity.json) records reference
source hashes, 59 independent numerical/behavior regressions and remaining
differences (including deterministic geometry, bounded cloud storage and stable
identity safeguards). It explicitly does not claim bitwise equivalence or a
completed real-cache replay.

New online behavior: stable object IDs/aliases, provisional instances, geometry
epochs after interactions, current effective masks for action dispatch, separately
tracked free/occupied/unknown space and visual coverage, FMM on a fine grid,
frontier exploration and bounded local camera scans. Original observations and
object–view/grid–view indices remain available. The prior FILM runtime perception
and RGB-histogram association are removed, without a compatibility fallback.
Using the existing Python 3.8 simulator interpreter does not import FILM's policy,
depth estimator or object detector.

No NR3D/OpenEQA/SceneFunc3D preprocessing or evaluator behavior is changed. The
NR3D processed_v1 package uses its existing GT proposal producer; it is not
relabelled as this detector pipeline. Generated local policy programs are outside
this iteration.

## Depth comparison predeclaration

Candidate sources: [DA3Metric-Large](https://github.com/ByteDance-Seed/Depth-Anything-3)
and [UniDepthV2 ViT-L/14](https://github.com/lpiccinelli-eth/UniDepth).
These are available strong metric models, not a verified claim of best ALFRED
performance. OPEx-S reports a domain-finetuned ZoeDepth variant, but its exact
released weights were not verified for reuse. Model selection uses measured
train-only evidence, not the validation outcomes.

Capture exactly 60 RGB/GT-depth frames from deterministic train trajectories in
FloorPlan1/201/301/401: three scripted stations, five pitches (-30, 0, 30, 45, 60).
Freeze source/data hashes and the capture plan before running. Record unsuccessful
movement attempts rather than replacing them with GT-guided routes. True depth
stays in the private calibration output, never in either predictor or the actor.
AI2-THOR 2.1.0 supplies millimeters; convert once by 1,000 and separately count
5 m saturation. Do not use median alignment, GT scale fitting, or a validation
frame to select/configure the depth backend.

Rank complete models (at least 99% valid coverage) by equal-frame macro AbsRel,
then near-range (0.3–2 m) AbsRel, depth-edge gradient error, and RMSE, with 1e-8 tie
tolerance. Report per-scene/per-pitch metrics and runtime. Plane and multiview
consistency are diagnostic only; unavailable ground-plane labels must remain
explicitly unavailable. DA3Metric's raw canonical output is converted with the
resized focal length exactly once; UniDepth retains its native meter scale.
The selected backend must be frozen in production before pilot15 starts.

Exact prepared commands: [train capture](v9_assets/capture_train60.sh) and
[depth comparison on physical GPU 4](v9_assets/compare_train60.sh). These are
commands to be executed after the source commit, not evidence of completed runs.

## Assets and environment

- [Model revisions and SHA-256](v9_assets/model_records.json).
- [Source revisions](v9_assets/source_records.json).
- [Core package versions](v9_assets/core_packages.json).
- Public production configuration: `configs/alfred/scene.toml`.
- Local setup and exact evaluation command: [ALFRED guide](../../codex_agent/alfred.md).
- Inference GPU: physical 4 (user coordinated other jobs onto 0–3), stage-wise
  CPU offload, GPU-specific interprocess file lock. Physical index is resolved to
  a UUID before setting visibility; model-local device remains `cuda:0`.
  Renderer: existing display :5 on physical GPU 6. Physical GPU 1 never used.
- Models are external; existing OpenCLIP weights are reused. Asset and actual
  editable-source hashes are also recorded with each run.

Dependency setup initially failed because new tmux sessions lacked the shell's
proxy environment, and GroundingDINO's older extension used the removed Torch
`Tensor.type()` dispatch API. The successful build uses the recorded two-line
scalar-type patch, CUDA 12.6/GCC 11 and architecture 8.9. Missing build dependency
`editables` was then installed for DA3. Source installs deliberately omit full
training/export dependency sets, preserving shared Torch 2.8, NumPy 1.26.4 and
Transformers 4.57.1. These setup failures are not scored episodes. Detailed setup
logs currently reside in `tmp/alfred_v9/`; the revisions and necessary source patch
are durably stored here and under `scripts/alfred/patches/`.

## Matched historical reference

After freezing pilot15, the saved v8 scores were restricted to those exact
identities, without rerunning any task. The matched reference is **3/15 successes
(SR 20%), GC 11/37 = 29.73%, PLWSR 12.11%, PLWGC 26.24%**. Seen is 1/7; unseen
is 2/8. This differs from the full pilot50 distribution. The deterministic fold
selection preceded this lookup; it must not be resampled based on the reference.
[Exact per-task scores and source hashes](v9_assets/matched_v8_pilot15.json) and
[reaggregation script](v9_assets/derive_matched_baseline.py) preserve the calculation.

## Validation and result ledger

Completed implementation checks: all five neural checkpoint loaders passed on
CPU; a generated 300×300 RGB image passed the actual DA3/DINO/SAM/CLIP worker
forward (six masks and unit CLIP features), a separate SAM box forward, and
UniDepthV2's actual forward. This validates interfaces and numerical output, not
accuracy on indoor scenes. Evidence: [checkpoint loading](v9_assets/cpu_model_loading.json)
and [synthetic forwards](v9_assets/synthetic_cpu_forward.json). The adapter suite
passed 440 tests with 91.34% branch-aware coverage. The first MkDocs build passed;
pre-existing unrelated navigation/link warnings remain.

Final repository gates passed: Ruff, Black, mypy, **1432 source tests**, MkDocs
build and `uv lock --check`. Together with the 440 adapter tests, this is 1872
passing tests. The existing simulator Python 3.8 compiled the capture worker.
The real symlinked simulator interpreter exposed a venv-selection bug during review;
a failing regression was added, `.absolute()` now preserves the venv entry point,
and all 15 capture tests pass.
[Exact verification commands, logs and hashes](v9_assets/quality_gates.json).

The actual worker also passed the same generated-image forward on physical GPU 4:
5.82 seconds per frame, 5.60 GiB peak CUDA memory, six masks, valid depth and
normalized CLIP features. Its reported GPU UUID and source digest were verified;
GPU memory returned to baseline after shutdown.
[GPU execution and actual worker environment](v9_assets/synthetic_gpu_forward.json).

Implementation commit: `ea444a0ff8bd982ad082a38d7d5df8b927fb3eb0`; executed-source
SHA-256 `d4ccbe9f5e77a4e23844f3173720e13c1d8ef1aab20ced195dae3e0ec1b4ddec`,
with a clean tracked tree. A real native Responses synthetic vision/schema probe
passed in 5.24 seconds on that commit ([receipt](v9_assets/native_responses_probe.json)).

The first capture launcher stopped during conda activation because shell nounset
rejected an unset compiler variable; Python and the simulator were not started.
The corrected launcher enables nounset after activation.
[Failed launch and fix](v9_assets/capture_launch_attempt1.json).

The corrected launcher captured all 60 train frames at clean HEAD `52b30b8`;
start/end code hashes agree. FloorPlan201 recorded three blocked movement attempts
as specified; other scenes had none. The first depth-comparison launch rejected
signed-pitch frame names containing `+` before loading a model. The validator now
accepts that safe character; the capture-to-consumer integration regression and
all 24 capture/comparison tests pass. All frozen frame hashes validate unchanged.
The complete source gates were rerun: all 1432 source tests, Ruff, Black, mypy,
MkDocs and lock validation pass
([post-fix verification](v9_assets/quality_gates_frame_id_fix.json)).
[Failed comparison launch and fix](v9_assets/depth_compare_launch_attempt1.json).

The second comparison completed on clean HEAD `0e89511`, using unchanged frozen
frames and physical GPU 4. Both models cover all valid GT pixels. DA3Metric-Large
wins the predeclared primary AbsRel criterion; its existing production setting
remains frozen. Neither model received GT or a fitted scale factor.

| Backend | AbsRel | Near AbsRel (0.3–2 m) | RMSE (m) | Edge gradient MAE (m) | 60-frame inference (s) |
|---|---:|---:|---:|---:|---:|
| DA3Metric-Large | 0.19530 | 0.19976 | 0.31422 | 0.24870 | 20.44 |
| UniDepthV2 ViT-L/14 | 0.41473 | 0.43347 | 0.51951 | 0.23543 | 18.28 |

GT saturation excludes 366,755/5,400,000 pixels (6.79%); 5,033,245 valid pixels
remain. Edge metrics exist for 51 frames. UniDepth has the lower edge error,
which does not override the predeclared AbsRel ranking. These are four-room,
60-frame train diagnostics, not a claim of ALFRED-wide depth superiority.
[Full per-frame/scene/pitch comparison](v9_assets/depth_comparison.json) and
[executed code, model hashes and command configuration](v9_assets/depth_protocol.json).

Pending: real perception/mapping checks and pilot15 official scoring. Run IDs,
code hashes, commands, artifacts and actual outcomes will be added as each stage
executes; no pending stage is presented as passed. Historical v1–v8 and the old
50 scores remain unchanged.
