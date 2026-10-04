# SCOPE reproduction setup — 2026-10-05

Status: **all required assets ready; first public episode completed**.

The [v1 public episode](v1_public_episode_20261005.md) exited 0 with all five
original subtasks recorded: distance SR 80.00% and SPL 68.84%. The 36-scene /
278-subtask evaluation is running in Linux `tmux`; the full 2,669-subtask
evaluation is incomplete.

The user selected SCOPE for research point 3 and subsequently authorized
`gemini-3.8-flash` as a replacement for the original GPT model. Compute and data
preparation run on `ssh bupt`; the model API must be reached from the Mac and
bridged to Linux through an SSH reverse tunnel.

## Source and storage

- Project branch: `feat/realtime-model`.
- Project commit at start: `1332f039e412590d3ac22bfb8a476b5ef8fe5365`.
- Official SCOPE commit: `46f1e1f78077219c30e35440540e4ef22995194c`.
- Source: `/home/ysh/codecase/SCOPE-repro/SCOPE`.
- Assets, logs, scripts, manifests and outputs:
  `/home/ysh/Datasets/SCOPE` (resolves under `/data1/ysh/Datasets`).
- External reproduction environment: Conda `scope-repro`, pinned to the
  upstream README's Python 3.9 / Torch 2.4.1 / PyTorch3D 0.7.8 combination.
  The project's existing `conceptgraph` environment is unchanged.
- GPU 0 is used for verification; GPU 1 is excluded by `CLAUDE.md`.
- Secret configuration lives outside tracked files; this record contains no
  API keys, access tokens, or credential-bearing URLs.

## Episode and data scope

The original task JSON files contain 36 scenes, 360 episodes and 2,669
subtasks. `--split 1` selects the first episode of each scene: 36 episodes and
278 subtasks. `--split 1` through `--split 10` are needed for the full set.

Existing geometry and navmeshes for all 36 scenes were found in
`/home/ysh/Datasets/open-eqa/data/scene_datasets/hm3d/val`. Their semantic mesh
and annotation text files were initially absent. The geometry is linked into
the task asset directory; the missing semantic files were subsequently filled
from the verified public mirror described below.

The official public HM3D example overlaps one required scene,
`00861-GLAQ4DNUx5U`, and supplies all four required files. The original task
JSON is copied unchanged to:

```text
/home/ysh/Datasets/SCOPE/datasets/goat_bench/public_example/content/GLAQ4DNUx5U.json
```

Its first episode is episode ID `0`, with five subtasks. This is an engineering
end-to-end check on an available official scene, not the 36-scene paper subset
or a decision-grade pilot. No synthetic substitute tasks or semantic labels
are created. The user authorized public dataset mirrors. The remaining
semantic assets were obtained from the public
[zhuhu00/hm3d mirror](https://huggingface.co/datasets/zhuhu00/hm3d), pinned to
revision `0bf2970b74226e2d84b3137bed68a9805f0cc88a`. The 2,145,935,360-byte
`hm3d-val-semantic-annots-v0.2.tar` matches its published LFS SHA256. All 36
required scenes now have their four files. The example scene's mirror semantic
files match the official example hashes; the annotation config matches the
original SCOPE config byte for byte. All 1,246 distinct scene/goal-object pairs
resolve to semantic labels, with no missing IDs.

An unauthenticated Matterport download returned login HTML, which was not
treated as a downloaded dataset. Existing geometry and verified mirror assets
removed that runtime blocker.

## Model and protocol

- Paper model: `gpt-4o-2024-11-20`.
- User-authorized replacement: `gemini-3.8-flash`, with thinking requested as
  `include_thoughts=true`, `budget_tokens=2000`.
- The Mac-local ModelHub bridge listens on `127.0.0.1:8899`, forwarded to the
  same Linux loopback port. Linux SDK image requests returned actual
  `gemini-3.8-flash` completions and usage. The bridge records reasoning and
  cached prompt tokens without private prompts or credentials. Missing response
  ID, creation time and choice index are filled as transport metadata only.
- The existing local GPT-4o credential returned HTTP 401 in a preflight;
  it is not used as evidence that the paper model is available.
- Algorithm settings retain upstream prompts, temperature 0.7, `top_p=0.95`,
  max tokens 4,096, frequency/presence penalties 0, persistent memory,
  potential estimation and self-refinement.
- Original observations: 1,280 × 1,280 RGB-D, HFOV 120 degrees; seven views at
  the first step of each subtask and three at its subsequent steps.
- Primary metrics: `success_by_distance` and `spl_by_distance`, using
  geodesic distance to annotated goal viewpoints and a 1 m threshold.
- Infrastructure API failure terminates the bootstrap before upstream
  API-failure defaults can be accepted. Nonempty semantic-format fallbacks
  remain part of the pinned upstream implementation, as detailed below.

### Effective protocol and fidelity boundaries

The settings below were checked against the pinned official source and the
frozen evaluation YAML. The YAML changes eight resource/output settings; it keeps
all navigation parameters. ModelHub forwards the requested sampling fields
and adds the user-authorized model and thinking configuration. Sampling
requests match [paper Appendix A, Table 3](https://arxiv.org/html/2511.08935v1);
Gemini with thinking remains a separate model configuration.

| Aspect | Effective released-code behavior | Pinned source evidence |
| --- | --- | --- |
| Sampling and prompt images | Temperature 0.7, `top_p=0.95`, max tokens 4,096, frequency/presence penalties 0; prompt images are 360 × 360. | [API request](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/eval_utils_gpt_goatbench.py#L49-L57), [image settings](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/cfg/eval_goatbench.yaml#L41-L43). |
| Memory lifetime | Scene memory, TSDF and potential graph persist across subtasks within an episode. A new episode initializes all three again; ten episodes from one scene are separate memory sessions. | [Episode initialization and optional clearing](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L135-L234), [`clear_up_memory_every_subtask=false`](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/cfg/eval_goatbench.yaml#L33-L36). |
| Potential graph | Runtime grid resolution is 1.0 m, temporal decay 0.95 and influence radius 3.0 m. Potential estimation is enabled by default. The entrypoint overrides the class constructor's grid/radius defaults. | [Runtime defaults](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L164-L170), [estimator switch](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L362-L386). |
| Navigation and sensing budget | Each subtask permits `max(50, int(2 * sqrt(scene_size)))` macro-steps; `scene_size` is the scene's horizontal bounding-box area. Its first step captures seven views, then three per step. | [Step budget](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L149-L151), [per-subtask loop and views](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L207-L255), [area calculation](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/geom.py#L15-L39). |
| Goal selection | The original helper retains descriptions with at most 55 space-separated words, includes child categories for category goals, and selects the referenced instance for description/image goals. The frozen request contains 278 split-1 and 2,669 full-set IDs. | [Unmodified goal helper](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/goatbench_utils.py#L6-L62), [frozen selection](assets/scope_20261005/full2669_frozen_manifest.json). |
| Stop and measurement | Arrival at a selected snapshot ends normal navigation. Distance success independently checks whether the final pose is strictly less than 1 m geodesic distance from an annotated viewpoint. | [Arrival condition](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L534-L553), [distance measurement](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L589-L613). |
| Seed and repetitions | Python and NumPy use seed 3407; the entrypoint sets no Torch seed. The queued evaluation is one pass. It provides no five-repeat mean/standard-deviation or significance result. | [Seed initialization](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L44-L45), [paper Appendix B](https://arxiv.org/html/2511.08935v1). |

[Paper Equation 9](https://arxiv.org/html/2511.08935v1) describes a weighted
exploration score with a visitation penalty. The released implementation
[computes `exploration_value`](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/potential_graph.py#L317-L352),
but the policy's frontier-score
[getter returns `node.potential_score`](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/potential_graph.py#L381-L394).
This behavior is retained. Results reproduce the pinned released implementation;
they do not establish that every paper equation is executed literally.

Self-refinement applies to snapshot choices for every goal modality, despite
the YAML comment mentioning descriptions. The upstream code retains three
semantic fallbacks: unmatched potential fields start at neutral 3.0;
nonempty confirmation text outside `CONFIRM`/`REJECT` is accepted; and a choice
rejected three times can be accepted on its next occurrence. See
[potential parsing](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/potential_graph.py#L199-L209),
[confirmation parsing](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/eval_utils_gpt_goatbench.py#L445-L474)
and [refinement/cycle handling](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/eval_utils_gpt_goatbench.py#L653-L698).
The injected client terminates on infrastructure errors, model mismatch and
empty completions; it does not rewrite these semantic parsing rules.

After five unsuccessful action-format/refinement attempts, the policy can
return `None` and end the current subtask's navigation loop. The evaluator
still records its final-pose distance metrics and later saves the episode.
Consequently, complete ID coverage can include action-format failures; it
cannot certify that every decision parsed successfully. An already nearby
final pose can even pass the distance threshold after such a failure. This
differs from an infrastructure `SystemExit`, which interrupts the run before
the current episode is saved. The paths are documented in
[retry exhaustion](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/eval_utils_gpt_goatbench.py#L514-L567),
[loop termination](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L449-L454)
and [unconditional metric recording and episode save](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L589-L628).
Final run records must report observed parser/fallback events separately from
coverage and SR/SPL; no partial-log observation is a full-run guarantee.

## Evidence and evaluation status

Remote durable manifests include:

```text
manifests/goat_tasks.json
manifests/hm3d_required_scene_audit.json
manifests/reused_scene_geometry.json
manifests/public_example_overlap.json
manifests/hm3d_sem_mirror_audit.json
manifests/task_semantic_compatibility.json
```

Public model downloads are `yolov8x-world.pt`, `sam_l.pt`, OpenCLIP
`ViT-B-32 / laion2b_s34b_b79k`, and the independent OpenAI CLIP encoder used by
YOLOWorld. Model hashes and archive integrity checks passed. GPU 0 CUDA,
PyTorch3D KNN, Habitat EGL, real annotated-scene RGB-D rendering, and SCOPE
imports passed. A real 1,280-square frame produced 17 YOLO boxes, a SAM mask
with 37,132 positive pixels, and finite image/text CLIP features. These are
preflight checks, not navigation success.

The [fourth encoder integrity record](assets/scope_20261005/yolo_clip_integrity.json)
verifies the separate 353,976,522-byte OpenAI CLIP checkpoint against its public
registry SHA. The [runtime environment audit](assets/scope_20261005/environment_runtime_audit_20261005.json)
also verifies package versions and offline cache revision. The recorded
[Conda explicit export](assets/scope_20261005/env_conda_explicit.txt) must be
followed by the [pip freeze](assets/scope_20261005/env_pip_freeze.txt): Conda's
NumPy metadata says 1.26.4, while the actual pip overlay is 1.23.5. Using only
the Conda record does not restore the observed runtime.

The public episode ran from clean project commit
`d5c24a5e676d8729fd1f6d0f890d5bf5760b5827`, using the original SCOPE runtime
commit listed above. The selected five IDs were frozen before launch. Its
terminal status, raw distance-metric shards, task metadata, config, run
manifest, summary and SQL reproduction commands are retained in the
[v1 process record](v1_public_episode_20261005.md). The record reports the
Gemini replacement as its own configuration.

The broader run freezes all 2,669 IDs and the first-episode 278-ID selection
before execution. Two workers on GPUs 0 and 2 are processing split 1 in
`tmux`. Completion requires terminal worker status and exact frozen-ID
coverage; running processes and partial metrics are not a completed result.

## Current evaluation queue

Each split uses two disjoint 18-scene shards. The first split uses physical
GPUs 0 and 2.
Its frozen file order, seed 3407 and exact IDs are in
[full2669_frozen_manifest.json](assets/scope_20261005/full2669_frozen_manifest.json).
Split 1 contains 137 and 141 subtasks, respectively. Both workers started on
2026-10-05 at 03:28 Asia/Shanghai and continue with the original navigation
budget. The controller must verify all 278 IDs before advancing to splits 2–10.
An active process is not a completed result.

The ingestion boundary now accepts the original integer episode IDs and
normalizes them to strings. It reads primary distance shards ahead of derived
aggregate files, checks matching success/SPL IDs and rejects conflicting data.
The original frozen metadata is unchanged. For splits 2–10, only `exp_name`
changes between stage configs so each split/worker writes to its own directory;
prompts, sensors, scene selection and algorithm parameters are identical.
Each terminal shard is checked against its exact IDs before being published
to the central output directory. A resumed controller reuses existing workers
and validates terminal artifacts rather than starting duplicate work.

After the 278-ID gate passes, the controller collects batches of three splits:
2–4, 5–7, then 8–10. Each batch has six workers on GPUs 0, 2, 3, 4, 5 and 6,
which were verified available before queue preparation. GPU 1 is excluded.
Each process retains the original half-scene selection and independently
resets its episode memory, as in the released evaluator. The model and
navigation parameters remain identical to the first-split configuration.
The successor [parallel storage plan](assets/scope_20261005/storage_plan_parallel_v2.json)
preserves the immutable original plan and all twenty gates. Its preparation
passed under the actual Linux `scope-repro` Python 3.9 environment.
At 04:30 on 2026-10-05, the four idle GPUs started split 3 and split 4 while
the original two first-split workers continued. These episodes have independent
memory and output directories. The controller will reuse them and start split 2
when GPUs 0 and 2 are free. This advances the full set while retaining the
canonical result gate and avoiding duplicate work. The
[early worker launch record](assets/scope_20261005/early_parallel_workers_20261005.json)
captures their exact PIDs and commits. Future phases remain queued.
The full-set API ledger now contains interleaved phases; it is only a full-run
usage ledger and cannot provide per-subtask or canonical278 token attribution.
The [parallel revision checks](assets/scope_20261005/parallel_validation_20261005.json)
include 76 targeted tests and five controller integration cases on Linux with
real file locking. The dispatch case uses synthetic terminal workers and
checks six distinct GPU/config/output assignments per batch.
The [parallel controller launch](assets/scope_20261005/parallel_controller_launch_20261005.json)
records its deployed SHA, matching script hashes and retained first-split PIDs.

## A-EQA release and asset boundary

The paper also reports A-EQA, but the official repository's A-EQA instructions
are commented out and neither the evaluator nor config exists in its eight
public commits. The author's [open issue response](https://github.com/mrwangyou/SCOPE/issues/2#issuecomment-3688968547)
says that evaluation code was being prepared. This check found no official
public A-EQA release or predictions.

A [third-party fork](https://github.com/NohJanghan/SCOPE/tree/60cc58cd0af615ee0755eca5639ba61c8b70c5e3)
added A-EQA code in September 2026 and uses SCOPE's potential graph and
refinement. Its model defaults, API-failure handling and blind-LLM answer
replacement during judging differ materially from the current transport and
coverage requirements. It has not been executed here and cannot establish
reproduction of the paper's A-EQA results.

The existing OpenEQA subset matches all 184 question IDs and their metadata;
its 57 distinct HM3D scenes have valid GLB headers/file lengths and navmeshes
that load in Habitat with finite positive navigable area. This verifies assets,
not complete rendering or navigation. Detailed sources and bounds are in the
[A-EQA audit](assets/scope_20261005/aeqa_release_audit_20261005.json).

Remote control records:

```text
tmux: scope-full-20261005; scope-s1-w0-20261005; scope-s1-w1-20261005
manifests/storage_plan.json
manifests/storage_plan_parallel_v2.json
manifests/full_project_commit.txt
manifests/full_controller_commit.txt
logs/full_run.log; logs/full_run.exit
logs/full_split{split}_worker{worker}.log / .exit
outputs/REPRO_20261005_SCOPE_Gemini38Flash_full2669
```

The first workers were launched from project commit
`89dd5337df953374c26b0ebcb67b3bb8663b9ad8`. The controller revision is recorded
separately in `full_controller_commit.txt`; it changes storage and validation,
while the pinned upstream runtime and injected model transport stay the same.

The storage revision passed the project quality gate and 75 targeted regression
tests. The four controller integration cases also passed on Linux with real
`flock`; they use isolated fixtures and provide no navigation performance
evidence. The exact checks are in
[storage_validation_20261005.json](assets/scope_20261005/storage_validation_20261005.json).
The [controller launch record](assets/scope_20261005/controller_launch_20261005.json)
captures the deployed commit and matching code hashes. Only the controller was
replaced; both navigation worker PIDs were retained.

### Transport recovery and partial checkpoint

At 05:08:56 on 2026-10-05, first-split worker 1 exited 1 after an actual
`APITimeoutError`. The bridge ledger recorded one upstream 504 after about
190 seconds; the SDK's 180-second limit expired before that response arrived.
The other five workers continued. The failed exit, both logs and the incomplete
episode's visualizations were archived under `failed_attempts/`.

Bootstrap revision `08fdeba` restores the SDK's two bounded transport retries
and uses a 240-second client timeout above the bridge's 180-second upstream
limit. Final API errors still terminate instead of accepting default scores.
Prompts, model sampling, sensor settings and navigation budgets are unchanged.
Only the failed worker restarted, at 05:39, with PID 2696054; real image requests
returned HTTP 200. Its first episode had no saved metric checkpoint and is
replayed from the original initial state. Earlier log-only successes are not
merged into new results. The [recovery record](assets/scope_20261005/transport_recovery_20261005.json)
retains the original failure and exact revision. Older live workers retain their
original clients; future launches use the new bootstrap.

At 05:40, first-split worker 0 had persisted all seven subtasks of scene
`00824-Dd4bFSTQ8gi`, episode 0. An immutable observation was ingested separately
from the eventual canonical result: **7/278 IDs**, five successes, observed SR
71.4286% and SPL 45.6954%. These are incomplete progress measurements, not a
new evaluation fold or a paper comparison. The frozen request remains 278 IDs.
The [checkpoint summary](assets/canonical278_progress_20261005_0540/ingested_summary.json)
explicitly reports `complete=false` and retains matching primary shards and
original metadata. It does not overwrite the controller's immutable final gates.

At 05:50, a second saved episode from split 4 worker 0 supplied five more
subtasks. The separate [full-set progress snapshot](assets/full2669_progress_20261005_0550/ingested_summary.json)
contains **12/2,669 IDs** across two original episodes, ten successes, observed
SR 83.3333% and SPL 55.4600%. Its requested set remains the full 2,669 IDs and
`complete=false`. The metadata symlink reuses the exact immutable original
metadata already archived for the first checkpoint. This snapshot is a progress
observation and is not a comparison with the paper's full-set metrics.

## Sources

- [Official SCOPE code](https://github.com/mrwangyou/SCOPE).
- [SCOPE paper](https://arxiv.org/html/2511.08935v1).
- [Official Habitat dataset download requirements](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).
