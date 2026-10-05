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

A fresh Linux read on 2026-10-05 at 11:43:35 verified 144 nonempty scene
files (36 each of geometry, navmesh, semantic mesh and semantic text), plus
two annotation configs. All 1,246 scene/goal-object pairs still resolve.
Both mirror archives and all 74 extracted files match their recorded SHA256.
The old `hm3d_required_scene_audit.json` describes the state before the public
mirror filled the missing semantic files; it is retained as historical evidence.

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

The original metadata and frozen IDs independently establish these final
denominators:

| Requested selection | Scenes | Episodes | Object | Image | Description | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First episode per scene | 36 | 36 | 99 | 88 | 91 | 278 |
| All original episodes | 36 | 360 | 991 | 822 | 856 | 2,669 |

Final acceptance requires the controller and every relevant worker to exit 0,
paired primary success/SPL shards with exactly the frozen IDs, and ingestion
with `complete=true`, `expected_ids_verified=true`, and `modality_complete=true`.
The SQLite samples must independently reproduce overall and modality SR/SPL
with the denominators above. Final raw shards, frozen IDs, metadata, config,
runtime revision and all transport recovery records must be archived before
publishing a completed local result. Full-run API usage remains aggregate;
per-subtask token counts are unavailable from these upstream artifacts.

### Final metric analysis

`scripts/scope/summarize_metrics.py` produces overall, modality and zero-based
subtask-index SR/SPL, retaining each group's denominator and a one-based
ordinal. It requires explicit original metadata indices, paired primary shard
suffixes and exact frozen-ID coverage. It records SHA256 and byte sizes for
the inputs actually consumed, checks that they did not change during analysis,
and publishes the JSON atomically. Partial folds, missing indices, conflicting
metrics and metric-namespace report paths are rejected.

After the corresponding terminal-status gate passes, run on Linux from the
asset root, using the project analysis environment:

```bash
cd /home/ysh/Datasets/SCOPE
/home/ysh/miniconda3/envs/conceptgraph/bin/python -m scripts.scope.summarize_metrics \
  --output-dir outputs/REPRO_20261005_SCOPE_Gemini38Flash_canonical278 \
  --expected-subtasks 278 \
  --report-file outputs/REPRO_20261005_SCOPE_Gemini38Flash_canonical278/fold_report.json
```

For the full fold, use the `full2669` directory and expected count 2,669.
Revision `dd33c67` passed 26 regression cases and the project quality gate.
Its actual Linux Python 3.10.19 CLI reproduced the existing seven-subtask
complete episode's SR 71.4286% and SPL 45.6954%, with seven index groups and
four input fingerprints. A real 278-ID partial snapshot was rejected without
producing a report. This was analysis-tool verification on saved data; it
started no benchmark run. See the
[verification record](assets/scope_20261005/summary_validation_20261005.json).

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

At 07:06:54, one additional frozen future worker, split 5 worker 1, started on
physical GPU 2 alongside first-split worker 1. It processes 128 requested IDs
in its existing isolated output directory. The other six workers and controller
were retained. A 20-second resource observation found 14,951 MiB free before
launch; after real rendering and model inference, both GPU-2 processes occupied
9,100 MiB each, with 5,836 MiB remaining. The new worker reached step 3 and all
original PIDs remained live. This observation supports current capacity and does
not establish a bound for every future scene's peak memory.

Models, prompts, task selection, memory resets and navigation budgets retain
the frozen configuration. The existing controller will reuse this worker when
split 5 is reached. Concurrency is seven workers; further sharing has not been
started. The [launch record](assets/scope_20261005/co_gpu_worker_launch_20261005.json)
preserves the project head, exact PID, requested IDs and resource evidence.

After 27 minutes of GPU-2 sharing with no terminal error or memory growth,
the complementary split 5 worker 0 started on GPU 0 at 07:34:34. Its frozen
shard contains 129 IDs, so both split-5 workers process 257 IDs in their separate
directories. Before launch GPU 0 had 14,839 MiB free. After real potential-score
requests and the first navigation query, GPU 0 used 17,906 MiB with 6,178 MiB
free; GPU 2 retained 5,836 MiB free. All existing PIDs remained live. Concurrency
is now eight workers with the same frozen configuration. The
[paired launch record](assets/scope_20261005/co_gpu_paired_worker_launch_20261005.json)
retains the exact preflight and runtime evidence.

At 08:21:43, split 6 workers 0 and 1 started early on GPUs 3 and 4, alongside
the existing split-3 workers. Their disjoint 118- and 133-ID sets exactly match
the frozen selection. Configs differ from the baseline only in `exp_name`, and
their actual output paths match the isolated storage plan. Both new workers
completed real rendering and potential-score requests; all eight previous
workers and the controller remained live. After startup GPUs 3 and 4 retained
7,769 and 6,860 MiB free, respectively. Ten navigation workers are now active.
See the [split-6 launch record](assets/scope_20261005/early_split6_workers_20261005.json)
for exact commits, PIDs and preflight evidence.

At 08:34:55, split 7 workers started early on GPUs 5 and 6 alongside split 4.
Their disjoint 134- and 142-ID sets match the frozen data, and their configs
retain all baseline settings except the isolated output name. Both workers
rendered real observations and received potential-score responses. At 08:38,
all twelve navigation workers and the controller remained live; GPUs 5 and 6
retained 6,674 and 6,061 MiB free. All 3,539 completed API calls since recovery
had HTTP 200 status. The [split-7 launch record](assets/scope_20261005/early_split7_workers_20261005.json)
captures source, exact PIDs and resource evidence. This starts the remaining
independent middle episodes early without changing the canonical gate.

Further dispatch must retain at most two active workers per healthy GPU. For
GPUs 0 and 2, reserve one slot for the controller's split-1-to-split-2 sequence:
an early split-8 worker may use the other slot only after the corresponding
split-5 worker has completed and passed its exact-ID check. On GPUs 3/4 and
5/6, split 9/10 can use a released slot from split 3/6 or 4/7; the controller
reuses those recorded stages. GPU 1 remains excluded. Check live processes,
terminal status and current free VRAM before each dispatch.

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

At 07:42, split 4 worker 0 had also saved all five subtasks of scene
`00808-y9hTuugGdiq`, episode 3. The next immutable
[full-set progress observation](assets/full2669_progress_20261005_0742/ingested_summary.json)
contains **17/2,669 IDs across three complete episodes**, with 13 successes,
observed SR 76.4706% and SPL 52.2125%. Its expected set remains the original
2,669 IDs and `complete=false`; the canonical278 progress remains seven IDs.
Both remote and local SQLite samples independently reproduce the metrics.
The [manifest](assets/full2669_progress_20261005_0742/run_manifest.json)
preserves the five new IDs, raw primary shard hashes, original metadata and
the distinct launch/recovery revisions. The source workers continue, and final
controller gates are untouched.

At 08:00, the next immutable
[full progress snapshot](assets/full2669_progress_20261005_0800/ingested_summary.json)
contains **38/2,669 IDs in seven complete episodes**, with 24 successes,
observed SR 63.1579% and SPL 42.0837%. Its new 21 IDs come from the next
first-episode scene, two further split-4 episodes and the first saved split-5
episode. A separate
[canonical progress snapshot](assets/canonical278_progress_20261005_0800/ingested_summary.json)
contains **12/278 IDs in two complete episodes**, with eight successes,
observed SR 66.6667% and SPL 40.7949%. Both requested sets are unchanged and
both summaries explicitly retain `complete=false`. Remote and local SQLite
reconstruct both metrics independently; raw primary shards, exact IDs and
hashes are archived. These are incomplete observations and remain separate
from completed benchmark comparisons.

The 08:55 immutable
[full progress snapshot](assets/full2669_progress_20261005_0855/ingested_summary.json)
contains **79/2,669 IDs in 13 complete episodes**, with 57 successes,
observed SR 72.1519% and SPL 47.5695%. Compared with 08:00, 41 further IDs
were saved in six complete episodes. The accompanying
[canonical progress snapshot](assets/canonical278_progress_20261005_0855/ingested_summary.json)
contains **20/278 IDs in three complete episodes**, with 15 successes,
observed SR 75.0000% and SPL 45.5455%; it adds eight IDs from the next scene's
original first episode. Both preserve their original requested ID sets and
`complete=false`. Every observed episode contains all its original subtasks;
raw paired shards match the SQLite rows exactly. The database was transferred
using SQLite backup, passed integrity checks, and retains all earlier runs.

| 08:55 observation | Modality | Recorded | Observed SR (%) | Observed SPL (%) |
| --- | --- | ---: | ---: | ---: |
| Full requested set | Description | 24 | 54.1667 | 29.4364 |
| Full requested set | Image | 24 | 79.1667 | 62.9485 |
| Full requested set | Object | 31 | 80.6452 | 49.7017 |
| Canonical requested set | Description | 5 | 80.0000 | 35.4597 |
| Canonical requested set | Image | 8 | 87.5000 | 73.8630 |
| Canonical requested set | Object | 7 | 57.1429 | 20.3868 |

The [full manifest](assets/full2669_progress_20261005_0855/run_manifest.json)
and [canonical manifest](assets/canonical278_progress_20261005_0855/run_manifest.json)
retain the exact new IDs, complete episode groups, input hashes and contributing
launch records. Initial and resumed attempts contributed to the checkpoints;
split 5's workers have their own recorded launch commits. Global API calls
remain interleaved without per-fold or per-subtask attribution. These partial
observations are separate from completed benchmark comparisons.

The subsequent [09:05 observation](assets/full2669_progress_20261005_0905/ingested_summary.json)
records **95/2,669 IDs in 15 complete episodes**, with 64 successes,
observed SR 67.3684% and SPL 45.5028%. It adds the complete episode 3 of
`00810-CrMo8WxCyVb` (six subtasks) and episode 2 of `00876-mv2HUxq3B53`
(ten subtasks). Description/image/object counts are 30/30/35, with observed
SR 43.3333/83.3333/74.2857% and SPL 23.5491/67.3118/45.6269%. The
[manifest](assets/full2669_progress_20261005_0905/run_manifest.json) retains
exact new IDs and hashes. Raw paired artifacts match the consistent remote
SQLite backup row by row, complete episode sets match the original metadata,
and all prior rows in all four database tables were preserved. Canonical
coverage remains 20/278; no duplicate canonical snapshot was created.

At 09:05:41 all 12 worker PIDs were live and upstream logs had changed in the
previous 0–13 seconds, with real HTTP 200 responses and navigation activity.
Split 1 worker 1 was on its sixth of nine subtasks in its first episode;
split 3 worker 0 was on its sixth of six. Split 3 worker 1 had saved all ten
subtasks of its first episode and entered the next scene. No active worker
had a terminal exit status or a recorded transport/OOM failure in the current
attempt. These checks establish ongoing execution, not benchmark completion.

The 09:15 [full snapshot](assets/full2669_progress_20261005_0915/ingested_summary.json)
contains **106/2,669 IDs in 17 complete episodes**, with 72 successes,
observed SR 67.9245% and SPL 46.4292%. Since 09:05 it adds
`00800-TEEsavR23oF` episode 0 (five subtasks) and `00877-4ok3usBNeis`
episode 4 (six subtasks). The corresponding
[canonical snapshot](assets/canonical278_progress_20261005_0915/ingested_summary.json)
contains **25/278 IDs in four complete episodes**, with 19 successes,
observed SR 76.0000% and SPL 50.2028%. Both remain `complete=false`.
Independent remote and local checks reproduced overall and per-modality
metrics, exact paired raw/SQLite rows and full original subtask coverage
within every saved episode. SQLite backup preserves all earlier table rows.
The [full manifest](assets/full2669_progress_20261005_0915/run_manifest.json)
and [canonical manifest](assets/canonical278_progress_20261005_0915/run_manifest.json)
retain the new IDs, launch provenance and input hashes.

A [provenance correction](assets/scope_20261005/provenance_corrections_20261005.json)
clarifies two historical canonical manifests: their `initial_launch_heads`
also included the early split-3/4 launch `6625cdf`, which contributed later
episodes. Canonical metrics came from the original split-1 launch `89dd533`
and recovery `59bc0d3`. The correction pins the historical manifest hashes;
those original artifacts and their numerical results are retained. The 09:15
canonical manifest lists the contributing split-1 launch explicitly.

The [next immutable observation](assets/full2669_progress_20261005_0950/ingested_summary.json),
captured at **09:47:10** with archive tag `0950`, contains **122/2,669 IDs in
20 complete episodes**, with 81 successes, observed SR 66.3934% and SPL
46.0037%. It adds `00808-y9hTuugGdiq` episode 4 (five subtasks),
`00824-Dd4bFSTQ8gi` episode 2 (six), and `00829-QaLdnwvtxbs` episode 3
(five). Description/image/object counts are 42/37/43, with observed SR
45.2381/83.7838/72.0930% and SPL 27.7772/67.8790/44.9834%.

The [manifest](assets/full2669_progress_20261005_0950/run_manifest.json)
retains exact new IDs, complete episode groups, input hashes and contributing
launch provenance. Independent remote and local checks reproduced raw/SQLite
rows, the original frozen-ID subset, explicit subtask indices, complete
observed episodes, and overall/per-modality SQL metrics. SQLite backup retains
all previous rows in all four tables. The original 2,669-ID request is unchanged
and `complete=false`; per-qid usage remains `NULL`. Canonical coverage remains
25/278, so no duplicate canonical observation was created. No worker had
reached a terminal status at the preceding 09:45 live check.

The [10:11:24 observation](assets/full2669_progress_20261005_1010/ingested_summary.json),
stored under tag `1010`, contains **132/2,669 IDs in 21 complete episodes**,
with 83 successes, observed SR 62.8788% and SPL 43.7466%. Its ten new IDs
are the full original episode 4 of `00890-6s7QHgap2fW`; two are successful.
Description/image/object counts are 44/39/49, with observed SR
43.1818/82.0513/65.3061% and SPL 26.5146/66.6594/40.9837%.
The [manifest](assets/full2669_progress_20261005_1010/run_manifest.json)
retains exact new IDs, original episode groups, hashes and contributing launch
provenance. Independent checks reproduced raw/SQLite rows, frozen IDs and
explicit indices, complete episode sets, and all grouped SQL metrics. The
consistent SQLite backup preserves every prior row. Canonical coverage remains
25/278, per-qid usage remains `NULL`, and full coverage remains incomplete.

The `1020` observations were captured at **10:21:42–43**. The
[full snapshot](assets/full2669_progress_20261005_1020/ingested_summary.json)
contains **137/2,669 IDs in 22 complete episodes**, with 88 successes,
observed SR 64.2336% and SPL 44.5790%. The
[canonical snapshot](assets/canonical278_progress_20261005_1020/ingested_summary.json)
contains **30/278 IDs in five complete episodes**, with 24 successes,
observed SR 80.0000% and SPL 52.9278%. Both add all five original subtasks
of `00839-zt1RVoi7PcG` episode 0, which are all successful.

Description/image/object counts for the full snapshot are 45/42/50, with
observed SR 44.4444/83.3333/66.0000% and SPL 27.8056/67.5907/40.3452%.
For canonical they are 9/11/10, with observed SR 88.8889/90.9091/60.0000%
and SPL 57.9325/75.4541/23.6446%. The
[full manifest](assets/full2669_progress_20261005_1020/run_manifest.json)
and [canonical manifest](assets/canonical278_progress_20261005_1020/run_manifest.json)
retain exact new IDs, input hashes, complete episode groups and contributing
launch records. Independent checks reproduced raw/SQLite rows, original IDs
and explicit indices, complete saved episodes, and grouped SQL metrics.
The consistent database backup preserves all earlier table rows. Both keep
their original expected sets and `complete=false`; per-qid usage remains
`NULL`. These observed metrics do not represent completed fold results.

The `1100` observations were captured at **11:01:20**. The
[full snapshot](assets/full2669_progress_20261005_1100/ingested_summary.json)
contains **164/2,669 IDs in 26 complete episodes**, with 106 successes,
observed SR 64.6341% and SPL 45.0962%. Compared with `1020`, it adds:

| Scene | Original episode | Subtasks | Successes | Worker |
| --- | ---: | ---: | ---: | --- |
| `00808-y9hTuugGdiq` | 2 | 8 | 4 | split 3 / worker 0 |
| `00824-Dd4bFSTQ8gi` | 6 | 5 | 4 | split 7 / worker 0 |
| `00876-mv2HUxq3B53` | 0 | 9 | 6 | split 1 / worker 1 |
| `00876-mv2HUxq3B53` | 5 | 5 | 4 | split 6 / worker 1 |

The [canonical snapshot](assets/canonical278_progress_20261005_1100/ingested_summary.json)
was derived solely from the split-1 pairs in that immutable full snapshot.
It contains **39/278 IDs in six complete episodes**, with 30 successes,
observed SR 76.9231% and SPL 52.8084%. Its nine new IDs are the complete
original first episode saved by worker 1. The canonical values exactly match
the corresponding frozen subset of the full observation.

Full description/image/object counts are 55/53/56, with observed SR
45.4545/83.0189/66.0714% and SPL 29.1220/67.3947/39.6812%.
Canonical counts are 11/15/13, with observed SR 72.7273/86.6667/69.2308%
and SPL 47.3993/75.3330/31.3952%. The
[full manifest](assets/full2669_progress_20261005_1100/run_manifest.json)
also adds the split-6 and split-7 launch records and their actual head commits,
`22927e8` and `154d11e`; both retain the same upstream runtime and recovery
bootstrap. The [canonical manifest](assets/canonical278_progress_20261005_1100/run_manifest.json)
retains only its contributing split-1 provenance. Independent checks verified
raw/SQLite rows, metadata bytes, frozen IDs and explicit indices, complete
observed episodes, and all grouped SQL metrics. The consistent database backup
preserves all prior rows; neither requested set is complete and per-qid usage
remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1120/ingested_summary.json),
captured at **11:15:08** under tag `1120`, contains **177/2,669 IDs in 28
complete episodes**, with 117 successes, observed SR 66.1017% and SPL
46.1734%. It adds two complete original episodes of `00810-CrMo8WxCyVb`:
episode 2 has six subtasks and four successes; episode 5 has seven subtasks
and seven successes. Description/image/object counts are 58/55/64, with
observed SR 48.2759/83.6364/67.1875% and SPL 31.3712/68.4838/40.4148%.
The [manifest](assets/full2669_progress_20261005_1120/run_manifest.json)
retains exact new IDs, input hashes, complete episode groups and all
contributing launch records. Independent checks reproduced paired raw/SQLite
rows, original metadata bytes, frozen IDs and explicit indices, complete
observed episodes, and all grouped SQL metrics. The consistent backup retains
all earlier table rows. Canonical coverage remains 39/278, full coverage is
incomplete, and per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1145/ingested_summary.json),
captured at **11:43:13** under tag `1145`, contains **185/2,669 IDs in 29
complete episodes**, with 124 successes, observed SR 67.0270% and SPL
46.5721%. It adds all eight original subtasks of `00877-4ok3usBNeis` episode
3, with seven successes. Description/image/object counts are 59/56/70,
with observed SR 47.4576/83.9286/70.0000% and SPL
30.8395/68.2509/42.4893%. The
[manifest](assets/full2669_progress_20261005_1145/run_manifest.json) retains
paired input hashes, exact new IDs, complete episode groups and contributing
launch records. Raw metrics, frozen IDs, explicit indices and grouped SQL
metrics passed independent checks; the consistent database backup preserves
all earlier table rows. Canonical coverage remains 39/278. Both requested
selections remain incomplete, and absent per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1150/ingested_summary.json),
captured at **11:48:28** under tag `1150`, contains **196/2,669 IDs in 31
complete episodes**, with 128 successes, observed SR 65.3061% and SPL
45.3675%. It adds `00823-7MXmsvcQjpJ` episode 4 (five subtasks, zero
successes) and `00824-Dd4bFSTQ8gi` episode 5 (six subtasks, four successes).
Description/image/object counts are 62/62/72, with observed SR
45.1613/80.6452/69.4444% and SPL 29.3472/66.0836/41.3238%. The
[manifest](assets/full2669_progress_20261005_1150/run_manifest.json) retains
the exact added IDs, paired input hashes, original episode groups and launch
provenance. Independent checks matched all paired rows to SQLite and verified
original metadata, frozen IDs, explicit indices, complete episodes and grouped
metrics. All earlier database rows are preserved. Canonical coverage remains
39/278; both requested selections are incomplete and per-qid usage is `NULL`.

At 11:49:40, a separate read-only process/log check found all 12 navigation
workers alive and advancing relative to the 10:52 observation. Each worker
had 78–196 additional model responses, its latest response was HTTP 200,
and its FileHandler log had updated within 1–43 seconds. No new terminal
exit file, OOM, traceback or unrecovered API error was observed. These are
execution-health observations, not additional saved benchmark results.

The [next full observation](assets/full2669_progress_20261005_1155/ingested_summary.json),
captured at **11:55:35** under tag `1155`, contains **206/2,669 IDs in 32
complete episodes**, with 136 successes, observed SR 66.0194% and SPL
46.4499%. It adds all ten original subtasks of `00829-QaLdnwvtxbs` episode
2, with eight successes. Description/image/object counts are 68/65/73,
with observed SR 47.0588/81.5385/69.8630% and SPL
31.4379/67.0091/42.1276%. The
[manifest](assets/full2669_progress_20261005_1155/run_manifest.json) retains
exact new IDs, paired input hashes, original episode groups and all
contributing launch records. Paired raw rows match SQLite; frozen IDs,
original metadata and explicit indices, complete episodes and grouped metrics
were verified. The consistent database backup preserves all earlier rows.
Canonical coverage remains 39/278; both requested selections remain
incomplete, and absent per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1200/ingested_summary.json),
captured at **11:58:22** under archive tag `1200`, contains **213/2,669 IDs
in 33 complete episodes**, with 142 successes, observed SR 66.6667% and SPL
47.4006%. It adds all seven original subtasks of `00829-QaLdnwvtxbs`
episode 5, with six successes. Description/image/object counts are 69/71/73,
with observed SR 46.3768/83.0986/69.8630% and SPL
30.9822/68.7780/42.1276%. The
[manifest](assets/full2669_progress_20261005_1200/run_manifest.json) retains
the precise capture time, exact new IDs, paired input hashes, original episode
groups and contributing launch records. Independent checks reproduced paired
raw/SQLite rows, frozen IDs, original metadata and explicit indices, complete
episodes and grouped metrics. All earlier database rows are preserved.
Canonical coverage remains 39/278; both requested selections are incomplete,
and per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1205/ingested_summary.json),
captured at **12:00:40** under archive tag `1205`, contains **220/2,669 IDs
in 34 complete episodes**, with 148 successes, observed SR 67.2727% and SPL
47.6189%. It adds all seven original subtasks of `00815-h1zeeAwLh9Z`
episode 4, with six successes. Description/image/object counts are 70/74/76,
with observed SR 45.7143/83.7838/71.0526% and SPL
30.5396/68.5161/43.0026%. The
[manifest](assets/full2669_progress_20261005_1205/run_manifest.json) retains
exact capture time, new IDs, paired input hashes, original episode groups and
contributing launch records. Independent checks reproduced paired raw/SQLite
rows, original metadata, frozen IDs and explicit indices, complete episodes
and grouped metrics. The consistent database backup preserves all earlier
rows. Canonical coverage remains 39/278; both requested selections are
incomplete, and absent per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1210/ingested_summary.json),
captured at **12:05:35** under archive tag `1210`, contains **226/2,669 IDs
in 35 complete episodes**, with 150 successes, observed SR 66.3717% and SPL
47.2397%. It adds all six original subtasks of `00890-6s7QHgap2fW`
episode 3, with two successes. Description/image/object counts are 71/78/77,
with observed SR 45.0704/80.7692/71.4286% and SPL
30.1095/66.2845/43.7428%. The
[manifest](assets/full2669_progress_20261005_1210/run_manifest.json) retains
exact capture time, new IDs, paired input hashes, original episode groups and
contributing launch records. Independent checks reproduced paired raw/SQLite
rows, original metadata, frozen IDs and explicit indices, complete episodes
and grouped metrics. All earlier database rows are preserved. Canonical
coverage remains 39/278; both requested selections are incomplete, and
absent per-qid usage remains `NULL`.

The `1215` [full snapshot](assets/full2669_progress_20261005_1215/ingested_summary.json)
and [canonical snapshot](assets/canonical278_progress_20261005_1215/ingested_summary.json),
captured at **12:09:23**, contain **240/2,669 IDs in 37 complete episodes**
and **48/278 IDs in seven complete episodes**. The full observation has 160
successes, observed SR 66.6667% and SPL 47.7761%; canonical has 38 successes,
observed SR 79.1667% and SPL 57.2741%. Both add `00810-CrMo8WxCyVb`
episode 0 (nine subtasks, eight successes); full also adds `00803-k1cupFYWXJ6`
episode 4 (five subtasks, two successes). Full description/image/object
counts are 74/85/81, with observed SR 44.5946/80.0000/72.8395% and SPL
30.2402/66.4806/44.1684%. Canonical counts are 12/21/15, with observed SR
75.0000/85.7143/73.3333% and SPL 51.7827/76.6978/34.4740%.

Canonical raw metrics and metadata are copied only from the immutable full
snapshot's split1 artifacts and retain its capture time. Its
[manifest](assets/canonical278_progress_20261005_1215/run_manifest.json)
records only the original split1 launch and recovery provenance; later-stage
launches belong in the
[full manifest](assets/full2669_progress_20261005_1215/run_manifest.json).
Paired raw rows, frozen IDs, original metadata and explicit indices, complete
episode groups and grouped database metrics were verified. The second,
uniquely named database backup includes both observations and preserves all
earlier rows. The original full-only backup is retained. Both requested
selections remain incomplete, and absent per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1220/ingested_summary.json),
captured at **12:19:17** under archive tag `1220`, contains **249/2,669 IDs
in 38 complete episodes**, with 169 successes, observed SR 67.8715% and SPL
47.5456%. It adds all nine original subtasks of `00877-4ok3usBNeis`
episode 2, all successful. Description/image/object counts are 76/88/85,
with observed SR 46.0526/80.6818/74.1176% and SPL
30.3838/65.4632/44.3402%. The
[manifest](assets/full2669_progress_20261005_1220/run_manifest.json) retains
exact capture time, new IDs, paired input hashes, original episode groups and
contributing launch records. Independent checks reproduced paired raw/SQLite
rows, original metadata, frozen IDs and explicit indices, complete episodes
and grouped metrics. All earlier database rows are preserved. Canonical
coverage remains 48/278; both requested selections are incomplete, and
absent per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1235/ingested_summary.json),
captured at **12:31:22** under archive tag `1235`, contains **256/2,669 IDs
in 39 complete episodes**, with 173 successes, observed SR 67.5781% and SPL
47.2156%. It adds all seven original subtasks of `00827-BAbdmeyTvMZ`
episode 4, with four successes. Description/image/object counts are 80/90/86,
with observed SR 46.2500/80.0000/74.4186% and SPL
30.3602/65.1196/44.1584%. The
[manifest](assets/full2669_progress_20261005_1235/run_manifest.json) retains
exact capture time, new IDs, paired input hashes, original episode groups and
contributing launch records. Independent checks reproduced paired raw/SQLite
rows, original metadata, frozen IDs and explicit indices, complete episodes
and grouped metrics. All earlier database rows are preserved. Canonical
coverage remains 48/278; both requested selections are incomplete, and
absent per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1240/ingested_summary.json),
captured at **12:33:39** under archive tag `1240`, contains **266/2,669 IDs
in 40 complete episodes**, with 177 successes, observed SR 66.5414% and SPL
46.3435%. It adds all ten original subtasks of `00853-5cdEh9F2hJL`
episode 3, with four successes. Description/image/object counts are 83/94/89,
with observed SR 45.7831/78.7234/73.0337% and SPL
30.4275/63.8288/42.7190%. The
[manifest](assets/full2669_progress_20261005_1240/run_manifest.json) retains
exact capture time, new IDs, paired input hashes, original episode groups and
contributing launch records. Independent checks reproduced paired raw/SQLite
rows, original metadata, frozen IDs and explicit indices, complete episodes
and grouped metrics. All earlier database rows are preserved. Canonical
coverage remains 48/278; both requested selections are incomplete, and
absent per-qid usage remains `NULL`.

The [next full observation](assets/full2669_progress_20261005_1245/ingested_summary.json),
captured at **12:35:54** under archive tag `1245`, contains **282/2,669 IDs
in 42 complete episodes**, with 188 successes, observed SR 66.6667% and SPL
46.3513%. It adds `00800-TEEsavR23oF` episode 4 (seven subtasks, five
successes) and `00876-mv2HUxq3B53` episode 6 (nine subtasks, six successes).
Description/image/object counts are 92/96/94, with observed SR
48.9130/78.1250/72.3404% and SPL 32.9279/63.4358/42.0412%. The
[manifest](assets/full2669_progress_20261005_1245/run_manifest.json) retains
exact capture time, new IDs, paired input hashes, original episode groups and
contributing launch records. Independent checks reproduced paired raw/SQLite
rows, original metadata, frozen IDs and explicit indices, complete episodes
and grouped metrics. All earlier database rows are preserved. Canonical
coverage remains 48/278; both requested selections are incomplete, and
absent per-qid usage remains `NULL`.

The [09:28 global API ledger audit](assets/scope_20261005/global_api_usage_20261005_0928.json)
pins the complete 2,186,908-byte, 7,784-record ledger prefix by SHA256.
After excluding the earlier preflight/public-episode rows, the full run has
7,620 recorded bridge requests: 7,619 HTTP 200 and one earlier HTTP 504.
The recovery interval beginning at ledger row 2,388 contains 5,396 requests,
all HTTP 200. Every recorded full-run request names `gemini-3.8-flash` and
thinking budget 2,000. Recovery request latency has median 7.149 seconds and
95th percentile 32.095 seconds; these are bridge-request timings.

The full-run known token totals are 130,596,004 prompt tokens, 399,546
completion tokens, and 133,536,530 provider total tokens. One request has
missing usage; reasoning-token usage is absent from 1,528 records. Each
field retains its own known/missing denominator in the audit. These totals
include ongoing episodes and earlier attempts, and cannot be assigned to
the completed subtasks or either requested fold. Missing values remain
unknown; no per-subtask usage was inserted into SQLite. The archived prefix
hash was independently rechecked against the append-only ledger. The audit
contains only model, transport and numeric usage metadata.

### SSH tunnel recovery

At 06:01, the Mac-to-Linux SSH reverse tunnel disconnected after an operation
timeout. The Mac model bridge remained healthy, but all six workers raised
`APIConnectionError` and exited 1; the controller also exited 1. Their terminal
statuses, stdout, upstream logs, checkpoint copies and incomplete episode
visualizations were preserved under
`failed_attempts/network_disconnect_20261005_0601/`.

Revision `59bc0d3` adds `scripts/scope/run_tunnel.sh`, which reconnects after
SSH exits with delays of 2, 4, 8, 16 and at most 30 seconds. It runs in the
Mac `scope-api-tunnel-20261005` tmux session, uses the private `bupt-scope` SSH
alias, and stops its own child on TERM/INT. A real owned-SSH termination before
worker restart recovered automatically in two seconds; Linux bridge health
returned HTTP 200. Five fake integration tests, 53 keyframe tests, lint,
formatting, type checking and shell syntax checks passed.

At 06:14, all six workers and the controller restarted from a clean project
checkout at `59bc0d30e40f326544fb2524956754bd094e08fb`. Each worker now uses the
two-retry/240-second bootstrap. The original source skips the two completed
episodes containing **12 saved subtasks**; incomplete episodes replay from
their original initial poses. Log-only successes from failed attempts are not
merged. Scene assets, model, prompts, sampling and navigation budgets are
unchanged. The full evaluation remains incomplete. Exact PIDs, code hashes,
archive paths and verification results are in the
[network recovery record](assets/scope_20261005/network_recovery_20261005.json).
The [checkpoint audit](assets/scope_20261005/network_checkpoint_audit_20261005.json)
verifies each shard's primary metric ID sets, frozen requested IDs and exact
archived incomplete episode folders.

A separate [semantic fallback log audit](assets/scope_20261005/semantic_fallback_audit_20261005.json)
captures the resumed attempt through 06:37:53. It uses only six upstream logs,
with captured prefix lengths and hashes, excluding duplicated stdout and older
attempts. It found 12 navigation-output validation warnings, including nine
parse warnings, and six decisions that recovered through the original retries.
Two retry sequences remained pending at the snapshot boundary. No marked retry
exhaustion or forced/default confirmation was observed in that snapshot.
The same-candidate rejection counter accumulates within one exploration step;
it does not require consecutive rejections. Overall potential-value fallback
counts remain unobservable because some default paths have no log marker.

The updated [09:30:54 audit](assets/scope_20261005/semantic_fallback_audit_20261005_0930.json)
captures fixed prefixes from all 12 current-attempt FileHandler logs. Every
prefix has one run header, and an independent reread of the live files matched
all twelve captured SHA256 values. It excludes duplicated stdout and old
failed attempts. Counts below are observed markers or decisions in these
prefixes, not failed task or episode counts.

| Observed marker/decision | Count |
| --- | ---: |
| Navigation response validation warnings | 101 |
| Format warnings, included in validation warnings | 80 |
| Valid-selection markers after validation retry | 53 |
| Valid-selection markers after format retry | 41 |
| Format retry exhaustion returning `None` | 3 |
| Generic retry exhaustion returning `None` | 13 |
| Validation warnings reporting zero remaining retries | 4 |
| Default `CONFIRM` from unclear/API-none self-refinement | 0 |
| Same-candidate acceptance after three rejections | 6 |

Exhaustion markers and zero-remaining warnings can overlap, so they must not
be added as distinct failed tasks. The same-candidate counter accumulates
within an exploration step. A valid-selection marker establishes syntactic
recovery, not correct grounding; the six acceptance events belong to the
unchanged released policy. The fixed prefixes contain no pending or unclosed
validation-warning query, which does not imply that navigation has finished.
Silent potential-score defaults remain `null`/unobservable. The 1,241 logged
potential bodies all contain the numeric pattern checked by this audit, but
this observable subset does not measure every neutral-score path. The report
retains sanitized line, elapsed-time, task and step locations, and does not
change the running algorithm.

## Sources

- [Official SCOPE code](https://github.com/mrwangyou/SCOPE).
- [SCOPE paper](https://arxiv.org/html/2511.08935v1).
- [Official Habitat dataset download requirements](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).
