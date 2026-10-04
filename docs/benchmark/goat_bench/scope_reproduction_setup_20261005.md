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
- Algorithm settings retain upstream prompts, temperature 0.7, max tokens
  4,096, persistent memory, potential estimation and self-refinement.
- Original observations: 1,280 × 1,280 RGB-D, HFOV 120 degrees; seven initial
  views and three subsequent views.
- Primary metrics: `success_by_distance` and `spl_by_distance`, using
  geodesic distance to annotated goal viewpoints and a 1 m threshold.
- API failure must terminate the run or invalidate it; upstream defaults that
  silently replace failed potential estimates or confirmations do not count
  as a successful reproduction.

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

The full-set queue uses two disjoint 18-scene shards on physical GPUs 0 and 2.
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
changes between the two worker configs so they write to separate directories;
prompts, sensors, scene selection and algorithm parameters are identical.
Each terminal shard is checked against its exact IDs before being published
to the central output directory. A resumed controller reuses existing workers
and validates terminal artifacts rather than starting duplicate work.

Remote control records:

```text
tmux: scope-full-20261005; scope-s1-w0-20261005; scope-s1-w1-20261005
manifests/storage_plan.json
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

## Sources

- [Official SCOPE code](https://github.com/mrwangyou/SCOPE).
- [SCOPE paper](https://arxiv.org/html/2511.08935v1).
- [Official Habitat dataset download requirements](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).
