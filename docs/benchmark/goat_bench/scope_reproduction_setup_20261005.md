# SCOPE reproduction setup — 2026-10-05

Status: **in progress; no end-to-end benchmark result yet**.

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
and annotation text files are absent. They are linked into the task asset
directory without altering the source assets.

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

## Evidence and pending verification

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

Before evaluation, capture the clean project and runtime code commits and
freeze the selected task IDs. Record each terminal exit status, exact command,
environment package list, bridge checks and raw output directory. Completed
evaluation results will receive a separate version record and SQLite ingestion.

## Sources

- [Official SCOPE code](https://github.com/mrwangyou/SCOPE).
- [SCOPE paper](https://arxiv.org/html/2511.08935v1).
- [Official Habitat dataset download requirements](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).
