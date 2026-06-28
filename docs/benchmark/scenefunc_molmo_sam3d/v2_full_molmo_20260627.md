# SceneFunc Molmo SAM3D v2 full Molmo smoke, 2026-06-27

This rerun completes the TASA-style front-half smoke on SceneFuncVal-CG:

1. Molmo predicts a 2D point for a small object.
2. SAM turns that point into a 2D mask.
3. SceneFuncVal-CG depth, intrinsics, and pose lift the mask pixels to a 3D PLY.

Reference method:

- TASA paper: https://arxiv.org/html/2511.11702v1
- TASA repo outline: https://github.com/LianHe00/TASA-main
- Molmo model card: https://huggingface.co/allenai/Molmo-7B-D-0924

## Environment

- Worker: Merlin worker `974456`
- GPU: `NVIDIA-B200`
- Python env:
  `/mlx_devbox/users/yueshuhao/miniforge3/envs/conceptgraph`
- Dataset root:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG`
- Local Molmo directory:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/models/Molmo-7B-D-0924`
- SAM checkpoint:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/models/Grounded-Segment-Anything/sam_vit_h_4b8939.pth`

The local Molmo directory was completed under `/nas` with seven safetensors
shards, about 30 GB total. The worker `conceptgraph` env was updated to
`transformers 5.9.0` and `tokenizers 0.22.2` to satisfy Molmo remote-code
imports. That upgrade reports dependency conflicts with older packages such as
`ram`; this smoke only used Molmo, SAM, torch, PIL, numpy, and depth/pose IO.

## Compatibility Notes

Molmo remote code needed local compatibility patches for the current worker
stack:

- `image_preprocessing_molmo.py`: lazy-load TensorFlow only if the TensorFlow
  resize path is used. The smoke uses the default torch bilinear path.
- `modeling_molmo.py`: add `all_tied_weights_keys = {}` and accept extra
  `tie_weights` arguments expected by newer `transformers`.
- `modeling_molmo.py`: guard `cache_position` updates when generation falls
  back to legacy tuple-style KV cache.
- `assets/molmo_sam3d_smoke.py`: force legacy generation cache for this Molmo
  process because the downloaded remote code expects tuple-style
  `past_key_values`, not `DynamicCache`.

## Repro Command

The runner records the exact scene, frame, prompt, model path, and log path:

```bash
docs/benchmark/scenefunc_molmo_sam3d/assets/run_full_421254_000050.sh
```

Default SAM candidate selection is highest score:

```bash
/bin/bash docs/benchmark/scenefunc_molmo_sam3d/assets/run_full_421254_000050.sh
```

Small-object candidate selection uses the smallest non-empty SAM multimask
candidate:

```bash
SAM_SELECTION=smallest \
OUTPUT_DIR=/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmo_sam3d_full_smallest_20260627 \
LOG_PATH=/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/logs/molmo_sam3d_full_smallest_421254_000050.log \
/bin/bash docs/benchmark/scenefunc_molmo_sam3d/assets/run_full_421254_000050.sh
```

Both commands were run inside worker tmux session
`molmo_sam3d_full_20260627`.

## Full Molmo Result

Target:

- Scene/frame: `421254/000050`
- Description id: `af0b7790-028c-4eed-945b-d90386d4f16b`
- Prompt target: small dark round drawer knob handle on the lower wooden
  cabinet drawer near the right side of the image.

Molmo output:

```xml
<point x="81.0" y="61.9" alt="small dark round drawer knob handle on the lower wooden cabinet drawer near the right side of the image">small dark round drawer knob handle on the lower wooden cabinet drawer near the right side of the image</point>
```

Parsed pixel point:

| x | y | Source |
|---:|---:|---|
| 1166.4 | 1188.48 | `molmo_percent` |

## SAM Candidate Selection Comparison

| Selection | SAM score | Mask pixels | Image coverage | Lifted points | Saved PLY vertices | Visual result |
|---|---:|---:|---:|---:|---:|---|
| `score` | 0.9252 | 865,821 | 31.3159% | 863,809 | 50,000 | Fails small-object segmentation; selects most of the cabinet front. |
| `smallest` | 0.8247 | 1,119 | 0.0405% | 1,119 | 1,119 | Passes small-object segmentation; overlay isolates the knob region. |

`score` artifacts:

- Summary:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmo_sam3d_full_20260627/421254/000050/summary.json`
- Overlay:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmo_sam3d_full_20260627/421254/000050/mask_00/overlay.jpg`
- PLY:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmo_sam3d_full_20260627/421254/000050/mask_00/lifted_points.ply`
- Log:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/logs/molmo_sam3d_full_421254_000050.log`

`smallest` artifacts:

- Summary:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmo_sam3d_full_smallest_20260627/421254/000050/summary.json`
- Overlay:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmo_sam3d_full_smallest_20260627/421254/000050/mask_00/overlay.jpg`
- PLY:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmo_sam3d_full_smallest_20260627/421254/000050/mask_00/lifted_points.ply`
- Log:
  `/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/logs/molmo_sam3d_full_smallest_421254_000050.log`

## Interpretation

Molmo + SAM + 3D lifting is feasible for this SceneFuncVal-CG small-object
case, but not with naive highest-score SAM candidate selection. Molmo produced a
point close enough to the target drawer knob for SAM to include a compact knob
candidate. Selecting the smallest non-empty multimask candidate produced a
1,119-point 3D lifted object cloud and a visual overlay on the knob.

The default highest-score SAM mask is a false positive for this use case because
it segments the drawer/cabinet surface instead of the small handle. Practical
small-object use should keep SAM multimask candidates and select or filter by
area, crop, box, or geometry before 3D lifting.
