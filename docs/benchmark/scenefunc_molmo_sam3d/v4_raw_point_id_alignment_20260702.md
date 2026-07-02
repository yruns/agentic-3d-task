# SceneFunc3D raw point-id alignment, 2026-07-02

This record covers the first real 20-case scored run after changing
SceneFunc3D lift output to write raw/source mesh vertex ids instead of
conceptgraph filtered mesh ordinals.

## Run Identity

- Branch: `feat/scenefunc3d-agent-tools`
- Head commit at launch: `142ca15525db77bf6feeb64ab17f8f32f7864c11`
- Dataset root: `data/SceneFun3D`
- Sample ids:
  `tmp/scenefunc3d/artifacts/scenefunc_parallel20_20260701_2346/sample_ids.json`
- Output root:
  `tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_workers20_20260702`
- Local run summary:
  `tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_workers20_20260702/raw_point_id_alignment_summary.json`

## Invocation

```bash
PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d \
  --dataset-root data/SceneFun3D \
  --sample-ids-path tmp/scenefunc3d/artifacts/scenefunc_parallel20_20260701_2346/sample_ids.json \
  --backend-config configs/scenefunc3d_backends.toml \
  --output-dir tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_workers20_20260702 \
  --score \
  --workers 20
```

Before launch, the local ModelHub adapter health check passed at
`http://127.0.0.1:8787/health`. The configured remote Molmo/SAM sidecar was
used through `configs/scenefunc3d_backends.toml`.

## Result

`evaluation_summary.json`:

| Metric | Value |
|---|---:|
| Samples | 20 |
| Completed | 15 |
| Failed | 5 |
| Scored | 15 |
| Mean IoU | 0.10025787969057885 |
| Mean precision | 0.15875931344565872 |
| Mean recall | 0.19503100133342768 |
| Mean F1 | 0.16177028571995186 |

Completed-case IoU distribution:

| Statistic | Value |
|---|---:|
| Non-zero IoU cases | 9 |
| Zero IoU cases | 6 |
| Min IoU | 0.0 |
| Median IoU | 0.040983606557377046 |
| Max IoU | 0.32637571157495254 |

Failure categories:

| Category | Count |
|---|---:|
| Stream disconnected before completion, `max_output_tokens` | 1 |
| Invalid `review_artifacts` path invented outside the run directory | 2 |
| Accepted fragment missing a matched SAM provenance chain | 2 |

## Raw Id Validation

The raw mesh for `421254` has `2,705,592` vertices. All 15 completed fused
`mask_data.npz` files were checked after the run:

```text
checked_completed_results=15
raw_id_violations=[]
```

This confirms the completed predictions are now written in the raw/source mesh
vertex-id space expected by SceneFunc3D annotations. The scorer no longer relies
on conceptgraph filtered mesh ordinals for these completed cases.

## Interpretation

The point-space bug is fixed for completed samples: predicted ids are bounded by
the raw mesh vertex count and overlap is now measurable. The run is still not a
stable benchmark-quality agent loop because 5/20 cases failed before scoring and
6/15 scored completions still have zero IoU.

The remaining failures are controller/playbook reliability issues, not raw-id
alignment issues. The next improvement should make `fuse_accepted_masks`
arguments deterministic or runner-controlled so the model cannot invent review
artifact paths or accept fragments without the validated Molmo -> SAM -> lift
tool chain.
