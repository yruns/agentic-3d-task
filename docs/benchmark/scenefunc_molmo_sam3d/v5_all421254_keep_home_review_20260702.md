# SceneFunc3D all-421254 keep-home review, 2026-07-02

This record covers the full `421254` SceneFunc3D run after raw point-id
alignment, with per-turn Codex run homes preserved for inspection.

## Run Identity

- Branch: `feat/scenefunc3d-agent-tools`
- Head commit at launch: `4b5cf51`
- Scene: `421254`
- Sample count: `23`
- Sample ids:
  `tmp/scenefunc3d/artifacts/scenefunc_421254_all23_20260702/sample_ids.json`
- Run root:
  `tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702`
- Review summary:
  `tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702/all23_review_summary.json`
- Run-home map:
  `tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702/run_home_map.json`

## Invocation

```bash
CODEX_AGENT_KEEP_RUN_HOME=1 PYTHONPATH=src \
python -m codex_agent.cli.run_scenefunc3d \
  --dataset-root data/SceneFun3D \
  --sample-ids-path tmp/scenefunc3d/artifacts/scenefunc_421254_all23_20260702/sample_ids.json \
  --backend-config configs/scenefunc3d_backends.toml \
  --output-dir tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702 \
  --score \
  --workers 20
```

The local ModelHub adapter health check passed before launch. The run used the
remote Molmo/SAM sidecar from `configs/scenefunc3d_backends.toml`.

## Aggregate Result

`evaluation_summary.json`:

| Metric | Value |
|---|---:|
| Samples | 23 |
| Completed | 12 |
| Failed | 11 |
| Scored | 12 |
| Mean IoU | 0.12577322279095213 |
| Mean precision | 0.1970906101792578 |
| Mean recall | 0.23287560602089719 |
| Mean F1 | 0.19641372995755657 |

Completed-case IoU distribution:

| Statistic | Value |
|---|---:|
| Non-zero IoU cases | 8 |
| Zero IoU cases | 4 |
| Min IoU | 0.0 |
| Median IoU | 0.07643043411566192 |
| Max IoU | 0.47580645161290325 |

Raw-id validation still passed for completed cases:

```text
raw_mesh_vertex_count=2705592
completed_results_checked=12
raw_id_violations=[]
```

The completed-case scores were recomputed directly from
`fused/mask_data.npz:point_indices` and
`data/SceneFun3D/421254/421254_annotations.json`; they matched
`evaluation_summary.json`. Completed-case low or zero IoU is therefore not a
known point-space mismatch.

## Failure Review

Failure categories:

| Category | Count | Interpretation |
|---|---:|---|
| `invalid_encrypted_content` | 10 | ModelHub/Responses encrypted reasoning state could not be decrypted or parsed. |
| `max_output_tokens` stream disconnect | 1 | Codex stream ended before completion. |

The 10 encrypted-content failures happened after normal successful SceneFunc3D
tool events, and no `failure.json` files were produced for this run. The error
is upstream Responses API validation, not a SceneFunc3D mask/tool error.

Most likely cause: encrypted reasoning state is not portable across all
upstream routing choices. The current stack has partial stickiness through
`chat_run_id`, but the adapter still has paths that do not fully prove same
AK/upstream/session affinity for every continuation:

- runtime sets `CODEX_AGENT_MODELHUB_EXTRA_HEADER` with a per-turn
  `chat_run_id`;
- adapter preserves `chat_run_id` and TOML upstream selection can hash on it;
- plain AK-pool selection is not fully tied to `chat_run_id`;
- adapter creates a fresh HTTP client per request;
- 429 retry can rebuild a continuation request against another upstream alias.

Recommended next fix: add redacted adapter diagnostics for `chat_run_id`,
selected upstream alias/key alias, status, and log id; then make continuation
requests sticky to the first successful upstream for that `chat_run_id`.

## Codex Run Homes

`CODEX_AGENT_KEEP_RUN_HOME=1` worked for this run. Completed samples have
first-class `turn.run_home` in their `result.json`, and those directories exist
under `.codex-home/runs/`.

Failed samples do not have first-class `run_home` in `evaluation_summary.json`;
their homes were identified from preserved session traces by the review pass.
The run-home mapping is recorded in:

```text
tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702/run_home_map.json
```

Observed retention properties:

| Item | Value |
|---|---:|
| Current run homes identified | 23 |
| Completed first-class `run_home` entries | 12 |
| Failed trace-inferred run homes | 11 |
| `failure.json` files | 0 |
| Pruned app-server `.tmp` / SQLite state | yes |

The current gap is durability for failed turns: future runner code should record
run-home metadata for run-level failures even when no `result.json` or
`failure.json` is written.

## IoU=0 Review

Completed cases with IoU `0.0`:

| Sample | Task | Pred / GT | Main cause |
|---|---|---:|---|
| `609f3c02` | Unplug the TV from the power supply | `318 / 208` | Occluded affordance; selected visible cord/plug-like region rather than GT unplug affordance. |
| `9773cbd7` | Open the third drawer of the cabinet left of the TV | `143 / 143` | Drawer ordinal / instance mismatch despite compact knob mask. |
| `bd171354` | Open the third drawer of the cabinet with the TV on top | `179 / 101` | Drawer ordinal / instance mismatch. |
| `f57e5506` | Turn on the red table lamp next to the bed | `46 / 64` | Action-affordance ambiguity; selected an inline cord switch while GT likely expects a lamp push/tip control. |

Very low nonzero cases:

| Sample | IoU | Main cause |
|---|---:|---|
| `4668a5f5` | `0.005758` | Broad right-window/blinds panel selected for a `hook_turn` affordance. |
| `6afc3b66` | `0.047170` | Door handle partly found, but SAM/lift included too much door surface. |

The dominant completed-case failures are semantic target selection, affordance
scale, and ordinal disambiguation. They are not empty outputs and not raw-id
alignment failures.

## Stronger Cases

- `55b56254`, IoU `0.475806`: best case. It selected bottom-drawer knob
  evidence from two nearby frames and fused compact, consistent fragments.
- `c9cde8d0`, IoU `0.317919`: high score, but visually suspicious; review
  recommends checking overlay-to-depth behavior for frame `000057`.
- `71e753ae`, IoU `0.215686`: good top-right TV-cabinet knob selection.
- `a1574eea`, IoU `0.189349`, and `af0b7790`, IoU `0.151899`: reasonable
  compact drawer-knob targeting, still sensitive to row/ordinal ambiguity.

## Improvement Opportunities

1. Add an affordance-size prior by motion label. For `hook_turn`,
   `pinch_pull`, `tip_push`, and `key_press`, broad panels and large 3D extents
   should be rejected or down-ranked unless the task explicitly targets a
   surface.
2. Score SAM candidates by compactness and point containment, not only SAM
   confidence or visual plausibility.
3. Add drawer ordinal validation: enumerate visible drawers, cabinet side, and
   row index before accepting a knob for “third”, “fifth”, “top”, or “bottom”
   descriptions.
4. Use multiview to confirm the same raw-id cluster. Reject follow-up fragments
   that add many new raw ids with weak overlap.
5. Treat “no good follow-up view” as lower confidence for occluded or ambiguous
   affordances rather than a high-confidence single-view success.
6. Record `run_home` for failed turns directly in batch failure rows or failure
   artifacts, so future reviews do not need trace inference.
