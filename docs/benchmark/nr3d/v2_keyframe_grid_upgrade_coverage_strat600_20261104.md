# NR3D v2 — `keyframe` selector grid & schema upgrade coverage (strat600)

Paired before/after Stage-1 keyframe target-id coverage evaluation on the
canonical `strat600` fold (`600` samples across `119` ScanNet scenes),
comparing the pre-upgrade `KeyframeSelector` baseline (`legacy`, commit
`4f4b29c`) against the grid-enhanced and schema-healed pipeline (`upgraded`).

## Question

> On the exact same `strat600` fold and LLM backend (`gemini-3.8-flash`), how
> much does upgrading `QueryParser` (structured `SCENE GRID LAYOUT`, node-level
> `candidate_object_ids` & `grid_cells`, self-healing JSON/categories),
> `QueryExecutor` (instance/grid soft priors, virtual grid anchors for unknown
> anchors), and `KeyframeSelector` (`grid_grounded` fallback + tiered padding)
> improve `hit@K` and target-object grounding accuracy?

## Run configuration

| Item | Value |
|---|---|
| Branch / base commit | `feat/upgrade_keyframe_selector` / `b4d04c1` (+ parser/executor/eval script) |
| Legacy baseline reference | `4f4b29c` (pre-grid `QueryParser` + `QueryExecutor` + `KeyframeSelector`) |
| Evaluation script | `scripts/eval_keyframe_coverage_nr3d.py` |
| Fold | `/home/ysh/Datasets/nr3d/processed_v1/folds/strat600_sample_ids.json` |
| Fold size | `600` (`Easy 290` / `Hard 310` ; `View-Dep 211` / `View-Indep 389`) |
| Data root | `/home/ysh/Datasets/nr3d/processed_v1/scannet` (`119` scenes) |
| Pack | `pack_nr3d_v9_catalog_first` + `conceptgraph/indices/{visibility_index,grid_visibility_index}.pkl` |
| Call shape | `select_keyframes_v2(query, k=3, use_visual_context=False)` |
| LLM | `gemini-3.8-flash` (`configs/llm.toml`, `temperature=0.0`) |
| CLIP | disabled (`HAS_CLIP=False`) |
| Date | `2026-11-04` |

Exact command:

```bash
python scripts/eval_keyframe_coverage_nr3d.py \
  --fold /home/ysh/Datasets/nr3d/processed_v1/folds/strat600_sample_ids.json \
  --data-root /home/ysh/Datasets/nr3d/processed_v1/scannet \
  --output-dir tmp/nr3d_coverage_latest_w64_20261104 \
  --workers 64
```

## Headline comparison (`baseline` vs `current`, 600/600)

| Metric | Baseline (Pre-change, rerun) | v1 Historical (`2026-06-08`) | **Current (Post-change)** | **Delta vs Baseline** |
|---|---:|---:|---:|---:|
| **Overall `hit@1`** | 65.67% (394/600) | 64.33% | **77.00% (462/600)** | **+11.33 pp (+68)** |
| **Overall `hit@2`** | 76.00% (456/600) | 77.00% | **87.00% (522/600)** | **+11.00 pp (+66)** |
| **Overall `hit@3`** | 79.17% (475/600) | 80.17% | **89.50% (537/600)** | **+10.33 pp (+62)** |
| Easy `hit@3` (`n=290`) | 83.79% | 85.52% | **90.00%** | **+6.21 pp** |
| **Hard `hit@3` (`n=310`)** | 74.84% | 75.16% | **89.03%** | **+14.19 pp** |
| **View-Dep `hit@3` (`n=211`)** | 81.99% | 80.09% | **91.47%** | **+9.48 pp** |
| View-Indep `hit@3` (`n=389`) | 77.63% | 80.21% | **88.43%** | **+10.80 pp** |
| Mean `Recall@1` | 65.67% | 64.33% | **77.00%** | **+11.33 pp** |
| Mean `Recall@2` | 60.58% | 58.75% | **68.50%** | **+7.92 pp** |
| Mean `Recall@3` | 60.78% | 58.56% | **67.89%** | **+7.11 pp** |
| Mean 1st-hit rank (when `hit@3`) | 1.211 | 1.237 | **1.168** | **-0.043 (earlier)** |
| **Target Object Top-1 Hit** | 46.33% (278/600) | — | **55.83% (335/600)** | **+9.50 pp (+57)** |
| **Target Object Top-5 Hit** | 47.50% (285/600) | — | **59.50% (357/600)** | **+12.00 pp (+72)** |
| **Empty pred frames (`no_evidence`)** | 40 (6.67%) | 28 (4.67%) | **0 (0.00%)** | **-40 (-100%)** |

## Paired sample-by-sample attribution (`n=600`)

| Paired Outcome (`hit@3`) | Count | Share |
|---|---:|---:|
| Both `baseline` and `current` hit | 454 | 75.67% |
| **`current`-only hit (recovered)** | **83** | **13.83%** |
| `baseline`-only hit (regressed) | 21 | 3.50% |
| Both miss | 42 | 7.00% |
| **Net gain** | **+62 samples** | **+10.33 pp** |

Where the **84 recovered `hit@3` samples** come from (`legacy status -> upgraded status`):

1. **Disambiguation among multiple same-category distractors (`direct_grounded -> direct_grounded`: 43 samples)**
   - With `SCENE GRID LAYOUT` (`(r, c): #id category` vs `#id category*`) and node-level `candidate_object_ids` + `grid_cells`, `QueryExecutor` applies soft instance (`1.35x`) and grid (`1.25x` / `1.18x` / `1.08x`) priors and prioritizes high-prior reference anchors in superlative distance checks.
   - This drives **Hard `hit@3` up by +14.19 pp** (`74.84% -> 89.03%`) and **Target Object Top-1 Hit up by +10.34 pp** (`46.33% -> 56.67%`).
2. **Elimination of `no_evidence` empty returns (`40 -> 0` empty samples; `33 / 40 = 82.5%` now hit GT target frames at `k=3`)**
   - `no_evidence -> grid_grounded` (**18 recovered hits**): when the target category is absent from `scene_categories` (`["UNKNOW"]`), `target_grids` / `anchor_grids` still recall views of the target 3D region via `grid_visibility_index.pkl`. Across all 24 `grid_grounded` samples in `strat600`, **20 / 24 (83.3%)** contain a depth-visible GT target frame in top-3.
   - `no_evidence -> direct_grounded` (**14 recovered hits**) & `proxy_grounded -> direct_grounded` (**6 recovered hits**): fuzzy category alignment (case/underscore/plural/token matching) and virtual 3D grid anchors (`_synthesize_virtual_grid_anchors`) allow `DIRECT` hypotheses with `UNKNOW` anchors to execute geometrically instead of being skipped or failing completely.

## Parse status distribution

| Status | Legacy (`4f4b29c` rerun) | v1 (`2026-06-08`) | **Upgraded** |
|---|---:|---:|---:|
| `direct_grounded` | 511 | 501 | **571** |
| `proxy_grounded` | 49 | 60 | **5** |
| `context_only` | 0 | 11 | **0** |
| `grid_grounded` | — | — | **24** |
| `no_evidence` (empty) | 40 | 28 | **0** |

## Artifacts

- Durable comparison summary: `docs/benchmark/nr3d/assets/v2_keyframe_grid_upgrade_coverage_strat600_20261104_summary.json`
- Reusable evaluation script: `scripts/eval_keyframe_coverage_nr3d.py`
- Full per-sample local outputs (gitignored): `tmp/nr3d_coverage_strat600_20261104/results_{legacy,upgraded}.json`
