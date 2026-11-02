# v6 — Offline anchor-centric fusion sweep (2026-07-02)

Date: 2026-07-02
Status: offline ablation (training-free), no full-pipeline run

## What this is

An **offline** ablation of the new training-free, anchor-centric layered fusion,
re-scored on the **already-saved v5 fragments** — no Codex/ModelHub adapter, no
Molmo/SAM sidecar, no model calls. It replaces the old "union of all
back-projected points" with: multi-view agreement gate → anchor radius gate →
single-linkage clustering (keep the anchor's component) → `motion_type` size
prior.

New code exercised (all committed on this branch):

- `src/codex_agent/scenefunc3d/backends/motion_priors.py`
- `src/codex_agent/scenefunc3d/backends/anchor.py`
- `src/codex_agent/scenefunc3d/backends/fusion.py`
- `src/codex_agent/scenefunc3d/evaluation/offline_fusion.py`

Design/plan: `docs/superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md`,
`docs/superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-fusion.md`.

## Run provenance

- Branch: `feat/scenefunc3d-agent-tools`
- Head commit at run time: `0266b7d` (no worktree drift; run-time code == head)
- Source run root (v5 fragments): `tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702`
- Raw artifact: `tmp/scenefunc3d/artifacts/offline_fusion_sweep_20260702.json`
- Fold: the **13** sample dirs that contain `fragments/` under the v5 run root
  (the 12 completed cases + `471254::471d04be…` which failed late on
  `max_output_tokens` but left lifted fragments). Denominator = 13.

## Command

```bash
source .venv/bin/activate
PYTHONPATH=src python -m codex_agent.scenefunc3d.evaluation.offline_fusion \
  --run-root tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702 \
  --data-root data/SceneFun3D \
  --output tmp/scenefunc3d/artifacts/offline_fusion_sweep_20260702.json
```

## Sweep results (n=13)

| agreement_tau | cluster_eps_m | Mean IoU | Mean precision | AP25 | AP50 |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.02 | 0.18293 | 0.29987 | 0.30769 | 0.15385 |
| 0.3 | 0.04 | 0.18255 | 0.29781 | 0.30769 | 0.15385 |
| **0.5** | **0.02** | **0.18293** | **0.29987** | **0.30769** | **0.15385** |
| 0.5 | 0.04 | 0.18255 | 0.29781 | 0.30769 | 0.15385 |
| 0.7 | 0.02 | 0.13106 | 0.25318 | 0.23077 | 0.07692 |
| 0.7 | 0.04 | 0.13068 | 0.25113 | 0.23077 | 0.07692 |

Best: `agreement_tau=0.5, cluster_link_eps_m=0.02` (identical to `tau=0.3` — see
caveats). `min_cluster_points=10`, `radius_scale=1.0` fixed.

## Cross-version comparison

Baseline is v5 union scoring
(`v5_all421254_keep_home_review_20260702.md`), computed over the 12 completed
scored cases.

| Metric | v5 union (n=12) | v6 offline fusion (n=13) | Δ |
|---|---:|---:|---:|
| Mean IoU | 0.1258 | 0.1829 | +0.0571 |
| Mean precision | 0.1971 | 0.2999 | +0.1028 |
| AP25 | 0.1667 | 0.3077 | +0.1410 |
| AP50 | 0.0000 | 0.1538 | +0.1538 |

The precision jump (~+52% relative) is the direct over-selection fix, and AP50
crosses from 0 to 0.154 (2 cases now reach IoU ≥ 0.5). Deltas are **directional**
— see caveat on the 12-vs-13 denominator.

## Interpretation

- The win is precision/compactness, exactly as designed: the anchor radius gate +
  single-linkage clustering discard the far/broad points the union kept. Example
  `421254::4668a5f5…` fell from ~1990 union points to 160 fused points.
- `tau=0.3` and `tau=0.5` are identical because most saved cases have a single
  fragment, so per-vertex agreement is trivially 1.0; the agreement gate only
  bites at `tau=0.7`, which over-prunes (Mean IoU 0.183 → 0.131). This is a
  property of the offline proxy, not of the online pipeline.
- `cluster_link_eps_m` 0.02 vs 0.04 barely differs on this fold.

## Caveats

- **Offline proxy anchor.** The anchor is built from the *largest saved
  fragment*, not an agent-verified seed. Cases whose original seed was on the
  wrong object (e.g. `4668a5f5`, a near-total miss in v5) stay at IoU 0 — offline
  fusion cannot fix a wrong seed. The Phase-2 online pipeline's agent
  seed-verification is expected to recover several of these; this offline number
  is therefore a **lower bound** on the design's potential.
- **Agreement dimension is weak here.** With 1–2 fragments per case, this run
  isolates the *geometric refinement* (anchor gate + clustering + size prior),
  not multi-view voting. True multi-view agreement needs the semi-online run
  (Phase 2: fixed seed → visibility projection → per-frame Molmo/SAM → fuse),
  which needs the sidecar but still not the ModelHub adapter.
- **Denominator mismatch.** n=13 (dirs with fragments) vs the v5 baseline's n=12
  completed. Treat the deltas as directional, not exact.
- **Small-fold overfitting.** Params are tuned on 13 cases from one visit
  (`421254`); confirm on a larger fold and semi-online before treating any
  `(tau, eps)` as decision-grade.
- **AP definition.** `AP25`/`AP50` here remain the internal IoU-threshold success
  rate (per the archive's metric contract), not the official ranked-instance AP.
  The fusion output already carries per-instance confidence for the later
  official-AP seam.
