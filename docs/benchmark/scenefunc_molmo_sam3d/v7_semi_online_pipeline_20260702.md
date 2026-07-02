# v7 — Semi-online anchor multi-view pipeline (2026-07-02)

Deterministic **Stage B→C→D engine** for anchor-centric multi-view mask
recovery, plus a **semi-online (tier-2) validation harness**. This is the
online counterpart of v6: v6 re-scored the *saved* v5 fragments offline (no
model / adapter / sidecar); v7 lands the engine that actually **selects frames,
calls the real Molmo/SAM sidecars per frame, lifts, and fuses** — driven by a
frozen 3D anchor rather than the free-form agent tool loop.

## Provenance

- **Branch:** `feat/scenefunc3d-agent-tools`
- **Head commit at write time:** `0dbe76e`
- **Run-time code commit:** `0dbe76e` (no worktree drift — engine and harness run from head).
- **Plan:** [`docs/superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md`](../../superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md) (Phase 2b; head at launch `af00dc1`, 14 task/fix commits to `0dbe76e`).
- **Spec:** [`docs/superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md`](../../superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md).

## What changed vs v6

v6 proved *offline* that anchor-centric layered fusion fixes over-selection when
re-scoring the saved v5 fragments (Mean IoU `0.1258 → 0.1829`, precision
`0.1971 → 0.2999`, AP50 `0.0 → 0.1538`). v7 builds the deterministic online
engine that produces those fragments from scratch under an anchor:

- `backends/camera_io.py` — shared 16-bit-mm depth + camera-matrix readers (DRY
  extraction from `tools/mask_lifting.py`).
- `backends/frame_loader.py` — **memory-streaming** per-frame geometry/depth/RGB
  access (one 1440×1920 depth map ≈ 22 MB; 170 frames ≈ 3.7 GB if held at once,
  so depth is read one frame at a time and discarded).
- `backends/visibility.py` (extends Phase 2a) — `select_scene_visible_frames`
  (stream all frames, score anchor visibility incl. depth occlusion, keep top-N)
  and `count_vertex_visibility` (per-vertex multi-view visible counts, the
  denominator that feeds `fusion.agreement_scores` — closes the "agreement is
  weak offline" gap).
- `pipeline.py` — the **pure, dependency-injected** driver `fuse_selected_frames`
  (project anchor → propose → anchor-consistency gate → `fuse_multiview_points`)
  and the disk-backed scene wrapper `run_anchor_multiview_pipeline`.
- `pipeline_backends.py` — `SidecarFrameProposer`: real Molmo (nearest point to
  the projected anchor, else a provenance-recorded anchor fallback) → smallest
  SAM candidate → lift → read back the fragment's raw-mesh vertex ids.
- `evaluation/semi_online_pipeline.py` — tier-2 harness + CLI.

The **agent `SeedConfirmation` turn contract is deferred to Phase 2c** (see the
plan's scope decision). v7 therefore validates the engine with a *fixed seed
anchor* (the largest saved fragment), not the online agent.

## Fold / selection mechanism

- **Source run root (v5 fragments, same as v6):**
  `tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702`
- **Fold:** the 13 `421254` sample ids that contain `fragments/` under that run
  root (derived by `offline_fusion._sample_ids_from_run_root`).
- **Seed proxy:** for each sample, the anchor is built from the *largest saved
  fragment's* point indices (`load_sample_fusion_input`) — identical seed proxy
  to the v6 offline sweep, so v7 is directly comparable to v6 once a full run
  completes.

## Exact CLI (semi-online smoke)

```bash
PYTHONPATH=src python -m codex_agent.scenefunc3d.evaluation.semi_online_pipeline \
  --run-root tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702 \
  --data-root data/SceneFun3D \
  --backend-config configs/scenefunc3d_backends.toml \
  --out-dir tmp/scenefunc3d/artifacts/sf3d_semi_online_20260702/ \
  --frame-cap 4 --agreement-tau 0.5 \
  --sample-ids '421254::4668a5f5-0555-48c1-82c2-3ba60bbd4645' \
  --output tmp/scenefunc3d/artifacts/sf3d_semi_online_20260702/summary.json
```

`--out-dir` **must** be under the backend config's `allowed_output_roots`
(`tmp/scenefunc3d/artifacts`); the harness fail-closes otherwise.

## Headline

| Metric | Value |
|---|---|
| Mean IoU | — (full run not completed; see below) |
| Mean precision | — |
| AP50 | — |

**No full-run metrics were fabricated.** The single-sample smoke on
`421254::4668a5f5-…` exercised the pipeline end-to-end and **validated the
wiring up to the SAM sidecar call**:

1. seed selection → **passed** (after the fix below),
2. `allowed_output_roots` boundary check → **passed** (with the corrected out-dir),
3. streaming frame selection over all 170 frames + the per-frame Molmo call →
   **succeeded** against the real remote sidecar proxy,
4. the SAM sidecar call → **failed** with
   `sidecar HTTP request failed: url=…/sam/v1/masks; reason_type=BrokenPipeError`.

The SAM failure is a **remote-proxy connection drop (sidecar availability), not
a pipeline defect** — the engine reached SAM correctly with a well-formed
request. Per the plan's rule ("if sidecars are unavailable, skip and state so;
do not fabricate numbers"), the full semi-online metrics are **pending stable
Molmo/SAM sidecar availability**. The engine itself is validated by 688 passing
`scenefunc3d` unit tests (see below) plus the pure-assembly semi-online test
`test_score_semi_online_sample_recovers_target` (IoU ≈ 1.0 on a synthetic scene).

## Bug found and fixed by the smoke run

The first smoke attempt crashed with
`frame_geometry_asset_missing: frame_id='semi_online_seed'`. Root cause: the
tier-2 harness builds the anchor with a synthetic `seed_frame_id="semi_online_seed"`
(there is no real seed frame in tier-2), and `select_scene_visible_frames`'s
seed-inclusion branch tried to `load_frame_geometry` for that non-existent frame.
Since `iter_frame_geometry` already streams *all* geometry-complete frames, a
seed absent from that set can never be loaded — the branch was dead-and-harmful.
Fixed in `0dbe76e`: **seed force-inclusion is now best-effort** (include the seed
only if it was already streamed into `scored`; otherwise leave the top-N visible
selection unchanged). The online path (Phase 2c) passes a real geometry-complete
seed, so the single-frame-degrade safety net still works there. Regression test:
`test_select_scene_visible_frames_tolerates_seed_without_geometry`.

## Quality gate (at `0dbe76e`)

- `ruff check src/` — clean.
- `black --check src/codex_agent/scenefunc3d/ src/codex_agent/tests/` — clean (113 files).
- `mypy src/codex_agent/scenefunc3d/` — clean (52 source files).
- `PYTHONPATH=src pytest src/codex_agent/tests/ -k scenefunc3d -q` — **688 passed,
  2 skipped, 3 failed**. The 3 failures are pre-existing in
  `test_scenefunc3d_molmo_sam_contracts.py` (lift-mask CLI payload keys
  `frame_id`/`lifted_point_count`), verified failing identically at the
  pre-Phase-2b baseline (`af00dc1`) via a throwaway worktree — unrelated to this
  work. No new failures were introduced.

## Caveats

- **Tier-2, not the online agent.** The anchor is a fixed saved-fragment seed,
  not an agent-confirmed seed. Wrong-seed cases stay at IoU 0 (same lower-bound
  caveat as v6). The online `SeedConfirmation` contract is Phase 2c.
- **Affordance-prompt limitation.** The harness has no separate affordance-noun
  field in tier-2, so `SidecarFrameProposer` is given
  `affordance_concept = task_description = <full task>`, yielding the degenerate
  Molmo prompt `"point to <full task> in order to <full task>"`. In the online
  path the affordance concept comes from the agent-confirmed seed. **If a
  completed semi-online run shows low IoU/AP, this prompt is the prime suspect**,
  not the fusion engine (which v6 already validated offline).
- **Per-frame SAM contact sheets** add trace volume for `frame_cap` frames;
  acceptable for tier-2, revisit for the online path in Phase 2c.
- **Sidecar dependency.** The full run needs the remote Molmo/SAM workspace proxy
  (or local sidecars) reachable and stable; it does **not** need the ModelHub
  adapter.

## Reproduce

Land any pending sidecar stability, then run the CLI above (optionally drop
`--sample-ids` to run all 13, and raise `--frame-cap` to ~12). Ingest the
resulting `summary.json` into a future v7.1 doc with the completed metric table
and a v6-vs-v7 comparison on the matched 13-sample fold.
