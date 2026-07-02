# v8 — 2D seed-confirmation online contract (2026-07-03)

Phase 2c lands the **online agent contract** that v7 deferred. The single LLM
turn now only establishes and confirms **one high-quality seed at the 2D level**
(a small `SeedConfirmation` JSON), and the deterministic Phase-2b
`run_anchor_multiview_pipeline` produces the final fused mask. This replaces the
fragile free-form "agent fuses everything and self-reports a big final JSON"
loop (P0-B) and the geometric over-selection it caused (P1).

This doc records the **contract landing + full quality gate + a real online
smoke** on `421254`. The smoke found and fixed **two** genuine contract-breaking
bugs (a strict-schema defect that made every online turn fail, and a
result-path defect that broke `--score`); see *Bugs found and fixed by the smoke
run* below. The scored single-sample result is **IoU 0.72**.

## Provenance

- **Branch:** `feat/scenefunc3d-agent-tools`
- **Head commit at write time:** `992a991` (includes the Phase-2c contract +
  the two Task-8 smoke-surfaced fixes).
- **Run-time code commit:** the online smoke ran at `c372a83` (the strict-schema
  fix); the scoring-path fix `992a991` landed afterward and was validated on the
  same fused mask (see *Bugs found and fixed*). **No worktree drift** — gate,
  smoke, and docs all run from this branch's head.
- **Plan:** [`docs/superpowers/plans/2026-07-02-scenefunc3d-seed-confirmation-contract-phase2c.md`](../../superpowers/plans/2026-07-02-scenefunc3d-seed-confirmation-contract-phase2c.md)
  (Phase 2c; head at launch `5ae92ef`, tasks 1–8 in commits `20c884e … 992a991`).
- **Phase 2b engine plan:** [`docs/superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md`](../../superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md).
- **Spec:** [`docs/superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md`](../../superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md).

## What changed vs v7

v7 landed the deterministic Stage B→C→D engine but validated it with a **fixed
saved-fragment seed** (tier-2), explicitly deferring the online agent contract
to Phase 2c. v8 is that contract:

- **The agent turn shrinks to a 2D seed.** Output schema goes from the 7-field
  `SceneFunc3dMaskDecision` (final-artifact JSON validated by the ~1000-line
  `final_mask_artifacts` + fuse-provenance machinery) to `SeedConfirmation`
  (`seed_frame_id`, `seed_candidate_id`, `affordance_concept`, `approval_actions`,
  `uncertainties`). The agent does **no 3D work**: it verifies the SAM candidate
  mask drawn on the *original RGB* and stops at "approve SAM candidate". The
  approval gate ends at `SAM_CANDIDATE_AGENT_APPROVED` (new
  `task.validate_seed_approval_actions`), not `FIRST_LIFT_AGENT_APPROVED`.
- **The 3D lift moved into the runner.** `SceneFunc3dMaskTask.parse_response`
  parses the schema, replays the SAM-candidate approval gates, and recovers the
  confirmed seed's **2D SAM candidate mask NPZ** from `events.jsonl`
  (`seed_recovery.recover_seed_sam_candidate`). `run_single_sample` then
  **deterministically lifts that SAM mask to 3D** (`_lift_seed_fragment`),
  freezes a `TargetAnchor` from the lifted points (`build_anchor`), and runs the
  deterministic anchor multi-view pipeline through the real Molmo/SAM sidecars
  (`_run_pipeline_for_seed` → `run_anchor_multiview_pipeline`).
- **New pipeline-shaped result.** The runner writes a
  `result_kind="anchor_multiview_pipeline"` `result.json` plus the fused
  `mask_data.npz` / `lifted_points.ply` (`pipeline_result.py`). Scoring goes
  through a new `score_pipeline_result_file` that scores the fused NPZ via the
  unchanged `score_mask_npz` / `metrics.py`; `score_result_file` dispatches on
  `result_kind` so legacy free-form result files still score.
- **Playbook trimmed to the 2D seed stage.** `SCENEFUNC3D_TOOL_NAMES` drops
  `lift_mask_to_3d`, `inspect_mask_artifact`, `suggest_additional_views`,
  `fuse_accepted_masks`; the final-JSON instructions describe `SeedConfirmation`.
- **Dead code removed.** The free-form decision/outcome types, the fuse
  self-report validators, and their tool-event models were deleted:
  `runner.py` shrank by **2,230 deleted / 201 inserted lines**; combined with the
  `tools/mask_artifacts.py` cleanup (−103) and the new seed/pipeline modules, the
  SceneFun3D source tree netted **−1,596 lines** (≈2.1k lines of dead
  free-form/fuse machinery removed). The removed runner tests also drop the
  `scenefunc3d` selected-test count from v7's 693 to 641 (636 passed +
  3 pre-existing failed + 2 skipped).

Fail-closed behavior (no-fallback rule): a missing confirmed SAM event →
`CodexResponseError` → `failure.json`; an unliftable seed → the lift raises →
`failure.json`; an empty fused mask → `SceneFunc3dDataError` → `failure.json`.
The runner never emits a zero-point / low-quality mask.

## Bugs found and fixed by the smoke run

The online smoke surfaced **two genuine contract-breaking defects** that the unit
tests could not catch (they validate parsing/serialization, not the live API's
strict-schema contract nor CWD-relative CLI paths). Both are small, both are now
fixed with regression tests, and both block the online path completely — so they
are exactly what the "quality gate for the phase 2c contract" is meant to catch.

**Bug 1 — strict output schema rejected (fixed in `c372a83`).** The first online
attempt failed immediately on the first agent turn:

```
RuntimeError: {"error":{"message":"code: invalid_json_schema; message: Invalid
schema for response_format 'codex_output_schema': In context=(), 'required' is
required to be supplied and to be an array including every key in properties.
Missing 'uncertainties'.", ... "code":"-4003"}}
```

Root cause: Codex structured outputs run with `strict: true`, which rejects a
schema unless **every** key in `properties` also appears in `required`.
`SeedConfirmation.uncertainties` had a Pydantic default (`= ()`), so Pydantic
dropped it from `required` and every online turn failed before doing any work.
Fix: make `uncertainties` a **required** field (matching the all-required
convention of the working `Nr3dGroundingDecision` and the legacy
`SceneFunc3dMaskDecision`); the agent emits an empty list when nothing is
ambiguous (playbook updated). A regression test asserts the emitted schema's
`required` set covers every property and `additionalProperties` is `False`.

**Bug 2 — pipeline result stored CWD-relative artifact paths (fixed in
`992a991`).** After Bug 1, the agent turn + full pipeline completed and wrote a
valid `result.json` + fused mask, but the `--score` step crashed:

```
FileNotFoundError: .../55b56254.../tmp/scenefunc3d/artifacts/.../fused/mask_data.npz
SceneFunc3dDataError: could not load predicted point-id NPZ ... (path doubled)
```

Root cause: `write_pipeline_fused_mask` returned CWD-relative paths when
`--output-dir` was relative, so `result.json` stored e.g.
`tmp/.../fused/mask_data.npz`; the scorer's `_resolve_artifact_member_path`
re-joins a **relative** member under the `result.json` directory, doubling the
run-dir prefix. (Legacy free-form results avoided this because the fuse tool
emitted absolute paths.) Fix: `write_pipeline_fused_mask` absolutizes `fused_dir`
so the durable `result.json` references absolute paths. A regression test runs
the writer under a relative dir (via `chdir`) and asserts absolute outputs.

After both fixes the online turn runs normally (BEV + frame evidence + Molmo/SAM
tool events), the deterministic pipeline fuses the mask, and `score_result_file`
loads and scores the fused NPZ — the metric below was produced by the fixed
scorer on this run's fused mask.

## Exact CLI (one-sample online smoke)

```bash
PYTHONPATH=src python -m codex_agent.scenefunc3d.runner \
  --dataset-root data/SceneFun3D \
  --sample-id '421254::55b56254-acf7-47f9-80f4-368e10816622' \
  --output-dir tmp/scenefunc3d/artifacts/sf3d_phase2c_20260703/ \
  --backend-config configs/scenefunc3d_backends.toml \
  --score
```

- **Sample:** `421254::55b56254-acf7-47f9-80f4-368e10816622` — *"Open the bottom
  drawer of the nightstand with the red table lamp on top"* (single annotation;
  a clean drawer-handle affordance). A different sample than v7's "right window"
  smoke, so the two n=1 numbers are not directly comparable.
- **Raw artifact dir:** `tmp/scenefunc3d/artifacts/sf3d_phase2c_20260703/`
  (must be under the backend config's `allowed_output_roots`).
- **Env at run time:** ModelHub adapter healthy on `127.0.0.1:8787`
  (`gpt-5.4-2026-03-05`), Molmo + SAM sidecars healthy via the workspace proxy,
  runtime preflight green. Run in `tmux` (session `sf3d_2c`) per `AGENTS.md`.

## Headline

**End-to-end status: VERIFIED (online).** The agent-confirmed 2D seed drove the
deterministic anchor multi-view pipeline to a **scored** fused mask through the
real ModelHub adapter + Molmo/SAM sidecars. The agent confirmed seed frame
`000011`, SAM candidate `mask_00`, affordance concept **"the bottom drawer
knob"** (seed lift = 1,877 points); the pipeline selected 30 frames and accepted
22 (8 rejected by the anchor gate), fusing to a mask with confidence `0.888`.
Agent turn duration ≈ 185 s.

Single-sample smoke on `421254::55b56254-acf7-47f9-80f4-368e10816622`
(runner defaults: `frame_cap=30`, `agreement_tau=0.5`, `radius_scale=1.0`):

| Metric | Value (n=1 smoke) |
|---|---|
| Mean IoU | **0.7174** |
| Precision | 0.7952 |
| Recall | 0.8800 |
| F1 | 0.8354 |
| AP25 (IoU ≥ 0.25) | 1.00 (1/1) |
| AP50 (IoU ≥ 0.50) | 1.00 (1/1) |
| Predicted point count | 83 |
| GT point count | 75 |

**No numbers fabricated.** This is the **first online run of the agent-confirmed
2D seed contract**, and — unlike v7's fixed-seed-proxy smoke (IoU 0.0 on a
harder "right window" target) — the agent selected a clean drawer-knob seed and
the pipeline produced a tight, high-overlap mask (IoU 0.72, above both AP
thresholds). This is a single, encouraging data point that proves the online
wiring *and* a plausible accuracy regime; it is **not** decision-grade. A full
fold (the 13-sample `421254` set used by v6/v7, or a stratified `421254` fold)
is required before any v7-vs-v8 or leaderboard claim.

The contract is additionally validated by **636 passing `scenefunc3d` unit tests**
(see the gate below), including new regression tests for both smoke-surfaced bugs.

## Quality gate (at `992a991`, macOS `.venv`)

- `ruff check src/` — **clean**.
- `black --check src/codex_agent/scenefunc3d/ src/codex_agent/tests/` — **clean**
  (122 files).
- `mypy src/codex_agent/scenefunc3d/` — **clean** (56 source files).
- `PYTHONPATH=src pytest src/codex_agent/tests/ -k scenefunc3d -q` —
  **636 passed, 2 skipped, 3 failed, 329 deselected**.

The **3 failures are pre-existing and unrelated** to Phase 2c. They are all in
`test_scenefunc3d_molmo_sam_contracts.py` (lift-mask CLI payload keys):

- `test_cli_sam_crop_candidate_lifts_against_full_frame_depth` — `KeyError: 'lifted_point_count'`
- `test_cli_lift_mask_writes_deterministic_artifacts` — `KeyError: 'frame_id'`
- `test_cli_lift_mask_resolves_frame_geometry_when_paths_are_omitted` — `KeyError: 'frame_id'`

That test file and the lift-CLI implementation (`tools/mask_lifting.py`,
`tools/cli.py`) are **untouched since v7** (`git log 5ae92ef..HEAD` on those paths
is empty), and v7 already documented the same 3 failing identically at the
pre-Phase-2b baseline `af00dc1`. No new failures were introduced by Phase 2c.
The higher pass count vs the raw v7 count reflects the three net-new regression
tests added with the two smoke-surfaced fixes (two for the strict-schema fix,
one for the scoring-path fix), offset by the free-form runner tests removed in
Phase 2c.

## Caveats

- **Online headline is n=1.** One sample proves the online wiring
  (agent-confirmed 2D seed → deterministic lift → anchor multi-view pipeline →
  fused mask → score); it does **not** establish accuracy. A decision-grade
  number needs a full fold (v6/v7 used the 13-sample `421254` fragment fold).
- **This is the first online run of the agent-confirmed seed.** v6/v7 used a
  fixed largest-fragment seed proxy; v8's seed comes from the LLM turn. So a
  v7-vs-v8 IoU comparison is only meaningful on a matched fold *and* accounting
  for the seed source difference.
- **The seed frame is re-processed by the pipeline** (Stage C runs Molmo→SAM→lift
  on every selected frame, including the seed frame). Accepted for Phase 2c;
  reusing the confirmed seed lift for its own frame is a follow-up.
- **`motion_type` uses the sample's first motion hint** to size the anchor.
- **Sidecar + adapter dependency.** The online path needs the ModelHub adapter
  (`127.0.0.1:8787`) *and* the remote Molmo/SAM workspace proxy reachable and
  stable. The unit/integration path (636 passing tests) proves the contract and
  wiring without either.

## Reproduce

With the adapter (`curl -sf http://127.0.0.1:8787/health`) and the Molmo/SAM
sidecars (`check_sidecar_health(configs/scenefunc3d_backends.toml)`) healthy, run
the CLI above (drop `--sample-id` for a `--sample-ids-path` fold, add `--workers`
for concurrency). Ingest the scored `result.json` values into a v8.1 doc with a
completed metric table and a v6/v7-vs-v8 comparison on the matched fold.
