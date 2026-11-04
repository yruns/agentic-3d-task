# NR3D v15 — Tool suite & prompt disambiguation upgrade on `flip41` diagnostic fold (`gemini-3.8-flash`), 2026-11-04

Targeted evaluation of the **v15 NR3D tool suite & prompt upgrade** (`caebb4a` +
`898b3fa`) on the frozen **41-sample `flip41` diagnostic fold**
(`docs/benchmark/nr3d/assets/v7_v8_flip41_sample_ids_20260701.json`, the 41 hard /
high-variance samples from `strat600`), using **`gemini-3.8-flash`** via
`codex_adapter`.

---

## Summary of NR3D pipeline upgrades (`v14` $\rightarrow$ `v15`)

1. **`select_by_proposal` (`src/codex_agent/nr3d/tools/frame_tools.py`):**
   - Replaced naive `sorted(frame_ids)[:3]` selection with 2D bbox view-quality
     scoring (area + centering + border-clipping penalty) and trajectory pose/frame
     diversity filtering.
   - `require_all=True` now automatically falls back to **maximal partial
     co-visibility** when $>2$ candidate IDs cannot fit in a single frame, and
     returns `per_proposal_best_frames` + `requested_left_to_right`.
   - `require_all=False` uses **per-proposal balanced coverage + spatial
     de-duplication**. On the full 600-sample `strat600` fold, offline target miss
     rate at $k=3$ dropped from **46.5% (279/600) to 0.3% (2/600)** and adjacent
     duplicate frame batches dropped from **94.2% (565/600) to 0.3% (2/600)**.
2. **`compare_proposals_spatial` & `compare_candidates_to_anchors` (`spatial_tools.py`):**
   - Fixed self-anchor contamination when `anchor_id` is included in `candidate_ids`.
   - Added support for `in_front_of`, `behind` (co-visible camera-to-object depth
     voting), `between` (group centrality + exact 3D line-segment distance
     `between_segment_distances`), and `opposite`.
   - Added **3D Virtual Observer Viewpoint projection** (`observer_id` +
     `facing_left_to_right_ids` and `facing_front_to_back_ids`), eliminating the
     tie-break failure when `shared_frame_count == 0`.
3. **`mark_frame_with_bbox` & `view_bev` (`frame_annotation.py` / `bev_tools.py`):**
   - `mark_frame_with_bbox` now accepts `frame_ids` (up to 4 frames in one call,
     composing a single `contact_sheet`) and automatically generates
     `crops_sheet_path` (high-resolution, unoccluded crops of each target from raw
     RGB without thick bbox strokes).
   - `view_bev` projects and draws oriented 3D bounding box footprints on the top-down
     map and returns `highlighted_objects` with 3D/2D coordinates.
4. **Prompt & Playbook Disambiguation (`grounding.py` / `playbook.py`):**
   - Exposed `color=...; nearby=[...]; loc=...` directly on every proposal line in
     the initial prompt (fixing the 30.8% color truncation rate on `strat600`).
   - Annotated `Proposals by category` with instance counts (`>= 2 instances` vs
     `1 instance — context/anchor only`) and added explicit guards against
     foreground-occluder VLM note pollution (`33.3%` of `strat600`) and false `-1`
     refusals.

---

## Results on `flip41` (41 hard / unstable samples)

`v15` completed **41 / 41 samples (`0` target-absent `-1` refusals, `0` failed
turns)**, reaching **35 / 41 = 85.37% Acc@0.25** (**+34.15 pp** over `v8 repeat` and
**+24.39 pp** over `v7`):

| Slice | n | v7 (`gpt-5.4`) | v8 (`gpt-5.4`) | v8 repeat (`gpt-5.4`) | **v15 `gemini-3.8-flash`** | $\Delta$ (`v15` − `v8 repeat`) | $\Delta$ (`v15` − `v7`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 41 | 60.98% (25) | 39.02% (16) | 51.22% (21) | **85.37% (35)** | **+34.15 pp (+14)** | **+24.39 pp (+10)** |
| **Easy** | 13 | 69.23% (9) | 30.77% (4) | 53.85% (7) | **92.31% (12)** | **+38.46 pp (+5)** | **+23.08 pp (+3)** |
| **Hard** | 28 | 57.14% (16) | 42.86% (12) | 50.00% (14) | **82.14% (23)** | **+32.14 pp (+9)** | **+25.00 pp (+7)** |
| **View-Dep** | 23 | 73.91% (17) | 26.09% (6) | 47.83% (11) | **86.96% (20)** | **+39.13 pp (+9)** | **+13.05 pp (+3)** |
| **View-Indep** | 18 | 44.44% (8) | 55.56% (10) | 55.56% (10) | **83.33% (15)** | **+27.77 pp (+5)** | **+38.89 pp (+7)** |

---

## Representative recovered failure cases

- **`scannet/scene0169_00::26::28815`** (*"The smaller cabinet."*, GT=`#26`):
  Previously failed as `-1` (`target_absent`) in `v11` because `#26`'s offline VLM
  note described an occluding black drafting chair. `v15` verified the 3D size and
  selected `#26` (`IoU = 1.0`).
- **`scannet/scene0221_00::51::24914`** (*"The towel in the furthest corner of the room"*, GT=`#51`):
  Previously failed as `-1` in `v8 repeat` because `#51`'s offline note described a
  wall-mounted hairdryer cord. `v15` selected `#51` (`IoU = 1.0`).
- **`scannet/scene0063_00::10::1029`** (*"The larger of the cabinets."*, GT=`#10`):
  With singleton categories marked as `1 instance — context/anchor only`, `v15`
  compared the two `kitchen cabinet` instances (`#10` vs `#16`) and selected `#10`
  (`IoU = 1.0`).
- **`scannet/scene0435_00::60::27767`** (*"Facing the foot of the bed, the suitcase on the left."*, GT=`#60`):
  With the external-facing viewpoint rule and virtual-observer spatial projection,
  `v15` selected `#60` (`IoU = 1.0`).

---

## Run metadata & reproduction

- **Branch:** `feat/upgrade_keyframe_selector`
- **Commits:** `caebb4a` + `898b3fa` (clean working tree).
- **Fold:** `docs/benchmark/nr3d/assets/v7_v8_flip41_sample_ids_20260701.json` (41 samples).
- **Model:** `gemini-3.8-flash` (`reasoning_effort=provider-default`, `--tools`).
- **Committed artifact:** `docs/benchmark/nr3d/assets/v15_flip41_tools_upgrade_20261104_summary.json`.
- **SQLite (`docs/benchmark/nr3d/runs.sqlite`):** `v15_flip41_tools_upgrade_20261104`.
