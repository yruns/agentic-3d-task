# NR3D v16 — Canonical `strat600` with upgraded tool suite & prompt disambiguation (`gemini-3.8-flash`), 2026-11-04

Full **canonical `strat600` (600 samples across 119 ScanNet scenes)** evaluation of
the **v15 NR3D tool suite & prompt disambiguation overhaul** (`caebb4a` + `898b3fa`,
launched at `8aa55a9`) using **`gemini-3.8-flash`** via `codex_adapter`.

---

## Headline

- **`gemini-3.8-flash` (`v16`, upgraded tools & prompt):**
  - **Overall `Acc@0.25` (`== Acc@0.50`):** **92.50% (555 / 600)** (**+1.00 pp / +6 net samples** over `v14 gemini-3.8-flash` at `91.50%`, **+6.50 pp** over `v7 gpt-5.4` at `86.00%`, and **+7.67 pp** over `v14 gpt-5.4` at `84.83%`).
  - **Mean 3D IoU:** **0.9253** (vs `0.9159` in `v14`).
  - **Easy (`n = 290`):** **94.14% (273 / 290)** (**+0.69 pp** over `v14`).
  - **Hard (`n = 310`):** **90.97% (282 / 310)** (**+1.29 pp** over `v14` `89.68%` — **first run to break 90% on `Hard`**).
  - **View-Dep (`n = 211`):** **90.05% (190 / 211)** (**+0.48 pp** over `v14` `89.57%` — **first run to break 90% on `View-Dep`**).
  - **View-Indep (`n = 389`):** **93.83% (365 / 389)** (**+1.29 pp** over `v14` `92.54%`).
  - **`flip41` hard/high-variance subset inside `strat600` (`n = 41`):** **82.93% (34 / 41)** (consistent with the standalone [v15 `flip41` diagnostic](v15_flip41_tools_upgrade_20261104.md) at `85.37% (35/41)` and **+31.71 pp** over `v8 repeat` at `51.22%`).
  - **Target-absent (`selected_object_id = -1`) false refusals:** **0 / 600** (`0.0%`).

| Slice | n | v7 (`gpt-5.4`) | v8 (`gpt-5.4`) | v14 (`gpt-5.4`) | v14 (`gemini-3.8-flash`) | **v16 (`gemini-3.8-flash` + v15 tools)** | $\Delta$ (`v16` − `v14 gemini`) | $\Delta$ (`v16` − `v14 gpt`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | 86.00% (516) | 84.50% (507) | 84.83% (509) | 91.50% (549) | **92.50% (555)** | **+1.00 pp (+6)** | **+7.67 pp (+46)** |
| **Easy** | 290 | 90.34% (262) | 88.62% (257) | 88.28% (256) | 93.45% (271) | **94.14% (273)** | **+0.69 pp (+2)** | **+5.86 pp (+17)** |
| **Hard** | 310 | 81.94% (254) | 80.65% (250) | 81.61% (253) | 89.68% (278) | **90.97% (282)** | **+1.29 pp (+4)** | **+9.36 pp (+29)** |
| **View-Dep** | 211 | 82.46% (174) | 77.25% (163) | 80.09% (169) | 89.57% (189) | **90.05% (190)** | **+0.48 pp (+1)** | **+9.96 pp (+21)** |
| **View-Indep** | 389 | 87.92% (342) | 88.43% (344) | 87.40% (340) | 92.54% (360) | **93.83% (365)** | **+1.29 pp (+5)** | **+6.43 pp (+25)** |

### 2×2 Sub-slice Breakdown (`is_easy × is_view_dep`)

| Sub-slice | n | Correct | `Acc@0.25` (`== Acc@0.50`) | Mean IoU |
|---|---:|---:|---:|---:|
| **Easy + View-Indep** | 198 | 187 | **94.44%** | 0.9446 |
| **Easy + View-Dep** | 92 | 86 | **93.48%** | 0.9359 |
| **Hard + View-Indep** | 191 | 178 | **93.19%** | 0.9321 |
| **Hard + View-Dep** | 119 | 104 | **87.39%** | 0.8741 |

---

## Summary of NR3D pipeline upgrades validated in `v16`

1. **View-Quality & Diversity-Aware Frame Selection (`select_by_proposal`):**
   - Replaced naive `sorted(frame_ids)[:3]` selection with 2D bbox view-quality
     scoring ($0.65 \cdot \text{AreaFrac} + 0.35 \cdot \text{CenterBonus}$ with
     border-clipping penalty) and camera pose/frame separation ($\ge 0.35\text{m}$,
     $\ge 0.25\text{rad}$).
   - `require_all=True` automatically falls back to maximal partial co-visibility when
     $>2$ candidates cannot fit in a single frame and returns
     `per_proposal_best_frames` + `requested_left_to_right`.
   - `require_all=False` uses per-proposal balanced round-robin coverage (reducing
     offline target miss rate at $k=3$ on `strat600` from **46.5% to 0.3%**).
2. **3D Virtual Observer Viewpoint Projection & Extended Spatial Relations (`spatial_tools.py`):**
   - Added `in_front_of`, `behind`, `between` (with exact 3D line-segment distance
     `between_segment_distances`), and `opposite`.
   - Added `facing_left_to_right_ids` and `facing_front_to_back_ids` computed from a
     virtual observer standing at the room centroid (or explicit `observer_id`)
     looking toward the target/anchor group, eliminating tie-break failures when
     candidates have `shared_frame_count == 0`.
3. **Batch Contact Sheets & Unoccluded High-Res Crops (`mark_frame_with_bbox` & `view_bev`):**
   - `mark_frame_with_bbox` supports batch `frame_ids` (up to 4 frames in one call)
     and automatically generates `crops_sheet_path` (high-resolution unoccluded crops
     of each target from raw RGB with a 22% margin before downscaling or drawing
     thick bounding box strokes).
   - `view_bev` projects and draws oriented 3D bounding box footprints on the top-down
     map.
4. **Prompt & Playbook Disambiguation (`grounding.py` / `playbook.py`):**
   - Inlined `color=...; nearby=[...]; loc=...` on every proposal line in the initial
     catalog.
   - Annotated `Proposals by category` with instance counts (`>= 2 instances` vs
     `1 instance — context/anchor only`) and enforced external-facing perspective
     rules (`facing the X`) plus anti-refusal (`-1` ban).

---

## Run health & adapter telemetry

| Metric | `v14 gemini-3.8-flash` | **`v16 gemini-3.8-flash`** |
|---|---:|---:|
| Sample records | 600 / 600 | **600 / 600** |
| Final status | 600 completed, 0 failed | **600 completed, 0 failed** |
| Target-absent (`-1`) selections | >0 | **0 / 600** |
| Mean 3D IoU | 0.9159 | **0.9253** |
| Prompt cache hit rate | 0.3133 | **0.3233** |
| Mean cache ratio | 0.2259 | **0.2432** |
| Turn latency (median / p90 / max) | 79.0 s / 217.7 s / 768.0 s | **81.9 s / 263.2 s / 915.7 s** (at `--workers 24`) |

---

## Run metadata & reproduction

- **Branch:** `feat/upgrade_keyframe_selector`
- **Head commit at launch:** `8aa55a9` (code commits `caebb4a` + `898b3fa`; clean working tree `head == executed code`).
- **Fold:** `/home/ysh/Datasets/nr3d/processed_v1/folds/strat600_sample_ids.json` (canonical `strat600`: 600 samples across 119 scenes).
- **Data root:** `/home/ysh/Datasets/nr3d/processed_v1/scannet`
- **Pack:** `pack_nr3d_v9_catalog_first`
- **Concurrency:** Main pass at `--workers 24` (`597 / 600` completed in ~71 min), followed by a 3-sample low-concurrency recovery pass at `--workers 3` (`600 / 600` completed in ~2 min).

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
export no_proxy=127.0.0.1,localhost,::1
export NO_PROXY=127.0.0.1,localhost,::1

python -m codex_agent.cli.run_nr3d \
  --data-root /home/ysh/Datasets/nr3d/processed_v1/scannet \
  --sample-ids /home/ysh/Datasets/nr3d/processed_v1/folds/strat600_sample_ids.json \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_v15_strat600_gemini38flash_20261104_8aa55a9 \
  --model gemini-3.8-flash \
  --reasoning-effort provider-default \
  --tools \
  --turn-timeout 900 \
  --max-tool-calls 18 \
  --sample-retries 5 \
  --workers 24
```

## Durable artifacts & SQLite ingestion

- **Committed leaderboard summary:** `docs/benchmark/nr3d/assets/v16_strat600_tools_upgrade_20261104_leaderboard.json`
- **SQLite database (`docs/benchmark/nr3d/runs.sqlite`):** `run_id = v16_strat600_tools_upgrade_20261104` (600 sample rows ingested via `scripts/ingest_nr3d_run.py`).
