# OpenEQA v6 — Map-First initialization, tool suite overhaul, and category guidance on `pilot300` (`gemini-3.8-flash`), 2026-11-04

Evaluation of the **v6 OpenEQA pipeline overhaul** (`625414a` + `80d894f` + `4e092d5`)
on the **canonical 300-question OpenEQA pilot fold**
(`assets/v5_pilot_fold_300_20260612.json`, stratified across all 7 categories over
85 ScanNet clips), using **`gemini-3.8-flash`** (stateful `gemini_chat` codec via
`codex_adapter`) and **`gemini-2.5-pro`** as the official `mmbench` LLM-as-judge.

---

## Summary of pipeline changes (v5 $\rightarrow$ v6)

1. **Map-First Initialization (zero initial first-person RGB frames preserved):**
   - Attaches the cached top-down **Schematic BEV map** (`1` overview image) to the
     initial turn and inlines `total_frames`, `frame_id_range`, `trajectory_stride`,
     `Scene Grid Layout` (`(row, col)` $\rightarrow$ `#id category`), and a compact
     `Detected 3D Object Catalog Preview` (including each object's `grid=(r,c)` and
     top `best_frames=[...]`).
   - Eliminates blind orientation tool calls at the start of every turn while keeping
     all first-person visual evidence strictly on-demand.
2. **Lightweight ConceptGraph Cache & Index Alignment (~140$\times$ faster tool loads):**
   - Enabled `ensure_lightweight_pcd=True` and precomputed `.light.pkl.gz` sidecars
     across all 89 OpenEQA scenes, reducing per-tool scene load latency from
     **~10,000 ms to ~72 ms** and eliminating `write_stdin` polling storms.
   - Rebuilt `visibility_index.pkl` and `grid_visibility_index.pkl` at `stride=2`
     for the 5 scenes (`002`, `003`, `012`, `013`, `014`) that previously carried
     stale `stride=5` / `sam3` index metadata.
3. **Tool Suite Bug Fixes & Upgrades:**
   - **`view_crop`:** Fixed two critical bugs: (a) native `1296x968` detection boxes
     were previously scaled by `height / 680` (+42% vertical offset error);
     (b) `_best_view_and_box` previously returned `(view_id, None)` on the first
     ranked view lacking a 2D detection box instead of scanning subsequent ranked
     views. Added joint `{"object_id": id, "frame_id": fid}` support.
   - **`list_objects`:** Exposed full enrichment metadata (`grid_cell`,
     `best_frame_ids`, `color`, `location`, `nearby_objects`, `usability`, `note`)
     and added multi-field keyword `query`, `object_ids`, and `grid_cells` filters.
   - **`view_frame`:** Added `around_frame_id` + `step` trajectory neighborhood
     panning, `grid_cells` frame lookup, and per-frame `visible_objects` with
     normalized 2D boxes (`bbox_norm`) ready for `view_crop`.
   - **`keyframe_selector` & `view_bev`:** Enabled multimodal BEV visual context by
     default (`use_bev=True`), added `grounded_objects` and per-frame
     `visible_objects` (`bbox_norm`), and added `highlight_grids` to `view_bev`.
4. **Category-Specific Prompt & Playbook Guidance:**
   - Added explicit guards against treating the 3D object catalog as complete or
     infallible (preventing false "no / impossible" answers when wall-mounted or
     small desktop items are omitted from 3D detection).
   - Added per-category instructions (`functional reasoning`, `world knowledge`,
     `object localization`, `spatial understanding`, `object recognition`,
     `attribute recognition`, `object state recognition`), including naming both the
     specific dedicated item and its surface/location.

---

## Headline results (`pilot300`)

`v6` completed **300 / 300 questions (`100.0%` completion)** in **~33 minutes** total
wall-clock time (down from **~120 minutes** in `v5`), achieving **MNAS = 83.08**:

- **+4.16 MNAS** over `v5 gemini-3.8-flash` (`78.92` $\rightarrow$ `83.08`)
- **+13.83 MNAS** over `v5 gpt-5.4-2026-03-05` (`69.25` $\rightarrow$ `83.08`)
- **+18.16 MNAS** over the `v4` baseline on the same 300 questions (`64.92` $\rightarrow$ `83.08`)
- **4.0$\times$ lower median turn latency** (`165.1 s` $\rightarrow$ **`41.0 s`**)

| Metric | v4 baseline (same 300 IDs) | v5 `gpt-5.4-2026-03-05` | v5 `gemini-3.8-flash` | **v6 `gemini-3.8-flash`** | $\Delta$ (`v6` vs `v5 gemini`) | $\Delta$ (`v6` vs `v4`) |
|---|---:|---:|---:|---:|---:|---:|
| **Overall MNAS** | 64.92 | 69.25 | 78.92 | **83.08** | **+4.16** | **+18.16** |
| Questions (`n_judged / n`) | 300 / 300 | 300 / 300 | 300 / 300 | **300 / 300** | — | — |
| Completion rate | 100.0% | 100.0% | 100.0% | **100.0%** | 0.0% | 0.0% |
| Prompt cache hit rate | 97.7% | 97.0% | 85.0% | **74.7%** | −10.3% | −23.0% |
| Mean cache ratio | 46.6% | 84.3% | 59.3% | **56.4%** | −2.9% | +9.8% |
| **Median turn latency (s)** | ~114.0 s | 189.7 s | 165.1 s | **41.0 s** | **−124.1 s (−75.2%)** | **−73.0 s** |
| **Mean turn latency (s)** | — | 203.5 s | 178.7 s | **54.6 s** | **−124.1 s (−69.4%)** | — |
| **Judge score = 5 (perfect)** | 135 (45.0%) | 149 (49.7%) | 180 (60.0%) | **200 (66.7%)** | **+20 (+6.7 pp)** | **+65 (+21.7 pp)** |
| **Judge score = 1 (miss)** | 66 (22.0%) | 51 (17.0%) | 29 (9.7%) | **19 (6.3%)** | **−10 (−3.4 pp)** | **−47 (−15.7 pp)** |

Judge-score histograms (1–5):

- **`v5 gpt-5.4-2026-03-05`:** `1: 51, 2: 13, 3: 39, 4: 48, 5: 149`
- **`v5 gemini-3.8-flash`:** `1: 29, 2: 3, 3: 40, 4: 48, 5: 180`
- **`v6 gemini-3.8-flash`:** `1: 19, 2: 6, 3: 34, 4: 41, 5: 200`

*(Note on cache hit rate: in `v6`, ~25% of questions resolve in a single fast tool
round because the initial prompt already includes the BEV map and `best_frames`, so
fewer turns require multi-round prefix-cached continuations.)*

---

## Per-category MNAS (`pilot300`)

| Category | n | v4 baseline | v5 `gpt-5.4` | v5 `gemini-3.8-flash` | **v6 `gemini-3.8-flash`** | $\Delta$ (`v6` − `v5 gemini`) | $\Delta$ (`v6` − `v4`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **object state recognition** | 45 | 77.78 | 87.22 | 90.56 | **95.00** | **+4.44** | **+17.22** |
| **attribute recognition** | 45 | 72.78 | 79.44 | 85.56 | **88.89** | **+3.33** | **+16.11** |
| **object localization** | 45 | 67.22 | 62.78 | 82.22 | **84.44** | **+2.22** | **+17.22** |
| **object recognition** | 43 | 52.33 | 59.88 | 75.58 | **81.40** | **+5.82** | **+29.07** |
| **world knowledge** | 39 | 65.38 | 66.67 | **80.13** | 79.49 | −0.64 | **+14.11** |
| **spatial understanding** | 42 | 51.79 | 60.71 | 74.40 | **76.19** | **+1.79** | **+24.40** |
| **functional reasoning** | 41 | 65.85 | 66.46 | 62.20 | **74.39** | **+12.19** | **+8.54** |
| **Overall** | **300** | **64.92** | **69.25** | **78.92** | **83.08** | **+4.16** | **+18.16** |

---

## Representative question-level improvements (`v5` $\rightarrow$ `v6`)

| Category | Question | GT Answer | v5 Prediction (Score) | **v6 Prediction (Score)** |
|---|---|---|---|---|
| `functional reasoning` | How can I call 911? | using the phone hanging on the wall | You cannot because there is no phone in the room. (1) | **use the wall-mounted phone above the chest of drawers (5)** |
| `functional reasoning` | The printer ran out of paper, where can I get more? | Open one of the green boxes of paper | on top of the filing cabinet (1) | **from the stacks of green paper reams on top of the filing cabinet (5)** |
| `functional reasoning` | Is there something to wipe my nose? If so where is it? | Yes, over the chest of drawers | no (1) | **yes, a tissue box on the wooden cabinet in front of the window (4)** |
| `object recognition` | What is the red object next to the sofa? | vacuum cleaner | fire extinguisher (1) | **vacuum cleaner (5)** |
| `attribute recognition` | What color are the patio chairs? | White | dark brown (1) | **white (5)** |
| `attribute recognition` | What color is the deadbolt | Black | silver (1) | **black (5)** |
| `object state recognition` | Is the door near the brown paper box open or closed? | closed | open (1) | **closed (5)** |
| `object state recognition` | Is the dryer closed? | no | yes (1) | **no (5)** |
| `spatial understanding` | What is to the left of the painting? | A world map | television (1) | **a world map (5)** |
| `world knowledge` | How many people are intended to use this space for work? | 2 | 1 (1) | **2 people (5)** |

---

## Run metadata & reproduction commands

- **Branch:** `feat/upgrade_keyframe_selector`
- **Head commit at launch:** `4e092d5` (`fix(openeqa): treat transient 409/502/504/indeterminate adapter errors as retryable across sample attempts`). Working tree clean (`head == executed code`).
- **Fold:** `docs/benchmark/openeqa/assets/v5_pilot_fold_300_20260612.json` (300 questions across 85 ScanNet clips).
- **Answerer model:** `gemini-3.8-flash` (`reasoning_effort=provider-default`).
- **Judge model:** `gemini-2.5-pro` via `configs/llm.toml`.
- **Sandbox & loop caps:** `workspace_write` + network; `max_tool_calls=32`, `max_repeated_tool_calls=4`, `turn_timeout=900`.
- **Concurrency:** Launched in `tmux` session `openeqa_v6_pilot300_20261104` at `--workers 16` (~31 min main pass, 298/300 completed), followed by a 2-sample recovery pass at `--workers 2` (~2 min) $\rightarrow$ **300/300 completed**.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
source scripts/datasets/processed_v1.sh

python -m codex_agent.cli.run_openeqa \
  --questions "$OPENEQA_QUESTIONS" \
  --data-root "$OPENEQA_DATA_ROOT" \
  --question-ids docs/benchmark/openeqa/assets/v5_pilot_fold_300_20260612.json \
  --output-dir tmp/openeqa_v6_pilot300_gemini38flash_20261104_4e092d5 \
  --model gemini-3.8-flash \
  --judge-model gemini-2.5-pro \
  --reasoning-effort provider-default \
  --turn-timeout 900 \
  --sample-retries 4 \
  --workers 16
```

## Durable artifacts & SQLite ingestion

- **Committed assets:**
  - `docs/benchmark/openeqa/assets/v6_mapfirst_pilot300_gemini38flash_20261104_summary.json`
  - `docs/benchmark/openeqa/assets/v6_mapfirst_pilot300_gemini38flash_20261104_predictions.json`
- **SQLite (`docs/benchmark/openeqa/runs.sqlite`):**
  - Run ID: `v6_mapfirst_pilot300_gemini38flash_20261104`
