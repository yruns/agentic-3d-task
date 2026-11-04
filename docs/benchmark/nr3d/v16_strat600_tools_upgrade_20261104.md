# NR3D v16 — Canonical `strat600` with upgraded tool suite & prompt disambiguation (`gemini-3.8-flash`), 2026-11-04

Full **canonical `strat600` (600 samples across 119 ScanNet scenes)** evaluation of
the **v15 NR3D tool suite & prompt disambiguation overhaul** (`caebb4a` + `898b3fa`,
launched at `8aa55a9`) using **`gemini-3.8-flash`** (stateful `gemini_chat` codec via
`codex_adapter`).

All 600 samples run with the self-contained inline tool playbook (`--tools`),
`workspace_write` sandbox + network access, and the `pack_nr3d_v9_catalog_first`
proposal pool, scored by deterministic oriented 3D IoU against the ground-truth 9-DOF
bounding box (`Acc@0.25 == Acc@0.50` because proposals come from the GT pool).

---

## 1. Headline & Historical Progression on Canonical `strat600`

- **`gemini-3.8-flash` (`v16`, upgraded tools & prompt):**
  - **Overall `Acc@0.25` (`== Acc@0.50`):** **92.50% (555 / 600)** — sets a new
    benchmark high on `strat600` (**+1.00 pp / +6 net samples** over `v14 gemini-3.8-flash` at `91.50%`, **+6.50 pp / +39 samples** over `v7 gpt-5.4` at `86.00%`, **+7.67 pp / +46 samples** over `v14 gpt-5.4` at `84.83%`, and **+28.67 pp / +172 samples** over the `v1` prompt-only floor at `63.83%`).
  - **Mean 3D IoU:** **0.9253** (up from `0.9159` in `v14`).
  - **All four canonical slices exceed 90% accuracy for the first time:**
    - **Easy (`n = 290`):** **94.14% (273 / 290)** (**+0.69 pp** over `v14`).
    - **Hard (`n = 310`):** **90.97% (282 / 310)** (**+1.29 pp** over `v14` `89.68%` — first run to surpass 90% on `Hard`).
    - **View-Dep (`n = 211`):** **90.05% (190 / 211)** (**+0.48 pp** over `v14` `89.57%` — first run to surpass 90% on `View-Dep`).
    - **View-Indep (`n = 389`):** **93.83% (365 / 389)** (**+1.29 pp** over `v14` `92.54%`).
  - **`flip41` hard/high-variance diagnostic subset inside `strat600` (`n = 41`):**
    **82.93% (34 / 41)** (consistent with the standalone [v15 `flip41` run](v15_flip41_tools_upgrade_20261104.md) at `85.37% (35/41)` and **+31.71 pp** over `v8 repeat` at `51.22%`).
  - **Zero target-absent (`selected_object_id = -1`) false refusals:** **0 / 600** (`0.0%`).

| Slice | n | v1 (`gpt-5.4` floor) | v4 (`gpt-5.4` tools) | v7 (`gpt-5.4` passthrough) | v8 (`gpt-5.4` fail-closed) | v11 (`gpt-5.4` Skill) | v14 (`gpt-5.4`) | v14 (`gemini-3.8-flash`) | **v16 (`gemini-3.8-flash` + v15 tools)** | $\Delta$ (`v16` − `v14 gemini`) | $\Delta$ (`v16` − `v14 gpt`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | 63.83% (383) | 78.33% (470) | 86.00% (516) | 84.50% (507) | 84.50% (507) | 84.83% (509) | 91.50% (549) | **92.50% (555)** | **+1.00 pp (+6)** | **+7.67 pp (+46)** |
| **Easy** | 290 | 73.10% (212) | 82.76% (240) | 90.34% (262) | 88.62% (257) | 89.31% (259) | 88.28% (256) | 93.45% (271) | **94.14% (273)** | **+0.69 pp (+2)** | **+5.86 pp (+17)** |
| **Hard** | 310 | 55.16% (171) | 74.19% (230) | 81.94% (254) | 80.65% (250) | 80.00% (248) | 81.61% (253) | 89.68% (278) | **90.97% (282)** | **+1.29 pp (+4)** | **+9.36 pp (+29)** |
| **View-Dep** | 211 | 51.66% (109) | 68.25% (144) | 82.46% (174) | 77.25% (163) | 79.15% (167) | 80.09% (169) | 89.57% (189) | **90.05% (190)** | **+0.48 pp (+1)** | **+9.96 pp (+21)** |
| **View-Indep** | 389 | 70.44% (274) | 83.80% (326) | 87.92% (342) | 88.43% (344) | 87.40% (340) | 87.40% (340) | 92.54% (360) | **93.83% (365)** | **+1.29 pp (+5)** | **+6.43 pp (+25)** |

---

## 2. Fine-Grained Multi-Dimensional Breakdown

### 2.1 2×2 Sub-slice Breakdown (`is_easy × is_view_dep`)

| Sub-slice | n | Correct | Errors | `Acc@0.25` (`== Acc@0.50`) | Mean IoU |
|---|---:|---:|---:|---:|---:|
| **Easy + View-Indep** | 198 | 187 | 11 | **94.44%** | 0.9446 |
| **Easy + View-Dep** | 92 | 86 | 6 | **93.48%** | 0.9359 |
| **Hard + View-Indep** | 191 | 178 | 13 | **93.19%** | 0.9321 |
| **Hard + View-Dep** | 119 | 104 | 15 | **87.39%** | 0.8741 |
| **Total** | **600** | **555** | **45** | **92.50%** | **0.9253** |

### 2.2 Breakdown by Same-Category Instance Multiplicity ($k$)

Accuracy remains above **93%** up to 4 same-category instances in a room, and only
drops to **84.00%** when a scene contains 5–6 identical-category distractors (e.g.,
rows of office chairs or conference-room seating):

| Same-Category Instances in Scene ($k$) | n | Correct | Errors | `Acc@0.25` |
|---|---:|---:|---:|---:|
| **$k = 2$ (`Easy`)** | 290 | 273 | 17 | **94.14%** |
| **$k \in \{3, 4\}$ (`Hard` moderate)** | 235 | 219 | 16 | **93.19%** |
| **$k \in \{5, 6\}$ (`Hard` dense)** | 75 | 63 | 12 | **84.00%** |

### 2.3 Breakdown by Target Object Category ($n \ge 10$)

Across the 18 most frequent object categories in `strat600` (409 / 600 samples), 6
categories achieve **100.0% accuracy** (`window`, `desk`, `box`, `picture`, `lamp`,
`backpack`), and 16 of 18 exceed **90%**:

| Category | n | Correct | `Acc@0.25` | Category | n | Correct | `Acc@0.25` |
|---|---:|---:|---:|---|---:|---:|---:|
| `window` | 27 | 27 | **100.00%** | `chair` | 55 | 51 | **92.73%** |
| `desk` | 18 | 18 | **100.00%** | `couch` | 13 | 12 | **92.31%** |
| `box` | 17 | 17 | **100.00%** | `cabinet` | 24 | 22 | **91.67%** |
| `picture` | 17 | 17 | **100.00%** | `keyboard` | 11 | 10 | **90.91%** |
| `lamp` | 12 | 12 | **100.00%** | `table` | 43 | 39 | **90.70%** |
| `backpack` | 11 | 11 | **100.00%** | `door` | 42 | 38 | **90.48%** |
| `trash can` | 30 | 29 | **96.67%** | `kitchen cabinets` | 10 | 9 | **90.00%** |
| `pillow` | 29 | 28 | **96.55%** | `monitor` | 27 | 24 | **88.89%** |
| `shelf` | 21 | 20 | **95.24%** | `armchair` | 13 | 10 | **76.92%** |

### 2.4 Self-Reported Confidence Calibration

The agent's structured output `confidence` field is strongly calibrated with actual
grounding accuracy:

| Confidence Bucket | Samples ($n$) | Share of Fold | Correct | Empirical `Acc@0.25` |
|---|---:|---:|---:|---:|
| **`confidence >= 0.95`** | 567 | 94.5% | 534 | **94.18%** |
| **`0.90 <= confidence < 0.95`** | 11 | 1.8% | 8 | **72.73%** |
| **`0.80 <= confidence < 0.90`** | 21 | 3.5% | 12 | **57.14%** |
| **`confidence < 0.80`** | 1 | 0.2% | 1 | **100.00%** |

---

## 3. Summary of NR3D Pipeline Upgrades Validated (`v14` $\rightarrow$ `v15`/`v16`)

1. **View-Quality & Diversity-Aware Frame Selection (`src/codex_agent/nr3d/tools/frame_tools.py`):**
   - Replaced naive `sorted(frame_ids)[:3]` selection in `select_by_proposal` with a
     2D bounding-box view-quality score:
     \[
       Q(v) = \bigl(0.65 \cdot \text{AreaFrac}(v) + 0.35 \cdot \text{CenterBonus}(v)\bigr) \times
       \begin{cases} 0.75 & \text{if clipped at image border} \\ 1.0 & \text{otherwise} \end{cases}
     \]
     combined with trajectory camera-pose separation ($\Delta \text{pos} \ge 0.35\text{m}$, $\Delta \theta \ge 0.25\text{rad}$, $\Delta \text{frame} \ge 4$).
   - **`require_all=True` fallback:** When $>2$ candidate proposals cannot fit into a
     single camera frustum, `select_by_proposal` automatically falls back to
     **maximal partial co-visibility** (`co_visible_count_per_frame`) and returns
     `per_proposal_best_frames` + `requested_left_to_right`.
   - **`require_all=False` balanced round-robin:** Allocates top diverse frames
     across every requested candidate proposal ID. On the full 600-sample `strat600`
     fold, offline target miss rate at $k=3$ dropped from **46.5% (279/600) to 0.3% (2/600)** and adjacent duplicate frame batches dropped from **94.2% (565/600) to 0.3% (2/600)**.

2. **3D Virtual Observer Viewpoint Projection & Extended Spatial Relations (`src/codex_agent/nr3d/tools/spatial_tools.py`):**
   - Fixed self-anchor contamination when `anchor_id` is included in `candidate_ids`.
   - Added `in_front_of`, `behind` (co-visible camera-to-object depth voting),
     `between` (group centrality + exact 3D line-segment distance
     `between_segment_distances`), and `opposite`.
   - Added **3D Virtual Observer Viewpoint projection**: computes the horizontal
     sightline from the room centroid (or an explicit `observer_id` via
     `--observer-id`) to the anchor/target centroid:
     \[
       \vec{f} = \frac{\vec{p}_{\text{target}} - \vec{p}_{\text{obs}}}{\|\vec{p}_{\text{target}} - \vec{p}_{\text{obs}}\|}, \quad \vec{r} = (f_y, -f_x)
     \]
     returning `facing_left_to_right_ids` and `facing_front_to_back_ids`. This
     eliminates the tie-break failure mode when candidates on opposite sides of a
     room have `shared_frame_count == 0`.

3. **Batch Contact Sheets & Unoccluded High-Res Crops (`frame_annotation.py` & `bev_tools.py`):**
   - `mark_frame_with_bbox` accepts batch `frame_ids` (up to 4 frames in one call,
     composing a single `contact_sheet`) and automatically generates
     `crops_sheet_path` — high-resolution unoccluded crops of each target from raw
     RGB with a 22% margin before downscaling or drawing thick bounding-box strokes.
   - `view_bev` projects and renders oriented 3D bounding-box footprints on the
     top-down map and returns `highlighted_objects` with 3D/2D coordinates.

4. **Prompt & Playbook Disambiguation (`grounding.py` & `playbook.py`):**
   - Inlined `color=...; nearby=[...]; loc=...` directly on every proposal line in
     the initial catalog (eliminating the 30.8% color truncation rate when notes
     were sliced to 100 chars).
   - Annotated `Proposals by category` with instance counts (`>= 2 instances` vs
     `1 instance — context/anchor only`), preventing false choices when singleton
     context objects share partial category words.
   - Added explicit guards against foreground-occluder VLM note pollution (`33.3%`
     of `strat600`) and enforced external-facing perspective rules (`"facing the X"`
     means standing in the room looking at `X`) plus anti-refusal (`-1` ban).

---

## 4. Run Health, Latency, & Token Telemetry

| Metric | `v14 gpt-5.4` (`workers=10`) | `v14 gemini-3.8-flash` (`workers=10`) | **`v16 gemini-3.8-flash` (`workers=24`)** |
|---|---:|---:|---:|
| Sample records | 600 / 600 | 600 / 600 | **600 / 600** |
| Final status | 599 completed, 1 refusal | 600 completed, 0 failed | **600 completed, 0 failed** |
| Target-absent (`-1`) selections | 1 | >0 | **0 / 600** |
| Mean 3D IoU | 0.8503 | 0.9159 | **0.9253** |
| Prompt cache hit rate | 0.9783 | 0.3133 | **0.3233** |
| Mean cache ratio | 0.8794 | 0.2259 | **0.2432** |
| Turn latency (median / p90 / max) | 89.0 s / 172.0 s / 583.4 s | 79.0 s / 217.7 s / 768.0 s | **81.9 s / 263.2 s / 915.7 s** |

### Per-Slice Latency & Token Consumption (`v16`)

| Slice | n | Median Latency (s) | Mean Latency (s) | p90 Latency (s) | Median Input Tokens | Median Output Tokens |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | **81.9 s** | 126.0 s | 263.2 s | 85,843 | 1,450 |
| **Easy** | 290 | **78.3 s** | 109.4 s | 232.8 s | 79,815 | 1,292 |
| **Hard** | 310 | **90.6 s** | 141.5 s | 323.5 s | 97,552 | 1,739 |
| **View-Dep** | 211 | **97.8 s** | 160.5 s | 393.9 s | 106,852 | 1,991 |
| **View-Indep** | 389 | **76.6 s** | 107.3 s | 225.9 s | 81,440 | 1,317 |

---

## 5. Representative Recovered Failure Cases

| Sample ID | Slice | Query | GT | Prior Failure Mode | **`v16` Outcome & Reasoning Summary** |
|---|---|---|---|---|---|
| `scannet/scene0435_00::21::13508` | `Easy+VI` | *"The door to the bathroom."* | `#21` | `v11` returned `-1` (`target_absent`), claiming the bathroom door was absent from the proposal pool. | **Selected `#21` (`IoU = 1.0`, `conf = 0.95`)** — verified bathroom door `#21` via BEV + first-person frame inspection without false refusal. |
| `scannet/scene0221_00::51::24914` | `Easy+VI` | *"The towel in the furthest corner of the room"* | `#51` | `v8 repeat` returned `-1` because `#51`'s offline VLM note described a wall-mounted hairdryer cord. | **Selected `#51` (`IoU = 1.0`, `conf = 0.95`)** — discounted noisy VLM note and confirmed `#51` is in the furthest alcove corner. |
| `scannet/scene0169_00::26::28815` | `Easy+VI` | *"The smaller cabinet."* | `#26` | `v11` returned `-1` because `#26`'s offline VLM note described an occluding black drafting chair. | **Selected `#26` (`IoU = 1.0`, `conf = 0.98`)** — compared 3D dimensions of `#26` vs `#32` and verified via unoccluded crop sheet. |
| `scannet/scene0063_00::10::1029` | `Hard+VI` | *"The larger of the cabinets."* | `#10` | Prior runs confused singleton `cabinet #15` with the two `kitchen cabinet` instances (`#10`, `#16`). | **Selected `#10` (`IoU = 1.0`, `conf = 0.98`)** — multi-instance header marked `cabinet` as singleton context and `kitchen cabinet (2 instances)` as the target pool. |
| `scannet/scene0435_00::60::27767` | `Hard+VD` | *"Facing the foot of the bed, the suitcase on the left."* | `#60` | Inverted intrinsic vs. external observer left/right and selected `#34`. | **Selected `#60` (`IoU = 1.0`, `conf = 0.95`)** — external-facing viewpoint projection placed `#60` on the viewer's left when facing the foot of the bed. |

---

## 6. Taxonomy of Remaining 45 Errors (`555 / 600` Correct, `7.50%` Error Rate)

A full audit of all 45 incorrect predictions on `strat600` shows that **100% (45 / 45)**
returned a valid proposal ID (`0` crashes, `0` `-1` refusals), falling into four
distinct categories:

### Category A — Fine-Grained ScanNet vs. Human-Utterance Lexical Split (`11 / 45` errors, `1.83%` of fold)

In Nr3D, human annotators write a generic noun (`"chair"`, `"cabinet"`, `"table"`,
`"shelf"`, `"door"`, `"recycling bin"`), while the ScanNet scene contains multiple
fine-grained category labels that **both** have $\ge 2$ instances. Because the
evaluator does not leak the ground-truth ScanNet stimulus class name to the agent,
when an instance of the sister category satisfies the spatial/visual clue better
than the GT class, the agent selects the sister-category object:

- `scannet/scene0606_00::21::18870` & `::3157` (*"chair closest to the clock on the wall"* / *"This chair is closer to the wall clock"*): GT `#21` is labeled `sofa chair` (distance ~2.4m to clock `#9`), whereas dining chair `#28` (labeled `chair`, 4 instances) is physically closer to the wall clock.
- `scannet/scene0591_00::23::33573` (*"The cabinet closest to the window"*) & `::6::38064` (*"cabinet closest to the black sofa"*): Scene has both `file cabinet` (`#6`, `#23`) and `cabinet` (`#4`, `#30`, `#31`); `#30` and `#4` are physically closer to the window and black sofa, respectively.
- `scannet/scene0187_00::10::17549` (*"the table closest to the small square beige stool"*): GT `#10` is `coffee table`, while `#8` (`table`) sits immediately beside the beige stool `#2` (`1.08 m`).
- `scannet/scene0208_00::1::5894` & `::20428` (*"bookcase in the corner next to the long set of shelves"*): GT `#1` is `bookshelf`, while `#15` is `book rack` tucked in the corner next to shelf `#106`.
- `scannet/scene0164_00::14::23053` & `scannet/scene0149_00::21::1743`: `kitchen cabinet` (singular) vs `kitchen cabinets` (plural) label split within the same kitchen.
- `scannet/scene0050_00::32::30122`: `doors` (double French doors `#32`) vs `door` (single door `#10`).
- `scannet/scene0412_00::0::10547`: `recycling bin` (`#0`) vs `trash can` (`#13`, which is a grey bin with a green recycling lid in front of `#14`).

### Category B — Referent-Target Mismatch in Human Utterance (`2 / 45` errors, `0.33%` of fold)

- `scannet/scene0690_00::14::34892`: Query is *"Standing in the middle of the room looking at both lamps, it's the lamp on the right."* The speaker explicitly asked for **the lamp on the right** (`#10`, `lamp`), which the agent selected with `confidence = 0.99`, whereas the underlying Nr3D target box was the end `table` (`#14`) supporting that lamp.
- `scannet/scene0187_00::8::34273`: Query is *"There is a green stool next to this square table"* (declarative statement without an explicit head noun); the agent selected the green stool `#5` instead of the square table `#8`.

### Category C — Complex View-Dependent Frame-of-Reference Ambiguity (`16 / 45` errors, `2.67%` of fold)

- **Dynamic entry path / walking trajectory (`"As you enter the room..."`, `"With the bed on the far wall..."`):** e.g., `scannet/scene0644_00::43::2885`, `scannet/scene0221_00::17::39165`, `scannet/scene0356_00::23::23827`.
- **2D image-plane corner terminology (`"upper right corner"`, `"lower left corner"` when facing an anchor):** In `scannet/scene0616_00::13::389` (*"Facing the fireplace, the chair in the upper right corner"*) and `scannet/scene0549_00::11::3698` (*"Facing the windows, the arm chair in the lower left corner"*), annotators used `"upper/lower"` on the 3D web viewer canvas to mean `"farther/closer"` along the depth axis.

### Category D — Dense / Symmetric Multi-Instance Disambiguation (`16 / 45` errors, `2.67%` of fold)

- Circular or symmetric arrangements around tables (`scannet/scene0314_00::4::24825` *"The chair opposite of the laptop"* around a round table; `scannet/scene0423_00::3::12539` *"3 chairs that are aligned, the chair is in the middle"* along a curved planter wall).
- Subtle visual wear or lighting-dependent attributes (`scannet/scene0423_00::4::38676` *"chair with the second most damage to the seat cushion"*; `scannet/scene0432_00::2::29554` daylight washout near a window).

---

## 7. Run Metadata & Exact Reproduction Commands

- **Branch:** `feat/upgrade_keyframe_selector`
- **Head commit at launch:** `8aa55a9` (code commits `caebb4a` + `898b3fa`; clean working tree `head == executed code`).
- **Fold:** `/home/ysh/Datasets/nr3d/processed_v1/folds/strat600_sample_ids.json` (canonical `strat600`: 600 samples across 119 scenes).
- **Data root:** `/home/ysh/Datasets/nr3d/processed_v1/scannet`
- **Pack:** `pack_nr3d_v9_catalog_first`
- **Model:** `gemini-3.8-flash` (`--reasoning-effort provider-default`, `--tools`).
- **Concurrency:** Main pass at `--workers 24` in `tmux` session `nr3d_v15_strat600_20261104` (`597 / 600` completed in ~71 min), followed by a 3-sample low-concurrency recovery pass at `--workers 3` (`600 / 600` completed in ~2 min).

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

## 8. Durable Artifacts & SQLite Ingestion

- **Committed leaderboard summary:** `docs/benchmark/nr3d/assets/v16_strat600_tools_upgrade_20261104_leaderboard.json`
- **Committed full 600-sample summary:** `docs/benchmark/nr3d/assets/v16_strat600_tools_upgrade_20261104_summary.json`
- **SQLite database (`docs/benchmark/nr3d/runs.sqlite`):** `run_id = v16_strat600_tools_upgrade_20261104` (600 sample rows ingested via `scripts/ingest_nr3d_run.py`).
