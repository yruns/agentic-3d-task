# v12: Single-turn Codex macro-tool architecture, 85-category vocabulary pruning, and GPU-resident perception on `prefix5` (`gemini-3.8-flash`), 2026-11-05

Evaluation of the **v12 ALFRED architecture and performance overhaul** (`8d6c739` +
`94b1c0b` + `c0947b5` + `a58f909`) on the **5-task `prefix5` cohort** (indices
`0..4` of `folds/pilot15.json`), using **`gemini-3.8-flash`** via `codex_adapter`.

---

## 1. Summary of architectural & performance changes (`v10` $\rightarrow$ `v12`)

1. **Single-Turn Codex Agent Loop + Stateful `EpisodeToolServer` + `SceneDelta` (`execution.py`, `episode_server.py`, `delta.py`, `macro_controller.py`):**
   - Replaced the stateless 60-turn reactive JSON loop with **one continuous Codex
     Agent SDK turn per episode**.
   - Before Turn 0, `EpisodeExecutor` executes an automatic 360-degree scan at
     $30^\circ$ pitch and attaches the initial annotated RGB + top-down BEV map
     alongside the inlined 3D Object Catalog and Scene Region Layout.
   - Removed atomic primitive actions (`MoveAhead`, `RotateLeft/Right`,
     `LookUp/Down`) from the model's decision space; exposed 10 high-level CLI
     tools (`navigate_to`, `explore_frontier`, `scan_surroundings`,
     `interact_object`, `list_objects`, `keyframe_selector`, `view_crop`,
     `view_frame`, `view_bev`, `finish_episode`) returning compact incremental
     `SceneDelta` diffs against a `RevisionWatermark`.
   - Separated unsubtracted `current_raw_detection_masks` (for `OpenObject` /
     `CloseObject` on containers) from child-subtracted `current_detection_masks`
     (for 3D point clouds and `PickupObject`).
2. **Detector Vocabulary Pruned from 114 to 85 Task-Relevant Categories (`vocabulary.py`):**
   - Full audit of all **7,080 unique trajectories (22,664 annotations)** across
     `train` (6,574), `valid_seen` (251), and `valid_unseen` (255) showed that
     only **85 categories** (53 pickup targets + 5 `*Sliced` forms + 24 stationary
     receptacles + 2 lamps + `Faucet`) ever appear in PDDL goals or expert actions.
   - Excluded the **29 zero-occurrence background/architectural/duplicate-shell
     categories** (`Window`, `Blinds`, `Curtains`, `Mirror`, `Painting`, `Poster`,
     `ShowerDoor`, `ShowerGlass`, `LightSwitch`, `StoveKnob`, `Chair`, `Footstool`,
     `TVStand`, `Television`, `Sink` [duplicate of `SinkBasin`], `Bathtub`
     [duplicate of `BathtubBasin`], etc.), reducing GroundingDINO passes from
     **5 batches/frame to 3 batches/frame (`caption_batch_size = 29`)** and
     eliminating background mask hollowing.
3. **GPU-Resident Perception Models & AABB/OBB Caching (`model_runtime.py`, `object_mapping.py`):**
   - Kept `DA3Metric-Large`, `GroundingDINO`, `SAM1 ViT-H`, and `OpenCLIP ViT-H-14`
     resident on the assigned 24 GB RTX 4090 GPU (~11.24 GiB peak VRAM), avoiding
     17 GB/frame of CPU$\leftrightarrow$GPU PCIe transfers and 4 synchronous
     `torch.cuda.empty_cache()` calls per frame.
   - Added $O(1)$ axis-aligned bounding-box (AABB) rejection and `_ENVELOPE_CACHE`
     before `ConvexHull` fitting in `oriented_envelopes_overlap()`.

---

## 2. Speedup Breakdown (`v11` Before Optimization vs. `v12` After Optimization)

| Stage / Metric | `v11` (114 classes, CPU offload, no AABB gate) | **`v12` (85 classes, GPU-resident, AABB+OBB cache)** | Speedup |
|---|---:|---:|---:|
| **Total Perception Inference / Frame** | **3.48 s / frame** | **0.72 – 0.78 s / frame** | **4.6$\times$ faster** |
| — `DA3Metric-Large` depth | 0.42 s / frame | **0.05 s / frame** | **8.4$\times$ faster** |
| — `GroundingDINO` detection | 0.62 s / frame (5 batches) | **0.21 s / frame** (3 batches) | **3.0$\times$ faster** |
| — `SAM 1 ViT-H` segmentation | 0.96 s / frame | **0.28 s / frame** | **3.4$\times$ faster** |
| — `OpenCLIP ViT-H-14` features | 1.44 s / frame | **0.17 s / frame** | **8.5$\times$ faster** |
| **Macro Step Execution (Perception + 3D Mapping + FMM + Sim)** | **8.0 – 19.2 s / step** (spikes to **73.6 s** at frame 40) | **1.82 – 2.25 s / step** (**0 spikes** even at 77 frames) | **4.5$\times$ – 30$\times$ faster** |

---

## 3. Official Results on `prefix5` (`pilot15` indices `0..4`)

| Run ID | Shard | Task Goal | Completed / Total Goals (`GC`) | Success (`SR`) | Primitive Actions / Failed | High-Level Tool Calls | Wall Time (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| `ALFRED_V12_R2_20261105_pilot15_i00` | `i00` | *Carry a purple pillow while turning on a lamp* | **2 / 2** | **True (1/1)** | 76 / 1 | 22 | 434.5 s |
| `ALFRED_V12_R3_20261105_pilot15_i01` | `i01` | *Put the spray bottle on the back of the toilet* | 0 / 1 | False | 34 / 4 | 23 | 400.6 s |
| `ALFRED_V12_R2_20261105_pilot15_i02` | `i02` | *Place a chilled martini glass with a fork in it on the table* | 0 / 3 | False | 30 / 4 | 26 | 395.6 s |
| `ALFRED_V12_R2_20261105_pilot15_i03` | `i03` | *Rinse off a ladle and move it to the table* | **1 / 3** | False | 18 / 1 | 24 | 384.5 s |
| `ALFRED_V12_R2_20261105_pilot15_i04` | `i04` | *Cool off a white plate* | 0 / 3 | False | 76 / 8 | 27 | 611.2 s |

### Cohort Comparison on the Same 5 Tasks (`prefix5`)

| Configuration | Model | Scored | **SR** | **GC** | **PLWSR** | **PLWGC** | Total Actions / Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| **`v10` (Old reactive per-step loop + 114 classes)** | `gpt-5.4-2026-03-05` | 5 / 5 | 0.00% (0/5) | 8.33% (1/12) | 0.00% | 3.78% | 317 / 32 |
| **`v11` (Single-turn macro tools + 114 classes)** | `gemini-3.8-flash` | 5 / 5 | 0.00% (0/5) | 16.67% (2/12) | 0.00% | 9.52% | 205 / 13 |
| **`v12` (Single-turn macro tools + 85 classes + GPU-resident + AABB cache)** | **`gemini-3.8-flash`** | **5 / 5** | **20.00% (1/5)** | **25.00% (3/12)** | **7.68%** | **14.34%** | **234 / 18** |
| *(Reference: `v8` domain-trained FILM Mask R-CNN)* | `gpt-5.4-2026-03-05` | 5 / 5 | 20.00% (1/5) | 25.00% (3/12) | 15.71% | 31.44% | 164 |

- **First full task success (`i00`, `GC = 2/2, SR = 100%`) with zero-shot open-vocabulary online mapping:** In `i00`, the agent explored the room with `explore_frontier`, navigated to the bed and armchair, used `view_crop` to verify the purple pillow (`#50`), picked it up (`PickupObject(#50) succeeded`), navigated to the desk lamp (`#7`), turned it on while holding the pillow (`ToggleObjectOn(#7) succeeded`), and called `finish_episode`.
- **Matches the historical `v8` FILM-trained baseline on both `SR` (`20.00%`, 1/5) and `GC` (`25.00%`, 3/12)** while improving over `v10` from **0% $\rightarrow$ 20% SR** and **8.33% $\rightarrow$ 25.00% GC (+16.67 pp)**.
