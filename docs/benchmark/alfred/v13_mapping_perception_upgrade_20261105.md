# ALFRED v13: Incremental Dynamic Mapping, GPU-Batched Perception & Ground-Plane Calibration

- **Date:** 2026-11-05
- **Branch:** `feat/alfred-benchmark`
- **Commits:** `e0bf8b5` (`feat(alfred): upgrade incremental dynamic mapping, perception speed, and floor calibration`) & `abd557b` (`fix(alfred): preserve visited footprint in spatial map and use raw mask for PutObject`)
- **Fold:** `docs/benchmark/alfred/folds/pilot15.json` (`prefix5`: shards `0..4`, all `valid_seen`)
- **Model:** `gemini-3.8-flash` (`codex_adapter`, `reasoning_effort = "provider-default"`)
- **Scene Config:** `configs/alfred/scene.toml` (`85` task-relevant categories, `caption_batch_size = 29`, `box_threshold = 0.25`, `text_threshold = 0.25`, `association_distance_m = 0.12`, `envelope_margin_m = 0.10`, `min_clip_similarity = 0.185`, `erode_mask_pixels = 1`, `depth_trim_quantile = 0.15`, `same_category_bonus = 0.20`)

---

## 1. Root-Cause Audit of `v12` Perception & Dynamic Mapping

Offline replay and ground-truth alignment across the 5 `v12` episodes (239 frames, 4,052 instance masks, 1,156 created object nodes) exposed four structural bottlenecks in the perception and online 3D mapping stack:

1. **Severe 3D Object Fragmentation (`0.78%` merge rate across 1,156 nodes):**
   - `point_overlap()` used `voxel_size_m = 0.025 m` (`2.5 cm` LiDAR-grade radius) with `association_threshold = 1.2` and `merge_overlap_threshold = 0.70`, while `oriented_envelopes_overlap()` enforced zero normal margin.
   - Under `DA3Metric-Large` monocular depth (`RMSE = 0.314 m`), 2.5D surface shells shift by `5–18 cm` across camera viewpoints and pitches. Consequently, 1 physical `Bed` in `i00` fragmented into **19 IDs**, 1 `Faucet` in `i01` into **17 IDs**, and 1 `Fridge` in `i02` into **7 IDs**.
   - Additionally, bilinear depth upsampling (`cv2.INTER_LINEAR`) created foreground-to-wall depth veil points along SAM mask boundaries, pulling small object centroids `20–50 cm` toward background walls.

2. **Detector Confidence Bug, `*Sliced` Modifier Hijacking & Threshold Mismatch:**
   - `decode_grounded_boxes()` assigned `logits[index].max()` (over all 256 BERT tokens, including `.` and `[CLS]`) as box confidence and scored multi-token phrases using `sub_tokens.max()`. Consequently, any weak activation on `"sliced"` or shared noun stems caused `*Sliced` classes (`TomatoSliced`, `PotatoSliced`, `AppleSliced`) to hijack ordinary objects (`41%–53%` of retained detections belonged to categories absent from the room).
   - `scene.toml` set `box_threshold = 0.20` while `MappingConfig` set `min_mask_confidence = 0.25`, wasting **26%–30% of SAM1 ViT-H and OpenCLIP ViT-H-14 compute** on `[0.20, 0.25)` boxes that were immediately rejected upon entering `mapping_masks()`.

3. **Monocular Depth Scale Bias Turning 80%–100% of Floor into Obstacle Walls:**
   - With camera height $z_{\text{cam}} = 1.576\text{ m}$ and `obstacle_min_height_m = 0.12 m`, a mere `-8%` to `-18%` depth underestimation by `DA3Metric-Large` lifts projected floor pixels to $z = 1.576 \times (1 - \hat{d}/d) \in [+0.13\text{ m}, +0.28\text{ m}]$, causing `SpatialMap._ray_evidence()` to mark **79%–100% of open floor** ahead of the agent as `occupied` obstacles.
   - Furthermore, CPU ray marching (`SpatialMap.update`) took **`341–509 ms/frame`** due to 80 per-frame 2D `np.unique(..., axis=0)` calls over `3.2M` sampled 3D ray points.

4. **GPU Perception & Disk I/O Bottlenecks:**
   - `GroundingDINO` ran the Swin-T image backbone 3 separate times per frame for the 3 fixed caption batches and re-tokenized static class prompts on every frame (`220 ms/frame`).
   - `SAM1 ViT-H` (`285 ms/frame`) and `OpenCLIP ViT-H-14` (`180–273 ms/frame`) ran in FP32 with serial box loops.
   - `validate_prediction_artifacts()` was invoked both inside `perception_worker` and again in `ScenePerception.predict`, and `np.savez_compressed` spent `40–55 ms/frame` running synchronous `zlib` compression on transient arrays.

---

## 2. Architectural & Algorithmic Upgrades in `v13`

1. **Batched `GroundingDINO`, `bfloat16` `SAM1`/`OpenCLIP`, and Single-Pass Validation (`vision_models.py`, `perception_worker.py`):**
   - Pre-encoded and cached the 3 fixed caption batches in `_cached_text_dict`; `GroundingDetector.predict()` now runs the Swin-T image backbone **once per frame**, expands multi-scale features to `batch_size = 3`, and evaluates only the final decoder layer.
   - Enabled `torch.autocast(bfloat16)` and batched `predict_torch()` in `PromptedSegmenter` (`SAM1 ViT-H`) and `batch_size = 32` `bfloat16` encoding in `ClipEncoder` (`OpenCLIP ViT-H-14`).
   - Aligned `box_threshold = text_threshold = 0.25`, enforced all-token support in `decode_grounded_boxes(..., require_all_span_tokens=True)`, and added OpenCLIP image-text cosine gating (`min_clip_similarity = 0.185`) in `mapping_masks()`.

2. **Monocular-Scale 3D Association, Mask Erosion & Depth Quantile Trimming (`object_mapping.py`, `memory.py`):**
   - Added 1-pixel morphological mask erosion (`erode_mask_pixels = 1`) and `[p02, p85]` depth quantile trimming (`depth_trim_quantile = 0.15`) in `_project_detections()`, plus `cv2.INTER_NEAREST` depth upsampling in `DA3MetricDepth`.
   - Decoupled `association_distance_m = 0.12 m` and `envelope_margin_m = 0.10 m` from `voxel_size_m = 0.025 m`, added `same_category_bonus = 0.20`, and cached both `cKDTree` (`_KDTREE_CACHE`) and AABB/OBB envelopes (`_ENVELOPE_CACHE`).
   - Added category consistency checks to post-interaction reacquisition (`!node.geometry_valid`) and preserved raw SAM masks for `PutObject` on containers/surfaces so child-subtracted masks do not miss `SinkBasin` or open appliance interiors.

3. **Adaptive Ground-Plane Calibration, Visited Footprint Protection & Vectorized Raycasting (`spatial_map.py`):**
   - `SpatialMap._ray_evidence()` estimates local flat floor elevation (`floor_offset`) from the low-height surface distribution (`p50 - p10 <= 0.09 m`) so range underestimation does not turn flat floors into obstacle walls.
   - Traversed robot footprints (`_visited_footprint`) are protected from subsequent noisy far-range obstacle votes unless a physical collision (`record_collision`) occurs.
   - Packed 2D grid coordinates into 64-bit integers (`_pack` / `_unpack_set`) and deduplicated ray endpoints by `(cell_x, cell_y, height_bin)` before ray marching, cutting `SpatialMap.update()` from **`341–509 ms/frame` down to `13–22 ms/frame` (~20$\times$ faster)**.

---

## 3. Quantitative Comparison: `v11` vs. `v12` vs. `v13` (`prefix5`)

### 3.1 Latency & Perception Breakdown

| Metric / Stage | `v11` (114 cats, CPU offload) | `v12` (85 cats, GPU resident) | **`v13` (Batched `bf16` + Fast Raycast)** | Speedup (`v13` vs `v12` / `v11`) |
|---|---:|---:|---:|---:|
| **Total Macro Step Time** | `8.2 – 73.6 s / step` | `1.86 – 2.96 s / step` | **`0.85 – 1.13 s / step`** | **2.4$\times$ / >10$\times$ faster** |
| **Perception RPC / Frame** | `3,480 ms / frame` | `785 ms / frame` | **`398 ms / frame`** | **2.0$\times$ / 8.7$\times$ faster** |
| — `DA3Metric-Large` depth | `420 ms` | `48 ms` | **`47 ms`** | `1.0x` / `8.9x` |
| — `GroundingDINO` (3 batches) | `1,150 ms` | `222 ms` | **`114 ms`** | **1.95$\times$ / 10.1$\times$** |
| — `SAM1 ViT-H` segmentation | `1,120 ms` | `292 ms` | **`142 ms`** | **2.06$\times$ / 7.9$\times$** |
| — `OpenCLIP ViT-H-14` features | `790 ms` | `212 ms` | **`81 ms`** | **2.62$\times$ / 9.8$\times$** |
| **CPU `Memory.integrate()` / Frame** | `>1,500 ms` | `460 ms` | **`135 ms`** | **3.4$\times$ / >11$\times$ faster** |
| — `SpatialMap.update()` raycast | `~450 ms` | `425 ms` | **`18 ms`** | **23.6$\times$ faster** |

### 3.2 3D Object Fragmentation, Precision & Free-Space Mapping

| Episode (`valid_seen`) | `v12` Memory Objects per 50 Frames (Raw) | **`v13` Memory Objects per 50 Frames (Raw)** | Fragmentation Reduction | `v12` $\rightarrow$ `v13` Room Category Precision | `v12` $\rightarrow$ `v13` Free / Occupied Cell Ratio |
|---|---:|---:|---:|---:|---:|
| `i00` (`look_at_obj_in_light`, FloorPlan302) | `219.5` (`338` over 77f) | **`47.0`** (`94` over 100f) | **-78.6%** | `47.6%` $\rightarrow$ `46.8%` | `0.23` (`1100/4760`) $\rightarrow$ **`0.30`** (`1436/4760`) |
| `i01` (`pick_and_place_simple`, FloorPlan415) | `228.6` (`160` over 35f) | **`98.5`** (`128` over 65f) | **-56.9%** | `56.2%` $\rightarrow$ **`60.9%`** | `0.50` (`1824/3618`) $\rightarrow$ `0.42` (`1839/4334`) |
| `i02` (`pick_and_place_with_movable_recep`, FloorPlan23) | `233.9` (`145` over 31f) | **`105.1`** (`82` over 39f) | **-55.1%** | `59.3%` $\rightarrow$ **`62.2%`** | `0.33` (`1352/4050`) $\rightarrow$ **`1.14`** (`3280/2870`) |
| `i03` (`pick_clean_then_place_in_recep`, FloorPlan4) | `305.3` (`116` over 19f) | **`89.8`** (`79` over 44f) | **-70.6%** | `53.4%` $\rightarrow$ **`58.2%`** | `0.11` (`372/3330`) $\rightarrow$ **`0.87`** (`1720/1979`) |
| `i04` (`pick_cool_then_place_in_recep`, FloorPlan1) | `251.9` (`388` over 77f) | **`76.3`** (`171` over 112f) | **-69.7%** | `60.6%` $\rightarrow$ **`65.5%`** | `0.72` (`3640/5041`) $\rightarrow$ **`0.86`** (`4228/4906`) |
| **Mean across 5 Episodes** | **`247.8 / 50f`** | **`83.3 / 50f`** | **-66.4% (-3.0$\times$)** | `55.4%` $\rightarrow$ **`60.7%`** | `0.38` $\rightarrow$ **`0.72` (+89%)** |

---

## 4. Per-Episode Execution & Remaining Behavioral Bottlenecks

| Shard | Episode ID | Task Type | `v13` R1 / R2 Outcome | Key Behavioral Finding |
|---:|---|---|---|---|
| `i00` | `look_at_obj_in_light-Pillow-None-DeskLamp-302` | `look_at_obj_in_light` | **GC `1/2` (50%)**, 93–99 actions | Picked up the purple pillow (`GC 1/2`), navigated to `DeskLamp #7`, and retained clear corridor footprints in R2; missed `ToggleObjectOn` because `DeskLamp` was not re-detected at close-up `60°` pitch. |
| `i01` | `pick_and_place_simple-SprayBottle-None-Cabinet-415` | `pick_and_place_simple` | **GC `0/1` (0%)**, 64 actions | Discovered `SprayBottle #101` and navigated to `0.81 m`, but the small bottle on the counter edge lacked a close-up SAM mask after framing. |
| `i02` | `pick_and_place_with_movable_recep-Ladle-Pan-SinkBasin-23` | `pick_and_place_with_movable_recep` | **GC `0/3` (0%)**, 38 actions | High-level instruction says *"Put a fork in a glass then put them in the sink"* (while PDDL target is `Ladle` + `Pan` + `SinkBasin` on this annotation); agent searched for `Cup`/`Fork`. |
| `i03` | `pick_clean_then_place_in_recep-Ladle-None-DiningTable-4` | `pick_clean_then_place_in_recep` | **Full workflow executed** (`Pickup(Ladle)` $\rightarrow$ `Put(Sink)` $\rightarrow$ `ToggleOn/Off(Faucet)` $\rightarrow$ `Put(DiningTable)`) | In R1, the agent executed all 4 semantic stages of `clean_in_sink` + `PutObject(DiningTable)`. Exposed two simulator-boundary bugs fixed in `abd557b`: (1) subtracted `SinkBasin` mask hit the outer `Sink` rim instead of inner `SinkBasin`, and (2) post-interaction reacquisition without category gating swapped `Ladle #30` with `ButterKnife` inside the sink. |
| `i04` | `pick_cool_then_place_in_recep-Plate-None-CounterTop-1` | `pick_cool_then_place_in_recep` | **GC `1/3` (33%)** in R2 (`OpenObject(Fridge)` + `CloseObject(Fridge)` + `CounterTop` navigation) | In R1, agent picked up `Plate #87`, opened `Fridge #41`, and attempted `PutObject(Fridge)`; fixed in `abd557b` by serving raw unsubtracted SAM masks and pitching up (`LookUp`) to expose upper refrigerator shelves. |

---

## 5. Artifact Locations

- `outputs/alfred/ALFRED_V13_20261105_pilot15_i00` .. `i04` (Commit `e0bf8b5`)
- `outputs/alfred/ALFRED_V13_R2_20261105_pilot15_i00`, `i03`, `i04` (Commit `abd557b`)
