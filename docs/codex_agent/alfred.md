# ALFRED: online OpenEQA scene mapping, single-turn Codex loop, and high-level macro tools

ALFRED uses our OpenEQA object–view mapping method incrementally inside a
**single interactive Codex Agent SDK turn per episode**:

1. **Map-First Turn 0 Initialization:** Before the Codex turn begins,
   `EpisodeExecutor` automatically performs an initial 360-degree scan at a
   $30^\circ$ downward pitch, integrating the surrounding room geometry, 3D
   objects, and 5 cm traversability grid, and attaches the initial annotated
   first-person RGB image + top-down BEV map alongside the inlined 3D Object
   Catalog and Scene Region Layout.
2. **High-Level Macro Tool Loop (`codex_agent.alfred.tools`):** Codex never
   issues low-level atomic movements (`MoveAhead`, `RotateLeft`, `LookUp`,
   `LookDown`) directly. Instead, Codex invokes semantic CLI tools
   (`navigate_to`, `explore_frontier`, `scan_surroundings`, `interact_object`,
   `list_objects`, `keyframe_selector`, `view_crop`, `view_frame`, `view_bev`,
   `finish_episode`) which communicate over loopback HTTP with an in-process
   `EpisodeToolServer` owned by `EpisodeExecutor`.
3. **Closed-Loop FMM Execution & Incremental `SceneDelta` Updates:** When Codex
   invokes a macro tool (e.g., `navigate_to(object_id=14)` or
   `interact_object(action="clean_in_sink", object_id=14)`),
   `MacroActionController` steps AI2-THOR closed-loop, runs neural perception
   and 3D mapping on every intermediate frame (persisting all raw frames, depth
   maps, masks, and CLIP features to disk), and returns **only a compact
   incremental `SceneDelta` JSON** (`delta.py`) computed against the previous
   `RevisionWatermark`.

The implementation replaces the earlier FILM perception and per-step reactive
JSON polling loop. Historical code/configuration and 50 official scores remain
archived in [v8](../benchmark/alfred/v8_pilot50_complete_20261105.md) and
[v10](../benchmark/alfred/v10_prefix5_20261105.md).

---

## Inputs and information boundary

The actor receives the high-level language instruction, initial/requested RGB
views, gridded BEV maps, yaw/pitch proprioception, successful-action odometry
and action-success feedback. H+L is an explicit option. Each episode turn runs
inside an isolated temporary workspace (`/tmp/alfred_actor_ws_*`) containing
only `session_endpoint.json` and published tool images so the agent cannot
inspect parent run directories or dataset files on disk.
The pose setting is unchanged: successful `MoveAhead` advances 0.25 m, the
standing camera is 1.576 m high, and calibrated vertical FOV is 60 degrees.

Simulator object identities, world positions, reachable-position lists, true
masks/depth, expert plans, task types, PDDL parameters and terminal goal conditions
remain behind the evaluator boundary. Action error strings redact coordinate-bearing
object IDs. A separate train-only depth calibration process reads true depth
offline; it is never an actor input.
The existing official simulator and scorer stay pinned to AI2-THOR 2.1.0, build
`201909061227`.

---

## Why the detector vocabulary is pruned from 114 to 85 task-relevant categories

### 1. Where the original 114 categories came from
The initial 114-class list (`OBJECTS_WSLICED` in FILM's `alfworld_constants.py`)
was formed by taking all **108 Unity GameObject modeling categories** from
ALFRED's `gen/constants.py` (`OBJECTS`), adding the **5 sliced food forms**
(`AppleSliced`, `BreadSliced`, `LettuceSliced`, `PotatoSliced`, `TomatoSliced`),
and adding **`Faucet`**.

### 2. Full dataset audit across all 7,080 ALFRED trajectories (`train` + `valid_seen` + `valid_unseen`)
An exhaustive scan of `pddl_params` (`object_target`, `mrecep_target`,
`parent_target`, `toggle_target`, `object_sliced`) and `plan.low_actions` across
all **7,080 unique trajectories (22,664 annotations)** in `json_2.1.0`
establishes the exact set of categories that ever participate in any ALFRED task:

| Split | Unique trajectories / annotations | `object_target` (base) | `mrecep_target` | `parent_target` | `toggle_target` | Union of PDDL targets (incl. `*Sliced`) | Union of all interacted categories (incl. `Faucet`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train` | 6,574 / 21,023 | 53 | 7 | 24 | 2 | 84 | **85** |
| `valid_seen` | 251 / 820 | 49 | 7 | 24 | 2 | 80 | 81 |
| `valid_unseen` | 255 / 821 | 42 | 6 | 15 | 2 | 64 | 65 |
| **All Splits Union** | **7,080 / 22,664** | **53** | **7** | **24** | **2** | **84** | **85** |

Every category appearing in `valid_seen` or `valid_unseen` is a strict subset of
the **85 task-relevant categories** in `train` (53 pickup targets + 5 `*Sliced`
forms + 24 stationary receptacles + 2 lamps + `Faucet`).

### 3. Why the 29 non-task background/duplicate-shell categories are excluded (`EXCLUDED_BACKGROUND_CATEGORIES`)
The remaining **29 categories** in the raw 114-class list have **zero
occurrences** across all 7,080 trajectories:
- **Architectural & wall background (10):** `Blinds`, `Curtains`, `Window`,
  `Mirror`, `Painting`, `Poster`, `ShowerDoor`, `ShowerGlass`, `LightSwitch`,
  `StoveKnob`.
- **Non-target background furniture (7):** `Chair` (tasks only use `ArmChair`),
  `Footstool` (tasks only use `Ottoman`), `TVStand`, `Television`, `HousePlant`,
  `LaundryHamper`, `LaundryHamperLid`.
- **Duplicate outer fixture shells (2):** `Sink` (tasks exclusively target
  `SinkBasin`) and `Bathtub` (tasks exclusively target `BathtubBasin`).
- **Unused props/variants (10):** `Boots`, `HandTowelHolder`, `PaperTowel`,
  `PaperTowelRoll`, `ScrubBrush`, `TeddyBear`, `Toaster`, `ToiletPaperRoll`
  (tasks use `ToiletPaper`), `Towel`, `TowelHolder`.

*(Note: FILM's own detector `alfworld_constants.py:1202` similarly strips
`Blinds`, `Boots`, `Chair`, `Curtains`, `Footstool`, `Mirror`, `LightSwitch`,
`Painting`, `Poster`, `ShowerGlass`, and `Window` from `OBJECTS_DETECTOR`.)*

When queried with GroundingDINO at `box_threshold = 0.20`, those 29 categories
caused three concrete failure modes observed in `v10` and `v11`:
1. **Severe false-positive box clutter & mask hollowing:** Large background
   detections (`Window`, `Curtains`, `Mirror`, `Footstool`) overlapped
   appliances and hollowed out `Fridge` / `CounterTop` masks during
   contained-mask subtraction (`v10` `i02`).
2. **Duplicate outer-shell confusion:** Detecting both `Sink` and `SinkBasin`
   on the same fixture caused the agent to target `Sink` (whose mask swallowed
   the `Faucet`) instead of `SinkBasin` (`v11` `i03`).
3. **Caption batch overhead & 3D clutter:** Pruning 114 $\rightarrow$ 85
   categories with `caption_batch_size = 29` reduces GroundingDINO forward
   passes from **5 batches/frame to 3 batches/frame (-40%)** and cuts spurious
   provisional 3D object fragments by more than half.

---

## Perception and 3D mapping performance optimizations (`v12` & `v13`)

Profiling of the 5-task runs identified four major structural bottlenecks across
the GPU perception worker, 2D open-vocabulary grounding, 3D point-cloud
association, and CPU raycasting (while AI2-THOR simulation itself takes
`< 0.08 s/step`):

1. **GPU-Resident Perception Models & `bfloat16` Batching (`model_runtime.py`, `vision_models.py`):**
   - Previously, `active_model()` moved all four neural models (~8.5 GB total:
     `DA3Metric-Large` 1.34 GB, `GroundingDINO` 0.70 GB, `SAM1 ViT-H` 2.56 GB,
     `OpenCLIP ViT-H-14` 3.94 GB) from CPU to GPU and back to CPU—followed by a
     synchronous `torch.cuda.empty_cache()`—four times on every single frame,
     adding ~2.5 s/frame of pure PCIe transfer and synchronization overhead.
   - `active_model()` now keeps each model resident on its assigned 24 GB RTX
     4090 GPU (~9.2 GB total footprint), eliminating per-frame CPU$\leftrightarrow$GPU
     weight thrashing.
   - In `v13`, `GroundingDetector` precomputes and caches the BERT text encoder
     features (`_cached_text_dict`) for the 3 fixed vocabulary batches once at
     startup, runs the Swin-T image backbone once per frame, expands the
     multi-scale visual pyramid to `batch_size=3`, and evaluates only the final
     decoder layer (`111 ms/frame`, **2.0$\times$ faster**).
   - `PromptedSegmenter` (`SAM1 ViT-H`) and `ClipEncoder` (`OpenCLIP ViT-H-14`)
     now run under `torch.autocast(bfloat16)` with batched `predict_torch` box
     prompts (`140 ms/frame` for SAM1 and `75 ms/frame` for OpenCLIP, cutting
     total perception RPC time from `3.48 s/frame` in `v11` $\rightarrow$
     `0.75 s/frame` in `v12` $\rightarrow$ **`0.38 s/frame` in `v13`**).

2. **Compound Token Span Scoring & CLIP Semantic Gating (`vision_models.py`, `object_mapping.py`):**
   - Previously, `decode_grounded_boxes()` scored multi-word categories by
     `sub_tokens.max()` and assigned `logits.max()` (across all 256 tokens
     including punctuation) as box confidence, causing `*Sliced` categories to
     hijack base categories (`41%–53%` hallucinated room categories) and wasting
     `26%–30%` of SAM1+CLIP compute on `0.20 < conf < 0.25` boxes that were
     immediately discarded by `MappingConfig(min_mask_confidence=0.25)`.
   - `v13` aligns `box_threshold = text_threshold = 0.25`, requires all tokens in
     compound spans (`require_all_span_tokens=True`), and gates low-confidence
     detections in `mapping_masks()` using OpenCLIP image-text cosine similarity
     (`min_clip_similarity = 0.185`).

3. **Monocular-Scale 3D Point-Cloud Association & Veil Removal (`object_mapping.py`, `memory.py`):**
   - Previously, `point_overlap()` used a `2.5 cm` (`voxel_size_m`) radius with
     `association_threshold = 1.2` and zero-margin `oriented_envelopes_overlap()`.
     Under monocular depth (`DA3Metric-Large`, RMSE `~0.31 m`), 2.5D surface
     shells drift by `5–15 cm` across viewpoints, causing `99.2%` of cross-view
     merges to fail (fragmenting 1 Fridge into 7 IDs and 1 Bed into 19 IDs).
   - `v13` erodes SAM masks by 1 px (`erode_mask_pixels = 1`) and trims the
     farthest 15% depth veil points (`depth_trim_quantile = 0.15`) before 3D
     projection, decouples `association_distance_m = 0.12 m` and
     `envelope_margin_m = 0.10 m` from `voxel_size_m`, caches `cKDTree` and
     AABB/OBB envelopes per point cloud, and incorporates `same_category_bonus = 0.20`
     during both frame-to-frame association and `maintain_objects()` consolidation—cutting
     duplicate 3D object fragmentation by **60%–73%**.

4. **Adaptive Ground-Plane Calibration & Vectorized 2D Ray Rasterization (`depth_models.py`, `spatial_map.py`):**
   - When `DA3Metric-Large` underestimates range by `8%–18%` on indoor floors,
     back-projected floor pixels from a `1.576 m` camera land at $z = +0.12\text{..}+0.24\text{ m}$,
     exceeding `obstacle_min_height_m = 0.12 m` and causing `80%–100%` of open
     floor to be misclassified as obstacle walls.
   - `SpatialMap._ray_evidence()` now detects flat horizontal floor modes to
     calibrate local floor elevation (`floor_offset`), protects physically
     traversed agent footprints (`_visited_footprint`) from noisy遠-range depth
     overwrites, and replaces 80 per-frame 2D `np.unique(..., axis=0)` calls
     with 64-bit integer key packing (`_pack` / `_unpack_set`), reducing CPU
     raycasting latency from **`340–510 ms/frame` to `~20 ms/frame` (20$\times$ faster)**.

---

## Module responsibilities

| Module | Responsibility |
|---|---|
| `dataset`, `settings`, `vocabulary` | Frozen fold, strict scene TOML config, and 85 task-relevant ALFRED categories |
| `perception`, `perception_worker`, `vision_models`, `depth_models` | Local GPU-resident GroundingDINO, SAM1, OpenCLIP, and DA3Metric-Large depth |
| `geometry`, `object_mapping`, `object_store`, `memory` | Registered observations, AABB/OBB-cached object fusion, raw/effective masks, and indexed evidence |
| `delta` | `RevisionWatermark` snapshots and compact incremental `SceneDelta` computation |
| `spatial_map`, `navigation`, `exploration`, `rendering` | Coverage, traversability, progressive FMM, frontier selection, and BEV rendering |
| `macro_controller`, `episode_server`, `tools`, `playbook` | Closed-loop FMM navigation + Auto-Framing, compound appliance macros, loopback RPC server, and 10 CLI tools |
| `policy`, `execution`, `actor_runtime` | Single-turn Map-First prompt, isolated temporary workspace, and tool-enabled Codex SDK runtime |
| `simulator`, `simulator_worker` | Official AI2-THOR 2.1.0 scene lifecycle, mask dispatch, and terminal scoring |
| `calibration_capture`, `depth_calibration`, `depth_calibration_runner` | Private train observations and independent raw metric depth comparison |
| `metrics`, `results`, `provenance` | Official aggregation, immutable attempts, `tool_calls.jsonl`, and code/asset provenance |

---

## Frozen pilot15 invocation

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
curl -fsS http://127.0.0.1:8787/health
export XAUTHORITY=/home/ysh/codebase/agentic-3d-task/tmp/alfred_setup/Xauthority
NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}" \
no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}" \
python -m codex_agent.cli.run_alfred \
  --data-root /home/ysh/Datasets/FILM/alfred/data/json_2.1.0 \
  --splits /home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json \
  --fold docs/benchmark/alfred/folds/pilot15.json \
  --output-dir outputs/alfred/pilot15_<date>_<attempt> \
  --simulator-python /home/ysh/codebase/FILM/.venv/bin/python \
  --alfred-root /home/ysh/Datasets/FILM/alfred \
  --scene-config configs/alfred/scene.toml \
  --x-display 5 \
  --physical-gpu 4 \
  --model gemini-3.8-flash \
  --reasoning-effort provider-default \
  --max-actions 1000 --max-failures 10 --max-decisions 35 --limit 5 \
  --preflight-only
```
