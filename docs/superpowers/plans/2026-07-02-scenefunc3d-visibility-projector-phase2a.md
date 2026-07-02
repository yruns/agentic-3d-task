# SceneFun3D Visibility Projector (Phase 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-geometry `backends/visibility.py` that projects the frozen 3D `TargetAnchor` into camera frames, applies a depth-occlusion test ("注意遮挡"), scores each frame's anchor visibility, and selects the top-N visible frames — the deterministic Stage B of the anchor-centric pipeline.

**Architecture:** One new module, `src/codex_agent/scenefunc3d/backends/visibility.py`, reusing `CameraGeometry` / `FloatArray` / `BoolArray` / `_validate_points_world` from `backends/lift_3d.py`. It is the inverse of `lift_3d.backproject_mask_to_world`: world→camera via the inverse of `camera_to_world`, then pinhole projection to pixels, then an occlusion test against the observed depth image (mirroring `tools/fusion_util`-style `|z − observed| ≤ tol·observed`). All functions operate on in-memory arrays / a `FrameCamera` value object, so the whole module is unit-testable with synthetic cameras — no dataset, sidecar, or adapter.

**Tech Stack:** Python 3.11+, numpy, Pydantic-free (frozen dataclasses), pytest. Reuses `backends/lift_3d.py`. Must satisfy `docs/python_code_agent_quality_guide.md` (§5 typing, §6 boundary validation, §7 exceptions, §13 ban list).

Reference spec: `docs/superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md` (Stage B). This plan is **Phase 2a only**; disk loading of frames and the online pipeline are Phase 2b (roadmap at end).

Environment (macOS): `source .venv/bin/activate`. Tests: `PYTHONPATH=src pytest <path> -v`.

---

## File Structure (Phase 2a)

- Create: `src/codex_agent/scenefunc3d/backends/visibility.py` — projection, per-point occlusion visibility, frame scoring, frame selection. One responsibility: 3D→2D visibility of the anchor.
- Test: `src/codex_agent/tests/test_scenefunc3d_visibility.py`

No existing files are modified (only imported).

---

## Task 1: World→pixel projection

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/visibility.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_visibility.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_visibility.py
"""Tests for SceneFunc3D 3D->2D visibility projection."""

from __future__ import annotations

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.backends.visibility import project_world_to_pixels


def _identity_camera(fx: float = 100.0, cx: float = 50.0, cy: float = 50.0) -> CameraGeometry:
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    return CameraGeometry(intrinsics=intrinsics, camera_to_world=np.eye(4))


def test_projects_on_axis_point_to_principal_point() -> None:
    geometry = _identity_camera()
    pixels, camera_z = project_world_to_pixels(np.array([[0.0, 0.0, 2.0]]), geometry)
    assert pixels[0] == pytest.approx((50.0, 50.0))
    assert camera_z[0] == pytest.approx(2.0)


def test_projects_off_axis_point_with_perspective() -> None:
    geometry = _identity_camera()
    pixels, camera_z = project_world_to_pixels(np.array([[0.1, 0.0, 2.0]]), geometry)
    # u = fx * x / z + cx = 100 * 0.1 / 2 + 50 = 55
    assert pixels[0][0] == pytest.approx(55.0)
    assert pixels[0][1] == pytest.approx(50.0)


def test_behind_camera_point_has_non_positive_z() -> None:
    geometry = _identity_camera()
    _pixels, camera_z = project_world_to_pixels(np.array([[0.0, 0.0, -1.0]]), geometry)
    assert camera_z[0] < 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.backends.visibility`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/backends/visibility.py
"""3D->2D visibility projection for SceneFun3D anchor-driven frame selection."""

from __future__ import annotations

import numpy as np

from codex_agent.scenefunc3d.backends.lift_3d import (
    CameraGeometry,
    FloatArray,
    _validate_points_world,
)


def project_world_to_pixels(
    points_world: FloatArray, geometry: CameraGeometry
) -> tuple[FloatArray, FloatArray]:
    """Project world points into one camera; return (pixels_uv (N,2), camera_z (N,)).

    Inverse of ``lift_3d.backproject_mask_to_world``: world -> camera via the
    inverse of ``camera_to_world``, then pinhole projection. Pixel u/v are NaN
    for points at or behind the image plane (``camera_z <= 0``); callers must
    gate on ``camera_z > 0`` before using pixels.
    """
    points = _validate_points_world(points_world, field_name="points_world")
    world_to_camera = np.linalg.inv(geometry.camera_to_world)
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    camera = homogeneous @ world_to_camera.T
    camera_xyz = camera[:, :3]
    camera_z = camera_xyz[:, 2]
    fx = geometry.intrinsics[0, 0]
    fy = geometry.intrinsics[1, 1]
    cx = geometry.intrinsics[0, 2]
    cy = geometry.intrinsics[1, 2]
    safe_z = np.where(camera_z > 0.0, camera_z, np.nan)
    u = fx * camera_xyz[:, 0] / safe_z + cx
    v = fy * camera_xyz[:, 1] / safe_z + cy
    pixels: FloatArray = np.column_stack((u, v)).astype(np.float64)
    return pixels, camera_z.astype(np.float64)


__all__ = ["project_world_to_pixels"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/visibility.py src/codex_agent/tests/test_scenefunc3d_visibility.py
git commit -m "feat(scenefunc3d): add world-to-pixel projection for visibility"
```

---

## Task 2: Per-point occlusion visibility

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/visibility.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_visibility.py`

- [ ] **Step 1: Write the failing test** — append:

```python
from codex_agent.scenefunc3d.backends.visibility import point_visibility


def _flat_depth(value: float = 2.0, size: int = 100) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float64)


def test_point_on_depth_surface_is_visible() -> None:
    geometry = _identity_camera()
    visible = point_visibility(np.array([[0.0, 0.0, 2.0]]), geometry, _flat_depth(2.0))
    assert bool(visible[0]) is True


def test_point_behind_surface_is_occluded() -> None:
    geometry = _identity_camera()
    # Observed depth is 1.0 (a near wall); the point sits at z=2.0 behind it.
    visible = point_visibility(np.array([[0.0, 0.0, 2.0]]), geometry, _flat_depth(1.0))
    assert bool(visible[0]) is False


def test_out_of_frame_point_is_not_visible() -> None:
    geometry = _identity_camera()
    # x=10 -> u = 100*10/2 + 50 = 550, outside the 100px image.
    visible = point_visibility(np.array([[10.0, 0.0, 2.0]]), geometry, _flat_depth(2.0))
    assert bool(visible[0]) is False


def test_behind_camera_point_is_not_visible() -> None:
    geometry = _identity_camera()
    visible = point_visibility(np.array([[0.0, 0.0, -1.0]]), geometry, _flat_depth(2.0))
    assert bool(visible[0]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: FAIL with `ImportError: cannot import name 'point_visibility'`.

- [ ] **Step 3: Write minimal implementation** — add (import `BoolArray` from `lift_3d`; add `point_visibility` + the constant to `__all__`):

```python
from codex_agent.scenefunc3d.backends.lift_3d import BoolArray

_DEFAULT_DEPTH_TOLERANCE = 0.25


def point_visibility(
    points_world: FloatArray,
    geometry: CameraGeometry,
    depth_meters: FloatArray,
    *,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> BoolArray:
    """Per-point un-occluded visibility in one frame.

    A point is visible when it is in front of the camera, projects inside the
    image, and its camera-space depth matches the observed depth within
    ``depth_tolerance`` (relative), i.e. ``|z - observed| <= tol * observed``.
    """
    depth = np.asarray(depth_meters, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_meters must be 2D; got shape {depth.shape}")
    if not np.isfinite(depth_tolerance) or depth_tolerance < 0.0:
        raise ValueError(f"depth_tolerance must be finite and non-negative: {depth_tolerance!r}")
    pixels, camera_z = project_world_to_pixels(points_world, geometry)
    height, width = int(depth.shape[0]), int(depth.shape[1])
    columns = np.floor(pixels[:, 0]).astype(np.int64, copy=False)
    rows = np.floor(pixels[:, 1]).astype(np.int64, copy=False)
    in_bounds = (
        (camera_z > 0.0)
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
        & (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    safe_rows = np.where(in_bounds, rows, 0)
    safe_columns = np.where(in_bounds, columns, 0)
    observed = depth[safe_rows, safe_columns]
    valid_observed = np.isfinite(observed) & (observed > 0.0)
    occlusion_ok = np.abs(camera_z - observed) <= depth_tolerance * observed
    visible: BoolArray = (in_bounds & valid_observed & occlusion_ok).astype(np.bool_)
    return visible
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/visibility.py src/codex_agent/tests/test_scenefunc3d_visibility.py
git commit -m "feat(scenefunc3d): add per-point depth-occlusion visibility"
```

---

## Task 3: Anchor visibility score per frame

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/visibility.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_visibility.py`

Score = `unoccluded_fraction * centeredness`. `unoccluded_fraction` is the fraction of a 7-point probe (anchor centroid + centroid ± radius along each axis) that is visible. `centeredness = 1 - dist(projected_centroid, image_center)/half_diagonal` (clamped to `[0,1]`). A frame is `visible` only if the centroid itself is visible.

- [ ] **Step 1: Write the failing test** — append:

```python
from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.visibility import (
    FrameVisibility,
    score_anchor_visibility,
)


def _centered_anchor() -> object:
    # Symmetric points -> centroid exactly (0, 0, 2), projecting to (50, 50)
    # so centeredness is exactly 1.0.
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.02, 0.0, 2.0],
            [-0.02, 0.0, 2.0],
            [0.0, 0.02, 2.0],
            [0.0, -0.02, 2.0],
        ]
    )
    return build_anchor(points, motion_type="pinch_pull", seed_frame_id="000001")


def test_fully_visible_centered_anchor_scores_high() -> None:
    result = score_anchor_visibility(
        "000001", _centered_anchor(), _identity_camera(), _flat_depth(2.0)
    )
    assert isinstance(result, FrameVisibility)
    assert result.visible is True
    assert result.unoccluded_fraction == pytest.approx(1.0)
    assert result.centeredness == pytest.approx(1.0)
    assert result.quality_score == pytest.approx(1.0)


def test_occluded_anchor_scores_zero_and_not_visible() -> None:
    result = score_anchor_visibility(
        "000001", _centered_anchor(), _identity_camera(), _flat_depth(1.0)
    )
    assert result.visible is False
    assert result.quality_score == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: FAIL with `ImportError: cannot import name 'FrameVisibility'`.

- [ ] **Step 3: Write minimal implementation** — add (import `TargetAnchor` from `anchor`, `dataclass`; add `FrameVisibility`, `score_anchor_visibility` to `__all__`):

```python
from dataclasses import dataclass

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor


@dataclass(frozen=True)
class FrameVisibility:
    """Anchor visibility of one frame."""

    frame_id: str
    visible: bool
    unoccluded_fraction: float
    centeredness: float
    quality_score: float


def _anchor_probe_points(anchor: TargetAnchor) -> FloatArray:
    center = np.asarray(anchor.centroid, dtype=np.float64)
    radius = anchor.radius_m
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float64,
    )
    return (center[np.newaxis, :] + offsets).astype(np.float64)


def score_anchor_visibility(
    frame_id: str,
    anchor: TargetAnchor,
    geometry: CameraGeometry,
    depth_meters: FloatArray,
    *,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> FrameVisibility:
    """Score how well one frame sees the anchor (probe visibility x centeredness)."""
    probes = _anchor_probe_points(anchor)
    visible = point_visibility(
        probes, geometry, depth_meters, depth_tolerance=depth_tolerance
    )
    centroid_visible = bool(visible[0])
    if not centroid_visible:
        return FrameVisibility(
            frame_id=frame_id,
            visible=False,
            unoccluded_fraction=0.0,
            centeredness=0.0,
            quality_score=0.0,
        )
    unoccluded_fraction = float(np.mean(visible))
    centeredness = _centeredness(anchor, geometry, depth_meters)
    quality_score = unoccluded_fraction * centeredness
    return FrameVisibility(
        frame_id=frame_id,
        visible=True,
        unoccluded_fraction=unoccluded_fraction,
        centeredness=centeredness,
        quality_score=quality_score,
    )


def _centeredness(
    anchor: TargetAnchor, geometry: CameraGeometry, depth_meters: FloatArray
) -> float:
    height, width = int(depth_meters.shape[0]), int(depth_meters.shape[1])
    pixels, _camera_z = project_world_to_pixels(
        np.asarray([anchor.centroid], dtype=np.float64), geometry
    )
    u, v = float(pixels[0, 0]), float(pixels[0, 1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return 0.0
    center_u, center_v = width / 2.0, height / 2.0
    distance = float(np.hypot(u - center_u, v - center_v))
    half_diagonal = float(np.hypot(center_u, center_v))
    if half_diagonal <= 0.0:
        return 0.0
    return max(0.0, 1.0 - distance / half_diagonal)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/visibility.py src/codex_agent/tests/test_scenefunc3d_visibility.py
git commit -m "feat(scenefunc3d): score per-frame anchor visibility"
```

---

## Task 4: Frame selection (top-N visible)

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/visibility.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_visibility.py`

- [ ] **Step 1: Write the failing test** — append:

```python
from codex_agent.scenefunc3d.backends.visibility import FrameCamera, select_visible_frames


def test_selection_drops_occluded_and_ranks_by_quality() -> None:
    anchor = _centered_anchor()
    geometry = _identity_camera()
    cameras = (
        FrameCamera(frame_id="000001", geometry=geometry, depth_meters=_flat_depth(2.0)),
        FrameCamera(frame_id="000002", geometry=geometry, depth_meters=_flat_depth(1.0)),
    )
    selected = select_visible_frames(anchor, cameras, frame_cap=10)
    assert tuple(v.frame_id for v in selected) == ("000001",)
    assert all(v.visible for v in selected)


def test_selection_respects_frame_cap() -> None:
    anchor = _centered_anchor()
    geometry = _identity_camera()
    cameras = tuple(
        FrameCamera(
            frame_id=f"{index:06d}", geometry=geometry, depth_meters=_flat_depth(2.0)
        )
        for index in range(5)
    )
    selected = select_visible_frames(anchor, cameras, frame_cap=3)
    assert len(selected) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: FAIL with `ImportError: cannot import name 'FrameCamera'`.

- [ ] **Step 3: Write minimal implementation** — add (import `Sequence` from `collections.abc`; add `FrameCamera`, `select_visible_frames` to `__all__`):

```python
from collections.abc import Sequence


@dataclass(frozen=True)
class FrameCamera:
    """One frame's in-memory camera + depth, for visibility scoring."""

    frame_id: str
    geometry: CameraGeometry
    depth_meters: FloatArray


def select_visible_frames(
    anchor: TargetAnchor,
    cameras: Sequence[FrameCamera],
    *,
    frame_cap: int,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> tuple[FrameVisibility, ...]:
    """Score all frames, keep the ones that see the anchor, return the top N.

    Ties break deterministically by ascending ``frame_id``.
    """
    if frame_cap <= 0:
        raise ValueError(f"frame_cap must be positive: {frame_cap!r}")
    scored = [
        score_anchor_visibility(
            camera.frame_id,
            anchor,
            camera.geometry,
            camera.depth_meters,
            depth_tolerance=depth_tolerance,
        )
        for camera in cameras
    ]
    visible = [result for result in scored if result.visible]
    visible.sort(key=lambda result: (-result.quality_score, result.frame_id))
    return tuple(visible[:frame_cap])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/visibility.py src/codex_agent/tests/test_scenefunc3d_visibility.py
git commit -m "feat(scenefunc3d): select top-N anchor-visible frames"
```

---

## Task 5: Quality gate

**Files:** none (verification only).

- [ ] **Step 1: Run the gate**

```bash
source .venv/bin/activate
ruff check src/
black --check src/codex_agent/scenefunc3d/backends/visibility.py src/codex_agent/tests/test_scenefunc3d_visibility.py
mypy src/codex_agent/scenefunc3d/backends/visibility.py
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -q
```
Expected: ruff clean, black clean, mypy no issues, 11 passed.

- [ ] **Step 2: Fix any issues and re-run until clean, then commit any fixes.**

---

## Phase 2b Roadmap (separate plan after 2a + reading the sidecar backends)

To be expanded into a full TDD plan once 2a lands. Requires reading
`backends/{molmo_rpc,sam_rpc,image_payload,mask_codec}.py`, `tools/{molmo_pointing,sam_masking,mask_lifting,dispatch}.py`, and the Codex turn/executor (`CodexTurnRequest`, `CodexExecutor`, `runner.run_single_sample`).

- **Frame loader**: `FrameCamera` from a scene's `source_frames.json` (reuse `scene_context` paths + shared depth/intrinsics/pose readers; extract the private `_read_depth_meters`/`_read_matrix` from `tools/mask_lifting.py` into a shared `backends/camera_io.py` — DRY).
- **Per-vertex visibility counts**: project candidate raw-mesh vertices into the selected frames → `visibility_counts` feeding `fusion.MultiViewLiftBundle` (closes the "agreement is weak offline" gap; can also augment the offline harness).
- **Pipeline driver** `pipeline.py::AnchorMultiViewPipeline`: seed → freeze anchor → `select_visible_frames` → per-frame projected-anchor Molmo point (nearest to projected anchor; anchor fallback recorded) → smallest-SAM candidate → lift → anchor-consistency gate → `fuse_multiview_points` → write existing final-artifact shape.
- **Agent seed-confirmation contract**: change `SceneFunc3dMaskTask` turn to emit `SeedConfirmation` (frame_id + candidate_id + affordance_concept + approval_actions); bounded correction loop (≤3), fail-closed; `runner.run_single_sample` runs the pipeline after confirmation; trim `playbook.py` to the seed stage.
- **Semi-online validation**: fix each saved case's confirmed seed as the anchor, run full B/C/D (needs Molmo/SAM sidecar, not the ModelHub adapter), compare to v6.

> **Status:** Implemented in [`docs/superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md`](2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md) (frame loader, per-vertex visibility counts, pure + scene-backed pipeline driver, sidecar proposer, semi-online harness — landed `0dbe76e`, recorded in [`docs/benchmark/scenefunc_molmo_sam3d/v7_semi_online_pipeline_20260702.md`](../../benchmark/scenefunc_molmo_sam3d/v7_semi_online_pipeline_20260702.md)). The **agent `SeedConfirmation` contract is deferred to Phase 2c** (separate plan).

---

## Self-Review

**1. Spec coverage (Phase-2a scope = Stage B geometry):** projection (Task 1), depth-occlusion visibility (Task 2), per-frame anchor score (Task 3), top-N frame selection (Task 4). ✓ Disk loading, per-vertex counts, pipeline, and agent contract are explicitly Phase-2b (roadmap). ✓ No scope creep.

**2. Placeholder scan:** No "TBD"/"add validation"/"similar to Task N". Every code step has full code. Occlusion tolerance and probe geometry are concrete. ✓

**3. Type consistency:** `project_world_to_pixels` (returns `(FloatArray, FloatArray)`), `point_visibility` (`-> BoolArray`, kw-only `depth_tolerance`), `FrameVisibility`, `score_anchor_visibility(frame_id, anchor, geometry, depth_meters, *, depth_tolerance)`, `FrameCamera`, `select_visible_frames(anchor, cameras, *, frame_cap, depth_tolerance)` are used with identical names/signatures across Tasks 1–4. Reuses `CameraGeometry`, `FloatArray`, `BoolArray`, `_validate_points_world` from `lift_3d.py`, and `TargetAnchor`/`build_anchor` from `anchor.py` (names verified against those committed modules). ✓
