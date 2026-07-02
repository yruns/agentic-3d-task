# SceneFun3D Anchor-Centric Multi-View Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SceneFun3D "union of all back-projected SAM pixels" with a training-free, anchor-centric layered fusion (multi-view agreement + anchor radius gate + connected-component clustering + `motion_type` size prior), and validate it offline on the 12 saved completed cases with zero model / adapter / sidecar calls.

**Architecture:** Phase 1 (this plan) builds four pure, strongly-typed modules under `src/codex_agent/scenefunc3d/backends/` (`motion_priors.py`, `anchor.py`, `fusion.py`) plus an offline re-scoring harness under `evaluation/offline_fusion.py`. Fusion operates in raw-mesh vertex-id space using canonical vertex coordinates from `raw/mesh.ply`; it consumes per-frame lifted point-id sets and a frozen `TargetAnchor`, and emits a `FusedMask` with a confidence and ranked instances (the seam for official ranked AP later). Phases 2–3 (roadmap at the end) add the visibility projector + per-frame deterministic stage + agent seed-confirmation contract, then the official-AP evaluator.

**Tech Stack:** Python 3.11+, numpy, scipy (`cKDTree`, optional with numpy fallback matching `lift_3d.py`), Pydantic v2 (boundaries only), pytest. Reuses `backends/lift_3d.py` (`FloatArray`, `IntArray`, `load_scene_mesh_vertices`) and `evaluation/scorer.py` (`score_point_ids`, `load_gt_point_ids`).

Reference spec: `docs/superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md`.

Environment (macOS): `source .venv/bin/activate`. Tests: `PYTHONPATH=src pytest <path> -v`.

---

## File Structure (Phase 1)

- Create: `src/codex_agent/scenefunc3d/backends/motion_priors.py` — `motion_type` → size prior lookup (radius + bbox caps). One responsibility: the affordance-size prior table.
- Create: `src/codex_agent/scenefunc3d/backends/anchor.py` — `TargetAnchor` + `build_anchor()` (centroid + robust radius, capped by motion prior).
- Create: `src/codex_agent/scenefunc3d/backends/fusion.py` — fusion types + `fuse_multiview_points()` (agreement gate → anchor gate → connected components → primary/instances + confidence).
- Create: `src/codex_agent/scenefunc3d/evaluation/offline_fusion.py` — build `MultiViewLiftBundle`s from a saved run root, sweep params, re-score with the existing scorer, emit a metrics table.
- Test: `src/codex_agent/tests/test_scenefunc3d_motion_priors.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_anchor.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_fusion.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_offline_fusion.py`

No existing files are modified in Phase 1 (the scorer and lift backend are imported, not changed).

---

## Task 1: Motion size priors

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/motion_priors.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_motion_priors.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_motion_priors.py
"""Tests for SceneFunc3D motion-type affordance-size priors."""

from __future__ import annotations

import pytest

from codex_agent.scenefunc3d.backends.motion_priors import (
    MotionSizePrior,
    motion_size_prior,
)


def test_small_control_motion_has_tight_prior() -> None:
    prior = motion_size_prior("key_press")
    assert isinstance(prior, MotionSizePrior)
    assert prior.max_radius_m == pytest.approx(0.12)
    assert prior.max_bbox_extent_m == pytest.approx(0.20)


def test_pull_motion_has_larger_prior_than_button() -> None:
    button = motion_size_prior("tip_push")
    pull = motion_size_prior("pinch_pull")
    assert pull.max_radius_m > button.max_radius_m


def test_unknown_motion_type_falls_back_to_default() -> None:
    prior = motion_size_prior("totally_unknown_motion")
    assert prior == motion_size_prior("")
    assert prior.max_radius_m == pytest.approx(0.25)


def test_motion_type_is_normalized_case_insensitively() -> None:
    assert motion_size_prior("KEY_PRESS") == motion_size_prior("key_press")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_motion_priors.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.backends.motion_priors`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/backends/motion_priors.py
"""Affordance-size priors keyed by SceneFun3D motion type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionSizePrior:
    """Upper bounds on the physical size of an affordance for one motion type."""

    max_radius_m: float
    max_bbox_extent_m: float


_DEFAULT_PRIOR = MotionSizePrior(max_radius_m=0.25, max_bbox_extent_m=0.45)

# Initial priors; calibrate against the offline sweep (see plan Task 8).
_PRIORS_BY_MOTION: dict[str, MotionSizePrior] = {
    "rotate": MotionSizePrior(0.12, 0.20),
    "key_press": MotionSizePrior(0.12, 0.20),
    "tip_push": MotionSizePrior(0.12, 0.20),
    "hook_turn": MotionSizePrior(0.12, 0.20),
    "pinch_pull": MotionSizePrior(0.20, 0.35),
    "hook_pull": MotionSizePrior(0.20, 0.35),
    "foot_push": MotionSizePrior(0.20, 0.35),
    "plug_in": MotionSizePrior(0.15, 0.25),
    "unplug": MotionSizePrior(0.15, 0.25),
}


def motion_size_prior(motion_type: str) -> MotionSizePrior:
    """Return the affordance-size prior for a SceneFun3D motion type."""
    return _PRIORS_BY_MOTION.get(motion_type.strip().lower(), _DEFAULT_PRIOR)


__all__ = ["MotionSizePrior", "motion_size_prior"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_motion_priors.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/motion_priors.py src/codex_agent/tests/test_scenefunc3d_motion_priors.py
git commit -m "feat(scenefunc3d): add motion-type affordance-size priors"
```

---

## Task 2: Target anchor construction

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/anchor.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_anchor.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_anchor.py
"""Tests for SceneFunc3D target-anchor construction."""

from __future__ import annotations

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor, build_anchor
from codex_agent.scenefunc3d.tools.models import ToolInputError


def test_anchor_centroid_is_point_mean() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    anchor = build_anchor(points, motion_type="pinch_pull", seed_frame_id="000012")
    assert isinstance(anchor, TargetAnchor)
    assert anchor.centroid == pytest.approx((1.0 / 3.0, 1.0 / 3.0, 0.0))
    assert anchor.seed_frame_id == "000012"
    assert anchor.source_point_count == 3


def test_anchor_radius_capped_by_motion_prior() -> None:
    # Points spread ~1 m from centroid; pinch_pull caps radius at 0.20 m.
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    anchor = build_anchor(points, motion_type="pinch_pull", seed_frame_id="000001")
    assert anchor.radius_m == pytest.approx(0.20)


def test_anchor_radius_uses_percentile_when_below_cap() -> None:
    points = np.zeros((100, 3))
    points[:, 0] = np.linspace(0.0, 0.10, 100)  # spread 0.10 m along x
    anchor = build_anchor(
        points, motion_type="pinch_pull", seed_frame_id="000001", radius_percentile=95.0
    )
    assert 0.02 < anchor.radius_m < 0.20


def test_empty_points_raise() -> None:
    with pytest.raises(ToolInputError):
        build_anchor(np.zeros((0, 3)), motion_type="rotate", seed_frame_id="000001")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_anchor.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.backends.anchor`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/backends/anchor.py
"""Frozen 3D target anchor derived from a confirmed lifted mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codex_agent.scenefunc3d.backends.lift_3d import FloatArray, _validate_points_world
from codex_agent.scenefunc3d.backends.motion_priors import motion_size_prior

_MIN_ANCHOR_RADIUS_M = 0.02


@dataclass(frozen=True)
class TargetAnchor:
    """A frozen 3D anchor used as a visibility probe and noise filter."""

    centroid: tuple[float, float, float]
    radius_m: float
    motion_type: str
    seed_frame_id: str
    source_point_count: int


def build_anchor(
    points_world: FloatArray,
    *,
    motion_type: str,
    seed_frame_id: str,
    radius_percentile: float = 95.0,
) -> TargetAnchor:
    """Build a target anchor (centroid + motion-capped robust radius)."""
    points = _validate_points_world(points_world, field_name="anchor_points_world")
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    robust_radius = float(np.percentile(distances, radius_percentile))
    capped_radius = min(robust_radius, motion_size_prior(motion_type).max_radius_m)
    radius_m = max(capped_radius, _MIN_ANCHOR_RADIUS_M)
    return TargetAnchor(
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        radius_m=radius_m,
        motion_type=motion_type,
        seed_frame_id=seed_frame_id,
        source_point_count=int(points.shape[0]),
    )


__all__ = ["TargetAnchor", "build_anchor"]
```

Note: `_validate_points_world` already raises `ToolInputError` on empty / wrong-shape input, satisfying `test_empty_points_raise`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_anchor.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/anchor.py src/codex_agent/tests/test_scenefunc3d_anchor.py
git commit -m "feat(scenefunc3d): build motion-capped target anchor from lifted points"
```

---

## Task 3: Fusion data types

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/fusion.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_fusion.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_fusion.py
"""Tests for SceneFunc3D layered multi-view fusion."""

from __future__ import annotations

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import (
    FrameLift,
    FusionParams,
    MultiViewLiftBundle,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError


def test_frame_lift_rejects_negative_index() -> None:
    with pytest.raises(ToolInputError):
        FrameLift(frame_id="000001", point_indices=(-1, 2))


def test_bundle_requires_at_least_one_frame() -> None:
    anchor = build_anchor(
        np.zeros((3, 3)), motion_type="rotate", seed_frame_id="000001"
    )
    with pytest.raises(ToolInputError):
        MultiViewLiftBundle(frames=(), anchor=anchor)


def test_fusion_params_defaults() -> None:
    params = FusionParams()
    assert params.agreement_tau == pytest.approx(0.5)
    assert params.radius_scale == pytest.approx(1.0)
    assert params.cluster_link_eps_m == pytest.approx(0.02)
    assert params.min_cluster_points == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.backends.fusion`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/backends/fusion.py
"""Layered, anchor-centric multi-view point fusion for SceneFun3D."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor
from codex_agent.scenefunc3d.tools.models import ToolInputError


@dataclass(frozen=True)
class FrameLift:
    """One frame's lifted raw-mesh vertex ids."""

    frame_id: str
    point_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ToolInputError("FrameLift.frame_id must be non-empty")
        if any(index < 0 for index in self.point_indices):
            raise ToolInputError("FrameLift.point_indices must be non-negative")


@dataclass(frozen=True)
class MultiViewLiftBundle:
    """Per-frame lifted vertex ids plus the frozen anchor for one sample."""

    frames: tuple[FrameLift, ...]
    anchor: TargetAnchor
    visibility_counts: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ToolInputError("MultiViewLiftBundle requires at least one frame")


@dataclass(frozen=True)
class FusionParams:
    """Tunable fusion thresholds (calibrated by the offline sweep)."""

    agreement_tau: float = 0.5
    radius_scale: float = 1.0
    cluster_link_eps_m: float = 0.02
    min_cluster_points: int = 10


@dataclass(frozen=True)
class FusedInstance:
    """One fused instance cluster."""

    point_indices: tuple[int, ...]
    confidence: float
    bbox_extent_m: float
    size_prior_ok: bool


@dataclass(frozen=True)
class FusedMask:
    """Final fused mask (primary instance) plus ranked instances for AP."""

    point_indices: tuple[int, ...]
    confidence: float
    instances: tuple[FusedInstance, ...]
    params: FusionParams


__all__ = [
    "FrameLift",
    "MultiViewLiftBundle",
    "FusionParams",
    "FusedInstance",
    "FusedMask",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/fusion.py src/codex_agent/tests/test_scenefunc3d_fusion.py
git commit -m "feat(scenefunc3d): add layered fusion data types"
```

---

## Task 4: Agreement scoring + gate

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/fusion.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_fusion.py`

- [ ] **Step 1: Write the failing test**

Append to `src/codex_agent/tests/test_scenefunc3d_fusion.py`:

```python
from codex_agent.scenefunc3d.backends.fusion import agreement_scores


def test_agreement_uses_visibility_denominator() -> None:
    frames = (
        FrameLift(frame_id="000001", point_indices=(5, 6)),
        FrameLift(frame_id="000002", point_indices=(5,)),
    )
    # vertex 5 hit in 2 frames, visible in 4 -> 0.5; vertex 6 hit 1, visible 2 -> 0.5
    visibility = {5: 4, 6: 2}
    scores = agreement_scores(frames, visibility)
    assert scores[5] == pytest.approx(0.5)
    assert scores[6] == pytest.approx(0.5)


def test_agreement_without_visibility_uses_frame_count() -> None:
    frames = (
        FrameLift(frame_id="000001", point_indices=(5, 6)),
        FrameLift(frame_id="000002", point_indices=(5,)),
    )
    scores = agreement_scores(frames, {})
    assert scores[5] == pytest.approx(1.0)  # 2 hits / 2 frames
    assert scores[6] == pytest.approx(0.5)  # 1 hit / 2 frames


def test_agreement_clamped_to_one() -> None:
    frames = (FrameLift(frame_id="000001", point_indices=(5, 5)),)  # dup ignored
    scores = agreement_scores(frames, {5: 1})
    assert scores[5] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: FAIL with `ImportError: cannot import name 'agreement_scores'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/codex_agent/scenefunc3d/backends/fusion.py` (imports at top, function before `__all__`; add `agreement_scores` to `__all__`):

```python
from collections.abc import Iterable


def agreement_scores(
    frames: Iterable[FrameLift],
    visibility_counts: Mapping[int, int],
) -> dict[int, float]:
    """Return per-vertex multi-view agreement in [0, 1].

    Agreement is ``hits / visibility`` when a per-vertex visibility count is
    known, else ``hits / n_frames`` over the frames in the bundle. Duplicate
    ids inside one frame count once for that frame.
    """
    frame_list = list(frames)
    n_frames = len(frame_list)
    hit_counts: dict[int, int] = {}
    for frame in frame_list:
        for index in set(frame.point_indices):
            hit_counts[index] = hit_counts.get(index, 0) + 1

    scores: dict[int, float] = {}
    for index, hits in hit_counts.items():
        denominator = visibility_counts.get(index, n_frames)
        if denominator <= 0:
            denominator = n_frames
        scores[index] = min(1.0, hits / denominator)
    return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/fusion.py src/codex_agent/tests/test_scenefunc3d_fusion.py
git commit -m "feat(scenefunc3d): add multi-view agreement scoring"
```

---

## Task 5: Anchor gate + connected-component clustering

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/fusion.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_fusion.py`

- [ ] **Step 1: Write the failing test**

Append to `src/codex_agent/tests/test_scenefunc3d_fusion.py`:

```python
from codex_agent.scenefunc3d.backends.fusion import (
    anchor_component_indices,
    connected_components,
)


def test_connected_components_splits_far_blobs() -> None:
    # Two tight blobs 1 m apart, eps = 0.05 -> two components.
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]
    )
    components = connected_components(coords, eps=0.05)
    assert len(components) == 2
    assert {frozenset(c) for c in components} == {frozenset({0, 1}), frozenset({2, 3})}


def test_anchor_component_keeps_blob_nearest_anchor() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]
    )
    kept = anchor_component_indices(
        coords, anchor_centroid=(0.0, 0.0, 0.0), eps=0.05, min_points=1
    )
    assert set(kept) == {0, 1}


def test_anchor_component_drops_small_components() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    kept = anchor_component_indices(
        coords, anchor_centroid=(0.0, 0.0, 0.0), eps=0.05, min_points=5
    )
    assert kept == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: FAIL with `ImportError: cannot import name 'connected_components'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/codex_agent/scenefunc3d/backends/fusion.py` (add `numpy as np`, and `FloatArray`/`IntArray` imports from `lift_3d`; add both functions to `__all__`):

```python
import numpy as np

from codex_agent.scenefunc3d.backends.lift_3d import FloatArray


def _neighbor_pairs(coords: FloatArray, eps: float) -> list[tuple[int, int]]:
    """Return index pairs within ``eps`` (scipy KD-tree, numpy fallback)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        deltas = coords[:, None, :] - coords[None, :, :]
        distances_sq = np.einsum("ijk,ijk->ij", deltas, deltas)
        upper = np.triu(distances_sq <= eps * eps, k=1)
        rows, cols = np.nonzero(upper)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    tree = cKDTree(coords)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    return [(int(a), int(b)) for a, b in pairs]


def connected_components(coords: FloatArray, eps: float) -> list[tuple[int, ...]]:
    """Single-linkage components of ``coords`` at radius ``eps``."""
    n_points = int(coords.shape[0])
    parent = list(range(n_points))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in _neighbor_pairs(coords, eps):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    groups: dict[int, list[int]] = {}
    for node in range(n_points):
        groups.setdefault(find(node), []).append(node)
    return [tuple(sorted(members)) for members in groups.values()]


def anchor_component_indices(
    coords: FloatArray,
    *,
    anchor_centroid: tuple[float, float, float],
    eps: float,
    min_points: int,
) -> tuple[int, ...]:
    """Return the component containing the point nearest the anchor centroid.

    Returns an empty tuple when the chosen component is smaller than
    ``min_points`` (treated as a failed cluster).
    """
    if coords.shape[0] == 0:
        return ()
    centroid = np.asarray(anchor_centroid, dtype=np.float64)
    nearest_row = int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))
    for component in connected_components(coords, eps):
        if nearest_row in component:
            if len(component) < min_points:
                return ()
            return component
    return ()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/fusion.py src/codex_agent/tests/test_scenefunc3d_fusion.py
git commit -m "feat(scenefunc3d): add anchor gate and connected-component clustering"
```

---

## Task 6: Assemble `fuse_multiview_points`

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/fusion.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_fusion.py`

- [ ] **Step 1: Write the failing test**

Append to `src/codex_agent/tests/test_scenefunc3d_fusion.py`:

```python
from codex_agent.scenefunc3d.backends.fusion import FusedMask, fuse_multiview_points


def _scene_vertices() -> np.ndarray:
    # 0..2 = tight target blob near origin; 3..12 = spurious far wall points.
    target = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
    )
    wall = np.stack(
        [np.linspace(1.0, 1.09, 10), np.zeros(10), np.zeros(10)], axis=1
    )
    return np.concatenate([target, wall], axis=0)


def test_fusion_keeps_consensus_target_drops_far_wall_by_anchor_gate() -> None:
    anchor = build_anchor(
        _scene_vertices()[:3], motion_type="pinch_pull", seed_frame_id="000001"
    )
    # Both frames hit the target; frame 2 also hits the far wall. The wall passes
    # the agreement gate (0.5) but is removed by the anchor radius gate.
    frames = (
        FrameLift(frame_id="000001", point_indices=(0, 1, 2)),
        FrameLift(frame_id="000002", point_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
    )
    bundle = MultiViewLiftBundle(frames=frames, anchor=anchor)
    params = FusionParams(agreement_tau=0.5, cluster_link_eps_m=0.05, min_cluster_points=2)
    fused = fuse_multiview_points(bundle, _scene_vertices(), params)
    assert isinstance(fused, FusedMask)
    assert set(fused.point_indices) == {0, 1, 2}
    assert fused.confidence == pytest.approx(1.0)


def test_fusion_empty_when_nothing_passes_agreement() -> None:
    anchor = build_anchor(
        _scene_vertices()[:3], motion_type="pinch_pull", seed_frame_id="000001"
    )
    frames = (FrameLift(frame_id="000001", point_indices=(0,)),)
    bundle = MultiViewLiftBundle(frames=frames, anchor=anchor, visibility_counts={0: 10})
    fused = fuse_multiview_points(bundle, _scene_vertices(), FusionParams(agreement_tau=0.5))
    assert fused.point_indices == ()
    assert fused.confidence == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: FAIL with `ImportError: cannot import name 'fuse_multiview_points'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/codex_agent/scenefunc3d/backends/fusion.py` (add the `motion_size_prior` import; add `fuse_multiview_points` to `__all__`):

```python
from codex_agent.scenefunc3d.backends.motion_priors import motion_size_prior


def _bbox_extent_m(coords: FloatArray) -> float:
    if coords.shape[0] == 0:
        return 0.0
    return float(np.max(coords.max(axis=0) - coords.min(axis=0)))


def _empty_fused_mask(params: FusionParams) -> FusedMask:
    return FusedMask(point_indices=(), confidence=0.0, instances=(), params=params)


def fuse_multiview_points(
    bundle: MultiViewLiftBundle,
    scene_vertices: FloatArray,
    params: FusionParams,
) -> FusedMask:
    """Fuse per-frame lifted vertex ids into one anchor-centric mask.

    Pipeline: agreement gate -> anchor radius gate -> connected components ->
    keep the anchor component. Confidence is the mean agreement of kept
    vertices; the kept component is also emitted as the primary ranked instance.
    """
    scores = agreement_scores(bundle.frames, bundle.visibility_counts)
    kept_ids = [
        index for index, score in scores.items() if score >= params.agreement_tau
    ]
    if not kept_ids:
        return _empty_fused_mask(params)

    centroid = np.asarray(bundle.anchor.centroid, dtype=np.float64)
    gate_radius = bundle.anchor.radius_m * params.radius_scale
    kept_coords = scene_vertices[kept_ids]
    within_radius = np.linalg.norm(kept_coords - centroid, axis=1) <= gate_radius
    gated_ids = [index for index, keep in zip(kept_ids, within_radius) if keep]
    if not gated_ids:
        return _empty_fused_mask(params)

    gated_coords = scene_vertices[gated_ids]
    component_rows = anchor_component_indices(
        gated_coords,
        anchor_centroid=bundle.anchor.centroid,
        eps=params.cluster_link_eps_m,
        min_points=params.min_cluster_points,
    )
    if not component_rows:
        return _empty_fused_mask(params)

    final_ids = tuple(sorted(gated_ids[row] for row in component_rows))
    confidence = float(np.mean([scores[index] for index in final_ids]))
    extent = _bbox_extent_m(scene_vertices[list(final_ids)])
    size_ok = extent <= motion_size_prior(bundle.anchor.motion_type).max_bbox_extent_m
    instance = FusedInstance(
        point_indices=final_ids,
        confidence=confidence,
        bbox_extent_m=extent,
        size_prior_ok=size_ok,
    )
    return FusedMask(
        point_indices=final_ids,
        confidence=confidence,
        instances=(instance,),
        params=params,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_fusion.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/fusion.py src/codex_agent/tests/test_scenefunc3d_fusion.py
git commit -m "feat(scenefunc3d): assemble anchor-centric layered fusion"
```

---

## Task 7: Offline re-scoring harness

**Files:**
- Create: `src/codex_agent/scenefunc3d/evaluation/offline_fusion.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_offline_fusion.py`

The harness rebuilds a `MultiViewLiftBundle` per sample from a saved run root's `fragments/*/mask_data.npz` (each has `points_world` + `point_indices`), constructs the offline anchor from the union of the sample's fragment points, fuses with a set of `FusionParams`, and re-scores against hidden GT via the existing scorer. It uses `data_root` for GT + `raw/mesh.ply` + `motion_type`; it makes **no** model / adapter / sidecar calls.

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_offline_fusion.py
"""Tests for the offline SceneFunc3D fusion re-scoring harness."""

from __future__ import annotations

import numpy as np

from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.evaluation.offline_fusion import (
    SampleFusionInput,
    fuse_and_score_sample,
)


def _line_scene(n: int = 40) -> np.ndarray:
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 0.39, n)  # 1 cm spacing
    return coords


def test_fuse_and_score_recovers_precision_over_union() -> None:
    scene = _line_scene()
    gt_ids = frozenset(range(0, 5))  # tight target
    # frame A hits target; frame B hits target + far spurious tail. The anchor is
    # built from the seed fragment (the target), not the polluted union.
    sample = SampleFusionInput(
        sample_id="420673::demo",
        motion_type="pinch_pull",
        frames_point_indices=(
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 4, 30, 31, 32, 33, 34, 35),
        ),
        anchor_point_indices=(0, 1, 2, 3, 4),
        scene_vertices=scene,
        gt_ids=gt_ids,
    )
    params = FusionParams(agreement_tau=0.5, cluster_link_eps_m=0.03, min_cluster_points=2)
    score = fuse_and_score_sample(sample, params)
    assert score.metrics.precision == 1.0  # spurious tail removed
    assert score.metrics.iou > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_offline_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.evaluation.offline_fusion`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/evaluation/offline_fusion.py
"""Offline SceneFunc3D fusion sweep: rebuild bundles from saved artifacts,
re-fuse, and re-score against hidden GT. No model/adapter/sidecar calls."""

from __future__ import annotations

import argparse
import json
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import (
    FrameLift,
    FusionParams,
    MultiViewLiftBundle,
    fuse_multiview_points,
)
from codex_agent.scenefunc3d.backends.lift_3d import (
    FloatArray,
    load_scene_mesh_vertices,
)
from codex_agent.scenefunc3d.evaluation.scorer import (
    SceneFunc3dScore,
    load_gt_point_ids,
    score_point_ids,
)
from codex_agent.scenefunc3d.sample import load_sample, scene_dir_for


@dataclass(frozen=True)
class SampleFusionInput:
    """Everything needed to fuse + score one sample offline."""

    sample_id: str
    motion_type: str
    frames_point_indices: tuple[tuple[int, ...], ...]
    anchor_point_indices: tuple[int, ...]
    scene_vertices: FloatArray
    gt_ids: Set[int]


def fuse_and_score_sample(
    sample: SampleFusionInput, params: FusionParams
) -> SceneFunc3dScore:
    """Fuse one sample's frames and score the result against its GT ids.

    The anchor is built from ``anchor_point_indices`` (the seed fragment), not
    the union, so a far spurious fragment cannot pull the anchor off-target.
    """
    if not sample.anchor_point_indices or not any(sample.frames_point_indices):
        return score_point_ids(
            sample_id=sample.sample_id, predicted_ids=frozenset(), gt_ids=sample.gt_ids
        )
    anchor = build_anchor(
        sample.scene_vertices[list(sample.anchor_point_indices)],
        motion_type=sample.motion_type,
        seed_frame_id="offline_seed",
    )
    frames = tuple(
        FrameLift(frame_id=f"offline_{position:03d}", point_indices=indices)
        for position, indices in enumerate(sample.frames_point_indices)
    )
    bundle = MultiViewLiftBundle(frames=frames, anchor=anchor)
    fused = fuse_multiview_points(bundle, sample.scene_vertices, params)
    return score_point_ids(
        sample_id=sample.sample_id,
        predicted_ids=frozenset(fused.point_indices),
        gt_ids=sample.gt_ids,
    )


def load_sample_fusion_input(
    *, run_root: Path, data_root: Path, sample_id: str
) -> SampleFusionInput:
    """Build a SampleFusionInput from a saved run root + dataset root."""
    sample = load_sample(data_root, sample_id)
    motion_type = sample.motion_hints[0].motion_type if sample.motion_hints else ""
    fragment_dir = run_root / sample.visit_id / sample.desc_id / "fragments"
    frames: list[tuple[int, ...]] = []
    for npz_path in sorted(fragment_dir.glob("*/mask_data.npz")):
        with np.load(npz_path) as archive:
            indices = np.asarray(archive["point_indices"]).astype(np.int64).ravel()
        frames.append(tuple(int(value) for value in indices))
    # Offline seed proxy: use the largest fragment as the anchor source. In the
    # online pipeline the anchor comes from the agent-confirmed seed instead.
    anchor_point_indices = max(frames, key=len) if frames else ()
    scene_vertices = load_scene_mesh_vertices(
        scene_dir_for(data_root, sample.visit_id) / "raw" / "mesh.ply"
    )
    return SampleFusionInput(
        sample_id=sample_id,
        motion_type=motion_type,
        frames_point_indices=tuple(frames),
        anchor_point_indices=anchor_point_indices,
        scene_vertices=scene_vertices,
        gt_ids=load_gt_point_ids(data_root, sample_id),
    )


def _sample_ids_from_run_root(run_root: Path) -> tuple[str, ...]:
    sample_ids: list[str] = []
    for visit_dir in sorted(p for p in run_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for desc_dir in sorted(p for p in visit_dir.iterdir() if (p / "fragments").is_dir()):
            sample_ids.append(f"{visit_dir.name}::{desc_dir.name}")
    return tuple(sample_ids)


def run_offline_sweep(
    *, run_root: Path, data_root: Path, param_grid: tuple[FusionParams, ...]
) -> dict[str, object]:
    """Fuse+score every sample under run_root for each params set."""
    sample_ids = _sample_ids_from_run_root(run_root)
    results: list[dict[str, object]] = []
    for params in param_grid:
        ious: list[float] = []
        precisions: list[float] = []
        hits25 = 0
        hits50 = 0
        for sample_id in sample_ids:
            sample = load_sample_fusion_input(
                run_root=run_root, data_root=data_root, sample_id=sample_id
            )
            score = fuse_and_score_sample(sample, params)
            ious.append(score.metrics.iou)
            precisions.append(score.metrics.precision)
            hits25 += int(score.metrics.iou >= 0.25)
            hits50 += int(score.metrics.iou >= 0.50)
        n = max(len(sample_ids), 1)
        results.append(
            {
                "params": {
                    "agreement_tau": params.agreement_tau,
                    "radius_scale": params.radius_scale,
                    "cluster_link_eps_m": params.cluster_link_eps_m,
                    "min_cluster_points": params.min_cluster_points,
                },
                "mean_iou": sum(ious) / n,
                "mean_precision": sum(precisions) / n,
                "ap25": hits25 / n,
                "ap50": hits50 / n,
            }
        )
    return {"sample_count": len(sample_ids), "results": results}


def _default_param_grid() -> tuple[FusionParams, ...]:
    grid: list[FusionParams] = []
    for tau in (0.3, 0.5, 0.7):
        for eps in (0.02, 0.04):
            grid.append(FusionParams(agreement_tau=tau, cluster_link_eps_m=eps))
    return tuple(grid)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run the offline fusion sweep and print JSON."""
    parser = argparse.ArgumentParser(description="Offline SceneFunc3D fusion sweep")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_offline_sweep(
        run_root=args.run_root, data_root=args.data_root, param_grid=_default_param_grid()
    )
    text = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_offline_fusion.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/evaluation/offline_fusion.py src/codex_agent/tests/test_scenefunc3d_offline_fusion.py
git commit -m "feat(scenefunc3d): add offline fusion re-scoring harness"
```

---

## Task 8: Run the offline sweep on the 12 saved cases + record numbers

**Files:**
- Create: `docs/benchmark/scenefunc_molmo_sam3d/v6_offline_fusion_sweep_20260702.md`

- [ ] **Step 1: Run the sweep on the saved run root**

Run:
```bash
source .venv/bin/activate
PYTHONPATH=src python -m codex_agent.scenefunc3d.evaluation.offline_fusion \
  --run-root tmp/scenefunc3d/artifacts/scenefunc_421254_all23_keep_home_20260702 \
  --data-root data/SceneFun3D \
  --output tmp/scenefunc3d/artifacts/offline_fusion_sweep_20260702.json
```
Expected: JSON with `sample_count` = number of completed cases with `fragments/`, and a `results` row per params set (mean_iou / mean_precision / ap25 / ap50).

- [ ] **Step 2: Compare against the union baseline**

The current baseline (union) is Mean IoU `0.1258`, AP50 `0.0`, mean precision `0.197` (from the v5 review). Confirm at least one params set raises mean precision and mean IoU above baseline; note the best `(agreement_tau, cluster_link_eps_m)`.

- [ ] **Step 3: Write the benchmark record**

Create `docs/benchmark/scenefunc_molmo_sam3d/v6_offline_fusion_sweep_20260702.md` following the mandatory benchmark-doc format (branch + commit at run time; exact CLI; raw artifact path `tmp/scenefunc3d/artifacts/offline_fusion_sweep_20260702.json`; the sweep table; per-sample deltas vs union; caveat that agreement is weak offline because old runs have 1–2 frames/case, so this isolates the anchor-gate + clustering effect). Update `docs/benchmark/scenefunc_molmo_sam3d/README.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/scenefunc_molmo_sam3d/v6_offline_fusion_sweep_20260702.md docs/benchmark/scenefunc_molmo_sam3d/README.md
git commit -m "docs(scenefunc3d): record offline fusion sweep vs union baseline"
```

---

## Task 9: Quality gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full quality gate**

Run:
```bash
source .venv/bin/activate
ruff check src/
black --check src/
mypy src/codex_agent/scenefunc3d/backends/motion_priors.py \
     src/codex_agent/scenefunc3d/backends/anchor.py \
     src/codex_agent/scenefunc3d/backends/fusion.py \
     src/codex_agent/scenefunc3d/evaluation/offline_fusion.py
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_motion_priors.py \
     src/codex_agent/tests/test_scenefunc3d_anchor.py \
     src/codex_agent/tests/test_scenefunc3d_fusion.py \
     src/codex_agent/tests/test_scenefunc3d_offline_fusion.py -q
```
Expected: ruff clean, black clean, mypy no issues on the four new modules, all new tests pass.

- [ ] **Step 2: Fix any lint/type issues and re-run until clean.**

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore(scenefunc3d): satisfy quality gate for fusion phase 1"
```

---

## Phase 2–3 Roadmap (expanded into full plans after Phase 1 numbers)

**Phase 2 — visibility projector + per-frame stage + agent seed contract** (needs Molmo/SAM sidecar, not the ModelHub adapter for the deterministic part):
- `backends/visibility.py`: `SceneToImageProjector` (3D→2D projection + depth-occlusion test, reusing `CameraGeometry`), per-frame visibility + quality score, per-vertex visibility counts feeding `MultiViewLiftBundle.visibility_counts`.
- `pipeline.py`: `AnchorMultiViewPipeline` driving B/C/D after seed confirmation; per-frame projected-anchor Molmo prompt + nearest-point selection + anchor fallback (recorded) + smallest-SAM candidate + anchor-consistency gate.
- `runner.py` / `task.py` / `playbook.py`: change the agent turn contract to emit `SeedConfirmation`; runner runs the pipeline and writes the existing final-artifact shape (scorer unchanged); seed correction loop bounded to 3, fail-closed.
- Semi-online validation: fix each saved case's confirmed seed as the anchor, run full B/C/D, compare to Phase 1 offline numbers.

**Phase 3 — official ranked AP seam:**
- `evaluation/ap.py`: port Fun3DU `scripts/sun3d/eval/eval_utils/eval_script.py` + `rle.py`; pcd→RLE bridge; consume `FusedMask.instances` + confidence; keep success-only Mean IoU/AP25/AP50 alongside official AP.

---

## Self-Review

**1. Spec coverage (Phase 1 scope):**
- Layered fusion (agreement + anchor gate + DBSCAN/components + size prior) → Tasks 4–6. ✓
- `motion_type` size prior → Task 1, used in anchor cap (Task 2) and bbox flag (Task 6). ✓
- Confidence = mean agreement of kept vertices → Task 6. ✓
- Ranked instances seam for official AP → `FusedMask.instances` (Task 6); wiring is Phase 3. ✓
- Offline validation on 12 saved cases, no adapter/model → Task 7–8. ✓
- Anchor = centroid + robust radius capped by motion prior → Task 2. ✓
- Fail-closed (empty result is explicit, not silent; dependency errors raise) → `_empty_fused_mask` + `_validate_points_world` + scipy/numpy equivalence. ✓
- Deferred to Phase 2/3 (visibility projector, per-frame stage, agent contract, official AP) → Roadmap. ✓ (explicitly out of Phase 1 scope)

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". `size_cap_by_motion` values are concrete in Task 1 (calibration is a Task 8 action, not a code gap). ✓

**3. Type consistency:** `TargetAnchor`, `FrameLift`, `MultiViewLiftBundle` (`visibility_counts` default `{}`), `FusionParams` (`agreement_tau`/`radius_scale`/`cluster_link_eps_m`/`min_cluster_points`), `FusedInstance`, `FusedMask`, `agreement_scores`, `connected_components`, `anchor_component_indices`, `fuse_multiview_points`, `SampleFusionInput`, `fuse_and_score_sample` are used with identical names/signatures across Tasks 3–8. `score_point_ids(sample_id=, predicted_ids=, gt_ids=)` and `load_gt_point_ids(data_root, sample_id)` match `evaluation/scorer.py`. `load_scene_mesh_vertices` and `FloatArray` match `backends/lift_3d.py`. ✓
