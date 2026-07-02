# SceneFun3D Anchor Multi-View Pipeline (Phase 2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic Stage B→C→D engine that turns a frozen `TargetAnchor` into a fused 3D mask — a memory-streaming frame loader, streaming anchor-visibility frame selection, per-vertex multi-view visibility counts, an injectable `AnchorMultiViewPipeline` driver, a sidecar-backed per-frame proposer (Molmo→SAM→lift), and an offline-mirroring **semi-online** re-scoring harness — all runnable without the ModelHub adapter.

**Architecture:** New modules under `src/codex_agent/scenefunc3d/backends/` (`camera_io.py`, `frame_loader.py`, plus additions to the committed `visibility.py`) and a new `src/codex_agent/scenefunc3d/pipeline.py` (pure, dependency-injected driver) + `pipeline_backends.py` (sidecar-backed `SidecarFrameProposer`). A new `evaluation/semi_online_pipeline.py` reuses saved run-root seeds as **fixed anchors** and runs the full pipeline through the real Molmo/SAM sidecars, re-scoring with the existing scorer. The pure driver depends only on injected callables (frame geometry, a `FrameProposer`, a visibility counter) so it is unit-testable with synthetic cameras and fakes; the scene wrapper wires the disk-backed implementations.

**Tech Stack:** Python 3.11+, numpy, Pydantic v2 (only at CLI/JSON boundaries), pytest. Reuses `backends/{lift_3d,anchor,fusion,visibility,motion_priors}.py`, `tools/{molmo_pointing,sam_masking,mask_lifting,scene_context}.py`, `backends/{molmo_rpc,config,frame_assets}.py`, `evaluation/scorer.py`. Must satisfy `docs/python_code_agent_quality_guide.md` (§5 typing, §6 boundary validation, §7 exceptions, §13 ban list) and `AGENTS.md` §1.

Reference spec: `docs/superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md` (Stages B/C/D + validation tiers 1–2). Phase 2a (`backends/visibility.py`) is already committed (`b6a5904`, `d0f4158`).

Environment (macOS): `source .venv/bin/activate`. Tests: `PYTHONPATH=src pytest <path> -v`.

---

## Scope decision (read first)

The reference spec lists five Phase-2b items. This plan implements **four** of them —
frame loader, per-vertex visibility counts, the deterministic `AnchorMultiViewPipeline`
driver, and semi-online validation — plus the sidecar proposer they need.

**The agent `SeedConfirmation` turn-contract change (runner/task/playbook) is deferred to a
separate Phase 2c plan.** Rationale:

1. **Semi-online validation does not need it.** Tier-2 validation uses a *saved seed as a fixed
   anchor* (spec §"验证分三档" tier 2), so it exercises the full B/C/D + agreement dimension
   with zero changes to the agent contract.
2. **De-risk before surgery.** The contract change edits the 111 KB `runner.py` self-report
   validation machinery and its 3500-line test file. Proving the engine semi-online first tells
   us whether the online agreement dimension is worth that cost.
3. It matches the Phase 2a roadmap, which already listed "Agent seed-confirmation contract" and
   "Semi-online validation" as **separate** bullets.

Every module in this plan is a prerequisite for Phase 2c anyway (the contract change just calls
`run_anchor_multiview_pipeline`), so nothing here is throwaway.

---

## Real data layout (verified 2026-07-02 on `data/SceneFun3D/421254/raw`)

- **170 frames.** Per frame: `NNNNNN-rgb.jpg`, `NNNNNN-depth.png`, `NNNNNN-intrinsic.txt`, `NNNNNN.txt` (pose).
- **RGB and depth are both 1440×1920** (same resolution → SAM mask shape matches depth shape, so `lift_3d.backproject_mask_to_world` works with no resize).
- **Depth PNG is 16-bit (`I;16`), millimetres** → `read_depth_meters` divides by 1000 (matches `mask_lifting._read_depth_meters`).
- **Per-frame `-intrinsic.txt` == `intrinsic_depth.txt` == `intrinsic_color.txt`**, a 3×3 with `cx≈718.2 (≈1440/2)`, `cy≈954.6 (≈1920/2)` → intrinsics match the 1440×1920 grid used by both the mask and the depth.
- **Pose `NNNNNN.txt` is a 4×4 camera-to-world** with last row `[0,0,0,1]`.
- `resolve_available_frame_geometry_assets` (in `backends/frame_assets.py`) already resolves these exact patterns (`{id}-depth.png`, `{id}-intrinsic.txt`, `{id}.txt`).

**Memory constraint (drives the streaming design):** one depth array is `1440*1920*8 ≈ 22 MB`.
170 frames ≈ **3.7 GB** if held at once. Therefore the frame loader yields geometry lazily and
depth is read **one frame at a time and discarded** during visibility scoring and visibility
counting. `scene_vertices` (raw mesh, ~2.7 M × 3 × 8 ≈ 65 MB) is loaded once per sample.

---

## File Structure (Phase 2b)

- Create `src/codex_agent/scenefunc3d/backends/camera_io.py` — shared `read_depth_meters`, `read_camera_matrix`, `load_camera_geometry`. `tools/mask_lifting.py` is refactored to reuse these (DRY; removes its private `_read_depth_meters`/`_read_matrix`).
- Create `src/codex_agent/scenefunc3d/backends/frame_loader.py` — lazy `iter_frame_geometry`, single-frame `load_frame_geometry`, `read_frame_depth`, `frame_rgb_path`. One responsibility: turn a `SceneFunc3dToolScene` into per-frame cameras/depth on demand.
- Modify `src/codex_agent/scenefunc3d/backends/visibility.py` — add streaming `select_scene_visible_frames` (Task 3) and `count_vertex_visibility` (Task 4). No behaviour change to Phase-2a functions.
- Create `src/codex_agent/scenefunc3d/pipeline.py` — pure, injectable driver: `FrameProposal`, `FrameProposer` (Protocol), `PerFrameOutcome`, `AnchorMultiViewResult`, `fuse_selected_frames` (pure), `run_anchor_multiview_pipeline` (scene wrapper).
- Create `src/codex_agent/scenefunc3d/pipeline_backends.py` — `SidecarFrameProposer` (Molmo RPC → smallest SAM candidate → lift → read back point ids).
- Create `src/codex_agent/scenefunc3d/evaluation/semi_online_pipeline.py` — tier-2 harness + CLI, mirroring `evaluation/offline_fusion.py` conventions.
- Modify `src/codex_agent/scenefunc3d/tools/molmo_pointing.py` — add public `points_from_molmo_response` (empty→`()`, no overlay, no raise-on-empty) for the proposer.
- Create test helper `src/codex_agent/tests/scenefunc3d_synthetic_scene.py` (importable, not a test module) — writes a tiny on-disk scene (frames + 16-bit depth PNG + intrinsic/pose txt + binary PLY mesh + `source_frames.json`).
- Tests: `test_scenefunc3d_camera_io.py`, `test_scenefunc3d_frame_loader.py`, `test_scenefunc3d_visibility.py` (extend), `test_scenefunc3d_pipeline.py`, `test_scenefunc3d_pipeline_backends.py`, `test_scenefunc3d_semi_online_pipeline.py`, `test_scenefunc3d_molmo_pointing.py` (extend or new).

---

## Task 1: Shared camera I/O (`camera_io.py`) + DRY refactor of `mask_lifting.py`

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/camera_io.py`
- Modify: `src/codex_agent/scenefunc3d/tools/mask_lifting.py` (replace private readers with imports)
- Test: `src/codex_agent/tests/test_scenefunc3d_camera_io.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_camera_io.py
"""Tests for shared SceneFunc3D camera/depth I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codex_agent.scenefunc3d.backends.camera_io import (
    load_camera_geometry,
    read_camera_matrix,
    read_depth_meters,
)
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.tools.models import ToolInputError


def _write_depth_png(path: Path, depth_mm: np.ndarray) -> None:
    Image.fromarray(depth_mm.astype(np.uint16)).save(path)


def test_read_depth_meters_scales_millimetres(tmp_path: Path) -> None:
    depth_png = tmp_path / "000000-depth.png"
    _write_depth_png(depth_png, np.array([[1000, 2000], [3000, 0]], dtype=np.uint16))
    depth = read_depth_meters(depth_png)
    assert depth.shape == (2, 2)
    assert depth[0, 0] == pytest.approx(1.0)
    assert depth[1, 0] == pytest.approx(3.0)


def test_read_camera_matrix_reshapes(tmp_path: Path) -> None:
    intrinsic_txt = tmp_path / "000000-intrinsic.txt"
    intrinsic_txt.write_text(
        "100 0 50\n0 100 60\n0 0 1\n", encoding="utf-8"
    )
    matrix = read_camera_matrix(
        intrinsic_txt, expected_shape=(3, 3), field_name="intrinsics"
    )
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(100.0)


def test_read_camera_matrix_rejects_wrong_size(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ToolInputError):
        read_camera_matrix(bad, expected_shape=(3, 3), field_name="intrinsics")


def test_load_camera_geometry_builds_validated_geometry(tmp_path: Path) -> None:
    (tmp_path / "000000-intrinsic.txt").write_text(
        "100 0 50\n0 100 60\n0 0 1\n", encoding="utf-8"
    )
    (tmp_path / "000000.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8"
    )
    geometry = load_camera_geometry(
        intrinsics_path=tmp_path / "000000-intrinsic.txt",
        pose_path=tmp_path / "000000.txt",
    )
    assert isinstance(geometry, CameraGeometry)
    assert geometry.intrinsics[1, 1] == pytest.approx(100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_camera_io.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.backends.camera_io`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/backends/camera_io.py
"""Shared depth-image and camera-matrix readers for SceneFunc3D backends.

Extracted from ``tools/mask_lifting.py`` so the multi-view pipeline and the
lift tool read camera geometry through one validated path (DRY). Depth PNGs are
16-bit millimetre ARKit depth; matrices are whitespace-delimited text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry, FloatArray
from codex_agent.scenefunc3d.tools.models import ToolInputError

_DEPTH_MILLIMETRES_PER_METRE = 1000.0


def read_depth_meters(depth_path: Path) -> FloatArray:
    """Read a single-channel 16-bit depth PNG as float64 metres."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "numpy and Pillow are required to read depth images; install the "
            "'vision' extra"
        ) from exc

    try:
        with Image.open(depth_path) as image:
            depth_pixels = np.asarray(image, dtype=np.float64)
    except OSError as exc:
        raise ToolInputError(
            "could not read depth image: "
            f"path={depth_path}; error_type={exc.__class__.__name__}"
        ) from exc
    except ValueError as exc:
        raise ToolInputError(
            "could not parse depth image: "
            f"path={depth_path}; error_type={exc.__class__.__name__}"
        ) from exc

    if depth_pixels.ndim != 2:
        raise ToolInputError(
            "depth image must be a single-channel 2D image: "
            f"path={depth_path}; shape={depth_pixels.shape}"
        )
    depth_meters: FloatArray = depth_pixels / _DEPTH_MILLIMETRES_PER_METRE
    return depth_meters


def read_camera_matrix(
    matrix_path: Path,
    *,
    expected_shape: tuple[int, int],
    field_name: str,
) -> FloatArray:
    """Read a whitespace-delimited matrix and reshape to ``expected_shape``."""
    expected_size = expected_shape[0] * expected_shape[1]
    try:
        raw_matrix = np.loadtxt(matrix_path, dtype=np.float64)
        matrix = np.asarray(raw_matrix, dtype=np.float64)
    except OSError as exc:
        raise ToolInputError(
            "could not read camera matrix file: "
            f"field={field_name}; path={matrix_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    except ValueError as exc:
        raise ToolInputError(
            "could not parse camera matrix file: "
            f"field={field_name}; path={matrix_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    if matrix.size != expected_size:
        raise ToolInputError(
            "camera matrix has wrong element count: "
            f"field={field_name}; path={matrix_path}; "
            f"expected={expected_size}; actual={matrix.size}"
        )
    reshaped_matrix: FloatArray = matrix.reshape(expected_shape)
    return reshaped_matrix


def load_camera_geometry(
    *, intrinsics_path: Path, pose_path: Path
) -> CameraGeometry:
    """Load intrinsics + camera-to-world pose into a validated ``CameraGeometry``."""
    intrinsics = read_camera_matrix(
        intrinsics_path, expected_shape=(3, 3), field_name="intrinsics"
    )
    camera_to_world = read_camera_matrix(
        pose_path, expected_shape=(4, 4), field_name="camera_to_world"
    )
    try:
        return CameraGeometry(
            intrinsics=intrinsics, camera_to_world=camera_to_world
        )
    except ToolInputError as exc:
        raise ToolInputError(
            "invalid camera geometry: "
            f"intrinsics_path={intrinsics_path}; pose_path={pose_path}; error={exc}"
        ) from exc


__all__ = ["load_camera_geometry", "read_camera_matrix", "read_depth_meters"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_camera_io.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: DRY-refactor `mask_lifting.py` to reuse `camera_io`**

In `src/codex_agent/scenefunc3d/tools/mask_lifting.py`:
1. Add near the top-level imports (module scope):

```python
from codex_agent.scenefunc3d.backends.camera_io import (
    load_camera_geometry,
    read_camera_matrix,
    read_depth_meters,
)
```

2. In `lift_mask_to_3d`, replace the depth/intrinsics/pose/geometry block:

```python
    mask = load_mask_npz(args.mask_path)
    depth_meters = read_depth_meters(args.depth_path)
    geometry = load_camera_geometry(
        intrinsics_path=args.intrinsics_path,
        pose_path=args.pose_path,
    )
    points_world = backproject_mask_to_world(mask, depth_meters, geometry)
```

   (Delete the local `intrinsics = _read_matrix(...)`, `camera_to_world = _read_matrix(...)`,
   the `try/except CameraGeometry(...)` block, and drop `CameraGeometry` from the local
   `from ...lift_3d import (...)`.)

3. Delete the now-unused private helpers `_read_depth_meters` and `_read_matrix` from
   `mask_lifting.py`. Keep `_read_matrix`'s callers: there are none other than lifting (verify
   with `rg "_read_matrix|_read_depth_meters" src/`).

- [ ] **Step 6: Verify the refactor keeps lifting green**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_mask_lifting.py -q`
(If that file does not exist, run the lifting-related tests: `PYTHONPATH=src pytest src/codex_agent/tests/ -k "lift" -q`.)
Expected: PASS (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/camera_io.py \
        src/codex_agent/scenefunc3d/tools/mask_lifting.py \
        src/codex_agent/tests/test_scenefunc3d_camera_io.py
git commit -m "refactor(scenefunc3d): extract shared camera/depth IO from mask lifting"
```

---

## Task 2: Synthetic scene test helper + frame loader (`frame_loader.py`)

**Files:**
- Create: `src/codex_agent/tests/scenefunc3d_synthetic_scene.py` (importable helper, not a test)
- Create: `src/codex_agent/scenefunc3d/backends/frame_loader.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_frame_loader.py`

- [ ] **Step 1: Write the synthetic-scene helper**

```python
# src/codex_agent/tests/scenefunc3d_synthetic_scene.py
"""Write a tiny on-disk SceneFunc3D scene for pipeline/loader tests.

Produces the exact raw layout the loaders expect: per-frame ``-rgb.jpg``,
``-depth.png`` (16-bit mm), ``-intrinsic.txt`` (3x3), ``<id>.txt`` (4x4 pose),
a binary-little-endian ``mesh.ply``, and ``source_frames.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene


@dataclass(frozen=True)
class SyntheticFrame:
    """One synthetic frame's pose and constant depth (metres)."""

    frame_id: str
    camera_to_world: np.ndarray
    depth_value_m: float


def write_synthetic_scene(
    scene_root: Path,
    *,
    frames: tuple[SyntheticFrame, ...],
    vertices_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] = (40, 40),
) -> SceneFunc3dToolScene:
    """Write a minimal scene and return the loaded ``SceneFunc3dToolScene``."""
    raw_dir = scene_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    for frame in frames:
        Image.new("RGB", (width, height), (10, 20, 30)).save(
            raw_dir / f"{frame.frame_id}-rgb.jpg"
        )
        depth_mm = np.full(
            (height, width), int(round(frame.depth_value_m * 1000.0)), dtype=np.uint16
        )
        Image.fromarray(depth_mm).save(raw_dir / f"{frame.frame_id}-depth.png")
        _write_matrix(raw_dir / f"{frame.frame_id}-intrinsic.txt", intrinsics)
        _write_matrix(raw_dir / f"{frame.frame_id}.txt", frame.camera_to_world)
    _write_binary_ply(raw_dir / "mesh.ply", vertices_world)
    _write_source_frames_json(raw_dir / "source_frames.json", frames)
    return SceneFunc3dToolScene.load(scene_root)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    rows = "\n".join(
        " ".join(f"{value:.10f}" for value in row) for row in np.asarray(matrix)
    )
    path.write_text(rows + "\n", encoding="utf-8")


def _write_source_frames_json(path: Path, frames: tuple[SyntheticFrame, ...]) -> None:
    import json

    payload = [
        {
            "frame_id": frame.frame_id,
            "rgb": f"{frame.frame_id}-rgb.jpg",
            "depth": f"{frame.frame_id}-depth.png",
            "intrinsic": f"{frame.frame_id}-intrinsic.txt",
            "pose": f"{frame.frame_id}.txt",
        }
        for frame in frames
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_binary_ply(path: Path, vertices_world: np.ndarray) -> None:
    vertices = np.asarray(vertices_world, dtype=np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes(order="C"))


__all__ = ["SyntheticFrame", "write_synthetic_scene"]
```

- [ ] **Step 2: Write the failing frame-loader test**

```python
# src/codex_agent/tests/test_scenefunc3d_frame_loader.py
"""Tests for the SceneFunc3D per-frame camera/depth loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.frame_loader import (
    frame_rgb_path,
    iter_frame_geometry,
    load_frame_geometry,
    read_frame_depth,
)
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.errors import SceneFunc3dDataError
from codex_agent.tests.scenefunc3d_synthetic_scene import (
    SyntheticFrame,
    write_synthetic_scene,
)

_INTRINSICS = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])


def _scene(tmp_path: Path) -> object:
    frames = (
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        SyntheticFrame("000001", np.eye(4), depth_value_m=3.0),
    )
    return write_synthetic_scene(
        tmp_path / "scene",
        frames=frames,
        vertices_world=np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        intrinsics=_INTRINSICS,
    )


def test_iter_frame_geometry_yields_all_frames(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    items = list(iter_frame_geometry(scene))
    assert [frame_id for frame_id, _geometry in items] == ["000000", "000001"]
    assert all(isinstance(g, CameraGeometry) for _f, g in items)


def test_load_frame_geometry_reads_one_frame(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    geometry = load_frame_geometry(scene, "000001")
    assert geometry.intrinsics[0, 0] == pytest.approx(20.0)


def test_read_frame_depth_scales_metres(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    depth = read_frame_depth(scene, "000000")
    assert depth.shape == (40, 40)
    assert float(depth[0, 0]) == pytest.approx(2.0)


def test_frame_rgb_path_exists(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    rgb = frame_rgb_path(scene, "000000")
    assert rgb.is_file()
    assert rgb.name == "000000-rgb.jpg"


def test_iter_frame_geometry_fails_closed_when_no_geometry(tmp_path: Path) -> None:
    scene_root = tmp_path / "empty"
    (scene_root / "raw").mkdir(parents=True)
    (scene_root / "raw" / "000000-rgb.jpg").write_bytes(b"")
    (scene_root / "raw" / "source_frames.json").write_text(
        '[{"frame_id": "000000", "rgb": "000000-rgb.jpg"}]', encoding="utf-8"
    )
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

    scene = SceneFunc3dToolScene.load(scene_root)
    with pytest.raises(SceneFunc3dDataError):
        list(iter_frame_geometry(scene))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_frame_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.backends.frame_loader`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/backends/frame_loader.py
"""Lazy per-frame camera/depth/RGB access for a prepared SceneFunc3D scene.

Depth is read one frame at a time (never all frames at once): a single depth
array is ~22 MB at 1440x1920, so materialising all 170 frames would need
~3.7 GB. Callers stream frames and discard depth after use.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.backends.camera_io import (
    load_camera_geometry,
    read_depth_meters,
)
from codex_agent.scenefunc3d.backends.frame_assets import (
    resolve_available_frame_geometry_assets,
    resolve_frame_geometry_assets,
)
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry, FloatArray
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

_RGB_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def iter_frame_geometry(
    scene: SceneFunc3dToolScene,
) -> Iterator[tuple[str, CameraGeometry]]:
    """Yield ``(frame_id, CameraGeometry)`` for every frame with full geometry.

    Frames missing any of depth/intrinsics/pose are skipped. Fails closed when
    no frame in the scene has complete geometry, so downstream stages never
    silently see an empty frame set.
    """
    yielded = 0
    for frame_id in scene.rgb_frame_ids:
        assets = resolve_available_frame_geometry_assets(scene, frame_id)
        if (
            assets.depth_path is None
            or assets.intrinsics_path is None
            or assets.pose_path is None
        ):
            continue
        geometry = load_camera_geometry(
            intrinsics_path=assets.intrinsics_path,
            pose_path=assets.pose_path,
        )
        yielded += 1
        yield frame_id, geometry
    if yielded == 0:
        raise SceneFunc3dDataError(
            "no SceneFunc3D frame has complete depth/intrinsics/pose geometry: "
            f"scene_root={scene.scene_root}"
        )


def load_frame_geometry(
    scene: SceneFunc3dToolScene, frame_id: str
) -> CameraGeometry:
    """Load one frame's validated camera geometry."""
    assets = resolve_frame_geometry_assets(scene, frame_id)
    return load_camera_geometry(
        intrinsics_path=assets.intrinsics_path,
        pose_path=assets.pose_path,
    )


def read_frame_depth(scene: SceneFunc3dToolScene, frame_id: str) -> FloatArray:
    """Read one frame's depth image as float64 metres."""
    assets = resolve_frame_geometry_assets(scene, frame_id)
    return read_depth_meters(assets.depth_path)


def frame_rgb_path(scene: SceneFunc3dToolScene, frame_id: str) -> Path:
    """Resolve one frame's RGB image path, failing closed when absent."""
    indexed = scene.source_frame_raw_rgb_path(frame_id)
    if indexed is not None and indexed.is_file():
        return indexed
    for suffix in _RGB_SUFFIXES:
        candidate = scene.raw_dir / f"{frame_id}-rgb{suffix}"
        if candidate.is_file():
            return candidate
    raise ToolInputError(
        "frame RGB image is missing: "
        f"frame_id={frame_id!r}; raw_dir={scene.raw_dir}"
    )


__all__ = [
    "frame_rgb_path",
    "iter_frame_geometry",
    "load_frame_geometry",
    "read_frame_depth",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_frame_loader.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/frame_loader.py \
        src/codex_agent/tests/scenefunc3d_synthetic_scene.py \
        src/codex_agent/tests/test_scenefunc3d_frame_loader.py
git commit -m "feat(scenefunc3d): add lazy per-frame camera/depth/RGB loader"
```

---

## Task 3: Streaming scene-level frame selection (`visibility.py`)

Adds `select_scene_visible_frames`: stream every frame's geometry, load its depth **one at a
time**, score anchor visibility (reusing Phase-2a `score_anchor_visibility`), keep the top-N,
and **always include the anchor's seed frame** (spec: degrade to single-frame when only the
seed sees the anchor).

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/visibility.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_visibility.py` (append)

- [ ] **Step 1: Write the failing test** — append:

```python
from codex_agent.scenefunc3d.backends.visibility import select_scene_visible_frames
from codex_agent.tests.scenefunc3d_synthetic_scene import (
    SyntheticFrame,
    write_synthetic_scene,
)


def _pipeline_scene(tmp_path):
    intrinsics = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])
    frames = (
        # 000000 sees the anchor at depth 2.0 (unoccluded).
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        # 000001 has a near wall at 1.0 -> anchor at z=2.0 is occluded.
        SyntheticFrame("000001", np.eye(4), depth_value_m=1.0),
    )
    return write_synthetic_scene(
        tmp_path / "scene",
        frames=frames,
        vertices_world=np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0]]),
        intrinsics=intrinsics,
    )


def test_select_scene_visible_frames_streams_and_ranks(tmp_path) -> None:
    scene = _pipeline_scene(tmp_path)
    anchor = build_anchor(
        np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0], [-0.02, 0.0, 2.0]]),
        motion_type="pinch_pull",
        seed_frame_id="000000",
    )
    selected = select_scene_visible_frames(anchor, scene, frame_cap=10)
    assert "000000" in {v.frame_id for v in selected}
    # The occluded frame must be dropped unless it is the seed frame.
    assert "000001" not in {v.frame_id for v in selected}


def test_select_scene_visible_frames_always_includes_seed(tmp_path) -> None:
    scene = _pipeline_scene(tmp_path)
    # Seed is the occluded frame; it must still appear (single-frame degrade).
    anchor = build_anchor(
        np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0], [-0.02, 0.0, 2.0]]),
        motion_type="pinch_pull",
        seed_frame_id="000001",
    )
    selected = select_scene_visible_frames(anchor, scene, frame_cap=10)
    assert "000001" in {v.frame_id for v in selected}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -k scene -v`
Expected: FAIL with `ImportError: cannot import name 'select_scene_visible_frames'`.

- [ ] **Step 3: Write minimal implementation** — add to `visibility.py` (import the loader lazily inside the function to avoid an import cycle risk, add name to `__all__`):

```python
def select_scene_visible_frames(
    anchor: TargetAnchor,
    scene: "SceneFunc3dToolScene",
    *,
    frame_cap: int,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> tuple[FrameVisibility, ...]:
    """Stream a scene's frames, score anchor visibility, return top-N + seed.

    Depth is read one frame at a time and discarded to bound memory. The
    anchor's ``seed_frame_id`` is always included (when it has geometry) so a
    fully-occluded scene degrades to the single-frame seed rather than an empty
    selection.
    """
    if frame_cap <= 0:
        raise ValueError(f"frame_cap must be positive: {frame_cap!r}")

    from codex_agent.scenefunc3d.backends.frame_loader import (
        iter_frame_geometry,
        load_frame_geometry,
        read_frame_depth,
    )

    scored: list[FrameVisibility] = []
    seen_frame_ids: set[str] = set()
    for frame_id, geometry in iter_frame_geometry(scene):
        depth = read_frame_depth(scene, frame_id)
        scored.append(
            score_anchor_visibility(
                frame_id, anchor, geometry, depth, depth_tolerance=depth_tolerance
            )
        )
        seen_frame_ids.add(frame_id)

    visible = [result for result in scored if result.visible]
    visible.sort(key=lambda result: (-result.quality_score, result.frame_id))
    selected = list(visible[:frame_cap])
    selected_ids = {result.frame_id for result in selected}

    if anchor.seed_frame_id not in selected_ids:
        seed_score = next(
            (r for r in scored if r.frame_id == anchor.seed_frame_id), None
        )
        if seed_score is not None:
            selected.append(seed_score)
        elif anchor.seed_frame_id not in seen_frame_ids:
            # Seed frame absent from the streamed set: score it directly so the
            # pipeline can still process it (fail-closed handled by loaders).
            seed_geometry = load_frame_geometry(scene, anchor.seed_frame_id)
            seed_depth = read_frame_depth(scene, anchor.seed_frame_id)
            selected.append(
                score_anchor_visibility(
                    anchor.seed_frame_id,
                    anchor,
                    seed_geometry,
                    seed_depth,
                    depth_tolerance=depth_tolerance,
                )
            )
    return tuple(selected)
```

Also add the `TYPE_CHECKING` import for the annotation at the top of `visibility.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: PASS (13 passed — 11 Phase-2a + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/visibility.py \
        src/codex_agent/tests/test_scenefunc3d_visibility.py
git commit -m "feat(scenefunc3d): stream scene-level anchor-visible frame selection"
```

---

## Task 4: Per-vertex multi-view visibility counts (`visibility.py`)

Adds `count_vertex_visibility`: for a set of candidate raw-mesh vertices, count in how many of
the given frames each vertex is geometrically visible (in-front + in-bounds + unoccluded). This
is the denominator that `fusion.agreement_scores` needs (`agreement = hits / visibility`),
closing the "agreement is weak offline" gap.

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/visibility.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_visibility.py` (append)

- [ ] **Step 1: Write the failing test** — append:

```python
from codex_agent.scenefunc3d.backends.visibility import count_vertex_visibility


def test_count_vertex_visibility_counts_unoccluded_frames(tmp_path) -> None:
    scene = _pipeline_scene(tmp_path)
    # Vertex 0 at z=2.0 is visible in 000000 (depth 2.0) but occluded in 000001
    # (near wall at 1.0). Vertex ids are arbitrary raw-mesh ids for the counter.
    vertex_ids = [7, 9]
    vertex_coords = np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0]])
    counts = count_vertex_visibility(
        vertex_ids, vertex_coords, scene, ("000000", "000001")
    )
    assert counts[7] == 1
    assert counts[9] == 1


def test_count_vertex_visibility_rejects_length_mismatch(tmp_path) -> None:
    scene = _pipeline_scene(tmp_path)
    with pytest.raises(ValueError):
        count_vertex_visibility([1], np.zeros((2, 3)), scene, ("000000",))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -k count_vertex -v`
Expected: FAIL with `ImportError: cannot import name 'count_vertex_visibility'`.

- [ ] **Step 3: Write minimal implementation** — add to `visibility.py` (add name to `__all__`; import `Sequence` is already present):

```python
def count_vertex_visibility(
    vertex_ids: Sequence[int],
    vertex_coords: FloatArray,
    scene: "SceneFunc3dToolScene",
    frame_ids: Sequence[str],
    *,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> dict[int, int]:
    """Count, per candidate vertex, how many ``frame_ids`` see it un-occluded.

    Reads each frame's depth once (streamed, discarded) and tests all vertices
    against it. ``vertex_ids[k]`` labels row ``k`` of ``vertex_coords``.
    """
    coords = np.asarray(vertex_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"vertex_coords must have shape (N, 3): {coords.shape}")
    if len(vertex_ids) != coords.shape[0]:
        raise ValueError(
            "vertex_ids length must match vertex_coords rows: "
            f"ids={len(vertex_ids)}; rows={coords.shape[0]}"
        )
    if coords.shape[0] == 0:
        return {}

    from codex_agent.scenefunc3d.backends.frame_loader import (
        load_frame_geometry,
        read_frame_depth,
    )

    counts = {int(vertex_id): 0 for vertex_id in vertex_ids}
    for frame_id in frame_ids:
        geometry = load_frame_geometry(scene, frame_id)
        depth = read_frame_depth(scene, frame_id)
        visible = point_visibility(
            coords, geometry, depth, depth_tolerance=depth_tolerance
        )
        for row, vertex_id in enumerate(vertex_ids):
            if bool(visible[row]):
                counts[int(vertex_id)] += 1
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_visibility.py -v`
Expected: PASS (15 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/backends/visibility.py \
        src/codex_agent/tests/test_scenefunc3d_visibility.py
git commit -m "feat(scenefunc3d): add per-vertex multi-view visibility counts"
```

---

## Task 5: Pure pipeline driver (`pipeline.py`, `fuse_selected_frames`)

The **pure** Stage C+D driver: given the frozen anchor, `scene_vertices`, the selected frames'
`FrameVisibility` + a `frame_id -> CameraGeometry` map, a `FrameProposer` (injected), and a
visibility-counter callable (injected), it projects the anchor into each frame, calls the
proposer, applies the **anchor-consistency gate**, builds the `MultiViewLiftBundle`, and calls
`fuse_multiview_points`. No disk, no sidecar → unit-tested with fakes.

**Files:**
- Create: `src/codex_agent/scenefunc3d/pipeline.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_pipeline.py
"""Tests for the pure anchor multi-view pipeline driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry, FloatArray
from codex_agent.scenefunc3d.backends.visibility import FrameVisibility
from codex_agent.scenefunc3d.pipeline import (
    FrameProposal,
    fuse_selected_frames,
)

_INTRINSICS = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])


def _geometry() -> CameraGeometry:
    return CameraGeometry(intrinsics=_INTRINSICS, camera_to_world=np.eye(4))


class _FakeProposer:
    """Returns pre-baked point ids per frame; records projected anchors seen."""

    def __init__(self, per_frame: Mapping[str, tuple[int, ...]]) -> None:
        self._per_frame = per_frame
        self.seen_projected: list[tuple[float, float] | None] = []

    def propose(
        self,
        *,
        frame_id: str,
        geometry: CameraGeometry,
        projected_anchor_xy: tuple[float, float] | None,
    ) -> FrameProposal:
        self.seen_projected.append(projected_anchor_xy)
        return FrameProposal(
            frame_id=frame_id,
            point_indices=self._per_frame.get(frame_id, ()),
            molmo_fallback_used=False,
        )


def _all_visible_counts(
    vertex_ids: Sequence[int], vertex_coords: FloatArray, frame_ids: Sequence[str]
) -> Mapping[int, int]:
    return {int(v): len(frame_ids) for v in vertex_ids}


def test_pipeline_fuses_consensus_target_and_drops_off_anchor_frame() -> None:
    # 5 tightly-clustered target vertices near the anchor, 1 far wall vertex.
    scene_vertices = np.array(
        [
            [0.00, 0.0, 2.0],
            [0.01, 0.0, 2.0],
            [0.00, 0.01, 2.0],
            [0.01, 0.01, 2.0],
            [0.005, 0.005, 2.0],
            [3.00, 3.0, 2.0],  # id 5: far wall
        ]
    )
    anchor = build_anchor(
        scene_vertices[:5], motion_type="pinch_pull", seed_frame_id="000000"
    )
    selected = (
        FrameVisibility("000000", True, 1.0, 1.0, 1.0),
        FrameVisibility("000001", True, 1.0, 1.0, 1.0),
        FrameVisibility("000002", True, 1.0, 1.0, 1.0),
    )
    geometry_by_frame = {fid: _geometry() for fid in ("000000", "000001", "000002")}
    proposer = _FakeProposer(
        {
            "000000": (0, 1, 2, 3, 4),
            "000001": (0, 1, 2, 3, 4),
            "000002": (5,),  # off-anchor -> must be gated out
        }
    )
    result = fuse_selected_frames(
        anchor=anchor,
        scene_vertices=scene_vertices,
        selected_frames=selected,
        geometry_by_frame=geometry_by_frame,
        proposer=proposer,
        params=FusionParams(agreement_tau=0.5, min_cluster_points=3),
        visibility_counts_fn=_all_visible_counts,
    )
    assert set(result.fused.point_indices) == {0, 1, 2, 3, 4}
    rejected = {o.frame_id for o in result.per_frame if not o.accepted}
    assert rejected == {"000002"}
    # The anchor was projected to the principal point (20, 20).
    assert proposer.seen_projected[0] == pytest.approx((20.0, 20.0))


def test_pipeline_returns_empty_when_no_frame_accepted() -> None:
    scene_vertices = np.array([[0.0, 0.0, 2.0], [5.0, 5.0, 2.0]])
    anchor = build_anchor(
        np.array([[0.0, 0.0, 2.0]]), motion_type="rotate", seed_frame_id="000000"
    )
    selected = (FrameVisibility("000000", True, 1.0, 1.0, 1.0),)
    result = fuse_selected_frames(
        anchor=anchor,
        scene_vertices=scene_vertices,
        selected_frames=selected,
        geometry_by_frame={"000000": _geometry()},
        proposer=_FakeProposer({"000000": (1,)}),  # off-anchor only
        params=FusionParams(),
        visibility_counts_fn=_all_visible_counts,
    )
    assert result.fused.point_indices == ()
    assert all(not o.accepted for o in result.per_frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.pipeline`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/pipeline.py
"""Deterministic anchor-centric multi-view mask pipeline (Stages B/C/D).

The pure driver ``fuse_selected_frames`` depends only on injected callables
(a ``FrameProposer`` and a visibility counter) plus in-memory geometry, so it is
unit-testable without a scene, sidecar, or the ModelHub adapter. The scene
wrapper ``run_anchor_multiview_pipeline`` wires the disk/sidecar-backed
implementations (frame selection, geometry loading, per-vertex visibility).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor
from codex_agent.scenefunc3d.backends.fusion import (
    FrameLift,
    FusedMask,
    FusionParams,
    MultiViewLiftBundle,
    fuse_multiview_points,
)
from codex_agent.scenefunc3d.backends.lift_3d import FloatArray
from codex_agent.scenefunc3d.backends.visibility import (
    FrameVisibility,
    project_world_to_pixels,
)

VisibilityCounter = "Callable[[Sequence[int], FloatArray, Sequence[str]], Mapping[int, int]]"


@dataclass(frozen=True)
class FrameProposal:
    """One frame's proposed lift: raw-mesh vertex ids + Molmo-fallback flag."""

    frame_id: str
    point_indices: tuple[int, ...]
    molmo_fallback_used: bool


class FrameProposer(Protocol):
    """Proposes lifted raw-mesh vertex ids for one frame (Molmo->SAM->lift)."""

    def propose(
        self,
        *,
        frame_id: str,
        geometry: "object",
        projected_anchor_xy: tuple[float, float] | None,
    ) -> FrameProposal:
        """Return the frame's lifted vertex ids (empty tuple when it found none)."""
        ...


@dataclass(frozen=True)
class PerFrameOutcome:
    """Provenance for one processed frame."""

    frame_id: str
    point_indices: tuple[int, ...]
    centroid: tuple[float, float, float] | None
    accepted: bool
    reject_reason: str
    molmo_fallback_used: bool


@dataclass(frozen=True)
class AnchorMultiViewResult:
    """Fused mask plus per-frame provenance for one sample."""

    fused: FusedMask
    selected_frames: tuple[FrameVisibility, ...]
    per_frame: tuple[PerFrameOutcome, ...]


def _project_anchor_xy(
    anchor: TargetAnchor, geometry: object
) -> tuple[float, float] | None:
    from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry

    assert isinstance(geometry, CameraGeometry)
    pixels, camera_z = project_world_to_pixels(
        np.asarray([anchor.centroid], dtype=np.float64), geometry
    )
    u, v = float(pixels[0, 0]), float(pixels[0, 1])
    if camera_z[0] <= 0.0 or not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (u, v)


def fuse_selected_frames(
    *,
    anchor: TargetAnchor,
    scene_vertices: FloatArray,
    selected_frames: Sequence[FrameVisibility],
    geometry_by_frame: Mapping[str, object],
    proposer: FrameProposer,
    params: FusionParams,
    visibility_counts_fn: object,
    gate_radius_scale: float = 1.0,
) -> AnchorMultiViewResult:
    """Run Stage C (per-frame propose + anchor gate) and Stage D (fusion)."""
    vertices = np.asarray(scene_vertices, dtype=np.float64)
    centroid = np.asarray(anchor.centroid, dtype=np.float64)
    gate_radius = anchor.radius_m * gate_radius_scale

    per_frame: list[PerFrameOutcome] = []
    accepted_lifts: list[FrameLift] = []
    for frame_vis in selected_frames:
        geometry = geometry_by_frame[frame_vis.frame_id]
        projected_xy = _project_anchor_xy(anchor, geometry)
        proposal = proposer.propose(
            frame_id=frame_vis.frame_id,
            geometry=geometry,
            projected_anchor_xy=projected_xy,
        )
        outcome = _evaluate_proposal(
            proposal, vertices=vertices, centroid=centroid, gate_radius=gate_radius
        )
        per_frame.append(outcome)
        if outcome.accepted:
            accepted_lifts.append(
                FrameLift(
                    frame_id=proposal.frame_id,
                    point_indices=proposal.point_indices,
                )
            )

    if not accepted_lifts:
        empty = fuse_multiview_points(
            MultiViewLiftBundle(
                frames=(FrameLift(frame_id="__none__", point_indices=()),),
                anchor=anchor,
            ),
            vertices,
            params,
        )
        return AnchorMultiViewResult(
            fused=empty,
            selected_frames=tuple(selected_frames),
            per_frame=tuple(per_frame),
        )

    union_ids = sorted({index for lift in accepted_lifts for index in lift.point_indices})
    union_coords = vertices[union_ids]
    accepted_frame_ids = [lift.frame_id for lift in accepted_lifts]
    counts = visibility_counts_fn(union_ids, union_coords, accepted_frame_ids)  # type: ignore[operator]
    bundle = MultiViewLiftBundle(
        frames=tuple(accepted_lifts),
        anchor=anchor,
        visibility_counts=dict(counts),
    )
    fused = fuse_multiview_points(bundle, vertices, params)
    return AnchorMultiViewResult(
        fused=fused,
        selected_frames=tuple(selected_frames),
        per_frame=tuple(per_frame),
    )


def _evaluate_proposal(
    proposal: FrameProposal,
    *,
    vertices: FloatArray,
    centroid: FloatArray,
    gate_radius: float,
) -> PerFrameOutcome:
    if not proposal.point_indices:
        return PerFrameOutcome(
            frame_id=proposal.frame_id,
            point_indices=(),
            centroid=None,
            accepted=False,
            reject_reason="empty_lift",
            molmo_fallback_used=proposal.molmo_fallback_used,
        )
    frame_centroid = vertices[list(proposal.point_indices)].mean(axis=0)
    distance = float(np.linalg.norm(frame_centroid - centroid))
    accepted = distance <= gate_radius
    return PerFrameOutcome(
        frame_id=proposal.frame_id,
        point_indices=proposal.point_indices,
        centroid=(
            float(frame_centroid[0]),
            float(frame_centroid[1]),
            float(frame_centroid[2]),
        ),
        accepted=accepted,
        reject_reason="" if accepted else "anchor_gate",
        molmo_fallback_used=proposal.molmo_fallback_used,
    )


__all__ = [
    "AnchorMultiViewResult",
    "FrameProposal",
    "FrameProposer",
    "PerFrameOutcome",
    "fuse_selected_frames",
]
```

Note: `VisibilityCounter` is documented as a string alias only; the parameter is typed
`object` and called positionally to keep the pure driver import-light. In Task 6 the scene
wrapper passes a concrete function; mypy will still check that function's own signature.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/pipeline.py \
        src/codex_agent/tests/test_scenefunc3d_pipeline.py
git commit -m "feat(scenefunc3d): add pure anchor multi-view fusion driver"
```

---

## Task 6: Scene wrapper (`run_anchor_multiview_pipeline`)

Wires the disk-backed pieces: `select_scene_visible_frames` (Task 3) → build a
`frame_id -> CameraGeometry` map for the selected frames (Task 2) → bind
`count_vertex_visibility` (Task 4) as the visibility counter → call `fuse_selected_frames`
(Task 5). Integration-tested with the synthetic scene + fake proposer.

**Files:**
- Modify: `src/codex_agent/scenefunc3d/pipeline.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_pipeline.py` (append)

- [ ] **Step 1: Write the failing test** — append:

```python
from codex_agent.scenefunc3d.backends.lift_3d import load_scene_mesh_vertices
from codex_agent.scenefunc3d.pipeline import run_anchor_multiview_pipeline
from codex_agent.tests.scenefunc3d_synthetic_scene import (
    SyntheticFrame,
    write_synthetic_scene,
)


def test_run_anchor_multiview_pipeline_end_to_end(tmp_path) -> None:
    intrinsics = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])
    # 5 target vertices near (0,0,2) + 1 far wall vertex.
    vertices = np.array(
        [
            [0.00, 0.0, 2.0],
            [0.01, 0.0, 2.0],
            [0.00, 0.01, 2.0],
            [0.01, 0.01, 2.0],
            [0.005, 0.005, 2.0],
            [3.00, 3.0, 2.0],
        ]
    )
    frames = (
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        SyntheticFrame("000001", np.eye(4), depth_value_m=2.0),
    )
    scene = write_synthetic_scene(
        tmp_path / "scene", frames=frames, vertices_world=vertices, intrinsics=intrinsics
    )
    scene_vertices = load_scene_mesh_vertices(scene.raw_mesh_path)
    anchor = build_anchor(
        vertices[:5], motion_type="pinch_pull", seed_frame_id="000000"
    )
    result = run_anchor_multiview_pipeline(
        anchor=anchor,
        scene=scene,
        scene_vertices=scene_vertices,
        proposer=_FakeProposer({"000000": (0, 1, 2, 3, 4), "000001": (0, 1, 2, 3, 4)}),
        params=FusionParams(agreement_tau=0.5, min_cluster_points=3),
        frame_cap=10,
    )
    assert set(result.fused.point_indices) == {0, 1, 2, 3, 4}
    assert result.fused.confidence == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline.py -k end_to_end -v`
Expected: FAIL with `ImportError: cannot import name 'run_anchor_multiview_pipeline'`.

- [ ] **Step 3: Write minimal implementation** — add to `pipeline.py` (add name to `__all__`):

```python
_DEFAULT_FRAME_CAP = 30


def run_anchor_multiview_pipeline(
    *,
    anchor: TargetAnchor,
    scene: object,
    scene_vertices: FloatArray,
    proposer: FrameProposer,
    params: FusionParams,
    frame_cap: int = _DEFAULT_FRAME_CAP,
    depth_tolerance: float = 0.25,
    gate_radius_scale: float = 1.0,
) -> AnchorMultiViewResult:
    """Scene-backed Stage B->C->D: select frames, propose, gate, fuse."""
    from codex_agent.scenefunc3d.backends.frame_loader import load_frame_geometry
    from codex_agent.scenefunc3d.backends.visibility import (
        count_vertex_visibility,
        select_scene_visible_frames,
    )
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

    assert isinstance(scene, SceneFunc3dToolScene)
    selected = select_scene_visible_frames(
        anchor, scene, frame_cap=frame_cap, depth_tolerance=depth_tolerance
    )
    geometry_by_frame = {
        frame_vis.frame_id: load_frame_geometry(scene, frame_vis.frame_id)
        for frame_vis in selected
    }

    def _counts(
        vertex_ids: Sequence[int],
        vertex_coords: FloatArray,
        frame_ids: Sequence[str],
    ) -> Mapping[int, int]:
        return count_vertex_visibility(
            vertex_ids,
            vertex_coords,
            scene,
            frame_ids,
            depth_tolerance=depth_tolerance,
        )

    return fuse_selected_frames(
        anchor=anchor,
        scene_vertices=scene_vertices,
        selected_frames=selected,
        geometry_by_frame=geometry_by_frame,
        proposer=proposer,
        params=params,
        visibility_counts_fn=_counts,
        gate_radius_scale=gate_radius_scale,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/pipeline.py \
        src/codex_agent/tests/test_scenefunc3d_pipeline.py
git commit -m "feat(scenefunc3d): wire scene-backed anchor multi-view pipeline"
```

---

## Task 7: Molmo response helper + sidecar proposer (`pipeline_backends.py`)

Two pieces:
1. A public `points_from_molmo_response` in `molmo_pointing.py` that returns parsed points or
   `()` for a **successful** response with no usable points (no overlay, no raise-on-empty).
   Transport/sidecar failures still propagate (they raise before a response exists).
2. `SidecarFrameProposer`: projects → Molmo (nearest point to projected anchor, else the
   projected anchor as an explicit, provenance-recorded fallback) → SAM (smallest candidate) →
   `lift_mask_to_3d` → reads the fragment's `point_indices` back.

**Files:**
- Modify: `src/codex_agent/scenefunc3d/tools/molmo_pointing.py` (add public helper)
- Create: `src/codex_agent/scenefunc3d/pipeline_backends.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_pipeline_backends.py`

- [ ] **Step 1: Write the failing test** (monkeypatches the RPC + SAM + lift; no real sidecar)

```python
# src/codex_agent/tests/test_scenefunc3d_pipeline_backends.py
"""Tests for the sidecar-backed per-frame proposer (Molmo->SAM->lift)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry


def _write_rgb(path: Path, size: tuple[int, int] = (40, 40)) -> None:
    Image.new("RGB", size, (0, 0, 0)).save(path)


def test_points_from_molmo_response_returns_empty_on_no_points() -> None:
    from codex_agent.scenefunc3d.servers.schemas import MolmoPointResponse
    from codex_agent.scenefunc3d.tools.molmo_pointing import (
        points_from_molmo_response,
    )

    response = MolmoPointResponse(request_id="r", raw_text="no tags", image_points=())
    assert points_from_molmo_response(response, image_width=40, image_height=40) == ()


def test_sidecar_proposer_picks_nearest_molmo_point_and_smallest_sam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.scenefunc3d import pipeline_backends
    from codex_agent.scenefunc3d.servers.schemas import (
        MolmoImagePoint,
        MolmoPointResponse,
    )
    from codex_agent.scenefunc3d.tools.molmo_pointing import MolmoPoint
    from codex_agent.scenefunc3d.tools.sam_masking import SamCandidate, SamMaskResult
    from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskResult

    scene_root = tmp_path / "scene"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    _write_rgb(raw / "000000-rgb.jpg")
    (raw / "000000-depth.png").write_bytes(b"")  # not read in this fake path
    (raw / "000000-intrinsic.txt").write_text("20 0 20\n0 20 20\n0 0 1\n")
    (raw / "000000.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "source_frames.json").write_text(
        '[{"frame_id": "000000", "rgb": "000000-rgb.jpg", "depth": "000000-depth.png",'
        ' "intrinsic": "000000-intrinsic.txt", "pose": "000000.txt"}]'
    )
    (raw / "mesh.ply").write_bytes(b"")  # raw_mesh_path passed through, not read here
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

    scene = SceneFunc3dToolScene.load(scene_root)

    # Molmo returns two points; the one nearest (20,20) must be chosen.
    def _fake_request_molmo_point(settings, *, request_id, args):
        return MolmoPointResponse(
            request_id=request_id,
            raw_text="",
            image_points=(
                MolmoImagePoint(x_px=21.0, y_px=20.0, source="molmo", label="a"),
                MolmoImagePoint(x_px=39.0, y_px=39.0, source="molmo", label="b"),
            ),
        )

    captured_points: list[tuple[float, float]] = []

    def _fake_sam_mask(args, *, out_dir, backend_config_path):
        captured_points.append((args.points[0].x_px, args.points[0].y_px))
        big = tmp_path / "big.npz"
        small = tmp_path / "small.npz"
        big.write_bytes(b"")
        small.write_bytes(b"")
        return SamMaskResult(
            frame_id="000000",
            candidates=(
                SamCandidate("cbig", 0.9, 500, 10.0, big, big),
                SamCandidate("csmall", 0.8, 50, 1.0, small, small),
            ),
            contact_sheet_path=tmp_path / "sheet.jpg",
        )

    def _fake_lift(args, *, out_dir, raw_mesh_path):
        npz = tmp_path / "fragments_000000_csmall.npz"
        np.savez_compressed(
            npz,
            points_world=np.array([[0.0, 0.0, 2.0]]),
            point_indices=np.array([3, 4, 5], dtype=np.int64),
        )
        return LiftMaskResult(
            frame_id="000000",
            candidate_id=args.candidate_id,
            lifted_point_count=3,
            mask_npz_path=npz,
            mask_ply_path=tmp_path / "x.ply",
            overlay_path=tmp_path / "x.txt",
        )

    monkeypatch.setattr(
        pipeline_backends, "request_molmo_point", _fake_request_molmo_point
    )
    monkeypatch.setattr(pipeline_backends, "sam_mask", _fake_sam_mask)
    monkeypatch.setattr(pipeline_backends, "lift_mask_to_3d", _fake_lift)
    monkeypatch.setattr(
        pipeline_backends, "load_backend_settings", lambda path: object()
    )

    proposer = pipeline_backends.SidecarFrameProposer(
        scene=scene,
        out_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backend.toml",
        affordance_concept="the handle",
        task_description="open the drawer",
    )
    geometry = CameraGeometry(
        intrinsics=np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]),
        camera_to_world=np.eye(4),
    )
    proposal = proposer.propose(
        frame_id="000000", geometry=geometry, projected_anchor_xy=(20.0, 20.0)
    )
    assert proposal.point_indices == (3, 4, 5)
    assert proposal.molmo_fallback_used is False
    # SAM must have been prompted with the nearest Molmo point (21, 20).
    assert captured_points[0] == pytest.approx((21.0, 20.0))


def test_sidecar_proposer_falls_back_to_projected_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.scenefunc3d import pipeline_backends
    from codex_agent.scenefunc3d.servers.schemas import MolmoPointResponse
    from codex_agent.scenefunc3d.tools.sam_masking import SamCandidate, SamMaskResult
    from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskResult

    scene_root = tmp_path / "scene"
    raw = scene_root / "raw"
    raw.mkdir(parents=True)
    _write_rgb(raw / "000000-rgb.jpg")
    (raw / "000000-depth.png").write_bytes(b"")
    (raw / "000000-intrinsic.txt").write_text("20 0 20\n0 20 20\n0 0 1\n")
    (raw / "000000.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    (raw / "source_frames.json").write_text(
        '[{"frame_id": "000000", "rgb": "000000-rgb.jpg", "depth": "000000-depth.png",'
        ' "intrinsic": "000000-intrinsic.txt", "pose": "000000.txt"}]'
    )
    (raw / "mesh.ply").write_bytes(b"")
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

    scene = SceneFunc3dToolScene.load(scene_root)

    captured_points: list[tuple[float, float]] = []

    monkeypatch.setattr(
        pipeline_backends,
        "request_molmo_point",
        lambda settings, *, request_id, args: MolmoPointResponse(
            request_id=request_id, raw_text="", image_points=()
        ),
    )

    def _fake_sam_mask(args, *, out_dir, backend_config_path):
        captured_points.append((args.points[0].x_px, args.points[0].y_px))
        npz = tmp_path / "m.npz"
        npz.write_bytes(b"")
        return SamMaskResult(
            frame_id="000000",
            candidates=(SamCandidate("c0", 0.8, 40, 1.0, npz, npz),),
            contact_sheet_path=tmp_path / "sheet.jpg",
        )

    def _fake_lift(args, *, out_dir, raw_mesh_path):
        npz = tmp_path / "frag.npz"
        np.savez_compressed(npz, point_indices=np.array([1], dtype=np.int64))
        return LiftMaskResult(
            frame_id="000000",
            candidate_id=args.candidate_id,
            lifted_point_count=1,
            mask_npz_path=npz,
            mask_ply_path=tmp_path / "x.ply",
            overlay_path=tmp_path / "x.txt",
        )

    monkeypatch.setattr(pipeline_backends, "sam_mask", _fake_sam_mask)
    monkeypatch.setattr(pipeline_backends, "lift_mask_to_3d", _fake_lift)
    monkeypatch.setattr(
        pipeline_backends, "load_backend_settings", lambda path: object()
    )

    proposer = pipeline_backends.SidecarFrameProposer(
        scene=scene,
        out_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backend.toml",
        affordance_concept="the handle",
        task_description="open the drawer",
    )
    geometry = CameraGeometry(
        intrinsics=np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]),
        camera_to_world=np.eye(4),
    )
    proposal = proposer.propose(
        frame_id="000000", geometry=geometry, projected_anchor_xy=(15.0, 16.0)
    )
    assert proposal.molmo_fallback_used is True
    assert captured_points[0] == pytest.approx((15.0, 16.0))
    assert proposal.point_indices == (1,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline_backends.py -v`
Expected: FAIL with `ImportError` (`points_from_molmo_response` / `pipeline_backends`).

- [ ] **Step 3a: Add `points_from_molmo_response` to `molmo_pointing.py`**

Add this function (and add its name to `__all__`):

```python
def points_from_molmo_response(
    response: "MolmoPointResponse", *, image_width: int, image_height: int
) -> tuple[MolmoPoint, ...]:
    """Return parsed points from a Molmo response, or ``()`` when it found none.

    Prefers the structured ``image_points``; falls back to parsing ``raw_text``.
    A *successful* response with no usable points yields ``()`` (the caller
    records a designed anchor fallback). Transport/sidecar failures raise
    earlier in ``request_molmo_point`` and are not reached here.
    """
    try:
        points = _points_from_sidecar_image_points(
            response.image_points,
            image_width=image_width,
            image_height=image_height,
        )
        if not points:
            points = parse_molmo_points(
                response.raw_text,
                image_width=image_width,
                image_height=image_height,
            )
    except SceneFunc3dDataError:
        return ()
    return points
```

Add the import for the type (top of file):

```python
from ..servers.schemas import MolmoImagePoint, MolmoPointResponse
```

(`MolmoImagePoint` is already imported; add `MolmoPointResponse` to that same import line.)

- [ ] **Step 3b: Write `pipeline_backends.py`**

```python
# src/codex_agent/scenefunc3d/pipeline_backends.py
"""Sidecar-backed per-frame proposer for the anchor multi-view pipeline.

Implements ``pipeline.FrameProposer`` by calling the real Molmo sidecar (RPC,
no overlay), the SAM tool (smallest candidate), and the lift tool, then reading
the fragment's raw-mesh vertex ids back. Molmo emptiness is a *designed*
fallback to the projected anchor, recorded in provenance (spec Stage C step 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.backends.config import load_backend_settings
from codex_agent.scenefunc3d.backends.frame_assets import (
    resolve_frame_geometry_assets,
)
from codex_agent.scenefunc3d.backends.frame_loader import frame_rgb_path
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.backends.molmo_rpc import request_molmo_point
from codex_agent.scenefunc3d.pipeline import FrameProposal
from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskArgs, lift_mask_to_3d
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.molmo_pointing import (
    MolmoPoint,
    MolmoPointArgs,
    points_from_molmo_response,
)
from codex_agent.scenefunc3d.tools.sam_masking import (
    SamMaskArgs,
    SamPointInput,
    sam_mask,
)
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

# Molmo points farther than this fraction of the image diagonal from the
# projected anchor are treated as "too far" and trigger the anchor fallback.
_MAX_MOLMO_ANCHOR_DISTANCE_FRACTION = 0.15
_FRAGMENT_POINT_INDICES_KEY = "point_indices"


@dataclass(frozen=True)
class SidecarFrameProposer:
    """Propose one frame's lifted vertex ids via Molmo -> SAM -> lift."""

    scene: SceneFunc3dToolScene
    out_dir: Path
    backend_config_path: Path
    affordance_concept: str
    task_description: str

    def propose(
        self,
        *,
        frame_id: str,
        geometry: object,
        projected_anchor_xy: tuple[float, float] | None,
    ) -> FrameProposal:
        """Return the frame's lifted raw-mesh vertex ids (empty when unusable)."""
        assert isinstance(geometry, CameraGeometry)
        rgb_path = frame_rgb_path(self.scene, frame_id)
        image_width, image_height = _image_size(rgb_path)
        prompt = (
            f"point to {self.affordance_concept} in order to "
            f"{self.task_description}"
        )
        settings = load_backend_settings(self.backend_config_path)
        response = request_molmo_point(
            settings,
            request_id=f"{frame_id}_molmo_mv",
            args=MolmoPointArgs(
                frame_id=frame_id,
                image_path=rgb_path,
                prompt=prompt,
                image_width=image_width,
                image_height=image_height,
            ),
        )
        molmo_points = points_from_molmo_response(
            response, image_width=image_width, image_height=image_height
        )
        prompt_xy, fallback_used = _select_prompt_point(
            molmo_points,
            projected_anchor_xy=projected_anchor_xy,
            image_width=image_width,
            image_height=image_height,
        )
        if prompt_xy is None:
            return FrameProposal(
                frame_id=frame_id, point_indices=(), molmo_fallback_used=fallback_used
            )
        sam_result = sam_mask(
            SamMaskArgs(
                frame_id=frame_id,
                image_path=rgb_path,
                points=(
                    SamPointInput(
                        x_px=prompt_xy[0],
                        y_px=prompt_xy[1],
                        source="anchor_fallback" if fallback_used else "molmo",
                        label=self.affordance_concept,
                    ),
                ),
            ),
            out_dir=self.out_dir,
            backend_config_path=self.backend_config_path,
        )
        smallest = _smallest_candidate(sam_result.candidates)
        if smallest is None:
            return FrameProposal(
                frame_id=frame_id, point_indices=(), molmo_fallback_used=fallback_used
            )
        lift_result = lift_mask_to_3d(
            LiftMaskArgs(
                frame_id=frame_id,
                candidate_id=smallest.candidate_id,
                mask_path=smallest.mask_npz_path,
                **_geometry_asset_paths(self.scene, frame_id),
            ),
            out_dir=self.out_dir,
            raw_mesh_path=self.scene.raw_mesh_path,
        )
        point_indices = _read_fragment_point_indices(lift_result.mask_npz_path)
        return FrameProposal(
            frame_id=frame_id,
            point_indices=point_indices,
            molmo_fallback_used=fallback_used,
        )


def _select_prompt_point(
    molmo_points: tuple[MolmoPoint, ...],
    *,
    projected_anchor_xy: tuple[float, float] | None,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float] | None, bool]:
    diagonal = float(np.hypot(image_width, image_height))
    max_distance = _MAX_MOLMO_ANCHOR_DISTANCE_FRACTION * diagonal
    if projected_anchor_xy is None:
        if molmo_points:
            return (molmo_points[0].x_px, molmo_points[0].y_px), False
        return None, True
    if molmo_points:
        anchor = np.asarray(projected_anchor_xy, dtype=np.float64)
        distances = [
            float(np.hypot(point.x_px - anchor[0], point.y_px - anchor[1]))
            for point in molmo_points
        ]
        best = int(np.argmin(distances))
        if distances[best] <= max_distance:
            return (molmo_points[best].x_px, molmo_points[best].y_px), False
    return projected_anchor_xy, True


def _smallest_candidate(candidates: tuple[object, ...]) -> object | None:
    usable = [c for c in candidates if getattr(c, "pixel_count", 0) > 0]
    if not usable:
        return None
    return min(usable, key=lambda c: c.pixel_count)


def _geometry_asset_paths(
    scene: SceneFunc3dToolScene, frame_id: str
) -> dict[str, str]:
    assets = resolve_frame_geometry_assets(scene, frame_id)
    return {
        "depth_path": str(assets.depth_path),
        "intrinsics_path": str(assets.intrinsics_path),
        "pose_path": str(assets.pose_path),
    }


def _image_size(rgb_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to read frame image size; install the 'vision' extra"
        ) from exc
    with Image.open(rgb_path) as image:
        return int(image.width), int(image.height)


def _read_fragment_point_indices(npz_path: Path) -> tuple[int, ...]:
    try:
        with np.load(npz_path) as archive:
            if _FRAGMENT_POINT_INDICES_KEY not in archive.files:
                raise SceneFunc3dDataError(
                    "lift fragment NPZ is missing required key "
                    f"{_FRAGMENT_POINT_INDICES_KEY!r}: npz_path={npz_path}"
                )
            indices = np.asarray(archive[_FRAGMENT_POINT_INDICES_KEY]).astype(
                np.int64
            ).ravel()
    except SceneFunc3dDataError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise SceneFunc3dDataError(
            "could not load lift fragment NPZ: "
            f"npz_path={npz_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return tuple(int(value) for value in indices)


__all__ = ["SidecarFrameProposer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline_backends.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/pipeline_backends.py \
        src/codex_agent/scenefunc3d/tools/molmo_pointing.py \
        src/codex_agent/tests/test_scenefunc3d_pipeline_backends.py
git commit -m "feat(scenefunc3d): add sidecar-backed per-frame proposer"
```

---

## Task 8: Semi-online validation harness (`evaluation/semi_online_pipeline.py`)

Tier-2 validation (spec §"半在线"): for each sample saved under a completed run root, take the
**largest saved fragment as the fixed seed anchor** (same seed proxy as the offline sweep), then
run the **full B/C/D pipeline** with the real Molmo/SAM sidecars and re-score with the existing
scorer. Needs the Molmo/SAM sidecars healthy; does **not** need the ModelHub adapter. Mirrors
`evaluation/offline_fusion.py` conventions (typed argparse, TypedDict payload, fail-closed IO).

**Files:**
- Create: `src/codex_agent/scenefunc3d/evaluation/semi_online_pipeline.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_semi_online_pipeline.py`

- [ ] **Step 1: Write the failing test** (pure assembly with a fake proposer; no sidecar)

```python
# src/codex_agent/tests/test_scenefunc3d_semi_online_pipeline.py
"""Tests for the semi-online SceneFunc3D pipeline harness (fake proposer)."""

from __future__ import annotations

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.evaluation.semi_online_pipeline import (
    score_semi_online_sample,
)
from codex_agent.scenefunc3d.pipeline import FrameProposal
from codex_agent.tests.scenefunc3d_synthetic_scene import (
    SyntheticFrame,
    write_synthetic_scene,
)


class _FakeProposer:
    def __init__(self, indices: tuple[int, ...]) -> None:
        self._indices = indices

    def propose(self, *, frame_id, geometry, projected_anchor_xy) -> FrameProposal:
        return FrameProposal(
            frame_id=frame_id, point_indices=self._indices, molmo_fallback_used=False
        )


def test_score_semi_online_sample_recovers_target(tmp_path) -> None:
    intrinsics = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])
    vertices = np.array(
        [
            [0.00, 0.0, 2.0],
            [0.01, 0.0, 2.0],
            [0.00, 0.01, 2.0],
            [0.01, 0.01, 2.0],
            [0.005, 0.005, 2.0],
            [3.00, 3.0, 2.0],
        ]
    )
    frames = (
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        SyntheticFrame("000001", np.eye(4), depth_value_m=2.0),
    )
    scene = write_synthetic_scene(
        tmp_path / "scene", frames=frames, vertices_world=vertices, intrinsics=intrinsics
    )
    anchor = build_anchor(
        vertices[:5], motion_type="pinch_pull", seed_frame_id="000000"
    )
    score = score_semi_online_sample(
        sample_id="123::0",
        scene=scene,
        anchor=anchor,
        gt_ids=frozenset({0, 1, 2, 3, 4}),
        proposer=_FakeProposer((0, 1, 2, 3, 4)),
        params=FusionParams(agreement_tau=0.5, min_cluster_points=3),
        frame_cap=10,
    )
    assert score.metrics.iou == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_semi_online_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` for `semi_online_pipeline`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/evaluation/semi_online_pipeline.py
"""Semi-online SceneFunc3D validation (spec tier 2).

For each sample under a completed run root, take the largest saved fragment as
a fixed seed anchor and run the full deterministic B/C/D pipeline through the
real Molmo/SAM sidecars, then re-score against hidden GT. Needs healthy Molmo
and SAM sidecars; does NOT need the ModelHub adapter.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence, Set
from pathlib import Path
from typing import TypedDict

from codex_agent.errors import CodexAgentError
from codex_agent.scenefunc3d.backends.anchor import TargetAnchor, build_anchor
from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.backends.lift_3d import load_scene_mesh_vertices
from codex_agent.scenefunc3d.evaluation.offline_fusion import (
    load_sample_fusion_input,
)
from codex_agent.scenefunc3d.evaluation.scorer import (
    SceneFunc3dScore,
    load_gt_point_ids,
    score_point_ids,
)
from codex_agent.scenefunc3d.pipeline import (
    FrameProposer,
    run_anchor_multiview_pipeline,
)
from codex_agent.scenefunc3d.pipeline_backends import SidecarFrameProposer
from codex_agent.scenefunc3d.sample import load_sample, scene_dir_for
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene


class SemiOnlineRowPayload(TypedDict):
    """JSON-ready metrics for one semi-online sample."""

    sample_id: str
    iou: float
    precision: float
    recall: float
    predicted_count: int


class SemiOnlineResultPayload(TypedDict):
    """JSON-ready result of a semi-online run."""

    sample_count: int
    mean_iou: float
    mean_precision: float
    ap50: float
    rows: list[SemiOnlineRowPayload]


def score_semi_online_sample(
    *,
    sample_id: str,
    scene: SceneFunc3dToolScene,
    anchor: TargetAnchor,
    gt_ids: Set[int],
    proposer: FrameProposer,
    params: FusionParams,
    frame_cap: int,
) -> SceneFunc3dScore:
    """Run the full pipeline for one sample and score it (pure assembly)."""
    scene_vertices = load_scene_mesh_vertices(scene.raw_mesh_path)
    result = run_anchor_multiview_pipeline(
        anchor=anchor,
        scene=scene,
        scene_vertices=scene_vertices,
        proposer=proposer,
        params=params,
        frame_cap=frame_cap,
    )
    return score_point_ids(
        sample_id=sample_id,
        predicted_ids=frozenset(result.fused.point_indices),
        gt_ids=gt_ids,
    )


def run_semi_online(
    *,
    run_root: Path,
    data_root: Path,
    backend_config_path: Path,
    out_dir: Path,
    params: FusionParams,
    frame_cap: int,
    sample_ids: Sequence[str],
) -> SemiOnlineResultPayload:
    """Run the semi-online pipeline for each sample and aggregate metrics."""
    rows: list[SemiOnlineRowPayload] = []
    ious: list[float] = []
    precisions: list[float] = []
    hits50 = 0
    for sample_id in sample_ids:
        sample = load_sample(data_root, sample_id)
        scene = SceneFunc3dToolScene.load(scene_dir_for(data_root, sample.visit_id))
        fusion_input = load_sample_fusion_input(
            run_root=run_root, data_root=data_root, sample_id=sample_id
        )
        anchor = build_anchor(
            fusion_input.scene_vertices[list(fusion_input.anchor_point_indices)],
            motion_type=fusion_input.motion_type,
            seed_frame_id="semi_online_seed",
        )
        proposer: FrameProposer = SidecarFrameProposer(
            scene=scene,
            out_dir=out_dir / sample.visit_id / sample.desc_id,
            backend_config_path=backend_config_path,
            affordance_concept=sample.task_description,
            task_description=sample.task_description,
        )
        score = score_semi_online_sample(
            sample_id=sample_id,
            scene=scene,
            anchor=anchor,
            gt_ids=load_gt_point_ids(data_root, sample_id),
            proposer=proposer,
            params=params,
            frame_cap=frame_cap,
        )
        ious.append(score.metrics.iou)
        precisions.append(score.metrics.precision)
        hits50 += int(score.metrics.iou >= 0.50)
        rows.append(
            {
                "sample_id": sample_id,
                "iou": score.metrics.iou,
                "precision": score.metrics.precision,
                "recall": score.metrics.recall,
                "predicted_count": score.metrics.predicted_count,
            }
        )
    n = max(len(sample_ids), 1)
    return {
        "sample_count": len(sample_ids),
        "mean_iou": sum(ious) / n,
        "mean_precision": sum(precisions) / n,
        "ap50": hits50 / n,
        "rows": rows,
    }


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value: object = getattr(args, name)
    if not isinstance(value, Path):
        raise TypeError(f"argparse field {name!r} must be a Path")
    return value


def _sample_ids_arg(args: argparse.Namespace) -> tuple[str, ...]:
    from codex_agent.scenefunc3d.evaluation.offline_fusion import (
        _sample_ids_from_run_root,
    )

    value: object = getattr(args, "sample_ids")
    if value:
        if not isinstance(value, list):
            raise TypeError("sample_ids must be a list")
        return tuple(str(item) for item in value)
    return _sample_ids_from_run_root(_namespace_path(args, "run_root"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semi-online SceneFunc3D pipeline")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frame-cap", type=int, default=12)
    parser.add_argument("--agreement-tau", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: run the semi-online pipeline and print JSON."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    tau: object = getattr(args, "agreement_tau")
    frame_cap: object = getattr(args, "frame_cap")
    try:
        summary = run_semi_online(
            run_root=_namespace_path(args, "run_root"),
            data_root=_namespace_path(args, "data_root"),
            backend_config_path=_namespace_path(args, "backend_config"),
            out_dir=_namespace_path(args, "out_dir"),
            params=FusionParams(agreement_tau=float(tau)),  # type: ignore[arg-type]
            frame_cap=int(frame_cap),  # type: ignore[arg-type]
            sample_ids=_sample_ids_arg(args),
        )
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    text = json.dumps(summary, indent=2)
    output: object = getattr(args, "output")
    if isinstance(output, Path):
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


__all__ = [
    "SemiOnlineResultPayload",
    "SemiOnlineRowPayload",
    "main",
    "run_semi_online",
    "score_semi_online_sample",
]


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_semi_online_pipeline.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/evaluation/semi_online_pipeline.py \
        src/codex_agent/tests/test_scenefunc3d_semi_online_pipeline.py
git commit -m "feat(scenefunc3d): add semi-online full-pipeline validation harness"
```

---

## Task 9: Quality gate, semi-online smoke run, and docs

**Files:** verification + docs only.

- [ ] **Step 1: Run the full quality gate**

```bash
source .venv/bin/activate
ruff check src/
black --check src/codex_agent/scenefunc3d/ src/codex_agent/tests/
mypy src/codex_agent/scenefunc3d/ 
PYTHONPATH=src pytest src/codex_agent/tests/ -k "scenefunc3d" -q
```
Expected: ruff clean, black clean, mypy no issues, all scenefunc3d tests pass.
Fix any issues, then commit fixes with `git commit -m "fix(scenefunc3d): quality gate for phase 2b engine"`.

- [ ] **Step 2 (optional, needs sidecars — run in tmux): semi-online smoke on the local scene**

Preflight (per `AGENTS.md`): confirm Molmo + SAM sidecars are healthy. Then run against the
one local run root that has saved fragments (the v5/v6 run root; find it under `tmp/` from
`docs/benchmark/scenefunc_molmo_sam3d/v6_offline_fusion_sweep_20260702.md`).

```bash
tmux new-session -d -s sf3d_semi "source .venv/bin/activate && \
  PYTHONPATH=src python -m codex_agent.scenefunc3d.evaluation.semi_online_pipeline \
    --run-root <v5_or_v6_run_root> \
    --data-root data/SceneFun3D \
    --backend-config <backend.toml> \
    --out-dir tmp/sf3d_semi_online_$(date +%Y%m%d)/ \
    --frame-cap 12 --agreement-tau 0.5 \
    --output tmp/sf3d_semi_online_$(date +%Y%m%d)/summary.json 2>&1 | tee /tmp/sf3d_semi.log"
```
Monitor: `tmux capture-pane -t sf3d_semi -p -S -20`.

If sidecars are unavailable, **skip** the run (state so explicitly) — the harness + unit test
still land. Do not fabricate numbers.

- [ ] **Step 3: Write the benchmark doc**

Create `docs/benchmark/scenefunc_molmo_sam3d/v7_semi_online_pipeline_<YYYYMMDD>.md` with the
mandatory content (branch + tip commit, exact CLI, raw artifact dir, fold size, what changed vs
v6, headline mean IoU / precision / AP50, and caveats — note this is tier-2 semi-online with
fixed seeds, not the online agent). If Step 2 was skipped, record the harness landing + "run
pending sidecar availability" and leave the metrics table empty. Update
`docs/benchmark/scenefunc_molmo_sam3d/README.md` version timeline in the same commit.

```bash
git add docs/benchmark/scenefunc_molmo_sam3d/
git commit -m "docs(scenefunc3d): record v7 semi-online pipeline validation"
```

- [ ] **Step 4: Update the Phase 2a roadmap pointer**

In `docs/superpowers/plans/2026-07-02-scenefunc3d-visibility-projector-phase2a.md`, under the
"Phase 2b Roadmap" section, add a line: "Implemented in
`docs/superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md`. Agent
`SeedConfirmation` contract deferred to Phase 2c."

```bash
git add docs/superpowers/plans/2026-07-02-scenefunc3d-visibility-projector-phase2a.md
git commit -m "docs(scenefunc3d): link phase 2b plan from phase 2a roadmap"
```

---

## Phase 2c (separate plan — agent SeedConfirmation contract)

Out of scope here; write a dedicated plan after semi-online validation. High-level shape:

- New `SeedConfirmation` output schema: `frame_id`, `candidate_id`, `affordance_concept`,
  `approval_actions` (must replay `task.FRAGMENT_APPROVAL_ACTIONS`, ending at
  `FIRST_LIFT_AGENT_APPROVED` — the existing `task.validate_fragment_approval_actions` is reused
  as-is).
- `SceneFunc3dMaskTask.build_turn_request` emits the `SeedConfirmation` schema; `parse_response`
  validates the confirmed seed fragment exists in `events.jsonl`; a bounded correction loop
  (≤3, fail-closed).
- `run_single_sample`: after confirmation, build the anchor from the seed fragment's lift, call
  `run_anchor_multiview_pipeline` (from this plan), write the final mask artifact + `result.json`
  in the existing shape so `scorer` / `evaluation/__main__` keep working.
- Trim `playbook.py` to the seed stage; rework the fuse-tool self-report validation
  (`_validate_outcome_against_fuse_tool_event`) since the runner now produces the fused mask.
- Rework `test_scenefunc3d_runner.py` accordingly.

---

## Self-Review

**1. Spec coverage (Phase-2b scope = Stages B/C/D disk + online engine + tier-2 validation):**
- Frame loader (spec "帧加载器"): Task 1 (`camera_io`) + Task 2 (`frame_loader`). ✓
- Per-vertex visibility counts (spec "逐顶点可见次数"): Task 4. ✓
- Streaming Stage B selection over a real scene (spec Stage B, memory-safe): Task 3. ✓
- Deterministic B/C/D driver (spec "pipeline.py 确定性驱动"): Tasks 5 (pure) + 6 (scene wrapper). ✓
- Per-frame Molmo+SAM+lift with projected-anchor fallback + smallest candidate + anchor gate
  (spec Stage C): Task 7 (proposer) + Task 5 (gate). ✓
- Layered fusion with agreement fed by real visibility counts (spec Stage D): Tasks 5/6 wire
  `fuse_multiview_points` with `visibility_counts`. ✓
- Semi-online validation (spec tier 2): Task 8 + Task 9 Step 2. ✓
- Agent `SeedConfirmation` contract (spec item 4): **explicitly deferred to Phase 2c** with
  rationale (scope decision at top); not a gap. ✓
- Non-goals respected: no ranked-AP wiring (only `FusedMask.instances`/confidence already
  produced), no training head, no adapter changes, no `metrics.py` math changes. ✓

**2. Placeholder scan:** No "TBD"/"add validation"/"similar to Task N". Every code step has full
code; every command has an expected result. The one intentional judgement — "find the v5/v6 run
root" in Task 9 Step 2 — points at the exact doc that records it and is explicitly optional/gated
on sidecar availability. ✓

**3. Type consistency:**
- `read_depth_meters(Path)->FloatArray`, `read_camera_matrix(Path,*,expected_shape,field_name)`,
  `load_camera_geometry(*,intrinsics_path,pose_path)->CameraGeometry` — defined Task 1, reused
  in `mask_lifting` (Task 1), `frame_loader` (Task 2). ✓
- `iter_frame_geometry(scene)->Iterator[(str,CameraGeometry)]`, `load_frame_geometry(scene,frame_id)->CameraGeometry`,
  `read_frame_depth(scene,frame_id)->FloatArray`, `frame_rgb_path(scene,frame_id)->Path` —
  Task 2; consumed by Tasks 3, 4, 7. ✓
- `select_scene_visible_frames(anchor,scene,*,frame_cap,depth_tolerance)->tuple[FrameVisibility,...]`
  (Task 3) and `count_vertex_visibility(vertex_ids,vertex_coords,scene,frame_ids,*,depth_tolerance)->dict[int,int]`
  (Task 4) reuse Phase-2a `score_anchor_visibility`/`point_visibility`/`FrameVisibility` (names
  verified against committed `visibility.py`). ✓
- `FrameProposal(frame_id,point_indices,molmo_fallback_used)`, `FrameProposer.propose(*,frame_id,geometry,projected_anchor_xy)->FrameProposal`,
  `PerFrameOutcome`, `AnchorMultiViewResult(fused,selected_frames,per_frame)`,
  `fuse_selected_frames(*,anchor,scene_vertices,selected_frames,geometry_by_frame,proposer,params,visibility_counts_fn,gate_radius_scale)`
  (Task 5) and `run_anchor_multiview_pipeline(*,anchor,scene,scene_vertices,proposer,params,frame_cap,depth_tolerance,gate_radius_scale)`
  (Task 6) — signatures identical where reused in Tasks 7/8 tests. ✓
- `SidecarFrameProposer(scene,out_dir,backend_config_path,affordance_concept,task_description)`
  implements `FrameProposer.propose` (Task 7); constructed the same way in Task 8. ✓
- Reuses verified upstream signatures: `fusion.{FrameLift,FusionParams,MultiViewLiftBundle,FusedMask,fuse_multiview_points}`,
  `anchor.{TargetAnchor,build_anchor}`, `lift_3d.{CameraGeometry,FloatArray,load_scene_mesh_vertices}`,
  `molmo_rpc.request_molmo_point`, `sam_masking.{SamMaskArgs,SamPointInput,sam_mask,SamCandidate,SamMaskResult}`,
  `mask_lifting.{LiftMaskArgs,lift_mask_to_3d,LiftMaskResult}`, `config.load_backend_settings`,
  `scorer.{score_point_ids,load_gt_point_ids,SceneFunc3dScore}`,
  `offline_fusion.{load_sample_fusion_input,_sample_ids_from_run_root}`,
  `sample.{load_sample,scene_dir_for}`, `scene_context.SceneFunc3dToolScene`,
  `servers.schemas.{MolmoPointResponse,MolmoImagePoint}`. ✓

**Known follow-ups (not blockers):** (a) `scorer.SceneFunc3dScore` = `{sample_id,
metrics: MaskMetrics, failure_type}` and `MaskMetrics` = `{iou, precision, recall, f1,
predicted_count, gt_count}` — verified against `scorer.py`/`metrics.py`, so Task 8 uses
`score.metrics.iou/.precision/.recall/.predicted_count` (confirmed, already corrected in the
plan). (b) The `VisibilityCounter` string alias in `pipeline.py` is documentation-only; the parameter is
typed `object` to keep the pure module import-light — mypy still checks the concrete function in
Task 6. (c) Per-frame SAM contact sheets add trace volume for `frame_cap` frames; acceptable for
tier-2, revisit for the online path in Phase 2c.
