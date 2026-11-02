# SceneFunc3D Raw Point ID Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every SceneFunc3D final mask artifact write `point_indices` in the same raw/source scan vertex-id space used by hidden GT annotations.

**Architecture:** Keep scorer math unchanged and move the fix to the deterministic lift path. `SceneFunc3dToolScene` exposes the raw mesh path, dispatcher passes that path into `lift_mask_to_3d`, and the lift tool assigns nearest raw mesh vertex ids directly. Filtered conceptgraph-to-source mapping remains a fail-closed compatibility helper, not the main path.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, SciPy `cKDTree` when installed, Pillow for lift tests, pytest, ruff, black, mypy.

---

## File Structure

- Modify `src/codex_agent/scenefunc3d/tools/scene_context.py`
  - Add the raw scene mesh path as a typed property of `SceneFunc3dToolScene`.
- Modify `src/codex_agent/scenefunc3d/tools/dispatch.py`
  - Pass `tool_scene.raw_mesh_path` to `lift_mask_to_3d`.
- Modify `src/codex_agent/scenefunc3d/tools/mask_lifting.py`
  - Rename the lift assignment mesh parameter to `raw_mesh_path`.
  - Assign `point_indices` directly against raw mesh vertices.
  - Remove silent filtered-index fallback from the main path.
  - Make filtered-to-source mapping fail when the crop mask is missing.
- Modify `src/codex_agent/scenefunc3d/backends/lift_3d.py`
  - Add a header-only `load_scene_mesh_vertex_count()` helper for scorer bounds checks.
- Modify `src/codex_agent/scenefunc3d/evaluation/scorer.py`
  - Validate predicted point ids against the raw mesh vertex count before scoring.
- Modify `src/codex_agent/tests/test_scenefunc3d_lift3d.py`
  - Add raw mesh path, dispatcher, raw assignment, and fail-closed mapping tests.
- Modify `src/codex_agent/tests/test_scenefunc3d_metrics.py`
  - Add raw mesh fixture data and scorer bounds test.
- Add a benchmark record under `docs/benchmark/scenefunc_molmo_sam3d/` only after code verification and a real scored run.

## Task 1: Expose Raw Mesh Path On Tool Scene

**Files:**
- Modify: `src/codex_agent/scenefunc3d/tools/scene_context.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_lift3d.py`

- [ ] **Step 1: Write the failing test**

In `src/codex_agent/tests/test_scenefunc3d_lift3d.py`, add this test after `test_resolve_frame_geometry_assets_reports_missing_asset`:

```python
def test_tool_scene_exposes_raw_mesh_path(tmp_path: Path) -> None:
    scene = _make_tool_scene(tmp_path)

    assert scene.raw_mesh_path == tmp_path / "scene" / "raw" / "mesh.ply"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_tool_scene_exposes_raw_mesh_path -q
```

Expected: FAIL with an `AttributeError` mentioning `raw_mesh_path`.

- [ ] **Step 3: Implement the raw mesh property**

In `src/codex_agent/scenefunc3d/tools/scene_context.py`, add this property to `SceneFunc3dToolScene` immediately after `raw_dir`:

```python
    @property
    def raw_mesh_path(self) -> Path:
        """Raw/source scan mesh whose vertex ids align with annotation indices."""
        return self.raw_dir / "mesh.ply"
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_tool_scene_exposes_raw_mesh_path -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/codex_agent/scenefunc3d/tools/scene_context.py src/codex_agent/tests/test_scenefunc3d_lift3d.py
git commit -m "Add SceneFunc3D raw mesh path"
```

## Task 2: Make Lift Tool Write Raw Mesh Vertex IDs

**Files:**
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Modify: `src/codex_agent/scenefunc3d/tools/mask_lifting.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_lift3d.py`

- [ ] **Step 1: Add the dispatcher/raw-assignment failing test**

In `src/codex_agent/tests/test_scenefunc3d_lift3d.py`, add this import near the existing SceneFunc3D imports:

```python
from codex_agent.scenefunc3d.tools.dispatch import run_tool
```

Then add this test after `test_lift_mask_to_3d_maps_filtered_mesh_indices_to_source_scan_ids`; this test intentionally makes raw mesh ids differ from conceptgraph mesh ordinals:

```python
def test_run_tool_lift_mask_to_3d_assigns_raw_mesh_point_indices(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image

    scene = _make_tool_scene(tmp_path)
    _write_binary_scene_mesh(
        scene.raw_mesh_path,
        points=((9.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
    )
    _write_binary_scene_mesh(
        scene.conceptgraph_dir / "mesh.ply",
        points=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
    )
    mask_path = tmp_path / "mask.npz"
    depth_path = scene.raw_dir / "000001-depth.png"
    intrinsics_path = scene.raw_dir / "000001-intrinsic.txt"
    pose_path = scene.raw_dir / "pose" / "000001.txt"
    np.savez_compressed(mask_path, mask=np.array([[True, True]], dtype=np.bool_))
    Image.fromarray(np.array([[1000, 1000]], dtype=np.uint16)).save(depth_path)
    intrinsics_path.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    pose_path.parent.mkdir()
    pose_path.write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
        encoding="utf-8",
    )

    payload = run_tool(
        scene,
        "lift_mask_to_3d",
        {
            "frame_id": "000001",
            "candidate_id": "mask_00",
            "mask_npz_path": str(mask_path),
            "depth_path": str(depth_path),
            "intrinsics_path": str(intrinsics_path),
            "pose_path": str(pose_path),
        },
        out_dir=tmp_path / "out",
    ).to_payload()

    with np.load(Path(str(payload["mask_npz_path"]))) as archive:
        np.testing.assert_array_equal(
            archive["point_indices"], np.array([1, 2], dtype=np.int64)
        )
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_run_tool_lift_mask_to_3d_assigns_raw_mesh_point_indices -q
```

Expected: FAIL because the current dispatcher assigns against `conceptgraph/mesh.ply` and writes `[0, 1]`.

- [ ] **Step 3: Change `lift_mask_to_3d` to use raw mesh ids directly**

In `src/codex_agent/scenefunc3d/tools/mask_lifting.py`, replace the `lift_mask_to_3d` function signature and body with:

```python
def lift_mask_to_3d(
    args: LiftMaskArgs, *, out_dir: Path, raw_mesh_path: Path
) -> LiftMaskResult:
    """Lift one 2D mask candidate into deterministic raw-scene point artifacts."""
    from codex_agent.scenefunc3d.backends.lift_3d import (
        CameraGeometry,
        assign_nearest_scene_point_indices,
        backproject_mask_to_world,
        load_mask_npz,
        load_scene_mesh_vertices,
        write_lift_npz,
        write_lift_ply,
    )

    mask = load_mask_npz(args.mask_path)
    depth_meters = _read_depth_meters(args.depth_path)
    intrinsics = _read_matrix(
        args.intrinsics_path,
        expected_shape=(3, 3),
        field_name="intrinsics",
    )
    camera_to_world = _read_matrix(
        args.pose_path,
        expected_shape=(4, 4),
        field_name="camera_to_world",
    )
    try:
        geometry = CameraGeometry(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
    except ToolInputError as exc:
        raise ToolInputError(
            "invalid lift camera geometry: "
            f"intrinsics_path={args.intrinsics_path}; pose_path={args.pose_path}; "
            f"error={exc}"
        ) from exc
    points_world = backproject_mask_to_world(mask, depth_meters, geometry)
    raw_scene_points_world = load_scene_mesh_vertices(raw_mesh_path)
    raw_point_indices = assign_nearest_scene_point_indices(
        points_world,
        raw_scene_points_world,
        max_distance_meters=_MAX_SCENE_POINT_ASSIGNMENT_DISTANCE_METERS,
    )
    fragment_dir = _fragment_dir(
        out_dir, frame_id=args.frame_id, candidate_id=args.candidate_id
    )
    mask_npz_path = write_lift_npz(
        fragment_dir / "mask_data.npz",
        points_world,
        point_indices=raw_point_indices,
    )
    mask_ply_path = write_lift_ply(fragment_dir / "lifted_points.ply", points_world)
    overlay_path = _write_lift_summary(
        fragment_dir / "lift_overlay.txt",
        args=args,
        raw_mesh_path=raw_mesh_path,
        lifted_point_count=int(points_world.shape[0]),
    )
    return LiftMaskResult(
        frame_id=args.frame_id,
        candidate_id=args.candidate_id,
        lifted_point_count=int(points_world.shape[0]),
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        overlay_path=overlay_path,
    )
```

Replace `_write_lift_summary` with:

```python
def _write_lift_summary(
    overlay_path: Path,
    *,
    args: LiftMaskArgs,
    raw_mesh_path: Path,
    lifted_point_count: int,
) -> Path:
    summary = (
        f"frame_id={args.frame_id}\n"
        f"candidate_id={args.candidate_id}\n"
        f"lifted_point_count={lifted_point_count}\n"
        f"mask_path={args.mask_path}\n"
        f"depth_path={args.depth_path}\n"
        f"intrinsics_path={args.intrinsics_path}\n"
        f"pose_path={args.pose_path}\n"
        f"raw_mesh_path={raw_mesh_path}\n"
    )
    try:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(summary, encoding="utf-8")
    except OSError as exc:
        raise ToolInputError(
            "could not write lift overlay summary: "
            f"path={overlay_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return overlay_path
```

- [ ] **Step 4: Pass raw mesh path from dispatcher**

In `src/codex_agent/scenefunc3d/tools/dispatch.py`, replace the `lift_mask_to_3d` branch with:

```python
    if name == "lift_mask_to_3d":
        from .mask_lifting import LiftMaskArgs, lift_mask_to_3d

        return lift_mask_to_3d(
            _parse(LiftMaskArgs, _with_resolved_lift_geometry(tool_scene, raw_args)),
            out_dir=out_dir,
            raw_mesh_path=tool_scene.raw_mesh_path,
        )
```

- [ ] **Step 5: Update direct lift tests to use `raw_mesh_path` keyword**

In `src/codex_agent/tests/test_scenefunc3d_lift3d.py`, any direct call that still uses the old `scene_mesh_path` keyword must use the new raw mesh keyword:

```python
        raw_mesh_path=mesh_path,
```

Do not keep `scene_mesh_path=` in tests or production code for the lift function.

- [ ] **Step 6: Run the focused test and verify it passes**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_run_tool_lift_mask_to_3d_assigns_raw_mesh_point_indices -q
```

Expected: PASS.

- [ ] **Step 7: Continue to Task 3 before committing**

Do not commit yet. At this point the focused raw-assignment test passes, but the
old filtered-mapping test still describes the previous main-path behavior. Task 3
replaces that test and makes the combined change committable.

## Task 3: Make Filtered-To-Source Mapping Fail Closed

**Files:**
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Modify: `src/codex_agent/scenefunc3d/tools/mask_lifting.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_lift3d.py`

- [ ] **Step 1: Import the module for private compatibility helper tests**

In `src/codex_agent/tests/test_scenefunc3d_lift3d.py`, add:

```python
from codex_agent.scenefunc3d.tools import mask_lifting as mask_lifting_module
```

- [ ] **Step 2: Replace the obsolete direct lift mapping test**

Replace `test_lift_mask_to_3d_maps_filtered_mesh_indices_to_source_scan_ids` with these two tests:

```python
def test_filtered_mesh_source_mapping_uses_crop_mask_when_present(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    mesh_path = _write_binary_scene_mesh(
        tmp_path / "SceneFuncVal-CG" / "421254" / "conceptgraph" / "mesh.ply",
        points=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
    )
    crop_mask = np.array([False, True, False, True], dtype=np.bool_)
    crop_mask_path = tmp_path / "SceneFunVal" / "421254" / "421254_crop_mask.npy"
    crop_mask_path.parent.mkdir(parents=True)
    np.save(crop_mask_path, crop_mask)

    mapped_indices = mask_lifting_module._map_filtered_mesh_indices_to_source_scan_ids(
        np.array([0, 1], dtype=np.int64),
        scene_mesh_path=mesh_path,
        scene_point_count=2,
    )

    np.testing.assert_array_equal(mapped_indices, np.array([1, 3], dtype=np.int64))


def test_filtered_mesh_source_mapping_requires_crop_mask(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    mesh_path = _write_binary_scene_mesh(
        tmp_path / "SceneFun3D" / "421254" / "conceptgraph" / "mesh.ply",
        points=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
    )

    with pytest.raises(ToolInputError, match="source crop mask"):
        mask_lifting_module._map_filtered_mesh_indices_to_source_scan_ids(
            np.array([0], dtype=np.int64),
            scene_mesh_path=mesh_path,
            scene_point_count=2,
        )
```

- [ ] **Step 3: Run the new fail-closed test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_filtered_mesh_source_mapping_requires_crop_mask -q
```

Expected: FAIL because current code returns the original filtered index when crop mask is absent.

- [ ] **Step 4: Implement fail-closed behavior**

In `src/codex_agent/scenefunc3d/tools/mask_lifting.py`, replace the start of `_map_filtered_mesh_indices_to_source_scan_ids` with:

```python
    crop_mask_path = _find_source_crop_mask_path(scene_mesh_path)
    if crop_mask_path is None:
        raise ToolInputError(
            "SceneFunc3D source crop mask is required to map filtered mesh "
            "indices back to source scan ids: "
            f"scene_mesh_path={scene_mesh_path}; scene_point_count={scene_point_count}"
        )
```

Keep the existing NumPy import, crop mask loading, dtype checks, true-count checks, and indexed return below this block.

- [ ] **Step 5: Run the mapping tests**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_filtered_mesh_source_mapping_uses_crop_mask_when_present \
  src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_filtered_mesh_source_mapping_requires_crop_mask \
  -q
```

Expected: PASS.

- [ ] **Step 6: Run the full lift test file**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/codex_agent/scenefunc3d/tools/mask_lifting.py src/codex_agent/tests/test_scenefunc3d_lift3d.py
git add src/codex_agent/scenefunc3d/tools/dispatch.py
git commit -m "Write SceneFunc3D lift masks with raw point ids"
```

## Task 4: Validate Scored Predictions Against Raw Mesh Vertex Count

**Files:**
- Modify: `src/codex_agent/scenefunc3d/backends/lift_3d.py`
- Modify: `src/codex_agent/scenefunc3d/evaluation/scorer.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_lift3d.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_metrics.py`

- [ ] **Step 1: Add a vertex-count helper test**

In `src/codex_agent/tests/test_scenefunc3d_lift3d.py`, add `load_scene_mesh_vertex_count` to the existing lift backend import:

```python
    load_scene_mesh_vertex_count,
```

Then add this test after `test_load_scene_mesh_vertices_reads_binary_little_endian_ply`:

```python
def test_load_scene_mesh_vertex_count_reads_binary_ply_header(
    tmp_path: Path,
) -> None:
    mesh_path = _write_binary_scene_mesh(
        tmp_path / "mesh.ply",
        points=((1.0, 0.0, 2.0), (3.5, 4.0, 5.0), (7.0, 8.0, 9.0)),
    )

    assert load_scene_mesh_vertex_count(mesh_path) == 3
```

- [ ] **Step 2: Run the vertex-count test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_load_scene_mesh_vertex_count_reads_binary_ply_header -q
```

Expected: FAIL with an import error or name error for `load_scene_mesh_vertex_count`.

- [ ] **Step 3: Implement header-only vertex count loading**

In `src/codex_agent/scenefunc3d/backends/lift_3d.py`, add this function immediately after `load_scene_mesh_vertices`:

```python
def load_scene_mesh_vertex_count(mesh_ply_path: Path) -> int:
    """Load only the vertex count from a SceneFunc3D binary mesh PLY header."""
    try:
        with mesh_ply_path.open("rb") as handle:
            layout = _read_binary_ply_vertex_layout(handle, mesh_ply_path)
    except OSError as exc:
        raise ToolInputError(
            "could not read SceneFunc3D mesh PLY header: "
            f"path={mesh_ply_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return layout.vertex_count
```

Also add `"load_scene_mesh_vertex_count",` to `__all__`.

- [ ] **Step 4: Run the vertex-count test and verify it passes**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py::test_load_scene_mesh_vertex_count_reads_binary_ply_header -q
```

Expected: PASS.

- [ ] **Step 5: Add scorer bounds fixture support**

In `src/codex_agent/tests/test_scenefunc3d_metrics.py`, add this import near the top:

```python
import struct
```

In `_write_scoring_scene`, create the raw directory and raw mesh before writing JSON files:

```python
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir()
    _write_binary_scene_mesh(raw_dir / "mesh.ply", vertex_count=128)
```

Add this helper near the other test helpers:

```python
def _write_binary_scene_mesh(path: Path, *, vertex_count: int) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for index in range(vertex_count):
            handle.write(
                struct.pack(
                    "<fffBBB",
                    float(index),
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                )
            )
```

- [ ] **Step 6: Add scorer bounds failing test**

In `src/codex_agent/tests/test_scenefunc3d_metrics.py`, add this test after `test_score_mask_npz_loads_prediction_and_hidden_gt`:

```python
def test_score_mask_npz_rejects_predicted_ids_outside_raw_mesh(
    tmp_path: Path,
) -> None:
    _write_scoring_scene(tmp_path)
    mask_npz_path = tmp_path / "mask_data.npz"
    np.savez_compressed(mask_npz_path, point_indices=np.array([3, 128]))

    with pytest.raises(SceneFunc3dDataError, match="outside raw mesh"):
        score_mask_npz(
            data_root=tmp_path,
            sample_id="421254::desc-a",
            mask_npz_path=mask_npz_path,
        )
```

- [ ] **Step 7: Run the scorer bounds test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_metrics.py::test_score_mask_npz_rejects_predicted_ids_outside_raw_mesh -q
```

Expected: FAIL because scorer currently accepts id `128` without checking raw mesh vertex count.

- [ ] **Step 8: Implement scorer bounds validation**

In `src/codex_agent/scenefunc3d/evaluation/scorer.py`, add imports:

```python
from codex_agent.scenefunc3d.backends.lift_3d import load_scene_mesh_vertex_count
from codex_agent.scenefunc3d.tools.models import ToolInputError
```

Replace `score_mask_npz` with:

```python
def score_mask_npz(
    *,
    data_root: Path,
    sample_id: str,
    mask_npz_path: Path,
    failure_type: str = "",
) -> SceneFunc3dScore:
    """Score a mask NPZ containing raw/source-scene point ids against hidden GT."""
    predicted_ids = load_predicted_point_ids(mask_npz_path)
    _validate_predicted_point_ids_within_raw_mesh(
        data_root=data_root,
        sample_id=sample_id,
        predicted_ids=predicted_ids,
        mask_npz_path=mask_npz_path,
    )
    return score_point_ids(
        sample_id=sample_id,
        predicted_ids=predicted_ids,
        gt_ids=load_gt_point_ids(data_root, sample_id),
        failure_type=failure_type,
    )
```

Add this helper below `score_mask_npz`:

```python
def _validate_predicted_point_ids_within_raw_mesh(
    *,
    data_root: Path,
    sample_id: str,
    predicted_ids: Set[int],
    mask_npz_path: Path,
) -> None:
    if not predicted_ids:
        return
    root = Path(data_root)
    sample = load_sample(root, sample_id)
    raw_mesh_path = scene_dir_for(root, sample.visit_id) / "raw" / "mesh.ply"
    try:
        raw_vertex_count = load_scene_mesh_vertex_count(raw_mesh_path)
    except ToolInputError as exc:
        raise SceneFunc3dDataError(
            "could not validate predicted point ids against raw SceneFunc3D mesh: "
            f"sample_id={sample_id!r}; mask_npz_path={mask_npz_path}; "
            f"raw_mesh_path={raw_mesh_path}; error={exc}"
        ) from exc

    max_predicted_id = max(predicted_ids)
    if max_predicted_id >= raw_vertex_count:
        raise SceneFunc3dDataError(
            "predicted point id is outside raw mesh vertex range: "
            f"sample_id={sample_id!r}; mask_npz_path={mask_npz_path}; "
            f"raw_mesh_path={raw_mesh_path}; raw_vertex_count={raw_vertex_count}; "
            f"max_predicted_id={max_predicted_id}"
        )
```

- [ ] **Step 9: Run focused scoring tests**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_metrics.py::test_score_mask_npz_loads_prediction_and_hidden_gt \
  src/codex_agent/tests/test_scenefunc3d_metrics.py::test_score_mask_npz_rejects_predicted_ids_outside_raw_mesh \
  -q
```

Expected: PASS.

- [ ] **Step 10: Run full lift and metrics tests**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_lift3d.py \
  src/codex_agent/tests/test_scenefunc3d_metrics.py \
  -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

Run:

```bash
git add src/codex_agent/scenefunc3d/backends/lift_3d.py src/codex_agent/scenefunc3d/evaluation/scorer.py src/codex_agent/tests/test_scenefunc3d_lift3d.py src/codex_agent/tests/test_scenefunc3d_metrics.py
git commit -m "Validate SceneFunc3D predictions in raw mesh space"
```

## Task 5: Verification, Smoke Run, And Benchmark Record

**Files:**
- Modify after real run: `docs/benchmark/scenefunc_molmo_sam3d/README.md`
- Create after real run: `docs/benchmark/scenefunc_molmo_sam3d/v4_raw_point_id_alignment_20260702.md`

- [ ] **Step 1: Run the mandatory focused test suite**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py src/codex_agent/tests/test_scenefunc3d_metrics.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the project quality gate**

Run:

```bash
source .venv/bin/activate
ruff check src/
black --check src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

Expected: every command exits `0`. If a command fails because of unrelated pre-existing issues, capture the exact failing command and first actionable error before changing scope.

- [ ] **Step 3: Check local ModelHub adapter before any Codex benchmark run**

Run:

```bash
curl -sf http://127.0.0.1:8787/health
```

Expected: JSON health response and exit `0`.

If it fails, start the adapter in tmux from `codex_modelhub_adapter`:

```bash
tmux new-session -d -s modelhub_adapter_8787 'cd /Users/bytedance/project/agentic-3d-task/codex_modelhub_adapter && python3 -m uvicorn adapter.app:app --host 127.0.0.1 --port 8787'
curl -sf http://127.0.0.1:8787/health
```

Expected: second health check exits `0`.

- [ ] **Step 4: Create a JSON sample id list for the existing 20-case fold**

Run:

```bash
.venv/bin/python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

source_path = Path("tmp/scenefunc3d/artifacts/scenefunc_parallel20_20260701_2346/sample_ids.txt")
target_path = Path("tmp/scenefunc3d/artifacts/scenefunc_parallel20_20260701_2346/sample_ids.json")
sample_ids = tuple(
    line.strip()
    for line in source_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
)
if len(sample_ids) != 20:
    raise SystemExit(f"expected 20 sample ids, got {len(sample_ids)}")
target_path.write_text(json.dumps(list(sample_ids), indent=2) + "\n", encoding="utf-8")
print(target_path)
PY
```

Expected: prints `tmp/scenefunc3d/artifacts/scenefunc_parallel20_20260701_2346/sample_ids.json`.

- [ ] **Step 5: Run a scored 20-case SceneFunc3D smoke in tmux**

Run:

```bash
tmux new-session -d -s scenefunc_raw_id_20 'cd /Users/bytedance/project/agentic-3d-task && source .venv/bin/activate && PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d --dataset-root data/SceneFun3D --sample-ids-path tmp/scenefunc3d/artifacts/scenefunc_parallel20_20260701_2346/sample_ids.json --backend-config configs/scenefunc3d_backends.toml --output-dir tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_20260702 --score 2>&1 | tee tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_20260702.log'
```

Expected: tmux session starts. Monitor with:

```bash
tmux capture-pane -pt scenefunc_raw_id_20
```

Expected after completion: scored output under `tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_20260702`.

- [ ] **Step 6: Summarize the 20-case result**

Run:

```bash
.venv/bin/python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

run_root = Path("tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_20260702")
result_paths = sorted(run_root.glob("**/result.json"))
if len(result_paths) != 20:
    raise SystemExit(f"expected 20 result.json files, found {len(result_paths)}")
ious: list[float] = []
nonzero = 0
for result_path in result_paths:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    score = payload.get("score")
    if not isinstance(score, dict):
        raise SystemExit(f"missing score in {result_path}")
    metrics = score.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"missing metrics in {result_path}")
    iou = float(metrics["iou"])
    ious.append(iou)
    if iou > 0.0:
        nonzero += 1
summary = {
    "count": len(ious),
    "mean_iou": sum(ious) / len(ious),
    "nonzero_iou": nonzero,
}
summary_path = run_root / "raw_point_id_alignment_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(summary_path)
PY
```

Expected: `count` is `20`; `mean_iou` should be in the same broad range as the diagnostic rawNN result near `0.165`; `nonzero_iou` should be far above the broken `1/20` baseline. The command writes `tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_20260702/raw_point_id_alignment_summary.json`.

- [ ] **Step 7: Add benchmark record**

Run this command to create `docs/benchmark/scenefunc_molmo_sam3d/v4_raw_point_id_alignment_20260702.md` from the actual summary JSON:

```bash
.venv/bin/python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

summary_path = Path(
    "tmp/scenefunc3d/artifacts/scenefunc_raw_id_alignment_20_20260702/"
    "raw_point_id_alignment_summary.json"
)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
doc_path = Path(
    "docs/benchmark/scenefunc_molmo_sam3d/"
    "v4_raw_point_id_alignment_20260702.md"
)
doc_path.write_text(
    "# SceneFunc3D v4 Raw Point ID Alignment, 2026-07-02\n\n"
    "## Purpose\n\n"
    "Validate that SceneFunc3D final `mask_data.npz.point_indices` now uses "
    "raw/source scan vertex ids aligned with hidden annotation `indices`.\n\n"
    "## Run\n\n"
    "- Branch: `feat/scenefunc3d-agent-tools`\n"
    "- Sample ids: `tmp/scenefunc3d/artifacts/"
    "scenefunc_parallel20_20260701_2346/sample_ids.json`\n"
    "- Output root: `tmp/scenefunc3d/artifacts/"
    "scenefunc_raw_id_alignment_20_20260702`\n"
    "- Command:\n\n"
    "```bash\n"
    "PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d \\\n"
    "  --dataset-root data/SceneFun3D \\\n"
    "  --sample-ids-path tmp/scenefunc3d/artifacts/"
    "scenefunc_parallel20_20260701_2346/sample_ids.json \\\n"
    "  --backend-config configs/scenefunc3d_backends.toml \\\n"
    "  --output-dir tmp/scenefunc3d/artifacts/"
    "scenefunc_raw_id_alignment_20_20260702 \\\n"
    "  --score\n"
    "```\n\n"
    "## Result\n\n"
    "```json\n"
    f"{json.dumps(summary, indent=2)}\n"
    "```\n\n"
    "## Interpretation\n\n"
    "This run validates the deterministic point-id alignment fix. Remaining "
    "failures should be treated as agent, SAM candidate, view selection, or "
    "task ambiguity issues rather than scorer index-space bugs.\n",
    encoding="utf-8",
)
print(doc_path)
PY
```

Update `docs/benchmark/scenefunc_molmo_sam3d/README.md` by adding a new top timeline row:

```markdown
| [v4_raw_point_id_alignment_20260702](v4_raw_point_id_alignment_20260702.md) | 2026-07-02 | SceneFunc3D raw/source point-id alignment validation on the 20-case `421254` fold. | Confirms final `mask_data.npz.point_indices` are scored in the same raw mesh id space as hidden annotation indices. Remaining failures are preserved as real model/tool quality issues. |
```

- [ ] **Step 8: Commit benchmark record**

Run:

```bash
git add docs/benchmark/scenefunc_molmo_sam3d/README.md docs/benchmark/scenefunc_molmo_sam3d/v4_raw_point_id_alignment_20260702.md
git commit -m "Record SceneFunc3D raw point ID alignment run"
```

## Final Verification

- [ ] **Step 1: Confirm no unstaged changes except expected runtime artifacts**

Run:

```bash
git status --short
```

Expected: no tracked source or docs changes. Runtime files under `tmp/` and gitignored config files may appear ignored or absent from status.

- [ ] **Step 2: Push after all commits are in place**

Run:

```bash
git push
```

Expected: push succeeds.
