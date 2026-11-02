"""Capture real denoise inputs without importing models or changing evaluation.

Capture is explicitly disabled by default. A complete immutable directory is
published atomically; loading validates its JSON, SHA256 and non-pickle arrays.
NumPy is imported only when a point-cloud operation is explicitly requested.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, Union, cast
from zipfile import ZIP_STORED, ZipFile

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float32] | NDArray[np.float64]

MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
_SOURCE_FILE = "src/conceptgraph/slam/utils.py"
Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
CommitHash = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
ArrayDtype = Literal["float32", "float64"]


class DenoiseParameters(BaseModel):
    """All actual denoise call parameters, including its recorded source device."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    downsample_voxel_size: float = Field(gt=0, allow_inf_nan=False)
    dbscan_remove_noise: bool
    dbscan_eps: float = Field(gt=0, allow_inf_nan=False)
    dbscan_min_points: int = Field(gt=0)
    spatial_sim_type: str = Field(min_length=1)
    device: str = Field(min_length=1)
    run_dbscan: Literal[True] = True


def _array(value: object, *, name: str) -> FloatArray:
    import numpy as np

    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape N x 3")
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"{name} dtype must be float32 or float64")
    if array.nbytes > MAX_CAPTURE_BYTES or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain bounded finite values")
    copied = array.copy(order="C")
    copied.setflags(write=False)
    return cast("FloatArray", copied)


@dataclass(frozen=True, eq=False)
class PointCloudInput:
    """An original integer ID with owned read-only copies of XYZ and colors."""

    object_id: int
    points: FloatArray
    colors: FloatArray

    def __post_init__(self) -> None:
        if type(self.object_id) is not int:
            raise ValueError("object_id must be an integer, not a boolean")
        points = _array(self.points, name="points")
        colors = _array(self.colors, name="colors")
        if len(colors) not in (0, len(points)):
            raise ValueError("colors must have N rows or be empty")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors", colors)

    @classmethod
    def from_arrays(
        cls, object_id: int, points: object, colors: object
    ) -> PointCloudInput:
        """Validate the external array boundary without touching original arrays."""
        return cls(object_id, cast("FloatArray", points), cast("FloatArray", colors))


class _ObjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    object_id: int
    point_count: int = Field(ge=0)
    color_count: int = Field(ge=0)
    points_dtype: ArrayDtype
    colors_dtype: ArrayDtype


class CaptureMetadata(BaseModel):
    """Small validated provenance and array descriptors; no models or credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    parameters: DenoiseParameters
    source_commit_sha: CommitHash
    source_file_sha256: Digest
    archive_sha256: Digest
    archive_bytes: int = Field(gt=0)
    objects: tuple[_ObjectMetadata, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_objects(self) -> CaptureMetadata:
        ids = tuple(item.object_id for item in self.objects)
        if len(set(ids)) != len(ids):
            raise ValueError("captured object IDs must be unique")
        if not any(item.point_count > 1 for item in self.objects):
            raise ValueError(
                "capture requires at least one cloud with more than one point"
            )
        if any(item.color_count not in (0, item.point_count) for item in self.objects):
            raise ValueError("captured colors must match points or be empty")
        return self


@dataclass(frozen=True)
class CapturedInputs:
    metadata: CaptureMetadata
    inputs: tuple[PointCloudInput, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_denoise_inputs(
    inputs: Sequence[PointCloudInput],
    parameters: DenoiseParameters,
    *,
    capture_dir: Path,
    upstream_root: Path,
    enabled: bool = False,
) -> CaptureMetadata | None:
    """Publish a real snapshot once; disabled capture has no IO side effects.

    The destination must not already exist. The caller must reject normals or
    other point-cloud attributes, because this contract captures XYZ/colors only.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if capture_dir.exists():
        raise FileExistsError("capture directory already exists")
    snapshots = tuple(inputs)
    if not snapshots or len({item.object_id for item in snapshots}) != len(snapshots):
        raise ValueError("capture requires nonempty unique object IDs")
    if (
        sum(item.points.nbytes + item.colors.nbytes for item in snapshots)
        > MAX_CAPTURE_BYTES
    ):
        raise ValueError("capture arrays exceed the size limit")
    source_sha = _sha256(upstream_root / _SOURCE_FILE)
    commit = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    capture_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pointcloud-", dir=capture_dir.parent))
    try:
        import numpy as np

        arrays: dict[str, FloatArray] = {}
        descriptors: list[_ObjectMetadata] = []
        for index, item in enumerate(snapshots):
            arrays[f"points_{index}"] = item.points
            arrays[f"colors_{index}"] = item.colors
            descriptors.append(
                _ObjectMetadata(
                    object_id=item.object_id,
                    point_count=len(item.points),
                    color_count=len(item.colors),
                    points_dtype=cast(ArrayDtype, item.points.dtype.name),
                    colors_dtype=cast(ArrayDtype, item.colors.dtype.name),
                )
            )
        archive = temporary / "inputs.npz"
        with archive.open("wb") as destination:
            with ZipFile(destination, "w") as zipped:
                for key, array in arrays.items():
                    with zipped.open(key + ".npy", "w") as member:
                        np.lib.format.write_array(member, array, allow_pickle=False)
            destination.flush()
            os.fsync(destination.fileno())
        if archive.stat().st_size > MAX_CAPTURE_BYTES:
            raise ValueError("capture archive exceeds the size limit")
        metadata = CaptureMetadata(
            parameters=parameters,
            source_commit_sha=commit,
            source_file_sha256=source_sha,
            archive_sha256=_sha256(archive),
            archive_bytes=archive.stat().st_size,
            objects=tuple(descriptors),
        )
        metadata_json = metadata.model_dump_json(indent=2) + "\n"
        if len(metadata_json.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError("capture metadata exceeds the size limit")
        with (temporary / "capture.json").open("w", encoding="utf-8") as destination:
            destination.write(metadata_json)
            destination.flush()
            os.fsync(destination.fileno())
        os.rename(temporary, capture_dir)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_capture(capture_dir: Path) -> CapturedInputs:
    """Validate a bounded snapshot; never execute code or deserialize pickle."""
    metadata_path = capture_dir / "capture.json"
    if metadata_path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("capture metadata exceeds the size limit")
    metadata = CaptureMetadata.model_validate_json(metadata_path.read_bytes())
    archive = capture_dir / "inputs.npz"
    size = archive.stat().st_size
    if size > MAX_CAPTURE_BYTES or size != metadata.archive_bytes:
        raise ValueError("capture archive size mismatch or exceeds limit")
    if _sha256(archive) != metadata.archive_sha256:
        raise ValueError("capture archive SHA256 mismatch")
    keys = {
        f"{kind}_{index}"
        for index in range(len(metadata.objects))
        for kind in ("points", "colors")
    }
    import numpy as np

    with ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) != len(keys) or {item.filename for item in members} != {
            key + ".npy" for key in keys
        }:
            raise ValueError("capture archive array keys mismatch")
        if sum(item.file_size for item in members) > MAX_CAPTURE_BYTES:
            raise ValueError("uncompressed capture exceeds the size limit")
        for index, descriptor in enumerate(metadata.objects):
            for kind, row_count, dtype_name in (
                ("points", descriptor.point_count, descriptor.points_dtype),
                ("colors", descriptor.color_count, descriptor.colors_dtype),
            ):
                member = zipped.getinfo(f"{kind}_{index}.npy")
                if member.compress_type != ZIP_STORED:
                    raise ValueError("capture requires uncompressed NPZ members")
                with zipped.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                            stream, max_header_size=4096
                        )
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                            stream, max_header_size=4096
                        )
                    else:
                        raise ValueError("capture supports NPY versions 1.0 and 2.0")
                    if (
                        shape != (row_count, 3)
                        or fortran
                        or dtype.hasobject
                        or dtype != np.dtype(dtype_name)
                    ):
                        raise ValueError("capture NPY header descriptor mismatch")
                    expected_bytes = stream.tell() + row_count * 3 * dtype.itemsize
                    if (
                        expected_bytes > MAX_CAPTURE_BYTES
                        or expected_bytes != member.file_size
                    ):
                        raise ValueError(
                            "capture NPY declared size exceeds budget or payload"
                        )

    inputs: list[PointCloudInput] = []
    with np.load(archive, allow_pickle=False) as arrays:
        for index, descriptor in enumerate(metadata.objects):
            item = PointCloudInput.from_arrays(
                descriptor.object_id,
                arrays[f"points_{index}"],
                arrays[f"colors_{index}"],
            )
            if (
                len(item.points),
                len(item.colors),
                item.points.dtype.name,
                item.colors.dtype.name,
            ) != (
                descriptor.point_count,
                descriptor.color_count,
                descriptor.points_dtype,
                descriptor.colors_dtype,
            ):
                raise ValueError("capture array descriptor mismatch")
            inputs.append(item)
    return CapturedInputs(metadata, tuple(inputs))


class PointCloudComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    object_id: int
    exact_match: bool
    max_absolute_difference: Union[float, None]  # noqa: UP007 - Pydantic on Python 3.9


def _sorted_point_colors(item: PointCloudInput) -> FloatArray:
    import numpy as np

    rows = (
        np.concatenate((item.points, item.colors), axis=1)
        if len(item.colors)
        else item.points
    )
    indices = np.lexsort(
        tuple(rows[:, column] for column in reversed(range(rows.shape[1])))
    )
    return cast("FloatArray", rows[indices])


def compare_pointcloud_outputs(
    reference: Sequence[PointCloudInput], actual: Sequence[PointCloudInput]
) -> tuple[PointCloudComparison, ...]:
    """Compare coordinates and associated colors jointly after row sorting."""
    import numpy as np

    reference_ids = tuple(item.object_id for item in reference)
    actual_by_id = {item.object_id: item for item in actual}
    if (
        len(set(reference_ids)) != len(reference_ids)
        or len(actual_by_id) != len(actual)
        or set(reference_ids) != set(actual_by_id)
    ):
        raise ValueError("replayed object IDs must match exactly")
    comparisons: list[PointCloudComparison] = []
    for item in reference:
        expected = _sorted_point_colors(item)
        observed = _sorted_point_colors(actual_by_id[item.object_id])
        same_shape = expected.shape == observed.shape
        difference: float | None = None
        if same_shape:
            difference = (
                float(np.max(np.abs(expected - observed))) if expected.size else 0.0
            )
        comparisons.append(
            PointCloudComparison(
                object_id=item.object_id,
                exact_match=same_shape and bool(np.array_equal(expected, observed)),
                max_absolute_difference=difference,
            )
        )
    return tuple(comparisons)


class _ReplayPointCloud(Protocol):
    points: object
    colors: object


class _Geometry(Protocol):
    def PointCloud(self) -> _ReplayPointCloud: ...


class _Utility(Protocol):
    def Vector3dVector(self, values: FloatArray) -> object: ...


class _Open3D(Protocol):
    geometry: _Geometry
    utility: _Utility


class _ProcessPointCloud(Protocol):
    def __call__(
        self,
        pcd: _ReplayPointCloud,
        downsample_voxel_size: float,
        dbscan_remove_noise: bool,
        dbscan_eps: float,
        dbscan_min_points: int,
        *,
        run_dbscan: bool,
    ) -> _ReplayPointCloud: ...


class _ObjectReplay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    object_id: int
    outcome: Literal["skipped_small_input", "processed", "restored_small_output"]
    processed_point_count: Union[int, None]  # noqa: UP007 - Pydantic on Python 3.9


class _ReplayRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    thread_count: int = Field(gt=0, le=96)
    elapsed_process_seconds: float = Field(ge=0, allow_inf_nan=False)
    objects: tuple[_ObjectReplay, ...]


class _ThreadComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    reference_threads: int
    actual_threads: int
    outcomes_match: bool
    pointclouds: tuple[PointCloudComparison, ...]


class _CalibrationReport(BaseModel):
    """Denoise point-cloud core only; excludes reconstruction, bbox and merge."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    input_archive_sha256: Digest
    input_metadata_sha256: Digest
    parameters: DenoiseParameters
    source_commit_sha: CommitHash
    source_file_sha256: Digest
    process_pcd_only: Literal[True] = True
    includes_bbox_merge_or_evaluation: Literal[False] = False
    runs: tuple[_ReplayRun, ...]
    comparisons: tuple[_ThreadComparison, ...]
    equivalent: bool


def _load_original_process(
    upstream_root: Path, metadata: CaptureMetadata, open3d: _Open3D
) -> _ProcessPointCloud:
    """Compile only the two original functions from verified source bytes."""
    import numpy as np

    source_path = upstream_root / _SOURCE_FILE
    payload = source_path.read_bytes()
    head = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        head != metadata.source_commit_sha
        or hashlib.sha256(payload).hexdigest() != metadata.source_file_sha256
    ):
        raise ValueError("replay source does not match captured runtime and SHA256")
    names = {"process_pcd", "pcd_denoise_dbscan"}
    functions = [
        node
        for node in ast.parse(payload).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if (
        len(functions) != 2
        or {node.name for node in functions} != names
        or any(node.decorator_list for node in functions)
    ):
        raise ValueError("original process_pcd function boundary changed")
    namespace: dict[str, object] = {"np": np, "o3d": open3d, "Counter": Counter}
    body: list[ast.stmt] = list(functions)
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    processor = namespace.get("process_pcd")
    if not callable(processor):
        raise ValueError("original process_pcd is not callable")
    return cast(_ProcessPointCloud, processor)


def _replay_child(
    *,
    capture_dir: Path,
    upstream_root: Path,
    output_dir: Path,
    thread_count: int,
    environment: Mapping[str, str],
) -> None:
    if environment.get("CUDA_VISIBLE_DEVICES") != "" or any(
        environment.get(key) != str(thread_count) for key in _THREAD_ENVIRONMENT_KEYS
    ):
        raise ValueError(
            "CPU replay child requires explicit disabled CUDA and thread limits"
        )
    import numpy as np

    captured = load_capture(capture_dir)
    open3d = cast(_Open3D, importlib.import_module("open3d.cpu.pybind"))
    process = _load_original_process(upstream_root, captured.metadata, open3d)
    parameters = captured.metadata.parameters
    outputs: list[PointCloudInput] = []
    results: list[_ObjectReplay] = []
    elapsed = 0.0
    for item in captured.inputs:
        processed_count: int | None = None
        if len(item.points) <= 1:
            output = item
            outcome: Literal[
                "skipped_small_input", "processed", "restored_small_output"
            ] = "skipped_small_input"
        else:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(item.points)
            pcd.colors = open3d.utility.Vector3dVector(item.colors)
            started = time.perf_counter()
            processed = process(
                pcd,
                parameters.downsample_voxel_size,
                parameters.dbscan_remove_noise,
                parameters.dbscan_eps,
                parameters.dbscan_min_points,
                run_dbscan=parameters.run_dbscan,
            )
            elapsed += time.perf_counter() - started
            output = PointCloudInput.from_arrays(
                item.object_id,
                np.asarray(processed.points),
                np.asarray(processed.colors),
            )
            processed_count = len(output.points)
            outcome = "processed"
            if processed_count < 4:
                output = item
                outcome = "restored_small_output"
        outputs.append(output)
        results.append(
            _ObjectReplay(
                object_id=item.object_id,
                outcome=outcome,
                processed_point_count=processed_count,
            )
        )
    capture_denoise_inputs(
        outputs,
        parameters,
        capture_dir=output_dir,
        upstream_root=upstream_root,
        enabled=True,
    )
    run = _ReplayRun(
        thread_count=thread_count,
        elapsed_process_seconds=elapsed,
        objects=tuple(results),
    )
    (output_dir / "replay.json").write_text(run.model_dump_json(), encoding="utf-8")


_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _cpu_environment(base: Mapping[str, str], thread_count: int) -> dict[str, str]:
    environment = dict(base)
    environment.update({key: str(thread_count) for key in _THREAD_ENVIRONMENT_KEYS})
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_THREAD_LIMIT"] = str(thread_count)
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["MKL_DYNAMIC"] = "FALSE"
    environment["NUMEXPR_MAX_THREADS"] = str(thread_count)
    return environment


def run_thread_calibration(
    *,
    capture_dir: Path,
    upstream_root: Path,
    threads: tuple[int, ...],
    report_file: Path,
    environment: Mapping[str, str],
) -> bool:
    """Run explicit CPU children and require exact sorted output equivalence."""
    if (
        len(threads) < 2
        or len(set(threads)) != len(threads)
        or any(type(count) is not int or not 1 <= count <= 96 for count in threads)
    ):
        raise ValueError("threads require at least two unique integers in 1..96")
    if capture_dir.resolve() in (report_file.resolve(), *report_file.resolve().parents):
        raise ValueError("replay report must not modify the input capture directory")
    captured = load_capture(capture_dir)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    runs: list[_ReplayRun] = []
    comparisons: list[_ThreadComparison] = []
    with tempfile.TemporaryDirectory(
        prefix=".denoise-replay-", dir=report_file.parent
    ) as temporary:
        root = Path(temporary)
        for count in threads:
            output = root / f"thread_{count}"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_cpu-child",
                    "--capture-dir",
                    str(capture_dir),
                    "--upstream-root",
                    str(upstream_root),
                    "--threads",
                    str(count),
                    "--output-dir",
                    str(output),
                ],
                env=_cpu_environment(environment, count),
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            runs.append(
                _ReplayRun.model_validate_json((output / "replay.json").read_bytes())
            )
        reference = load_capture(root / f"thread_{threads[0]}")
        for index, count in enumerate(threads[1:], 1):
            actual = load_capture(root / f"thread_{count}")
            comparisons.append(
                _ThreadComparison(
                    reference_threads=threads[0],
                    actual_threads=count,
                    outcomes_match=runs[0].objects == runs[index].objects,
                    pointclouds=compare_pointcloud_outputs(
                        reference.inputs, actual.inputs
                    ),
                )
            )
    equivalent = all(
        comparison.outcomes_match
        and all(item.exact_match for item in comparison.pointclouds)
        for comparison in comparisons
    )
    report = _CalibrationReport(
        input_archive_sha256=captured.metadata.archive_sha256,
        input_metadata_sha256=_sha256(capture_dir / "capture.json"),
        parameters=captured.metadata.parameters,
        source_commit_sha=captured.metadata.source_commit_sha,
        source_file_sha256=captured.metadata.source_file_sha256,
        runs=tuple(runs),
        comparisons=tuple(comparisons),
        equivalent=equivalent,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".denoise-report-", dir=report_file.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(report.model_dump_json(indent=2) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, report_file)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return equivalent


class _ReplayOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    command: Literal["replay"]
    capture_dir: Path
    upstream_root: Path
    threads: tuple[Annotated[int, Field(gt=0, le=96, strict=True)], ...] = Field(
        strict=False, min_length=2
    )
    report_file: Path


class _ChildOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    command: Literal["_cpu-child"]
    capture_dir: Path
    upstream_root: Path
    threads: int = Field(gt=0, le=96)
    output_dir: Path


_CliOptions = Annotated[
    Union[_ReplayOptions, _ChildOptions],  # noqa: UP007 - evaluated on Python 3.9
    Field(discriminator="command"),
]
_CLI_ADAPTER: TypeAdapter[_CliOptions] = TypeAdapter(_CliOptions)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options; heavy CPU operations remain in explicit child commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay")
    child = commands.add_parser("_cpu-child", help=argparse.SUPPRESS)
    for command in (replay, child):
        command.add_argument("--capture-dir", type=Path, required=True)
        command.add_argument("--upstream-root", type=Path, required=True)
    replay.add_argument("--threads", type=int, nargs="+", required=True)
    replay.add_argument("--report-file", type=Path, required=True)
    child.add_argument("--threads", type=int, required=True, choices=range(1, 97))
    child.add_argument("--output-dir", type=Path, required=True)
    options = _CLI_ADAPTER.validate_python(vars(parser.parse_args(argv)))
    if isinstance(options, _ChildOptions):
        _replay_child(
            capture_dir=options.capture_dir,
            upstream_root=options.upstream_root,
            output_dir=options.output_dir,
            thread_count=options.threads,
            environment=os.environ,
        )
        return 0
    return (
        0
        if run_thread_calibration(
            capture_dir=options.capture_dir,
            upstream_root=options.upstream_root,
            threads=options.threads,
            report_file=options.report_file,
            environment=os.environ,
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
