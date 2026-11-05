"""Train-only capture planning and fake simulator regressions; no real capture."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from codex_agent.alfred import calibration_capture as capture
from codex_agent.alfred.calibration_capture_worker import (
    _CalibrationWorker,
    depth_meters,
)
from codex_agent.cli.capture_alfred_calibration import build_arg_parser


def _sources(root: Path) -> tuple[Path, Path]:
    alfred, thor = root / "alfred", root / "ai2thor"
    for path in (
        alfred / "env/thor_env.py",
        alfred / "gen/constants.py",
        thor / "server.py",
        thor / "controller.py",
        thor / "__init__.py",
        thor / "_builds.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic pinned source\n")
    return alfred, thor


def _plan(root: Path) -> capture.CapturePlan:
    alfred, thor = _sources(root)
    unity = root / "synthetic-unity"
    unity.write_bytes(b"synthetic binary fixture")
    entries = []
    for scene in capture.SCENES:
        for suffix in ("a", "b"):
            task = f"task-{scene}/trial_{suffix}"
            path = root / "data/train" / task / "traj_data.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"scene": {"floor_plan": scene}}))
            entries.extend(
                [{"task": task, "repeat_idx": 0}, {"task": task, "repeat_idx": 1}]
            )
    splits = root / "splits.json"
    splits.write_text(json.dumps({"train": entries, "valid_seen": []}))
    return capture.create_capture_plan(
        data_root=root / "data",
        splits_path=splits,
        alfred_root=alfred,
        ai2thor_root=thor,
        unity_executable=unity,
        code={"branch": "test", "executed_code_sha256": "a" * 64},
    )


def test_plan_selection_uses_fixed_scene_coverage_and_trajectory_identity(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    assert tuple(item.scene_id for item in plan.trajectories) == capture.SCENES
    for selected in plan.trajectories:
        candidates = [
            f"task-{selected.scene_id}/trial_{suffix}" for suffix in ("a", "b")
        ]
        assert selected.task == min(candidates, key=capture._selection_hash)
    capture.validate_capture_plan(plan)
    path = tmp_path / "capture_plan.json"
    capture.write_capture_plan(plan, path)
    assert capture.CapturePlan.model_validate_json(path.read_bytes()) == plan
    with pytest.raises(FileExistsError):
        capture.write_capture_plan(plan, path)


def test_capture_plan_rejects_validation_split_and_source_drift(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    payload = plan.model_dump(mode="json")
    payload["split"] = "valid_seen"
    with pytest.raises(ValidationError):
        capture.CapturePlan.model_validate(payload)
    (plan.ai2thor_root / "server.py").write_text("# different implementation\n")
    with pytest.raises(ValueError, match="frozen capture plan"):
        capture.validate_capture_plan(plan)


def test_frozen_plan_rejects_train_membership_or_trajectory_change(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    selected = plan.trajectories[0]
    selected.trajectory_path.write_text(
        json.dumps({"scene": {"floor_plan": selected.scene_id}, "different": True})
    )
    with pytest.raises(ValueError, match="frozen capture plan"):
        capture.validate_capture_plan(plan)


def test_unsafe_task_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="safe relative"):
        capture.CaptureTrajectory.safe_task("../outside")


def test_removing_selected_train_membership_cannot_silently_reselect(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    split = json.loads(plan.splits_path.read_text())
    split["train"] = [
        entry for entry in split["train"] if entry["task"] != plan.trajectories[0].task
    ]
    plan.splits_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="frozen capture plan"):
        capture.validate_capture_plan(plan)


def test_depth_conversion_uses_absolute_millimeter_scale_and_preserves_saturation() -> (
    None
):
    depth = np.full((300, 300), 2000, dtype=np.float32)
    depth[0, :3] = [0, 500, 5000]
    converted = depth_meters(depth)
    np.testing.assert_allclose(converted[0, :4], [0, 0.5, 5, 2])
    assert converted.dtype == np.float32
    assert converted[200, 200] == 2  # No min/max or median scale correction.


@pytest.mark.parametrize("invalid", [np.nan, -1.0, 5001.0])
def test_depth_invalid_values_fail_explicitly(invalid: float) -> None:
    depth = np.full((300, 300), 1000.0)
    depth[0, 0] = invalid
    with pytest.raises(ValueError, match="millimeters"):
        depth_meters(depth)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.last_event = SimpleNamespace(
            frame=np.full((300, 300, 3), 20, dtype=np.uint8),
            depth_frame=np.full((300, 300), 1500.0, dtype=np.float32),
            metadata={
                "lastActionSuccess": True,
                "errorMessage": "",
                "agent": {"cameraHorizon": 30.0, "rotation": {"y": 0.0}},
            },
        )
        self.commands: list[str] = []
        self.move_count = 0

    def step(self, action: dict[str, object]) -> SimpleNamespace:
        name = str(action["action"])
        self.commands.append(name)
        if name == "MoveAhead":
            self.move_count += 1
        success = not (name == "MoveAhead" and self.move_count == 2)
        self.last_event.metadata.update(
            lastActionSuccess=success, errorMessage="blocked" if not success else ""
        )
        return self.last_event

    def look_angle(
        self, angle: float, *, render_settings: dict[str, bool]
    ) -> SimpleNamespace:
        assert (
            render_settings["renderDepthImage"]
            and not render_settings["renderObjectImage"]
        )
        self.last_event.metadata["agent"]["cameraHorizon"] += angle
        self.last_event.metadata.update(lastActionSuccess=True, errorMessage="")
        return self.last_event


def test_scripted_stations_keep_failed_moves_and_never_reselect_views(
    tmp_path: Path,
) -> None:
    worker = _CalibrationWorker(
        tmp_path, tmp_path, tmp_path, "5", tmp_path / "unused-unity"
    )
    environment = _FakeEnvironment()
    worker.environment = environment
    frames = worker._capture_stations("FloorPlan1")
    assert len(frames) == 15
    assert [frame["pitch_degrees"] for frame in frames] == list(capture.PITCHES) * 3
    assert environment.commands == [
        "RotateRight",
        "MoveAhead",
        "MoveAhead",
        "RotateLeft",
        "MoveAhead",
        "MoveAhead",
    ]
    receipts = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl").read_text().splitlines()
    ]
    assert sum(record["success"] is False for record in receipts) == 1
    assert all(
        "camera_to_world" not in frame for frame in frames
    )  # No guessed camera height.
    with Path(str(frames[0]["gt_depth_path"])).open("rb") as stream:
        depth = np.load(stream, allow_pickle=False)
    assert np.all(depth == 1.5)
    result = capture._CapturedScene.model_validate(
        {
            "scene_id": "FloorPlan1",
            "frames": frames,
            "actions_path": str(tmp_path / "actions.jsonl"),
            "actions_sha256": capture.file_sha256(tmp_path / "actions.jsonl"),
            "depth_units": "meters_from_ai2thor_2.1_mm_div_1000",
            "depth_saturation_m": 5.0,
        }
    )
    capture._validate_captured_scene(result, "FloorPlan1", tmp_path)


def test_worker_remains_python38_syntax_and_imports_no_actor() -> None:
    path = Path(capture.__file__).with_name("calibration_capture_worker.py")
    tree = ast.parse(path.read_text(), feature_version=(3, 8))
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(name and name.startswith("codex_agent") for name in modules)
    assert "get_goal_conditions_met" not in path.read_text()


def test_cli_plan_and_capture_are_separate_operations() -> None:
    parsed = build_arg_parser().parse_args(
        [
            "capture",
            "--plan",
            "plan.json",
            "--output-dir",
            "new-capture",
            "--simulator-python",
            "/existing/python",
        ]
    )
    assert parsed.command == "capture" and parsed.x_display == "5"
    assert not hasattr(parsed, "model")


def test_capture_requires_clean_committed_code_before_creating_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def dirty(_: Path) -> dict[str, object]:
        raise ValueError("dirty source")

    monkeypatch.setattr(capture, "code_provenance", dirty)
    with pytest.raises(ValueError, match="dirty source"):
        capture.run_capture(
            plan_path=tmp_path / "plan.json",
            output_dir=tmp_path / "output",
            project_root=tmp_path,
            simulator_python=tmp_path / "python",
            x_display="5",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("symlinked_interpreter", [False, True])
def test_complete_offline_manifest_contains_fixed_60_train_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlinked_interpreter: bool
) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "capture_plan.json"
    capture.write_capture_plan(plan, plan_path)
    interpreter = tmp_path / "fake-python"
    interpreter.write_text("not executed")
    if symlinked_interpreter:
        environment_python = tmp_path / "venv" / "bin" / "python"
        environment_python.parent.mkdir(parents=True)
        environment_python.symlink_to(interpreter)
        interpreter = environment_python
    closed: list[bool] = []

    class FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            command = kwargs["command"]
            assert isinstance(command, list)
            assert command[0] == str(interpreter.absolute())
            output = Path(command[command.index("--output-dir") + 1])
            self.worker = _CalibrationWorker(
                plan.alfred_root, plan.ai2thor_root, output, "5", plan.unity_executable
            )

        def __enter__(self) -> FakeWorker:
            return self

        def __exit__(self, *_: object) -> None:
            closed.append(True)

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            return self.worker.capture(payload)

    def initialize(worker: _CalibrationWorker, _: dict[str, object]) -> None:
        worker.environment = _FakeEnvironment()

    monkeypatch.setattr(capture, "code_provenance", lambda _: plan.code)
    monkeypatch.setattr(capture, "JsonWorker", FakeWorker)
    monkeypatch.setattr(_CalibrationWorker, "_initialize", initialize)
    output = tmp_path / "captured"
    path = capture.run_capture(
        plan_path=plan_path,
        output_dir=output,
        project_root=tmp_path,
        simulator_python=interpreter,
        x_display="5",
    )
    manifest = json.loads(path.read_text())
    assert manifest["split"] == "train"
    assert (
        len(manifest["frames"])
        == len({row["frame_id"] for row in manifest["frames"]})
        == 60
    )
    assert {row["scene_id"] for row in manifest["frames"]} == set(capture.SCENES)
    assert manifest["provenance"]["depth_saturation_m"] == 5
    assert manifest["provenance"]["learned_checkpoints"] == {}
    assert len(closed) == 4
    assert (
        json.loads((output / "capture_status.json").read_text())["status"]
        == "completed"
    )
