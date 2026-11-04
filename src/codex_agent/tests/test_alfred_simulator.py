"""Simulator boundary tests, independent of Unity and external ALFRED imports."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from codex_agent.alfred.simulator import (
    SimulatorConfig,
    SimulatorError,
    SimulatorObservation,
    ThorSimulator,
)
from codex_agent.alfred.simulator_worker import (
    _SimulatorWorker,
    load_interaction_mask,
    sanitize_action_error,
    verify_simulator_version,
)


@pytest.mark.parametrize(
    ("version", "build"), [("2.2.0", "201909061227"), ("2.1.0", "202001010000")]
)
def test_simulator_version_guard_rejects_package_or_build_drift(
    version: str, build: str
) -> None:
    with pytest.raises(RuntimeError, match="ai2thor==2.1.0.*201909061227"):
        verify_simulator_version(version, build)


def test_simulator_version_guard_accepts_pinned_release() -> None:
    verify_simulator_version("2.1.0", "201909061227")


def test_mask_loading_preserves_predicted_foreground(tmp_path: Path) -> None:
    path = tmp_path / "mask.npy"
    np.save(path, np.array([[0, 1], [1, 0]], dtype=np.uint8))
    assert np.array_equal(load_interaction_mask(path, shape=(2, 2)), [[0, 1], [1, 0]])


@pytest.mark.parametrize("mask", [np.zeros((2, 2)), np.full((2, 2), 0.5)])
def test_mask_loading_rejects_empty_or_soft_masks(
    tmp_path: Path, mask: np.ndarray
) -> None:
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    with pytest.raises(ValueError, match="interaction mask"):
        load_interaction_mask(path, shape=(2, 2))


def test_error_feedback_does_not_leak_simulator_coordinates() -> None:
    objects = [{"objectId": "Apple|+00.24|+00.60|-01.40", "objectType": "Apple"}]
    message = sanitize_action_error(
        "Apple|+00.24|+00.60|-01.40 is not visible", objects
    )
    assert message == "object is not visible"
    assert "|" not in sanitize_action_error("Unknown|+01.00|-02.00|+03.00", [])


def test_observation_rejects_privileged_fields() -> None:
    payload = _observation_payload()
    payload["objects"] = [{"objectId": "Apple|1|2|3"}]
    with pytest.raises(ValidationError, match="Extra inputs"):
        SimulatorObservation.model_validate(payload)


def test_worker_exports_only_rgb_and_proprioception(tmp_path: Path) -> None:
    worker = _SimulatorWorker(alfred_root=tmp_path, output_dir=tmp_path, x_display="5")
    worker.environment = SimpleNamespace(
        last_event=SimpleNamespace(
            frame=np.full((2, 2, 3), 120, dtype=np.uint8),
            metadata={
                "agent": {
                    "rotation": {"y": 90.0},
                    "cameraHorizon": 30.0,
                    "position": {"x": 99, "y": 99, "z": 99},
                },
                "objects": [{"objectId": "Hidden|1|2|3"}],
                "inventoryObjects": [{"objectId": "Held|1|2|3"}],
            },
            depth_frame=np.ones((2, 2)),
            instance_masks={"Hidden|1|2|3": np.ones((2, 2))},
        )
    )
    observation = SimulatorObservation.model_validate(
        worker._observe(success=True, error="")
    )
    assert observation.rgb_path.is_file()
    assert observation.yaw_degrees == 90.0
    assert set(observation.model_dump()) == set(_observation_payload())
    assert list(tmp_path.rglob("*")) == [tmp_path / "frames", observation.rgb_path]


def test_terminal_score_preserves_state_history_and_ends_episode(
    tmp_path: Path,
) -> None:
    worker = _SimulatorWorker(alfred_root=tmp_path, output_dir=tmp_path, x_display="5")
    worker.environment = SimpleNamespace(
        get_goal_satisfied=lambda: False,
        get_goal_conditions_met=lambda: (1, 3),
        last_event=SimpleNamespace(metadata={"objects": []}),
        cleaned_objects={"cleaned"},
        cooled_objects={"cooled"},
        heated_objects={"heated"},
    )
    score = worker.score()
    assert score["success"] is False
    assert score["completed_goal_conditions"] == 1
    assert score["total_goal_conditions"] == 3
    snapshot = (tmp_path / "terminal_evaluation.json").read_text()
    assert all(state in snapshot for state in ("cleaned", "cooled", "heated"))
    with pytest.raises(RuntimeError, match="terminal scoring"):
        worker.step("MoveAhead", None)


def test_rpc_failure_does_not_fabricate_observation(tmp_path: Path) -> None:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        " print(json.dumps({'ok':False,'error_type':'ValueError',"
        "'message':'bad scene'}), flush=True)\n",
        encoding="utf-8",
    )
    config = _config(tmp_path)
    with ThorSimulator(config, worker_path=worker) as simulator:
        with pytest.raises(SimulatorError, match="bad scene"):
            simulator.reset(tmp_path / "trajectory.json")


def test_rpc_timeout_stops_owned_worker(tmp_path: Path) -> None:
    worker = tmp_path / "fake_worker.py"
    worker.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    config = _config(tmp_path, timeout_seconds=0.05)
    simulator = ThorSimulator(config, worker_path=worker)
    with pytest.raises(SimulatorError, match="timed out"):
        simulator.reset(tmp_path / "trajectory.json")
    assert simulator.is_closed


def test_rpc_rejects_invalid_action_before_sending(tmp_path: Path) -> None:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " print(json.dumps({'ok':True,'result':{}}),flush=True)\n",
        encoding="utf-8",
    )
    with ThorSimulator(_config(tmp_path), worker_path=worker) as simulator:
        with pytest.raises(ValueError, match="action"):
            simulator.step("TeleportFull")
        with pytest.raises(ValueError, match="mask_path"):
            simulator.step("PickupObject")


def _config(tmp_path: Path, *, timeout_seconds: float = 2.0) -> SimulatorConfig:
    (tmp_path / "trajectory.json").write_text("{}", encoding="utf-8")
    return SimulatorConfig(
        python_executable=Path(sys.executable),
        alfred_root=tmp_path,
        output_dir=tmp_path / "output",
        timeout_seconds=timeout_seconds,
    )


def _observation_payload() -> dict[str, object]:
    return {
        "frame_id": 0,
        "rgb_path": "rgb.png",
        "yaw_degrees": 0,
        "horizon_degrees": 0,
        "last_action_success": True,
        "last_action_error": "",
    }
