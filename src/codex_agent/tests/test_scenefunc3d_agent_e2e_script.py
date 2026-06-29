"""Tests for the SceneFunc3D real agent-runner E2E script contract."""

from __future__ import annotations

from pathlib import Path


def test_agent_e2e_script_runs_runner_with_sidecars_and_scoring() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert "codex_agent.scenefunc3d.servers.molmo_point_server" in script_text
    assert "codex_agent.scenefunc3d.servers.sam2_mask_server" in script_text
    assert "codex_agent.scenefunc3d.runner" in script_text
    assert '"--score"' in script_text
    assert "--backend-config" in script_text
    assert "check_adapter_ready()" in script_text
    assert "ensure_codex_home()" in script_text


def test_agent_e2e_script_does_not_manual_chain_mask_tools() -> None:
    script_text = _agent_e2e_script_path().read_text(encoding="utf-8")

    assert '"molmo_point",' not in script_text
    assert '"sam_mask",' not in script_text
    assert '"lift_mask_to_3d",' not in script_text
    assert '"fuse_accepted_masks",' not in script_text


def _agent_e2e_script_path() -> Path:
    return (
        Path.cwd()
        / "docs"
        / "benchmark"
        / "scenefunc_molmo_sam3d"
        / "assets"
        / "run_agent_runner_e2e_20260629.sh"
    )
