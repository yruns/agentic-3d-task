"""Frozen-fold sharding is deterministic, exhaustive, and never silently empty."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from codex_agent.alfred.dataset import AlfredEpisode, AlfredInstruction
from codex_agent.alfred.execution import ExecutionConfig
from codex_agent.alfred.simulator import SimulatorConfig
from codex_agent.cli.run_alfred import build_arg_parser
from codex_agent.config import CodexAgentConfig
from codex_agent.evaluation import alfred_runner
from codex_agent.tests.test_alfred_settings import local_scene_config


@pytest.fixture
def shard_config(tmp_path: Path) -> alfred_runner.AlfredRunConfig:
    fold = tmp_path / "fold.json"
    splits = tmp_path / "splits.json"
    fold.write_text("{}")
    splits.write_text("{}")
    return alfred_runner.AlfredRunConfig(
        source_root=tmp_path,
        data_root=tmp_path,
        splits_path=splits,
        fold_path=fold,
        output_dir=tmp_path / "run",
        run_id="shard-test",
        database_path=tmp_path / "runs.sqlite",
        simulator=SimulatorConfig(Path(sys.executable), tmp_path, tmp_path / "run"),
        perception=local_scene_config(tmp_path / "assets"),
        runtime=CodexAgentConfig(project_root=tmp_path),
        execution=ExecutionConfig(),
        command=("python", "-m", "codex_agent.cli.run_alfred"),
    )


def _manifest() -> tuple[AlfredEpisode, ...]:
    return tuple(
        AlfredEpisode(
            split="valid_seen" if index < 25 else "valid_unseen",
            task=f"type/trial{index}",
            repeat_idx=0,
            task_id=f"trial{index}",
            task_type="pick_and_place_simple",
            trajectory_path=Path(f"/trial{index}.json"),
            trajectory_sha256="a" * 64,
            expert_actions=10,
            instruction=AlfredInstruction("Move the object.", ()),
        )
        for index in range(50)
    )


def test_four_shards_cover_full_fifty_once_with_stable_order(
    shard_config: alfred_runner.AlfredRunConfig,
) -> None:
    manifest = _manifest()
    shards = [
        alfred_runner._selected_episodes(
            replace(shard_config, shard_index=index, num_shards=4), manifest
        )
        for index in range(4)
    ]
    assert [len(shard) for shard in shards] == [13, 13, 12, 12]
    assert shards == [manifest[index::4] for index in range(4)]
    assert Counter(
        episode.episode_id for shard in shards for episode in shard
    ) == Counter(episode.episode_id for episode in manifest)


def test_default_shard_preserves_full_fold_and_limit_applies_after_sharding(
    shard_config: alfred_runner.AlfredRunConfig,
) -> None:
    manifest = _manifest()
    assert alfred_runner._selected_episodes(shard_config, manifest) == manifest
    config = replace(shard_config, shard_index=2, num_shards=4, limit=2)
    assert alfred_runner._selected_episodes(config, manifest) == (
        manifest[2],
        manifest[6],
    )


@pytest.mark.parametrize(
    ("index", "count"), [(-1, 4), (4, 4), (0, 0), (0, -1), (True, 4), (0, 1.5)]
)
def test_rejects_invalid_shard_parameters(
    shard_config: alfred_runner.AlfredRunConfig, index: object, count: object
) -> None:
    with pytest.raises(ValueError, match="shard"):
        replace(shard_config, shard_index=cast(int, index), num_shards=cast(int, count))


def test_rejects_empty_shard_and_duplicates_anywhere_in_full_manifest(
    shard_config: alfred_runner.AlfredRunConfig,
) -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="no ALFRED episodes.*shard_index=3"):
        alfred_runner._selected_episodes(
            replace(shard_config, shard_index=3, num_shards=4), manifest[:2]
        )
    with pytest.raises(ValueError, match="duplicate"):
        alfred_runner._selected_episodes(
            replace(shard_config, shard_index=0, num_shards=4),
            manifest + (manifest[1],),
        )


def test_shard_metadata_retains_full_fold_size_and_is_not_a_diagnostic_prefix(
    shard_config: alfred_runner.AlfredRunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alfred_runner, "external_source_provenance", lambda path: {})
    monkeypatch.setattr(
        alfred_runner,
        "model_source_provenance",
        lambda: {"groundingdino": {"version": "fixture"}},
    )
    config = replace(shard_config, shard_index=2, num_shards=4)
    selected = alfred_runner._selected_episodes(config, _manifest())
    metadata = alfred_runner._run_metadata(config, selected, {}, full_fold_size=50)
    assert metadata["full_fold_size"] == 50
    assert metadata["shard_index"] == 2
    assert metadata["num_shards"] == 4
    assert metadata["diagnostic_prefix"] is False
    assert len(cast(list[object], metadata["episodes"])) == 12


def test_cli_exposes_explicit_shard_configuration() -> None:
    required = [
        argument
        for name in (
            "data-root",
            "splits",
            "fold",
            "output-dir",
            "simulator-python",
            "alfred-root",
            "scene-config",
        )
        for argument in (f"--{name}", "/fixture")
    ]
    parser = build_arg_parser()
    defaults = parser.parse_args(required)
    assert (defaults.shard_index, defaults.num_shards) == (0, 1)
    explicit = parser.parse_args([*required, "--shard-index", "3", "--num-shards", "4"])
    assert (explicit.shard_index, explicit.num_shards) == (3, 4)
