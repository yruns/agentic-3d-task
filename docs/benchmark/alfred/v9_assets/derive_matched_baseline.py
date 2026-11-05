"""Reaggregate saved v8 scores for frozen pilot15 without rerunning a task."""

import json
from pathlib import Path

from codex_agent.alfred.metrics import EpisodeScore, aggregate_scores
from codex_agent.alfred.provenance import file_sha256


def main() -> None:
    project = Path(__file__).resolve().parents[4]
    fold_path = project / "docs/benchmark/alfred/folds/pilot15.json"
    source_path = (
        project
        / "docs/benchmark/alfred/assets/v8_recovery_complete_20261105/audits/pilot50_analysis.json"
    )
    fold = json.loads(fold_path.read_text())
    saved = json.loads(source_path.read_text())
    by_id = {row["episode_id"]: row for row in saved["episodes"]}
    identities = [
        f"{row['split']}/{row['task']}#{row['repeat_idx']}" for row in fold["samples"]
    ]
    assert len(set(identities)) == 15
    selected = [by_id[identity] for identity in identities]
    metrics = {
        split: aggregate_scores(
            [
                EpisodeScore.model_validate(row["score"])
                for row in selected
                if split == "combined" or row["split"] == split
            ]
        ).model_dump()
        for split in ("combined", "valid_seen", "valid_unseen")
    }
    record = {
        "scope": "Saved v8 results restricted to frozen pilot15; no new actor, perception, or simulator execution",
        "fold_sha256": file_sha256(fold_path),
        "source_path": str(source_path.relative_to(project)),
        "source_sha256": file_sha256(source_path),
        "script_sha256": file_sha256(Path(__file__)),
        "aggregation_source_sha256": file_sha256(
            project / "src/codex_agent/alfred/metrics.py"
        ),
        "metrics": metrics,
        "episodes": [
            {
                key: row[key]
                for key in (
                    "episode_id",
                    "split",
                    "task_type",
                    "run_id",
                    "score",
                    "execution",
                )
            }
            for row in selected
        ],
    }
    output = Path(__file__).with_name("matched_v8_pilot15.json")
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
