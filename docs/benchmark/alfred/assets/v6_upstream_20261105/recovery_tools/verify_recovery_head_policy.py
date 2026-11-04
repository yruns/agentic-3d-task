"""Exercise recovery identity comparison using copied metadata, without any run."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from analyze_pilot50 import ROOT, read_json
from analyze_recovered_pilot50 import sha256, validate_run_configuration


def main() -> None:
    manifest_path = ROOT / "tmp/alfred_setup/pilot50_cohort.json"
    manifest = read_json(manifest_path)
    original = read_json(Path(manifest["original_runs"][0]["run_json"]))
    head_only = copy.deepcopy(original)
    head_only["code"]["head_sha"] = "a" * 40
    accepted = validate_run_configuration(head_only, manifest)
    assert accepted["head_differs"] is True
    assert accepted["actual_head_sha"] == "a" * 40
    assert accepted["reference_head_sha"] == original["code"]["head_sha"]
    rejected: list[str] = []
    for field, value, message in (
        ("executed_code_sha256", "0" * 64, "executed source/config SHA changed"),
        ("branch", "different-branch", "source branch changed"),
        ("tracked_worktree_drift", " M src/example.py", "source tree is not clean"),
        ("head_sha", "short-sha", "actual HEAD must be a recorded full Git SHA"),
    ):
        changed = copy.deepcopy(head_only)
        changed["code"][field] = value
        try:
            validate_run_configuration(changed, manifest)
        except ValueError as exc:
            assert message in str(exc), str(exc)
            rejected.append(field)
        else:
            raise AssertionError(f"forbidden {field} difference was accepted")
    for field in ("model", "execution_budget", "input_path", "turn_timeout"):
        changed = copy.deepcopy(head_only)
        if field == "model":
            changed["model"] = "another-model"
        elif field == "execution_budget":
            changed["execution"]["max_decisions"] += 1
        else:
            option = "--data-root" if field == "input_path" else "--turn-timeout"
            changed["command"][changed["command"].index(option) + 1] = (
                "/different-input" if field == "input_path" else "301"
            )
        try:
            validate_run_configuration(changed, manifest)
        except ValueError as exc:
            assert "actor/model/input/budget configuration changed" in str(exc)
            rejected.append(field)
        else:
            raise AssertionError(f"forbidden {field} difference was accepted")
    report = {
        "scope": "Metadata-only boundary checks; no models, simulator, evaluation or artifact rewriting.",
        "cohort_sha256": sha256(manifest_path),
        "script_sha256": sha256(Path(__file__)),
        "comparator_script_sha256": sha256(
            Path(__file__).with_name("analyze_recovered_pilot50.py")
        ),
        "accepted_head_only_difference": accepted,
        "rejected_changed_fields": rejected,
        "checks_passed": len(rejected) + 1,
    }
    output = ROOT / "tmp/alfred_setup/recovery_head_policy_checks.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
