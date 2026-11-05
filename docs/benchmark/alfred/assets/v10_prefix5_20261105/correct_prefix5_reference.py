"""Correct explanatory comparability fields without repeating verified request IO."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import audit_pilot15 as audit

from codex_agent.alfred.dataset import load_manifest
from codex_agent.config import DEFAULT_PROJECT_ROOT

ROOT = DEFAULT_PROJECT_ROOT
WORK = ROOT / "tmp/alfred_v9"


def main() -> None:
    source = WORK / "pilot5_final_audit_01.json"
    output = WORK / "pilot5_final_audit_02.json"
    audit.require(not output.exists(), "Corrected audit output already exists")
    report = audit.read_json(source)
    audit.require(
        report["status"] == "complete_verified_user_prefix5",
        "Original audit is not complete",
    )
    references = load_manifest(
        ROOT / "docs/benchmark/alfred/folds/pilot15.json",
        Path("/home/ysh/Datasets/FILM/alfred/data/json_2.1.0"),
        Path("/home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json"),
    )[:5]
    reference = audit.historical_v8_reference(references, report["configuration"])
    audit.require(
        reference["metrics"]
        == report["historical_v8_same_identity_reference"]["metrics"],
        "Correction unexpectedly changed historical metrics",
    )
    audit.require(
        reference["foundation_model_matches_current"]
        and reference["reasoning_effort_matches_current"],
        "Per-run comparison does not confirm equal model/effort",
    )
    report["historical_v8_same_identity_reference"] = reference
    report["explanatory_correction"] = {
        "corrected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_final_audit_path": str(source),
        "source_final_audit_sha256": audit.sha256(source),
        "changed_fields": ["historical_v8_same_identity_reference"],
        "reason": "The original comparison note incorrectly asserted different reasoning configuration; every matched v8 run actually uses the same gpt-5.4-2026-03-05 model and medium effort. Numeric metrics and all task/actor/SQLite/official-AST audit results are unchanged.",
        "core_audit_helper_sha256": report["script_sha256"],
        "core_audit_helper_snapshot": str(
            WORK / "helper_versions/final01/audit_pilot15.py"
        ),
        "reference_check_helper_sha256": audit.sha256(Path(audit.__file__)),
        "correction_script_sha256": audit.sha256(Path(__file__)),
        "request_artifacts_rescanned": False,
    }
    audit.require(
        audit.sha256(
            Path(report["explanatory_correction"]["core_audit_helper_snapshot"])
        )
        == report["script_sha256"],
        "Original executed audit helper snapshot changed",
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "same_model": True,
                "same_effort": True,
                "matched_source_runs": reference["matched_count"],
                "historical_metrics_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
