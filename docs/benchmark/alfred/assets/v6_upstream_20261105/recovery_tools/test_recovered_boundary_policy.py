"""Local provenance comparator regressions; no evaluation/model/simulator calls."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from audit_pilot_boundary import BoundaryAudit, JsonObject, read_json, sha256
from audit_recovered_pilot_boundary import validate_run, validate_source_identity


class SourceIdentityPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = read_json(Path(__file__).with_name("pilot50_cohort.json"))
        self.descriptor = self.manifest["original_runs"][0]
        self.metadata = read_json(Path(self.descriptor["run_json"]))
        self.summary = read_json(Path(self.descriptor["summary_json"]))
        self.temporary = tempfile.TemporaryDirectory(prefix="alfred-audit-policy-")
        self.root = Path(self.temporary.name)
        self.audit = BoundaryAudit(self.root / "unused-sdk-runs")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _source_codes(self, **changes: object) -> set[str]:
        metadata = copy.deepcopy(self.metadata)
        metadata["code"].update(changes)
        validate_source_identity(metadata, self.manifest, self.audit, self.root)
        return {violation["code"] for violation in self.audit.violations}

    def _write_run(self, *, recovery: bool) -> tuple[JsonObject, JsonObject]:
        metadata = copy.deepcopy(self.metadata)
        summary = copy.deepcopy(self.summary)
        if recovery:
            metadata["status"] = "completed"
            metadata["episodes"] = metadata["episodes"][:1]
            metadata["code"]["head_sha"] = "f" * 40
            summary.update(selected=1, completed=1, errors=0, interrupted=0, pending=0)
            summary["attempts"] = summary["attempts"][:1]
            summary["metrics"]["num_evals"] = 1
        (self.root / "run.json").write_text(json.dumps(metadata))
        (self.root / "summary.json").write_text(json.dumps(summary))
        descriptor = {
            "run_id": metadata["run_id"],
            "output_dir": str(self.root),
            "status": metadata["status"],
            "run_json_sha256": sha256(self.root / "run.json"),
            "summary_json_sha256": sha256(self.root / "summary.json"),
        }
        return descriptor, metadata

    def test_head_only_difference_is_accepted(self) -> None:
        self.assertEqual(self._source_codes(head_sha="f" * 40), set())

    def test_executed_code_difference_is_rejected(self) -> None:
        self.assertIn(
            "executed_code", self._source_codes(executed_code_sha256="f" * 64)
        )

    def test_branch_difference_is_rejected(self) -> None:
        self.assertIn("branch_provenance", self._source_codes(branch="wrong-branch"))

    def test_dirty_tree_is_rejected(self) -> None:
        self.assertIn(
            "worktree_provenance",
            self._source_codes(tracked_worktree_drift=" M src/example.py"),
        )

    def test_actual_head_must_remain_recorded(self) -> None:
        self.assertIn("actual_head", self._source_codes(head_sha="unknown"))

    def test_recovery_with_head_only_change_retains_full_configuration(self) -> None:
        descriptor, metadata = self._write_run(recovery=True)
        _, actual, _ = validate_run(
            descriptor, self.manifest, self.audit, recovery=True
        )
        self.assertEqual(self.audit.violations, [])
        self.assertEqual(actual["code"]["head_sha"], metadata["code"]["head_sha"])

    def test_changed_turn_timeout_is_rejected(self) -> None:
        descriptor, metadata = self._write_run(recovery=True)
        metadata["command"].extend(["--turn-timeout", "301"])
        (self.root / "run.json").write_text(json.dumps(metadata))
        validate_run(descriptor, self.manifest, self.audit, recovery=True)
        self.assertIn(
            "matched_cli_configuration",
            {item["code"] for item in self.audit.violations},
        )

    def test_changed_execution_budget_is_rejected(self) -> None:
        descriptor, metadata = self._write_run(recovery=True)
        metadata["execution"]["max_decisions"] += 1
        (self.root / "run.json").write_text(json.dumps(metadata))
        validate_run(descriptor, self.manifest, self.audit, recovery=True)
        self.assertIn(
            "matched_configuration", {item["code"] for item in self.audit.violations}
        )

    def test_original_head_cannot_be_rewritten_despite_head_policy(self) -> None:
        descriptor, metadata = self._write_run(recovery=False)
        metadata["code"]["head_sha"] = "f" * 40
        (self.root / "run.json").write_text(json.dumps(metadata))
        validate_run(descriptor, self.manifest, self.audit, recovery=False)
        self.assertIn(
            "original_artifact_hashes", {item["code"] for item in self.audit.violations}
        )


if __name__ == "__main__":
    unittest.main()
