"""Validate the completed ALFRED documentation and build its local site."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path("/home/ysh/codebase/agentic-3d-task")
WORK = ROOT / "tmp/alfred_setup"
ARCHIVE = ROOT / "docs/benchmark/alfred"
PAGES = [
    ARCHIVE / name
    for name in (
        "README.md",
        "leaderboard.md",
        "v7_recovery_failures_20261105.md",
        "v8_pilot50_complete_20261105.md",
    )
] + [ROOT / "docs/benchmark/index.md", ROOT / "docs/codex_agent/alfred.md"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks: dict[str, object] = {}
    links: set[tuple[str, str]] = set()
    for page in PAGES:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", page.read_text()):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            resolved = (page.parent / unquote(url.path)).resolve()
            if not resolved.is_file():
                raise ValueError(f"Missing documentation target: {page.name}: {target}")
            links.add((str(page.relative_to(ROOT)), target))
    checks["relative_links_checked"] = len(links)
    checks["relative_links"] = sorted(links)
    asset_paths: list[Path] = []
    for folder in ("v7_recovery_failures_20261105", "v8_recovery_complete_20261105"):
        base = ARCHIVE / "assets" / folder
        hashes = json.loads((base / "sha256.json").read_text())
        for name, digest in hashes.items():
            if sha(base / name) != digest:
                raise ValueError(f"Archive checksum mismatch: {folder}/{name}")
        checks[folder + "_hashes_checked"] = len(hashes)
        asset_paths.extend(path for path in base.rglob("*") if path.is_file())
    completion = ARCHIVE / "assets/v8_completion_validation_20261105.json"
    asset_paths.append(completion)
    checks["completion_validation_sha256"] = sha(completion)
    manifest = json.loads((WORK / "pilot50_cohort_recovery3.json").read_text())
    errors = [
        attempt["error"]
        for record in manifest["original_runs"]
        for attempt in json.loads(Path(record["summary_json"]).read_text())["attempts"]
        if attempt["status"] == "error"
    ]
    secret = re.compile(
        r"Bearer\s+\S+|sk-[A-Za-z0-9]{12,}|"
        r"-----BEGIN (?:(?:RSA|DSA|EC|OPENSSH|ENCRYPTED) )?PRIVATE KEY-----|"
        r"[?&](?:ak|api[_-]?key|access[_-]?token)=[^&\s]+",
        re.IGNORECASE,
    )
    for path in asset_paths + PAGES:
        raw = path.read_bytes()
        text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode()
        if secret.search(text) or any(error in text for error in errors):
            raise ValueError(f"Privacy check failed for {path.relative_to(ROOT)}")
    checks["privacy_scanned_files"] = len(asset_paths) + len(PAGES)
    checks["raw_historical_error_strings_excluded"] = len(errors)
    for name, command in {
        "existing_executable_files_unchanged_from_base": [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=MDRT",
            "debd961b7d6d848518d415635afd55b85dc1f41b",
            "--",
            "src",
            "scripts",
            "pyproject.toml",
        ],
        "working_executable_tree_unchanged": [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "src",
            "scripts",
            "configs",
            "pyproject.toml",
        ],
    }.items():
        result = subprocess.check_output(command, cwd=ROOT, text=True)
        if result.strip():
            raise ValueError(f"Unexpected executable changes: {name}")
        checks[name] = True
    tracked = subprocess.check_output(
        ["git", "ls-files", "docs/benchmark/alfred"], cwd=ROOT, text=True
    ).splitlines()
    history = [path for path in tracked if re.search(r"/(?:assets/)?v[1-6]_", path)]
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "7010ba8f54088bdad50ad30ba8ac709765c0d1c6",
            "--",
            *history,
        ],
        cwd=ROOT,
        check=True,
    )
    checks["unchanged_v1_to_v6_tracked_files"] = len(history)
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    command = ["mkdocs", "build", "--site-dir", "tmp/alfred_setup/site_v8"]
    log = WORK / "mkdocs_v8.txt"
    with log.open("x") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    checks["mkdocs"] = {
        "command": command,
        "exit_code": result.returncode,
        "log_sha256": sha(log),
    }
    checks["passed"] = result.returncode == 0
    checks["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    checks["scope"] = (
        "Documentation publication validation only; no model, simulator, or benchmark evaluation; existing executable checks reused from v6."
    )
    output = WORK / "v8_publication_validation.json"
    with output.open("x") as stream:
        json.dump(checks, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {key: value for key, value in checks.items() if key != "relative_links"},
            indent=2,
        )
    )
    if not checks["passed"]:
        raise RuntimeError("Documentation build failed")


if __name__ == "__main__":
    main()
