"""Fail-closed code provenance for ALFRED evaluation invocations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from importlib import metadata as distribution_metadata
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PERCEPTION_PACKAGES = (
    "groundingdino",
    "segment_anything",
    "depth-anything-3",
    "open_clip_torch",
    "transformers",
    "torch",
    "torchvision",
    "numpy",
)
GEOMETRY_PACKAGES = ("numpy", "scipy", "scikit-fmm")


class ExternalSourceProvenance(BaseModel):
    """Exact local source content for an editable external Git checkout."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    path: str
    head_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    tracked_diff_sha256: Sha256
    tracked_dirty: bool
    python_source_sha256: Sha256
    python_source_count: int = Field(ge=0)


class PackageProvenance(BaseModel):
    """Installed package version and its wheel record or actual editable source."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: str = Field(min_length=1)
    distribution_record_sha256: Sha256 | None = None
    source: ExternalSourceProvenance | None = None
    native_extensions: dict[str, Sha256] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_fingerprint(self) -> PackageProvenance:
        if self.distribution_record_sha256 is None and self.source is None:
            raise ValueError(
                "package provenance requires a distribution record or source digest"
            )
        return self


class RuntimeProvenance(BaseModel):
    """A process reports its own interpreter, project modules and package records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    executable: str
    prefix: str
    package_root: str
    package_source_sha256: Sha256
    packages: dict[str, PackageProvenance]
    visible_devices: str | None = None

    @model_validator(mode="after")
    def validate_process_identity(self) -> RuntimeProvenance:
        for name in ("executable", "prefix", "package_root"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"runtime provenance {name} must be an absolute path")
        if not self.packages:
            raise ValueError("runtime provenance must contain package fingerprints")
        return self


def file_sha256(path: Path) -> str:
    """Hash large checkpoints without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_source_provenance(path: Path) -> dict[str, object]:
    """Hash actual Python sources, including untracked imports, plus Git revision."""
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--"])
    source_paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ]
    ).split(b"\0")
    digest = hashlib.sha256()
    paths = sorted({relative for relative in source_paths if relative})
    for relative in paths:
        digest.update(relative + b"\0")
        digest.update((path / relative.decode("utf-8")).read_bytes())
        digest.update(b"\0")
    return {
        "path": str(path),
        "head_sha": head,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_dirty": bool(diff),
        "python_source_sha256": digest.hexdigest(),
        "python_source_count": len(paths),
    }


def package_source_provenance(
    package_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Record installed inference libraries and actual editable source contents."""
    result: dict[str, dict[str, object]] = {}
    for name in package_names:
        try:
            distribution = distribution_metadata.distribution(name)
        except distribution_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required perception package is not installed: {name}"
            ) from exc
        record: dict[str, object] = {"version": distribution.version}
        installed_record = distribution.read_text("RECORD")
        if installed_record:
            record["distribution_record_sha256"] = hashlib.sha256(
                installed_record.encode()
            ).hexdigest()
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        url = direct.get("url")
        if isinstance(url, str) and urlsplit(url).scheme == "file":
            source = Path(unquote(urlsplit(url).path)).resolve(strict=True)
            if (source / ".git").exists():
                record["source"] = external_source_provenance(source)
                record["native_extensions"] = {
                    str(path.relative_to(source)): file_sha256(path)
                    for path in sorted(source.rglob("_C*.so"))
                    if "build" not in path.relative_to(source).parts
                }
        result[name] = record
    return result


def model_source_provenance() -> dict[str, dict[str, object]]:
    """Capture model packages from the process executing this function."""
    return package_source_provenance(PERCEPTION_PACKAGES)


def alfred_source_sha256() -> str:
    """Identify actual online-module bytes imported by a worker or main process."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def capture_runtime_provenance(*, model_worker: bool) -> RuntimeProvenance:
    """Capture only the calling process; never infer another interpreter's state."""
    packages = (
        model_source_provenance()
        if model_worker
        else package_source_provenance(GEOMETRY_PACKAGES)
    )
    return RuntimeProvenance(
        executable=str(Path(sys.executable).absolute()),
        prefix=str(Path(sys.prefix).absolute()),
        package_root=str(Path(__file__).resolve().parent),
        package_source_sha256=alfred_source_sha256(),
        packages={
            name: PackageProvenance.model_validate(record)
            for name, record in packages.items()
        },
        visible_devices=(
            os.environ.get("CUDA_VISIBLE_DEVICES", "") if model_worker else None
        ),
    )


def code_provenance(project_root: Path) -> dict[str, object]:
    """Require a clean tracked tree and committed executable source/config.

    An untracked source file is executable drift even when ``git diff`` is empty.
    Evaluation cannot be enabled by merely leaving new implementation untracked.
    """

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments], text=True
        ).strip()

    tracked = git("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise ValueError(
            "evaluation requires a clean tracked working tree; commit the reviewed change first"
        )
    untracked = git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "src",
        "scripts",
        "configs",
        "pyproject.toml",
    )
    if untracked:
        raise ValueError(
            "evaluation source/config contains untracked files; commit the reviewed change first"
        )
    paths = git("ls-files", "src", "scripts", "configs", "pyproject.toml").splitlines()
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode() + b"\0")
        digest.update((project_root / relative).read_bytes())
        digest.update(b"\0")
    return {
        "branch": git("branch", "--show-current"),
        "head_sha": git("rev-parse", "HEAD"),
        "executed_code_sha256": digest.hexdigest(),
        "tracked_worktree_drift": tracked,
    }
