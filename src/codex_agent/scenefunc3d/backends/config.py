"""Configuration loading for SceneFunc3D local sidecar backends."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

from codex_agent.scenefunc3d.tools.models import ToolInputError

LOCAL_BACKEND_HOSTS: frozenset[str] = frozenset(("127.0.0.1", "localhost"))


@dataclass(frozen=True)
class SceneFunc3dBackendSettings:
    """Validated settings for SceneFunc3D sidecar backend calls."""

    molmo_url: str
    sam_url: str
    request_timeout_seconds: float
    artifact_staging_root: Path
    allowed_image_roots: tuple[Path, ...]
    allowed_output_roots: tuple[Path, ...]

    def __post_init__(self) -> None:
        _validate_backend_url(self.molmo_url, field_name="molmo_url")
        _validate_backend_url(self.sam_url, field_name="sam_url")
        if self.request_timeout_seconds <= 0 or not math.isfinite(
            self.request_timeout_seconds
        ):
            raise ToolInputError(
                "request_timeout_seconds must be a finite positive number; "
                f"got {self.request_timeout_seconds}"
            )
        object.__setattr__(
            self,
            "artifact_staging_root",
            _normalize_path(
                self.artifact_staging_root, field_name="artifact_staging_root"
            ),
        )
        object.__setattr__(
            self,
            "allowed_image_roots",
            _normalize_roots(
                self.allowed_image_roots,
                field_name="allowed_image_roots",
            ),
        )
        object.__setattr__(
            self,
            "allowed_output_roots",
            _normalize_roots(
                self.allowed_output_roots,
                field_name="allowed_output_roots",
            ),
        )


class _BackendSettingsDocument(BaseModel):
    """Pydantic boundary model for untrusted backend TOML content."""

    model_config = ConfigDict(extra="forbid")

    molmo_url: str
    sam_url: str
    request_timeout_seconds: float = Field(gt=0)
    artifact_staging_root: Path
    allowed_image_roots: tuple[Path, ...] = Field(min_length=1)
    allowed_output_roots: tuple[Path, ...] = Field(min_length=1)

    def to_settings(self) -> SceneFunc3dBackendSettings:
        """Convert the validated document into the immutable runtime settings."""
        return SceneFunc3dBackendSettings(
            molmo_url=self.molmo_url,
            sam_url=self.sam_url,
            request_timeout_seconds=self.request_timeout_seconds,
            artifact_staging_root=self.artifact_staging_root,
            allowed_image_roots=self.allowed_image_roots,
            allowed_output_roots=self.allowed_output_roots,
        )


def load_backend_settings(path: Path) -> SceneFunc3dBackendSettings:
    """Load SceneFunc3D backend settings from a TOML file.

    Raises:
        ToolInputError: If the file is missing, unreadable, invalid TOML, or
            fails config validation.
    """
    if not path.is_file():
        raise ToolInputError(f"backend config is missing: {path}")

    try:
        with path.open("rb") as handle:
            raw_document: object = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ToolInputError(f"backend config has invalid TOML: {path}: {exc}") from exc
    except OSError as exc:
        raise ToolInputError(
            f"backend config could not be read: {path}: {exc}"
        ) from exc

    try:
        document = _BackendSettingsDocument.model_validate(raw_document)
    except ValidationError as exc:
        raise ToolInputError(
            "backend config validation failed for "
            f"{path}: {_format_validation_errors(exc)}"
        ) from exc

    return document.to_settings()


def _validate_backend_url(url: str, *, field_name: str) -> None:
    url_summary = _summarize_url(url)
    parsed_url = urlparse(url)
    if parsed_url.scheme != "http":
        raise ToolInputError(
            f"{field_name} must use http://127.0.0.1:<port> or "
            f"http://localhost:<port>; got {url_summary}"
        )
    if parsed_url.hostname not in LOCAL_BACKEND_HOSTS:
        raise ToolInputError(
            f"{field_name} must target 127.0.0.1 or localhost; got {url_summary}"
        )
    if parsed_url.username or parsed_url.password:
        raise ToolInputError(
            f"{field_name} must not include credentials; got {url_summary}"
        )
    if parsed_url.params or parsed_url.query or parsed_url.fragment:
        raise ToolInputError(
            f"{field_name} must be a local backend base URL without params, "
            f"query, or fragment; got {url_summary}"
        )
    if parsed_url.path not in ("", "/"):
        raise ToolInputError(
            f"{field_name} must be a local backend base URL without a path; "
            f"got {url_summary}"
        )
    try:
        port = parsed_url.port
    except ValueError as exc:
        raise ToolInputError(
            f"{field_name} has an invalid explicit port: {url_summary}"
        ) from exc
    if port is None or port <= 0:
        raise ToolInputError(
            f"{field_name} must include an explicit port; expected "
            f"http://127.0.0.1:<port> or http://localhost:<port>; got {url_summary}"
        )


def _normalize_path(path: Path, *, field_name: str) -> Path:
    if not isinstance(path, Path):
        raise ToolInputError(f"{field_name} must be a filesystem path; got {path!r}")
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ToolInputError(
            f"{field_name} could not be resolved: {exc.__class__.__name__}"
        ) from exc


def _normalize_roots(roots: Sequence[Path], *, field_name: str) -> tuple[Path, ...]:
    if not roots:
        raise ToolInputError(f"{field_name} must contain at least one path")
    return tuple(
        _normalize_path(root, field_name=f"{field_name}[{index}]")
        for index, root in enumerate(roots)
    )


def _format_validation_errors(exc: ValidationError) -> str:
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    parts: list[str] = []
    for error in errors:
        location = ".".join(str(item) for item in error.get("loc", ())) or "(root)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


def _summarize_url(url: str) -> str:
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme or "<missing-scheme>"
    host = parsed_url.hostname or "<missing-host>"
    credentials = "<credentials>@" if parsed_url.username or parsed_url.password else ""
    try:
        port = parsed_url.port
    except ValueError:
        port_text = ":<invalid-port>"
    else:
        port_text = f":{port}" if port is not None else ""
    suffix = ""
    if parsed_url.path not in ("", "/") or parsed_url.params or parsed_url.query:
        suffix = "/..."
    if parsed_url.fragment:
        suffix = "/..."
    return f"{scheme}://{credentials}{host}{port_text}{suffix}"
