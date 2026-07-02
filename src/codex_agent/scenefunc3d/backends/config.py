"""Configuration loading for SceneFunc3D sidecar backends."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from urllib.parse import ParseResult, urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

from codex_agent.scenefunc3d.tools.models import ToolInputError

LOCAL_BACKEND_HOSTS: frozenset[str] = frozenset(("127.0.0.1", "localhost"))
_HEADER_NAME_RE: Pattern[str] = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


@dataclass(frozen=True)
class HttpHeader:
    """One validated HTTP request header loaded from private config."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if not _HEADER_NAME_RE.fullmatch(self.name):
            raise ToolInputError(f"unsafe request header name: {self.name!r}")
        if not self.value:
            raise ToolInputError(f"unsafe request header value: header={self.name!r}")
        if "\r" in self.value or "\n" in self.value:
            raise ToolInputError(f"unsafe request header value: header={self.name!r}")


@dataclass(frozen=True)
class SceneFunc3dBackendSettings:
    """Validated settings for SceneFunc3D sidecar backend calls."""

    molmo_url: str
    sam_url: str
    request_timeout_seconds: float
    allowed_image_roots: tuple[Path, ...]
    allowed_output_roots: tuple[Path, ...]
    request_headers: tuple[HttpHeader, ...] = ()

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

    sidecar_base_url: str = ""
    molmo_url: str = ""
    sam_url: str = ""
    request_headers_path: Path | None = None
    request_timeout_seconds: float = Field(gt=0)
    allowed_image_roots: tuple[Path, ...] = Field(min_length=1)
    allowed_output_roots: tuple[Path, ...] = Field(min_length=1)

    def to_settings(self, *, config_dir: Path) -> SceneFunc3dBackendSettings:
        """Convert the validated document into the immutable runtime settings."""
        molmo_url, sam_url = _resolve_service_urls(
            sidecar_base_url=self.sidecar_base_url,
            molmo_url=self.molmo_url,
            sam_url=self.sam_url,
        )
        return SceneFunc3dBackendSettings(
            molmo_url=molmo_url,
            sam_url=sam_url,
            request_timeout_seconds=self.request_timeout_seconds,
            allowed_image_roots=self.allowed_image_roots,
            allowed_output_roots=self.allowed_output_roots,
            request_headers=_load_request_headers(
                self.request_headers_path,
                config_dir=config_dir,
            ),
        )


class _RequestHeadersDocument(BaseModel):
    """Pydantic boundary model for private sidecar header TOML content."""

    model_config = ConfigDict(extra="forbid")

    headers: dict[str, str] = Field(default_factory=dict)


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

    return document.to_settings(config_dir=path.parent)


def ensure_path_under_roots(
    path: Path,
    *,
    roots: tuple[Path, ...],
    field_name: str,
) -> Path:
    """Return a normalized path only when it is inside an allowed root."""
    normalized_path = _normalize_path(path, field_name=field_name)
    for root in roots:
        if normalized_path == root or root in normalized_path.parents:
            return normalized_path
    raise ToolInputError(
        f"{field_name} is outside configured roots: "
        f"path={normalized_path}; roots={[str(root) for root in roots]}"
    )


def _validate_backend_url(url: str, *, field_name: str) -> None:
    url_summary = _summarize_url(url)
    parsed_url = urlparse(url)
    if parsed_url.scheme == "http":
        _validate_local_backend_url(
            parsed_url,
            field_name=field_name,
            url_summary=url_summary,
        )
        return
    if parsed_url.scheme == "https":
        _validate_remote_backend_url(
            parsed_url,
            field_name=field_name,
            url_summary=url_summary,
        )
        return
    raise ToolInputError(
        f"{field_name} must use a local http://127.0.0.1:<port>[/path], "
        f"local http://localhost:<port>[/path], or remote https://<host>[/path]; "
        f"got {url_summary}"
    )


def _validate_local_backend_url(
    parsed_url: ParseResult,
    *,
    field_name: str,
    url_summary: str,
) -> None:
    if parsed_url.hostname not in LOCAL_BACKEND_HOSTS:
        raise ToolInputError(
            f"{field_name} must target 127.0.0.1 or localhost for http; "
            f"got {url_summary}"
        )
    _validate_url_has_no_unsafe_parts(
        parsed_url,
        field_name=field_name,
        url_summary=url_summary,
    )
    try:
        port = parsed_url.port
    except ValueError as exc:
        raise ToolInputError(
            f"{field_name} has an invalid explicit port: {url_summary}"
        ) from exc
    if port is None or port <= 0:
        raise ToolInputError(
            f"{field_name} must include an explicit port for local http; "
            f"got {url_summary}"
        )


def _validate_remote_backend_url(
    parsed_url: ParseResult,
    *,
    field_name: str,
    url_summary: str,
) -> None:
    if not parsed_url.hostname:
        raise ToolInputError(
            f"{field_name} must include a remote host; got {url_summary}"
        )
    _validate_url_has_no_unsafe_parts(
        parsed_url,
        field_name=field_name,
        url_summary=url_summary,
    )
    try:
        port = parsed_url.port
    except ValueError as exc:
        raise ToolInputError(
            f"{field_name} has an invalid explicit port: {url_summary}"
        ) from exc
    if port is not None and port <= 0:
        raise ToolInputError(
            f"{field_name} has an invalid explicit port: {url_summary}"
        )


def _validate_url_has_no_unsafe_parts(
    parsed_url: ParseResult,
    *,
    field_name: str,
    url_summary: str,
) -> None:
    if parsed_url.username or parsed_url.password:
        raise ToolInputError(
            f"{field_name} must not include credentials; got {url_summary}"
        )
    if parsed_url.params or parsed_url.query or parsed_url.fragment:
        raise ToolInputError(
            f"{field_name} must be a backend base URL without params, query, "
            f"or fragment; got {url_summary}"
        )


def _resolve_service_urls(
    *,
    sidecar_base_url: str,
    molmo_url: str,
    sam_url: str,
) -> tuple[str, str]:
    resolved_molmo_url = molmo_url.strip()
    resolved_sam_url = sam_url.strip()
    base_url = sidecar_base_url.strip()
    if base_url:
        _validate_backend_url(base_url, field_name="sidecar_base_url")
        if not resolved_molmo_url:
            resolved_molmo_url = _append_url_path(base_url, "molmo")
        if not resolved_sam_url:
            resolved_sam_url = _append_url_path(base_url, "sam")
    if not resolved_molmo_url:
        raise ToolInputError("backend config must set molmo_url or sidecar_base_url")
    if not resolved_sam_url:
        raise ToolInputError("backend config must set sam_url or sidecar_base_url")
    return resolved_molmo_url, resolved_sam_url


def _append_url_path(base_url: str, path_component: str) -> str:
    parsed_url = urlparse(base_url)
    base_path = parsed_url.path.rstrip("/")
    next_path = f"{base_path}/{path_component}" if base_path else f"/{path_component}"
    return urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            next_path,
            "",
            "",
            "",
        )
    )


def _load_request_headers(
    request_headers_path: Path | None,
    *,
    config_dir: Path,
) -> tuple[HttpHeader, ...]:
    if request_headers_path is None:
        return ()
    header_path = request_headers_path
    if not header_path.is_absolute():
        header_path = config_dir / header_path
    normalized_header_path = _normalize_path(
        header_path,
        field_name="request_headers_path",
    )
    if not normalized_header_path.is_file():
        raise ToolInputError(
            f"request headers config is missing: {normalized_header_path}"
        )

    try:
        with normalized_header_path.open("rb") as handle:
            raw_document: object = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ToolInputError(
            f"request headers config has invalid TOML: {normalized_header_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise ToolInputError(
            f"request headers config could not be read: {normalized_header_path}: {exc}"
        ) from exc

    try:
        document = _RequestHeadersDocument.model_validate(raw_document)
    except ValidationError as exc:
        raise ToolInputError(
            "request headers config validation failed for "
            f"{normalized_header_path}: {_format_validation_errors(exc)}"
        ) from exc
    return tuple(
        HttpHeader(name=name, value=value) for name, value in document.headers.items()
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


__all__ = [
    "HttpHeader",
    "SceneFunc3dBackendSettings",
    "ensure_path_under_roots",
    "load_backend_settings",
]
