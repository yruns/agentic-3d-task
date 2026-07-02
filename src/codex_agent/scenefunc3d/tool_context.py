"""SceneFunc3D tool context file boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    ValidationError,
    field_validator,
)

from codex_agent.errors import SceneFunc3dDataError


@dataclass(frozen=True)
class SceneFunc3dToolContext:
    """Runtime paths shared by one SceneFunc3D tool turn."""

    sample_id: str
    scene_root: Path
    backend_config_path: Path
    out_dir: Path


class _SceneFunc3dToolContextDocument(BaseModel):
    """Pydantic boundary model for untrusted tool context JSON."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    scene_root: DirectoryPath
    backend_config_path: FilePath
    out_dir: Path

    @field_validator("scene_root", "backend_config_path", "out_dir", mode="before")
    @classmethod
    def _expand_and_resolve_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            raw_path = value
        elif isinstance(value, str):
            raw_path = Path(value)
        else:
            raise ValueError("path value must be a string or Path")
        return raw_path.expanduser().resolve()

    @field_validator("out_dir")
    @classmethod
    def _require_absolute_out_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(
                "out_dir must be absolute after expanduser() and resolve()"
            )
        return value

    def to_context(self) -> SceneFunc3dToolContext:
        """Convert validated JSON data into the immutable runtime context."""
        return SceneFunc3dToolContext(
            sample_id=self.sample_id,
            scene_root=self.scene_root,
            backend_config_path=self.backend_config_path,
            out_dir=self.out_dir,
        )


def write_tool_context(path: Path, context: SceneFunc3dToolContext) -> Path:
    """Write a validated SceneFunc3D tool context JSON file."""
    normalized_path = Path(path).expanduser().resolve()
    try:
        document = _SceneFunc3dToolContextDocument.model_validate(
            {
                "sample_id": context.sample_id,
                "scene_root": context.scene_root,
                "backend_config_path": context.backend_config_path,
                "out_dir": context.out_dir,
            }
        )
    except ValidationError as exc:
        raise _context_error(normalized_path, "validation", exc) from exc

    try:
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise _context_error(normalized_path, "write", exc) from exc
    return normalized_path


def load_tool_context(path: Path) -> SceneFunc3dToolContext:
    """Load and validate a SceneFunc3D tool context JSON file."""
    normalized_path = Path(path).expanduser().resolve()
    try:
        raw_payload: object = json.loads(normalized_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise _context_error(normalized_path, "json", exc) from exc
    except OSError as exc:
        raise _context_error(normalized_path, "read", exc) from exc

    try:
        return _SceneFunc3dToolContextDocument.model_validate(raw_payload).to_context()
    except ValidationError as exc:
        raise _context_error(normalized_path, "validation", exc) from exc


def _context_error(
    path: Path, error_type: str, exc: BaseException
) -> SceneFunc3dDataError:
    if isinstance(exc, ValidationError):
        error_text = _format_validation_errors(exc)
    else:
        error_text = str(exc)
    return SceneFunc3dDataError(
        "SceneFunc3D tool context failed: "
        f"path={path}; error_type={error_type}; error={error_text}"
    )


def _format_validation_errors(exc: ValidationError) -> str:
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    parts: list[str] = []
    for error in errors:
        location = ".".join(str(item) for item in error.get("loc", ())) or "(root)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


__all__ = [
    "SceneFunc3dToolContext",
    "load_tool_context",
    "write_tool_context",
]
