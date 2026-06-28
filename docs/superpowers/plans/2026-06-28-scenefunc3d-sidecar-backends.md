# SceneFunc3D Sidecar Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect SceneFunc3D `molmo_point`, `sam_mask`, and `lift_mask_to_3d` tools to local sidecar backends and a deterministic 3D lifting implementation so one fixed SceneFuncVal-CG case can run single-view end to end.

**Architecture:** Keep MolmoPoint and SAM2.1 as local `127.0.0.1` sidecar HTTP JSON services under the current repository. The agent tool layer remains responsible for validation, durable artifacts, contact sheets, approval state, and recoverable errors; 3D lifting stays as local deterministic geometry code.

**Tech Stack:** Python 3.10+, Pydantic v2, standard-library `http.server.ThreadingHTTPServer`, `urllib.request`, dataclasses, pytest, optional Pillow/numpy/scipy/torch/transformers/sam2 imports behind lazy boundaries.

---

## Pre-Execution Notes

- Work in the isolated branch/worktree `feat/scenefunc3d-agent-tools`.
- Read `AGENTS.md`, `docs/python_code_agent_quality_guide.md`, and `docs/superpowers/specs/2026-06-28-scenefunc3d-sidecar-backends-design.md` before editing.
- Do not commit model weights, private credentials, or local `configs/scenefunc3d_backends.toml`.
- Keep default tests GPU-free. Heavy Molmo/SAM tests must be explicit scripts or marked tests.
- Keep `import codex_agent` and `import codex_agent.scenefunc3d` working without torch, transformers, sam2, Pillow, or OpenCV.
- Use `ToolInputError` for agent-recoverable tool failures. Do not return fake success for unavailable backends, crop rendering, object index, SAM output, or lifting output.

## File Structure

Create backend and server modules:

- `src/codex_agent/scenefunc3d/backends/__init__.py`: package marker and public exports.
- `src/codex_agent/scenefunc3d/backends/config.py`: TOML loader, URL/path validation, backend settings.
- `src/codex_agent/scenefunc3d/backends/http_client.py`: standard-library JSON HTTP client with timeout and schema validation.
- `src/codex_agent/scenefunc3d/backends/molmo_rpc.py`: Molmo RPC client plus artifact conversion helper.
- `src/codex_agent/scenefunc3d/backends/sam_rpc.py`: SAM RPC client plus mask candidate loader helper.
- `src/codex_agent/scenefunc3d/backends/frame_assets.py`: resolve RGB/depth/intrinsics/pose for a frame.
- `src/codex_agent/scenefunc3d/backends/lift_3d.py`: mask backprojection, scene point mapping, NPZ/PLY writing.
- `src/codex_agent/scenefunc3d/servers/__init__.py`: package marker.
- `src/codex_agent/scenefunc3d/servers/schemas.py`: Pydantic request/response models shared by servers and clients.
- `src/codex_agent/scenefunc3d/servers/http_json.py`: reusable local JSON server handler helpers.
- `src/codex_agent/scenefunc3d/servers/molmo_point_server.py`: Molmo sidecar CLI and model runner.
- `src/codex_agent/scenefunc3d/servers/sam2_mask_server.py`: SAM2.1 sidecar CLI and model runner.

Modify existing tools:

- `src/codex_agent/scenefunc3d/tools/__main__.py`: add `--backend-config`.
- `src/codex_agent/scenefunc3d/tools/dispatch.py`: pass backend settings into heavy tools.
- `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`: call Molmo RPC, write raw text and point overlay.
- `src/codex_agent/scenefunc3d/tools/sam_masking.py`: call SAM RPC, write candidate overlays/contact sheet.
- `src/codex_agent/scenefunc3d/tools/mask_lifting.py`: call local lifting implementation.
- `src/codex_agent/scenefunc3d/tools/frame_views.py`: reuse frame asset resolver where it avoids duplicate path logic.

Create scripts and config examples:

- `configs/scenefunc3d_backends.example.toml`: safe local endpoint example.
- `scripts/scenefunc3d/serve_molmo_point.sh`: tmux-friendly Molmo server launcher.
- `scripts/scenefunc3d/serve_sam2.sh`: tmux-friendly SAM2.1 server launcher.
- `scripts/scenefunc3d/check_sidecars.sh`: health check both sidecars.
- `scripts/scenefunc3d/run_single_case_e2e.sh`: fixed-case smoke wrapper.

Add tests:

- `src/codex_agent/tests/test_scenefunc3d_backend_config.py`
- `src/codex_agent/tests/test_scenefunc3d_http_client.py`
- `src/codex_agent/tests/test_scenefunc3d_server_schemas.py`
- `src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py`
- `src/codex_agent/tests/test_scenefunc3d_sam_rpc.py`
- `src/codex_agent/tests/test_scenefunc3d_lift3d.py`
- Extend existing `test_scenefunc3d_tools_cli.py`, `test_scenefunc3d_molmo_sam_contracts.py`, and `test_scenefunc3d_metrics.py`.

## Task 1: Backend Config And Shared Server Schemas

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/__init__.py`
- Create: `src/codex_agent/scenefunc3d/backends/config.py`
- Create: `src/codex_agent/scenefunc3d/servers/__init__.py`
- Create: `src/codex_agent/scenefunc3d/servers/schemas.py`
- Create: `configs/scenefunc3d_backends.example.toml`
- Test: `src/codex_agent/tests/test_scenefunc3d_backend_config.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_server_schemas.py`

- [ ] **Step 1: Write failing config tests**

Create `src/codex_agent/tests/test_scenefunc3d_backend_config.py`:

```python
"""Tests for SceneFunc3D sidecar backend configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_agent.scenefunc3d.backends.config import (
    SceneFunc3dBackendSettings,
    load_backend_settings,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError


def test_load_backend_settings_from_toml(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    allowed_root.mkdir()
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
molmo_url = "http://127.0.0.1:8711"
sam_url = "http://127.0.0.1:8712"
request_timeout_seconds = 3.5
artifact_staging_root = "{allowed_root}"
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{allowed_root}"]
""",
        encoding="utf-8",
    )

    settings = load_backend_settings(config_path)

    assert settings.molmo_url == "http://127.0.0.1:8711"
    assert settings.sam_url == "http://127.0.0.1:8712"
    assert settings.request_timeout_seconds == 3.5
    assert settings.artifact_staging_root == allowed_root
    assert settings.allowed_image_roots == (tmp_path,)
    assert settings.allowed_output_roots == (allowed_root,)


def test_load_backend_settings_missing_file_is_recoverable(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="backend config is missing"):
        load_backend_settings(tmp_path / "missing.toml")


def test_backend_settings_rejects_non_local_url(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="127.0.0.1"):
        SceneFunc3dBackendSettings(
            molmo_url="http://10.1.2.3:8711",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=3.0,
            artifact_staging_root=tmp_path,
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )


def test_backend_settings_rejects_non_positive_timeout(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="request_timeout_seconds"):
        SceneFunc3dBackendSettings(
            molmo_url="http://127.0.0.1:8711",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=0.0,
            artifact_staging_root=tmp_path,
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )
```

- [ ] **Step 2: Write failing schema tests**

Create `src/codex_agent/tests/test_scenefunc3d_server_schemas.py`:

```python
"""Tests for shared SceneFunc3D sidecar HTTP schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_agent.scenefunc3d.servers.schemas import (
    HealthResponse,
    MolmoPointRequest,
    MolmoPointResponse,
    SamMaskCandidateResponse,
    SamMaskRequest,
    SamMaskResponse,
    SamPointPrompt,
)


def test_molmo_point_schema_round_trip(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    request = MolmoPointRequest(
        request_id="req-1",
        image_path=image_path,
        prompt="point to the handle",
        image_width=640,
        image_height=480,
    )
    response = MolmoPointResponse(
        request_id=request.request_id,
        model_name="MolmoPoint-8B",
        raw_text='<point x="50" y="50">handle</point>',
        latency_ms=12,
    )

    assert request.model_dump(mode="json")["image_path"] == str(image_path)
    assert response.model_dump(mode="json")["raw_text"].startswith("<point")


def test_sam_mask_schema_round_trip(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    mask_path = staging_dir / "mask_00.npz"
    mask_path.write_bytes(b"mask")
    request = SamMaskRequest(
        request_id="req-2",
        image_path=image_path,
        points=(
            SamPointPrompt(
                x_px=10.0,
                y_px=20.0,
                label="handle",
                source='<point x="1" y="2">handle</point>',
            ),
        ),
        staging_dir=staging_dir,
    )
    response = SamMaskResponse(
        request_id=request.request_id,
        model_name="SAM2.1-Hiera-L",
        candidates=(
            SamMaskCandidateResponse(
                candidate_id="mask_00",
                score=0.9,
                mask_npz_path=mask_path,
                pixel_count=11,
                coverage_percent=0.5,
            ),
        ),
        latency_ms=34,
    )

    assert request.model_dump(mode="json")["points"][0]["label"] == "handle"
    assert response.model_dump(mode="json")["candidates"][0]["candidate_id"] == "mask_00"


def test_health_response() -> None:
    payload = HealthResponse(
        status="ok",
        model_name="MolmoPoint-8B",
        model_loaded=True,
    )

    assert payload.model_dump(mode="json")["status"] == "ok"


def test_sam_request_requires_point(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    with pytest.raises(ValidationError):
        SamMaskRequest(
            request_id="req-3",
            image_path=image_path,
            points=(),
            staging_dir=staging_dir,
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_backend_config.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  -q
```

Expected: fail with import errors for `codex_agent.scenefunc3d.backends` and `codex_agent.scenefunc3d.servers`.

- [ ] **Step 4: Implement config and schemas**

Create `src/codex_agent/scenefunc3d/backends/__init__.py`:

```python
"""SceneFunc3D backend integration package."""

from __future__ import annotations

__all__: list[str] = []
```

Create `src/codex_agent/scenefunc3d/servers/__init__.py`:

```python
"""Local SceneFunc3D sidecar server package."""

from __future__ import annotations

__all__: list[str] = []
```

Create `src/codex_agent/scenefunc3d/backends/config.py`:

```python
"""Configuration for local SceneFunc3D sidecar backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from ..tools.models import ToolInputError


@dataclass(frozen=True)
class SceneFunc3dBackendSettings:
    """Validated local sidecar backend settings."""

    molmo_url: str
    sam_url: str
    request_timeout_seconds: float
    artifact_staging_root: Path
    allowed_image_roots: tuple[Path, ...]
    allowed_output_roots: tuple[Path, ...]

    def __post_init__(self) -> None:
        """Validate URL, timeout, and root contracts."""
        _validate_local_http_url("molmo_url", self.molmo_url)
        _validate_local_http_url("sam_url", self.sam_url)
        if self.request_timeout_seconds <= 0.0:
            raise ToolInputError(
                "request_timeout_seconds must be positive; got "
                f"{self.request_timeout_seconds!r}"
            )
        if not self.allowed_image_roots:
            raise ToolInputError("allowed_image_roots must not be empty")
        if not self.allowed_output_roots:
            raise ToolInputError("allowed_output_roots must not be empty")


class _BackendSettingsFile(BaseModel):
    """Pydantic boundary model for TOML backend settings."""

    model_config = ConfigDict(extra="forbid")

    molmo_url: str = Field(min_length=1)
    sam_url: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0.0)
    artifact_staging_root: Path
    allowed_image_roots: tuple[Path, ...] = Field(min_length=1)
    allowed_output_roots: tuple[Path, ...] = Field(min_length=1)


def load_backend_settings(path: Path) -> SceneFunc3dBackendSettings:
    """Load sidecar backend settings from a TOML file."""
    if not path.is_file():
        raise ToolInputError(f"SceneFunc3D backend config is missing: {path}")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ToolInputError(f"invalid backend config TOML {path}: {exc}") from exc
    try:
        parsed = _BackendSettingsFile.model_validate(payload)
    except ValidationError as exc:
        raise ToolInputError(f"invalid backend config {path}: {exc}") from exc
    return SceneFunc3dBackendSettings(
        molmo_url=parsed.molmo_url,
        sam_url=parsed.sam_url,
        request_timeout_seconds=parsed.request_timeout_seconds,
        artifact_staging_root=parsed.artifact_staging_root,
        allowed_image_roots=parsed.allowed_image_roots,
        allowed_output_roots=parsed.allowed_output_roots,
    )


def _validate_local_http_url(field_name: str, url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ToolInputError(f"{field_name} must be an http://127.0.0.1 URL; got {url!r}")
    if parsed.port is None:
        raise ToolInputError(f"{field_name} must include an explicit port; got {url!r}")


__all__ = ["SceneFunc3dBackendSettings", "load_backend_settings"]
```

Create `src/codex_agent/scenefunc3d/servers/schemas.py` with Pydantic models:

```python
"""Shared JSON schemas for local SceneFunc3D sidecar servers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FilePath, NonNegativeFloat


class HealthResponse(BaseModel):
    """Sidecar server health response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    model_name: str = Field(min_length=1)
    model_loaded: bool


class MolmoPointRequest(BaseModel):
    """Molmo point inference request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    image_path: FilePath
    prompt: str = Field(min_length=1)
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)


class MolmoPointResponse(BaseModel):
    """Molmo point inference response."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    raw_text: str
    latency_ms: NonNegativeFloat


class SamPointPrompt(BaseModel):
    """One SAM point prompt in pixel coordinates."""

    model_config = ConfigDict(extra="forbid")

    x_px: float = Field(ge=0.0, strict=True)
    y_px: float = Field(ge=0.0, strict=True)
    label: str = ""
    source: str = ""


class SamMaskRequest(BaseModel):
    """SAM candidate mask request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    image_path: FilePath
    points: tuple[SamPointPrompt, ...] = Field(min_length=1)
    staging_dir: Path


class SamMaskCandidateResponse(BaseModel):
    """One SAM candidate mask response."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    mask_npz_path: FilePath
    pixel_count: int = Field(ge=0, strict=True)
    coverage_percent: float = Field(ge=0.0, le=100.0)


class SamMaskResponse(BaseModel):
    """SAM candidate mask response."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    candidates: tuple[SamMaskCandidateResponse, ...]
    latency_ms: NonNegativeFloat


__all__ = [
    "HealthResponse",
    "MolmoPointRequest",
    "MolmoPointResponse",
    "SamMaskCandidateResponse",
    "SamMaskRequest",
    "SamMaskResponse",
    "SamPointPrompt",
]
```

Create `configs/scenefunc3d_backends.example.toml`:

```toml
molmo_url = "http://127.0.0.1:8711"
sam_url = "http://127.0.0.1:8712"
request_timeout_seconds = 30.0
artifact_staging_root = "/tmp/scenefunc3d_artifacts"
allowed_image_roots = [
  "/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG",
  "/tmp/scenefunc3d_artifacts",
]
allowed_output_roots = [
  "/tmp/scenefunc3d_artifacts",
]
```

- [ ] **Step 5: Run tests and quality checks**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_backend_config.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  -q
ruff check src/codex_agent/scenefunc3d/backends src/codex_agent/scenefunc3d/servers src/codex_agent/tests/test_scenefunc3d_backend_config.py src/codex_agent/tests/test_scenefunc3d_server_schemas.py
black --check src/codex_agent/scenefunc3d/backends src/codex_agent/scenefunc3d/servers src/codex_agent/tests/test_scenefunc3d_backend_config.py src/codex_agent/tests/test_scenefunc3d_server_schemas.py
mypy src/codex_agent/scenefunc3d/backends src/codex_agent/scenefunc3d/servers src/codex_agent/tests/test_scenefunc3d_backend_config.py src/codex_agent/tests/test_scenefunc3d_server_schemas.py
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add \
  configs/scenefunc3d_backends.example.toml \
  src/codex_agent/scenefunc3d/backends/__init__.py \
  src/codex_agent/scenefunc3d/backends/config.py \
  src/codex_agent/scenefunc3d/servers/__init__.py \
  src/codex_agent/scenefunc3d/servers/schemas.py \
  src/codex_agent/tests/test_scenefunc3d_backend_config.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py
git commit -m "feat: add scenefunc3d sidecar backend schemas"
```

## Task 2: JSON HTTP Client And Fake Sidecar Server Harness

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/http_client.py`
- Create: `src/codex_agent/scenefunc3d/servers/http_json.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_http_client.py`

- [ ] **Step 1: Write failing HTTP client tests**

Create `src/codex_agent/tests/test_scenefunc3d_http_client.py`:

```python
"""Tests for local SceneFunc3D JSON HTTP client/server helpers."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from typing import TypeAlias

import pytest
from pydantic import BaseModel, ConfigDict

from codex_agent.scenefunc3d.backends.http_client import post_json
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.tools.models import ToolInputError

JsonObject: TypeAlias = dict[str, object]


class _EchoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    message: str


def test_post_json_round_trip() -> None:
    server = _start_server(
        {
            "/echo": lambda payload: {
                "ok": True,
                "message": str(payload["message"]),
            }
        }
    )
    try:
        response = post_json(
            f"http://127.0.0.1:{server.server_port}/echo",
            payload={"message": "hello"},
            response_model=_EchoResponse,
            timeout_seconds=2.0,
        )
    finally:
        server.shutdown()

    assert response.ok is True
    assert response.message == "hello"


def test_json_server_get_health_round_trip() -> None:
    server = _start_server(
        {
            "/health": lambda payload: {
                "ok": True,
                "message": "healthy",
            }
        }
    )
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/health", timeout=2.0
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()

    assert payload == {"ok": True, "message": "healthy"}


def test_post_json_http_500_is_recoverable() -> None:
    server = _start_server({"/fail": lambda payload: (_ for _ in ()).throw(RuntimeError("boom"))})
    try:
        with pytest.raises(ToolInputError, match="HTTP 500"):
            post_json(
                f"http://127.0.0.1:{server.server_port}/fail",
                payload={"message": "hello"},
                response_model=_EchoResponse,
                timeout_seconds=2.0,
            )
    finally:
        server.shutdown()


def test_post_json_schema_mismatch_is_recoverable() -> None:
    server = _start_server({"/bad": lambda payload: {"ok": "yes", "extra": 1}})
    try:
        with pytest.raises(ToolInputError, match="invalid response"):
            post_json(
                f"http://127.0.0.1:{server.server_port}/bad",
                payload={"message": "hello"},
                response_model=_EchoResponse,
                timeout_seconds=2.0,
            )
    finally:
        server.shutdown()


def _start_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    handler_type = make_json_handler(routes)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_http_client.py -q
```

Expected: fail with import errors for `backends.http_client` and `servers.http_json`.

- [ ] **Step 3: Implement HTTP client**

Create `src/codex_agent/scenefunc3d/backends/http_client.py`:

```python
"""JSON HTTP client for local SceneFunc3D sidecar backends."""

from __future__ import annotations

import json
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

from ..tools.models import ToolInputError

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def post_json(
    url: str,
    *,
    payload: dict[str, object],
    response_model: type[ResponseT],
    timeout_seconds: float,
) -> ResponseT:
    """POST a JSON object to a local sidecar and validate the JSON response."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ToolInputError(f"{url}: HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise ToolInputError(f"{url}: sidecar request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ToolInputError(f"{url}: sidecar request timed out") from exc
    try:
        decoded: object = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"{url}: invalid response JSON: {exc}") from exc
    try:
        return response_model.model_validate(decoded)
    except ValidationError as exc:
        raise ToolInputError(f"{url}: invalid response schema: {exc}") from exc


__all__ = ["post_json"]
```

- [ ] **Step 4: Implement JSON server helper**

Create `src/codex_agent/scenefunc3d/servers/http_json.py`:

```python
"""Small JSON HTTP server helpers for local SceneFunc3D sidecars."""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]
JsonRoute: TypeAlias = Callable[[JsonObject], JsonObject]


def make_json_handler(routes: dict[str, JsonRoute]) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class for a fixed mapping of POST routes."""

    class JsonHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            route = routes.get(self.path)
            if route is None:
                self._write_json(404, {"error": f"unknown route {self.path}"})
                return
            try:
                response = route({})
            except Exception as exc:  # route boundary converts to HTTP error
                self._write_json(500, {"error": str(exc)})
                return
            self._write_json(200, response)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            route = routes.get(self.path)
            if route is None:
                self._write_json(404, {"error": f"unknown route {self.path}"})
                return
            try:
                payload = self._read_json()
                response = route(payload)
            except Exception as exc:  # route boundary converts to HTTP error
                self._write_json(500, {"error": str(exc)})
                return
            self._write_json(200, response)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress noisy default access logs in tests and tmux."""

        def _read_json(self) -> JsonObject:
            length_header = self.headers.get("Content-Length", "0")
            content_length = int(length_header)
            raw_body = self.rfile.read(content_length).decode("utf-8")
            decoded: object = json.loads(raw_body)
            if not isinstance(decoded, dict):
                raise ValueError("request body must be a JSON object")
            return dict(decoded)

        def _write_json(self, status_code: int, payload: JsonObject) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return JsonHandler


__all__ = ["JsonObject", "JsonRoute", "make_json_handler"]
```

- [ ] **Step 5: Run tests and quality checks**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_http_client.py -q
ruff check src/codex_agent/scenefunc3d/backends/http_client.py src/codex_agent/scenefunc3d/servers/http_json.py src/codex_agent/tests/test_scenefunc3d_http_client.py
black --check src/codex_agent/scenefunc3d/backends/http_client.py src/codex_agent/scenefunc3d/servers/http_json.py src/codex_agent/tests/test_scenefunc3d_http_client.py
mypy src/codex_agent/scenefunc3d/backends/http_client.py src/codex_agent/scenefunc3d/servers/http_json.py src/codex_agent/tests/test_scenefunc3d_http_client.py
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/backends/http_client.py \
  src/codex_agent/scenefunc3d/servers/http_json.py \
  src/codex_agent/tests/test_scenefunc3d_http_client.py
git commit -m "feat: add scenefunc3d sidecar http client"
```

## Task 3: Wire Molmo Tool To RPC With Fake Server Tests

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/molmo_rpc.py`
- Modify: `src/codex_agent/scenefunc3d/tools/__main__.py`
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Modify: `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`

- [ ] **Step 1: Write failing Molmo RPC tests**

Create `src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py`:

```python
"""Tests for SceneFunc3D Molmo RPC backend integration."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.backends.config import SceneFunc3dBackendSettings
from codex_agent.scenefunc3d.backends.molmo_rpc import request_molmo_point
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.molmo_pointing import MolmoPointArgs


def test_request_molmo_point_returns_raw_text(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": '<point x="50" y="50">handle</point>',
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(tmp_path, molmo_url=f"http://127.0.0.1:{server.server_port}")
    args = MolmoPointArgs(
        frame_id="000050",
        image_path=image_path,
        prompt="drawer handle",
        image_width=100,
        image_height=80,
    )
    try:
        response = request_molmo_point(settings, request_id="req-1", args=args)
    finally:
        server.shutdown()

    assert response.raw_text == '<point x="50" y="50">handle</point>'


def test_request_molmo_point_server_down_is_recoverable(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    settings = _settings(tmp_path, molmo_url="http://127.0.0.1:9")
    args = MolmoPointArgs(
        frame_id="000050",
        image_path=image_path,
        prompt="drawer handle",
        image_width=100,
        image_height=80,
    )

    with pytest.raises(ToolInputError, match="sidecar request failed"):
        request_molmo_point(settings, request_id="req-2", args=args)


def _settings(tmp_path: Path, *, molmo_url: str) -> SceneFunc3dBackendSettings:
    return SceneFunc3dBackendSettings(
        molmo_url=molmo_url,
        sam_url="http://127.0.0.1:8712",
        request_timeout_seconds=2.0,
        artifact_staging_root=tmp_path,
        allowed_image_roots=(tmp_path,),
        allowed_output_roots=(tmp_path,),
    )


def _start_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_json_handler(routes))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
```

- [ ] **Step 2: Add failing CLI integration test for configured Molmo RPC**

Append to `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`:

```python
def test_cli_molmo_point_uses_configured_fake_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene(tmp_path)
    server = _start_json_server(
        {
            "/v1/point": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "MolmoPoint-8B",
                "raw_text": '<point x="50" y="50">handle</point>',
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{server.server_port}",
        sam_url="http://127.0.0.1:8712",
    )
    try:
        code = main(
            [
                "molmo_point",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "prompt": "drawer handle",
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    finally:
        server.shutdown()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["points"][0]["label"] == "handle"
    assert Path(payload["raw_text_path"]).exists()
    assert Path(payload["overlay_path"]).exists()
```

Also add helper functions in the same test file:

```python
def _write_backend_config(tmp_path: Path, *, molmo_url: str, sam_url: str) -> Path:
    config_path = tmp_path / "scenefunc3d_backends.toml"
    output_root = tmp_path / "out"
    output_root.mkdir()
    config_path.write_text(
        f"""
molmo_url = "{molmo_url}"
sam_url = "{sam_url}"
request_timeout_seconds = 2.0
artifact_staging_root = "{output_root}"
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{output_root}"]
""",
        encoding="utf-8",
    )
    return config_path
```

Use `_start_json_server()` with the same implementation as Task 2 tests.

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py::test_cli_molmo_point_uses_configured_fake_backend \
  -q
```

Expected: fail with import error for `backends.molmo_rpc` or CLI argument error for missing `--backend-config`.

- [ ] **Step 4: Implement Molmo RPC client**

Create `src/codex_agent/scenefunc3d/backends/molmo_rpc.py`:

```python
"""Molmo sidecar RPC client for SceneFunc3D tools."""

from __future__ import annotations

from .config import SceneFunc3dBackendSettings
from .http_client import post_json
from ..servers.schemas import MolmoPointResponse
from ..tools.molmo_pointing import MolmoPointArgs


def request_molmo_point(
    settings: SceneFunc3dBackendSettings,
    *,
    request_id: str,
    args: MolmoPointArgs,
) -> MolmoPointResponse:
    """Call the configured local Molmo sidecar."""
    return post_json(
        f"{settings.molmo_url.rstrip('/')}/v1/point",
        payload={
            "request_id": request_id,
            "image_path": str(args.image_path),
            "prompt": args.prompt,
            "image_width": args.image_width,
            "image_height": args.image_height,
        },
        response_model=MolmoPointResponse,
        timeout_seconds=settings.request_timeout_seconds,
    )


__all__ = ["request_molmo_point"]
```

- [ ] **Step 5: Thread backend config through CLI and dispatch**

Modify `src/codex_agent/scenefunc3d/tools/__main__.py`:

```python
parser.add_argument(
    "--backend-config",
    type=Path,
    default=None,
    help="Optional SceneFunc3D sidecar backend TOML config.",
)
```

After parsing args:

```python
backend_config_path = args.backend_config
payload = run_tool(
    tool_scene,
    args.tool,
    raw_args,
    out_dir=out_dir,
    backend_config_path=backend_config_path,
)
```

Modify `run_tool()` signature in `dispatch.py`:

```python
def run_tool(
    tool_scene: SceneFunc3dToolScene,
    name: str,
    raw_args: Mapping[str, object],
    *,
    out_dir: Path,
    backend_config_path: Path | None = None,
) -> ToolPayload:
```

Existing non-heavy branches ignore `backend_config_path`. Heavy branches pass it to tool functions.

- [ ] **Step 6: Implement Molmo tool call and artifacts**

Modify `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`:

```python
def molmo_point(
    args: MolmoPointArgs,
    *,
    out_dir: Path,
    backend_config_path: Path | None,
) -> MolmoPointResult:
    """Call Molmo sidecar, parse points, and write raw/overlay artifacts."""
    if backend_config_path is None:
        raise ToolInputError("molmo_point backend config is required")
    from ..backends.config import load_backend_settings
    from ..backends.molmo_rpc import request_molmo_point

    settings = load_backend_settings(backend_config_path)
    request_id = f"{args.frame_id}_molmo"
    response = request_molmo_point(settings, request_id=request_id, args=args)
    raw_text_path = _write_raw_text(out_dir, args.frame_id, response.raw_text)
    points = parse_molmo_points(
        response.raw_text,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    overlay_path = _write_point_overlay(args.image_path, out_dir, args.frame_id, points)
    return MolmoPointResult(
        frame_id=args.frame_id,
        prompt=args.prompt,
        points=points,
        raw_text_path=raw_text_path,
        overlay_path=overlay_path,
    )
```

Add helpers:

```python
def _write_raw_text(out_dir: Path, frame_id: str, raw_text: str) -> Path:
    path = out_dir / "molmo" / f"{frame_id}_raw.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_text, encoding="utf-8")
    return path
```

For `_write_point_overlay()`, lazy import Pillow, open image, draw one red circle per point, save to `out_dir / "molmo" / f"{frame_id}_points.jpg"`, and wrap `OSError`/`ImportError` in `ToolInputError`.

In `dispatch.py`, replace the `molmo_point` stub with:

```python
if name == "molmo_point":
    from .molmo_pointing import MolmoPointArgs, molmo_point

    return molmo_point(
        _parse(MolmoPointArgs, raw_args),
        out_dir=out_dir,
        backend_config_path=backend_config_path,
    )
```

- [ ] **Step 7: Run tests and quality checks**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py \
  -q
ruff check src/codex_agent/scenefunc3d/backends/molmo_rpc.py src/codex_agent/scenefunc3d/tools/__main__.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/scenefunc3d/tools/molmo_pointing.py src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
black --check src/codex_agent/scenefunc3d/backends/molmo_rpc.py src/codex_agent/scenefunc3d/tools/__main__.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/scenefunc3d/tools/molmo_pointing.py src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
mypy src/codex_agent/scenefunc3d/backends/molmo_rpc.py src/codex_agent/scenefunc3d/tools/__main__.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/scenefunc3d/tools/molmo_pointing.py src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
```

Expected: all commands pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/backends/molmo_rpc.py \
  src/codex_agent/scenefunc3d/tools/__main__.py \
  src/codex_agent/scenefunc3d/tools/dispatch.py \
  src/codex_agent/scenefunc3d/tools/molmo_pointing.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
git commit -m "feat: connect scenefunc3d molmo tool to sidecar"
```

## Task 4: Wire SAM Tool To RPC With Fake Server Tests

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/sam_rpc.py`
- Modify: `src/codex_agent/scenefunc3d/tools/sam_masking.py`
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_sam_rpc.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`

- [ ] **Step 1: Write failing SAM RPC tests**

Create `src/codex_agent/tests/test_scenefunc3d_sam_rpc.py`:

```python
"""Tests for SceneFunc3D SAM RPC backend integration."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.config import SceneFunc3dBackendSettings
from codex_agent.scenefunc3d.backends.sam_rpc import request_sam_masks
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.sam_masking import SamMaskArgs


def test_request_sam_masks_returns_candidate_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    mask_path = staging_dir / "mask_00.npz"
    np.savez_compressed(mask_path, mask=np.array([[True, False]], dtype=bool))
    server = _start_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.9,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 1,
                        "coverage_percent": 50.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    settings = _settings(tmp_path, sam_url=f"http://127.0.0.1:{server.server_port}")
    args = SamMaskArgs(
        frame_id="000050",
        image_path=image_path,
        points=({"x_px": 10.0, "y_px": 20.0, "label": "handle", "source": ""},),
    )
    try:
        response = request_sam_masks(
            settings,
            request_id="req-1",
            args=args,
            staging_dir=staging_dir,
        )
    finally:
        server.shutdown()

    assert response.candidates[0].candidate_id == "mask_00"
    assert response.candidates[0].mask_npz_path == mask_path


def test_request_sam_masks_server_down_is_recoverable(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    settings = _settings(tmp_path, sam_url="http://127.0.0.1:9")
    args = SamMaskArgs(
        frame_id="000050",
        image_path=image_path,
        points=({"x_px": 10.0, "y_px": 20.0, "label": "handle", "source": ""},),
    )

    with pytest.raises(ToolInputError, match="sidecar request failed"):
        request_sam_masks(settings, request_id="req-2", args=args, staging_dir=staging_dir)


def _settings(tmp_path: Path, *, sam_url: str) -> SceneFunc3dBackendSettings:
    return SceneFunc3dBackendSettings(
        molmo_url="http://127.0.0.1:8711",
        sam_url=sam_url,
        request_timeout_seconds=2.0,
        artifact_staging_root=tmp_path,
        allowed_image_roots=(tmp_path,),
        allowed_output_roots=(tmp_path,),
    )


def _start_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_json_handler(routes))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
```

- [ ] **Step 2: Add failing CLI integration test for configured SAM RPC**

Append to `test_scenefunc3d_molmo_sam_contracts.py`:

```python
def test_cli_sam_mask_uses_configured_fake_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir, image_path = _write_cli_scene(tmp_path)
    staging_dir = tmp_path / "out" / "sam_staging"
    staging_dir.mkdir(parents=True)
    mask_path = staging_dir / "mask_00.npz"
    np.savez_compressed(mask_path, mask=np.array([[True, False]], dtype=bool))
    server = _start_json_server(
        {
            "/v1/masks": lambda payload: {
                "request_id": payload["request_id"],
                "model_name": "SAM2.1-Hiera-L",
                "candidates": [
                    {
                        "candidate_id": "mask_00",
                        "score": 0.9,
                        "mask_npz_path": str(mask_path),
                        "pixel_count": 1,
                        "coverage_percent": 50.0,
                    }
                ],
                "latency_ms": 1.0,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url="http://127.0.0.1:8711",
        sam_url=f"http://127.0.0.1:{server.server_port}",
    )
    try:
        code = main(
            [
                "sam_mask",
                "--scene-root",
                str(scene_dir),
                "--backend-config",
                str(config_path),
                "--args",
                json.dumps(
                    {
                        "frame_id": "000000",
                        "image_path": str(image_path),
                        "points": [{"x_px": 1.0, "y_px": 1.0, "label": "handle", "source": ""}],
                    }
                ),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    finally:
        server.shutdown()

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["candidates"][0]["candidate_id"] == "mask_00"
    assert Path(payload["candidates"][0]["mask_npz_path"]).exists()
    assert Path(payload["contact_sheet_path"]).exists()
```

Add `import numpy as np` at the top of the test file.

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py::test_cli_sam_mask_uses_configured_fake_backend \
  -q
```

Expected: fail with import error for `backends.sam_rpc` or current backend-not-configured stub.

- [ ] **Step 4: Implement SAM RPC client**

Create `src/codex_agent/scenefunc3d/backends/sam_rpc.py`:

```python
"""SAM sidecar RPC client for SceneFunc3D tools."""

from __future__ import annotations

from pathlib import Path

from .config import SceneFunc3dBackendSettings
from .http_client import post_json
from ..servers.schemas import SamMaskResponse
from ..tools.sam_masking import SamMaskArgs


def request_sam_masks(
    settings: SceneFunc3dBackendSettings,
    *,
    request_id: str,
    args: SamMaskArgs,
    staging_dir: Path,
) -> SamMaskResponse:
    """Call the configured local SAM sidecar."""
    return post_json(
        f"{settings.sam_url.rstrip('/')}/v1/masks",
        payload={
            "request_id": request_id,
            "image_path": str(args.image_path),
            "points": [point.to_payload() for point in args.points],
            "staging_dir": str(staging_dir),
        },
        response_model=SamMaskResponse,
        timeout_seconds=settings.request_timeout_seconds,
    )


__all__ = ["request_sam_masks"]
```

- [ ] **Step 5: Add mask path to SAM result contract**

Modify `src/codex_agent/scenefunc3d/tools/sam_masking.py` so the accepted SAM mask artifact path is visible to the agent and can be passed directly into `lift_mask_to_3d`:

```python
class SamCandidatePayload(TypedDict):
    """JSON-ready payload for one SAM mask candidate."""

    candidate_id: str
    score: float
    pixel_count: int
    coverage_percent: float
    mask_npz_path: str
    overlay_path: str


@dataclass(frozen=True)
class SamCandidate:
    """One SAM mask candidate proposed from a Molmo point prompt."""

    candidate_id: str
    score: float
    pixel_count: int
    coverage_percent: float
    mask_npz_path: Path
    overlay_path: Path

    def __post_init__(self) -> None:
        """Validate directly constructed candidate contracts."""
        if not self.candidate_id.strip():
            raise SceneFunc3dDataError("candidate_id must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise SceneFunc3dDataError(
                f"candidate score must be in [0, 1]; got {self.score!r}"
            )
        if self.pixel_count < 0:
            raise SceneFunc3dDataError(
                f"candidate pixel_count must be non-negative; got {self.pixel_count!r}"
            )
        if not 0.0 <= self.coverage_percent <= 100.0:
            raise SceneFunc3dDataError(
                "candidate coverage_percent must be in [0, 100]; "
                f"got {self.coverage_percent!r}"
            )

    def to_payload(self) -> SamCandidatePayload:
        """Return this candidate as a JSON-ready mapping."""
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "pixel_count": self.pixel_count,
            "coverage_percent": self.coverage_percent,
            "mask_npz_path": str(self.mask_npz_path),
            "overlay_path": str(self.overlay_path),
        }
```

- [ ] **Step 6: Implement SAM tool artifacts**

Modify `src/codex_agent/scenefunc3d/tools/sam_masking.py` to add:

```python
def sam_mask(
    args: SamMaskArgs,
    *,
    out_dir: Path,
    backend_config_path: Path | None,
) -> SamMaskResult:
    """Call SAM sidecar and render candidate review artifacts."""
    if backend_config_path is None:
        raise ToolInputError("sam_mask backend config is required")
    from ..backends.config import load_backend_settings
    from ..backends.sam_rpc import request_sam_masks

    settings = load_backend_settings(backend_config_path)
    staging_dir = out_dir / "sam" / args.frame_id / "candidates"
    staging_dir.mkdir(parents=True, exist_ok=True)
    response = request_sam_masks(
        settings,
        request_id=f"{args.frame_id}_sam",
        args=args,
        staging_dir=staging_dir,
    )
    candidates = tuple(_candidate_from_response(item, out_dir, args) for item in response.candidates)
    contact_sheet_path = _write_contact_sheet(args.image_path, candidates, out_dir, args.frame_id)
    return SamMaskResult(
        frame_id=args.frame_id,
        candidates=candidates,
        contact_sheet_path=contact_sheet_path,
    )
```

Implement `_candidate_from_response()` to load `item.mask_npz_path`, verify `mask` key exists and is 2D bool-like, write overlay to `out_dir / "sam" / frame_id / f"{candidate_id}_overlay.jpg"`, and return:

```python
return SamCandidate(
    candidate_id=item.candidate_id,
    score=item.score,
    pixel_count=item.pixel_count,
    coverage_percent=item.coverage_percent,
    mask_npz_path=item.mask_npz_path,
    overlay_path=overlay_path,
)
```

Implement `_write_contact_sheet()` with Pillow by horizontally stacking overlays with candidate ids. Wrap missing arrays, invalid shapes, Pillow failures, and file IO failures in `ToolInputError`.

In `dispatch.py`, replace the `sam_mask` stub:

```python
if name == "sam_mask":
    from .sam_masking import SamMaskArgs, sam_mask

    return sam_mask(
        _parse(SamMaskArgs, raw_args),
        out_dir=out_dir,
        backend_config_path=backend_config_path,
    )
```

- [ ] **Step 7: Run tests and quality checks**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py \
  -q
ruff check src/codex_agent/scenefunc3d/backends/sam_rpc.py src/codex_agent/scenefunc3d/tools/sam_masking.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_sam_rpc.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
black --check src/codex_agent/scenefunc3d/backends/sam_rpc.py src/codex_agent/scenefunc3d/tools/sam_masking.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_sam_rpc.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
mypy src/codex_agent/scenefunc3d/backends/sam_rpc.py src/codex_agent/scenefunc3d/tools/sam_masking.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_sam_rpc.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
```

Expected: all commands pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/backends/sam_rpc.py \
  src/codex_agent/scenefunc3d/tools/sam_masking.py \
  src/codex_agent/scenefunc3d/tools/dispatch.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
git commit -m "feat: connect scenefunc3d sam tool to sidecar"
```

## Task 5: Deterministic 3D Lifting

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/lift_3d.py`
- Create: `src/codex_agent/scenefunc3d/backends/frame_assets.py`
- Modify: `src/codex_agent/scenefunc3d/tools/mask_lifting.py`
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_lift3d.py`
- Extend: `src/codex_agent/tests/test_scenefunc3d_metrics.py`

- [ ] **Step 1: Write failing synthetic lift tests**

Create `src/codex_agent/tests/test_scenefunc3d_lift3d.py`:

```python
"""Tests for deterministic SceneFunc3D 2D-mask-to-3D lifting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.lift_3d import (
    CameraGeometry,
    backproject_mask_to_world,
    load_mask_npz,
    write_lift_npz,
    write_lift_ply,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError


def test_backproject_mask_to_world_identity_pose() -> None:
    mask = np.array([[False, True], [False, False]], dtype=bool)
    depth = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    geometry = CameraGeometry(
        intrinsics=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
        camera_to_world=np.eye(4, dtype=np.float64),
    )

    points = backproject_mask_to_world(mask, depth, geometry)

    assert points.shape == (1, 3)
    assert points[0].tolist() == [1.0, 0.0, 2.0]


def test_backproject_rejects_empty_valid_depth() -> None:
    mask = np.array([[True]], dtype=bool)
    depth = np.array([[0.0]], dtype=np.float64)
    geometry = CameraGeometry(
        intrinsics=np.eye(3, dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )

    with pytest.raises(ToolInputError, match="lift_too_sparse"):
        backproject_mask_to_world(mask, depth, geometry)


def test_mask_npz_round_trip(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.npz"
    np.savez_compressed(mask_path, mask=np.array([[True, False]], dtype=bool))

    loaded = load_mask_npz(mask_path)

    assert loaded.dtype == bool
    assert loaded.shape == (1, 2)


def test_write_lift_artifacts(tmp_path: Path) -> None:
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    npz_path = write_lift_npz(tmp_path / "mask_data.npz", points)
    ply_path = write_lift_ply(tmp_path / "lifted_points.ply", points)

    assert npz_path.is_file()
    assert ply_path.read_text(encoding="utf-8").startswith("ply")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_lift3d.py -q
```

Expected: fail with import error for `backends.lift_3d`.

- [ ] **Step 3: Implement lift_3d core**

Create `src/codex_agent/scenefunc3d/backends/lift_3d.py`:

```python
"""Deterministic 2D mask to 3D point lifting for SceneFunc3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..tools.models import ToolInputError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CameraGeometry:
    """Camera intrinsics and camera-to-world pose."""

    intrinsics: FloatArray
    camera_to_world: FloatArray

    def __post_init__(self) -> None:
        """Validate camera matrix contracts."""
        if self.intrinsics.shape != (3, 3):
            raise ToolInputError(f"intrinsics must be 3x3, got {self.intrinsics.shape}")
        if self.camera_to_world.shape != (4, 4):
            raise ToolInputError(
                f"camera_to_world must be 4x4, got {self.camera_to_world.shape}"
            )


def load_mask_npz(path: Path) -> BoolArray:
    """Load a 2D bool mask from a compressed NPZ file."""
    try:
        with np.load(path) as payload:
            if "mask" not in payload:
                raise ToolInputError(f"lift_invalid_mask: {path} has no 'mask' array")
            mask = np.asarray(payload["mask"], dtype=bool)
    except OSError as exc:
        raise ToolInputError(f"lift_invalid_mask: could not read {path}: {exc}") from exc
    if mask.ndim != 2:
        raise ToolInputError(f"lift_invalid_mask: mask must be 2D, got {mask.shape}")
    return mask


def backproject_mask_to_world(
    mask: BoolArray,
    depth_meters: FloatArray,
    geometry: CameraGeometry,
) -> FloatArray:
    """Backproject masked valid-depth pixels into world coordinates."""
    if mask.shape != depth_meters.shape:
        raise ToolInputError(
            f"lift_invalid_mask: mask shape {mask.shape} != depth shape {depth_meters.shape}"
        )
    valid = mask & np.isfinite(depth_meters) & (depth_meters > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        raise ToolInputError("lift_too_sparse: no valid depth pixels under mask")
    z = depth_meters[rows, cols]
    fx = float(geometry.intrinsics[0, 0])
    fy = float(geometry.intrinsics[1, 1])
    cx = float(geometry.intrinsics[0, 2])
    cy = float(geometry.intrinsics[1, 2])
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    cam_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
    world_h = (geometry.camera_to_world @ cam_points.T).T
    return np.ascontiguousarray(world_h[:, :3], dtype=np.float64)


def write_lift_npz(path: Path, points_world: FloatArray) -> Path:
    """Write lifted points to NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points_world=points_world)
    return path


def write_lift_ply(path: Path, points_world: FloatArray) -> Path:
    """Write lifted points as an ASCII PLY point cloud."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points_world.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{x:.8f} {y:.8f} {z:.8f}" for x, y, z in points_world)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "BoolArray",
    "CameraGeometry",
    "FloatArray",
    "backproject_mask_to_world",
    "load_mask_npz",
    "write_lift_npz",
    "write_lift_ply",
]
```

- [ ] **Step 4: Implement frame asset resolver**

Create `src/codex_agent/scenefunc3d/backends/frame_assets.py`:

```python
"""Resolve frame-level RGB/depth/intrinsics/pose assets for SceneFunc3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..tools.models import ToolInputError
from ..tools.scene_context import SceneFunc3dToolScene


@dataclass(frozen=True)
class FrameGeometryAssets:
    """Paths required to lift one frame's mask into 3D."""

    frame_id: str
    depth_path: Path
    intrinsics_path: Path
    pose_path: Path


def resolve_frame_geometry_assets(
    tool_scene: SceneFunc3dToolScene, frame_id: str
) -> FrameGeometryAssets:
    """Resolve depth, intrinsics, and pose paths for one frame."""
    depth_path = _first_existing(
        (
            tool_scene.raw_dir / f"{frame_id}-depth.png",
            tool_scene.raw_dir / f"{frame_id}_depth.png",
        ),
        error_code="lift_depth_missing",
    )
    intrinsics_path = _first_existing(
        (
            tool_scene.raw_dir / f"{frame_id}-intrinsics.txt",
            tool_scene.raw_dir / f"{frame_id}.pincam",
            tool_scene.raw_dir / "intrinsics.txt",
        ),
        error_code="lift_intrinsics_missing",
    )
    pose_path = _first_existing(
        (
            tool_scene.raw_dir / f"{frame_id}-pose.txt",
            tool_scene.raw_dir / f"{frame_id}.pose.txt",
        ),
        error_code="lift_pose_missing",
    )
    return FrameGeometryAssets(
        frame_id=frame_id,
        depth_path=depth_path,
        intrinsics_path=intrinsics_path,
        pose_path=pose_path,
    )


def _first_existing(paths: tuple[Path, ...], *, error_code: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise ToolInputError(f"{error_code}: checked {[str(path) for path in paths]}")


__all__ = ["FrameGeometryAssets", "resolve_frame_geometry_assets"]
```

This first resolver is intentionally narrow and testable. Later tasks may extend it to parse `source_frames.json` depth/intrinsic fields and ConceptGraph trajectories for real SceneFuncVal-CG scenes.

- [ ] **Step 5: Implement tool-level lifting call**

Modify `mask_lifting.py` to add:

```python
def lift_mask_to_3d(args: LiftMaskArgs, *, out_dir: Path) -> LiftMaskResult:
    """Lift one accepted 2D mask into a 3D point artifact."""
    from PIL import Image

    from ..backends.lift_3d import (
        CameraGeometry,
        backproject_mask_to_world,
        load_mask_npz,
        write_lift_npz,
        write_lift_ply,
    )

    mask = load_mask_npz(args.mask_path)
    depth = np.asarray(Image.open(args.depth_path), dtype=np.float64) / 1000.0
    intrinsics = np.loadtxt(args.intrinsics_path, dtype=np.float64).reshape(3, 3)
    pose = np.loadtxt(args.pose_path, dtype=np.float64).reshape(4, 4)
    points = backproject_mask_to_world(
        mask,
        depth,
        CameraGeometry(intrinsics=intrinsics, camera_to_world=pose),
    )
    fragment_dir = out_dir / "fragments" / f"{args.frame_id}_{args.candidate_id}"
    npz_path = write_lift_npz(fragment_dir / "mask_data.npz", points)
    ply_path = write_lift_ply(fragment_dir / "lifted_points.ply", points)
    overlay_path = fragment_dir / "lift_overlay.txt"
    overlay_path.write_text(
        f"lifted_point_count={points.shape[0]}\n",
        encoding="utf-8",
    )
    return LiftMaskResult(
        frame_id=args.frame_id,
        candidate_id=args.candidate_id,
        lifted_point_count=int(points.shape[0]),
        mask_npz_path=npz_path,
        mask_ply_path=ply_path,
        overlay_path=overlay_path,
    )
```

Add lazy imports for `numpy` and `PIL` inside the function, wrapping `ImportError`, `OSError`, and `ValueError` into `ToolInputError`. Keep `LiftMaskArgs` file fields as `FilePath`.

In `dispatch.py`, replace the `lift_mask_to_3d` stub:

```python
if name == "lift_mask_to_3d":
    from .mask_lifting import LiftMaskArgs, lift_mask_to_3d

    return lift_mask_to_3d(_parse(LiftMaskArgs, raw_args), out_dir=out_dir)
```

- [ ] **Step 6: Run tests and quality checks**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_lift3d.py \
  src/codex_agent/tests/test_scenefunc3d_metrics.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  -q
ruff check src/codex_agent/scenefunc3d/backends/lift_3d.py src/codex_agent/scenefunc3d/backends/frame_assets.py src/codex_agent/scenefunc3d/tools/mask_lifting.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_lift3d.py src/codex_agent/tests/test_scenefunc3d_metrics.py
black --check src/codex_agent/scenefunc3d/backends/lift_3d.py src/codex_agent/scenefunc3d/backends/frame_assets.py src/codex_agent/scenefunc3d/tools/mask_lifting.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_lift3d.py src/codex_agent/tests/test_scenefunc3d_metrics.py
mypy src/codex_agent/scenefunc3d/backends/lift_3d.py src/codex_agent/scenefunc3d/backends/frame_assets.py src/codex_agent/scenefunc3d/tools/mask_lifting.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_lift3d.py src/codex_agent/tests/test_scenefunc3d_metrics.py
```

Expected: all commands pass.

- [ ] **Step 7: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/backends/lift_3d.py \
  src/codex_agent/scenefunc3d/backends/frame_assets.py \
  src/codex_agent/scenefunc3d/tools/mask_lifting.py \
  src/codex_agent/scenefunc3d/tools/dispatch.py \
  src/codex_agent/tests/test_scenefunc3d_lift3d.py \
  src/codex_agent/tests/test_scenefunc3d_metrics.py
git commit -m "feat: add scenefunc3d deterministic mask lifting"
```

## Task 6: Real MolmoPoint Sidecar Server

**Files:**
- Create: `src/codex_agent/scenefunc3d/servers/molmo_point_server.py`
- Create: `scripts/scenefunc3d/serve_molmo_point.sh`
- Test: `src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py`

- [ ] **Step 1: Write server CLI tests using a fake runner**

Create `src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py`:

```python
"""Tests for Molmo sidecar server CLI pieces that do not load the model."""

from __future__ import annotations

from codex_agent.scenefunc3d.servers.molmo_point_server import build_arg_parser


def test_molmo_server_arg_parser() -> None:
    args = build_arg_parser().parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "8711",
            "--model-name",
            "MolmoPoint-8B",
            "--model-path",
            "/models/molmo",
            "--device",
            "cuda:0",
        ]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8711
    assert args.model_name == "MolmoPoint-8B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py -q
```

Expected: fail with import error for `servers.molmo_point_server`.

- [ ] **Step 3: Implement Molmo server CLI and lazy runner**

Create `src/codex_agent/scenefunc3d/servers/molmo_point_server.py`:

```python
"""Local MolmoPoint sidecar server for SceneFunc3D."""

from __future__ import annotations

import argparse
import time
from collections.abc import Protocol
from http.server import ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from .http_json import JsonObject, make_json_handler
from .schemas import HealthResponse, MolmoPointRequest, MolmoPointResponse


class MolmoRunner(Protocol):
    """Runtime interface for Molmo model inference."""

    model_name: str

    def point(self, request: MolmoPointRequest) -> str:
        """Return raw Molmo text for one request."""


class TransformersMolmoRunner:
    """Molmo runner backed by Hugging Face transformers."""

    def __init__(self, *, model_name: str, model_path: Path, device: str) -> None:
        self.model_name = model_name
        self._device = device
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        except ImportError as exc:
            raise RuntimeError(
                "Molmo server requires torch and transformers in the server environment"
            ) from exc
        self._torch = torch
        self._generation_config_type = GenerationConfig
        self._processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self._model.to(device)

    def point(self, request: MolmoPointRequest) -> str:
        """Run Molmo pointing inference and return raw generated text."""
        from PIL import Image

        image = Image.open(request.image_path).convert("RGB")
        inputs = self._processor.process(images=[image], text=request.prompt)
        inputs = {key: value.to(self._model.device).unsqueeze(0) for key, value in inputs.items()}
        output = self._model.generate_from_batch(
            inputs,
            self._generation_config_type(max_new_tokens=128, stop_strings="<|endoftext|>"),
            tokenizer=self._processor.tokenizer,
        )
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        return self._processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build Molmo sidecar argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def serve(runner: MolmoRunner, *, host: str, port: int) -> None:
    """Serve Molmo JSON endpoints forever."""
    routes = {
        "/health": lambda payload: HealthResponse(
            status="ok", model_name=runner.model_name, model_loaded=True
        ).model_dump(mode="json"),
        "/v1/point": lambda payload: _handle_point(runner, payload),
    }
    server = ThreadingHTTPServer((host, port), make_json_handler(routes))
    server.serve_forever()


def _handle_point(runner: MolmoRunner, payload: JsonObject) -> JsonObject:
    started = time.perf_counter()
    try:
        request = MolmoPointRequest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid Molmo request: {exc}") from exc
    raw_text = runner.point(request)
    return MolmoPointResponse(
        request_id=request.request_id,
        model_name=runner.model_name,
        raw_text=raw_text,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    ).model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    """Start the Molmo sidecar server."""
    args = build_arg_parser().parse_args(argv)
    runner = TransformersMolmoRunner(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
    )
    serve(runner, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Before changing the `TransformersMolmoRunner.point()` implementation, verify the local API with:

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoProcessor
print(AutoProcessor)
print(hasattr(AutoModelForCausalLM, "from_pretrained"))
PY
```

If the installed Molmo package exposes a different call signature, change only `TransformersMolmoRunner.point()` and add one focused unit test with a fake processor/model object. Keep the `MolmoRunner` protocol and HTTP schema stable.

- [ ] **Step 4: Add launcher script**

Create `scripts/scenefunc3d/serve_molmo_point.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m codex_agent.scenefunc3d.servers.molmo_point_server "$@"
```

Set executable bit:

```bash
chmod +x scripts/scenefunc3d/serve_molmo_point.sh
```

- [ ] **Step 5: Run tests and quality checks**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py -q
ruff check src/codex_agent/scenefunc3d/servers/molmo_point_server.py src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py
black --check src/codex_agent/scenefunc3d/servers/molmo_point_server.py src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py
mypy src/codex_agent/scenefunc3d/servers/molmo_point_server.py src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py
```

Expected: all commands pass without importing torch unless `TransformersMolmoRunner` is instantiated.

- [ ] **Step 6: Commit**

```bash
git add \
  scripts/scenefunc3d/serve_molmo_point.sh \
  src/codex_agent/scenefunc3d/servers/molmo_point_server.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py
git commit -m "feat: add scenefunc3d molmo sidecar server"
```

## Task 7: Real SAM2.1 Sidecar Server

**Files:**
- Create: `src/codex_agent/scenefunc3d/servers/sam2_mask_server.py`
- Create: `scripts/scenefunc3d/serve_sam2.sh`
- Test: `src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py`

- [ ] **Step 1: Write server CLI tests**

Create `src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py`:

```python
"""Tests for SAM2 sidecar server CLI pieces that do not load the model."""

from __future__ import annotations

from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser


def test_sam2_server_arg_parser() -> None:
    args = build_arg_parser().parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "8712",
            "--model-name",
            "SAM2.1-Hiera-L",
            "--checkpoint-path",
            "/models/sam2.pt",
            "--config-path",
            "/models/sam2.yaml",
            "--device",
            "cuda:0",
        ]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8712
    assert args.model_name == "SAM2.1-Hiera-L"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py -q
```

Expected: fail with import error for `servers.sam2_mask_server`.

- [ ] **Step 3: Implement SAM2 server CLI and lazy runner**

Create `src/codex_agent/scenefunc3d/servers/sam2_mask_server.py`:

```python
"""Local SAM2.1 sidecar server for SceneFunc3D."""

from __future__ import annotations

import argparse
import time
from collections.abc import Protocol
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .http_json import JsonObject, make_json_handler
from .schemas import (
    HealthResponse,
    SamMaskCandidateResponse,
    SamMaskRequest,
    SamMaskResponse,
)


class Sam2Runner(Protocol):
    """Runtime interface for SAM2 model inference."""

    model_name: str

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        """Return candidate mask paths for one request."""


class OfficialSam2Runner:
    """SAM2 runner backed by facebookresearch/sam2."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: Path,
        config_path: Path,
        device: str,
    ) -> None:
        self.model_name = model_name
        try:
            from PIL import Image
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 server requires pillow and the facebookresearch sam2 package"
            ) from exc
        self._image_type = Image
        model = build_sam2(str(config_path), str(checkpoint_path), device=device)
        self._predictor = SAM2ImagePredictor(model)

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        """Run SAM2 image prediction for point prompts."""
        image = np.asarray(self._image_type.open(request.image_path).convert("RGB"))
        self._predictor.set_image(image)
        point_coords = np.array(
            [[point.x_px, point.y_px] for point in request.points],
            dtype=np.float32,
        )
        point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        request.staging_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[SamMaskCandidateResponse] = []
        height, width = image.shape[:2]
        for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
            candidate_id = f"mask_{index:02d}"
            mask_bool = np.asarray(mask, dtype=bool)
            mask_path = request.staging_dir / f"{candidate_id}.npz"
            np.savez_compressed(mask_path, mask=mask_bool)
            pixel_count = int(mask_bool.sum())
            candidates.append(
                SamMaskCandidateResponse(
                    candidate_id=candidate_id,
                    score=float(score),
                    mask_npz_path=mask_path,
                    pixel_count=pixel_count,
                    coverage_percent=100.0 * pixel_count / float(height * width),
                )
            )
        return tuple(candidates)
```

Add `build_arg_parser()`, `serve()`, `_handle_masks()`, and `main()` mirroring Task 6, with `/health` and `/v1/masks`.

- [ ] **Step 4: Add launcher script**

Create `scripts/scenefunc3d/serve_sam2.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m codex_agent.scenefunc3d.servers.sam2_mask_server "$@"
```

Set executable bit:

```bash
chmod +x scripts/scenefunc3d/serve_sam2.sh
```

- [ ] **Step 5: Run tests and quality checks**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py -q
ruff check src/codex_agent/scenefunc3d/servers/sam2_mask_server.py src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py
black --check src/codex_agent/scenefunc3d/servers/sam2_mask_server.py src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py
mypy src/codex_agent/scenefunc3d/servers/sam2_mask_server.py src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py
```

Expected: all commands pass without importing `sam2` unless `OfficialSam2Runner` is instantiated.

- [ ] **Step 6: Commit**

```bash
git add \
  scripts/scenefunc3d/serve_sam2.sh \
  src/codex_agent/scenefunc3d/servers/sam2_mask_server.py \
  src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py
git commit -m "feat: add scenefunc3d sam2 sidecar server"
```

## Task 8: Single-Case Runtime Task, E2E CLI, And Health Checks

**Files:**
- Create: `scripts/scenefunc3d/check_sidecars.sh`
- Create: `scripts/scenefunc3d/run_single_case_e2e.sh`
- Modify: `src/codex_agent/scenefunc3d/runner.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_runner.py`
- Modify test: `src/codex_agent/tests/test_scenefunc3d_playbook.py`

- [ ] **Step 1: Write failing runner/task tests**

Create `src/codex_agent/tests/test_scenefunc3d_runner.py`:

```python
"""Tests for the SceneFunc3D single-case runner task."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.scenefunc3d.runner import (
    SceneFunc3dMaskOutcome,
    SceneFunc3dMaskTask,
    SceneFunc3dRunnerConfig,
    build_arg_parser,
    check_sidecar_health,
    run_single_sample,
)
from codex_agent.scenefunc3d.sample import SceneFunc3dSample, SceneFuncMotionHint
from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.tasks.base import CodexTask


class _FakeExecutor:
    def execute(
        self, task: CodexTask[SceneFunc3dMaskOutcome]
    ) -> CodexTaskResult[SceneFunc3dMaskOutcome]:
        final_response = json.dumps(
            {
                "mask_artifact_path": "/tmp/scenefunc/final.json",
                "mask_npz_path": "/tmp/scenefunc/final.npz",
                "mask_ply_path": "/tmp/scenefunc/final.ply",
                "selected_frame_ids": ["000050"],
                "accepted_fragment_ids": ["000050_mask_00"],
                "confidence": 0.8,
                "uncertainties": [],
            }
        )
        outcome = task.parse_response(final_response)
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=outcome,
            turn=CodexTurnResult(
                final_response=final_response,
                metadata=CodexTurnMetadata(turn_id="turn-1"),
            ),
        )


def test_runner_arg_parser_accepts_single_case_arguments() -> None:
    args = build_arg_parser().parse_args(
        [
            "--dataset-root",
            "/data/SceneFuncVal-CG",
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            "/tmp/scenefunc3d_backends.toml",
            "--output-dir",
            "/tmp/scenefunc3d_e2e",
        ]
    )

    assert args.sample_id == "421254::desc-a"
    assert args.backend_config == Path("/tmp/scenefunc3d_backends.toml")


def test_task_prompt_contains_tool_entrypoint_and_approval_flow(tmp_path: Path) -> None:
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=tmp_path / "421254",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    request = task.build_turn_request()

    assert "python -m codex_agent.scenefunc3d.tools <tool>" in request.prompt
    assert "--backend-config" in request.prompt
    assert "Molmo point" in request.prompt
    assert "SAM candidates" in request.prompt
    assert request.image_paths == ()
    assert request.skills == ()


def test_task_parses_strict_final_json(tmp_path: Path) -> None:
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=tmp_path / "421254",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    outcome = task.parse_response(
        json.dumps(
            {
                "mask_artifact_path": "/tmp/scenefunc/final.json",
                "mask_npz_path": "/tmp/scenefunc/final.npz",
                "mask_ply_path": "/tmp/scenefunc/final.ply",
                "selected_frame_ids": ["000050"],
                "accepted_fragment_ids": ["000050_mask_00"],
                "confidence": 0.8,
                "uncertainties": ["single view only"],
            }
        )
    )

    assert outcome.mask_npz_path == Path("/tmp/scenefunc/final.npz")
    assert outcome.selected_frame_ids == ("000050",)


def test_check_sidecar_health_passes_for_healthy_servers(tmp_path: Path) -> None:
    molmo = _start_server(
        {
            "/health": lambda payload: {
                "status": "ok",
                "model_name": "MolmoPoint-8B",
                "model_loaded": True,
            }
        }
    )
    sam = _start_server(
        {
            "/health": lambda payload: {
                "status": "ok",
                "model_name": "SAM2.1-Hiera-L",
                "model_loaded": True,
            }
        }
    )
    config_path = _write_backend_config(
        tmp_path,
        molmo_url=f"http://127.0.0.1:{molmo.server_port}",
        sam_url=f"http://127.0.0.1:{sam.server_port}",
    )
    try:
        check_sidecar_health(config_path)
    finally:
        molmo.shutdown()
        sam.shutdown()


def test_run_single_sample_writes_result(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    backend_config = _write_backend_config(
        tmp_path,
        molmo_url="http://127.0.0.1:8711",
        sam_url="http://127.0.0.1:8712",
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path,
        output_dir=tmp_path / "out",
        backend_config_path=backend_config,
    )

    result_path = run_single_sample(
        config,
        sample_id="421254::desc-a",
        executor=_FakeExecutor(),
        check_sidecars=False,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["outcome"]["mask_npz_path"] == "/tmp/scenefunc/final.npz"


def _sample() -> SceneFunc3dSample:
    return SceneFunc3dSample(
        sample_id="421254::desc-a",
        visit_id="421254",
        desc_id="desc-a",
        task_description="Open the lower drawer.",
        annotation_ids=("annot-a",),
        motion_hints=(
            SceneFuncMotionHint(
                motion_id="motion-a",
                annotation_id="annot-a",
                motion_type="trans",
                motion_dir=(1.0, 0.0, 0.0),
            ),
        ),
    )


def _start_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_json_handler(routes))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _write_backend_config(tmp_path: Path, *, molmo_url: str, sam_url: str) -> Path:
    output_root = tmp_path / "out"
    output_root.mkdir(exist_ok=True)
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
molmo_url = "{molmo_url}"
sam_url = "{sam_url}"
request_timeout_seconds = 2.0
artifact_staging_root = "{output_root}"
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{output_root}"]
""",
        encoding="utf-8",
    )
    return config_path


def _write_scene(root: Path) -> None:
    scene_dir = root / "421254"
    scene_dir.mkdir(parents=True)
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a"],
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps({"visit_id": "421254", "annotations": [{"annot_id": "annot-a"}]}),
        encoding="utf-8",
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_runner.py -q
```

Expected: fail with missing runner exports such as `SceneFunc3dMaskTask` and `build_arg_parser`.

- [ ] **Step 3: Implement strict runtime task and health check**

Modify `src/codex_agent/scenefunc3d/runner.py`:

```python
"""SceneFunc3D single-sample Codex runtime task and CLI."""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ..config import CodexAgentConfig
from ..errors import CodexResponseError
from ..json_extraction import extract_json_object
from ..models import CodexTurnRequest
from ..runtime import CodexAgentRuntime
from ..tasks.base import CodexExecutor
from .backends.config import load_backend_settings
from .playbook import SCENEFUNC3D_TOOLS_PLAYBOOK
from .sample import SceneFunc3dSample, load_sample, scene_dir_for, safe_sample_id
from .servers.schemas import HealthResponse

TASK_NAME = "scenefunc3d_mask_generation"
DEFAULT_TOOL_CLI_MODULE = "codex_agent.scenefunc3d.tools"


class SceneFunc3dMaskDecision(BaseModel):
    """Strict final JSON requested from the SceneFunc3D agent."""

    model_config = {"extra": "forbid"}

    mask_artifact_path: str = Field(min_length=1)
    mask_npz_path: str = Field(min_length=1)
    mask_ply_path: str = Field(min_length=1)
    selected_frame_ids: list[str] = Field(min_length=1)
    accepted_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: list[str]


@dataclass(frozen=True)
class SceneFunc3dMaskOutcome:
    """Parsed SceneFunc3D mask-generation result."""

    mask_artifact_path: Path
    mask_npz_path: Path
    mask_ply_path: Path
    selected_frame_ids: tuple[str, ...]
    accepted_fragment_ids: tuple[str, ...]
    confidence: float
    uncertainties: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        """Return JSON-ready outcome data."""
        return {
            "mask_artifact_path": str(self.mask_artifact_path),
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "selected_frame_ids": list(self.selected_frame_ids),
            "accepted_fragment_ids": list(self.accepted_fragment_ids),
            "confidence": self.confidence,
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class SceneFunc3dRunnerConfig:
    """Configuration for a SceneFunc3D single-sample run."""

    dataset_root: Path
    output_dir: Path
    backend_config_path: Path


class SceneFunc3dMaskTask:
    """A CodexTask for one SceneFunc3D mask-generation sample."""

    def __init__(
        self,
        *,
        sample: SceneFunc3dSample,
        scene_root: Path,
        output_dir: Path,
        backend_config_path: Path,
        tool_cli_module: str = DEFAULT_TOOL_CLI_MODULE,
    ) -> None:
        self.sample = sample
        self.scene_root = scene_root
        self.output_dir = output_dir
        self.backend_config_path = backend_config_path
        self.tool_cli_module = tool_cli_module

    @property
    def task_name(self) -> str:
        return TASK_NAME

    def build_turn_request(self) -> CodexTurnRequest:
        return CodexTurnRequest(
            prompt=self._build_prompt(),
            output_schema=SceneFunc3dMaskDecision.model_json_schema(),
            skills=(),
            image_paths=(),
        )

    def is_valid_response(self, response_text: str) -> bool:
        try:
            self._parse_decision(response_text)
        except CodexResponseError:
            return False
        return True

    def parse_response(self, response_text: str) -> SceneFunc3dMaskOutcome:
        decision = self._parse_decision(response_text)
        return SceneFunc3dMaskOutcome(
            mask_artifact_path=Path(decision.mask_artifact_path),
            mask_npz_path=Path(decision.mask_npz_path),
            mask_ply_path=Path(decision.mask_ply_path),
            selected_frame_ids=tuple(decision.selected_frame_ids),
            accepted_fragment_ids=tuple(decision.accepted_fragment_ids),
            confidence=decision.confidence,
            uncertainties=tuple(decision.uncertainties),
        )

    def _parse_decision(self, response_text: str) -> SceneFunc3dMaskDecision:
        payload = extract_json_object(response_text)
        try:
            return SceneFunc3dMaskDecision.model_validate(payload)
        except ValidationError as exc:
            raise CodexResponseError(
                f"Codex response does not match the SceneFunc3D mask schema: {exc}"
            ) from exc

    def _build_prompt(self) -> str:
        return build_prompt(self.sample) + self._tools_section()

    def _tools_section(self) -> str:
        return (
            "\nHow to run a tool (in the shell):\n"
            f"- scene_root: {self.scene_root}\n"
            f"- backend_config: {self.backend_config_path}\n"
            f"- out_dir: {self.output_dir}\n"
            f"- invoke: python -m {self.tool_cli_module} <tool> "
            f"--scene-root {self.scene_root} "
            f"--backend-config {self.backend_config_path} "
            f"--out-dir {self.output_dir} --args '<json>'\n"
            "\nYou are NOT exploring or editing a codebase. Use ONLY the "
            "SceneFunc3D tools above plus view_image. Do not read repository "
            "source files, docs, AGENTS.md, or SKILL.md during the task turn.\n"
            "\nOutput JSON schema:\n"
            + json.dumps(SceneFunc3dMaskDecision.model_json_schema(), ensure_ascii=False)
        )


def build_prompt(sample: SceneFunc3dSample) -> str:
    """Build the prompt prefix for one SceneFunc3D sample."""
    context_json = json.dumps(sample.agent_context, ensure_ascii=False, indent=2)
    return (
        f"{SCENEFUNC3D_TOOLS_PLAYBOOK}\n\n"
        "Task context:\n"
        f"{context_json}\n\n"
        "Generate a SceneFunc3D 3D mask artifact for this task."
    )


def load_runner_sample(
    config: SceneFunc3dRunnerConfig, sample_id: str
) -> SceneFunc3dSample:
    """Load the sample a runtime turn will solve."""
    _ = config.output_dir
    _ = config.backend_config_path
    return load_sample(config.dataset_root, sample_id)


def check_sidecar_health(backend_config_path: Path) -> None:
    """Fail before the agent turn if either configured sidecar is unhealthy."""
    settings = load_backend_settings(backend_config_path)
    for label, base_url in (("molmo", settings.molmo_url), ("sam", settings.sam_url)):
        url = base_url.rstrip("/") + "/health"
        with urllib.request.urlopen(url, timeout=settings.request_timeout_seconds) as response:
            raw_payload = json.loads(response.read().decode("utf-8"))
        health = HealthResponse.model_validate(raw_payload)
        if not health.model_loaded:
            raise RuntimeError(f"{label} sidecar is not loaded at {url}: {raw_payload}")


def run_single_sample(
    config: SceneFunc3dRunnerConfig,
    *,
    sample_id: str,
    executor: CodexExecutor,
    check_sidecars: bool = True,
) -> Path:
    """Run one SceneFunc3D sample and write a durable result JSON."""
    if check_sidecars:
        check_sidecar_health(config.backend_config_path)
    sample = load_runner_sample(config, sample_id)
    sample_output_dir = config.output_dir / safe_sample_id(sample_id)
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    task = SceneFunc3dMaskTask(
        sample=sample,
        scene_root=scene_dir_for(config.dataset_root, sample.visit_id),
        output_dir=sample_output_dir,
        backend_config_path=config.backend_config_path,
    )
    result = executor.execute(task)
    result_path = sample_output_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": result.task_name,
                "sample_id": sample_id,
                "outcome": result.outcome.to_payload(),
                "turn": result.turn.metadata.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_path
```

- [ ] **Step 4: Add runner CLI**

Append to `runner.py`:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the single-case SceneFunc3D runner parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-sidecar-health-check",
        action="store_true",
        help="Skip /health checks; intended only for unit tests with fake executors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one SceneFunc3D sample through CodexAgentRuntime."""
    args = build_arg_parser().parse_args(argv)
    result_path = run_single_sample(
        SceneFunc3dRunnerConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            backend_config_path=args.backend_config,
        ),
        sample_id=args.sample_id,
        executor=CodexAgentRuntime(CodexAgentConfig.from_env()),
        check_sidecars=not args.skip_sidecar_health_check,
    )
    print(json.dumps({"result_path": str(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TOOL_CLI_MODULE",
    "TASK_NAME",
    "SceneFunc3dMaskDecision",
    "SceneFunc3dMaskOutcome",
    "SceneFunc3dMaskTask",
    "SceneFunc3dRunnerConfig",
    "build_arg_parser",
    "build_prompt",
    "check_sidecar_health",
    "load_runner_sample",
    "main",
    "run_single_sample",
]
```

- [ ] **Step 5: Update existing playbook tests for runner config**

Modify each `SceneFunc3dRunnerConfig(...)` construction in `src/codex_agent/tests/test_scenefunc3d_playbook.py` to pass the new backend config path:

```python
config = SceneFunc3dRunnerConfig(
    dataset_root=tmp_path,
    output_dir=tmp_path / "out",
    backend_config_path=tmp_path / "backends.toml",
)
```

Keep the existing assertions for `build_prompt()` and `load_runner_sample()` unchanged.

- [ ] **Step 6: Add scripts**

Create `scripts/scenefunc3d/check_sidecars.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

molmo_url="${1:-http://127.0.0.1:8711}"
sam_url="${2:-http://127.0.0.1:8712}"

python - <<'PY' "$molmo_url" "$sam_url"
import json
import sys
import urllib.request

for url in (sys.argv[1], sys.argv[2]):
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "ok" or not payload.get("model_loaded"):
        raise SystemExit(f"unhealthy sidecar: {url}: {payload}")
    print(f"ok: {url}: {payload.get('model_name')}")
PY
```

Create `scripts/scenefunc3d/run_single_case_e2e.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m codex_agent.scenefunc3d.runner "$@"
```

Set executable bits:

```bash
chmod +x scripts/scenefunc3d/check_sidecars.sh scripts/scenefunc3d/run_single_case_e2e.sh
```

- [ ] **Step 7: Run tests and quality checks**

Run:

```bash
bash -n scripts/scenefunc3d/check_sidecars.sh
bash -n scripts/scenefunc3d/run_single_case_e2e.sh
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_runner.py \
  src/codex_agent/tests/test_scenefunc3d_playbook.py \
  -q
ruff check src/codex_agent/scenefunc3d/runner.py src/codex_agent/tests/test_scenefunc3d_runner.py
black --check src/codex_agent/scenefunc3d/runner.py src/codex_agent/tests/test_scenefunc3d_runner.py
mypy src/codex_agent/scenefunc3d/runner.py src/codex_agent/tests/test_scenefunc3d_runner.py
```

Expected: scripts parse; runner/playbook tests pass; lint, formatting, and typing pass.

- [ ] **Step 8: Commit**

```bash
git add \
  scripts/scenefunc3d/check_sidecars.sh \
  scripts/scenefunc3d/run_single_case_e2e.sh \
  src/codex_agent/scenefunc3d/runner.py \
  src/codex_agent/tests/test_scenefunc3d_runner.py
git commit -m "feat: add scenefunc3d single-case runtime runner"
```

## Task 9: Final Verification And Heavy Smoke Instructions

**Files:**
- Verify all files touched in Tasks 1-8.
- Create: `docs/benchmark/scenefunc_molmo_sam3d/sidecar_e2e_20260628.md`

- [ ] **Step 1: Run focused SceneFunc3D tests**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_backend_config.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  src/codex_agent/tests/test_scenefunc3d_http_client.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_lift3d.py \
  src/codex_agent/tests/test_scenefunc3d_sample.py \
  src/codex_agent/tests/test_scenefunc3d_state.py \
  src/codex_agent/tests/test_scenefunc3d_artifacts.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  src/codex_agent/tests/test_scenefunc3d_metrics.py \
  src/codex_agent/tests/test_scenefunc3d_playbook.py \
  -q
```

Expected: all focused SceneFunc3D tests pass.

- [ ] **Step 2: Run repository quality gate**

Run:

```bash
ruff check src/
black src/
mypy src/
PYTHONPATH=src /mlx_devbox/users/yueshuhao/miniforge3/envs/conceptgraph/bin/python -m pytest src/keyframe/tests -q
```

Expected:

- `ruff check src/` exits 0.
- `black src/` exits 0. If it reformats files, inspect and commit the formatting-only diff.
- `mypy src/` exits 0.
- keyframe tests pass in the conceptgraph env.

- [ ] **Step 3: Document heavy smoke command**

If Molmo/SAM model paths are available, run in tmux:

```bash
tmux new -s scenefunc-molmo
scripts/scenefunc3d/serve_molmo_point.sh \
  --port 8711 \
  --model-name MolmoPoint-8B \
  --model-path /path/to/MolmoPoint-8B \
  --device cuda:0

tmux new -s scenefunc-sam2
scripts/scenefunc3d/serve_sam2.sh \
  --port 8712 \
  --model-name SAM2.1-Hiera-L \
  --checkpoint-path /path/to/sam2.1_hiera_large.pt \
  --config-path /path/to/sam2.1_hiera_l.yaml \
  --staging-root /path/to/scenefunc3d_runs \
  --device cuda:0
```

Then:

```bash
scripts/scenefunc3d/check_sidecars.sh configs/scenefunc3d_backends.toml
```

Create `docs/benchmark/scenefunc_molmo_sam3d/sidecar_e2e_20260628.md` and record whether heavy smoke was run. If model paths are not available in the environment, write the exact missing paths/config keys.

- [ ] **Step 4: Commit benchmark status note**

Run:

```bash
mkdir -p docs/benchmark/scenefunc_molmo_sam3d
git add docs/benchmark/scenefunc_molmo_sam3d/sidecar_e2e_20260628.md
git commit -m "docs: record scenefunc3d sidecar e2e status"
```

- [ ] **Step 5: Final code review**

Dispatch a final code reviewer over the branch range:

```bash
BASE_SHA=$(git merge-base HEAD main)
HEAD_SHA=$(git rev-parse HEAD)
```

Ask the reviewer to prioritize:

- `src/codex_agent/scenefunc3d/backends/**`
- `src/codex_agent/scenefunc3d/servers/**`
- heavy optional import boundaries
- no fake success in model/lift paths
- default tests remain GPU-free

Fix Critical/Important findings before finishing.

## Task 10: Real MolmoPoint And SAM2.1 API Alignment

**Files:**
- Modify: `src/codex_agent/scenefunc3d/servers/molmo_point_server.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py`
- Modify: `src/codex_agent/scenefunc3d/servers/sam2_mask_server.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py`
- Modify if needed: `scripts/scenefunc3d/serve_sam2.sh`

- [ ] **Step 1: Add MolmoPoint-8B API regression test**

Add a fake Transformers module test proving `TransformersMolmoRunner` loads
`AutoModelForImageTextToText`, calls `processor.apply_chat_template(...,
return_pointing_metadata=True)`, calls `model.generate(...)`, post-processes raw
text, and does not require the old `generate_from_batch` API.

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py -q
```

Expected before implementation: fail because the current runner still requires
`AutoModelForCausalLM` / `generate_from_batch`.

- [ ] **Step 2: Implement true MolmoPoint runner path**

Update `TransformersMolmoRunner` so the real path follows the successful
`MolmoPoint-8B` batch script:

```text
AutoModelForImageTextToText.from_pretrained(local_model_path, trust_remote_code=True)
AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True, padding_side="left")
processor.apply_chat_template(..., return_pointing_metadata=True)
model.generate(..., logits_processor=model.build_logit_processor_from_inputs(...))
processor.post_process_image_text_to_text(...)
model.extract_image_points(...) as a non-artifact sanity check
```

Keep heavy imports lazy, CUDA fail-fast, raw-text-only response, and no server
artifact writes.

- [ ] **Step 3: Add transformers SAM2 backend regression test**

Add a fake Transformers test proving the SAM server CLI can select a
`transformers` backend and that the runner uses `Sam2Model` + `Sam2Processor`
to produce masks from point prompts.

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py -q
```

Expected before implementation: fail because the CLI only supports official
`--checkpoint-path` / `--config-path`.

- [ ] **Step 4: Implement SAM2 backend selection**

Add explicit `--backend official|transformers`. Keep the official backend
unchanged. For `transformers`, accept `--model-path` for local HF snapshots or
`--model-id` for model names, require `--staging-root`, run CUDA fail-fast, and
reuse the existing mask/scores validation and candidate NPZ writing helpers.

- [ ] **Step 5: Verify and commit**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py -q
ruff check src/codex_agent/scenefunc3d/servers/molmo_point_server.py src/codex_agent/scenefunc3d/servers/sam2_mask_server.py src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py
black --check src/codex_agent/scenefunc3d/servers/molmo_point_server.py src/codex_agent/scenefunc3d/servers/sam2_mask_server.py src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py
mypy src/codex_agent/scenefunc3d/servers/molmo_point_server.py src/codex_agent/scenefunc3d/servers/sam2_mask_server.py src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py
```

Expected: all focused checks pass without loading real model weights.

## External API References

- MolmoPoint server implementation should follow the actual
  `MolmoPoint-8B` Hugging Face API: `AutoModelForImageTextToText`,
  `AutoProcessor`, `apply_chat_template`, `generate`, and
  `extract_image_points`.
- SAM2 server implementation should support both the official
  `facebookresearch/sam2` image prediction API and the Hugging Face
  `Sam2Model` / `Sam2Processor` API used by `facebook/sam2.1-hiera-large`.
