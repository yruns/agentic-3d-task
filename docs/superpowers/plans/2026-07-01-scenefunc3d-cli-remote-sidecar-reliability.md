# SceneFunc3D CLI Remote Sidecar Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SceneFunc3D run through the same benchmark CLI shape as NR3D/OpenEQA, fail fast when Codex or sidecars are unavailable, and support the remote Molmo/SAM sidecar without leaking remote paths into local artifacts.

**Architecture:** Keep task-specific benchmark logic in the existing runners, but move shared Codex provider checks into `codex_agent.cli`. SceneFunc3D uses a per-sample tool context file so the agent no longer copies long filesystem paths, and the sidecar protocol accepts inline images plus local materialization of returned masks. Failure artifacts are written at the SceneFunc3D sample boundary and then propagated as real failures.

**Tech Stack:** Python 3.10+, stdlib `urllib`/`http.server`/`tomllib`, Pydantic v2, dataclasses, numpy for mask materialization, pytest, ruff, black, mypy.

---

## File Structure

- Create `src/codex_agent/cli/runtime_preflight.py`: shared Codex provider config parsing and local provider health/connectivity checks for benchmark CLIs.
- Modify `src/codex_agent/cli/run_nr3d.py`: call the shared runtime preflight after building `CodexAgentConfig` and before constructing `CodexAgentRuntime`.
- Modify `src/codex_agent/cli/run_openeqa.py`: same preflight wiring as NR3D.
- Modify `src/codex_agent/cli/run_scenefunc3d.py`: parse with the CLI wrapper parser and call a runner function that accepts parsed args, so the visible `prog` remains `codex_agent.cli.run_scenefunc3d`.
- Modify `src/codex_agent/scenefunc3d/runner.py`: remove `--skip-sidecar-health-check`, always check sidecars, write per-sample tool context, use context-only prompt command, and write `failure.json` when a Codex turn/response fails.
- Create `src/codex_agent/scenefunc3d/tool_context.py`: Pydantic model plus read/write helpers for `tool_context.json`.
- Modify `src/codex_agent/scenefunc3d/tools/__main__.py`: add `--context`, require explicit context or explicit legacy paths, and remove default scratch fallback when context is absent.
- Create `src/codex_agent/scenefunc3d/backends/image_payload.py`: local image-root validation, MIME/sha256/base64 encoding, and server-side inline image materialization helpers.
- Create `src/codex_agent/scenefunc3d/backends/mask_codec.py`: row-major boolean-mask RLE encode/decode helpers.
- Modify `src/codex_agent/scenefunc3d/servers/schemas.py`: allow `image_path` or `image` request input, add `InlineImagePayload`, `SamMaskRle`, and make SAM candidate mask content path/base64/RLE aware.
- Modify `src/codex_agent/scenefunc3d/backends/molmo_rpc.py`: send inline image payload for remote HTTPS backends.
- Modify `src/codex_agent/scenefunc3d/backends/sam_rpc.py`: send inline image payload for remote HTTPS backends and stop requiring a remote staging path for HTTPS.
- Modify `src/codex_agent/scenefunc3d/tools/sam_masking.py`: materialize SAM candidates from RLE/base64/local path into the sample output directory.
- Modify `src/codex_agent/scenefunc3d/servers/molmo_point_server.py`, `src/codex_agent/scenefunc3d/servers/sam2_mask_server.py`, and `src/codex_agent/scenefunc3d/servers/scenefunc_sidecar_server.py`: resolve inline image requests and return SAM masks by value.
- Create `src/codex_agent/scenefunc3d/failure_artifacts.py`: typed failure artifact payload and writer.
- Add or modify focused tests under `src/codex_agent/tests/`.
- Update SceneFunc3D benchmark docs/scripts only where they still show unsupported direct runner entrypoints or old tool command shapes.

---

### Task 1: Add Shared Codex Runtime Preflight

**Files:**
- Create: `src/codex_agent/cli/runtime_preflight.py`
- Create: `src/codex_agent/tests/test_runtime_preflight.py`

- [ ] **Step 1: Write failing tests for local provider health**

Create `src/codex_agent/tests/test_runtime_preflight.py` with tests that cover:

```python
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from codex_agent.cli.runtime_preflight import preflight_codex_runtime
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError


def test_preflight_checks_local_provider_health(tmp_path: Path) -> None:
    server = _start_health_server(status_code=200, body=b'{"status":"ok"}')
    codex_home = _write_codex_home(
        tmp_path, base_url=f"http://127.0.0.1:{server.server_port}/v1"
    )
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")
    try:
        result = preflight_codex_runtime(config, timeout_seconds=1.0)
    finally:
        server.shutdown()
        server.server_close()

    assert result.status == "checked"
    assert result.model_provider == "modelhub_adapter"
    assert result.health_url.endswith("/health")


def test_preflight_fails_before_codex_turn_when_local_provider_down(
    tmp_path: Path,
) -> None:
    server = _start_health_server(status_code=200, body=b'{"status":"ok"}')
    port = server.server_port
    server.shutdown()
    server.server_close()
    codex_home = _write_codex_home(tmp_path, base_url=f"http://127.0.0.1:{port}/v1")
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")

    with pytest.raises(CodexConfigError, match="Codex model provider is unavailable"):
        preflight_codex_runtime(config, timeout_seconds=0.2)


def test_preflight_skips_remote_provider_without_network_probe(tmp_path: Path) -> None:
    codex_home = _write_codex_home(
        tmp_path,
        base_url="https://example.invalid/api/modelhub/online/responses",
    )
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")

    result = preflight_codex_runtime(config, timeout_seconds=0.2)

    assert result.status == "skipped_remote"
    assert result.base_url.startswith("https://")


def test_preflight_reports_missing_provider_section(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model_provider = "modelhub_adapter"\n',
        encoding="utf-8",
    )
    config = CodexAgentConfig(codex_home=codex_home, model_provider="modelhub_adapter")

    with pytest.raises(CodexConfigError, match="model provider config is missing"):
        preflight_codex_runtime(config, timeout_seconds=0.2)


def _write_codex_home(tmp_path: Path, *, base_url: str) -> Path:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                'model = "gpt-5.4-2026-03-05"',
                'model_provider = "modelhub_adapter"',
                "",
                "[model_providers.modelhub_adapter]",
                'name = "ModelHub local adapter"',
                f"base_url = {json.dumps(base_url)}",
                'wire_api = "responses"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return codex_home


def _start_health_server(*, status_code: int, body: bytes) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/codex_agent/tests/test_runtime_preflight.py -q
```

Expected: import failure for `codex_agent.cli.runtime_preflight`.

- [ ] **Step 3: Implement `runtime_preflight.py`**

Implement:

- `CodexRuntimePreflightStatus = Literal["checked", "skipped_remote", "skipped_non_http"]`
- `CodexRuntimePreflightResult` as `@dataclass(frozen=True)` with `model_provider`, `base_url`, `health_url`, and `status`.
- A Pydantic boundary model for `config.toml` with `model_providers`.
- `preflight_codex_runtime(config: CodexAgentConfig, *, timeout_seconds: float = 1.0) -> CodexRuntimePreflightResult`.
- Local URL detection for `http://127.0.0.1` and `http://localhost`.
- Health URL derivation where `http://127.0.0.1:8787/v1` becomes `http://127.0.0.1:8787/health`.

Errors must be `CodexConfigError` and must not include headers, tokens, or full env values.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/codex_agent/tests/test_runtime_preflight.py -q
```

Expected: all tests in `test_runtime_preflight.py` pass.

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/cli/runtime_preflight.py src/codex_agent/tests/test_runtime_preflight.py
git commit -m "Add Codex runtime preflight"
```

---

### Task 2: Wire Benchmark CLIs And Remove SceneFunc3D Skip Health

**Files:**
- Modify: `src/codex_agent/cli/run_nr3d.py`
- Modify: `src/codex_agent/cli/run_openeqa.py`
- Modify: `src/codex_agent/cli/run_scenefunc3d.py`
- Modify: `src/codex_agent/scenefunc3d/runner.py`
- Modify: `src/codex_agent/tests/test_run_nr3d_cli.py`
- Modify: `src/codex_agent/tests/test_run_openeqa_cli.py`
- Modify: `src/codex_agent/tests/test_run_scenefunc3d_cli.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_runner.py`

- [ ] **Step 1: Write failing CLI wiring tests**

Add tests with these assertions:

- `run_nr3d.main()` calls `preflight_codex_runtime(config)` before constructing `CodexAgentRuntime`.
- `run_openeqa.main()` calls `preflight_codex_runtime(config)` before constructing `CodexAgentRuntime`.
- `run_scenefunc3d.build_arg_parser().format_usage()` contains `codex_agent.cli.run_scenefunc3d`.
- `run_scenefunc3d.main()` calls `runner.run_from_args(parsed_args)` instead of `runner.main(argv)`.
- `runner.build_arg_parser().format_help()` does not contain `--skip-sidecar-health-check`.
- `runner.main()` always passes `check_sidecars=True` to single-sample and batch paths.

For NR3D/OpenEQA tests, monkeypatch `preflight_codex_runtime`, `CodexAgentRuntime`, and the downstream run function so the test never starts a real Codex turn.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_run_nr3d_cli.py \
  src/codex_agent/tests/test_run_openeqa_cli.py \
  src/codex_agent/tests/test_run_scenefunc3d_cli.py \
  src/codex_agent/tests/test_scenefunc3d_runner.py \
  -q
```

Expected: failures showing missing preflight calls, SceneFunc3D still delegates to `runner.main`, and `--skip-sidecar-health-check` still exists.

- [ ] **Step 3: Update NR3D and OpenEQA**

In both CLI modules:

1. Import `preflight_codex_runtime` from `codex_agent.cli.runtime_preflight`.
2. Build the `CodexAgentConfig` as today.
3. Call `preflight_codex_runtime(config)`.
4. Construct `CodexAgentRuntime(config)` only after the preflight returns.

Do not add a CLI flag to skip the preflight.

- [ ] **Step 4: Refactor SceneFunc3D CLI parser flow**

Change `runner.build_arg_parser()` to accept a keyword-only `prog` argument:

```python
def build_arg_parser(*, prog: str = "codex_agent.scenefunc3d.runner") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run SceneFunc3D mask-generation cases with Codex.",
    )
    return parser
```

Add a parsed-args execution function:

```python
def run_from_args(args: argparse.Namespace) -> int:
    config = SceneFunc3dRunnerConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        backend_config_path=args.backend_config,
    )
    return _run_from_config_and_args(config, args)
```

Move the current body of `runner.main()` after `args = parser.parse_args(...)` into `run_from_args(args)`. If the implementation needs a helper, name it `_run_from_config_and_args(config: SceneFunc3dRunnerConfig, args: argparse.Namespace) -> int`.

Change `run_scenefunc3d.build_arg_parser()` to call:

```python
return runner.build_arg_parser(prog="codex_agent.cli.run_scenefunc3d")
```

Change `run_scenefunc3d.main()` to parse with that parser and call `runner.run_from_args(args)`.

- [ ] **Step 5: Remove SceneFunc3D skip health**

Remove the parser argument `--skip-sidecar-health-check`.

Remove the `not args.skip_sidecar_health_check` branches.

Single sample and batch calls should pass `check_sidecars=True` explicitly or rely on the default `True`.

- [ ] **Step 6: Preflight SceneFunc3D runtime construction**

Update `_build_executor()`:

```python
def _build_executor() -> CodexExecutor:
    from ..cli.runtime_preflight import preflight_codex_runtime
    from ..runtime import CodexAgentRuntime

    config = _tool_writable_runtime_config(CodexAgentConfig.from_env())
    preflight_codex_runtime(config)
    return CodexAgentRuntime(config)
```

- [ ] **Step 7: Run targeted tests and verify GREEN**

Run the same targeted pytest command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/codex_agent/cli/run_nr3d.py \
  src/codex_agent/cli/run_openeqa.py \
  src/codex_agent/cli/run_scenefunc3d.py \
  src/codex_agent/scenefunc3d/runner.py \
  src/codex_agent/tests/test_run_nr3d_cli.py \
  src/codex_agent/tests/test_run_openeqa_cli.py \
  src/codex_agent/tests/test_run_scenefunc3d_cli.py \
  src/codex_agent/tests/test_scenefunc3d_runner.py
git commit -m "Wire benchmark runtime preflight"
```

---

### Task 3: Add SceneFunc3D Tool Context

**Files:**
- Create: `src/codex_agent/scenefunc3d/tool_context.py`
- Modify: `src/codex_agent/scenefunc3d/runner.py`
- Modify: `src/codex_agent/scenefunc3d/tools/__main__.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_runner.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`

- [ ] **Step 1: Write failing tool context tests**

Add tests for:

- `SceneFunc3dMaskTask.build_turn_request().prompt` contains `--context <sample_output_dir>/tool_context.json`.
- The prompt no longer contains `--scene-root`, `--backend-config`, or `--out-dir` in the tool command.
- `run_single_sample()` writes `tool_context.json` before executing the Codex task.
- `codex_agent.scenefunc3d.tools --context <path>` loads `scene_root`, `backend_config_path`, and `out_dir` from the context.
- Missing context exits nonzero.
- A context whose `out_dir` is not under `allowed_output_roots` exits nonzero.
- Legacy direct tool invocation is still accepted only when `--scene-root` and `--out-dir` are explicit; there is no default scratch fallback.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_runner.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py \
  -q
```

Expected: failures showing `tool_context.py` is missing, prompt still uses explicit long paths, and CLI still has default scratch behavior.

- [ ] **Step 3: Implement `tool_context.py`**

Create a Pydantic v2 model:

- `sample_id: str`
- `scene_root: DirectoryPath`
- `backend_config_path: FilePath`
- `out_dir: Path`

Add a validator requiring `out_dir` to be absolute after `expanduser().resolve()`.

Add immutable dataclass `SceneFunc3dToolContext` with `sample_id`, `scene_root`, `backend_config_path`, and `out_dir`.

Add helpers:

- `write_tool_context(path: Path, context: SceneFunc3dToolContext) -> Path`
- `load_tool_context(path: Path) -> SceneFunc3dToolContext`

Both helpers raise `SceneFunc3dDataError` with path and error type when read/write/validation fails.

- [ ] **Step 4: Write context from runner**

In `run_single_sample()`:

1. Create `sample_output_dir`.
2. Initialize `events.jsonl`.
3. Build and write `tool_context.json`.
4. Construct `SceneFunc3dMaskTask` with `tool_context_path`.

Change `SceneFunc3dMaskTask.__init__()` to accept `tool_context_path: Path`.

Change `_build_prompt()` tool command to:

```bash
python -m codex_agent.scenefunc3d.tools <tool> --context <tool_context_path> --args '<json>'
```

Keep the runtime path section short and avoid repeating `scene_root`, `backend_config`, and `out_dir` as separate copyable command arguments.

- [ ] **Step 5: Update tools CLI context loading**

In `tools/__main__.py`:

- Add optional `--context`.
- Make `--scene-root` optional in argparse, but require it after parsing when `--context` is absent.
- Remove `_DEFAULT_OUT_DIR`.
- Require `--out-dir` when `--context` is absent.
- Reject combinations where `--context` is present together with `--scene-root`, `--backend-config`, or `--out-dir`.
- When context is present, load the backend config and validate `context.out_dir` with `ensure_path_under_roots`.
- Pass loaded `tool_scene`, `backend_config_path`, and `out_dir` into existing `run_tool()`.

- [ ] **Step 6: Run targeted tests and verify GREEN**

Run the same targeted pytest command from Step 2.

Expected: selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/tool_context.py \
  src/codex_agent/scenefunc3d/runner.py \
  src/codex_agent/scenefunc3d/tools/__main__.py \
  src/codex_agent/tests/test_scenefunc3d_runner.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py
git commit -m "Add SceneFunc3D tool context"
```

---

### Task 4: Add Inline Image Payload Support

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/image_payload.py`
- Modify: `src/codex_agent/scenefunc3d/servers/schemas.py`
- Modify: `src/codex_agent/scenefunc3d/backends/molmo_rpc.py`
- Modify: `src/codex_agent/scenefunc3d/backends/sam_rpc.py`
- Modify: `src/codex_agent/scenefunc3d/servers/molmo_point_server.py`
- Modify: `src/codex_agent/scenefunc3d/servers/sam2_mask_server.py`
- Modify: `src/codex_agent/scenefunc3d/servers/scenefunc_sidecar_server.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_server_schemas.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_sam_rpc.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_molmo_server_cli.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_sam2_server_cli.py`

- [ ] **Step 1: Write failing inline image tests**

Add tests for:

- `MolmoPointRequest` accepts exactly one of `image_path` or `image`.
- `SamMaskRequest` accepts exactly one of `image_path` or `image`.
- `image.sha256` must match decoded `data_base64`.
- For remote HTTPS backend URLs, `request_molmo_point()` sends `image` and does not send `image_path`.
- For remote HTTPS backend URLs, `request_sam_masks()` sends `image` and does not send `image_path`.
- For local HTTP backend URLs, existing `image_path` payload remains supported.
- Standalone and combined sidecar handlers can process inline image requests by materializing a temporary local image path for the existing runner call.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py \
  -q
```

Expected: schema validation and request-payload tests fail because `image` is not supported.

- [ ] **Step 3: Implement image payload helper**

In `image_payload.py`, implement:

- `InlineImagePayload` if the shared schema does not own it directly.
- `build_inline_image_payload(image_path: Path, *, allowed_roots: tuple[Path, ...]) -> InlineImagePayload`
- `materialize_inline_image(payload: InlineImagePayload, *, parent_dir: Path) -> Path`
- `is_remote_backend_url(url: str) -> bool`

Use `mimetypes.guess_type()` and allow only `image/jpeg`, `image/png`, and `image/webp`. Compute `sha256` over bytes and verify it on decode.

- [ ] **Step 4: Update schemas**

Add `InlineImagePayload` to `servers/schemas.py` with:

- `filename: NonEmptyString`
- `mime_type: Literal["image/jpeg", "image/png", "image/webp"]`
- `sha256: str` validated as 64 lowercase hex chars
- `data_base64: NonEmptyString`

Change `MolmoPointRequest` and `SamMaskRequest`:

- `image_path: Path | None = None`
- `image: InlineImagePayload | None = None`
- model validator requires exactly one.

Expose a small method or property that lets handlers resolve whether the request was path-based or inline-based without inspecting raw fields throughout the server modules.

- [ ] **Step 5: Update RPC clients**

In `molmo_rpc.py` and `sam_rpc.py`:

- Keep root validation with `ensure_path_under_roots`.
- If `is_remote_backend_url(settings.molmo_url)` or `is_remote_backend_url(settings.sam_url)`, send `"image": build_inline_image_payload(image_path, allowed_roots=settings.allowed_image_roots).model_dump(mode="json")`.
- If local HTTP, keep existing `"image_path": str(image_path)`.
- For SAM HTTPS, omit `"staging_dir"` from the request. For local HTTP, keep it for compatibility while server support is being migrated.

- [ ] **Step 6: Update sidecar handlers**

In each handler:

- Validate the request schema.
- If the request has `image_path`, call the existing runner path.
- If the request has `image`, materialize it under a `tempfile.TemporaryDirectory()` and call the existing runner with a path-backed request object.
- Do not persist inline image files after the handler returns.

- [ ] **Step 7: Run targeted tests and verify GREEN**

Run the same targeted pytest command from Step 2.

Expected: selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/backends/image_payload.py \
  src/codex_agent/scenefunc3d/servers/schemas.py \
  src/codex_agent/scenefunc3d/backends/molmo_rpc.py \
  src/codex_agent/scenefunc3d/backends/sam_rpc.py \
  src/codex_agent/scenefunc3d/servers/molmo_point_server.py \
  src/codex_agent/scenefunc3d/servers/sam2_mask_server.py \
  src/codex_agent/scenefunc3d/servers/scenefunc_sidecar_server.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py
git commit -m "Support inline sidecar images"
```

---

### Task 5: Return And Materialize SAM Masks By Value

**Files:**
- Create: `src/codex_agent/scenefunc3d/backends/mask_codec.py`
- Modify: `src/codex_agent/scenefunc3d/servers/schemas.py`
- Modify: `src/codex_agent/scenefunc3d/tools/sam_masking.py`
- Modify: `src/codex_agent/scenefunc3d/servers/sam2_mask_server.py`
- Modify: `src/codex_agent/scenefunc3d/servers/scenefunc_sidecar_server.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_server_schemas.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_sam_rpc.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`

- [ ] **Step 1: Write failing RLE tests**

Add tests for:

- `encode_bool_mask_rle()` and `decode_bool_mask_rle()` round-trip a 2D bool numpy mask.
- RLE decode rejects negative counts, wrong total pixel count, and non-2D shapes.
- `SamMaskCandidateResponse` accepts `mask_rle` without `mask_npz_path`.
- `sam_mask()` writes a local `.npz` under `out_dir` when the response contains RLE.
- Remote-looking `mask_npz_path` is not accepted unless inline base64 or RLE is present.
- Server responses for SAM include `mask_rle` and omit remote-only paths in the primary payload.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  -q
```

Expected: failures showing `mask_codec.py` is missing and `SamMaskCandidateResponse` still requires path-like mask payloads.

- [ ] **Step 3: Implement row-major RLE codec**

In `mask_codec.py`, implement:

- `MaskRlePayload` as `@dataclass(frozen=True)` with `height`, `width`, and `counts: tuple[int, ...]`.
- `encode_bool_mask_rle(mask: npt.NDArray[np.bool_]) -> MaskRlePayload`.
- `decode_bool_mask_rle(payload: MaskRlePayload) -> npt.NDArray[np.bool_]`.

Encoding starts with the count of `False` pixels in row-major order, alternates `False` and `True`, and requires the counts sum to `height * width`.

- [ ] **Step 4: Update SAM candidate schema**

Add Pydantic schema `SamMaskRle`:

- `encoding: Literal["row_major_counts"]`
- `height: int = Field(gt=0, strict=True)`
- `width: int = Field(gt=0, strict=True)`
- `counts: tuple[int, ...] = Field(min_length=1)`

Change `SamMaskCandidateResponse`:

- `mask_npz_path: Path | None = None`
- `mask_rle: SamMaskRle | None = None`
- keep `mask_npz_base64` and `mask_npz_sha256` as compatibility fields.

Add a model validator requiring at least one of `mask_rle`, `mask_npz_base64`, or `mask_npz_path`.

- [ ] **Step 5: Update local materialization**

In `sam_masking.py`, update `_resolve_candidate_mask_npz_path()`:

1. If `candidate_response.mask_rle` is present, decode it and write `staging_dir / f"{safe_candidate_id}.npz"`.
2. Else if `mask_npz_base64` is present, keep existing base64 materialization.
3. Else if `mask_npz_path` is a readable local file under allowed output roots, keep the compatibility path.
4. Else raise `ToolInputError` with candidate id and payload type summary.

- [ ] **Step 6: Update SAM sidecar responses**

In standalone and combined SAM handlers:

- Convert runner-produced local mask NPZ candidates into RLE response candidates before returning HTTP JSON.
- Keep `pixel_count`, `coverage_percent`, `candidate_id`, and `score`.
- Do not rely on client access to server-local files.

- [ ] **Step 7: Run targeted tests and verify GREEN**

Run the same targeted pytest command from Step 2.

Expected: selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/backends/mask_codec.py \
  src/codex_agent/scenefunc3d/servers/schemas.py \
  src/codex_agent/scenefunc3d/tools/sam_masking.py \
  src/codex_agent/scenefunc3d/servers/sam2_mask_server.py \
  src/codex_agent/scenefunc3d/servers/scenefunc_sidecar_server.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
git commit -m "Materialize SAM masks from value payloads"
```

---

### Task 6: Write Structured Failure Artifacts

**Files:**
- Create: `src/codex_agent/scenefunc3d/failure_artifacts.py`
- Modify: `src/codex_agent/scenefunc3d/runner.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_runner.py`

- [ ] **Step 1: Write failing failure-artifact tests**

Add tests for:

- When executor raises `CodexTurnError`, `run_single_sample()` writes `<sample_output_dir>/failure.json` and re-raises.
- When final response validation raises `CodexResponseError`, `run_single_sample()` writes `failure.json` and re-raises.
- `failure.json` includes `task_name`, `sample_id`, `status="failed"`, `failure_stage`, `error_type`, `error_message`, `events_path`, `tool_context_path`, and a `turn` object with nullable metadata fields.
- Batch mode records the failed sample and points to `failure.json` if the failure happened after sample output initialization.
- Single-sample CLI returns nonzero for these failures through the raised exception path; it must not print a success payload.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_runner.py -q
```

Expected: failures showing `failure.json` is not written.

- [ ] **Step 3: Implement failure artifact model and writer**

In `failure_artifacts.py`, implement:

- `SceneFunc3dFailureStage = Literal["codex_turn", "response_validation", "run"]`
- `SceneFunc3dFailureTurnPayload` as a Pydantic model mirroring nullable `CodexTurnMetadataPayload`.
- `SceneFunc3dFailureArtifact` as a Pydantic model with `status: Literal["failed"]`.
- `write_failure_artifact(path: Path, artifact: SceneFunc3dFailureArtifact) -> Path`.

`write_failure_artifact()` creates parents, writes UTF-8 JSON with a trailing newline, and raises `SceneFunc3dDataError` on `OSError`.

- [ ] **Step 4: Integrate failure writes in `run_single_sample()`**

After sample loading and output directory initialization, wrap the Codex execution, response parsing, and outcome validation path:

- `CodexTurnError` maps to `failure_stage="codex_turn"`.
- `CodexResponseError` maps to `failure_stage="response_validation"`.
- Other `CodexAgentError` maps to `failure_stage="run"`.

Write `failure.json`, then re-raise the original exception.

- [ ] **Step 5: Include failure path in batch summary**

Extend `SceneFunc3dBatchSampleResult` and `SceneFunc3dBatchSampleResultPayload` with:

```python
failure_path: str | None = None
```

When a sample fails and `<sample_output_dir>/failure.json` exists, include that path in the summary. Keep `result_path=None` for failed runs.

- [ ] **Step 6: Run targeted tests and verify GREEN**

Run the same targeted pytest command from Step 2.

Expected: selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add \
  src/codex_agent/scenefunc3d/failure_artifacts.py \
  src/codex_agent/scenefunc3d/runner.py \
  src/codex_agent/tests/test_scenefunc3d_runner.py
git commit -m "Write SceneFunc3D failure artifacts"
```

---

### Task 7: Align Docs, Scripts, And End-To-End Checks

**Files:**
- Modify: `docs/benchmark/scenefunc_molmo_sam3d/README.md`
- Modify: `docs/benchmark/scenefunc_molmo_sam3d/assets/run_agent_runner_e2e_20260629.sh`
- Modify: `docs/benchmark/scenefunc_molmo_sam3d/assets/run_sidecar_tool_smoke_20260629.sh`
- Modify tests for the scripts if assertions drift.

- [ ] **Step 1: Update docs and scripts**

Ensure SceneFunc3D docs/scripts show only:

```bash
PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d
```

Do not document `codex_agent.scenefunc3d.runner` as a benchmark entrypoint.

Ensure script output roots remain under repo `tmp/scenefunc3d/`.

- [ ] **Step 2: Run focused docs/script tests**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_run_scenefunc3d_cli.py \
  src/codex_agent/tests/test_scenefunc3d_agent_e2e_script.py \
  src/codex_agent/tests/test_scenefunc3d_sidecar_smoke_script.py \
  -q
```

Expected: selected tests pass.

- [ ] **Step 3: Run mandatory quality gate**

Run:

```bash
source .venv/bin/activate
ruff check src/
black --check src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

Expected:

- `ruff check src/` exits 0.
- `black --check src/` exits 0.
- `mypy src/` exits 0.
- keyframe tests pass.

- [ ] **Step 4: Run focused Codex Agent tests**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_runtime_preflight.py \
  src/codex_agent/tests/test_run_nr3d_cli.py \
  src/codex_agent/tests/test_run_openeqa_cli.py \
  src/codex_agent/tests/test_run_scenefunc3d_cli.py \
  src/codex_agent/tests/test_scenefunc3d_runner.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py \
  -q
```

Expected: selected tests pass.

- [ ] **Step 5: Run local help smoke**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d --help
```

Expected:

- Usage begins with `usage: codex_agent.cli.run_scenefunc3d`.
- Help output does not contain `--skip-sidecar-health-check`.

- [ ] **Step 6: Run remote sidecar smoke in tmux**

Because a real Codex agent turn can run for minutes, start it in tmux:

```bash
tmux new-session -d -s scenefunc_remote_smoke_20260701 \
  'cd /Users/bytedance/project/agentic-3d-task && \
   source .venv/bin/activate && \
   PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d \
     --dataset-root data/SceneFun3D \
     --sample-id 421254::6afc3b66-ea3e-4b8e-b630-ca6174da502a \
     --backend-config configs/scenefunc3d_backends.toml \
     --output-dir tmp/scenefunc3d/artifacts/remote_smoke_20260701 \
     --score \
     2>&1 | tee tmp/scenefunc3d/remote_smoke_20260701.log'
```

Expected:

- If Codex adapter, remote sidecar, and private headers are available, the run reaches Molmo/SAM and writes either `result.json` or `failure.json` under the sample output directory.
- If an external service is unavailable, the CLI fails before a Codex turn for preflight/health errors, or leaves `failure.json` for turn/response failures.

- [ ] **Step 7: Commit docs and final adjustments**

```bash
git add \
  docs/benchmark/scenefunc_molmo_sam3d/README.md \
  docs/benchmark/scenefunc_molmo_sam3d/assets/run_agent_runner_e2e_20260629.sh \
  docs/benchmark/scenefunc_molmo_sam3d/assets/run_sidecar_tool_smoke_20260629.sh \
  src/codex_agent/tests/test_run_scenefunc3d_cli.py \
  src/codex_agent/tests/test_scenefunc3d_agent_e2e_script.py \
  src/codex_agent/tests/test_scenefunc3d_sidecar_smoke_script.py
git commit -m "Align SceneFunc3D benchmark entrypoints"
```

---

## Final Acceptance

- [ ] `PYTHONPATH=src python -m codex_agent.cli.run_nr3d --help` works.
- [ ] `PYTHONPATH=src python -m codex_agent.cli.run_openeqa --help` works.
- [ ] `PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d --help` works and shows `codex_agent.cli.run_scenefunc3d`.
- [ ] SceneFunc3D help output does not include `--skip-sidecar-health-check`.
- [ ] SceneFunc3D prompt uses `--context <tool_context.json>` and does not ask the agent to copy `scene_root`, `backend_config`, or `out_dir` into the tool command.
- [ ] Remote HTTPS sidecar requests send inline image payloads.
- [ ] SAM candidate masks are materialized locally from RLE/base64/path compatibility input under the sample `out_dir`.
- [ ] Failed SceneFunc3D turns write `failure.json` and do not produce fake success payloads.
- [ ] Mandatory gate passes:

```bash
source .venv/bin/activate
ruff check src/
black --check src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

- [ ] Focused Codex Agent tests pass:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_runtime_preflight.py \
  src/codex_agent/tests/test_run_nr3d_cli.py \
  src/codex_agent/tests/test_run_openeqa_cli.py \
  src/codex_agent/tests/test_run_scenefunc3d_cli.py \
  src/codex_agent/tests/test_scenefunc3d_runner.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py \
  src/codex_agent/tests/test_scenefunc3d_server_schemas.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_sam_rpc.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py \
  -q
```
