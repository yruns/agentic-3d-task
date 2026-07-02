# SceneFunc3D Native Sidecar Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-native single-process SceneFunc3D HTTP server that listens on one port and serves both MolmoPoint and SAM2 endpoints.

**Architecture:** Keep the existing Molmo and SAM runner implementations as the inference boundary. Add a new combined server module that constructs both runners in one process, exposes aggregate and component health endpoints, and routes point/mask POST requests without a separate gateway process.

**Tech Stack:** Python stdlib `http.server`, existing Pydantic request/response schemas, existing `TransformersMolmoRunner`, `OfficialSam2Runner`, and `TransformersSam2Runner`.

---

### Task 1: Add Combined Server Tests

**Files:**
- Create: `src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py`

- [ ] **Step 1: Write failing route tests**

Create tests that instantiate fake Molmo and SAM runners, call the combined route builder through `make_json_handler`, and assert:
- `GET /health` returns aggregate component status.
- `GET /molmo/health` returns Molmo health.
- `GET /sam/health` returns SAM health.
- `POST /molmo/v1/point` returns one parsed point and latency.
- `POST /sam/v1/masks` returns one mask candidate and latency.
- CUDA OOM-like runner failures map to HTTP 503.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py -q
```

Expected: import failure because `codex_agent.scenefunc3d.servers.scenefunc_sidecar_server` does not exist.

### Task 2: Implement Combined Server

**Files:**
- Create: `src/codex_agent/scenefunc3d/servers/scenefunc_sidecar_server.py`

- [ ] **Step 1: Add dataclasses and route builder**

Create an immutable `SceneFuncSidecarRunners` dataclass containing `MolmoRunner` and `Sam2Runner`. Add `build_routes(runners)` returning routes for `/health`, `/molmo/health`, `/sam/health`, `/molmo/v1/point`, and `/sam/v1/masks`.

- [ ] **Step 2: Add request handlers**

Validate requests using existing Pydantic schemas. Build existing response schemas with measured `latency_ms`. Convert validation errors to `JsonHttpError(400, ...)`, CUDA resource failures to `JsonHttpError(503, ...)`, and SAM artifact/image/staging errors to the same stable errors as the standalone SAM server.

- [ ] **Step 3: Add CLI parser and runner construction**

Support one process with:

```bash
python -m codex_agent.scenefunc3d.servers.scenefunc_sidecar_server \
  --host :: \
  --port 9001 \
  --device cuda:0 \
  --molmo-model-path /path/to/MolmoPoint-8B \
  --sam-backend transformers \
  --sam-model-path /path/to/SAM2.1-Hiera-L \
  --staging-root /path/to/output
```

Also support official SAM with `--sam-checkpoint-path` and `--sam-config-path`.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run the same targeted pytest command. Expected: all tests pass.

### Task 3: Add Startup Documentation

**Files:**
- Modify: `docs/benchmark/scenefunc_molmo_sam3d/v3_agent_loop_status_20260630.md`

- [ ] **Step 1: Document native server command**

Add a short note that 9001 should now be served by the repo-native single-process server, with the current MolmoPoint and SAM2.1 paths.

- [ ] **Step 2: Run focused documentation-free checks**

No doc build is required for this operational note. Verify source formatting and tests.

### Task 4: Quality Gate

- [ ] **Step 1: Run formatting**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
black src/
```

- [ ] **Step 2: Run lint**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
ruff check src/
```

- [ ] **Step 3: Run type check**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
mypy src/
```

- [ ] **Step 4: Run mandatory keyframe tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
PYTHONPATH=src pytest src/keyframe/tests -q
```

- [ ] **Step 5: Run new focused tests**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_native_sidecar_server.py -q
```
