# AGENTS.md

Operating rules for any AI coding agent working in this repository.

This file is binding. Read it before writing or changing code. It works
together with:

- [`docs/python_code_agent_quality_guide.md`](docs/python_code_agent_quality_guide.md)
  — the **mandatory Python coding standard** for this project.
- [`CLAUDE.md`](CLAUDE.md) — environment, package management, tmux for
  long-running jobs, GPU rules, and benchmark process documentation.

## 1. Python coding standard (MANDATORY)

All Python code produced or modified in this repo **must** comply with
[`docs/python_code_agent_quality_guide.md`](docs/python_code_agent_quality_guide.md).
Treat that document as the source of truth for naming, structure, function/class
design, typing, input validation, exceptions, logging, configuration, and tests.

Every code change must be **strictly checked against that standard before you
claim it is done.** Self-review against the guide's §15 review checklist is not
optional.

### Non-negotiable rules (enforced here)

These mirror the guide's §13 ban list and this repo's existing conventions:

- **Strong typing.** Every function/method parameter and return value is fully
  type-annotated. No `Any`. No bare `dict` / `Dict` / `list` to model a real
  object — use `dataclass(frozen=True)`, `TypedDict`, `Enum`/`Literal`, or a
  Pydantic v2 model. Keep `Any` and untyped `dict` only at deserialization
  boundaries, and convert to a concrete type immediately.
- **Minimize `Optional` parameters.** Prefer empty collections / default
  objects over `None`; keep `T | None` only where "absent" is a real domain
  state.
- No mutable default arguments. No bare `except:`. No swallowing exceptions
  into fake success.
- No business logic inside CLI `main` / API handlers / ORM models.
- No global implicit initialization of heavy clients; inject dependencies.
- No scattered `os.environ` reads in business code; centralize config.
- No catch-all `utils.py`; name modules by capability/domain.
- Never treat external input (HTTP, CLI, env, files, DB, MQ, third-party API,
  LLM/tool output) as trusted — validate at the boundary (Pydantic v2 / argparse).
- Never log secrets, tokens, passwords, or full private data.
- Never claim completion based on manual inspection alone — run the gate below.

## 2. Mandatory quality gate (run before claiming done)

Activate the project environment first (see `CLAUDE.md` §Package Management):

```bash
# macOS (Darwin): uv + .venv
source .venv/bin/activate

# Linux (Ubuntu): conda
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
```

Then run, and confirm each passes, on the code you touched:

```bash
ruff check src/                       # lint + import order (guide §11: `ruff check`)
black src/                            # formatting (this repo's formatter; guide's `ruff format` maps to black here)
mypy src/                             # static type check; must report no issues
PYTHONPATH=src pytest src/keyframe/tests -q   # tests must pass
```

Notes:

- This project formats with **black** (configured in `pyproject.toml` and
  `CLAUDE.md`), not `ruff format`. Use `black --check src/` in CI-style checks.
- Tests need `PYTHONPATH=src` (or the configured `pythonpath = ["src"]`).
- If a tool or test genuinely cannot run, say so explicitly and state the risk
  — do not silently skip it.

## 3. Workflow

Follow the guide's §12 workflow:

1. **Before coding** — understand the existing structure, dependencies, tests,
   and the naming / error-handling / layering style already in the package you
   are editing. Keep changes small and focused; do not opportunistically
   refactor unrelated modules.
2. **While coding** — prefer adding/adjusting a minimal regression test first
   (explain if TDD does not fit); reuse existing helpers and abstractions; do
   not add dependencies unless clearly justified.
3. **After coding** — run the §2 gate, then report: which files changed and
   why, which verification commands ran and their result, anything not run and
   why, and remaining risks or follow-ups.

## 4. Repository-specific reminders

- **Secrets:** real LLM endpoints/keys live in `configs/llm.toml`, which is
  gitignored. Keep `configs/llm.example.toml` (placeholders) tracked. Never
  commit real AKs or paste them into docs, logs, or commits (see `CLAUDE.md`).
- **Optional dependencies:** heavy/vision deps (`opencv-python`, `pillow`,
  `plyfile`, `torch`, `open-clip-torch`) are optional extras. Keep
  `import keyframe` working without them — lazy-import optional deps inside the
  functions that need them.
- **Long-running tasks** (training, evaluation, batch processing, large test
  suites, data prep) must run inside `tmux` — see `CLAUDE.md` §Long-Running
  Tasks.
- **ModelHub adapter preflight:** before starting project workflows that use
  the Codex Agent SDK or benchmark CLIs, first check the local adapter health:
  `curl -sf http://127.0.0.1:8787/health`. If the service is not listening or
  the health check fails, start it from the repo-local adapter directory before
  launching the run:

  ```bash
  cd codex_modelhub_adapter
  python3 -m uvicorn adapter.app:app --host 127.0.0.1 --port 8787
  ```

  For long benchmark runs, keep this adapter process in `tmux`. Do not proceed
  to `codex_agent.cli.run_*` commands while this health check is down; runtime
  preflight will fail before a Codex turn. Never commit adapter `.env` or
  upstream TOML files.
- **Documentation organization:** project documentation under `docs/` is managed
  as a MkDocs Material site (`mkdocs.yml`) while keeping Markdown as the single
  source of truth. Agents should continue reading/editing the source `.md` files
  directly; humans can browse the rendered site. When adding or reorganizing
  `docs/` content, update the relevant chapter index page and `mkdocs.yml`
  navigation in the same change. Top-level `README.md`, `AGENTS.md`, and
  `CLAUDE.md` are operational entrypoints and are intentionally not part of the
  MkDocs navigation.
- **Benchmark runs** must leave a durable process record under
  `docs/benchmark/<name>/` — see `CLAUDE.md` §Benchmark Process Documentation.
