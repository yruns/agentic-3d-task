# MkDocs Material Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a MkDocs Material site that renders the existing `docs/` Markdown as a chaptered human-readable documentation site.

**Architecture:** Keep Markdown as the only content source. Add root tooling configuration in `mkdocs.yml`, add three lightweight routing pages under `docs/`, and keep existing benchmark records and engineering notes in place. Exclude `docs/superpowers/**` from the generated site because those files are internal workflow artifacts, not project documentation.

**Tech Stack:** MkDocs, MkDocs Material, existing Markdown documents, `pyproject.toml` development dependencies.

---

### Task 1: Add MkDocs Tooling Configuration

**Files:**
- Create: `mkdocs.yml`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [x] **Step 1: Add the dev dependency**

Add `mkdocs-material>=9.5.0` to the `[project.optional-dependencies].dev` list in `pyproject.toml`, directly after `black>=24.0.0`.

- [x] **Step 2: Create `mkdocs.yml`**

Create root-level `mkdocs.yml` with `docs_dir: docs`, Material theme settings, Markdown extensions, explicit chapter navigation, and `exclude_docs: superpowers/**`.

- [x] **Step 3: Ignore generated site output**

Add `site/` to `.gitignore` so `mkdocs build` output remains local.

- [x] **Step 4: Verify the config file can be parsed**

Run: `python -c "import yaml; yaml.safe_load(open('mkdocs.yml', encoding='utf-8'))"`

Expected: command exits 0 and prints no output.

### Task 2: Add Chapter Index Pages

**Files:**
- Create: `docs/index.md`
- Create: `docs/codex_agent/index.md`
- Create: `docs/benchmark/index.md`

- [x] **Step 1: Create the documentation homepage**

Create `docs/index.md` with links to development standards, Codex Agent notes, and benchmark archive pages.

- [x] **Step 2: Create the Codex Agent chapter index**

Create `docs/codex_agent/index.md` with short descriptions and links for each existing Codex Agent engineering note.

- [x] **Step 3: Create the benchmark chapter index**

Create `docs/benchmark/index.md` with links to NR3D and OpenEQA timeline pages and a note that benchmark assets remain beside their records.

### Task 3: Build And Fix MkDocs Site

**Files:**
- Modify only files touched by Task 1 or Task 2 if build output exposes issues.

- [x] **Step 1: Install documentation tooling if needed**

Run: `python -m mkdocs --version`

Expected: if the module exists, it prints a version. If it is missing, run `uv pip install -e ".[dev]"`.

- [x] **Step 2: Build the site**

Run: `mkdocs build`

Expected: build exits 0 and writes the static site to `site/`.

- [x] **Step 3: Rebuild after any fixes**

If `mkdocs build` reports broken config or missing nav files, edit only the MkDocs config or new index pages, then rerun `mkdocs build`.

Expected: final build exits 0.

- [x] **Step 4: Inspect changed files**

Run: `git diff --stat` and `git diff -- mkdocs.yml pyproject.toml .gitignore docs/index.md docs/codex_agent/index.md docs/benchmark/index.md`.

Expected: changes are limited to MkDocs configuration, the new chapter index pages, the dev dependency, and ignoring generated site output.
