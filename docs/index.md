# Agentic 3D Task Docs

This site is the human-readable view over the Markdown files in `docs/`.
The Markdown remains the source of truth, so agents can keep reading the same
files directly while people browse the rendered MkDocs Material site.

## Reading Route

- [Linux agentic3d Environment](agentic3d_environment.md): shared model/source
  storage, Conda activation, native model services, and verification records.
- [Local Dataset Locations](dataset_locations.md): canonical NR3D/OpenEQA/ScanNet
  paths on this machine and the verified NR3D centralized-copy record.
- [Python Code Agent Quality Guide](python_code_agent_quality_guide.md): coding,
  testing, typing, and review standards for Python changes in this repository.
- [Codex Agent](codex_agent/index.md): engineering notes, design records, and
  runtime investigations for Codex-based 3D reasoning agents.
- [Research Reports](research/index.md): literature reviews, benchmark protocol
  audits, and research planning for indoor 3D embodied tasks.
- [Benchmarks](benchmark/index.md): durable evaluation records for NR3D and
  OpenEQA, including timelines, version notes, and preserved artifacts.
- [Latest Handoff](handoff_2026-07-02_1313.md): current SceneFunc3D/Codex
  Agent SDK state, risks, metrics, and next steps for agent continuity.

## Local Preview

Install the development dependencies and run the MkDocs server from the
repository root:

```bash
uv pip install -e ".[dev]"
mkdocs serve
```

For a static build, run:

```bash
mkdocs build
```
