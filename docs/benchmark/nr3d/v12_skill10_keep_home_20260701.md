# v12 Skill10 Keep Run Home (2026-07-01)

Diagnostic rerun of the first 10 strat600 NR3D samples in external Skill mode,
with `CODEX_AGENT_KEEP_RUN_HOME=1` enabled so each Codex SDK app-server
`CODEX_HOME` copy remains available under `.codex-home/runs/<id>`.

> Raw artifacts: `tmp/nr3d_tools_skill10_keep_home_20260701/`.

## Result

| Metric | Value |
|---|---:|
| Completed | 10/10 |
| Acc@0.25 | 80.00% |
| Acc@0.50 | 80.00% |
| Mean IoU | 0.8000 |
| Cache hit rate | 0.9000 |
| Mean cache ratio | 0.8733 |

Wrong samples matched the earlier first-10 Skill smoke set:

- `scannet/scene0500_00::25::13471`: selected `26`, IoU `0.0`.
- `scannet/scene0608_00::9::17679`: selected `8`, IoU `0.0`.

## Run Homes

Base Codex home: `.codex-home`.

| Sample | Selected | IoU | Run home |
|---|---:|---:|---|
| `scannet/scene0608_00::9::17679` | 8 | 0.0 | `.codex-home/runs/27d0f2797cd34be18b90cb31ea5fc257` |
| `scannet/scene0207_00::20::12189` | 20 | 1.0 | `.codex-home/runs/37ab6613ffc44d92b525f006c3439eff` |
| `scannet/scene0246_00::5::12312` | 5 | 1.0 | `.codex-home/runs/5ded491b5bb846e2afdf253ba26aa8ba` |
| `scannet/scene0426_00::22::9868` | 22 | 1.0 | `.codex-home/runs/5fde4bf94aed4656944e45bee120dff2` |
| `scannet/scene0591_00::14::19171` | 14 | 1.0 | `.codex-home/runs/7c5facd387a84374a5289139f88e6425` |
| `scannet/scene0221_00::47::4274` | 47 | 1.0 | `.codex-home/runs/cd8ad373b5394f6b80761c1392b69cbd` |
| `scannet/scene0699_00::26::40486` | 26 | 1.0 | `.codex-home/runs/cfde09603977411b8675f8abca238cec` |
| `scannet/scene0500_00::25::13471` | 26 | 0.0 | `.codex-home/runs/d5a67a13e66446e8b5da03c00c62506f` |
| `scannet/scene0653_00::18::29686` | 18 | 1.0 | `.codex-home/runs/eaad6b05299043a28245bf71ad67698f` |
| `scannet/scene0462_00::6::40854` | 6 | 1.0 | `.codex-home/runs/f55526593b5149cf8f2c375b35556aa2` |

Each run home contains its own `sessions/2026/07/01/rollout-*.jsonl`.

## Reproduction

Launch script:
`docs/benchmark/nr3d/assets/run_v12_skill10_keep_home_20260701.sh`

Key settings:

```bash
export CODEX_AGENT_KEEP_RUN_HOME=1
export AIDP_CODEX_PROXY_UPSTREAM_API=responses
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office

PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_skill10_keep_home_20260701 \
  --pack-name pack_nr3d_v9_catalog_first \
  --limit 10 \
  --workers 10 \
  --sample-retries 2 \
  --tools \
  --skill-path .agents/skills/nr3d-codex-tools/SKILL.md \
  --reasoning-summary auto \
  --turn-timeout 900
```
