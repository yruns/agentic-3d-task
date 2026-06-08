# 3DVLMReasoning

Two-Stage Framework for 3D Scene Understanding with Evidence-Seeking VLM Agents

## Overview

This repository implements a two-stage framework for 3D scene understanding:

- **Stage 1** (`query_scene`): Task-conditioned keyframe retrieval from 3D scene graphs
- **Stage 2** (`agents`): VLM agentic reasoning over retrieved visual evidence

### Academic Innovation Points

1. **Adaptive Evidence Acquisition**: VLM agent dynamically decides when to request more visual evidence
2. **Symbolic-to-Visual Repair**: Stage 2 validates and corrects Stage 1 scene graph hypotheses
3. **Evidence-Grounded Uncertainty**: Explicit uncertainty output when evidence is insufficient
4. **Unified Multi-Task Policy**: Single agent handles QA, grounding, navigation, manipulation

## Installation

```bash
# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # Linux/macOS

# Install in development mode
uv pip install -e ".[dev]"

# With all optional dependencies
uv pip install -e ".[dev,full]"
```

### Alternative: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### Stage 1: Keyframe Retrieval

```python
from query_scene import KeyframeSelector

selector = KeyframeSelector(scene_path="/path/to/scene")
result = selector.select_keyframes_v2(
    query="Find the red pillow on the sofa",
    k=5
)
print(result.keyframes)
```

### Stage 1 with Dataset Adapter

```python
from query_scene import run_query_with_dataset
from dataset import get_adapter

# Load dataset through adapter
adapter = get_adapter("replica", data_root="/path/to/replica")

# Run query on specific scene
result = run_query_with_dataset(
    adapter=adapter,
    scene_id="room0",
    query="the lamp on the table",
    k=5
)
```

### Stage 2: Agentic Reasoning

```python
from agents import Stage2DeepResearchAgent
from agents.stage1_adapters import build_stage2_evidence_bundle

# Build evidence from Stage 1
evidence = build_stage2_evidence_bundle(stage1_result)

# Run agent
agent = Stage2DeepResearchAgent()
result = agent.run(task_spec)
print(result.structured_response)
```

## Running Tests

```bash
# All tests
pytest src/ -v

# Stage 1 tests
pytest src/query_scene/tests/ -v

# Stage 2 tests
pytest src/agents/tests/ -v

# Benchmark tests
pytest src/benchmarks/tests/ -v

# Migration equivalence tests
pytest tests/migration/ -v

# Integration tests
pytest tests/integration/ -v
```

## Project Structure

```
src/
├── query_scene/           # Stage 1: Keyframe retrieval
│   ├── keyframe_selector.py
│   ├── query_parser.py
│   ├── query_executor.py
│   ├── spatial_relations.py
│   ├── core/              # Core types and structures
│   ├── parsing/           # Parser infrastructure
│   └── tests/
├── agents/                # Stage 2: VLM agentic reasoning
│   ├── stage2_deep_agent.py
│   ├── models.py
│   ├── stage1_adapters.py
│   ├── benchmark_adapters.py
│   ├── core/              # Agent configuration
│   ├── adapters_pkg/      # Benchmark adapters
│   ├── tools/             # Agent tools
│   └── tests/
├── dataset/               # Multi-dataset support
│   ├── base.py            # DatasetAdapter interface
│   ├── registry.py        # Adapter registration
│   ├── replica_adapter.py
│   └── scannet_adapter.py
├── benchmarks/            # Benchmark loaders
│   ├── openeqa_loader.py
│   ├── sqa3d_loader.py
│   └── scanrefer_loader.py
├── evaluation/            # Evaluation framework
│   ├── batch_eval.py
│   ├── metrics.py
│   ├── scripts/
│   └── ablations/
├── config/                # Configuration utilities
│   └── datasets.yaml
└── utils/                 # Shared utilities
    └── llm_client.py

scripts/
├── run_migration_scorecard.py  # Migration validation
└── generate_migration_ground_truth.py

tests/
├── migration/             # Migration equivalence tests
│   └── ground_truth/      # Ground truth data
└── integration/           # End-to-end tests
```

## Dataset Support

### Supported Datasets

| Dataset | Adapter Name | Aliases |
|---------|--------------|---------|
| Replica | `replica` | `replica-imap`, `replica-v1` |
| ScanNet | `scannet` | `scannet-v2` |

### Using Dataset Adapters

```python
from dataset import get_adapter, list_adapters

# List available adapters
print(list_adapters())  # ['replica', 'replica-imap', 'replica-v1', 'scannet', 'scannet-v2']

# Get adapter instance
adapter = get_adapter("replica", data_root="/path/to/replica")

# Get scene metadata
metadata = adapter.get_scene_metadata("room0")
print(metadata.scene_id, metadata.num_frames)

# Iterate frames
for frame in adapter.iter_frames("room0", stride=5):
    process(frame.rgb, frame.depth, frame.pose)
```

### Configuring Datasets

Edit `src/config/datasets.yaml`:

```yaml
replica:
  data_root: /path/to/replica
  default_stride: 5
  coordinate_system: replica

scannet:
  data_root: /path/to/scannet
  default_stride: 10
  coordinate_system: scannet
```

## Documentation

- [Contributing Guide](CONTRIBUTING.md)
- [Architecture Overview](docs/architecture.md)
- [Dataset Guide](docs/dataset_guide.md)
- [Configuration Guide](docs/configuration_guide.md)
- [Stage 2 Design](docs/stage2_vlm_agent_design.md)
- [Agent Handoff](docs/stage2_agent_handoff.md)
- [Migration Report](MIGRATION_REPORT.md)

## Development

### Code Quality

```bash
# Formatting
black src/

# Linting
ruff check src/

# Type checking
mypy src/
```

### Running Migration Scorecard

```bash
python scripts/run_migration_scorecard.py --verbose
```

## License

MIT License
