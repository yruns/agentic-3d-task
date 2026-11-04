# Linux agentic3d environment and shared model store

Provisioned on **2026-11-02** for the current `keyframe` / `codex_agent` source tree.
The existing `conceptgraph` environment and its older model caches were not modified.

## Activate and link

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
cd /home/ysh/codebase/agentic-3d-task
scripts/link_agentic3d_assets.sh
```

The named environment is `/home/ysh/miniconda3/envs/agentic3d`.
Activation sets `HF_HOME`, `HF_HUB_CACHE`, `HF_MODULES_CACHE`,
`AGENTIC3D_ASSET_ROOT`, `MOLMO_MODEL_PATH`, `SAM_MODEL_PATH`, and a matching
`CUDA_HOME`. **The default physical GPU is 0.** Override `CUDA_VISIBLE_DEVICES`
after activation when needed, but never select broken physical GPU 1.

The linker is idempotent and refuses to replace unrelated existing paths.
For another checkout:

```bash
scripts/link_agentic3d_assets.sh --project-root /path/to/checkout
```

Reinstall that checkout with `python -m pip install --no-deps -e /path/to/checkout`
if it should replace this environment's editable project installation.

## Shared assets

`/home/ysh/Models` is an existing symlink to `/data1/ysh/Models`. Assets are stored
under `/home/ysh/Models/agentic3d`, not in Git:

```text
agentic3d/
├── models/                         # Stable aliases to complete pinned snapshots
│   ├── MolmoPoint-8B/
│   ├── sam2.1-hiera-large/
│   └── CLIP-ViT-H-14-laion2B-s32B-b79K/
├── third_party/                    # Real source checkouts, plus exact SDK release
│   ├── sam2/
│   ├── open_clip/
│   ├── molmo2/
│   ├── transformers/
│   ├── accelerate/
│   ├── codex/
│   └── openai-codex/                # Alias to the exact 0.163.0 source release
├── cache/huggingface/               # Copied/downloaded weights and dynamic modules
├── wheelhouse/                      # Ubuntu-compatible local Codex CLI wheel
├── manifests/                      # Versions, revisions, checksums, validation
├── logs/
└── validation/                     # Generated synthetic-image smoke artifacts
```

The checkout contains ignored links:

```text
external/models      -> /home/ysh/Models/agentic3d/models
external/third_party -> /home/ysh/Models/agentic3d/third_party
```

The Hugging Face blob storage is inside the shared store. The copied OpenCLIP
weights do not depend on the old `/home/ysh/.cache/huggingface` directory.

### Pinned models

| Model | Upstream | Revision |
|---|---|---|
| MolmoPoint-8B | `allenai/MolmoPoint-8B` | `188130f961c8e0888a34e11121a1423c461a01ba` |
| SAM2.1 Large | `facebook/sam2.1-hiera-large` | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |
| OpenCLIP ViT-H-14 | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | `1c2b8495b28150b8a4922ee1c8edee224c284c0c` |

MolmoPoint includes all eight weight shards and its local custom Python code.
SAM2 includes both the Transformers safetensors and official `.pt` checkpoint.
All **11 weight files** were SHA256-checked against their upstream HF LFS blob
identifiers; see `manifests/model-checksums.json`.

SAM3, SAM3D Objects, Florence-2, GroundingDINO, RAM, and original SAM are **not
current production dependencies** and were not added to this deployment.
Standalone base Qwen/SigLIP checkpoints are not additionally required: the
MolmoPoint checkpoint already contains its trained text and vision weights.

### Source provenance

Exact Git commits are in `manifests/third_party.tsv`:

| Repository | Selected source |
|---|---|
| `facebookresearch/sam2` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| `mlfoundations/open_clip` | `v3.2.0` |
| `allenai/molmo2` | `f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a` |
| `huggingface/transformers` | `v4.57.1` |
| `huggingface/accelerate` | `v1.10.1` |
| `openai/codex` | `rust-v0.163.0` |

SAM2, OpenCLIP, Transformers, Accelerate, and the exact Codex Python SDK are
installed editable from this store. MolmoPoint inference uses the pinned
checkpoint's custom code. The `molmo2` training repository is preserved for
reference, not installed as a second conflicting runtime implementation.

The actual runtime uses the SHA256-verified PyPI `openai-codex==0.163.0`
source release under `third_party/openai-codex`. The matching Git repository is
preserved separately for CLI/source reference; do not replace the verified SDK
installation with an unverified checkout.

## Environment versions and reproduction records

Key versions:

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch / torchvision | `2.8.0+cu126` / `0.23.0+cu126` |
| Transformers | `4.57.1` |
| Accelerate | `1.10.1` |
| OpenCLIP | `3.2.0` |
| NumPy / SciPy | `1.26.4` / `1.15.3` |
| OpenCV / Pillow | `4.11.0.86` / `11.3.0` |
| Codex SDK / CLI | `0.163.0` / `0.163.0` |

The environment also includes all project extras (`dev,vision,clip,codex`),
Open3D for compatibility with older prepared scene pickles, Hydra/OmegaConf,
MolmoPoint's `einops`/`decord2` dependencies, and the ModelHub adapter dependencies.

Full records:

- `manifests/agentic3d-environment.yml`
- `manifests/agentic3d-conda-explicit.txt`
- `manifests/agentic3d-pip-freeze.txt`
- `manifests/models.json`
- `manifests/preflight.json`
- `manifests/check-results.tsv`
- `manifests/model-smoke-*.json`
- `manifests/native-http-smoke.json`

### Codex SDK/CLI upgrade and Linux wheel support

SDK and CLI were upgraded together to **0.163.0**, the latest stable PyPI
releases checked on 2026-11-02. This CLI publishes an official
`manylinux_2_17_x86_64` wheel supported by Ubuntu: **no retagging or binary
modification is needed**. The verified original wheel is saved in `wheelhouse/`.
The SDK loads its bundled CLI directly; a separately installed `codex` shell
command can have another version and is not the SDK runtime version check.

```bash
python -m pip install --find-links /home/ysh/Models/agentic3d/wheelhouse \
  'openai-codex==0.163.0'
```

Upgrade provenance, startup/RPC checks, test results, and the previous working
environment are under `manifests/codex-upgrade-20261102/`. The previous SDK
`0.1.0b2`, CLI `0.132.0`, and repaired old wheel are retained only for rollback;
the old platform-tag repair is not a requirement for the current installation.
Model-stack versions were kept unchanged. No remote model call or scored
benchmark was made as part of this upgrade.

### Official SAM2 CUDA extension

The extension was built, imported, and exercised on a real CUDA tensor; it was
**not silently omitted**. Build settings were:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
export CPATH="$(find "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia" \
  -maxdepth 3 -type d -name include | paste -sd: -)${CPATH:+:$CPATH}"
export SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0
python -m pip install --no-build-isolation --no-deps -e external/third_party/sam2
```

CUDA 12.6 compiler/dev packages are installed in `agentic3d`; do not compile this
PyTorch 12.6 extension with the host's old `/usr/local/cuda-11.8` toolkit.
Conda compiler hooks are not `set -u` safe: activate Conda before enabling
`nounset` in automation. Run build tools from the project root, passing external
source paths as arguments, not from inside downloaded source directories.

## Start local model services

The combined native service fits on one 24 GB RTX 4090 in the synthetic smoke.
Start it in tmux; it intentionally binds **127.0.0.1**, not a public interface:

```bash
tmux new-session -d -s agentic3d-sidecar 'bash -lc "\
  source /home/ysh/miniconda3/etc/profile.d/conda.sh && \
  conda activate agentic3d && \
  cd /home/ysh/codebase/agentic-3d-task && \
  scripts/scenefunc3d/serve_local_agentic3d.sh"'
curl -fsS http://127.0.0.1:9001/health
```

The generated, ignored `configs/scenefunc3d_backends.toml` points to this service.
It restricts input/output roots to this checkout and the synthetic validation
images; extend those roots explicitly if using datasets stored elsewhere.
The validation service was stopped after testing; it is not left occupying GPU
memory. Stop a service you start with `tmux send-keys -t agentic3d-sidecar C-c`.

For the **official** SAM2 backend instead of the combined Transformers backend:

```bash
python -m codex_agent.scenefunc3d.servers.sam2_mask_server \
  --backend official \
  --checkpoint-path external/models/sam2.1-hiera-large/sam2.1_hiera_large.pt \
  --config-path external/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --staging-root tmp/scenefunc3d/artifacts \
  --device cuda:0
```

Use a YAML **inside the installed SAM2 package** for that official backend.
The wrapper converts its resolved filesystem path to the Hydra package-relative
config name. It rejects unrelated external YAML rather than silently substituting
another model configuration. Exact floating-point 0/1 masks returned by official
SAM2 are accepted; soft logits, nonbinary values, NaN, and infinity remain errors.

## Active Codex adapter

The maintained service now lives in `src/codex_adapter/` and uses the root
project's `adapter` extra and `uv.lock`. The old standalone
`codex_modelhub_adapter/` launch commands are retired. See the
[adapter operations guide](codex_agent/adapter.md) for explicit model registration,
per-model URL/AK/weight upstream pools, private file-based AKs, state
initialization, and durable session/upstream bindings. Use the existing Linux `agentic3d` environment; do not create another
venv or modify the vision model stack for this gateway.

GPT stays native Responses end to end. Gemini uses a separate Chat codec with
buffered Responses SSE and explicit capability checks; compact is unsupported.
A successful local readiness or SDK startup check is not upstream authentication,
strict-schema acceptance, or evidence of a prefix-cache hit.

## Validation and boundaries

Validation uses generated local images, not a benchmark fold:

- Native imports, NumPy/OpenCV ABI, bf16 GPU matmul, and the compiled SAM2 kernel.
- Actual offline OpenCLIP text encoding, MolmoPoint pointing, and SAM2 masks.
- Both official and Transformers SAM2 implementations.
- A real combined localhost HTTP request chain: health → Molmo point → SAM masks.
- Codex SDK starts and initializes the real CLI app-server, without a paid LLM call.
- Complete source/adapter tests, Ruff, Black, mypy, MkDocs, and CLI help checks.

The setup also corrects three stale test meshes to the current `raw/mesh.ply`
contract and adds type annotations compatible with the pinned NumPy/mypy stack;
it does not add a production geometry fallback.

**Remote LLM credentials and prepared benchmark data are still required for real
NR3D/OpenEQA/SceneFunc3D evaluations.** No real API keys were invented, copied into
this store, logged, or committed. No scored benchmark or paid model turn was run.
The local ModelHub adapter must still be configured and pass its health preflight
before an agent evaluation. CLI help and app-server startup are not claims that
remote LLM authentication has been verified.
