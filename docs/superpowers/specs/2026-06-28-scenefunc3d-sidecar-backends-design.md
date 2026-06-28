# SceneFunc3D Sidecar Backends 设计

日期：2026-06-28

## 目标

把当前 SceneFunc3D agent tools 从 contract/stub 状态推进到单 case
end-to-end 可跑通：

```text
agent 选证据帧
-> molmo_point 调本地 Molmo sidecar
-> agent 查看 point overlay 并审批
-> sam_mask 调本地 SAM2.1 sidecar
-> agent 查看 candidates contact sheet 并审批
-> lift_mask_to_3d 本地几何 lifting
-> agent 查看 3D lift artifact 并审批
-> final 3D mask artifact
```

采用 B 路线：Molmo 和 SAM 作为常驻本地 sidecar server，agent tools 通过
本地 RPC 调用；3D lifting、artifact metadata、approval state、final output
仍保留在当前项目代码中。

Server 代码、启动脚本、测试和配置示例都必须放在当前仓库中，不能依赖临时
外部脚本。

## 设计决策

1. Molmo 和 SAM 服务用本地 HTTP JSON RPC。
2. 服务只监听 `127.0.0.1`，默认不暴露到网络。
3. 不新增 FastAPI/uvicorn 等 Web 依赖；首版使用标准库
   `http.server.ThreadingHTTPServer` 加 Pydantic v2 schema。
4. 模型加载生命周期属于 server；tool 调用不加载重模型。
5. Tool 层负责 agent 可见 artifacts：raw text、point overlay、SAM contact
   sheet、mask metadata、lifted PLY/NPZ。
6. SAM server 可以写 tool 指定的候选 mask staging 文件，因为 mask array 体积
   较大；但 metadata 和 contact sheet 仍由 tool 生成并登记。
7. 3D lifting 先做本地 deterministic 实现，不放入模型 server。
8. 单视角 E2E 优先，多视角建议和 fusion 作为第二阶段。

## Package Layout

新增模块放在现有 `src/codex_agent/scenefunc3d/` 下：

```text
src/codex_agent/scenefunc3d/
  backends/
    __init__.py
    config.py
    http_client.py
    molmo_rpc.py
    sam_rpc.py
    lift_3d.py
    frame_assets.py
  servers/
    __init__.py
    http_json.py
    schemas.py
    molmo_point_server.py
    sam2_mask_server.py
  tools/
    molmo_pointing.py
    sam_masking.py
    mask_lifting.py
    frame_views.py
    dispatch.py
```

新增脚本放在当前仓库：

```text
scripts/scenefunc3d/
  serve_molmo_point.sh
  serve_sam2.sh
  run_single_case_e2e.sh
  check_sidecars.sh
```

新增配置示例：

```text
configs/scenefunc3d_backends.example.toml
```

真实模型路径和端口配置放在 gitignored 本地配置：

```text
configs/scenefunc3d_backends.toml
```

`configs/scenefunc3d_backends.toml` 不能提交真实私有路径、token 或凭证。模型
路径不是 secret，但如果路径暴露内部机器结构，仍只放本地配置。

## Backend 配置

集中配置对象：

```text
SceneFunc3dBackendSettings
  molmo_url: str
  sam_url: str
  request_timeout_seconds: float
  artifact_staging_root: Path
  allowed_image_roots: tuple[Path, ...]
  allowed_output_roots: tuple[Path, ...]
```

加载顺序：

1. Tool CLI 显式 `--backend-config <path>`。
2. 默认 `configs/scenefunc3d_backends.toml`，如果存在。
3. 不存在配置时，保持当前 recoverable `ToolInputError`，不能假成功。

业务代码不散落读取环境变量。Shell 脚本可以接收环境变量或参数来拼启动命令，
但 Python 模块通过 CLI 参数和配置对象获得设置。

## Molmo Sidecar

### 启动方式

```bash
python -m codex_agent.scenefunc3d.servers.molmo_point_server \
  --host 127.0.0.1 \
  --port 8711 \
  --model-name MolmoPoint-8B \
  --model-path /path/to/molmo \
  --device cuda:0
```

`scripts/scenefunc3d/serve_molmo_point.sh` 包装上述命令，便于 tmux 中启动。

### Endpoints

```text
GET /health
POST /v1/point
```

`/health` 返回：

```json
{
  "status": "ok",
  "model_name": "MolmoPoint-8B",
  "model_loaded": true
}
```

`/v1/point` request：

```json
{
  "request_id": "421254_desc-a_000050_molmo_001",
  "image_path": "/abs/path/000050.jpg",
  "prompt": "point to the small handle used to open the lower drawer",
  "image_width": 1440,
  "image_height": 1920
}
```

`/v1/point` response：

```json
{
  "request_id": "421254_desc-a_000050_molmo_001",
  "model_name": "MolmoPoint-8B",
  "raw_text": "<point x=\"81.0\" y=\"61.9\">drawer handle</point>",
  "latency_ms": 812
}
```

Molmo server 不写 durable artifacts。`molmo_point` tool 负责：

- 保存 raw output text。
- 用现有 `parse_molmo_points()` 解析 points。
- 生成 point overlay。
- 返回 agent 可见 `raw_text_path`、`overlay_path`、points。

这样 Molmo parser、artifact layout 和 agent approval state 保持在当前 tool 层，
server 只负责推理。

## SAM2.1 Sidecar

### 启动方式

```bash
python -m codex_agent.scenefunc3d.servers.sam2_mask_server \
  --host 127.0.0.1 \
  --port 8712 \
  --model-name SAM2.1-Hiera-L \
  --checkpoint-path /path/to/sam2.1_hiera_large.pt \
  --config-path /path/to/sam2.1_hiera_l.yaml \
  --device cuda:0
```

`scripts/scenefunc3d/serve_sam2.sh` 包装上述命令。

### Endpoints

```text
GET /health
POST /v1/masks
```

`/v1/masks` request：

```json
{
  "request_id": "421254_desc-a_000050_sam_001",
  "image_path": "/abs/path/000050.jpg",
  "points": [
    {
      "x_px": 1166.4,
      "y_px": 1188.48,
      "label": "drawer handle",
      "source": "<point x=\"81.0\" y=\"61.9\">drawer handle</point>"
    }
  ],
  "staging_dir": "/abs/path/run/fragments/000050_mask_candidates"
}
```

`/v1/masks` response：

```json
{
  "request_id": "421254_desc-a_000050_sam_001",
  "model_name": "SAM2.1-Hiera-L",
  "candidates": [
    {
      "candidate_id": "mask_00",
      "score": 0.91,
      "mask_npz_path": "/abs/path/run/fragments/000050_mask_candidates/mask_00.npz",
      "pixel_count": 1119,
      "coverage_percent": 0.0405
    }
  ],
  "latency_ms": 430
}
```

SAM server 只写 mask arrays 到 tool 提供的 staging dir。`sam_mask` tool 负责：

- 验证 staging dir 在允许的 output root 下。
- 读取 candidate masks。
- 生成每个 candidate overlay。
- 生成包含所有 candidates 的 contact sheet。
- 返回 agent 可见 `contact_sheet_path` 和候选 metadata。

## 3D Lifting

`lift_mask_to_3d` 保持本地 deterministic 工具，不经过 sidecar server。

新增 `backends/lift_3d.py`：

```text
load_mask_npz(path) -> 2D bool mask
load_depth(path) -> depth meters
load_intrinsics(path) -> 3x3 K
load_pose(path) -> 4x4 camera-to-world
backproject_mask(mask, depth, K, pose) -> Nx3 world points
map_to_scene_points(points, scene_point_cloud) -> nearest scene point ids
write_lift_artifacts(...) -> NPZ + PLY + overlay metadata
```

新增 `backends/frame_assets.py` 负责解析每个 frame 的几何输入：

- RGB path。
- depth path。
- intrinsics path。
- pose path。

解析来源按优先级：

1. `raw/source_frames.json` 中的显式字段。
2. `raw/` 下按 frame id 匹配的 depth/intrinsics/pose 文件。
3. `conceptgraph/` 下可发现的 camera trajectory 或 pose files。

如果某项缺失，`lift_mask_to_3d` 返回 recoverable `ToolInputError`，错误类型应能
区分：

- `lift_depth_missing`
- `lift_intrinsics_missing`
- `lift_pose_missing`
- `lift_invalid_mask`
- `lift_too_sparse`

首版 lifting 输出：

```json
{
  "frame_id": "000050",
  "candidate_id": "mask_00",
  "lifted_point_count": 1119,
  "mask_npz_path": "/abs/path/fragments/000050_mask_00/mask_data.npz",
  "mask_ply_path": "/abs/path/fragments/000050_mask_00/lifted_points.ply",
  "overlay_path": "/abs/path/overlays/000050_mask_00_lift.jpg"
}
```

## Tool 接入

现有 tool names 不变：

- `molmo_point`
- `sam_mask`
- `lift_mask_to_3d`

`dispatch.py` 仍只做 lazy routing。每个工具内部：

1. 用 Pydantic 校验 CLI JSON 参数。
2. 加载 backend settings。
3. 调 RPC 或本地 lift。
4. 写 durable artifacts。
5. 返回 JSON-ready payload。

所有 model server 调用错误都转成 recoverable `ToolInputError`：

- server 未启动。
- `/health` 不健康。
- timeout。
- schema mismatch。
- backend 500/503。
- GPU OOM。
- 输出文件缺失或不可读。

不能出现 fake success。没有真实 crop、object index、model output、lift result 时，
tool 必须返回 recoverable error 或明确的 unavailable 状态。

## Agent 审批流

单视角 E2E 必须保留当前 approval gates：

1. `molmo_point` 返回 point overlay 后，agent 必须查看 overlay。
2. Agent 明确接受某个 point 后，才能调用 `sam_mask`。
3. `sam_mask` 返回 candidates contact sheet 后，agent 必须查看同一张图上所有
   candidates。
4. Agent 明确接受某个 SAM candidate 后，才能调用 `lift_mask_to_3d`。
5. `lift_mask_to_3d` 返回 first lift artifact 后，agent 必须查看 overlay 和
   artifact summary。
6. Agent 接受 first lift 后，才能 final 或进入多视角阶段。

首版 E2E runner 可以先只要求单视角 final。多视角逻辑在后续阶段启用。

## 多视角第二阶段

单视角跑通后，再接：

- `suggest_additional_views`：基于 first lift 的 camera coverage、遮挡、点云稀疏
  和相邻 frame visibility 推荐补充视角。
- 对每个补充视角重复 Molmo point、SAM candidates、3D lift 三个审批点。
- `fuse_accepted_masks`：合并 agent 接受的 3D fragments，输出 fused NPZ/PLY。

第二阶段不使用 CLIP top-K 作为首要路径。视角候选优先来自当前 SceneFuncVal-CG
和 ConceptGraph 的 frame/pose/visibility 结构；后续如果需要再接 CLIP rerank。

## Server 生命周期

长时间运行的模型服务必须在 tmux 中启动。推荐流程：

```bash
tmux new -s scenefunc-molmo
scripts/scenefunc3d/serve_molmo_point.sh ...

tmux new -s scenefunc-sam2
scripts/scenefunc3d/serve_sam2.sh ...
```

`scripts/scenefunc3d/check_sidecars.sh` 检查：

- Molmo `/health`。
- SAM `/health`。
- 模型名是否匹配配置。
- 简单 schema round-trip。

Runner 在开始单 case E2E 前必须先 health check。Health check 失败时直接给出
可定位错误，不让 agent 进入半可用状态。

## 测试策略

不依赖 GPU 的默认测试：

- Pydantic request/response schema tests。
- Backend config loader tests。
- HTTP client timeout/schema/error handling tests。
- Fake Molmo HTTP server tests：返回固定 raw text，tool 生成 raw text file 和
  point overlay。
- Fake SAM HTTP server tests：返回固定 mask npz，tool 生成 overlays 和 contact
  sheet。
- Lifting unit tests：用小型 synthetic depth、intrinsics、pose、mask 验证
  backprojection。
- CLI tests：server unavailable、invalid config、invalid file path 都是
  recoverable JSON error。

GPU / heavy integration tests 用显式 marker 或脚本，不进入默认质量门禁：

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_heavy_e2e.py \
  -m scenefunc3d_heavy
```

单 case smoke：

```bash
scripts/scenefunc3d/run_single_case_e2e.sh \
  --dataset-root /mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG \
  --sample-id 421254::<desc_id> \
  --output-dir /tmp/scenefunc3d_e2e
```

默认质量门禁仍然是：

```bash
ruff check src/
black src/
mypy src/
PYTHONPATH=src /mlx_devbox/users/yueshuhao/miniforge3/envs/conceptgraph/bin/python -m pytest src/keyframe/tests -q
```

## 里程碑

### P1：Sidecar 协议和 fake-server E2E

- 增加 `servers/schemas.py`、`servers/http_json.py`。
- 增加 `backends/http_client.py`、`backends/config.py`。
- `molmo_point` 接 fake Molmo server。
- `sam_mask` 接 fake SAM server。
- 无 GPU tests 跑通。

### P2：本地 3D lifting

- 解析 frame depth、intrinsics、pose。
- 实现 mask backprojection。
- 输出 fragment NPZ/PLY。
- 用 synthetic test 和一个真实 frame smoke 验证。

### P3：真实 MolmoPoint-8B sidecar

- 实现模型加载。
- 输入 image + prompt。
- 返回 raw text。
- 单帧 `molmo_point` tool 生成 point overlay。

### P4：真实 SAM2.1-Hiera-L sidecar

- 实现模型加载。
- 输入 approved points。
- 输出 candidates mask NPZ。
- `sam_mask` tool 生成 overlays 和 contact sheet。

### P5：单 case E2E

- 启动两个 sidecar。
- 跑一个固定 SceneFuncVal-CG case。
- 保存完整 artifacts。
- 手工或 agent 审批 point、SAM candidate、first lift。

### P6：多视角和 fusion

- 根据 first lift coverage 推荐补充视角。
- 重复审批链。
- 合并 accepted fragments。

## 风险和缓解

- Molmo 输出格式漂移：保存 raw text，并继续用严格 parser；坏格式是
  recoverable model-output error。
- SAM 候选分错大面：必须展示所有 candidates contact sheet，agent 选择后才能
  lift。
- Server 冷启动慢：常驻 sidecar，只在 tmux 中启动一次。
- GPU OOM：server 返回结构化 503，tool 转成 recoverable `ToolInputError`。
- 数据几何字段不统一：`frame_assets.py` 做路径发现和错误分类，首版先覆盖
  SceneFuncVal-CG 当前两场景。
- Artifact 写入失败：tool 包装为 recoverable error，不吞异常。
- 端口冲突：启动脚本允许显式端口；health check 报告实际 endpoint。

## 非目标

- 不把 Molmo/SAM 权重提交到仓库。
- 不把私有 token、AK、服务凭证写入配置示例。
- 不在默认 import 路径加载 torch 或模型。
- 不在首版做自动批量 benchmark。
- 不在首版实现 CLIP top-K view retrieval。

## 完成标准

首个可接受版本必须满足：

1. `molmo_point` 能通过本地 sidecar 拿到真实 Molmo raw output，并生成 point
   overlay。
2. `sam_mask` 能通过本地 sidecar 生成多个 mask candidates，并生成同图 contact
   sheet。
3. `lift_mask_to_3d` 能把一个 approved candidate lift 成 3D points，并写出
   NPZ/PLY。
4. 一个固定 SceneFuncVal-CG sample 能跑通单视角 E2E。
5. 所有失败路径不能假成功，必须有可恢复错误和 durable diagnostic artifact。
6. 默认无 GPU 测试和仓库质量门禁通过。
