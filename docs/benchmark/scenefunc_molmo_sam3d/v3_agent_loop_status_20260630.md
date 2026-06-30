# SceneFunc3D Molmo/SAM agent loop status, 2026-06-30

本文记录当前 SceneFunc3D agent 的真实运行状态：系统提示词、CLI 工具、外部模型、已经能稳定完成的环节、最新 E2E 轨迹中失败的位置，以及下一步应该如何收敛。

## 当前结论

截至 `2026-06-30`，代码已经在分支 `feat/scenefunc3d-agent-tools` 上提交并推送：

```text
35ea6a5 Harden SceneFunc3D Molmo SAM agent loop
```

当前结论分三层：

1. Repo 集成的工具链本身已经打通。MolmoPoint-8B、SAM2.1-Hiera-L、2D mask 到 3D lift、artifact inspect、single/multi-fragment fuse 这些 tool 都能独立工作。
2. Runner 已经有严格的 prompt、tool sequence guard 和 final provenance validation，能阻止一批明显错误的路径，例如 SAM 后直接 inspect、重复相同 Molmo/SAM/lift、两个有效 lift 后继续乱找。
3. 端到端 agent loop 仍未稳定跑通最新目标。失败不主要在 Molmo 或 SAM sidecar，而是在 LLM agent 的控制流：它看到 tool guard 的错误后，不能稳定自我纠正到下一步；或者已经有足够 fragment 后，不稳定调用 `fuse_accepted_masks`。

因此当前状态是：

```text
tool chain viable
runner validation hardened
latest free-form agent loop not yet reliable
no latest r22/r23 result.json
```

## 目标工作流

SceneFunc3D 和 NR3D 的入口不同：

- NR3D 目标是从现有 proposal pool 里选择一个 bbox id。
- SceneFunc3D 目标是生成一个 3D mask artifact。

由于通用 agent 不能自己完成 2D 分割和 3D mask lift，它必须通过外部工具完成以下流程：

```text
scene_summary
keyframe_selector
view_frame / view_crop / view_bev / frame_objects
molmo_point
sam_mask
lift_mask_to_3d
inspect_mask_artifact
suggest_additional_views, optional
repeat follow-up Molmo/SAM/lift/inspect, optional
fuse_accepted_masks
final JSON
```

理想情况下，agent 的职责不是直接生成 mask，而是：

1. 根据任务和多视角证据找到目标部件。
2. 对小 knob、handle、switch、dial 先抽取 affordance concept。
3. 用完整任务约束构造 Molmo point prompt。
4. 检查 Molmo 原始输出和 point overlay。
5. 检查 SAM contact sheet，选择紧凑且语义正确的 candidate。
6. 将 candidate lift 到 3D。
7. 通过 artifact geometry 判断 3D fragment 是否过大或偏离目标。
8. 根据遮挡和相机视角决定是否需要更多 view。
9. 将已批准的 3D fragments fuse 成最终 artifact。

## 当前 agent prompt

当前没有单独的外部 system prompt 文件。Runner 直接把完整任务 prompt 放进 `CodexTurnRequest.prompt`。

关键代码：

```text
src/codex_agent/scenefunc3d/runner.py
src/codex_agent/scenefunc3d/playbook.py
```

`build_turn_request()` 当前传入：

```python
CodexTurnRequest(
    prompt=self._build_prompt(),
    output_schema=SceneFunc3dMaskDecision.model_json_schema(),
    skills=(),
    image_paths=(),
)
```

也就是说，当前 SceneFunc3D runner 没有注入外部 Codex skill。真正起作用的是内联 playbook 文本和 CLI 工具约束。

Prompt 由以下部分拼成：

1. `SCENEFUNC3D_TOOLS_PLAYBOOK`
2. 当前 sample 的 `Task context`
3. runtime paths：
   - `scene_root`
   - `backend_config`
   - `out_dir`
4. CLI 调用方式
5. hard limits
6. final JSON schema

Prompt 中最重要的硬规则是：

```text
No image is evidence until you open it with view_image.
```

也就是说，任何工具返回的 image path 都必须用 `view_image` 打开后才能被 agent 当作视觉证据。

Prompt 还明确要求：

- `molmo_point` args 只能是 `image_path`, `image_width`, `image_height`, `prompt`。
- `sam_mask` args 只能是 `frame_id`, `image_path`, `point_xy`。
- `lift_mask_to_3d` args 只能是 `frame_id`, `candidate_id`, `mask_npz_path`。
- `inspect_mask_artifact` 必须使用 `lift_mask_to_3d` 返回的 `mask_npz_path`, `mask_ply_path`, `lift_overlay_path`。
- SAM candidate 不能只看 score。
- 小部件优先选紧凑 mask，拒绝 broad panel、radiator body、cabinet face、wall patch、pipe、shadow。
- SAM 成功后，如果 candidate plausibly compact，下一步必须是 `lift_mask_to_3d`。
- 两个有效 inspected fragments 已经覆盖同一个小 affordance 后，应立即 `fuse_accepted_masks`。
- final answer 必须来自 `fuse_accepted_masks`，不能直接用 fragment artifact。

## 当前 CLI tool 列表

统一入口：

```bash
python -m codex_agent.scenefunc3d.tools <tool> \
  --scene-root <scene_root> \
  --backend-config <backend_config> \
  --out-dir <out_dir> \
  --args '<json>'
```

当前可用工具：

| Tool | 当前用途 |
|---|---|
| `scene_summary` | 汇总 scene asset 和 task context。 |
| `keyframe_selector` | 找任务相关 first-person frames，返回 matched objects 和 recommended crops。 |
| `view_frame` | 渲染或打开完整 RGB frame。 |
| `view_crop` | 根据 bbox 或 pixel crop 放大小目标区域。 |
| `view_bev` | 查看 top-down BEV，如果当前 scene 没有 BEV asset 会失败。 |
| `frame_objects` | 列出某帧可见对象。 |
| `molmo_point` | 调外部 MolmoPoint，根据图像和 prompt 生成 point。 |
| `sam_mask` | 调外部 SAM2.1，用 point 生成 2D mask candidates。 |
| `lift_mask_to_3d` | 用 frame depth、pose、intrinsics 将 2D mask lift 成 3D fragment。 |
| `inspect_mask_artifact` | 检查 3D fragment、overlay 和 geometry metadata。 |
| `suggest_additional_views` | 根据第一块有效 3D fragment 推荐多视角补充帧。 |
| `fuse_accepted_masks` | 将已批准 fragments 融合为最终 3D mask artifact。 |

对于长耗时工具，prompt 要求 shell 调用设置：

```text
yield_time_ms=30000
```

这些工具包括：

```text
molmo_point
sam_mask
lift_mask_to_3d
inspect_mask_artifact
suggest_additional_views
fuse_accepted_masks
```

如果 shell 返回 session id，agent 必须继续 wait 同一个 session，不能并发启动另一个 SceneFunc3D tool。

## 当前外部模型和运行环境

最新目标配置是：

```text
MolmoPoint-8B + SAM2.1-Hiera-L
```

### Native single-port sidecar server

新的正式服务入口是 repo 内的单进程 server：

```text
src/codex_agent/scenefunc3d/servers/scenefunc_sidecar_server.py
```

它在一个 Python 进程中同时加载 MolmoPoint 和 SAM2.1，并只监听一个对外端口。对 `mlx export --port=9001` 场景，推荐直接让该 server 监听 IPv6 `::` 的 `9001`：

```bash
PYTHONPATH=src scripts/scenefunc3d/serve_native_sidecar.sh \
  --host :: \
  --port 9001 \
  --device cuda:0 \
  --molmo-model-path /mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/sidecar_services_20260630/molmopoint_merged_model \
  --sam-backend transformers \
  --sam-model-path /mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/molmopoint_sam21_20260628/hf_home/transformers/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610 \
  --staging-root /mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/sidecar_services_20260630 \
  --base-path /s/qnpoSRas
```

### Native server API contract

所有 `POST` 请求都使用 JSON body，并设置：

```text
Content-Type: application/json
```

如果部署在会保留 URL prefix 的 reverse proxy 后面，用 `--base-path` 增加挂载前缀。例如 `--base-path /s/<export-id>` 会同时支持 `/health` 和 `/s/<export-id>/health`。下面只写 server 看到的 path；挂载前缀只是在这些 path 前面加一段，不改变 method 和 body。

#### Aggregate health

```text
Method: GET
Path: /health
Body: none
```

Response:

```json
{
  "status": "ok",
  "molmo": {
    "status": "ok",
    "model_name": "MolmoPoint-8B",
    "model_loaded": true
  },
  "sam": {
    "status": "ok",
    "model_name": "SAM2.1-Hiera-L",
    "model_loaded": true
  }
}
```

#### Molmo health

```text
Method: GET
Path: /molmo/health
Body: none
```

Response:

```json
{
  "status": "ok",
  "model_name": "MolmoPoint-8B",
  "model_loaded": true
}
```

#### SAM health

```text
Method: GET
Path: /sam/health
Body: none
```

Response:

```json
{
  "status": "ok",
  "model_name": "SAM2.1-Hiera-L",
  "model_loaded": true
}
```

#### Molmo point

```text
Method: POST
Path: /molmo/v1/point
```

Body:

```json
{
  "request_id": "case-000073-molmo-01",
  "image_path": "/absolute/path/to/frame_or_crop.jpg",
  "prompt": "Point to the center of the green circular radiator temperature dial face.",
  "image_width": 252,
  "image_height": 403
}
```

Field requirements:

| Field | Type | Required | Meaning |
|---|---:|---:|---|
| `request_id` | string | yes | Caller-provided id echoed in the response. |
| `image_path` | string path | yes | Existing image file path visible to the server process. |
| `prompt` | string | yes | Pointing prompt sent to MolmoPoint. |
| `image_width` | integer | yes | Image width in pixels. Must be positive. |
| `image_height` | integer | yes | Image height in pixels. Must be positive. |

Response:

```json
{
  "request_id": "case-000073-molmo-01",
  "model_name": "MolmoPoint-8B",
  "raw_text": "<points ...>...</point>",
  "image_points": [
    {
      "x_px": 107.28666666666666,
      "y_px": 153.27141304347828,
      "source": "<points ...>...</point>",
      "label": "green circular radiator temperature dial face"
    }
  ],
  "latency_ms": 918.1
}
```

Notes:

- `image_points` can contain zero, one, or multiple points. The caller must review the point overlay or raw output before using it as a SAM prompt.
- Coordinates are absolute pixel coordinates in the input image coordinate system.

#### SAM masks

```text
Method: POST
Path: /sam/v1/masks
```

Body:

```json
{
  "request_id": "case-000073-sam-01",
  "image_path": "/absolute/path/to/frame_or_crop.jpg",
  "points": [
    {
      "x_px": 107.28666666666666,
      "y_px": 153.27141304347828,
      "label": "green circular radiator temperature dial face",
      "source": "molmo_point"
    }
  ],
  "staging_dir": "/absolute/path/to/output/sam/000073/candidates"
}
```

Field requirements:

| Field | Type | Required | Meaning |
|---|---:|---:|---|
| `request_id` | string | yes | Caller-provided id echoed in the response. |
| `image_path` | string path | yes | Existing image file path visible to the server process. |
| `points` | array | yes | One or more approved point prompts. |
| `points[].x_px` | float | yes | Point x coordinate in image pixels. |
| `points[].y_px` | float | yes | Point y coordinate in image pixels. |
| `points[].label` | string | no | Human-readable point label. |
| `points[].source` | string | no | Point provenance, usually Molmo raw output or `molmo_point`. |
| `staging_dir` | string path | yes | Directory where mask `.npz` artifacts are written. Must be under the server staging root. |

Response:

```json
{
  "request_id": "case-000073-sam-01",
  "model_name": "SAM2.1-Hiera-L",
  "candidates": [
    {
      "candidate_id": "mask_00",
      "score": 0.890625,
      "mask_npz_path": "/absolute/path/to/output/sam/000073/candidates/mask_00.npz",
      "pixel_count": 1579,
      "coverage_percent": 0.05711082175925926
    }
  ],
  "latency_ms": 69.7
}
```

Notes:

- SAM may return multiple candidates for one point prompt. The caller must choose a candidate after visual review; do not select by score alone.
- `mask_npz_path` is a server-written artifact path. Downstream `lift_mask_to_3d` should consume the selected candidate's path.

#### Common errors

Request validation failures return:

```text
Status: 400
Body:
```

```json
{
  "error": "invalid_request",
  "details": [
    {
      "field": "image_path",
      "message": "Path does not point to a file"
    }
  ]
}
```

GPU resource failures return:

```text
Status: 503
Body:
```

```json
{
  "error": "molmo_resource_unavailable"
}
```

or:

```json
{
  "error": "sam_resource_unavailable"
}
```

SAM-specific artifact or image failures use stable error names:

```text
invalid_staging_dir
sam_image_load_failed
sam_invalid_output
sam_artifact_write_failed
```

旧的 `molmo_point_server.py` 和 `sam2_mask_server.py` 仍保留为单模型调试入口，但外部服务不再需要额外的 NAS wrapper 或 `8711/8712` 双端口暴露。

E2E launcher 现在在 Linux 上默认启动项目内 ModelHub adapter：

```text
codex_modelhub_adapter
```

私有上游配置放在 gitignored 文件：

```text
codex_modelhub_adapter/.modelhub_upstreams.toml
```

该文件不能提交，不能写入文档，不能打印真实 AK。

当前 Linux 环境要使用的 ModelHub endpoint 策略已经进入项目脚本：

```text
Linux: project-local adapter, private ModelHub upstream config
macOS or non-Linux: keep previous path unless explicitly overridden
```

最新 runner 默认关键参数：

```text
CODEX_AGENT_MODEL=gpt-5.4-2026-03-05
CODEX_AGENT_MODEL_PROVIDER=modelhub_adapter
CODEX_AGENT_TURN_TIMEOUT_S=900
CODEX_AGENT_MAX_TOOL_CALLS=128
CODEX_AGENT_MAX_REPEATED_TOOL_CALLS=6
```

## 当前 tool sequence guard

为减少 agent 自由发挥，CLI 入口现在读取 `events.jsonl`，并按历史成功 tool 约束下一步。

### Checkpoint 后的允许转移

当前约束：

| 最近成功 checkpoint | 下一步允许 tool |
|---|---|
| `molmo_point` | `sam_mask` 或另一次 changed-args `molmo_point` |
| `sam_mask` | `lift_mask_to_3d` 或另一次 changed-args `sam_mask` |
| `lift_mask_to_3d` | `inspect_mask_artifact` |
| `suggest_additional_views` | `view_crop`, `view_frame`, `molmo_point`, `fuse_accepted_masks` |

这直接修复了一类失败：agent 在 Molmo 或 SAM 后跳回普通探索工具。

### SAM 后禁止直接 inspect

当前明确禁止：

```text
sam_mask -> inspect_mask_artifact
```

因为 `inspect_mask_artifact` 只能检查 lift 之后的 artifact。SAM 输出只是 2D candidate，不包含 3D fragment 的 `mask_ply_path` 和 `lift_overlay_path`。

正确链路必须是：

```text
sam_mask
lift_mask_to_3d
inspect_mask_artifact
```

### lift 后 inspect 参数必须来自真实 lift 输出

如果最近成功 tool 是 `lift_mask_to_3d`，下一次 `inspect_mask_artifact` 必须包含：

```text
mask_npz_path
mask_ply_path
lift_overlay_path
```

并且这些路径要来自真实 lift 输出，不能凭空猜路径。

### 禁止重复成功的高成本 checkpoint

以下 tool 成功后，不能用完全相同 args 再跑：

```text
molmo_point
sam_mask
lift_mask_to_3d
```

如果要重试，必须换 crop、frame、prompt、point 或 candidate。

### Molmo 成功次数预算

SAM 之前最多允许连续 3 次成功 Molmo：

```text
after 3 successful molmo_point calls before SAM, call sam_mask
```

目的是阻止 agent 在 point 阶段无限比较。

### Inspect 成功次数预算

当前最多允许 2 次成功 `inspect_mask_artifact` 后继续探索：

```text
after 2 successful inspect_mask_artifact calls, call fuse_accepted_masks
```

目的是强制 agent 在已有两个有效 3D fragments 后停止找第三视角。

### suggest 后 follow-up evidence 预算

`suggest_additional_views` 成功后，agent 可以看少量 follow-up frame/crop，但不能无限打开更多 view。

超过预算后必须：

```text
molmo_point on selected follow-up evidence
```

或者：

```text
fuse_accepted_masks with rejected_suggested_frame_ids
```

## 当前 final validation

Runner 不再只相信 final JSON 文本。它会重新验证：

1. final path 必须在 sample output directory 里面。
2. `mask_artifact_path` 必须是 `.json`。
3. `mask_npz_path` 必须是 `.npz`。
4. `mask_ply_path` 必须是 `.ply`。
5. 三个路径必须真实存在。
6. final outcome 必须匹配本次 `events.jsonl` 里成功的 `fuse_accepted_masks` event。
7. 每个 accepted fragment 必须能追溯到标准链路：

```text
view_frame or view_crop
molmo_point
sam_mask
lift_mask_to_3d
inspect_mask_artifact
fuse_accepted_masks
```

8. `multi_view_decision` 必须和 `suggest_additional_views` 的事件一致。

这意味着 agent 不能绕过 fuse，也不能在 final 里填一个 fragment 路径冒充最终 artifact。

## 已经可复查的 6/29 E2E 结果

现有落盘文档是：

```text
docs/benchmark/scenefunc_molmo_sam3d/agent_runner_e2e_20260629.md
```

该次运行的 case：

```text
421393::729dc0d8-571c-44e5-9dc4-05045524dcf5
Adjust the room's temperature using the radiator dial
```

模型：

```text
MolmoPoint-8B
SAM2.1-Hiera-L
```

该次运行证明了：repo-integrated agent path 能生成完整 artifact：

```text
summary.json
result.json
events.jsonl
fused/mask_artifact.json
fused/mask_data.npz
fused/lifted_points.ply
```

但评测分数是 0：

```text
iou=0.0
precision=0.0
recall=0.0
f1=0.0
predicted_count=99
gt_count=199
```

6/29 的 tool trace 里，agent 最终选择了：

```text
selected_frame_ids = ["000010"]
accepted_fragment_ids = ["000010_mask_02"]
multi_view_decision.action = "stop"
```

这说明 6/29 的状态是：

```text
artifact-generation success
semantic/evaluation failure
```

不是 runner 崩溃，也不是 sidecar 不可用。

## 6/29 成功轨迹细节

6/29 成功轨迹按阶段拆开如下。

### Evidence phase

`keyframe_selector` 返回了关键帧：

```text
000073
000010
000011
000072
```

agent 打开了这些 frame，并在 `000011` 和 `000010` 上生成 crop。

该阶段可成功。

已知 recoverable failures：

1. `keyframe_selector` 第一次带了多余字段 `top_k`，被 Pydantic 拒绝。
2. agent 去掉多余字段后重试成功。
3. `view_bev` 第一次带了多余字段 `task_description`，被拒绝。
4. `view_bev` 后续又因为 scene 没有 `conceptgraph/bev/scene_bev.png` 失败。

这些失败没有阻塞主流程。

### Molmo phase

agent 在 `000010` crop 上调用 Molmo。

输入 image：

```text
421393/000010_crop_pixel_xyxy_41aa437cdb42.jpg
```

prompt 约束目标为 radiator temperature dial/thermostatic knob，要求避免 radiator panel、wall、pipe、label box、shadow。

Molmo 解析出的点：

```text
x_px = 143.2
y_px = 185.98148148148147
```

Molmo raw text 和 point overlay 都保存成功。

该阶段可成功。

### SAM phase

SAM 对 `000010` 返回了 3 个 candidate：

| Candidate | Score | Pixels | Coverage |
|---|---:|---:|---:|
| `mask_00` | `0.06103515625` | `157` | `0.005678530092592593` |
| `mask_01` | `0.384765625` | `10442` | `0.37767650462962965` |
| `mask_02` | `0.859375` | `4128` | `0.14930555555555555` |

agent 选择了 `mask_02`。

该阶段可成功，但质量不稳定。这里有一个重要教训：最高 score 不一定是最适合小 affordance 的 mask。小 dial/knob 场景里，应该优先看 contact sheet、pixel count、coverage 和几何紧凑性。

### Lift and inspect phase

agent 将 `000010_mask_02` lift 到 3D，并成功 inspect。

该阶段可成功。

### Suggest/fuse phase

`suggest_additional_views` 返回了 follow-up frames：

```text
000101
000100
000099
000086
```

agent 看过 follow-up frames 后选择停止，只 fuse 了一个 fragment。

该阶段技术上可成功，但策略上不理想：最终 3D mask 与 hidden GT 完全无交集。

## 最新 6/30 r22 轨迹

6/30 的 r22 是在最新多视角策略和 tool guard 强化过程中观察到的运行。

临时 run root：

```text
/tmp/scenefunc_agent_runner_e2e_421393_20260630_r22
```

注意：该目录是临时 worker 路径，本机当前没有可复查的完整落盘文件；以下记录来自当次运行观察和状态总结。

### r22 成功的部分

r22 证明了新链路可以走得比 6/29 更远：

1. MolmoPoint-8B sidecar 可用。
2. SAM2.1-Hiera-L sidecar 可用。
3. agent 可以找到至少两个视角的目标证据。
4. agent 可以完成两次：

```text
Molmo point
SAM mask
lift_mask_to_3d
inspect_mask_artifact
```

当次得到两个有效 3D fragments：

```text
000073_mask_00
000010_mask_00
```

观察到的 geometry 尺度：

```text
000073_mask_00 max_extent_meters ~= 0.1979
000010_mask_00 max_extent_meters ~= 0.2165
```

这说明：

```text
多视角证据 -> Molmo -> SAM -> 3D lift -> inspect
```

这条链路本身是可跑通的。

### r22 失败的位置

r22 在第二个有效 inspected fragment 后，没有调用 `fuse_accepted_masks`。

实际失败模式：

```text
two valid inspected fragments exist
agent should fuse
agent instead drifts to a third view
third-view Molmo fails
no result.json
```

这说明 r22 的失败不在 sidecar，也不在 3D lift，而是在 agent policy：

```text
after enough valid fragments, the LLM did not terminate through fuse
```

### r22 暴露的问题

r22 暴露了 prompt-only 策略的核心缺陷：

- prompt 已经说“两块有效 fragment 后 fuse”。
- 但是 free-form agent 仍然会继续搜索第三视角。
- 如果第三视角 Molmo 失败，它会把本来已经足够生成 artifact 的轨迹拖入失败。

因此只靠 prompt 不能保证终止动作。

## 最新 6/30 r23 轨迹

r23 是在 r22 之后加入更强 tool guard 后的测试。

临时 run root：

```text
/tmp/scenefunc_agent_runner_e2e_421393_20260630_r23
```

同样，该目录是临时 worker 路径，本机当前没有可复查的完整落盘文件；以下记录来自当次运行观察和状态总结。

### r23 成功的部分

r23 说明新增 guard 有效果：

1. 初始 Molmo 失败后，agent 能换证据或 prompt 恢复。
2. 后续得到两个成功 Molmo points。
3. agent 在 Molmo 后试图非法调用 `view_crop`，tool guard 成功阻止。
4. agent 随后能继续到 `sam_mask`。
5. SAM 成功返回 candidates。

这说明：

```text
guard can block invalid exploration after Molmo
Molmo failure can be recovered
SAM sidecar works
```

### r23 失败的位置

r23 的关键失败发生在 SAM 之后。

正确下一步应该是：

```text
sam_mask
lift_mask_to_3d
inspect_mask_artifact
```

但 agent 实际尝试：

```text
sam_mask
inspect_mask_artifact with empty args
inspect_mask_artifact with guessed/placeholder paths
```

tool guard 正确拒绝了这两次非法调用，因为 SAM 输出还不是 3D artifact，不能直接 inspect。

拒绝后，agent 没有稳定自我纠正到：

```text
lift_mask_to_3d
```

最终没有生成 `result.json`。

### r23 暴露的问题

r23 说明 tool guard 只能阻止错误动作，但不能保证 LLM 读懂错误并走正确下一步。

当前工具错误消息已经包含提示：

```text
After sam_mask, call lift_mask_to_3d first by copying candidate_id and mask_npz_path from the selected sam_mask candidate.
```

但 agent 仍然可能继续错误地构造 inspect 参数，或者陷入无效修正。

这说明下一步要从“错误提示”升级为“控制流接管”。

## 哪些环节已经可靠

### 数据入口

SceneFuncVal-CG 的 conceptgraph 形式数据已准备好，当前 sample 能被 runner 读取。

可用：

```text
SceneFuncVal-CG/<scene_id>
rgb frames
depth
pose
intrinsics
conceptgraph mesh
task context
```

### Keyframe and crop retrieval

`keyframe_selector` 可以返回相关 frame 和 crop 建议。

当前已加入小 affordance 场景的 recommended crop 策略：

- radiator dial / valve：优先 right/lower end。
- drawer/cabinet pull：看 nearby frames，优先真实 knob、handle、pull tab、recessed grip。
- switch/button：看 control panel 区域。

该环节可以工作，但仍可能返回 broad object bbox，需要 agent 或后续 controller 做裁剪判断。

### Molmo point

MolmoPoint-8B 能输出可解析 point。

已支持：

- raw text 保存。
- point overlay 保存。
- 多 point 或多结果可视化。
- prompt 中明确约束 target affordance，避免 broad object。

不稳定点：

- 小部件上点位可能偏到 panel、shadow、pipe 或 label。
- 失败时 agent 必须换 crop、frame 或 prompt。
- 一个 query 可能得到多个候选点或多个解析结果，因为 Molmo 本质是 VLM 文本输出，point 是从 text 中解析出来的，后处理会保留多个可解析候选以便人工/agent 判断。

### SAM mask

SAM2.1-Hiera-L 能从 Molmo point 生成 multimask candidates。

已支持：

- contact sheet。
- candidate overlay。
- mask npz。
- score、pixel_count、coverage metadata。

不稳定点：

- 最高 score 可能是 broad object。
- 小 affordance 场景应优先 compact candidate，不应 score-only。
- 如果 Molmo point 偏了，SAM 会稳定分割错误区域。

### 2D to 3D lift

`lift_mask_to_3d` 可以用 depth、pose、intrinsics lift 成 3D fragment。

已支持：

- 自动解析 frame geometry asset。
- 输出 fragment NPZ。
- 输出 lifted PLY。
- 输出 lift overlay。

该环节目前不是主要失败点。

### Inspect artifact

`inspect_mask_artifact` 可以读取 lift 后 artifact，并提供 geometry 字段。

关键字段：

```text
bbox_extent_xyz
max_extent_meters
point_count
overlay path
```

它能帮助判断小 affordance 是否被 broad surface 污染。

### Fuse artifact

`fuse_accepted_masks` 可以把一个或多个 fragments 变成最终 artifact。

输出：

```text
fused/mask_artifact.json
fused/mask_data.npz
fused/lifted_points.ply
```

Runner 会验证 final JSON 必须来自该 tool。

该环节本身可工作。当前失败在 agent 不稳定调用它。

## 哪些环节仍会失败

### 失败类型 1：artifact 成功但语义错

代表：6/29 E2E。

轨迹：

```text
agent produces result.json
fuse succeeds
scoring succeeds
IoU/F1 = 0
```

原因：

- Molmo/SAM/lift 链路生成了合法 artifact。
- 但 2D point 或 SAM candidate 没有对准 hidden GT 对应的小 affordance。
- 单视角停止导致最终 mask 覆盖不足或语义偏离。

这类失败是 perception/selection quality 问题。

### 失败类型 2：已有足够 fragments 但不 fuse

代表：6/30 r22。

轨迹：

```text
first valid lift inspected
second valid lift inspected
agent should call fuse_accepted_masks
agent opens/queries more evidence
third Molmo fails
no result.json
```

原因：

- Prompt 要求不足以约束 termination。
- Tool guard 在当时还不够强，或者只阻止部分错误，不能主动完成 fuse。

这类失败是 control-flow termination 问题。

### 失败类型 3：SAM 后跳过 lift 直接 inspect

代表：6/30 r23。

轨迹：

```text
sam_mask success
agent calls inspect_mask_artifact
guard rejects because lift artifact does not exist
agent retries inspect with guessed paths
guard rejects again
agent does not recover to lift_mask_to_3d
no result.json
```

原因：

- LLM 将 2D SAM mask 和 3D lifted artifact 混淆。
- Tool guard 能拒绝错误，但不能保证下一次 action 正确。

这类失败是 phase-state confusion 问题。

### 失败类型 4：Molmo point 小目标不稳定

代表：多个小 knob/dial/switch 查询。

轨迹：

```text
crop/frame opened
Molmo point lands near target but not on operable part
SAM follows wrong point
SAM candidate may look plausible but lift geometry too broad or shifted
```

原因：

- Molmo 是纯 VLM，point 能力通过文本输出表达，不是一个几何分割模型。
- 小 affordance 在大物体上像素占比很低。
- 如果 crop 不够任务约束，模型容易指向 object body 或视觉显著区域。

缓解方式：

- 先抽 affordance concept。
- 用完整任务约束 prompt。
- 使用 recommended crop，而不是整帧。
- Molmo 后必须看 overlay。

### 失败类型 5：SAM candidate 选择不稳定

代表：6/29 `mask_02` score 最高但最终 IoU 为 0。

轨迹：

```text
Molmo point plausible
SAM returns multiple masks
agent selects high-score candidate
lift succeeds
final artifact valid
hidden GT mismatch
```

原因：

- SAM multimask 会返回不同粒度候选。
- 对小 affordance，score 不是唯一指标。
- broad candidate 在 SAM score 上可能更高。

缓解方式：

- contact sheet 必须视觉检查。
- 小部件优先低 coverage、紧凑 candidate。
- inspect geometry 后拒绝过大的 fragment。

## 当前不是主要问题的部分

### 不是主要问题：SAM sidecar 是否能运行

SAM2.1-Hiera-L 已经多次返回 candidates、contact sheet 和 mask npz。

失败主要不是 SAM 服务不可用。

### 不是主要问题：3D lift 是否能运行

`lift_mask_to_3d` 已经成功产出 fragment NPZ、PLY 和 overlay。

失败主要不是 lift 代码不可用。

### 不是主要问题：final validation 太弱

当前 final validation 已经很强：

- final path 必须存在。
- final path 必须来自 fuse。
- fuse event 必须能追溯到 view/Molmo/SAM/lift/inspect。
- multi-view decision 必须和 suggestion event 一致。

当前问题反而是 agent 经常到不了 final validation。

## 当前真正的问题

当前核心问题是：

```text
free-form LLM agent is not a reliable finite-state controller for this workflow
```

具体表现：

1. 它可以理解局部视觉证据，但不稳定维护 workflow phase。
2. 它能读到 tool 错误，但不稳定修正到唯一合法下一步。
3. 它能生成有效 fragment，但不稳定在 sufficient evidence 后停止。
4. 它有时把 SAM 的 2D mask artifact 和 lift 后的 3D artifact 混淆。
5. 它有时继续探索，导致本来可成功的轨迹被后续失败拖垮。

换句话说，当前失败不是“Molmo/SAM 不行”的单点问题，而是：

```text
agent decision loop lacks deterministic phase control
```

## 当前 prompt-only 方案的上限

已经加入的 prompt 规则包括：

- SAM 成功后下一步必须 lift。
- 两个 valid fragments 后必须 fuse。
- 不要重复相同 tool。
- 不要读 repo。
- 只能用 SceneFunc3D tools。
- final 必须来自 fuse。

这些规则能降低失败率，但不能保证端到端成功。r23 说明：即使 tool error 明确告诉它 “call lift_mask_to_3d first”，agent 仍可能继续构造错误 inspect。

因此 prompt-only 已经接近收益上限。

## 建议的下一步技术路线

下一步应该把关键阶段从 prompt 迁移到 runner-side deterministic controller。

### 方案 A：严格 FSM controller

Runner 维护状态机：

```text
START
EVIDENCE_SELECTED
MOLMO_PROPOSED
MOLMO_APPROVED
SAM_PROPOSED
SAM_APPROVED
LIFT_CREATED
LIFT_APPROVED
FOLLOWUP_SUGGESTED
READY_TO_FUSE
FUSED
FINAL
```

LLM 只负责在允许的候选中做判断：

- 选择哪个 frame/crop。
- 是否接受 Molmo point。
- 选择哪个 SAM candidate。
- 是否接受 lift geometry。
- 是否需要更多 view。

Runner 负责执行唯一合法下一步。

优点：

- 可以彻底解决 r23 的 SAM 后直接 inspect。
- 可以彻底解决 r22 的两个 fragment 后不 fuse。
- final provenance 更容易保证。

缺点：

- 需要改 runner 架构。
- LLM 的自由探索能力降低。

### 方案 B：auto-repair layer

保留当前 free-form shell agent，但在 tool guard 拒绝时返回更强的 machine-readable repair。

例如 SAM 后 inspect 被拒绝时，工具返回：

```json
{
  "error": "...",
  "required_next_tool": "lift_mask_to_3d",
  "required_args_from_last_sam_candidate": {
    "frame_id": "...",
    "candidate_id": "...",
    "mask_npz_path": "..."
  }
}
```

甚至 runner 可以直接代执行 `lift_mask_to_3d`。

优点：

- 改动较小。
- 保留当前 prompt/tool 模型。

缺点：

- 仍依赖 LLM 读 repair。
- 遇到 termination 问题仍可能需要额外 auto-fuse。

### 方案 C：runner-side auto-fuse

当 `events.jsonl` 中已经存在足够有效 inspected fragments，runner 自动调用 fuse。

触发条件：

```text
successful inspect_mask_artifact count >= 2
fragments target-consistent
latest inspection valid
no conflicting geometry signal
```

或者更保守：

```text
one valid fragment + suggest_additional_views action stop
```

优点：

- 直接修复 r22。
- 不需要等待 LLM 说 final。

缺点：

- 需要定义 “valid inspected fragment” 的结构化字段。
- 需要明确 rejected suggested frames 的记录策略。

### 方案 D：phase-specific mini prompts

把一个长 turn 拆成多个短 phase：

1. select evidence
2. approve Molmo point
3. approve SAM candidate
4. approve lift
5. decide follow-up or fuse

每个 phase 输出严格 JSON，不允许 shell 自由调用下一阶段工具。

优点：

- 比完全 FSM 灵活。
- 每步 schema 更小，错误更容易定位。

缺点：

- Runner 要编排更多 turn。
- Token 和调用次数增加。

## 推荐实现顺序

当前最推荐路线：

```text
FSM controller + auto-fuse fallback
```

具体优先级：

1. 在 runner 侧实现 phase state，不再让 LLM 自由决定 checkpoint 后下一步 tool。
2. SAM 成功后，由 runner 要求 LLM 只输出 selected candidate id；runner 调用 `lift_mask_to_3d`。
3. lift 成功后，由 runner 调用 inspect，再让 LLM 判定 accept/reject。
4. 两个 accepted fragments 后，runner 自动调用 `fuse_accepted_masks`。
5. 如果只有一个 accepted fragment，runner 根据 `suggest_additional_views` 的 action 决定是否继续。
6. final JSON 由 runner 基于 fuse result 生成，LLM 只补 confidence 和 uncertainties。

这能把当前最脆弱的 free-form control path 改成确定流程。

## 当前应保留的能力

即使引入 FSM，也应保留以下已实现能力：

- Molmo raw output 保存。
- Molmo point overlay。
- SAM contact sheet。
- selected SAM candidate overlay。
- lift overlay。
- fragment geometry summary。
- final artifact provenance。
- multi-view decision 审计字段。
- private ModelHub adapter 配置文件 gitignored。
- Linux 和 macOS endpoint 策略分流。

这些是后续 debug 和 benchmark 复现所需的审计证据。

## 当前代码验证状态

当前分支上的代码质量 gate 已通过：

```text
black src/
ruff check src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_*.py -q
PYTHONPATH=codex_modelhub_adapter pytest codex_modelhub_adapter/tests -q
black --check codex_modelhub_adapter/adapter/app.py codex_modelhub_adapter/adapter/proxy.py codex_modelhub_adapter/tests/test_app.py
git diff --check
```

已观察到的结果：

```text
src/keyframe/tests: 37 passed, 1 skipped
src/codex_agent/tests/test_scenefunc3d_*.py: 544 passed
codex_modelhub_adapter/tests: 35 passed
mypy: no issues
```

这些验证说明代码层面已经通过静态检查和单元测试；不代表最新 E2E agent loop 已经成功。

## 当前分支和提交状态

当前开发分支：

```text
feat/scenefunc3d-agent-tools
```

当前已推送提交：

```text
35ea6a5 Harden SceneFunc3D Molmo SAM agent loop
```

该提交包含：

- SceneFunc3D tool sequence guard。
- Prompt/playbook 强化。
- Runner final provenance validation。
- Linux ModelHub adapter launcher 集成。
- Adapter 429 retry/upstream rotation。
- 相关单元测试。

当前私有 upstream 配置文件仍是 ignored 文件，不应提交：

```text
codex_modelhub_adapter/.modelhub_upstreams.toml
```

## 当前状态一句话

当前不是“模型服务没有跑通”，而是“视觉工具链已经能产生可用中间结果，但 free-form agent 不能稳定按必须的 Molmo -> SAM -> lift -> inspect -> fuse 状态机走完”。下一步应把状态机和 fuse 终止条件移到 runner 侧，让 LLM 只做视觉判断和候选选择。
