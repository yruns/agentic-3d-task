# SceneFunc3D Agent 工具设计

日期：2026-06-28

## 目标

为已经准备好的 ConceptGraph 数据集增加一个 SceneFunc3D agent 工作流：

```text
/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG
```

这个工作流用于解决语言条件下的功能部位 3D 分割任务。Agent 先通过
keyframe、切换帧、crop、BEV 等工具找到视觉证据；然后调用 Molmo 生成
任务约束下的 2D point；再调用 SAM 生成 2D mask candidates；经 agent 审核
通过后，把选中的 2D mask 通过 depth、intrinsics、pose lift 到 3D；最后可
以根据第一个通过审核的 3D seed mask 判断是否需要从其他视角补充更多 3D
points，得到更完整的功能部位 mask。

SceneFunc3D 和 NR3D 是两个不同任务：

- NR3D：从已有候选中选一个 bbox/proposal id。
- SceneFunc3D：生成一个新的 3D mask artifact。

因此 SceneFunc3D 必须有单独的 task entrypoint、prompt/playbook、final answer
schema、tool contract 和 evaluation 逻辑，不能复用 NR3D 的 bbox id 输出语义。

## 当前项目上下文

仓库中已经有两个 Codex Agent 任务包：

- `src/codex_agent/nr3d/`
- `src/codex_agent/openeqa/`

它们提供了可复用的工程模式：

- task-specific sample loader
- playbook
- runner
- tool dispatcher
- keyframe retrieval
- frame rendering
- crop rendering
- BEV rendering
- tests

SceneFuncVal-CG 当前包含两个完整的 ConceptGraph 形式场景：

- `421254`
- `421393`

每个场景包含：

- `<visit_id>_descriptions.json`
- `<visit_id>_annotations.json`
- `<visit_id>_motions.json`
- `conceptgraph/` artifacts

现有 `docs/benchmark/scenefunc_molmo_sam3d/` 记录说明 Molmo + SAM + 3D
lifting 是可行的，但也暴露了一个关键问题：SAM 最高分 candidate 可能会把
大面积表面分出来，而不是小的 handle、knob、switch、dial 等目标部位。

## 决策

新增独立的 SceneFunc3D task package：

```text
src/codex_agent/scenefunc3d/
```

这个 package 可以复用 NR3D/OpenEQA 的底层证据获取思路，但不能复用 NR3D
的 task entrypoint、final answer schema 或 proposal id 语义。

首版实现目标是一个实际可落地的 end-to-end workflow：

1. 加载 SceneFuncVal-CG sample。
2. 让 agent 搜索并查看视觉证据。
3. Molmo point 生成后，必须让 agent 看 overlay 并审核。
4. SAM mask candidates 生成后，必须让 agent 看同图 contact sheet 并审核。
5. 只有 agent 审核通过的 mask 才能 lift 到 3D。
6. 第一个审核通过的 3D mask 作为 seed。
7. Agent 根据 seed 的遮挡、相机视角、点云稀疏度和几何覆盖判断是否需要
   多视角补充。
8. 保存可审计的 mask artifact 和结构化运行 metadata。
9. 增加对隐藏 GT annotation indices 的 scoring hook。

## 范围

首版包含：

- 新的 SceneFunc3D task namespace、runner、playbook、sample loader、tool
  dispatcher、artifact schema 和 scoring skeleton。
- 类似 OpenEQA/NR3D 的证据工具：scene summary、keyframe selection、
  frame viewing、crop viewing、BEV viewing、visible object listing。
- Molmo point tool：保存 raw text、parsed points 和 point overlay。
- SAM mask tool：保存所有 candidate masks，并在同一张 contact sheet 上展示。
- 3D lifting tool：使用 frame depth、intrinsics、camera pose 生成 3D points。
- Molmo、SAM、first lift、多视角扩展之间的 agent approval gates。
- 用于调试和评估的 durable artifacts。
- 不依赖 GPU 的单元测试：schema、parsing、state transition、sample loading、
  scoring math。

首版不包含：

- 训练或微调 Molmo、SAM 或 3D refinement 模型。
- 不经 agent 审核的全自动盲目 top-K multi-view fusion。
- 替换已有 NR3D/OpenEQA runners。
- 提交模型权重、私有凭证或大型生成 artifacts。
- 直接把 benchmark smoke script 当作生产接口使用。

## Package Layout

使用独立、强类型、小模块的 package：

```text
src/codex_agent/scenefunc3d/
  __init__.py
  sample.py
  task.py
  playbook.py
  runner.py
  tools/
    __init__.py
    __main__.py
    dispatch.py
    models.py
    scene_context.py
    frame_views.py
    keyframe_retrieval.py
    molmo_pointing.py
    sam_masking.py
    mask_lifting.py
    mask_artifacts.py
    mask_inspection.py
  evaluation/
    __init__.py
    metrics.py
    scorer.py
```

SceneFunc3D tool CLI 必须独立：

```bash
python -m codex_agent.scenefunc3d.tools <tool> \
  --scene-root <scene_dir> \
  --json '<validated-json-payload>'
```

单 case 或 batch runner 也必须是 SceneFunc3D 专用入口：

```bash
python -m codex_agent.scenefunc3d.runner \
  --dataset-root /mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG \
  --visit-id 421254 \
  --desc-id <description_id> \
  --output-dir <run_dir>
```

## Sample Model

Sample loader 为每条 description 生成强类型 sample：

```text
SceneFunc3dSample
  sample_id: "<visit_id>::<desc_id>"
  visit_id: "421254"
  desc_id: "<description_id>"
  task_description: "<natural language task>"
  annotation_ids: [...]
  motion_hints: [...]
```

`<visit_id>_annotations.json` 中的 GT annotation indices 不能进入 agent prompt，
也不能通过 prediction tools 暴露给 agent。它们只用于 scorer。

Motion metadata 可以作为 task context 暴露给 agent，前提是它用于理解任务本身，
而不是泄露 GT mask。例如 `rot`、`trans`、motion direction 可以帮助 agent 区
分 knob、dial、handle、switch、drawer face、button 等功能部位。

## Agent 状态机

Runner 必须强制执行阶段状态机：

```text
evidence_selected
-> molmo_point_proposed
-> molmo_point_agent_approved
-> sam_candidates_proposed
-> sam_candidate_agent_approved
-> first_lift_created
-> first_lift_agent_approved
-> optional_multiview_expansion
-> fused_mask_created
-> final_answer
```

状态转移规则：

- Agent 必须查看 Molmo raw output 和 point overlay 后，才能进入 SAM。
- 如果 agent 没有接受 point，SAM 不能运行。
- Agent 必须查看所有 SAM candidates 的同图 contact sheet 后，才能进入 3D lift。
- 如果 agent 没有接受某个 SAM candidate，3D lift 不能运行。
- 第一个通过 agent 审核的 3D lift 结果成为 seed mask。
- 只有 agent 查看并接受 first 3D seed mask 后，才能考虑额外视角。
- 任何 mask fragment 进入最终 fused artifact 前，都必须经过 point、2D mask、
  3D lift 三个审核点。

这些 gate 不能只写在 prompt 里。代码层也要验证状态转移，避免 prompt 错误或
tool misuse 跳过审核。

## Tool Contracts

### Evidence Tools

Evidence tools 让 agent 在调用 Molmo 前先找到并查看视觉证据：

| Tool | 目的 |
| --- | --- |
| `scene_summary` | 返回可用 frames、object counts 和 indexed assets。 |
| `keyframe_selector` | 根据 task description 或 target concept 找候选关键帧。 |
| `view_frame` | 渲染一张 RGB frame，供 `view_image` 查看。 |
| `view_crop` | 围绕 visible object、bbox 或 image region 渲染 zoomed crop。 |
| `view_bev` | 渲染 top-down scene context，可选 highlights。 |
| `frame_objects` | 列出某个 frame 中可见的 ConceptGraph objects。 |

Playbook 必须明确要求：只有 agent 真正看过返回图片后，该 frame 或 crop 才能
被当作视觉证据引用。

### Molmo Point Tool

`molmo_point` 在一张已查看的 frame 或 crop 上运行配置好的 Molmo-family 模型，
返回 raw output 和 parsed points：

```json
{
  "frame_id": "000050",
  "prompt": "Point to the small drawer knob used to open the bottom drawer.",
  "points": [
    {
      "x_px": 1166.4,
      "y_px": 1188.48,
      "source": "molmo_percent",
      "label": "small drawer knob"
    }
  ],
  "raw_text_path": "/abs/path/raw_outputs/molmo_000050.txt",
  "overlay_path": "/abs/path/overlays/molmo_points_000050.jpg"
}
```

工具运行后，agent 必须查看 overlay，并做出以下决策之一：

- `accept_point`
- `retry_with_crop`
- `retry_with_new_prompt`
- `try_another_frame`

对于 knobs、handles、switches、dials、buttons 这类小目标，playbook 应要求
agent 先抽取 affordance concept，再使用完整 task 约束生成 Molmo point prompt。

Molmo backend 必须通过配置注入。首版可以包装现有本地 Molmo smoke 路径；生产
配置要允许使用更适合 point 的 backend，例如运行环境中可用的 MolmoPoint-8B。
如果请求的模型不可用，工具必须显式失败，不能默默切换到更弱模型。

### SAM Mask Tool

`sam_mask` 接收 agent 已接受的 Molmo point，返回所有有意义的 mask candidates：

```json
{
  "frame_id": "000050",
  "points": [{"x_px": 1166.4, "y_px": 1188.48}],
  "candidates": [
    {
      "candidate_id": "mask_00",
      "score": 0.8247,
      "pixel_count": 1119,
      "coverage_percent": 0.0405,
      "overlay_path": "/abs/path/overlays/mask_00.jpg"
    }
  ],
  "contact_sheet_path": "/abs/path/overlays/sam_candidates_000050.jpg"
}
```

Agent 必须查看 contact sheet，并做出以下决策之一：

- `accept_candidate`
- `retry_with_different_point`
- `retry_with_crop`
- `try_another_frame`
- `reject_case`

SAM 不能只返回最高分 candidate。候选选择策略必须显式记录。支持策略包括：

- `highest_score`
- `smallest_non_empty`
- `area_range`
- `crop_local`
- `agent_selected`

如果环境里安装并验证了 SAM2.1-Hiera-L，优先使用它。原始 SAM ViT-H 可以作为
可复现实验 baseline，但不能作为隐藏 fallback。

### 3D Lifting Tool

`lift_mask_to_3d` 把 agent 接受的 2D mask 通过 depth、intrinsics 和 camera pose
投影到 scene coordinates：

```json
{
  "frame_id": "000050",
  "candidate_id": "mask_00",
  "lifted_point_count": 1119,
  "mask_npz_path": "/abs/path/mask_data.npz",
  "mask_ply_path": "/abs/path/lifted_points.ply",
  "overlay_path": "/abs/path/overlays/selected_mask_000050.jpg"
}
```

Agent 必须查看 selected-mask overlay 和 lifted artifact summary，才能把该 mask
作为 accepted seed。

长期更理想的 artifact 是和 scene point cloud 或 ConceptGraph point index space
对齐的 mask。首版可以同时保存 lifted coordinates 和 nearest scene-point ids，
这样 visualization 和 scoring 可以分开演进。

### Multi-View Expansion Tool

第一个 3D seed 被接受后，`suggest_additional_views` 根据 seed 找其他可能有用的
frames。判断信号包括：

- seed region 的 camera coverage
- 预计遮挡情况
- depth availability
- view angle diversity
- first lift 是否稀疏或只覆盖局部
- 功能部位是否可能存在未看到的背面或侧面

是否扩展由 agent 决定：

- 如果 first lift 已经足够覆盖目标部位，则停止。
- 如果存在遮挡、depth 缺失、点云稀疏、几何不完整，则继续补充视角。

每个新增视角必须重复同一条 approval path：

```text
Molmo point -> agent approval -> SAM candidates -> agent approval -> 3D lift -> agent approval
```

### Mask Fusion Tool

`fuse_accepted_masks` 只能融合已经被 agent 接受的 3D fragments。可用信号包括：

- nearest scene-point ids
- connected components
- spatial proximity
- ConceptGraph object membership
- per-view confidence and coverage metadata

Fused artifact 必须保留所有 per-view fragments，保证失败时能回溯到具体 point、
SAM candidate 或 lift 阶段。

### Inspection Tool

`inspect_mask_artifact` 汇总任意中间或最终 artifact，并返回可供 `view_image`
查看的图片路径：

- Molmo point overlay
- SAM all-candidate contact sheet
- selected-mask overlay
- lifted-mask stats
- optional BEV 或 point-cloud projection preview

这是 agent 用于 approve、reject、retry 的主要检查工具。

## Artifact Layout

每个 sample run 写入一个 durable artifact directory：

```text
<output_dir>/<visit_id>/<desc_id>/
  summary.json
  events.jsonl
  raw_outputs/
    molmo_<frame_id>_<attempt_id>.txt
  overlays/
    frame_<frame_id>.jpg
    crop_<frame_id>_<attempt_id>.jpg
    molmo_points_<frame_id>_<attempt_id>.jpg
    sam_candidates_<frame_id>_<attempt_id>.jpg
    selected_mask_<frame_id>_<attempt_id>.jpg
  fragments/
    <frame_id>_<candidate_id>/
      mask_data.npz
      lifted_points.ply
      summary.json
  fused/
    mask_data.npz
    lifted_points.ply
    summary.json
```

`summary.json` 记录：

- sample id、visit id、desc id、task description
- model backends 和 checkpoints
- selected frames 和 crops
- Molmo prompts、raw output paths、parsed points、approval decisions
- SAM candidate metadata 和 approval decisions
- lift stats 和 approval decisions
- multi-view expansion decisions
- fusion inputs 和 output paths
- final status；如果失败，则记录 failure type

如果一个 case 有多个 Molmo points 或多个 SAM candidates，artifact 必须把它们
放在同一张图中展示，方便 agent 和人类复盘。

## Final Answer Contract

Agent final answer 使用紧凑 JSON：

```json
{
  "status": "success",
  "sample_id": "421254::<description_id>",
  "mask_artifact_path": "/abs/path/summary.json",
  "mask_npz_path": "/abs/path/fused/mask_data.npz",
  "mask_ply_path": "/abs/path/fused/lifted_points.ply",
  "selected_frame_ids": ["000050"],
  "accepted_fragment_ids": ["000050_mask_00"],
  "confidence": "medium",
  "uncertainties": ["single-view mask may miss hidden geometry"]
}
```

失败时，final answer 要包含标准化 failure type 和最新 artifact path，保证失败
run 仍然可检查。

## 错误处理和重试策略

标准 failure types：

- `no_relevant_frame`
- `molmo_point_off_target`
- `molmo_point_ambiguous`
- `sam_empty`
- `sam_over_segmented`
- `sam_wrong_part`
- `lift_depth_missing`
- `lift_too_sparse`
- `multiview_fusion_failed`
- `artifact_invalid`

标准 run stop reasons，和模型阶段 failure type 分开记录：

- `single_view_complete`
- `multiview_not_needed`
- `max_retry_budget_reached_with_partial_mask`

重试规则：

- 不要用完全相同参数重复昂贵的 Molmo 或 SAM 调用，除非上一次是进程级失败。
- 放弃 case 前，优先尝试 crop refinement 或 prompt refinement。
- 每个阶段的 retry 上限由强类型 runner config 控制。
- Rejected points、masks、lift fragments 不能覆盖，必须记录下来。
- 进程失败和模型质量失败要分开记录。

## Evaluation

Evaluation 和 generation 分离。

Process evaluation：

- Agent 是否在引用证据前查看了图片。
- Agent 是否在 SAM 前审核了 Molmo points。
- Agent 是否在 lift 前审核了 SAM candidates。
- Agent 是否在 multi-view expansion 前审核了 first lift。
- 每个阶段发生了多少次 retry。
- 失败发生在哪个阶段。

Mask evaluation：

- 从最终 artifact 读取 predicted point ids 或 nearest scene points。
- 根据 sample 的 annotation ids 读取隐藏 GT annotation indices。
- 报告 IoU、precision、recall、F1、predicted point count、GT point count、
  failure type。

Scorer 必须同时支持 single-fragment artifact 和 fused multi-fragment artifact。

## 测试策略

不依赖 GPU 的 tests：

- `descriptions`、`motions`、`annotations` 的 sample loading
- task id construction
- hidden-GT 和 agent-visible sample context 分离
- Molmo raw text parsing
- SAM candidate metadata parsing
- state transition validation
- artifact schema validation
- retry 和 failure-type handling
- scorer metric math

可选 integration tests：

- 基于已保存 smoke artifacts 测试 wrappers
- 在 worker tmux session 中对一个手动选择的小 case 跑 Molmo/SAM/lift
- 对一两个 descriptions 比较 single-view 和 multi-view artifacts

任何 Python 代码变更都必须运行 `AGENTS.md` 中的质量门：

```bash
ruff check src/
black src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

如果 optional tests 需要 heavy vision dependencies 或 GPU，run record 必须明确
说明。

## 实现标准

所有生产代码必须遵循 `docs/python_code_agent_quality_guide.md`。

设计要求：

- samples、tool inputs、tool outputs、artifact metadata、state transitions、
  scoring records 使用 `dataclass(frozen=True)` 或 Pydantic v2 models。
- 除反序列化边界外，不使用 `Any` 或 untyped dictionaries；外部输入进入边界后
  立即转换为强类型模型。
- 每个 CLI JSON payload 必须先验证，再执行工具。
- CLI dispatch 保持薄；业务逻辑放到强类型模块中。
- Molmo、SAM、image、point cloud 相关 heavy optional dependencies 必须 lazy
  import。
- model paths 和 backend names 通过强类型配置注入。
- 不在业务逻辑里散落读取 `os.environ`。
- 不记录 secrets、tokens、private keys 或完整私有数据。
- 请求 best model 时不能默默 fallback 到 baseline model。Fallback 必须在配置中
  显式指定，或者作为失败返回。

## Milestones

P0：loaders 和 schemas。

- 增加 typed sample loader 和 task models。
- 增加 artifact 和 state-machine models。
- 增加 sample loading 和 hidden-GT separation tests。

P1：evidence tools。

- 增加 SceneFunc3D `scene_summary`、`keyframe_selector`、`view_frame`、
  `view_crop`、`view_bev`、`frame_objects`。
- 复用 OpenEQA/NR3D 模式，但保持 SceneFunc3D entrypoint 独立。

P2：Molmo、SAM、lift tools。

- 把 benchmark smoke 逻辑重构成强类型生产模块。
- 增加 point、mask、lift、inspection、artifact-writing tools。
- 保留 raw outputs 和所有 candidates。

P3：agent runner 和 approval gates。

- 增加 SceneFunc3D playbook。
- 在代码层强制 state transitions。
- 在准备好的两个 scenes 上跑小规模 single-view smoke。

P4：multi-view expansion 和 fusion。

- 用 first accepted 3D seed 推荐额外 frames。
- 每个 view 都要求 approval gates。
- 融合 accepted fragments，生成最终 mask artifact。

P5：scoring 和 benchmark record。

- 对隐藏 annotation indices 计算分数。
- 对有意义的 runs，在 `docs/benchmark/scenefunc_molmo_sam3d/` 写 durable
  benchmark records。

## Open Risks

- 当任务需要功能推理而不是简单 object naming 时，Molmo 可能 point 到错误部位。
- SAM 可能返回多个看起来合理的 masks，其中高分 mask 可能是错误的大面积表面，
  尤其是 handles、knobs、switches 等小部位。
- Single-view depth lifting 可能因为遮挡或 depth 缺失漏掉隐藏几何。
- Multi-view fusion 如果 seed projection 找到相似但错误的部件，可能引入噪声。
- macOS 和 Linux worker 上 best-model backend 可用性可能不同。实现必须检测并
  报告配置 backend 不可用，不能静默切换模型。
