# SceneFun3D 锚点中心多视角 Mask 生成设计

日期：2026-07-02
状态：设计已确认，待实现
分支：`feat/scenefunc3d-agent-tools`

## 目标

把 SceneFun3D 的 mask 生成从"自由 agent 循环 + 所有反投影点取并集"改造成
**锚点中心的多视角确定性管线**，核心解决**过度选择（over-selection）**导致的低
精度 / 低 AP：

- agent 只负责建立并**验证一个高质量种子**（选帧 → Molmo+SAM → 看 overlay →
  确认或纠正）。
- 种子确认后冻结一个 **3D 锚点（anchor）**；之后可见性投影、逐帧 Molmo+SAM、
  反投影 lifting、去噪融合全部是**确定性 runner 代码，不再调 LLM**。
- 锚点同时充当"可见性探针"（决定哪些帧真正看得到目标、注意遮挡）和"噪声过滤器"
  （把远离锚点的散点/错误实例剔除）。
- 融合方案（本设计的重点）采用**分层去噪**：多视角 agreement + 锚点半径门控 +
  DBSCAN 连通簇 + `motion_type` 尺寸上限。
- 全流程 **training-free**，且可在已保存的 completed artifact 上离线验证、不依赖
  ModelHub adapter；同时为"官方 ranked AP"和"TASA 式训练精修头"预留干净接口。

## 背景与当前问题

### 当前管线

现状是一次自由 ReAct turn：一条巨型 playbook 让 agent 自己 shell 调用
`molmo_point → sam_mask → lift_mask_to_3d → inspect → fuse_accepted_masks`，
runner 再对 self-reported 产物做大量事后校验。lifting
（`tools/mask_lifting.py::lift_mask_to_3d`）把 SAM mask 里**每一个有有效深度的
像素**反投影，映射到 `raw/mesh.ply` 上 0.05m 内最近的顶点，**没有任何 3D 聚类 /
尺寸 / 紧致度约束**（`backends/lift_3d.py::assign_nearest_scene_point_indices`）。

### 过度选择证据

最新 all-`421254` run（12 个 completed）：

- Mean IoU 仅 `0.1258`，**AP50 = 0.0**。
- 掩码紧且命中即高：最好一例 `55b56254` 预测 108 / GT 75 → IoU 0.476。
- 掩码宽即精度崩：`4668a5f5` 预测 1990 / GT 106 → precision 0.006；
  `6afc3b66` 561 / 105。
- raw mesh 有 270 万顶点、GT 仅 49–446 点（都是小 affordance）。

结论：raw point-id 对齐已修复，低 IoU 的主因是**几何层的过度选择**，不是点空间
错位。当前"多视角"是把各帧 lift 结果取**并集**（命中 ≥1 视角即保留），与抑制
噪声的方向相反。

### 借鉴的外部工作（论文 + 源码考古结论）

- **Fun3DU**（CVPR25，已开源，最同族）：真实融合是**逐点跨视角投票 + 阈值**，
  代码里等价于"保留 `vote > 0.7 × peak_vote` 的点"（`evaluate.py`），与我们的
  并集相反，是过度选择的直接解药；`PointCloudToImageMapper.compute_mapping`
  （投影 + 深度遮挡）可干净移植。
- **TASA**（AAAI26，已开源但残缺）："geometric refinement" 实为**训练的
  Point Transformer**（shipped 代码有 SyntaxError、无 checkpoint、loss 只有 Dice、
  用 SAM1）。但仓库里躺着一个**未被调用**的训练无关
  `crop_extraneous_points_from_point_cloud`（DBSCAN 最大簇 + bbox 裁剪）——正是
  我们要的几何先验。
- **UniFunc3D**（arXiv26，暂无代码）：**MLLM overlay 验证**"红色区域是否只包含
  目标部件"用于拒绝父物体 mask —— 对应我们阶段 A 的 agent 验证。
- 共同硬伤：遮挡 / 非附着 affordance（如遮挡的拔插头）在三篇里都是**已知未解**。

我们相对它们的**独有优势**：有一个 **agent 验证过的种子**，因此能做**锚点中心**
融合；Fun3DU/TASA 没有可信种子，只能用较弱的纯视角一致性。

## 非目标（本期）

- 不实现 TASA 式训练精修头（只留接口）。
- 不在本期完成官方 ranked AP 的接线（只留接口 + 置信度产出）。
- 不解决 ModelHub adapter 的加密状态稳定性（独立 workstream）。
- 不承诺解决遮挡 / 非附着 affordance（已知未解）。
- 不改 `metrics.py` 中 IoU/precision/recall/F1 的数学定义。
- 不引入 SAM/Molmo server 协议变更。

## 架构总览

```
[agent 在环 — 唯一 LLM 阶段]                 [确定性 runner — 不再调 LLM]
A. 种子 + 验证  ──确认──▶  B. 可见性投影  ──▶  C. 逐帧          ──▶  D. 分层融合 ──▶ 最终 mask
   ▲        │               把锚点投影到          逐可见帧            投票→锚点门控        (+置信度)
   └─纠正───┘               所有帧 + 遮挡测试      Molmo+SAM+lift      →DBSCAN→尺寸上限
   (≤3 次, 否则 fail-closed)                     + 锚点一致性门控
```

边界：agent 的输出从"大 final-JSON"缩减为一个**小的种子确认 JSON**（确认的
`frame_id` / `candidate_id` / affordance 概念 + 审批轨迹）；runner 接管 B/C/D 并
写出最终 artifact。这既是质量改造，也顺带把自由循环收敛成 FSM 驱动。

## 阶段 A — 种子建立 + agent 验证/纠正

**输入**：`task_description`、`motion_hints`（`motion_type` 如 `hook_turn` +
3D `motion_dir`）、`keyframe_selector` 候选帧。

**流程**：
1. agent 选一帧，`molmo_point`，提示词用 Fun3DU 形式
   `"point to the {affordance} in order to {完整任务}"`（去掉任务描述在 Fun3DU
   消融里掉 14.4 AP25）。
2. `sam_mask` → 候选 mask + overlay。
3. agent **看 overlay** 后二选一：
   - **确认**：部件对、实例对、mask 紧致。
   - **纠正**：换帧 / 重跑 Molmo / 重跑 SAM。最多 3 次，否则 fail-closed。
4. 确认后 `lift_mask_to_3d` 得到该 mask 的 raw-mesh 点；由这些点构造**锚点**：
   - `centroid` = 反投影点质心。
   - `radius` = 95 分位距质心距离，并被 `motion_type` 尺寸先验封顶。
   - 记录 `frame_id`、`candidate_id`、affordance 概念。

## 阶段 B — 可见性投影（新确定性工具）

新增 `SceneToImageProjector`（移植 Fun3DU/TASA 的 `PointCloudToImageMapper`
模式）：

1. 取锚点球的代表点（质心 + 球内采样点），用内参 + 位姿投影到**每一帧**。
2. **深度遮挡测试**：`|投影深度 − 实测深度| ≤ 0.25 · 实测深度`，只保留锚点
   **真正未被遮挡**的帧（对应"注意遮挡"）。
3. 帧质量分 = 未遮挡比例 · 居中度 · 距离 · 视角朝向；按分排序取 **top-30**，
   并**始终包含种子帧**。
4. 同时产出每个 raw 顶点的"可见次数"计数，供阶段 D 的 agreement 归一使用。

## 阶段 C — 逐帧 Molmo+SAM+lift（确定性）

对每个选中帧：
1. 把锚点投影到该帧 → 期望 2D 点。
2. `molmo_point`（同一 affordance 概念）→ 取**离投影锚点最近**的点；若 Molmo
   返回空 / 全部太远，则**显式记录**并用投影锚点作为 SAM 提示点兜底（这是设计内
   的确定性行为并写入 provenance，不是静默降级）。
3. `sam_mask` → 选**最紧致**的候选（T-FunS3D 的最小 mask 技巧）。
4. `lift_mask_to_3d` → raw-mesh 顶点 id + 逐帧 provenance。
5. **锚点一致性门控**：该帧 lift 质心距锚点 > radius 则整帧丢弃（该帧点错 / 命中
   错误实例）。

## 阶段 D — 分层融合（本设计重点）

输入：所有被接受帧的 lifted 顶点集合 + 阶段 B 的逐顶点可见次数。

1. **多视角 agreement**：逐顶点 `agreement = 命中帧数 / 可见帧数`。
2. **锚点门控**：保留 `agreement ≥ τ` **且**落在锚点 radius 内的顶点。
3. **几何精修**：对存活顶点做 `DBSCAN(eps)`，保留**含锚点的连通簇**；可选
   `motion_type` 尺寸上限（bbox 最大延展超限则拒绝）。
4. **输出**：保留顶点 id = 最终 mask；**置信度 = 保留顶点平均 agreement**。
   DBSCAN 的次级簇可作为 ranked 实例候选（供官方 AP，见下）。

参数 `(τ, radius, eps, 尺寸上限)` 由**离线扫参**确定（见"参数与离线扫参"）。

## 数据结构与接口

全部强类型（`dataclass(frozen=True)` 领域对象；边界用 Pydantic v2）：

- `TargetAnchor`：`centroid: tuple[float,float,float]`、`radius_m: float`、
  `seed_frame_id: str`、`seed_candidate_id: str`、`affordance_concept: str`、
  `motion_type: str`。
- `FrameVisibility`：`frame_id`、`visible: bool`、`unoccluded_fraction: float`、
  `quality_score: float`、`projected_xy: tuple[float,float] | None`。
- `PerFrameLift`：`frame_id`、`candidate_id`、`point_indices: tuple[int,...]`、
  `centroid`、`accepted: bool`、`reject_reason: str`、`molmo_fallback_used: bool`。
- `FusedMask`：`point_indices: tuple[int,...]`、`confidence: float`、
  `instances: tuple[FusedInstance,...]`、`params: FusionParams`。
- `SeedConfirmation`（agent 输出 schema）：`frame_id`、`candidate_id`、
  `affordance_concept`、`approval_actions`。

## 模块 / 文件划分

新增（`src/codex_agent/scenefunc3d/backends/`）：

- `visibility.py`：`SceneToImageProjector`（3D→2D 投影 + 深度遮挡）、帧质量打分、
  逐顶点可见计数。复用 `lift_3d.py` 的 `CameraGeometry` 与矩阵读取。
- `anchor.py`：由确认后的 lift 结果构造 `TargetAnchor`（质心 + 稳健半径 + 尺寸
  先验封顶）。
- `motion_priors.py`：`motion_type → 尺寸先验（半径 / bbox 上限）` 表。
- `fusion.py`：`fuse_multiview_points(...) -> FusedMask`（agreement + 锚点门控 +
  DBSCAN + 尺寸上限 + 置信度 + 实例）。

新增管线驱动（`src/codex_agent/scenefunc3d/`）：

- `pipeline.py`：`AnchorMultiViewPipeline`，在 agent 确认种子后确定性执行
  B/C/D，并写出最终 artifact。

改动：

- `runner.py`：`SceneFunc3dMaskTask` 的 turn 契约改为产出 `SeedConfirmation`；
  `run_single_sample` 在解析确认后调用 `AnchorMultiViewPipeline`。保留现有
  batch summary / failure artifact / raw-id 校验。
- `playbook.py`：改写为"只到种子确认为止"的精简 playbook（去掉后续多视角 / 融合
  的散文规则，那些改由 runner 保证）。
- `task.py`：现有 FSM 从"事后验证器"转为"种子阶段驱动 + 确认门"。
- `evaluation/`：`scorer.py` / `metrics.py` 不变（继续对单个 fused mask 打分）。

预留接口（本期只留 seam）：

- `evaluation/ap.py`：官方 ranked AP + RLE（移植 Fun3DU
  `scripts/sun3d/eval/eval_utils/eval_script.py` + `rle.py`），消费 `FusedMask`
  的实例 + 置信度。
- `fusion.py` 的阶段 3 crop（锚点 KNN 球 + 初始指示）即 TASA 训练精修头的输入，
  未来可整体替换 DBSCAN。

## 参数与离线扫参

| 参数 | 含义 | 初值 |
|---|---|---|
| `max_seed_attempts` | 种子纠正上限 | 3 |
| `frame_cap` | 可见帧保留上限 | 30 |
| `vis_thres` | 深度遮挡容差 | 0.25 |
| `tau` | agreement 阈值 | 0.5（扫 0.3–0.7） |
| `anchor_radius_percentile` | 锚点半径分位 | 95 |
| `dbscan_eps` | 聚类半径(m) | 0.02 |
| `dbscan_min_points` | 最小簇点数 | 10 |
| `size_cap_by_motion` | 各 `motion_type` bbox 上限 | 待离线标定 |

验证分三档，均**不碰 ModelHub adapter**：

1. **纯离线（无 sidecar、无 adapter）**：用 12 个 completed 已保存的
   `fragments/*/mask_data.npz`（`points_world` + `point_indices`）+ 已确认种子帧，
   跑**阶段 D 的锚点门控 + DBSCAN + 尺寸上限**并用现有 scorer 重打分。因老 run
   每 case 仅 1–2 帧，agreement 维度较弱，此档主要量化**几何精修**对
   precision/IoU 的恢复。
2. **半在线（需 Molmo/SAM sidecar，无 LLM adapter）**：把每个 case 的已确认种子
   当作固定锚点，跑完整 B/C/D（新选帧 + 逐帧 Molmo/SAM），验证完整管线与
   agreement 维度。
3. **全在线**：agent 也参与（需 adapter），最终 benchmark。

离线扫参脚本：`src/codex_agent/scenefunc3d/evaluation/offline_fusion.py`
（读取 run root + `run_home_map.json`，网格扫 `(tau, radius, eps, size_cap)`，
输出每组的 mean IoU / precision / AP25 / AP50）。

## 输出与评测接口

- **本期主输出**：单个 fused mask（`mask_data.npz` 带 `point_indices`），保持现有
  scorer 与 final-artifact 校验可用。
- **官方 AP seam**：`FusedMask.instances` + 置信度已产出，后续接
  `evaluation/ap.py` 即可算 AP25/AP50/mAP@0.5:0.95；置信度用保留点平均 agreement，
  避免 Fun3DU 那种 conf=1.0 退化。
- **训练精修 seam**：阶段 D 的锚点 crop 直接可喂 TASA 式 Point Transformer。

## 错误处理 / fail-closed（遵守仓库强制无兜底规则）

- 种子 3 次仍不通过 → 该 case 失败并写 failure artifact，不静默产出低质 mask。
- 阶段 B 无可见帧（仅种子帧可见）→ 退化为单帧管线并**显式标记**，不报错但记录。
- 锚点退化（点太少 / 半径非有限）→ 失败并给出清晰原因。
- 依赖缺失（numpy/scipy/Pillow、深度/位姿/内参文件、raw mesh）→ 立即失败，
  绝不用 bbox 矩形或 conceptgraph mesh 兜底。
- 逐帧 Molmo 兜底（用投影锚点点）**写入 provenance**，是设计行为而非静默降级。
- 最终 `point_indices` 必须在 `raw/mesh.ply` 顶点范围内（沿用现有 raw-id 校验）。

## 测试计划

- 单元测试：
  - `visibility.py`：合成一个已知位姿/深度的小场景，验证投影 + 遮挡判定 +
    可见计数。
  - `anchor.py`：给定 lift 点集验证质心/半径/尺寸先验封顶。
  - `fusion.py`：构造含离群点 + 双簇的合成点集，验证 agreement/锚点门控/DBSCAN/
    尺寸上限/置信度。
  - `motion_priors.py`：各 `motion_type` 的先验查表。
- 集成测试：`pipeline.py` 用 mock 的 Molmo/SAM/lift（固定返回）跑完 B/C/D，
  校验最终 artifact 结构 + raw-id 校验。
- 回归护栏：现有 `tests/test_scenefunc3d_*.py` 全绿；离线扫参脚本对 12 case
  产出可复现的指标表。
- 质量门：`ruff check src/`、`black src/`、`mypy src/`、
  `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_*.py -q`。

## 风险与未决问题

- **多实例目标**（如两个把手）：DBSCAN 只取锚点簇会漏掉第二个实例；官方 AP 接线
  时需允许多实例输出。缓解：阶段 D 保留次级簇作为候选实例。
- **坏种子**：agent 确认了错误部件，则整条确定性管线放大错误。缓解：种子验证
  提示词强调"红色区域是否只含目标部件"；必要时加一次 overlay 复核。
- **`motion_type` 尺寸先验覆盖**：需要对 9 类 motion 标定 bbox 上限，可能有长尾。
- **参数过拟合 12 case**：离线扫参易过拟合小集；结论需在半在线 / 更大集上复核，
  才能作为决策级结论。
- **半径单位与坐标系**：锚点半径在 raw mesh 世界坐标（米）下定义，需与
  `_MAX_SCENE_POINT_ASSIGNMENT_DISTANCE_METERS=0.05` 的量级一致。
