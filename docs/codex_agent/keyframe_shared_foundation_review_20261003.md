# Keyframe Selector 统一基础组件缺口（待处理）

- 记录日期：2026-10-03（Asia/Shanghai）。
- Review 基线：`feat/realtime-model`，提交 `d4f15fda3e16811fb07a9c0d47a594968e701d97`。
- 状态：**已记录，暂缓实施；架构问题尚未解决。**
- 范围：核对 OpenEQA、NR3D、SceneFun3D 是否实际依赖共享选帧方法。本次只记录问题，不修改运行逻辑，不重跑 benchmark。

## 论文定位与判定标准

论文第一章以 keyframe selector 为连接 3D 场景与 LLM 的基础组件，后续的
OpenEQA、NR3D 和 SceneFun3D 应建立在它产生的视觉证据之上。

需要分别判断三个层次：

1. 是否复用 `keyframe` 包中的数据加载、对象模型或 BEV 绘图代码。
2. 是否提供调用共享 `KeyframeSelector.select_keyframes_v2()` 的入口。
3. 下游是否实际消费该方法产生的证据，并能追溯其来源与贡献。

只有包复用、工具同名，或者在提示词中建议调用一次，都不足以证明第三点。

## 当前结论

**当前仓库尚未形成三个下游统一依赖第一章选帧算法的架构。**
已经共享的是 Codex 运行框架；语义检索和视觉证据获取仍存在独立实现与旁路。

| 下游 | 共享选帧算法接入情况 | 当前证据路径 | 与论文定位的差距 |
|---|---|---|---|
| OpenEQA | 自由文本 `keyframe_selector` 工具调用共享算法 | 零初始图片；Agent 也可直接取帧、裁图、查看对象和 BEV | 共享选帧是可选路径，没有必经或实际消费约束 |
| NR3D | 工具模式中的自由文本检索调用共享算法；工具模式本身可关闭 | 初始提供 BEV 与 proposal catalog；推荐使用空间比较、按 proposal 取共视帧和画框确认 | 共享算法处于辅助位置，主流程可绕过 |
| SceneFun3D | 同名工具没有调用共享算法 | 独立的帧编号匹配、对象关键词匹配、均匀覆盖；种子确认后另做几何扩视角 | 尚未接入第一章方法 |

上述结论来自调用链与接口核对。代码允许绕过不等于每个样本都绕过；真实调用率、
实际查看的图片及最终采用的证据，需要另行按样本 trace 统计。

## 代码证据

以下源码路径与行号对应 review 基线提交；该提交尚未推送，故使用仓库内路径记录。
后续修改后应重新核查，不能把本记录当作永久现状。

### OpenEQA：共享算法是可选取证工具

- Runner（`src/codex_agent/evaluation/openeqa_runner.py:148`）
  直接构造任务并调用 runtime，没有先执行共享选帧。
- 任务输入与规则（`src/codex_agent/openeqa/qa.py:94`）
  设置 `image_paths=()`，要求 Agent 自主取证，但允许不同工具入口。
- 自由文本检索（`src/codex_agent/openeqa/tools/keyframe_retrieval.py:151`）
  确实调用 `select_keyframes_v2()`。
- 对象列表（`src/codex_agent/openeqa/tools/object_tools.py:73`）
  创建 selector 后只读取对象；
  BEV 工具（`src/codex_agent/openeqa/tools/bev_tools.py:57`）
  只调用绘图方法。这些属于包复用，不能算选帧算法执行。

### NR3D：catalog 路径可以独立完成任务

- Runner（`src/codex_agent/evaluation/nr3d_runner.py:135`）
  加载场景后直接启动任务；
  初始请求（`src/codex_agent/nr3d/grounding.py:111`）
  附带 BEV，提示词另提供 proposal catalog。
- Playbook（`src/codex_agent/nr3d/playbook.py:44`）
  把自由文本 selector 定位为 catalog 候选不明确时使用；推荐流程主要使用 catalog、
  空间比较和 `select_by_proposal`。
- 按 proposal 取帧（`src/codex_agent/nr3d/tools/frame_tools.py:96`）
  从可见帧集合取交集或并集，不经过共享语言选帧算法。

### SceneFun3D：初始检索与后续扩视角均为独立实现

- 同名工具实现（`src/codex_agent/scenefunc3d/tools/keyframe_retrieval.py:243`）
  依次执行显式 frame ID 匹配、对象关键词匹配和均匀覆盖，不调用共享 parser、
  query executor 或联合覆盖选帧。
- 种子后的流水线（`src/codex_agent/scenefunc3d/pipeline.py:203`）
  调用独立的 `select_scene_visible_frames()`，按三维锚点的可见性选择后续帧。
- BEV 备用绘图（`src/codex_agent/scenefunc3d/tools/frame_views.py:551`）
  复用 `keyframe.bev.schematic`，不构成主选帧算法接入。

### 公共接口：没有统一的证据消费契约

CodexTask（`src/codex_agent/tasks/base.py:23`）
只要求构造请求、验证和解析响应；
CodexTurnRequest（`src/codex_agent/models.py:34`）
接收 prompt、图片与 schema，不要求共享检索的证据记录。

KeyframeResult（`src/keyframe/models/results.py:122`）
可以作为统一输出的起点，但当前尚不是三个任务共同消费的跨数据集证据接口。

## 对论文实验归因的影响

- NR3D 的整套 Agent 工具收益不能直接归因于共享 selector。
  [v4 网络修复记录](../benchmark/nr3d/v4_network_fix_strat600_20260610.md)
  记载：v3 的 selector 因网络问题失效，v4 恢复后，同一 strat600 上的 Overall
  都是 78.33%。这说明替代路径可以支撑该实验结果；不证明 selector 普遍无效。
- [OpenEQA v3/v4 记录](../benchmark/openeqa/README.md)
  比较的是有无 8 张初始图片，不能作为论文 selector 相对均匀/随机检索的单变量消融。
- [SceneFun3D v8.1 记录](../benchmark/scenefunc_molmo_sam3d/v8_1_full20_421254_20260703.md)
  对应其独立检索和下游流水线，不能作为共享 selector 已跨任务验证的证据。

上述数字是已有实验归档，本次没有重跑，也没有计算当前实现的调用比例或效果增益。

## 后续收敛方向（未实施）

建议目标依赖为：数据集适配层 → 共享 keyframe selector → 统一证据包 → 各下游任务。
以下待办用于后续设计，不代表当前已支持这些能力。

- [ ] 定义跨数据集场景输入：稳定对象/帧 ID、RGB/depth/pose/intrinsics 对应关系、
  坐标约定、可见性与可选 BEV；把文件布局适配与选帧算法分离。
- [ ] 定义统一证据包：逐帧身份与路径、目标/参照对象、选择依据、检索器版本、
  hypothesis/fallback 来源，以及关联的检索 ID。
- [ ] 让三个任务的首轮候选证据由共享方法生成并实际消费；可保留 OpenEQA 的零初始图片模式，
  但需通过运行时和工具接口建立可核验的依赖，不能仅靠提示词要求调用。
- [ ] 接入 SceneFun3D adapter；区分父对象/上下文命中与真实操作部位命中。
  当前独立检索器保留为明确命名的 baseline，而非混作论文方法。
- [ ] 将后续语言补充检索收敛到共享接口；保留读取、裁图、画框、Molmo/SAM 和种子后几何扩视角，
  记录这些操作与初始证据的关系及额外成本，不要求每次读取图片都重跑 parser。
- [ ] 固定下游模型、对象信息、图像分辨率与检索预算，对比均匀采样、可见性选帧、
  catalog 检索和论文方法；分别报告独立检索指标、下游效果、延迟与实际证据使用情况。

## 关闭问题的验收条件

三个任务都能追溯首轮证据到共享方法；改变检索策略时，下游实际消费的证据同步改变；
替代路径和 fallback 有明确标记；集成测试与受控消融能够验证依赖和贡献。
仅统一工具名称、包目录，或调用后丢弃返回结果，不算完成。

本记录只覆盖架构依赖与论文归因问题。排序、帧映射、图片路径隔离等具体实现问题，
应在后续修复时另行界定范围并验证。
