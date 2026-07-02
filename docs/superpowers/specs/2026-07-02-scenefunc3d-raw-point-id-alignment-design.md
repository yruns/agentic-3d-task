# SceneFunc3D Raw Point ID 对齐设计

日期：2026-07-02

## 目标

修复 SceneFunc3D 当前 scored run 中 prediction 与 GT 点索引空间不一致的问题。
本次改动只解决确定性的对齐层：

- final `mask_data.npz.point_indices` 必须是 `raw/mesh.ply` 的 vertex ids。
- scorer 继续把 annotation `indices` 当作 GT raw/source scan ids。
- 缺少 raw mesh、点云最近邻失败、或产物无法证明在 raw id 空间内时，直接失败。

这次不优化 agent prompt、SAM candidate 选择、多视角策略或功能部位质量。对齐修复
之后，真实的模型/工具失败会被保留下来，作为后续质量改进的输入。

## 当前问题

`tmp/scenefunc3d/artifacts/scenefunc_parallel20_20260701_2346` 的 20 case
结果显示：

- direct scorer mean IoU 为 `9.866473722291653e-06`。
- 只有 `1/20` 个样本 IoU 非零。
- 每个样本都有非空 `predicted_count` 和 `gt_count`，不是空 mask 问题。

只读诊断确认了索引空间错位：

- `data/SceneFun3D/421254/raw/mesh.ply` 顶点数为 `2,705,592`。
- `data/SceneFun3D/421254/conceptgraph/mesh.ply` 顶点数为 `2,546,020`。
- `421254_annotations.json` 的 annotation `indices` 范围为 `3837..2694883`，
  全部落在 raw mesh 范围内，但有大量 index 超出 conceptgraph mesh 顶点数。
- 当前 `lift_mask_to_3d` 通过 dispatcher 固定使用
  `tool_scene.conceptgraph_dir / "mesh.ply"` 做最近邻 assignment。
- `_map_filtered_mesh_indices_to_source_scan_ids` 需要 `<visit_id>_crop_mask.npy`
  才能把 conceptgraph filtered index 映射回 source scan id；当前
  `data/SceneFun3D/421254/421254_crop_mask.npy` 不存在。
- 找不到 crop mask 时，当前代码直接 `return point_indices`，把 conceptgraph
  index 写成 final `point_indices`，scorer 随后把它当 raw GT id 比较。

临时把同一批 final `points_world` 最近邻到 `raw/mesh.ply` 后重算，mean IoU
恢复到约 `0.1649`，`16/20` 个样本非零。多数 nearest distance 是毫米级，
说明 lift 后的 world coordinate 基本对齐；主要错误是写出的点 id 空间不对。

## 非目标

- 不改 IoU/precision/recall/F1 的数学定义。
- 不让 scorer 用 `points_world` 兜底修正错误 artifact。
- 不依赖缺失的 `crop_mask.npy` 作为当前数据集的必需输入。
- 不调整 Molmo/SAM server 协议。
- 不增加 prompt rule 来解决 window mask 过大、相邻 drawer 混淆或左右 remote
  混淆。
- 不引入新的 benchmark 启动方式或可选评分模式。

## 设计决策

### 1. final artifact 的唯一评分索引空间是 raw mesh

SceneFunc3D final mask artifact 必须满足：

```text
mask_data.npz.point_indices[i] == raw/mesh.ply vertex id
```

`point_indices` 不再允许表示 conceptgraph filtered mesh ordinal。这个契约由
lift tool 写入时保证，由 final artifact validation 和 scorer 消费时共同检查。

`points_world` 仍表示 2D mask 中正深度像素反投影得到的 world-space XYZ 点。
它用于可视化和审计，但不作为正式 scoring id。scoring id 必须显式写在
`point_indices` 中。

### 2. lift 阶段直接对 raw mesh 做最近邻 assignment

`lift_mask_to_3d` 的主要数据流调整为：

```text
SAM mask + depth + intrinsics + camera_to_world
  -> points_world
  -> nearest neighbor against data/SceneFun3D/<visit_id>/raw/mesh.ply
  -> raw_point_indices
  -> mask_data.npz(points_world, point_indices=raw_point_indices)
```

dispatcher 不再把 `conceptgraph/mesh.ply` 作为 scoring assignment mesh 传给
`lift_mask_to_3d`。它需要提供 raw mesh path，或提供一个强类型 scene geometry
对象，其中明确区分：

- conceptgraph scene directory：用于 keyframe、view、inspection、agent evidence。
- raw mesh path：用于 scoring id assignment。

如果 raw mesh 不存在或不可读，tool 直接失败。不要 fallback 到 conceptgraph mesh。

### 3. crop-mask 映射逻辑不作为本次主路径

保留 filtered-to-source 映射可以作为未来兼容路径，但它不能在当前数据缺失
`crop_mask.npy` 时静默成功。若实现中仍保留 `_map_filtered_mesh_indices_to_source_scan_ids`，
它的规则必须改为：

- 找到 crop mask 时，严格校验 dtype、shape、true count 和 filtered mesh vertex count。
- 找不到 crop mask 且调用方仍要求 filtered-to-source mapping 时，抛出错误。
- 不能返回原始 filtered `point_indices` 作为 scoring ids。

采用 raw mesh 直接 assignment 后，正常 SceneFun3D run 不需要调用这个 mapping。

### 4. scorer 不负责修复错误产物

scorer 的职责保持简单：

- 读取 final NPZ 的 `point_indices` 或 `point_ids`。
- 读取 GT annotation `indices`。
- 比较两个 raw id set。

scorer 可以增加边界校验，例如在有 `data_root` 和 `sample_id` 的上下文中，确认
predicted ids 不超过 raw mesh vertex count。它不应该在发现 `points_world` 时自动
最近邻 raw mesh 后评分，因为那会掩盖 artifact contract 错误，并让 inspection、
fusion、复现实验继续使用错误 id。

## 错误处理

所有对齐失败都必须 fail closed：

- raw mesh path 缺失或不是文件。
- raw mesh PLY 无法读取、vertex 数为 0、或坐标非法。
- lifted `points_world` 无法在阈值内匹配 raw mesh 最近邻。
- 写入的 `point_indices` 不是一维整数数组。
- `point_indices` 数量与 `points_world` 数量不一致。
- 任一 `point_indices` 小于 0 或大于等于 raw mesh vertex count。
- filtered-to-source mapping 被调用但缺少有效 crop mask。

错误消息要带上 sample/visit、mesh path、npz path、最大距离或越界 index 等定位信息，
但不能打印私有 header、AK、cookie 或完整 remote credentials。

## 测试设计

新增或调整不依赖 GPU 的测试，覆盖以下行为：

1. `lift_mask_to_3d` 对 raw mesh 做 assignment，写出的 `point_indices` 等于 raw
   nearest-neighbor ids。
2. dispatcher 为 lift tool 传入 raw mesh path，而不是 conceptgraph mesh path。
3. 缺少 raw mesh 时，lift tool 失败。
4. 缺少 crop mask 时，filtered-to-source mapping 不再 no-op 成功。
5. final artifact validation 能捕获 `point_indices` 数量、dtype、维度和越界错误。
6. scorer 继续用 raw id set 计算指标，不从 `points_world` 自动修正。

测试数据应使用临时小 PLY 和小 NPZ，不依赖本机绝对路径或真实 SceneFun3D 大数据。

## 验证计划

实现后先做快速验证：

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d*.py -q
ruff check src/
black --check src/
mypy src/
```

然后用当前 20 case 重新跑或重新 lift 后 score，预期 direct scorer 与诊断期 rawNN
重算结果同量级：

- mean IoU 接近 `0.165`。
- 非零样本数接近 `16/20`。
- drawer/nightstand/left remote 中多例从 0 变为明显非零。

这个结果只证明点 id 对齐修复有效，不代表 SceneFunc3D 任务已经达到最终质量要求。
600 case benchmark 应在这个修复后再跑，作为真实质量基线。

## 对齐后仍会保留的失败类型

对齐修复后，以下失败仍会存在，并且应该在后续质量阶段处理：

- window mask 过大：left/right window 的 SAM mask 覆盖大面积窗帘或窗户区域，
  precision 很低。
- 相邻 drawer ordinal 混淆：部分 TV cabinet drawer 仍可能选到相邻 drawer。
- 左右 remote 混淆：right remote case 在 rawNN 诊断中仍为 0，空间距离约
  `2.97m`，更像选错目标。
- 局部或边界覆盖不足：door、drawer 的部分 case 有交集但 IoU 偏低，可能只覆盖
  handle、edge 或局部面。

这些是 agent/SAM/多视角策略问题，不应该通过 scorer 或 point-id 对齐层隐藏。

## 文档和 benchmark 记录

`docs/superpowers/**` 被 `mkdocs.yml` 排除，是内部设计和计划记录；新增本设计不需要
更新 MkDocs navigation。

实现并验证后，需要在 `docs/benchmark/scenefunc_molmo_sam3d/` 追加一条 durable run
记录，说明：

- 修复前 direct scorer 的错位指标。
- 修复后 20 case 或 600 case 的 raw-aligned direct scorer 指标。
- 仍失败样本的分类，避免把二级质量问题误判成对齐问题。

## 采用方案

采用方案 1：lift 阶段直接输出 raw/source scan ids。

不采用方案 2：继续对 conceptgraph mesh assignment 后依赖 crop mask。当前数据缺少
`crop_mask.npy`，数据依赖脆弱。

不采用方案 3：在 scorer 中用 `points_world` 临时 rawNN 打分。它适合诊断，但会让
artifact contract 保持错误，不适合作为正式修复。
