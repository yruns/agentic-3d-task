# 本机数据目录

本机统一数据入口是 **`/home/ysh/Datasets`**，它是指向
`/data1/ysh/Datasets` 的软链接。新任务和配置应使用这个共享入口，
不要把其他项目 checkout 下的旧副本作为默认数据源。

以下位置在 **2026-11-02** 核查；换机器后必须重新确认，不能照搬绝对路径。

## 规范路径

| 数据 | 共享路径 |
|---|---|
| NR3D 原始标注 | `/home/ysh/Datasets/nr3d/raw/nr3d.csv` |
| NR3D 原始场景划分 | `/home/ysh/Datasets/nr3d/raw/train_scans.txt`、`test_scans.txt` |
| NR3D 已提取的 RGB / depth / poses | `/home/ysh/Datasets/nr3d/scannet/<scene>/raw/` |
| NR3D 实例、分割辅助标注 | `/home/ysh/Datasets/nr3d/scannet_aux/` |
| NR3D 已有 ConceptGraph 输出 | `/home/ysh/Datasets/nr3d/scannet/<scene>/conceptgraph/` |
| ScanNet 原始 `.sens` / mesh | `/home/ysh/Datasets/ScanNet/scans/` |
| OpenEQA 官方问题标注 | `/home/ysh/Datasets/open-eqa/data/open-eqa-v0.json` |
| OpenEQA ScanNet episode 帧 | `/home/ysh/Datasets/open-eqa/data/frames/scannet-v0/` |
| OpenEQA HM3D episode 帧 | `/home/ysh/Datasets/open-eqa/data/frames/hm3d-v0/` |
| OpenEQA 原始下载归档 | `/home/ysh/Datasets/open-eqa/downloads/` |
| OpenEQA 已有 ConceptGraph 输出 | `/home/ysh/Datasets/OpenEQA/scannet/<clip>/conceptgraph/` |

注意大小写：`open-eqa` 是官方数据 checkout；`OpenEQA` 是已有处理结果。
原始 `.sens`、提取后的帧、任务标注、ConceptGraph 输出是不同层级的数据，
不能直接互相替代。

## NR3D 集中复制记录

来源是旧项目的 `/home/ysh/codecase/3DVLMReasoning/data/nr3d/`，目标为
`/home/ysh/Datasets/nr3d/`。复制包含 `raw/`、`scannet/`、`scannet_aux/`、
`splits/`，保留已有场景处理结果，避免只复制标注而遗漏帧和索引。

- **复制而非移动**：旧副本保留以兼容旧项目，但不再作为新任务的规范入口。
- 复制时解引用源软链接，目标目录不依赖旧项目中的文件或链接；数据集内部
  的硬链接关系可保留，不与旧项目跨文件系统共享硬链接。
- 先复制到同一磁盘的临时目录，逐文件进行 rsync checksum / 元数据校验，
  无差异后再发布为正式 `nr3d/` 目录。
- 详细记录、统计和校验差异文件位于
  `/home/ysh/Datasets/.manifests/nr3d-copy-20261102/`。

OpenEQA 官方数据、episode 帧和 ScanNet 源数据原本就位于
`/data1/ysh/Datasets`，即上述共享入口内，因此没有再创建重复副本。

## 使用边界

- 当前仓库没有新增 `data/` 目录或数据副本；运行时显式传共享路径，
  不要通过旧项目寻找数据。
- NR3D 已有数据不包含当前 loader 默认要求的
  `pack_nr3d_v9_catalog_first`。原始数据与旧版处理结果存在，不代表该评测包
  已准备好；不要静默改用其他 pack。
- 本次不重新生成处理结果，不改写原文件中的历史绝对路径。它们是 provenance；
  实际读取应以新的共享根目录及当前 loader 的路径规则为准。
- 已核查目录中未找到 `SceneFuncVal-CG`。不要把其他机器的 NAS 路径、
  合成 smoke 图片或未经验证的原始 SceneFun3D 数据当作该预处理数据包。
- 本次是数据集中整理及校验，没有运行 benchmark 或远程模型调用。
