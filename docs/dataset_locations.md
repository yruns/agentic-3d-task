# 本机数据目录

本机统一数据入口是 **`/home/ysh/Datasets`**，指向 `/data1/ysh/Datasets`。
以下位置在 **2026-11-03** 核查；新任务使用共享目录，不再依赖其他项目 checkout。

## 当前评测使用 processed_v1

2026-11-02 传来的两个数据包和独立清单包已于 2026-11-03 解压。
**它们与之前保存的原始标注、原始帧、旧版处理结果分开，不覆盖旧数据。**

| 用途 | 路径 |
|---|---|
| NR3D 数据根目录，即 CLI `--data-root` | `/home/ysh/Datasets/nr3d/processed_v1/scannet/` |
| NR3D 全量清单，8,584 条 | `/home/ysh/Datasets/nr3d/processed_v1/folds/full_test_sample_ids.json` |
| NR3D canonical pilot，600 条 | `/home/ysh/Datasets/nr3d/processed_v1/folds/strat600_sample_ids.json` |
| NR3D proposal pack | `<data-root>/<scene>/pack_nr3d_v9_catalog_first/` |
| OpenEQA 数据根目录，即 CLI `--data-root` | `/home/ysh/Datasets/open-eqa/processed_v1/scannet/` |
| OpenEQA 问题/答案，即 CLI `--questions` | `/home/ysh/Datasets/open-eqa/processed_v1/open-eqa-v0.json` |
| OpenEQA pilot，300 条 | `/home/ysh/Datasets/open-eqa/processed_v1/folds/v5_pilot_fold_300.json` |

目录布局：

```text
/home/ysh/Datasets/
├── nr3d/
│   ├── raw/                         # 旧原始标注，未改动
│   ├── scannet/                     # 旧提取帧/处理结果，未改动
│   ├── scannet_aux/                 # 旧辅助标注，未改动
│   └── processed_v1/                # 本次 nr3d-data.tar
│       ├── raw/                     # 数据包自带标注；不是上面的 raw/
│       ├── scannet/<scene>/
│       │   ├── raw/
│       │   ├── conceptgraph/
│       │   └── pack_nr3d_v9_catalog_first/
│       ├── scannet_aux/
│       ├── scannet_aux_meshes/
│       ├── splits/
│       ├── folds/                   # 独立清单包里的 NR3D 清单
│       └── transfer_metadata/
├── open-eqa/
│   ├── data/                        # 官方原始数据，未改动
│   ├── downloads/                   # 旧下载归档，未改动
│   └── processed_v1/                # 本次 openeqa-data.tar
│       ├── open-eqa-v0.json
│       ├── scannet/<clip>/
│       │   ├── raw/
│       │   └── conceptgraph/
│       ├── folds/                   # 独立清单包里的 OpenEQA 清单
│       └── transfer_metadata/
├── ScanNet/scans/                   # 原始 .sens / mesh，未改动
├── OpenEQA/scannet/                 # 旧 ConceptGraph 结果，非 processed_v1
└── .manifests/processed-v1-import-20261103/
```

### 当前代码如何指定路径

通用 CLI 原本就要求显式数据路径，不需要修改加载器或硬编码本机默认目录。
新增路径 helper 只检查输入是否存在并导出变量，**不会运行评测、调用 API 或修改数据**：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
source scripts/datasets/processed_v1.sh
```

换机器/数据盘可在 source 前设置 `AGENTIC3D_DATASETS_ROOT`，保持上述相对布局。
CLI 不会自行读取这些变量，须按下面示例显式传参。

以下是**实际评测命令示例，本次没有执行**。启动前必须满足 `AGENTS.md` 中的
干净工作区、代码 SHA、tmux、有效 API 配置和 adapter health 前置条件。

NR3D canonical pilot：

```bash
python -m codex_agent.cli.run_nr3d \
  --sample-ids "$NR3D_PILOT_SAMPLE_IDS" \
  --data-root "$NR3D_DATA_ROOT" \
  --pack-name "$NR3D_PACK_NAME" \
  --output-dir tmp/nr3d_processed_v1_strat600 \
  --workers 1 --tools --reasoning-summary auto
```

全量评测将 `--sample-ids` 换为 `"$NR3D_FULL_SAMPLE_IDS"`，输出目录另取唯一 run ID。
该全量清单有 **8,584 条、130 个涉及场景**，不是历史过滤后 **7,805 条**的列，
不要把两者指标当作同一 fold 比较。

OpenEQA pilot：

```bash
python -m codex_agent.cli.run_openeqa \
  --questions "$OPENEQA_QUESTIONS" \
  --data-root "$OPENEQA_DATA_ROOT" \
  --question-ids "$OPENEQA_PILOT_QUESTION_IDS" \
  --output-dir tmp/openeqa_processed_v1_pilot300 \
  --workers 1
```

OpenEQA 全量运行省略 `--question-ids`，使用 **1,079 道 ScanNet 问题、89 个 clip**。
问题文件包含 HM3D 条目，但本次没有 HM3D 场景包，当前 loader 默认只取 ScanNet。
不能用原始官方问题文件替换数据包版本并假定 question ID/fold 一致。

### 清单与完整性

- NR3D `folds/`：`full_test_sample_ids.json`（8,584）、`strat600_sample_ids.json`
  （600）、`strat10_smoke_sample_ids.json`（10）、`historical_random100_sample_ids.json`
  （100）、`v7_v8_flip41_sample_ids.json`（41）。后两类诊断/历史清单不能替代 canonical pilot。
- OpenEQA `folds/`：`v5_pilot_fold_300.json`（300）、`smoke10_sample_ids.json`（10）。
- 两份 NR3D strat600 清单（大包自带/独立清单包）经当前 ID loader 比较，内容与顺序一致。
- 三个 tar 的 SHA-256 均与随包 `.sha256` 文件一致；先检查全部成员路径和文件类型，
  再在各自新建的空临时目录内以安全 data filter 解压，验证文件数/长度后发布。
- 原包目录中的 `TRANSFER_STATUS` 仍标 `INCOMPLETE`；未改写它。
  这里的成功结论基于本机重新计算的 SHA-256 和解压/加载结果，不依赖旧标记。

源归档保留在 `/home/ysh/openeqa-nr3d-transfer-20261102.NUyCRUIf/`：

| 包 | 常规文件数 | 文件内容字节数 |
|---|---:|---:|
| `nr3d-data.tar` | 99,993 | 31,258,871,949 |
| `openeqa-data.tar` | 187,704 | 77,399,618,006 |
| `evaluation-input-lists.tar` | 8 | 1,344,444 |

解压总内容约 **108.7 GB**，不包括文件系统开销和少量清单/记录副本。
源归档、独立清单包原始布局、SHA 和验证报告保留；未删除源包以腾空间。

永久记录在 `/home/ysh/Datasets/.manifests/processed-v1-import-20261103/`：

- `archive-audit.json`：每包 SHA、路径安全检查、文件数、解压大小。
- `extraction.json`：源归档、隔离暂存目录、最终发布位置。
- `loader-validation.json`：当前仓库 loader 对全部 ID、场景和 fold 的检查结果。
- `evaluation-input-lists/MANIFEST.json`：发送端说明及原始 fold 布局。

验证覆盖 **8,584 条 NR3D 样本**、130 套场景资产、**22,202 张引用 RGB 帧**；
OpenEQA 的 **1,079 道问题、89 个 clip、53,193 张 RGB 帧**；
每个 fold 的唯一性/成员关系、BEV 图片读取和旧绝对路径在新根目录下的 RGB 重定位。
PCD/visibility pickle 仅检查文件存在，未为了验收而反序列化不可信 pickle。
**这不是正式评测或完整 Agent 推理结果**；模型、LLM 凭据和实际任务仍需运行前检查。

额外数值检查发现 **13 个 OpenEQA clip 的轨迹共含 926 个非有限 4×4 位姿**
（例如 `103-scannet-scene0527_00` 的 `conceptgraph/traj.txt` 中第 333–339
个零基位姿为 `-inf`）。这些值存在于源包中，不是解压损坏；当前 JSON/scene
loader 可以加载，但使用这些位姿的 keyframe/BEV 几何工具可能受影响。
未删除帧、伪造位姿或修改源数据。具体 clip、文件和零基 pose 索引保存在
`nonfinite-poses.jsonl`。因此导入和路径验收通过，不代表几何数值完整性全通过。
若后续评测使用姿态相关工具，应单独明确无效位姿处理策略并留下新版本记录。

导入后目标盘只剩约 **40 GB** 可用空间；正式全量评测前检查容量，输出/缓存不要
无限写入该盘。不要未经授权删除原始数据、旧副本或源归档。

## 原始数据和旧处理结果

这些目录继续保留，**不要与当前 processed_v1 混用**：

| 数据 | 共享路径 |
|---|---|
| NR3D 原始标注/划分 | `/home/ysh/Datasets/nr3d/raw/` |
| NR3D 旧提取帧/处理结果 | `/home/ysh/Datasets/nr3d/scannet/` |
| NR3D 旧实例/分割元数据 | `/home/ysh/Datasets/nr3d/scannet_aux/` |
| ScanNet 源 `.sens` / mesh | `/home/ysh/Datasets/ScanNet/scans/` |
| OpenEQA 官方原始问题 | `/home/ysh/Datasets/open-eqa/data/open-eqa-v0.json` |
| OpenEQA 官方 episode 帧 | `/home/ysh/Datasets/open-eqa/data/frames/{scannet-v0,hm3d-v0}/` |
| OpenEQA 旧 ConceptGraph | `/home/ysh/Datasets/OpenEQA/scannet/<clip>/conceptgraph/` |

2026-11-02 从旧项目复制 NR3D 原始数据的记录仍在
`/home/ysh/Datasets/.manifests/nr3d-copy-20261102/`。那份旧 `nr3d/scannet`
缺少 `pack_nr3d_v9_catalog_first`，**本次 `processed_v1/scannet` 已包含该 pack**。

当前 checkout 仍没有 `data/` 目录。历史绝对路径只作 provenance；优先使用
上面的共享根目录。已核查目录中未找到 `SceneFuncVal-CG` 预处理包，不能直接
套用其他机器的 NAS 路径。本次没有运行 benchmark 或远程模型调用。
