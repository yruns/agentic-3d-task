# 研究点二：室内多任务推理

> 对应论文 3.2 节。调研截至 2026-11-10。表中“与本文关系”只描述机制差异，不代表性能优劣。

## 最接近的 5 篇

1. **ZSVG3D**（CVPR 2024，arXiv:2311.15383）：可执行视觉程序 + 房间中心虚拟相机，直接覆盖本文“虚拟观察者”参考系。
2. **GraphPad**（CVPR 2025 Workshop，arXiv:2506.01174）：OpenEQA 上的可变场景图 + API 循环检索 + 记忆更新，是“迭代工具取证”的直接近邻。
3. **MemTree3D**（ECCV 2026，arXiv:2608.18009）：记忆树先导 + 关键帧检索，是“先读紧凑表示，再取原图证据”的近邻。
4. **Fun3DU**（CVPR 2025，arXiv:2411.16310）：任务解析、视图选择、Molmo 指点、SAM 分割、多视角一致性过滤，是功能分割流程最接近的前作。
5. **AFFORDMEM**（arXiv:2605.11616, 2026）：可见性归一化投票 + 聚类，直接覆盖本文式（30）的核心归一化。

补充直接前作：**VLM-Grounder**（CoRL 2024）、**CoV**（arXiv:2601.05172, 2026）、**MaskClustering**（CVPR 2024）、**PartSLIP++**（arXiv:2312.03015, 2023）、**GuideGround**（arXiv:2608.00518, 2026）。

## 文献清单

| 方法 | 标题 | 作者/出处 | arXiv/DOI | 核心思想 | 与本文关系 |
|---|---|---|---|---|---|
| OpenEQA | Embodied Question Answering in the Era of Foundation Models | Majumdar et al., CVPR 2024 | — | 开放式具身问答基准，EM-EQA/A-EQA | 任务基准；含对象 crop 与去重基线 |
| ScanQA | 3D Question Answering for Spatial Scene Understanding | Azuma et al., CVPR 2022 | 2112.10482 | 点云 + 对象候选融合回答 | 三维问答基础；非工具取证 |
| SQA3D | Situated Question Answering in 3D Scenes | Ma et al., ICLR 2023 | 2210.07474 | 显式给出观察者位置与朝向 | 视角依赖推理基准 |
| Explore until Confident | Efficient Exploration for Embodied EQA | Ren et al., RSS 2024 | 2403.15941 | VLM 语义地图 + conformal 停止准则 | 主动探索；选择未来观测 |
| Map-based Modular EQA | Map-based Modular Approach for Zero-shot EQA | Sakamoto et al., IROS 2024 | 2405.16559 | 探索建图 + 基础模型推理的模块化零样本 EQA | 地图 + 基础模型路线近邻 |
| 3D-VisTA | Pre-trained Transformer for 3D Vision and Text Alignment | Zhu et al., ICCV 2023 | 2308.04352 | 统一 Transformer 编码点云与文本 | 表征层多任务统一先例 |
| 3D-LLM | Injecting the 3D World into Large Language Models | Hong et al., NeurIPS 2023 | 2307.12981 | 三维点云/多视图特征注入 LLM | 模型层多任务统一先例 |
| Scene-LLM | Extending Language Model for 3D Visual Understanding and Reasoning | Fu et al., WACV 2025 | 2403.11401 | 场景级 + 自我中心混合表征，支持在线更新 | 全局—局部结合先例 |
| Chat-Scene | Bridging 3D Scene and LLMs with Object Identifiers | Huang et al., NeurIPS 2024 | 2312.08168 | 对象标识符 token + 多任务联合训练 | 对象级统一接口先例 |
| LEO | An Embodied Generalist Agent in 3D World | Huang et al., ICML 2024 | 2311.12871 | 统一感知、接地、推理、规划与行动 | 多任务统一模型先例 |
| 3D-R1 | Enhancing Reasoning in 3D VLMs for Unified Scene Understanding | Huang et al., arXiv 2025 | 2507.23478 | 思维链冷启动 + GRPO + 动态视角选择 | 强化学习 + 动态选视角近邻 |
| ScanRefer | 3D Object Localization in RGB-D Scans using Natural Language | Chen et al., ECCV 2020 | 1912.08830 | 对象提案 + 语言匹配的三维框定位 | 三维指代任务基础 |
| ReferIt3D | Neural Listeners for Fine-Grained 3D Object Identification | Achlioptas et al., ECCV 2020 | — | Sr3D/Nr3D，视角依赖/无关指代 | 任务与评测基础 |
| ViewRefer | Grasp the Multi-view Knowledge for 3D Visual Grounding | Guo et al., ICCV 2023 | 2303.16894 | 视角相关语言扩写 + 多视角原型融合 | 学习式视角融合近邻 |
| VLM-Grounder | A VLM Agent for Zero-Shot 3D Visual Grounding | Xu et al., CoRL 2024 | 2410.13860 | 动态拼图、反馈重试、多视图投影 | 智能体式定位近邻 |
| ZSVG3D | Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding | Yuan et al., CVPR 2024 | 2311.15383 | LLM 生成可执行程序 + 虚拟相机 | 虚拟观察者直接先例 |
| SeeGround | See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding | Li et al., CVPR 2025 | 2412.04383 | 查询对齐渲染视角 + 空间文本 | 视角自适应近邻 |
| OriGround | Orientation-Aware Neuro-Symbolic Zero-Shot 3D Visual Grounding | Li et al., Findings EMNLP 2026 | — | 对象朝向 + 观察者参考系 | 观察者推断近邻 |
| SceneFun3D | Fine-Grained Functionality and Affordance Understanding | Delitzas et al., CVPR 2024 | — | 功能/可供性细粒度标注与任务 | 功能分割基准 |
| OpenMask3D | Open-Vocabulary 3D Instance Segmentation | Takmaz et al., NeurIPS 2023 | 2306.13631 | 类别无关三维掩码 + 多视图特征 | 可见视图与深度过滤先例 |
| Fun3DU | Functionality Understanding and Segmentation in 3D Scenes | Corsetti et al., CVPR 2025 | 2411.16310 | 任务解析、选帧、Molmo 指点、SAM、多视图提升 | 功能分割流程最接近前作 |
| UniFunc3D | Unified Active Spatial-Temporal Grounding for 3D Affordance Segmentation | Lin, Xu, arXiv 2026 | 2603.23478 | 主动时空取证 + 掩码视觉核验 | 视觉核验近邻；单种子约束可区分 |
| AFFORDMEM | Grounding by Remembering: Cross-Scene and In-Scene Memory for 3D Functional Affordances | Wang et al., arXiv 2026 | 2605.11616 | 功能示例记忆 + 可见性归一化投票 + 聚类 | 式（30）归一化的直接先例 |
| GraphPad | Inference-Time 3D Scene Graph Updates for EQA | Ali et al., CVPR 2025 Workshop | 2506.01174 | 可变场景图 + API 循环检索 + 记忆更新 | OpenEQA 迭代取证近邻 |
| MemTree3D | Memory Tree Guided Key Frame Querying for Efficient 3D QA | Huang et al., ECCV 2026 | 2608.18009 | 位置—对象—检测记忆树 + 关键帧检索 | 紧凑表示先导 + 原图取证近邻 |
| MapGPT | Map-Guided Prompting with Adaptive Path Planning | Chen et al., ACL 2024 | 2401.07314 | 在线语言拓扑地图 + 多步规划 | 地图先导近邻；非 OpenEQA |
| IVSGround | Where to Look Matters: Learning Influential Views | Chiang et al., ECCV 2026 | 2609.04741 | 学习查询相关影响视角 | 学习式选帧近邻 |
| VLMaps | Visual Language Maps for Robot Navigation | Huang et al., ICRA 2023 | 2210.05714 | 视觉语言特征写入栅格地图 | 栅格语言检索先例 |
| AffordBot | 3D Fine-grained Embodied Reasoning via Multimodal LLMs | Wang et al., NeurIPS 2025 | — | 多模态 LLM 做细粒度可供性推理 | 功能推理近邻 |
| CoV | Chain-of-View Prompting for Spatial Reasoning | Zhao et al., arXiv 2026 | 2601.05172 | 测试时选视角、离散相机动作、由粗到细探索 | OpenEQA 视角探索最接近；本文强调 BEV 先验与 O2V 帧号内联 |
| MaskClustering | View Consensus based Mask Graph Clustering for Open-Vocabulary 3D Instance Segmentation | Yan et al., CVPR 2024 | 2401.07745 | 视图共识图聚类 | 一致度门控 + 聚类的最强概念先行者 |
| PartSLIP++ | Enhancing Low-Shot 3D Part Segmentation via Multi-View Instance Segmentation and MLE | Zhou et al., arXiv 2023 | 2312.03015 | 多视图 SAM 掩码 + 概率化 3D 部件融合 | 2D 种子 + 多视图传播的最直接可比工作 |
| GuideGround | VLM-guided Semantic Understanding and Viewpoint-aware Reasoning for 3D Visual Grounding | Wang et al., arXiv 2026 | 2608.00518 | 保留逐视图假设并由 VLM 仲裁 | 视角依赖指代最近对照 |
| T-FunS3D | Task-Driven Hierarchical Open-Vocabulary 3D Functionality Segmentation | Feng, Sabzevari, arXiv 2026 | 2606.05975 | 任务驱动层次化功能分割 | 功能分割近邻 |
| Task-Aware 3D Affordance | Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement | He et al., AAAI | — | 二维引导 + 几何细化的可供性分割 | 二维引导三维细化近邻 |

## Novelty verdict

| 本文候选创新 | 判定 | 最接近证据 | 可保留的贡献边界 |
|---|---|---|---|
| 共享表征 + 三类任务统一 | 宽泛主张创新性低；接口层统一可辩护 | 3D-VisTA、Chat-Scene、LEO、Scene-LLM | 统一发生在证据获取接口层，不是统一模型训练 |
| 迭代工具取证 + 几何约束后验推断 | 范式层面低；具体调度可增量 | GraphPad、ZSVG3D、VLM-Grounder | 需定义何时取证、何时核验、冲突如何处理、何时停止 |
| Map-First：首轮俯视图 + 目录 | 具体初始化有增量 | MapGPT、GraphPad、MemTree3D | 网格化 BEV + 对象目录 + 最佳帧号内联 |
| 多尺度渐进取证 | 低至中等 | MemTree3D、GraphPad、UniFunc3D | 问题驱动的尺度切换规则与预算分配 |
| 原图高分辨率外扩裁剪 | 工程设计 | OpenEQA、OpenMask3D、UniFunc3D | 避免额外下采样损失；不能单独包装成算法创新 |
| 视点质量 + 位姿分离 + 部分共视回退 + 候选轮询 | 中等潜力，建议保留 | OpenMask3D、VLM-Grounder、MemTree3D | 固定预算下兼顾候选覆盖、质量与冗余 |
| 三维虚拟观察者 + 左右/前后排序 | 直接先例，独立创新性低 | ZSVG3D、SeeGround、OriGround | 确定性方位近似，与学习方法互补 |
| contact sheet + 无遮挡 crops sheet | 输入组织改进，独立创新性低 | VLM-Grounder、SeeGround、OpenMask3D | 空间标记图与外观图分离，需直接验证 |
| 二维功能种子确认 | 低至中等 | Fun3DU、UniFunc3D、AFFORDMEM | 只提交一个明确目标与粒度的种子，并约束后续处理 |
| 冻结三维锚点 | 中等潜力，建议保留 | Fun3DU、AFFORDMEM、OpenMask3D | 已确认单种子 → 固定位置与尺度 → 后续视角不得任意改写 |
| 可见性归一化投票 + 半径门控 + 连通分量 | 单个算子低；组合有增量 | AFFORDMEM、MaskClustering、PartSLIP++、Fun3DU、UniFunc3D | 片段级、点级、连通分量级约束围绕同一冻结锚点串联 |
| 测试时视角探索 / 多尺度取证 | 增量改进 | CoV、Explore-EQA、3D-R1 | 本文增量在 BEV 先验、O2V 帧号内联与原图裁剪坐标对齐 |
| 逐视图假设仲裁 / 视角依赖推理 | 已有工作覆盖或增量 | GuideGround、ViewRefer、ViewSRD | 本文是确定性虚拟观察者排序，不是学习式仲裁 |
| 多视图 SAM + 概率化 3D 部件融合 | 已有工作覆盖 | PartSLIP++、MaskClustering、OpenMask3D | 本文差异在唯一种子 + 冻结锚点 + 分层净化 |

## 可直接写入论文的区别表述

1. 与 3D-VisTA、Chat-Scene 和 LEO 主要通过表征对齐或多任务训练建立统一能力不同，本文将统一性落实于共享场景表示及证据访问接口，在同一对象—网格—视图关联体系上，为场景问答、指代定位和功能部件分割提供任务特化的视觉取证与几何约束流程。
2. 相较于 GraphPad 的可变场景图推理和 MemTree3D 的记忆树引导关键帧查询，本文采用网格化俯视图、对象目录及最佳观测帧索引构成首轮证据，并通过区域检索、邻域视图扩展与原始分辨率局部裁剪逐步补充信息，以兼顾场景布局理解与细粒度状态辨识。
3. 在已有虚拟视点几何推理和视觉提示方法的基础上，本文进一步关注有限观测预算下的候选证据覆盖，通过视点质量评估、位姿分离和候选轮询分配选择真实观测帧，并分别提供空间标记视图与无标记局部外观图，支持同类实例的联合消歧。
4. 相较于 Fun3DU 的多视图功能定位流程及 UniFunc3D 的视觉掩码核验，本文将目标身份与分割粒度的确认集中于唯一二维种子，并将其转化为位置和尺度固定的三维锚点，作为后续跨视图定位与融合的共同约束。

## 高风险表述

- “统一高精度求解” → 改为“共享接口下的任务协同”；需定义语义评分、Ψ_geo、冲突更新与置信度收敛。
- “彻底消除裁剪坐标误差” → 改为“减小”；原图裁剪不能保证成像、检测框、接口编码与模型内部处理均无损。
- “冻结锚点保证正确” → 改为“抑制漂移，但依赖种子质量”；需比较固定锚点、更新锚点与无锚点。
- “AP25/AP50” → 实为 IoU 阈值样本比例，应称 SR@IoU 或 Acc@IoU；与官方 mean Average Precision 不同。
- “最终管线全面提升” → 只主张平均精确率与 F1 改善；平均 IoU、严格阈值指标并未全面提升。
- “同模型比较” → 表 3-6/3-9 混有不同推理模型，需拆开模型差异与方法差异。
- “虚拟观察者独立创新” → ZSVG3D 已有直接先例；本文只能主张扩展与集成。
- “二维种子 + SAM / 可见性归一化投票” → Fun3DU、UniFunc3D、AFFORDMEM 已有；本文主张“唯一种子 + 冻结锚点 + 分层净化”的约束链。
- “一致度门控 + 聚类” → MaskClustering 已做视图共识图聚类；本文需强调部件级、锚点中心、可见性归一化与运动先验。
- “多视图 SAM + 3D 部件融合” → PartSLIP++ 已有；本文需强调唯一种子与冻结锚点的身份约束。
- “视角依赖推理” → GuideGround、ViewRefer、ViewSRD 已有逐视图假设或视角自适应；本文只能主张确定性虚拟观察者排序与检索前置约束。
- SceneFun3D 的 AP 值形态接近 k/13，样本量过小是当前最大统计风险；需补全量样本与官方评测器结果。
