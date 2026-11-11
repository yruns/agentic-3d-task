# 研究点一：三维场景表征与关键帧检索

> 对应论文 3.1 节。调研截至 2026-11-10。表中“与本文关系”只描述机制差异，不代表性能优劣。

## 最接近的 5 篇

1. **3D-Mem**（CVPR 2025，arXiv:2411.17735）：覆盖驱动的 Memory Snapshots，按对象簇选共视最多帧，每对象唯一分配，同源帧合并。与本文“对象覆盖 + 固定预算选帧”最接近；差异是无网格通道、无固定 k；本文主实验使用对象—网格联合覆盖贪心，驻留/视锥项为可选、默认关闭。
2. **IVSGround / Where to Look Matters**（ECCV 2026，arXiv:2609.04741）：学习查询相关的“影响视角”选择器，用 VLM 反馈做拒绝采样监督。与本文“选帧”最接近；差异是学习式选择器，无对象/网格可见性索引。
3. **GraphEQA**（CoRL 2025，arXiv:2412.14480）：实时 3D 度量—语义场景图 + 任务相关图像记忆。与本文“场景索引 + 图像证据做 EQA”最接近；差异是图检索为主，无逐帧深度遮挡索引与预算选帧。
4. **ZSVG3D**（CVPR 2024，arXiv:2311.15383）：用可执行视觉程序与房间中心虚拟相机处理视角依赖关系。与本文“虚拟观察者”最接近；本文不能主张该参考系首创。
5. **CSES / Thorne et al.**：CSES（arXiv:2608.00714, 2026）做覆盖驱动自适应关键帧选择；Thorne et al.（ICRA 2025，arXiv:2410.05576）把次模优化用于 SLAM 关键帧选择。二者覆盖本文“覆盖/次模选帧”的数学形式。

## 文献清单

| 方法 | 标题 | 作者/出处 | arXiv/DOI | 核心思想 | 与本文关系 |
|---|---|---|---|---|---|
| ReferIt3D | Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes | Achlioptas et al., ECCV 2020 | — | 提出 Sr3D/Nr3D，区分视角依赖/无关指代 | 任务与评测基础；不是视觉证据检索器 |
| 3DSSG | Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions | Wald, Dhamo, Navab, Tombari, CVPR 2020 | 2004.03967 | 从三维重建学习对象—关系图 | 场景图生成；非图像预算选帧 |
| OpenScene | 3D Scene Understanding with Open Vocabularies | Peng et al., CVPR 2023 | 2211.15654 | 多视角二维语义融合到三维开放词汇特征 | 已有深度一致性测试；非原图检索 |
| ConceptGraphs | Open-Vocabulary 3D Scene Graphs for Perception and Planning | Gu et al., ICRA 2024 | 2309.16650 | 对象点云、特征、关系图，保留来源视图 | 已有对象—视图关联；非联合覆盖选帧 |
| HOV-SG | Hierarchical Open-Vocabulary 3D Scene Graphs | Werby et al., RSS 2024 | 2403.17846 | 楼层—房间—对象层次场景图 | 多粒度区域查询；非预算选帧 |
| OpenMask3D | Open-Vocabulary 3D Instance Segmentation | Takmaz et al., NeurIPS 2023 | 2306.13631 | 三维实例掩码 + 每实例可见视图 + 深度过滤 | 对象—视图可见性已有；本文不能主张首创 |
| RegionPLC | Regional Point-Language Contrastive Learning | Yang et al., CVPR 2024 | 2304.00962 | 区域点—语言表示 | 区域语义；不是可寻址 BEV 网格回查 |
| EmbodiedScan | Holistic Multi-Modal 3D Perception Suite | Wang et al., CVPR 2024 | 2312.16170 | 多视图、三维框、占据与可见实例 | 数据/感知基础设施；非检索算法 |
| OVSG | Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs | Chang et al., CoRL 2023 | 2309.15940 | 对象、智能体、区域节点及关系 | 已有区域节点与三维框；非缺失参照代理 |
| MapGPT | Map-Guided Prompting with Adaptive Path Planning | Chen et al., ACL 2024 | 2401.07314 | 在线语言拓扑地图 + 多步路径规划 | 地图先导；不是历史图像预算检索 |
| VLMaps | Visual Language Maps for Robot Navigation | Huang et al., ICRA 2023 | 2210.05714 | 视觉语言特征写入栅格地图 | 网格语言检索先例；本文 G2V 以可见性为键 |
| NS3D | Neuro-Symbolic Grounding of 3D Objects and Relations | Hsu, Mao, Wu, CVPR 2023 | 2303.13483 | 对象特征 + 嵌套符号程序 | 已有关系执行与软对象选择 |
| ZSVG3D | Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding | Yuan et al., CVPR 2024 | 2311.15383 | LLM 生成可执行程序，房间中心虚拟相机看向参照 | 虚拟观察者最直接先例 |
| ViewRefer | Grasp the Multi-view Knowledge for 3D Visual Grounding | Guo et al., ICCV 2023 | 2303.16894 | 视角相关语言扩写 + 多视角原型融合 | 学习式视角融合；非确定性参考系 |
| ViewSRD | 3D Visual Grounding via Structured Multi-View Decomposition | Huang et al., ICCV 2025 | 2507.11261 | 透视感知的多锚点分解与跨视角一致融合 | 视角依赖近邻；学习式多视角 |
| OriGround | Orientation-Aware Neuro-Symbolic Zero-Shot 3D Visual Grounding | Li et al., Findings EMNLP 2026 | — | 对象朝向 + 观察者参考系 + 视觉提示 | 观察者推理近邻；需全文核对 |
| VLM-Grounder | A VLM Agent for Zero-Shot 3D Visual Grounding | Xu et al., CoRL 2024 | 2410.13860 | 动态拼图、反馈重试、多视图投影 | 视觉搜索—反馈—三维输出 |
| SeeGround | See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding | Li et al., CVPR 2025 | 2412.04383 | 查询对齐渲染视角 + 三维空间文本 | 视角自适应；非真实历史帧检索 |
| OpenEQA | Embodied Question Answering in the Era of Foundation Models | Majumdar et al., CVPR 2024 | — | 开放式具身问答基准，EM-EQA/A-EQA | 任务基准；含对象 crop 与去重基线 |
| 3D-Mem | 3D Scene Memory for Embodied Exploration and Reasoning | Yang et al., CVPR 2025 | 2411.17735 | 覆盖驱动 Memory Snapshots，对象唯一分配 | 覆盖式记忆最接近；无网格通道/固定 k |
| GraphEQA | Using 3D Semantic Scene Graphs for Real-time EQA | Saxena et al., CoRL 2025 | 2412.14480 | 实时场景图 + 任务相关图像记忆 | 场景索引 + 图像证据的系统级近邻 |
| Explore until Confident | Efficient Exploration for Embodied Question Answering | Ren et al., RSS 2024 | 2403.15941 | VLM 语义地图 + 置信度校准停止 | 主动探索；选择未来观测而非已有帧 |
| MemTree3D | Memory Tree Guided Key Frame Querying for Efficient 3D QA | Huang et al., ECCV 2026 | 2608.18009 | 位置—对象—检测记忆树，先读树再检索关键帧 | 一次建索引、少帧检索；漏检上下文回退 |
| Opti3D | Query-Driven Evidence Retrieval for Efficient 3D QA | Liu, Wu, IEEE Access 2026 | 10.1109/ACCESS.2026.3720265 | BEV + 查询引导局部证据 + 跨视图去重 | 全局 BEV + 局部原图；本文差异在可寻址网格与联合集合目标 |
| FocusGraph | Graph-Structured Frame Selection for Embodied Long Video QA | Zemskova et al., arXiv 2026 | 2603.04349 | 时间片段场景图 + 最多 K 帧选帧 | 关系记忆定位片段，再做二维选帧 |
| SCONE | Surface Coverage Optimization in Unknown Environments | Guédon et al., NeurIPS 2022 | 2208.10449 | 预测可见性与表面覆盖增益的下一最佳视点 | NBV 覆盖优化；非语言条件历史证据 |
| CSES | Coverage-Driven Adaptive Keyframe Selection for Video Understanding | Zhang et al., arXiv 2026 | 2608.00714 | 相关性 + 自适应数量 + 覆盖饱和 | 覆盖式选帧数学形式近邻 |
| 全景 EM-EQA 视图选择 | Informative Viewpoint Selection for Episodic-Memory EQA | Kitamura, Kanezaki, ACCV 2026 | 2609.33288 | 全景历史 + cubemap 透视候选 + 固定 K | 固定预算视图选择；去冗余用特征相似度 |
| IVSGround | Where to Look Matters: Learning Influential Views | Chiang et al., ECCV 2026 | 2609.04741 | 学习查询相关影响视角，VLM 反馈监督 | 学习式选帧；无对象/网格索引 |
| Thorne et al. | Submodular Optimization for Keyframe Selection and Usage in SLAM | Thorne et al., ICRA 2025 | 2410.05576 | 次模优化用于定位约束与地图摘要 | “次模选帧”最近对照；目标不是语言检索 |
| MSGNav | Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation | Huang et al., CVPR 2026 | — | 多模态场景图 + 关键子图 + 可见性视点 | 图像关联与视点决策近邻 |
| VideoAgent | Long-form Video Understanding with LLM as Agent | Wang et al., ECCV 2024 | — | 多轮检索组织长视频证据 | 任务驱动证据检索；非三维可见性 |
| CAMCUE | （预印本） | arXiv 2026 | 2602.06041 | 想象视角/参考系学习 | 视角依赖近邻；未核实全文 |
| LeRF | （预印本） | arXiv 2026 | 2609.36219 | 学习参考系/视角表示 | 视角依赖近邻；未核实全文 |

## Novelty verdict

| 本文候选创新 | 判定 | 最接近证据 | 可保留的贡献边界 |
|---|---|---|---|
| 对象级 + 区域级双通道可见性索引 | 增量改进；组合可辩护 | ConceptGraphs、OpenMask3D、OpenScene、RegionPLC、HOV-SG | 同一表征中并行维护对象级与网格级索引；网格级在漏检时提供区域回退 |
| BEV 网格 + 深度遮挡核验 + G2V | 增量改进；具体 G2V 索引可辩护 | OpenScene、Opti3D、VLMaps | “每帧对网格包络的深度遮挡核验结果”作为检索键 |
| 虚拟观察者解决视角依赖 | 概念已有；确定性算子属增量 | ZSVG3D、NS3D、ViewSRD、SeeGround、OriGround | 无共视帧时的确定性横向/纵向排序，作为候选筛选前置约束 |
| 对象—网格联合覆盖（主实验）+ 可选视锥去冗余 | 增量改进；联合公式可辩护 | 3D-Mem、CSES、SCONE、MemTree3D、Thorne | 主实验使用对象/网格联合覆盖贪心；`pose_aware=True` 的驻留与视锥惩罚为可选变体，默认关闭 |
| 多假设解析 + 实例/网格软先验 | 增量改进 | NS3D、ZSVG3D、OVSG | 需补假设概率 p_m 的聚合公式；否则改称“优先级权重” |
| 未知参照物虚拟网格锚点 | 增量改进；局部可辩护 | OVSG、SeeGround、MemTree3D | 需与虚拟观察者拆分为独立消融 |
| 72 ms 场景索引 | 工程贡献 | MemTree3D、3D-Mem、FocusGraph | 只能作为效率结果，不作为研究新颖点 |

## 可直接写入论文的区别表述

1. 与 ConceptGraphs、3D-Mem 等对象中心表征相比，本文在同一场景表征中并行维护对象级与网格级两套可见性索引：对象级保留开放词汇语义与高分辨率观测证据，网格级在检测漏检时提供不依赖检测标签的区域级回退；二者在查询解析阶段通过软先验融合，目标是提高证据召回下界，而非替代对象中心语义图。
2. 与 VLMaps、MapGPT 等将语言特征或拓扑节点写入地图的方法相比，本文的 G2V 索引以“每帧对网格三维包络的深度遮挡核验结果”为键，检索单位是“某空间区域是否被某帧无遮挡观测”，而不是“某地图位置与文本嵌入的相似度”。前者回答“去哪里找证据帧”，后者回答“目标在地图上的哪里”，可以组合使用。
3. 与 ViewRefer、ViewSRD、SeeGround 等多视角融合/视角自适应方法相比，本文不学习跨视角特征融合；当目标与参照物无共视帧时，以虚拟观察者坐标系给出确定性的横向/纵向排序，作为候选筛选与选帧前的几何约束。该方法与学习方法互补：前者保证无共视条件下的方位可解算性，后者提升语义判别力。

## 高风险表述

- “次模贪心” → 改为“边际增益贪心”；式（17）不保证标准次模近似，不应声称 1−1/e 近似。
- **代码事实**：`select_keyframes_v2` 的 `pose_aware` 默认 `False`，NR3D/OpenEQA 工具封装与评测脚本均未传 `True`；主实验只使用对象—网格联合覆盖贪心。`dwell_score = 1/(1+5*平滑速度)` 与 `_pose_aware_gain` 存在，但属于未启用的可选路径。论文不应把驻留/视锥项写成主公式。
- “从机制上消除检测漏检” → 改为“降低空返回、提供区域级回退”；空返回率 0% 部分是回退机制的定义性结果，需补网格回退精确率/召回率。
- “解决视角依赖指代” → 改为“在无共视帧条件下提供确定性方位近似”，并说明房间中心近似说话者视角的假设。
- 式（13）含 0.35×对象可见度，“无需检测标签”应改为“基础可见性项不依赖检测标签，检测仅用于加权”。
- 式（3）未实际使用假设概率 p_m，需补聚合公式或改称“优先级权重”。
- 缺 k=1/2/5/10 预算敏感性、δ_occ/τ_ov/λ_g 敏感性、虚拟锚点与虚拟观察者分离消融。
