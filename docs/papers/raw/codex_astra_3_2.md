**本文 3.2 有成立为硕士论文研究贡献的空间，但应把创新性落在“证据如何调度、目标身份如何保持、几何约束如何执行”的具体机制上。**“共享表示支持多任务”“工具增强三维推理”“虚拟观察者”“多视角投票”均已有明确先例。其中，**ZSVG3D 的虚拟相机方法、Fun3DU 的功能分割管线，以及 AFFORDMEM 的可见性归一化投票，必须正面比较。**

已阅读两份材料的完整 3.2 节：[sec3.md](/tmp/sec3.md:187)、[draft.xml](/home/ysh/codebase/agentic-3d-task/draft_96fcab98_folder/draft.xml:201)。两稿方法主线一致，XML 对部分绝对化表述有所收敛，但实验协议和归因问题仍存在。全程未修改文件。

检索范围为 2020—2026 年，截至 **2026-11-10**。以下判断依据原论文、正式会议页面、作者项目及官方评测说明；“未发现完全相同实现”不作为首创证明。

**1．建议纳入的 25 篇核心文献如下。**

作者较多时列首作者等；会议年份与预印本年份不同时予以区分。

| 编号 | 英文标题及方法名 | 作者 | 出处、年份 |
|---|---|---|---|
| 1 | [ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes](https://referit3d.github.io/) | Panos Achlioptas 等 | ECCV 2020；Nr3D/Sr3D 来源 |
| 2 | [ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language](https://arxiv.org/abs/1912.08830) | Dave Zhenyu Chen、Angel X. Chang、Matthias Nießner | ECCV 2020；预印本 2019 |
| 3 | [ScanQA: 3D Question Answering for Spatial Scene Understanding](https://arxiv.org/abs/2112.10482) | Daichi Azuma、Taiki Miyanishi、Shuhei Kurita、Motoaki Kawanabe | CVPR 2022 |
| 4 | [SQA3D: Situated Question Answering in 3D Scenes](https://arxiv.org/abs/2210.07474) | Xiaojian Ma、Silong Yong 等 | ICLR 2023 |
| 5 | [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://openaccess.thecvf.com/content/CVPR2024/html/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.html) | Arjun Majumdar、Anurag Ajay、Xiaohan Zhang 等 | CVPR 2024 |
| 6 | [3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment](https://arxiv.org/abs/2308.04352) | Ziyu Zhu、Xiaojian Ma 等 | ICCV 2023 |
| 7 | [3D-LLM: Injecting the 3D World into Large Language Models](https://arxiv.org/abs/2307.12981) | Yining Hong、Haoyu Zhen 等 | NeurIPS 2023 |
| 8 | [Data-Efficiently Learn Large Language Model for Universal 3D Scene Perception](https://aclanthology.org/2025.findings-naacl.18/) — Chat-3D | Zehan Wang、Haifeng Huang 等 | Findings of NAACL 2025；早期版本 arXiv:2308.08769 |
| 9 | [Chat-Scene: Bridging 3D Scene and Large Language Models with Object Identifiers](https://arxiv.org/abs/2312.08168) | Haifeng Huang、Yilun Chen 等 | NeurIPS 2024 |
| 10 | [Scene-LLM: Extending Language Model for 3D Visual Reasoning](https://openaccess.thecvf.com/content/WACV2025/html/Fu_Scene-LLM_Extending_Language_Model_for_3D_Visual_Reasoning_WACV_2025_paper.html) | Rao Fu、Jingyu Liu、Xilun Chen、Yixin Nie、Wenhan Xiong | WACV 2025；预印本 2024 |
| 11 | [An Embodied Generalist Agent in 3D World](https://proceedings.mlr.press/v235/huang24ae.html) — LEO | Jiangyong Huang、Silong Yong、Xiaojian Ma 等 | ICML 2024 |
| 12 | [LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_LLaVA-3D_A_Simple_yet_Effective_Pathway_to_Empowering_LMMs_with_ICCV_2025_paper.html) | Chenming Zhu、Tai Wang、Wenwei Zhang、Jiangmiao Pang、Xihui Liu | ICCV 2025；arXiv:2409.18125 |
| 13 | [Uni3D: Exploring Unified 3D Representation at Scale](https://arxiv.org/abs/2310.06773) | Junsheng Zhou、Jinsheng Wang、Baorui Ma 等 | ICLR 2024 |
| 14 | [MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation](https://aclanthology.org/2024.acl-long.529/) | Jiaqi Chen、Bingqian Lin 等 | ACL 2024 |
| 15 | [GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering](https://arxiv.org/abs/2506.01174) | Muhammad Qasim Ali、Saeejith Nair 等 | CVPR 2025 **3D-LLM/VLA Workshop**；arXiv:2506.01174 |
| 16 | [Memory Tree Guided Key Frame Querying for Efficient 3D Question Answering](https://arxiv.org/abs/2608.18009) — MemTree3D | Hsiang-Wei Huang、Fu-Chen Chen 等 | 2026；arXiv:2608.18009，作者注明 ECCV 2026 |
| 17 | [Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding](https://arxiv.org/abs/2311.15383) — ZSVG3D | Zhihao Yuan、Jinke Ren、Chun-Mei Feng 等 | CVPR 2024 |
| 18 | [VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding](https://proceedings.mlr.press/v270/xu25c.html) | Runsen Xu、Zhiwei Huang、Tai Wang 等 | CoRL 2024；PMLR 论文集年份 2025；arXiv:2410.13860 |
| 19 | [SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](https://arxiv.org/abs/2412.04383) | Rong Li、Shijie Li、Lingdong Kong、Xulei Yang、Junwei Liang | CVPR 2025 |
| 20 | [OriGround: Orientation-Aware Neuro-Symbolic Zero-Shot 3D Visual Grounding](https://aclanthology.org/2026.findings-emnlp.2237/) | Haochen Li、Jiaxin Shi、Ruonan Liu、Luo Liufu | Findings of EMNLP 2026，官方页面标注 2026 年 10 月 |
| 21 | [SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes](https://scenefun3d.github.io/) | Alexandros Delitzas、Ayça Takmaz 等 | CVPR 2024 |
| 22 | [OpenMask3D: Open-Vocabulary 3D Instance Segmentation](https://arxiv.org/abs/2306.13631) | Ayça Takmaz、Elisabetta Fedele 等 | NeurIPS 2023 |
| 23 | [Functionality Understanding and Segmentation in 3D Scenes](https://arxiv.org/abs/2411.16310) — Fun3DU | Jaime Corsetti、Francesco Giuliari、Alice Fasoli、Davide Boscaini、Fabio Poiesi | CVPR 2025 |
| 24 | [UniFunc3D: Unified Active Spatial-Temporal Grounding for 3D Affordance Segmentation](https://arxiv.org/abs/2603.23478) | Jiaying Lin、Dan Xu | 2026；arXiv:2603.23478；作者注明 NeurIPS 2026 已接收 |
| 25 | [Grounding by Remembering: Cross-Scene and In-Scene Memory for 3D Functional Affordances](https://arxiv.org/abs/2605.11616) — AFFORDMEM | Qirui Wang、Jingyi He、Yining Pan、Xulei Yang、Shijie Li | 2026；arXiv:2605.11616，正式会议出处未核实 |

名称需要统一：

- **Fun3DU** 是上述功能理解方法；**SceneFun3D** 是数据集。本次未核验到这里所指的独立“Fun3D”方法。
- **3DVLM** 尚不能由该缩写唯一确定为一篇室内场景论文，不能擅自替换成 3D-LLM、3D-VisTA 或其他名称。
- Chat-3D、Scene-LLM、LLaVA-3D 的正式发表标题与早期预印本存在变化，参考文献应与实际引用版本对应。
- OriGround 本次核验到官方元数据和摘要，未完整读取其 PDF；下文不对其未核验的具体公式和消融作断言。

**2．方法对照中，需要区分“使用三维信息”和“执行显式几何核验”。**

下表的“几何核验”指推理时明确计算投影、可见性、距离或一致性约束；“无独立核验”不表示该方法没有空间能力。“多任务统一”也区分共享模型、共享表示和单任务内部多个阶段。

| 方法 | 任务范围 | 场景表示 | 证据获取 | 是否几何核验 | 是否多任务统一 | 主要假设 | 与本文差异 |
|---|---|---|---|---|---|---|---|
| ReferIt3D [1] | 指代实例选择 | 分割对象点云、对象关系 | 给定候选集合编码 | 无独立核验 | 否 | 候选实例可用 | 建立任务；不做按需原图取证 |
| ScanRefer [2] | 指代三维框定位 | 场景点云、检测候选 | 学习式语言—对象匹配 | 无独立核验 | 否 | 检测与监督数据可用 | 输出检测框，与固定 proposal 选择不同 |
| ScanQA / SQA3D [3–4] | 场景问答／情境问答 | 点云、对象及语言情境 | 编码场景后回答 | 基线无独立核验 | 主要为问答 | 场景与情境信息足够 | 不以可调用证据工具为主要贡献 |
| 3D-VisTA [6] | 定位、问答、情境推理 | 对象与文本 token | 预训练及下游适配 | 表征内编码 | 是，共享骨干 | 三维—语言训练数据 | 本文侧重推理时工具编排 |
| 3D-LLM / Chat-3D [7–8] | 多类三维语言任务 | 三维特征／对象特征 | 特征对齐与指令输入 | 通常无独立硬核验 | 是，覆盖范围不同 | 对齐训练、场景特征可靠 | 统一主要发生在模型层 |
| Chat-Scene [9] | 定位、描述、问答 | 对象 ID＋二维／三维对象特征 | 将场景对象序列输入模型 | 无独立硬核验 | 是，统一问答格式 | proposal 质量及训练数据 | 本文另设原图检索与几何执行接口 |
| Scene-LLM [10] | 描述、问答、交互规划 | 全局与自我中心三维特征 | 场景／帧特征及状态更新 | 空间信息进入表示 | 是 | 特征对齐、位姿与场景状态 | 已有全局—局部结合，本文采用外部工具 |
| LEO [11] | 感知、推理、导航、操作 | 对象点云＋自我中心图像 | 统一 VL/VLA 指令处理 | 主要由模型学习 | **是，统一模型与目标** | 多任务数据与训练 | 不能声称本文首次统一室内多任务 |
| LLaVA-3D [12] | 二维／三维理解、定位、问答 | 图像 patch＋三维位置 | 多视图编码与池化 | 几何进入 token | 是 | 位姿／三维位置及指令微调 | 本文保留外部可查询几何与源图 |
| Uni3D [13] | 分类、检索、部件等迁移任务 | 对齐图文空间的点云表示 | 大规模表征预训练 | 非任务后验核验 | 表征复用 | 规模化跨模态数据 | 是表征基础，非室内工具推理直接基线 |
| OpenEQA 的 SVM 基线 [5] | 开放场景问答 | 全局对象实例、三维框、图像裁剪 | 跨帧关联、最佳对象 crop、描述 | 有实例关联与几何处理 | 否 | 检测、深度及跨帧关联可靠 | 已有去重和裁剪；本文增加交互式区域取证 |
| MapGPT [14] | 视觉语言导航 | 在线语言拓扑地图 | 地图引导多步路径规划 | 拓扑约束为主 | 否 | 导航节点及观测可用 | 不是 OpenEQA，也不是零初始 RGB 的 BEV 策略 |
| GraphPad [15] | OpenEQA | 可变场景图、帧日志、笔记 | API 循环检索、分析并更新记忆 | 部分空间关系处理 | 否 | 初始记忆与帧检索可靠 | 本文突出预建共享表示及任务特化几何算子 |
| MemTree3D [16] | 三维／具身问答 | 位置—对象—检测记忆树 | 先读紧凑树，再检索相关关键帧 | 位姿用于组织与选帧 | 否 | 位姿与对象索引可靠 | 已有表示先导取证；本文采用网格 BEV 和交互 crop |
| ZSVG3D [17] | 三维指代定位 | 对象位置、框及语言—对象关联 | LLM 生成可执行视觉程序 | **是，含虚拟相机关系计算** | 否 | 对象与参照系假设可靠 | 本文增加真实观测质量、候选覆盖及外观复核 |
| VLM-Grounder [18] | 三维指代定位 | 多视图、二维定位及三维提升 | 动态拼图、反馈重试、多视图投影 | 是，投影融合 | 否 | RGB-D／投影条件可用 | 它生成三维框；本文选择已有 proposal |
| SeeGround [19] | 三维指代定位 | 对象空间文本＋查询对齐渲染 | 调整虚拟视点、可见性标记 | 是，投影与可见性 | 否 | 可渲染三维场景及对象表示 | 本文重点是已有真实 RGB 帧及无标记 crop |
| OriGround [20] | 三维指代定位 | 对象朝向、观察者参考系 | 符号程序＋参考系对齐视觉提示 | 是，摘要明确几何评分 | 否 | 朝向和观察者可估计 | 已覆盖观察者推断与视觉—几何联合消歧 |
| SceneFun3D [21] | 功能分割、任务驱动定位、运动估计 | 高分辨率扫描、RGB-D、动作标注 | 基准输入与任务定义 | 依基线而定 | 多任务基准 | 扫描及标注对齐 | 不是证明本文多任务统一的既有方法 |
| OpenMask3D [22] | 开放词汇实例分割 | 类别无关三维掩码＋多视图特征 | 可见视图、SAM、多尺度 crop、CLIP 聚合 | 是，投影／可见性 | 否 | 初始三维实例掩码能覆盖目标 | 本文从语义确认的细部种子开始 |
| Fun3DU [23] | 任务驱动功能分割 | RGB-D／点云对应 | 任务解析、选帧、Molmo 指点、SAM、提升 | **是，多视图一致性** | 否 | 上下文对象与功能部件可定位 | 本文加入唯一种子、冻结锚点和分层门控 |
| UniFunc3D [24] | 三维功能分割 | 多尺度视频证据、掩码与三维对应 | 主动时空取证、掩码视觉核验 | 是，多视图提升／一致性 | 单任务统一推理阶段 | MLLM 与分割器可靠 | 已有视觉核验；本文可强调固定单种子约束 |
| AFFORDMEM [25] | 三维功能定位／分割 | 跨场景示例记忆＋场景内候选图 | 功能示例引导、候选融合与选择 | **是，可见性归一化投票及聚类** | 否 | 源场景示例、深度与候选可靠 | 本文先锁定一个目标，再限制其跨视图扩张 |
| 本文 3.2 | QA＋实例选择＋功能部件分割 | 共享对象／网格／源图关联表示 | 任务驱动工具调用与分层取证 | VG、Func 明确；QA 部分属于视觉核查 | **共享表示与接口层统一** | 预建场景、可靠位姿、候选及种子 | 贡献应由具体调度与约束机制体现 |

**3．逐项 novelty verdict：多数基础组件已有先例，两个组合设计值得重点保留。**

这里的“低”表示不宜独立作为核心原创点；“中等潜力”表示存在可论证的具体差异，但仍需直接比较和消融。

| 本文候选创新 | 最接近的 2—3 篇证据 | Novelty verdict |
|---|---|---|
| **共享多模态表示支撑三类任务** | [3D-VisTA](https://arxiv.org/abs/2308.04352)、[Chat-Scene](https://arxiv.org/abs/2312.08168)、[LEO](https://proceedings.mlr.press/v235/huang24ae.html) | **宽泛主张创新性低。**共享表示与多任务统一已有充分先例。可以主张本文将文本回答、对象 ID 和部件顶点集合接入同一证据服务体系，但这属于表示／系统接口统一，不等于统一模型训练。 |
| **迭代工具取证＋几何约束后验推断** | [GraphPad](https://arxiv.org/html/2506.01174v1)、[ZSVG3D](https://arxiv.org/html/2311.15383v2)、[VLM-Grounder](https://arxiv.org/html/2410.13860v1) | **范式层面创新性低，具体策略可能有增量。**必须定义何时取证、何时核验、冲突如何处理、何时停止；仅写式（19）的策略—执行—输出形式不足以形成新算法。 |
| **Map-First：首轮 BEV＋目录，不输入第一人称帧** | [MapGPT](https://aclanthology.org/2024.acl-long.529/)、[GraphPad](https://arxiv.org/abs/2506.01174)、[MemTree3D](https://arxiv.org/abs/2608.18009) | **具体初始化策略有增量，广义“地图先导”不新。**较清楚的差异是网格化 BEV、对象目录与最佳帧号内联，以及首轮输入配置。MapGPT 不是同任务方法，不能直接当作等价实现。 |
| **多尺度渐进取证** | [MemTree3D](https://arxiv.org/abs/2608.18009)、[GraphPad](https://arxiv.org/abs/2506.01174)、[UniFunc3D](https://arxiv.org/html/2603.23478v2) | **低至中等。**从紧凑概览到相关帧、再到局部细节已有近邻。贡献应落在问题驱动的尺度切换规则和预算分配，而非“全局＋局部”组合本身。 |
| **原图高分辨率外扩裁剪、坐标对齐** | [OpenEQA 补充材料 F](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Majumdar_OpenEQA_Embodied_Question_CVPR_2024_supplemental.pdf)、[OpenMask3D](https://arxiv.org/abs/2306.13631)、[UniFunc3D](https://arxiv.org/html/2603.23478v2) | **主要是有效工程设计。**原图裁剪和多尺度 crop 已有先例；归一化坐标乘原图尺寸是标准实现。可以论证其避免额外下采样损失，不能将坐标修正单独包装成算法创新。 |
| **视点质量＋位姿分离＋部分共视回退＋候选轮询** | [OpenMask3D](https://arxiv.org/abs/2306.13631)、[VLM-Grounder](https://arxiv.org/html/2410.13860v1)、[MemTree3D](https://arxiv.org/abs/2608.18009) | **中等潜力，是建议保留的重点之一。**面积、中心度、阈值本身新意有限；固定预算下兼顾每个候选的可见覆盖、质量和冗余，可能形成可验证的方法贡献。 |
| **三维虚拟观察者＋左右／前后排序及几何关系工具** | [ZSVG3D §3.3](https://arxiv.org/html/2311.15383v2)、[SeeGround](https://arxiv.org/html/2412.04383v2)、[OriGround](https://aclanthology.org/2026.findings-emnlp.2237/) | **存在直接先例，独立创新性低。**ZSVG3D 已在房间中心放置虚拟相机、朝向参照物，通过投影横坐标和深度判断左右前后。返回排序 ID 是接口实现差异。 |
| **contact sheet＋无遮挡高清 crops sheet** | [VLM-Grounder](https://arxiv.org/html/2410.13860v1)、[SeeGround](https://arxiv.org/html/2412.04383v2)、[OpenMask3D](https://arxiv.org/abs/2306.13631) | **输入组织改进，独立创新性低。**可强调把空间标记图与外观图分开，降低标记对细小纹理的遮挡；需直接验证，不能由“拼图＋22% padding”推出新算法。 |
| **二维功能种子确认与 SeedConfirmation** | [Fun3DU](https://arxiv.org/html/2411.16310v5)、[UniFunc3D §3.4](https://arxiv.org/html/2603.23478v2#S3.S4)、[AFFORDMEM](https://arxiv.org/html/2605.11616v1) | **低至中等。**Molmo→SAM、掩码视觉核验已有先例。可保留的是“只提交一个明确目标及粒度的种子，并约束后续处理”的机制；数据结构命名本身不是创新。 |
| **冻结三维锚点：质心＋95% 半径＋运动尺度上限** | [Fun3DU](https://arxiv.org/html/2411.16310v5)、[AFFORDMEM](https://arxiv.org/html/2605.11616v1)、[OpenMask3D](https://arxiv.org/abs/2306.13631) | **中等潜力，是建议保留的重点之二。**本次核心文献中，这一“已确认单种子→固定位置与尺度→后续视角不得任意改写”的完整约束链具有较清楚差异；尚不足以宣称首次。 |
| **可见性归一化投票＋半径门控＋连通分量净化** | [AFFORDMEM §3.3](https://arxiv.org/html/2605.11616v1#S3.SS3)、[Fun3DU §3.5](https://arxiv.org/html/2411.16310v5)、[UniFunc3D §3.5](https://arxiv.org/html/2603.23478v2) | **单个算子创新性低，组合有增量空间。**AFFORDMEM 已明确使用前景支持次数／可见次数并聚类。可主张的是片段级、点级、连通分量级约束围绕同一冻结锚点串联。 |

三个特别关键的事实不能遗漏：

- **虚拟观察者不是仅有“相似思想”。**ZSVG3D 的 §3.3 与本文式（15）—（16）在几何构造上非常接近：房间中心、朝向参照对象、投影后解释方向关系。[ZSVG3D 原文](https://arxiv.org/html/2311.15383v2)
- **Fun3DU 不能被概括成“逐帧分割后直接取并集”。**它已经包含视图选择和多视图一致性过滤。本文直接并集方案可以作为内部基线，但不能代替与该前作的比较。[Fun3DU 原文](https://arxiv.org/html/2411.16310v5)
- **式（30）的核心归一化已有直接先例。**AFFORDMEM 使用 \(n_{\mathrm{fg}}/n_{\mathrm{vis}}\)，并设置最小可见次数，再进行 DBSCAN 聚类；因此原创性应放在冻结锚点和分层约束组合上。[AFFORDMEM §3.3](https://arxiv.org/html/2605.11616v1#S3.SS3)

**4．若只能优先补充五篇，建议补充以下工作。**

这里的“遗漏”指最容易在当前方法论证中被忽略的直接前作，不表示已检查整篇论文的参考文献表。

| 优先补充 | 为什么重要 | 应补入的位置 |
|---|---|---|
| **ZSVG3D** | 直接覆盖虚拟相机、视点关系和确定性几何模块 | 3.2.1、3.2.3 |
| **VLM-Grounder** | 已有智能体反馈、动态图像拼接、多视图提升；也是 contact sheet 的直接比较对象 | 3.2.3 |
| **GraphPad** | 在 OpenEQA 上已有结构记忆与 API 迭代取证 | 3.2.1、3.2.2 |
| **Fun3DU** | 功能任务、Molmo 指点、SAM 分割、三维提升和一致性过滤均高度接近 | 3.2.4 |
| **AFFORDMEM** | 直接影响可见性归一化投票及几何融合的原创性判断 | 3.2.4 |

对于明确覆盖到 2026 年的论文，还应在上述讨论中同时处理 **OriGround、UniFunc3D、MemTree3D**：它们分别进一步收窄观察者参考系、视觉掩码核验、紧凑表示先导检索的创新空间。

**5．以下五条可以直接改写进论文“与现有工作区别”。**

这些表述刻意限定为方法差异，不预先承诺尚未充分验证的性能优势。

1. **共享框架的区别**

   > 与 3D-VisTA、Chat-Scene 和 LEO 主要通过表征对齐或多任务训练建立统一能力不同，本文将统一性落实于共享场景表示及证据访问接口，在同一对象—网格—视图关联体系上，为场景问答、指代定位和功能部件分割提供任务特化的视觉取证与几何约束流程。

2. **开放问答的区别**

   > 相较于 GraphPad 的可变场景图推理和 MemTree3D 的记忆树引导关键帧查询，本文采用网格化俯视图、对象目录及最佳观测帧索引构成首轮证据，并通过区域检索、邻域视图扩展与原始分辨率局部裁剪逐步补充信息，以兼顾场景布局理解与细粒度状态辨识。

3. **指代定位的区别**

   > 在已有虚拟视点几何推理和视觉提示方法的基础上，本文进一步关注有限观测预算下的候选证据覆盖，通过视点质量评估、位姿分离和候选轮询分配选择真实观测帧，并分别提供空间标记视图与无标记局部外观图，支持同类实例的联合消歧。

4. **功能种子与锚点的区别**

   > 相较于 Fun3DU 的多视图功能定位流程及 UniFunc3D 的视觉掩码核验，本文将目标身份与分割粒度的确认集中于唯一二维种子，并将其转化为位置和尺度固定的三维锚点，作为后续跨视图定位与融合的共同约束。

5. **几何融合的区别**

   > 在多视图一致性与可见性归一化支持度的基础上，本文围绕冻结三维锚点依次执行片段质心筛选、顶点支持度与半径过滤，以及锚点邻近连通分量选择，形成针对细粒度功能部件的分层融合流程，以抑制错误视角引起的目标身份漂移和背景扩张。

其中第 3、4、5 条最值得配套直接消融；其余主要用于准确界定研究路线。

**6．当前证据不足或需要修正的 claim，主要有以下几类。**

**第一，“统一高精度求解”“几何约束后验推断”尚需更明确的操作定义。**

三类任务分别调用不同工具并得到结果，能够证明框架复用；不能自动证明跨任务知识迁移、统一优化或统一模型能力。式（19）可以作为形式化描述，但若要称为可执行后验推断，应定义语义评分如何取得、\(\Psi_{\mathrm{geo}}\) 如何计算、冲突证据如何更新，以及何谓“置信度收敛”。

另外，**高清 crop 是视觉证据核查，不直接构成三维几何一致性证明**。建议总称使用“任务特化证据核验与几何约束”，再区分真正执行三维测量的部分。

**第二，“彻底消除误差”“保证覆盖”“严格排序即可解决视角歧义”超出当前公式能够支持的范围。**

- 原图裁剪可避免先缩小全帧再裁剪造成的额外损失，但不能保证原始成像、检测框、接口编码和模型内部处理均无损。
- 边界截断的 0.75 软惩罚不能保证选中帧“无截断”；有限预算也不能无条件保证每个候选都有代表帧。
- 式（25）要求平移、旋转和帧距**同时**过阈值，会排除纯旋转、同朝向平移中的有效观测，应交代回退条件。
- 给定坐标系后可以精确计算左右，但**语言所指的观察者位置及物体内在朝向仍可能不确定**。“面向床尾”需要识别床尾，不能仅由床的质心解决。OriGround 对对象朝向和观察者推断的处理尤其值得比较。[OriGround 官方摘要](https://aclanthology.org/2026.findings-emnlp.2237/)

**第三，功能分割中的“冻结”和“一致度”不等于正确性保证。**

冻结锚点可以限制漂移，也可能固化错误种子。应比较固定锚点、更新锚点和无锚点，并测试种子偏移、过分割、漏分割及尺度先验误差。

式（30）还存在低观测数问题：一个点只被看到一次并命中一次，也会得到一致度 1。AFFORDMEM 额外设置了最小可见次数，因此本文至少应分析低支持点的可信度。[AFFORDMEM §3.3](https://arxiv.org/html/2605.11616v1#S3.SS3)

“跨视角过程完全交由几何完成”也与阶段 C 再次调用 Molmo／SAM 的描述不一致。更准确的说法是：**跨视角语义预测受固定几何锚点约束，融合与去噪由确定性算子执行。**

**第四，SceneFun3D 指标存在可以确定的命名错配。**

官方 task-driven affordance grounding 使用 IoU 阈值下的 **mean Average Precision**，要求提交实例掩码及置信度；并允许一条任务描述对应多个功能实例。[官方任务与指标](https://scenefun3d.github.io/documentation/benchmarks/task2/)

草稿则将“\(\mathrm{IoU}\ge0.25/0.50\) 的样本比例”称为 AP25/AP50。只读核查还发现，[semi_online_pipeline.py](/home/ysh/codebase/agentic-3d-task/src/codex_agent/scenefunc3d/evaluation/semi_online_pipeline.py:129) 确实累计 `int(iou >= 0.50)`，随后以样本比例返回 `ap50`。

因此：

- 现有数值应称 **SR@IoU** 或 **Acc@IoU**；
- 与官方 AP 榜单比较，需要官方评测器；
- 单种子、单连通分量输出的适用范围也需说明，不能默认覆盖官方多实例指令。

**第五，Nr3D 的指标与输入条件应和文献对齐。**

Nr3D 的经典任务是给定候选对象后的实例识别，ScanRefer 则涉及三维框定位。不能把 proposal 选择准确率、IoU 阈值准确率和端到端检测定位能力混为一谈。[ReferIt3D](https://referit3d.github.io/)、[ScanRefer](https://arxiv.org/abs/1912.08830)

“Acc@0.25 等价于 Acc@0.50”只可能在特定候选框与真值关系下成立，需要验证，不能作为 Nr3D 的一般性质。与 VLM-Grounder 等比较时，还须交代候选是否来自真值实例、是否包含全部场景对象，以及评测子集是否一致。

**第六，当前表格不能完全支持模块增益的因果归因。**

- OpenEQA 的 78.92→83.08 同时涉及 Map-First 与裁剪坐标修正，不能全部归因于地图初始化。
- Nr3D 表 3-6 将 86.00 与 91.50 分别标为不同模型，表 3-9 又将相同数值呈现为递进模块消融。应澄清模型是否变化，否则无法把增益归于选帧模块。
- “各组采用同一推理模型”的文字与包含不同模型的比较表不一致。
- 两份草稿的状态保持实验叙述也不完全一致：Markdown 正文出现 85.33，XML 和表格使用 86.00，应回溯运行记录统一。

这些问题可直接从两份 [3.2 原稿](/home/ysh/codebase/agentic-3d-task/draft_96fcab98_folder/draft.xml:282) 看出，尚不能据此判定哪组数值正确。

**第七，“最终功能分割方案全面提升”不受表 3-8 支持。**

从草稿自身数值看，最终在线方案相对最佳离线融合：

- Mean Precision：29.99→32.40；
- Mean IoU：18.29→18.20；
- 所称 AP25：30.77→25.00；
- 所称 AP50：15.38→5.00。

因此目前可以描述为“精确率提高”，不能写成所有指标全面改善；若样本集合不同，则上述纵向比较本身也需要重新限定。[表 3-8 所在原稿](/home/ysh/codebase/agentic-3d-task/draft_96fcab98_folder/draft.xml:359)

最后，**现有内部基线不足以支持公开基准 SOTA 声明**。至少需要披露完整样本数量与划分、proposal 来源、模型和 judge 的固定版本、工具及图像预算，并在匹配设置下比较上述直接前作。最有价值的补充实验是：候选质量／覆盖／冗余的分项消融，以及种子确认／锚点冻结／可见性归一化／分层门控的分项消融。