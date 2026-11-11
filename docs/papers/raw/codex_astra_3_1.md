**3.1 可以形成可辩护的系统方法贡献，但目前不宜将对象—视图关联、深度遮挡核验、虚拟观察者或次模覆盖分别宣称为原创。**较有说服力的贡献边界是：面向检测目录不完整和图像预算受限的场景，将对象与空间网格两类检索通道、缺失参照的虚拟网格锚点、空间查询执行及互补选帧联合起来。

已阅读 [/tmp/sec3.md](/tmp/sec3.md) 和 [draft.xml 的 3.1 节](/home/ysh/codebase/agentic-3d-task/draft_96fcab98_folder/draft.xml:7)，并抽查了有关评测记录。全文未修改任何文件。检索截止 **2026-11-10**；以下区分原文已核实机制、合理推断与尚未核实事项。“可辩护新颖点”表示值得作为贡献验证，不代表已经证明全球首创。

先澄清两个名称：

- **3DVLM-Query／3DVLM-Control**：现有证据指向旧仓库 `3DVLMReasoning` 的内部实现及历史版本，而非独立发表论文。基准记录明确列出其仓库、提交及 `query_scene` 版本，应标注为“内部实现基线”，补充算法说明和版本信息。[本地基准记录](/home/ysh/codebase/agentic-3d-task/docs/benchmark/nr3d/v1_keyframe_targetid_coverage_strat600_20260608.md:21)
- **Map-First**：在本项目中是初始化策略名称；没有核实到与之唯一对应的独立论文。**MapGPT** 则应引用下表的 ACL 2024 导航论文，两者不能互换。[Map-First 记录](/home/ysh/codebase/agentic-3d-task/docs/benchmark/openeqa/v6_mapfirst_tools_overhaul_pilot300_20261104.md:11)

**文献清单选取 25 篇。**作者较多时采用首作者等的常用缩写，链接指向论文或官方出版页面。

| 编号 | 英文标题／方法 | 作者 | 年份、会议或编号 |
|---|---|---|---|
| 1 | [ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf) | Panos Achlioptas 等 | ECCV 2020 |
| 2 | [Learning 3D Semantic Scene Graphs From 3D Indoor Reconstructions](https://arxiv.org/abs/2004.03967)，3DSSG | Johanna Wald、Helisa Dhamo、Nassir Navab、Federico Tombari | CVPR 2020；2004.03967 |
| 3 | [OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/abs/2211.15654) | Songyou Peng 等 | CVPR 2023；2211.15654 |
| 4 | [ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://concept-graphs.github.io/) | Qiao Gu 等 | ICRA 2024；2309.16650 |
| 5 | [OpenMask3D: Open-Vocabulary 3D Instance Segmentation](https://arxiv.org/abs/2306.13631) | Ayça Takmaz 等 | NeurIPS 2023；2306.13631 |
| 6 | [RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding](https://arxiv.org/abs/2304.00962) | Jihan Yang、Runyu Ding、Weipeng Deng、Zhe Wang、Xiaojuan Qi | CVPR 2024；2304.00962 |
| 7 | [EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI](https://arxiv.org/abs/2312.16170) | Tai Wang 等 | CVPR 2024；2312.16170 |
| 8 | [Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs](https://arxiv.org/abs/2309.15940)，OVSG | Haonan Chang 等 | CoRL 2023；2309.15940 |
| 9 | [MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation](https://aclanthology.org/2024.acl-long.529/) | Jiaqi Chen、Bingqian Lin、Ran Xu、Zhenhua Chai、Xiaodan Liang、Kwan-Yee K. Wong | ACL 2024；2401.07314 |
| 10 | [NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations](https://arxiv.org/abs/2303.13483) | Joy Hsu、Jiayuan Mao、Jiajun Wu | CVPR 2023；2303.13483 |
| 11 | [Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding](https://arxiv.org/abs/2311.15383)，ZSVG3D | Zhihao Yuan 等 | CVPR 2024；2311.15383 |
| 12 | [ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance](https://arxiv.org/abs/2303.16894) | Zoey Guo 等 | ICCV 2023；2303.16894；会议版题名省略末尾 “with GPT and Prototype Guidance” |
| 13 | [OriGround: Orientation-Aware Neuro-Symbolic Zero-Shot 3D Visual Grounding](https://aclanthology.org/2026.findings-emnlp.2237/) | Haochen Li、Jiaxin Shi、Ruonan Liu、Luo Liufu | Findings of EMNLP 2026，10月；2026.findings-emnlp.2237 |
| 14 | [VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding](https://proceedings.mlr.press/v270/xu25c.html) | Runsen Xu、Zhiwei Huang、Tai Wang、Yilun Chen、Jiangmiao Pang、Dahua Lin | CoRL 2024；PMLR 270，出版年份 2025 |
| 15 | [SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](https://arxiv.org/abs/2412.04383) | Rong Li、Shijie Li、Lingdong Kong、Xulei Yang、Junwei Liang | CVPR 2025；2412.04383 |
| 16 | [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://open-eqa.github.io/) | Arjun Majumdar、Anurag Ajay、Xiaohan Zhang 等 | CVPR 2024 |
| 17 | [3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning](https://arxiv.org/abs/2411.17735) | Yuncong Yang、Han Yang、Jiachen Zhou、Peihao Chen、Hongxin Zhang、Yilun Du、Chuang Gan | CVPR 2025；2411.17735 |
| 18 | [GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering](https://saumyasaxena.github.io/grapheqa/) | Saumya Saxena 等 | CoRL 2025；2412.14480 |
| 19 | [Explore until Confident: Efficient Exploration for Embodied Question Answering](https://arxiv.org/abs/2403.15941) | Allen Z. Ren、Jaden Clark、Anushri Dixit、Masha Itkina、Anirudha Majumdar、Dorsa Sadigh | RSS 2024；2403.15941 |
| 20 | [Memory Tree Guided Key Frame Querying for Efficient 3D Question Answering](https://arxiv.org/abs/2608.18009)，MemTree3D | Hsiang-Wei Huang 等 | ECCV 2026；2608.18009 |
| 21 | [Query-Driven Evidence Retrieval for Efficient 3-D Question Answering](https://doi.org/10.1109/ACCESS.2026.3720265)，Opti3D | Huihui Liu、Haoyang Wu | IEEE Access 2026；DOI:10.1109/ACCESS.2026.3720265 |
| 22 | [FocusGraph: Graph-Structured Frame Selection for Embodied Long Video Question Answering](https://arxiv.org/abs/2603.04349) | Tatiana Zemskova 等 | arXiv:2603.04349，2026；采用10月1日 v2 |
| 23 | [SCONE: Surface Coverage Optimization in Unknown Environments by Volumetric Integration](https://papers.neurips.cc/paper_files/paper/2022/hash/828c6d69bdf91fca7f2b97c4dc214e94-Abstract-Conference.html) | Antoine Guédon、Pascal Monasse、Vincent Lepetit | NeurIPS 2022；2208.10449 |
| 24 | [Coverage-Driven Adaptive Keyframe Selection for Video Understanding](https://arxiv.org/abs/2608.00714)，CSES | Junyang Zhang、Puhan Luo、Chen Tang、Yuxi Shi、Xiang-Yang Li | arXiv:2608.00714，2026；未核实正式会议 |
| 25 | [Informative Viewpoint Selection for Episodic-Memory Embodied Question Answering using Omnidirectional Images](https://arxiv.org/abs/2609.33288) | Kaname Kitamura、Asako Kanezaki | ACCV 2026 已接收；2609.33288 |

核验边界：**OriGround 目前核实了官方元数据与摘要，具体算法细节未完成全文核验**；Opti3D 的作者官方页面与出版商检索正文相互支持主要机制，但出版商全文直接访问受限。因此，不能据此断言二者“没有”某个未在已读内容中出现的模块。

**下面按机制比较。**“预算”特指论文实际约束：每对象特征提取帧数、每轮记忆窗口、机器人探索步数，与本文的单查询全局 \(k\) 帧预算不同。“未专门处理／未核实”不等于不存在；“多视图融合”也不等于解决语言中的观察者视角歧义。

| 方法 | 场景表示 | 检索／选择粒度 | 预算约束 | 几何／遮挡核验 | 视角依赖指代 | 主要假设 | 与本文差异 |
|---|---|---|---|---|---|---|---|
| [ReferIt3D](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf) | 点云实例与对象上下文 | 候选实例 | 非选帧问题 | 三维对象关系 | 基准包含此类描述 | 给定场景实例候选 | 提供任务与评测基础，不是视觉证据检索器 |
| [3DSSG](https://arxiv.org/abs/2004.03967) | 对象—关系图 | 节点、关系、图匹配 | 非图像预算 | 从三维重构学习关系 | 非专门观察者求解 | 重构场景及监督数据 | 重点是场景图生成、匹配 |
| [OpenScene](https://arxiv.org/html/2211.15654v2) | 稠密点级开放词汇特征 | 点／语义区域 | 非问答选帧预算 | **投影与深度遮挡测试** | 未专门处理 | 配准图像、深度、三维点 | 融合特征，而非返回原始证据图 |
| [ConceptGraphs](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf) | 对象点云、特征及关系图 | 对象及来源视图裁剪 | 描述生成阶段最多10个信息视图 | 三维几何关联；记录视图点贡献 | 关系语义，非本文式坐标执行 | 分割及跨帧实例融合可靠 | 已有对象来源视图与代表视图；未采用本文对象—网格联合目标 |
| [OpenMask3D](https://arxiv.org/html/2306.13631v2) | 三维实例掩码与多视图特征 | 每实例可见视图 | **每实例 Top-\(k\)**，实验 \(k=5\) | **对象点投影、深度遮挡核验** | 未专门处理 | 三维实例掩码及配准 RGB-D | 预算用于实例特征提取，不是查询相关的全局集合选择 |
| [RegionPLC](https://arxiv.org/html/2304.00962v3) | 区域点—语言表示 | 语义区域／点 | 非证据图预算 | 二维区域到三维关联；具体遮挡细节未核 | 未专门处理 | 区域语言监督与表示学习 | “区域”不是可寻址 BEV 网格回查通道 |
| [EmbodiedScan](https://arxiv.org/html/2312.16170v1) | 多视图、三维框、占据表示 | 关键帧及可见实例 | 数据预处理选帧 | **预计算可见实例与占据掩码** | 包含 grounding 任务，非独立观察者机制 | 多模态数据和标注 | 数据／感知基础设施，不宜作为主要检索算法对手 |
| [OVSG](https://arxiv.org/html/2309.15940) | 对象、智能体、区域节点及关系 | 查询子图／实体 | 非图像预算 | 三维包围盒与空间关系评分 | 空间关系建模，非本文虚拟观察者规则 | 区域／智能体位置作为输入 | 已有区域实体和查询图；未发现预测网格自动合成缺失参照的同一机制 |
| [3D-Mem](https://arxiv.org/html/2411.17735v3) | 三维对象与 Memory Snapshots | 共视对象簇／快照 | 覆盖压缩；类别 Top-\(K\) 不等于图像 \(K\) | 几何关联、共视性 | 主要由视觉模型推理 | 已检测对象支持快照组织 | 已有覆盖式视觉记忆；本文增加网格补充通道及查询时联合覆盖 |

| 方法 | 场景表示 | 检索／选择粒度 | 预算约束 | 几何／遮挡核验 | 视角依赖指代 | 主要假设 | 与本文差异 |
|---|---|---|---|---|---|---|---|
| [MapGPT](https://aclanthology.org/2024.acl-long.529/) | 在线语言化拓扑地图 | 导航节点／路径 | 导航过程约束 | 拓扑关系；非深度核验检索 | 导航方向提示，非专门指代算子 | 可行动节点与在线观察 | 地图服务路径规划，不是离线历史图像检索 |
| [NS3D](https://jiajunwu.com/papers/ns3d_cvpr.pdf) | 对象特征、嵌套符号程序 | 对象与关系 | 非图像预算 | 学习关系评分、软对象选择 | **高阶关系处理 viewpoint anchoring** | 关系模块接受训练 | 查询树、软执行已有；本文侧重确定性几何及网格回退 |
| [ZSVG3D](https://arxiv.org/html/2311.15383) | 三维对象、可执行视觉程序 | 对象／参照 | 非全局选帧预算 | 距离、投影和关系算子 | **房间中心虚拟相机朝向参照** | 检测候选与参考系假设 | 本文虚拟观察者的最直接先例 |
| [ViewRefer](https://arxiv.org/abs/2303.16894) | 多视角对象特征与扩写语言 | 对象／视角特征 | 非原图检索预算 | 几何一致描述与跨视角融合 | **显式处理** | 训练、视角扩写及原型 | 学习视角融合，不是几何执行后选原帧 |
| [OriGround](https://aclanthology.org/2026.findings-emnlp.2237/) | 朝向感知神经符号表示 | 对象／参照／观察者 | 具体图像预算未核实 | 摘要明确参照及观察者几何评分 | **显式推断观察者视角** | 多视图支持朝向估计 | 进一步压缩“观察者推理”的新颖性边界；网格机制待全文核查 |
| [VLM-Grounder](https://proceedings.mlr.press/v270/xu25c.html) | 图像序列、动态拼图、投影结果 | 图像／目标区域 | 交互式搜索，非本文集合目标 | 多视图集成投影 | 由视觉推理处理，非同一固定参考系 | 可定位目标的图像与投影信息 | 视觉搜索—反馈—三维输出，非预构建双索引 |
| [SeeGround](https://arxiv.org/abs/2412.04383) | 查询对齐渲染图＋空间文字 | 虚拟渲染视角／对象 | 渲染输入规模，非历史帧覆盖目标 | 三维投影与图文对齐 | **查询相关视角适配** | 可渲染三维场景 | 已处理文字目录信息不足；本文检索真实历史帧并合成网格代理参照 |

| 方法 | 场景表示 | 检索／选择粒度 | 预算约束 | 几何／遮挡核验 | 视角依赖指代 | 主要假设 | 与本文差异 |
|---|---|---|---|---|---|---|---|
| [OpenEQA](https://open-eqa.github.io/) | 记录历史或可探索环境 | 取决于基线 | 取决于基线 | 取决于基线 | 包含空间问题 | EM-EQA 与 A-EQA 两种设定 | 是任务基准，不是统一检索算法 |
| [GraphEQA](https://arxiv.org/html/2412.14480v2) | 在线三维场景图＋视觉记忆 | 相关历史帧、当前帧、探索目标 | **Top-2 记忆帧＋当前帧** | 三维建图；选帧按 SigLIP 相关性 | VLM 推理 | 在线建图、机器人行动 | 图像窗口有限，但非对象／网格边际覆盖选帧 |
| [Explore until Confident](https://arxiv.org/html/2403.15941v3) | TSDF／探索地图＋语义价值 | 后续探索方向 | 探索成本及停止条件 | 深度建图 | 非专门指代机制 | 能移动、置信度可校准 | 选择未来观测，本文选择已有观测 |
| [MemTree3D](https://arxiv.org/html/2608.18009v1) | 位置—对象—检测三层树 | 位置段→代表帧 | **Top-\(k\) 位置，每位置一帧** | 位姿分组；选帧按检测分数 | LLM 空间线索，非同一几何执行器 | 位姿、检测、跟踪 | 已有一次建索引、少帧检索和漏检上下文回退 |
| [Opti3D](https://henricwu.github.io/publication/2026-05-04-query-driven-evidence-retrieval-3d-qa) | BEV＋跨视图实例组 | 实例代表视图 | Top-\(K\) 实例证据 | **三维射线一致性去重**；深度遮挡细节未核 | 视点质量辅助，非已核实的观察者程序 | 位姿、点云、检测候选 | 已有全局 BEV＋局部优质原图；本文差异在可寻址网格和联合集合目标 |
| [FocusGraph](https://arxiv.org/html/2603.04349v2) | 时间片段场景图记忆 | 片段→原始关键帧 | **最多 \(K\) 帧** | 光流／外观去冗余；非三维遮挡核验 | 非专门观察者推理 | 训练片段选择器、视频图构建 | 关系记忆定位相关片段，再做二维视觉选帧 |
| [SCONE](https://papers.neurips.cc/paper_files/paper/2022/hash/828c6d69bdf91fca7f2b97c4dc214e94-Abstract-Conference.html) | 点云、占据概率、可见性预测 | 下一相机位姿 | 逐步采集，非历史图像 \(k\) | **预测可见性与表面覆盖增益** | 非语言指代 | 未知环境、深度传感器 | NBV 优化重建覆盖，不是任务相关历史证据 |
| [CSES](https://arxiv.org/html/2608.00714v1) | 帧特征、时间、查询相关性 | 待打分帧／输入关键帧 | 自适应数量及 \(K_{\max}\) | 非三维几何；视觉—时间相似性 | 未专门处理 | 图文相关性及覆盖饱和有效 | 与本文覆盖函数同属加权 max-sum；覆盖元素不同 |
| [全景 EM-EQA 视图选择](https://arxiv.org/html/2609.33288v1) | 全景历史及 cubemap 透视图 | 透视候选视图 | **固定 \(K\)** | 投影转换；去冗余用特征相似性 | 处理成像视角，不等于指代参考系 | 全景观察、微调 BLIP-2 | 相关性＋最大相似度阈值去冗余，与本文实际选择规则值得直接比较 |

**逐项 novelty verdict 如下。**判断依据是机制是否已有，而非论文是否使用 O2V、G2V 等相同缩写。

| 3.1 中的创新点 | Verdict | 最接近的 2–3 篇证据 | 可保留的贡献边界 |
|---|---|---|---|
| RGB-D 掩码反投影、几何＋语义跨视图实例融合 | **已有** | [ConceptGraphs](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf)、[OpenScene](https://arxiv.org/html/2211.15654v2)、[3D-Mem](https://arxiv.org/html/2411.17735v3) | 作为基础表征构建流程；具体清洗和融合策略可报告工程改进 |
| O2V/V2O 与原始视觉证据保留 | **已有；双向存储属于实现增量** | [OpenMask3D §3.2.1](https://arxiv.org/html/2306.13631v2)、[ConceptGraphs §II-A／A2](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf)、[3D-Mem §3.1](https://arxiv.org/html/2411.17735v3) | 强调查询接口、访问效率与质量权重；不能说首次连接对象和源图 |
| 面积、截断、距离、朝向、稳定性加权的视图质量分数 | **增量** | [OpenMask3D](https://arxiv.org/html/2306.13631v2)、[Opti3D](https://doi.org/10.1109/ACCESS.2026.3720265)、[MemTree3D](https://arxiv.org/html/2608.18009v1) | 特定质量指标组合及其效果；固定权重本身不是强方法创新 |
| BEV 网格＋G2V/V2G＋深度核验＋漏检回退 | **组件已有；联合机制可辩护新颖** | [OpenScene](https://arxiv.org/html/2211.15654v2) 已有深度测试；[Opti3D](https://henricwu.github.io/publication/2026-05-04-query-driven-evidence-retrieval-3d-qa) 已有 BEV＋局部证据；[MemTree3D](https://arxiv.org/html/2608.18009v1) 已有上下文定位漏检对象 | 贡献应限定为“可寻址网格通过几何可见性直接回查原帧，并与对象通道联合使用” |
| 多假设查询树＋实例／网格软先验＋递归执行 | **增量；组合有价值** | [NS3D](https://jiajunwu.com/papers/ns3d_cvpr.pdf)、[ZSVG3D](https://arxiv.org/html/2311.15383)、[OVSG](https://arxiv.org/html/2309.15940) | 需突出多假设如何融合、网格先验如何降低误过滤；不是首次神经符号解析或软评分 |
| 房间中心／指定位置的三维虚拟观察者 | **核心已有** | [ZSVG3D §3.3](https://arxiv.org/html/2311.15383)、[NS3D](https://jiajunwu.com/papers/ns3d_cvpr.pdf)、[OriGround](https://aclanthology.org/2026.findings-emnlp.2237/) | ZSVG3D 已明确“房间中心虚拟相机看向参照”；本文只能主张扩展、集成及适用情形 |
| 缺失参照时“预测 BEV 格→虚拟三维框→关系评分” | **最可辩护的局部新颖点，但尚需验证** | [OVSG](https://arxiv.org/html/2309.15940) 已有区域节点及三维框；[SeeGround](https://arxiv.org/abs/2412.04383) 应对目录信息不足；[MemTree3D](https://arxiv.org/html/2608.18009v1) 通过上下文恢复检索 | 在已核验文献中未发现完全相同的自动网格代理机制；应强调代理的不确定性和失效条件 |
| 对象—网格加权 max-sum 覆盖与边际增益选帧 | **数学形式已有；覆盖对象与任务结合是增量** | [CSES](https://arxiv.org/html/2608.00714v1)、[3D-Mem](https://arxiv.org/html/2411.17735v3)、[SCONE](https://papers.neurips.cc/paper_files/paper/2022/hash/828c6d69bdf91fca7f2b97c4dc214e94-Abstract-Conference.html) | 将三维对象与区域作为共同覆盖元素具有任务针对性；不能主张首次覆盖优化或首次次模选帧 |
| 视锥重叠抑制＋相机驻留质量 | **增量** | [全景 EM-EQA 视图选择](https://arxiv.org/html/2609.33288v1)、[FocusGraph](https://arxiv.org/html/2603.04349v2)、[Opti3D](https://doi.org/10.1109/ACCESS.2026.3720265) | 相对于特征／时间去冗余，明确验证三维重叠的额外收益；驻留只是图像质量代理 |
| 轻量几何缓存、原始视觉证据按需访问 | **工程增量** | [MemTree3D](https://arxiv.org/html/2608.18009v1)、[3D-Mem](https://arxiv.org/html/2411.17735v3)、[FocusGraph](https://arxiv.org/html/2603.04349v2) | 可作为效率贡献；需要加载、查询、建图及存储成本分开报告 |
| 整体双通道检索框架 | **可辩护的系统方法贡献** | [3D-Mem](https://arxiv.org/html/2411.17735v3)、[MemTree3D](https://arxiv.org/html/2608.18009v1)、[Opti3D](https://henricwu.github.io/publication/2026-05-04-query-driven-evidence-retrieval-3d-qa) | 证明双通道、缺失参照代理与联合选帧之间存在稳定协同，而不只是各模块串接 |

其中，虚拟观察者部分的先例尤其明确。ZSVG3D 已给出：

\[
R,T=\operatorname{Lookat}(P_{\mathrm{center}},P_{\mathrm{anchor}},up),
\]

再用投影横坐标判断左右、相机深度判断前后。本文式（15）—（16）采用水平前向轴和右向轴，属于同一参考系构造思路。**“不要求目标与参照同帧共视”是三维模型中进行虚拟投影的自然能力，不能单独作为新颖性证据。**[ZSVG3D §3.3](https://arxiv.org/html/2311.15383)

**最值得检查是否漏引的五篇是以下工作。**由于没有完整参考文献表，这里指“应优先核对的遗漏候选”，不认定它们已经被遗漏。

1. **ZSVG3D，CVPR 2024**：直接覆盖虚拟观察者和可执行三维关系程序；漏引会明显影响创新性论证。
2. **MemTree3D，ECCV 2026**：与“一次构建场景索引、查询时推理、只检索少量原图”高度接近，还讨论了目标漏检时的上下文定位。
3. **Opti3D，IEEE Access 2026**：同时具有全局 BEV、查询引导局部目标、跨视图几何去重和优质视图选择。
4. **SeeGround，CVPR 2025**：明确利用视觉证据弥补对象文字描述不足，并按查询适配观察视角。
5. **OriGround，EMNLP Findings 2026**：最新的观察者／对象朝向感知神经符号方法；必须先完成全文核查，再确定本文视角推理的剩余差异。

以上五篇的证据链接和元数据见清单；此外，**CSES 应用于约束次模优化的表述，OpenMask3D 应用于约束可见性索引的表述**。

**以下五条可直接用于“与现有工作区别”，但应配套引用与实验。**

1. > 在已有跨视图实例融合与对象级视觉记忆的基础上，本文进一步将规则空间网格作为可直接寻址的检索单元，构建对象和网格到原始观测的互补访问通道，使证据检索在目标未进入对象目录时仍可通过预测区域继续执行。

   对应比较：ConceptGraphs、OpenMask3D、3D-Mem。

2. > 与依据位置分组或实例代表视图排序获取证据的方法相比，本文在统一图像预算下评价新增视图对任务相关对象和空间区域的边际覆盖，并结合三维视锥重叠抑制重复观测。

   对应比较：MemTree3D、Opti3D；不要把该句扩展成未经证明的最优性保证。

3. > 本文沿用可执行空间关系与观察者参考系的三维指代求解思路，将语言解析产生的实例和网格预测作为软先验，并在参照实例缺失时由预测网格构造空间代理，以降低不完整对象目录对关系执行的影响。

   对应比较：NS3D、ZSVG3D、OVSG。这里明确承认参考系与程序执行的已有基础。

4. > 与基于图像嵌入或时间相邻性控制冗余的选帧方法相比，本文利用配准 RGB-D 观测估计区域可见性，并以三维空间重叠辅助选择互补证据；其收益主要面向视角变化大、重复观测多的室内扫描序列。

   对应比较：全景 EM-EQA 视图选择、FocusGraph、CSES；最后一句应由对应子集实验支持。

5. > 与通过机器人移动获取新观测的主动问答和下一最佳视点方法不同，本文研究已采集观测序列上的预算受限证据检索，重点优化查询相关性、几何可见性和输入紧凑性。

   对应比较：Explore until Confident、GraphEQA、SCONE。此区别界定问题设定，不应据此推导性能优劣。

**目前证据不足或需要修正的 claim，主要有以下几类。**

1. **“已有方法只存文本，丢失原始视觉细节”。**  
   这一概括不成立。ConceptGraphs 保存来源视图并选取信息视图；3D-Mem 直接采用图像快照；GraphEQA 联合场景图和视觉记忆。可以批评“仅使用文本摘要的具体推理接口”，不能概括整个场景图研究方向。[ConceptGraphs](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf)、[3D-Mem](https://arxiv.org/html/2411.17735v3)、[GraphEQA](https://arxiv.org/html/2412.14480v2)

2. **“首次提出对象—视图可见性或深度遮挡核验”。**  
   OpenMask3D 已明确构造对象—视图可见度并用深度过滤遮挡；OpenScene 也有深度一致性测试。双向索引主要是可见性关系的正向／反向访问组织，不能因命名不同就认定新颖。[OpenMask3D §3.2.1](https://arxiv.org/html/2306.13631v2)、[OpenScene](https://arxiv.org/html/2211.15654v2)

3. **“消除漏检证据断链”“保障小物体召回”。**  
   网格通道提供补充机会，但仍取决于区域预测、原始扫描是否拍到目标、深度有效性和选帧预算。并且式（13）包含对象可见度分支，因此完整得分不是完全独立于检测。宜写“缓解”“提供补充召回通道”，并分别测试目标漏检、参照漏检、两者同时漏检。

4. **“虚拟观察者解决视角依赖”。**  
   房间中心观察者只是一个假设，未必等于说话者视角或对象固有朝向。“从门口看”“面对柜子”“椅子自身左侧”可能对应不同参考系。应分别报告显式观察者、隐式观察者和参考系歧义样本；`between` 通常也不必归入观察者依赖关系。

5. **“完整算法具有标准次模贪心保证”。**  
   固定查询、非负权重下，式（2）确实是单调次模函数；加入非负驻留模块项仍成立。减去非负两两重叠后，目标可以保持次模性，却可能失去单调性。更关键的是，**式（18）的阈值乘性惩罚不是式（1）加性两两重叠目标的真实边际增益**，不能直接继承 \(1-1/e\) 保证。CSES 的保证针对其明确的 max-sum 目标和真实边际算法。[CSES §4.2／附录 E–F](https://arxiv.org/html/2608.00714v1)  
   XML 稿改用“贪心”较稳妥，但“近似求解式（1）”仍应说明是启发式，或统一目标与算法。

6. **“联合覆盖保障目标—参照共视”。**  
   \(\sum_o\max_{v\in S}\psi(v,o)\) 允许目标和参照分别出现在不同帧；它保障的是集合层面的各自覆盖，并不奖励同帧共视。若要主张共视优化，需要显式共视项、约束，或独立统计“至少一帧同时清楚包含目标与参照”的指标。

7. **若干公式与术语的解释超出了实际定义。**  
   式（12）的 \(|d_{\mathrm{proj}}-D|\le\delta\) 更接近“接近观测表面的深度一致性”，在体积内均匀采样时不等同于一般自由空间无遮挡；式（8）的光轴夹角体现画面方向／中心性，缺少表面法向，不能严格称为“入射角”；式（13）正文称对象“平均可见度”，公式实际为最大值。多假设的 \(p_m\) 如何进入最终融合也未在所给公式中交代。

8. **“各模块消融证明独立贡献”。**  
   目前找到的配对记录支持总体 `hit@3` 从 **79.17% 提升到 89.50%**，但同时包含解析 schema 修复、类别对齐、软先验、虚拟锚点、网格回退等变化；不能由这个整体升级直接归因于单个模块。表 3-4 的 **83.50%、86.83%、88.17%** 中间结果，在本次抽查的基准记录中尚未找到对应实验。它们需要各自的配置、代码版本及逐样本日志，而非只保留递进表格。[配对评测记录](/home/ysh/codebase/agentic-3d-task/docs/benchmark/nr3d/v2_keyframe_grid_upgrade_coverage_strat600_20261104.md:1)

9. **实验来源和比较口径需要纠正。**  
   草稿 OpenEQA 表中的 **94.2%→0.3%** 重复帧变化，目前可定位的来源是 **NR3D `strat600` 的 `select_by_proposal` 诊断**，不能直接作为 OpenEQA 跨区域检索结果。旧 3DVLM 数字来自历史代码／配置，相关记录还明确提示代码演进与 CLIP 开关混杂，因此“所有方法统一同一模型”的表述需要逐行核实。[NR3D 诊断来源](/home/ysh/codebase/agentic-3d-task/docs/benchmark/nr3d/v15_flip41_tools_upgrade_20261104.md:12)、[历史比较限制](/home/ysh/codebase/agentic-3d-task/docs/benchmark/nr3d/v1_keyframe_targetid_coverage_strat600_20260608.md:88)

10. **指标和效率证据需补全。**  
    当前代码的 `Recall@k` 分母为 \(\min(k,|GT|)\)，不是标准检索召回率的 \(|GT|\)；应显式给出定义，或改称“归一化 Top-\(k\) 可见帧命中比例”。`hit@3` 只证明有一张目标可见帧，不证明文字、材质、状态等答案证据充分。[指标实现](/home/ysh/codebase/agentic-3d-task/scripts/eval_keyframe_coverage_nr3d.py:141)  
    约 **10秒→72毫秒** 的加载优化有运行记录支持，但“完全无精度损失”仍需等价性检查；**165.1秒→41.0秒** 则伴随缓存、索引对齐、裁剪修复、工具接口和提示调整，不能全部归因于场景表征或选帧算法。[OpenEQA 升级记录](/home/ysh/codebase/agentic-3d-task/docs/benchmark/openeqa/v6_mapfirst_tools_overhaul_pilot300_20261104.md:11)

论文当前最值得优先补强的证据，是在同一模型、同一候选序列、同一图像预算下，对比 **3D-Mem／MemTree3D 式检索、逐帧相关性 Top-\(k\)、对象覆盖、网格覆盖、联合覆盖**，并单独报告缺失参照与目标漏检子集。这样才能把“有效的系统组合”进一步论证为有明确作用机制的方法贡献。