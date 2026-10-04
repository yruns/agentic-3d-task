# 基于多模态大模型的室内长程视觉语言导航研究报告

调研截止日期：2026 年 10 月 3 日。

研究路线补充：若选择导航方向，优先关注冻结通用 LLM/VLM，允许仅训练一个小模块。[补充报告](zero_shot_indoor_llm_navigation_20261003.md)据此重新筛选零样本方法并调整基线建议。本篇仍作为导航领域背景综述，其中推荐 StreamVLN、DualVLN 为实现主干的部分建立在允许导航任务训练的条件下。是否选择导航作为研究点，另参考[室内三维跨任务能力报告](indoor_3d_llm_capability_survey_20261003.md)。

本报告服务于研究生学位论文选题“基于多模态大模型的室内长程视觉语言导航方法研究”，重点回答四个问题：该领域目前如何定义任务和评价效果，主流方法如何组织语言、视觉、历史与动作，近年研究取得了哪些可核实的进展，以及哪些问题仍值得开展独立研究。

范围限定为室内三维环境中的视觉语言导航。无人机、室外驾驶、设备操作教学与机械臂控制不作为本报告的研究对象。目标搜索、具身问答和长期记忆仅在帮助辨明任务边界或提供相邻技术时讨论。本报告不要求新研究复用已有课题的算法或代码。

## 一 主要判断与证据范围

**研究方向成立，但题目中的每一个关键词都需要明确。** “室内三维”描述交互环境，不意味着必须将点云直接输入大模型；“长程”可能指较长路线、较长语言指令或连续多个子任务，三者不能互相替代；“视觉语言导航”要求动作影响后续观察，静态图像问答不构成导航实验。

**方法已经从专用导航网络发展为多种并行路线。** 一类方法冻结通用语言或视觉语言模型，以地图、观察描述和工具接口组织决策；另一类把导航轨迹用于多模态模型训练，使其直接预测动作或局部目标；还有方法把高层推理与快速执行分开。新研究应选择一种清楚的比较对象，不能把不同传感器、训练数据、动作接口下的成绩视为同一排行榜。

**“进度感知”“主动观察”“历史记忆”和“回退”均已有研究。** 早期工作已经学习任务进度及回退策略，近期 Progress-Think、SeekVLN 等进一步把语义进度、证据获取和策略学习结合。报告后文将区分这些已有能力与拟议的错误状态修正，不能以“现有方法不会纠错”为研究前提。[Self-Monitoring](https://arxiv.org/abs/1901.03035)、[Progress-Think](https://arxiv.org/abs/2511.17097)、[SeekVLN](https://arxiv.org/abs/2609.37353)

**此前对“进度修正与恢复”的创新性判断需要收紧。** CorrectNav 已经使用自然错误轨迹训练纠正行为，BudVLN 研究错误历史与纠正监督的重新对齐，Navigation Heads 已连接偏航检测与回退控制，AVERT-VLN 则加入独立语义监测和人工恢复。它们在自主性、传感器和训练方式上不同，但都属于必须比较的直接近邻。[CorrectNav](https://arxiv.org/html/2508.10416v1)、[BudVLN](https://arxiv.org/html/2602.06356v1)、[Navigation Heads](https://arxiv.org/html/2603.13782v1)、[AVERT-VLN](https://arxiv.org/html/2609.39579v1)

**基准应先于具体模型确定。** R2R-CE 和 RxR-CE 适合作为连续空间指令导航的基础，LHPR-VLN 可补充多阶段任务评价；GOAT-Bench 和 LangMap 对应连续开放目标搜索，适合另一种任务定位。它们的能力要求相关，但不相同。[VLN-CE 官方实现](https://github.com/jacobkrantz/VLN-CE)、[LH-VLN 官方实现](https://github.com/HCPLab-SYSU/LH-VLN)、[GOAT-Bench](https://github.com/Ram81/goat-bench)、[LangMap](https://arxiv.org/html/2602.02220v3)

### 证据使用说明

本报告以论文原文、会议论文集、作者项目页和官方代码仓库为主要依据。重点方法核对了方法或实验段落；其他研究作为脉络条目时，仅使用其原文摘要、正式论文信息或官方发布说明所能支持的结论。正文中的“本文建议”“可研究问题”属于分析判断，不是论文已经证实的结果。

报告中的公开状态指截止日期可见的数据、代码、权重或运行说明，不等同于本机已复现。此次未下载大规模场景资产、未训练模型、未运行导航基准。2026 年预印本单独注明，不能与经过同行评审的结论赋予完全相同的证据强度。报告不将不同论文的最高分拼成总体排名，也不将小规模真机展示视为长期自主部署证明。

阅读顺序上，第四节用于确定基准，第五、六节用于了解主要方法及直接近邻，第七至九节讨论训练、适应与效率，第十、十一节给出研究判断和实验路线。第十三节汇集主要文献。对论文的限制分析区分“作者已报告的限制”“本次代码核验发现”和“尚需实验验证的判断”。

## 二 任务定义与研究边界

### 2.1 视觉语言导航的基本闭环

视觉语言导航（Vision-and-Language Navigation，VLN）要求智能体根据自然语言指令和第一视角观察，在环境中连续选择动作，最终满足任务要求。一个典型过程为：接收指令，获取当前图像，将观察与指令、历史经历对应，选择动作，执行后获得新的观察，再判断下一步或停止。

对应的现实能力是：机器人进入不熟悉的室内环境后，能够执行“沿走廊经过第二个门，进入右侧房间，在靠窗的桌子旁停下”一类指令，并在相似房间、遗漏地标或执行偏差出现时继续完成任务。这个示例是任务说明，不是已完成的系统演示。它强调开放语言、环境理解和连续行动，而不要求预先为每种表达编写路径脚本。

可将策略写为 `a_t = π(I, o_0:t, a_0:t-1)`。其中，`I` 是指令，`o_t` 是当前协议允许的观察，`a_t` 可以是转向、前进、停止，也可以是局部航点或短时轨迹。环境完整状态通常不可见，因此只依据当前图像很难区分重复走廊、相似房间或同一路段的不同任务阶段。这是部分可观测决策问题，而不只是图文匹配。

R2R 将真实室内图像、语言路线和离散导航环境结合，奠定了这一任务的常用形式；VLN-CE 随后移除固定导航图上的跳转假设，使智能体通过动作在连续可通行空间中移动。[R2R 原论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html)、[VLN-CE 官方说明](https://github.com/jacobkrantz/VLN-CE)

### 2.2 几类相近任务不能混用

| 任务 | 主要输入 | 需要解决的问题 | 典型成功条件 |
| --- | --- | --- | --- |
| 指令跟随导航 | 包含地标、转向或阶段顺序的路线描述 | 当前执行到哪段指令，下一步如何行动 | 正确到达并停止，必要时满足路线忠实度 |
| 开放目标导航 | 类别、图片或实例描述 | 目标可能在哪里，如何搜索和确认 | 到达正确目标实例或有效观察位置 |
| 多阶段导航 | 若干具有顺序或依赖的子任务 | 保持阶段关系，避免前序失败传递 | 整任务及各阶段满足原协议 |
| 主动具身问答 | 关于环境的问题 | 获取足以回答问题的观察 | 答案正确，同时考虑取证和探索代价 |
| 持续巡逻 | 区域或检查点及持续访问要求 | 重复覆盖、访问频率、任务分配 | 重访间隔、覆盖率或事件发现延迟 |

表中分类是为了组织研究问题，不意味着任务之间完全隔离。同一个系统可以完成多种任务，但应分别遵守各自的信息边界、动作和评价协议。例如，GOAT-Bench 的连续目标搜索不能直接证明它能忠实执行 RxR 的详细路线；R2R 的终点到达也不能证明全天周期巡逻有效。

### 2.3 长程导航至少包括三个维度

第一是**运动过程长**：更多动作和更大移动距离，使定位、感知及执行误差累积。第二是**指令依赖长**：描述包含较多地标、顺序、转向或指代，需要持续对齐语言与视觉。第三是**任务阶段多**：完成一个局部目标后继续执行后续子任务，需要保存阶段状态和历史经验。

这三个维度应分别统计。增加输入词数不一定增加运动距离；串联目标不一定要求理解复杂路线；大环境也不必然意味着每个 episode 很长。对于同一方法，应按实际动作数、参考路径长度、指令长度和子任务数量分组分析，而不只汇报一个平均分。

“连续环境”还不等于“动态环境”。常用室内扫描场景主要是静态世界；智能体在连续空间中运动，不代表实验包含移动行人、物体被搬走或门状态变化。若研究这些扰动，需要使用相应公开协议，或明确作为自行增加的鲁棒性实验。

### 2.4 大模型与三维导航系统的分工

本报告中的 LLM 指语言模型，VLM 或 MLLM 指能够结合视觉与语言输入的模型，VLA 则强调模型输出与动作执行的连接。三者的命名在不同论文中并不完全统一；因此，判断一种方法是否适合作为比较对象，应检查实际输入、输出和训练方式。BERT 式跨模态编码器也不应仅因为参数较多就与通用生成式 MLLM 混称。

多模态大模型的潜在优势主要是开放语言理解、视觉语义关联、任务阶段推断及对执行反馈的解释。几何模块可以承担可达性判断、局部路径生成和避障；控制模块负责执行。在研究设计中，将语言推理置于高层、固定低层执行器，是隔离算法贡献的一种合理选择，但不是唯一正确架构。

显式三维地图也不是性能提升的先验保证。LangMap 已提供较强的 RGB-only 基线。地图是否值得引入，需要比较它对实例身份、空间关系、历史回访和动作可执行性的收益，以及建图与维护成本。[LangMap](https://arxiv.org/html/2602.02220v3)

## 三 研究发展脉络

### 3.1 从序列学习到任务进度与回退

早期 VLN 通常将指令编码为语言特征，将视觉观察及历史压缩到循环状态，再预测下一导航动作。Speaker-Follower 进一步用“根据轨迹生成指令”的反向模型扩充训练数据，并评价候选轨迹与指令的对应程度。这说明数据构造和语言对齐从一开始就是主要问题，而非大模型出现后才有的需求。[Speaker-Follower](https://papers.neurips.cc/paper_files/paper/2018/hash/6a81681a7af700c6385d36577ebec359-Abstract.html)

此后研究开始显式估计进度、使用辅助监督并根据进度变化决定是否回退。这条线为今天的语义进度模型提供了直接历史基础。需要区分的是，早期标量进度、当前语言化的任务状态，以及实际物理回退是三个层次；它们相互联系，但不能使用同一名称概括。

### 3.2 从任务专用表示到预训练和历史建模

PREVALENT 将图像、指令和动作三元组用于预训练，再适配导航任务；Recurrent VLN-BERT 通过循环跨模态状态支持历史相关决策；HAMT 对历史中的图像、全景及时间关系分层编码。这些方法说明，导航性能既受局部语言视觉匹配影响，也受历史组织方式影响。[PREVALENT](https://arxiv.org/abs/2002.10638)、[Recurrent VLN-BERT](https://openaccess.thecvf.com/content/CVPR2021/html/Hong_VLN_BERT_A_Recurrent_Vision-and-Language_BERT_for_Navigation_CVPR_2021_paper.html)、[HAMT](https://proceedings.nips.cc/paper_files/paper/2021/hash/2e5c2cb8d13e8fba78d95211440ba326-Abstract.html)

这类视觉语言预训练模型与今天的通用生成式 MLLM 有联系，但不宜混为一类。论文综述中应说明：它们是否使用通用语言生成骨干，是否通过 token 序列输出动作，以及是否具备通用指令跟随接口。

### 3.3 地图和分层控制成为重要支撑

DUET 在线构建拓扑地图，在全局地图推理和局部细粒度语言视觉匹配之间动态融合；BEVBert 引入地图形式的多模态预训练。地图的作用不仅是保存对象名称，还可以支持远距离回访和全局动作选择。[DUET](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Think_Global_Act_Local_Dual-Scale_Graph_Transformer_for_Vision-and-Language_Navigation_CVPR_2022_paper.html)、[BEVBert](https://arxiv.org/abs/2212.04385)

ETPNav 将预测航点组织为在线拓扑图，通过跨模态规划器选择目标，再由避障控制器执行。其研究特别讨论了禁止沿障碍物滑动后性能变化及脱困问题。因此，后续若讨论“导航错误恢复”，必须与地图回访、局部避障脱困等已有能力区分。[ETPNav 正文](https://arxiv.org/html/2304.03047v3)

### 3.4 通用大模型形成多条方法路线

2023 年以来，一类工作通过提示、工具和环境表示将冻结 LLM 用于导航；另一类使用导航轨迹微调 VLM，使其从图像或视频直接预测动作。随后，流式记忆、高低频协同、进度推理、主动观察和测试时适应逐渐成为研究重点。下表给出理解这些方法的统一框架，具体论文和证据在后文展开。

| 方法路线 | 大模型承担的工作 | 常见输入与输出 | 主要收益 | 主要代价或风险 |
| --- | --- | --- | --- | --- |
| 冻结模型与工具规划 | 理解任务、选择候选位置、调用工具 | 文本化地图或图像加历史，输出动作或航点 | 快速迁移，便于审计模块作用 | 观察文本化损失、调用延迟、模型版本波动 |
| 导航轨迹微调 | 学习语言视觉到动作的对应 | 当前图像、历史视频、指令，输出动作序列 | 统一学习与快速推理 | 数据和训练成本，分布外误差 |
| 地图辅助决策 | 在位置、对象或区域结构上规划 | 拓扑图、BEV、局部观察，输出下一目标 | 支持空间记忆和回访 | 地图错误可能持续影响决策 |
| 分层与快慢协同 | 大模型处理语义，快速策略处理执行 | 语义子目标与当前观察，输出局部轨迹 | 缓解高层推理与动作频率不匹配 | 目标过期、接口失配、不同模块贡献难区分 |
| 进度与主动观察 | 判断执行阶段、是否需要更多信息 | 指令片段、历史、补充视图，输出进度及动作 | 处理局部观察不足和阶段混淆 | 额外观察和推理成本，进度标签可靠性 |
| 测试时适应 | 根据新环境交互调整策略 | 观察转移、反馈或历史经验 | 改善新环境适应 | 错误自监督、负迁移、评测信息泄漏 |

上述分类可以重叠。地图不是与大模型竞争的独立路线，主动观察也不必然要求在线更新模型参数。真正需要比较的是信息、决策、训练与执行接口，而不是方法名称。

## 四 公共基准与评价协议

以下内容分别核对论文、公开数据和代码实现。每个基准内部固定比较条件，跨基准保留各自原有协议。

### 4.1 R2R 与连续环境移植

#### 4.1.1 原始离散 R2R

R2R 来自真实建筑的全景图像。原论文报告 90 个场景、7,189 条路径，每条有 3 条人工指令，共 21,567 条，平均约 29 个词。智能体从指定起始位姿出发，根据语言选择可达的相邻扫描视点并调整视角，最后停止。它不是在任意 3D 坐标上走动的机器人模拟；导航图提供了可达候选。原始 RGB 设置与后来模拟器支持 RGB-D 的能力也应区分。[R2R 原论文](https://arxiv.org/pdf/1711.07280)、[Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator)

官方公开 train、val-seen、val-unseen、test，test 为未见建筑；每条路径的三种表述与路径 ID 不能被当作三个独立场景。官方仓库提供任务下载、seq2seq 基线、训练和本地验证 evaluator；test 轨迹通过外部评测。MP3D 图像资产需要另行申请。现代 panorama 方法经常使用 36 个离散朝向视图，但这不是所有 R2R 方法天然相同的输入。[R2R 官方任务说明](https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/README.md)

**使用建议：**如果主论文面向连续空间的室内执行，R2R 离散图结果不应替代 R2R-CE。它可以用于快速研究语义阶段推断，但沿图选节点与实际执行动作之间有明显能力差距。

#### 4.1.2 R2R-CE 的精确公开数据口径

VLN-CE 将 R2R 的路线移植到 Habitat 的 MP3D 重建网格。建议固定 **R2R_VLNCE_v1-3**，它在 2022-02-03 修正了早期版本与 R2R 不一致的初始朝向，并重算 oracle 路径。官方公开的数量如下；这里的 episode 对应一次指令条件导航，不是建筑或去重路线数量。[官方数据版本与统计](https://jacobkrantz.github.io/vlnce/data)

| 划分 | 指令 episode | 出现场景数 | 用途 |
|---|---:|---:|---|
| train | 10,819 | 61 | 训练 |
| val_seen | 778 | 53 | 训练建筑内验证 |
| val_unseen | 1,839 | 11 | 未见建筑的主要公开评测 |
| test | 3,408 | 18 | 隐藏 GT 的外部评测 |

v1-3 minimal 文件约 3 MB；preprocessed 版约 250 MB，另外含 EnvDrop 增广、词向量及训练/验证 GT 文件。应单独披露是否使用增广，不把增广轨迹当成人工指令数据。`goals`、`reference_path`、GT actions、距离等字段即使存在于 JSON 中，也不能在测试时交给策略。

#### 4.1.3 默认传感器、动作和成功条件

官方默认 `vlnce_task.yaml`：最大 500 个环境步，前进 0.25 m，左右转 15°，动作 STOP/FORWARD/LEFT/RIGHT；RGB 224×224、HFOV 90°，depth 256×256；允许 sliding；成功距离阈值 3 m。默认任务同时列有 shortest-path/oracle-progress sensors，它们是训练与诊断设施，不能据此认定测试策略可以读取 oracle。[官方 R2R-CE 配置](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/vlnce_task.yaml)

这只是明确可定位的一套配置。2025–2026 的大模型方法可能采用单 RGB 图像、不同分辨率、动作块、12 视图 panorama 或 waypoint。研究者必须选定一个强基线的完整评测配置并对自己的方法保持一致，不能只凭都叫 R2R-CE 就横向比较。

#### 4.1.4 指标和可用性

主指标建议 SR、SPL，并报告 NE、OSR、TL；若研究路径忠实性和偏航，还应报告 nDTW/SDTW。SR 回答最终是否完成，OSR 回答途中是否曾到达有效区域，二者差距有助于识别终止失败；它不能独自证明语言理解错误。

官方仓库有 train/eval/inference 入口、CMA 基线与预训练资源、轨迹导出方式。其原始环境固定 Python 3.6、Habitat 0.1.7，环境年代较早。README 还记录同一模型因硬件/Habitat build 在论文与 leaderboard 上有差异，所以新项目必须保存环境版本和完整 evaluator 配置。[VLN-CE 仓库](https://github.com/jacobkrantz/VLN-CE)

### 4.2 RxR-CE 长路线与长语言的核心基准

#### 4.2.1 为什么它比 R2R-CE 更贴近本题

原始 RxR 支持英语、印地语、泰卢固语，并有 Guide/Follower 轨迹与密集的语言—视觉时间对齐。它有意弱化“指令等于最短路”的偏置。原始 RxR 的总规模不能直接写成 RxR-CE 的可用 episode 数，因为连续移植涉及轨迹筛选。[RxR 原始公开仓库](https://github.com/google-research-datasets/RxR)

ETPNav 对所用 CE 数据的统计为：R2R-CE 平均 9.89 m/32 words；RxR-CE 平均 15.23 m/120 words。该表中的 RxR-CE train/val_seen/val_unseen/test_unseen 分别为 60,300/6,746/11,006/9,557 条指令，涉及 59/57/11/17 个建筑。该数量用于描述该论文的三语言 CE 设置，不应替代项目实际加载清单。[ETPNav 表 I](https://arxiv.org/html/2304.03047v3#S4)

**项目清点要求：**分别保存语言、guide/follower、scene、instruction_id、trajectory_id、split 的数量和交集。尤其不能将“英语 val_unseen”与“三语言 val_unseen”都写成完整 RxR-CE 而不注明。开发时的小样本与最终完整结果应保留两个名称。

#### 4.2.2 官方 challenge 配置

官方 RxR 英语任务配置规定：RGB-D 640×480、HFOV 79°，相机/agent height 0.88 m、radius 0.18 m；不允许 sliding；左右转 30°、抬头低头 30°，STOP/前进/左右转/抬低头，共 500 步，成功阈值 3 m；英语筛选为 en-US、en-IN，默认使用 guide。[RxR task 配置](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/rxr_vlnce_english_task.yaml)

官方 challenge 还明确要求前进 0.25 m，并指出 panorama waypoint 模型不能未经适配直接提交；正式 test-challenge 需要三种语言全部 episode 的预测。论文中常见的英语子集、单 RGB 大模型结果可以成立，但必须明确它是相应研究设置，不冒充全语言 challenge 排名。[RxR-Habitat 官方提交说明](https://github.com/jacobkrantz/VLN-CE#rxr-habitat-challenge)

#### 4.2.3 指标选择与协议陷阱

建议 RxR-CE 的主表同时包含 **SR、SPL、nDTW、SDTW**。对于非最短路线，单看是否到终点会奖励抄近路；单看轨迹相似度又可能让接近正确路线但最终失败的方法显得良好。nDTW 衡量有序轨迹匹配，SDTW 将其约束到成功轨迹。[nDTW/SDTW 原论文](https://arxiv.org/abs/1907.05446)

需要固定：使用 guide 还是 follower 参考；坐标序列怎样采样；是否执行相同的成功阈值；macro action 是否展开计入实际步数和路长；转头获取额外图像的物理和计算成本。不要为了提升 SPL，将跟随非最短语言路线改成知道目标坐标后的最短路规划。

公开资源包括 CE episodes、英语/印地语/泰卢固语 CMA checkpoint、BERT 特征、验证 evaluator 与预测合并脚本。完整仿真仍依赖 MP3D 许可资产。本次核验到的是资源入口及配置，未验证所有外链权重可完整下载、也未试交隐藏测试。

### 4.3 LHPR-VLN 多阶段任务与公开实现

#### 4.3.1 多阶段目标任务与路线指令

LH-VLN 是任务定义，LHPR-VLN 是基准，NavGen 是数据生成流程，MGDM 是论文方法。任务包含依次到达不同对象/房间的多阶段指令，同时生成与相关轨迹对应的 step-by-step 指令。公开数据 loader 将 `lh_task` 与 `st_task` 列表分别返回；验证入口也分别记录长任务和 step 任务指标。[论文](https://arxiv.org/html/2412.09082v2)、[dataset.py](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/utils/dataset.py#L97)、[train.py](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/train.py#L188)

“把物品从 A 带到 B”是任务语言形式，但公开 action space 只有左右转、前进、STOP；不能宣称模型完成了抓取、携带与放置。论文的 dynamic memory 也不意味着这个基准已经包含运动的人、移动物体或实时变化的建筑。

论文报告 216 场景、3,260 个 multi-stage 和 step-by-step VLN tasks，平均约 150 动作步。**不应未经核对就将 3,260 写成完整多阶段 episode 数。**[论文 §3.2](https://arxiv.org/html/2412.09082v2#S3.SS2)、[CVPR 2025 正式论文入口](https://openaccess.thecvf.com/content/CVPR2025/html/Song_Towards_Long-Horizon_Vision-Language_Navigation_Platform_Benchmark_and_Method_CVPR_2025_paper.html)

#### 4.3.2 对当前公开 episode 包的实际清点

本次仅在内存读取约 281 KB 的 `episode_task.zip`，未下载场景、图像、模型或运行评测。数据 revision 为 `af66785e0b456639088d673bc100dcc63f2df997`（2025-05-30）。8 个 batch 合计 **1,052 条 lh_task、2,290 条 st_task**，两种任务条目相加为 3,342。它与论文概括数量不同，应该以“当前公开发布版清点”标记。[固定版本 episode 包](https://huggingface.co/datasets/Starry123/LHPR-VLN/resolve/af66785e0b456639088d673bc100dcc63f2df997/episode_task.zip)、[数据集页面](https://huggingface.co/datasets/Starry123/LHPR-VLN)

官方 `split_by_scene=True` 时，会先合并 batch，再按 Scene 数字前缀分割：`<700` train，`700–799` valid，`>=800` test。所以配置里 batch_1–5/6/7–8 不能直接等同最终 train/valid/test。[划分实现](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/utils/dataset.py#L297)、[默认配置](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/configs/lh_vln.yaml)

| 当前公开版，按上述代码划分 | 完整 LH 任务 | 配对 ST 任务 | 场景数 | LH 子任务数分布：2 / 3 / 4 |
|---|---:|---:|---:|---:|
| train | 756 | 1,659 | 112 | 327 / 352 / 77 |
| valid | 91 | 193 | 14 | 45 / 43 / 3 |
| test | 205 | 438 | 27 | 82 / 111 / 12 |

这些数字是对公开 JSON 的确定性计数，不是作者论文原实验 split 的恢复，也不是成功运行后的有效 episode 数。场景是否可加载、是否存在不可达目标、数据字段是否都有效，仍需要后续 simulator preflight。四子任务测试只有 12 条，细分结论的统计不确定性必须如实报告。

#### 4.3.3 传感器、动作、STOP 与预算

公开配置最大步数为 **整条任务 500 步**，success_dis 为 1 m。`actor()` 遇到 STOP 会无条件推进 stage，不论这一阶段是否成功；最终阶段 STOP 或总预算耗尽结束任务。成功检测使用 geodesic distance。模型因此必须在 STOP 之前修正错误的完成判断；按原协议，错误 STOP 后不能撤销环境阶段并把之前失败目标补判成功。[simulation.py:73–132](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/habitat_base/simulation.py#L73)

官方 challenge 文字描述默认三视图 `+60°/0°/-60°`、366×366 RGB，可提供 depth 和世界坐标 pose；而公开 simulator config 渲染 512×512，Spot/其他 embodiment 相机高度也不同。代码动作前进 0.25 m、转向 30°。应报告渲染分辨率、视觉编码器 resize 后分辨率和真实喂给策略的模态，不能混成一个数字。[challenge 说明](https://github.com/HCPLab-SYSU/LH-VLN/blob/master/contest/DetailedDescription.md)、[simulator config](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/habitat_base/config.py)

公开 `validate()` 给策略的是 instruction、view features、pose；`get_info()` 另含 target coordinate 和 geodesic distance，验证过程还计算并打印 expert action。这些服务 evaluator/诊断的数据不应进入大模型提示词或恢复模块。适配大模型时应采用白名单组装 observation，不能直接序列化整个 `info`。[策略输入边界](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/utils/agent.py#L118)

#### 4.3.4 指标的真实含义

公开实现的 SR 由 `all(result['successes'])` 得到，已经是整条多阶段任务的全部成功；OSR 对各阶段 oracle 标志同样取 all。ISR 是所有子任务成功数除以子任务总数。CSR 对当前和相邻前一阶段成功作加权，**不等于整任务 AND**。CGT 进一步按 GT 子任务步数加权。公开 `spl()` 用 `gt_steps/path_steps`，因此不是通常以米为长度单位的 SPL；报告中可标为“官方 SPL（动作步数口径）”。[聚合调用](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/train.py#L207)、[指标实现](https://github.com/HCPLab-SYSU/LH-VLN/blob/3492080382aba031e9d82a4f630c2b1cdf57df81/utils/metrics.py)

CSR 的直观反例：三阶段成功标志 `[1,0,1]` 的官方 CSR 为 `(3+0+1)/9=4/9`，但整任务 SR 为 0。这是按代码公式演算，不是模型实验。只报告 CSR 提升会掩盖整项工作仍然失败。

challenge 又增加 TAR，最终组合分为 `0.4*TAR+0.2*ISR+0.2*CSR+0.2*CGT`。该组合不是原论文全部实验的统一主指标；论文的 60°水平视域成功条件也没有在本次读到的公开 `actor()` 成功判断中出现。建议保留原代码结果，另将严格实现论文条件的结果作为独立协议，不能悄悄修改 evaluator 后仍横比旧表。[challenge 指标页](https://github.com/HCPLab-SYSU/LH-VLN/blob/master/contest/metrics.md)、[论文成功定义](https://arxiv.org/html/2412.09082v2#S3.SS2)

#### 4.3.5 可用性和选用判断

目前公开数据有 Hugging Face、ModelScope 两个入口；有 Habitat-Sim 0.3.1 环境说明、NavGen、训练/验证、SFT 代码。仓库明确说当前提供的是适配 Llama/Qwen 的简化基线；基础视觉/语言 encoder 权重存在不等于原论文 MGDM 训练后 checkpoint 已完整发布。因此，其公开资源可作为搭建起点，但完整复现条件仍需核验。[官方 README](https://github.com/HCPLab-SYSU/LH-VLN)

**建议：**保留 LHPR 作为第三个核心压力测试，但先固定公开 evaluator commit 和完整观察配置，在 91 个 valid LH 任务上开发并复现基线，冻结方法后评测 205 个 test LH 任务；不要把它作为唯一支撑创新的主基准。

### 4.4 其他室内基准的定位

#### 4.4.1 R4R 长路线与路径忠实性

R4R 将 R2R 中末端与下一条起点接近的路径和指令拼接。官方提供生成器；默认连接阈值 3 m，不限制 heading。README 的默认输出是 train 233,613 条指令/25,930 条路径，val_seen 1,035/115，val_unseen 45,162/5,018；平均参考路径约 20 m。它增加了曲折、非最短路线，而不是新增数量相等的独立房屋。[R4R 生成代码及统计](https://github.com/google-research/google-research/tree/master/r4r)、[原论文，ACL 2019](https://arxiv.org/abs/1905.12255)

建议报告 SR、CLS（覆盖范围与路径长度共同度量）、nDTW、SDTW，避免只看 SPL。R4R 保留离散图环境，本次未确认一套像 R2R-CE v1-3 一样广泛使用的官方 R4R-CE 移植。因此它适合机制验证，加入主工作量之前应评估维护第二套 simulator 是否划算。不同论文对子集采样与 instruction/path 去重口径也需核对。

#### 4.4.2 REVERIE 导航与目标指代定位

REVERIE 给的是远处对象的指代表达，要求移动到适当位置并从干扰候选中指出目标；不是详细路线跟随。官方数据页报告 86 个建筑、10,318 个全景、4,140 个目标对象、21,702 条指令。[官方数据页](https://yuankaiqi.github.io/REVERIE_Challenge/dataset.html)、[CVPR 2020 论文](https://arxiv.org/abs/1904.10151)

应将导航 SR/SPL 与 remote grounding success（以及其长度加权版）分开。站在目标附近不等于指对目标对象。官方提供对象 proposals、baseline grounding 权重和 val evaluator；数据已有多个版本，2026-03 公告说明 test 改为人工处理提交，不能沿用旧 EvalAI 文案声称自动服务可用。[官方仓库及当前公告](https://github.com/YuankaiQi/REVERIE)、[验证 evaluator](https://github.com/YuankaiQi/REVERIE/blob/master/tasks/REVERIE/eval_release.py)

#### 4.4.3 SOON/FAO 目标描述驱动的长距离搜索

SOON 是 **CVPR 2021** 工作；配套数据名为 FAO。任务描述目标属性、关系和所在/邻近区域，起点可以变化。论文报告 3,848 组指令、3,923 个对象、6,326 个框，并采样超过 30K 条导航轨迹。它有 seen-instruction、seen-house/unseen-instruction、unseen-house 等不同泛化维度；这些划分不能简化成同一个“unseen”。[CVPR 入口](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_SOON_Scenario_Oriented_Object_Navigation_With_Graph-Based_Exploration_CVPR_2021_paper.html)、[论文 §5](https://arxiv.org/pdf/2103.17138)

论文同时评估导航 NE/OSR/SR/SPL 与目标定位 SFPL。官方有训练与 eval 代码，但 README 的数据、完整预训练权重准备说明较稀疏，本次未验证完整 runnable bundle。它适合验证场景语言泛化，不建议和 R2R/RxR/LHPR 同时作为必须完成的第四主基准。[官方 SOON 代码](https://github.com/ZhuFengdaaa/SOON)

#### 4.4.4 GOAT-Bench 多模态连续目标导航

GOAT-Bench 每条 episode 有 5–10 个依次到来的目标，目标模态为类别、语言描述、图像；当前子任务结束或预算耗尽后才收到下一目标，同场景中保留经验。官方描述传感器为 RGB 360×640、depth、相对位姿/GPS+compass；前进 0.25 m、转向/抬低头 30°、STOP；每目标 500 步，STOP 时离目标实例 1 m 欧氏距离以内成功。[GOAT-Bench 官方任务协议](https://github.com/Ram81/goat-bench#robot-goat-bench)

因此未来目标列表不能输入规划器，不允许偷看未来目标做全局旅行商排序。研究点应是如何利用已经经历的目标及观察改善后来目标，而非提前优化整个序列。

原论文使用 145 train/36 validation 的 HM3D-Sem 场景；不同章节对总 episode/category 的统计表述存在不一致，本报告不将摘要中的 680K/312 与正文更细的训练统计混写成严格发布规模。正式使用时应按下载版本清点。公开仓库提供 train/val_seen/val_seen_synonyms/val_unseen、SenseAct-NN checkpoint、goal embeddings 和 eval 配置；“seen/unseen”还包含目标词汇维度，不能只按名字认定它是训练建筑与未见建筑划分。[论文 §4](https://arxiv.org/html/2404.06609v1#S4)、[数据与权重](https://github.com/Ram81/goat-bench#floppy_disk-dataset)

报告每种目标模态及序列位置的 SR/SPL。若增加全序列成功率，明确它是补充指标并保留相同 episode；不要将后几个子任务表现改善直接归因于学习，因为起点距离和已探索空间也在变化。

#### 4.4.5 LangMap 层级语言目标与评测协议差异

LangMap 的 HieraNav 用语言规定 scene/category、room type、特定 region、唯一 instance 四层目标。论文 v3 覆盖 36 个 HM3D-Sem validation 场景，414 个标注类别中 349 用于导航，共 18,479 个任务，其中有 720 个五目标 episode（3,600 个子任务）。目标在同楼层，跨目标不重置位姿。它测试目标理解与搜索，不是路线忠实性。[论文 §3](https://arxiv.org/html/2602.02220v3#S3)

特别需要保留以下差异：论文 §3.1 写“500 步内 STOP，距有效目标 1 m”；公开代码 commit `a37db3b1b06cec80bfefe3e05e6084efddd70698`（2026-10-01）却按**最终位置距 goal viewpoint 的 geodesic <0.25 m** 计算成功，注释明确无论 STOP、early stop 或 step limit 均如此。目标物体与 goal viewpoint 不是同一坐标，两个阈值不能互换。[run.py:194](https://github.com/bo-miao/LangMap/blob/a37db3b1b06cec80bfefe3e05e6084efddd70698/run.py#L194)、[run_seq.py:531](https://github.com/bo-miao/LangMap/blob/a37db3b1b06cec80bfefe3e05e6084efddd70698/run_seq.py#L531)

数据和 PlaNaVid 代码已公开；基础 Uni-NaVid/EVA checkpoint 有下载说明，VLM planner 可本地或 API 使用。官方 eval 脚本默认 `CHECK=1` 只测一个场景，完整评测需 `CHECK=0`。代码按子任务生成当前 instruction 并将上一任务末位姿传入下一任务；不应把 evaluator 中可读取的 `task_sequence` 全部交给策略。[README](https://github.com/bo-miao/LangMap)、[安装及 checkpoint](https://github.com/bo-miao/LangMap/blob/main/INSTALL.md)、[顺序任务执行](https://github.com/bo-miao/LangMap/blob/a37db3b1b06cec80bfefe3e05e6084efddd70698/run_seq.py#L356)

PlaNaVid 本身是 RGB-only、有界多样性记忆与高层规划结合的强基线。因此“加一个大模型和记忆”并不构成新贡献，显式 3D 表征是否值得也应通过相同观察预算的消融证明。LangMap 更适合将来目标导航扩展；本题的路线跟随主实验不必强行加入。

### 4.5 核心测评方案

| 核心实验 | 必须回答的问题 | 主要数据 | 首要指标 | 关键补充 |
|---|---|---|---|---|
| R2R-CE | 在成熟未见环境上，方法是否改善总体执行而非只增加探索？ | v1-3 完整 val_unseen 1,839；最终争取官方 test | SR、SPL | NE/OSR/TL、nDTW、总模型成本 |
| RxR-CE | 在更长语言、路线和历史上是否保持语义/路径一致？ | 明确的英语 guide 或三语言 guide 完整划分 | SR、SDTW | SPL/nDTW、按长度分桶、语言分组 |
| LHPR-VLN | 顺序多目标执行、阶段切换与停止是否可靠？ | 固定公开 revision 的 LH test；ST 单列 | 整任务 SR、ISR | CSR/CGT、官方步数 SPL、2/3/4 阶段分组 |

建议先完成英语路线主线。如果没有可靠的印地语/泰卢固语能力与训练预算，可以公开承认 RxR-CE-English，避免以小语种失败掩盖导航方法分析；但要与同英语设置的基线比较，不能挑三语言论文中的最优数字。若论文目标明确强调多语言泛化，再增加全语言测试。

### 4.6 常用指标及其解释

| 指标 | 计算含义 | 主要反映的能力与限制 |
| --- | --- | --- |
| NE，Navigation Error | 最终位置到目标的导航距离，通常采用 geodesic distance | 距离更近不保证正确 STOP 或忠实执行路线 |
| SR，Success Rate | 满足该协议成功条件的 episode 比例 | 必须说明距离、停止、视域与目标实例要求 |
| OSR，Oracle Success Rate | 路径中曾进入成功区域的 episode 比例 | 与 SR 的差距可帮助分析终止问题，不能视为实际完成 |
| SPL | 每条成功乘以 `L / max(L, P)` 再取平均；L 为起点到目标的最短路长度，P 为实际路长 | 衡量到达效率；非最短路线任务还需评价语言忠实性 |
| nDTW | 对参考轨迹与实际轨迹的动态时间规整距离进行长度及阈值归一化 | 同时考虑位置及访问顺序；受参考轨迹和采样口径影响 |
| SDTW | 成功标志乘以 nDTW 后汇总 | 同时要求完成与路线一致，失败路径不贡献成功得分 |
| TL，Trajectory Length | 实际移动轨迹长度 | 必须与转向、宏动作及执行预算口径一起报告 |
| ISR、CSR、CGT | LHPR 的子任务、相邻阶段条件及目标步数加权指标 | 以第四节核对的具体实现为准，不能代替整任务 SR |

SPL 的定义依据导航评价建议，nDTW/SDTW 依据原论文；LHPR 的官方步数 SPL 是前文列明的实现差异。[SPL 评价建议](https://arxiv.org/abs/1807.06757)、[nDTW/SDTW](https://arxiv.org/abs/1907.05446)

## 五 多模态大模型导航方法

本节聚焦从图像或视频预测动作的模型，以及近期围绕记忆、推理与执行效率的扩展。冻结语言模型与地图规划方法在第六节合并比较。训练规模与运行速度均为作者报告，不是本次实测。

### 5.1 NaVid 视频历史到可执行动作

- **出处**：2024-02-24 首发，RSS 2024；[论文](https://arxiv.org/abs/2402.15852)、[方法全文](https://arxiv.org/html/2402.15852v1)、[官方评测仓库](https://github.com/jzhzhang/NaVid-VLN-CE)。
- **模型**：EVA-CLIP、指令查询视觉 token、Vicuna-7B/LLaMA-VID。每帧保留一个指令相关 token；指令无关部分当前帧用 64 个 token、历史每帧用 4 个。因此它并非简单把所有帧等分辨率塞给聊天模型。
- **动作与监督**：输出前进/转向/停止及距离、角度等语言形式参数，再解析执行。论文使用约 550K 导航样本与 665K 网络数据，混合动作规划和指令推理，包含偏离专家轨迹后的 DAgger 数据。
- **成本与实机**：原论文报告 24 张 A100、约 28 小时（672 GPU 小时）；远端推理约 1.2–1.5 秒/动作，Turtlebot4 实机实验。该延迟不是高频电机控制周期。
- **重要版本边界**：当前官方仓库的下载权重按 Uni-NaVid 的训练策略使用 R2R 和 RxR 训练集，分数与 RSS 原始版本不同。原论文的 R2R→RxR 零样本迁移结论，不能直接套在这个新版 checkpoint 上。仓库还说明开源实现存在因许可而重写的部分。
- **本题价值**：清楚展示 VLM 泛化如何通过导航动作监督落地；适合作为旧基线，研究主干优先选较新的开放实现。

### 5.2 Uni-NaVid 近远历史分层压缩与多任务训练

- **出处**：2024-12 首发，RSS 2025；[论文](https://arxiv.org/abs/2412.06224)、[全文](https://arxiv.org/html/2412.06224v1)、[训练仓库](https://github.com/jzhzhang/Uni-NaVid)、[项目页](https://pku-epic.github.io/Uni-NaVid/)。
- **记忆**：当前帧 64 token、短期每帧 4 token、长期每帧 1 token；短期缓冲长度 64，再在线合并时空冗余。这是按时间尺度分层的压缩，不是显式可检索的房间—地标—指令进度记忆。
- **训练与动作**：约 3.6M 导航样本联合 VLN、ObjectNav、EQA、跟随任务，EVA-CLIP+Vicuna-7B，一次生成未来 4 个动作。论文报告 40 张 H800、35 小时（1,400 GPU 小时）。
- **效率口径**：A100 上约 0.2 秒生成 4 个动作，约 5 Hz 是模型推理更新频率。附录另计约 0.3 秒互联网通信，机器人通过本地控制器执行，并异步上传图像；不能由“5 Hz”推断观测到控制全链路 200 ms。
- **发布边界**：训练代码、VLN-CE 评测和权重公开；官方明确只发布部分训练数据，且因许可未沿用 L3MVN 的 ObjectNav 模仿来源，可能影响 ObjectNav 结果。不能宣称完整 3.6M 数据管线已经无条件一键复现。
- **本题价值**：多任务共训、防遗忘和历史压缩的扎实对照；“近期精细、远期压缩”本身已经不是空白创新。

### 5.3 NaVILA 语义导航与腿式机器人执行解耦

- **出处**：2024-12-05 首发，RSS 2025；[论文](https://arxiv.org/abs/2412.04453)、[方法全文](https://arxiv.org/html/2412.04453v1)、[官方代码](https://github.com/AnjieCheng/NaVILA)、[项目页](https://navila-bot.github.io/)。
- **高层模型**：VILA-8B 接收当前图像与均匀抽样历史，保留首帧，利用文字提示区分当前与历史；实际部署使用 8 帧记忆配置。输出诸如前进 75 cm、右转 30°的中层语言动作。
- **学习方式**：用 R2R/RxR、EnvDrop、ScanQA、通用 VQA，以及从约 2K 人类游览视频形成的约 20K 轨迹做 SFT。底层视觉 locomotion policy 另用 PPO 在 IsaacLab 训练，将中层命令变成关节控制。
- **关键传感边界**：高层 VLN 预测可以是单视角 RGB；真实 Go2 底层避障/步态使用 LiDAR 高度图和本体感知。把整套机器人系统写成“仅 RGB 无额外传感”是不准确的。
- **效率与实机**：论文报告 VLA 推理约 0.6 秒，包含图像传输时动作间等待约 1 秒；这与低层运动策略频率不同。报告真实机器人 25 条指令实验，不能外推成任意建筑中的可靠自主系统。
- **发布边界**：高层训练/评测/权重已发布，底层与 Isaac 评测有独立仓库；人类视频仅发布 IDs 与标注，原始视频受版权限制，EnvDrop 还需自行渲染。老 Habitat 依赖需要适配。
- **本题价值**：很适合借鉴“研究语义决策，复用执行器”的边界，不必把毕业论文扩展为步态控制研究。

### 5.4 StreamVLN 流式上下文与计算复用

- **出处**：2025-07-07 首发，ICRA 2026，2026-07-09 发布 v2；[论文](https://arxiv.org/abs/2507.05240)、[v2 全文](https://arxiv.org/html/2507.05240v2)、[官方代码](https://github.com/InternRobotics/StreamVLN)。
- **模型与上下文**：LLaVA-Video-7B（Qwen2-7B 语言骨干），把观测—动作组织成多轮对话；快通道保留滑动窗口 KV cache，慢通道保留抽样历史。论文通常用 8 轮活动窗口、每轮 4 个动作，历史再固定数量采样，而非每步重新 prefill 全历史。
- **3D 压缩边界**：可选体素剪枝把图像 patch 借助深度投影到共同 3D 坐标，重复体素保留较新 token。使用此组件时必须披露 depth/pose 来源；RGB 主结果与几何辅助剪枝不能混为“同样输入”。v2 明确保留了这一点。
- **训练**：先 oracle 轨迹 SFT，再收集 DAgger，混合导航与通用多模态数据继续训练；包含 R2R、RxR、EnvDrop、约 150K ScaleVLN 子集，以及 VideoQA/ScanQA/MMC4 等共训来源。论文完整训练约 1,500 A100 GPU 小时。
- **效率**：Go2+远端 RTX 4090 报告约 0.27 秒/4 动作，室内通信约 0.2 秒；这是推理与通信分项，不是闭环安全反应时间保证。
- **当前发布**：独立仓库有训练、DAgger、共训、实机部署、HF 权重与数据说明。2025-09 更新为 R2R_VLNCE_v1-3 checkpoint；README 还记录过 `num_history` 未正确传递的修复，需锁定 commit、数据与权重版本。InternNav 的集成列表写 coming soon，不表示 StreamVLN 独立仓库未发布。
- **本题价值**：最适合研究有限视觉上下文下的长程导航。改进要回答“保留哪些与未完成指令有关的证据、忘记是否造成特定错误”，而不是只再加一个窗口或 cache。

### 5.5 InternVLA-N1 / DualVLN 慢语义目标与快轨迹的开放主干

- **出处**：InternVLA-N1 技术报告为 2025 年；DualVLN 论文 2025-12-09 首发，ICLR 2026，正式题名 *Ground Slow, Move Fast*。[DualVLN 论文](https://arxiv.org/abs/2512.08186)、[全文](https://arxiv.org/html/2512.08186v1)、[InternNav](https://github.com/InternRobotics/InternNav)、[训练文档](https://internrobotics.github.io/user_guide/internnav/tutorials/training.html)。
- **必须区分版本**：InternVLA-N1 是模型家族/工程框架；当前列出 System2+外接控制器、与 NavDP* 联调的 RGB-D 双系统、RGB DualVLN。不能把这些 checkpoint 的结果混用。
- **动作接口**：Qwen2.5-VL-7B 的 System2 输出图像中的像素目标，或主动转头/上下看/停止；追加少量可学习 query 提取 latent goal。System1 将目标潜变量与新旧 RGB 编码融合，经小型 DiT/flow matching 生成 32 点连续轨迹。
- **训练边界**：先全量调优 System2 的目标 grounding；再冻结 System2，训练 latent query 与轨迹模块。像素目标监督来自 3D 专家轨迹投影、深度遮挡过滤；因此“推理 RGB-only”不等于训练没有几何特权监督。
- **速度口径**：示意图慢系统约 2 Hz、快系统约 30 Hz；实机报告 RTX 4090、约 20GB 显存，System2 有 cache 时约 0.7 秒，System1 TensorRT 约 0.03 秒。应采用具体部署测量，不把示意频率相乘或宣称整个大模型 30 Hz。
- **实机证据**：Turtlebot4、Go2、G1 的作者实验，远端系统输出由 odometry 转世界坐标、MPC 跟踪；算法 RGB 输入与整机仍需里程计/执行器是两个层次。
- **发布边界**：InternNav 已有训练文档、不同权重、VLN-CE/VLN-PE 评测和实机部署代码。适合在公开权重上改长期目标状态/记忆与恢复，同时固定低层执行器；若重训全部模块，数据和计算成本明显提高。

### 5.6 LiveVLN 异步动作续接与执行等待

- **出处**：2026-04-21 预印本；官方仓库自标 RA-L 2026，arXiv 元数据本次仅确认预印本。[论文](https://arxiv.org/abs/2604.19536)、[全文](https://arxiv.org/html/2604.19536v1)、[官方代码](https://github.com/NIneeeeeem/LiveVLN)。
- **性质**：training-free runtime，不重新训练导航模型。将动作续列拆成已执行部分、已承诺的 guard buffer、可改写的 tail；执行线程使用 guard，推理线程用最新观测刷新后续动作。
- **自适应逻辑**：以最近端到端 sense/inference 延迟的平滑估计加安全余量，选择执行时间足够覆盖它的最短动作前缀。刷新超时且无有效备份时停止；不等于无限制继续旧动作。
- **兼容条件**：backbone 要接受最新观测和已承诺的 action prefix，能输出有序多步动作。不是任意只有单步黑盒 API 都可以原样接入。
- **证据**：同权重对比 StreamVLN 与 NaVIDA，Unitree G1+远端 RTX 5090。平均等待时间分别减少 77.7% / 72.8%，整段任务墙钟时间减少 12.6% / 19.6%；后者远小于等待降低，说明物理执行占主要时间。实机表分别报告 40 次任务，成功定义为停在目标 2 m 内。
- **发布**：代码、真实机器人 service/client 入口已公开，复用原导航 checkpoint。它主要处理连续性，没有证明任意动态变化下执行旧前缀都安全。
- **本题价值**：若研究效率，至少要比较它；更有价值的问题是“等待减少后是否因为语义状态过时而损伤长程正确性”。

### 5.7 SPARK-VLN 推理期间的隐藏状态流与动态室内场景

- **出处**：2026-07-18 预印本，作者页标 Under Review；[论文](https://arxiv.org/abs/2607.16806)、[全文](https://arxiv.org/html/2607.16806v1)、[实际项目页](https://hutslib.github.io/SPARK-VLN.dc.html)。
- **贡献**：VILA-8B 逐 token 提取第 16–23 层隐藏状态，经 8 个 latent slots 的跨注意力桥接，持续更新 fast planner；快模块用 5 步 Euler 的 rectified flow 生成轨迹，再用安全 critic 筛选。
- **不同于 LiveVLN**：LiveVLN 重排完整动作块的执行时间；SPARK 在慢模型尚未完成生成时就给快模型新的中间表征。它需要读取本地模型隐藏状态并训练桥接/策略，普通 token 文本流 API 不提供这个条件。
- **任务与数据边界**：快速规划器先学 PointNav，再增加语言潜变量条件；VLN 评测不提供目标坐标。主要新证据是含行人的室内动态仿真，对照环境在推理时暂停/继续运行。论文把更长、多子目标指令列为后续方向，不能因标题就认为已解决超长程导航。
- **效率口径**：单 RTX 4090、batch=1 下，stream 模式有效更新延迟 0.185 秒，等待生成结束是 0.788 秒；并报告推理期间行人位移。这里测的是可用指导更新速度，不是整个 episode 时间或实机网络闭环。
- **发布边界**：项目页的 Code 按钮目前链接为 `#`；本次没有核验可下载训练代码、权重或动态 benchmark 资产。论文实验主要为仿真，不能写成已经部署真实人群环境。
- **本题价值**：它直接覆盖“慢快模型+中间 latent streaming+动态时延”方向；若将效率作为贡献，需要拿出更明确的指令记忆、过时目标检测或恢复问题，而不是重复双系统结构。

### 5.8 VerNav 判别式动作选择与选择性生成

- **出处**：2026-09-01 预印本；[论文](https://arxiv.org/abs/2609.00920)、[全文](https://arxiv.org/html/2609.00920v1)。
- **模型与训练**：文本 Qwen2.5-3B-Instruct，对候选动作批量问“是否合适”，读 Yes 的首 token 分数，按候选分布熵决定是否调用生成器补充状态证据。用 LoRA r=16/alpha=32，先 VPO chosen/rejected 排序，再用 step-level RFT；作者在一张 H100 80GB 上训练。
- **监督信号**：局部奖励基于离目标更近/发现指令地标，另罚重复视点；这些几何与成功信号用于训练，不给推理时的 verifier。训练确实是关键，raw verifier 的 SR 很低，不能把结果归功于一次简单 prompt 变化。
- **协议边界**：主实验为离散 R2R 图导航，模型消费文本化描述与候选，不是 RGB 视频 VLA。作者报告 80 ms 仅涵盖“提交候选 verification prompts 到取出 Yes logits”；加自适应证据生成约 294 ms。视觉描述产生、环境转移、通信与控制不在同一测量范围。
- **发布与实机**：本次未核验官方代码/权重，也未见论文的真实机器人闭环证据。仅可作为近期判别式决策的近邻，不能直接加入 CE 复现清单并承诺现成可用。
- **本题价值**：提示可以训练轻量进度/动作验证器，减少不必要生成；但候选动作可获得性、描述错误与熵是否校准都要单独验证。

### 5.9 VLingNav 自适应推理与视觉语言记忆

- **出处**：2026-01-13 预印本；[论文](https://arxiv.org/abs/2601.08665)、[全文](https://arxiv.org/html/2601.08665v1)、[项目页](https://shaoanwang.github.io/VLingNav-web/)。
- **核心**：预测 think_on/think_off 决定是否显式推理；VLingMem 组合精简语言总结与关键视觉特征，动作 head 利用最后隐藏状态预测未来轨迹。论文自己将它归为单系统，不能等同 DualVLN 的两个并行网络。
- **训练**：Nav-AdaCoT-2.9M，Qwen2.5-VL-72B 生成推理标注；三阶段含开放视频预训练、导航共训、专家辅助在线后训练。PPO 类目标与 SFT 混合，失败/卡住时专家恢复数据进入 buffer；报告使用 128 张 A100。
- **任务边界**：主定量基准为 ObjectNav、开放词汇 ObjectNav、ImageNav 和 Tracking，并非 R2R/RxR 指令路线跟随。目标搜索记忆与路线指令进度有交集，但结果不直接可比。
- **效率与实机**：论文报告远端 RTX 4090，500 帧范围内推理低于 300 ms，加约 100 ms 通信约 2.5 FPS；有四足实机案例。FPS 不证明长期记忆事实正确，更不是任意长历史的无损保留。
- **发布边界**：项目页有论文与视频，GitHub 是网站源码；本次没有核验模型 checkpoint、2.9M 数据和完整训练程序。适合论证方法新颖性边界，不宜让主实验依赖它未来开源。

### 5.10 Qwen-RobotNav 可配置观测上下文

- **出处**：2026-06-16 技术报告；[论文](https://arxiv.org/abs/2606.18112)、[全文](https://arxiv.org/html/2606.18112v1)、[官方仓库](https://github.com/QwenLM/Qwen-RobotNav)。
- **模型与动作**：Qwen3-VL 骨干+轻量 4 层 MLP action head，输出 8 个 `(x,y,theta)` waypoint。上层 planner 调同一模型的不同 task mode；本文只关注其中室内 VLN/搜索部分。
- **上下文接口**：调用时配置视觉 token 预算、时间衰减、相机权重、随机/最近帧采样；训练时随机化这些参数。即“任务需要什么历史”成为可控模型接口，不只是固定 prompt 文案。
- **训练**：15.6M 多任务样本、轨迹监督与 vision-language 共训，报告 2B→8B 扩展和室内 R2R/RxR 结果。如此规模的训练可作为趋势证据，不是本课题必须复制的工作量目标。
- **泛化证据与边界**：有作者真实机器人演示/结果，但不能把多域演示当本项目跨建筑、超长指令的统一评测；也不能从参数小、waypoint head 简单就推断本地时延。
- **最重要发布事实**：官方 README 明确表示目前没有发布 Qwen-RobotNav 权重的计划，仓库主要是报告资源。因此它是必须讨论的近邻方法，不是可立即训练/消融的开放 baseline，也不应写成“选 Qwen-RobotNav 部署即可”。

### 5.11 近期进展对研究问题的影响

2026-04-03 至 2026-10-03 内，至少有四个直接影响本课题定位的进展：

1. **LiveVLN（4 月）**把执行连续性做成可独立接在 checkpoint 上的 runtime，并报告机器人端等待、暂停与总耗时。再做“双线程异步执行”需要明显超出这一基线。
2. **Qwen-RobotNav（6 月）**把 token budget、历史衰减和采样策略参数化，并让训练覆盖不同观测配置。“动态调上下文”这一宽泛描述已被直接覆盖。
3. **SPARK-VLN（7 月）**把慢快接口推进到生成期间的隐藏状态流，使用推理不暂停的动态仿真。只是调用大模型加一个快控制器，已经不足以成为结构创新。
4. **VerNav（9 月）**表明动作决策可以以判别验证为主、只在必要时生成，且通过偏好训练和闭环 RFT 对齐。它的限定是图导航与决策阶段测时，不代表视觉 CE 已完整解决。

此外，StreamVLN 在 2026-07 更新论文版本，说明不能只读最初摘要：其几何剪枝仍需要深度，开放实现的版本/训练集仍决定可比性。上述作品共同说明：**实时、记忆、推理触发都已存在直接近邻；本题的贡献应落在它们究竟改善了哪一种长程导航失败，以及为何能迁移到新建筑和新表达。**

### 5.12 公开资源与基线选择

| 级别 | 方法 | 当前适合的用途 |
|---|---|---|
| 可作为主要实现候选 | StreamVLN、InternNav/DualVLN | 有开放权重、训练/评测入口；适合锁版本后做完整消融 |
| 可作为历史/结构对照 | NaVid、Uni-NaVid、NaVILA、NavGPT-2 | 有代码与权重，但训练数据发布程度、checkpoint更新和模拟器版本需核对 |
| 可作为运行时对照 | LiveVLN | 已有代码，基于相同预训练权重比较等待与正确性 |
| 可作为冻结 Agent 对照 | NavGPT 或按同一 CE action/sensor contract 重建的冻结 VLM Agent | 不应直接搬离散图分数；比较时控制候选动作与额外工具 |
| 用于创新空间审查，不作为立即可复现承诺 | SPARK-VLN、VerNav、VLingNav | 本次未核验完整模型/训练/评测资产，SPARK项目Code还是占位链接 |
| 明确不作为开放训练底座 | Qwen-RobotNav | 官方无发布权重计划 |

**建议主线**：先用 StreamVLN 的指定公开 checkpoint 建立 RGB VLN-CE 基线；再在同一观测与执行协议下接入 DualVLN 作为较强异构验证。若研究内容与动作形式耦合很深，先完成一个骨干的全套证据，比同时移植大量模型更有价值。若重点是规划、记忆和恢复，低层控制器应固定，不以更换执行器获得的提升代替高层方法效果。

## 六 任务进度与错误恢复方法的深入比较

### 6.1 早期进度估计与回退

#### 6.1.1 Self-Monitoring 进度监督与规则回退

输入为指令、当前全景候选方向特征与循环历史状态。视觉—文本共同注意力定位当前相关指令；progress monitor 结合视觉、隐状态和文本注意力预测完成程度。其监督来自归一化目标距离，起点约为 0，接近目标为 1；当当前位置到目标的距离大于初始位置到目标的距离时，可为负。训练联合动作交叉熵和进度 MSE。推理既研究 beam 重排序，也明确给出 progress inference：新一步进度下降时返回前一 viewpoint，再尝试次优方向。评测为离散 R2R 的 NE/SR/OSR/SPL。

因此“观察进度下降→真实路径回退→重新选路”早有先例；其边界是标量目标距离并不等于精确的指令阶段判断，且离散图回退不涉及完整连续控制。正文证据：[§2.2–2.4、§3](https://arxiv.org/html/1901.03035v1)；[官方实现的 progress inference 说明](https://github.com/chihyaoma/selfmonitoring-agent)。

#### 6.1.2 Regretful Agent 学习何时回退以及回退后去哪

它把前后两步进度差输入 Regret Module，学习融合“继续前进”和“返回上一位置”的表示；Progress Marker 记录已访问方向及其进度，辅助回退后的方向选择。动作交叉熵、进度 MSE 外加入熵正则，让训练策略有机会走错并学习恢复；推理使用贪心动作选择。其核心是导航图上的局部搜索与回退，而非仅增加解释文本。

对拟议方法的压力是：回退触发、历史失败分支记忆、避免重新选择错误方向，都已有端到端方案。不能将这些组合换成大模型便称新。正文证据：[§4.1–4.3 与组件消融](https://arxiv.org/html/1903.01602v1)；[CVPR 官方论文页](https://openaccess.thecvf.com/content_CVPR_2019/html/Ma_The_Regretful_Agent_Heuristic-Aided_Navigation_Through_Progress_Estimation_CVPR_2019_paper.html)。

### 6.2 大模型推理与地图规划

#### 6.2.1 NavGPT 零样本推理控制闭环的早期代表

NavGPT 将全景图像转为描述，并结合历史观察、动作、推理和可行方向组成提示，GPT 产生下一步思考与动作。导航任务上不进行专门参数训练。它已讨论子目标分解、进度跟踪与异常后的计划调整；局限主要是图像转文字及历史摘要的信息损失。环境仍是预定义导航图。

它是“生成解释—接收新观察—继续决策”的基线，不能因为没有专门 recovery loss 就断言无法恢复；但定性推理示例也不能证明可靠的错误归因或进度纠正。应通过干预实验检验解释内容是否真正改变动作。正文证据：[§3 与失败案例附录](https://arxiv.org/html/2305.16986v2)；[AAAI 2024 正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/28597)。

#### 6.2.2 MapGPT 全局拓扑上下文和可修订的多步计划

输入包括原始指令、可导航方向的图像或文字、动作历史，以及在线拓扑图。图中记录已经探索、当前可达和暂不可达的已知节点；节点连接关系由模拟器提供。模型生成并随观察调整多步计划，能够返回特定节点重新探索。方法以提示为主，不训练新的导航网络，评测离散 R2R 和 REVERIE。

它直接覆盖“维护地图、发现当前方向不符、回到之前岔路重规划”。与连续 RGB-only 模型比较时，不能忽略候选节点和连通关系这一输入差别。官方实现的抽样快捷评测也不能替代全量主实验。正文证据：[§3.1–3.3](https://arxiv.org/html/2401.07314v3)；[ACL 正式论文](https://aclanthology.org/2024.acl-long.529.pdf)；[代码与抽样协议](https://github.com/chen-judge/MapGPT/)。

#### 6.2.3 NavGPT-2 大模型表征与专门策略网络分工

Q-former 将多视图与指令对齐到冻结 LLM，隐藏表示再交给图导航策略。先用 GPT-4V 生成的约 10K 单步推理样本训练视觉桥接部分，再冻结 VLM，使用 BC＋DAgger 训练导航策略。在线图保留已访问节点和相邻未探索节点，策略可选择全图目标，正文明确包含走错后的 back-tracing。

它的意义是分离“通用多模态表征”和“导航动作学习”；可生成解释不等于每步必须先输出长推理。它也削弱了“引入 LLM 后才能全局回退”的论点。正文证据：[§3.1–3.3](https://arxiv.org/html/2407.12366v2)；[ECCV 2024 正式论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01143.pdf)。

#### 6.2.4 ETPNav 连续环境已有物理脱困与全局纠偏

ETPNav 通过预测 waypoint 构建在线拓扑图，跨模态规划器选全局目标，再由低层控制器执行。Tryout 检查前进后位置是否改变；卡住时尝试若干转向并前进一步，脱困后恢复原朝向继续。正文还有全局规划避免两个位置来回振荡、纠正错误路线的案例。它评测 R2R-CE/RxR-CE，并专门分析是否允许沿障碍滑行的差别。

所以“物理恢复”不是新领域；真正需要证明的是语言阶段误判与控制恢复之间的联系，而非再加一个卡住检测器。它需要几何／深度等条件，不能当作与单目模型完全同输入的基线。正文证据：[控制器及 §IV-C 消融](https://arxiv.org/html/2304.03047v2)；[TPAMI 2024 官方代码](https://github.com/MarSaKi/ETPNav)。

### 6.3 语义进度与主动感知

#### 6.3.1 Progress-Think 语义进度与策略联合学习

当前 RGB 和采样视觉历史输入进度模块，输出已完成的指令式文本，供动作模块使用。训练分为前缀软对齐与单调损失预训练、固定进度模块的动作监督、GRPO 式联合优化；后者奖励动作前缀正确性、格式和进度文本长度。策略预测三步动作，在连续环境执行，主评测为 R2R-CE/RxR-CE。

重要边界：单调性是训练正则，不能写成推理阶段硬性禁止修正。长度奖励只限制文本不过长，也不直接证明完成判断语义正确。将其错误前缀置于反证观察后，是否能重新推断，是待测问题。正文证据：[§3、§4.3、§5.4](https://arxiv.org/html/2511.17097v2)；[CVPR 2026 正式记录](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Progress-Think_Semantic_Progress_Reasoning_for_Vision-Language_Navigation_CVPR_2026_paper.html)。

#### 6.3.2 SeekVLN 主动观察与进度依据获取

策略先选 NAV 或 SEEK；后者获取左、前、右视图，再生成完成任务、下一任务和关键证据字段后行动。FRG 利用未来专家动作生成取证监督并 SFT；C2PO 从相同模拟状态比较 SEEK/NAV 分支的短期进展，结合终局结果做 PPO。它是“证据不足→补充观察→更新语义进度→行动”的直接最近邻。

测试使用 R2R-CE/RxR-CE；FRG/C2PO 组件消融在 val-unseen 的 613 条去重轨迹上评测，触发策略消融使用另一个 100-episode 子集。这些是消融评测规模，不是训练数据量或完整主评测分母。模拟 SEEK 的观察资源与实体转头成本应另计；真人实验的 stop 是模型声明，非独立成功证明。不能据正文未专门测试错误前缀撤销，就声称它做不到恢复。正文证据：[§3、§4.1–4.3、附录 D/E](https://arxiv.org/html/2609.37353v1)。

#### 6.3.3 FantasyVLN 多模态推理训练与隐式动作预测

在 LH-VLN 上混合直接动作、文本 CoT、紧凑视觉 CoT 和多模态 CoT 训练；视觉未来观察以冻结 VAR 的低尺度潜在表示编码，并对齐不同模式的动作决策。推理走直接动作分支，以保留训练收益并减少生成开销。它的重点是推理表征与效率，没有据此建立独立“已纠正错误状态”的因果证明。

它使“加长解释就更可靠”的假设需要重新检验。LH-VLN 的整任务 SR、子任务 ISR、条件 CSR/CGT 不等于 R2R 单目标 SR。官方训练／评估代码公开，但 README 明确文本 CoT 标注未公开，需重新生成。正文证据：[§3–4](https://arxiv.org/html/2601.13976v2)；[CVPR 正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zuo_FantasyVLN_Unified_Multimodal_Chain-of-Thought_Reasoning_for_Vision-and-Language_Navigation_CVPR_2026_paper.pdf)；[代码](https://github.com/Fantasy-AMAP/fantasy-vln)。

### 6.4 错误轨迹与纠正学习

#### 6.4.1 CorrectNav 自纠正数据与迭代训练

模型在训练集上滚动，按与参考路线的距离定位偏差点；从该错误位置规划经过后续参考点的纠正轨迹，用前缀提供历史，只对修正动作学习。另取偏差前、中、后三帧，利用大模型生成地标描述和空间 VQA，补足纠正感知监督。反复收集新错误并后训练形成 flywheel。基础是 RGB 视频 VLA，部署不额外生成整段纠正解释。

这与“让模型知道为何走错并学习返回正确路线”高度重合。它使用训练时参考轨迹不等于部署泄露目标；应公平保留这种训练权限。拟议方法若只多一个结构化进度字段，须证明超过该闭环，而非直接宣称新。正文证据：[Self-correction Flywheel，Steps 1–4](https://arxiv.org/html/2508.10416v1)；[AAAI 2026 正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/38942)。

#### 6.4.2 BudVLN 历史重对齐与纠正监督

它指出离轨位置的返回动作可能与原指令“直行”等表面语义冲突。训练先用 greedy probe；已掌握样本用 GRPO 改善效率，困难样本走 SFT。困难轨迹按离轨、停滞、错误停止等条件定位失败，回到最远有效参考点对应的历史状态，保留有效历史、丢弃错误后缀，reset 模拟器并用 oracle 合成向前示范。

其“retrospective rollback”发生在训练数据构造中，不证明机器人在线物理返回；但“有效前缀＋错误后缀＋重新锚定”已被明确研究。恢复监督如何同时表示任务方向和纠正意图，必须与它比较。正文证据：[§3.2–3.5，尤其 History-Aware Rollback](https://arxiv.org/html/2602.06356v1)。

#### 6.4.3 SACA 逐步审计、偏差点与失败样本信用分配

PGSA 将指令拆为地标，结合图文相似度、检测与分割产生软进度分数和硬边界。不同结果的 rollout 组分别进行修复重采样或全失败救援；强化有效前缀，在偏差点进行对比纠正。局部教师动作可使用训练时模拟器最短路径信号。评测为连续 VLN。

它同样不是运行时“文字进度状态数据库”，但已覆盖“错误不应否定整段轨迹、应该找首次偏差并局部学习”。其地标匹配边界对遮挡、重复实例及关系指令的稳定性，是可检测的限制，不能未经实验断言失效。正文证据：[§3 与附录算法](https://arxiv.org/html/2603.09740v1)。

### 6.5 在线偏差检测与恢复

#### 6.5.1 Navigation Heads／HiVLA 内部信号检测与低层恢复

从冻结 NaVILA 的注意力头中选择对指令—历史对齐和偏航敏感的少数头，监测相对熵变化；正常时维护位姿、历史等检查点，异常时切换 PPO 训练的低层策略返回。物理系统还使用 LiDAR costmap、定位和 ROS 2。

“training-free”指检测器不新增可训练参数，不是整套系统无需数据：头选择和阈值在训练集校准，恢复控制器另外训练。它已直接覆盖检测到恢复链；比较时要区分 episode detection 与 step F1、几何偏航标签与语义误判，亦不能给纯 RGB 方案免费加入其定位权限。正文证据：[§3.2–3.6、§4](https://arxiv.org/html/2603.13782v1)；[作者项目](https://jaehwan-j.github.io/projects/hivla/hivla.html)。

#### 6.5.2 AVERT-VLN 独立语义监控、弃权、人工恢复和离线 DPO

独立 VLM Monitor 根据指令、视觉历史与当前图给 Pass/Lost；数据定义明确区分几何离轨与语义违令。异步 Lost 被接受后停止尚未执行动作、清除相关待决请求，从当前位置接受人类像素目标或转向纠正。离线回溯最近 Pass 边界，在同一上下文构造偏好对做 DPO。

它已覆盖“语义一致性监控＋可信历史边界＋纠正学习”，且提供 LostAware：1095 对 Pass/Lost、2190 状态。在线主成绩含人类恢复，不能混入完全自主排名；训练数据是否可下载仍需独立核验。正文证据：[§III-B–E、§IV-A 与主表注释](https://arxiv.org/html/2609.39579v1)。

#### 6.5.3 Talk2Escape 对话反馈与导航恢复

循环或严重偏航触发求助，MLLM 把场景转为问题，oracle／人类给方向提示再注入导航上下文。模拟 oracle 使用目标相对方位与距离；实验还包括物理 Go2。论文的 R2R-CE 和 RxR-CE 主比较为 100/260 episode 子集。

它证明求助可能挽救运行，但不直接证明自主状态推理优于先前工作。目标位置提示、触发信息和人工指点改变了输入协议。论文把许多单轮指令策略统称“open-loop”也不宜照搬：每步接收视觉反馈的导航，本来就有控制闭环，缺少对话不能等同完全开环。正文证据：[§IV-B 与 §V](https://arxiv.org/html/2609.28296v1)。

### 6.6 主要方法的能力覆盖与研究边界

| 要研究的环节 | 已有直接研究 | 拟议工作需要回答的问题 |
|---|---|---|
| 知道自己走到指令哪一步 | Progress-Think、SeekVLN | 不能只增加“已完成／待完成”字段；要证明错误情况下校准与修正更好 |
| 发现执行与指令冲突 | AVERT-VLN、SACA、Navigation Heads | 说明检测对象、可用传感器、假阳性和反馈延迟；不能只报告最终成功率 |
| 找到错误开始的位置 | CorrectNav、BudVLN、SACA | 证明比几何距离／有效前缀分割更准确，尤其区分合法替代路线 |
| 看不清时主动再观察 | SeekVLN | 同等视图与动作预算下比较；泛泛的主动感知不是贡献 |
| 返回历史位置并换分支 | Regretful、MapGPT、NavGPT-2、ETPNav | 证明怎样选择返回点、如何避免循环，以及真实执行成本 |
| 修复失败数据并持续训练 | CorrectNav、BudVLN、SACA、AVERT-VLN | 不能仅把 SFT 换成 DPO/GRPO；需有新的监督信号或解决明确失败机制 |
| 降低长推理成本 | FantasyVLN | 比较隐式策略，不能默认更多文本推理更好 |

“这些论文未单列错误前缀撤销指标”只能说明**论文证据尚未覆盖我们提出的专门评估**，不构成“方法不支持”的结论。相反，需要把这些方法接入统一实验，让它们有公平使用历史、重新观察和回退的机会。

## 七 训练数据与学习方法

本文中，SFT 指监督微调，BC 指行为克隆，二者在导航中通常用专家观察—动作样本训练策略；DAgger 强调收集学习者实际访问状态上的专家监督。RL 指根据回报优化策略，PPO、GRPO 是文献采用的不同优化方法；DPO 使用优选与劣选回答或动作的偏好对。LoRA、adapter 描述参数更新方式，本身不规定导航监督或奖励。这些概念分属数据、目标与参数化三个层次，组合时应分别说明。

### 7.1 专家模仿能够建立能力但不能覆盖全部错误状态

导航训练通常首先使用专家轨迹学习观察、指令到动作的对应。专家示范中的观察是正确动作带来的观察；模型部署时一旦走错，就可能进入训练数据很少覆盖的状态。长任务因此不仅需要记得更久，也需要处理自身行为改变输入分布的问题。DAgger 的基本思想是在学习者访问的状态上补充专家监督，而非只重复学习理想轨迹。[DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)

在 VLN 中，简单把偏航位置连接回目标未必产生正确的语言监督。例如，原指令要求“经过第二个门后左转”，从错误房间直接走最短路径可能到达终点，却没有遵循原路线。训练目标必须说明是在学习目标到达、路线跟随，还是偏差后的任务恢复。几何正确与指令一致应分别检查。

### 7.2 三类数据增强解决不同问题

| 数据增强类型 | 构造方式 | 主要用途 | 需要控制的问题 |
| --- | --- | --- | --- |
| 语言增强 | 同一路线生成多种描述或改写 | 提高语言表达泛化 | 改写不能改变目标、顺序、指代和可见地标 |
| 环境与轨迹增强 | 新布局、外观变化、新路线 | 扩大视觉及空间分布 | 新轨迹必须可执行，训练和测试场景隔离 |
| 错误与恢复增强 | 收集策略自然失败，补充纠正或后续示范 | 学习处理自身错误状态 | 不能把不符合原指令的恢复动作当无条件正例 |

这三类增强需要独立统计规模与贡献。使用更强教师模型批量生成 CoT 或动作，不等于标签天然可靠；应抽样核验目标、动作方向、历史引用及进度判断，并保留自动生成比例。若标注利用未来专家动作或真值路径，这可以属于训练监督，但测试输入不得保留这些信息。

### 7.3 推理监督与动作监督

显式推理监督可以要求模型预测子任务、关键地标或任务进度，再输出动作；隐式推理方法则将这些能力通过训练融入模型，在测试时直接产生动作。两种方法的比较需要同时记录训练监督量、教师模型、测试输出长度及推理成本。

一个合理的研究问题不是“输出更多推理是否更好”，而是哪些中间变量对后续动作具有预测价值。可以分别测试只有动作监督、动作加进度监督、动作加观察依据监督，以及完整联合训练。若没有中间变量正确性的独立检验，流畅的推理文本仍可能只是对错误动作的事后解释。

### 7.4 强化学习中的奖励与信用分配

终点成功奖励容易计算，但很难说明长序列中的哪一步有效。距离变化、路径忠实度、阶段完成、观察收益和动作成本可以提供更细反馈，但各自存在偏差。距离奖励可能鼓励抄近路；进度奖励可能鼓励过早宣布完成；过强的动作惩罚可能使模型过早停止。

因此，奖励应与任务协议一致，且保留官方指标作为主要评价。对于某一机制，例如主动观察，比较同一状态下采用与不采用该机制的后续行为，是理解它是否真正有效的一种研究设计；训练中的反事实分支或模拟器状态克隆不能被计为测试时免费观察。报告训练环境交互量与推理阶段成本也同样必要。

### 7.5 泛化应分开评价

未见场景、未见表达、较长任务、不同传感器和真实机器人迁移是不同维度。方法只在 R2R 训练后测试 RxR，可说明一种跨数据集迁移，但不能自动证明跨语言、动态人群或任意机器人泛化。类似地，在新场景中允许持续更新模型，与每个 episode 重置的标准离线测试，也应分别报告。

## 八 测试时适应与持续经验利用

测试时适应研究部署过程中如何利用新观察调整策略。它与保存历史图像不同：有些方法更新提示、动作重排序器或模型参数，因而需要明确哪些状态跨任务保留，以及在线学习获得什么反馈。

| 工作 | 更新依据与机制 | 证据边界 |
| --- | --- | --- |
| [FeedTTA，ICML 2025](https://proceedings.mlr.press/v267/kim25ad.html) | 使用每个 episode 结束后的二值成功反馈进行适应，并约束更新以平衡稳定性和可塑性 | 依赖外部结果反馈；不能直接等同于完全无人反馈的部署 |
| [ATENA，2025 首发](https://arxiv.org/abs/2506.06630) | 研究主动获取反馈与测试时适应 | 需单独记录人类反馈或查询预算，不能与完全自主设置直接混比 |
| [IDEA，2026](https://arxiv.org/abs/2605.23257) | 将适应得到的软提示及域信息积累为历史资产，用于新域初始化 | 即使主干冻结，也包含适应和优化开销；不能仅按名称理解为没有学习 |
| [DART-VLN，2026 年 9 月版本](https://arxiv.org/html/2607.01043v4) | 在记忆读取时衰减旧证据，并惩罚局部循环动作，不重训主干 | 主要是离散 VLN 中的测试时控制；不能当连续空间真实回程的直接证明 |
| [CGPI，2026 年 9 月 29 日预印本](https://arxiv.org/html/2609.37591v1) | 从动作导致的观察转移提取相对信用，更新轻量动作重排序器；验证后保留或撤销更新 | 撤销的是策略更新，不是机器人位置或语言任务阶段；主要在既有导航主干上验证 |
| [TAP 与 MemCtrl，2026 年 9 月预印本](https://arxiv.org/html/2609.25666v1) | 分别研究场景习惯引导的搜索与主动记忆管理 | 覆盖相邻具身任务；其收益不能全部归入长程路线跟随 |

从这些工作可以推断，长期导航正在从“模型是否能记住”推进到“哪些经验应保留，何种更新值得接受”。这为研究提供了机会，但也使“加入历史经验或在线更新”不足以构成独立创新。

### 8.1 四种修正需要明确区分

| 修正对象 | 例子 | 是否必然改变机器人位置 |
| --- | --- | --- |
| 语义任务判断 | 撤销“已经经过第二个门”的错误判断 | 否 |
| 物理路线 | 回到已访问岔口，改走另一方向 | 是 |
| 模型或策略参数 | 撤销导致后续表现变差的在线更新 | 否 |
| 外部指令 | 用户纠正“我说的是右边那个房间” | 不一定 |

不能因为某论文含有 rollback、reflection 或 correction，就认定它实现了上表全部能力。反过来，也不能因为正文没有使用某个词，就断言其模型无法产生相应行为。需要阅读执行流程、测试输入及评价指标，必要时通过对照实验确认。

## 九 效率与实际部署的评价方法

### 9.1 将延迟拆到完整执行链

导航系统的时间开销至少包括传感读取、图像或深度处理、历史检索、模型排队、视觉编码、语言解码、动作解析、局部规划和运动执行。首 token 延迟、动作决策延迟与整个任务完成时间不是同一指标。

建议同时记录：每次决策的中位数和 P95 时间、每个任务的模型调用次数和输入输出 token、整个 episode 的墙钟时间、执行器等待时间，以及渲染和环境初始化是否计入统计。若使用闭源 API，还应固定模型版本、推理参数和重试规则；若使用本地模型，应注明硬件、精度、batch、缓存及图像分辨率。

### 9.2 暂停式仿真与实时执行

在许多导航评测中，模型推理时环境不会继续推进。这样的设置适合比较导航决策质量，也可以测计算耗时，但不能单独证明系统能在行人运动或机器人持续行进时实时工作。

异步系统还需要处理观测时刻与执行时刻不一致：模型依据旧视角产生的新目标是否仍有效，局部执行器是否需要等待、取消或重新规划。快慢协同的研究价值应体现在完整任务结果及端到端时间，而不是只比较一个模型的生成速度。

### 9.3 上下文管理的四种机制

| 机制 | 核心做法 | 对长程导航的潜在影响 |
| --- | --- | --- |
| 固定时间窗口 | 只保留最近若干帧或动作 | 成本稳定，但可能丢失远处岔口和早期约束 |
| 图像检索 | 从历史挑选少量相关观察 | 保留视觉细节，但可能检索到外观相似的错误位置 |
| 结构化状态 | 保存地标、任务阶段、拓扑连接和依据 | 便于长时推理，但错误摘要可能长期传播 |
| 学习式压缩 | 将历史编码为 latent、memory token 或外部记忆 | 有效性依赖训练分布，解释与错误定位较困难 |

比较时不应只控制帧数。不同方法的分辨率、全景视图数量、额外深度和预计算特征都会改变信息量及成本。最好同时做固定信息预算、固定计算预算和完整系统三种比较，并说明研究结论适用于哪一种。

### 9.4 真机实验的证据等级

扫描真实房间得到的仿真环境、一次实物机器人演示、多个未见房间的定量实验和连续长期部署，提供的证据逐级增强。论文出现机器人视频不等于方法已经解决部署问题。应记录成功与失败次数、任务选择、操作员介入、碰撞或接管、地图与定位来源，以及模型是否使用了额外设备信息。

对于毕业论文，公开仿真基准可以构成主要实证；如具备条件，真机实验可检验感知噪声、延迟和控制接口。若尚无真机条件，应将结论明确限定为室内三维仿真导航，不必用未经验证的部署描述扩大结论。

## 十 当前已经取得的进展与仍然存在的问题

### 10.1 已有较明确进展

**语言视觉到动作的统一建模已经可行。** 专用导航表征、冻结大模型工具系统和导航微调 VLM 均能在公开任务中形成完整观察—动作闭环。研究重点已不再是证明语言模型能否输出导航动作，而是它在哪些条件下可靠、成本如何、是否能泛化。

**历史信息的组织方式明显影响长任务。** 循环状态、分层 Transformer、地图、外部记忆与流式模型分别提供不同折中。领域已经从“加入历史即可”进入如何保留相关信息、减少冗余并防止错误状态持续传播的问题。

**进度与反馈已成为显式学习对象。** 进度预测、语义阶段、主动观察、纠正轨迹训练、偏航监测以及在线适应都已有独立研究。对于研究生选题，必须在这些近邻之上提出可检验差异。

**效率开始进入系统设计。** 短时动作序列、快慢协同、隐式推理、候选动作验证和异步执行均有探索。但局部算子加速与完整导航效率之间仍需要实测联系。

### 10.2 论文内部的量化进展

下表只比较同一论文中的模型变体，全部是作者报告，未在本项目复现。SR、SPL 和 nDTW 均按论文的 0–100 标度列示，差值为该标度上的增量；成功率差值使用百分点。各行不构成跨论文排名。

| 控制问题与 baseline → method | 已核验的导航指标 | 原文与版本 | 控制变量、数据与分母 | 能支持什么；不能外推什么 |
|---|---|---|---|---|
| StreamVLN：记忆 `2×196` → `8×196` token | R2R-CE val-unseen：SR **37.3→45.5，+8.2 pp**；SPL **34.2→41.6，+7.4 pp**。`all` 历史的 SR=40.0、SPL=36.4，低于有限记忆 | [2026-07-09 v2，Table IV](https://arxiv.org/html/2507.05240v2#S4.T4) | 相同架构、窗口固定8轮；只用第一阶段 oracle 导航训练。不是声称同一 checkpoint 的无训练替换。该表未给 episode N/标注包版本 | 记忆容量影响导航，更多并非总好；R2R消融不能证明超长任务或事件记忆已解决 |
| StreamVLN：不加 → 加 ScaleVLN | R2R：SR **52.8→56.4，+3.6 pp**；SPL **47.2→50.2，+3.0 pp**。RxR：SR **48.6→54.4，+5.8 pp**；SPL **42.5→45.4，+2.9 pp**；nDTW **60.2→63.7，+3.5** | [v2 Table III](https://arxiv.org/html/2507.05240v2#S4.T3)核对训练配置，[Table I](https://arxiv.org/html/2507.05240v2#S3.T1)核对RxR | 同骨干，RGB、不用几何剪枝；已有R2R+RxR+DAgger+VidQA/MMC4，再加约150K ScaleVLN轨迹。val-unseen；表未列N | 有较长RxR上的收益，但它来自额外训练数据；**不是零样本RxR，不是单独记忆模块收益** |
| NaVILA：8 → 64帧记忆 | R2R-CE val-unseen：SR **49.7→50.1，+0.4 pp**；SPL **45.5→45.4，−0.1 pp**；16帧SR=48.6，32帧SR=49.5 | [2024-12-05 v1，Table 8](https://arxiv.org/html/2412.04453v1#A4.T8) | 同方法家族的历史长度消融，单视角RGB；表未声称同一checkpoint，也未列N或标注包版本 | 盲目增加帧数未呈现稳定收益；无方差/显著性证据，不能把+0.4认定为可靠改进；不是RxR超长程证明 |
| StreamVLN同权重：原runtime → LiveVLN | R2R：SR **56.4→57.2**、SPL **50.2→50.0**；RxR：SR **54.4→53.7**、SPL **45.4→44.5**、nDTW **63.7→63.0** | [2026-04-21，Table I](https://arxiv.org/html/2604.19536v1#S3.T1) | 论文明确同预训练checkpoint、training-free wrapper、RGB-only、val-unseen；双方额外导航训练数据一致。该表未给N | runtime效率可以与能力近似保持同时出现，但不是各指标都提升；不能把更顺畅直接解释为更强长程推理 |

#### 同权重效率的补充证据

LiveVLN **Table II** 的 StreamVLN 真实机器人对照中，停在目标2m内是 **15/40→16/40**；等待 **7.32→1.63秒**，任务墙钟 **41.98→36.71秒**。机器人、场景与checkpoint匹配；Unitree G1、远端RTX 5090。它支持减少执行等待，**一例成功之差不足以证明导航能力显著提升**，也不能把40次实机实验当仿真表的分母。[原文 Table II](https://arxiv.org/html/2604.19536v1#S4.T2)

#### 数字使用与版本边界

- 本表选择明确的版本锚点。**StreamVLN v2 的 Table III 单元格已变化，但紧邻解释段仍含旧版增益**；上面的增益按表中数字计算，不照抄段落的旧差值。
- StreamVLN 官方仓库目前推荐 `R2R_VLNCE_v1-3`，并记录了2025-09 checkpoint更新；这不能代替论文每张表的精确数据包证据。正式复现应锁定数据包、checkpoint和commit，并实际统计评测episode数。[官方README](https://github.com/InternRobotics/StreamVLN)
- 不使用完整RxR数据集的126K指令数作为val-unseen评测分母；论文表没有列出的分母明确保留为未核验，不根据惯常配置猜测。
- 这四组结果的研究含义是：数据与闭环训练仍有明显影响；历史容量存在有效区间；执行效率与导航正确性必须分别测。它们不构成跨论文排名，也不保证新方法收益。

### 10.3 尚未充分解决的问题

| 问题 | 为何现有结果仍不足 | 更有解释力的评价 |
| --- | --- | --- |
| 长任务中错误持续传播 | 平均成功率难定位哪次地标或阶段判断造成后续失败 | 按错误起点、阶段数及剩余预算分组 |
| 偏航检测的误报 | 正常绕行可能被误判为失败，反而触发多余回退 | 错误识别精确率与召回率，错误干预率 |
| 恢复是否真正符合指令 | 到终点的最短修正路线未必满足原有顺序要求 | SR 与 nDTW/SDTW、阶段条件联合评价 |
| 主动观察是否值得 | 更多视图、动作和计算本身可提高成功率 | 固定预算及等成本对照，观察净收益 |
| 长期记忆中的错误事实 | 生成摘要可能丢失依据，把推测变为既定状态 | 状态正确性、依据一致性、错误传播范围 |
| 在线适应是否稳定 | 自生成监督可能强化原有偏差 | 跨 episode 回归、更新保留收益与负迁移 |
| 泛化与协议可比性 | 感知、训练和控制条件不同可能解释分差 | 同主干、同传感器、同执行器的成对实验 |

表中的判断来自多条研究路线的综合分析，不表示每篇现有论文都存在同样缺陷，更不表示这些问题完全无人研究。

### 10.4 对拟议研究内容的重新定位

“基于多模态大模型的室内长程视觉语言导航方法研究”可以继续作为研究点标题，但应先保留具体方法的开放性。更充分的近邻检索已经表明，单独提出任务进度、主动补充观察、偏航后回退或恢复轨迹训练，均不足以确立贡献。

可优先检验的一个问题是：**在没有额外人类纠正的情况下，如何区分应当修正的错误任务判断与可以接受的执行偏差，并在原动作预算内采取有效调整。** 这里的重要区别是“何时修正、修正什么、是否造成新的错误”，而不是仅增加一个恢复分支。

该问题是否值得最终立题，取决于自然失败审计：若大多数失败由基础感知或低层控制决定，或强基线已经能有效恢复，则不应继续围绕任务状态强行增加模块。可以转向受计算预算约束的决策、跨阶段停止判断，或更清楚的场景泛化问题。以上均是候选方向，而非已验证的创新结论。

## 十一 建议的实验路线

### 11.1 先建立可复核的基线

主实验选择 R2R-CE 与 RxR-CE，各基准内部统一不同方法的视觉输入、动作、控制器和环境版本，跨基准保留原协议差异。LHPR-VLN 用于补充多阶段任务评价，但必须先确认其数据组成及 STOP、阶段推进、成功判定和代价计算方式。若论文主旨改为连续目标搜索，则独立采用 GOAT-Bench 与 LangMap，不把两种任务的成功率求一个混合均值。

基线至少应覆盖传统地图或规划方法、一个公开的导航微调 VLM，以及最接近拟议机制的方法。可以从 ETPNav、StreamVLN 和进度/纠错类模型中选择；不能只与直接提示通用大模型的弱基线比较。代码或权重未公开的方法可作为文献近邻，但若自行复现，应标明复现版本，不能把它等同于作者实现。

### 11.2 失败审计与立题门槛

建议首先从训练或专门开发划分中收集一批完整 episode，包含成功和失败。保存每步观察、动作、任务解释、STOP、环境反馈及耗时。人工审计可使用以下分类：目标或地标误识别、左右及序数关系错误、阶段推进错误、历史遗忘、路线循环、局部控制失败、过早停止和预算耗尽。

分类时允许多种原因共同出现，并记录首次可识别错误与最终失败之间的距离。不要只对失败终帧做解释，也不要用模型生成的理由替代独立标注。若研究恢复，还应估计剩余预算内存在可行恢复方案的比例；使用真值进行上界分析时，应明确它只是离线诊断，不是部署策略输入。

### 11.3 训练与开发划分

监督数据生成和模型训练使用训练划分；提示选择、模型选择与模块阈值使用预先固定的开发划分；测试及预留最终报告划分在方案冻结后评测。若反复使用官方 val_unseen 调参，应披露其开发用途，并另保留最终检验集。恢复数据可以包括基线自然偏航、受控动作偏差和相似地标干扰，但应分别标明来源。测试集的真值目标、未来视图、参考路径和隐藏成功标签不能作为在线判断工具。

如果使用未来专家动作生成训练标签，必须剥离未来信息后构造测试接口。若原协议只提供当前目标，则不能让规划器提前访问整串未来目标。跨任务记忆和在线更新需要固定保留规则，避免用“测试适应”掩盖额外标注输入。

### 11.4 用实验隔离状态修正与动作恢复

以下消融用于研究假设验证，不代表已有方法：

| 实验条件 | 不开启显式恢复回访模块 | 开启显式恢复回访模块 |
| --- | --- | --- |
| 不显式修正任务判断 | 同主干基线 | 简单历史位置回退 |
| 显式修正任务判断 | 仅更新内部进度与指令对应 | 完整联合方法 |

这组对照还需加入等调用、等观察预算设置。若完整方法只是用了更多图像或更长动作预算，不能将收益全部归因于状态建模。对于主动观察，应增加固定频率补看和随机补看的同预算对照；对于记忆，应比较最近窗口、均匀采样及等容量检索。

“不开启恢复模块”不禁止原策略的合法折返，尤其不能删除 RxR 指令本来要求的回访。在 LHPR 中，内部阶段修正应发生在不可逆的 STOP 之前；错误 STOP 后只能改善后续阶段，不能撤销环境已经记录的失败。标准主表保留原起点、观察和动作协议；人为偏航、定位噪声、遮挡等实验另列为扰动测试，并固定样本与随机种子。

### 11.5 指标与统计报告

主要结果保留官方指标：成功率、路径效率、路径忠实度，以及多阶段任务的整任务和子任务成绩。机制分析可补充错误干预率、偏航后的恢复率、恢复增加的路程和动作、过早停止率、模型调用量及耗时。

报告结果时注明全部分母，避免只展示“被检测出的错误中的恢复率”而忽略漏检。优先采用以场景为簇的成对 bootstrap，同一路线的多条指令、同一长任务的子阶段整体保留，不能把每帧当独立样本。R2R/RxR 的未见验证建筑只有 11 个，LHPR 当前四阶段测试只有 12 条，应同时提供原始成功数与不确定性。对于随机模型和在线训练，保留多个随机种子；重复采样同一条任务不能冒充新增任务数量。

### 11.6 适合一章论文的工作组织

| 阶段 | 必须形成的产物 | 继续推进的依据 |
| --- | --- | --- |
| 任务与协议确定 | 各基准内可执行的统一配置、环境版本和信息边界 | 能稳定复现至少一个强基线 |
| 问题诊断 | 有分母的失败类型、可恢复上界和代价分析 | 拟研究问题占比足够且未被已有机制覆盖 |
| 方法设计 | 状态表示、决策机制、训练目标或明确的推理算法 | 每个组件对应一个已观察到的问题 |
| 系统与训练 | 数据构造、模型训练或策略实现、可追溯日志 | 不依赖测试期特权信息 |
| 实验验证 | 官方主结果、机制消融、跨长度与跨场景分析 | 收益不能由额外输入、计算或控制器替换解释 |

这一安排既能体现系统和实验工作量，也能避免仅靠模块堆叠形成“复杂系统”。研究贡献最终应落在一个被明确识别和有效改善的问题上。

## 十二 文献阅读与选题推进建议

第一轮阅读建议关注任务和评价：R2R、VLN-CE、RxR、nDTW、LHPR-VLN。读完后应能解释为什么到达终点、遵循路线、完成多个阶段和长期巡逻是不同的目标，并能明确拟使用的观察与动作协议。

第二轮关注方法基础：HAMT、DUET、ETPNav，以及 NavGPT、MapGPT、NaVid。重点比较历史保存在何处、地图由谁生成、语言如何进入动作决策、训练和测试分别可以使用什么信息。

第三轮关注当前直接近邻：StreamVLN、Progress-Think、SeekVLN、CorrectNav、BudVLN，以及偏航监测和交互纠正方法。阅读时为每篇填写同一张表：错误类型、检测依据、干预动作、训练监督、测试期反馈、额外代价及评价指标。

第四轮再选择扩展：效率问题阅读快慢协同及验证式决策；持续适应问题阅读 FeedTTA、CGPI 等；若转向开放目标搜索，则另读 GOAT-Bench 和 LangMap。无需一开始将所有分支接入同一系统。

现阶段可以保留学位论文研究点名称，但开题中的具体创新表述应在基线审计后确定。较稳妥的措辞是“研究任务进度感知、环境反馈利用及导航偏差修正方法”，而不是提前宣称提出首个具有恢复能力的导航系统。

## 十三 主要文献与公开资源

正文已在具体结论附近给出直接来源。下列索引帮助按问题阅读；它不是逐项复现实验清单。论文已公开、代码已公开、权重已公开和完整训练数据已公开是不同状态。

### 任务与评价

| 标题 / 资源 | 年份与会议/性质 | 直接链接 | 本次读到的证据类型 |
|---|---|---|---|
| Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments | CVPR 2018 | [论文](https://arxiv.org/abs/1711.07280) | PDF：任务、数据收集、图动作、规模 |
| Matterport3D Simulator / R2R | 官方代码与任务 | [仓库](https://github.com/peteanderson80/Matterport3DSimulator) | README、任务格式、eval 源码 |
| Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments | ECCV 2020 | [论文](https://arxiv.org/abs/2004.02857) / [代码](https://github.com/jacobkrantz/VLN-CE) | 官方 README、R2R/RxR yaml、数据版本页、权重/提交入口 |
| Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding | EMNLP 2020 | [论文](https://arxiv.org/abs/2010.07954) / [数据](https://github.com/google-research-datasets/RxR) | 官方数据说明；CE移植由 VLN-CE 配置交叉核对 |
| ETPNav: Evolving Topological Planning for VLN in Continuous Environments | TPAMI 2024；本文核验 v3 2024-01 | [论文](https://arxiv.org/html/2304.03047v3) | 全文：CE 规模表、panorama/pose 假设、指标、controller |
| Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation | ACL 2019 | [论文](https://arxiv.org/abs/1905.12255) / [生成代码](https://github.com/google-research/google-research/tree/master/r4r) | 官方 README：生成方式、阈值与输出数量 |
| General Evaluation for Instruction Conditioned Navigation Using Dynamic Time Warping | 2019，ViGIL@NeurIPS / arXiv | [论文](https://arxiv.org/abs/1907.05446) | 原论文摘要和指标定义说明 |
| Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method | CVPR 2025 | [正式入口](https://openaccess.thecvf.com/content/CVPR2025/html/Song_Towards_Long-Horizon_Vision-Language_Navigation_Platform_Benchmark_and_Method_CVPR_2025_paper.html) / [全文](https://arxiv.org/html/2412.09082v2) | 正式出版元数据、全文任务定义/规模/指标 |
| LH-VLN 公开实现 | commit 3492080，2025-08-20 | [仓库](https://github.com/HCPLab-SYSU/LH-VLN/tree/3492080382aba031e9d82a4f630c2b1cdf57df81) | loader、split、actor、policy输入、metrics、config、challenge规则 |
| LHPR-VLN 公开数据 | HF revision af66785，2025-05-30 | [固定 episode 包](https://huggingface.co/datasets/Starry123/LHPR-VLN/resolve/af66785e0b456639088d673bc100dcc63f2df997/episode_task.zip) | 内存解压8个JSON并计数；未跑simulator |
| REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments | CVPR 2020 | [论文](https://arxiv.org/abs/1904.10151) / [仓库](https://github.com/YuankaiQi/REVERIE) | 官方数据页、任务、权重/eval入口、2026测试提交公告 |
| SOON: Scenario Oriented Object Navigation with Graph-Based Exploration | CVPR 2021 | [论文](https://arxiv.org/abs/2103.17138) / [代码](https://github.com/ZhuFengdaaa/SOON) | PDF任务、划分、数量和指标；README训练/eval入口 |
| GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation | CVPR 2024 | [论文](https://arxiv.org/abs/2404.06609) / [代码](https://github.com/Ram81/goat-bench) | 原文和README：逐目标揭示、传感器、预算、划分、checkpoint |
| LangMap: A Human-Verified Benchmark for Hierarchical Open-Vocabulary Goal Navigation | 2026；官方仓库标为 NeurIPS 2026 | [论文v3](https://arxiv.org/html/2602.02220v3) / [固定代码](https://github.com/bo-miao/LangMap/tree/a37db3b1b06cec80bfefe3e05e6084efddd70698) | 全文、README、INSTALL、run/run_seq/analyze 源码；记录论文/实现差异 |
| Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation | CVPR 2026；本文核验 v2 | [论文](https://arxiv.org/abs/2511.17097) | 摘要、方法与实验设置；作为近邻机制约束 |
| Seek Before You Move: Evidence Seeking for Progress Grounding in Vision-Language Navigation | 2026-09-29 预印本 | [论文](https://arxiv.org/abs/2609.37353) | 原文/摘要：主动补充观察、反事实训练；作为近邻机制约束 |


### 历史方法与通用训练基础

| 工作 | 年份与出处 | 本报告使用的内容 |
| --- | --- | --- |
| [Speaker-Follower Models for Vision-and-Language Navigation](https://papers.neurips.cc/paper_files/paper/2018/hash/6a81681a7af700c6385d36577ebec359-Abstract.html) | NeurIPS 2018 | 轨迹到语言的数据增强与候选排序 |
| [Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training](https://arxiv.org/abs/2002.10638) | CVPR 2020，PREVALENT | 视觉语言动作预训练 |
| [VLN BERT: A Recurrent Vision-and-Language BERT for Navigation](https://openaccess.thecvf.com/content/CVPR2021/html/Hong_VLN_BERT_A_Recurrent_Vision-and-Language_BERT_for_Navigation_CVPR_2021_paper.html) | CVPR 2021 | 循环跨模态状态 |
| [History Aware Multimodal Transformer for Vision-and-Language Navigation](https://proceedings.nips.cc/paper_files/paper/2021/hash/2e5c2cb8d13e8fba78d95211440ba326-Abstract.html) | NeurIPS 2021，HAMT | 分层历史建模 |
| [Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Think_Global_Act_Local_Dual-Scale_Graph_Transformer_for_Vision-and-Language_Navigation_CVPR_2022_paper.html) | CVPR 2022，DUET | 全局拓扑与局部匹配 |
| [BEVBert: Multimodal Map Pre-training for Language-guided Navigation](https://arxiv.org/abs/2212.04385) | ICCV 2023 | 地图表示预训练 |
| [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) | AISTATS 2011，DAgger | 学习者状态分布上的专家监督 |
| [On Evaluation of Embodied Navigation Agents](https://arxiv.org/abs/1807.06757) | 2018 | 导航评价与 SPL 的基础 |

### 视频动作模型与系统效率

| 工作 | 本次确认的版本与出处 | 主要来源 |
| --- | --- | --- |
| NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation | RSS 2024 | [论文](https://arxiv.org/abs/2402.15852)、[评测实现](https://github.com/jzhzhang/NaVid-VLN-CE) |
| Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks | RSS 2025 | [论文](https://arxiv.org/abs/2412.06224)、[代码](https://github.com/jzhzhang/Uni-NaVid) |
| NaVILA: Legged Robot Vision-Language-Action Model for Navigation | RSS 2025 | [论文](https://arxiv.org/abs/2412.04453)、[代码](https://github.com/AnjieCheng/NaVILA) |
| StreamVLN | ICRA 2026；2026-07 v2 | [论文](https://arxiv.org/abs/2507.05240)、[代码](https://github.com/InternRobotics/StreamVLN) |
| DualVLN / Ground Slow, Move Fast | ICLR 2026 | [论文](https://arxiv.org/abs/2512.08186)、[InternNav](https://github.com/InternRobotics/InternNav) |
| LiveVLN | 2026-04 预印本；仓库标为 RA-L 2026 | [论文](https://arxiv.org/abs/2604.19536)、[代码](https://github.com/NIneeeeeem/LiveVLN) |
| Qwen-RobotNav | 2026-06 技术报告 | [论文](https://arxiv.org/abs/2606.18112)、[仓库](https://github.com/QwenLM/Qwen-RobotNav) |
| SPARK-VLN | 2026-07 预印本；作者页为 Under Review | [论文](https://arxiv.org/abs/2607.16806)、[项目页](https://hutslib.github.io/SPARK-VLN.dc.html) |
| VerNav | 2026-09 预印本 | [论文](https://arxiv.org/abs/2609.00920) |
| VLingNav | 2026-01 预印本；相邻目标导航任务 | [论文](https://arxiv.org/abs/2601.08665)、[项目页](https://shaoanwang.github.io/VLingNav-web/) |

### 任务进度与恢复

“方法正文”表示已阅读方法／实验段落；“官方仓库说明”不表示已运行代码。预印本仅说明本次确认的状态，不推断一定未被任何会议接收。

| 工作 | 首次公开日期 | 截至本次核验的会议／版本状态 | 方法证据与官方来源 |
|---|---|---|---|
| Self-Monitoring Navigation Agent via Auxiliary Progress Estimation | 2019-01-10 | ICLR 2019 | [arXiv](https://arxiv.org/abs/1901.03035)，正文 §2.2–2.4；[代码](https://github.com/chihyaoma/selfmonitoring-agent) |
| The Regretful Agent: Heuristic-Aided Navigation through Progress Estimation | 2019-03-05 | CVPR 2019 | [arXiv 正文](https://arxiv.org/html/1903.01602v1)，§4；[正式记录](https://openaccess.thecvf.com/content_CVPR_2019/html/Ma_The_Regretful_Agent_Heuristic-Aided_Navigation_Through_Progress_Estimation_CVPR_2019_paper.html) |
| ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments | 2023-04-06 | TPAMI 2024 | [arXiv](https://arxiv.org/abs/2304.03047)，控制器及消融正文；[官方仓库](https://github.com/MarSaKi/ETPNav) |
| NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models | 2023-05-26 | AAAI 2024 | [arXiv](https://arxiv.org/abs/2305.16986)，方法正文；[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28597) |
| MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation | 2024-01-14 | ACL 2024 | [ACL](https://aclanthology.org/2024.acl-long.529.pdf)，方法正文；[代码](https://github.com/chen-judge/MapGPT/) |
| NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models | 2024-07-17 | ECCV 2024 | [arXiv](https://arxiv.org/abs/2407.12366)，方法正文；[ECVA 正式论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01143.pdf) |
| CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model | 2025-08-14 | AAAI 2026 | [arXiv 正文](https://arxiv.org/html/2508.10416v1)；[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38942) |
| Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation | 2025-11-21 | 2026-04-14 v2；CVPR 2026 | [arXiv](https://arxiv.org/abs/2511.17097)，方法／奖励正文；[CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Progress-Think_Semantic_Progress_Reasoning_for_Vision-Language_Navigation_CVPR_2026_paper.html) |
| FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation | 2026-01-20 | 2026-01-23 v2；CVPR 2026 | [arXiv](https://arxiv.org/abs/2601.13976)，方法正文；[官方代码](https://github.com/Fantasy-AMAP/fantasy-vln)，[CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zuo_FantasyVLN_Unified_Multimodal_Chain-of-Thought_Reasoning_for_Vision-and-Language_Navigation_CVPR_2026_paper.pdf) |
| Nipping the Drift in the Bud: Retrospective Rectification for Robust Vision-Language Navigation | 2026-02-06 | arXiv v1；未独立确认会议录用 | [arXiv](https://arxiv.org/abs/2602.06356)，§3.2–3.5 方法正文 |
| Let's Reward Step-by-Step: Step-Aware Contrastive Alignment for Vision-Language Navigation in Continuous Environments | 2026-03-10 | arXiv v1；未独立确认会议录用 | [arXiv](https://arxiv.org/abs/2603.09740)，§3 和算法正文 |
| Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection | 2026-03-14 | arXiv v1；本次未确认会议 | [arXiv](https://arxiv.org/abs/2603.13782)，§3/4；[作者项目](https://jaehwan-j.github.io/projects/hivla/hivla.html) |
| Talk2Escape: Conversational Grounding for Vision-and-Language Navigation | 2026-09-23 | arXiv v1，预印本 | [正文](https://arxiv.org/html/2609.28296v1)，oracle、触发和子集协议 |
| Seek Before You Move: Evidence Seeking for Progress Grounding in Vision-Language Navigation | 2026-09-29 | arXiv v1，预印本 | [arXiv](https://arxiv.org/abs/2609.37353)，§3、附录 D/E |
| AVERT-VLN: Abstention-aware Visual Error Recovery and Training for Vision-and-Language Navigation | 2026-09-30 | arXiv v1，预印本 | [arXiv](https://arxiv.org/abs/2609.39579)，§III/IV；人工辅助主表注释 |

### 测试时适应与持续经验

| 工作 | 本次使用的来源 | 研究对象 |
| --- | --- | --- |
| FeedTTA | [ICML 2025 论文集](https://proceedings.mlr.press/v267/kim25ad.html) | 二值结果反馈下的测试时适应 |
| ATENA | [2025 年论文](https://arxiv.org/abs/2506.06630) | 主动反馈与适应 |
| IDEA | [2026 年论文](https://arxiv.org/abs/2605.23257) | 软提示与历史域资产 |
| DART-VLN | [2026-09 v4](https://arxiv.org/html/2607.01043v4) | 记忆读取衰减与循环控制 |
| CGPI | [2026-09-29 v1](https://arxiv.org/html/2609.37591v1) | 无外部成功反馈的信用引导策略更新 |
| TAP 与 MemCtrl | [2026-09-22 v1](https://arxiv.org/html/2609.25666v1) | 相邻任务中的搜索经验和记忆管理 |
