**研究点三具有可论证的系统方法贡献，但目前不足以把“增量场景记忆”“程序化执行”“前置条件修复”或“完成核验”分别宣称为首创。** 更有区分度的定位是：以共享的三维实例身份、容器拓扑和操作状态证据，支持受限调用预算下的局部子目标修复，并在终止时检查跨步骤目标是否一致满足。

我已对照阅读 [Markdown 原稿](/tmp/sec3.md) 与 [XML 版 3.3](/home/ysh/codebase/agentic-3d-task/draft_96fcab98_folder/draft.xml:402)。以下核验范围是文稿与公开原始文献，未复现实验、未独立验证运行日志；未修改任何文件。检索范围截至 **2026 年 11 月 10 日**。

**文献清单共 25 篇。** 长作者名单以“等”缩写；会议年份与论文集出版年份不同时分别注明。

| 编号 | 英文标题 | 作者 | 出处与年份 |
|---|---|---|---|
| 1 | [ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks](https://arxiv.org/abs/1912.01734) | Mohit Shridhar、Jesse Thomason、Daniel Gordon 等 | CVPR 2020；arXiv:1912.01734，初稿 2019 |
| 2 | [Factorizing Perception and Policy for Interactive Instruction Following](https://openaccess.thecvf.com/content/ICCV2021/html/Singh_Factorizing_Perception_and_Policy_for_Interactive_Instruction_Following_ICCV_2021_paper.html)（MOCA） | Kunal Pratap Singh、Suvaansh Bhambri、Byeonghwi Kim、Roozbeh Mottaghi、Jonghyun Choi | ICCV 2021；arXiv:2012.03208 |
| 3 | [A Persistent Spatial Semantic Representation for High-level Natural Language Instruction Execution](https://proceedings.mlr.press/v164/blukis22a.html)（HLSM） | Valts Blukis、Chris Paxton、Dieter Fox、Animesh Garg、Yoav Artzi | CoRL 2021；PMLR 164，2022；arXiv:2107.05612 |
| 4 | [FILM: Following Instructions in Language with Modular Methods](https://arxiv.org/abs/2110.07342) | So Yeon Min、Devendra Singh Chaplot、Pradeep Ravikumar、Yonatan Bisk、Ruslan Salakhutdinov | ICLR 2022；arXiv:2110.07342 |
| 5 | [LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models](https://openaccess.thecvf.com/content/ICCV2023/html/Song_LLM-Planner_Few-Shot_Grounded_Planning_for_Embodied_Agents_with_Large_Language_ICCV_2023_paper.html) | Chan Hee Song、Jiaman Wu、Clayton Washington 等 | ICCV 2023；arXiv:2212.04088 |
| 6 | [Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models](https://aclanthology.org/2023.findings-emnlp.226/)（HELPER） | Gabriel Sarch、Yue Wu、Michael J. Tarr、Katerina Fragkiadaki | Findings of EMNLP 2023；arXiv:2310.15127 |
| 7 | [Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents](https://arxiv.org/abs/2308.07241)（CAPEAM） | Byeonghwi Kim、Jinyeon Kim、Yuyeong Kim、Cheolhong Min、Jonghyun Choi | ICCV 2023；arXiv:2308.07241 |
| 8 | [Pre-emptive Action Revision by Environmental Feedback for Embodied Instruction Following Agents](https://proceedings.mlr.press/v270/kim25b.html)（PRED） | Jinyeon Kim、Cheolhong Min、Byeonghwi Kim、Jonghyun Choi | CoRL 2024；PMLR 270，2025 |
| 9 | [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)（SayCan） | Michael Ahn 等，按所链 arXiv 版本 | arXiv:2204.01691，2022；正式版 CoRL 2022 |
| 10 | [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753) | Jacky Liang、Wenlong Huang、Fei Xia 等 | ICRA 2023；arXiv:2209.07753 |
| 11 | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | Shunyu Yao、Jeffrey Zhao、Dian Yu 等 | ICLR 2023；arXiv:2210.03629 |
| 12 | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf) | Noah Shinn、Federico Cassano、Ashwin Gopinath、Karthik Narasimhan、Shunyu Yao | NeurIPS 2023；arXiv:2303.11366 |
| 13 | [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://proceedings.mlr.press/v205/huang23c.html) | Wenlong Huang、Fei Xia、Ted Xiao 等 | CoRL 2022；PMLR 205，2023；arXiv:2207.05608 |
| 14 | [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | Guanzhi Wang、Yuqi Xie、Yunfan Jiang 等 | **TMLR 2024**；arXiv:2305.16291，2023 |
| 15 | [LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477) | Bo Liu、Yuqian Jiang、Xiaohan Zhang 等 | arXiv:2304.11477，2023；本次未核实正式会议出处 |
| 16 | [ProgPrompt: Generating Situated Robot Task Plans using Large Language Models](https://arxiv.org/abs/2209.11302) | Ishika Singh、Valts Blukis、Arsalan Mousavian 等 | ICRA 2023；arXiv:2209.11302，2022 |
| 17 | [CAPE: Corrective Actions from Precondition Errors using Large Language Models](https://arxiv.org/abs/2211.09935) | Shreyas Sundara Raman、Vanya Cohen、Ifrah Idrees 等 | ICRA 2024；arXiv:2211.09935，2022 |
| 18 | [DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment](https://arxiv.org/abs/2307.00329) | Yanjiang Guo、Yen-Jen Wang、Lihan Zha、Jianyu Chen，按当前正式版本 | IROS 2024；arXiv:2307.00329，2023 |
| 19 | [ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://concept-graphs.github.io/) | Qiao Gu、Ali Kuwajerwala、Sacha Morin 等 | ICRA 2024；arXiv:2309.16650 |
| 20 | [RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation](https://proceedings.mlr.press/v270/jiang25c.html) | Hanxiao Jiang、Binghao Huang、Ruihai Wu 等 | CoRL 2024；PMLR 270，2025；arXiv:2402.15487 |
| 21 | [Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Language-Guided Mobile Manipulation](https://arxiv.org/abs/2410.11989)（DovSG） | Zhijie Yan、Shufei Li、Zuoxu Wang 等 | IEEE RA-L 2025；arXiv:2410.11989 |
| 22 | [Integrated Exploration and Sequential Manipulation on Scene Graph with LLM-based Situated Replanning](https://arxiv.org/abs/2602.04419)（EPoG） | Heqing Yang、Ziyuan Jiao、Shu Wang、Yida Niu、Si Liu、Hangxin Liu | ICRA 2026，作者记录注明已接收；arXiv:2602.04419 |
| 23 | [EmbodiedMemory-Bench: Benchmarking Embodied Memory for Long-Horizon Embodied Tasks](https://arxiv.org/abs/2609.28236)（提出 EMem） | Lizhou Liang、Xinyu Zhong、Miao Pan 等 | **arXiv 预印本** 2609.28236，2026；未核实同行评审录用 |
| 24 | [Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://papers.nips.cc/paper/2021/hash/021bbc7ee20b71134d53e20206bd6feb-Abstract.html) | Andrew Szot、Alexander Clegg、Eric Undersander 等 | NeurIPS 2021；arXiv:2106.14405 |
| 25 | [BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments](https://proceedings.mlr.press/v164/srivastava22a.html) | Sanjana Srivastava、Chengshu Li、Michael Lingelbach 等 | CoRL 2021；PMLR 164，2022；arXiv:2108.03332 |

**比较时需要区分三件事：可行性评分、实际执行核验和任务完成核验。** SayCan 的成功概率不是显式前置条件；动作 API 返回成功不等于全部任务目标满足；PDDL 模型内的正确性也不等于真实物理执行正确性。

下表的“程序级执行”指高层计划是否以代码、DSL 或显式程序结构表示并解释执行；“未报告”仅表示本次核对未找到相应机制。为便于阅读，对照表分为三组。

| 方法 | 任务 | 记忆／状态表示 | 规划粒度 | 程序级执行 | 前／后置条件核验 | 失败回退 | 主要假设 | 与本文差异 |
|---|---|---|---|---|---|---|---|---|
| [MOCA](https://arxiv.org/abs/2012.03208) | ALFRED | 循环隐状态、二维实例关联 | 原子动作 | 否 | 无完整显式契约；有受阻检测 | 屏蔽受阻动作、选择其他动作 | 专家监督、视觉交互掩码 | 无本文实例拓扑账本，但不能称其没有记忆或恢复 |
| [HLSM](https://proceedings.mlr.press/v164/blukis22a.html) | ALFRED | 持续三维语义体素、可观测性地图 | 高层子目标＋低层控制 | 子目标接口，非生成程序 | 技能内控制与交互检查；未报告同类全局终止门控 | 探索、导航、交互位姿搜索 | 学习感知模型；位姿与持物观测 | 增量空间记忆已有；本文增加实例拓扑、加工状态及门控 |
| [FILM](https://arxiv.org/abs/2110.07342) | ALFRED | 语义地图、障碍与探索信息 | 任务模板→子任务→动作 | 模板式确定性执行 | 图像变化判断执行结果；非通用谓词核验 | 搜索与执行启发式 | 任务类型模板、预测深度与分割 | **5 cm 栅格和 FMM 已有**；本文差异在状态与程序保护 |
| [LLM-Planner](https://arxiv.org/abs/2212.04088) | ALFRED | 已见对象、完成子目标；底层 HLSM 地图 | LLM 高层计划 | 子目标序列 | 环境反馈接地；非全面契约 | 失败／超时触发 LLM 重规划 | 少量示例、既有低层执行器 | 本文把部分恢复移至不重新调用 LLM 的执行器 |
| [HELPER](https://aclanthology.org/2023.findings-emnlp.226/) | 原论文主评 TEACh | 语言—程序记忆＋在线三维对象及属性状态 | API 程序＋视觉运动例程 | 是 | **执行前检查并补足前提**；结合失败反馈 | VLM 识别错误，LLM 生成纠正程序 | 有限机器人 API、预定义错误描述及感知模块 | 强近邻；本文需证明本地候选回退与终止谓词的增量作用 |
| [CAPEAM](https://arxiv.org/abs/2308.07241) | ALFRED | 语义空间地图、对象迁移与掩码缓存 | 上下文子目标＋详细规划 | 非 LLM 生成程序 | 跟踪操作后的对象状态；无同类终止门控证据 | 状态跟踪及掩码记忆减少重复、错误交互 | 学习式规划和感知 | 已处理“搬过哪个对象”；本文须细化状态与拓扑语义 |
| [PRED](https://proceedings.mlr.press/v270/kim25b.html) | ALFRED、TEACh、机器人 | 对象存在、外观、属性、关系反馈 | 子目标／动作计划修订 | 计划修订，非通用代码策略 | **执行前环境差异检查**，并核验对象身份 | LLM 修订目标和动作 | 可提取环境差异、既有执行系统 | 本文主要区别是常见异常在本地解释器内处理 |

| 方法 | 任务 | 记忆／状态表示 | 规划粒度 | 程序级执行 | 前／后置条件核验 | 失败回退 | 主要假设 | 与本文差异 |
|---|---|---|---|---|---|---|---|---|
| [SayCan](https://arxiv.org/abs/2204.01691) | 真实机器人家务技能 | 当前状态、历史技能、价值函数 | 技能选择 | 否 | 执行前软可行性评分；非显式谓词修复 | 原版未形成同类错误诊断恢复环 | 已训练技能和成功价值函数 | 本文以明确状态和局部修复约束执行 |
| [Code as Policies](https://arxiv.org/abs/2209.07753) | 桌面操作、移动机器人控制 | 感知 API、程序变量 | 含分支／循环的策略程序 | 是 | 可编写反馈控制；非默认完整前后验系统 | 可由程序表达，非统一恢复器 | 可靠感知／控制 API、指令可执行 | 本文采用受限子目标语言与固定解释器 |
| [ReAct](https://arxiv.org/abs/2210.03629) | ALFWorld、问答、网页任务 | 文本思考—行动—观测历史 | 单步行动 | 否 | 依赖环境反馈；无几何谓词核验器 | 在线修订行动 | 文本接口与可观察反馈 | 没有本文三维记忆和局部几何执行层 |
| [Reflexion](https://papers.nips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf) | ALFWorld、推理、编程 | 跨试次语言反思记忆 | 轨迹／试次 | 依任务而异 | 有显式 Evaluator，形式依任务而异 | 反思后重新尝试 | 可获得评价，允许多次尝试 | 跨试次反思不同于当前技能内兄弟候选切换 |
| [Inner Monologue](https://arxiv.org/abs/2207.05608) | 仿真与真实操作 | 场景描述、成功反馈、子目标完成清单 | 技能级闭环规划 | 否 | **技能成功检测；部分实验逐项检查目标关系** | 根据反馈重规划／重试 | 技能库；部分设置使用状态或人工反馈 | 本文侧重实例、嵌套拓扑及终止调用硬门控 |
| [Voyager](https://arxiv.org/abs/2305.16291) | Minecraft | 可执行技能库、状态、错误历史 | 任务级代码技能 | 是 | GPT-4 critic 检查任务完成 | 反馈修代码；有限重试后可换任务 | Minecraft 高层 API、结构化状态 | 已有验证后推进；本文使用确定性记忆谓词 |
| [LLM+P](https://arxiv.org/abs/2304.11477) | 符号规划 | PDDL 初始状态、谓词、目标 | 完整符号计划 | 符号计划，非视觉控制程序 | 模型内前置条件、效果与目标满足 | 原框架非感知失败恢复重点 | 已知且正确的领域模型 | 本文难点是噪声感知与实际执行，而非目标合取 |
| [ProgPrompt](https://arxiv.org/abs/2209.11302) | VirtualHome、桌面操作 | 对象列表、环境状态图 | 带断言与恢复分支的程序 | 是 | **运行时前置断言**；真实桌面实验未实现同等断言闭环 | 断言失败执行恢复动作 | 有限动作 API；语义状态可检查 | 最接近 guarded program；本文需突出三维接地与局部确定性恢复 |
| [CAPE](https://arxiv.org/abs/2211.09935) | 抽象具身任务规划 | 前置命题、错误信息、行动历史 | 动作及纠正步骤 | 主要为动作序列 | 显式前置错误；非视觉物理后验 | 重新提示 LLM 生成纠正动作 | 已知前提、可提供错误反馈；抽象低层控制 | 本文处理掩码／姿态等几何问题，且部分修复不调用 LLM |
| [DoReMi](https://arxiv.org/abs/2307.00329) | 机械臂、人形机器人 | 当前视觉与 LLM 生成约束 | 技能＋持续监测 | 非代码策略主体 | **执行中约束持续核验** | 违约中止技能并重规划 | 既有技能、可靠性有限的 VLM 检测 | 本文关注本地候选替代和最终目标检查 |

| 方法／基准 | 任务 | 记忆／状态表示 | 规划粒度 | 程序级执行 | 前／后置条件核验 | 失败回退 | 主要假设 | 与本文差异 |
|---|---|---|---|---|---|---|---|---|
| [ConceptGraphs](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf) | 开放词汇建图、导航、操作示例 | 对象点云、视觉语义特征、三维关系图 | 图查询与任务计划 | 非核心贡献 | 部分任务有目标核验，非完整家务契约 | 有重新定位／搜索示例 | posed RGB-D 与基础视觉模型 | 不能称其只支持静态场景；本文扩展加工状态与执行门控 |
| [RoboEXP](https://arxiv.org/html/2402.15487v2) | 交互探索、揭露隐藏物体、操作 | 对象—动作混合图、空间关系与动作依赖 | 图节点和依赖动作 | 图式技能执行 | **Action verifier；显式添加前置动作** | 移障、开容器、重新观察与调整 | 机器人技能库、RGB-D 感知、LMM | 已结合增量记忆和前提补足；本文面向指定家务目标验收 |
| [DovSG](https://arxiv.org/html/2410.11989v6) | 长期语言引导移动操作 | 动态对象图、on／inside 等关系、局部几何更新 | 任务计划＋导航抓取 | 技能接口 | 到达后重新观察与对准；无同类完整目标门控证据 | 抓取候选及备用几何策略 | 初始扫描地图、标定与机器人能力 | 本文初始仅局部可见，并加入加工状态与目标谓词 |
| [EPoG](https://arxiv.org/html/2602.04419v2) | 探索与连续物体运输 | belief graph、goal graph、图编辑 | 全局依赖计划＋局部修复 | 图编辑／动作计划 | 检查不可达、遮挡、碰撞、不稳定等情况 | LLM 插入纠正动作后重试 | 进入房间后对象关系基本可观测；感知实现被抽象 | 本文若使用实际噪声视觉管线，其感知接地条件更具体 |
| [EMem](https://arxiv.org/html/2609.28236v1) | 长程交互、动态状态与经验记忆 | spatial、event、scene 三类记忆 | 记忆检索驱动行动序列 | 未以独立受保护 DSL 为核心 | 记录成功／失败和状态冲突；未核实同类提交门控 | 历史反馈支持纠正 | 多模态记忆管理与环境反馈 | 状态账本已有近邻；本文提供度量几何和确定性解释器 |
| [ALFRED](https://arxiv.org/abs/1912.01734) | 七类家务任务 | 仿真状态、PDDL 任务定义 | 基准而非单一规划器 | 不适用 | 官方评测目标谓词 | 不适用 | AI2-THOR 离散动作与掩码交互 | 提供评价标准，不证明智能体自己的核验器可靠 |
| [Habitat 2.0](https://arxiv.org/abs/2106.14405) | 长程移动操作、家庭重排 | 物理场景与任务状态 | 层级技能等基线 | 依基线 | 任务评测与技能接口 | 依基线 | 不同机器人、物理仿真及控制条件 | 支持技能衔接问题背景，不能直接横比 ALFRED SR |
| [BEHAVIOR](https://proceedings.mlr.press/v164/srivastava22a.html) | 日常家庭活动 | BDDL 对象状态与关系谓词 | 活动／技能 | 依基线 | 显式初始和目标条件 | 依基线 | 仿真物理状态及活动定义 | 状态谓词和目标合取已有；本文区别在运行时自检与补救 |

**对 3.3 各创新点的 novelty verdict 如下。** 这里的“新颖性较弱”指不能单独支撑通用方法首创，不意味着没有工程价值或硕士论文贡献。

| 本文主张／组成部分 | 最接近的 2–3 篇证据 | Novelty verdict |
|---|---|---|
| **在线三维实例融合、多视角类别稳定度** | HLSM [3]、ConceptGraphs [19]、DovSG [21] | **基础构件已有，新颖性较弱。** 增量几何—语义融合已有明确谱系；具体类别投票与 6 cm 关联阈值属于实现选择，需要证明其针对操作场景的独立效果。 |
| **容器拓扑＋操作状态账本** | HELPER [6]、CAPEAM [7]、EMem [23] | **有较强直接先例。** 可保留的贡献在于：同一实例经过拿取、加工、搬运后，其身份、状态与父子关系如何一致更新，而不是“引入状态记忆”本身。 |
| **手持物自遮挡过滤、容器减去子物体掩码** | RoboEXP [20]、DovSG [21]、CAPEAM [7] | **可能构成局部几何改进，但证据不足。** 这些论文提供动态记忆、遮挡与交互掩码相关机制；本次未核实完全相同的掩码差集公式，不能据此认定首创。需直接比较放置失败率。 |
| **细粒度占据＋粗粒度方向覆盖，配合 FMM** | FILM [4]、HLSM [3] | **主要为已有导航技术的具体组合。** FILM 已有 5 cm 地图与 FMM，HLSM 已区分空间语义和可观测性。方向覆盖与可通行性分离可以作为设计特点，尚不足以称新的导航算法。 |
| **一次提交受保护子目标程序** | Code as Policies [10]、ProgPrompt [16]、HELPER [6] | **程序化执行本身不新。** 本文可突出有限 DSL、解释器语义、预算控制及恢复边界；需说明它相对普通宏工具新增了什么控制结构。 |
| **前置条件自动满足，如开门、找刀** | ProgPrompt [16]、HELPER [6]、RoboEXP [20] | **直接已有。** HELPER 已展示切片前找刀，RoboEXP 有添加前置动作的算法。本文差异可能是基于在线实例记忆的本地确定性补齐。 |
| **导航后多俯仰角对准** | HLSM [3]、HELPER [6]、DovSG [21] | **工程性恢复策略，新颖性较弱。** 交互位姿搜索、换视角纠错、到达后对准均有先例；固定角度搜索应以消融和动作成本证明价值。 |
| **同类兄弟候选自动回退** | CAPE [17]、PRED [8]、EPoG [22] | **具体实现可能有区分度。** 局部替换对象、修订目标和异常恢复已有；应证明本文保留了任务约束和实例状态，且减少高层调用。简单候选重试不宜称为完整的 contingency-aware planning。 |
| **提交前目标谓词核验、拒绝过早结束** | Inner Monologue [13]、Voyager [14]、LLM+P [15] | **完成检查和补救思想已有。** 较具体的贡献是系统层拦截 `finish_episode`，检查独立实例计数、加工—放置对象一致性及嵌套容器关系。式（35）的合取本身没有算法新颖性。 |
| **上述组件组成完整执行闭环** | HELPER [6]、RoboEXP [20]、EPoG [22] | **可作为有条件成立的系统方法贡献。** 目前最需要证明的是组件之间共享状态带来的效果，以及本地修复与终止核验各自的因果贡献。 |

其中三处先例尤其影响创新表述：

- **HELPER 不仅有语言—程序检索记忆，也有在线三维对象记忆。** 其执行前检查和自动调整计划，与本文的前置条件自动满足直接重叠。[HELPER 正文与附录](https://aclanthology.org/2023.findings-emnlp.226.pdf)
- **Inner Monologue 已检查期望子目标是否满足，Voyager 已用 critic 验证完成后再推进。** 因此，“完成前核验→失败补救”不能作为没有先例的思想。[Inner Monologue 附录 A.1](https://arxiv.org/html/2207.05608v1)、[Voyager §2.3](https://arxiv.org/html/2305.16291v2)
- **RoboEXP 已将对象、动作与前置依赖放入同一场景图。** 这比仅引用静态语义地图更接近本文“记忆支持执行”的主线。[RoboEXP 算法与正文](https://arxiv.org/html/2402.15487v2)

**最值得优先补入相关工作讨论的五篇**，按与本文机制的直接重叠程度选择如下：

1. **HELPER**：同时覆盖三维对象记忆、状态更新、程序执行、前置检查和失败纠正，是必须正面对比的近邻。
2. **ProgPrompt**：直接覆盖带断言和恢复分支的程序，是“受保护子目标程序”最重要的程序范式先例。
3. **CAPEAM**：在同一 ALFRED 基准中维护物体迁移、状态和掩码，直接影响“双物体去重与动态记忆”的创新定位。
4. **RoboEXP**：覆盖隐藏物体探索、容器操作、增量三维记忆以及前置动作补足。
5. **PRED**：在 ALFRED／TEACh 中根据存在、属性、外观和关系差异提前修订动作，避免把已有环境反馈纠错描述为本文独有。

另外，正文若以 2026 年为文献截止时间，已经列入上表的 **EPoG 与 EMem** 也应讨论；EMem 必须标注为预印本。

**以下五条可作为论文“与现有工作区别”的写作基础。** 它们描述方法结构，不预先宣称性能优势。

> 1. 相较于 HLSM 和 FILM 以语义空间地图支持子目标执行的设计，本文进一步围绕可持续识别的对象实例，联合维护容器包含关系与操作状态，使目标获取、状态加工和最终放置能够依据同一实例的历史信息进行协调。

> 2. 相较于 Code as Policies 和 ProgPrompt 的程序生成范式，本文采用面向家务任务的声明式子目标接口，将导航接近、交互对准、前置条件补足和候选替换封装于受限解释器中，以明确高层任务决策与局部执行修复之间的职责边界。

> 3. 相较于 CAPE 和 LLM-Planner 将执行错误反馈给大模型以生成修订计划的方式，本文针对预定义的常见异常，在子目标执行期间利用场景记忆完成局部补救，仅在局部恢复失败或需要改变任务策略时返回高层决策。

> 4. 相较于采用自然语言完成判断的策略，本文在任务终止接口设置基于实例状态和容器拓扑的谓词检查，重点约束双物体任务的实例独立性、加工对象与放置对象的一致性，以及移动容器任务的嵌套关系。

> 5. 本文将研究重点放在初始局部可见条件下的在线感知与执行协同，考察增量三维记忆如何同时服务于对象选择、交互准备、失败恢复和终止判断；其贡献定位于这些环节的联合设计及受限预算下的实验验证。

第 4 条最好随后明确：这些谓词根据智能体自身的记忆求值，**并不天然构成真实环境成功的形式化保证**。

**现稿最需要收紧或补证据的 claim 有以下几类。**

**首先，不能把内部账本核验直接称为物理后置条件核验。** 如果执行过 `heat_held` 就写入 `heated`，随后式（35）读取同一标志，那么它检查的是内部记录是否一致。需要说明每种状态来自动作尝试、API 成功返回、视觉证据还是独立状态检测；还需要处理开门后关门、物体转移后旧父节点失效等更新。稿件已经提到“容器状态虚标”，这正是核验可靠性的关键限制。

实例身份同样重要：仅用 6 cm 三维包络距离，不能充分解释物体被抓起并搬到远处后如何继承原 ID。若身份发生分裂，两个记忆节点可能对应同一个物体，影响双物体计数；若身份错误合并，已加工状态可能被赋给其他实例。

**其次，“有效可放置表面”的名称强于公式（32）实际保证。** 减去子物体掩码只能得到排除部分遮挡物后的候选像素，不能直接证明其为水平支撑面、具有空余体积、无碰撞且可达。较稳妥的名称是“容器交互候选掩码”；若坚持“有效可放置表面”，应补充三维几何条件或实证定义。

还应澄清版本对应关系：Markdown 版方法部分已把子物体掩码裁剪写为现有机制，实验诊断却仍把“未裁剪干扰”列为当前瓶颈。需要说明该机制是否已包含在产生 v17 数字的执行版本中。

**第三，现有结果支持“Unseen 改善、总体小幅提高”，不支持整体大幅提升。**

根据原稿给出的数量和比例，可得到：

| 划分／指标 | 基线 | 本文 | 变化 |
|---|---:|---:|---:|
| Overall SR | 402/1641，24.50% | 407/1641，24.80% | **净增 5 个；约 +0.30 个百分点** |
| Seen SR | 249/820，30.37% | 224/820，27.32% | **减少 25 个；约 −3.05 个百分点** |
| Unseen SR | 153/821，18.64% | 183/821，22.29% | **增加 30 个；约 +3.65 个百分点** |
| 加热任务 Overall SR | 17.28% | 12.35% | −4.93 个百分点 |
| 冷却任务 Overall SR | 13.62% | 8.09% | −5.53 个百分点 |

这些是对文稿数字的核算，不是对运行结果的独立复核。XML 版删去了 Seen 总体行和退化任务，容易使读者高估整体收益。建议恢复完整结果，并在没有配对统计检验前避免使用统计意义上的“显著提升”。

**第四，决策耗尽率降低不直接证明执行效率提高。** 从原子宏工具改为较长子目标程序后，一次决策能够执行更多动作；在统一 35 轮上限下，预算容量本身已经改变。因此 32.42%→3.41% 可以支持“较少耗尽高层决策轮”，但不能单独证明更少物理动作、更低成本或更强规划能力。

至少需要分别报告：LLM 调用数与 token、低层动作数、失败动作数、总耗时，以及恢复成功率。消融应包括“仅程序”“仅终止核验”“二者结合”，再检查前置补足、视角恢复与候选替换。只比较 v16 和联合升级的 v17，无法把双物体提升归因于核验器。

**第五，与 HLSM／FILM／MOCA 等已发表方法的可比性有明确边界。**

| 边界 | 当前需要说明的内容 |
|---|---|
| **评测单位** | ALFRED 原论文的 820／821 是语言标注数量；应写“1,641 个语言指令评测 episode”，不能等同于 1,641 条独立专家轨迹。[官方数据说明](https://raw.githubusercontent.com/askforalfred/alfred/master/data/README.md) |
| **动作预算** | 官方评测入口默认 `max_steps=1000`、`max_fails=10`。本文 250 动作步、35 决策轮是自定义预算；可说采用官方指标，不能据此说完整协议完全一致。[官方入口](https://raw.githubusercontent.com/askforalfred/alfred/master/models/eval/eval_seq2seq.py) |
| **深度与状态来源** | HLSM、FILM 的标准管线使用预测深度；本文必须说明 RGB-D 中的 D 是否直接来自模拟器。位姿、持物状态、错误字符串、对象属性和目标参数也应逐项声明来源。HLSM 本身使用位姿与持物观测，不能反向夸大其限制。[HLSM](https://proceedings.mlr.press/v164/blukis22a/blukis22a.pdf)、[FILM](https://arxiv.org/abs/2110.07342) |
| **指令条件** | goal-only 与包含逐步语言指导的设置不同；训练数据、提示示例数、感知训练和低层控制器也不同。LLM-Planner 的不同设置不能混用。[LLM-Planner](https://arxiv.org/abs/2212.04088) |
| **验证集与测试集** | 本文 valid seen／unseen 不能直接和 test leaderboard 排名。当前实验表只证明与自己的基线相比发生了哪些变化。 |
| **核验器与评测器** | 内部预测状态可以用于自检；若决策器访问官方成功函数、隐藏目标参数或真值对象状态，则改变信息条件。现稿不足以判断是否发生，不能预先认定有泄漏。 |
| **指标实现** | `1503/4229=35.54%` 的 GC 汇总形式符合官方 micro 聚合；不应误改为逐任务 GC 的简单平均。PLW 还包含专家动作长度权重，应复用或逐项核对官方实现。[官方指标代码](https://raw.githubusercontent.com/askforalfred/alfred/master/models/eval/eval_task.py) |
| **泛化范围** | Unseen 表示 ALFRED 训练划分未包含的场景；不证明基础模型预训练从未接触相关数据，也不证明能推广到 BEHAVIOR 或真实长程操作。 |

另外，ReAct／Reflexion 的 **ALFWorld 文本环境结果不是 ALFRED 视觉执行结果**；HELPER 2023 原论文主要评测 TEACh，不能把后续扩展系统的 ALFRED 数字直接归给该论文。

论文目前最值得补强的证据，是一张终止核验统计表：**拦截多少错误终止、误拒绝多少已成功任务、拦截后多少任务真正恢复成功，以及为此新增多少动作和模型调用。** 配合共享实例记忆与各恢复组件的独立消融，才能把“完整系统实现”推进为清晰、可验证的方法贡献。