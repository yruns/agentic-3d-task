# 研究点三：增量场景记忆与具身执行

> 对应论文 3.3 节。调研截至 2026-11-10。表中“与本文关系”只描述机制差异，不代表性能优劣。

## 最接近的 5 篇

1. **AtomTree**（arXiv:2610.38744, 2026）：状态感知的子目标/动作树 + 符号状态与历史记忆 + 层次回溯；零样本 ALFRED Valid Unseen 42.39%。与本文“状态账本 + 兄弟回退”最接近。
2. **ReAcTree**（AAMAS 2026，arXiv:2511.02424）：子目标 agent 树 + 控制流节点 + 情节/工作记忆；ALFRED Valid Unseen 25.09%–39.83%。与“受保护子目标程序 + 回退”同构。
3. **HELPER**（Findings EMNLP 2023，arXiv:2310.15127）：语言—程序记忆 + 在线三维对象及属性状态 + 执行前检查与失败纠正。与本文“记忆 + 程序 + 检查”最接近。
4. **ProgPrompt**（ICRA 2023，arXiv:2209.11302）：带断言与恢复分支的程序；是“受保护子目标程序”的程序范式先例。
5. **CAPEAM / PRED / RoboEXP**：ALFRED 上的对象状态记忆、执行前环境差异检查、动作条件场景图与前置动作补足，分别覆盖本文“状态账本”“提前修订”“前置条件自动满足”。

## 文献清单

| 方法 | 标题 | 作者/出处 | arXiv/DOI | 核心思想 | 与本文关系 |
|---|---|---|---|---|---|
| ALFRED | A Benchmark for Interpreting Grounded Instructions for Everyday Tasks | Shridhar et al., CVPR 2020 | 1912.01734 | 长程家务指令、第一人称视觉、导航与交互 | 评测基准；不证明智能体核验器可靠 |
| MOCA | Factorizing Perception and Policy for Interactive Instruction Following | Singh et al., ICCV 2021 | 2012.03208 | 感知与策略分解，循环隐状态与二维实例关联 | 原子动作 + 受阻检测；无完整显式契约 |
| HLSM | A Persistent Spatial Semantic Representation for High-level Instruction Execution | Blukis et al., CoRL 2021 | 2107.05612 | 持久三维语义体素 + 高层子目标 + 低层控制 | 增量空间记忆最直接先行工作 |
| FILM | Following Instructions in Language with Modular Methods | Min et al., ICLR 2022 | 2110.07342 | 语义地图 + 语义搜索 + 模板化子任务 | 5 cm 栅格与 FMM 已有；本文差异在状态与程序保护 |
| LLM-Planner | Few-Shot Grounded Planning for Embodied Agents | Song et al., ICCV 2023 | 2212.04088 | LLM 高层计划 + 已完成步骤与对象反馈 | 子目标序列；失败/超时触发重规划 |
| HELPER | Open-Ended Instructable Embodied Agents with Memory-Augmented LLMs | Sarch et al., Findings EMNLP 2023 | 2310.15127 | 语言—程序记忆 + 在线三维对象/属性状态 + 执行前检查 | 强近邻；需证明本地候选回退与终止谓词的增量作用 |
| CAPEAM | Context-Aware Planning and Environment-Aware Memory | Kim et al., ICCV 2023 | 2308.07241 | 语义空间地图 + 对象迁移 + 掩码缓存 | 已处理“搬过哪个对象”；本文须细化状态与拓扑语义 |
| PRED | Pre-emptive Action Revision by Environmental Feedback | Kim et al., CoRL 2024 | — | 对象存在/外观/属性/关系差异检查 + 计划修订 | 执行前环境差异检查近邻 |
| SayCan | Do As I Can, Not As I Say | Ichter et al., CoRL 2022 | 2204.01691 | 技能有用性 + 可执行性评分 | 执行前软可行性；非显式谓词修复 |
| Code as Policies | Language Model Programs for Embodied Control | Liang et al., ICRA 2023 | 2209.07753 | 感知调用、几何运算、分支循环表达为策略程序 | 程序级执行先例；非统一恢复器 |
| ReAct | Synergizing Reasoning and Acting in Language Models | Yao et al., ICLR 2023 | 2210.03629 | 思考—行动—观测循环 | 单步行动；无三维记忆与几何核验 |
| Reflexion | Language Agents with Verbal Reinforcement Learning | Shinn et al., NeurIPS 2023 | 2303.11366 | 跨试次语言反思记忆 | 跨试次反思；非技能内兄弟候选切换 |
| Inner Monologue | Embodied Reasoning through Planning with Language Models | Huang et al., CoRL 2022 | 2207.05608 | 场景描述、成功反馈、子目标完成清单 | 已有成功检测与部分目标关系检查 |
| Voyager | An Open-Ended Embodied Agent with Large Language Models | Wang et al., TMLR 2024 | 2305.16291 | 可执行技能库 + 状态 + 错误历史 + critic | 已有验证后推进；本文使用确定性记忆谓词 |
| LLM+P | Empowering LLMs with Optimal Planning Proficiency | Liu et al., arXiv 2023 | 2304.11477 | LLM 生成 PDDL，符号规划器求解 | 模型内前置条件/效果；非噪声感知 |
| ProgPrompt | Generating Situated Robot Task Plans using LLMs | Singh et al., ICRA 2023 | 2209.11302 | 带断言与恢复分支的程序 | 最接近 guarded program；需突出三维接地 |
| CAPE | Corrective Actions from Precondition Errors using LLMs | Raman et al., ICRA 2024 | 2211.09935 | 显式前置错误 + 纠正动作 | 本文处理掩码/姿态等几何问题，部分修复不调用 LLM |
| DoReMi | Detecting and Recovering from Plan-Execution Misalignment | Guo et al., IROS 2024 | 2307.00329 | 技能 + 持续监测，违约中止并重规划 | 执行中约束核验近邻 |
| ConceptGraphs | Open-Vocabulary 3D Scene Graphs for Perception and Planning | Gu et al., ICRA 2024 | 2309.16650 | 对象点云、特征、三维关系图 | 对象拓扑记忆先例；非加工状态/终止门控 |
| RoboEXP | Action-Conditioned Scene Graph via Interactive Exploration | Jiang et al., CoRL 2024 | 2402.15487 | 对象—动作混合图 + 前置动作补足 + action verifier | 增量记忆 + 前置条件近邻 |
| DovSG | Dynamic Open-Vocabulary 3D Scene Graphs for Long-term Mobile Manipulation | Yan et al., RA-L 2025 | 2410.11989 | 动态对象图 + on/inside 关系 + 局部几何更新 | 长期记忆近邻；无终止门控证据 |
| EPoG | Integrated Exploration and Sequential Manipulation with LLM Replanning | Yang et al., ICRA 2026 | 2602.04419 | belief graph + goal graph + 图编辑 + 局部修复 | 图式记忆与重规划近邻 |
| EMem | EmbodiedMemory-Bench: Benchmarking Embodied Memory | Liang et al., arXiv 2026 | 2609.28236 | spatial/event/scene 三类记忆 | 状态账本近邻；未核实提交门控 |
| AtomTree | State-Aware Embodied Instruction Following with LLMs | Zhai et al., arXiv 2026 | 2610.38744 | 状态感知子目标/动作树 + 符号状态 + 层次回溯 | 与本文“状态账本 + 兄弟回退”最接近 |
| ReAcTree | Hierarchical LLM Agent Trees with Control Flow | Choi et al., AAMAS 2026 | 2511.02424 | 子目标 agent 树 + 控制流 + 情节/工作记忆 | 与“受保护子目标程序 + 回退”同构 |
| Think Twice, Act Once | Verifier-Guided Action Selection for Embodied Agents | Singhi et al., CVPR 2026 Findings | 2605.12620 | 动作候选采样 + 生成式验证器选择 | 动作级验证；本文是终止级符号谓词门控 |
| Habitat 2.0 | Training Home Assistants to Rearrange their Habitat | Szot et al., NeurIPS 2021 | 2106.14405 | 长程移动操作与家庭重排仿真 | 长程基准；不能直接横比 ALFRED SR |
| BEHAVIOR | Benchmark for Everyday Household Activities | Srivastava et al., CoRL 2021 | 2108.03332 | BDDL 对象状态与关系谓词 | 状态谓词/目标合取先例；不同仿真与协议 |

## Novelty verdict

| 本文候选创新 | 判定 | 最接近证据 | 可保留的贡献边界 |
|---|---|---|---|
| 增量记忆：对象拓扑 + 状态账本 + 占据栅格 + 观测覆盖 | 增量改进 | HLSM、FILM、ConceptGraphs、HOV-SG、HELPER、CAPEAM、EMem | 同一记忆同时服务候选排序、前置条件与终止谓词求值（memory-as-contract） |
| 受保护子目标程序 + 兄弟候选回退 | 增量改进 | Code as Policies、ProgPrompt、PDDL/PDDLStream、behavior trees、AtomTree、ReAcTree | 类型约束子目标 DSL + 确定性解释器 + 感知驱动回退 |
| 提交前目标谓词核验器拒绝终止 | 增量改进 | Inner Monologue、Reflexion、Think Twice、KnowNo、ALFRED 评测 | 终止级、符号化、记忆接地、拒绝 finish_episode 并返回未满足谓词诊断 |
| 程序级执行降低调用开销/过早终止 | 调用开销降低已有；终止减少证据不足 | Code as Policies、ProgPrompt、Voyager、LLM+P | 需补每任务 LLM 调用数、token、墙钟、环境步数与 success-per-cost |
| 容器有效可放置表面掩码（容器减子物体） | 微创新，证据不足 | RoboEXP、DovSG、CAPEAM | 可能构成局部几何改进；需比较放置失败率 |
| 6 cm 实例关联 / 多视角类别稳定度 | 实现选择 | ConceptGraphs、DovSG、HLSM | 需解释抓取搬移后的身份继承 |
| 细粒度占据 + 粗粒度方向覆盖 + FMM | 已有导航技术组合 | FILM、HLSM | 设计特点，不是新导航算法 |

## 可直接写入论文的区别表述

1. 与 HLSM、FILM 等以持久语义地图支撑目标定位与探索的模块化方法相比，本文将增量对象记忆同时作为子目标候选排序、前置条件满足与终止谓词求值的统一状态源；已有 ALFRED 智能体多将地图用于导航/定位，而较少将容器—内容物拓扑、操作状态账本与占据/覆盖证据统一暴露给执行与核验两个阶段。
2. 与 Code as Policies、ProgPrompt 等让大模型生成开放式代码的策略相比，本文不让模型直接生成任意控制流，而是提交受类型约束的子目标程序，由确定性解释器完成候选排序、自动开门、多俯仰角对准与兄弟候选回退；该设计把开放世界的不确定性限制在感知与候选选择层，使执行语义保持可检查、可复现。
3. 与 Reflexion、Inner Monologue 等基于事后反思或学习式成功检测的闭环方法相比，本文的提交前核验器在 finish_episode 提交点对任务目标约束集合进行符号化谓词求值，并在拒绝终止时返回未满足谓词诊断；其作用对象是终止决策而非单步动作，因而与动作级验证器（如 Think Twice, Act Once）互补。

## 高风险表述

- **内部账本一致性 ≠ 真实物理后置条件核验。** 如果执行 `heat_held` 就写入 `heated`，式（35）检查的是内部记录是否一致；需说明每种状态来自动作尝试、API 成功、视觉证据还是独立检测。
- **“有效可放置表面”强于式（32）实际保证。** 减去子物体掩码只得到候选像素，不能证明水平支撑、空余体积、无碰撞与可达；建议改称“容器交互候选掩码”。
- **6 cm 关联不能充分解释抓取搬移后的身份继承。** 身份分裂或错误合并会影响双物体计数与加工状态归属。
- **决策耗尽率降低 ≠ 执行效率提高。** 从原子宏工具改为较长子目标程序后，每轮决策承载更多环境步；需分别报告 LLM 调用数、token、低层动作数、失败动作数、总耗时与恢复成功率。
- **ALFRED 1,641 是语言指令评测 episode，不是独立专家轨迹。** 250 步动作预算与 35 轮决策上限是自定义设置，不能直接对应官方 1000 步/10 次失败协议或公开榜单排名。
- **Seen SR 由 30.37% 降至 27.32%，加热/冷却任务下降。** 初版可以淡化，但定稿前应恢复完整表并解释；不能声称整体大幅提升。
- **AtomTree / ReAcTree 报告了更高的 Unseen 成功率（42.39% / 25.09%–39.83%）。** 其训练/示例设置、动作预算与感知条件不同，不能直接横比；论文不宜声称 SOTA。
- **HELPER 已同时具备在线三维对象记忆、程序执行、执行前检查与失败纠正。** 本文不能把“程序 + 记忆 + 检查”组合本身称为首创。
