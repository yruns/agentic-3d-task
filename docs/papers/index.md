# 相关工作调研（2026-11-10）

本目录记录硕士学位论文《基于多模态室内场景表征的大模型具身推理与执行方法研究》三个研究点的相关工作交叉调研。调研用于回答两个问题：

1. 本文各“创新点”中，哪些已被现有工作覆盖，哪些只是增量改进，哪些可以作为可辩护的新颖点；
2. 论文初稿中哪些表述、指标或实验归因需要修正，避免答辩时被直接质疑。

## 调研方式

- **codex astra**：三个独立任务，分别调研研究点一、二、三；检索 2020–2026 年论文，核对 arXiv、会议页面与项目主页。
- **Claude 子代理**：三个独立任务，与研究点一一对应；重点做 novelty verdict、遗漏检查和“与现有工作区别”表述。
- **本地综述**：`docs/research/` 下的三份室内三维/长程任务综述，用于交叉核对 2026 年新工作与基准协议。
- **基准记录**：`docs/benchmark/` 下的 NR3D、OpenEQA、SceneFunc3D、ALFRED 运行记录，用于核对论文数字的来源与口径。

## 文件索引

- [研究点一：三维场景表征与关键帧检索](3_1_scene_representation_retrieval.md)
- [研究点二：室内多任务推理](3_2_multi_task_reasoning.md)
- [研究点三：增量场景记忆与具身执行](3_3_embodied_execution.md)
- [创新边界、遗漏文献与高风险表述](innovation_boundary_and_risks.md)
- 原始调研记录：
  - [codex astra：研究点一](raw/codex_astra_3_1.md)
  - [codex astra：研究点二](raw/codex_astra_3_2.md)
  - [codex astra：研究点三](raw/codex_astra_3_3.md)

## 总体结论

三个研究点都存在“单项组件已有、组合与系统设计可辩护”的情况。论文不宜把 BEV、对象—视图可见性、虚拟观察者、覆盖式选帧、Map-First、原图裁剪、SAM2、程序化执行、前置条件、完成核验等分别称为首创。可辩护的创新性应落在三个“联合”上：

1. **对象—区域双通道 + 预算约束联合覆盖 + 几何约束求解**的统一检索框架；
2. **共享表征 + 统一证据获取 + 任务特化几何核验**的三任务协同推理；
3. **增量记忆 + 程序级受保护执行 + 提交前谓词核验**的长程执行闭环。

## 最接近的前作

| 研究点 | 最接近的前作 | 关系 |
|---|---|---|
| 3.1 | 3D-Mem、MemTree3D、Opti3D、GraphEQA、ZSVG3D、OpenMask3D、CSES、SCONE、Thorne et al.、IVSGround | 覆盖式记忆/选帧、对象—视图可见性、虚拟观察者、次模选帧、视角选择 |
| 3.2 | ZSVG3D、GraphPad、MemTree3D、Fun3DU、UniFunc3D、AFFORDMEM、OriGround、SeeGround、VLM-Grounder | 观察者参考系、场景图/记忆树取证、二维种子+SAM、可见性归一化投票、多视角定位 |
| 3.3 | HELPER、ProgPrompt、CAPEAM、PRED、RoboEXP、AtomTree、ReAcTree、Think Twice Act Once、HLSM、FILM | 程序+记忆+前置条件、状态更新、失败回退、验证器、ALFRED 模块化执行 |

## 使用说明

- 本目录是**调研记录**，不是论文正文；论文正文使用 `3.5 与现有工作的比较` 与 `3.7 本节补充参考文献`。
- 表中“与本文关系”只描述机制差异，不预设性能优劣；不同论文的训练设置、动作预算、感知条件和评测协议不同，不能直接横向排名。
- 标注“预印本”或“未核实正式出处”的条目，引用前需再次核对；不要把这些标注带进正式论文。
