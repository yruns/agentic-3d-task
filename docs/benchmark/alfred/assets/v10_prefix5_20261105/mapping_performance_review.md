当前映射实现存在可直接从源码确认的重复计算：同一关联阶段内，不变的对象点云会按候选对重复拟合 OBB、运行 ConvexHull/PCA；便宜的 AABB 距离检查和 KD-tree 也没有在候选阶段复用。这是下一轮优先检查的 CPU 风险，但现有记录没有 mapper 分项计时，不能据此断言它占用了多少 CPU 秒，或承诺某个整体加速倍数。当前冻结的 pilot15 不作任何优化或参数调整。

本次只读源码与已完成的 `ALFRED_V9_20261105_pilot15_i02` 输出；未附加 profiler、重放 mapper、调用模型/GPU/模拟器或修改 tracked 文件。检查时 HEAD 为 `62e9f18a6654bbedb1558b3ae2a08cc7f2207a6f`，tracked tree clean；i02 的 recorded executed-code SHA 为 `547bf610324ba30dcdc76e4e4cfe9ccf8d01d593df8f84c4f0dddaaf3d0107aa`。

**证据口径**

- Root 提供的既有 psutil 观察：五个 parent 进程在约 1200 秒 elapsed 时，分别累计约 269–756 CPU 秒。这相当于各自平均约 0.22–0.63 个 CPU 核的累计用量，不是峰值利用率。本次未重新采样该数据，也没有原始 psutil 快照可用于逐进程归因。CPU 秒可能累计进程内多个线程，不能直接与单段 wall time 相减。
- Root 提供的“单帧推理约 5.8 秒”属于先前观察，不能自动当作 CUDA kernel 时间。当前代码使用 wall-clock timer；它包含 CPU 前后处理、模型搬运等。不同样本、并发等待和统计阶段之间不可直接对比。
- i02 的完成记录可直接核对：34 个观测、38 个决策、33 个动作、10 次失败，`stop_reason=failure_budget`，官方 GC=0/3。10 次失败均为 `OpenObject`；这不是 mapper 性能导致任务失败的因果证据。

| i02 已保存计时 | 总计 | 口径 |
|---|---:|---|
| run 开始至结束 | 1947.107 s | 包括启动、动作、模型等待、建图、存盘和评分等 |
| execution.policy_seconds | 1526.565 s | `runtime.execute(task)` 的 wall time，约总时长 78.4% |
| execution.perception_seconds | 211.643 s | `perception.predict` RPC 的 wall time，34 帧平均 6.225 s，约总时长 10.9% |
| 两项计时之外的剩余 wall time | 208.898 s | 约10.7%；包含建图，也包含初始化、模拟器动作、绘图/文件IO、评分等，不能标为纯 mapper 时间 |
| worker.inference_seconds | 116.633 s | `_predict_frame` wall time，34帧平均3.430 s；不是纯 GPU 时间 |
| worker.gpu_wait_seconds | 114.387 s | GPU lock 等待，包含初始化等待和帧请求等待，不应直接全部加到逐帧预测口径 |
| worker.load_seconds | 18.487 s | 模型初始化阶段 wall time |

`execution._observe` 在累加 `perception_seconds` 后才加载 depth/features/masks 并执行 `memory.integrate`，因此该计时不包含 mapper。`_decide` 中当前图标注、BEV 绘制、snapshot 和 prompt 准备发生在 `policy_seconds` 的起点之前。不能用 `run_elapsed - policy - perception` 作为 mapper 的分项测量。执行计时边界 (`src/codex_agent/alfred/execution.py:160`)、模型阶段与锁计时 (`src/codex_agent/alfred/perception_worker.py:80`)

**从 i02 日志可以推导的工作规模**

最终记忆包含167个节点，其中60个 provisional；167个节点的最终几何均有效，aliases 为空。这里的节点数不是经过真值验证的真实物体数，也不足以证明这些节点都是重复或错误实例。34帧的原始检测数 min/median/max 为1/18/45，映射后的不同当前实例数为1/11/33。

在本例中没有成功的物体交互或切片，也没有合并别名。因此，用 `observations.jsonl` 中逐帧已分配ID及冻结控制流，可重建以下关联循环规模。它们是日志与代码推导，不是采样 profiler 结果：

| 帧 | 原始检测 | 当前映射实例 D | 本帧之前节点 N | D×N 候选配对 |
|---:|---:|---:|---:|---:|
| 0 | 13 | 8 | 0 | 0 |
| 9 | 41 | 33 | 48 | 1584 |
| 19 | 12 | 10 | 96 | 960 |
| 29 | 44 | 33 | 143 | 4719 |
| 33 | 44 | 33 | 167 | 5511 |

- 全34帧累计56,155次 `_association_score` 配对。此例中的节点均走有效静态几何分支，每对调用一次 `oriented_envelopes_overlap`，其内部重新拟合两边包围盒。因此关联阶段对应112,310次 `_oriented_envelope` 调用；不是112,310次必然成功的 ConvexHull，退化点集可能不进入 ConvexHull 分支。
- 第20个观测，即 frame19，周期维护时已有101个节点，枚举5050个对象对，又对应10,100次 envelope 拟合。34帧仅触发这一次20帧周期维护。
- 若每个关联阶段只拟合一次每个当前检测和每个旧节点，且无候选时不拟合，纯调用数可从112,310降到3668；该次维护可从10,100降到101。这只是避免重复拟合的可推导工作量，不是30倍/100倍总耗时加速预测。实际耗时还取决于点数、rank/hull分支、KD查询、CPU调度与其他步骤。

**静态实现风险**

1. `_assign_regions` 的双层循环遍历所有 D×N 对，先计算 image/text cosine，再做 OBB 门控。`oriented_envelopes_overlap` 每次重新执行有限性扫描、rank/SVD、可能的 ConvexHull、协方差/eigh，以及坐标投影；静态节点之间没有 envelope 缓存。文本 cosine 在有效静态几何分支实际不参与 sim_sum，但当前仍会提前计算。`feature_cosine` 又会复制和校验两条向量并重算范数，形成额外的 O(D×N×feature_dim) Python/数组工作。关联循环 (`src/codex_agent/alfred/memory.py:309`)、包围盒拟合 (`src/codex_agent/alfred/object_mapping.py:303`)

2. `point_overlap` 虽然有 AABB 的半径拒绝检查，但它发生在 OBB 拟合之后。通过门控的每一对都会重新 `cKDTree(existing)`；同一节点在一个阶段中被不同检测查询时，树未复用。周期合并计算两个方向的 overlap 时，同样重复构建双方的树。点重叠核 (`src/codex_agent/alfred/object_mapping.py:274`)

3. `_merge_candidates` 在 V 个有效节点间执行 V(V−1)/2 对的 OBB 检查。共视实例禁止合并的检查及 feature 阈值检查在候选几何计算之后才做，部分最终会被拒绝的对已完成昂贵工作。不过 feature 门控不能轻率提前：真正执行合并时，前序合并可能已改变某个代表节点的特征，原调度是在那时检查更新后的特征。周期合并 (`src/codex_agent/alfred/memory.py:423`)

4. DBSCAN 先通过 `query_ball_point(points, eps)` 物化所有邻接列表，再用 Python deque/set 遍历核心点邻居。主要遍历工作为 O(P+E)，E 是物化的邻接项总数；一般密集邻域可接近平方级，不能只按 O(P log P) KD-tree 估计。不过当前固定体素尺寸限制了单位空间的点密度，所以不能据此断言此运行产生4096²邻居，或随任意点数始终呈平方增长。它在每个新对象投影时运行，周期维护还处理全部有效节点。4096点限制是在 voxel/DBSCAN **之后**，因此新检测或对象合并的中间点集不受4096直接约束。这是已声明的算法顺序，不能在当前评估中前置截断“修快”。目前没有保存逐对象 DBSCAN 的 pre/post 点数或邻接项计数，无法据此确认它就是主要 CPU 热点。DBSCAN (`src/codex_agent/alfred/object_mapping.py:207`)

5. 其他未分项计时的候选成本包括：原始/有效掩码的 NPZ 压缩；完整深度投影及 `np.unique`；空间图射线证据与集合更新；BEV 绘制、snapshot、历史深度读取。需要把这些成本与对象关联分开，不能将 parent CPU 全部归因于 ConvexHull。空间图更新 (`src/codex_agent/alfred/spatial_map.py:215`)

**后续可以验证且保持点重叠语义的优化顺序**

以下只作为当前15条全部结束之后的讨论方案，不修改当前运行。

- 先把**现有** `point_overlap` 中同一条 AABB 半径拒绝表达式移到静态对象的 OBB 之前，并复用每个点集的 min/max。在这条表达式成立时，现有 kernel 本来就会返回0，当前静态关联还明确要求 overlap>0，周期合并也要求 overlap>.7，因此不会减少任何已有合格匹配。必须保留动态 held/changed reacquisition 的独立逻辑，不能把它的0.6m重新接近范围误当作0.025m静态重叠半径。
- 更强的数学下界是 `g=max(0, a_min-b_max, b_min-a_max)`，`||g||₂` 是任意跨点集距离的下界。若下界不小于严格邻域半径 r，则两个方向的点重叠均为0。实现时须采用保守浮点边界；第一步优先复用已存在的拒绝表达式，避免新增边界舍入导致的不同结果。仅仅两个原始 AABB 不相交不足以拒绝，因为它们仍可能相距小于 r。
- **优先阶段内缓存**：关联阶段的旧节点和当前检测点云在全部分配结束前不变化；周期候选构造在实际执行任何合并前完成。因此可在各阶段预计算一次 AABB、OBB envelope 和每个目标点集的 cKDTree，重复调用相同 query 参数。保持原点数组、dtype、顺序、严格半径、overlap方向和分母，可以验证结果不变。不要缓存整对 score 跨越特征或几何更新。
- 持久缓存需要独立的 `geometry_revision`，或者严格受控的数组身份/内容版本。**仅用 geometry_epoch 不够**：它用于交互引起的位置版本，而普通 fuse、voxel/denoise、periodic merge 也会改变点云。缓存失效必须覆盖所有这些更新、alias retire和动态状态变化；特征缓存还要覆盖每次特征融合。阶段内缓存可以先避开这些失效风险。
- 可以缓存归一化校验结果/范数，并仅在动态分支计算 text cosine，前提是边界仍完整验证特征、保持当前真实 cosine 算式。不要为提速再次把近单位向量直接当成单位向量点积，也不要未经边界回归就改用不同精度或批量BLAS求和顺序。

不宜称为“无损性能优化”的改动包括：提前限制输入点数、减小DBSCAN工作集、降低检测数/类别范围、按类别硬排除匹配、删除 provisional 节点、改变维护周期/merge阈值、永久跳过未更新节点的再次 voxel/denoise、替换聚类的边界/最大簇tie规则。尤其 cloud-min voxel 格原点依赖输入，不能未经证明就假设反复后处理完全幂等。这些应作为独立方法变化或消融，不混入同一冻结fold。

未来验证应先比较优化前后：静态候选是否完全相同、两个方向的 overlap 值、按顺序的最终 assignments/aliases、当前有效mask索引、几何及feature版本、0.025m与0.8阈值边界，然后再比较单阶段 wall/CPU 时间和峰值内存。当前不执行这些额外实验，也不声称已有性能提升。

**只读证据位置与摘要**

- 完成run：run.json (`outputs/alfred/ALFRED_V9_20261105_pilot15_i02/run.json`)，SHA256 `50f956a13869f816fca68bdf9b6e1e6a720b7fc6067640d5db2890e494afc7d9`。
- 执行计时：execution.json (`outputs/alfred/ALFRED_V9_20261105_pilot15_i02/episodes/0001/execution.json`)，SHA256 `9c6160d32d08d82baf3afe61839ec9b0b5373965c4359e9efcb38ae0cb6f4244`。
- 最终节点：memory.json (`outputs/alfred/ALFRED_V9_20261105_pilot15_i02/episodes/0001/memory.json`)，SHA256 `e1b854ce84b51a157974b4c801684e7b3a8ab9dcfc3832a9a17d9398dbb4a6f5`。
- 逐帧映射：observations.jsonl (`outputs/alfred/ALFRED_V9_20261105_pilot15_i02/episodes/0001/observations.jsonl`)，SHA256 `a8958c8edef477682988463c97561a850a67428fda03e0c86f55e15bdb568899`。
- 动作：actions.jsonl (`outputs/alfred/ALFRED_V9_20261105_pilot15_i02/episodes/0001/actions.jsonl`)，SHA256 `ddba20f16365c9f0f58151455ddc5c7d815e9894e56122aaf033e4e8ee555868`。
- 本次读取的 `memory.py` SHA256：`a2ad4185dd48687856c717a3eda8302b5f6c2e6e313e1daa734f28c3735c9096`；`object_mapping.py`：`39d99f5aa657dfb01e91fcc53f89b219d3e367ce23419034ff6ab53d800ea05b`。
