# C++ 游戏引擎重写研究

> 目标：为 AlphaZero 式 MCTS 自对弈准备一个足够快的游戏模拟器。
> 研究日期：2026-08-04。本文只做研究和方案对比，不涉及实施。

## 1. 现状：仓库里其实有"三个引擎"，但只有一个在跑

| 引擎 | 位置 | 状态 | 用途 |
|------|------|------|------|
| **Python 引擎** | `Ant-Game/SDK/backend/engine.py`（1856 行） | ✅ 唯一在用的 | 我们所有 AI 代码（`code/my_ai/` 全部走它） |
| **官方 C++ 游戏** | `Ant-Game/game/`（game.hpp + game.cpp 2852 行等，共 4765 行） | ✅ 可编译、有官方测试验证 | 独立二进制，通过 stdin/stdout judger 协议对战 |
| **pybind11 包装** | `Ant-Game/SDK/native_antwar.cpp`（594 行）+ `native_adapter.py` | ❌ 从未编译（无 .pyd/.so、无构建脚本） | 试图把 game.hpp 包装成 Python 模块 |

三个关键事实：

1. **`native_antwar.cpp` 是死代码**。它从未被编译过（仓库里没有 `.pyd`/`.so`，没有任何构建脚本），`code/` 里也没有任何代码引用 `NativeBackend`。它用 `#define private public` 这种 hack 去访问 `Game` 的私有成员，本身就是"强行适配"的产物，不能作为重写的基础。
2. **官方 C++ 游戏是完整且经过验证的移植**。`Ant-Game/tests/test_cpp_runtime.py` 会对 Python 引擎逐回合对比 C++ 引擎的塔血量、蚂蚁位置/血量，且 `game/src/game.cpp` 包含完整的 Enhanced 移动寻路（`reverse_weighted_plan`、`prepare_enhanced_move_cache` 等，和 Python 引擎方法一一对应）。但它是 judger 协议二进制——进程式、靠 JSON 包通信，无法直接嵌入 MCTS 做百万次调用。
3. **Python 引擎是性能基准**：纯引擎约 2.7~6ms/回合（见下节实测）。

> 注：之前对话里说过"C++ 引擎 2.6ms/回合"——那个数字其实测的是 Python 引擎（`performance_analysis.md` 里的 2.7ms）。native 包装从未编译过，不存在可测的 C++ 引擎。这点已在本文更正。

## 1.5 重要补充：官方自带 AlphaZero 脚手架 + 我们 AI 的设计（2026-08-04 补读后修订）

重读规则文档和我们的设计文档后，有两个此前忽略的事实，直接改变了研究结论：

### A. 官方 SDK 已自带一整套 AlphaZero 脚手架

`Ant-Game/SDK/alphazero.py` + `Ant-Game/SDK/training/`（共 ~1800 行）已经实现：

- **`PriorGuidedMCTS`**：完整的 AlphaZero 式搜索。PUCT 选路、top-k 分支（root 16 / child 10）、Dirichlet 根噪声、温度采样、先验/价值混合（`prior_mix`/`value_mix`）。**动作空间 = 96 个 ActionBundle**（不是 8303 格意图空间）。
- **`AntWarSequentialEnv`**：AEC 顺序环境。**P0 决策 → P1 决策（能看到 P0 的 ops）→ 整回合结算**（`SDK/utils/turns.py` 的 `DecisionContext`：`PLAYER_0_DECISION → PLAYER_1_DECISION → advance_round`）。这和比赛评测流程完全一致——所以 MCTS 用交替树建模是**官方认定的做法**，不需要我们自己发明。
- **`AlphaZeroSelfPlayTrainer`**：自对弈收集 episode（每步 `search.search` 64 次迭代）+ 批量 SGD 更新。
- **`PolicyValueNet`**：玩具 numpy MLP（扁平化特征 ~10100 维 → 128 → 64 → policy/value），不是我们的 CNN。

**结论：MCTS 骨架已经存在，不需要从零写。** 真正的工作是：把玩具网络换成我们的 CNN、并大幅降低 per-node 成本（见 §2.2 和 §5）。

### B. 我们的 AI 设计：CNN 已经是 AlphaZero 架构，但输出是"意图空间"

`code/my_ai/network.py` + `decoder.py` + `agent.py`（**以代码为准，设计文档 `model_design.md` 已过时**）：

- **`AntWarNetwork`**（network.py）：`NUM_CLASSES=24`（0-22 动作类 + **23=HOLD**，不是文档说的 23 类）、`STATS_DIM=42`（不是 22）、`action_map_conv(64→24)` → `action_map(24,19,19)`、3 个分类头各 `Linear(64→24)`、**价值头是裸 Linear（无 Tanh）**（PPO 时代改的，文档还写 tanh）。~0.55M 参数。
- **意图解码**（decoder.py）：每个头在掩码后的意图空间选 `(class_id, x, y)`，映射成 Operation。**关键：意图→操作是多对多**——空地上的任意塔类意图（0-15）都解码成"建 Basic"；超级武器（17-20）在金币不足且 `intent_decoding` 开启时，位置改从 downgrade 通道（16）采样、动作变成"拆塔"；21/22 基地升级无位置；23=HOLD 跳过。3 个头合并成 0-3 个操作的 bundle（`can_apply_operation` 兜底）。
- **`NeuralAgent`**（agent.py）：一次前向 + 3 头解码；支持温度采样、`action_dropout`（随机用合法动作覆盖，用于行为克隆数据增强）。
- **我们已有的 MCTS（mcts_agent.py）不是 AlphaZero**：是"双专家选择式"——两个专家模型各出一个 bundle，分支因子=2，用独立 `ValueNetwork`（另训的价值网络）评估，选更好者。无 UCB/先验，深度 3-5。文档 `dual_expert_mcts_plan.md`。
- **特征输入**：28 通道 board + 42 维 stats（官方 `FeatureExtractor`，~2.8ms/次）。
- **训练管线**：主力是 ES 家族（es_train → ss_train → ga_ss_train，带 BC 行为克隆更新 + leaderboard 对手池）；PPO（ppo_train.py）是失败实验线（胜率退化，已判死刑）。
- **当前训练方法**：ES（进化策略）打固定对手；设计文档明确写了"当前输出可自然转化为 AlphaZero 先验"（`logits = head_logits[:,None,None] + action_map`）。

### C. 核心对接问题：意图空间 vs bundle 空间

官方 MCTS 分支在 **96 个 bundle** 上，我们的 CNN 输出 **意图空间**：21 个带位置的动作类 × 19×19 ≈ 7581 格 + 2 个基地升级 + 1 个 HOLD。用 CNN 做 MCTS 先验，必须把意图分布"折叠"成 bundle 分布。**难点在于多对多映射**：同一个操作可以被多个意图解码出（空地上"建 Basic"可由任意塔类 0-15 触发；超级武器金币不足会变成拆塔），同一个意图在不同局面上也解码出不同操作（空地→建塔，有塔→升级）。

- **方案 i（bundle 空间）**：MCTS 分支在 96 bundle，先验 = 对每个 bundle 聚合"能解码出它的所有意图组合的概率"。需要意图→bundle 反查表 + 3 头联合概率的近似（求和或取 max）。树浅（每回合 2 层），但先验有损。
- **方案 ii（意图空间）**：MCTS 直接分支在掩码后的意图格上，**每个头 = 一层树**：P0-head1 → P0-head2 → P0-head3 → P1-head1 → ... → 结算，每回合 6 层。先验**精确**（直接用 action_map + head logits），和我们网络天然贴合；但树深 6 倍，且和官方脚手架（单层 = 整回合 bundle）不兼容，要改树结构。
- **方案 iii（渐进替换）**：先用官方玩具网络把**管道**跑通（验证 MCTS 自对弈在 Python 引擎上能跑、能学），再逐步换 CNN + 加速引擎。**M3 之前先做这个，成本最低。**

> 折中思路（方案 i'）：MCTS 仍分支在 bundle 空间，但**候选 bundle 由我们的 CNN 生成**（3 头各自解码出 top 意图 + ActionCatalog 启发式），先验用"意图概率沿解码路径的近似聚合"。树保持官方结构，先验质量取决于候选覆盖度。

### D. 对引擎研究结论的影响

1. 引擎 API 不用自己设计——**官方脚手架已经定义了用法**：`state.clone()` + `apply_operation_list` + `advance_round`（P1 决策后）。C++ 引擎照这套接口做即可无缝替换。
2. **per-node 成本不只是引擎**：每个新节点 = `ActionCatalog.build`（内部 `_paired_candidates` 会 clone 状态做组合尝试，很贵）+ 特征提取（2.8ms）+ `clone` + 结算。**特征提取和 ActionCatalog 与引擎同级贵，甚至更贵**。所以"引擎 30~100μs"只是其中一块，MCTS 要快必须同时加速这三样。
3. 官方脚手架的 `SearchConfig`（64 迭代）如果直接用，单次决策 = 64 × (特征提取 + clone + 结算 + ActionCatalog) ≈ 数百 ms，勉强能跑但撑不起大规模自对弈——这正是 C++ 引擎 + 特征加速的用武之地。

## 2. 性能剖析：Python 引擎的 2.7~6ms 花在哪

实测（本机，空操作 120 回合，cProfile + 分阶段计时）：

| 阶段 | 每回合耗时 | 占比 |
|------|-----------|------|
| **`_move_ants`（蚂蚁移动）** | **2.5~6.0 ms** | **≈100%** |
| `_attack_ants`（塔攻击） | 0.004 ms | ~0% |
| `_spawn_ants`（产兵） | 0.002 ms | ~0% |
| `_update_pheromone` | 0.007 ms | ~0% |
| 其余（生命周期/效果/经济） | <0.01 ms | ~0% |

`_move_ants` 内部再拆解（关键在 Enhanced 移动策略的寻路）：

- **`_reverse_weighted_plan`（多源 Dijkstra，361 格）**：单次 0.67ms。每回合调用次数 = 每个玩家 2 次（worker 路径 + combat 路径）+ 每个敌方塔 1 次。空操作时 4 次/回合 ≈ 3ms；中后期 20+ 塔时 ≈ **44 次/回合**。
- **`_is_ant_walkable_cell`**：561k 次调用/120 回合 ≈ 1.45ms/回合（Dijkstra 和蚂蚁选步都大量调用）。
- **蚂蚁选步**（`_choose_ant_move_enhanced`）：~0.9ms/回合（平均每回合 ~10 只蚂蚁）。
- `_teleport_ants`：~0.6ms/回合。

**结论：引擎成本 ≈ 100% 是移动阶段，其中多源 Dijkstra 路径规划是绝对大头。** 攻击、产兵、经济这些"常规回合逻辑"加起来不到 0.02ms，完全可以忽略。

### 为什么 C++ 会快这么多

361 格 + 6 邻居的 Dijkstra，在 Python 里用 `heapq` 是 0.67ms；在 C++ 里用二叉堆是 **5~20μs**（快 30~100 倍），用 Dial 桶队列还能更快。`_is_ant_walkable_cell` 在 C++ 里是一个内联的数组查表。整个回合预计 **30~100μs**，即比 Python 快 **30~100 倍**。

### 别忘了：特征提取和引擎同级贵

`performance_analysis.md` 的完整回合拆解：FeatureExtractor ×2 = 5.6ms（47%）、网络前向 ×2 = 4.0ms（33%）、引擎 2.7ms（22%）、Decoder ×2 = 2.0ms（17%）。

**对 MCTS 来说，特征提取和引擎一样是瓶颈**——每次叶子评估都要先提特征再进网络。FeatureExtractor 目前 2.8ms/次，若每模拟一次，100 次模拟就是 280ms，完全不可行。所以重写方案必须同时覆盖**引擎 + 特征提取**（特征提取是纯计算：28 通道 19×19 棋盘 + 标量向量，非常适合 C++ 或批量 numpy 向量化）。

## 3. MCTS 对引擎的真实需求

AlphaZero 式 MCTS 每回合需要引擎做：

1. **整回合结算**：给定双方 operation 列表 → 新状态 + 奖励 + 终局。即 `apply_turn(ops0, ops1) → (reward, terminal, winner)`。这等价于 Python 的 `resolve_turn`，但**不要** judger 协议、不要 replay 输出、不要超时/合法性兜底之外的多余状态。
2. **廉价克隆**：`clone()`。MCTS 树节点需要大量状态副本（或从 root 重放操作序列，见 §5）。
3. **确定性 RNG**：RNG 状态必须属于状态本身（clone 后各自演化），且同 (seed, ops) 结果可复现。Python 引擎已是 LCG；C++ 引擎也是 LCG，移植时保持。
4. **自洽性优先**：自对弈训练时，两边都用同一个 C++ 引擎，只要内部一致即可；**不需要**与 Python 引擎逐位一致（评测时才用官方 Python 引擎）。

## 4. 重写方案对比

### 方案 A：编译 + 修复现有 `native_antwar.cpp`

- 写 pybind11 构建脚本，把 game.hpp + native_antwar.cpp 编成 `.pyd`
- 需要解决：`#define private public` hack 脆弱；`native_adapter.py` 每步同步 Python 影子状态（开销巨大，MCTS 场景下等于白干）
- 优点：代码已存在，改动最小
- 缺点：包装层是"给 Python 同步用"的设计，不是"MCTS 用"的设计；pybind11 每次调用的对象转换 + 影子状态同步会让实际收益大打折扣；hack 长期不可维护
- **结论：不推荐作为主路径**

### 方案 B：复用官方 `game/` 的 Game 类，做干净的可嵌入封装（推荐）

官方 `game/src/game.cpp` 已经是**完整、经测试**的游戏逻辑移植（test_cpp_runtime.py 验证过它和 Python 的一致性）。做法：

1. 把 `Game` 类从 judger 协议中剥离出来（不链接 `comm_judger.cpp`、`output.cpp`、`main.cpp`），只保留核心类 + `init/apply_operation/next_round`。
2. 用公开接口或 friend 暴露 MCTS 需要的成员（当前 native_antwar.cpp 用 `#define private public` 访问 `rng_state`、`trail_cells` 等——改成正规的访问方式）。
3. 写一个**极薄**的 pybind11 绑定，只暴露：`apply_turn(ops0, ops1) → (reward, terminal, winner)`、`clone()`、少量只读查询。**不要** `sync_public_round_state`、不要影子状态、不要行级导出。
4. 用官方测试 + 自写一致性脚本验证（随机 ops 逐回合对比 Python 引擎的状态快照）。
5. 基准测试：目标 <100μs/回合。

- 优点：游戏逻辑已验证，移植工作量最小；剥离后天然适配 MCTS
- 缺点：`game.cpp` 是为"单局运行"写的，状态是 `std::deque<DefenseTower>` + 指针重连（`rewire_map`）——克隆语义需要专门处理；需要装 C++ 工具链（见 §6）
- **推荐路径**：先验证 `game/` 能编译 + 现有测试通过，再剥离封装

### 方案 C：从零重写纯 C++ 模拟器

- 优点：可以按 MCTS 需求设计（SoA 布局、arena 分配、批量状态、无拷贝克隆），性能上限最高，可以彻底绕过 `game.cpp` 的一切历史包袱
- 缺点：要把 1856 行 Python 规则（Enhanced 寻路、所有塔型特殊能力、4 种超级武器、信息素、行为系统）全部重写——工作量最大、规则漂移风险最高
- 适用时机：方案 B 验证后仍不满足性能（不太可能）或需要全 C++ MCTS（最后阶段）
- **结论：作为远期选项，不做第一刀**

## 5. 关键设计决策（重写时要定的）

### 5.1 移动策略：保留 Enhanced

Enhanced 是官方对战默认策略。它在 Python 里贵是因为 Dijkstra 用 Python 跑；在 C++ 里 30~100μs 完全可接受。**保留 Enhanced**，另留 `legacy` 开关做对照实验（验证简化移动模型对训练收敛的影响）。

### 5.2 交替树建模：官方脚手架已按此实现 ✅

比赛流程其实是**先手盲动、后手知情**（README 评测流程：P0 操作验证后转发给 P1，P1 再决策，然后一起结算），不是严格同时。官方 `DecisionContext` 已经把它建模成 `P0 决策 → P1 决策 → advance_round` 的交替结构，`AntWarSequentialEnv` 和 `PriorGuidedMCTS` 都按这个跑。**这条不用我们发明，直接用。**

> 注意：树里引擎只在"双方都选完"（P1 决策后）结算一次，不是每个节点都结算。所以每模拟的引擎调用次数 ≈ 路径上的回合数（树深/2），不是路径节点数。

### 5.3 动作空间剪枝：官方已实现 top-k ✅

96 个 bundle 全展开太贵（国际象棋 ~35、围棋 ~250，我们 96×组合）。官方 `PriorGuidedMCTS` 已实现 KataGo 式**移动选择**：`SearchConfig.root_action_limit=16`、`child_action_limit=10`，按先验取 top-k 再加保底动作 0（hold）。直接用，不用重写。

### 5.4 树的状态存储：从 root 重放 vs 每节点存状态

- 每节点存克隆状态：内存大，但每个新孩子只需 1 次克隆 + 1 次结算
- 从 root 重放 ops：内存小，但每叶子要重放 O(路径回合数) 次结算
- KataGo 的做法：树只存 root 状态 + 当前批次的叶子状态，批量评估之间从 root 重放
- 引擎 API 两种都支持（`clone` 便宜 + `apply_turn` 快），具体由 MCTS 层决定

### 5.5 训练/评测分离

- **训练（自对弈）**：用 C++ 引擎，自洽即可，快
- **评测/提交**：用官方 Python 引擎（或官方 C++ 二进制），规则权威
- 风险：若 C++ 与 Python 规则漂移，训练出的策略可能不迁移。缓解：定期随机回放一致性校验（test_cpp_runtime.py 已有模式），漂移大时以 Python 侧为准修 C++ 侧

## 6. 前置条件：机器上没有任何 C++ 工具链 ⚠️

本机检查结果：**没有 g++、gcc、make、clang、cmake**，conda 环境里也没有编译器。pybind11 3.0.4 已装（Python 侧没问题），但 C++ 侧是空白。

装工具链的选项：
- **conda**：`conda install -c conda-forge gxx_win-64 make`（或 `c-compiler cxx-compiler`，装 m2w64 GCC）。和现有 `game/Makefile`（g++ 语法）最兼容。
- **MinGW-w64 独立包**（winlibs 或 MSYS2）：同样兼容 g++ 语法。
- **MSVC（Visual Studio Build Tools）**：Windows 上 pybind11 最标准，但 `game/Makefile` 是 g++ 写的，需要改或用 CMake。

**M0 里程碑必须先解决工具链**，然后验证 `game/` 能编译、`test_cpp_runtime.py` 能通过（这是方案 B 的地基）。

## 7. 推荐里程碑

> 与之前版本的差异：MCTS 骨架官方已有，所以里程碑从"写 MCTS"改为"加速 + 换网"。

| 里程碑 | 内容 | 验证标准 |
|--------|------|---------|
| **M0** | 装 C++ 工具链；`make` 编译 `game/`；跑 `test_cpp_runtime.py` | 全部测试通过 |
| **M1** | 从 `game/` 剥离 Game 类 + 极薄 pybind11 绑定（`apply_operation_list`/`advance_round`/`clone`，对齐官方脚手架的调用方式），去掉 judger 协议和影子状态 | 与 Python 引擎随机回放逐回合一致；单回合 <100μs |
| **M2** | 跑通官方脚手架（玩具网络 + Python 引擎）：`SDK/train_mcts.py` 能自对弈训练并保存 checkpoint | 一个 epoch 跑完不报错；搜索耗时可见 |
| **M3** | 意图→bundle 先验映射：写 `CNNPolicyValueNet` 包装我们的 `AntWarNetwork`，把 (23,19,19) 意图分布折叠成 96-bundle 先验（方案 i）；接进 `PriorGuidedMCTS` | 自对弈训练用上 CNN 先验；对固定对手胜率上升 |
| **M4** | 性能：特征提取加速（C++ 或批量 numpy，<50μs/次）+ C++ 引擎接进 env | 单次决策（64 迭代）从 ~300ms 降到 <50ms |
| **M5**（可选） | 树/搜索逻辑下沉 C++，或接入批量网络 | 每回合决策 <10s 内，模拟数上千 |

引擎工作（M0~M1）和管道跑通（M2）可以并行准备；M3 是真正的"我们的 AlphaZero"验证点。**先别急着上 C++ 引擎——先用官方脚手架（M2）确认这套 AlphaZero 管道在我们的游戏上真的能学，再投钱优化速度（M4）。**

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| C++ 与 Python 规则漂移（Enhanced 寻路细节、塔特殊能力） | 随机回放一致性测试作为 CI 门禁；训练/评测分离 |
| `game.cpp` 的克隆语义（`deque` 存储 + 指针重连 `rewire_map`） | M1 时专项压测 clone 正确性（克隆后再跑 N 回合，与原状态并行跑结果一致） |
| Windows 工具链安装失败/编译报错 | M0 先做最小冒烟（编译一个 hello pybind11 模块）再上完整移植 |
| 意图→bundle 先验映射有损（同一 bundle 可被多种意图组合解码出） | M3 先做"先验一致性"校验：CNN 先验 top-k 动作与 decoder argmax 动作的重合度 |
| 官方脚手架玩具网络可能学不动 | M2 就用小规模实验确认管道能学（即使慢）；不行再查脚手架本身 |
| 特征提取 + ActionCatalog 拖累 MCTS | M4 独立里程碑，不等 M1 完美 |

## 9. 一句话总结

**官方已有完整的 AlphaZero 脚手架（MCTS 交替树 + top-k 剪枝 + 训练器），我们的 CNN 也已是 AlphaZero 架构——缺的是把 CNN 意图分布映射成 bundle 先验、以及把每节点成本降下来。引擎侧推荐方案 B（把已验证的官方 `game/` C++ 移植剥成极薄 pybind11 模块，30~100μs/回合），但**先跑通官方脚手架管道（M2）确认能学，再投钱加速（M4）；第一步是装 C++ 工具链（M0）。**
