# 冠亚军 AI 逆向源码分析（Ant-War 2 / Saiblo 第 30 届）

> **基于已有的 `docs/distill_top2_expressivity_plan.md` §1 报告补充**，并交叉引用
> `docs/findings_runnerup_prior_structure.md`、`docs/findings_shadow_sim_fidelity.md`、
> `docs/other_versions_ai_sources.md`、`其他版本ai/CLAUDE.md`、`其他版本ai/omegafantasy的方法考古.md`。
> 旧报告 `distill_top2_expressivity_plan.md` §1 已经给出"两者都是零机器学习 + 方法族 C/对抗 MCTS"的结论骨架；
> **本文不重复该骨架，而是补齐它没有回答的 5 个问题**：搜索预算怎么花、rollout 多深、评估特征与权重全表、
> 座位处理、以及可被针对的弱点。旧报告有 3 处需要修正/加限定，见 §2.1、§2.3、§6.1。
>
> 只读分析：未修改、未编译、未运行任何被分析代码。所有结论带 `文件:行号`。

---

## §0 一手资料清单

### 0.1 读了什么

| 文件 | 行数 | 性质 | 读了哪些部分 |
|---|---|---|---|
| `其他版本ai/ant-war2-magica-v3/main.cpp` | **27,462** | 冠军全部源码（单文件合并产物，1.48 MB） | 结构图全量（`struct/class/namespace` 全表）；`MagicaMcts` 全类；`ComputeStateValue`/`ComputePlayerStateScoreBreakdown` 全部；prior 计算；`PolicyRunner::Run`；`SimValuePlanner::GenerateArrangement` 头部；约 25 处常量 |
| `其他版本ai/ant-war2-magica-v3/readme.md` | 6 段 | 作者自述（**GBK 编码，直接 Get-Content 是乱码**，需转码读） | 全文 |
| `.../cpp_lure_v4/ai_cpp_lure_v4.cpp` | 195 | 亚军入口薄壳 | 全文 |
| `.../cpp_lure_v4/include/antgame_ai/*.hpp` | **5,652**（27 个头文件） | 亚军策略全部逻辑 | `decision` `params` `evaluation` `terminal_eval` `rollout_score` `rollout_sampling` `root_plans` `core_values` `plan_types` 关键段；其余仅结构扫描 |
| `其他版本ai/saiblo-30th-AI/Game1/antgame_cpp_sdk/include/antgame_sdk/random_search_baseline.hpp` | **5,015** | 亚军的**前向模型** `rs::DefenseSimulator`（旧报告漏了：它不在 `cpp_lure_v4/` 里） | 结构全表 + `simulate_round`/`move_options_for`/缓存/`kMaxImportantAnts` |
| `.../antgame_sdk/native_sim.hpp` | 104 | 官方 native 引擎绑定（pimpl，实现在二进制里） | 结构 |

**冠军源码确认存在**（不是"只有二进制"）：`main.cpp` + `magica_v3_O0.exe` + `readme.md` 三个文件，
`git -C 其他版本ai/ant-war2-magica-v3 status --short` 只有 `?? magica_v3_O0.exe`（exe 未跟踪），
**`main.cpp` 与上游 pin `f33534d` 一致，无本地改动** ⇒ 分析的是纯净的提交源码。

### 0.2 已有的旧报告（不重复造轮子）

| 文档 | 已覆盖 | 本文如何处理 |
|---|---|---|
| `docs/distill_top2_expressivity_plan.md` §1（:25-79） | 两者方法族、冠军 RNG 控制、亚军 `evaluate_terminal` 公式 | **作为基础**，补全并升级为"可复现的数字表"；修正 3 处 |
| `docs/findings_runnerup_prior_structure.md` | 亚军的"强度来自搜索不来自先验"（312,771 条 plan 的逐项核对） | 直接引用其判决，不重做统计 |
| `docs/findings_shadow_sim_fidelity.md` | 亚军单边仿真器的保真度 | 引用其"单边、对手只有两个标量"结论 |
| `docs/other_versions_ai_sources.md:19-20` | 两个子模块的 url + pin 提交 | 用于确认源码纯净性 |
| `其他版本ai/CLAUDE.md` | "全程零机器学习（0 权重文件/0 ML 框架导入/0 训练脚本）" | 本文用 grep 独立复核（§1.1、§2.1） |

### 0.3 ⚠️ 工作区不纯净的 3 处（影响"这是谁写的"判断）

`git -C 其他版本ai/saiblo-30th-AI status --short` 给出 **2 处修改 + 1 个未跟踪文件**，全部是**我们自己**加的：

```
 M Game1/antgame_ai_cpp/cpp_lure_v4/ai_cpp_lure_v4.cpp          # 加 _setmode(_O_BINARY)（:157-162）
 M .../lure_strategy_v4_terminal_eval.hpp                        # 加 LURE_DUMP_VALUE / LURE_NEURAL_VALUE 分支
?? .../antgame_ai/entity_value_net.hpp                          # ← 我们的人工前向小价值网（新增文件）
```

⇒ **`entity_value_net.hpp` 不是亚军的代码**，是本地注入的探针（其第 2 行自述"对应 `checkpoints/entity_sim_value_small.pt`"）。
两个 `#ifdef`（`LURE_NEURAL_VALUE` / `LURE_DUMP_VALUE`，`terminal_eval.hpp:265-391`、`:441-443`）
**默认不编译**，因此不影响本文对亚军算法的判断；但任何"亚军用了神经网络"的说法都是误读。

### 0.4 没读 / 读不到

- 冠军的 `bundled_data`（`main.cpp:5962` 起约 5,000 行常量数据）、`Env`（:5855-11420，5,500 行仿真器）只看了结构，**未逐行**。
- 亚军 `rs::DefenseSimulator`（5,015 行）的塔伤害/蚂蚁行为细节只读了结构 + `simulate_round` 主循环，**未逐行**。
- `antgame_sdk/native_sim.hpp` 的实现在二进制里（`struct Impl`，:103）⇒ 该层**读不到**。
- 冠军 exe（11.9 MB）未反汇编——不需要，源码在。

---

## §1 冠军分析：`ant-war2-magica-v3`（"小小预言家-v3"，作者 yanlifeng）

### 1.1 整体架构：**手工启发式 + AlphaZero 式 MCTS（手写 prior + 手写 state value + PUCT），零机器学习**

**代码证据（三件套齐全，但不是神经网络）**：

| 要素 | 证据 |
|---|---|
| 树 + 访问计数 | `Node` 含 `visit_count/value_sum/prior`（`main.cpp:16241-16260`）；`BackupValue` 逐节点 `++node.visit_count; node.value_sum += root_value`（`:24603-24610`） |
| PUCT（AlphaZero 常数） | `pb_c_base = 19652.0f`、`pb_c_init = 1.25f`（`:16218-16219`）；`pb_c = log((N+base+1)/base)+init`（`:24496`）；`exploration = pb_c * child.prior * sqrt(parent_visits)/(1+child.visit_count)`（**`:24503`**） |
| 手写 prior | `prior_logit = total_contribute + effco_delta_logit`（`:20067`），`effco_delta_logit = effco_delta * 0.25`（`:20066`）；softmax 温度 3.0（`:17062`、`:22647`、`:22675`） |
| 手写 state value | `ComputeStateValue(...) = my.state_score - enemy.state_score`（`:16052`、`:16056-16065`） |
| 无 rollout | 见 §1.2 末段（叶子值直接取价值函数） |

**机器学习痕迹：0**。对 `其他版本ai/ant-war2-magica-v3/` 全目录 grep
`tensorflow|torch|onnxruntime|onnx|pytorch|keras|numpy|.pt|.h5|.bin|load_weights|batch_norm`
**只命中 3 行自写的 `softmax` 温度代码**（`:17062`、`:22647`、`:22675`）；
目录下无任何 `.onnx/.pt/.bin/.h5/.pth/.pb/.npz/.pkl/.safetensors` 文件；无训练脚本。
⇒ 与 `其他版本ai/CLAUDE.md` 的"全程零机器学习"一致，且我独立复核过。

**注意一个易误读的点**：`main.cpp:2126-2127` 确实复刻了 Java LCG（`RNG_MULTIPLIER = 25214903917`，
`:2272` `rng_state = (MULT * rng_state + INC) & MASK`）、
`:3613` 由游戏 seed 播种、`:2026`/`:3637` 支持 `use_raw_rng_state` 直接设 rng 状态 —— 这是**确定性规划**（前向模拟随机数流，
让未来的蚂蚁移动/生产类型落在有利分支），不是机器学习。旧报告 §1.1 已正确指出这点，我复核了这几行。

### 1.2 搜索方式

**(a) 候选动作怎么生成：按"目标敌蚁"分层枚举 arrangements，不是枚举全部动作**

- MCTS 的树按 `key = (round, target_ant_id)` 分层：`RoundInitPreprocessResult` 先挑出本回合要处理的敌蚁列表
  （`CollectMixedRoundInitTargetAnts`，`:22804`），随后**一次展开只针对一个目标蚁**（`:22956` `current_target_ant_id`）。
- 每个目标的候选 = `SimValuePlanner::GenerateArrangement`（`:22894`）产出的 **arrangements**
  （"用哪座塔打 / 派哪只战斗蚁顶 / 是否降级腾钱"的组合），再由
  `ActionProposer::BuildExpansion`（`:22762-22892`）装配成 `Proposal`。
- **每节点上界 32 个候选**：`kMagicaMctsMaxExpansionProposals = 32`（`:16116`），
  `ExpandNode` 硬校验 `expansion.proposals.size() > 32` 即失败（`:24515`）。
- **每个候选动作最多 8 个 op**：`kMctsMaxValidOpsPerMove = 8`（`:16120`）。
- **树的容量 10 万节点**：`kMagicaMctsMaxNodes = 100000`（`:16113`、`:24530`）。
- 子节点上界名义上是 `kMagicaMctsMaxChildrenPerNode = kMaxActions + 1`（`:16118`），实际以 32 为准（`:16117` 注释自述）。

⇒ **不是"枚举全部动作"**：枚举被两层剪枝压住 —— ① 只对"排好序的目标蚁"逐个展开；
② 每个目标内部由 planner 生成 32 个以内的 arrangement。

**(b) 评估函数：不是 rollout，是 state value 直接给叶子打分**

这是本文与旧报告相比最需要强调的一点。看 `Search` 的叶子处理（`:24148-24238`）：

```
24148  float leaf_root_value = MeanValue(nodes_[node_index]);              // 先用已有均值
24149  if (!nodes_[node_index].expanded) {
24163      if (IsHorizonCutoffLeaf(request, node, simulation_env)) {       // 到 max_round 深度上限
24164          leaf_root_value = action_proposer_.ComputeRoundBoundaryRootValueForMcts(simulation_env);
                                                                          // ↑ 直接调 state value，不 rollout
24180      } else {
24185          const Expansion expansion = action_proposer_.BuildExpansion(...);   // 继续展开
24230          if (expansion.root_value_kind == RootValueKind::kReal) {
24231              leaf_root_value = expansion.root_value;                  // arrangement 前向 5 回合的估值
24232          } else {
24234              leaf_root_value = MeanValue(nodes_[round_root_index]);    // 否则回落到"本回合根"的均值
```

- 深度：`request.max_round = root_env.RoundIndex() + 9`（**`:26823`**）⇒ **前瞻 9 个回合**（游戏上限 512 回合，`:11461`）。
- 采样数：`request.max_simulations = 50`（**`:26822`**）⇒ **每步只做 50 次模拟**。
- 无 MCTS 内的随机 rollout、无 MCTS 内的采样噪声：50 次模拟全是确定性的（除 RNG 复刻部分）。
- **有访问计数**（树节点 N），选子用 PUCT，最终选 `visit_count` 最大者（`FillResult`，`:24648-24657`）；
  平手时**优先非 pass、再比 prior**（`:24649-24650`）——这是个明确去偏设计。

**(c) 备份不翻符号（重要设计，也是弱点，见 §5）**

```
24603  void MagicaMcts::BackupValue(..., float root_value) {
24604      // Stored values remain root-relative at every depth, so backup never flips sign.
24605      for (path_index ...) { ++node.visit_count; node.value_sum += root_value; }
```

⇒ 全深度都存**根视角**值，不做 negamax 的符号翻转。这与 AlphaZero（单网络双视角）不同：
它要求叶子估值本身已经是"根视角"，因此对手回合节点也累加同一标量。

### 1.3 评估函数（`state value`）的特征与权重

**总公式**（`:15947-15949`）：

```
state_score(p) = base_hp
               + effco_value / 60
               + ant_expect_value
state_value    = state_score(me) - state_score(enemy)          （:16052）
```

**逐项权重表**（全部常量在 `:15803-15808`，DPS 表在 `:15816-15829`）：

| 特征 | 权重/公式 | 行号 |
|---|---|---|
| **基地血量** | ×1（直接当主单位） | `:15947` |
| **effco_value（等价金币）** | `coins + ComputeOptimalLiquidationRefund(最优清仓回收)`，再 **/60** | `:17901-17903`、`:15948` |
| `kStateValueHp2Coin`（血↔钱汇率） | **60** | `:15807` |
| **战斗蚁数量** | **×1.5** | `:15944` |
| **工人蚁期望损失** | `−Σ expect_loss / 60` | `:15939`、`:15945` |
| `kStateValueBaseHpEconomicLoss`（掉 1 血的等价损失） | **60** | `:15804` |
| `kStateValueAntArriveCoin`（蚁到达的附加损失） | **10** | `:15805` |
| `kStateValueCombatAntValue` | **1.5** | `:15808` |
| `kStateValueDistanceLooseFactor`（距离宽松系数） | **1.33** | `:15806` |
| `kStateValueHitSuccessProbability`（单回合命中率） | **0.9** | `:15803` |
| 击杀一只蚁折算的金币 | 战斗蚁 **18**；工人 lv0 **6** / lv1 **10** / lv2+ **14** | `:15839-15850` |
| DPS 查表 | 12 档 `effco_threshold` ∈ {15,45,90,105,150,165,180,210,240,270,300,360}，每档两组 DPS（有/无战斗蚁） | `:15816-15829`、`:15998-16022` |

**"竞速式"工人威胁模型**（冠军最精致的部分，`:15922-15942`）：

```
按 remain_distance 升序（:15915-15920）遍历敌方工人蚁：
  loss_value      = 60 + kill_coin + 10
  loose_threshold = remain_distance * 1.33
  if (t_sim + eliminate_time > loose_threshold)   → 全额损失 loss_value        // 塔来不及杀掉它
  else                                            → loss * (1-0.9)^fail_rounds // 还来得及
  t_sim += eliminate_time        // ← 关键的"塔的击杀产能"串行化
```

其中 `eliminate_time = (ceil(hp/standard_dps) + shield) / 0.9`（`:16024-16038`）——
即**把己方塔的 DPS 当成一条串行流水线**，前面的蚁占用产能会推迟后面蚁的击杀时刻。
这是"逐蚁 + 时间"的推理，不是静态距离加权。`ComputeOptimalLiquidationRefund`（`:17858`）
则是"把所有塔按最优顺序降级再卖"的回收价，等于**把塔折成钱**（亚军也有同样的概念，见 §2.3）。

**prior（候选打分）与 state value 的分工**（作者自述 + 代码印证）：

- `prior_logit = (target_contribute + 0.5 * other_contribute) + effco_delta * 0.25`（`:20064-20067`，
  `kOtherAntContributionWeight = 0.5`、`kEffcoPriorLogitWeight = 0.25`）。
- **硬门 `prior_gate`**：若"目标蚁之前的蚁"受到的伤害比 baseline 还少（`delta < -1e-8`），
  `prior_gate_ok = false`（`:20059-20061`）⇒ 该候选直接淘汰。
  **这是"否定第一感"的机制化**：不许为了救后面的蚁而牺牲前面的蚁。
- arrangement 的前向打分另有 `SCORE_W_ECONOMICS = 0.2`（`:17052`、`:18405`），
  与 `kDamageFlowHorizonRounds = 5`（`:17055`）配套 —— 与经济项相乘的是 **0.2**，比 prior 里的 0.25 略保守。

### 1.4 时间/算力使用

| 项 | 值 | 证据 |
|---|---|---|
| 每步模拟次数上限 | **50** | `:26822` |
| 每步墙钟上限 | **5000 ms**，且先扣掉闪电搜索耗时，**下限 2000 ms** | `:24901-24902`、`:24914-24917`、`:26824` |
| 超时处理 | 循环头 `IsSearchTimeLimitExceeded` 每 sim 检查一次，超了就 break（**保留已完成的部分结果**） | `:24030-24041`、`:23721-23730` |
| 单步总预算结构 | 先 `EvaluateThunderV2Phase1`（闪电专项搜索，`:26796`），再 MCTS 用剩余时间 | `:26794-26824` |
| 多线程 | **无**（`std::thread|std::async|pthread_create|#pragma omp` 全文件 0 命中） | grep |
| 缓存/前缀共享 | **无跨 sim 共享**：每次模拟都 `Env simulation_env = root_env_.Clone()` 然后从根重放路径 | `:24055`、`:23936` |
| 环境拷贝 | `ApplySearchMove` 也走 `Env staged_env = active_env.Clone()` | `:23936` |

⇒ 冠军的策略是"**用精确全引擎仿真换搜索宽度**"：50 次模拟 ×（9 回合 × 全引擎 clone）在 C++ 里够快；
没有缓存优化，因为模拟次数本来就只有 50。**这一点很重要：它没有"更快仿真"的压力，压力在"更好的先验/价值"。**

### 1.5 先手/后手处理

**结论：策略层完全对称，唯一的 `player` 分支是协议收发顺序。**

- 主循环 `PolicyRunner::Run`：`if (state_.SelfPlayer() == 0) { EmitRoundActions(); ReceiveOpponentTurn(); }`
  vs `else { ReceiveOpponentTurn(); EmitRoundActions(); }`（**`:27244-27284`**）。
  即 **P0 先出招再收对手操作；P1 先收对手操作再出招** —— 这是赛制决定的真实信息差（后手能看到先手本回合的出招），
  **不是** AI 自己加的特权。
- 策略/估值里没有任何 `player == 0/1` 的分支，只有镜像：`enemy_player = 1 - self_player`（`:16048`、`:23740`）。
- 引擎克隆里有大量 `player == 0/1`（`:4433-4805`），但那是**复刻裁判的操作顺序**，
  例如 `:4515 if (player == 1) // must cost coin first!!`（P1 的扣钱顺序不同）。
  这是"忠实复刻规则"，对双方同等施加。

⇒ **冠军没有"我方是先手所以怎么下"的特殊逻辑**。若我们观测到 27–47pp 的座位差
（口径见 `docs/prior_ceiling_20261026.md:127-133`：固定种子下 27pp，后手方 72%），
那**不是**冠军代码里的对称性破缺，更可能是**信息差本身**（P1 看得见 P0 的出招）+ 我们自己的适应性问题。
这条对我方很重要，见 §5 与 §6.2。

### 1.6 冠军的弱点（代码级）

按可利用程度排序，全部有行号：

1. **只 50 次模拟 + 每层 32 候选 ⇒ 对 prior 的依赖极强**（`:26822`、`:16116`）。
   若某类走法的 `prior_logit` 恒低（例如"先攒钱不动"），它**几乎不可能**被 50 次模拟翻出来。
2. **`prior_gate` 硬否决**（`:20059-20061`）：任何"牺牲目标蚁之前的蚁"的交换被结构性禁止。
   对付"多点同时压上、逼它在两个蚁之间取舍"的局面，它的候选集会被剪空（或退化为 pass）。
3. **备份不翻符号**（`:24603-24604`）：对手回合节点也用根视角的值累加，
   ⇒ 对"对手会针对性反制"的局面不敏感（近似 max 而非 minimax）。
4. **非回合边界叶子没有独立估值**，回落成"本回合根的均值"（`:24232-24237`）
   ⇒ 同一回合内的多个动作，价值上不可区分，只能靠 prior 排序。
5. **每个 move 最多 8 个 op**（`:16120`）⇒ 需要长组合操作（连续降级腾钱 + 建塔 + 升级）时被迫拆到多回合。
6. **视野只有 9 回合**（`:26823`），而游戏 512 回合 ⇒ 对"后期经济复利"类策略近视。
7. **强依赖 shadow env 同步**：一旦检测到与公开状态不一致就走 `force_pass`
   （`main.cpp:26748-26760` 的 `shadow_env_unhealthy` / `shadow_env_terminal` 分支；
   我们已在 `docs/experiment_log.md:3854` 观测到"第 1 回合 `age=0` vs `age=1` 差异导致冠军检测到 desync 后 force_pass"）。
   ⇒ 这是**它的脆弱点**，但**主动制造 desync 属于规则边缘**，不建议作为战术（见 §5 的合规章节）。

---

## §2 亚军分析：`saiblo-30th-AI/Game1/.../cpp_lure_v4`（作者 omegafantasy）

### 2.1 整体架构：**模板候选 + 扁平 UCB1 bandit + 单边 MC rollout + 手写终点估值，无树、零 ML**

**证据**：

- 决策管线只有三步（`lure_strategy_v4_decision.hpp:28-34`）：
  ```
  defense_root = make_defense_simulator(state, simulator, player); defense_root.ignore_enemy_spawns = true;  // :28-29
  root_plans   = generate_root_plans(state, &defense_root, player);                                          // :30
  evaluated    = evaluate_root_plans(state, defense_root, player, serial, root_plans);                        // :33-34
  final_ops    = evaluated.front().plan.ops;   // 直接取第一名                                        // :45
  ```
- **无树**：每个候选（arm）只持有一个 `rs::DefenseSimulator plan_root`（`evaluation.hpp:135`），
  **没有子节点、没有深搜**；`UcbEvaluationTrace` 里只有 `samples/batches`（`:50-68`）。
- **访问计数是 bandit 的 `n` 而不是树节点 N**：`UcbArm::samples`（`:139`），UCB 公式用 `arm.samples`（`:311-314`）。
- **ML 痕迹：0**。全部 27 个头文件 grep `tensorflow|torch|onnx|pytorch|keras|numpy` 只命中
  我们的 `entity_value_net.hpp:2`（本地注入，见 §0.3）；目录下无权重文件；无训练脚本。
  另有 `entity_value_net.hpp:328-391` 的 `LURE_NEURAL_VALUE` 分支，**默认不编译**。

**旧报告需要加的限定（修正 1）**：旧报告说亚军"rollout 打分"。更精确的说法是
**"单边简化前向模型的 K 回合展开 + 末端手写估值"**，其前向模型
`rs::DefenseSimulator` 在 `Game1/antgame_cpp_sdk/include/antgame_sdk/random_search_baseline.hpp`
（**5,015 行，不在 `cpp_lure_v4/` 里**，旧报告没点出位置），它**不是全引擎**：
只模拟己方塔 + 敌方蚁，且 `ignore_enemy_spawns = true`（`decision.hpp:29`、`random_search_baseline.hpp:1029`、`:3538`）。

### 2.2 搜索方式

**(a) 候选动作怎么生成：4 个模板族的"积"再按 key 去重**

`generate_root_plans`（`root_plans.hpp:175-182`）把四族拼起来：

| 族 | 生成器 | 规模 |
|---|---|---|
| base（建塔/升级/降级/换位） | `generate_base_candidates`（`base_plans.hpp:225`、`:398`） | **15 个 base 槽位**（`core_format.hpp:406` `std::array<int,15> base_codes()`）× {build, upgrade, downgrade}, 另有 `15×15` 的 swap 与 `15×15` 的 recycle-into-lure（`base_plans.hpp:79-91`、`:250-338`）；**受建塔上限约束** |
| lure（诱饵：在 `lure_codes()` 专用槽位上建/卖塔，操纵敌蚁寻路与回收） | `generate_lure_candidates`（`lure_plans.hpp:7`） | `lure_hold` + 每个己方 lure 槽位的"卖掉"（`:40-53`）+ 空位时"建造"（`:55-60`）；`forced_lure_sell_distance = 1`、`max_non_lure_towers = 2`（`params.hpp:31-33`） |
| lightning_prep / lightning_center（闪电前置与落点） | `lightning_plans.hpp` | 落点限制 `hex_distance(center) <= 5`（`params.hpp:28-29`） |

- **硬性上限：场上最多 2 座非 lure 塔**（`max_non_lure_towers = 2`、`rich_max_non_lure_towers = 2`，`params.hpp:32-33`；
  生效点 `base_rules.hpp:98-110` 的 `non_lure_tower_build_limit`，调用点 `base_plans.hpp:230`、`:403`）。
- 去重：按 `plan_key(ops, followup)` 去重，重复时**保留 heuristic 更高的那个**（`root_plans.hpp:220`、`:239-261`）。
- **最终 plan 数源码不给常量**：`RootPlanSet` 只有 `base_count/lure_count/.../raw_plan_count` 四个计数字段
  （`root_plans.hpp:189-193`），日志里报 `plans_total`（`decision.hpp:133`）。
  实测约 **40 个**（旧报告 `findings_runnerup_prior_structure.md` 转述其自身日志；
  `distill_top2_expressivity_plan.md` 也写了"亚军自己的 40 个"）——**这是观测值不是常量，见 §6.2**。

**(b) 评估：扁平 UCB1 + 每臂 N 次随机 rollout + 末端手写估值**

```
evaluate_root_plans（evaluation.hpp:90）：
  for each plan → UcbArm { plan_root = rollout_root->clone(); apply_operations(plan.ops) }   // :147-159 前缀共享
  分组：has_lightning 分开                                                              // :165-171
  ① 闪电组：run_ucb_group(lightning_arms, batch=2, total=600, exploration=300)          // :335-339
  ② 普通组：run_budgeted_action_group()                                                  // :344-460
  排序：mean_score = mean_rollout_score + plan.heuristic                                 // :475
```

- **每条臂的目标采样数 125**：`action_target_rollouts_per_action = 125`（`params.hpp:11`），
  单批上限 100（`:12`）；**UCB 探索常数 600**（`:6`）。
- **先探后算**：每臂先跑 `action_probe_min_samples = 5`（`params.hpp:10`）个样本（`evaluation.hpp:354-367`），
  用实测 `us_per_sample` 外推"剩余时间还能跑多少"（`:377-382`），
  再 `× action_target_total_multiplier = 1.25`（`params.hpp:8`）得动态目标（`:385-388`），
  **上限 `per_action_target_total = 125 × 臂数`**（`:388`、`:411-412`），且受 `action_time_budget_ms = 6000` 硬闸（`:390-392`、`:426-428`）。
- **rollout 深度 = plan.horizon**：普通/闪电前置 plan 用 `long_eval_horizon = 8`，
  纯闪电 plan 用 `lightning_horizon = 10`（`params.hpp:24,26`；赋值点 `root_plans.hpp:306/331/350/368/395/425/441/462`）。
- **mid-eval 实际是关的**：`mid_eval_horizon = 4` 但 `mid_eval_weight = 0.0`（`params.hpp:23,25`），
  而 `combine_eval_breakdowns` 用 `weight` 混 `mid`（`terminal_eval.hpp:447-455`，调用 `rollout_score.hpp:155`）
  ⇒ 权重 0 时只剩 terminal。**所以"4 回合中期评估"当前不生效**（第 4 回合的 `capture_mid_eval` 白算一次）。
- **rollout 里的随机性**：蚁移动按 `move_options_for` 的概率分布**分层抽样**（`rollout_sampling.hpp:29-105`），
  只对 **至多 5 只"重要蚁"** 强制走位（`kMaxImportantAnts = 5`，`random_search_baseline.hpp:63`；
  裁剪点 `rollout_sampling.hpp:136-139`）。
- **是否有 MCTS**：**没有**。无树、无 backprop 到祖先、无 expansion。UCB 只是"选哪条臂再多采样"的调度器。

**(c) 亚军的"强度来自搜索"已被旧报告钉死**

`docs/findings_runnerup_prior_structure.md` 用 312,771 条 plan 逐项核对得出：
`lure_heuristic == lightning_heuristic == followup_penalty == 0` 占 100%，
`heuristic = base + lure + lightning − op_pen − followup_pen` 恒等式 100% 成立，
且它自己重复同一决策的一致率上限约 **87.5%**（SE≈11 分）。
⇒ **本文不再重做该统计**，只引用。

### 2.3 评估函数的特征与权重（全表）

`evaluate_terminal`（`terminal_eval.hpp:393-445`）：

```
total_score = base_hp * 200                        # :396
            + tower_value_score + money_score      # :402-403（等价金币按比例拆成"塔的份"与"现金的份"）
            + c1_bonus                             # :405-406（★ 实为 0，见下）
            − worker_threat                        # :407-408
            − combat_threat                        # :409-410
            (+ future_threat_adjustment_score)     # :428-439（默认关）
```

**全部权重**（`params.hpp:57-104`）：

| 特征 | 权重 | 行号 |
|---|---|---|
| 基地血量 | **×200** | `:66` |
| 等价金币（现金 + 最优清仓回收） | **≤400 的部分 ×10，超出部分 ×8**（分段递减） | `:67-70`、`terminal_eval.hpp:93-99` |
| `tower_value_weight` | 10（**实际终局按比例拆分，见 §6.2**） | `:67`、`terminal_eval.hpp:401-404` |
| 工人威胁 | `Σ 160 * hp/max_hp / max(1, hex_dist(ant, own_base))` | `:73`、`:72-73` |
| 战斗威胁 | `Σ max(300*?, core_tower_anchor_threat) * behavior_scale`，`random→0.6`、`bewitched→0.3` | `:76-82` |
| combat 的 C1 锚点 | `coin_ratio = 0.2`，环 1 加成 `0.2`，环距 1 | `:77-79` |
| 闪电 bonus | 敌方 super 在场 **+100**，否则 **−200**；破盾 **25**；每点伤害 **1.5**；塔价值比 **0.5** | `:100-104` |
| hold 加分 | **+80**（在 heuristic 里，不是 terminal 里） | `:40` |
| 降级惩罚 | sniper 降级 **−400**；producer 降级 −200 / medic −800；medic 升级 +1000 | `:49-55` |
| C1 相关 | `c1_quick_trans_bonus = 100`、`c1_sniper_trans_bonus = 400` | `:46-47` |
| 操作惩罚 | `operation_penalty`（来自 `downgrade_penalty_for_ops`） | `root_plans.hpp:221` |

**`tower_full_salvage_value`（亚军最值得抄的一段，`core_values.hpp:93-131`）**：

```
对每座己方塔：while (type != Basic) { total += downgrade_tower_income(current);
                                       按 max_hp 比例缩放 hp; type /= 10; }
              收集成 BasicRefund{hp}
按 hp 降序排序，依次 destroy_tower_income(tower_count--, hp) 累加
```

⇒ 它精确复刻了"**把塔降级回基础再按 HP 从高到低卖**"的回收价（回收价随剩余塔数递减），
于是"塔"和"钱"被折算到同一单位 **equivalent_money**，这是它经济判断的基础。

**修正 2（旧报告的一处不精确）**：`c1_terminal_bonus` 在终局估值里**恒为 0**
（`terminal_eval.hpp:89-91`：`inline double c1_terminal_bonus(const rs::DefenseSimulator &, int) { return 0.0; }`）。
C1 加成只以 **root 层的 `c1_root_bonus_for_plan`** 形式进入（`evaluation.hpp:156`、`:229-232`）。
⇒ 旧报告把 `c1_bonus` 当成终局估值的活跃项，会高估该特征的作用。

**`future_threat` 默认关闭**：`future_threat_eval_enabled = false`（`params.hpp:84`），
即"再往前模拟 4 回合算残余威胁"（`terminal_eval.hpp:233-263`）这条路**默认不走**。

**闪电的反事实加成**（`rollout_score.hpp:46-76`、`:114-123`）：
对每条闪电 plan，额外跑一个"剥掉闪电操作"的对照仿真，比较敌方战斗蚁的**盾减少量**（+25）
与**掉血量**（+1.5/hp），这才是闪电被选中的真实原因（不是静态的 `lightning_static_bonus`）。

### 2.4 时间/算力使用

| 项 | 值 | 证据 |
|---|---|---|
| 目标时间 | `action_target_time_ms = 4500` | `params.hpp:7` |
| 硬时间闸 | `action_time_budget_ms = 6000`（超过就 `break`，用已有样本） | `params.hpp:13`、`evaluation.hpp:390-392`、`:426-428` |
| 采样数 | 自适应（probe 5 → 估 us/sample → 外推），名义上限 125/臂 ×1.25 | `params.hpp:8,10,11,12`；`evaluation.hpp:377-388` |
| 闪电组预算 | 总 600 次 rollout、批 2、探索 300 | `params.hpp:20-22` |
| 多线程 | **无**（4 个模式全目录 0 命中） | grep |
| 缓存 | `StaticRiskCache`（1024 条，默认开）、`MoveCacheMemo`（2048，默认开）、`ReversePathCache`（2048，**默认关**） | `params.hpp:14-19`；构造处 `evaluation.hpp:104-130` |
| 前缀共享 | **有**（两层）：① 每条臂的 `plan.ops` 只在 root 上 apply 一次，各 rollout 从 `plan_root.clone()` 起（`evaluation.hpp:152-153`、`rollout_score.hpp:110`）；② 同一投掷种子的控制组复用（`rollout_score.hpp:115-120`） |

⇒ 亚军与冠军的算力策略**完全相反**：冠军"少模拟 + 精确全引擎 + 无缓存"，
亚军"多采样 + 简化单边模型 + 三层缓存 + 时间自适应"。**单位时间的"决策质量/成本"是被亚军优化过的。**

### 2.5 先手/后手处理

**结论：同样完全对称，唯一分支是协议收发顺序。**

- 主循环：`if (runtime.player == 0) { perform_self_turn(); receive_and_apply_opponent(); }`
  vs `else { receive_and_apply_opponent(); perform_self_turn(); }`（**`ai_cpp_lure_v4.cpp:170-188`**）。
- 策略里只有镜像：`enemy = 1 - player`（`lightning_plans.hpp:9`、`offense.hpp:56/80/129/256`、
  `plan_types.hpp:100`），**无 `player == 0/1` 的战术分支**。
- 有一条与座位相关的**信息机制**：`ctx.opponent_ops_already_applied`
  （`ai_cpp_lure_v4.cpp:114`、`:86`），即"后手在决策前已经把先手本回合的操作应用到镜像里了"。
  这把 P1 的**信息优势**变成一个显式输入（P0 决策时该标志为 false）。
  ⚠️ 注意：该标志只用于 session/推断（`decision.hpp:25-27` 的 `apply_inferred_last_moves`），
  **并没有被用来给两条座位写不同策略**。

⇒ 两个对手**都没有**"先手特权/后手特化"。座位差 27–47pp 的成因必须在别处找（见 §6.2）。

### 2.6 亚军的弱点（代码级）

1. **最多 2 座非 lure 塔（`params.hpp:32-33`，生效 `base_rules.hpp:98-110`）** ⇒ 防守覆盖是硬性上限。
   对手若能用 3 路以上同时逼近，它的塔位分配必然漏。
2. **前向模型忽略对手的一切建造/出兵**（`decision.hpp:29` + `random_search_baseline.hpp:3538`）
   ⇒ 对"对手正在攒钱准备大招"完全无感知（旧报告 `findings_shadow_sim_fidelity.md:57` 已给出同样结论：
   对手只剩 `generation_level` / `ant_level` 两个标量）。
3. **只对 5 只蚁做强制走位**（`kMaxImportantAnts = 5`，`random_search_baseline.hpp:63`；
   `rollout_sampling.hpp:136-139`）⇒ 大规模蚁群时采样失真。
4. **视野 8 回合**（`long_eval_horizon = 8`，`params.hpp:24`）⇒ 对"10 回合后才见效"的经济/塔相性近视。
5. **mid-eval 关闭**（`mid_eval_weight = 0.0`）⇒ 4 回合内的中期信息全丢，只看 8 回合末的一个标量。
6. **`hold_bonus = 80` 是固定加分**（`params.hpp:40`）⇒ 在"必须立刻行动"的临界局面会偏保守。
   结合 `followup_plan_penalty = 0.0`（`:41`，默认不惩罚依赖后续回合的 plan）⇒ 倾向"等下一回合"。
7. **扁平 bandit 无法表达"先 A 后 B"的两步组合**（`evaluation.hpp` 结构上只有一层），
   `FollowupAction` 只覆盖 5 步且只能描述"延迟到第 N 回合做同一类操作"（`plan_types.hpp:26`）。
8. **单边模型下"对手塔"不存在** ⇒ 它**无法评估自己的蚁去打对手塔的收益**（进攻完全靠后处理：
   `choose_offensive_evasion/emp/ant_upgrade`，`decision.hpp:46-62`，是硬阈值规则而非搜索，
   阈值见 `params.hpp:106-116`）。

---

## §3 三者对比

| 维度 | 冠军 `magica-v3` | 亚军 `cpp_lure_v4` | 我们 `code/my_ai`（仅必要对照） |
|---|---|---|---|
| **方法族** | 手工 prior + 手工 state value + **树 MCTS(PUCT)** | 模板候选 + **扁平 UCB1** + 单边 MC rollout + 手写终端估值 | 神经网络（prior/value）+ **扁平 rollout / MC 仲裁** |
| **ML** | **0** | **0** | CNN/实体网（`network.py:116` `AntWarNetwork`，`:9-10` Value→标量 / Policy→3×23 类 logits） |
| **权重文件** | 无 | 无 | 有（`checkpoints/`，见 `docs/checkpoints.md`） |
| **候选生成** | 按目标敌蚁分层，每节点 ≤32（`:16116`） | 4 模板族穷举后 key 去重（`root_plans.hpp:175-182`），约 40 个 | 类级菜单 + 位置 top-k（`az_intent/menu3_candidates.py`），k=24/48 可配 |
| **搜索** | 树，N≤**50**，深 **9** 回合（`:26822-26823`），节点上限 10 万 | **无树**，扁平，每臂 N≈**125**，深 **8/10** 回合（`params.hpp:11,24,26`） | 扁平 rollout，N 由固定"总回合预算"反算（`flat_rollout_greedy.py:325-328`） |
| **叶子/末端评估** | 手写 state value（`my−enemy`，逐蚁时间竞速） | 手写 terminal score（血量200 + 等价金币分段 + 威胁） | 价值网 |
| **前向模型** | **全引擎 clone**（`Env{Game game_}`，`:5942`、`:24055`） | **单边简化模型**（`DefenseSimulator`，忽略敌方出兵） | 我们的 C++/Python 引擎（比对手慢，见 `docs/findings_flat_rollout_vs_tree.md`） |
| **单步预算** | 5000ms 软上限（先扣闪电，下限 2000），实际由 50 sims 卡住 | 4500ms 目标 / 6000ms 硬闸，**采样数自适应** | 固定总回合数预算，非时间自适应 |
| **并行** | 无 | 无 | 无（单步内） |
| **缓存** | 无 | 3 层（静态风险 1024 / 移动 2048 / 反向路径默认关） | 前缀树记忆化（`flat_rollout_greedy.py:336-345`，省 21–91%） |
| **座位处理** | 仅协议顺序（`:27244`） | 仅协议顺序（`:170-188`）+ 信息标志 | 我们观测到 27–47pp 座位差（`docs/prior_ceiling_20261026.md:127-133`） |
| **代码规模** | 27,462 行（单文件） | 5,652 + 5,015（SDK 仿真器）行 | 见 `code/my_ai/` |

---

## §4 可直接借鉴清单（按"落地成本 vs 预期收益"排序）

> 排序原则：先做"加特征"（不动搜索，只动数据管道）→ 再做"改预算"（动少量代码、立刻见效）
> → 最后做"改仲裁/加硬门"（会动策略分布，需重新评估）。

### ① 【最划算】把"塔血 + 等价金币 + 分段递减"补成观测特征

- **做什么**：把亚军的 `equivalent_money = coins + tower_full_salvage_value`（`core_values.hpp:93-131`）
  与它的**分段递减** `own_equivalent_money_score`（`terminal_eval.hpp:93-99`：≤400 部分 ×10、超出 ×8）
  作为一个/两个标量特征喂进我们的价值网。
- **为什么是它**：`docs/distill_top2_expressivity_plan.md:139` 已判定"塔血量不在任何通道"是"可能致命"的缺失，
  而 `docs/entity_transformer_value_results.md` 显示补上塔 HP 后价值网 R² 0.9969 / MAE 65 —— 说明**输入侧补齐的收益是实证过的**。
  亚军把"塔"折成"钱"再统一度量的做法，比我们直接用"塔数/等级和"更接近真实经济学
  （我们 `docs/findings_flat_rollout_vs_tree.md:219` 的结论也指向"经济管理是那 30 分差距"）。
- **我们改哪里**：
  1. 观测编码：`Ant-Game/SDK/utils/features.py`（board/stats 的 42 维）——**建议不改官方 SDK，而是在我们自己的 dump 编码器里追加通道**；
  2. 数据管道：`code/my_ai/collect_value_data.py`、`code/my_ai/az_intent/collect_value_prior.py`；
  3. 网络输入维度：`code/my_ai/network.py:197-200`（stats 通道 42 → 42+N）。
- **预期收益**：中高（离线指标必然改善；实战是否传导需按 `docs/prior_ceiling_20261026.md` 的教训做对外验证，别只看离线）。
- **风险**：改观测维度 ⇒ 所有旧 checkpoint 不兼容、必须重训；且"离线指标↑实战↓"我们已踩过三次（同上文档），
  **必须带 vs 亚军 16 局的外部判据**。

### ② 【次划算】"竞速式"逐蚁威胁度（冠军的时间-产能模型）

- **做什么**：照抄 `main.cpp:15922-15942` 的 `t_sim` 累积逻辑：
  把敌方工人蚁按到己方基地的距离升序排，累加"塔杀掉它需要的回合数"，
  判 `t_sim + eliminate_time > distance*1.33` ⇒ 全额损失，否则按 `(1-0.9)^fail_rounds` 折减。
  产出一到两个标量：`Σ expect_loss`、`#来不及杀的蚁数`。
- **为什么**：这是**"我能不能在它咬死我之前杀掉它"的直接判据**，而我们的 stats 里只有
  `hp_delta / hostile_distance / base_arc_coverage / tower_spacing` 这类静态代理
  （`docs/distill_top2_expressivity_plan.md:136`）。亚军对同一件事的建模是**静态距离加权**
  （`Σ 160*hp/max_hp/dist`，`params.hpp:72-73`），冠军的是**串行产能模型**——后者信息量更大。
- **我们改哪里**：和 ① 同一条管道（在 dump 编码阶段算好写进 npz），
  参考 `code/my_ai/distill_top2/lure_value_eval.py` 已有的"亚军价值复刻"结构照做一份"冠军威胁复刻"。
  需要塔的 DPS 表（可由我们的塔类型反推；冠军的 12 档表在 `main.cpp:15816-15829`，不要照抄它的 effco 阈值，
  那是它自己的经济刻度）。
- **预期收益**：中高（把一个 5 维隐式推理压成 2 个标量，网络不必自己学）。
- **风险**：依赖 ① 的 `tower_hp`；`1.33` / `0.9` / `60` 是它的手调常数（`main.cpp:15803-15808`），
  照抄等于把它的经济刻度搬过来，需在我们的价值尺度上重新标定。

### ③ 【低成本、立刻见效】按墙钟时间自适应的采样预算

- **做什么**：照抄亚军 `evaluation.hpp:352-392` 的三段式：
  ① 每个候选先跑 `probe = 5` 个样本；② 用实测 `us_per_sample` 外推剩余时间能跑多少；
  ③ 目标 = min(时间允许量 × 1.25, 名义上限)，且设一个硬闸。
- **为什么**：我们现在是**固定总回合数预算**（`flat_rollout_greedy.py:325-328`：
  `n_rollouts = min(rollouts_per_candidate, max_total_rounds/(n_cand*horizon))`），
  而我们的 Python 仿真单步耗时随局面（蚁数/塔数）能有数量级变化 ⇒
  复杂局面浪费/超时、简单局面欠采样。亚军这条正好补上。
- **我们改哪里**：`code/my_ai/az_intent/flat_rollout_greedy.py:325-330`
  （把 `per_candidate_rounds` 的静态反算替换成"probe → 估时 → 动态目标"），
  参数可放在我们的 `AZAI_*` 环境变量族里，便于做 A/B。
- **预期收益**：中（同等墙钟下采样数更贴合，且能去掉超时风险）。
- **风险**：低。但要给比赛单步超时留余量（亚军自己留了 4500/6000 的 25% 余量，`params.hpp:7,13`）。

### ④ 【中低成本】"搜索分 + 手写先验分"并列相加，而不是只靠网络 prior

- **做什么**：像亚军那样 `final_score = mean_rollout_score + plan.heuristic`（`evaluation.hpp:475`），
  其中 `heuristic` 里放**少量硬编码业务项**：`hold_bonus = 80`（`params.hpp:40`）、
  降级惩罚（sniper −400 / medic −800，`:49-55`）、C1 换位奖励（`:46-47`）。
- **为什么**：`docs/exception_audit_20261022.md:4-14` 与 `docs/findings_flat_rollout_vs_tree.md:164-165`
  都记录过"某一类动作（升级）长期 0 次执行"的病 —— 纯网络 prior 对**稀有类**没有兜底。
  亚军这套"手写分叠加"是给稀有类一个恒定偏置的手段。
- **我们改哪里**：`code/my_ai/az_intent/bundle_mcts.py` 的候选打分
  （以及 `flat_rollout_greedy.py` 的 `scores` 聚合处）；手写项集中在配置文件里，便于关掉做消融。
- **预期收益**：中低—中（对"整类动作从不出现"有效，对整体棋力未必）。
- **风险**：中。手调常数容易过拟合到某个对手（我们在 `docs/action_net_retrain_history.md:50`
  已经吃过"对 ExampleAI 90.6% / 对 rule_v4 1.6%"的亏）。**每一项都要能一键关闭。**

### ⑤ 【中成本】"硬否决门"（冠军的 `prior_gate`）

- **做什么**：照抄 `main.cpp:20059-20061`：若候选导致"排序在目标之前的蚁"受到的伤害变差，
  直接把该候选 `prior_gate_ok = false` 淘汰。
- **为什么**：这是一条**可解释的安全约束**（"不许为了救后面的蚁而牺牲前面的蚁"），
  与 ④ 的加分不同，它是**乘性硬门**，能挡住"看起来短期收益高、实则拆东墙"的候选。
  我们的候选过滤目前主要靠 ckpt 的 prior 分布，没有这种结构性保证。
- **我们改哪里**：`code/my_ai/az_intent/menu3_candidates.py`（菜单生成期过滤）
  或 `bundle_mcts.py` 的 `_expand`（更靠近冠军的位置），需要一份"按威胁排序的敌蚁列表"。
- **预期收益**：中（在"多蚁同时压上"的局面能避免明显坏棋）。
- **风险**：中高。硬门会剪掉创新走法，且与 ④ 的先验偏置可能互斥；建议**默认关、按局面开启**，并做消融。

### ⑥ 【低收益、不建议现在做】PUCT + prior softmax(T=3) 的树搜索

- **做什么/依据**：冠军 `pb_c_base=19652 / pb_c_init=1.25`（`:16218-16219`）、
  PUCT（`:24503`）、`kMaxActions` 宽的 softmax 温度 3.0（`:17062`、`:22647`）。
- **为什么排最后**：我们已有内部证据"flat rollout ≥ 树"（`docs/findings_flat_rollout_vs_tree.md`），
  且我们的单步仿真比 C++ 慢两个数量级 ⇒ 冠军"50 sims × 全引擎 clone"这套在我们这儿
  **一次决策都做不完**。抄它等于抄"我们没有算力支付的那部分"。
- **我们改哪里**：`code/my_ai/az_intent/mcts.py` / `bundle_mcts.py`（若将来算力/仿真提速再谈）。
- **预期收益**：低（现阶段）。
- **风险**：高（时间超预算 + 我们的价值网方差比它的手写价值大，树搜索会被噪声带偏）。

### 明确"不值得抄"的一项

**冠军的 RNG 序列控制**（LCG 复刻 `:2125-2127`、`:3613`、`use_raw_rng_state` `:3637`）。
旧报告 §3 已判定它为"逐 seed 的确定性规划、不泛化"，我确认并加一条：它**依赖对裁判实现细节的完全复刻**，
一旦平台换了随机数实现就整体失效 —— 这是把 AI 绑在特定实现上的高风险设计，不适合我们。

---

## §5 对手弱点清单（可被针对性利用的具体模式）

> 每条都给"机制 → 证据 → 怎么打 → 置信度"。
> ⚠️ 合规前提：全部限定在**规则内的对局策略**。任何靠"制造 desync / 拖超时 / 发非法操作流"的路径都**不建议**
> （见 §5.9）。

### 5.1 冠军：`prior_gate` 的交换盲区（置信度：高）

- **机制**：目标蚁**之前**的蚁只要多受一点伤，候选就被 `prior_gate_ok=false` 淘汰（`main.cpp:20059-20061`）。
- **怎么打**：同时在**两个方向**压上，且让"先到的那只"故意显得威胁小（例如先派一只血厚但慢的），
  迫使它在"必须先处理 A、但处理 A 会漏 B"的局面上，候选集被剪空、退化到 pass 或低质量 arrangement。
- **可用性**：中——需要我们能同时维持两路有威胁的兵线（我们的出招率历史上比对手低很多，
  见 `docs/experiment_log.md:1643`，这是我们的弱项，不是它的）。

### 5.2 冠军：非回合边界无独立估值 ⇒ 同回合内的动作不可区分（置信度：高，但难利用）

- **机制**：`:24232-24237` 回落成 `MeanValue(round_root_index)`。
- **含义**：一个回合里它先做哪个 op 只由 prior 决定；如果 prior 对该回合的"最佳首步"判断错，
  整条路径都会错（搜索无法纠正）。
- **怎么打**：难直接利用；但它解释了为什么它对**顺序敏感**的骚扰（比如卡它的造塔位）有效。

### 5.3 冠军：备份不翻符号 ⇒ 对"会被反制"的局面不敏感（置信度：中）

- **机制**：`:24603-24604` 全深度存根视角值。
- **怎么打**：做"看起来我亏、但能引出它过度投入再反打"的局（诱饵类战术）。
- **旁证（弱）**：亚军自带一整套 `lure` 诱饵族（`lure_plans.hpp:7`、`is_lure_slot_code`、`forced_lure_sell_distance = 1`），
  说明"用廉价塔位操纵对手寻路"在这个游戏里是**已被验证有效的战术范式**；
  但我**没有**任何证据表明它专门用来打冠军——这是外推，不是事实。

### 5.4 亚军：非 lure 塔上限 2 ⇒ 防守覆盖有硬顶（置信度：高）

- **机制**：`max_non_lure_towers = 2`、`rich_max_non_lure_towers = 2`（`params.hpp:32-33`），
  生效点 `base_rules.hpp:98-110`。
- **怎么打**：三路以上同时逼近（或"一路佯攻 + 两路真压"），它的塔位分配必然漏掉一路。
  这也是它为什么要用 lure（把蚁引到已有的塔位）来补偿。

### 5.5 亚军：看不到对手的建造/攒钱（置信度：高）

- **机制**：`ignore_enemy_spawns = true`（`decision.hpp:29`），且 `DefenseSimulator` 里根本没有对手塔；
  对手只剩 `generation_level` / `ant_level` 两个标量（`docs/findings_shadow_sim_fidelity.md:57`）。
- **怎么打**：**长时间囤钱 → 一次性爆发**（连续建塔/连续升级/配合闪电）。
  它的 rollout 里这段完全不发生，因此"我囤到 600 再动手"在它的估值里是免费的。
  `docs/findings_flat_rollout_vs_tree.md:219` 的观察（"对手常囤到 300–600 金币做一次性大动作"）
  正是这个机制的对偶面 —— 说明**它自己就这么打，也最怕这个**。
- **可用性**：高。这是我们最该利用的一条。

### 5.6 亚军：`hold_bonus = 80` 与 `followup_plan_penalty = 0` ⇒ 偏保守/偏拖延（置信度：中高）

- **机制**：`params.hpp:40-41`。hold 有固定加分，依赖后续回合的 plan 反而**不**被罚。
- **怎么打**：制造"看起来还能再等一回合"的假象（例如把兵线维持在一个刚好不致命的位置），
  让它的 80 分把决策推向 hold；下一回合我们完成爆发。**需要有耐心/能控节奏的战术。**

### 5.7 亚军：只对 5 只蚁做强制走位（置信度：中）

- **机制**：`kMaxImportantAnts = 5`（`random_search_baseline.hpp:63`、`rollout_sampling.hpp:136-139`）。
- **怎么打**：**大规模蚁群**（>5 只有威胁的蚁同时在场）时它的 rollout 会失真，
  它对这些蚁的移动预测退化为默认分布。⇒ 用"多蚁海"战术逼它在失真的模型上做决策。

### 5.8 两者的共同盲区：都不做"对手建模"（置信度：高）

- 冠军：MCTS 里对手的走法由 `Env` 的默认策略推进（它没有对手的 prior/value）；
- 亚军：单边模型直接忽略对手。
- ⇒ 两者对"**非标准的、反常的**走法"都没有专门的应对。这是"我们如果下出它们没见过的棋"的唯一优势来源。
  但注意：这不等于"乱下能赢"——`docs/champ_ladder_plan.md:21` 的实测是冠军 vs 亚军 4:5
  （9 局全打满 512 回合），说明**同一水平线上的两个不同风格已经打平**；
  我们的问题不是"没被它们见过"，而是"绝对强度差 30 分"（`docs/findings_flat_rollout_vs_tree.md:217-219`）。

### 5.9 ⚠️ 不建议使用的弱点（合规/稳定性风险）

- **冠军的 shadow env desync → force_pass**（`main.cpp:26748-26760`；我们已在 `docs/experiment_log.md:3854` 复现过）。
  主动制造 desync 属于**操纵裁判/协议**，且冠军一方"整局 pass"会让对局形态失去意义。
  **列为"它的脆弱点"，不作为战术。**
- **亚军的时间预算**：`action_target_time_ms = 4500` / `action_time_budget_ms = 6000`（`params.hpp:7,13`）
  ⇒ 它对墙钟敏感。但**拖满它的预算是靠环境/平台实现，不是靠棋**，同样不建议。

---

## §6 诚实边界

### 6.1 我读到的代码事实（可直接引用）

- 冠军：27,462 行单文件；`max_simulations = 50`（`:26822`）、`max_round = +9`（`:26823`）、
  `kMagicaMctsMaxNodes = 100000`（`:16113`）、每节点 32 候选（`:16116`）、每 move 8 op（`:16120`）、
  PUCT 常数与公式（`:16218-16219`、`:24496`、`:24503`）、备份不翻符号（`:24603-24604`）、
  state value 全公式与全部常量（`:15803-15808`、`:15878-15950`、`:16052`）、
  prior 公式与 `prior_gate`（`:20059-20067`）、预算 5000/2000ms（`:24901-24902`、`:24914-24917`）、
  座位分支只在协议层（`:27244-27284`）。
- 亚军：`params.hpp` **全部 60 余个手调常数**（`:5-117`）、UCB1 与动态预算（`evaluation.hpp:286-460`）、
  `evaluate_terminal` 全公式（`terminal_eval.hpp:393-445`）、`c1_terminal_bonus` 恒 0（`:89-91`）、
  `own_equivalent_money_score` 分段（`:93-99`）、`tower_full_salvage_value`（`core_values.hpp:93-131`）、
  rollout 深度与 mid-eval 关闭（`params.hpp:23-26`、`rollout_score.hpp:128-155`）、
  单边模型与 `kMaxImportantAnts=5`（`decision.hpp:29`、`random_search_baseline.hpp:63/1029/3527`）、
  `base_codes()` = 15（`core_format.hpp:406`）、建塔上限 2（`params.hpp:32-33`、`base_rules.hpp:98-110`）。
- 两侧**均无** ML 框架、权重文件、训练脚本（grep 全目录，0 命中；唯一命中是我们自己注入的
  `entity_value_net.hpp`，且 `git status` 显示它未跟踪）。
- 两侧**均无**多线程（4 个模式 0 命中）。

### 6.2 推断（有依据但**不是**读到的常量）—— 请勿当事实引用

1. **"亚军每步约 40 个候选"**：源码没有这个常量（`RootPlanSet` 只有计数字段，`root_plans.hpp:189-193`），
   40 来自其运行日志（旧报告转述）。实际数字随局面变化（取决于 15 个 base 槽位、建塔上限、去重后剩多少）。
2. **"亚军每局约 1.9 万次价值评估"**（`terminal_eval.hpp:267` 的注释自述）：这是**亚军源码里的注释**，
   不是我们实测；且注释未说明统计口径（是每步还是每局）。
3. **"备份不翻符号 ⇒ 近似 max 而非 minimax"**：机制（不翻符号）是事实（`:24604` 注释自述），
   "等价于 max"是我的解释性推断（严格说它依赖叶子估值已成根视角，因此是否算缺陷取决于价值函数的定义域）。
   我按"可能缺陷"列入 §5.3，并标了置信度中。
4. **`tower_value_weight = 10`（`params.hpp:67`）是否真的生效**：我在 `terminal_eval.hpp:397-404`
   只看到 `tower_value_score` 与 `money_score` 按 **比例** 拆分 `own_equivalent_money_score` 的结果，
   没看到这个权重被直接乘上去。⇒ **它可能是历史遗留的未使用参数**，但我没有穷举全 27 个头文件确认，**不下结论**。
5. **"冠军/亚军没有座位特化"**：我 grep 了 `player == 0/1`、`self_player`、`root_player` 等，
   只找到协议顺序与镜像。但我**没有逐行读完** 27,462 行（实读约 1,600 行 ≈ 6%）⇒
   "完全没有任何座位相关策略差异"这句话，我只能担保"在我读到的范围内没有"。
6. **座位差 27–47pp 的成因**：`docs/prior_ceiling_20261026.md:131-133` 的归因是
   "后手能看到先手的出招（`match_via_sdk.py:436`）"，那是**我们自己的对局管线**的属性；
   本文只确认"两个对手都没有座位特化"，**不**替那份文档背书其归因。

### 6.3 读不到 / 未覆盖

- `antgame_sdk/native_sim.hpp` 的实现（pimpl 进二进制，`:103`）—— 官方 native 引擎的细节读不到。
- 冠军 `Env`（`:5855-11420`）与 `bundled_data`（`:5962` 起）未逐行：其中的距离表/地块常量如何生成，
  以及 `Env` 对"裁判随机数"的复刻精度，**未验证**。
- 亚军 `random_search_baseline.hpp`（5,015 行）只读了结构 + `simulate_round` 主循环；
  塔伤害/蚁行为/传送的具体公式**未逐行**（因此 §2.2 关于仿真保真度的结论依赖
  `docs/findings_shadow_sim_fidelity.md` 的既有工作，而非我独立复核）。
- 我们自己的 `code/my_ai` **按任务要求只做必要对比**（`network.py` 的头部、`flat_rollout_greedy.py` 的预算段），
  §3 里"我们"那一列**不是**对我方架构的完整描述，请以 `docs/model_design.md`、`docs/checkpoints.md` 为准。

### 6.4 与旧报告的 3 处差异（本文的修正）

| # | 旧报告说法 | 本文修正 | 依据 |
|---|---|---|---|
| 1 | 亚军"rollout 打分"，未指出前向模型在哪 | 前向模型是 `rs::DefenseSimulator`，在 `antgame_cpp_sdk/.../random_search_baseline.hpp`（**5,015 行，不在 `cpp_lure_v4/`**），且**单边 + 忽略敌方出兵** | `decision.hpp:29`、`random_search_baseline.hpp:63/1016/3527/3538` |
| 2 | 把 `c1_bonus` 当作亚军终局估值的活跃项 | `c1_terminal_bonus` **恒返回 0**（`:89-91`）；C1 只以 root 层加分形式存在（`evaluation.hpp:156/229-232`） | 同左 |
| 3 | 亚军"4/8 回合 MC rollout" | 实际 **`mid_eval_weight = 0.0`** ⇒ 4 回合中期评估**不生效**，只有 8 回合（闪电 10 回合）末端 | `params.hpp:23-26`、`rollout_score.hpp:128-155`、`terminal_eval.hpp:447-455` |

---

## 附：一句话总结

两个前两名都是**零机器学习的手写程序**，但它们的钱花在完全相反的地方：
**冠军用手写先验 + 手写价值 + 50 次精确全引擎 MCTS 买"深度"**，
**亚军用手写模板 + 单边简化仿真 + 时间自适应的上千次采样买"广度"**。
对我们的直接含义是：**"更快仿真"这条路亚军已经走到极致（简化模型 + 3 层缓存 + 时间预算），
而我们短期抄不动它的 C++ 速度；但"更好的评估"这一侧，两者缺的东西（塔血/等价金币、逐蚁时间竞速）
恰恰是我们最便宜能补的两块**（§4 ①②）。
