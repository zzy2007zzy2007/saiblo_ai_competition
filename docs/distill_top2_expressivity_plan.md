# 计划：蒸馏前两名（冠军/亚军）做「架构表达力」可行性验证

> 2026-10 新方向（**独立于旧的 goal 章程**，见 §7 合规说明）。
> 目的：**不为了刷分，而是回答一个被旧章程挡住、但其实必须回答的问题**——
> "我们现有的网络/架构，表达能力到底够不够追上**非机器学习方法的 SOTA**？"
> 如果连"监督式蒸馏一个强老师"都拟合不出来，那自己从头 RL（信号更弱）只会更难；
> 如果蒸馏得出来，至少说明**瓶颈在训练/发现，不在表示**。

---

## 0. 一句话动机（用户想法，已确认）

- 前两名（冠军 `magica-v3`、亚军 `saiblo-30th-AI/Game1`）都是**零机器学习**的规则/搜索程序（见 §1）。
- 用 ML 蒸馏它们，**不一定做行为克隆**，也可以**蒸馏它们的「先验」和「价值函数」**。
- 判读：
  - 蒸馏成功 ⇒ 架构**表达力够**，主要问题在训练 ⇒ 可以放心投 RL；
  - 蒸馏失败 ⇒ 架构连"表示"都做不到 ⇒ 自己 RL 更难（但先要排除"输入缺信息"这个混淆，见 §5）。

⚠️ **一个要提前说清的技术限定**：蒸馏是**稠密监督**（每一步都有目标），RL 是**稀疏监督**（只有局末胜负）。
所以"蒸馏成功"只证明**能表示（representable）**，不证明**能被稀疏奖励发现（discoverable）**。
即：**蒸馏失败 = 强负信号；蒸馏成功 = 弱正信号（必要不充分）**。

---

## 1. 前两名的方法（已读完代码，带 `文件:行号`）

### 1.1 冠军：`其他版本ai/ant-war2-magica-v3`（作者 yanlifeng，"小小预言家-v3"）

单文件 `main.cpp`（1.48 MB，多文件合并产物）。作者在 `readme.md` 自述核心点（按重要度排序）：

| 作者自述 | 代码证据 |
|---|---|
| **RNG 序列控制**（通过操作控制随机数消费序列；手段=杀蚂蚁时机、闪电；可控内容=蚂蚁行为/生产类型） | `main.cpp:2125-2127` 复刻了游戏引擎的 LCG（`MULT=25214903917, INC=11, MASK=2^48-1` = Java LCG）；`:3613` `rng_state=(seed^MULT)&MASK` 由游戏 seed 播种；`:3637` 支持 `use_raw_rng_state` 直接设 rng 状态、`:2026/:3633` 可克隆并前向模拟 RNG |
| **MCTS 回合分层** `key=(round, target_ant_id)` | `:20691/:21005/:21053` 日志 `[MCTS:Expand:CandidateSearch] key=(round, ant_id)` |
| **Plan 结构**（解决 pass 的 prior logit 不平衡） | `:22599` `pass_arrangement.score.prior_logit = 0.0` + prior softmax `:22640-22675` |
| **state value 设计** | `:15694-16069`（`src/state_value.cpp` 内嵌）；`ComputeStateValue = my.state_score - enemy.state_score`（`:16052`） |
| **ForcedSet 设计** | `:18961` `BuildRoundInitCombatLureSelection`、`:18108` `[MCTS:Expand:ForcedDowngrade]`、`:18239` `ForcedGiveUp`、LureSet |
| **prior 与 state value 分工** | `readme.md:23-26`：prior 求快求全（别漏最优），state value 求"逻辑推理否定第一感" |

冠军 state value 的关键常量（`:15803-15808`）：
`kStateValueBaseHpEconomicLoss=60`、`kStateValueAntArriveCoin=10`、`kStateValueDistanceLooseFactor=1.33`、
`kStateValueHp2Coin=60`、`kStateValueCombatAntValue=1.5`、`kStateValueHitSuccessProbability=0.9` + 12 档 `DpsTier` 表。
prior 关键常量（`:17061-17062`）：`kEffcoPriorLogitWeight=0.25`、`kPriorSoftmaxTemperature=3.0`；`prior_logit = total_contribute + effco_delta_logit`（`:20067`）。

### 1.2 亚军：`其他版本ai/saiblo-30th-AI/Game1`（作者 omegafantasy，历届连冠选手）

`antgame_ai_cpp/cpp_lure_v4/`（"lure 诱饵"策略）。**方法族 C：模板候选 + 扁平 UCB1 bandit + 4/8 回合 MC rollout + 手写终点评估，无树**。

决策管线（`lure_strategy_v4_decision.hpp:7-64`）：
```
generate_root_plans(state, defense_root, player)   # 模板候选（先验）
  -> evaluate_root_plans(...)                        # 扁平 UCB1 bandit + rollout 打分
  -> 取最优 plan -> 追加进攻性 EMP/回避/升级后处理
```

价值函数（`lure_strategy_v4_terminal_eval.hpp:255-288` `evaluate_terminal`）——**干净的 state→标量手写函数**：
```
total_score = base_hp * base_hp_weight                       (200)
            + stepped_own_equivalent_money_score              (等价金币=金币+全塔最优回收价值)
            + c1_bonus
            - worker_threat
            - combat_threat
            + future_threat_adjustment                        (可选，默认关)
```
其中：
- `worker_threat = Σ worker_threat_unit(160) * hp/max_hp / max(1, hex_dist(ant, own_base))`（`:params.hpp:72-73`）；
- `combat_threat = Σ max(base_threat, core_tower_anchor_threat) * behavior_scale`（`random→0.6, bewitched→0.3`）；
- 全部权重集中在 `lure_strategy_v4_params.hpp`（`base_hp_weight=200` `:66`、`money_weight=10` `:68`、`money_decay_threshold=400` `:69`、`worker_threat_unit=160` `:73`、`combat_base_threat_unit=300` `:76`…）。

### 1.3 共同点与差别

| | 冠军 magica-v3 | 亚军 lure_v4 |
|---|---|---|
| 搜索骨架 | **MCTS（回合分层，有树）** | 扁平 UCB1 bandit（**无树**） |
| 先验 | 手写 `prior_logit`（softmax T=3.0） | 模板候选 `generate_root_plans` + 手写 heuristic |
| 价值 | 手写 `state_value = my - enemy`（逐蚁精细推理） | 手写 `terminal_score`（线性加权） |
| **额外** | **RNG 序列控制（LCG 复刻 + 前向模拟）** | 无 |
| ML | 0 | 0 |

---

## 2. 可蒸馏目标（具体到函数）

| 目标 | 冠军 | 亚军 | 可蒸馏性 |
|---|---|---|---|
| **价值函数** | `ComputeStateValue(state)` → `my.state_score - enemy.state_score` | `evaluate_terminal(state)` → `total_score` | ✅ 都是 state→标量；**亚军最干净（首选）** |
| **先验** | `prior_logit(candidate)`（contribution + effco_delta） | `root_plans` 候选 + `plan_heuristic` | ✅ 候选→logit；但候选空间结构复杂（变长 op-list） |
| **RNG 序列控制** | LCG 复刻 + `use_raw_rng_state` + 杀蚁/闪电时机 | 无 | ❌ **不可作为 Markovian state→action 蒸馏**（见 §3） |

---

## 3. 用户结论的确认与一处修正

**确认（用户直觉对）**：
1. 冠军的 **RNG 序列控制**很难模仿 ✓——它不是"state→action"的纯函数，而是"**通过操作序列消费随机数流，让未来的随机事件（蚂蚁移动/攻击/传送/生产类型）落在有利分支**"的控制流能力。要蒸馏它，网络得内部学会**复刻引擎的 LCG + 精确的 RNG 消费顺序**，本质是逐 seed 的确定性规划，**不泛化**（换个 seed 就失效）。
2. 亚军相对"正常"，更易模仿 ✓——它就是"候选 + bandit + 手写线性估值"，没有 RNG 控制。

**修正（一处）**：
- 说"冠军 RNG 难模仿"是对的，但要**区分**：冠军的 **MCTS + prior + state value 这三件其实和我们的架构同构**（我们也是 prior + value + MCTS），**只有 RNG 控制那一件难**。所以对冠军，可以**只蒸它的 state value / prior**，跳过 RNG。
- 两者价值函数都是**"手写特征的线性/查表组合"**，理论上是可学的。真正的问题是：**我们的输入表示（board/stats）有没有包含计算这些手写特征所需的信息**（§5）——这个才是表达力验证的关键。

---

## 4. 蒸馏的具体方案（先做亚军价值函数，最干净）

### 4.1 数据生成
- 用**引擎跑大量局面**（随机 / 自对弈 / 前两名自己的对局），在每个决策点调用亚军 `evaluate_terminal`（或其等价实现）得到 `(state, score)` 对。
- 复用现有采集设施：`az_bridge_ai.py` 的 `AZAI_DUMP_RAW_DIR` 已能 dump `(board, stats, player)` 原始观测（`az_bridge_ai.py:74-88`）；只需在 dump 时**并行调用亚军的 evaluate_terminal** 存 score 标签。
- **训练/评测分离**：数据 seed 与判据 seed `7..70` / 随机段**不重叠**。

### 4.2 训练（单变量）
- 冻结 `class_state` / `pos_state`，**只训 `value_state`** 回归 `total_score`（MSE）。
- 对照组：① 官方 SDK 的线性 `FeatureExtractor.evaluate`（`features.py:280-309`）当"线性特征基线"；② 常数预测。看我们的价值网能否**显著超过线性基线**并逼近手写 score。

### 4.3 判读（预注册）
- **蒸馏成功**（离线 MAE 显著 < 线性基线，且 R² 高）⇒ 价值网架构 + 输入能表达这份手写估值 ⇒ 表达力够，瓶颈在发现。
- **蒸馏失败** ⇒ 先跑 §5 的输入审计，区分"输入缺信息" vs "架构不足"。

### 4.4 后续步骤（按优先级）
1. 亚军**价值函数**蒸馏（本轮）。
2. 亚军**先验**蒸馏（`root_plans` 候选分布 → 位置头/类头的先验）。
3. 冠军 **state value / prior** 蒸馏（跳过 RNG）。
4. （明确**不做**）冠军 RNG 序列控制——判定为逐 seed、不泛化，不追。

---

## 5. ⚠️ 关键：先审计我们的输入编码，再下"表达力"结论

我们的观测 = `board`（28 通道 × MAP × MAP）+ `stats`（43 维），来源 `Ant-Game/SDK/utils/features.py`：

**board 已含**：
- 塔：位置（ch4/5）、等级（ch6）、射程（ch7）、冷却（ch8）、伤害（ch9）
- 蚁：位置（ch10/11，血量按格**累加**）、等级（ch12）、年龄（ch13）、冻结（ch16）、行为 random/bewitched/control_free（ch17-19）
- 地形（path/highland/barrier，ch0-3）、信息素（ch14/15）、超武器效果（ch20-27）

**stats 已含**：`hp_delta`、`coin_ratio`、`safe_coin`、塔数/等级和、`kill_delta`、`old_delta`、`hostile_distance`、`base_arc_coverage`、`tower_spacing`… + extras 里 `coins[双方]/300`、`base_hp[双方]/50`、8 个武器冷却、超武器使用数等。

**缺失（可能致命）**：
1. **塔血量（tower.hp）不在任何通道** —— 亚军的 `tower_full_salvage_value`（回收价值）**直接依赖塔 HP**（`core_values.hpp:93-135`）。缺它 ⇒ 网络无法精确算"等价金币"。
2. **塔类型 / C1 锚点语义**未显式编码（只有等级/射程/伤害的弱代理）——亚军的 `c1_bonus`、`combat_threat` 的 anchor 概念依赖它。
3. **蚁血量按格累加**（`+= hp/max_hp`）——同一格多只蚁会混在一起，丢失逐蚁精度（冠军的逐蚁"消除时间/伤害窗口"推理需要逐蚁状态）。

⇒ **结论**：若蒸馏失败，**先补这些特征再试一次**，才能区分"输入缺信息" vs "架构表达力不足"。这一步本身也有价值——它能告诉我们"架构到底需要哪些输入"。

---

## 6. 成本估计（粗略）

| 步骤 | 成本 |
|---|---|
| 数据生成（亚军价值标签，~1-3 千局 × dump） | 数小时（复用采集设施） |
| 训练 value_state 回归（单变量） | ~1-2 h |
| 输入审计 + 特征补丁 | ~半天（若需要） |
| 冠军 value/prior 蒸馏 | 视亚军结果而定 |

---

## 7. 合规说明（新方向，与旧章程解耦）

- 旧 goal 章程（`goal_longrun_plan.md` / `task_ruleV4_90_charter.md`）里的**反捷径①（不许蒸馏前两名）是上一个 agent 任务的约束，本方向明确与之解耦**（用户 2026-10 确认）。
- 本方向**目的不是刷分/提交**，而是**表达力可行性验证**；蒸馏结果**只当"架构诊断"**，不据此宣布任何里程碑达成。
- 仍遵守两条硬约束：① **训练/评测分离**（不碰判据 seed 的对局数据）；② **不把前两名代码搬进 submission**（`其他版本ai/CLAUDE.md` 与 AAArena 说明里的合规提醒）。
- 参考元文档：`其他版本ai/CLAUDE.md`、`其他版本ai/omegafantasy的方法考古.md`、`其他版本ai/AAArena说明.md`。

---

## 8. 第一轮执行结果（已完成）

- [x] **`evaluate_terminal` 能否独立调用**：不能直接调（C++ header，依赖 `rs::DefenseSimulator`），
  但它是**纯 state→标量函数**，已用 Python 逐条复刻在 `code/my_ai/distill_top2/lure_value_eval.py`
  （价值公式/权重/threat/回收价值/槽位码，全部带 `文件:行号` 对照）。
- [x] **`MAP_SIZE` / 塔 HP**：`MAP_SIZE=19`（`EDGE=10`）；塔 HP 在 SDK 里有（`model.py:250`），
  但**不在 board 编码里**（`features.py` 只编码塔位置/等级/射程/冷却/伤害）⇒ 已把 `tower_hp` / `tower_type`
  两张辅助图一并 dump，方便离线测"补塔HP"变体。
- [x] **数据生成脚本**：`code/my_ai/distill_top2/gen_value_dataset.py`。
  状态来源 = **C++ 引擎 `GameStateFacade`**（权威；不用官方 Python 引擎，见下"引擎坑"）。
  已生成 `training_history/distill_top2/value_ds_12g.npz`：**5942 样本 / 12 局**，
  value −168 ~ 11018（std 2801），塔类型覆盖 `[0,1,2,3,4,12,22]`（含三级塔）。
- [ ] **下一步**：§4 的亚军价值函数蒸馏（只动 `value_state`，回归 `value`）+ 判读。

### 8.1 ⚠️ 引擎坑（用户 2026-10 提醒，已按此调整）

**官方 `Ant-Game` 的 Python 引擎与 C++ 引擎对不上；C++ 引擎才符合规则；README 规则也有错。**
⇒ 本方案**状态一律取自 C++ 引擎**（`GameStateFacade`，`code/my_ai/az_intent/game_state_facade.py`，
底层是 `code/cpp_engine` 的 native binding）；官方 Python 引擎**只用于 SDK 常量/几何/特征编码**（这些是固定的）。

### 8.2 运行前提：`native_game.pyd` 是构建产物

`GameStateFacade` 依赖 `code/cpp_engine/native_game.pyd`（`build.sh` 产出，**gitignore、不跟仓库**）。
本工作区已从主工作区拷入 `native_game.pyd` + 3 个 MinGW DLL；用**训练环境 python**（`pytorch-gpu`，3.11）导入。
若要重建：改 `build.sh` 里的 `REPO` 指向本工作区再跑（或直接从主工作区拷）。

