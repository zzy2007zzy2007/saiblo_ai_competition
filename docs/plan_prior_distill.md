# 计划：把亚军的**先验（prior）**抽成可训练标签，判决"我们的架构能不能复刻它"

> 2026-10-20 起。这是**主线**的最后一段。价值函数一侧已有结论（R² 0.9969 / 0.9987），
> 先验一侧一直缺工具，本文档记录打通方式。

## 1. 主线是什么（对齐一下定义）

用户原话：**"我们的主线是实验当前架构能不能蒸馏/复刻出亚军的先验和价值函数，来验证我们的架构表达力够不够"**。

亚军每一步的决策分两段：

```
final_score(plan) = heuristic(plan)  +  mean_rollout_score(plan)
                     ^^^^^^^^ 纯函数   ^^^^^^^^^^^^^^^^^^ 搜索（每候选 ~125 次 rollout）
```

- **prior = `heuristic`**。它是**纯函数、µs 级**、不依赖任何随机数，逐项可解释：
  `heuristic = base_heuristic + lure_heuristic + lightning_heuristic
             − operation_penalty − followup_penalty`
  （`lure_strategy_v4_root_plans.hpp:223-224`）
- **value = `evaluate_terminal`**（rollout 终止时对局面的打分），已蒸馏完成。
- **rollout 是 search，不是 prior**。这点之前混淆过，已澄清。

⇒ 所以主线要回答的是一个**回归问题**：
> 给定状态 + 候选动作，我们的网络能不能拟合出亚军 `generate_root_plans` 给出的 `heuristic`？

R² 高 ⇒ 架构表达力够（能装下这份先验）；R² 低 ⇒ 装不下，是我们架构的真正瓶颈。

## 2. 为什么之前卡住（数据链路）

要拿标签必须在**我们的状态**上跑亚军的 `generate_root_plans`。原计划绕道 replay：

```
我们的 state_dump.json → replay.json → sdk_lure_inspector(plan_rollouts) → 标签
```

replay 格式有三层坑（`load_replay_round_start` 要求 `[{seed,op0,op1,round_state?}]`，
`round_state` 又必须是**对象** schema，而我们的 dump 是**数组**），且 replay 只能表达
"从 seed 重新推演"这一条路径 —— 我们的中局状态**没有 op 历史**，根本写不进去。

**结论：这条路是死的，绕开它。**

## 3. 现在的做法：直接吃状态 JSON

新工具 `code/my_ai/distill_top2/cpp_test/dump_plan_prior.cpp`：

```
stdin  每行一个状态（就是 ents_seed*.jsonl 里的那种数组 schema）
  {"seed":7,"player":0,"round_index":12,
   "coins":[c0,c1],"bases":[[hp,gen,ant],[hp,gen,ant]],
   "towers":[[id,player,x,y,type,hp,cd],...],
   "ants":[[id,player,x,y,hp,max_hp,kind,age,level,behavior,status,frozen],...],
   "weapon_cooldowns":[[5],[5]],"super_weapon_usage":[2],
   "active_effects":[[type,player,x,y,dur],...]}
stdout 每行一个结果
  {"ok":1,"round_index":12,"n_plans":71,"n_base":..,"n_lure":..,"n_lightning":..,
   "plans":[{"key":..,"name":..,"heuristic":..,"base_heuristic":..,"lure_heuristic":..,
             "lightning_heuristic":..,"operation_penalty":..,"followup_penalty":..,
             "lightning_static_bonus":..,"has_lightning":..,"horizon":..,
             "followup":"..","ops":[[type,arg0,arg1],...]}]}
```

链路：数组行 → `PublicRoundState` → `NativeSimulator(seed).sync_public_round_state()`
→ `PublicState` → `rs::make_defense_simulator` → `ls::generate_root_plans`
→ 每个候选的 `heuristic` 分解。**不需要 replay，不需要 op 历史，不需要 rollout**，
所以每个状态只要 µs 级、可以流水线刷百万条。

`ignore_enemy_spawns = true` 与 `sdk_lure_inspector.cpp:1263` 保持一致（亚军评估 root plan 时也是这么设的）。

## 4. 数据来源

`training_history/distill_top2/abl_linear2_raw/ents_seed*.jsonl`：亚军 vs 亚军 linear 变体的
**逐回合完整状态**（64 局 × 512 回合 = 32768 个状态，奇偶两方都要）。分布就是亚军自己的
部署分布，正好。

- 只取 `player` = 该回合实际决策的一方（每回合两个玩家各一份，都要跑一遍标签）。
- 需要更多数据时可以再用 `MVS_RAW_STATE_DIR` 采（桥接已支持逐回合 dump）。

## 5. 训练与判决

1. 标签：每 (状态,玩家,候选计划) 一条 `heuristic`。预计 32768 × 2 × ~70 ≈ 460 万条。
2. 特征：沿用 entity Transformer 的实体编码 + 候选 op 编码（`az_intent`）。
3. 目标：回归 `heuristic`（也试 pairwise 排序，因为决策只用 argmax）。
4. **判决**：
   - 同一状态内的 R² / Spearman；
   - **top-1 一致率**：我们的 argmax 与亚军 heuristic 的 argmax 是否同一个候选
     ← 这才是"复刻先验"最贴近决策的指标（价值那侧我们就是栽在这个指标上：候选差 ~8 vs MAE ~80）。

## 6. 不做

- 不复刻 `mean_rollout_score`（search，不是 prior；要复刻它等于复刻整个搜索引擎）。
- 不再碰 replay → inspector 那条链路。
