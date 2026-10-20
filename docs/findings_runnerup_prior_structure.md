# 发现：亚军"先验"的真面目（3 值阈值规则），以及它对主线判决的影响

> 2026-10-20。工具：`code/my_ai/distill_top2/cpp_test/dump_plan_prior.cpp`
> + `code/my_ai/distill_top2/dump_prior_labels.py` + `analyze_prior_decomposition.py`
> 计划文档：`docs/plan_prior_distill.md`

## 0. 一句话

**亚军的"先验"不是价值函数那种平滑打分，而是一个 3 值阈值规则：**
`heuristic = 80·[不动] − 400·[降级狙击塔] + 0（其余）`。
它的作用等价于"**除非搜索证据比不动好 80 分以上，否则别动**"——
实际决策里 **82.6% 就是不动**。

## 1. 工具怎么来的（顺便关掉一条死路）

原来的链路是 `我们的状态 → replay.json → sdk_lure_inspector`。这条路是死的：

- `load_replay_round_start` 只能**从 seed 逐回合推演**，而我们手上只有中局状态快照，没有 op 历史；
- 它要的 `round_state` 是**对象** schema，我们的 dump 是**数组** schema，直接报
  `cannot use value() with array`；
- 折腾两小时的结论：**不要 replay**。

改成新工具 `dump_plan_prior.exe`：状态 JSON（数组 schema）→ `PublicRoundState`
→ `PublicState::sync_public_round_state` → `rs::make_defense_simulator`
→ `ls::generate_root_plans`，逐候选打印 `heuristic` 分解。**不需要 replay、不需要 rollout**，
每状态只要几百 µs。

### 对齐验证（先证明工具是对的）

`seed732 / round100 / player0`，我的工具 vs 亚军自己的 stderr 日志：

| 指标 | 结果 |
|---|---|
| 候选 key 集合 | **209 / 209 完全一致** |
| heuristic 逐位比较 | **max diff = 0.0**（209 个全等） |

对齐脚本：`python code/my_ai/distill_top2/dump_prior_labels.py --globs <ents_glob> --align-log-dir match_results --align-seed 732 --align-round 100` → `PASS`。

## 2. 先验只有 3 个取值

`training_history/distill_top2/abl_linear2_raw/ents_seed00732.jsonl`（512 回合 × 2 玩家）

| heuristic 取值 | 占比 | 含义 |
|---|---|---|
| `0.0` | **89.99%** | 普通动作 |
| `−400.0` | 9.39% | 该计划会降级狙击塔（`sniper_downgrade_penalty`） |
| `+80.0` | 0.62% | 不动（`hold_bonus`） |

**不同取值个数 = 3**。各分项非零比例：

```
base_heuristic          0.62%   ← 只有 hold 那一条
lure_heuristic          0.00%
lightning_heuristic     0.00%
operation_penalty       9.39%   ← 就是那 400 的降级罚
followup_penalty        0.00%
lightning_static_bonus  0.00%
```

原因在源码里一目了然（`lure_strategy_v4_params.hpp` 的手调常数表）：

```cpp
double hold_bonus = 80.0;
double c1_build_bonus = 0.0;          // 0 ⇒ 建塔不加分
double c1_quick_trans_bonus = 100.0;  // 但默认配置下没被触发到
double c1_sniper_trans_bonus = 400.0;
double sniper_downgrade_penalty = 400.0;
double producer_upgrade_bonus = 200.0;
double medic_upgrade_bonus = 1000.0;  // enable_producer_medic_branch=false ⇒ 关
bool   hold_followup_enabled = false; // ⇒ followup 那一支也关
```

`c1_build_heuristic(code)` 的实现就一句：`return code == C1 ? c1_build_bonus : 0.0;`
—— 它压根不是函数，是**规则查表**。

## 3. 先验对决策的真实作用：阈值（82.6% 不动）

用 1024 个决策（`ai0_runnerup_seed732` + `ai1_ai3_linear_seed732` 两条日志）统计：

| 指标 | 数值 |
|---|---|
| 决策数 | 1024（候选计划共 161,365，均值 157.6/决策） |
| 先验取值种类 | **3** |
| **最终选择 = `hold`** | **846 / 1024 = 82.6%** |
| `argmax(heuristic) == 最终选择` | 82.6%（因为 argmax(heuristic) **恒为 hold**） |
| `argmax(mean_terminal_score) == 最终选择` | 44.7% |
| 先验翻盘（搜索最佳 ≠ 最终选择） | **52.6%** |
| top1 − top2 的 `score` 分差 | 均值 62.2 / **中位 71.1** / p90 100.0 |
| top1 − top2 的 `heuristic` 分差 | 恒 80.0 |
| `rank==1` 与 `max(score)` 是否一致 | **1024 / 1024**（日志自身的校验） |
| `score == mean_terminal + heuristic + 闪电 − 反应式` | 156,759 / 161,365（97.1%） |

注意第 3、5、6 行合起来说明的事：

- `argmax(heuristic)` **永远是 hold**（80 是唯一正数），所以"只看先验的 argmax 对 82.6%"这句话
  的真实意思是"**82.6% 的决策就是不动**"，而不是"先验很准"。
- top1−top2 的 `score` 中位分差 **71.1**，而 hold 的加成正好是 **80** ⇒
  **先验的 80 分就是为了卡在这个分差量级上当阈值**：动的收益要超过 ~80 才值得动。
- 所以 `prior` 的信息量约等于"**该不该动**"这 1 bit（外加减级狙击塔的 −400 硬约束）。

## 4. 这对主线意味着什么（重要）

主线是"**验证我们的架构能不能复刻亚军的先验和价值函数**"。现在的答案是：

| 目标 | 结论 | 证据 |
|---|---|---|
| **先验** `heuristic` | **可表达，而且是平凡的** | 3 值查表规则，任何架构都能精确复刻；不存在"表达力不够"的问题 |
| **价值** `evaluate_terminal` | **可表达** | entity Transformer R² **0.9969**（单边仿真版 0.9987） |
| **决策真正依赖的量** `mean_terminal_score` | **还没蒸，而且这才是难点** | 它是 **125 次随机 rollout 的终止评估均值**，即"未来的蒙特卡洛期望"，不是状态的确定性函数 |

**我们的 0/32 现在有了解释**：决策的 top1−top2 中位分差只有 **71**，而我们蒸馏的
`evaluate_terminal` MAE ≈ **80** —— 误差和分差同一量级，argmax 必然被噪声决定。
这不是"架构表达力不够"，而是**精度/估计方差**问题。

### 由此得到的下一步（真正有意义的那个）

亚军靠 **125 次 rollout 平均**把方差压下来，而不是靠更强的函数类。所以要回答
"我们的架构能不能替代它的搜索"，必须量两个数：

1. **rollout 噪声下限**：固定状态、固定计划，`mean_terminal_score` 在 rollout 之间的
   σ 是多少？125 次平均后 SE = σ/√125 是多少？
2. **我们的单次前向误差** vs 那个 SE：如果我们的 MAE 已经低于 SE，那"输给搜索"就纯属
   方差问题，用集成/多次采样也能压下去；如果高于 SE，说明函数类真的不够。

`mean_terminal_score` 本身（期望值）也许同样是我们架构能拟合的——但那需要先有标签，
即对每个状态跑若干 rollout 取均值，这正是 1 要测的。

## 5. 复现命令

```powershell
# 1) 编译（ASCII junction + mingw64 PATH，坑见 docs/runnerup_build_note.md）
& code\my_ai\distill_top2\cpp_test\build_dump_plan_prior.ps1

# 2) 对齐验证（拿亚军自己的日志核对）
python code\my_ai\distill_top2\dump_prior_labels.py `
  --globs "training_history/distill_top2/abl_linear2_raw/ents_seed*.jsonl" `
  --align-log-dir match_results --align-seed 732 --align-round 100

# 3) 先验值分布
python code\my_ai\distill_top2\dump_prior_labels.py `
  --globs "training_history/distill_top2/abl_linear2_raw/ents_seed00732.jsonl" --stats-only --limit 1

# 4) 先验 vs 搜索的归因
python code\my_ai\distill_top2\analyze_prior_decomposition.py `
  match_results\ai0_runnerup_seed732.stderr.log match_results\ai1_ai3_linear_seed732.stderr.log
```
