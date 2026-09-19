# ppo_v2 监控表格字段补齐需求

> 本文档基于 ppo_v1 最完整时期（train_09 完成时）的字段全集，
> 对比 ppo_v2 当前输出的 5 张监控表格，逐表列出缺失字段及补齐需求。

---

## 缺失总览

下面从 4 个维度对比 ppo_v1 与 ppo_v2 的字段覆盖：

| 维度 | ppo_v1 | ppo_v2 | 差距 |
|------|--------|--------|------|
| JSONL 写入字段数 | ~50+ 项 | ~30 项 | 缺失 ~20+ 项 |
| 使用的聚合上下文方法 | `_build_training_context()` (5 个子方法) | 无独立上下文构建 | 缺少经济、奖励分解、对手统计 |
| 字段名一致性 | 统一命名 | 部分字段名不同（loss vs avg_loss, gradient_norm vs grad_norm） | 需对齐 |
| action_rewards 格式 | 扁平字段 `actrw_*` / `actcnt_*` | 嵌套 dict `action_rewards.*` | 格式不同但内容等价的 |

---

## 1. SelfPlay 技术参数表（train_metrics_table）

### 1.1 当前 ppv_v2 输出列

```
Ep  Reward  PolicyLoss  ValueLoss  Entropy  EntCoef  LR  Rounds
```

### 1.2 需要补齐的字段

| 缺失字段 | 原 ppo_v1 列名 | 补齐优先级 | 补齐理由 |
|----------|---------------|-----------|----------|
| `avg_our_coins` + `avg_enemy_coins` | OurCoin / EnCoin | 高 | 经济对抗性是评估策略健康度的核心维度，经济崩盘往往是策略退化的前兆 |
| `avg_our_cumulative_coins` + `avg_enemy_cumulative_coins` | OurCum / EnCum | 高 | 累积经济总量反映长线资源积累能力，区分短期波动 |
| `aux_tower_loss` + `aux_gold_loss` | TowerLoss / GoldLoss | 中 | 辅助预测任务的收敛状态直接影响主任务梯度质量 |
| `aux_enemy_tower_loss` + `aux_enemy_gold_loss` | — | 低 | ppo_v2 已新增敌方辅助预测，当前工具未展示此列，建议补充 |

### 1.3 语义优化建议

**问题**: 当前 `Reward` 列显示的是 `avg_reward`（PPO update batch 的窗口均值），但用户往往混淆 "本轮 episode 的累计奖励" 与 "窗口均值奖励"。

**建议改动**:
- 将 `Reward` 列拆分为 `Reward`（窗口均值，当前逻辑）和 `LastRew`（最后 1 局 episode 累计奖励），后者从 `per_episode_stats.jsonl` 的最新记录读取
- 添加 `RwdMean/RwdStd` 两个辅助列在表底统计行中，提供奖励分布的全貌参考

**建议新增列**:
- `GradNorm`：总梯度范数（来源于 `grad_norm` 字段），替代仅在末尾统计行显示
- `ClipFrac`：PPO 裁剪率（`clip_fraction`），< 0.1 表示策略更新保守，> 0.5 表示策略变化过快

**建议调整**:
- `EntCoef` 和 `LR` 作为超参快照，移动至表尾统计区更合理，但为保持横向对比保留在当前表头位置
- `Rounds` 建议加单位后缀 → `Rounds(step)`

### 1.4 完整表格设计（建议最终态）

```
    Ep    Reward   LastRew   OurCoin   EnCoin  OurCum   EnCum  GradNorm  ClipFrac  TowerLoss  GoldLoss  PolicyLoss   ValueLoss    Entropy  EntCoef              LR   Rounds
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     1     42.15     45.0     120.5    115.3   450.0    420.0    2.34     0.12      0.0423     0.0032     0.123456    0.089012   2.3456   0.2000   0.0000500000     38
     5     38.20     30.0     118.0    112.0   438.0    415.0    1.89     0.08      0.0385     0.0028     0.118000    0.085000   2.4000   0.2000   0.0000490000     36
```

---

## 2. 动作执行次数统计表（train_action_table）

### 2.1 当前 ppv_v2 输出列

```
Ep  Noop  Build  Upgrd  Downg  LS  Emp  Defl  Eva  Tech  Total  Rounds
```

当前输出列已经完整，此表本身设计合理，无需增减列。

### 2.2 语义优化建议

**问题**:
- 各动作类型的列名缩写不够直观（如 `LS`, `Emp`, `Defl`, `Eva`），新用户不易理解
- `Total` 与 `Rounds` 的语义边界模糊：Total = 所有动作执行次数之和，Rounds = 回合步数

**建议改动**:
- 在表格底部添加图例行，说明各缩写的全称（如 `LS = Lightning Storm`）
- 将 `Rounds` 改为 `Steps`，与 JSONL 中 `avg_rounds` 的命名一致
- 在表底添加 "动作分布比例" 辅助行，将各列除以 Total 显示 %

### 2.3 完整表格设计（建议最终态）

```
    Ep    Noop    Build    Upgrd    Downg       LS      Emp     Defl      Eva     Tech     Total    Steps
--------------------------------------------------------------------------------------------------------
     1    12.50    8.20    3.10    0.50    2.30    1.80    0.90    0.40    1.20    30.90     38.0
  Dist    40.5%   26.5%   10.0%    1.6%    7.4%    5.8%    2.9%    1.3%    3.9%
```

---

## 3. Action Reward 统计表（train_action_reward_table）

### 3.1 当前 ppv_v2 输出列

```
Ep  Noop%  Build%  Upgrd%  Downg%  Super%  HPBase%  HPTower%  ActSum  AvgRwd  Batch
```

### 3.2 需要补齐的字段

当前表已基本完整，但有以下语义问题需要优化：

| 问题 | 当前状态 | 建议修改 |
|------|---------|---------|
| HPBase% / HPTower% | 始终为 0.0%，无用 | **删除**这两列，或新增 `actrw_hp_attack_base` / `actrw_hp_attack_tower` 字段支持 |
| Super% 聚合 | 4 类超级武器打包显示 | **拆分**为 `LS%`、`Emp%`、`Defl%`、`Eva%` 四列，提供更细粒度的奖励来源分析 |
| 绝对值分母 | 分母使用 `sum(\|各分量\|)`，正负抵消信息 | 保留此设计，同时**新增** `PosSum`（正奖励分量和）+ `NegSum`（负奖励分量和）两列 |

### 3.3 奖励来源分解（rw_*）支持

ppv_v2 当前**完全未记录**奖励来源分解字段（`rw_hp_attack_base`, `rw_coin_gain`, `rw_balance` 等 8 个字段）。应恢复与 ppo_v1 等价的奖励来源聚合：

**恢复方式**:
- 在 `selfplay.py` 的 `_collect_episode_with_swap()` 中，记录每步的 `reward_detail` 分解
- 新增 `_aggregate_reward_sources()` 方法聚合 batch 内所有 step 的奖励来源
- 将结果写入 JSONL 的 `rw_*` 字段

### 3.4 完整表格设计（建议最终态）

```
V2 Action Reward 统计（各分量占绝对值之和比例 %）:

    Ep    Noop%   Build%   Upgrd%   Downg%     LS%     Emp%    Defl%     Eva%   PosSum   NegSum    ActSum    AvgRwd   Batch
---------------------------------------------------------------------------------------------------------------------------
     1     5.2%   45.3%    12.1%     0.8%     9.2%     5.1%    3.0%     1.3%    245.0    -62.5     182.5      42.2     1024
```

---

## 4. Baseline 对战横表（train_battle_table）

### 4.1 当前 ppv_v2 输出列

```
Opponent  Battles  Wins  Losses  Draws  WinRate  AvgReward  AvgRounds
```

### 4.2 需要补齐的字段

| 缺失特性 | 补齐优先级 | 补齐理由 |
|----------|-----------|----------|
| Per-Round 逐轮对战详情 | 高 | ppv_v2 已支持 `battle_evaluations.jsonl`，但工具仅当 `selfplay_battles/` 目录存在时使用 `--selfplay-stats` 模式，不会自动回退到 `--evaluations` 模式 |

### 4.3 语义优化建议

**问题**:
- 当前仅显示累积汇总表，缺少逐轮变化趋势
- `AvgReward` 和 `AvgRounds` 在累积表中会随时间漂移，失去参考意义

**建议改动**:
- 累积汇总表和 Per-Round 评估表**并列输出**，互不替代
- 新增 `WinRateTrend` 辅助行：各轮次胜率的变化范围（min ~ max），直观反映稳定性
- 表底添加 `Trend` 行：`[E20:75% → E40:85% → E60:85%]`

### 4.4 完整表格设计（建议最终态）

```
Baseline Battle Stats (cumulative)
==========================================================================================
Opponent              Battles    Wins  Losses  Draws  WinRate  AvgReward  AvgRounds
------------------------------------------------------------------------------------------
BasicRandomAI              40      38       2       0   95.0%      65.20       42.3
BasicTowerAI               40      30       6       4   75.0%      42.10       45.1
MediumRuleAI               20      12       5       3   60.0%      18.50       48.7
------------------------------------------------------------------------------------------
TOTAL                     100      80      13       7   80.0%
Trend                    E20:95% → E40:90% → E60:88%

Baseline Battle Stats (by evaluation round)
===================================================================================================
Episode    BasicRandomAI        BasicTowerAI          WinRate    AvgRounds
---------------------------------------------------------------------------------------------------
E20        8/10(80%)            7/10(70%)              75%        42.5
E40        9/10(90%)            8/10(80%)              85%        43.0
E60        10/10(100%)          7/10(70%)              85%        44.1
---------------------------------------------------------------------------------------------------
Total      27/30(90%)           22/30(73%)             82%        43.2
===================================================================================================
```

---

## 5. 异常扫描（scan_anomalies）

### 5.1 当前监测能力

ppv_v2 的 `scan_anomalies.sh` 可扫描以下项目：
- NaN/Inf 检测
- 熵坍缩检测（`entropy` < 阈值）
- 梯度爆炸检测（`grad_norm` > 阈值）
- 奖励归零检测

### 5.2 需要补齐的检测项

| 缺失检测项 | 所需字段 | 补齐优先级 | 检测逻辑 |
|-----------|---------|-----------|---------|
| ratio 异常 | `ratio_mean`, `ratio_std` | 高 | `ratio_mean` 偏离 1.0 超过 ±0.2 或 `ratio_std` > 0.5 |
| 价值预测发散 | `value_pred_mean`, `value_pred_std` | 高 | `value_pred_mean` 持续偏离 0 或 `value_pred_std` → 0 |
| TD 误差异常 | `td_error_mean`, `td_error_std` | 中 | `td_error_mean` 绝对值 > 10 或 `td_error_std` → 0 |
| 有效动作崩塌 | `valid_actions_min` | 中 | 有效动作数持续下降至接近 0 |
| 经济崩溃 | `avg_our_coins`, `avg_enemy_coins` | 中 | 己方平均金币 < 敌方平均金币 × threshold |
| NO-OP 比例异常 | `type_noop_ratio` | 低 | 同动作坍缩检测，但阈值可调 |
| 损失发散 | `policy_loss`, `value_loss` | 中 | 连续 N 轮损失上升超过 baseline × 2 |
| 学习率异常 | `learning_rate` | 低 | LR 衰减梯度过大或停止衰减 |

---

## 6. 数据源字段完整性要求

为支持以上所有表格的补齐，JSONL 数据源 `training_metrics_history.jsonl` 需要写入以下 **新增字段**：

### 6.1 恢复字段（ppv_v1 有，ppv_v2 需恢复）

```
avg_our_coins, avg_enemy_coins
avg_our_cumulative_coins, avg_enemy_cumulative_coins
our_coins_lt_30, our_coins_ge_30_lt_60, our_coins_ge_60_lt_100, our_coins_ge_100
enemy_coins_lt_30, enemy_coins_ge_30_lt_60, enemy_coins_ge_60_lt_100, enemy_coins_ge_100
rw_hp_attack_base, rw_hp_attack_tower, rw_coin_gain, rw_tower_survival
rw_balance, rw_tech_bonus, rw_die_penalty, rw_end_reward
entropy_type, entropy_target
value_input_mean, value_input_std, value_pred_mean, value_pred_std
ratio_mean, ratio_std
valid_actions_mean, valid_actions_min
td_error_mean, td_error_std
num_collected
```

### 6.2 字段名对齐（统一命名）

| ppv_v1 旧字段 | ppv_v2 当前字段 | 对齐后统一字段 |
|--------------|----------------|--------------|
| `gradient_norm` | `grad_norm` | `grad_norm` |
| `nan_skip_count` | `skipped_minibatches` | `nan_skip_count` |
| `avg_loss` | `loss` | `avg_loss` |

### 6.3 新增字段（ppv_v2 可提供但之前未记录的）

```
# 敌方辅助损失（ppv_v2 已计算但未写入 JSONL）
aux_enemy_tower_loss, aux_enemy_gold_loss
```
