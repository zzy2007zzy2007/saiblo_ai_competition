# monitor_training.sh 完整统计表格与字段说明

> 本文档汇编了监控脚本 5 个子命令的完整输出表格及每个字段的含义。
> 基线版本为 ppo_v1 完成期（train_09 前后），此时监控系统拥有最全面的字段集合。

---

## 目录

1. [SelfPlay 技术参数表](#1-selfplay-技术参数表)
2. [动作执行次数统计](#2-动作执行次数统计)
3. [Action Reward 统计](#3-action-reward-统计)
4. [Baseline 对战横表](#4-baseline-对战横表)
5. [异常扫描](#5-异常扫描)

---

## 1. SelfPlay 技术参数表

**命令**: `train_metrics_table`  
**格式化脚本**: [tools/monitor/format_metrics_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_metrics_table.py)  
**数据源**: `training/training_metrics_history.jsonl`

### 1.1 完整表格样例

```
    Ep     Reward   OurCoin   EnCoin   OurCum    EnCum  TowerLoss  GoldLoss  PolicyLoss   ValueLoss    Entropy  EntCoef              LR   Rounds
------------------------------------------------------------------------------------------------------------------------------------------------
     1     42.15    120.5    115.3    450.0    420.0    0.042300   0.003200    0.123456    0.089012   2.3456   0.2000   0.0000500000     38.0
     5     38.20    118.0    112.0    438.0    415.0    0.038500   0.002800    0.118000    0.085000   2.4000   0.2000   0.0000490000     36.5
```

### 1.2 字段清单

| 序号 | 字段 | 列名 | 类型 | 说明 |
|------|------|------|------|------|
| 1 | `episode` | Ep | int | 训练轮次编号 |
| 2 | `avg_reward` | Reward | float | 最近 100 轮窗口平均奖励 |
| 3 | `avg_our_coins` | OurCoin | float | 本轮己方每步平均金币持有量 |
| 4 | `avg_enemy_coins` | EnCoin | float | 本轮敌方每步平均金币持有量 |
| 5 | `avg_our_cumulative_coins` | OurCum | float | 己方累积金币（整局累计、不重置）均值 |
| 6 | `avg_enemy_cumulative_coins` | EnCum | float | 敌方累积金币均值 |
| 7 | `aux_tower_loss` | TowerLoss | float | 己方防御塔伤害预测辅助损失 |
| 8 | `aux_gold_loss` | GoldLoss | float | 己方金币收入预测辅助损失 |
| 9 | `policy_loss` | PolicyLoss | float | PPO-Clip 策略损失 |
| 10 | `value_loss` | ValueLoss | float | 价值网络损失（clipped MSE） |
| 11 | `entropy` | Entropy | float | 策略熵（动作分布不确定性） |
| 12 | `entropy_coef` | EntCoef | float | 熵正则化系数（超参数） |
| 13 | `learning_rate` | LR | float | 策略网络当前学习率 |
| 14 | `avg_rounds` | Rounds | float | 最近 100 轮平均回合步数 |

### 1.3 字段说明

**Reward**: 由 `TrainingLogger.log_training_metrics()` 计算，取 `episode_rewards[-window:100]` 的均值。  
**OurCoin / EnCoin**: 采集自每步环境 state 中 `coins[player]` 的逐帧均值，经 `_aggregate_coin_stats()` 聚合。  
**OurCum / EnCum**: 整局累积金币的均值，初始值为 50，反映经济积累效果。  
**TowerLoss / GoldLoss**: 辅助预测网络（auxiliary head）对己方塔伤害和金币收入的预测误差（MSE），系数由 `aux_tower_coef` / `aux_gold_coef` 缩放后加入总损失。  
**PolicyLoss / ValueLoss / Entropy**: 标准 PPO 三大损失项。  
**EntCoef**: 当前使用的熵系数（可为固定值或退火值）。  
**LR**: 当前学习率（支持固定 / 余弦退火 / warmup）。  
**Rounds**: 最近 100 轮平均每局步数，反映对战时长趋势。

### 1.4 条件性显示的字段

脚本会根据数据中是否存在以下字段自动调整表头：

- `avg_our_coins` / `avg_enemy_coins` → 显示 OurCoin / EnCoin 列
- `avg_our_cumulative_coins` / `avg_enemy_cumulative_coins` → 显示 OurCum / EnCum 列
- `aux_tower_loss` / `aux_gold_loss` → 显示 TowerLoss / GoldLoss 列

当所有条件字段都存在时，输出 **14 列完整表格**。

---

## 2. 动作执行次数统计

**命令**: `train_action_table`  
**格式化脚本**: [tools/monitor/format_action_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_action_table.py)  
**数据源**: `training/training_metrics_history.jsonl`

### 2.1 完整表格样例

```
    Ep     Noop   Build   Upgrd    Downg       LS      Emp     Defl      Eva     Tech   Total  Rounds
---------------------------------------------------------------------------------------------------
     1    12.50    8.20    3.10    0.50    2.30    1.80    0.90    0.40    1.20   30.90   38.0
     5    11.80    9.00    3.50    0.30    2.00    1.50    1.00    0.50    1.40   31.00   36.5
```

### 2.2 字段清单

| 序号 | 字段 | 列名 | 类型 | 说明 |
|------|------|------|------|------|
| 1 | `episode` | Ep | int | 训练轮次编号 |
| 2 | `type_noop_ratio` × `avg_rounds` | Noop | float | 无操作动作平均每局执行次数 |
| 3 | `type_build_tower_ratio` × `avg_rounds` | Build | float | 建造防御塔平均次数 |
| 4 | `type_upgrade_tower_ratio` × `avg_rounds` | Upgrd | float | 升级防御塔平均次数 |
| 5 | `type_downgrade_tower_ratio` × `avg_rounds` | Downg | float | 降级防御塔平均次数 |
| 6 | `type_lightning_storm_ratio` × `avg_rounds` | LS | float | 闪电风暴平均次数（超级武器） |
| 7 | `type_emp_blaster_ratio` × `avg_rounds` | Emp | float | EMP 冲击波平均次数（超级武器） |
| 8 | `type_deflector_ratio` × `avg_rounds` | Defl | float | 偏转护盾平均次数（超级武器） |
| 9 | `type_evasion_ratio` × `avg_rounds` | Eva | float | 闪避平均次数（超级武器） |
| 10 | `type_tech_upgrade_ratio` × `avg_rounds` | Tech | float | 科技升级平均次数 |
| 11 | 前列之和 | Total | float | 该轮所有动作总执行次数 |
| 12 | `avg_rounds` | Rounds | float | 该轮平均步数 |

### 2.3 算法规格

```
count(type_X) = type_X_ratio × avg_rounds
Total = sum(count(type_X))
```

即：各动作类型的比例（`type_*_ratio` = 该类动作数 / 总动作数）乘以平均回合步数，得到近似每局执行次数。

### 2.4 动作类型与 Action ID 映射

| 动作类型 | flat_start | flat_end | type_*_ratio 字段 |
|----------|-----------|----------|-------------------|
| noop | 0 | 1 | `type_noop_ratio` |
| build_tower | 1 | 3 | `type_build_tower_ratio` |
| upgrade_tower | 3 | 9 | `type_upgrade_tower_ratio` |
| downgrade_tower | 9 | 11 | `type_downgrade_tower_ratio` |
| lightning_storm | 11 | 35 | `type_lightning_storm_ratio` |
| emp_blaster | 35 | 46 | `type_emp_blaster_ratio` |
| deflector | 46 | 74 | `type_deflector_ratio` |
| evasion | 74 | 83 | `type_evasion_ratio` |
| tech_upgrade | 83 | 84 | `type_tech_upgrade_ratio` |

---

## 3. Action Reward 统计

**命令**: `train_action_reward_table`  
**格式化脚本**: [tools/monitor/format_action_reward_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_action_reward_table.py)  
**数据源**: `training/training_metrics_history.jsonl`

### 3.1 完整表格样例

```
V2 Action Reward 统计（action_rewards 各分量占绝对值之和比例 %）:

    Ep     Noop%    Build%    Upgrd%    Downg%    Super%   HPBase%   HPTower%    ActSum    AvgRwd   Batch
---------------------------------------------------------------------------------------------------------
     1      5.2%    45.3%    12.1%     0.8%     18.6%     0.0%      0.0%      182.5      42.2     1024
     5      4.8%    46.0%    11.5%     0.6%     19.0%     0.0%      0.0%      190.0      38.2     1024
```

### 3.2 字段清单

| 序号 | 列名 | 字段来源 | 类型 | 说明 |
|------|------|----------|------|------|
| 1 | Ep | `episode` | int | 训练轮次编号 |
| 2 | Noop% | `action_rewards.noop` | % | NO-OP 动作奖励占绝对值之和的比例 |
| 3 | Build% | `action_rewards.build_tower` | % | 建造塔奖励占比 |
| 4 | Upgrd% | `action_rewards.upgrade_tower` | % | 升级塔奖励占比 |
| 5 | Downg% | `action_rewards.downgrade_tower` | % | 降级塔奖励占比 |
| 6 | Super% | 4 类超级武器奖励之和 | % | 超级武器（闪电+EMP+偏转+闪避）奖励占比 |
| 7 | HPBase% | V2 未分离 | % | 攻击兵营奖励占比（当前始终 0.0%） |
| 8 | HPTower% | V2 未分离 | % | 攻击防御塔奖励占比（当前始终 0.0%） |
| 9 | ActSum | `action_rewards` 各分量之和 | float | 所有 action reward 总和 |
| 10 | AvgRwd | `avg_reward` | float | 参考用整体平均奖励 |
| 11 | Batch | `batch_size` | int | 该 PPO update 的 batch size |

### 3.3 算法规格

```
abs_sum = sum(|action_rewards[type]| for all types)
pct(type) = action_rewards[type] / abs_sum × 100
ActSum   = sum(action_rewards[type] for all types)
```

分母使用绝对值之和，避免正负抵消导致比例失真。

---

## 4. Baseline 对战横表

**命令**: `train_battle_table`  
**格式化脚本**: [tools/monitor/format_battle_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_battle_table.py)  
**数据源**: `selfplay/selfplay_battles/selfplay_battle_stats.json` + `selfplay/selfplay_battles/battle_evaluations.jsonl`

### 4.1 完整表格样例

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
```

### 4.2 Per-Round 评估横表

```
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

### 4.3 字段清单

**累积统计表**:

| 序号 | 列名 | 字段来源 | 类型 | 说明 |
|------|------|----------|------|------|
| 1 | Opponent | stats key | str | 对手/基线名称 |
| 2 | Battles | `wins + losses + draws` | int | 对战总场次 |
| 3 | Wins | `wins` | int | 胜场数 |
| 4 | Losses | `losses` | int | 负场数 |
| 5 | Draws | `draws` | int | 平局数 |
| 6 | WinRate | `wins / total` | % | 胜率 |
| 7 | AvgReward | `avg_reward` | float | 平均奖励 |
| 8 | AvgRounds | `avg_rounds` | float | 平均回合步数 |

**Per-Round 评估表**:

| 序号 | 列名 | 说明 |
|------|------|------|
| 1 | Episode | 评估轮次（E20 = episode 20 时评估） |
| 2 | {BaselineName} | 格式 `wins/total(win_rate%)` |
| 3 | WinRate | 该轮对所有基线的综合胜率 |
| 4 | AvgRounds | 该轮平均步数 |

---

## 5. 异常扫描

**命令**: `scan_anomalies`  
**脚本**: [tools/monitor/scan_anomalies.sh](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/scan_anomalies.sh)  
**数据源**: `training/training_metrics_history.jsonl`, `training/error.log`, `training/warning.log`

### 5.1 扫描项目

| 序号 | 检查项 | 检测逻辑 | 严重等级 |
|------|--------|----------|----------|
| 1 | NaN 检测 | 扫描所有数值字段，检测 NaN/Inf 出现次数 | critical |
| 2 | ratio 异常 | `ratio_mean` 严重偏离 1.0 或 `ratio_std` 过大 | warning |
| 3 | 熵坍缩 | `entropy` < 阈值（如 0.1）超过 N 轮 | critical |
| 4 | 梯度爆炸 | `grad_norm`/`grad_norm_max_layer` 超过阈值 | warning |
| 5 | loss 发散 | `loss` 或 `policy_loss` 持续增长超过趋势线 | warning |
| 6 | reward 异常 | `avg_reward` 长期趋于零或大幅下降 | warning |
| 7 | 动作坍缩 | `type_noop_ratio` > 0.9 表明策略退化为全 NO-OP | critical |

### 5.2 输出格式

```
[异常扫描] 扫描 250 个 episode 记录...

❌ [CRITICAL] 熵坍缩: episode 100-150 连续 50 轮 entropy < 0.1
   last_entropy=0.087 | threshold=0.100 | affected_episodes=50

⚠️ [WARNING] 梯度爆炸: episode 125 grad_norm_max_layer=45.2 > 20.0
   total_grad_norm=32.5 | grad_norm_policy=28.0 | grad_norm_value=4.5

✅ 动作分布正常: noop_ratio=0.12 (threshold=0.90)
✅ 奖励稳定: reward_mean=42.5, reward_std=8.3
✅ NaN/Inf 检测: 0 次
```

---

## 附录 A: 数据流架构

```
SelfPlayTrainer 训练循环
  │
  ├─ collect_episode_with_swap()
  │   ├─ action_counts: Dict[str, int]      ← 每局每类动作执行次数
  │   ├─ action_rewards: Dict[str, float]   ← 每局每类动作的奖励总和
  │   └─ step_snapshots (tower_hp, coins)   ← 逐帧经济/血量快照
  │
  ├─ _ppo_update()
  │   ├─ PPO 损失 (policy/value/entropy)
  │   ├─ 辅助损失 (aux_tower/aux_gold)
  │   ├─ 梯度统计 (grad_norm_*)
  │   ├─ 价值诊断 (value_input/pred_*)
  │   ├─ 比率诊断 (ratio_mean/std)
  │   └─ 动作类型统计 (type_*_ratio/logit/prob)
  │
  ├─ _build_training_context()  [ppo_v1 only]
  │   ├─ _extract_ppo_metric_fields()       ← 提取 PPO 指标
  │   ├─ _aggregate_coin_stats()            ← 经济统计
  │   ├─ _aggregate_reward_sources()        ← 奖励来源分解
  │   ├─ _aggregate_action_reward_stats()   ← 动作奖励聚合
  │   └─ _aggregate_opponent_stats()        ← 对手统计
  │
  └─ Logger.save_training_metrics()        → training_metrics_history.jsonl
     ↓
  format_metrics_table.py  ← 各 format 脚本读取 JSONL 渲染表格
  format_action_table.py
  format_action_reward_table.py
  format_battle_table.py
```

---

## 附录 B: JSONL 历史数据字段全集

以下为 `training_metrics_history.jsonl` 中 ppo_v1 最完整时期记录的所有字段：

```
# 基础标识
episode, timestamp

# PPO 损失
policy_loss, value_loss, entropy, entropy_type, entropy_target, loss

# 辅助损失
aux_tower_loss, aux_gold_loss, aux_enemy_tower_loss, aux_enemy_gold_loss

# 奖励统计
avg_reward, reward_mean, reward_std, reward_min, reward_max

# GAE 诊断
advantages_std, return_mean, return_std

# 裁剪诊断
clip_fraction

# 梯度诊断
gradient_norm, grad_norm_policy, grad_norm_value, grad_norm_max_layer

# 价值诊断
value_input_mean, value_input_std, value_pred_mean, value_pred_std

# 重要性采样比率诊断
ratio_mean, ratio_std

# 有效动作诊断
valid_actions_mean, valid_actions_min

# TD 误差诊断
td_error_mean, td_error_std

# 批次信息
batch_size, num_collected, nan_skip_count, steps

# 超参数快照
entropy_coef, learning_rate

# 经济统计
avg_our_coins, avg_enemy_coins, avg_our_cumulative_coins, avg_enemy_cumulative_coins
our_coins_lt_30, our_coins_ge_30_lt_60, our_coins_ge_60_lt_100, our_coins_ge_100
enemy_coins_lt_30, enemy_coins_ge_30_lt_60, enemy_coins_ge_60_lt_100, enemy_coins_ge_100

# 奖励来源分解（rw_* = reward component）
rw_hp_attack_base, rw_hp_attack_tower, rw_coin_gain, rw_tower_survival
rw_balance, rw_tech_bonus, rw_die_penalty, rw_end_reward

# 动作类型统计
type_noop_ratio, type_build_tower_ratio, type_upgrade_tower_ratio, type_downgrade_tower_ratio
type_lightning_storm_ratio, type_emp_blaster_ratio, type_deflector_ratio, type_evasion_ratio
type_tech_upgrade_ratio

# 动作类型 Logit/Prob
logit_noop, logit_build_tower, ...
prob_noop, prob_build_tower, ...

# 动作奖励/计数（扁平字段）
actrw_noop, actrw_build_tower, actrw_upgrade_tower, actrw_downgrade_tower
actrw_lightning_storm, actrw_emp_blaster, actrw_deflector, actrw_evasion
actcnt_noop, actcnt_build_tower, actcnt_upgrade_tower, actcnt_downgrade_tower
actcnt_lightning_storm, actcnt_emp_blaster, actcnt_deflector, actcnt_evasion

# 对手统计（动态字段）
opponent_win_rate_{oid}, opponent_reward_{oid}
```
