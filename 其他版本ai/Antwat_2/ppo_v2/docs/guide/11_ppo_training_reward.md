# PPO 奖励系统设计

## 1. 概述

### 1.1 奖励系统的地位

奖励信号是强化学习的核心驱动力。在 AntWar PPO 训练系统中，奖励函数定义了什么是"好的行为"，是策略网络学习的唯一目标信号。奖励设计的质量直接决定了智能体的最终行为模式。

### 1.2 奖励系统架构概览

```
游戏回合
   │
   ├── ① 动作级奖励（Action Reward）
   │     在动作执行前计算，即时评估动作质量
   │     包括：建塔分级奖励、升级奖励、降级惩罚、超级武器奖励、科技升级奖励、noop 惩罚
   │
   ├── ② 回合级奖励（Step / Battle Reward）
   │     在回合结束后计算，评估回合的综合收益
   │     包括：HP 伤害、塔伤害、金币收入、塔存活、科技加成、余额奖励、死亡惩罚
   │
   └── ③ 终局奖励（Terminal Reward）
         在对局结束时计算，评估最终胜负
         包括：胜利 +500.0，失败 -500.0
```

这三层奖励通过 **叠加** 构成每一步的最终奖励信号：

```
每步奖励 = 动作级奖励 + 回合级奖励 + 终局奖励（仅最后一步）
```

---

## 2. 奖励的三层结构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────┐
│                    一步的时间线                       │
│                                                     │
│  动作执行前                 回合推进               终局检查
│  ──────────              ──────────             ──────────        │
│  │                    │                    │                     │
│  ▼                    ▼                    ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐               │
│  │动作级奖励 │    │回合级奖励 │    │  终局奖励    │               │
│  │(即时评估) │ +  │(综合评估) │ +  │  (胜负判定)  │               │
│  └──────────┘    └──────────┘    └──────────────┘               │
│       │               │               │                          │
│       └───────────────┼───────────────┘                          │
│                       │                                          │
│                       ▼                                          │
│               clip(-500, +500)                                   │
│                       │                                          │
│                       ▼                                          │
│             ┌──────────────────┐                                │
│             │  最终步奖励       │                                │
│             │  (收集到 Batch)  │                                │
│             └──────────────────┘                                │
└─────────────────────────────────────────────────────┘
```

### 2.2 各层职责

| 层次 | 计算时机 | 职责 | 典型幅度 |
|------|---------|------|---------|
| 动作级 | 动作解码后、回合推进前 | 评估单次决策质量，提供即时反馈 | -9.0 ~ +15.0 |
| 回合级 | 回合推进后的状态变化 | 评估回合综合收益（HP/金币/塔），提供全局反馈 | -500 ~ +500 |
| 终局级 | 对局终止时 | 提供终极胜负信号 | ±500.0 |

---

## 3. 动作级奖励

### 3.1 设计原理

动作级奖励是对策略的 **即时行为引导**，类似于 "即时反馈"。它直接奖励好的动作、惩罚坏的动作，让策略在稀疏的回合级奖励到来之前就能获得学习信号。

### 3.2 调度机制

动作级奖励通过 `_ACTION_REWARD_DISPATCH` 字典调度到对应的计算方法：

```python
_ACTION_REWARD_DISPATCH = {
    "build_tower":      "_compute_build_tower_reward",
    "upgrade_tower":    "_compute_upgrade_tower_reward",
    "downgrade_tower":  "_compute_downgrade_tower_reward",
    "lightning_storm":  "_compute_fixed_deploy_reward",
    "emp_blaster":      "_compute_fixed_deploy_reward",
    "deflector":        "_compute_fixed_deploy_reward",
    "evasion":          "_compute_fixed_deploy_reward",
    "tech_upgrade":     "_compute_tech_upgrade_reward",
}
```

计算入口为 [AntWarEnv._compute_action_reward()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/antwar_env.py#L427)。

### 3.3 各动作类型的奖励明细

#### 3.3.1 Noop（空操作）

```python
# 渐进式惩罚：连续 noop 次数越多，惩罚越大
noop_streak += 1
reward = max(noop_streak * noop_base_penalty, noop_max_penalty)
#       = max(noop_streak × (-0.0025), -0.005)
```

| 连续 Noop 次数 | 单次惩罚 |
|--------------|---------|
| 1 | -0.0025 |
| 2 | -0.005 |
| 3+ | -0.005（封顶） |

设计意图：轻微惩罚不做任何操作，防止策略陷入 "什么都不做是安全的" 的局部最优。

#### 3.3.2 建造防御塔（build_tower）

```python
# 分级递减奖励：同一位置多次建塔，奖励递减
tiers = [0.6, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1]  # build_tower_tiers

reward = tiers[min(count, len(tiers) - 1)]
# count 为该位置塔被摧毁后重建的次数
```

| 建塔次数 | 奖励 |
|---------|------|
| 第 1 次（新位置） | +0.6 |
| 第 2 次（修复后重建） | +0.5 |
| 第 3 次 | +0.5 |
| 第 4-6 次 | +0.3 |
| 第 7 次及以上 | +0.1 |

设计意图：鼓励积极探索建造新的防守位置，同一位置反复重建的价值递减。

#### 3.3.3 升级防御塔（upgrade_tower）

```python
if tower.level == 1:
    reward = +3.0    # upgrade_tower_l2
elif tower.level == 2:
    reward = +5.0    # upgrade_tower_l3
else:
    reward = 0.0     # 已是最高等级
```

设计意图：高等级升级奖励更大，鼓励投入资源强化已有塔位。

#### 3.3.4 降级防御塔（downgrade_tower）

```python
reward = -9.0    # downgrade_tower_penalty
```

设计意图：强烈惩罚降级塔的行为。降级塔极少是正确的战略选择（仅在紧急需要金币时可能考虑），训练初期需要通过强烈负信号来抑制此行为。另外，**前 1000 回合会通过动作掩码直接禁用此操作**。

#### 3.3.5 超级武器（四类）

| 超级武器 | 奖励值 | 键 |
|---------|--------|----|
| 闪电风暴 (Lightning Storm) | +7.5 | `deploy_lightning_storm` |
| EMP 冲击波 (EMP Blaster) | +9.0 | `deploy_emp_blaster` |
| 偏转器 (Deflector) | +1.0 | `deploy_deflector` |
| 紧急规避 (Emergency Evasion) | +0.8 | `deploy_evasion` |

设计意图：
- EMP 和闪电风暴是进攻性武器，高奖励鼓励使用
- 偏转器和规避是防御性武器，奖励较低因为它们的效果更间接

#### 3.3.6 科技升级（tech_upgrade）

```python
# target_idx == 0: 升级蚂蚁生成速度
if generation_level == 1:
    reward = +15.0    # upgrade_gen_speed_l1
elif generation_level == 2:
    reward = +10.0    # upgrade_gen_speed_l2

# target_idx == 1: 升级蚂蚁属性（HP/攻击）
if ant_level == 1:
    reward = +7.5     # upgrade_gen_ant_l1
elif ant_level == 2:
    reward = +2.5     # upgrade_gen_ant_l2
```

设计意图：
- 早期科技升级奖励非常高（+15.0），引导策略尽早发展经济
- 蚂蚁生成速度 > 蚂蚁属性（前者是经济的根基）
- 后期科技升级奖励递减（边际效应递减）

---

## 4. 回合级奖励（Battle Reward）

### 4.1 设计原理

回合级奖励在回合结束后计算，通过比较回合前后的状态变化来评估回合的综合价值。这是 **主要的奖励信号来源**，占总奖励的绝大部分。

### 4.2 计算入口

[AntWarEnv._compute_battle_rewards()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/antwar_env.py#L228)

接收参数：
- `old`：回合开始前的状态快照
- `new`：回合结束后的状态快照
- `terminated`：是否终局
- `winner`：胜利方
- `coins_after_ops`：操作执行后、回合推进前的金币量（用于分离被动收入）

### 4.3 八个奖励子组件

#### 组件 1：HP 攻击奖励

```python
hp_dmg = max(0, old.enemy_hp - new.enemy_hp)
reward += hp_dmg × base_hp_attack_weight  # 2.0
```

**含义**：对敌方基地造成的 HP 伤害，每点 HP 伤害 = +2.0 奖励。

**权重**：`base_hp_attack_weight = 2.0`

#### 组件 2：塔伤害奖励

```python
tower_dmg = Σ max(0, old.enemy_tower_hp[k] - new.enemy_tower_hp[k])
reward += tower_dmg × tower_hp_attack_weight  # 0.2
```

**含义**：对敌方防御塔造成的累计 HP 伤害，每点塔伤害 = +0.2 奖励。

**权重**：`tower_hp_attack_weight = 0.2`

#### 组件 3：金币收入奖励（被动收入）

```python
# ★ 关键：仅计算"被动收入"，排除主动花费
coin_gain = new_coins - coins_after_ops  # 回合结束金币 - 操作后金币
reward += coin_gain × own_coin_gain_weight               # 0.05
reward -= enemy_coin_gain × enemy_coin_gain_weight       # 0.01
```

**含义**：
- 己方被动金币收入 = +0.05/金币（鼓励经济发展）
- 敌方被动金币收入 = -0.01/金币（轻微惩罚敌方经济增长）

**设计要点**：`coins_after_ops` 是操作执行后（建塔等花费已扣除）、回合推进前（被动收入尚未产生）的金币量。这样 `coin_gain` 仅反映 "蚂蚁采集产生的被动收入"，而非"操作花费后的净变化"。

#### 组件 4：塔存活奖励（非零和）

```python
# 己方塔存活
own_survival = Σ tower_survival_per_tower × level_multiplier
    # tower_survival_per_tower = 0.02
    # level_multipliers = [1.0, 1.5, 2.5]

# 敌方塔存活惩罚
enemy_survival = enemy_tower_count × enemy_tower_survival_per_tower
    # enemy_tower_survival_per_tower = 0.01

tower_survival = own_survival - enemy_survival
```

**含义**：
- 己方塔存活 → 正奖励（等级越高奖励越多：L1×1.0, L2×1.5, L3×2.5）
- 敌方塔存活 → 负惩罚（按数量计）

**非零和设计**：双方各自独立计算，不是互为相反数。

#### 组件 5：科技加成

```python
tech = own_gen_level × tech_speed_per_level        # 1.5
     + own_ant_level × tech_hp_per_level           # 1.0
```

**含义**：每级生成速度科技 = +1.5，每级蚂蚁属性科技 = +1.0。

设计意图：持续奖励已获得的技术优势，维持科技升级的动力。

#### 组件 6：余额奖励

```python
balance = own_coins × balance_reward_weight  # 0.01
```

**含义**：每 100 金币余额 = +1.0 奖励。

设计意图：鼓励保持一定的金币储备，避免将所有金币全部花光的"零余额"策略。

#### 组件 7：死亡惩罚

```python
own_die_delta = new.own_die_count - old.own_die_count
enemy_die_delta = new.enemy_die_count - old.enemy_die_count

die_penalty = own_die_delta × own_die_penalty_per_ant     # -0.05
            - enemy_die_delta × own_die_penalty_per_ant   # +0.05（敌方死亡=己方收益）
```

**含义**：
- 己方蚂蚁死亡 = -0.05/只（惩罚）
- 敌方蚂蚁死亡 = +0.05/只（奖励）

#### 组件 8：终局奖励（在回合级奖励中计算）

终局奖励在回合级奖励的同一函数中计算（因为终止状态在回合推进后检测）。详见第 5 节。

### 4.4 奖励来源追踪

[EpisodeCollector._update_reward_sources()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/episode_collector.py#L214) 从 `info['reward_detail']` 中提取并累计 8 个奖励来源：

```python
source_keys = [
    'rw_hp_attack_base',    # HP 基础攻击 → hp_dmg_reward_{player}
    'rw_hp_attack_tower',   # 塔攻击    → tower_dmg_reward_{player}
    'rw_coin_gain',         # 金币收入   → coin_gain_reward_{player}
    'rw_tower_survival',    # 塔存活   → tower_survival_{player}
    'rw_balance',           # 余额     → balance_{player}
    'rw_tech_bonus',        # 科技加成  → tech_bonus_{player}
    'rw_die_penalty',       # 死亡惩罚  → die_penalty_{player}
    'rw_end_reward',        # 终局奖励  → end_reward_{player}
]
```

这些数据最终写入 `selfplay_battle_log.jsonl`，可用于分析哪些奖励来源对胜率贡献最大。

---

## 5. 终局奖励

### 5.1 计算逻辑

```python
if terminated and winner is not None:
    if winner == 0:
        end_0 = +500.0   # win_reward
        end_1 = -500.0   # loss_reward
    elif winner == 1:
        end_0 = -500.0
        end_1 = +500.0
```

### 5.2 设计分析

| 参数 | 值 | 作用 |
|------|---|------|
| `win_reward` | +500.0 | 胜利奖励 |
| `loss_reward` | -500.0 | 失败惩罚 |

终局奖励的 ±500 远远超过任何单步的回合级奖励（被 clip 到 ±500），确保 "胜利" 是最强的学习信号。同时 ±500 相同设为对称值，保证游戏是零和的（仅在终局层面）。

终局奖励在 [AntWarEnv._compute_battle_rewards()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/antwar_env.py#L317-L325) 中与回合级奖励在同一函数中计算，最终叠加到回合级奖励之上。

---

## 6. 奖励裁剪与标准化

### 6.1 步级裁剪（Step-Level Clipping）

```python
clip_val = REWARD_CONFIG["step_reward_clip"]  # 500.0
reward = max(-clip_val, min(clip_val, reward))
```

所有动作级+回合级+终局奖励的叠加结果被裁剪到 `[-500.0, +500.0]` 范围内。

**目的**：
- 防止极端奖励值导致梯度爆炸
- 限制单步奖励对总回报的影响（尤其是终局 ±500 已经接近裁剪上界）

### 6.2 GAE 中的裁剪（GAE-Level Clipping）

在 [PPOTrainer._compute_gae()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py#L609) 中，GAE 计算完成后对 advantages 和 returns 分别裁剪：

```python
# 默认值（来自 clip_values 配置）
advantages = clamp(advantages, -1e4, +1e4)   # 优势值裁剪
returns    = clamp(returns,    -1e6, +1e6)   # 回报值裁剪
```

### 6.3 GAE 奖励统计（Reward Statistics）

在 [PPOTrainer._compute_reward_stats()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py#L154) 中统计：

```python
{
    'reward_mean':  rewards.mean(),
    'reward_std':   rewards.std(),
    'reward_min':   rewards.min(),
    'reward_max':   rewards.max(),
}
```

这些统计指标写入 `batch_metrics.jsonl` 和 TensorBoard，用于监控奖励分布是否健康。

---

## 7. 奖励信号的处理管线

### 7.1 从环境到训练数据的完整流程

```
1. AntWarEnv 计算原始奖励
   │
   ├── _compute_action_reward()    → action_reward     (~±15)
   ├── _compute_battle_rewards()   → battle_reward     (±500)
   │     ├── HP 伤害、塔伤害、金币、塔存活...
   │     ├── 终局奖励（如有）
   │     └── clip(battle_reward, -500, +500)
   │
   └── final_step_reward = action_reward + battle_reward
   │
2. EpisodeCollector 收集到 EpisodeBatch
   │
   ├── 每步: rewards.append(final_step_reward)
   ├── 累计 reward_sources（8 个来源分量）
   └── 累计 action_counts / action_rewards（监督用）
   │
3. Batch.to_tensors() → rewards (T,) float32
   │
4. PPOTrainer.ppo_update()
   │
   ├── compute_gae(rewards, values, dones)
   │     ├── δ = r + γ×V_next×(1-done) - V
   │     ├── GAE = δ + γλ×(1-done)×GAE_prev
   │     └── advantages = GAE, returns = advantages + values
   │
   ├── clamp(advantages, -1e4, +1e4)
   ├── clamp(returns, -1e6, +1e6)
   │
   ├── 多 epoch/minibatch PPO 训练
   │     ├── ratio = π_new / π_old
   │     ├── policy_loss = -min(ratio×adv, clip(ratio)×adv)
   │     └── value_loss = MSE(V_pred, returns)
   │
   └── 指标: reward_mean, reward_std, reward_min, reward_max
```

### 7.2 奖励来源 → 训练指标的数据流

```
AntWarEnv._compute_battle_rewards()
    │
    └── reward_detail (info 字段)
        │
        ▼
EpisodeCollector._update_reward_sources()
    │
    ├── rw_hp_attack_base, rw_hp_attack_tower, rw_coin_gain
    ├── rw_tower_survival, rw_balance, rw_tech_bonus
    ├── rw_die_penalty, rw_end_reward
    │
    └──→ battle_details['reward_sources']
         │
         ▼
SelfPlayBattleWriter.write_episode()
    │
    └──→ selfplay_battle_log.jsonl (每行含 reward_sources)
```

---

## 8. 辅助任务中的奖励关联

### 8.1 塔伤害预测标签

辅助任务 "塔伤害预测" 的标签来自 **HP 变化量**，与回合奖励中的 "HP 攻击奖励" 和 "塔伤害奖励" 直接相关：

```python
# aux_labels.py: compute_aux_labels_from_trajectory()
for h in [1, 2, 4, 8, 16]:  # 5 个未来视界
    tower_labels[t, h] = max(0, own_hp_sum[t] - own_hp_sum[t+h])
```

**关联**：辅助任务预测的 "未来塔 HP 减少量" 对应 `REWARD_CONFIG` 中 `tower_hp_attack_weight` 加权的奖励分量。

### 8.2 金币收入预测标签

```python
gold_labels[t, h] = coins[t+h] - coins[t]
```

**关联**：辅助任务预测的 "未来金币净变化" 对应 `REWARD_CONFIG` 中 `own_coin_gain_weight` 加权的奖励分量。

### 8.3 辅助任务设计意义

通过在策略网络的编码器上同时训练未来塔伤害和金币收入的预测，让特征表示提前 "看到" 未来的奖励相关事件。这种辅助学习信号比奖励信号更稠密（每步都有 10 个预测目标），加速了特征学习。

---

## 9. 配置参数汇总

### 9.1 完整 REWARD_CONFIG

全部奖励参数定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/action_constants.py) 的 `REWARD_CONFIG` 中。

#### 回合级奖励权重

| 参数 | 默认值 | 关联组件 |
|------|--------|---------|
| `base_hp_attack_weight` | 2.0 | HP 攻击奖励 |
| `tower_hp_attack_weight` | 0.2 | 塔伤害奖励 |
| `own_coin_gain_weight` | 0.05 | 己方金币收入奖励 |
| `enemy_coin_gain_weight` | 0.01 | 敌方金币收入惩罚 |
| `tower_survival_per_tower` | 0.02 | 每个己方塔存活奖励基数 |
| `enemy_tower_survival_per_tower` | 0.01 | 每个敌方塔存活惩罚基数 |
| `tower_survival_level_multipliers` | [1.0, 1.5, 2.5] | L1/L2/L3 塔的奖励倍率 |
| `balance_reward_weight` | 0.01 | 余额奖励权重 |
| `own_die_penalty_per_ant` | -0.05 | 每个己方蚂蚁死亡的惩罚 |
| `tech_speed_per_level` | 1.5 | 每级速度科技的加成 |
| `tech_hp_per_level` | 1.0 | 每级蚂蚁属性科技的加成 |

#### 动作级奖励值

| 参数 | 默认值 | 关联动作 |
|------|--------|---------|
| `build_tower_tiers` | [0.6, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1] | 建塔分级奖励 |
| `upgrade_tower_l2` | 3.0 | 升级到 L2 |
| `upgrade_tower_l3` | 5.0 | 升级到 L3 |
| `downgrade_tower_penalty` | -9.0 | 降级塔惩罚 |
| `deploy_lightning_storm` | 7.5 | 闪电风暴 |
| `deploy_emp_blaster` | 9.0 | EMP 冲击波 |
| `deploy_deflector` | 1.0 | 偏转器 |
| `deploy_evasion` | 0.8 | 紧急规避 |
| `upgrade_gen_speed_l1` | 15.0 | 升级生成速度 L1 |
| `upgrade_gen_speed_l2` | 10.0 | 升级生成速度 L2 |
| `upgrade_gen_ant_l1` | 7.5 | 升级蚂蚁属性 L1 |
| `upgrade_gen_ant_l2` | 2.5 | 升级蚂蚁属性 L2 |
| `noop_base_penalty` | -0.0025 | 无操作基础惩罚 |
| `noop_max_penalty` | -0.005 | 无操作最大惩罚 |

#### 终局奖励

| 参数 | 默认值 |
|------|--------|
| `win_reward` | 500.0 |
| `loss_reward` | -500.0 |

#### 裁剪

| 参数 | 默认值 | 作用阶段 |
|------|--------|---------|
| `step_reward_clip` | 500.0 | 步级裁剪（环境层） |
| `adv_clip` (clip_values.advantages_*) | ±1e4 | GAE 优势裁剪（训练层） |
| `returns_clip` (clip_values.returns_*) | ±1e6 | GAE 回报裁剪（训练层） |

---

## 10. 设计理念与决策

### 10.1 为什么采用多组件奖励而非简单的胜率奖励？

如果只给终局 ±500，其他步奖励为 0，则奖励信号极其稀疏（一场可能 200+ 回合的对局仅有最后一步有信号）。多组件奖励提供了 **稠密的中间反馈**，让策略在每一步都能学到一些东西，大大加速了训练。

### 10.2 为什么回合级奖励是非零和的？

回合级奖励中，塔存活、科技加成、余额奖励、死亡惩罚等组件对双方独立计算（不是互为相反数），这使得回合级奖励 **非零和**。这样设计的原因：

- 提供比零和博弈更丰富的学习信号
- 鼓励双方发展经济（科技、塔），而非纯粹进攻
- 终局奖励 ±500 确保了游戏的最终目标是零和的

### 10.3 动作级奖励和回合级奖励的关系？

| 维度 | 动作级奖励 | 回合级奖励 |
|------|----------|----------|
| 粒度 | 单个动作 | 完整回合 |
| 延迟 | 即时 | 回合结束后 |
| 范围 | ±15 | ±500 |
| 反馈内容 | 动作"对 or 错"的启发式判断 | 动作"实际效果"的客观衡量 |

两者的叠加使策略既能获得即时的行为引导，又能从长期后果中学习。

### 10.4 为什么前 1000 回合禁用降级塔？

降级塔的动作奖励为 -9.0（强烈惩罚），但在训练初期模型可能随机探索到此动作。通过前 1000 回合的动作掩码直接屏蔽降级选项，是 **课程学习（Curriculum Learning）** 策略的一部分。

### 10.5 金币收入的"被动收入"修复

`coins_after_ops` 机制是一个关键修复点：如果简单用 `new_coins - old_coins` 计算 coin_gain，会把建塔/升级的花费计入"收入"，导致错误的奖励信号。分离出操作花费，只奖励蚂蚁采集的被动收入，是 v2 版本对 v1 的重要改进。

---

## 11. 调优与扩展建议

### 11.1 如何调整奖励以改变策略行为

| 希望策略更... | 调整方式 |
|-------------|---------|
| 激进进攻 | 提高 `base_hp_attack_weight`、`tower_hp_attack_weight` |
| 注重防御 | 提高 `tower_survival_per_tower`、等级倍率 |
| 快速发展经济 | 提高 `own_coin_gain_weight`、科技升级动作奖励 |
| 使用超级武器 | 提高对应武器的 `deploy_*` 值 |
| 避免无操作 | 提高 `noop_base_penalty` 的绝对值 |

### 11.2 奖励幅度的平衡原则

- 所有回合级奖励分量的典型值应在同一数量级
- 动作级奖励的典型值应明显小于回合级奖励（保持层级结构清晰）
- 终局奖励的绝对值应远大于任何单步奖励（强调最终目标）
- clip 范围应足够容纳大部分有效奖励而不被截断

### 11.3 监控指标

建议关注以下指标来判断奖励设计是否合理：

- `reward_mean` / `reward_std`：奖励分布是否稳定
- `reward_sources` 中各分量的占比：是否有某分量主导
- `value_loss`：价值网络是否能准确预测回报
- `explained_variance`：价值预测与实际回报的相关性
