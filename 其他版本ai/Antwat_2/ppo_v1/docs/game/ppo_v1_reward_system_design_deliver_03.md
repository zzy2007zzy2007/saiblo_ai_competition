# PPO v1 奖励系统现状 — Deliver 03

> 版本：v3.0（基于实际代码审计重建）
> 日期：2026-05-25
> 
> **声明**：本文档所有数值和逻辑均直接提取自源代码，与代码完全一致。
> 如未来修改代码，请同步更新本文档。
>
> 源代码参考：
> - [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L154-L187) — `REWARD_CONFIG` 定义
> - [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) — 奖励计算逻辑
> - [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) — GAE/裁剪/归一化
> - [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py) — reward 提取
> - [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml) — PPO 超参数

---

## 1. 系统架构

奖励计算分为两个层级：

1. **Env 层** — `AntWarEnv` 负责每步的原始奖励计算，包含动作奖励和回合奖励
2. **Trainer 层** — `PPOTrainer` 负责 GAE 折现、returns 归一化、values/advantages 裁剪

Env 层的奖励计算在 [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) 中实现，Trainer 层的处理在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) 中实现。

### 1.1 数据流

```
env._compute_action_reward()      → 动作奖励（每步，立即计算）
env._resolve_turn_from_ops()      → 回合奖励（每回合结算时计算）
         ↓
action_reward + 回合级子项 → raw_reward（每步）
         ↓
step_reward_clip [-20, +20]               ← env 层裁剪
         ↓
selfplay.py: rewards.get("player_{p}", 0) ← 根据 swap 提取己方 reward
         ↓
EpisodeBatch.rewards.append()             → batch_pool 收集
         ↓
tensors['rewards']                       → ppo_trainer.py 转为 Tensor
         ↓
GAE(γ=0.99, λ=0.95)                     → _compute_gae()
         → advantages, returns
         ↓
values clamp [-200, 200]                 ← VALUES_MIN / VALUES_MAX
returns clamp [-1e6, 1e6]                ← RETURNS_MIN / RETURNS_MAX
advantages clamp [-1e4, 1e4]             ← ADVANTAGES_MIN / ADVANTAGES_MAX
         ↓
returns 全局归一化（零均值单位方差）
         ↓
mini-batch 内 advantages 再归一化（零均值单位方差）
         ↓
PPO 更新: clipped ratio × advantages
```

### 1.2 奖励层级

| 层级 | 方法 | 粒度 | 代码位置 |
|:----|:-----|:-----|:---------|
| 动作奖励 | `_compute_action_reward()` | 每步（立即），先于游戏逻辑 | L216-L292 |
| 回合奖励 | `_resolve_turn_from_ops()` 中的回合子项 | 每回合结算时，后于 `advance_round()` | L142-L180 |
| 终局奖励 | `_resolve_turn_from_ops()` 中的终局分支 | 终局回合 | L184-L194 |

每步最终 reward = **(动作奖励 + 所有回合子项)**，再经 `step_reward_clip` 裁剪到 [-20, +20]。

---

## 2. REWARD_CONFIG（精确值）

所有奖励参数集中在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L154-L187) 的 `REWARD_CONFIG` 字典中：

```python
REWARD_CONFIG = {
    # === 回合级奖励 ===
    "hp_attack_weight": 0.3,
    "own_coin_gain_weight": 0.02,
    "enemy_coin_gain_weight": 0.01,
    "tower_survival_per_tower": 0.20,
    "enemy_tower_survival_per_tower": 0.10,
    "tower_survival_level_multipliers": [1.0, 1.5, 2.5],
    "own_die_penalty_per_ant": -0.05,

    # === 动作奖励 ===
    "build_tower_tiers": [0.6, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1],
    "upgrade_tower_l2": 0.6,
    "upgrade_tower_l3": 1.0,
    "downgrade_tower_penalty": -3.15,
    "upgrade_gen_speed": [1.0, 0.8],
    "upgrade_gen_ant": [0.5, 0.1],
    "deploy_ls": 0.5,
    "deploy_emp": 0.8,
    "deploy_deflector": 0.3,
    "deploy_evasion": 0.2,

    # === 科技持续加成 ===
    "tech_speed_per_level": 0.10,
    "tech_hp_per_level": 0.06,

    # === NO-OP 惩罚 ===
    "noop_tolerance": 3,
    "noop_base_penalty": -0.02,
    "noop_max_penalty": -0.40,

    # === 终局奖励 ===
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "efficiency_max_rounds": 200,
    "efficiency_bonus_max": 30.0,

    # === 裁剪 ===
    "step_reward_clip": 20.0,
}
```

---

## 3. 动作奖励（代码验证精确）

### 3.1 方法

[antwar_env.py:_compute_action_reward()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L216-L292)

每步在 `_resolve_turn_from_ops()` 中先于游戏逻辑执行（L113-L114），立即计算该步动作的奖励值。

### 3.2 各动作奖励值

| 动作 | 奖励值 | 条件说明 |
|:----|:------:|:---------|
| NO-OP（第 1~3 步） | 0.0 | `noop_tolerance=3`，前 3 步不惩罚 |
| NO-OP（第 4 步） | -0.02 | 见下方公式 |
| NO-OP（第 8 步） | -0.04 | 每 5 步阶梯递增 |
| NO-OP（第 13 步） | -0.06 | |
| NO-OP（第 18 步） | -0.08 | |
| NO-OP（第 23 步） | -0.10 | |
| NO-OP（第 28 步） | -0.12 | |
| NO-OP（第 33 步） | -0.14 | |
| NO-OP（第 38 步） | -0.16 | |
| NO-OP（第 43 步） | -0.18 | |
| NO-OP（第 48 步） | -0.20 | |
| NO-OP（第 53 步） | -0.22 | |
| NO-OP（第 58 步） | -0.24 | |
| NO-OP（第 63 步） | -0.26 | |
| NO-OP（第 68 步） | -0.28 | |
| NO-OP（第 73 步） | -0.30 | |
| NO-OP（第 78 步） | -0.32 | |
| NO-OP（第 83 步） | -0.34 | |
| NO-OP（第 88 步） | -0.36 | |
| NO-OP（第 93 步） | -0.38 | |
| **NO-OP（第 98 步+）** | **-0.40（封顶）** | 达到 `noop_max_penalty` |
| BUILD_TOWER（同一塔位第 1 次） | +0.6 | |
| BUILD_TOWER（同一塔位第 2 次） | +0.5 | |
| BUILD_TOWER（同一塔位第 3 次） | +0.5 | |
| BUILD_TOWER（同一塔位第 4 次） | +0.3 | |
| BUILD_TOWER（同一塔位第 5 次） | +0.3 | |
| BUILD_TOWER（同一塔位第 6 次） | +0.3 | |
| BUILD_TOWER（同一塔位第 7 次+） | +0.1 | 超出 tiers 长度后取 `tiers[-1]` |
| UPGRADE_TOWER → L2 | +0.6 | `target_type < 100` |
| UPGRADE_TOWER → L3 | +1.0 | `target_type >= 100` |
| DOWNGRADE_TOWER | -3.15 | |
| UPGRADE_GENERATION_SPEED L0→L1 | +1.0 | `generation_level - 1 = 0` |
| UPGRADE_GENERATION_SPEED L1→L2 | +0.8 | `generation_level - 1 = 1` |
| UPGRADE_GENERATED_ANT L0→L1 | +0.5 | `ant_level - 1 = 0` |
| UPGRADE_GENERATED_ANT L1→L2 | +0.1 | `ant_level - 1 = 1` |
| USE_LIGHTNING_STORM | +0.5 | |
| USE_EMP_BLASTER | +0.8 | |
| USE_DEFLECTOR | +0.3 | |
| USE_EMERGENCY_EVASION | +0.2 | |

### 3.3 NO-OP 惩罚公式（代码精确）

```python
if n > cfg["noop_tolerance"]:  # 超过 tolerance(3) 步才开始惩罚
    penalty = cfg["noop_base_penalty"] + cfg["noop_base_penalty"] * ((n - cfg["noop_tolerance"]) // 5)
    reward = max(penalty, cfg["noop_max_penalty"])
```

其中：
- `n` = 连续 NO-OP 步数（包括当前步）
- `noop_tolerance` = 3（前 3 步不惩罚）
- `noop_base_penalty` = **-0.02**
- 步进除数 = **//5**（每 5 步递增一个阶梯）
- `noop_max_penalty` = **-0.40**（封顶）

公式展开：

```
penalty = -0.02 + (-0.02) × ((n - 3) // 5)
```

实际惩罚值表：

| n（连续 NO-OP 步数） | (n - 3) // 5 | penalty | 最终 reward（取 max） |
|:---:|:---:|:---:|:---:|
| 1~3 | — | — | 0.0（未超过 tolerance） |
| 4~7 | 0 | -0.02 | -0.02 |
| 8~12 | 1 | -0.04 | -0.04 |
| 13~17 | 2 | -0.06 | -0.06 |
| 18~22 | 3 | -0.08 | -0.08 |
| ... | ... | ... | ... |
| 93~97 | 18 | -0.38 | -0.38 |
| 98+ | 19 | -0.40 | **-0.40（封顶）** |

### 3.4 建塔奖励递减机制

首次在某塔位建塔递进 `tiers[0]`，后续该塔位每重建一次递减：

| 在该塔位的建塔次数（含本次） | 奖励 |
|:---:|:---:|
| 第 1 次 | 0.6 |
| 第 2 次 | 0.5 |
| 第 3 次 | 0.5 |
| 第 4 次 | 0.3 |
| 第 5 次 | 0.3 |
| 第 6 次 | 0.3 |
| 第 7 次及以后 | 0.1（最低值） |

追踪变量 `_tower_build_slot_counts[player][local_idx]` 在 `reset()` 时重置。

---

## 4. 回合奖励（代码验证精确）

### 4.1 方法

[antwar_env.py:_resolve_turn_from_ops()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L105-L214)

每回合双方行动后、`state.advance_round()` 之后计算。逻辑内联在该方法中。

### 4.2 执行时序

```
1. 记录 hp_before, die_before                              ← L110-L111
2. _compute_action_reward(p0)                              ← L113
3. _compute_action_reward(p1)                              ← L114
4. 应用双方操作（apply_self_operations + apply_opponent_operations） ← L119-L122
5. 记录 coins_after_ops                                    ← L124
6. 记录 tower_hp_before                                    ← L126-L129
7. state.advance_round()                                   ← L131
8. 计算回合级奖励                                           ← L142-L194
9. 叠加动作奖励到回合奖励                                   ← L196
10. 裁剪到 [-20, +20]                                     ← L210-L212
```

### 4.3 回合奖励公式

```python
cfg = REWARD_CONFIG
rewards = {"player_0": action_reward_0, "player_1": action_reward_1}  # 先以动作奖励为基底

for p in (0, 1):
    enemy = 1 - p
    agent = f"player_{p}"
    round_reward = 0.0                                    # 用于累加回合级子项

    # ── 4.3.1 HP 攻击加权 ──
    enemy_base_damage = hp_before[enemy] - state.bases[enemy].hp
    enemy_towers_after = {t.tower_id: t.hp for t in state.towers_of(enemy)}
    enemy_tower_damage = 0
    for tid, hp_b4 in tower_hp_before[enemy].items():
        hp_after = enemy_towers_after.get(tid, 0)
        enemy_tower_damage += max(0, hp_b4 - hp_after)
    hp_attack_val = (enemy_base_damage + enemy_tower_damage) * cfg["hp_attack_weight"]
    round_reward += hp_attack_val

    # ── 4.3.2 金币收入加权 ──
    own_income = state.coins[p] - coins_after_ops[p]
    enemy_income = state.coins[enemy] - coins_after_ops[enemy]
    coin_gain_val = own_income * cfg["own_coin_gain_weight"] - enemy_income * cfg["enemy_coin_gain_weight"]
    round_reward += own_income * cfg["own_coin_gain_weight"]
    round_reward -= enemy_income * cfg["enemy_coin_gain_weight"]

    # ── 4.3.3 塔存活加权 ──
    multipliers = cfg["tower_survival_level_multipliers"]
    own_tower_value = 0
    for t in state.towers_of(p):
        level = min(t.level, len(multipliers) - 1)
        own_tower_value += cfg["tower_survival_per_tower"] * multipliers[level]
    enemy_tower_value = len(state.towers_of(enemy)) * cfg["enemy_tower_survival_per_tower"]
    tower_survival_val = own_tower_value - enemy_tower_value
    round_reward += own_tower_value
    round_reward -= enemy_tower_value

    # ── 4.3.4 科技等级加成 ──
    tech_bonus_val = state.bases[p].generation_level * cfg["tech_speed_per_level"] \
                     + state.bases[p].ant_level * cfg["tech_hp_per_level"]
    round_reward += tech_bonus_val

    # ── 4.3.5 蚂蚁死亡惩罚 ──
    own_die_delta = state.die_count[p] - die_before[p]
    enemy_die_delta = state.die_count[enemy] - die_before[enemy]
    die_penalty_val = own_die_delta * cfg["own_die_penalty_per_ant"] \
                      - enemy_die_delta * cfg["own_die_penalty_per_ant"]
    round_reward += own_die_delta * cfg["own_die_penalty_per_ant"]
    round_reward -= enemy_die_delta * cfg["own_die_penalty_per_ant"]

    # ── 4.3.6 终局奖励 ──
    end_reward_val = 0.0
    speed_bonus_val = 0.0
    if terminated:
        if state.winner == p:
            total_rounds = state.round_index
            speed_bonus_val = max(0, (cfg["efficiency_max_rounds"] - total_rounds)
                                  / cfg["efficiency_max_rounds"]) \
                              * cfg["efficiency_bonus_max"]
            end_reward_val = cfg["win_reward"] + speed_bonus_val    # 100.0 + bonus
            round_reward += cfg["win_reward"]                       # +100.0
            round_reward += speed_bonus_val                         # +bonus
        elif state.winner == enemy:
            end_reward_val = cfg["loss_reward"]                     # -100.0
            round_reward += cfg["loss_reward"]                      # -100.0

    # 回合级奖励叠加到动作奖励上
    rewards[agent] += round_reward                                   # L196
```

### 4.4 各子项公式

#### 4.4.1 HP 攻击加权

```
hp_attack_val = (敌方基地扣血量 + 敌方塔扣血量总和) × 0.3
```

- `enemy_base_damage` = 本回合开始时的 HP − 本回合结束时的 HP（即敌方基地本回合受到的伤害）
- `enemy_tower_damage` = 本回合敌方所有塔 HP 减少量之和（每座塔分别计算 `max(0, before - after)`）
- 权重：**0.3**

#### 4.4.2 金币收入加权

```
coin_gain_val = own_income × 0.02 - enemy_income × 0.01
```

- `own_income` = `state.coins[p] - coins_after_ops[p]`（操作执行后到回合结算之间的金币增量）
- `enemy_income` = 同理
- 己方权重：**0.02**，敌方权重：**0.01**

#### 4.4.3 塔存活加权

```
own_tower_value   = Σ(0.20 × 等级倍数)     for each own tower
enemy_tower_value = 敌方塔数量 × 0.10
tower_survival_val = own_tower_value - enemy_tower_value
```

等级倍数：L0=×1.0（=0.20/塔），L1=×1.5（=0.30/塔），L2=×2.5（=0.50/塔）

```python
level = min(t.level, len(multipliers) - 1)  # 防止 level 超出数组长度
own_tower_value += cfg["tower_survival_per_tower"] * multipliers[level]
```

#### 4.4.4 科技等级加成

```
tech_bonus_val = gen_speed_level × 0.10 + ant_hp_level × 0.06
```

- 生产速度每级 +0.10/回合
- 兵蚁血量每级 +0.06/回合

#### 4.4.5 蚂蚁死亡惩罚

```
die_penalty_val = own_die_delta × (-0.05) - enemy_die_delta × (-0.05)
                = own_die_delta × (-0.05) + enemy_die_delta × 0.05
```

- 己方每死一只蚂蚁：**-0.05**
- 敌方每死一只蚂蚁：**+0.05**

#### 4.4.6 终局奖励

**胜方**：
```
end_reward_val = 100.0 + speed_bonus_val
speed_bonus_val = max(0, (200 - total_rounds) / 200) × 30.0
```
- 基础胜利奖励：**+100.0**
- 速度奖励上限：**+30.0**
- 速度奖励仅在终局回合发放

速度奖励示例：
| 终局回合数 | speed_bonus |
|:---------:|:-----------:|
| 200 | 0.0 |
| 100 | +15.0 |
| 50 | +22.5 |
| 1 | +30.0 |

**败方**：
```
end_reward_val = -100.0
```
- 失败惩罚：**-100.0**

### 4.5 重要：终局奖励的裁剪效果

所有奖励在回合结束时统一经 `step_reward_clip` 裁剪到 [-20, +20]：

```python
clip = cfg["step_reward_clip"]    # 20.0
rewards[agent] = max(min(rewards[agent], clip), -clip)
```

这意味着：**终局回合的 win_reward=100 会被裁剪到 +20，loss_reward=-100 会被裁剪到 -20。** 实际 agent 在终局回合接收到的 terminal reward 是 [-20, +20] 范围内的值，不是 100 或 -100。

---

## 5. 奖励裁剪

### 5.1 Env 层裁剪

[antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L210-L212)：

```python
clip = cfg["step_reward_clip"]    # 20.0
rewards[agent] = max(min(rewards[agent], clip), -clip)
```

每步总 reward（动作奖励 + 回合奖励 + 终局奖励）裁剪到 [-20, +20]。

### 5.2 Trainer 层裁剪

[ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L155-L167) 在 GAE 计算后执行裁剪，参数来自 [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L32-L46)：

| 参数 | 范围 | 代码变量 | 裁剪时机 |
|:----|:----:|:---------|:---------|
| logits | [-20, 20] | `LOGITS_MIN/MAX` | 前向传播时 |
| values | [-200, 200] | `VALUES_MIN/MAX` | GAE 计算前（old values），新 value 输出后 |
| returns | [-1e6, 1e6] | `RETURNS_MIN/MAX` | GAE 计算后 |
| advantages | [-1e4, 1e4] | `ADVANTAGES_MIN/MAX` | GAE 计算后 |
| ratio | [1e-5, 1e5] | `RATIO_MIN/MAX` | importance sampling ratio |
| loss | ≤ 1e6 | `LOSS_MAX` | 总 loss 上限保护 |

```yaml
# ppo_antwar.yaml L32-L46
clip_values:
  logits_min: -20
  logits_max: 20
  values_min: -200
  values_max: 200
  returns_min: -1e6
  returns_max: 1e6
  advantages_min: -1e4
  advantages_max: 1e4
  prob_min: 1e-10
  ratio_min: 1e-5
  ratio_max: 1e5
  loss_max: 1e6
  nan_threshold: 5
```

### 5.3 Returns 归一化

[ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L694-L696)：

```python
return_mean = full_returns.mean()
return_std = full_returns.std() + 1e-8
full_returns = (full_returns - return_mean) / return_std
```

### 5.4 Mini-batch 内 Advantages 再归一化

[ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L720)：

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## 6. PPO 训练中的 Reward 处理流程

### 6.1 数据收集

[selfplay.py:_collect_episode_with_swap()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L581-L585)：

```python
if not swap_positions:
    reward = rewards.get("player_0", 0.0)
else:
    reward = rewards.get("player_1", 0.0)
batch.rewards.append(reward)
```

### 6.2 GAE 计算

[ppo_trainer.py:_compute_gae()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L836-L858)：

```python
gamma = 0.99        # ppo_antwar.yaml L13
gae_lambda = 0.95   # ppo_antwar.yaml L14

for t in reversed(range(len(rewards))):
    if t == len(rewards) - 1:
        next_value = 0.0
    else:
        next_value = values[t + 1]
    delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
    running_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * running_advantage
    advantages[t] = running_advantage

returns = advantages + values.detach()
```

### 6.3 完整数据流

```
raw reward (env)                     [-20, +20]          step_reward_clip
    ↓
values clamp                         [-200, 200]         VALUES_MIN / VALUES_MAX
    ↓
GAE(γ=0.99, λ=0.95)
    → returns                        [-1e6, 1e6]         RETURNS_MIN / RETURNS_MAX
    → advantages                     [-1e4, 1e4]         ADVANTAGES_MIN / ADVANTAGES_MAX
    ↓
returns 全局归一化                   零均值单位方差
    ↓
mini-batch advantages 再归一化      零均值单位方差
    ↓
PPO 更新:
  - policy loss: -E[ min(r×A, clip(r, 1-ε, 1+ε)×A) ]
  - value loss:  clipped MSE
  - entropy bonus: S(π) × ent_coef
  - hard entropy penalty: if entropy < target
```

---

## 7. 状态追踪变量

[antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L21-L26)：

```python
self._tower_build_slot_counts: dict[int, dict[int, int]] = {
    0: {i: 0 for i in range(8)},   # player_0 的 8 个塔位
    1: {i: 0 for i in range(8)},   # player_1 的 8 个塔位
}
self._player_noop_streak: dict[int, int] = {0: 0, 1: 0}
self._last_reward_detail: dict = {}
```

在 [reset()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L28-L45) 中重置：

```python
self._tower_build_slot_counts = {
    0: {i: 0 for i in range(8)},
    1: {i: 0 for i in range(8)},
}
self._player_noop_streak = {0: 0, 1: 0}
self._last_reward_detail = {}
```

---

## 8. _last_reward_detail 结构

每回合结束时构造，写入到 `self._last_reward_detail[agent]`：

[antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L198-L208)：

```python
self._last_reward_detail[agent] = {
    'action_reward': round(action_reward_val, 4),     # 动作奖励
    'hp_attack': round(hp_attack_val, 4),             # HP 攻击加权
    'coin_gain': round(coin_gain_val, 4),             # 金币收入加权
    'tower_survival': round(tower_survival_val, 4),   # 塔存活加权
    'tech_bonus': round(tech_bonus_val, 4),           # 科技等级加成
    'die_penalty': round(die_penalty_val, 4),         # 蚂蚁死亡惩罚
    'end_reward': round(end_reward_val, 4),           # 终局奖励（不含 speed_bonus）
    'speed_bonus': round(speed_bonus_val, 4),         # 速度奖励（仅终局回合非零）
}
```

该结构被 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L589-L598) 写入对战日志 `battle_details['actions'][step]['reward_detail']`。

**注意**：各子项**不裁剪**，只有最终叠加后的总 reward 会被 `step_reward_clip` 裁剪。

---

## 9. 与 SDK `_round_rewards()` 的对比

| 维度 | SDK `_round_rewards()` (Ant-Game/SDK/training/env.py) | 当前 `AntWarEnv` (ppo/src/ppo_antwar/env/antwar_env.py) |
|:----|:-----------------------------------------------------|:-------------------------------------------------------|
| 文件位置 | Ant-Game/SDK/training/env.py | ppo/src/ppo_antwar/env/antwar_env.py |
| 导入 REWARD_CONFIG | 否（硬编码常数） | 是（从 action_constants.py 导入） |
| 动作奖励 | 无 | 有（`_compute_action_reward`） |
| NO-OP 惩罚 | 无 | 有（渐进惩罚，封顶 -0.40） |
| HP 攻击加权 | ×10.0（硬编码） | ×0.3（配置化） |
| 金币收入 | 己方 ×0.05 / 敌方 ×0.02 | 己方 ×0.02 / 敌方 ×0.01 |
| 塔存活奖励 | 无 | 有（含等级加权） |
| 科技等级加成 | 无 | 有 |
| 蚂蚁死亡惩罚 | 无 | 有（±0.05/只） |
| 非法动作惩罚 | -1.0 | 无（非法动作被 mask 过滤） |
| 终局奖励 | ±100.0 | ±100.0（裁剪后实际 ±20） |
| 单步裁剪 | 无 | [-20, +20] |
| 建塔奖励递减 | 无 | 7 级递减 (0.6→0.1) |
| 降塔惩罚 | 无 | -3.15 |
| 超级武器奖励 | 无 | 0.2~0.8 |

当前 PPO v1 训练使用的 `AntWarEnv` 位于 [ppo/src/ppo_antwar/env/antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py)。SDK 的 `_round_rewards()` 位于 [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L152-L168)，未被当前训练代码引用。

---

## 10. 代码索引

| 文件 | 位置 | 内容 |
|:----|:------|:-----|
| [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py) | L154-L187 | `REWARD_CONFIG` 定义 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L11 | 导入 `REWARD_CONFIG` |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L21-L26 | 状态追踪变量初始化 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L28-L45 | `reset()` — 变量重置 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L76-L82 | `_resolve_turn()` — action_id → op → 回合结算 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L105-L214 | `_resolve_turn_from_ops()` — 回合奖励 + 终局奖励 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L142-L194 | 回合奖励计算循环 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L198-L208 | `_last_reward_detail` 构造 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L210-L212 | `step_reward_clip` 裁剪 [-20, +20] |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L216-L292 | `_compute_action_reward()` — 动作奖励 |
| [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py) | L581-L585 | reward 提取（根据 swap 取 player_0/1） |
| [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py) | L589-L598 | reward_detail 写入对战日志 |
| [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | L666-L696 | 奖励统计 + values 裁剪 + GAE + returns 裁剪/归一化 |
| [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | L720 | mini-batch advantages 再归一化 |
| [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | L836-L867 | `_compute_gae()` / `_compute_returns()` |
| [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml) | L13-L14 | gamma=0.99, gae_lambda=0.95 |
| [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml) | L32-L46 | `clip_values` 全部边界参数 |
