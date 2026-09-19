# PPO v1 奖励系统现状 — Deliver 02

> 版本：v2.0
> 日期：2026-05-24
> 本文档记录 PPO v1 当前实际运行的奖励系统代码。
> 所有数值和逻辑均可追溯至源代码：
> - [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py)
> - [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py)
> - [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py)
> - [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py)

---

## 1. 系统架构

奖励计算分为两个层级：

1. **Env 层** — `AntWarEnv` 负责每步的原始奖励计算，包含动作奖励和回合奖励
2. **Trainer 层** — `PPOTrainer` 负责 GAE 折现、returns 归一化、values/advantages 裁剪

Env 层的奖励计算在 [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) 中实现，Trainer 层的处理在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) 中实现。

### 1.1 数据流

```
env._compute_action_reward()      → 动作奖励（每步，立即）
env._resolve_turn_from_ops()      → 回合奖励（每回合结束）
         ↓
raw reward (float)
         ↓
EpisodeBatch.rewards.append()     → selfplay.py 收集
         ↓
tensors['rewards']                → ppo_trainer.py 转为 Tensor
         ↓
GAE(γ=0.99, λ=0.95)             → _compute_gae()
         returns = advantages + values
         ↓
values clamp [-200, 200]
returns clamp [-1e6, 1e6]
advantages clamp [-1e4, 1e4]
         ↓
returns 全局归一化（零均值单位方差）
         ↓
mini-batch 内 advantages 再归一化（零均值单位方差）
```

### 1.2 奖励来源

| 层级 | 方法 | 粒度 |
|:----|:-----|:-----|
| 动作奖励 | `_compute_action_reward()` | 每步 |
| 回合奖励 | `_resolve_turn_from_ops()` | 每回合 |
| 终局奖励 | `_resolve_turn_from_ops()` 中的条件分支 | 终局 |

---

## 2. REWARD_CONFIG

所有奖励参数集中在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L154-L187) 的 `REWARD_CONFIG` 字典中：

```python
REWARD_CONFIG = {
    # 回合级
    "hp_attack_weight": 0.3,
    "own_coin_gain_weight": 0.02,
    "enemy_coin_gain_weight": 0.01,
    "tower_survival_per_tower": 0.20,
    "enemy_tower_survival_per_tower": 0.10,
    "tower_survival_level_multipliers": [1.0, 1.5, 2.5],
    "own_die_penalty_per_ant": -0.05,

    # 动作奖励
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

    # 科技持续加成
    "tech_speed_per_level": 0.10,
    "tech_hp_per_level": 0.06,

    # NO-OP 惩罚
    "noop_tolerance": 3,
    "noop_base_penalty": -0.04,
    "noop_max_penalty": -0.80,

    # 终局
    "win_reward": 2500.0,
    "loss_reward": -2500.0,
    "efficiency_max_rounds": 200,
    "efficiency_bonus_max": 30.0,

    # 裁剪
    "step_reward_clip": 20.0,
}
```

`REWARD_CONFIG` 的消费方是 [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L11)：

```python
from ..utils.action_constants import REWARD_CONFIG, OperationType, TOWER_POSITIONS
```

### 调优历史

| 轮次 | 改动 | 对应字段 |
|:----|:-----|:---------|
| 第 1 轮 | 初始值 | `noop_base_penalty: -0.005`, `win_reward: 100.0` |
| 第 7 轮结束 | NO-OP 惩罚翻倍 | `noop_base_penalty: -0.005 → -0.02` |
| 第 8 轮开始 | NO-OP 再翻倍 + win_reward 提升 5 倍 | `noop_base_penalty: -0.02 → -0.04`, `win_reward: 100 → 500` |
| 第 9 轮开始 | win_reward 再提升 5 倍 | `win_reward: 500 → 2500` |

---

## 3. 动作奖励

### 3.1 方法

[antwar_env.py:_compute_action_reward()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L216-L292)

每步在 `_resolve_turn_from_ops()` 中先于游戏逻辑执行，立即计算该步动作的奖励值。

### 3.2 逻辑

```python
def _compute_action_reward(self, player: int, op) -> float:
    cfg = REWARD_CONFIG
    reward = 0.0

    if op is None or op.op_type == OperationType.NO_OP:
        self._player_noop_streak[player] += 1
        n = self._player_noop_streak[player]
        if n > cfg["noop_tolerance"]:
            penalty = cfg["noop_base_penalty"] + cfg["noop_base_penalty"] * ((n - cfg["noop_tolerance"]) // 10)
            reward = max(penalty, cfg["noop_max_penalty"])
        return reward

    self._player_noop_streak[player] = 0

    op_type = op.op_type

    if op_type == OperationType.BUILD_TOWER:
        x, y = op.arg0, op.arg1
        try:
            global_idx = TOWER_POSITIONS.index((x, y))
        except ValueError:
            global_idx = -1
        if global_idx >= 0:
            local_idx = global_idx if player == 0 else global_idx - 8
            if 0 <= local_idx <= 7:
                count = self._tower_build_slot_counts[player][local_idx]
                tiers = cfg["build_tower_tiers"]
                if count < len(tiers):
                    reward += tiers[count]
                else:
                    reward += tiers[-1]
                self._tower_build_slot_counts[player][local_idx] += 1

    elif op_type == OperationType.UPGRADE_TOWER:
        target_type = op.arg1
        if target_type >= 100:
            reward += cfg["upgrade_tower_l3"]
        else:
            reward += cfg["upgrade_tower_l2"]

    elif op_type == OperationType.DOWNGRADE_TOWER:
        reward += cfg["downgrade_tower_penalty"]
        # 递减该位置建塔计数
        tower_id = op.arg0
        state = self._runtime.state
        tower = state.tower_by_id(tower_id) if hasattr(state, 'tower_by_id') else None
        if tower is not None:
            try:
                global_idx = TOWER_POSITIONS.index((tower.x, tower.y))
            except ValueError:
                global_idx = -1
            if global_idx >= 0:
                local_idx = global_idx if player == 0 else global_idx - 8
                if 0 <= local_idx <= 7:
                    old = self._tower_build_slot_counts[player][local_idx]
                    if old > 0:
                        self._tower_build_slot_counts[player][local_idx] = old - 1

    elif op_type == OperationType.UPGRADE_GENERATION_SPEED:
        level = self._runtime.state.bases[player].generation_level - 1
        if 0 <= level < len(cfg["upgrade_gen_speed"]):
            reward += cfg["upgrade_gen_speed"][level]

    elif op_type == OperationType.UPGRADE_GENERATED_ANT:
        level = self._runtime.state.bases[player].ant_level - 1
        if 0 <= level < len(cfg["upgrade_gen_ant"]):
            reward += cfg["upgrade_gen_ant"][level]

    elif op_type == OperationType.USE_LIGHTNING_STORM:
        reward += cfg["deploy_ls"]
    elif op_type == OperationType.USE_EMP_BLASTER:
        reward += cfg["deploy_emp"]
    elif op_type == OperationType.USE_DEFLECTOR:
        reward += cfg["deploy_deflector"]
    elif op_type == OperationType.USE_EMERGENCY_EVASION:
        reward += cfg["deploy_evasion"]

    return reward
```

### 3.3 各动作奖励值

| 动作 | 奖励值 |
|:----|:------:|
| NO-OP（1~3步） | 0.0 |
| NO-OP（第4步） | -0.04 |
| NO-OP（第14步） | -0.08 |
| NO-OP（第24步） | -0.12 |
| NO-OP（第34步） | -0.16 |
| NO-OP（第44步） | -0.20 |
| NO-OP（第54步） | -0.24 |
| NO-OP（第64步） | -0.28 |
| NO-OP（第74步） | -0.32 |
| NO-OP（≥第80步） | -0.80（封顶） |
| BUILD_TOWER（同一塔位第1次） | +0.6 |
| BUILD_TOWER（同一塔位第2次） | +0.5 |
| BUILD_TOWER（同一塔位第3次） | +0.5 |
| BUILD_TOWER（同一塔位第4次） | +0.3 |
| BUILD_TOWER（同一塔位第5次） | +0.3 |
| BUILD_TOWER（同一塔位第6次） | +0.3 |
| BUILD_TOWER（同一塔位第7次+） | +0.1 |
| UPGRADE_TOWER → L2 | +0.6 |
| UPGRADE_TOWER → L3 | +1.0 |
| DOWNGRADE_TOWER | -3.15 |
| UPGRADE_GENERATION_SPEED L0→L1 | +1.0 |
| UPGRADE_GENERATION_SPEED L1→L2 | +0.8 |
| UPGRADE_GENERATED_ANT L0→L1 | +0.5 |
| UPGRADE_GENERATED_ANT L1→L2 | +0.1 |
| USE_LIGHTNING_STORM | +0.5 |
| USE_EMP_BLASTER | +0.8 |
| USE_DEFLECTOR | +0.3 |
| USE_EMERGENCY_EVASION | +0.2 |

### 3.4 NO-OP 惩罚公式

```
penalty = noop_base_penalty + noop_base_penalty * ((n - noop_tolerance) // 10)
reward = max(penalty, noop_max_penalty)
```

其中：
- `n` = 连续 NO-OP 步数（包括当前步）
- `noop_tolerance` = 3（前3步不惩罚）
- `noop_base_penalty` = -0.04
- `noop_max_penalty` = -0.80

### 3.5 状态追踪变量

[antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L21-L26)：

```python
self._tower_build_slot_counts: dict[int, dict[int, int]] = {
    0: {i: 0 for i in range(8)},   # player_0 的 8 个塔位
    1: {i: 0 for i in range(8)},   # player_1 的 8 个塔位
}
self._player_noop_streak: dict[int, int] = {0: 0, 1: 0}
self._last_reward_detail: dict = {}
```

在 `reset()` 中重置：

```python
self._tower_build_slot_counts = {
    0: {i: 0 for i in range(8)},
    1: {i: 0 for i in range(8)},
}
self._player_noop_streak = {0: 0, 1: 0}
self._last_reward_detail = {}
```

---

## 4. 回合奖励

### 4.1 方法

[antwar_env.py:_resolve_turn_from_ops()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L105-L214)

每回合双方行动后、`state.advance_round()` 之后计算。逻辑内联在该方法中。

### 4.2 执行时序

```
1. 记录 hp_before, die_before
2. _compute_action_reward(p0)
3. _compute_action_reward(p1)
4. 应用双方操作（apply_self_operations + apply_opponent_operations）
5. 记录 coins_after_ops
6. 记录 tower_hp_before
7. state.advance_round()
8. 计算回合级奖励
9. 叠加动作奖励
10. 裁剪到 [-20, +20]
```

### 4.3 回合奖励公式

```python
cfg = REWARD_CONFIG
rewards = {"player_0": action_reward_0, "player_1": action_reward_1}

for p in (0, 1):
    enemy = 1 - p
    agent = f"player_{p}"
    round_reward = 0.0

    # 4.3.1 HP 攻击加权
    enemy_base_damage = hp_before[enemy] - state.bases[enemy].hp
    enemy_towers_after = {t.tower_id: t.hp for t in state.towers_of(enemy)}
    enemy_tower_damage = 0
    for tid, hp_b4 in tower_hp_before[enemy].items():
        hp_after = enemy_towers_after.get(tid, 0)
        enemy_tower_damage += max(0, hp_b4 - hp_after)
    hp_attack_val = (enemy_base_damage + enemy_tower_damage) * cfg["hp_attack_weight"]
    round_reward += hp_attack_val

    # 4.3.2 金币收入加权
    own_income = state.coins[p] - coins_after_ops[p]
    enemy_income = state.coins[enemy] - coins_after_ops[enemy]
    round_reward += own_income * cfg["own_coin_gain_weight"]
    round_reward -= enemy_income * cfg["enemy_coin_gain_weight"]

    # 4.3.3 塔存活加权
    multipliers = cfg["tower_survival_level_multipliers"]
    own_tower_value = 0
    for t in state.towers_of(p):
        level = min(t.level, len(multipliers) - 1)
        own_tower_value += cfg["tower_survival_per_tower"] * multipliers[level]
    enemy_tower_value = len(state.towers_of(enemy)) * cfg["enemy_tower_survival_per_tower"]
    round_reward += own_tower_value
    round_reward -= enemy_tower_value

    # 4.3.4 科技等级加成
    tech_bonus_val = state.bases[p].generation_level * cfg["tech_speed_per_level"] \
                     + state.bases[p].ant_level * cfg["tech_hp_per_level"]
    round_reward += tech_bonus_val

    # 4.3.5 蚂蚁死亡惩罚
    own_die_delta = state.die_count[p] - die_before[p]
    enemy_die_delta = state.die_count[enemy] - die_before[enemy]
    round_reward += own_die_delta * cfg["own_die_penalty_per_ant"]
    round_reward -= enemy_die_delta * cfg["own_die_penalty_per_ant"]

    # 4.3.6 终局奖励
    end_reward_val = 0.0
    speed_bonus_val = 0.0
    if terminated:
        if state.winner == p:
            total_rounds = state.round_index
            speed_bonus_val = max(0, (cfg["efficiency_max_rounds"] - total_rounds) / cfg["efficiency_max_rounds"]) \
                              * cfg["efficiency_bonus_max"]
            end_reward_val = cfg["win_reward"] + speed_bonus_val
            round_reward += cfg["win_reward"]
            round_reward += speed_bonus_val
        elif state.winner == enemy:
            end_reward_val = cfg["loss_reward"]
            round_reward += cfg["loss_reward"]

    rewards[agent] += round_reward
```

### 4.4 子项说明

#### 4.4.1 HP 攻击加权

```
hp_attack_val = (enemy_base_damage + enemy_tower_damage) × 0.3
```

- `enemy_base_damage` = 本回合敌方基地 HP 减少量
- `enemy_tower_damage` = 本回合敌方所有塔 HP 减少量之和
- 权重：0.3

#### 4.4.2 金币收入加权

```
coin_gain_val = own_income × 0.02 - enemy_income × 0.01
```

- `own_income` = 本回合己方金币增加量 = 当前金币 - 操作执行后的金币快照
- `enemy_income` = 本回合敌方金币增加量
- 己方权重：0.02，敌方权重：0.01

#### 4.4.3 塔存活加权

```
own_tower_value = Σ(0.20 × 等级倍数)
enemy_tower_value = 敌方塔数量 × 0.10
tower_survival_val = own_tower_value - enemy_tower_value
```

等级倍数：L0=×1.0（=0.20/塔），L1=×1.5（=0.30/塔），L2=×2.5（=0.50/塔）

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

- 己方每死一只蚂蚁：-0.05
- 敌方每死一只蚂蚁：+0.05

#### 4.4.6 终局奖励

胜方：
```
end_reward_val = 2500.0 + speed_bonus_val
speed_bonus_val = max(0, (200 - total_rounds) / 200) × 30.0
```

败方：
```
end_reward_val = -2500.0
```

速度奖励公式：`max(0, (200 - total_rounds) / 200) × 30.0`
- 100 回合结束：+15.0
- 50 回合结束：+22.5
- 1 回合结束：+30.0

### 4.5 终局回合的奖励计算

终局回合的总奖励 = 动作奖励 + 回合奖励 + 终局奖励，通过 `step_reward_clip` 裁剪到 [-20, +20]。

---

## 5. 奖励裁剪

### 5.1 Env 层裁剪

[antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py#L210-L212)：

```python
clip = cfg["step_reward_clip"]    # 20.0
rewards[agent] = max(min(rewards[agent], clip), -clip)
```

每步总 reward 裁剪到 [-20, +20]。

### 5.2 Trainer 层裁剪

[ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) 在 GAE 计算后执行裁剪，参数来自 [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L32-L46)：

| 参数 | 范围 | 裁剪时机 |
|:----|:----:|:---------|
| VALUES | [-200, 200] | GAE 计算前（old values），新 value 输出后 |
| RETURNS | [-1e6, 1e6] | GAE 计算后 |
| ADVANTAGES | [-1e4, 1e4] | GAE 计算后 |
| RATIO | [1e-5, 1e5] | importance sampling ratio |
| LOSS | ≤ 1e6 | 总 loss 上限保护 |

```python
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

---

## 6. PPO 训练中的 Reward 处理流程

### 6.1 数据收集

[selfplay.py:_collect_episode_with_swap()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L617-L622)：

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
gamma = 0.99
gae_lambda = 0.95

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

### 6.3 Returns 归一化

[ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L694-L696)：

```python
return_mean = full_returns.mean()
return_std = full_returns.std() + 1e-8
full_returns = (full_returns - return_mean) / return_std
```

### 6.4 Mini-batch 内 Advantages 再归一化

[ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L720)：

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 6.5 数据流

```
raw reward (env)                     [-20, +20]          step_reward_clip
    ↓
values clamp                         [-200, 200]         VALUES_MIN/VALUES_MAX
    ↓
GAE(γ=0.99, λ=0.95)
    → returns                        [-1e6, 1e6]         RETURNS_MIN/RETURNS_MAX
    → advantages                     [-1e4, 1e4]         ADVANTAGES_MIN/ADVANTAGES_MAX
    ↓
returns 全局归一化                   零均值单位方差
    ↓
mini-batch advantages 再归一化      零均值单位方差
    ↓
PPO 更新:
  - policy loss: clipped ratio × advantages
  - value loss:  clipped MSE
  - entropy bonus: S(π) × ent_coef
```

---

## 7. 与 SDK `_round_rewards()` 的对比

| 维度 | SDK `_round_rewards()` (Ant-Game/SDK/training/env.py) | 当前 `AntWarEnv` (ppo/src/ppo_antwar/env/antwar_env.py) |
|:----|:-----------------------------------------------------|:-------------------------------------------------------|
| 文件 | Ant-Game/SDK/training/env.py | ppo/src/ppo_antwar/env/antwar_env.py |
| 导入 REWARD_CONFIG | 否（硬编码常数） | 是 |
| 动作奖励 | 无 | 有（`_compute_action_reward`） |
| NO-OP 惩罚 | 无 | 有（渐进惩罚，封顶 -0.80） |
| HP 攻击加权 | ×10.0（硬编码） | ×0.3（配置化） |
| 金币收入 | 己方 ×0.05 / 敌方 ×0.02 | 己方 ×0.02 / 敌方 ×0.01 |
| 塔存活奖励 | 无 | 有（含等级加权） |
| 科技等级加成 | 无 | 有 |
| 蚂蚁死亡惩罚 | 无 | 有（±0.05/只） |
| 非法动作惩罚 | -1.0 | 无（非法动作被 mask 过滤） |
| 终局奖励 | ±100.0 | ±2500.0（含速度奖励最高 +30） |
| 单步裁剪 | 无 | [-20, +20] |
| 建塔奖励递减 | 无 | 7 级递减 (0.6→0.1) |
| 降塔惩罚 | 无 | -3.15 |
| 超级武器奖励 | 无 | 0.2~0.8 |

当前 PPO v1 训练使用的 `AntWarEnv` 位于 [ppo/src/ppo_antwar/env/antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py)。SDK 的 `_round_rewards()` 位于 [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L152-L168)，未被当前训练代码引用。

---

## 8. 代码索引

| 文件 | 位置 | 内容 |
|:----|:------|:-----|
| [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py) | L154-L187 | `REWARD_CONFIG` 定义 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L11 | 导入 `REWARD_CONFIG` |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L21-L26 | 状态追踪变量初始化 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L105-L214 | `_resolve_turn_from_ops()` — 回合奖励 + 终局奖励 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L139-L212 | 回合奖励计算 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L210-L212 | `step_reward_clip` 裁剪 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L216-L292 | `_compute_action_reward()` — 动作奖励 |
| [antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/env/antwar_env.py) | L28-L45 | `reset()` — 变量重置 |
| [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py) | L617-L622 | reward 提取 |
| [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | L666-L696 | 奖励统计 + values 裁剪 + GAE + returns 裁剪/归一化 |
| [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | L720 | mini-batch advantages 再归一化 |
| [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | L836-L867 | `_compute_gae()` / `_compute_returns()` |
| [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml) | L13-L14 | gamma=0.99, gae_lambda=0.95 |
| [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml) | L32-L46 | `clip_values` 全部边界参数 |
