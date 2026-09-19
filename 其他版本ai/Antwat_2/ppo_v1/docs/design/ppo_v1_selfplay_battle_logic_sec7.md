# 7. 奖励函数

## 7.1 奖励来源

奖励来自 `env.step()` 返回的 `rewards` 字典。env 实例由 `env_factory` 回调函数外部注入（参见 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L63) 的 `env_factory` 参数）：

```python
# 在 PPOTrainer.__init__() 中：
self.env_factory = env_factory  # 回调函数，由外部注入

# 在 train() 中：
env = self.env_factory()        # 运行时创建 env 实例
```

奖励的实际计算逻辑位于 **Ant-Game SDK** 的 `AntWarSequentialEnv` 类中，核心方法为 `_round_rewards()`。见 [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L152-L168)：

```python
# rewards 字典格式
rewards = {"player_0": float, "player_1": float}
```

**核心源文件**：

| 文件 | 说明 |
|------|------|
| [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py) | `AntWarSequentialEnv._round_rewards()` 奖励计算逻辑 |
| [trainer/selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py) | `_collect_episode_with_swap()` 中提取奖励 |
| [trainer/ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | `collect_episode()` 中提取奖励、GAE 计算 |

**补充说明**：PPO 训练框架不直接依赖 SDK 文件。env 由 `env_factory` 回调注入，框架通过标准的 `env.step()` / `env.reset()` API 与之交互。`env_factory` 返回的 env 实例应包装或适配 SDK 的 `AntWarSequentialEnv`，以提供兼容的接口。

---

## 7.2 SelfPlay 中奖励的获取方式

在 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L574-L588) 的 `_collect_episode_with_swap()` 中，每轮执行两次 `env.step()`：

```python
if not swap_positions:
    player_0_action = action
    player_1_action = opponent_action
else:
    player_0_action = opponent_action
    player_1_action = action

obs, _, _, _, _ = env.step(player_0_action)       # 先手 step，奖励恒为 0
obs, rewards, terminated, truncated, info = env.step(player_1_action)  # 后手 step，奖励在此返回

if not swap_positions:
    reward = rewards.get("player_0", 0.0)           # Agent 是先手 → 取 player_0 奖励
else:
    reward = rewards.get("player_1", 0.0)           # Agent 是后手 → 取 player_1 奖励
```

> **说明**：上述代码中 `env.step()` 接受单个整数动作并返回 5 元组。SDK 原生的 `AntWarSequentialEnv.step(action)` 在单步模式下返回 `None`（见 [env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L238-L242)），只有字典形式的联合步进才返回完整元组。实际训练时，`env_factory` 返回的 env 实例应封装适配层，使其 `step(int)` 返回标准 Gymnasium 风格的 5 元组，且内部调用 SDK 的 `_step_single()` 并收集结果。

关键设计要点：

1. **先手 step 的奖励恒为零**：先手 step（`player_0_action`）执行后回合尚未完成，SDK 内部判断 `completed_round=False`，不会触发 `_round_rewards()`。因此第一行用 `_` 丢弃。

2. **后手 step 返回奖励**：后手 step（`player_1_action`）执行时，SDK 内部判断 `player == 1`，执行 `advance_round()` 推进回合，`completed_round=True`，触发 `_round_rewards()` 计算非零奖励。

3. **swap_positions 与奖励选择**：
   - `swap_positions=False`：Agent 控制 player_0（先手），取 `rewards["player_0"]`
   - `swap_positions=True`：Agent 控制 player_1（后手），取 `rewards["player_1"]`

   无论先手后手，Agent 始终取自己控制的那一方的奖励。

---

## 7.3 终端奖励

当游戏因一方大本营 HP 归零而结束时，发放终端奖励。见 [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L162-L166)：

```python
if self.state.terminal:
    if self.state.winner == index:
        reward += 100.0
    elif self.state.winner == enemy:
        reward -= 100.0
```

| 结果 | winner 值 | player_0 奖励 | player_1 奖励 |
|------|-----------|--------------|--------------|
| player_0 胜利 | `0` | **+100.0**（叠加回合奖励后） | -100.0（叠加回合奖励后） |
| player_1 胜利 | `1` | -100.0（叠加回合奖励后） | **+100.0**（叠加回合奖励后） |

**重要**：终端奖励是在回合奖励（HP 伤害差 + 金币差）的基础上叠加的，并非单独发放。即终端回合的最终奖励 = 回合奖励 + `±100.0`。

`winner` 的值由 SDK 后端判定，基于大本营 HP 比较：一方 HP ≤ 0 即判负。

---

## 7.4 增量奖励（回合奖励）

非终端状态下，每个完整回合发放一次增量奖励。奖励由以下两个分量加权求和构成。见 [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L152-L168) 的 `_round_rewards()`：

```python
def _round_rewards(self) -> dict[str, float]:
    rewards: dict[str, float] = {}
    previous_hp = self._round_start_hp
    previous_coins = self._round_start_coins

    for index, agent in enumerate(self.possible_agents):
        enemy = 1 - index

        # 对敌人造成的 HP 伤害
        reward = (previous_hp[enemy] - self.state.bases[enemy].hp) * 10.0

        # 自身受到的 HP 伤害
        reward -= (previous_hp[index] - self.state.bases[index].hp) * 10.0

        # 自身金币变化
        reward += (self.state.coins[index] - previous_coins[index]) * 0.05

        # 敌方金币变化（负向）
        reward -= (self.state.coins[enemy] - previous_coins[enemy]) * 0.02

        # 终端奖励（仅在游戏结束时叠加）
        if self.state.terminal:
            if self.state.winner == index:
                reward += 100.0
            elif self.state.winner == enemy:
                reward -= 100.0

        rewards[agent] = float(reward)

    return rewards
```

### 各分量详解

| 分量 | 权重 | 说明 |
|------|------|------|
| **对敌 HP 伤害** | `×10.0` | 本回合自身对敌方大本营造成的 HP 伤害（`hp_before_enemy - hp_after_enemy`）。正值产生奖励。 |
| **自身 HP 受损** | `×(-10.0)` | 本回合自身大本营受到的 HP 伤害（`hp_before_self - hp_after_self`）。HP 减少时 `(previous - current) > 0`，乘以 `-10.0` 产生惩罚。 |
| **自身金币变化** | `×0.05` | 本回合自身金币变化量（`coins_after - coins_before`）。鼓励玩家积累金币。 |
| **敌方金币变化** | `×(-0.02)` | 本回合敌方金币变化量。敌方金币增加时对自身产生小量惩罚，鼓励压制对方经济。 |

### 设计意图

当前 `_round_rewards()` 的设计可以分解为两个核心维度：

- **HP 维度**（权重 10.0）：奖励对敌人造成的伤害，惩罚自身受到的伤害。净效果 = `(对敌伤害 - 自身受伤) × 10.0`。这是回合奖励中权重最高的分量，驱动物体优先攻击敌方大本营并保护自身大本营。

- **金币维度**（权重 0.05 / -0.02）：轻微奖励自身金币积累，轻微惩罚敌方金币积累。金币权重远小于 HP 权重，在训练中起辅助作用，鼓励经济发展但不喧宾夺主。

整体设计偏向 **对抗性**：HP 分量直接以对敌我双方的伤害差作为奖励信号，比"仅惩罚自身受伤"或"仅奖励对敌伤害"更具对抗性——既要进攻，也要防守。

---

## 7.5 非法动作惩罚

当玩家选择了非法动作（不在 `_bundles` 中的动作或 `action` 为 `None`）时，施加固定惩罚。见 [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L141-L150) 和 [env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L204-L205)：

```python
def _selected_bundle(self, player: int, action: int | None) -> tuple[ActionBundle, bool]:
    bundles = self._bundles[player]
    if not bundles:
        return ActionBundle(name="hold", score=0.0, tags=("noop",)), True
    if action is None:
        return bundles[0], True
    selected = int(action)
    if 0 <= selected < len(bundles):
        return bundles[selected], False
    return bundles[0], True

# 在 _step_single() 中：
if illegal:
    rewards[agent] -= 1.0
```

| 属性 | 值 |
|------|-----|
| 惩罚值 | **-1.0** |
| 触发条件 | 动作不在合法 bundle 列表中、action 为 None、或 bundles 为空 |
| 适用范围 | 仅惩罚执行非法动作的玩家 |

**设计意图**：惩罚非法动作选择，鼓励智能体在合法动作空间内进行决策。非法动作通常会导致选择默认动作（bundles[0]），施加惩罚可以避免智能体倾向于越界选择。

**惩罚时机**：非法动作惩罚在 `_round_rewards()` 计算完成后施加，叠加在回合奖励（或回合奖励 + 终端奖励）之上。

> **注意**：SDK 的非法动作惩罚（-1.0）与本文档旧版中描述的"NO_OP 惩罚（-0.3）"不同。SDK 不区分 NO_OP 与其他合法动作——只要动作在 `_bundles` 列表中即为合法。NO_OP（动作 ID = 0，定义于 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L74)）在 action_mask 中始终合法（`mask[0] = 1.0`），因此不会被判定为非法。

---

## 7.6 奖励值范围

SDK 的 `_round_rewards()` **不包含奖励裁剪逻辑**。回合奖励（不含终端部分）的值域分析：

| 极端情况 | HP 伤害差 ×10 | 金币差 ×0.05/-0.02 | 回合奖励估算 |
|----------|--------------|--------------------|------------|
| 单回合对敌方造成大量 HP 伤害 | ~+50~100 | — | ~+50~100 |
| 单回合被敌方造成大量 HP 伤害 | ~-50~-100 | — | ~-50~-100 |
| 平和发展（无 HP 变化） | 0 | ~±5 | ~±5 |

终端奖励：±100.0（若游戏结束，在此基础上叠加回合奖励）。

**重要说明**：

1. **无裁剪**：`_round_rewards()` 不裁剪奖励值。奖励信号直接传递给训练框架。

2. **PPO 层的数值安全**：训练框架在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L114-L131) 中对 values、returns 和 advantages 进行裁剪（不在 env 层对原始 rewards 裁剪），裁剪参数由配置文件 `clip_values` 段读取：

   | 裁剪参数 | 范围 |
   |----------|------|
   | `VALUES_MIN` / `VALUES_MAX` | `[-10, 10]` |
   | `RETURNS_MIN` / `RETURNS_MAX` | `[-1e6, 1e6]` |
   | `ADVANTAGES_MIN` / `ADVANTAGES_MAX` | `[-1e4, 1e4]` |

---

## 7.7 奖励发放时机

奖励的发放时机由 SDK 的 `_step_single()` 内部控制。见 [Ant-Game/SDK/training/env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L170-L220)：

```python
# _step_single() 中的关键逻辑：
rewards = {name: 0.0 for name in self.possible_agents}   # 默认零奖励

self._context = self._context.next_turn()

completed_round = False
if not self.state.terminal and player == 1:               # 后手玩家执行后推进回合
    self.state.advance_round()
    completed_round = True

if completed_round or self.state.terminal:                 # 回合完成或游戏结束时计算奖励
    rewards = self._round_rewards()
    if completed_round and not self.state.terminal:
        self._capture_round_start()                        # 记录下回合起始状态

if illegal:
    rewards[agent] -= 1.0                                  # 非法动作额外惩罚
```

### 时序图

```
每个完整回合包含两次 env.step()：

  ┌─────────────────────────────────────────────────────────┐
  │ env.step(player_0_action)                               │
  │   → player=0, completed_round=False                     │
  │   → rewards = {"player_0": 0.0, "player_1": 0.0}       │  奖励恒为零
  ├─────────────────────────────────────────────────────────┤
  │ env.step(player_1_action)                               │
  │   → player=1, advance_round(), completed=True           │
  │   → rewards = _round_rewards()                           │  奖励在此返回
  │   → HP 伤害差 ×10 + 金币差 ×0.05/-0.02                  │
  │   → + 非法动作惩罚 (-1.0，如有)                          │
  └─────────────────────────────────────────────────────────┘

游戏结束时（任一步骤后可能触发）：

  ┌─────────────────────────────────────────────────────────┐
  │ env.step(action) 且 state.terminal = True               │
  │   → rewards = _round_rewards()                           │  回合奖励 + 终端奖励 ±100.0
  └─────────────────────────────────────────────────────────┘
```

### 在 SelfPlay 中的实际表现

结合第 7.2 节，每个完整回合中：

1. **先手 step 后**：`env.step(player_0_action)` 返回全零奖励，被 `_` 丢弃
2. **后手 step 后**：`env.step(player_1_action)` 返回非零奖励，根据 `swap_positions` 取对应 player 的值

因此，在 `EpisodeBatch` 中（[selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L588)）：
- 奇数索引的 step（先手 step 对应的记录）reward = 0.0
- 偶数索引的 step（后手 step 对应的记录）reward = 非零（回合奖励或回合+终端奖励）

即 **每个完整回合只产生一个非零奖励记录**，对应一条 `batch.rewards` 条目。每个 episode 中非零奖励的个数约等于有效回合数。

---

## 7.8 奖励在 PPO 更新中的使用

在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L825-L847) 的 `_compute_gae()` 中，批次奖励通过 GAE（Generalized Advantage Estimation）计算 advantages 和 returns：

```python
def _compute_gae(
    self,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    running_advantage = 0.0
    gamma = self.ppo_config.gamma           # 折扣因子
    gae_lambda = self.ppo_config.gae_lambda  # GAE λ

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        running_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * running_advantage
        advantages[t] = running_advantage

    returns = advantages + values.detach()
    return advantages, returns
```

PPO 更新中的数值安全裁剪不直接作用于原始 rewards，而是作用于 values、returns 和 advantages。裁剪参数在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L114-L131) 中由配置文件 `clip_values` 段读取。

---

## 7.9 总结：奖励设计全景

```
                        ┌──────────────────────┐
                        │  env.step() 调用      │
                        │  (env_factory 注入)   │
                        └──────────┬───────────┘
                                   │
                         ┌─────────▼──────────┐
                         │ 回合完成或游戏结束?  │
                         │ (_step_single 判定) │
                         └────┬──────────┬────┘
                              │ yes      │ no
                              ▼          ▼
                    ┌─────────────┐  ┌──────────┐
                    │ 游戏已结束?   │  │ rewards  │
                    └──┬───────┬──┘  │ = 0, 0   │
                       │yes    │no   └──────────┘
                       ▼       ▼
                ┌──────────┐ ┌──────────────────────┐
                │ 终端奖励  │ │  回合奖励             │
                │          │ │  对敌HP伤害 × 10.0   │
                │ win+100.0│ │  自身HP受损 × (-10.0)│
                │ lose-100 │ │  自身金币 × 0.05     │
                └──────────┘ │  敌方金币 × (-0.02)  │
                             │         │            │
                             │    ┌────▼────────┐   │
                             │    │非法动作惩罚  │   │
                             │    │(-1.0)       │   │
                             │    └────┬────────┘   │
                             │         │            │
                             └─────────┼────────────┘
                                       ▼
                               rewards 字典
                         {"player_0": 非零值,
                          "player_1": 非零值}
                                       │
                         ┌─────────────▼─────────────┐
                         │ SelfPlay: 根据 swap 取对应方 │
                         │ PPO: GAE 计算 advantage     │
                         │ → 策略梯度更新               │
                         └───────────────────────────┘
```

**关键设计特点**：

| 特性 | 实现方式 |
|------|---------|
| 奖励来源 | SDK `AntWarSequentialEnv._round_rewards()` |
| 奖励形式 | 对抗性：对敌伤害 − 自身受伤 + 经济差异 |
| HP 权重 | ±10.0（最高优先级） |
| 金币权重 | +0.05（自身）/ -0.02（敌方） |
| 终端奖励 | ±100.0 |
| 非法动作惩罚 | -1.0（仅在动作不在合法 bundle 中时） |
| 奖励裁剪 | SDK 层不裁剪；PPO 层对 values/returns/advantages 裁剪 |
| 注入方式 | `env_factory` 回调 → `PPOTrainer` / `SelfPlayTrainer` |
