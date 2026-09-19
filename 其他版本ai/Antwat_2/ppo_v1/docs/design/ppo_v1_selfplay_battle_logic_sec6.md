# 6. Observation 编码（观察空间）

> 本章基于 `Ant-Game/SDK/utils/features.py`、`Ant-Game/SDK/utils/actions.py`、`ppo/src/ppo_antwar/configs/ppo_antwar.yaml` 分析观察空间的编码方式。PPO 训练中观察的生成由环境（`AntWarEnv`）负责，其底层编码逻辑来自 SDK 的 `FeatureExtractor`。

---

## 6.1 综合观察结构

每个玩家的观察（Observation）是一个 Python 字典，包含三个字段：

| 字段 | 数据类型 | 形状 | 含义 |
|------|----------|------|------|
| `board` | `np.ndarray (float32)` | `(28, 19, 19)` | 六边形网格棋盘的空间特征 |
| `global` | `np.ndarray (float32)` | `(33,)` | 全局标量特征（资源、兵力、科技等） |
| `action_mask` | `np.ndarray (float32)` | `(96,)` | 合法动作掩码，1.0=合法，0.0=非法 |

```python
# 观察结构示例
observation = {
    'board': np.ndarray(shape=(28, 19, 19), dtype=np.float32),
    'global': np.ndarray(shape=(33,), dtype=np.float32),
    'action_mask': np.ndarray(shape=(96,), dtype=np.float32),
}
```

**代码位置：**

- [features.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py) — `FeatureExtractor.encode_board()` 定义了 board 编码逻辑，`FeatureExtractor.summarize()` + `FeatureExtractor.encode_stats()` 定义了 global 特征
- 观察组装由环境在 `env.reset()` / `env.step()` 中调用编码器完成，生成 `player_0` 和 `player_1` 两个对称观察，并为每个观察添加 action_mask

---

## 6.2 Board 特征（28 × 19 × 19）

Board 特征由 `FeatureExtractor.encode_board()` 生成，共 28 个 channel，每个 channel 是一个 19×19 的二维网格，对应六边形地图的 offset 坐标。

### Channel 索引表

| Channel | 名称 | 编码含义 | 取值规则 |
|---------|------|----------|----------|
| 0 | 道路 (PATH) | 当前格是否为道路 | `1.0` 或 `0.0` |
| 1 | Player0 高地 | 是否为 Player0 的高地格 | 当前视角的玩家是 player 0 → `1.0`；player 1 → `0.5` |
| 2 | Player1 高地 | 是否为 Player1 的高地格 | 当前视角的玩家是 player 1 → `1.0`；player 0 → `0.5` |
| 3 | 障碍物 (BARRIER) | 当前格是否为障碍物 | `1.0` 或 `0.0` |
| 4 | 我方塔位置 | 我方塔所在的格子 | `1.0`（是）或 `0.0`（否） |
| 5 | 敌方塔位置 | 敌方塔所在的格子 | `1.0`（是）或 `0.0`（否） |
| 6 | 塔等级 | 塔的升级等级 | `tower.level / 2.0`，最大取 `1.0` |
| 7 | 塔攻击范围 | 塔的攻击半径 | `tower.attack_range / 6.0` |
| 8 | 塔冷却 | 塔的攻击冷却进度 | `tower.display_cooldown() / 6.0` |
| 9 | 塔伤害 | 塔的单发伤害 | `tower.damage / 50.0` |
| 10 | 我方蚂蚁（血量） | 我方蚂蚁在该格的生命值占比（多只取最大值） | `ant.hp / ant.max_hp`，范围 `[0, 1]` |
| 11 | 敌方蚂蚁（血量） | 敌方蚂蚁在该格的生命值占比（多只取最大值） | `ant.hp / ant.max_hp`，范围 `[0, 1]` |
| 12 | 蚂蚁等级 | 蚂蚁的进化等级 | `ant.level / 2.0`，最大取 `1.0` |
| 13 | 蚂蚁年龄 | 蚂蚁的年龄占比 | `ant.age / ANT_AGE_LIMIT`（`ANT_AGE_LIMIT = 64`） |
| 14 | 我方信息素 | 我方蚂蚁留下的信息素浓度 | `pheromone / (12 × PHEROMONE_SCALE)` |
| 15 | 敌方信息素 | 敌方蚂蚁留下的信息素浓度 | `pheromone / (12 × PHEROMONE_SCALE)` |
| 16 | 冰冻标记 | 蚂蚁是否被冻结 | `1.0`（冻结）或 `0.0`（未冻结） |
| 17 | RANDOM 行为 | 蚂蚁是否处于随机行走状态 | `1.0` 或 `0.0` |
| 18 | BEWITCHED 行为 | 蚂蚁是否处于魅惑状态 | `1.0` 或 `0.0` |
| 19 | CONTROL_FREE 行为 | 蚂蚁是否处于失控状态 | `1.0` 或 `0.0` |
| 20 | 我方闪电风暴 | 我方 LIGHTNING_STORM 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |
| 21 | 敌方闪电风暴 | 敌方 LIGHTNING_STORM 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |
| 22 | 我方 EMP 冲击波 | 我方 EMP_BLASTER 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |
| 23 | 敌方 EMP 冲击波 | 敌方 EMP_BLASTER 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |
| 24 | 我方偏转器 | 我方 DEFLECTOR 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |
| 25 | 敌方偏转器 | 敌方 DEFLECTOR 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |
| 26 | 我方紧急闪避 | 我方 EMERGENCY_EVASION 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |
| 27 | 敌方紧急闪避 | 敌方 EMERGENCY_EVASION 超级武器的覆盖范围 | `remaining_turns / duration`，范围 `[0, 1]` |

**编码逻辑要点：**

- **地形 channel（0-3）**：静态特征，不随回合变化。Player0 和 Player1 的高地（channel 1、2）的取值取决于当前观察的视角玩家是谁——若视角玩家是 player 0 则 channel 1 为 `1.0`、channel 2 为 `0.5`；视角玩家为 player 1 则相反。
- **塔 channel（4-9）**：channel 4/5 标记塔的所属方。channel 6-9 的属性贴在其所在格子上，视角玩家看到的始终是我方塔的数据。
- **蚂蚁 channel（10-19）**：channel 10/11 区分我方/敌方蚂蚁，同类多只蚂蚁共存于同一格时取 `hp` 最大值。channel 16-19 按蚂蚁行为类型独立标记。
- **信息素 channel（14-15）**：`PHEROMONE_SCALE = 10000`，除以 `12 × PHEROMONE_SCALE` 将值归一化到 `[0, 1]` 范围。
- **超级武器效果 channel（20-27）**：按武器类型成对排列（我方 / 敌方），值 = `remaining_turns / duration`。覆盖范围使用六边形距离（`hex_distance`）判断。

**参考源码：**

- [features.py#L161-L213](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py#L161-L213) — `FeatureExtractor.encode_board()`

---

## 6.3 Global 特征（33 维）

Global 特征由 `FeatureExtractor.summarize()` 和 `FeatureExtractor.encode_stats()` 生成，共 33 个归一化标量值（PPO 版本去掉了 SDK 版中的 4 维决策上下文 + 4 维 super_weapon_usage 等 extras，并重组了 super weapon cooldown 的顺序）。

### 特征索引表

| 索引 | 特征名称 | 计算方式 | 归一化除子 | 含义 |
|------|----------|----------|------------|------|
| 0 | `round_ratio` | `state.round_index / MAX_ROUND` | `512` | 当前回合进度 |
| 1 | `hp_delta` | `bases[player].hp - bases[enemy].hp` | `/ 50.0` | 我方与敌方基地 HP 差值 |
| 2 | `coin_ratio` | `coins[player] / max(coins[enemy], 1)` | — | 我方与敌方金币比值 |
| 3 | `safe_coin` | `max(coins[player] - threshold, 0)` | `/ 1000.0` | 超出安全阈值的富余金币 |
| 4 | `frontline_advantage` | `enemy_front - my_front` | `/ 19.0` | 前线距离优势（正=我方推进更深） |
| 5 | `enemy_front_distance` | 敌方最近蚂蚁到己方基地的距离 | `/ 19.0` | 敌方前线到己方基地的距离 |
| 6 | `my_front_distance` | 己方最近蚂蚁到敌方基地的距离 | `/ 19.0` | 己方前线到敌方基地的距离 |
| 7 | `enemy_progress` | 敌方蚂蚁的平均推进进度 | `/ ANT_AGE_LIMIT` | 敌方蚂蚁(剩余寿命 - 距离惩罚)的均值 |
| 8 | `my_progress` | 己方蚂蚁的平均推进进度 | `/ ANT_AGE_LIMIT` | 己方蚂蚁(剩余寿命 - 距离惩罚)的均值 |
| 9 | `tower_count` | `len(my_towers)` | `/ 20.0` | 我方防御塔数量 |
| 10 | `enemy_tower_count` | `len(enemy_towers)` | `/ 20.0` | 敌方防御塔数量 |
| 11 | `tower_level_sum` | `sum(tower.level)` | `/ 40.0` | 我方塔等级总和 |
| 12 | `enemy_tower_level_sum` | `sum(enemy_tower.level)` | `/ 40.0` | 敌方塔等级总和 |
| 13 | `kill_delta` | `die_count[enemy] - die_count[player]` | `/ 20.0` | 击杀数差值（正=我方多杀） |
| 14 | `old_delta` | `old_count[enemy] - old_count[player]` | `/ 20.0` | 老死数差值（正=敌方老死更多） |
| 15 | `tower_spread` | `state.tower_spread_score(player)` | `/ 10.0` | 我方塔的分布得分 |
| 16 | `slot_fill_ratio` | `len(my_towers) / len(HIGHLAND_CELLS[player])` | — | 我方高地格占用比例 |
| 17 | `generation_value` | 蚂蚁生成周期的升级进度 | — | 归一化到 `[0, 1]`，越大代表周期越短 |
| 18 | `ant_value` | 蚂蚁生命值的升级进度 | — | 归一化到 `[0, 1]`，越大代表 HP 越高 |
| 19 | `hostile_distance` | 最近的敌方蚂蚁到己方基地的距离 | `/ 19.0` | 敌方威胁距离 |
| 20 | `base_arc_coverage` | 基地前方弧线的塔覆盖比例 | — | 范围 `[0, 1]`，覆盖3个关键方向的平均情况 |
| 21 | `tower_spacing` | 塔间距惩罚得分 | — | 负数，塔越密集惩罚越大 |
| 22 | `my_base_hp` | `bases[player].hp` | `/ 50.0` | 我方基地当前 HP（50 为满血） |
| 23 | `enemy_base_hp` | `bases[enemy].hp` | `/ 50.0` | 敌方基地当前 HP（50 为满血） |
| 24 | `my_coins` | `coins[player]` | `/ 1000.0` | 我方当前金币数 |
| 25 | `weapon_ls_cd_me` | 我方 LIGHTNING_STORM 冷却 | `/ cooldown` | 归一化冷却进度 |
| 26 | `weapon_emp_cd_me` | 我方 EMP_BLASTER 冷却 | `/ cooldown` | 归一化冷却进度 |
| 27 | `weapon_def_cd_me` | 我方 DEFLECTOR 冷却 | `/ cooldown` | 归一化冷却进度 |
| 28 | `weapon_evade_cd_me` | 我方 EMERGENCY_EVASION 冷却 | `/ cooldown` | 归一化冷却进度 |
| 29 | `weapon_ls_cd_enemy` | 敌方 LIGHTNING_STORM 冷却 | `/ cooldown` | 归一化冷却进度 |
| 30 | `weapon_emp_cd_enemy` | 敌方 EMP_BLASTER 冷却 | `/ cooldown` | 归一化冷却进度 |
| 31 | `weapon_def_cd_enemy` | 敌方 DEFLECTOR 冷却 | `/ cooldown` | 归一化冷却进度 |
| 32 | `weapon_evade_cd_enemy` | 敌方 EMERGENCY_EVASION 冷却 | `/ cooldown` | 归一化冷却进度 |

**四大超级武器的冷却参数（参考 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L135-L139)）：**

| 武器类型 | 枚举值 | 持续时间 | 攻击范围 | 冷却回合 | 费用 |
|----------|--------|----------|----------|----------|------|
| LIGHTNING_STORM | 1 | 20 | 3 | 100 | 150 |
| EMP_BLASTER | 2 | 20 | 3 | 100 | 150 |
| DEFLECTOR | 3 | 10 | 3 | 50 | 100 |
| EMERGENCY_EVASION | 4 | 2 | 3 | 50 | 100 |

**编码逻辑要点：**

- 索引 0-3：游戏进度与资源状态（回合数、HP 差、金币比、富余金币）
- 索引 4-8：前线与推进（前线距离差、双方前沿距离、双方推进进度）
- 索引 9-16：建筑经济（塔数量、等级、分布、高地占用率）
- 索引 13-14：战损（击杀差、老死差）
- 索引 17-18：科技升级（蚂蚁生成速度、蚂蚁 HP）
- 索引 19-21：防御态势（威胁距离、基地覆盖、塔间距）
- 索引 22-24：基础状态（双方 HP、己方金币）
- 索引 25-32：超级武器冷却（前 4 个为我方，后 4 个为敌方）

**参考源码：**

- [features.py#L89-L159](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py#L89-L159) — `FeatureExtractor.summarize()`（22 个 named 特征的生成逻辑）
- [features.py#L219-L266](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py#L219-L266) — `FeatureExtractor.encode_stats()`（将 summarize 结果与 extras 拼接，产出最终的 stats 向量）

---

## 6.4 形状与网络输入

| 字段 | 形状 | 网络处理方式 |
|------|------|--------------|
| `board` | `(28, 19, 19)` | 经过 `HexCNNEncoder` 六边形 CNN 编码为固定长度向量 |
| `global` | `(33,)` | 经过 `MLPEncoder` 两层全连接 MLP 编码 |
| `action_mask` | `(96,)` | 不参与编码，直接用于屏蔽非法动作的 logit |

网络配置（[ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L66-L71)）：

```yaml
network:
  hidden_dim: 256
  board_shape:
    - 28
    - 19
    - 19
  global_dim: 33
  action_dim: 96
```

网络前向时，`board` 输入 `HexCNNEncoder`（[antwar_net.py:L83-L183](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L83-L183)），`global` 输入 `MLPEncoder`（[antwar_net.py:L186-L195](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L186-L195)），二者拼接后经过 Projection 层，再分别进入 Policy Head 和 Value Head：

```python
def forward(self, board, global_features, action_mask=None):
    board_encoded = self.cnn_encoder(board)       # (B, 28, 19, 19) → (B, hidden*2*5*5)
    global_encoded = self.mlp_encoder(global_features)  # (B, 33) → (B, hidden//2)
    merged = torch.cat([board_encoded, global_encoded], dim=1)
    merged = self.projection(merged)               # → (B, hidden)

    action_logits = self.policy_head(merged)       # → (B, 96)
    value = self.value_head(merged)                # → (B, 1)

    if action_mask is not None:
        action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

    return action_logits, value
```

参考：[antwar_net.py:L248-L266](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L248-L266)

---

## 6.5 player_0 与 player_1 观察的区别

环境的 `reset()` 和 `step()` 方法会为双方玩家生成各自的观察，返回的字典格式为：

```python
{
    "player_0": {"board": ..., "global": ..., "action_mask": ...},
    "player_1": {"board": ..., "global": ..., "action_mask": ...},
}
```

观察的编码逻辑来自 SDK 的 `FeatureExtractor`（[features.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py)），核心思路是：调用 `encode_board(state, player)` 和相应的 global 特征编码时，传入当前玩家的 ID 作为 `player` 参数，以该 ID 为"我方"视角进行编码。

**核心差异：`encode_board(state, player)` 和 global 特征编码中的 `player` 参数决定了视角。**

| 差异维度 | player_0 观察 | player_1 观察 |
|----------|---------------|---------------|
| 视角 | 以 player 0 为"我方" | 以 player 1 为"我方" |
| Board channel 1（Player0 高地） | `1.0`（自己是 player 0） | `0.5`（自己是外人） |
| Board channel 2（Player1 高地） | `0.5`（自己是外人） | `1.0`（自己是 player 1） |
| Board channel 4（我方塔位置） | player 0 的塔 | player 1 的塔 |
| Board channel 5（敌方塔位置） | player 1 的塔 | player 0 的塔 |
| Board channel 10（我方蚂蚁） | player 0 的蚂蚁 | player 1 的蚂蚁 |
| Board channel 11（敌方蚂蚁） | player 1 的蚂蚁 | player 0 的蚂蚁 |
| Board channel 14（我方信息素） | player 0 的信息素 | player 1 的信息素 |
| Board channel 15（敌方信息素） | player 1 的信息素 | player 0 的信息素 |
| Board channel 20,22,24,26（我方超级武器） | player 0 的效果 | player 1 的效果 |
| Board channel 21,23,25,27（敌方超级武器） | player 1 的效果 | player 0 的效果 |
| Global 特征（全部 33 维） | 以 player 0 为"我方"进行计算 | 以 player 1 为"我方"进行计算 |
| action_mask | player 0 的合法动作掩码 | player 1 的合法动作掩码 |

这种设计使得两个玩家看到的观察在语义上对称——每个玩家都总是把自己视为"我方"，把对手视为"敌方"。Board 中的高地 channel（1、2）通过 `0.5` 指示"非己方高地"，使网络能区分"自己的高地"和"对手的高地"。

**训练中的实际使用：**

在 `PPOTrainer.collect_episode()` 中（[ppo_trainer.py:L585-L586](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L585-L586)），训练 Agent 始终从 `player_0` 视角获取观察和奖励：

```python
player_obs = obs[f"player_0"]      # 训练 Agent 的观察
opponent_obs = obs["player_1"]     # 对手的观察（用于对手 AI 决策）
```

观察中的三个字段被分别提取并存入 `EpisodeBatch`（[ppo_trainer.py:L604-L606](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L604-L606)）：

```python
batch.observations_board.append(player_obs['board'])
batch.observations_global.append(player_obs['global'])
batch.observations_mask.append(player_obs['action_mask'])
```

在 `_select_action()` 中（[ppo_trainer.py:L628-L631](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L628-L631)），观察被转换为 tensor 输入网络：

```python
board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)
```

而在 SelfPlay 的先后手交替模式中（[selfplay.py:L538-L541](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L538-L541)），通过 `swap_positions` 参数切换训练 Agent 的视角：

```python
if not swap_positions:
    player_obs = obs["player_0"]   # Agent 先手
else:
    player_obs = obs["player_1"]   # Agent 后手（视为 player_1 的观察）
```

环境采用双 step 模式模拟一局游戏（[selfplay.py:L581-L582](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L581-L582)）：

```python
obs, _, _, _, _ = env.step(player_0_action)
obs, rewards, terminated, truncated, info = env.step(player_1_action)
```

无论哪种模式，`player_0` 和 `player_1` 两个观察都是完整的、从各自视角编码的对称结构。编码逻辑底层均来自 SDK 的 `FeatureExtractor`。

---

## 6.6 与 SDK 版编码的差异

`Ant-Game/SDK/utils/features.py` 中的 `FeatureExtractor` 提供了观察编码（用于 baseline agent 或旧版训练流程），PPO 训练中使用的观察编码是对其的简化适配：

| 对比项 | PPO 训练中使用的编码 | SDK FeatureExtractor |
|--------|------------------------|----------------------|
| 文件 | 环境层封装 + SDK [features.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py) | [features.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py) |
| global 字段名 | `'global'` | `'stats'` |
| global 维度 | 33 | 42（22 个 named + 20 个 extras） |
| global 编码方式 | 环境层直接硬编码列表（简化版 summarize + encode_stats） | `summarize()` → `encode_stats()` 两步 |
| 决策上下文 | 不包含 `to_play` / `settles_after_action` | 包含（extras 中的最后 4 维） |
| board 编码逻辑 | 几乎相同，语义一致 | 几乎相同，语义一致 |

PPO 版本去掉了决策上下文（PPO 训练中通过双 step 模式自然获得先后手信息），并将维度压缩到 33 维以保持紧凑。

---

## 6.7 相关配置参数速查

| 参数 | 配置段 | 值 | 含义 |
|------|--------|-----|------|
| `board_shape` | `network:` | `[28, 19, 19]` | Board 特征的形状（channels × height × width） |
| `global_dim` | `network:` | `33` | Global 特征的维度 |
| `action_dim` | `network:` | `96` | 动作空间维度（= action_mask 长度） |
| `MAX_ROUND` | [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L120) | `512` | 最大回合数（用于 round_ratio 归一化） |
| `ANT_AGE_LIMIT` | [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L121) | `64` | 蚂蚁最大年龄（用于年龄归一化） |
| `PHEROMONE_SCALE` | [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L124) | `10000` | 信息素缩放因子 |
