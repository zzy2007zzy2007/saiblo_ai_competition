# 模块1：游戏环境与动作空间

## 1. 概述

### 1.1 模块定位

游戏环境与动作空间模块是 PPO 训练系统与 AntWar 游戏后端的 **唯一交互层**。它负责将原始游戏状态转化为神经网络可处理的张量格式，将神经网络的输出解码为游戏操作，并计算每一步的奖励信号。

### 1.2 在整体架构中的位置

```
训练数据闭环的起点和终点：

  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │ AntWarEnv │────►│ Observation  │────►│ Network  │  (模块2)
  │ (游戏状态) │     │   Encoder    │     │ forward  │
  └────▲─────┘     └──────────────┘     └────┬─────┘
       │                                      │
       │           ┌──────────────┐           │
       │           │ ActionMask   │◄──────────┘
       │           │  .action_id  │
       │           │  _to_op()    │
       │           └──────┬───────┘
       │                  │
  ┌────┴─────┐            │
  │ env.step │◄───────────┘
  │ (执行动作)│
  └──────────┘
```

该模块被以下模块依赖：
- **模块3（PPO 训练引擎）**：通过 `EpisodeCollector` 驱动环境进行数据收集
- **模块5（对战评估系统）**：通过 `BattleSimulator` 驱动环境进行评估对战

### 1.3 业务目标

- 封装 AntWar 游戏的完整交互接口
- 定义标准化的观测空间、动作空间和奖励函数
- 提供 Gymnasium 风格的 `reset()` / `step()` 接口，无缝对接强化学习训练流程
- 通过动作掩码确保策略网络只输出合法动作

---

## 2. 背景与概念

### 2.1 AntWar 游戏简介

AntWar 是一款 **双人回合制塔防对战游戏**，发生在一个 19×19 的六边形棋盘上。

**地图地形**（定义在 [MAP_PROPERTY](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/action_constants.py)）：

| 地形类型 | 枚举值 | 含义 |
|---------|--------|------|
| `VOID` | -1 | 不可达区域（地图边缘） |
| `PATH` | 0 | 蚂蚁行进路径 |
| `BARRIER` | 1 | 障碍物（不可建造） |
| `PLAYER0_HIGHLAND` | 2 | 玩家 0 高地（可建塔） |
| `PLAYER1_HIGHLAND` | 3 | 玩家 1 高地（可建塔） |

**核心概念**：

- **基地血量（HP）**：每方拥有一个基地，HP 归零则该方失败
- **防御塔**：建造在高地上的防御建筑，可升级（提高攻击力）或降级
- **蚂蚁单位**：自动生成并沿路径向敌方基地移动的单位，对敌方基地造成伤害
- **科技升级**：提升蚂蚁生成速度或蚂蚁属性
- **超级武器**：特殊技能，包括闪电风暴（Lightning Storm）、EMP 冲击波、偏转器（Deflector）、紧急规避（Emergency Evasion）
- **金币**：用于建造、升级和科技研发的资源

### 2.2 Gymnasium 接口标准

本模块遵循 [Gymnasium](https://gymnasium.farama.org/) 强化学习环境接口标准：

- `reset()` → `(observation, info)`：重置环境到初始状态
- `step(action)` → `(observation, reward, terminated, truncated, info)`：执行一个动作并返回结果

### 2.3 对战模式

`AntWarEnv` 支持两种对战模式：

- **顺序模式（Sequential）**：每回合调用 `step(player_0_action)` 和 `step(player_1_action)` 两次，适合交替决策
- **批量模式（Batch）**：一次 `step({0: action_0, 1: action_1})` 同时传入双方动作，适合同时决策（如训练时的数据收集）

---

## 3. 架构设计

### 3.1 模块内部结构

```
┌─────────────────────────────────────────────────┐
│                 游戏环境与动作空间模块              │
│                                                   │
│  ┌─────────────────┐                             │
│  │   AntWarEnv      │  Gymnasium 风格环境封装      │
│  │                  │  reset / step / 奖励计算     │
│  └────────┬────────┘                             │
│           │ 依赖                                  │
│     ┌─────┴─────┐                                │
│     │           │                                 │
│  ┌──▼────────┐ ┌▼──────────────┐                 │
│  │Observation│ │ActionMask     │                 │
│  │ Encoder   │ │Handler        │                 │
│  │ (观测编码) │ │(动作掩码/解码) │                 │
│  └───────────┘ └───────────────┘                 │
│           │           │                           │
│           └─────┬─────┘                           │
│                 │                                  │
│     ┌───────────▼───────────┐                    │
│     │   action_constants.py │                    │
│     │   游戏级常量定义        │                    │
│     │   (类型/奖励/地图/归一化)│                    │
│     └───────────────────────┘                    │
└─────────────────────────────────────────────────┘
```

### 3.2 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| `AntWarEnv` | [env/antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/antwar_env.py) | 环境主入口，管理对局生命周期和奖励计算 |
| `ObservationEncoder` | [env/observation.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/observation.py) | 将游戏状态编码为神经网络可处理的三段式张量 |
| `ActionMaskHandler` | [env/action_mask.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/action_mask.py) | 动作合法性判断、掩码生成、action_id 到 Operation 转换 |
| `action_constants` | [utils/action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/action_constants.py) | 游戏级全局常量（地图、动作类型、奖励权重、归一化参数） |

---

## 4. 核心组件详解

### 4.1 AntWarEnv — 游戏环境封装

位置：[env/antwar_env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/antwar_env.py)

**初始化**：

```
AntWarEnv(player_id: int, backend_type: str, prefer_native: bool = False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `player_id` | `int` | 当前训练智能体在游戏中的玩家编号（0 或 1） |
| `backend_type` | `str` | 后端类型，如 `"python"` |
| `prefer_native` | `bool` | 是否优先使用原生后端（更快但可能不可用） |

初始化时创建 `ActionMaskHandler` 和 `ObservationEncoder`，并通过 `BackendAdapter` 创建游戏运行时。

**关键方法**：

| 方法 | 签名 | 返回 | 说明 |
|------|------|------|------|
| `reset()` | `() -> (obs_dict, info_dict)` | `observations: {0: obs, 1: obs}`, `info: {round_index: 0}` | 重置对局到初始状态，返回双方观测 |
| `step(action)` | `(action) -> (obs, rewards, terminated, truncated, info)` | 见下文 | 执行动作，推进对局 |

**`step()` 的两种调用模式**：

1. **顺序模式**（`action` 为 `int`）：
   - 第一次调用传入 player_0 的动作，内部缓存
   - 第二次调用传入 player_1 的动作，双方动作一起解析，推进回合
   - 返回当前玩家的观测、奖励和终止状态

2. **批量模式**（`action` 为 `Dict[int, int]`）：
   - 一次传入 `{0: action_0, 1: action_1}` 双方的完整动作
   - 立即解析双方动作、推进回合
   - 返回双方观测 `{0: obs_0, 1: obs_1}`、双方奖励 `{0: r0, 1: r1}`

**`step()` 返回值详解**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `observations` | `Dict[int, Dict]` | 双方观测，每个观测包含 `board`/`global`/`action_mask` |
| `rewards` | `Dict[int, float]` | 双方奖励值（含回合级奖励 + 动作级奖励） |
| `terminated` | `bool` | 是否因一方 HP ≤ 0 正常终局 |
| `truncated` | `bool` | 是否因超过最大回合数截断 |
| `info` | `Dict` | 额外信息（如当前回合数） |

**状态快照**：`get_state_snapshot(player: int) -> Dict[str, Any]` 获取指定玩家的当前状态（HP、金币、塔信息、科技等级等），用于训练数据中辅助标签的计算。

### 4.2 ObservationEncoder — 观测编码器

位置：[env/observation.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/observation.py)

将游戏原始状态转换为神经网络可处理的三段式张量。

**初始化**：

```
ObservationEncoder(action_mask_handler: Optional[ActionMaskHandler] = None)
```

初始化时校验 `OBS_NORMALIZATION` 是否包含所有必需的缩放因子（9 个），并初始化 SDK 的 `FeatureExtractor`（用于棋盘特征提取）。

**核心方法**：

```
encode(state, player: int) -> Dict[str, np.ndarray]
```

**返回的三段式观测**：

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `board` | `(28, 19, 19)` | `float32` | 28 通道棋盘特征图，由 SDK 的 `FeatureExtractor.encode_board()` 生成 |
| `global_vec` | `(33,)` | `float32` | 33 维全局特征向量，手工提取，经过归一化 |
| `action_mask` | `(96,)` | `float32` | 96 维动作合法性掩码（1.0=合法, 0.0=非法） |

**33 维全局向量的组成**（全部经过 `OBS_NORMALIZATION` 归一化）：

| 维度区间 | 含义 | 归一化缩放因子 |
|---------|------|--------------|
| 0 | 回合进度 | `progress_scale = 64.0` |
| 1 | 己方/敌方 HP 差 | `hp_scale = 50.0` |
| 2 | 金币比 | `coin_scale = 1000.0` |
| 3-4 | 前线优势距离 | `distance_scale = 19.0` |
| 5-6 | 塔数量/等级 | `tower_count_scale = 20.0` / `tower_level_scale = 40.0` |
| 7 | 击杀差 | `kill_scale = 20.0` |
| 8 | 塔扩散度 | `tower_spread_scale = 10.0` |
| 9 | 棋盘填充率 | — |
| 10-11 | 科技等级 | — |
| 12-13 | 超级武器冷却时间 | — |
| 14-17 | 己方/敌方 HP（缩放） | `hp_scale = 50.0` |
| 18-33 | 更多精细化特征 | 包含塔间距等 |

### 4.3 ActionMaskHandler — 动作掩码处理器

位置：[env/action_mask.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/action_mask.py)

负责两个核心功能：动作合法性掩码生成和 action_id ↔ Operation 双向转换。

**关键方法**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_action_mask(state, player)` | `-> np.ndarray shape=(96,)` | 生成动作合法性掩码 |
| `action_id_to_op(action_id, state, player)` | `-> Optional[Operation]` | action_id 转换为 SDK Operation |
| `flat_mask_to_type_mask(flat_mask)` | `-> np.ndarray shape=(9,)` | 扁平掩码 → 类型级掩码 |
| `flat_mask_to_target_mask(flat_mask, type_id)` | `-> np.ndarray` | 提取指定类型的子掩码 |

**特殊规则**：

- **前 1000 回合禁止降级塔**：防止策略过早学会破坏自己防御的负面行为，给学习过程提供更好的初始引导

**动作空间查询**：

- `TYPE_CONFIG` 定义了 9 种动作类型的扁平区间（见 4.4 节）
- `flat_mask_to_type_mask()`：检查每种类型下是否有至少一个合法动作
- `flat_mask_to_target_mask()`：提取某类型下的目标维度合法掩码

### 4.4 action_constants — 游戏常量定义

位置：[utils/action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/action_constants.py)

这是整个模块的 **基石文件**，定义了所有游戏级常量。

#### 4.4.1 地图常量

| 常量 | 值 | 说明 |
|------|---|------|
| `MAP_SIZE` | 19 | 棋盘尺寸（19×19） |
| `EDGE` | 10 | 前线边界位置 |
| `PLAYER_COUNT` | 2 | 玩家数量 |
| `PLAYER_BASES` | `((2,9), (16,9))` | 双方基地坐标 |
| `MAP_PROPERTY` | 19×19 元组 | 地图地形矩阵 |
| `TOWER_POSITIONS` | 16 个坐标 | 可建塔位置列表 |
| `MAX_ROUND` | 512 | 最大回合数 |

#### 4.4.2 动作空间定义（TYPE_CONFIG）

动作空间设计为 **96 维扁平空间**，由 9 种动作类型组成，每种类型占据一段连续的 flat_id 区间：

| 类型索引 | 类型名 | 含义 | flat 区间 | 大小 | 目标数 |
|---------|--------|------|-----------|------|--------|
| 0 | `noop` | 空操作 | [0, 1) | 1 | 1 |
| 1 | `build_tower` | 建造防御塔 | [1, 11) | 10 | 16 个位置 |
| 2 | `upgrade_tower` | 升级防御塔 | [11, 51) | 40 | 10 个位置 × 4 方向 |
| 3 | `downgrade_tower` | 降级防御塔 | [51, 61) | 10 | 10 个位置 |
| 4 | `lightning_storm` | 闪电风暴 | [61, 66) | 5 | 5 个释放位置 |
| 5 | `emp_blaster` | EMP 冲击波 | [66, 71) | 5 | 5 个释放位置 |
| 6 | `deflector` | 偏转器 | [71, 76) | 5 | 5 个释放位置 |
| 7 | `evasion` | 紧急规避 | [76, 81) | 5 | 5 个释放位置 |
| 8 | `tech_upgrade` | 科技升级 | [81, 83) | 2 | 2 种科技 |
| **合计** | | | **[0, 96)** | **96** | |

**重要**：每个类型的具体 flat_id 范围定义在 `TYPE_CONFIG` 的 `flat_start` 和 `flat_end` 字段中。上述区间是文档层面的推导，实际以代码中的 `TYPE_CONFIG` 为准。

#### 4.4.3 观测归一化参数（OBS_NORMALIZATION）

| 参数 | 值 | 应用维度 |
|------|---|---------|
| `hp_scale` | 50.0 | HP 相关特征 |
| `coin_scale` | 1000.0 | 金币相关特征 |
| `distance_scale` | 19.0 | 距离相关特征 |
| `progress_scale` | 64.0 | 回合进度 |
| `tower_count_scale` | 20.0 | 塔数量 |
| `tower_level_scale` | 40.0 | 塔等级 |
| `tower_spread_scale` | 10.0 | 塔扩散度 |
| `kill_scale` | 20.0 | 击杀数 |
| `tower_spacing_scale` | 1.0 | 塔间距 |

#### 4.4.4 超级武器位置（SUPER_WEAPON_POSITIONS）

不同超级武器有不同的可释放位置：

- **闪电风暴 / EMP 冲击波**：`[(7,9), (8,9), (9,9), (10,9), (11,9)]` — 中线区域
- **偏转器 / 紧急规避**：`[(4,9), (5,9), (6,9), (12,9), (13,9)]` — 两侧区域

---

## 5. 数据流与时序

### 5.1 观测编码流程

```
raw_state (游戏对象)
    │
    ├──► FeatureExtractor.encode_board(state)
    │       └──► board (28, 19, 19) float32
    │
    ├──► 手工特征提取 (33 维全局信息)
    │       ├── 回合进度、HP差、金币比
    │       ├── 前线距离、塔数量/等级/扩散
    │       ├── 科技等级、超级武器冷却
    │       └──► 归一化 (除以 OBS_NORMALIZATION 各因子)
    │       └──► global_vec (33,) float32
    │
    └──► ActionMaskHandler.get_action_mask(state, player)
            └──► 逐动作合法性检查
            └──► 特殊规则 (前1000回合禁止降级)
            └──► action_mask (96,) float32
```

### 5.2 动作执行流程

```
action_id (int, 0-95)
    │
    ├──► action_id_to_type_and_target()
    │       └──► type_id (0-8), target_k (0-max_targets)
    │
    ├──► ActionMaskHandler.action_id_to_op()
    │       ├── noop: 返回 None (空操作)
    │       ├── build_tower: 根据 target_k 选择建塔位置
    │       ├── upgrade_tower: 从目标塔位置 + 方向解析
    │       ├── downgrade_tower: 降级指定塔
    │       ├── super_weapon: 根据类型 + target_k 选择释放位置
    │       └── tech_upgrade: 选择科技类型
    │       └──► Operation 对象
    │
    └──► BackendAdapter (兼容层)
            └──► SDK.Operation 对象
            └──► 传递给 gamesdk resolve_turn()
```

### 5.3 奖励计算流程

奖励计算在 `AntWarEnv._compute_battle_rewards()` 中完成，每回合调用。

**回合级奖励（Step Reward）**：

```
step_reward = 
    base_hp_damage * base_hp_attack_weight          (对敌方HP的伤害)
  + tower_damage * tower_hp_attack_weight           (对敌方塔的伤害)
  + coin_gain * own_coin_gain_weight                 (己方金币净收入)
  + enemy_coin_gain * enemy_coin_gain_weight         (敌方金币净收入)
  + tower_survival_bonus                             (己方塔存活奖励)
  - enemy_tower_survival_penalty                     (敌方塔存在惩罚)
  + balance_bonus                                    (余额奖励)
  + own_die_penalty                                  (己方蚂蚁死亡惩罚)
  - tech_penalties                                   (科技投入惩罚)

final_step_reward = clip(step_reward, -step_reward_clip, +step_reward_clip)
```

**动作级奖励（Action Reward）**：

在 `compute_action_reward()` 中计算，与回合级奖励叠加：

| 动作类型 | 奖励逻辑 |
|---------|---------|
| `build_tower` | 按建塔优先级分级奖励 (0.6 → 0.1) |
| `upgrade_tower` (L2) | +3.0 |
| `upgrade_tower` (L3) | +5.0 |
| `downgrade_tower` | -9.0 (惩罚) |
| `upgrade_gen_speed` | +15.0 (L1) / +10.0 (L2) |
| `upgrade_gen_ant` | +7.5 (L1) / +2.5 (L2) |
| `deploy_lightning_storm` | +7.5 |
| `deploy_emp_blaster` | +9.0 |
| `deploy_deflector` | +1.0 |
| `deploy_evasion` | +0.8 |
| `noop` | 负惩罚 (渐进式, base: -0.0025, max: -0.005) |

**终局奖励（Terminal Reward）**：

- 胜利：+500.0
- 失败：-500.0

---

## 6. 对外交互

### 6.1 被依赖关系

| 调用方 | 场景 | 使用接口 |
|--------|------|---------|
| **EpisodeCollector** (模块3) | 训练数据收集 | `reset()` → 循环 `step()` → 收集观测/奖励 |
| **BattleSimulator** (模块5) | 评估对战 | 通过 `_execute_battle()` 使用 SDK 接口驱动对战 |

### 6.2 对模块8（基础设施）的依赖

- 通过 `BackendAdapter`（[compat/adapters.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/compat/adapters.py)）创建游戏运行时
- 使用 SDK 的 `FeatureExtractor` 进行棋盘编码
- 使用 SDK 的 `Operation` 类型定义动作

### 6.3 与模块2（神经网络）的交互

- `AntWarEnv` 不直接依赖模块2
- 其输出的 `{board, global_vec, action_mask}` 格式与 `AntWarPolicyValueNetwork.forward()` 的输入格式严格匹配
- 其消耗的 `action_id` 是网络 `get_action()` 的输出

---

## 7. 配置参数

### 7.1 YAML 配置中的环境参数

在配置文件的 `env` 字段中：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `player_id` | `int` | 0 | 智能体扮演的玩家编号 |
| `backend_type` | `str` | `"python"` | 游戏后端类型 |

### 7.2 核心超参数（REWARD_CONFIG）

所有奖励权重集中在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/action_constants.py) 的 `REWARD_CONFIG` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_hp_attack_weight` | 2.0 | 基础HP伤害权重 |
| `tower_hp_attack_weight` | 0.2 | 塔伤害权重 |
| `own_coin_gain_weight` | 0.05 | 己方金币收入权重 |
| `tower_survival_per_tower` | 0.02 | 每个塔存活奖励 |
| `win_reward` | 500.0 | 胜利奖励 |
| `loss_reward` | -500.0 | 失败惩罚 |
| `step_reward_clip` | 500.0 | 单步奖励裁剪上限 |

---

## 8. 常见问题与注意事项

### 8.1 动作空间为 96 维而非 9 维的原因

虽然只有 9 种动作类型，但由于每种类型包含不同的目标（建塔位置、升级方向、武器位置），直接使用 9 维分类无法表达完整决策。96 维扁平空间是 "类型 + 目标" 的组合枚举。

策略网络实际采用分层采样：先选类型（9维），再选目标（该类型下的合法子空间大小），通过 `StructuredActionHead` 实现。

### 8.2 前 1000 回合禁止降级塔的特殊规则

这是一个 **课程学习（Curriculum Learning）** 策略。在训练初期，模型可能随机探索到降级塔的操作，导致自身防御崩溃。通过在前 1000 回合屏蔽降级选项，引导模型先学会积极的建造和升级行为，之后再逐步开放降级操作（可能在某些场景下有效，如战略调整塔位）。

### 8.3 奖励函数是非零和的

回合级奖励中，己方的塔存活奖励和敌方的塔存活惩罚是独立计算的，因此双方奖励之和不为零（非零和）。这是有意设计的，旨在提供更丰富的学习信号。

### 8.4 观测归一化的必要性

33 维全局向量的各维度物理含义不同、量纲不同（HP 可达数千，进度最多 512），必须通过归一化缩放到相似的数值范围，否则神经网络训练会不稳定。`OBS_NORMALIZATION` 中的缩放因子是关键调优参数。

### 8.5 与 SDK 的耦合

`ObservationEncoder` 和 `ActionMaskHandler` 通过 `BackendAdapter` 间接依赖游戏 SDK。如果 SDK 版本更新，可能需要更新 `compat/adapters.py` 中的兼容适配代码。
