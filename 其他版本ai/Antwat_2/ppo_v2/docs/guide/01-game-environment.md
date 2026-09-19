# 游戏环境模块 (Game Environment)

## 1. 背景与定位

### 1.1 业务问题

在强化学习训练中，智能体（Agent）需要与一个可交互的环境进行互动：智能体观察当前状态、选择动作、环境执行动作并返回新的状态和奖励。在本项目中，底层游戏引擎是 Ant-Game SDK（一个回合制的双人对战游戏），它提供了底层的状态管理和回合推进能力，但其 API 风格并非为强化学习设计——它需要使用者手动管理操作对象、状态快照、奖励计算等细节。

**游戏环境模块**的核心业务目标就是：**在 Ant-Game SDK 之上构建一层强化学习标准接口**，将底层游戏引擎包装为符合 Gymnasium 风格的 `reset/step` 交互范式，使得上层训练代码可以像使用任何标准 RL 环境一样与游戏交互。

### 1.2 在整个系统中的位置

```
自对弈编排 (Self-Play Orchestration)
        │
        │  创建环境、调用 reset/step
        ▼
┌─────────────────────────────────────┐
│      游戏环境模块                    │
│  ┌──────────┐  ┌───────────────┐   │
│  │AntWarEnv │─▶│ Observation   │   │
│  │(主入口)  │  │ Encoder       │   │
│  │          │  └───────────────┘   │
│  │          │  ┌───────────────┐   │
│  │          │─▶│ ActionMask    │   │
│  │          │  │ Handler       │   │
│  └────┬─────┘  └───────────────┘   │
│       │                            │
│       ▼                            │
│  ┌──────────┐                      │
│  │Backend   │                      │
│  │Adapter   │                      │
│  └────┬─────┘                      │
└───────┼─────────────────────────────┘
        │
        ▼
    Ant-Game SDK (外部)
```

- **上层**：自对弈编排模块（`selfplay.py`）创建 `AntWarEnv` 实例，通过标准的 `reset()` 和 `step()` 方法与环境交互。
- **下层**：`BackendAdapter` 适配 Ant-Game SDK 的版本差异，统一创建游戏运行时。
- **同层协作**：`ObservationEncoder` 和 `ActionMaskHandler` 作为环境内部组件，分别负责观测编码和动作合法性校验。

### 1.3 与其他模块的关系

| 外部模块 | 关系 | 交互方式 |
|---------|------|---------|
| 基础设施 (`action_constants`) | 依赖 | 使用动作空间常量、奖励配置、观测归一化参数 |
| 自对弈编排 (`selfplay.py`) | 被依赖 | 创建 `AntWarEnv` 实例，调用 `reset/step` |
| 对战执行 (`agent_loader.py`) | 被依赖 | 使用 `ActionMaskHandler` 进行动作转换 |
| 策略网络 (`agent.py`) | 被依赖 | 使用 `ObservationEncoder` 进行观测编码 |

---

## 2. 核心概念

### 2.1 关键术语

| 术语 | 说明 |
|------|------|
| **Runtime** | Ant-Game SDK 中的游戏运行时，管理一盘对局的状态、操作应用和回合推进 |
| **Operation** | SDK 中的操作对象，表示一个具体的游戏动作（如"在位置(3,5)建塔"） |
| **Action ID** | 0-95 的整数，是强化学习中动作空间的扁平化表示。每个 ID 对应一个具体的 Operation |
| **观测 (Observation)** | 神经网络输入的三元组：`board` (28,19,19) 棋盘特征、`global` (33,) 全局特征、`action_mask` (96,) 动作合法性掩码 |
| **回合 (Round)** | 游戏的基本时间单位。每个回合双方各执行一系列操作，然后由 SDK 推进状态 |
| **批次模式 (Batch Mode)** | `step()` 同时接收双方动作，一次调用完成一个完整回合 |
| **顺序模式 (Sequential Mode)** | `step()` 每次只接收一个玩家的动作，需要两次调用完成一个回合 |

### 2.2 数据流概要

```
训练循环                     AntWarEnv                     SDK
   │                           │                           │
   ├─ env.reset() ────────────▶│                           │
   │                           ├─ BackendAdapter           │
   │                           │   .create_runtime() ─────▶│ 创建运行时
   │                           │                           │
   │                           ├─ ObservationEncoder       │
   │                           │   .encode(state, p) ─────▶│ FeatureExtractor
   │                           │                           │
   │  ◀── (obs, info) ────────┤                           │
   │                           │                           │
   │  [训练循环继续...]        │                           │
   │                           │                           │
   ├─ env.step(action_id) ────▶│                           │
   │                           ├─ ActionMaskHandler        │
   │                           │   .action_id_to_op() ────▶│ 转为 Operation
   │                           │                           │
   │                           ├─ runtime.apply_ops() ────▶│ 应用操作
   │                           ├─ state.advance_round() ──▶│ 推进回合
   │                           │                           │
   │                           ├─ _snapshot_state() ──────▶│ 捕获状态快照
   │                           ├─ _compute_battle_rewards()│ 计算奖励
   │                           ├─ ObservationEncoder       │
   │                           │   .encode() ─────────────▶│ 编码新观测
   │                           │                           │
   │  ◀── (obs, reward, done) ─┤                           │
   │                           │                           │
```

---

## 3. 架构总览

### 3.1 模块组成

```
game-environment/
├── antwar_env.py          # AntWarEnv - 环境主入口
├── observation.py         # ObservationEncoder - 观测编码器
├── action_mask.py         # ActionMaskHandler - 动作掩码处理器
└── compat/
    └── adapters.py        # BackendAdapter - SDK 适配层
```

### 3.2 组件关系

```
                    ┌──────────────────────┐
                    │     AntWarEnv         │  ← 对外唯一入口
                    │  (Gymnasium 风格)     │
                    ├──────────────────────┤
                    │  - player_id         │
                    │  - backend_type      │
                    │  - _obs_encoder      │───▶ ObservationEncoder
                    │  - _mask_handler     │───▶ ActionMaskHandler
                    │  - _runtime          │───▶ MatchRuntime (SDK)
                    │  - 内部状态缓存      │
                    ├──────────────────────┤
                    │  reset()             │
                    │  step()              │
                    │  close()             │
                    └──────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
   ┌────────────────┐ ┌──────────┐ ┌──────────────┐
   │ Observation    │ │ Action   │ │ Backend      │
   │ Encoder        │ │ Mask     │ │ Adapter      │
   │                │ │ Handler  │ │              │
   │ 编码棋盘状态   │ │ 生成合法 │ │ 创建 Runtime │
   │ 编码全局特征   │ │ 性掩码  │ │ 包装兼容层   │
   │ 生成观测三元组 │ │ ID→Op   │ │              │
   └────────────────┘ └──────────┘ └──────────────┘
                              │              │
                              ▼              ▼
                     ┌───────────────────────────┐
                     │     Ant-Game SDK          │
                     │  (外部依赖)                │
                     │  FeatureExtractor         │
                     │  MatchRuntime             │
                     │  Operation                │
                     │  state.can_apply_op()     │
                     └───────────────────────────┘
```

---

## 4. 核心组件详解

### 4.1 AntWarEnv (`env/antwar_env.py`)

#### 4.1.1 职责

封装游戏后端 SDK，提供 Gymnasium 风格的 `reset/step/close` 接口。核心功能包括：
- 创建和管理游戏运行时（Runtime）
- 执行回合（应用双方操作、推进状态、计算奖励）
- 编码观测输出
- 管理内部状态缓存（塔建造计数、空操作连续计数等）

#### 4.1.2 构造函数

```python
def __init__(self, player_id: int, backend_type: str, prefer_native: bool = False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `player_id` | int | 当前环境视角的玩家 ID（0 或 1） |
| `backend_type` | str | 后端类型标识符（传递给 SDK） |
| `prefer_native` | bool | 是否优先使用 C++ Native 后端（而非 Python 后端） |

构造时创建 `ObservationEncoder` 和 `ActionMaskHandler` 实例，并初始化以下内部状态：
- `_tower_build_slot_counts`: 记录每个位置建塔次数（用于建塔奖励阶梯递减）
- `_noop_streak`: 记录连续空操作次数（用于空操作惩罚递增）
- `_round_reward_stats`: 回合级奖励统计

#### 4.1.3 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `reset` | `() -> Tuple[Dict, Dict]` | 重置环境。创建新 Runtime，编码初始观测，返回 `(observations, info)`。observations 包含 `player_0` 和 `player_1` 两个视角 |
| `step` | `(action: Union[int, Dict]) -> Tuple[Dict, Dict, bool, bool, Dict]` | 执行动作。`int` 为顺序模式，`Dict[int, int]` 为批量模式。返回 `(obs, rewards, terminated, truncated, info)` |
| `close` | `() -> None` | 清理环境资源，清空所有内部状态 |

**step 返回值详解**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `observations` | `Dict[str, Dict]` | 键为 `'player_0'`/`'player_1'`，每个值为 `{board, global, action_mask}` |
| `rewards` | `Dict[str, float]` | 键为 `'player_0'`/`'player_1'`，值为标量奖励 |
| `terminated` | `Dict[str, bool]` | 是否因胜负而终止 |
| `truncated` | `Dict[str, bool]` | 是否因回合数上限而截断 |
| `info` | `Dict` | 附加信息（先手标记等） |

#### 4.1.4 关键算法：回合执行 (`_resolve_turn_from_ops`)

这是环境最核心的逻辑，控制一次回合的完整执行流程：

```
_resolve_turn_from_ops(ops_0, ops_1)
│
1. _snapshot_state(state)  →  捕获执行前的状态快照（old_health, old_coins, old_towers...）
│
2. runtime.apply_self_operations(player_0_ops)
   runtime.apply_opponent_operations(player_1_ops)    [视角0]
   或反之 [视角1]
│
3. _snapshot_state(state)  →  捕获操作后的中间态（mid_health, mid_coins）
│
4. state.advance_round()  →  SDK 推进一个回合（蚂蚁移动、金币产生、冷却减少等）
│
5. _snapshot_state(state)  →  捕获推进后的状态（new_health, new_coins...）
│
6. _compute_battle_rewards(old, mid, new, ...)  →  计算完整奖励
│
7. 检查终止条件（任意一方 HP ≤ 0 → terminated）
│
8. 返回 (rewards, terminated, truncated, info)
```

#### 4.1.5 关键算法：奖励计算 (`_compute_battle_rewards`)

奖励由**8 个分量**组成，加权求和后裁剪到对称范围 `[-step_reward_clip, step_reward_clip]`：

| 奖励分量 | 计算方式 | 说明 |
|---------|---------|------|
| 基地 HP 伤害 | `(old_hp - new_hp) * REWARD_CONFIG["hp_attack_weight"]` | 对敌方基地造成伤害的奖励 |
| 塔 HP 伤害 | `(old_tower_hp - new_tower_hp) * REWARD_CONFIG["tower_attack_weight"]` | 对敌方塔造成伤害的奖励 |
| 金币被动收入 | `(new_coins - old_coins) * REWARD_CONFIG["coin_income_weight"]` | 每回合被动获得金币的奖励 |
| 塔存活奖励 | 按等级加权的塔存活数量 × 权重 | 鼓励保护己方塔（高等级塔权重更高） |
| 科技加成奖励 | `科技等级变化 × REWARD_CONFIG["tech_upgrade_reward"]` | 科技升级带来的持续性收益 |
| 余额奖励 | `余额保有量 × 权重` | 鼓励合理管理金币（不浪费） |
| 蚂蚁死亡惩罚 | `死亡数 × REWARD_CONFIG["ant_death_penalty"]` | 对己方蚂蚁死亡的惩罚 |
| 终局胜负奖励 | 胜 `+win_bonus` / 负 `-win_bonus` / 平 `0` | 对局结束时的胜负奖励 |

此外，`_compute_action_reward` 计算**动作级即时奖励**：
- 空操作（NOOP）：连续性递增惩罚（`noop_streak * penalty_per_noop`，上限 `max_noop_penalty`），鼓励智能体不要无所作为
- 建塔/升塔/降塔/部署武器/科技升级：各有固定的分段奖励值

#### 4.1.6 内部调用链

```
reset()
  ├─ self._obs_encoder = ObservationEncoder()
  ├─ self._mask_handler = ActionMaskHandler()
  ├─ runtime = BackendAdapter.create_runtime(player, ...)
  ├─ obs = self._obs_encoder.encode(state, player)
  └─ return obs, info

step(action)
  ├─ 如果 action 是 int → _step_sequential(action)
  │    ├─ 第一次 step: 缓存 player_0 的 action，返回占位 obs
  │    └─ 第二次 step: 转换为 _step_batch({0: cached, 1: action})
  ├─ 如果 action 是 dict → _step_batch(actions)
  │    ├─ 将双方 action_id 转为 Operation（通过 ActionMaskHandler）
  │    ├─ _compute_action_reward()  →  动作级即时奖励
  │    ├─ _resolve_turn_from_ops()  →  执行回合
  │    └─ 编码新观测
  └─ return (obs, rewards, terminated, truncated, info)
```

---

### 4.2 ObservationEncoder (`env/observation.py`)

#### 4.2.1 职责

将游戏 SDK 的 `state` 对象编码为神经网络可以处理的张量格式。输出包含三个部分：
- **board**: 棋盘特征图，shape `(28, 19, 19)`，通过 SDK 的 `FeatureExtractor` 提取
- **global**: 全局特征向量，shape `(33,)`，手工编码的 33 维特征
- **action_mask**: 动作合法性掩码，shape `(96,)`，委托 `ActionMaskHandler` 生成

#### 4.2.2 构造函数

```python
def __init__(self)
```

无参数。构造时校验 `OBS_NORMALIZATION` 常量是否完整，创建 SDK 的 `FeatureExtractor` 和 `ActionMaskHandler` 实例。

#### 4.2.3 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `encode` | `(state, player: int) -> Dict[str, np.ndarray]` | 编码单个玩家视角的观测，返回 `{board, global, action_mask}` 字典 |

#### 4.2.4 全局特征向量（33 维）

| 索引范围 | 特征 | 归一化 |
|---------|------|--------|
| 0 | 回合进度 (round / MAX_ROUND) | — |
| 1 | 己方 HP - 敌方 HP 的差值 | `hp_scale` 归一化 |
| 2 | 己方金币 / 敌方金币 | `coin_scale` 归一化 |
| 3 | 安全金币（有保底的残值） | — |
| 4 | 前线优势距离 | `distance_scale` 归一化 |
| 5-6 | 敌方前线距离 / 己方前线距离 | `distance_scale` 归一化 |
| 7-8 | 己方推进进度 / 敌方推进进度 | — |
| 9-10 | 己方塔数量 / 敌方塔数量 | — |
| 11-12 | 己方塔等级总和 / 敌方塔等级总和 | — |
| 13-14 | 击杀差 / 阵亡差（老兵机制） | — |
| 15 | 塔分布集中度 | — |
| 16 | 插槽填充率 | — |
| 17-18 | 己方科技等级（生产速度/产兵质量） | — |
| 19 | 到最近敌方位置的距离 | `distance_scale` 归一化 |
| 20 | 基地弧度覆盖范围 | — |
| 21 | 塔间距 | — |
| 22-23 | 己方基地 HP / 敌方基地 HP | `hp_scale` 归一化 |
| 24 | 己方当前金币 | `coin_scale` 归一化 |
| 25-32 | 4 种超级武器的双方冷却比例 | 各归一化到 `[0, 1]` |

#### 4.2.5 内部调用链

```
encode(state, player)
  ├─ board = FeatureExtractor.encode_board(state, player_id)  →  (28, 19, 19)
  ├─ global = self._encode_global(state, player_id)
  │    ├─ summary = FeatureExtractor.summarize(state, player_id)
  │    └─ 手动组装 33 维向量（HP/金币/塔/科技/武器冷却等，除以 OBS_NORMALIZATION）
  ├─ action_mask = self._mask_handler.get_action_mask(state, player)  →  (96,)
  └─ return {board, global, action_mask}
```

---

### 4.3 ActionMaskHandler (`env/action_mask.py`)

#### 4.3.1 职责

管理 96 维扁平动作空间与 SDK `Operation` 对象之间的双向映射，并生成动作合法性掩码，确保策略网络只输出当前状态下合法的动作。

#### 4.3.2 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `get_action_mask` | `(state, player) -> np.ndarray` | 遍历 96 个 action_id，逐个检查合法性，生成 `(96,)` 的 float32 掩码 |
| `action_id_to_op` | `(action_id, state, player) -> Optional[Operation]` | 将 action_id 转为 SDK Operation（公开入口） |
| `flat_mask_to_type_mask` | `(flat_mask) -> np.ndarray` | 将 96 维扁平掩码收缩为 9 维类型掩码 |
| `flat_mask_to_target_mask` | `(flat_mask, type_id) -> np.ndarray` | 提取指定动作类型的 target 子掩码 |

#### 4.3.3 动作空间结构

| 动作类型 | action_id 范围 | 数量 | target 含义 |
|---------|---------------|------|------------|
| noop (空操作) | 0 | 1 | — |
| build_tower (建塔) | 1-10 | 10 | 塔位置索引 (0-9) |
| upgrade_tower (升塔) | 11-50 | 40 | 位置 (0-9) × 升级方向 (0-3) |
| downgrade_tower (降塔) | 51-60 | 10 | 塔位置索引 (0-9) |
| lightning_storm | 61-65 | 5 | 目标位置 (0-4) |
| emp_blaster | 66-70 | 5 | 目标位置 (0-4) |
| deflector | 71-75 | 5 | 目标位置 (0-4) |
| evasion | 76-80 | 5 | 目标位置 (0-4) |
| tech_upgrade (科技升级) | 81-82 | 2 | 0=生产速度, 1=蚂蚁血量 |

#### 4.3.4 关键逻辑

- **前 1000 回合强制禁用降塔操作**：这是一个有意的训练稳定性设计，防止模型学会"先建后拆"的投机行为模式。
- 合法性判断委托给 SDK 的 `state.can_apply_operation()`，例如：金币不足时无法建塔、科技已满时无法升级等。

---

### 4.4 BackendAdapter (`compat/adapters.py`)

#### 4.4.1 职责

封装 Ant-Game SDK 的版本差异，提供统一的游戏运行时创建接口。解决以下实际问题：
- SDK 不同版本中 `MatchRuntime.create()` 的参数签名可能不同（如 `movement_policy`、`cold_handle_rule_illegal` 为可选参数）
- 需要自动检测并优先使用 Native（C++）后端提升性能

#### 4.4.2 公开方法

| 方法 | 类型 | 签名 | 功能 |
|------|------|------|------|
| `create_runtime` | 静态 | `(player, seed, cold_handle_rule_illegal, prefer_native, movement_policy) -> MatchRuntimeWrapper` | 创建运行时，动态适配 SDK 版本差异 |
| `supports_native_backend` | 静态 | `() -> bool` | 检测是否支持 Native 后端 |
| `get_backend_info` | 静态 | `() -> Dict` | 获取后端版本和 CUDA 信息 |

#### 4.4.3 MatchRuntimeWrapper

包装原始的 `MatchRuntime`，新增 `resolve_turn(player_0_ops, player_1_ops)` 方法，兼容旧版 API：
- 内部根据自身视角，通过 `apply_self_operations`/`apply_opponent_operations` 应用双方操作
- 调用 `state.advance_round()` 推进回合
- 返回 `_SimulatedTurnResolution` 包装对象

---

## 5. 外部交互

### 5.1 本模块依赖的外部模块

| 模块 | 依赖内容 | 依赖方式 |
|------|---------|---------|
| 基础设施 (`action_constants`) | `ACTION_DIM`, `TYPE_CONFIG`, `REWARD_CONFIG`, `OBS_NORMALIZATION`, `TOWER_POSITIONS` | 直接 import |
| Ant-Game SDK (外部) | `MatchRuntime`, `load_backend`, `FeatureExtractor`, `Operation`, `OperationType`, `state.can_apply_operation()` | 直接 import |

### 5.2 依赖本模块的外部模块

| 模块 | 使用内容 | 使用方式 |
|------|---------|---------|
| 自对弈编排 (`selfplay.py`) | `AntWarEnv` | 创建实例，调用 `reset()/step()` |
| 对战执行 (`agent_loader.py`) | `ActionMaskHandler` | 创建实例，调用 `action_id_to_op()` 转换动作 |
| 对战执行 (`battle_simulator.py`) | `BackendAdapter` | 调用 `create_runtime()` 创建游戏运行时 |

### 5.3 典型交互场景

#### 场景 1：自对弈训练中收集一条轨迹

1. `SelfPlayTrainer` 调用 `env = AntWarEnv(player_id=0, backend_type="python")` 创建环境
2. 调用 `obs, info = env.reset()` 获得初始观测（含 board, global, action_mask）
3. 策略网络根据观测选择动作 `action_id`（96 维空间中的合法动作）
4. 调用 `obs, reward, terminated, truncated, info = env.step(action_id)` 执行动作
5. `AntWarEnv` 内部：`ActionMaskHandler.action_id_to_op()` 转为 Operation → SDK 执行 → 计算奖励 → 编码新观测
6. 重复 3-5 直到 `terminated=True` 或 `truncated=True`

#### 场景 2：对战执行中使用环境

1. `BattleSimulator._execute_battle()` 调用 `BackendAdapter.create_runtime(player=0)` 创建运行时
2. 在每回合中，分别从两个 `PPOWarAgent` 获取操作列表
3. 调用 `runtime.resolve_turn(ops_0, ops_1)` 推进回合
4. 直到任意一方 HP ≤ 0 或回合数达到上限

#### 场景 3：Agent 推理时使用观测编码

1. `PPOAgent.act(observation)` 接收来自环境的观测字典
2. 调用 `observation_to_tensors()` 将 numpy 字典转为 PyTorch 张量（加 batch 维度）
3. 策略网络 `AntWarPolicyValueNetwork.forward(board_t, global_t, mask_t)` 执行前向推理
4. 返回动作 logits，在 `action_mask` 的约束下采样合法动作

---

## 6. 关键设计决策

### 6.1 独立 Policy/Value 编码器（非共享编码器）

虽然 AntWarEnv 本身不涉及网络架构，但环境输出的观测结构（board + global）直接决定了策略网络采用双编码器设计。环境的设计从一开始就支持了策略网络的双编码器需求。

### 6.2 顺序模式 vs 批量模式

| 模式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 顺序模式 (Sequential) | 自对弈训练（交替收集双方数据） | 每次只需一个 action_id，接口简洁 | 两次 step() 才完成一回合 |
| 批量模式 (Batch) | 基线对战评估 | 一次 step() 完成整回合 | 需要双方动作同时传入 |

`step()` 通过参数类型自动分派模式，上层代码无需关心具体实现。

### 6.3 奖励计算的精细设计

奖励不是简单的"赢了 +1，输了 -1"，而是由 8 个回合级分量和 6 个动作级分量组成。这种设计：
- **塑造 (Shaping)**：给智能体中间过程的反馈，加速学习
- **多维度**：鼓励同时关注进攻（HP 伤害）、防守（塔存活）、经济（金币管理）和科技
- **空操作惩罚**：避免智能体学会"什么都不做"的退化策略
- **建塔阶梯递减**：同一位置反复建塔的收益递减，鼓励策略多样化

### 6.4 前 1000 回合禁用降塔

这是一个有意的约束设计：
- 防止模型在早期学到"先建塔再降级刷奖励"的投机行为
- 强制模型在前 1000 回合专注于正常的建塔和升级策略
- 1000 回合后恢复降塔操作，此时模型应已学到合理的建塔策略

### 6.5 BackendAdapter 的兼容性设计

`BackendAdapter.create_runtime()` 使用 `inspect.signature()` 动态检测 `MatchRuntime.create()` 的签名，仅在目标方法接受时传入额外参数。这种设计使代码能够兼容多个版本的 SDK，而不需要条件导入或版本判断。
