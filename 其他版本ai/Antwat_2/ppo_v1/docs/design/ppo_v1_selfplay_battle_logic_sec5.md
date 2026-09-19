# 5. 动作空间与 Action Mask

> **源码参考**：
> - `ppo/src/ppo_antwar/utils/action_constants.py` — 动作空间常量定义（完整）
> - `ppo/src/ppo_antwar/network/antwar_net.py` — 网络前向中 Action Mask 的处理
> - `ppo/src/ppo_antwar/trainer/ppo_trainer.py` — `_select_action()`、`PPOAgent.act()`、`_ppo_update()` 中 mask 的使用
> - `ppo/src/ppo_antwar/trainer/selfplay.py` — SelfPlay 中对手随机动作选择时 mask 的使用
> - `Ant-Game/SDK/utils/actions.py` — SDK 中 `ActionCatalog.build()` 与 `action_mask()` 动态生成合法动作
> - `Ant-Game/SDK/utils/features.py` — `FeatureExtractor.encode_observation()` 中 action_mask 的编码
> - `Ant-Game/SDK/training/env.py` — `AntWarSequentialEnv._action_mask_for_agent()` 提供 mask

---

## 5.1 动作空间结构（96 维）

PPO 框架使用一个固定的 **96 维离散动作空间**（`ACTION_DIM = 96`），定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L53-L54)：

```python
MAX_ACTIONS = 96
ACTION_DIM = 96
```

### 5.1.1 Action ID 完整映射表

| action_id | 动作类型 | 数量 | 参数说明 |
|-----------|----------|------|----------|
| 0 | `NO_OP` | 1 | 不执行任何操作；始终合法 |
| 1–10 | `BUILD_TOWER` | 10 | 在指定 HIGHLAND 位置建造防御塔，参数为 `(x, y)` |
| 11–15 | `UPGRADE_TOWER` | 5 | 升级已有防御塔（指定 tower_id 和方向） |
| 16–20 | `DOWNGRADE_TOWER` | 5 | 降级已有防御塔（指定 tower_id） |
| 21–25 | `USE_LIGHTNING_STORM` | 5 | 使用超级武器"闪电风暴"（指定目标位置 `(x, y)`） |
| 26–30 | `USE_EMP_BLASTER` | 5 | 使用超级武器"EMP 冲击炮"（指定目标位置 `(x, y)`） |
| 31–35 | `USE_DEFLECTOR` | 5 | 使用超级武器"偏折护盾"（指定目标位置 `(x, y)`） |
| 36–40 | `USE_EMERGENCY_EVASION` | 5 | 使用超级武器"紧急闪避"（指定目标位置 `(x, y)`） |
| 41 | `UPGRADE_GENERATION_SPEED` | 1 | 科技升级：提升蚂蚁生成速度 |
| 42 | `UPGRADE_GENERATED_ANT` | 1 | 科技升级：提升生成蚂蚁等级 |
| 43–95 | **保留/未使用** | 53 | 当前未分配，始终被 Mask 为不合法 |

**总计**：43 个已定义动作 + 53 个保留位 = 96 维。

> **注意**：实际对局中，SDK 的 `ActionCatalog.build()` 动态生成合法 ActionBundle 列表，列表中第 `i` 个元素对应 action_id = i。`action_constants.py` 中的固定偏移量是 PPO 框架对动作空间的逻辑分组，具体映射需要结合环境运行时状态。

---

## 5.2 `ACTION_OFFSETS` 与 `ACTION_SPACE_CONFIG`

### 5.2.1 `ACTION_OFFSETS` — 动作类型起始偏移

定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L96-L105)：

```python
ACTION_OFFSETS: Dict[str, int] = {
    'build_tower': 1,
    'upgrade_tower': 11,
    'downgrade_tower': 16,
    'lightning_storm': 21,
    'emp_blaster': 26,
    'deflector': 31,
    'evasion': 36,
    'tech_upgrade': 41,
}
```

- `build_tower` 从 ID=1 开始：用于在 10 个塔位建造
- `upgrade_tower` 从 ID=11 开始：5 个升级槽位
- `downgrade_tower` 从 ID=16 开始：5 个降级槽位
- 四种超级武器各从固定位置开始，各占 5 个槽位
- `tech_upgrade` 从 ID=41 开始：2 种科技升级

### 5.2.2 `ACTION_SPACE_CONFIG` — 动作空间分片

定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L107-L117)：

```python
ACTION_SPACE_CONFIG: Dict[str, dict] = {
    'build_tower': {'start': 1, 'end': 11},
    'upgrade_tower': {'start': 11, 'end': 16},
    'downgrade_tower': {'start': 16, 'end': 21},
    'use_lightning_storm': {'start': 21, 'end': 26},
    'use_emp_blaster': {'start': 26, 'end': 31},
    'use_deflector': {'start': 31, 'end': 36},
    'use_emergency_evasion': {'start': 36, 'end': 41},
    'upgrade_generation_speed': {'start': 41, 'end': 42},
    'upgrade_generated_ant': {'start': 42, 'end': 43},
}
```

每个动作类型对应一个 `[start, end)` 半开区间。例如：
- `build_tower`: ID ∈ [1, 11)，共 10 个
- `use_lightning_storm`: ID ∈ [21, 26)，共 5 个
- `upgrade_generation_speed`: ID ∈ [41, 42)，仅 action_id=41

这些配置也可用于反向解析：给定 action_id，可以通过区间判断其属于哪种动作类型。

---

## 5.3 位置映射

### 5.3.1 `TOWER_POSITIONS` — 可建造塔位

定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L91-L94)：

```python
TOWER_POSITIONS: List[Tuple[int, int]] = [
    (4, 2), (4, 9), (5, 6), (5, 9), (6, 7), (6, 14), (7, 8), (8, 7),
    (10, 7), (11, 5), (11, 14), (12, 6), (12, 9), (13, 9), (13, 15), (14, 9),
]
```

共 16 个六边形网格位置。这些是地图上的 HIGHLAND 单元格，用于建造防御塔。在 `BUILD_TOWER` 动作中，action_id 对应到前 10 个塔位（ID=1 对应 `TOWER_POSITIONS[0]`，依此类推）。

这些位置形似一条从左上到右下的"对角线"排布，覆盖了双方基地区域附近的战略要地。

### 5.3.2 `SUPER_WEAPON_POSITIONS` — 超级武器可投放位置

定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L84-L89)：

```python
SUPER_WEAPON_POSITIONS: Dict[SuperWeaponType, List[Tuple[int, int]]] = {
    SuperWeaponType.LIGHTNING_STORM: [(7, 9), (8, 9), (9, 9), (10, 9), (11, 9)],
    SuperWeaponType.EMP_BLASTER:    [(7, 9), (8, 9), (9, 9), (10, 9), (11, 9)],
    SuperWeaponType.DEFLECTOR:      [(4, 9), (5, 9), (6, 9), (12, 9), (13, 9)],
    SuperWeaponType.EMERGENCY_EVASION: [(4, 9), (5, 9), (6, 9), (12, 9), (13, 9)],
}
```

每种超级武器有 5 个可投放位置，按行 `y=9`（地图中线）排列：

| 超级武器 | 投放列坐标 | 覆盖区域 |
|----------|-----------|----------|
| `LIGHTNING_STORM` | x = 7, 8, 9, 10, 11 | 地图中部（双方交战主干道） |
| `EMP_BLASTER` | x = 7, 8, 9, 10, 11 | 地图中部（双方交战主干道） |
| `DEFLECTOR` | x = 4, 5, 6, 12, 13 | 地图两侧（掩护我方蚂蚁） |
| `EMERGENCY_EVASION` | x = 4, 5, 6, 12, 13 | 地图两侧（撤退路线） |

- 攻击型武器（LIGHTNING_STORM、EMP_BLASTER）投放位置在中部（x=7~11），对敌方蚂蚁/塔造成效果
- 防御/辅助型武器（DEFLECTOR、EMERGENCY_EVASION）投放位置在两侧（x=4~6, 12~13），保护我方单位

---

## 5.4 枚举定义

### 5.4.1 `SuperWeaponType` — 超级武器类型

定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L57-L61)：

```python
class SuperWeaponType(IntEnum):
    LIGHTNING_STORM = 1
    EMP_BLASTER = 2
    DEFLECTOR = 3
    EMERGENCY_EVASION = 4
```

| 枚举值 | 数值 | 效果 |
|--------|------|------|
| `LIGHTNING_STORM` | 1 | 对范围 3 内敌方单位造成伤害；持续 20 ticks，冷却 100 ticks，消耗 150 金币 |
| `EMP_BLASTER` | 2 | 对范围 3 内敌方防御塔造成效果；持续 20 ticks，冷却 100 ticks，消耗 150 金币 |
| `DEFLECTOR` | 3 | 为范围 3 内我方蚂蚁提供偏折护盾；持续 10 ticks，冷却 50 ticks，消耗 100 金币 |
| `EMERGENCY_EVASION` | 4 | 使范围 3 内我方蚂蚁紧急闪避；持续 2 ticks，冷却 50 ticks，消耗 100 金币 |

> 具体数值参见 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L135-L140) 中的 `SUPER_WEAPON_STATS`。

### 5.4.2 `OperationType` — 操作类型

定义在 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L64-L74)：

```python
class OperationType(IntEnum):
    BUILD_TOWER = 11
    UPGRADE_TOWER = 12
    DOWNGRADE_TOWER = 13
    USE_LIGHTNING_STORM = 21
    USE_EMP_BLASTER = 22
    USE_DEFLECTOR = 23
    USE_EMERGENCY_EVASION = 24
    UPGRADE_GENERATION_SPEED = 31
    UPGRADE_GENERATED_ANT = 32
    NO_OP = 0
```

`OperationType` 枚举用于 SDK 内部的 `Operation` 对象，是 ActionBundle 中包含的实际游戏操作。其数值是 SDK 协议层面的操作码，与 PPO 框架的 action_id 是不同的概念。二者的映射关系通过 ActionBundle 列表的顺序建立。

---

## 5.5 Action Mask 机制

### 5.5.1 Mask 在 Observation 中的表示

Action Mask 是 Observation 字典的一个字段，类型为 **96 维 float 数组**：

```python
observation = {
    'board': np.ndarray(28, 19, 19),      # 棋盘特征
    'global': np.ndarray(33,),              # 全局统计特征
    'action_mask': np.ndarray(96,),         # 动作掩码
}
```

SDK 中 `AntWarSequentialEnv._action_mask_for_agent()` 返回 mask，随后通过 `FeatureExtractor.encode_observation()` 打包进 observation：

- SDK 文件：[env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L89-L101)
- 特征编码文件：[features.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py#L260-L275)

**Mask 取值含义**：

| mask[i] 值 | 含义 |
|-----------|------|
| `1.0` | action_id = i 是当前状态下的合法动作 |
| `0.0` | action_id = i 是非法动作（不可选） |

### 5.5.2 SDK 中 Mask 的生成逻辑

SDK 的 `ActionCatalog.action_mask()` 方法：

```python
def action_mask(self, bundles: list[ActionBundle]) -> np.ndarray:
    mask = np.zeros(self.max_actions, dtype=np.int8)
    mask[: len(bundles)] = 1
    return mask
```

（参见 [actions.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/actions.py#L75-L78)）

这意味着 **mask 是连续的前缀形式**：如果 SDK 生成了 N 个合法 ActionBundle（含 NO_OP），则 mask[0:N] = 1，mask[N:96] = 0。

- **NO_OP 始终合法**：因为 `ActionCatalog.build()` 第一步就添加了 `ActionBundle(name="hold")`（参见 [actions.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/actions.py#L56)），所以 `mask[0]` 始终为 1.0。

### 5.5.3 网络前向中的处理

在 `AntWarPolicyValueNetwork.forward()` 中，action_mask 用于屏蔽非法动作的 logit：

```python
def forward(self, board, global_features, action_mask=None):
    # ... 特征编码、编码器前向 ...
    action_logits = self.policy_head(merged)      # shape: [B, 96]
    value = self.value_head(merged)                # shape: [B, 1]

    if action_mask is not None:
        action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

    return action_logits, value
```

（参见 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L248-L266)）

**关键处理**：
- `action_mask == 0` 的维度 → 对应 logit 被设为 `-inf`
- 经过后续 `softmax` 后，`exp(-inf) = 0`，非法动作的概率严格为 0

### 5.5.4 `get_action()` 中的完整推理流程

`get_action()` 封装了从 logit 到最终动作的全流程：

```python
def get_action(self, board, global_features, action_mask=None, deterministic=False):
    action_logits, value = self.forward(board, global_features, action_mask)

    # 数值稳定化
    action_logits_stable = action_logits - action_logits.max(dim=-1, keepdim=True)[0]

    if deterministic:
        action = torch.argmax(action_logits, dim=-1)       # 贪心选择
    else:
        probs = F.softmax(action_logits_stable, dim=-1)    # 采样
        probs = probs.clamp(min=1e-10)
        # ... 数值安全检查 ...
        action = torch.multinomial(probs, 1).squeeze(-1)

    log_prob = F.log_softmax(action_logits_stable, dim=-1)
    action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)

    return action, action_log_prob, value
```

（参见 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L268-L294)）

**操作模式**：
- `deterministic=False`（训练中采样）：对 softmax 概率分布进行 `multinomial` 采样
- `deterministic=True`（评估/对战中选择）：使用 `argmax` 选最高 logit 的动作

### 5.5.5 `evaluate_actions()` 中的 Mask 处理

在 PPO 更新时，`evaluate_actions()` 计算给定动作的 log_prob 和熵。其中 mask 的处理更为细致：

```python
def evaluate_actions(self, board, global_features, actions, action_mask=None):
    action_logits, value = self.forward(board, global_features, action_mask)

    log_prob_stable = action_logits - action_logits.max(dim=-1, keepdim=True)[0]
    log_prob = F.log_softmax(log_prob_stable, dim=-1)
    action_log_prob = log_prob.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # 重新构建合法的概率分布（用于熵计算）
    probs = torch.exp(log_prob)
    probs = probs.masked_fill(action_mask == 0, 0.0)       # mask 掉非法动作
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs_sum = probs_sum.clamp(min=1e-10)
    probs = probs / probs_sum                               # 重新归一化

    log_probs_masked = log_prob.masked_fill(action_mask == 0, 0.0)
    dist_entropy = -(probs * log_probs_masked).sum(dim=-1) # 仅对合法动作计算熵

    return action_log_prob, value.squeeze(-1), dist_entropy
```

（参见 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L296-L318)）

这里的 mask 处理与 forward 中不同：`evaluate_actions` 需要计算合法的概率分布，因此重新 softmax 后在 mask=0 维度填 0 并重新归一化，确保概率仅在合法动作上分配。

---

## 5.6 Action Mask 在训练/对战代码中的使用位置

### 5.6.1 `_select_action()` — 训练中 Agent 动作选择

在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L628-L660) 中，提取 observation 中的 action_mask 并传给网络：

```python
def _select_action(self, observation: Dict[str, np.ndarray]) -> tuple:
    board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
    global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
    action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)
    # ...
    with torch.no_grad():
        action, log_prob, value = self.policy.get_action(
            board, global_obs, action_mask, deterministic=False
        )
```

### 5.6.2 `PPOAgent.act()` — 对战中的动作选择

在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L904-L922) 中：

```python
def act(self, observation, deterministic=False):
    # ...
    action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)
    with torch.no_grad():
        action, log_prob, value = self.policy.get_action(
            board, global_obs, action_mask, deterministic=deterministic
        )
```

### 5.6.3 SelfPlay 对手随机动作

在 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L557-L562) 中，当对手没有 agent 时，通过 mask 随机选合法动作：

```python
opponent_mask = torch.FloatTensor(opponent_obs["action_mask"]).to(self.device)
valid_actions = torch.where(opponent_mask > 0)[0]
if len(valid_actions) > 0:
    opponent_action = int(np.random.choice(valid_actions.cpu().numpy()))
else:
    opponent_action = 0  # fallback to NO_OP
```

### 5.6.4 EpisodeBatch 中 Mask 的收集

在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L604-L606) 和 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L564-L566) 中，每步将 action_mask 存入 EpisodeBatch：

```python
batch.observations_mask.append(player_obs['action_mask'])
```

在 PPO 更新期间，这些 mask 通过 `evaluate_actions()` 传给网络，确保 loss 计算时也只考虑合法动作的分布。

---

## 5.7 `NO_OP`（action_id=0）始终合法

在整个系统中，`NO_OP`（action_id=0）被特殊保证始终合法：

1. **SDK 层面**：[actions.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/actions.py#L56) 中 `ActionCatalog.build()` 的第一步总是添加 `ActionBundle(name="hold")`，对应 action_id=0
2. **枚举定义**：[action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L74) 中 `OperationType.NO_OP = 0`
3. **Fallback 策略**：当所有动作都被 mask 时，代码回退到 `action = 0`（NO_OP）
4. **网络层面**：mask[0] 始终为 1.0，因此 NO_OP 的 logit 永远不会被设为 `-inf`

这意味着在任何状态下，Agent 都至少有一个合法动作可选。
