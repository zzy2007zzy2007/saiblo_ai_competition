# 动作空间语义改造 — 详细技术方案

> 基于 [action_space_redesign.md](./action_space_redesign.md) 的详细代码实现方案

## 1. 总体变更概览

### 1.1 核心变更

| 项目 | 改造前 | 改造后 |
|------|--------|--------|
| 动作空间维度 | 119（扁平 one-hot） | 27（4 段语义向量） |
| 网络输出头 | 1×type_head(9) + 9×target_head | 1×Linear(256→27) |
| 动作解码 | 分层 one-hot → flat action_id | 分段 softmax + 映射表 → (s,t,st,p) 四元组→Operation |
| 合法性检查 | 119-d 外部 mask 过滤 | 解码后 `can_apply_operation()` 检查 |
| 策略网络 | Policy + Value 双编码器 | 仅 Policy 编码器 |
| 辅助任务 | 6 个 Auxiliary Head | 全部删除 |
| 训练元数据 | `get_action` 返回 log_prob+value+type_probs | `get_action` 仅返回解码动作四元组 |
| 参数量 | ~400K | ~105K |

### 1.2 删改文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `network/heads.py` | **重写** | 删除 StructuredActionHead / ValueHead / 所有 AuxHead，新增 ActionHead |
| `network/ant_war_policy_value_network.py` | **重写** | 删除 Value 编码器、enable_auxiliary、evaluate_actions、get_value_only、logit_noise_std；重写 get_action |
| `utils/action_constants.py` | **重写** | ACTION_DIM=27；删除 TYPE_CONFIG/_UPGRADE_DIRECTIONS；新增映射表常量 |
| `env/action_mask.py` | **重写** | `_action_id_to_op` → `construct_operation(s,t,st,p)`；移除 flat_mask 相关逻辑 |
| `battle/ga_war_agent.py` | **修改** | 适配新 get_action 接口和解码流程 |
| `trainer/neural_agent.py` | **修改** | 删除 get_value / act_record；简化 act |
| `env/observation.py` | **修改** | 更新 mask 维度注释；移除 mask 依赖 |
| `config/config_parser.py` | **修改** | 删除 enable_auxiliary 配置项 |

### 1.3 无改动文件

| 文件 | 原因 |
|------|------|
| `genome/genome.py` | 通过 `named_parameters()` 自动适配新网络结构 |
| `network/hex_cnn_encoder.py` | 编码器本身不变 |
| `network/mlp_encoder.py` | 同上 |
| `network/hex_conv.py` | 同上 |

---

## 2. `utils/action_constants.py` — 常量定义改造

### 2.1 删除内容

```python
# 删除
ACTION_DIM = 119  →  改为 ACTION_DIM = 27

# 删除整个 TYPE_CONFIG 列表（9 种类型的 flat_start/flat_end/max_targets 映射）
TYPE_CONFIG = [...]  # 删除

# 删除
_UPGRADE_DIRECTIONS = 4

# 删除函数
validate_action_space_consistency()  # 删除
action_id_to_type_and_target()       # 删除
```

### 2.2 新增常量

```python
# ── 动作空间维度 ──
ACTION_DIM = 27

# ── 语义分段定义 ──
ACTION_SEGMENTS = {
    "strategy":  (0, 3),    # [0:3)   strategy logits
    "type":      (3, 9),    # [3:9)   type_idx logits
    "sub_type":  (9, 19),   # [9:19)  sub_type logits
    "position":  (19, 27),  # [19:27) position logits
}

SUB_TYPE_SEGMENTS = {
    "main_tree":    (9, 12),    # [9:12)  HEAVY/QUICK/MORTAR
    "sub_variant":  (12, 15),   # [12:15) variant A/B/C
    "producer":     (15, 19),   # [15:19) PRODUCER/FAST/SIEGE/MEDIC
}
```

### 2.3 新增多表位置映射常量

```python
# ── 塔操作：position 0-7 → 当前玩家的固定塔位 ──
# 两个玩家的塔位镜像对称，战略意义相同的 position 在不同玩家下映射到不同坐标
_PLAYER_TOWER_POS = {
    0: [  # Player 0
        (5, 9), (4, 9), (7, 8), (6, 7),   # pos 0-3: 中线核心 → 中央战场
        (8, 7), (5, 6), (6, 14), (4, 2),   # pos 4-7: 中央侧翼 → 角落防守
    ],
    1: [  # Player 1（镜像对称）
        (13, 9), (14, 9), (10, 7), (12, 6),  # pos 0-3
        (11, 5), (12, 9), (11, 14), (13, 15), # pos 4-7
    ],
}

# ── 超级武器：position 0-7 → 5 个预设位的冗余映射 ──
# pos 5-7 冗余映射到 pos 2-4（中线/核心位），提高合法率
_WEAPON_POS_REDUNDANCY = [0, 1, 2, 3, 4, 2, 3, 4]

# ── 科技升级：position 0-7 → 2 种科技类型的冗余映射 ──
_TECH_POS_MAP = {
    0: "generation_speed",  # pos 0-3 → 生产速度
    1: "generation_speed",
    2: "generation_speed",
    3: "generation_speed",
    4: "generated_ant",     # pos 4-7 → 蚂蚁类型
    5: "generated_ant",
    6: "generated_ant",
    7: "generated_ant",
}
```

### 2.4 新增动作解码映射表

```python
# ── (Strategy, Type) 联合映射表 ──
# strategy=0 防御, strategy=1 无操作, strategy=2 进攻
# 每个条目: (action_name, flags)
#   flags = {"use_sub_type": True} 表示该操作需要解析 sub_type
#   None 表示 → NOOP

ACTION_DECODE_TABLE = [
    # strategy=0 (防御)
    {
        0: ("build_tower",          {}),
        1: ("upgrade_tower",        {"use_sub_type": True}),
        2: ("lightning_storm",      {}),  # 杀伤敌方蚂蚁
        3: ("lightning_storm",      {}),  # 冗余
        4: ("lightning_storm",      {}),  # 冗余
        5: None,                          # 进攻独占 slot → NOOP
    },
    # strategy=1 (无操作) — 全部 NOOP
    {i: None for i in range(6)},
    # strategy=2 (进攻)
    {
        0: ("downgrade_tower",      {}),
        1: ("downgrade_tower",      {}),  # 冗余
        2: ("deflector",            {}),  # 保护己方蚂蚁
        3: ("evasion",              {}),  # 己方蚂蚁闪避
        4: ("emp_blaster",          {}),  # 压制敌方塔
        5: ("tech_upgrade",         {}),
    },
]
```

### 2.5 保留不变的内容

```python
# 以下保持不动
SUPER_WEAPON_POSITIONS            # 超级武器预设位置
TOWER_POSITIONS                   # 16 个可用塔位（用于兼容旧代码，新逻辑使用 _PLAYER_TOWER_POS）
SuperWeaponType                   # 枚举
REWARD_CONFIG                     # 奖励配置
OBS_NORMALIZATION                 # 观测归一化参数
is_valid_tower_type()             # 塔类型合法性校验
validate_upgrade_operation()      # 升级操作校验
```

---

## 3. `network/heads.py` — 输出头改造

### 3.1 删除内容

```python
# 删除整个类
StructuredActionHead    # 分层 type/target 结构不再需要
ValueHead               # PPO 价值头不再需要
TowerDamageHead         # 辅助预测头不再需要
GoldIncomeHead          # 辅助预测头不再需要
BaseDamageHead          # 辅助预测头不再需要
```

### 3.2 新增 ActionHead

```python
class ActionHead(nn.Module):
    """语义分段动作头 - 将 256 维特征投影为 27 维向量。

    27 维向量语义分段：
        [0:3)   strategy  logits  → argmax → {0,1,2}  [防御/无操作/进攻]
        [3:9)   type      logits  → argmax → {0..5}
        [9:19)  sub_type  logits  → argmax → {0..9} (分段解析见下文)
        [19:27) position  logits  → argmax → {0..7}

    注意：type_idx 无固有语义，需联合 strategy 查 ACTION_DECODE_TABLE 确定动作。
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.output_head = nn.Linear(hidden_dim, 27)
        self._init_weights()

    def _init_weights(self):
        orthogonal_init(self.output_head, gain=0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播：输出 27 维原始 logits

        Args:
            features: (batch, hidden_dim) 编码器输出

        Returns:
            logits: (batch, 27) 未分段 softmax 的原始 logits
        """
        return self.output_head(features)

    @staticmethod
    def decode(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """分段 argmax 解码（批处理版本）

        GA 确定性决策：softmax 是单调函数不改变 argmax 结果，故直接对 logits 取 argmax。

        Args:
            logits: (batch, 27) 网络输出的原始 logits

        Returns:
            strategy:   (batch,)  ∈ {0,1,2}
            type_idx:   (batch,)  ∈ {0..5}
            sub_type:   (batch,)  ∈ {0..9}
            position:   (batch,)  ∈ {0..7}
        """
        strategy = torch.argmax(logits[:, 0:3], dim=-1)
        type_idx = torch.argmax(logits[:, 3:9], dim=-1)
        sub_type = torch.argmax(logits[:, 9:19], dim=-1)
        position = torch.argmax(logits[:, 19:27], dim=-1)

        return strategy, type_idx, sub_type, position
```

### 3.3 导出调整

```python
__all__ = ["ActionHead"]
```

---

## 4. `network/ant_war_policy_value_network.py` — 主网络改造

### 4.1 构造函数改造

```python
class AntWarPolicyValueNetwork(nn.Module):
    """GA 策略网络 — 语义分段动作输出。

    本网络仅用于 GA 策略推断（子代生成/对战），不包含任何 PPO 组件：

    输入:
        board:      (batch, 29, 19, 19)  棋盘特征图
        global_vec: (batch, 33)           全局特征向量

    输出（通过 get_action）:
        strategy: int   ∈ {0,1,2}  战略选择
        type_idx: int   ∈ {0..5}   类型选择
        sub_type: int   ∈ {0..9}   子类选择
        position: int   ∈ {0..7}   位置选择

    动作解码在外部（GAWarAgent/ActionMaskHandler）完成：
        (strategy, type_idx) → 查 ACTION_DECODE_TABLE → 动作名称
        position → 查多表位置映射 → 具体坐标
        (动作名称, 坐标) → construct_operation() → Operation
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── 策略编码器（唯一编码器） ──
        self.policy_cnn = HexCNNEncoder(hidden_dim)
        self.policy_mlp = MLPEncoder()
        cnn_out_channels = HexCNNEncoder.CHANNEL_COMPRESS  # 128
        cnn_out_dim = (
            cnn_out_channels
            * HexCNNEncoder.POOL_OUTPUT_SIZE
            * HexCNNEncoder.POOL_OUTPUT_SIZE
        )
        mlp_out_dim = MLP_OUTPUT_DIM
        fused_dim = cnn_out_dim + mlp_out_dim
        self.policy_proj = nn.Linear(fused_dim, hidden_dim)
        self.policy_layer_norm = nn.LayerNorm(hidden_dim)

        # ── 动作输出头 ──
        self.action_head = ActionHead(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        orthogonal_init(self.policy_proj, gain=np.sqrt(2))
```

**关键删除项**（与 PPO 旧架构对比）：

| 删除项 | 行数 | 原因 |
|--------|------|------|
| `value_cnn` | ~6 行 | PPO 独立价值编码器 |
| `value_mlp` | ~6 行 | 同上 |
| `value_proj` | ~3 行 | 同上 |
| `value_layer_norm` | ~2 行 | 同上 |
| `value_head` | ~2 行 | PPO 价值头 |
| 6 个 Auxiliary Head 实例 | ~8 行 | 辅助预测任务 |
| `enable_auxiliary` 参数及条件分支 | ~3 行 | 不再需要 |
| `logit_noise_std` 属性 | ~1 行 | GA 不需要探索噪声 |

### 4.2 编码器方法改造

```python
def encode(self, board: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
    """单一编码器：棋盘 CNN + 全局 MLP → hidden_dim 向量

    Args:
        board:      (batch, 29, 19, 19)
        global_vec: (batch, 33)

    Returns:
        features:   (batch, hidden_dim)
    """
    cnn_out = self.policy_cnn(board)
    mlp_out = self.policy_mlp(global_vec)
    fused = torch.cat([cnn_out, mlp_out], dim=1)
    projected = self.policy_proj(fused)
    return self.policy_layer_norm(projected)
```

说明：`_encode_policy` 重命名为 `encode`（不再需要区分 policy/value 编码器）。

### 4.3 删除方法清单

```python
# 以下方法全部删除
def forward(self, ...)                    # PPO 训练正向传播
def get_value_only(self, ...)             # PPO 价值估值
def evaluate_actions(self, ...)           # PPO 训练评估
def compute_auxiliary_loss(self, ...)     # 辅助损失计算
```

### 4.4 get_action 方法重写

```python
@torch.no_grad()
def get_action(
    self,
    board: torch.Tensor,
    global_vec: torch.Tensor,
    deterministic: bool = True,
) -> Tuple[int, int, int, int]:
    """网络推理：输出 27 维向量并解码为四元组。

    Args:
        board:         (1, 29, 19, 19) 或 (batch, ...)
        global_vec:    (1, 33) 或 (batch, ...)
        deterministic: True 时 argmax，False 时按概率采样

    Returns:
        (strategy, type_idx, sub_type, position) 四个整数
    """
    features = self.encode(board, global_vec)   # (1, hidden_dim)
    logits = self.action_head(features)         # (1, 27)

    strategy, type_idx, sub_type, position = self.action_head.decode(logits)

    return (
        strategy.item(),
        type_idx.item(),
        sub_type.item(),
        position.item(),
    )
```

**接口变更说明**：

| 项目 | 改造前 | 改造后 |
|------|--------|--------|
| 返回值 | `(action_id, log_prob, value, type_probs)` | `(strategy, type_idx, sub_type, position)` |
| 输入参数 | `board, global, action_mask, deterministic` | `board, global, deterministic` |
| action_mask | 作为输入过滤非法动作 | 不再需要（解码后通过 `can_apply_operation` 检查） |
| log_prob/value | 返回给 PPO 训练 | 不再需要 |
| 采样/确定性 | 支持 | **始终确定性**（保持与 design doc 一致） |

---

## 5. `env/action_mask.py` — 动作掩码和操作构造改造

### 5.1 核心改造：从 action_id 到四元组

```python
class ActionMaskHandler:
    """动作解码处理器 - 将 (strategy, type_idx, sub_type, position) 转换为 Operation。

    不再生成 119 维扁平掩码。合法性检查通过构造 Operation 后调用
    state.can_apply_operation() 完成。
    """
```

### 5.2 新增核心方法

```python
def construct_operation(
    self,
    strategy: int,
    type_idx: int,
    sub_type: int,
    position: int,
    state,
    player: int,
):
    """将 (s, t, st, p) 四元组解码为 Operation

    解码流程：
    1. 查 ACTION_DECODE_TABLE[strategy][type_idx] 获取 action_name
    2. 根据 action_name 将 position 映射为具体坐标或科技类型
    3. 构造 Operation 对象
    4. 调用 can_apply_operation 检查合法性

    Args:
        strategy:  {0, 1, 2}
        type_idx:  {0..5}
        sub_type:  {0..9}（仅 UPGRADE_TOWER 使用）
        position:  {0..7}
        state:     游戏状态
        player:    玩家 ID

    Returns:
        Operation 对象，或 None（非法组合 → NOOP）
    """
    from ..utils.action_constants import (
        ACTION_DECODE_TABLE, _PLAYER_TOWER_POS, _WEAPON_POS_REDUNDANCY,
        _TECH_POS_MAP, SUPER_WEAPON_POSITIONS, SuperWeaponType,
    )

    action_meta = ACTION_DECODE_TABLE[strategy].get(type_idx)
    if action_meta is None:
        return None  # → NOOP

    action_name, flags = action_meta

    # === 塔操作：BUILD / UPGRADE / DOWNGRADE ===
    if action_name in {"build_tower", "upgrade_tower", "downgrade_tower"}:
        pos = _PLAYER_TOWER_POS[player][position]
        x, y = pos

        if action_name == "build_tower":
            tower_id = self._find_tower_id_at(state, player, x, y)
            if tower_id is not None:
                return None  # 该格已有塔
            # 原地建塔时塔类型由状态决定（自动选择当前可建类型）
            op = Operation(OperationType.BUILD_TOWER, x, y)
            return op if self._is_valid_operation(state, player, op) else None

        elif action_name == "upgrade_tower":
            tower_id = self._find_tower_id_at(state, player, x, y)
            if tower_id is None:
                return None
            tower = state.tower_by_id(tower_id)
            if tower is None:
                return None
            # sub_type 解析：依赖当前塔类型
            upgrade_target = self._resolve_upgrade_path(tower.tower_type, sub_type)
            if upgrade_target is None:
                return None
            op = Operation(OperationType.UPGRADE_TOWER, tower_id, upgrade_target)
            return op if self._is_valid_operation(state, player, op) else None

        elif action_name == "downgrade_tower":
            tower_id = self._find_tower_id_at(state, player, x, y)
            if tower_id is None:
                return None
            op = Operation(OperationType.DOWNGRADE_TOWER, tower_id)
            return op if self._is_valid_operation(state, player, op) else None

    # === 超级武器：LIGHTNING_STORM / EMP_BLASTER / DEFLECTOR / EVASION ===
    elif action_name in {"lightning_storm", "emp_blaster", "deflector", "evasion"}:
        weapon_map = {
            "lightning_storm": (SuperWeaponType.LIGHTNING_STORM, OperationType.USE_LIGHTNING_STORM),
            "emp_blaster":     (SuperWeaponType.EMP_BLASTER,     OperationType.USE_EMP_BLASTER),
            "deflector":       (SuperWeaponType.DEFLECTOR,       OperationType.USE_DEFLECTOR),
            "evasion":         (SuperWeaponType.EMERGENCY_EVASION, OperationType.USE_EMERGENCY_EVASION),
        }
        weapon_type, op_type = weapon_map[action_name]
        weapon_positions = SUPER_WEAPON_POSITIONS[weapon_type]
        mapped_idx = _WEAPON_POS_REDUNDANCY[position]
        pos = weapon_positions[mapped_idx]
        op = Operation(op_type, pos[0], pos[1])
        return op if self._is_valid_operation(state, player, op) else None

    # === 科技升级 ===
    elif action_name == "tech_upgrade":
        tech_type = _TECH_POS_MAP[position]
        if tech_type == "generation_speed":
            op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
        else:
            op = Operation(OperationType.UPGRADE_GENERATED_ANT)
        return op if self._is_valid_operation(state, player, op) else None

    return None
```

### 5.3 新增升级路径解析方法

```python
def _resolve_upgrade_path(self, current_tower_type: int, sub_type: int) -> Optional[int]:
    """根据当前塔类型和 sub_type 确定升级目标 TowerType

    sub_type 编码（仅当 action=UPGRADE_TOWER 时有意义）：
        [0:3]  main_tree   → HEAVY/QUICK/MORTAR 一级分支
        [3:6]  sub_variant → variant A/B/C       二级子类
        [6:10] producer    → PRODUCER/FAST/SIEGE/MEDIC 生产系

    升级逻辑：
        - 目标塔为 BASIC (0) → main_tree / producer 决定一级升级方向
        - 目标塔为 combat_type → main_tree 决定上哪个枝，sub_variant 决定二级子类
        - 目标塔为 PRODUCER → producer 决定二级子类
    """
    from SDK.utils.constants import TOWER_UPGRADE_TREE

    targets = TOWER_UPGRADE_TREE.get(current_tower_type, ())
    if len(targets) == 0:
        return None

    # BASIC → 一级升级：sub_type[0:3] 选主方向
    if current_tower_type == 0:  # BASIC
        idx = sub_type % len(targets)  # 取模防止越界
        return int(targets[idx])

    # 非 BASIC → 二级升级：需要联合解析 main_tree + sub_variant / producer
    main_tree_idx = sub_type // 7  # 粗略划分，实际需要根据 TOWER_UPGRADE_TREE 精确映射
    # 精确逻辑参见 6.3 节的完整升级路径解析
    ...
```

### 5.4 删除/废弃方法

```python
# 删除以下方法
get_action_mask(state, player)               # 不再需要 119-d 扁平 mask
_action_id_to_op(action_id, state, player)   # 改用 construct_operation
flat_mask_to_type_mask(flat_mask)            # 不再需要
flat_mask_to_target_mask(flat_mask, type_id) # 不再需要
```

**注意**：如果外部代码仍引用 `get_action_mask`，可以保留一个简化版本返回全 1 数组（长度为 27），但强烈建议直接移除所有外部引用。

### 5.5 保留的辅助方法

```python
# 以下方法保持不动
_find_tower_id_at(state, player, x, y)  # 查找坐标上的己方塔 ID
_is_valid_operation(state, player, op)   # 操作合法性检查
```

---

## 6. `battle/ga_war_agent.py` — GA 对战 Agent 改造

### 6.1 改造要点

| 项目 | 改造前 | 改造后 |
|------|--------|--------|
| `_act()` 返回值 | `int` (action_id) | `Tuple[int, int, int, int]` (strategy, type_idx, sub_type, position) |
| `choose_operations()` | `_action_id_to_op(action_id)` | `construct_operation(s, t, st, p)` |
| `get_action` 参数 | `board, global, action_mask, deterministic` | `board, global, deterministic` |
| `get_action` 返回值 | `(action_id, log_prob, value, type_probs)` | `(strategy, type_idx, sub_type, position)` |
| 非法动作处理 | 记录日志返回 [] | 同前（记录日志返回 []） |

### 6.2 修改后代码

```python
class GAWarAgent(AntWarAgent):
    """GA 个体驱动的对战 Agent — 语义分段动作版本。

    与改造前的关键区别：
    - 网络直接输出 (strategy, type_idx, sub_type, position) 四元组
    - 不再使用 119-d 扁平 action_id 和 TYPE_CONFIG
    - 通过 ACTION_DECODE_TABLE + 多表位置映射解码为 Operation
    - 不再需要 action_mask 输入
    """

    def __init__(self, player_id: int, network: AntWarPolicyValueNetwork, device: torch.device):
        super().__init__(player_id, name="GA")
        self._network = network
        self._device = device
        self._observation_encoder = ObservationEncoder()
        self._action_mask_handler = ActionMaskHandler()
        self._illegal_count = 0

    def choose_operations(self, state) -> List[Any]:
        """根据当前状态选择操作（确定性策略）。

        流程：
        1. 编码观测（注意：不再包含 action_mask）
        2. 网络推理 → (strategy, type_idx, sub_type, position)
        3. construct_operation() → Operation
        """
        obs = self._observation_encoder.encode(state, self.player_id)
        with np.errstate(invalid="ignore"):
            strategy, type_idx, sub_type, position = self._act(obs)

        # 使用新的四元组解码接口
        op = self._action_mask_handler.construct_operation(
            strategy, type_idx, sub_type, position, state, self.player_id
        )
        if op is not None:
            return [op]

        # 非法动作 → NOOP 惩罚
        self._illegal_count += 1
        logger.warning(
            f"[GAWarAgent] Illegal action "
            f"(strategy={strategy}, type_idx={type_idx}, "
            f"sub_type={sub_type}, position={position}) "
            f"for player={self.player_id}, NOOP penalty applied"
        )
        return []

    def _act(self, observation) -> Tuple[int, int, int, int]:
        """网络推理获取动作四元组（确定性）"""
        tensors = observation_to_tensors(observation, self._device)
        with torch.no_grad():
            strategy, type_idx, sub_type, position = self._network.get_action(
                tensors["board"],
                tensors["global"],
                deterministic=True,
            )
        return strategy, type_idx, sub_type, position

    @property
    def illegal_action_count(self) -> int:
        return self._illegal_count
```

### 6.3 `observation_to_tensors` 适配

```python
# utils/obs_utils.py 保持不变，但注意 observation 中不再需要 action_mask
# 如果 ObservationEncoder.encode() 仍返回 action_mask，需要确认是否删除
```

---

## 7. `env/observation.py` — 观测编码器适配

### 7.1 改动点

```python
class ObservationEncoder:
    """观测编码器 - 将游戏状态编码为神经网络可处理的张量格式。

    输出字典：
        board:       numpy.ndarray, shape (29, 19, 19)  棋盘特征图
        global_vec:  numpy.ndarray, shape (33,)           全局特征向量
        # 注：不再输出 action_mask，网络不再需要外部 mask 输入
    """
```

### 7.2 `encode()` 方法删除 action_mask

```python
def encode(self, state, player: int) -> Dict[str, np.ndarray]:
    board = self._encode_board(state, player)
    global_vec = self._encode_global(state, player)
    # action_mask 不再需要 — 合法性在 construct_operation 时由 state.can_apply_operation() 检查

    return {
        "board": board,
        "global": global_vec,
        # "action_mask" 已移除
    }
```

---

## 8. `trainer/neural_agent.py` — NeuralAgent 改造

### 8.1 改动点

NeuralAgent 原服务于 PPO 训练流程（`act_record` 返回 log_prob+value）。改造后：

- 删除 `get_value()` 方法（依赖 value encoder）
- 删除 `act_record()` 方法（回归测试中不再需要）
- 简化 `_forward()` 方法
- 适配新 `get_action()` 接口

```python
class NeuralAgent(RecordingAgent):
    """基于神经网络的 Agent（GA 版本）。

    简化版：不再提供 log_prob/value 记录功能。
    get_value 已被移除（value encoder 已删除）。
    """

    def __init__(self, policy: AntWarPolicyValueNetwork, device: torch.device):
        self._policy = policy
        self._device = device

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """根据观测返回动作（返回扁平 action_id 以兼容现有接口）。

        注意：这里的 action_id 为 0（NOOP）或 1（非 NOOP），
        完整解码需使用 GAWarAgent 中的 construct_operation。
        """
        # 实际使用时，建议直接使用 GAWarAgent，不走 NeuralAgent 路径
        raise NotImplementedError(
            "NeuralAgent.act() 已废弃，请直接使用 GAWarAgent"
        )
```

---

## 9. `config/config_parser.py` — 配置改造

### 9.1 删除配置项

```python
# 默认配置中删除
"networks": {"hidden_dim": 256, "enable_auxiliary": True}
# 改为
"networks": {"hidden_dim": 256}

# 配置类型验证中删除
"network.enable_auxiliary": bool
```

### 9.2 连锁影响

| 受影响的文件 | 改动 |
|-------------|------|
| `config/config_parser.py` | 删除 `enable_auxiliary` 配置 |
| `trainer/genetic_evolution_trainer.py` | 删除所有 `enable_auxiliary` 引用 |
| `evolution/population_manager.py` | 删除 `enable_auxiliary` 参数和引用 |
| `battle/opponent_agent.py` | 删除 `enable_auxiliary` 参数 |
| `battle/battle_config.py` | 删除 `enable_auxiliary` 字段 |
| `selection/round_robin.py` | 删除 `enable_auxiliary` 引用 |
| `selection/novice_pruner.py` | 删除 `enable_auxiliary` 引用 |
| `selection/touchstone_selector.py` | 删除 `enable_auxiliary` 引用 |
| `evaluation/baseline_evaluator.py` | 删除 `enable_auxiliary` 引用 |
| `trainer/checkpoint_manager.py` | 删除 `enable_auxiliary` 引用 |

**各文件的改动方式**：在所有创建 `AntWarPolicyValueNetwork` 实例的地方，删除 `enable_auxiliary=...` 参数（因为构造函数已移除该参数）。

---

## 10. `genome/genome.py` — 基因组适配

### 10.1 官方声明：无改动

```python
# genome.py 无需任何修改
#
# 原因：
# - extract_genome() 通过 network.named_parameters() 自动遍历所有参数
# - genome_to_state_dict() / load_genome_to_network() 按参数名匹配加载
# - 删除 value encoder / aux heads 后，旧基因组与新网络参数名不匹配，
#   但这是预期的行为——旧基因组不再兼容，新基因组自然适配新网络结构。
#
# 加载旧检查点的行为：
# - strict=False：匹配到的参数（policy encoder）加载，不匹配的参数（value encoder, aux heads）跳过
# - strict=True：会报 missing_keys / unexpected_keys 错误
# - 建议设置 strict=False 或升级检查点版本号
```

---

## 11. 调用链改动总结

### 11.1 正向调用链（GA WarAgent 执行路径）

```
[Game State]
    │
    ▼
ObservationEncoder.encode()
    │  board: (29,19,19) + global: (33,)
    │  (action_mask 已移除)
    ▼
GAWarAgent._act()
    │  board_tensor, global_tensor
    ▼
AntWarPolicyValueNetwork.get_action()
    │  encode() → ActionHead() → segment argmax
    ▼
(strategy, type_idx, sub_type, position)  ← 新接口
    │
    ▼
ActionMaskHandler.construct_operation()
    │  ACTION_DECODE_TABLE → 多表位置映射 → Operation
    ▼
Operation → state.can_apply_operation()
    │
    ├─ 合法 → 返回 [op]
    └─ 非法 → NOOP（返回 []）
```

### 11.2 旧代码调用点迁移指南

| 旧调用 | 新调用 | 说明 |
|--------|--------|------|
| `network.get_action(board, global, mask, det)` | `network.get_action(board, global, det)` | 去掉 mask 参数，返回四元组 |
| `mask_handler.get_action_mask(state, player)` | `mask_handler.construct_operation(s,t,st,p,state,player)` | 即时解码替代预计算掩码 |
| `mask_handler._action_id_to_op(aid, state, player)` | `mask_handler.construct_operation(s,t,st,p,state,player)` | 四元组替代 action_id |
| `obs.encode(state, player)["action_mask"]` | 已删除 | 不再需要 |
| `neural_agent.get_value(obs)` | 已删除 | value encoder 已移除 |
| `network.evaluate_actions(...)` | 已删除 | PPO 训练已移除 |
| `network.compute_auxiliary_loss(...)` | 已删除 | 辅助任务已移除 |

---

## 12. 升级路径与兼容性

### 12.1 检查点兼容性

| 场景 | 兼容性 | 建议 |
|------|--------|------|
| 旧检查点加载到新网络（strict=True） | **不兼容** | 旧检查点含 value encoder/aux heads 参数，新网络没有 |
| 旧检查点加载到新网络（strict=False） | **部分兼容** | policy encoder 参数可加载，其余被丢弃 |
| 新检查点加载到旧网络 | **不兼容** | 旧网络期望 119-d 输出头，新网络只有 27-d |

**建议**：
1. 新检查点使用版本号 `v11` 或 `v2` 标识，不与旧检查点混用
2. 在 checkpoint_manager 中增加版本校验逻辑
3. 升级后首次训练从随机初始化开始（不加载旧检查点）

### 12.2 配置兼容性

新配置文件删除了 `enable_auxiliary`，如果旧配置文件仍包含该字段：

```python
# 建议在 config_parser.py 中增加警告
if "enable_auxiliary" in config.get("network", {}):
    logger.warning("Config key 'network.enable_auxiliary' is deprecated, ignoring")
```

---

## 13. 验证与测试

### 13.1 单元测试清单

| 测试项 | 测试内容 | 关键断言 |
|--------|---------|----------|
| ActionHead 输出维度 | forward → logits.shape | `== (batch, 27)` |
| ActionHead.decode 边界 | 输入全 0 logits | 各段 argmax == 0 |
| 映射表完整性 | 所有 (s, t) 组合不抛 KeyError | `len(table[s]) == 6` |
| 塔位映射 | position 0-7 映射到己方高地格 | 坐标在 HIGHLAND_CELLS 中 |
| 超武冗余映射 | pos 5-7 映射到 pos 2-4 | `_WEAPON_POS_REDUNDANCY` 验证 |
| construct_operation | 合法组合返回 Operation | `isinstance(op, Operation)` |
| construct_operation | 非法组合返回 None | 如 strategy=1 任意 type_idx |

### 13.2 集成测试

- 网络输出维度：`network.get_action(board, global)` → 返回 4 个 int
- 端到端 GA 对战：GAWarAgent 可在 1v1 对战中完整执行 `choose_operations`

---

## 14. 实施建议

### 14.1 实施顺序

```
Phase 1: 数据层
  └── utils/action_constants.py（先定义新常量，暂不删除旧常量）

Phase 2: 网络层
  ├── network/heads.py（新增 ActionHead，保留旧类用于回归）
  ├── network/ant_war_policy_value_network.py（重写，保证编译通过）
  └── genome/genome.py（验证无改动）

Phase 3: 解码层
  ├── env/action_mask.py（新增 construct_operation，验证解码正确性）
  └── env/observation.py（移除 action_mask）

Phase 4: Agent 层
  ├── battle/ga_war_agent.py（适配新接口）
  └── trainer/neural_agent.py（清理 PPO 遗留代码）

Phase 5: 配置清理
  ├── config/config_parser.py
  ├── trainer/genetic_evolution_trainer.py
  ├── evolution/population_manager.py
  └── 其他 7 个引用 enable_auxiliary 的文件

Phase 6: 清理删除
  └── 删除旧常量、旧类、旧方法（确保无外部引用）
```

### 14.2 风险提示

1. **检查点不兼容**：所有现存的 `.pth` 检查点无法加载到新网络，需重新训练
2. **日志监控**：GAWarAgent 的 `illegal_count` 日志中出现新的 warning 格式，需更新日志解析脚本
3. **外部依赖**：如果 `battle_simulator.py` 或其他文件直接引用 `_action_id_to_op`，需同步更新
4. **观测维度变化**：`observation_to_tensors` 不再需要 `action_mask` 键，需确认所有调用方已适配
