# 动作空间语义改造方案

## 背景

当前 ppo_v10 的动作空间为 119 维扁平 one-hot 结构（9 类型 × N 靶位），网络输出为分层 one-hot（先选类型，再选靶位）。存在以下问题：

1. **语义混合**：UPGRADE_TOWER 的 64 个靶位混合了位置(16)和方向(4)两个正交语义维度
2. **合法率极低**：动作合法率高度不平衡——武器 ~80%，建塔/升级 ~10%，导致 97% 的操作被转为 NOOP
3. **GA 信号淹没**：97% NOOP 噪声中，胜负信号无法通过选择压力有效传导到 119 维的每个输出分量
4. **架构冗余**：遗留大量 PPO-only 组件（Value 编码器、Auxiliary Heads 等），GA 不使用

## 改造目标

1. 网络输出从 one-hot 改为**语义分段向量**，每段代表一个独立的语义维度
2. 删除所有与策略无关的 PPO 遗留结构
3. 动作合法率通过缩小靶空间自然提升，不依赖外部 mask

## 网络架构改造

### 删除结构

| 结构 | 原因 |
|------|------|
| Value 编码器（value_cnn / value_mlp / value_proj / value_layer_norm） | 完整克隆的独立编码器，PPO 估值用 |
| ValueHead | 价值头，PPO 用 |
| Auxiliary Heads（TowerDamage / GoldIncome / BaseDamage × 2 套） | 辅助预测任务，PPO 表征学习用 |
| `evaluate_actions` 方法 | 完整 PPO 训练方法（log_prob + entropy + value + aux_loss） |
| `get_value_only` 方法 | PPO 估值入口 |
| `forward` 方法 | PPO 训练入口（返回结构化 type/target logits） |
| `log_prob` 计算（get_action 内） | GA 不调用 |
| `logit_noise_std` | GA 确定性子代，不需要探索噪声 |

### 保留结构

| 结构 | 原因 |
|------|------|
| Policy 编码器（policy_cnn / policy_mlp / policy_proj / policy_layer_norm） | 策略特征提取核心 |

### 新输出头

单一 Linear 层，将编码器输出 256 维特征投影为 27 维向量：

```python
self.output_head = nn.Linear(hidden_dim, 27)  # 27 = 3 + 6 + 10 + 8
```

## 输出向量语义分段

### 分段定义

```
总长度 27 维

[0:3]     strategy  (3)  → [defensive_idx, noop_idx, offensive_idx]
[3:9]     type      (6)  → [type_idx_0 ~ type_idx_5]
[9:19]    sub_type  (10) → [main_tree(3) + sub_variant(3) + producer(4)]
[19:27]   position  (8)  → [pos_0 ~ pos_7]
```

sub_type 段进一步分解为三个子段，仅对 UPGRADE_TOWER 有意义：

```
sub_type = [9:12]  main_tree   (3) → [HEAVY, QUICK, MORTAR]      一级分支
           [12:15] sub_variant (3) → [variant_A, variant_B, variant_C]  二級子类
           [15:19] producer    (4) → [PRODUCER, PRODUCER_FAST,       生产系
                                      PRODUCER_SIEGE, PRODUCER_MEDIC]
```

**核心原则：type_idx 本身没有固有语义。同一 type_idx 在不同 strategy 下代表不同动作。** 只有 `(strategy, type_idx)` 联合起来才定义了一个具体的操作种类。

### 解码方式

分成两步：

**第一步**：每段独立 argmax，得到原始索引向量：

```
strategy  = argmax(logits[0:3])     ∈ {0, 1, 2}  [防御, 无操作, 进攻]
type_idx  = argmax(logits[3:9])     ∈ {0..5}
sub_type  = 见下方解码说明
position  = argmax(logits[19:27])   ∈ {0..7}
```

sub_type 解码（仅当 action = UPGRADE_TOWER 时使用）：

```
main_tree    = argmax(softmax(logits[9:12]))    ∈ {0, 1, 2}         [HEAVY, QUICK, MORTAR]
sub_variant  = argmax(softmax(logits[12:15]))   ∈ {0, 1, 2}         [tree 对应的三个子类]
producer     = argmax(softmax(logits[15:19]))   ∈ {0, 1, 2, 3}      [PRODUCER, FAST, SIEGE, MEDIC]

# 升级方向选择逻辑：
# - 若 target 塔为 BASIC → main_tree / producer 决定一级升级方向
# - 若 target 塔为 combat_type (HEAVY/QUICK/MORTAR) → main_tree 决定上哪个枝,
#   sub_variant 决定该枝的二級子类
# - 若 target 塔为 PRODUCER → producer 决定二級子类
```

**第二步**：用 `(strategy, type_idx)` 查映射表确定具体动作。

### (Strategy, Type) 联合映射表

type_idx 本身无语义，以下映射表唯一定义每个 `(strategy, type_idx)` 组合代表什么操作：

| type_idx | strategy=防御 | strategy=无操作 | strategy=进攻 |
|:--------:|:-------------:|:--------------:|:-------------:|
| **0** | BUILD_TOWER | NOOP | DOWNGRADE_TOWER |
| **1** | UPGRADE_TOWER | NOOP | DOWNGRADE_TOWER (冗余) |
| **2** | LIGHTNING_STORM | NOOP | DEFLECTOR |
| **3** | LIGHTNING_STORM (冗余) | NOOP | EVASION |
| **4** | LIGHTNING_STORM (冗余) | NOOP | EMP_BLASTER |
| **5** | (→NOOP) | NOOP | TECH_UPGRADE |

**映射设计原则**：

### 进攻/防守成对反向

每个 type_idx 下的防御方动作和进攻方动作互为"反向操作"。type_idx 共享，但由 strategy 决定是"加强己方"还是"削弱己方/敌方"：

| type_idx | 防守 | 进攻 | 说明 |
|:--------:|:----:|:----:|:----:|
| 0 | BUILD_TOWER (建己方塔) | DOWNGRADE_TOWER (降级己方塔) | 非超武·塔↔塔 |
| 1 | UPGRADE_TOWER (升己方塔) | DOWNGRADE_TOWER (降级己方塔·冗余) | 非超武·塔↔塔 |
| 2 | LIGHTNING_STORM (杀伤敌方蚂蚁) | DEFLECTOR (保护己方蚂蚁) | **超武↔超武**·蚂蚁 |
| 3 | LIGHTNING_STORM (杀伤敌方蚂蚁·冗余) | EVASION (己方蚂蚁闪避) | **超武↔超武**·蚂蚁 |
| 4 | LIGHTNING_STORM (杀伤敌方蚂蚁·冗余) | EMP_BLASTER (压制敌方塔) | **超武↔超武**·混合目标 |
| 5 | (→NOOP) | TECH_UPGRADE (加强己方蚂蚁) | 非超武·进攻方独有 |

### type_idx 无固有语义

索引值在不同 strategy 下代表完全不同的操作。例如 `type_idx=0` 在防御下是 BUILD_TOWER，在进攻下是 DOWNGRADE_TOWER。

### 冗余映射

- 防御方：type_idx 2、3、4 都映射为 LIGHTNING_STORM，给网络更多命中杀伤敌蚁的机会
- 进攻方：type_idx 0 和 1 都映射为 DOWNGRADE_TOWER

这给了网络更多表达能力——同一个动作可以有多个索引，让 GA 更容易命中。

### 非对称映射

进攻方比防御方多出 2 个动作（EMP_BLASTER 和 TECH_UPGRADE），因此 type_idx 4-5 在防御方下退化为 NOOP。这是设计允许的——type 空间不要求攻防对称。

### strategy=无操作时强制 NOOP

无论 type_idx 为何值，strategy=无操作时均→NOOP。

### sub_type 和 position 的联合语义

sub_type 和 position 的语义由 `(strategy, type_idx)→action` 联合确定：

| 动作 | sub_type 含义 | position 含义 |
|------|:------------:|:-------------:|
| BUILD_TOWER | 忽略 | 玩家塔位表索引（8 个固定塔位中选 1，见下） |
| UPGRADE_TOWER | 依赖当前塔类型:<br/>BASIC→main_tree/producer 选一级方向<br/>combat→sub_variant 选二級子类<br/>PRODUCER→producer 选二級子类 | 玩家塔位表索引 |
| DOWNGRADE_TOWER | 忽略 | 玩家塔位表索引 |
| LIGHTNING_STORM | 忽略 | 超武释放位置（5 个预设位含冗余，见下） |
| EMP_BLASTER | 忽略 | 同上 |
| DEFLECTOR | 忽略 | 同上 |
| EVASION | 忽略 | 同上 |
| TECH_UPGRADE | 忽略 | 科技类型（0-3→generation_speed, 4-7→generated_ant） |
| NOOP | 忽略 | 忽略 |

## 多表位置映射方案

position 段共 8 维（值 0-7），不同动作类型查不同映射表解码。**核心设计**：position 值不直接对应固定坐标，而是根据 `(动作类型, 玩家)` 查表确定实际坐标。

### 塔操作映射表（BUILD/UPGRADE/DOWNGRADE_TOWER）

每个 position 值直接映射到当前玩家的一个固定塔位，双方镜像对称：

| position | Player 0 坐标 | Player 1 坐标 | 战略意义 |
|:--------:|:------------:|:------------:|----------|
| 0 | (5, 9) | (13, 9) | **中线核心位** |
| 1 | (4, 9) | (14, 9) | 中线辅助位 |
| 2 | (7, 8) | (10, 7) | **中央战场** |
| 3 | (6, 7) | (12, 6) | 中央侧翼 |
| 4 | (8, 7) | (11, 5) | 中央侧翼 |
| 5 | (5, 6) | (12, 9) | 中轴前突 |
| 6 | (6, 14) | (11, 14) | 下侧翼 |
| 7 | (4, 2) | (13, 15) | 角落防守 |

**效果**：position 0-7 始终指向当前玩家的合法高地格。仅有该格已被建塔或被 EMP 覆盖时才会导致 NOOP，编码空间本身无浪费。

### 超级武器映射表（LIGHTNING_STORM / EMP_BLASTER / DEFLECTOR / EVASION）

8 个 position 值通过冗余映射覆盖 5 个释放位置，确保全员合法：

| position | 映射到预设位 | 说明 |
|:--------:|:----------:|------|
| 0 | 位置[0] | 直接映射 |
| 1 | 位置[1] | 直接映射 |
| 2 | 位置[2] | 直接映射（中线/核心） |
| 3 | 位置[3] | 直接映射 |
| 4 | 位置[4] | 直接映射 |
| 5 | 位置[2] | **冗余** → 中线/核心位 |
| 6 | 位置[3] | **冗余** |
| 7 | 位置[4] | **冗余** |

> 预设位置数组与现有 `SUPER_WEAPON_POSITIONS` 一致：闪电风暴/EMP 为中轴线 [(7,9),(8,9),(9,9),(10,9),(11,9)]，偏射盾/回避为己方半场 [(4,9),(5,9),(6,9),(12,9),(13,9)]。

### 科技升级映射表（TECH_UPGRADE）

| position | 映射到科技类型 | 冗余数 |
|:--------:|:------------:|:------:|
| 0-3 | generation_speed | 4 个冗余入口 |
| 4-7 | generated_ant | 4 个冗余入口 |

### 冗余设计收益

| 动作大类 | 编码利用率（8 值中） | vs 原 4 维方案 |
|---------|:-----------------:|:------------:|
| 塔操作 | **8/8 = 100%** | 4 维 region 分组约 70% |
| 超武 | **8/8 = 100%**（含冗余） | 4/5 = 80% （值 4 不可达） |
| 科技升级 | **8/8 = 100%** | 2/4 = 50% |

## 动作解码流程

### 合法性校验

```python
def is_legal_combination(strategy, type_id, sub_type, position, state, player):
    """检查 (s, t, st, p) 组合是否合法"""
    op = construct_operation(strategy, type_id, sub_type, position, state, player)
    return op is not None and state.can_apply_operation(player, op)
```

非法组合 → NOOP（与当前惩罚机制一致）。

### 动作构造（伪代码）

动作解码由一张**映射表**驱动，而不是硬编码的 if-else 分支：

```python
# ────── 多表位置映射常量 ──────

# 塔操作：position 0-7 → 当前玩家的固定塔位
_PLAYER_TOWER_POS = {
    0: [  # Player 0 的 8 个塔位（按战略意义排序）
        (5, 9), (4, 9), (7, 8), (6, 7),
        (8, 7), (5, 6), (6, 14), (4, 2),
    ],
    1: [  # Player 1 的 8 个塔位（镜像对称）
        (13, 9), (14, 9), (10, 7), (12, 6),
        (11, 5), (12, 9), (11, 14), (13, 15),
    ],
}

# 超级武器：position 0-7 → 5 个预设位的冗余映射
_WEAPON_POS_REDUNDANCY = [0, 1, 2, 3, 4, 2, 3, 4]
#                               ↑ pos 5-7 冗余到 2-4（中线/核心位）

# 科技升级：position 0-7 → 2 种科技类型的冗余映射
_TECH_POS_MAP = {
    0: "generation_speed",  # position 0-3
    1: "generation_speed",  # 冗余
    2: "generation_speed",  # 冗余
    3: "generation_speed",  # 冗余
    4: "generated_ant",     # position 4-7
    5: "generated_ant",     # 冗余
    6: "generated_ant",     # 冗余
    7: "generated_ant",     # 冗余
}


# ────── (Strategy, Type) 联合映射表 ──────
# 每个 type_idx 下，防御 vs 进攻互为反向操作
ACTION_DECODE_TABLE = [
    # strategy=0 (防御) — 非超武: idx0-1, 超武: idx2-4
    {
        0: ("build_tower",          {}),
        1: ("upgrade_tower",        {"use_sub_type": True}),
        2: ("lightning_storm",      {}),
        3: ("lightning_storm",      {}),  # 冗余
        4: ("lightning_storm",      {}),  # 冗余
        5: None,                    # (→NOOP，进攻独占 slot)
    },
    # strategy=1 (无操作) — 全部 NOOP
    {i: None for i in range(6)},
    # strategy=2 (进攻) — 非超武: idx0-1,5  超武: idx2-4
    {
        0: ("downgrade_tower",      {}),
        1: ("downgrade_tower",      {}),  # 冗余
        2: ("deflector",            {}),
        3: ("evasion",              {}),
        4: ("emp_blaster",          {}),
        5: ("tech_upgrade",         {}),
    },
]


def construct_operation(strategy, type_idx, sub_type, position, state, player):
    """用映射表解析 (strategy, type_idx, position) → 具体操作"""
    
    action_meta = ACTION_DECODE_TABLE[strategy][type_idx]
    if action_meta is None:
        return None  # → NOOP
    
    action_name, flags = action_meta
    
    # 统一派发
    if action_name in {"build_tower", "upgrade_tower", "downgrade_tower"}:
        # 查玩家塔位表：position 0-7 始终映射到当前玩家的高地格
        pos = _PLAYER_TOWER_POS[player][position]
        
        if action_name == "build_tower":
            if state.tower_at(pos[0], pos[1]) is not None:
                return None  # 该格已被占
            tower_type = pick_defensive_tower_type(state, pos[0], pos[1])
            op = Operation(BUILD_TOWER, pos[0], pos[1], tower_type)
            return op if state.can_apply_operation(player, op) else None
        
        elif action_name == "upgrade_tower":
            tower_id = find_tower_id_at(state, player, pos[0], pos[1])
            if tower_id is None:
                return None
            upgrade = select_upgrade_path(tower.tower_type, sub_type)
            op = Operation(UPGRADE_TOWER, tower_id, upgrade)
            return op if state.can_apply_operation(player, op) else None
        
        elif action_name == "downgrade_tower":
            tower_id = find_tower_id_at(state, player, pos[0], pos[1])
            if tower_id is None:
                return None
            op = Operation(DOWNGRADE_TOWER, tower_id)
            return op if state.can_apply_operation(player, op) else None
    
    elif action_name in {"lightning_storm", "emp_blaster", "deflector", "evasion"}:
        # 查超武冗余映射表：position 0-7 → 5 个预设位
        weapon_positions = SUPER_WEAPON_POSITIONS[action_name]
        mapped_pos = _WEAPON_POS_REDUNDANCY[position]  # 0-7 → 0-4
        pos = weapon_positions[mapped_pos]
        op = Operation(ACTION_NAME_TO_OP[action_name], pos[0], pos[1])
        return op if state.can_apply_operation(player, op) else None
    
    elif action_name == "tech_upgrade":
        # 查科技冗余映射表：position 0-3→speed, 4-7→ant
        tech_type = _TECH_POS_MAP[position]
        if tech_type == "generation_speed":
            op = Operation(UPGRADE_GENERATION_SPEED)
        else:
            op = Operation(UPGRADE_GENERATED_ANT)
        return op if state.can_apply_operation(player, op) else None
    
    return None
```

## 改动文件清单

| 文件 | 改动 |
|------|------|
| [ant_war_policy_value_network.py](../../src/ppo_ga/network/ant_war_policy_value_network.py) | 删除 Value/aux/evaluate/log_prob/logit_noise；新增 27 维向量输出头；重写 get_action |
| [heads.py](../../src/ppo_ga/network/heads.py) | 删除 StructuredActionHead / ValueHead / AuxHeads；不需要替换（逻辑移入主网络 + agent） |
| [action_constants.py](../../src/ppo_ga/utils/action_constants.py) | `ACTION_DIM=27`；新增 `_PLAYER_TOWER_POS`、`_WEAPON_POS_REDUNDANCY`、`_TECH_POS_MAP`；删除 `REGION_GROUPS`；简化 `TYPE_CONFIG` |
| [action_mask.py](../../src/ppo_ga/env/action_mask.py) | 重写 `_action_id_to_op` 为四元组解码器，集成多表位置映射 |
| [ga_war_agent.py](../../src/ppo_ga/battle/ga_war_agent.py) | 适配新的 get_action 接口 |
| [genome.py](../../src/ppo_ga/genome/genome.py) | 无改动（自动适配新参数） |

## 预期效果

| 指标 | 改造前（119 维） | 原设计方案（4 维 position） | 本方案（8 维 position） |
|------|:--------------:|:-------------------------:|:---------------------:|
| 动作空间维度 | 119 | 23 | **27** |
| 语义维度数 | 1（组合 one-hot） | 4 | **4**（strategy + type + sub_type + **8维position**） |
| BUILD_TOWER 合法率 | ~10-20% | ~70% | **~100%**（扣除占用） |
| UPGRADE_TOWER 合法率 | ~8% | ~50% | **~100%**（取决于塔存在） |
| 超武编码利用率 | ~80%（单 type 5/5） | 4/5=80%（第5位不可达） | **8/8=100%**（含冗余） |
| 科技升级编码利用率 | 2/2=100% | 2/4=50% | **8/8=100%**（含冗余） |
| 整体 NOOP 率 | ~97% | ~60-70% | **~50-60%** |
| 网络参数量 | ~400K | ~100K | ~105K（4维增量） |
| GA 有效信号密度 | 3% | 30-40% | **40-50%** |
| 语义可解释性 | 差（flat index） | 中（表格驱动的联合解码） | 高（player-aware 多表映射） |
| 位置编码方式 | 一维 flat index | region 分组（间接选址） | **多表直映射（player-specific）** |

## 后续优化方向

1. **position embedding**：当前为硬编码查表。未来若需精确到坐标级，可引入可学习的 position embedding 层，将各 action type 下的 position 语义交由网络隐式学习
2. **sub_type 扩展**：当前 sub_type 仅用于 UPGRADE 方向，后续可为其他类型扩展含义
3. **strategy 扩展**：当前为 3 值（防御/无操作/进攻），可扩展为更多值
4. **动态塔位表**：如需利用全部 33 个高地格，可引入动态筛选机制，根据棋盘状态实时选取最优 N 个候选位置替换固定查表
