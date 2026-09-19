# AntWar 游戏规则说明文档

> 版本：v2.0（基于 SDK v1 代码事实重构）
> 来源：[constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/constants.py)、[engine.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/backend/engine.py)、[model.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/backend/model.py)
> 原则：每一条规则均可追溯至 SDK 源代码中对应的位置。

---

## 1. 游戏概要

| 属性 | 值 | 来源 |
|------|:---:|------|
| 玩家数量 | 2 | `PLAYER_COUNT = 2` |
| 地图尺寸 | 19 × 19（六边形网格） | `MAP_SIZE = 19` |
| 初始金币 | 各 50 | `INITIAL_COINS = 50` |
| 基地初始血量 | 各 50 | `BASE_HP = 50` |
| 回合上限 | 512 回合 | `MAX_ROUND = 512` |
| 动作空间维度 | 96 | `MAX_ACTIONS = 96` |

游戏为**完全信息对称**的回合制策略游戏。双方在每一轮中交替执行操作（玩家 0 先手，玩家 1 后手），之后引擎推进一个游戏回合。

---

## 2. 地图系统

### 2.1 网格与坐标

- 地图为 19×19 的六边形网格。
- 六边形坐标的 6 个方向通过 `OFFSET` 表定义，偏移量根据 `y % 2`（奇偶行）不同。
- 坐标表示：（`x`, `y`），其中 `0 ≤ x < 19`，`0 ≤ y < 19`。

### 2.2 地形类型

`MAP_PROPERTY` 是一个 19×19 的矩阵，定义了每个格子的类型：

| 枚举值 | 类型 | 含义 | 可否行走 |
|:---:|------|------|:---:|
| -1 | VOID | 不可达区域 | ❌ |
| 0 | PATH | 可通行道路 | ✅ |
| 1 | BARRIER | 障碍物 | ❌ |
| 2 | PLAYER0_HIGHLAND | 玩家 0 高地（可建塔） | ✅（不可通行，但塔可建于此） |
| 3 | PLAYER1_HIGHLAND | 玩家 1 高地（可建塔） | ✅（不可通行，但塔可建于此） |

### 2.3 基地坐标

| 玩家 | 坐标 | 来源 |
|:---:|:---:|------|
| 玩家 0 | `(2, 9)` | `PLAYER_BASES[0]` |
| 玩家 1 | `(16, 9)` | `PLAYER_BASES[1]` |

### 2.4 可建造位置

玩家只能在**自己的高地格**（`Terrain.PLAYER0_HIGHLAND` / `Terrain.PLAYER1_HIGHLAND`）上建造防御塔。高地格列表由 `HIGHLAND_CELLS` 定义。

建造优先级顺序由 `STRATEGIC_BUILD_ORDER` 定义——这是一个按优先级排序的高地坐标列表，并非强制顺序，仅作为 AI 建议参考。

---

## 3. 经济系统

### 3.1 金币来源

| 来源 | 数值 | 来源 |
|------|:---:|------|
| 初始金币 | 50 | `INITIAL_COINS = 50` |
| 基础收入 | 3 金币 / 每 2 回合 | `BASIC_INCOME = 3`，`BASIC_INCOME_INTERVAL = 2` |
| 蚂蚁击杀奖励 | 参见 §4.5 | `ANT_KILL_REWARD = (6, 10, 14)`，`COMBAT_ANT_KILL_REWARD = 18` |
| 蚂蚁到达敌方基地 | 10 金币 | `ANT_BREACH_REWARD = 10` |
| 防御塔拆除退款 | 费用的 90% ×（塔当前 HP/塔最大 HP） | `TOWER_DOWNGRADE_REFUND_RATIO = 0.9` |

### 3.2 金币支出

| 支出 | 公式 | 来源 |
|------|------|------|
| 建造防御塔 | `15 × 3^(n//2)`，`n` 为偶数：`×1`，`n` 为奇数：`×2` | `tower_build_cost_for_count()` |
| 塔升级到 L2 | 60 金币 | `LEVEL2_TOWER_UPGRADE_COST = 60` |
| 塔升级到 L3 | 200 金币 | `LEVEL3_TOWER_UPGRADE_COST = 200` |
| 基地科技升级 | L0→L1: 200 金币，L1→L2: 250 金币 | `BASE_UPGRADE_COST = (200, 250)` |
| 超级武器 | 参见 §6 | `SUPER_WEAPON_STATS` |

### 3.3 建造费用表

| 已建塔数 | 建造费用 | 计算过程 |
|:---:|:---:|------|
| 0 | 15 | 15 × 3^0 × 1 |
| 1 | 30 | 15 × 3^0 × 2 |
| 2 | 45 | 15 × 3^1 × 1 |
| 3 | 90 | 15 × 3^1 × 2 |
| 4 | 135 | 15 × 3^2 × 1 |
| 5 | 270 | 15 × 3^2 × 2 |
| 6 | 405 | 15 × 3^3 × 1 |
| 7 | 810 | 15 × 3^3 × 2 |
| 8 | 1215 | 15 × 3^4 × 1 |
| 9 | 2430 | 15 × 3^4 × 2 |

---

## 4. 蚂蚁系统

### 4.1 蚂蚁种类

| 种类 | 枚举值 | 说明 |
|------|:---:|------|
| WORKER（工蚁） | `AntKind.WORKER = 0` | 普通蚂蚁，会沿路径向敌方基地前进，途中可攻击塔 |
| COMBAT（战斗蚂蚁） | `AntKind.COMBAT = 1` | 专门攻击敌方塔，不攻基地；免疫老死；可自毁 |

### 4.2 蚂蚁属性

| 属性 | 工蚁 L0→L1→L2 | 战斗蚂蚁 | 来源 |
|------|:---:|:---:|------|
| 最大血量 | 20 → 25 → 25 | 固定 30 | `ANT_MAX_HP`，`COMBAT_ANT_HP = 30` |
| 塔攻击伤害 | 1 → 2 → 4 | 5 | `WORKER_TOWER_ATTACK_DAMAGE`，`COMBAT_TOWER_ATTACK_DAMAGE = 5` |
| 年龄上限 | 64 回合 | **无限**（不受年龄限制） | `ANT_AGE_LIMIT = 64`，`refresh_status()` 中对 COMBAT 跳过年龄检查 |
| 击杀金币 | 6 → 10 → 14 | 18 | `ANT_KILL_REWARD`，`COMBAT_ANT_KILL_REWARD` |
| 破基地金币 | 10 | N/A（不破基地） | `ANT_BREACH_REWARD = 10` |

### 4.3 蚂蚁状态

| 枚举值 | 含义 | 判定逻辑（`refresh_status()`） |
|:---:|------|------|
| ALIVE（0） | 存活 | `hp > 0` 且不满足其他状态条件 |
| SUCCESS（1） | 到达敌方基地 | `(x, y) == 敌方基地坐标` |
| FAIL（2） | 死亡 | `hp ≤ 0` |
| TOO_OLD（3） | 老死 | `age > 64` 且非战斗蚂蚁 |
| FROZEN（4） | 被冰冻 | `frozen == True` |

### 4.4 蚂蚁行为模式

蚂蚁有 5 种行为模式（`AntBehavior` 枚举），由引擎引擎自动控制（AI 无法直接设置蚂蚁个体行为）：

| 枚举值 | 含义 | 对移动的影响 |
|:---:|------|------|
| DEFAULT（0） | 默认行为 | 按 `DEFAULT_MOVE_TEMPERATURE = 1.75` 的 softmax 概率移动 |
| CONSERVATIVE（1）| 保守行为 | 选择得分最高的方向（贪心移动），持续 `SPECIAL_BEHAVIOR_DECAY_TURNS` 回合 |
| RANDOM（2） | 随机行为 | 等概率随机选择合法方向，持续 `RANDOM_ANT_DECAY_TURNS = 5` 回合 |
| BEWITCHED（3）| 被迷惑 | 朝目标点移动，持续 `SPECIAL_BEHAVIOR_DECAY_TURNS` 回合 |
| CONTROL_FREE（4）| 不受控 | 免疫行为变更，当 evasion shield 耗尽时触发 |

蚂蚁生成时，行为模式由 `_draw_spawn_profile()` 按以下概率随机分配：

| 种类 | 行为 | 概率 |
|------|------|:---:|
| WORKER + DEFAULT | 40% |
| WORKER + CONSERVATIVE | 35% |
| WORKER + RANDOM | 10% |
| COMBAT + DEFAULT | 15% |

### 4.5 蚂蚁死亡

当蚂蚁 `hp ≤ 0` 时，其状态变为 `FAIL`。死亡后果：

1. 击杀方获得金币：工蚁 `ANT_KILL_REWARD[level]`，战斗蚂蚁 `COMBAT_ANT_KILL_REWARD`（`_resolve_ant_lifecycle()` 中 `self.coins[1 - ant.player] += ant.kill_reward`）
2. 死亡方 `die_count` +1
3. 蚂蚁路径上留下负信息素（`PHEROMONE_FAIL_BONUS_INT = -50000`）

**注意**：蚂蚁死亡来源（被塔攻击、被闪电风暴伤害、兵蚁自毁）在引擎层面不可区分，统一按上述 `FAIL` 逻辑处理。

### 4.6 蚂蚁受伤

蚂蚁通过 `ant.take_damage(amount, apply_freeze)` 方法承受伤害，处理优先级为：

1. **回避盾（shield）**：若 `shield > 0`，则消耗 1 层盾并完全抵消本次伤害（不使用偏射盾）
2. **偏射盾（deflector）**：若 `deflector == True` 且 `amount × 2 < max_hp`，则完全抵消
3. **直接扣血**：`hp -= amount`

### 4.7 战斗蚂蚁自毁

当战斗蚂蚁 `hp × 2 < max_hp`（即 HP < 15）时，攻击塔时触发自毁（`should_self_destruct_on_tower_attack`）：

- 对目标塔及其半径 1 格内的所有敌方塔造成 `COMBAT_SELF_DESTRUCT_DAMAGE = 10` 点伤害
- 战斗蚂蚁自身 `hp = 0`，状态变为 `FAIL`

### 4.8 蚂蚁生成

蚂蚁从以下来源生成：

**基地产出**：每回合调用 `base.should_spawn(round_index)`，按 `ANT_GENERATION_SCHEDULE` 判定：
- 0 级科技（0→0）：`(9, 2)` 调度，约 4.5 回合产 1 只
- 1 级科技（0→1）：`(4, 1)` 调度，约 4.0 回合产 1 只
- 2 级科技（1→2）：`(7, 2)` 调度，约 3.5 回合产 1 只

种类和行为由 `_draw_spawn_profile()` 随机决定。

**生产者塔产出**：当 PRODUCER 系列塔的冷却完毕时，在塔的相邻可通行格产出 1 只蚂蚁。如果塔为 `PRODUCER_SIEGE`，有 25% 额外概率产出一只战斗蚂蚁（`siege_spawn_chance = 0.25`）。

### 4.9 蚂蚁移动

蚂蚁每回合向 6 个方向之一移动 1 格。移动方向由 `_choose_ant_move()` 计算，受以下因素综合影响（通过加权评分）：

- **进度**（progress）：向目标靠近的速度得分
- **信息素**（pheromone）：路径上的信息素值
- **拥挤**（crowding）：同格和邻格友方蚂蚁数量的惩罚
- **伤害风险**（expected_damage）：移动后的预期伤害
- **控制风险**（control_risk）：冰塔、脉冲塔等控制效果风险
- **塔吸引力**（tower_pull）：对战斗蚂蚁，敌方塔的吸引力
- **效果场**（effect_pull）：偏射盾和紧急撤离的范围吸引力

**工蚁目标**：固定为敌方基地（`PLAYER_BASES[1 - ant.player]`）。

**战斗蚂蚁目标**：优先选择离自己最近、且离敌方基地最近的敌方防御塔；若无敌方塔，则目标为敌方基地。

**DEFAULT 行为**：按 softmax 概率选择方向，temperature = 1.75（`DEFAULT_MOVE_TEMPERATURE`）。

**BEWITCHED 行为**：按 softmax 概率，temperature = 1.5（`BEWITCH_MOVE_TEMPERATURE`），目标为随机选择的敌方半场格子。

**RANDOM 行为**：等概率随机选择合法方向。

**CONSERVATIVE / CONTROL_FREE 行为**：选择评分最高的方向（贪心移动）。

**移动约束**：
- 蚂蚁不能移动到 VOID 或 BARRIER 格
- 非 RANDOM/BEWITCHED 行为下，默认禁止立即回头（backtrack 限制）
- 位于敌方基地格、敌方塔格、己方基地格的视作合法落点（但塔格会触发攻击）

### 4.10 蚂蚁攻击塔

当蚂蚁移动到敌方塔所在的格子时（而非仅是邻格），不执行正常移动，而是攻击该塔：

- 工蚁：造成 `WORKER_TOWER_ATTACK_DAMAGE[level]`（1/2/4）伤害
- 战斗蚂蚁：造成 `COMBAT_TOWER_ATTACK_DAMAGE = 5` 伤害
- 战斗蚂蚁如满足自毁条件，触发自毁

### 4.11 蚂蚁传送

每 `ANT_TELEPORT_INTERVAL = 10` 回合，约 `ANT_TELEPORT_RATIO = 0.1`（10%）的蚂蚁会触发传送——它们在 3 个子步中按 2/3 概率走正常逻辑、1/3 概率随机走的混合策略进行移动（`_resolve_random_move_steps`）。

---

## 5. 防御塔系统

### 5.1 塔属性

所有塔（除 PRODUCER 系列外）具有以下可攻击属性：

| 塔类型 | 伤害 | 攻速(回合) | 攻击范围 | 最大HP | 特殊效果 |
|------|:---:|:---:|:---:|:---:|------|
| **BASIC**（基础） | 5 | 2.0 | 1 | 10 | — |
| **HEAVY**（重型） | 12 | 2.0 | 1 | 15 | — |
| **HEAVY_PLUS**（重型+） | 24 | 2.0 | 1 | 15 | — |
| **ICE**（冰冻） | 12 | 2.0 | 2 | 15 | 冻结目标蚂蚁 1 回合，之后蚂蚁行为变为 RANDOM |
| **BEWITCH**（迷惑） | 14 | 2.0 | 2 | 15 | 使目标蚂蚁行为变为 BEWITCHED，向随机敌方半场点移动 |
| **QUICK**（快速） | 6 | 1.0 | 1 | 15 | — |
| **QUICK_PLUS**（快速+） | 6 | 0.5 | 1 | 15 | 每回合攻击 2 次 |
| **DOUBLE**（双发） | 6 | 2.0 | 3 | 15 | 每攻击选择 2 个目标 |
| **SNIPER**（狙击） | 10 | 2.0 | 4 | 15 | — |
| **MORTAR**（迫击炮） | 12 | 4.0 | 2 | 15 | 范围伤害 |
| **MORTAR_PLUS**（迫击炮+） | 18 | 4.0 | 2 | 15 | 范围伤害 |
| **PULSE**（脉冲） | 14 | 4.0 | 2 | 15 | 范围伤害 + 范围控制（RANDOM 行为） |
| **MISSILE**（导弹） | 18 | 6.0 | 3 | 15 | 范围伤害 |
| **PRODUCER**（生产者） | 0 | 10 回合 | 0 | 15 | 每 10 回合产 1 只蚂蚁 |
| **PRODUCER_FAST**（快速产蚁） | 0 | 8 回合 | 0 | 15 | 每 8 回合产 1 只蚂蚁 |
| **PRODUCER_SIEGE**（攻城） | 0 | 10 回合 | 0 | 15 | 每 10 回合产 1 只蚂蚁 + 25% 概率额外产战斗蚂蚁 |
| **PRODUCER_MEDIC**（医疗） | 0 | 10 回合 | 0 | 15 | 每 10 回合产 1 只蚂蚁；每 4 回合治疗最前线友方蚂蚁（回满血 + 1 层盾） |

### 5.2 塔升级树

```
BASIC → HEAVY  → HEAVY_PLUS / ICE / BEWITCH
      → QUICK  → QUICK_PLUS / DOUBLE / SNIPER
      → MORTAR → MORTAR_PLUS / PULSE / MISSILE
      → PRODUCER → PRODUCER_FAST / PRODUCER_SIEGE / PRODUCER_MEDIC
```

升级到 L2 费用：60 金币，升级到 L3 费用：200 金币。

### 5.3 塔攻击逻辑

1. 每回合，所有非生产者塔检测冷却（`tower.tick()` 减少 1.0 冷却）
2. 冷却就绪的塔选择攻击目标：优先攻击范围内距离最近的敌方蚂蚁（`_find_targets()`）
3. 对每个目标造成伤害（`_damage_ant_from_tower()`），速度 < 1 的塔可以每回合多次检测和攻击
4. 特殊塔效果同步触发：
   - **ICE**：使目标 `frozen = True`，`pending_behavior = RANDOM`
   - **BEWITCH**：若蚂蚁在己方半场 → 目标设为己方基地；否则随机选择敌方半场点
   - **PULSE**：使目标行为变为 RANDOM
5. 攻击后塔冷却重置为 `tower.speed`

**范围伤害塔**：MORTAR/MORTAR_PLUS/MISSILE/PULSE 以主目标为中心，攻击范围内的所有敌方蚂蚁。PULSE 的范围控制也覆盖所有范围内蚂蚁。

### 5.4 塔被攻击

塔通过 `tower.take_damage(amount)` 受伤，`hp ≤ 0` 后从游戏中移除：

| 攻击来源 | 伤害 | 来源 |
|---------|:---:|------|
| 工蚁攻击塔 | 1/2/4（取决于等级） | `ant.tower_attack_damage` |
| 战斗蚂蚁攻击塔 | 5 | `COMBAT_TOWER_ATTACK_DAMAGE` |
| 战斗蚂蚁自毁 | 10（半径 1 格范围） | `COMBAT_SELF_DESTRUCT_DAMAGE` |
| 闪电风暴对塔伤害 | 3（每 5 回合触发一次） | `LIGHTNING_STORM_TOWER_DAMAGE` |

### 5.5 塔操作

**建造**（BUILD_TOWER）：
- 只能在己方高地（`is_highland(player, x, y)` 返回 `True`）上建造
- 建造的格子不能是基地或已有塔
- 该格子不能处于 EMP 效果下
- 新建塔为 BASIC 类型，冷却初始化为 `TowerStats.speed`（2.0）
- 费用由 `tower_build_cost_for_count(tower_count)` 计算（参见 §3.3）

**升级**（UPGRADE_TOWER）：
- 目标类型必须在当前塔的升级树中
- 塔所在格不能处于 EMP 效果下
- 升级后 HP 重置为满值，冷却重置为 `tower.speed`

**拆除/降级**（DOWNGRADE_TOWER）：
- BASIC 塔：完全拆除，退款 = 建造费用 × 0.9 × (当前HP/最大HP)
- 非 BASIC 塔：降级 1 级（`tower_type = tower_type // 10`），退款 = 升级费用 × 0.9 × (当前HP/最大HP)，HP 等比例缩放

### 5.6 风险场

防御塔会在可通行格上产生两个影响场：

- **伤害风险场**（`damage_risk_field`）：每个非生产者塔对自己的 attack_range 内所有可通行格累加 `tower.damage / 25.0`
- **控制风险场**（`control_risk_field`）：ICE 塔贡献 1.0，BEWITCH 塔贡献 1.3，PULSE 塔贡献 0.7

这些场用于蚂蚁移动时的风险评估。

---

## 6. 超级武器系统

### 6.1 超级武器参数

| 武器 | 费用 | 冷却 | 持续时间 | 范围 | 来源 |
|------|:---:|:---:|:---:|:---:|------|
| LIGHTNING_STORM | 90 | 35 | 15 | 3 | `SUPER_WEAPON_STATS` |
| EMP_BLASTER | 135 | 45 | 10 | 3 | 同上 |
| DEFLECTOR | 60 | 25 | 10 | 3 | 同上 |
| EMERGENCY_EVASION | 60 | 25 | **1**（瞬时） | 3 | 同上 |

### 6.2 各武器效果

**LIGHTNING_STORM（闪电风暴）**：
- 部署后立即对范围内所有敌方蚂蚁造成 `LIGHTNING_STORM_ANT_DAMAGE = 20` 伤害
- 每回合持续对范围内所有敌方蚂蚁造成 20 伤害
- 每 `LIGHTNING_STORM_TOWER_INTERVAL = 5` 回合对范围内所有敌方塔造成 `LIGHTNING_STORM_TOWER_DAMAGE = 3` 伤害
- 每回合随机漂移到相邻 1 格（`_drift_effect()`）
- 在伤害风险场中贡献 `20 / 25.0 = 0.8` 的伤害值

**EMP_BLASTER（EMP 轰炸）**：
- 使范围内所有敌方塔失效：不能攻击、不能产蚁、不能触发医疗
- 范围内不能建塔、不能升级塔
- 每回合随机漂移到相邻 1 格
- 持续时间 10 回合

**DEFLECTOR（重力偏射盾）**：
- 范围内己方蚂蚁获得 `deflector = True`
- 偏射盾抵消小额伤害（`amount × 2 < max_hp`）
- 效果结束后蚂蚁行为变为 CONTROL_FREE
- 在效果场中贡献 `DEFLECTOR_PATH_ATTRACTION = 1.0` 的吸引力

**EMERGENCY_EVASION（紧急回避装置）**：
- 部署时给范围内己方蚂蚁增加 `shield = max(当前值, 2)`、`evasion = True`
- 每回合对范围内己方蚂蚁增加 2 层 shield
- shield 耗尽后蚂蚁行为变为 CONTROL_FREE
- **实际持续时间 1 回合**：引擎在 `_tick_effects()` 中对 EMERGENCY_EVASION 不减 `remaining_turns`，一轮结束后直接移除
- 在效果场中贡献 `EMERGENCY_EVASION_PATH_ATTRACTION = 1.35` 的吸引力

### 6.3 超级武器冷却

冷却在部署时设置为对应的 cooldown 值，每回合减少 1。冷却期间不能再次使用同类型武器。

---

## 7. 基地与科技升级

### 7.1 基地属性

`Base` 数据结构（[model.py:L352-L386](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/backend/model.py#L352-L386)）包含：

| 属性 | 初始值 | 说明 |
|------|:---:|------|
| `hp` | 50 | 被蚂蚁到达时减 1，≤ 0 时游戏结束 |
| `generation_level` | 0 | 0→1→2，影响蚂蚁生成速度 |
| `ant_level` | 0 | 0→1→2，影响工蚁 HP 和塔攻击伤害 |

### 7.2 科技升级

| 升级操作 | 费用 | 效果 | 上限 |
|------|:---:|------|:---:|
| UPGRADE_GENERATION_SPEED | L0→L1: 200 / L1→L2: 250 | `generation_level += 1` | 2 |
| UPGRADE_GENERATED_ANT | L0→L1: 200 / L1→L2: 250 | `ant_level += 1` | 2 |

**注意**：升级 `UPGRADE_GENERATED_ANT` L1→L2 时，工蚁 HP 从 25→25（无变化），因此在 `_base_upgrade_candidates()` 中 `hp_gain > 0` 检查会导致此升级不被推荐，但引擎本身**不阻止**此操作。

### 7.3 基地 HP 减少

基地 HP 仅通过一种方式减少：敌方蚂蚁到达基地（`ant.status == SUCCESS`）：

```python
self.bases[1 - ant.player].hp -= 1       # 每次固定减 1
self.coins[ant.player] += 10              # 到达方获 10 金币
```

基地不能被蚂蚁直接"攻击"——蚂蚁只能"到达"。

---

## 8. 游戏回合流程

### 8.1 操作执行顺序

```
Step 1: 玩家0 提交操作列表 → apply_operation_list(0, ops)
        若操作非法且未设置 cold_handle_rule_illegal → 立即判负

Step 2: 玩家1 提交操作列表 → apply_operation_list(1, ops)
        若操作非法且未设置 cold_handle_rule_illegal → 立即判负

Step 3: advance_round()（游戏回合结算）
```

### 8.2 单回合结算流程（`advance_round()`）

按以下顺序执行：

1. **攻击结算**（`_attack_ants()`）
   - 准备蚂蚁（解除冰冻、更新偏射盾/回避状态）
   - 闪电风暴攻击（伤害蚂蚁 + 周期性伤害塔）
   - 防御塔攻击（按 ID 顺序，冷却就绪的塔攻击敌人）

2. **蚂蚁移动**（`_move_ants()`）
   - 所有存活蚂蚁移动 1 步
   - 移动过程中遇到敌方塔则攻击塔
   - 移动后触发蚂蚁传送（每 10 回合）

3. **信息素更新**（`_update_pheromone()`）
   - 全局衰减：`p = 0.97p + 0.03 * 10`（整数运算）
   - SUCCESS 蚂蚁：路径上 `+100000`
   - FAIL 蚂蚁：路径上 `-50000`
   - TOO_OLD 蚂蚁：路径上 `-30000`

4. **蚂蚁生命周期结算**（`_resolve_ant_lifecycle()`）
   - SUCCESS：敌方基地 HP -1，己方金币 +10
   - FAIL：击杀方获得 `kill_reward`，死亡方 `die_count` +1
   - TOO_OLD：`old_count` +1
   - 所有死亡蚂蚁从游戏中移除

5. **蚂蚁生成**（`_spawn_ants()`）
   - 基地按调度产出
   - 生产者塔冷却就绪时产出

6. **年龄增长**（`_increase_ant_age()`）
   - `ant.age += 1`，`ant.behavior_turns += 1`
   - RANDOM 行为 5 回合后自动恢复 DEFAULT
   - BEWITCHED 到达目标后恢复 DEFAULT
   - `behavior_expiry` 到期后恢复 DEFAULT

7. **金币收入**（每 2 回合）
   - 双方各 `+BASIC_INCOME = 3` 金币

8. **效果结算**（`_tick_effects()`）
   - 冷却 -1
   - 超级武器效果漂移
   - 效果持续时间 -1（EMERGENCY_EVASION 除外）

9. **回合推进**：`round_index += 1`

10. **超时判定**：若 `round_index >= MAX_ROUND(512)`，触发超时胜负判定

### 8.3 操作合法性检查

对每个操作，系统检查（`can_apply_operation()`）：

- **BUILD_TOWER**：位置必须是己方高地、不能是基地或已有塔、不能处于 EMP 下、且钱够
- **UPGRADE_TOWER**：塔属于己方、目标类型在升级树中、塔不处于 EMP 下
- **DOWNGRADE_TOWER**：塔属于己方、不处于 EMP 下
- **超级武器**：位置合法、不处于冷却中、钱够
- **科技升级**：等级未满 2 级

---

## 9. 操作类型（Action Space）

### 9.1 操作枚举

| 操作 | 枚举值 | 参数 | 说明 |
|------|:---:|------|------|
| BUILD_TOWER | 11 | x, y | 在坐标建造 BASIC 塔 |
| UPGRADE_TOWER | 12 | tower_id, target_type | 升级指定塔 |
| DOWNGRADE_TOWER | 13 | tower_id | 拆除/降级指定塔 |
| USE_LIGHTNING_STORM | 21 | x, y | 部署闪电风暴 |
| USE_EMP_BLASTER | 22 | x, y | 部署 EMP |
| USE_DEFLECTOR | 23 | x, y | 部署偏射盾 |
| USE_EMERGENCY_EVASION | 24 | x, y | 部署紧急回避 |
| UPGRADE_GENERATION_SPEED | 31 | — | 升级蚂蚁生产速度 |
| UPGRADE_GENERATED_ANT | 32 | — | 升级蚂蚁最大血量 |

### 9.2 协议格式

- `BUILD_TOWER`, `USE_LIGHTNING_STORM`, `USE_EMP_BLASTER`, `USE_DEFLECTOR`, `USE_EMERGENCY_EVASION`：3 个 token（`[op_type, x, y]`）
- `UPGRADE_TOWER`：3 个 token（`[op_type, tower_id, target_type]`）
- `DOWNGRADE_TOWER`：2 个 token（`[op_type, tower_id]`）
- `UPGRADE_GENERATION_SPEED`, `UPGRADE_GENERATED_ANT`：1 个 token（`[op_type]`）

### 9.3 Action Catalog 候选生成

PPO 使用的 `ActionCatalog` 将 96 维离散动作空间映射到候选 `ActionBundle` 列表：

1. **hold（NO_OP）**：始终可用，score = 0
2. **build**：对每个 `STRATEGIC_BUILD_ORDER` 中的高地位置，按优先级、本地敌人压力和费用评估生成候选
3. **upgrade**：对每个己方塔的每个合法升级方向，按塔类型适配度、等级、位置优先级评估
4. **downgrade**：对每个己方塔（本地敌人压力不大的）评估退款收益
5. **base upgrade**：`UPGRADE_GENERATION_SPEED` 和 `UPGRADE_GENERATED_ANT`
6. **superweapon**：4 种超级武器的部署候选
7. **paired**：两个低优先级操作的组合

---

## 10. 胜负判定

### 10.1 正常胜负

当任意一方基地 `hp ≤ 0` 时：
- 玩家 0 基地 HP ≤ 0 → 玩家 1 获胜
- 玩家 1 基地 HP ≤ 0 → 玩家 0 获胜
- 双方同时 ≤ 0 → 判定 `winner = 0`（平局时玩家 0 胜）

### 10.2 超时胜负

当 `round_index >= MAX_ROUND(512)` 时，按以下优先级判定胜负：

| 优先级 | 判定条件 | 胜者 |
|:---:|------|:---:|
| 1 | HP 差 | HP 更高的玩家 |
| 2 | die_count 差 | die_count 更多的玩家（死亡少 = 胜） |
| 3 | super_weapon_usage | 超级武器使用次数更少的玩家（少用 = 管理好 = 胜） |
| 4 | ai_time | AI 思考时间更短的玩家 |
| 5 | 默认 | 玩家 0 |

### 10.3 非法操作判负

如果玩家提交了引擎无法执行的非法操作，且 `cold_handle_rule_illegal = False`（正常模式），则该玩家立即判负。

---

## 11. 关键常数速查表

| 常数 | 值 | 说明 |
|------|:---:|------|
| `MAP_SIZE` | 19 | 地图宽度/高度 |
| `MAX_ROUND` | 512 | 最大回合数 |
| `BASE_HP` | 50 | 基地初始 HP |
| `INITIAL_COINS` | 50 | 初始金币 |
| `BASIC_INCOME` | 3 | 基础收入（每 2 回合） |
| `BASIC_INCOME_INTERVAL` | 2 | 收入间隔 |
| `ANT_AGE_LIMIT` | 64 | 工蚁最大年龄 |
| `ANT_MAX_HP` | (20, 25, 25) | L0/L1/L2 工蚁 HP |
| `COMBAT_ANT_HP` | 30 | 战斗蚂蚁 HP |
| `ANT_KILL_REWARD` | (6, 10, 14) | 击杀 L0/L1/L2 工蚁金币 |
| `COMBAT_ANT_KILL_REWARD` | 18 | 击杀战斗蚂蚁金币 |
| `ANT_BREACH_REWARD` | 10 | 蚂蚁到达敌方基地金币 |
| `LEVEL2_TOWER_UPGRADE_COST` | 60 | 塔升级到 L2 费用 |
| `LEVEL3_TOWER_UPGRADE_COST` | 200 | 塔升级到 L3 费用 |
| `BASE_UPGRADE_COST` | (200, 250) | 科技升级费用 |
| `TOWER_DOWNGRADE_REFUND_RATIO` | 0.9 | 拆除退款比例 |
| `LIGHTNING_STORM_ANT_DAMAGE` | 20 | LS 对蚂蚁伤害 |
| `LIGHTNING_STORM_TOWER_DAMAGE` | 3 | LS 对塔伤害 |
| `LIGHTNING_STORM_TOWER_INTERVAL` | 5 | LS 对塔伤害间隔 |
| `COMBAT_SELF_DESTRUCT_DAMAGE` | 10 | 兵蚁自毁伤害 |
| `COMBAT_SELF_DESTRUCT_RANGE` | 1 | 兵蚁自毁范围 |
| `MAX_ACTIONS` | 96 | 动作空间维度 |
| `DEFAULT_MOVE_TEMPERATURE` | 1.75 | 默认移动 softmax 温度 |
| `ANT_TELEPORT_INTERVAL` | 10 | 蚂蚁传送检查间隔 |
| `ANT_TELEPORT_RATIO` | 0.1 | 蚂蚁传送比例 |
| `COMBAT_INITIAL_EVASION` | 3 | 战斗蚂蚁初始回避层数 |
| `RANDOM_ANT_DECAY_TURNS` | 5 | 随机行为衰减回合 |

---

> 本文档完全基于 SDK 源代码事实编写。所有规则均可追溯至 `Ant-Game/SDK/backend/engine.py`、`Ant-Game/SDK/backend/model.py` 和 `Ant-Game/SDK/utils/constants.py` 中的对应实现。
