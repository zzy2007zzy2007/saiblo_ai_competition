# PPO v1 Reward 体系技术实现方案

> 版本：v1.0
> 日期：2026-05-16
> 基于：[ppo_v1_reward_system_design.md](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/game/ppo_v1_reward_system_design.md)
> 原则：每一处修改均可追溯至设计文档中的对应条目。

---

## 0. 代码现状分析

### 0.1 当前奖励计算

当前 `[env.py:_round_rewards()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L152-L168)` 仅在回合结算时计算一次 reward，公式为：

```python
reward = (Δenemy_hp - Δself_hp) × 10.0    # HP 伤害差
        + Δself_coins × 0.05                 # 己方金币增
        - Δenemy_coins × 0.02                # 敌方金币增
        ± 100.0 (terminal)                   # 终局
```

**核心问题**：每个 step（agent 操作）的 reward 仅在"推进回合后"才非零。大部分 step 的 reward=0，GAE 无法传播有效 advantage。

### 0.2 引擎事件追踪现状

当前 `[engine.py:advance_round()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/backend/engine.py)` 内部不追踪过程事件：
- 蚂蚁死亡统一处理为 `FAIL`，不区分死因（塔/闪电风暴/自毁）
- 塔攻击命中、塔被攻击等中间步没有事件记录
- `die_count`、`old_count` 仅是累加计数器，不含事件级信息
- **实现细粒度效果奖励需要先添加事件追踪**

### 0.3 操作检测现状

当前 `[env.py:_step_single()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L170-L219)` 通过 `ActionBundle.name` 可获取操作名称：
- `"hold"` → NO_OP
- `"build"` → BUILD_TOWER
- `"upgrade"` → UPGRADE_TOWER
- `"sell"` → DOWNGRADE_TOWER
- `"base_upgrade"` → 科技升级
- `"storm"/"emp"/"deflector"/"evasion"` → 超级武器

### 0.4 PPO 配置现状

当前 PPO trainer 使用 **相对裁剪**：
- `clip_eps_vf = 0.2`（values 相对裁剪幅度）
- **不存在绝对 `values_clip_range` 参数**
- `gamma = 0.99`, `gae_lambda = 0.95`, `ent_coef = 0.15`

**设计文档 §5.5 中"扩大 values_clip_range 到 200.0"的建议不适用于当前相对裁剪机制。** 当前 `clip_eps_vf=0.2` 已经可以处理任意范围的 values（相对裁剪），无需修改。

---

## 1. 文件修改总览

| 序号 | 文件 | 修改类型 | 内容 |
|:---:|------|:---:|------|
| 1 | `Ant-Game/SDK/backend/model.py` | 新增类 | `RoundEvents` 事件追踪数据结构 |
| 2 | `Ant-Game/SDK/backend/engine.py` | 修改 | 在 `advance_round()` 各子步骤中记录事件 |
| 3 | `Ant-Game/SDK/training/env.py` | **重写** | `_round_rewards()` + 新增 `_step_rewards()` + 新增 `_action_rewards()` + 新增 `_event_rewards()` + 新增 `REWARD_CONFIG` |
| 4 | `Ant-Game/SDK/training/env.py` | 修改 | `__init__`、`reset`、`_step_single` 适配新体系 |

---

## 2. 修改 1：新增 `RoundEvents` 数据结构

### 文件：`Ant-Game/SDK/backend/model.py`

在文件末尾添加 `RoundEvents` 类（或在 `GameState` 之前）：

```python
@dataclass
class RoundEvents:
    """每回合结算期间记录的战斗与效果事件。
    由 engine.advance_round() 各子步骤填充，
    供 env._round_rewards() 消费以计算细粒度奖励。
    """
    # --- 塔相关 ---
    tower_hits: list[tuple[int, int, int]]    # [(tower_id, ant_id, damage)]
    tower_kills: list[tuple[int, int, int]]    # [(tower_id, ant_id, player_of_ant)]
    towers_destroyed: list[tuple[int, int]]    # [(tower_id, player_owner)]

    # --- 蚂蚁相关 ---
    ant_breaches: list[tuple[int, int]]        # [(ant_id, breach_player)]
    ants_killed: list[tuple[int, int]]         # [(ant_id, player_owner)]
    ants_old: list[tuple[int, int]]            # [(ant_id, player_owner)]

    # --- 超级武器 ---
    ls_ant_kills: list[tuple[int, int, int]]   # [(ant_id, player_owner, damage)]
    ls_tower_damage: list[tuple[int, int]]     # [(tower_id, damage)]
    emp_disabled_towers: list[int]              # [tower_id]
    deflector_negations: list[tuple[int, int]]  # [(ant_id, negated_damage)]
    evasion_shields_granted: list[int]          # [ant_id]

    def clear(self) -> None:
        """清除所有事件，在每回合开始时调用。"""
        self.tower_hits.clear()
        self.tower_kills.clear()
        self.towers_destroyed.clear()
        self.ant_breaches.clear()
        self.ants_killed.clear()
        self.ants_old.clear()
        self.ls_ant_kills.clear()
        self.ls_tower_damage.clear()
        self.emp_disabled_towers.clear()
        self.deflector_negations.clear()
        self.evasion_shields_granted.clear()

    @classmethod
    def new(cls) -> "RoundEvents":
        return cls([], [], [], [], [], [], [], [], [], [], [])
```

**同时修改 `GameState` 类**，新增字段（在 `__init__` 或现有的初始化位置）：

```python
self.round_events: RoundEvents = RoundEvents.new()
```

---

## 3. 修改 2：引擎事件追踪

### 文件：`Ant-Game/SDK/backend/engine.py`

需要在 `advance_round()` 中添加一步：**回合开始，清除事件**，然后在各子步骤中插入记录逻辑。

#### 3.1 在 `advance_round()` 开头添加

```python
def advance_round(self) -> None:
    # ===== 新增：清除上回合事件 =====
    self.round_events.clear()
    
    # ... 原有代码不变 ...
    self._attack_ants()
    self._move_ants()
    # ...
```

#### 3.2 修改 `_tower_attack()` — 记录命中与击杀

在塔攻击蚂蚁的实现中，每造成一次伤害时：

```python
# 在 _damage_ant_from_tower() 或等效位置
def _damage_ant_from_tower(self, tower, ant, damage):
    ant.take_damage(damage)
    # ===== 新增：记录塔命中事件 =====
    self.round_events.tower_hits.append((tower.id, ant.id, damage))
    if ant.hp <= 0:
        # ===== 新增：记录塔击杀事件 =====
        self.round_events.tower_kills.append((tower.id, ant.id, ant.player))
```

**注意**：需要区分"塔攻击命中"和"塔击杀蚂蚁"。塔攻击命中是每次造成伤害时记录；塔击杀是伤害导致 HP ≤ 0 时记录。

#### 3.3 修改 `_resolve_ant_lifecycle()` — 记录蚂蚁生命周期

在遍历蚂蚁状态的处理中：

```python
# 在遍历蚂蚁 status 的逻辑中
for ant in all_ants:
    if ant.status == AntStatus.SUCCESS:
        # ===== 新增：记录破基地 =====
        self.round_events.ant_breaches.append((ant.id, ant.player))
    elif ant.status == AntStatus.FAIL:
        # ===== 新增：记录蚂蚁死亡 =====
        self.round_events.ants_killed.append((ant.id, ant.player))
    elif ant.status == AntStatus.TOO_OLD:
        # ===== 新增：记录蚂蚁老死 =====
        self.round_events.ants_old.append((ant.id, ant.player))
```

#### 3.4 修改 `_attack_tower_from_ant()` — 记录塔被摧毁

在蚂蚁攻击塔导致塔被摧毁时：

```python
def _attack_tower_from_ant(self, ant, tower):
    # ... 原有伤害逻辑 ...
    if tower.took_damage and tower.hp <= 0:
        # ===== 新增：记录塔被摧毁 =====
        self.round_events.towers_destroyed.append((tower.id, tower.player))
```

#### 3.5 修改 `_apply_lightning_effect()` — 记录 LS 效果

```python
def _apply_lightning_effect(self, effect):
    for ant in enemy_ants_in_range:
        ant.take_damage(LIGHTNING_STORM_ANT_DAMAGE)
        # ===== 新增 =====
        self.round_events.ls_ant_kills.append((ant.id, ant.player, LIGHTNING_STORM_ANT_DAMAGE))
    # ... tower damage every 5 rounds ...
    if self.round_index % LIGHTNING_STORM_TOWER_INTERVAL == 0:
        for tower in enemy_towers_in_range:
            tower.take_damage(LIGHTNING_STORM_TOWER_DAMAGE)
            self.round_events.ls_tower_damage.append((tower.id, LIGHTNING_STORM_TOWER_DAMAGE))
            if tower.hp <= 0:
                self.round_events.towers_destroyed.append((tower.id, tower.player))
```

#### 3.6 修改 `_tick_effects()` — 记录 EMP 禁用

```python
def _tick_effects(self):
    for effect in self.active_effects:
        if effect.type == EMP_BLASTER:
            for tower in enemy_towers_in_range:
                if tower.id not in self.round_events.emp_disabled_towers:
                    # ===== 新增：记录本回合新被禁用的塔 =====
                    self.round_events.emp_disabled_towers.append(tower.id)
```

#### 3.7 修改 `Ant.take_damage()` — 记录偏射盾抵消

在 `[model.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/backend/model.py)` 的 `take_damage()` 方法中：

```python
def take_damage(self, amount: int, apply_freeze: bool = True) -> int:
    # ... 回避盾逻辑 ...
    if self.deflector and amount * 2 < self.max_hp:
        # ===== 新增：记录偏射盾抵消 =====
        # 注意：这需要 GameState 的引用，可通过 engine 传入
        return 0  # 原逻辑，伤害被完全抵消
    # ...
```

**实现难点**：`take_damage()` 在 `model.py` 中，没有 `GameState` 引用。解决方案：
- **方案A**：在 engine 中包装 `_damage_ant_from_tower()`，在调用 `take_damage()` 前后检测偏射盾是否触发（通过 HP 变化判断）
- **方案B**：给 `take_damage()` 传入一个可选的 `event_callback`
- **推荐方案A**：在 engine 的 `_damage_ant_from_tower()` 中：

```python
def _damage_ant_from_tower(self, tower, ant, damage):
    hp_before = ant.hp
    ant.take_damage(damage)
    hp_after = ant.hp
    actual_damage = hp_before - hp_after
    if actual_damage == 0 and hp_before > 0:
        # 伤害被偏射盾抵消
        self.round_events.deflector_negations.append((ant.id, damage))
```

---

## 4. 修改 3：重写奖励计算（核心）

### 文件：`Ant-Game/SDK/training/env.py`

#### 4.1 新增 `REWARD_CONFIG`

在 `env.py` 文件顶部（`import` 之后、`class AntWarSequentialEnv` 之前）添加：

```python
REWARD_CONFIG = {
    # ---- 回合级 (round-level) ----
    "hp_diff_weight": 10.0,
    "own_coin_gain_weight": 0.10,
    "enemy_coin_gain_weight": 0.05,
    "tower_survival_per_tower": 0.02,
    "enemy_tower_survival_per_tower": 0.01,
    "emp_disabled_per_tower": 0.05,

    # ---- 动作奖励 (action rewards) ----
    "build_tower_tiers": [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1],
    "upgrade_tower_l2": 0.3,
    "upgrade_tower_l3": 0.5,
    "downgrade_tower_penalty": -0.3,
    "upgrade_gen_speed": [1.0, 0.8],
    "upgrade_gen_ant": [0.5, 0.1],
    "deploy_ls": 0.5,
    "deploy_emp": 0.8,
    "deploy_deflector": 0.3,
    "deploy_evasion": 0.2,

    # ---- NO_OP ----
    "noop_tolerance": 3,
    "noop_base_penalty": -0.02,
    "noop_max_penalty": -0.10,

    # ---- 效果奖励 (effect rewards) ----
    "breach_enemy_base": 1.0,
    "breached_by_enemy": -0.5,
    "tower_kill_ant": 0.2,
    "own_ant_killed": -0.05,
    "tower_hit_ant": 0.03,
    "destroy_enemy_tower": 0.8,
    "own_tower_destroyed": -1.0,
    "ls_kill_ant": 0.15,
    "ls_destroy_tower": 0.3,
    "deflector_negate": 0.05,
    "own_hp_damaged": -0.3,
    "enemy_hp_damaged": 0.5,

    # ---- 终局 (terminal) ----
    "win_reward": 100.0,
    "loss_reward": -100.0,

    # ---- 裁剪 ----
    "step_reward_clip": 20.0,
}
```

#### 4.2 修改 `__init__()` — 新增状态变量

```python
def __init__(self, ...) -> None:
    # ... 原有代码不变 ...
    
    # ==== 新增：动作奖励追踪 ====
    self._round_start_tower_count = (0, 0)      # 回合开始时各方的塔数量
    self._player_tower_build_counts: dict[int, int] = {0: 0, 1: 0}  # 历史建造数
    self._player_noop_streak: dict[int, int] = {0: 0, 1: 0}         # 连续 NO_OP 计数
```

#### 4.3 修改 `_capture_round_start()` — 新增塔数量快照

```python
def _capture_round_start(self) -> None:
    self._round_start_hp = tuple(base.hp for base in self.state.bases)
    self._round_start_coins = tuple(self.state.coins)
    # ==== 新增：记录回合开始塔数量 ====
    self._round_start_tower_count = tuple(len(self.state.towers[p]) for p in (0, 1))
```

#### 4.4 修改 `reset()` — 重置新状态

```python
def reset(self, ...):
    # ... 原有代码 ...
    self._capture_round_start()
    # ==== 新增 ====
    self._player_tower_build_counts = {0: 0, 1: 0}
    self._player_noop_streak = {0: 0, 1: 0}
    self._round_start_tower_count = (0, 0)
    # ... 后续代码不变 ...
```

#### 4.5 修改 `_step_single()` — 注入步骤级奖励

这是最关键的修改。需要在 agent 执行操作后、结算回合后分别给奖励：

```python
def _step_single(self, action: int | None) -> None:
    if not self.agents:
        return

    agent = self.agent_selection
    if self.terminations[agent] or self.truncations[agent]:
        self._was_dead_step(action)
        return

    self._clear_rewards()
    self._cumulative_rewards[agent] = 0.0
    player = self.player_index(agent)
    bundle, illegal = self._selected_bundle(player, action)

    # ===== 新增：记录操作前的塔数量（用于检测建塔/拆塔） =====
    tower_count_before = len(self.state.towers[player])

    invalid_ops = self.state.apply_operation_list(player, bundle.operations)
    self.infos[agent] = {
        **self.infos.get(agent, {}),
        "bundle": bundle,
        "illegal": illegal,
        "invalid_ops": invalid_ops,
    }

    # ===== 新增：操作后立即计算动作奖励 =====
    step_action_reward = self._action_rewards(player, bundle, tower_count_before)

    rewards = {name: 0.0 for name in self.possible_agents}
    self._context = self._context.next_turn()
    completed_round = False

    if not self.state.terminal and player == 1:
        self.state.advance_round()
        completed_round = True

    if completed_round or self.state.terminal:
        # ===== 修改：用新的奖励函数替代旧的 _round_rewards =====
        rewards = self._round_rewards_v2()
        if completed_round and not self.state.terminal:
            self._capture_round_start()

    if illegal:
        rewards[agent] -= 1.0

    # ===== 新增：将动作奖励叠加到总奖励中 =====
    rewards[agent] = rewards.get(agent, 0.0) + step_action_reward

    # ===== 新增：单步裁剪 =====
    for name in self.possible_agents:
        r = rewards.get(name, 0.0)
        rewards[name] = max(min(r, REWARD_CONFIG["step_reward_clip"]), -REWARD_CONFIG["step_reward_clip"])

    for name in self.possible_agents:
        self.rewards[name] = float(rewards[name])
        self.terminations[name] = self.state.terminal
        self.truncations[name] = False

    self._refresh_bundles()
    self._update_infos()
    self._accumulate_rewards()

    if self.state.terminal:
        self._deads_step_first()
    else:
        self.agent_selection = self.possible_agents[self._context.to_play]
```

#### 4.6 新增 `_action_rewards()` — 动作级奖励

```python
def _action_rewards(self, player: int, bundle: ActionBundle, tower_count_before: int) -> float:
    """根据 agent 操作类型计算过程奖励。"""
    cfg = REWARD_CONFIG
    name = bundle.name
    reward = 0.0

    # --- NO_OP 处理 ---
    if name == "hold" or "noop" in bundle.tags:
        self._player_noop_streak[player] += 1
        n = self._player_noop_streak[player]
        if n > cfg["noop_tolerance"]:
            penalty = cfg["noop_base_penalty"] + cfg["noop_base_penalty"] * ((n - cfg["noop_tolerance"]) // 10)
            reward = max(penalty, cfg["noop_max_penalty"])
    else:
        self._player_noop_streak[player] = 0

    # --- 建造塔 ---
    if name == "build":
        count = self._player_tower_build_counts[player]
        if count < len(cfg["build_tower_tiers"]):
            reward += cfg["build_tower_tiers"][count]
        else:
            reward += cfg["build_tower_tiers"][-1]
        self._player_tower_build_counts[player] += 1

    # --- 升级塔 ---
    if name == "upgrade":
        # 通过操作参数判断 L1→L2 还是 L2→L3
        for op in bundle.operations:
            # op_type: UPGRADE_TOWER, 参数: [op_type, tower_id, target_type]
            target_type = op[2]
            # L2 类型编号 < 100, L3 类型编号 >= 100
            if target_type >= 100:
                reward += cfg["upgrade_tower_l3"]
            else:
                reward += cfg["upgrade_tower_l2"]

    # --- 拆除/降级塔 ---
    if name == "sell":
        reward += cfg["downgrade_tower_penalty"]

    # --- 科技升级 ---
    if name == "base_upgrade":
        for op in bundle.operations:
            op_type = op[0]
            if op_type == 31:  # UPGRADE_GENERATION_SPEED
                base = self.state.bases[player]
                level = base.generation_level - 1  # 操作后 level
                if 0 <= level < len(cfg["upgrade_gen_speed"]):
                    reward += cfg["upgrade_gen_speed"][level]
            elif op_type == 32:  # UPGRADE_GENERATED_ANT
                base = self.state.bases[player]
                level = base.ant_level - 1
                if 0 <= level < len(cfg["upgrade_gen_ant"]):
                    reward += cfg["upgrade_gen_ant"][level]

    # --- 超级武器 ---
    if name == "storm":
        reward += cfg["deploy_ls"]
    elif name == "emp":
        reward += cfg["deploy_emp"]
    elif name == "deflector":
        reward += cfg["deploy_deflector"]
    elif name == "evasion":
        reward += cfg["deploy_evasion"]

    return reward
```

#### 4.7 新增 `_round_rewards_v2()` — 回合级 + 效果级奖励

完整替代旧的 `_round_rewards()`：

```python
def _round_rewards_v2(self) -> dict[str, float]:
    """新奖励体系：回合级 delta + 事件级 + 终局。"""
    cfg = REWARD_CONFIG
    rewards: dict[str, float] = {}
    previous_hp = self._round_start_hp
    previous_coins = self._round_start_coins
    previous_towers = self._round_start_tower_count

    events = self.state.round_events  # RoundEvents 实例

    for index, agent in enumerate(self.possible_agents):
        enemy = 1 - index
        reward = 0.0

        # ===== 1. 回合级 delta =====
        # HP 伤害差（保持）
        reward += (previous_hp[enemy] - self.state.bases[enemy].hp) * cfg["hp_diff_weight"]
        reward -= (previous_hp[index] - self.state.bases[index].hp) * cfg["hp_diff_weight"]

        # 金币差（权重提升）
        reward += (self.state.coins[index] - previous_coins[index]) * cfg["own_coin_gain_weight"]
        reward -= (self.state.coins[enemy] - previous_coins[enemy]) * cfg["enemy_coin_gain_weight"]

        # 塔存活（新增）
        reward += len(self.state.towers[index]) * cfg["tower_survival_per_tower"]
        reward -= len(self.state.towers[enemy]) * cfg["enemy_tower_survival_per_tower"]

        # ===== 2. 事件级效果 =====
        reward += self._event_rewards(index, events)

        # ===== 3. 终局 =====
        if self.state.terminal:
            if self.state.winner == index:
                reward += cfg["win_reward"]
            elif self.state.winner == enemy:
                reward += cfg["loss_reward"]

        rewards[agent] = float(reward)
    return rewards
```

#### 4.8 新增 `_event_rewards()` — 效果事件级奖励

```python
def _event_rewards(self, player: int, events: RoundEvents) -> float:
    """从 RoundEvents 计算本回合的效果奖励（player 视角）。"""
    cfg = REWARD_CONFIG
    enemy = 1 - player
    reward = 0.0

    # --- 蚂蚁破基地 ---
    for ant_id, breach_player in events.ant_breaches:
        if breach_player == player:
            reward += cfg["breach_enemy_base"]
        else:
            reward += cfg["breached_by_enemy"]

    # --- 塔击杀蚂蚁 ---
    for tower_id, ant_id, ant_player in events.tower_kills:
        # 找到塔的拥有者
        tower_owner = self._find_tower_owner(tower_id)
        if tower_owner == player and ant_player != player:
            reward += cfg["tower_kill_ant"]

    # --- 塔攻击命中 ---
    for tower_id, ant_id, damage in events.tower_hits:
        tower_owner = self._find_tower_owner(tower_id)
        ant_player = self._find_ant_player(ant_id)
        if tower_owner == player and ant_player != player:
            reward += cfg["tower_hit_ant"]

    # --- 蚂蚁被击杀 ---
    for ant_id, ant_player in events.ants_killed:
        if ant_player == player:
            reward += cfg["own_ant_killed"]

    # --- 塔被摧毁 ---
    for tower_id, tower_player in events.towers_destroyed:
        if tower_player == player:
            reward += cfg["own_tower_destroyed"]
        else:
            reward += cfg["destroy_enemy_tower"]

    # --- LS 效果 ---
    for ant_id, ant_player, damage in events.ls_ant_kills:
        if ant_player != player:
            reward += cfg["ls_kill_ant"]
    # LS 摧毁塔
    for tower_id, damage in events.ls_tower_damage:
        tower_owner = self._find_tower_owner(tower_id)
        if tower_owner != player:
            reward += cfg["ls_destroy_tower"]

    # --- EMP 禁用塔 ---
    for tower_id in events.emp_disabled_towers:
        tower_owner = self._find_tower_owner(tower_id)
        if tower_owner != player:
            reward += cfg["emp_disabled_per_tower"]

    # --- 偏射盾抵消 ---
    for ant_id, negated_damage in events.deflector_negations:
        ant_player = self._find_ant_player(ant_id)
        if ant_player == player:
            reward += cfg["deflector_negate"]

    # --- 渐进 HP 伤害 ---
    hp_damaged = self._round_start_hp[enemy] - self.state.bases[enemy].hp
    own_hp_lost = self._round_start_hp[player] - self.state.bases[player].hp
    reward += hp_damaged * cfg["enemy_hp_damaged"]
    reward += own_hp_lost * cfg["own_hp_damaged"]

    return reward
```

#### 4.9 新增辅助方法

```python
def _find_tower_owner(self, tower_id: int) -> int:
    """根据 tower_id 查找塔属于哪个玩家。"""
    for p in (0, 1):
        for tower in self.state.towers[p]:
            if tower.id == tower_id:
                return p
    return -1  # 理论上不会发生

def _find_ant_player(self, ant_id: int) -> int:
    """根据 ant_id 查找蚂蚁属于哪个玩家。"""
    for p in (0, 1):
        for ant in self.state.ants[p]:
            if ant.id == ant_id:
                return p
    return -1
```

#### 4.10 保留旧方法（兼容性）

将旧的 `_round_rewards()` 重命名为 `_round_rewards_legacy()`（保留，不与 v2 冲突），新方法命名为 `_round_rewards_v2()`。

---

## 5. 修改 4：PPO 训练配置

### 文件：`ppo/src/ppo_antwar/trainer/selfplay.py`（或等效的训练配置）

**当前 PPO 配置（相对裁剪）**：
```python
clip_eps_vf = 0.2    # values 相对裁剪
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.15
```

**建议修改**：降低 `ent_coef`，因为新系统过程奖励稠密，不需要依赖高熵探索来发现信号。

```python
# 建议调整
ent_coef = 0.05        # 从 0.15 → 0.05（更多信号 = 更少探索需求）
# clip_eps_vf = 0.2   # 保持（相对裁剪已足够）
# gamma = 0.99         # 保持
# gae_lambda = 0.95    # 保持
```

**关于 values_clip**：
- 设计文档 §5.5 建议将 `values_clip_range` 扩大至 200.0，但当前 PPO 实现使用的是**相对裁剪** `clip_eps_vf`，不是绝对范围裁剪
- `clip_eps_vf=0.2` 的含义：`clipped_value = old_value + clamp(value - old_value, -0.2×old_value, +0.2×old_value)`
- 这意味着 values 可以平滑地从 0 拓展到 ±100，不需要修改此参数
- **结论：不需要修改 values_clip 相关参数**

---

## 6. 改动影响范围分析

| 组件 | 影响 | 风险 |
|------|:---:|:---:|
| `model.py` | 新增 `RoundEvents` 类 + `GameState` 加字段 | 低 — 纯新增数据结构 |
| `engine.py` | 在 5-6 个方法中插入事件记录 | 中 — 需要在正确位置插入，不改变原有逻辑 |
| `env.py` | 重写 `_round_rewards()`、新增 3 个私有方法 | 中 — 核心逻辑改动，需验证数值正确 |
| PPO trainer | 微调 `ent_coef` | 低 — 单参数调整 |

---

## 7. 实现顺序建议

```
Step 1: model.py — 新增 RoundEvents 类 + GameState.round_events 字段
Step 2: engine.py — 插入事件记录（逐个方法：_attack_ants → _move_ants → ...)
Step 3: env.py  — 添加 REWARD_CONFIG + 新增 _action_rewards()
Step 4: env.py  — 新增 _event_rewards() + _round_rewards_v2()
Step 5: env.py  — 修改 _step_single() 集成 v2 奖励
Step 6: PPO config — 微调 ent_coef
Step 7: 验证 — 对一个对局采集完整 reward trace，核对数值
```

每个 Step 完成后可独立验证。Step 2 和 Step 3-4 可以部分并行（env 的 `_action_rewards()` 不依赖 engine 事件）。

---

## 8. 验证方案

### 8.1 单元验证

挂载 v2 奖励系统，让两个随机 agent 对战一局，采集每个 step 的：

```python
print(f"step={step}, player={p}, action={bundle.name}, "
      f"action_reward={action_r:.3f}, round_reward={round_r:.3f}, "
      f"event_reward={event_r:.3f}, total={total_r:.3f}")
```

### 8.2 检查项

- [ ] 每个 step 的 reward 不再大面积为 0（非零比例 > 80%）
- [ ] NO_OP 连续 3 步以上出现渐进负奖励
- [ ] 建塔后 step 立即获得正奖励
- [ ] 蚂蚁破基地后出现 +1.0 效果奖励
- [ ] 塔击杀蚂蚁后出现 +0.2 效果奖励
- [ ] 终局 ±100 正常触发
- [ ] 单步 reward 不超过 [-20, +20]
- [ ] 总和过程 reward 远小于 ±100

---

## 9. 回滚方案

如果 v2 奖励系统导致训练不稳定，可快速回滚：

```python
class AntWarSequentialEnv:
    _USE_V2_REWARDS = True  # 改为 False 即回退旧系统
    
    def _step_single(self, action):
        # ...
        if self._USE_V2_REWARDS:
            rewards = self._round_rewards_v2()
            rewards[agent] += self._action_rewards(...)
        else:
            rewards = self._round_rewards_legacy()
```

---

> 本文档基于 `ppo_v1_reward_system_design.md` 的完整设计编写，所有数值均可追溯至设计文档的 §2-§6。实现过程中如有偏差，以设计文档为准。
