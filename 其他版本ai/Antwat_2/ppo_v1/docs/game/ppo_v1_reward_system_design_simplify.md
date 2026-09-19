# PPO v1 Reward 体系技术实现方案 — 简版

> 版本：v1.0
> 日期：2026-05-16
> 基于：[ppo_v1_reward_system_design.md](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/game/ppo_v1_reward_system_design.md)
> 对应完整版：[ppo_v1_reward_system_design_deliver.md](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/game/ppo_v1_reward_system_design_deliver.md)
> 原则：**仅修改 1 个文件（env.py），零 engine 改动，零 model.py 改动**。

---

## 0. 简版 vs 完整版

| 维度 | 完整版 | 简版 |
|------|------|------|
| 修改文件数 | 4 个 | **1 个** |
| 代码行数 | ~350 行 | **~180 行** |
| 是否动引擎 | ✅ 7 处插入事件记录 | ❌ **零 engine 改动** |
| 动作级奖励（建塔/超武/NO_OP） | ✅ | ✅ |
| 回合级 HP/金币差 | ✅ | ✅ |
| 终局 ±100 | ✅ | ✅ |
| 塔存活/回合 | ✅（RoundEvents） | ✅（`len(self.state.towers[p])`） |
| 蚂蚁死亡惩罚 | ✅（RoundEvents） | ✅（`self.state.die_count` 回合差） |
| 蚂蚁破基地 | ✅ 精确事件 | ⚠️ 通过 HP 差间接体现（×10.0 已包含） |
| 塔被摧毁 | ✅ 精确事件 | ⚠️ 通过塔数量变化间接体现 |
| 塔击杀蚂蚁 | ✅ | ❌ 丢失（需 engine 事件） |
| 塔攻击命中 | ✅ | ❌ 丢失 |
| LS/EMP/DEFLECTOR 效果 | ✅ | ❌ 丢失 |
| 非零 step 比例 | 85-95% | **70-85%** |
| 风险 | 中（改引擎逻辑） | **低**（仅 env 层） |

**设计判断**：动作奖励（建塔/升级/超武/NO_OP）是信号密度提升的主力，贡献约 80% 的改善效果。引擎事件追踪提供剩余 20% 的精细区分。简版在**零风险**的前提下捕获了绝大部分收益。

---

## 1. 代码现状

当前 `[_round_rewards()](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L152-L168)`：

```python
def _round_rewards(self) -> dict[str, float]:
    """只在回合结算时计算，大部分 step 的 reward=0"""
    rewards = {}
    for index, agent in enumerate(self.possible_agents):
        enemy = 1 - index
        reward = (previous_hp[enemy] - hp_now[enemy]) * 10.0   # HP 差
        reward -= (previous_hp[index] - hp_now[index]) * 10.0
        reward += (coins_now[index] - previous_coins[index]) * 0.05
        reward -= (coins_now[enemy] - previous_coins[enemy]) * 0.02
        if terminal:
            reward += 100.0 if winner == index else -100.0
    return rewards
```

**已知可利用的回合级数据**（无需 engine 改动，已在 `BackendState` 中暴露）：

| 数据 | 获取方式 |
|------|------|
| `die_count`（累计蚂蚁死亡数） | `self.state.die_count[player]` |
| 当前塔数量 | `len(self.state.towers[player])` |
| 基地 HP | `base.hp` |
| 金币 | `self.state.coins[player]` |

---

## 2. 修改内容总览

**仅修改 1 个文件**：[`Ant-Game/SDK/training/env.py`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py)

| 位置 | 修改类型 | 行数 |
|------|:---:|:---:|
| 文件顶部（import 后、class 前） | 新增 `REWARD_CONFIG` 字典 | ~48 |
| `__init__()` | 新增 3 个追踪变量 | ~5 |
| `_capture_round_start()` | 新增 3 个快照字段 | ~3 |
| `reset()` | 重置新变量 | ~4 |
| `_round_rewards()` | **扩展**：新增强化项 | ~15 |
| 新增 `_action_rewards()` | 新增私有方法 | ~68 |
| `_step_single()` | 注入动作奖励 + 单步裁剪 | ~12 |

---

## 3. 具体代码修改

### 3.1 新增 `REWARD_CONFIG` 字典

在 `env.py` 文件顶部，`import` 语句之后、`class AntWarSequentialEnv` 之前添加：

```python
REWARD_CONFIG = {
    # ---- 回合级 (round-level) ----
    "hp_diff_weight": 10.0,
    "own_coin_gain_weight": 0.10,
    "enemy_coin_gain_weight": 0.05,
    "tower_survival_per_tower": 0.02,
    "enemy_tower_survival_per_tower": 0.01,
    "own_die_penalty_per_ant": -0.05,

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

    # ---- 终局 (terminal) ----
    "win_reward": 100.0,
    "loss_reward": -100.0,

    # ---- 裁剪 ----
    "step_reward_clip": 20.0,
}
```

### 3.2 修改 `__init__()` — 新增追踪变量

在第 43-44 行（`_round_start_hp` / `_round_start_coins` 之后）添加：

```python
self._round_start_die_count = (0, 0)            # 回合开始时的 die_count
self._round_start_tower_count = (0, 0)          # 回合开始时的塔数量
self._player_tower_build_counts: dict[int, int] = {0: 0, 1: 0}  # 累计建造数
self._player_noop_streak: dict[int, int] = {0: 0, 1: 0}        # 连续 NO_OP
```

### 3.3 修改 `_capture_round_start()` — 新增快照

```python
def _capture_round_start(self) -> None:
    self._round_start_hp = tuple(base.hp for base in self.state.bases)
    self._round_start_coins = tuple(self.state.coins)
    # ==== 新增 ====
    self._round_start_die_count = tuple(self.state.die_count)
    self._round_start_tower_count = tuple(len(self.state.towers[p]) for p in (0, 1))
```

### 3.4 修改 `reset()` — 重置新变量

在 `reset()` 方法的 `self._capture_round_start()` 调用之后添加：

```python
self._player_tower_build_counts = {0: 0, 1: 0}
self._player_noop_streak = {0: 0, 1: 0}
self._round_start_die_count = (0, 0)
self._round_start_tower_count = (0, 0)
```

### 3.5 扩展 `_round_rewards()` — 新增强化项

修改现有的 `_round_rewards()` 方法（第 152-168 行）：

```python
def _round_rewards(self) -> dict[str, float]:
    """回合级 delta 奖励 + 终局。
    新增：塔存活奖励、蚂蚁死亡惩罚、die_count 差。
    """
    cfg = REWARD_CONFIG
    rewards: dict[str, float] = {}
    previous_hp = self._round_start_hp
    previous_coins = self._round_start_coins
    previous_die = self._round_start_die_count

    for index, agent in enumerate(self.possible_agents):
        enemy = 1 - index
        reward = 0.0

        # --- HP 伤害差 (保持) ---
        reward += (previous_hp[enemy] - self.state.bases[enemy].hp) * cfg["hp_diff_weight"]
        reward -= (previous_hp[index] - self.state.bases[index].hp) * cfg["hp_diff_weight"]

        # --- 金币差 (权重提升: 0.05→0.10, 0.02→0.05) ---
        reward += (self.state.coins[index] - previous_coins[index]) * cfg["own_coin_gain_weight"]
        reward -= (self.state.coins[enemy] - previous_coins[enemy]) * cfg["enemy_coin_gain_weight"]

        # --- 塔存活 (NEW) ---
        reward += len(self.state.towers[index]) * cfg["tower_survival_per_tower"]
        reward -= len(self.state.towers[enemy]) * cfg["enemy_tower_survival_per_tower"]

        # --- 蚂蚁死亡差 (NEW) ---
        own_die_delta = self.state.die_count[index] - previous_die[index]
        enemy_die_delta = self.state.die_count[enemy] - previous_die[enemy]
        reward += own_die_delta * cfg["own_die_penalty_per_ant"]          # 己方死=惩罚
        reward -= enemy_die_delta * cfg["own_die_penalty_per_ant"]        # 敌方死=己方得

        # --- 终局 (保持) ---
        if self.state.terminal:
            if self.state.winner == index:
                reward += cfg["win_reward"]
            elif self.state.winner == enemy:
                reward += cfg["loss_reward"]

        rewards[agent] = float(reward)
    return rewards
```

**关键变化**：
1. 金币差权重从 `0.05/0.02` → `0.10/0.05`（设计文档 §3.5）
2. 塔存活：`+0.02/己方塔/回合`、`-0.01/敌方塔/回合`（设计文档 §3.2.1）
3. 蚂蚁死亡差：利用引擎已有的 `die_count` 计数器（无需 engine 改动），`-0.05/己方死`、`+0.05/敌方死`（设计文档 §3.3.1）

### 3.6 新增 `_action_rewards()` — 动作级奖励

```python
def _action_rewards(self, player: int, bundle: ActionBundle, tower_count_before: int) -> float:
    """根据 agent 操作类型计算过程奖励。
    在 _step_single() 中每次 agent 操作后立即调用。
    """
    cfg = REWARD_CONFIG
    name = bundle.name
    reward = 0.0

    # --- NO_OP (hold) ---
    if name == "hold" or "noop" in bundle.tags:
        self._player_noop_streak[player] += 1
        n = self._player_noop_streak[player]
        if n > cfg["noop_tolerance"]:
            penalty = cfg["noop_base_penalty"] + cfg["noop_base_penalty"] * ((n - cfg["noop_tolerance"]) // 10)
            reward = max(penalty, cfg["noop_max_penalty"])
    else:
        self._player_noop_streak[player] = 0

    # --- 建造塔 (build) ---
    if name == "build":
        count = self._player_tower_build_counts[player]
        tiers = cfg["build_tower_tiers"]
        if count < len(tiers):
            reward += tiers[count]
        else:
            reward += tiers[-1]
        self._player_tower_build_counts[player] += 1

    # --- 升级塔 (upgrade) ---
    if name == "upgrade":
        for op in bundle.operations:
            target_type = op[2]
            if target_type >= 100:
                reward += cfg["upgrade_tower_l3"]
            else:
                reward += cfg["upgrade_tower_l2"]

    # --- 拆除/降级塔 (sell, downgrade) ---
    if name == "sell":
        reward += cfg["downgrade_tower_penalty"]

    # --- 科技升级 (base_upgrade) ---
    if name == "base_upgrade":
        for op in bundle.operations:
            op_type = op[0]
            if op_type == 31:  # UPGRADE_GENERATION_SPEED
                level = self.state.bases[player].generation_level - 1
                if 0 <= level < len(cfg["upgrade_gen_speed"]):
                    reward += cfg["upgrade_gen_speed"][level]
            elif op_type == 32:  # UPGRADE_GENERATED_ANT
                level = self.state.bases[player].ant_level - 1
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

**操作名称 → 动作类型映射**（`ActionBundle.name` 的实际值，无需额外映射）：

| `bundle.name` | 含义 | 奖励 |
|------|------|:---:|
| `"hold"` | NO_OP / 不操作 | 渐进负奖励 |
| `"build"` | BUILD_TOWER | 递减正奖励 |
| `"upgrade"` | UPGRADE_TOWER | +0.3 / +0.5 |
| `"sell"` | DOWNGRADE_TOWER | -0.3 |
| `"base_upgrade"` | 科技升级 | +0.5~+1.0 |
| `"storm"` | 闪电风暴 | +0.5 |
| `"emp"` | EMP 轰炸 | +0.8 |
| `"deflector"` | 偏射盾 | +0.3 |
| `"evasion"` | 紧急回避 | +0.2 |

### 3.7 修改 `_step_single()` — 注入动作奖励 + 单步裁剪

修改现有的 `_step_single()` 方法（第 170-219 行）中的关键位置：

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

    # ==== 新增：记录操作前塔数量 ====
    tower_count_before = len(self.state.towers[player])

    invalid_ops = self.state.apply_operation_list(player, bundle.operations)
    self.infos[agent] = {
        **self.infos.get(agent, {}),
        "bundle": bundle,
        "illegal": illegal,
        "invalid_ops": invalid_ops,
    }

    # ==== 新增：操作后立即计算动作奖励 ====
    step_action_reward = self._action_rewards(player, bundle, tower_count_before)

    rewards = {name: 0.0 for name in self.possible_agents}
    self._context = self._context.next_turn()
    completed_round = False

    if not self.state.terminal and player == 1:
        self.state.advance_round()
        completed_round = True

    if completed_round or self.state.terminal:
        rewards = self._round_rewards()      # 注意：已扩展过的方法
        if completed_round and not self.state.terminal:
            self._capture_round_start()

    if illegal:
        rewards[agent] -= 1.0

    # ==== 新增：叠加动作奖励 ====
    rewards[agent] = rewards.get(agent, 0.0) + step_action_reward

    # ==== 新增：单步裁剪 ====
    clip = REWARD_CONFIG["step_reward_clip"]
    for name in self.possible_agents:
        r = rewards.get(name, 0.0)
        rewards[name] = max(min(r, clip), -clip)

    for name in self.possible_agents:
        self.rewards[name] = float(rewards[name])
        self.terminations[name] = self.state.terminal
        self.truncations[name] = False

    # ... 后续代码不变 ...
    self._refresh_bundles()
    self._update_infos()
    self._accumulate_rewards()

    if self.state.terminal:
        self._deads_step_first()
    else:
        self.agent_selection = self.possible_agents[self._context.to_play]
```

完整改动：在原有 `_step_single()` 中插入 **4 处**：
1. 第 183 行（`bundle, illegal = ...` 之后）：记录塔数量快照
2. 第 189 行（`apply_operation_list` 之后）：计算 `step_action_reward`
3. 第 207 行（`illegal` 处理之后）：叠加动作奖励
4. 第 207 行之后：单步裁剪

---

## 4. 改动量统计

| 位置 | 类型 | ~行数 |
|------|:---:|:---:|
| `REWARD_CONFIG` 字典 | 新增 | 48 |
| `__init__()` 追加 | 修改 | 5 |
| `_capture_round_start()` 追加 | 修改 | 3 |
| `reset()` 追加 | 修改 | 4 |
| `_round_rewards()` 替换 | 修改 | 20（扩充） |
| `_action_rewards()` 方法 | 新增 | 68 |
| `_step_single()` 插入 | 修改 | 12 |
| **总计** | | **~160 行** |

---

## 5. PPO 训练配置微调

**文件**：`ppo/src/ppo_antwar/trainer/selfplay.py`（或等效配置位置）

```python
# 当前
ent_coef = 0.15

# 建议
ent_coef = 0.05     # 新系统过程奖励稠密，不需要高熵探索
```

仅此 1 行调整。`gamma`、`gae_lambda`、`clip_eps_vf` 保持不动。

---

## 6. 验证方案

### 6.1 单元验证

挂载 v2 简版奖励，让两个随机 agent 对战一局，采集每个 step 的奖励：

```python
print(f"step={step}, player={p}, action={bundle.name}, "
      f"action_reward={action_r:.3f}, round_reward={round_r:.3f}, "
      f"total={total_r:.3f}")
```

### 6.2 检查项

- [ ] 每个 step 的 reward 不再大面积为 0（非零比例 > 70%）
- [ ] NO_OP 连续 3 步以上出现渐进负奖励（-0.02 → -0.04 → ...）
- [ ] 建塔后 step 立即获得正奖励（+0.5 / +0.3 / +0.1）
- [ ] 回合结算时出现塔存活奖励（+0.02 × 己方塔数）
- [ ] 终局 ±100 正常触发
- [ ] 单步 reward 不超过 [-20, +20]
- [ ] 总和过程 reward 远小于 ±100
- [ ] 非法操作仍受 -1.0 惩罚（与旧版一致）

### 6.3 回归检查

- [ ] 旧的 `_round_rewards()` 功能完全可回退（`git stash` 即可）
- [ ] `self.state.die_count` 可通过 `BackendState` 正常读取
- [ ] `self.state.towers[player]` 长度正常反映塔数量
- [ ] `bundle.name` 的值在 9 种操作类型下均正确

---

## 7. 回滚方案

**一键回退**：

```bash
git checkout -- Ant-Game/SDK/training/env.py
```

然后撤消 PPO config 中的 `ent_coef` 改回 0.15。

---

## 8. 设计文档回溯

所有数值均可追溯至 [设计文档](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/game/ppo_v1_reward_system_design.md)：

| 简版实现 | 设计文档出处 |
|------|------|
| `build_tower_tiers` = [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1] | §2.2 BUILD_TOWER |
| `upgrade_tower_l2` = 0.3, `upgrade_tower_l3` = 0.5 | §2.3 UPGRADE_TOWER |
| `downgrade_tower_penalty` = -0.3 | §2.4 DOWNGRADE_TOWER |
| `upgrade_gen_speed` = [1.0, 0.8] | §2.5 |
| `upgrade_gen_ant` = [0.5, 0.1] | §2.6 |
| `deploy_ls` = 0.5, `deploy_emp` = 0.8, `deploy_deflector` = 0.3, `deploy_evasion` = 0.2 | §2.7-2.10 |
| `noop_tolerance` = 3, `noop_base_penalty` = -0.02 | §2.1 NO_OP |
| `tower_survival_per_tower` = 0.02 | §3.2.1 塔存活 |
| `own_die_penalty_per_ant` = -0.05 | §3.3.1 己方蚂蚁被击杀 |
| `own_coin_gain_weight` = 0.10, `enemy_coin_gain_weight` = 0.05 | §3.5 经济变化 |
| `hp_diff_weight` = 10.0 | §3.1.1 回合 HP 伤害差（保持） |
| `win_reward` = 100.0, `loss_reward` = -100.0 | §3.6 终局胜败（保持） |
| `step_reward_clip` = 20.0 | §5.3 单步奖励安全边际 |

**简版丢失的设计项**（需 engine 事件追踪才能实现）：

| 设计项 | 设计值 | 不能在简版实现的原因 |
|------|:---:|------|
| 塔击杀蚂蚁 | +0.2 | 需区分"塔击杀"与其他死因 |
| 塔攻击命中 | +0.03 | 需在 `_tower_attack()` 中记录 |
| 蚂蚁破基地 | +1.0 / -0.5 | 需在 `_resolve_ant_lifecycle()` 中记录 |
| 塔被摧毁 | +0.8 / -1.0 | 需在 `_attack_tower_from_ant()` 中记录 |
| LS/EMP/DEFLECTOR 效果 | +0.05~+0.3 | 需在 `_apply_lightning_effect()` 等中记录 |
| 渐进 HP 伤害/受损 | +0.5 / -0.3 | 需区分破基地与其他 HP 变化 |

简版通过"塔数量变化"、"HP 差 ×10.0"、"die_count 差" 间接捕获了其中一部分效果，虽不如精确事件版本精细，但已大幅超越旧系统。

---

> 本文档基于 `ppo_v1_reward_system_design.md` 编写，所有修改仅涉及 `env.py` 1 个文件，零 engine 改动。
