# PPO v1 全面 Reward 体系设计

> 版本：v1.0
> 日期：2026-05-16
> 基于游戏规则文档 `docs/design/antwar_rules.md` 及 SDK 常量 [constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/constants.py)

---

## 1. 设计目标

### 1.1 核心目标

| 目标 | 说明 |
|------|------|
| **Dense 化** | 每个 step 都有非零奖励信号，消灭 "reward=0 死循环" |
| **课程引导** | 奖励信号形成自然的学习阶梯：建塔 → 升级 → 战斗 → 取胜 |
| **防退化** | 杜绝 BUILD/DOWNGRADE 套利循环、NO_OP 停滞等退化策略 |
| **量级兼容** | 各项奖励量级协调，与 PPO 的 values [-10, 10] / returns 兼容 |

### 1.2 设计原则

1. **"做了什么" 比 "得到了什么" 先奖励**：过程奖励引导行为，终局奖励评估结果
2. **小额、高频、叠加**：每个事件独立给分，多处叠加形成明确梯度
3. **对称性**：敌我对称，正负对称，便于 GAE 计算 advantage
4. **可调节**：所有权重集中配置，方便调参实验

---

## 2. 游戏事件全景

基于游戏规则，可产生奖励信号的游戏事件分为四个层级：

```
                         ┌──────────────────────────┐
                         │     L4: 终局奖励          │
                         │  Win/Loss, HP Margin     │
                         ├──────────────────────────┤
                         │     L3: 重大事件奖励       │
                         │  蚂蚁破基地、超级武器击杀   │
                         ├──────────────────────────┤
                         │     L2: 战斗事件奖励       │
                         │  塔击杀蚂蚁、蚂蚁击杀蚂蚁   │
                         ├──────────────────────────┤
                         │     L1: 运营事件奖励       │
                         │  建塔、升级、科技、塔存活   │
                         └──────────────────────────┘
```

### 2.1 全事件清单

| 层级 | 事件 | 触发时机 | 当前有无奖励 | 游戏内效果 |
|:---:|------|---------|:---:|------|
| L1 | 建造防御塔 | BUILD_TOWER 成功 | ❌ | 金币 -cost |
| L1 | 升级防御塔 | UPGRADE_TOWER 成功 | ❌ | 金币 -60/-200 |
| L1 | 拆除防御塔 | DOWNGRADE_TOWER 成功 | ❌ | 金币 +refund |
| L1 | 升级科技 | UPGRADE_GENERATION_SPEED / UPGRADE_GENERATED_ANT | ❌ | 金币 -200/-250 |
| L1 | 部署超级武器 | USE_XXX 成功 | ❌ | 金币 -cost，开始冷却 |
| L1 | 塔存活 | 每回合检查 | ❌ | — |
| L2 | 塔攻击命中蚂蚁 | 每回合塔攻击 | ❌ | 蚂蚁 HP 减少 |
| L2 | 塔击杀蚂蚁 | 蚂蚁 HP ≤ 0 | ❌ | 金币 +6~18 |
| L2 | 蚂蚁击杀蚂蚁 | 蚂蚁互相攻击 | ❌ | 金币 +6~18 |
| L3 | 蚂蚁到达敌方基地 | 蚂蚁 status=SUCCESS | ❌ | 敌方 HP -1，金币 +10 |
| L3 | 闪电风暴伤害 | 每回合 LS 造成伤害 | ❌ | 蚂蚁 HP -20 |
| L3 | EMP 使塔失效 | EMP 范围内塔被禁用 | ❌ | 塔暂时失效 |
| L3 | 偏射盾保护蚂蚁 | 蚂蚁受到伤害被抵消 | ❌ | 伤害被抵消 |
| L4 | 游戏胜利 | 敌方 HP ≤ 0 | ✅ +100 | — |
| L4 | 游戏失败 | 己方 HP ≤ 0 | ✅ -100 | — |
| L4 | HP 伤害差 | 每回合 | ✅ ×10 | — |
| L4 | 金币变化差 | 每回合 | ✅ ×0.05/-0.02 | — |

> 当前仅 L4 级别有稀疏奖励（回合 HP/金币差 + 终局胜败），L1-L3 全部缺失。这是训练失败的根因。

---

## 3. 奖励体系架构

### 3.1 总体框架

每个 step 的最终奖励 = **回合奖励（每 2 step 一次）** + **事件奖励（每 step）** + **终局奖励（游戏结束时）**

```
R_total = R_round + R_events + R_terminal
```

### 3.2 奖励发放位置

| 奖励类型 | 发放位置 | 频率 |
|---------|---------|:---:|
| `R_round` | SDK `_round_rewards()`（回合完成时） | 每 2 step 1 次 |
| `R_events` | PPO `_collect_episode_with_swap()`（每 step 后） | 每 step |
| `R_terminal` | SDK `_round_rewards()` + 叠加 | 仅终局 |

---

## 4. L1 — 运营事件奖励（Dense 核心）

这是最关键的层级。当前训练失败的根因是 agent 连 "建第一座塔" 都学不会——因为没有奖励信号。

### 4.1 BUILD_TOWER 奖励

```
R_build = +1.0 × tower_value_factor
```

| 参数 | 值 | 说明 |
|------|:---:|------|
| 基础奖励 | **+1.0** | 每次成功建造 |
| tower_value_factor | `1.0 / (1.0 + 0.1 × tower_count)` | 边际递减（第 1 座塔价值 > 第 10 座） |

**设计理由**：
- +1.0 是适中的量级——足够驱动学习，但不至于让 agent 盲目建塔
- 边际递减防止 agent 为了刷奖励而无脑堆塔（费效比越来越差）
- 第 1 座塔 = +1.0，第 5 座塔 ≈ +0.67，第 10 座塔 = +0.5

**对抗 BUILD/DOWNGRADE 循环的关键**：拆除塔时有惩罚条款，见 §4.3。

### 4.2 UPGRADE_TOWER 奖励

```
R_upgrade = +2.0 (L1→L2) / +4.0 (L2→L3)
```

| 升级阶段 | 费用 | 奖励 | 奖励/费用比 |
|---------|:---:|:---:|:---:|
| BASIC → L2 (HEAVY/QUICK/MORTAR/PRODUCER) | 60 | **+2.0** | 0.033 |
| L2 → L3 (HEAVY_PLUS/ICE/SNIPER 等) | 200 | **+4.0** | 0.020 |

**设计理由**：
- 升级费用高昂（60/200），对应给予更高奖励以鼓励长期投资
- 奖励/费用比递减，防止无意义升级
- 升级是 agent "脱离新手村" 的标志性行为

### 4.3 DOWNGRADE_TOWER 惩罚

```
R_downgrade = -1.0
```

| 参数 | 值 | 说明 |
|------|:---:|------|
| 惩罚 | **-1.0** | 每次拆除塔 |

**设计理由**：
- 与 BUILD +1.0 形成对称，拆除的惩罚抵消建造的奖励
- 建造再立即拆除 → 净收益 ≈ +1.0 - 1.0 = 0，不再是套利
- 这是对抗 BUILD/DOWNGRADE 循环的核心机制

**注意**：在合理的战略场景下（如拆除低价值塔腾出位置），-1.0 的惩罚由后续建造更高价值塔的 +1.0~+4.0 奖励覆盖，整体净正。

### 4.4 科技升级奖励

```
R_tech_upgrade = +3.0 (L0→L1) / +5.0 (L1→L2)
```

| 升级 | 费用 | 奖励 | 效果 |
|------|:---:|:---:|------|
| 蚂蚁生产速度 L0→L1 | 200 | **+3.0** | 生成间隔 4.5→4.0 轮 |
| 蚂蚁生产速度 L1→L2 | 250 | **+5.0** | 生成间隔 4.0→3.5 轮 |
| 蚂蚁血量 L0→L1 | 200 | **+3.0** | 蚂蚁 HP 20→25 |
| 蚂蚁血量 L1→L2 | 250 | **+5.0** | 蚂蚁 HP 25→25（兵蚁 30） |

**设计理由**：
- 科技升级是全局性投资，影响深远，值得鼓励
- 奖励量级介于塔升级之间，反映其战略价值

### 4.5 超级武器部署奖励

```
R_superweapon = +2.0 (偏射盾/紧急回避) / +3.0 (闪电风暴) / +4.0 (EMP)
```

| 超级武器 | 费用 | 冷却 | 奖励 |
|---------|:---:|:---:|:---:|
| DEFLECTOR (偏射盾) | 60 | 25 | **+2.0** |
| EMERGENCY_EVASION (紧急回避) | 60 | 25 | **+2.0** |
| LIGHTNING_STORM (闪电风暴) | 90 | 35 | **+3.0** |
| EMP_BLASTER (EMP 轰炸) | 135 | 45 | **+4.0** |

**设计理由**：
- 奖励与费用成正比，但不完全线性（避免 agent 学会"有钱就乱放"）
- 超级武器是进阶玩法，L1 阶段的 agent 可能不会用——没关系，L1 建塔奖励已足够驱动初期学习

### 4.6 塔存活奖励（关键防退化机制）

```
R_tower_survive = +0.02 × Σ(tower_value_multiplier)  （每回合）
```

| 塔类型 | multiplier | 说明 |
|--------|:---:|------|
| BASIC | ×1.0 | 基础塔 |
| L2 塔 (HEAVY/QUICK/MORTAR/PRODUCER) | ×1.5 | 已升级 |
| L3 塔 (HEAVY_PLUS/ICE/SNIPER 等) | ×2.5 | 高级塔 |

**示例计算**：
- 拥有 2 座 BASIC + 1 座 HEAVY → 奖励 = 0.02 × (1.0 + 1.0 + 1.5) = **+0.07/回合**
- 拥有 5 座 L2+ 塔 → 奖励 = 0.02 × 7.5 = **+0.15/回合**

**设计理由**：
- 这是对抗 BUILD→立即 DOWNGRADE 循环的核心机制
- 如果建塔后立即拆除，则无法获得持续存活奖励
- 建塔 + 保持 → 建造奖励 + 持续存活奖励，总收益远超建造+拆除的零和
- 0.02/tower/round 是微量级——在 256 回合中，5 塔可累计约 25.6，有意义但不主导

---

## 5. L2 — 战斗事件奖励

### 5.1 塔击杀蚂蚁奖励

```
R_tower_kill = +0.3 (L0 蚂蚁) / +0.5 (L1 蚂蚁) / +0.7 (L2 蚂蚁) / +0.9 (兵蚁)
```

| 被击杀蚂蚁 | 游戏内金币 | 奖励 | 奖励/金币比 |
|-----------|:---:|:---:|:---:|
| L0 工蚁 | 6 | **+0.3** | 5% |
| L1 工蚁 | 10 | **+0.5** | 5% |
| L2 工蚁 | 14 | **+0.7** | 5% |
| 兵蚁 | 18 | **+0.9** | 5% |

**设计理由**：
- 击杀奖励比游戏内金币奖励低一个数量级，因为金币本身已通过 §6 的经济奖励间接体现
- 直接奖励 "击杀行为" 而非 "击杀收益"，让 agent 更快学会塔的防御价值

### 5.2 塔攻击命中奖励（可选增强）

```
R_tower_hit = +0.02 × damage_dealt / 10.0  （每次塔攻击命中蚂蚁）
```

| 塔类型 | 伤害 | 单次命中奖励 |
|--------|:---:|:---:|
| BASIC | 5 | **+0.01** |
| HEAVY | 12 | **+0.024** |
| SNIPER | 10 | **+0.02** |
| MISSILE | 18 | **+0.036** |

**设计理由**：
- 比击杀奖励更 dense——每次塔攻击都产生信号
- 量级极小（0.01~0.04），累积效果在 256 回合中约 1~3 分，作为辅助信号
- 可选——如果觉得过于细碎，可仅保留 §5.1 的击杀奖励

---

## 6. L3 — 重大事件奖励

### 6.1 蚂蚁到达敌方基地

```
R_ant_breach = +1.0  （己方蚂蚁到达敌方基地）
R_ant_breached = -0.5  （敌方蚂蚁到达己方基地）
```

| 事件 | 游戏效果 | 奖励 |
|------|---------|:---:|
| 己方蚂蚁到达敌方基地 | 敌方 HP -1，己方金币 +10 | **+1.0** |
| 敌方蚂蚁到达己方基地 | 己方 HP -1，敌方金币 +10 | **-0.5** |

**设计理由**：
- 游戏内已通过 HP 伤害（×10 = 10.0）给回合奖励，这里的事件奖励是增量
- 非对称设计（+1.0 vs -0.5）：进攻比防守更值得鼓励（主动行为）

### 6.2 超级武器效果奖励

```
R_ls_damage = +0.05 × ants_hit_count  （闪电风暴每命中 1 只蚂蚁）
R_emp_disable = +0.5 × towers_disabled  （EMP 每禁用 1 座敌方塔）
R_deflector_protect = +0.1 × attacks_blocked  （偏射盾每抵消 1 次伤害）
```

| 事件 | 奖励 |
|------|:---:|
| 闪电风暴命中蚂蚁 | **+0.05/只** |
| EMP 使敌方塔失效 | **+0.5/座** |
| 偏射盾抵消伤害 | **+0.1/次** |

**设计理由**：
- 超级武器部署已有 L1 奖励，这里是"打得准/用得好"的额外奖励
- 引导 agent 学会在正确时机（敌蚁群密集、敌塔集中）使用超级武器

---

## 7. L4 — 终局与回合奖励（改进）

### 7.1 终局胜败奖励（保持现有）

```
R_win = +100.0  （叠加在回合奖励上）
R_loss = -100.0
```

✅ 保持现有设计不变。±100 的量级在游戏后期（HP 接近耗尽时）合理。

### 7.2 回合 HP 伤害奖励（改进）

```
R_hp = （对敌方造成 HP 伤害 - 己方受到 HP 伤害）× 10.0
```

✅ 保持现有权重的 `×10.0`。

**新增**：渐进式 HP 伤害奖励：

```
R_hp_progressive = +0.5 × Δenemy_hp  当 Δenemy_hp > 0  （对敌方造成 HP 伤害）
R_hp_progressive = -0.3 × Δself_hp    当 Δself_hp > 0    （己方受到 HP 伤害）
```

**设计理由**：
- 现有 `×10.0` 权重对 1 点 HP 伤害 = 10.0，但对 0 伤害 = 0——梯度断裂
- `+0.5` 的渐进奖励降低了 "首次造成伤害" 的门槛，让 agent 更快收到正向信号

### 7.3 回合经济奖励（改进）

```
R_coin = (自身金币变化) × 0.10 - (敌方金币变化) × 0.05
```

| 参数 | 旧值 | 新值 | 变化 |
|------|:---:|:---:|:---:|
| 自身金币权重 | 0.05 | **0.10** | ×2 |
| 敌方金币权重 | 0.02 | **0.05** | ×2.5 |

**设计理由**：
- 原权重过低（0.045/round），淹没在 HP 伤害信号中
- 提高到 0.10 使经济管理成为可见的学习信号
- 每回合基础收入 1.5 金币 → 奖励 = 1.5 × 0.10 - 1.5 × 0.05 = **+0.075/回合**

---

## 8. 负向信号与防退化机制

### 8.1 NO_OP 序列惩罚

```
R_noop_penalty = -0.02 × max(0, consecutive_noop_steps - 5)
```

| 连续 NO_OP 步数 | 惩罚 |
|:---:|:---:|
| 0~5 | 0（容忍短暂的思考/等待） |
| 10 | -0.10 |
| 50 | -0.90 |
| 200 | -3.90 |

**设计理由**：
- 5 步容忍窗口：给 agent 正常的 "等待金币积累" 行为留空间
- 超过 5 步开始惩罚：防止纯 NO_OP 停滞
- 量级温和：-0.02/步不足以主导 loss，但清晰指示 "什么都不做是不好的"

### 8.2 塔被摧毁惩罚

```
R_tower_destroyed = -0.5 × tower_multiplier  （己方塔被敌方蚂蚁摧毁）
```

| 被摧毁的塔 | 惩罚 |
|-----------|:---:|
| BASIC | **-0.5** |
| L2 塔 | **-0.75** |
| L3 塔 | **-1.25** |

**设计理由**：
- 鼓励 agent 保护自己的塔（通过升级 HP 或建造更多塔防守）
- 惩罚量级不大——塔被摧毁是战争中的正常损耗，不应过度惩罚

---

## 9. 奖励发放实现方案

### 9.1 整体架构

```
                    ┌─────────────────────────┐
                    │  SDK env.step(action)    │
                    │  返回: rewards, info     │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  R_round       │  │  R_events     │  │  R_terminal   │
    │  (SDK 层)      │  │  (PPO 框架层)  │  │  (SDK 层)     │
    │  HP/金币差     │  │  建塔/击杀/.. │  │  Win/Loss     │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │  R_total = sum(above)    │
                    │  传入 PPO _ppo_update()  │
                    └─────────────────────────┘
```

### 9.2 事件检测方式

事件奖励需要在 `_collect_episode_with_swap()` 中通过比较 step 前后的 `GameState` 来检测：

```python
# 伪代码：每 step 后检测事件
def _compute_event_rewards(self, state_before, state_after, player, action_taken):
    reward = 0.0
    
    # L1: 运营事件（由 action 类型直接判定）
    if action_type == BUILD_TOWER and success:
        reward += BUILD_REWARD / (1.0 + 0.1 * tower_count_before)
    elif action_type == UPGRADE_TOWER and success:
        reward += UPGRADE_REWARDS[upgrade_level]
    elif action_type == DOWNGRADE_TOWER and success:
        reward += DOWNGRADE_PENALTY
    # ... 科技、超级武器类似
    
    # L2: 战斗事件（比较 state 差异）
    ants_killed_by_towers = count_ants_killed(state_before, state_after, player)
    reward += sum(ANT_KILL_REWARDS[a.level] * KILL_REWARD_RATIO for a in ants_killed_by_towers)
    
    # L3: 重大事件
    breaches = count_breaches(state_before, state_after, player)
    reward += breaches * BREACH_REWARD
    
    # 防退化
    if action_taken == NO_OP:
        self.consecutive_noop += 1
        if self.consecutive_noop > 5:
            reward += NOOP_PENALTY_RATE * (self.consecutive_noop - 5)
    else:
        self.consecutive_noop = 0
    
    return reward
```

### 9.3 配置参数汇总

```yaml
# 建议在 ppo_antwar.yaml 中新增的 reward 配置段
reward:
  # L1 运营事件
  build_tower_base: 1.0
  build_tower_decay: 0.1          # tower_value_factor 中的衰减系数
  upgrade_tower_l1_to_l2: 2.0
  upgrade_tower_l2_to_l3: 4.0
  downgrade_tower_penalty: -1.0
  tech_upgrade_l0_to_l1: 3.0
  tech_upgrade_l1_to_l2: 5.0
  superweapon_deflector: 2.0
  superweapon_evasion: 2.0
  superweapon_lightning: 3.0
  superweapon_emp: 4.0
  tower_survive_per_round: 0.02   # 每座塔每回合

  # L2 战斗事件
  tower_kill_ant_l0: 0.3
  tower_kill_ant_l1: 0.5
  tower_kill_ant_l2: 0.7
  tower_kill_ant_combat: 0.9
  tower_hit_per_10_damage: 0.02   # 可选，每次塔攻击命中

  # L3 重大事件
  ant_breach_enemy: 1.0           # 己方蚂蚁到达敌方基地
  ant_breach_self: -0.5           # 敌方蚂蚁到达己方基地
  lightning_storm_per_ant: 0.05
  emp_per_tower_disabled: 0.5
  deflector_per_block: 0.1

  # L4 终局与回合（改进现有）
  terminal_win: 100.0             # 保持
  terminal_loss: -100.0           # 保持
  hp_damage_weight: 10.0          # 保持
  hp_damage_progressive: 0.5      # 新增：渐进式 HP 奖励
  coin_self_weight: 0.10          # 从 0.05 提升
  coin_enemy_weight: -0.05        # 从 -0.02 提升（取负值）

  # 防退化
  noop_penalty_rate: -0.02        # 每步惩罚（超过 5 步后）
  noop_tolerance_steps: 5         # 容忍窗口
  tower_destroyed_basic: -0.5
  tower_destroyed_l2: -0.75
  tower_destroyed_l3: -1.25
```

---

## 10. 量级分析与兼容性

### 10.1 典型 256 回合对局的奖励分解

假设一场 PPO vs BasicTowerAI 的对局（agent 行为正常）：

| 奖励来源 | 频率 | 单次量级 | 累计量级 |
|---------|:---:|:---:|:---:|
| 建造 3 座塔 | 3 次 | +1.0, +0.91, +0.83 | **+2.74** |
| 升级 2 座塔 (L1→L2) | 2 次 | +2.0 | **+4.0** |
| 塔存活奖励 (3塔×150回合) | 150 | +0.06/回合 | **+9.0** |
| 塔击杀蚂蚁 (~20只) | 20 | +0.3~0.7 | **~8.0** |
| 蚂蚁到达基地 (5次) | 5 | +1.0 | **+5.0** |
| 回合 HP 伤害 (累计 20 HP) | — | ×10 | **+200.0** |
| 回合 HP 受损 (累计 15 HP) | — | ×(-10) | **-150.0** |
| 回合金币变化 | 128 | ~0.075 | **~9.6** |
| 终局胜利 | 1 | +100 | **+100.0** |
| **总计** | — | — | **~188.3** |

| 奖励类型 | 占比 |
|---------|:---:|
| 过程奖励 (L1+L2+L3) | ~15% |
| 回合 HP 净差 | ~27% |
| 终局胜败 | ~53% |
| 其他 | ~5% |

**关键结论**：过程奖励占比 ~15%，终局奖励约占 53%。过程奖励足够 dense 来驱动初期学习，终局奖励足够大来激励获胜行为。两者比例协调，不会出现 "过程奖励主导、agent 刷分不取胜" 的问题。

### 10.2 与 PPO Values 裁剪的兼容性

当前 values 裁剪为 `[-10, 10]`：

- 单步奖励最大估计：建塔 +1.0 + 击杀 +0.9 + breach +1.0 = **~2.9**（极少数回合）
- 单步奖励典型值：**0.01 ~ 0.3**（大部分回合）
- 终端回合奖励：**±100** + 回合奖励
- 经 GAE 折现后，returns 可达 ±100 量级

**建议**：将 `values_min/max` 从 `[-10, 10]` 调整为 `[-200, 200]`，以容纳终端奖励量级。此建议与 §3.6 训练参数分析一致。

### 10.3 与对手同质问题的关系

即使在 SelfPlay 初期（双方都是随机网络），过程奖励仍会产生非零信号：

| 场景 | 旧系统 reward | 新系统 reward |
|------|:---:|:---:|
| 双方纯 NO_OP | 0.0 | -0.02×(256-5) = **-5.02**（NO_OP 惩罚） |
| Agent 建塔 + 对方 NO_OP | 0.0 | +1.0 + 存活奖励 ≈ **+3~5** |
| BUILD→DOWNGRADE 循环 | ≈0 | +1.0 -1.0 + 存活0 = **≈0**（零和，非套利） |

**新系统从根本上消灭了 "reward 恒为 0" 的可能性。**

---

## 11. 实施建议

### 11.1 分阶段实施

| 阶段 | 内容 | 预期效果 |
|:---:|------|------|
| **Phase 1** | L1 运营奖励（建塔、升级、拆除惩罚、塔存活） | agent 学会建塔并维持 |
| **Phase 2** | + L2 战斗奖励（塔击杀蚂蚁） | agent 学会布局塔位以最大化击杀 |
| **Phase 3** | + L3 重大事件（蚂蚁破基地、超级武器） | agent 学会进攻和超级武器使用 |
| **Phase 4** | + 防退化机制（NO_OP 惩罚） | 兜底保护 |

### 11.2 关键实施位置

| 改动 | 文件 | 位置 |
|------|------|------|
| 事件检测 + R_events 叠加 | [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L574-L588) | `_collect_episode_with_swap()` |
| 回合奖励权重修改 | [env.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/training/env.py#L158-L161) | `_round_rewards()` |
| values 裁剪调整 | [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L30-L31) | `clip_values.values_min/max` |
| 新增 reward 配置 | [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml) | 新增 `reward:` 段 |

### 11.3 Phase 1 最小实施（最快见到效果）

仅实施以下 4 项，预计 50~100 episode 内 agent 就能学会建塔：

1. **BUILD_TOWER 奖励 +1.0**（边际递减）
2. **DOWNGRADE_TOWER 惩罚 -1.0**
3. **塔存活奖励 +0.02/tower/round**
4. **NO_OP 序列惩罚 -0.02/步**（超过 5 步后）

这四个改动相互配合：
- BUILD 正奖励 → 鼓励建塔
- DOWNGRADE 负惩罚 → 阻止套利循环
- 塔存活奖励 → 鼓励建了就不拆
- NO_OP 惩罚 → 什么都不做比做点什么更差

---

## 12. 附录：完整奖励信号速查表

| 事件 | 条件 | 奖励值 | 类型 |
|------|------|:---:|:---:|
| 建造防御塔 | BUILD_TOWER 成功 | +1.0 / (1 + 0.1×count) | L1 事件 |
| 升级塔 L1→L2 | UPGRADE_TOWER 到 2 级 | +2.0 | L1 事件 |
| 升级塔 L2→L3 | UPGRADE_TOWER 到 3 级 | +4.0 | L1 事件 |
| 拆除防御塔 | DOWNGRADE_TOWER 成功 | -1.0 | L1 惩罚 |
| 科技升级 L0→L1 | UPGRADE_SPEED/HP | +3.0 | L1 事件 |
| 科技升级 L1→L2 | UPGRADE_SPEED/HP | +5.0 | L1 事件 |
| 部署偏射盾/紧急回避 | USE_DEFLECTOR/EVASION | +2.0 | L1 事件 |
| 部署闪电风暴 | USE_LIGHTNING_STORM | +3.0 | L1 事件 |
| 部署 EMP | USE_EMP_BLASTER | +4.0 | L1 事件 |
| 塔存活 | 每座塔每回合 | +0.02 × multiplier | L1 持续 |
| 塔击杀 L0 蚂蚁 | tower kill | +0.3 | L2 事件 |
| 塔击杀 L1 蚂蚁 | tower kill | +0.5 | L2 事件 |
| 塔击杀 L2 蚂蚁 | tower kill | +0.7 | L2 事件 |
| 塔击杀兵蚁 | tower kill | +0.9 | L2 事件 |
| 塔命中蚂蚁 | 每次攻击命中 | +0.02 × dmg/10（可选） | L2 可选 |
| 蚂蚁到达敌方基地 | breach | +1.0 | L3 事件 |
| 被敌方蚂蚁到达 | breached | -0.5 | L3 事件 |
| 闪电风暴命中蚂蚁 | LS hit | +0.05 / 只 | L3 事件 |
| EMP 禁用敌方塔 | EMP disable | +0.5 / 座 | L3 事件 |
| 偏射盾抵消伤害 | deflector block | +0.1 / 次 | L3 事件 |
| 塔被摧毁 (BASIC) | tower destroyed | -0.5 | L3 惩罚 |
| 塔被摧毁 (L2) | tower destroyed | -0.75 | L3 惩罚 |
| 塔被摧毁 (L3) | tower destroyed | -1.25 | L3 惩罚 |
| NO_OP 连续 >5 步 | consecutive NO_OP | -0.02 / 步 | 防退化 |
| 对敌 HP 伤害 | 每回合 | +10.0 / HP | L4 回合 |
| 己方 HP 受损 | 每回合 | -10.0 / HP | L4 回合 |
| 对敌 HP 伤害（渐进） | 每回合，≥1 HP | +0.5 / HP | L4 回合 |
| 自身金币变化 | 每回合 | +0.10 / coin | L4 回合 |
| 敌方金币变化 | 每回合 | -0.05 / coin | L4 回合 |
| 游戏胜利 | 敌方 HP ≤ 0 | +100.0 | L4 终局 |
| 游戏失败 | 己方 HP ≤ 0 | -100.0 | L4 终局 |

---

> **文档结束。本设计基于 AntWar 完整游戏规则，覆盖从建塔到取胜的全事件链，提供 dense 化、分层的奖励信号。**
