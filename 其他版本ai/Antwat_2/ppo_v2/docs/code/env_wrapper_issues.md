# PPO v2 环境封装模块代码审查报告

审查日期：2026-06-10
审查范围：`ppo_v2/src/ppo_antwar/env/` 及 `ppo_v2/src/ppo_antwar/compat/adapters.py`

---

## 1. [CRITICAL] 科技升级奖励：level==1 分支返回错误的奖励配置键

**文件**: `antwar_env.py` 第 548 行、第 557 行

**问题描述**: `_compute_tech_upgrade_reward` 方法中，当 `generation_level == 1` 和 `ant_level == 1` 时，均返回了 `l1` 级别的奖励配置，而非 `l2`。这是典型的复制粘贴错误——`level == 1` 分支从 `level == 0` 分支复制过来后未修改奖励配置键。

当前代码（第 543-549 行）：
```python
if target_idx == 0:
    level = player_state.generation_level
    if level == 0:
        return REWARD_CONFIG["upgrade_gen_speed_l1"]  # 15.0 ✓
    elif level == 1:
        return REWARD_CONFIG["upgrade_gen_speed_l1"]  # 15.0 ← BUG，应为 l2
    elif level == 2:
        return REWARD_CONFIG["upgrade_gen_speed_l2"]  # 10.0
```

当前代码（第 552-559 行）：
```python
elif target_idx == 1:
    level = player_state.ant_level
    if level == 0:
        return REWARD_CONFIG["upgrade_gen_ant_l1"]  # 7.5 ✓
    elif level == 1:
        return REWARD_CONFIG["upgrade_gen_ant_l1"]  # 7.5 ← BUG，应为 l2
    elif level == 2:
        return REWARD_CONFIG["upgrade_gen_ant_l2"]  # 2.5
```

**影响**: 科技从 level 1 升级到 level 2 时，奖励被高估。generation_speed 多给 5.0（15.0 vs 10.0），ant 多给 5.0（7.5 vs 2.5）。这会导致模型在 level 1 → 2 的升级决策上收到错误的奖励信号，影响科技升级策略的学习。

**修复建议**:
```python
# 第 548 行
elif level == 1:
    return REWARD_CONFIG["upgrade_gen_speed_l2"]  # 10.0

# 第 557 行
elif level == 1:
    return REWARD_CONFIG["upgrade_gen_ant_l2"]  # 2.5
```

---

## 2. [HIGH] step() 在 terminated 时返回空 observations，违反 Gymnasium 接口契约

**文件**: `antwar_env.py` 第 132-139 行

**问题描述**: 当游戏终止（`terminated=True`）时，`step()` 返回 `observations = {}`（空字典），而非包含 `"self"` 和 `"opponent"` 键的标准结构。Gymnasium 接口要求 `step()` 返回的 observation 始终与 `reset()` 返回的结构一致，即使游戏已终止。许多训练框架依赖最终 observation 进行 value bootstrapping（GAE 计算），空字典会导致 KeyError。

当前代码：
```python
if not terminated:
    state = self._runtime.state
    observations = {
        "self": self.observation_encoder.encode(state, self_player),
        "opponent": self.observation_encoder.encode(state, 1 - self_player),
    }
else:
    observations = {}  # ← 违反 Gymnasium 接口
```

**影响**: 训练框架在处理 episode 最后一步时，访问 `observations["self"]` 或 `observations["self"]["board"]` 会抛出 KeyError，导致训练崩溃或需要特殊处理终止状态。

**修复建议**: 终止时也返回完整的 observation 结构：
```python
else:
    observations = {
        "self": self.observation_encoder.encode(state, self_player),
        "opponent": self.observation_encoder.encode(state, 1 - self_player),
    }
```
注意：此时 `state` 已在 `_resolve_turn_from_ops` 中推进过，需要确保 `state` 变量在 terminated 分支中仍可用。当前代码中 `state` 变量在该分支前已被赋值，只需移除 `else` 分支的空字典覆盖即可。

---

## 3. [MEDIUM] reset() 中 "self" 观测始终映射为 player 0，与 step() 不一致

**文件**: `antwar_env.py` 第 62-68 行

**问题描述**: `reset()` 方法硬编码 `"self"` 为 `obs_0`（player 0 的观测），但 `step()` 方法根据 `self_first` 参数动态决定 `"self"` 对应的玩家：`self_player = 0 if self_first else 1`。当 `self_first=False` 时，`reset()` 返回的初始观测（player 0 视角）与后续 `step()` 返回的观测（player 1 视角）不一致。

当前代码（reset）：
```python
obs_0 = self.observation_encoder.encode(state, 0)
obs_1 = self.observation_encoder.encode(state, 1)
observations = {
    "self": obs_0,      # ← 始终为 player 0
    "opponent": obs_1,  # ← 始终为 player 1
}
```

step() 中的逻辑：
```python
self_player = 0 if self_first else 1
observations = {
    "self": self.observation_encoder.encode(state, self_player),      # ← 可能是 player 1
    "opponent": self.observation_encoder.encode(state, 1 - self_player),
}
```

**影响**: 如果训练中 `self_first` 可以为 `False`，则 episode 的第一个观测与后续观测来自不同玩家视角，神经网络会收到不一致的输入，导致策略学习混乱。观测编码器可能对棋盘进行翻转（基于 player 视角），视角不一致等同于输入分布突变。

**修复建议**: `reset()` 方法需要接收或推断 `self_first` 参数，确保初始观测与 `step()` 使用相同的玩家映射：
```python
def reset(self, self_first: bool = True):
    ...
    self_player = 0 if self_first else 1
    observations = {
        "self": self.observation_encoder.encode(state, self_player),
        "opponent": self.observation_encoder.encode(state, 1 - self_player),
    }
    ...
```

---

## 4. [MEDIUM] step() 异常处理路径返回空 observations

**文件**: `antwar_env.py` 第 122-130 行

**问题描述**: `_resolve_turn_from_ops` 抛出异常时，`step()` 的 except 分支设置 `observations = {}`，随后因为 `terminated = True`，在下方 `if not terminated` 分支中再次被设为 `{}`。训练代码访问 `observations["self"]` 时会触发 KeyError。

当前代码：
```python
except Exception as e:
    logger.error(f"[AntWarEnv] _resolve_turn_from_ops failed: {e}")
    observations = {}  # ← 空字典
    reward_self = 0.0
    reward_opponent = 0.0
    terminated = True
    truncated = False
    info = {"round_index": 0, "error": str(e)[:200]}
```

**影响**: 与 Issue #2 类似，但触发条件是运行时异常。虽然在正常游戏中不应发生，但一旦发生，训练进程会因 KeyError 崩溃，而非优雅降级。

**修复建议**: 在异常路径中也尝试生成观测，或至少返回与正常终止一致的观测结构。可以尝试从当前状态编码：
```python
except Exception as e:
    logger.error(f"[AntWarEnv] _resolve_turn_from_ops failed: {e}")
    try:
        state = self._runtime.state
        observations = {
            "self": self.observation_encoder.encode(state, self_player),
            "opponent": self.observation_encoder.encode(state, 1 - self_player),
        }
    except Exception:
        # 如果连观测都无法生成，用零张量兜底
        observations = {
            "self": {"board": np.zeros((28, 19, 19), dtype=np.float32),
                     "global": np.zeros(33, dtype=np.float32),
                     "action_mask": np.ones(ACTION_DIM, dtype=np.float32)},
            "opponent": {"board": np.zeros((28, 19, 19), dtype=np.float32),
                         "global": np.zeros(33, dtype=np.float32),
                         "action_mask": np.ones(ACTION_DIM, dtype=np.float32)},
        }
    ...
```

---

## 5. [MEDIUM] 战斗奖励中塔伤害计算只遍历旧状态的塔 ID，遗漏当回合新建塔的伤害

**文件**: `antwar_env.py` 第 228-242 行

**问题描述**: `_compute_battle_rewards` 方法在计算 `tower_dmg` 和 `tower_dmg_received` 时，仅遍历 `old` 快照中已存在的塔 ID。如果某方在当前回合建造了新塔，且该塔在同一回合的战斗阶段受到了伤害，这部分伤害不会被计入奖励。

当前代码：
```python
tower_dmg = sum(
    max(0.0, old.opponent_state.tower_hp.get(k, 0) - new.opponent_state.tower_hp.get(k, 0))
    for k in old.opponent_state.tower_hp  # ← 只遍历旧塔 ID
)
```

**影响**: 新建塔在当回合受到的伤害不计入 `tower_dmg` 或 `tower_dmg_received`，导致奖励信号不完整。虽然在游戏中新建塔立即受伤的情况不常见，但在对抗激烈的局面下（如对方蚂蚁已接近建造位置），可能出现。值得注意的是，`_resolve_turn_from_ops` 中已计算了 mid→new 的塔伤害（第 376-383 行），但仅用于日志，未纳入实际奖励。

**修复建议**: 将塔伤害计算改为遍历 `new` 和 `old` 中所有塔 ID 的并集，或使用 mid 快照中的塔 HP 作为基准：
```python
all_tower_ids = set(old.opponent_state.tower_hp) | set(new.opponent_state.tower_hp)
tower_dmg = sum(
    max(0.0, old.opponent_state.tower_hp.get(k, 0) - new.opponent_state.tower_hp.get(k, 0))
    for k in all_tower_ids
)
```
注意：新建塔在 `old` 中 HP 为 0，所以 `max(0, 0 - new_hp)` 为 0（不会把新塔 HP 算作伤害），只有当新塔受伤导致 HP 从满血下降时才会被遗漏。因此更精确的做法是结合 mid 快照的塔 HP 来计算战斗阶段造成的伤害。

---

## 6. [LOW] 降级禁用回合数（1000）超过最大回合数（512），降级实际永远不可用

**文件**: `action_mask.py` 第 17 行

**问题描述**: `_DOWNGRADE_BAN_ROUNDS = 1000`，而 `MAX_ROUND = 512`。由于游戏最多只有 512 回合，降级操作在所有合法回合中都被禁用。代码注释声称"1000回合后降级操作恢复正常可用"，但该条件永远无法满足。

当前代码：
```python
_DOWNGRADE_BAN_ROUNDS = 1000  # MAX_ROUND = 512
```

注释：
```
# 1000回合后降级操作恢复正常可用。
```

**影响**: 降级操作实际永远被禁用，但动作空间中仍保留了 16 个降级动作维度（action_id 81-96），浪费了动作空间。注释具有误导性，可能让后续开发者误以为降级在后期可用。

**修复建议**: 如果确认降级在训练中不需要，应从 `TYPE_CONFIG` 中移除 `downgrade_tower` 并相应调整 `ACTION_DIM`，释放动作空间。如果后期需要开放降级，应将 `_DOWNGRADE_BAN_ROUNDS` 调整为小于 `MAX_ROUND` 的合理值，并修正注释。

---

## 7. [LOW] close() 清理 _tower_build_slot_counts 的方式与 reset() 不一致

**文件**: `antwar_env.py` 第 568 行

**问题描述**: `reset()` 将 `_tower_build_slot_counts` 重置为 `{0: {}, 1: {}}`（含两个玩家键的字典），但 `close()` 调用 `.clear()` 将其清空为 `{}`（空字典）。如果在 `close()` 后、`reset()` 前调用了 `_compute_build_tower_reward`，访问 `self._tower_build_slot_counts[player]` 会抛出 KeyError。

当前代码：
```python
# reset() 第 57 行
self._tower_build_slot_counts = {0: {}, 1: {}}

# close() 第 568 行
self._tower_build_slot_counts.clear()  # 结果: {} ← 缺少 0/1 键
```

**影响**: 正常使用流程中（先 reset 再 step），此问题不会触发。但如果存在 close 后未 reset 就调用 step 的异常路径，会导致 KeyError。

**修复建议**: 使 `close()` 与 `reset()` 保持一致：
```python
def close(self):
    self._runtime = None
    self._tower_build_slot_counts = {0: {}, 1: {}}
    self._noop_streak.clear()
```

---

## 8. [LOW] reward_opponent 未包含对手的动作奖励，零和假设被破坏

**文件**: `antwar_env.py` 第 113 行、第 119 行、第 401 行

**问题描述**: `step()` 只计算了 `reward_self_action`（己方动作奖励），没有计算对手的动作奖励。最终 `reward_opponent = -reward_self`（第 401 行），而 `reward_self` 已包含动作奖励，导致 `reward_self + reward_opponent ≠ 0`，零和假设被打破。

当前代码：
```python
reward_self_action = self._compute_action_reward(action_self, self_player)  # 只算己方
...
reward_self, reward_opponent, ... = self._resolve_turn_from_ops(...)
reward_self += reward_self_action  # 己方加了动作奖励
# reward_opponent 未加对手动作奖励
```

代码注释声明 `reward_opponent 仅用于日志（零和假设）`，但实际上并非零和。

**影响**: 如果有训练代码使用 `reward_opponent` 进行对手策略更新或对称性利用，会得到不正确的奖励信号。当前如果仅用于日志则影响有限，但注释与实现不一致可能误导后续开发。

**修复建议**: 要么修正注释去掉"零和"说法，要么补充对手动作奖励计算：
```python
reward_opponent_action = self._compute_action_reward(action_opponent, 1 - self_player)
reward_opponent = -reward_self + reward_opponent_action
```
