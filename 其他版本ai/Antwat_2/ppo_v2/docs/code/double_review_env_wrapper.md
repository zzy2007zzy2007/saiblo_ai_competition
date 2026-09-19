# PPO V2 环境封装模块代码审查二次验证报告

二次审查日期：2026-06-10
审查对象：`ppo_v2/docs/code/env_wrapper_issues.md`
涉及源文件：
- `ppo_v2/src/ppo_antwar/env/antwar_env.py`（569 行）
- `ppo_v2/src/ppo_antwar/env/action_mask.py`（175 行）
- `ppo_v2/src/ppo_antwar/utils/action_constants.py`

项目原则参照：`.trae/rules/AntWar.md` + `.trae/rules/Principles.md`

---

## Issue 1: [CRITICAL] 科技升级奖励：level==1 分支返回错误的奖励配置键

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证（`antwar_env.py`）：

- **行 543-551**（generation_speed 分支）：
  ```python
  if target_idx == 0:
      level = player_state.generation_level
      if level == 0:
          return REWARD_CONFIG["upgrade_gen_speed_l1"]   # 15.0 ✓
      elif level == 1:
          return REWARD_CONFIG["upgrade_gen_speed_l1"]   # 15.0 ← BUG
      elif level == 2:
          return REWARD_CONFIG["upgrade_gen_speed_l2"]   # 10.0
  ```
  行 547-548：`level == 1` 分支返回 `upgrade_gen_speed_l1`（15.0），而非 `upgrade_gen_speed_l2`（10.0）。

- **行 552-560**（ant_level 分支）：
  ```python
  elif target_idx == 1:
      level = player_state.ant_level
      if level == 0:
          return REWARD_CONFIG["upgrade_gen_ant_l1"]    # 7.5 ✓
      elif level == 1:
          return REWARD_CONFIG["upgrade_gen_ant_l1"]    # 7.5 ← BUG
      elif level == 2:
          return REWARD_CONFIG["upgrade_gen_ant_l2"]    # 2.5
  ```
  行 556-557：`level == 1` 分支返回 `upgrade_gen_ant_l1`（7.5），而非 `upgrade_gen_ant_l2`（2.5）。

- **REWARD_CONFIG 确认**（`action_constants.py` 行 254-257）：
  ```python
  "upgrade_gen_speed_l1": 15.0,
  "upgrade_gen_speed_l2": 10.0,
  "upgrade_gen_ant_l1": 7.5,
  "upgrade_gen_ant_l2": 2.5,
  ```

- **设计文档确认**（`docs/guide/11_ppo_training_reward.md` 行 158-179）明确描述了 level→reward 的递减关系，与 `l1`/`l2` 键的语义一致。

这是典型的复制粘贴错误：`level == 1` 分支从 `level == 0` 复制后未修改配置键。

### 2. 问题级别是否准确

**✅ CRITICAL 准确。**

科技升级是核心策略决策之一。level 1→2 的升级奖励被高估（generation_speed 多 5.0，ant 多 5.0），直接扭曲了模型对升级时机的判断。此 bug 在每次升级到第二级时都会触发，频率高、影响面大。

### 3. 问题影响是否准确

**✅ 影响描述准确。**

- generation_speed：15.0 vs 10.0，多给 5.0（50% 偏差）
- ant：7.5 vs 2.5，多给 5.0（200% 偏差）
- 模型会倾向于过早将科技升到 level 2，因为奖励信号暗示 level 1→2 的价值等同于 level 0→1

### 4. 建议是否科学合理

**✅ 建议科学合理，完全符合项目原则。**

修复方案是最直接的代码修正，不涉及容错或兼容设计。

### 5. 具体修复方案

```python
# antwar_env.py 行 547-548，将：
elif level == 1:
    return REWARD_CONFIG["upgrade_gen_speed_l1"]

# 修改为：
elif level == 1:
    return REWARD_CONFIG["upgrade_gen_speed_l2"]

# antwar_env.py 行 556-557，将：
elif level == 1:
    return REWARD_CONFIG["upgrade_gen_ant_l1"]

# 修改为：
elif level == 1:
    return REWARD_CONFIG["upgrade_gen_ant_l2"]
```

### 6. 修复代价

**低。** 仅修改 2 行代码中的字符串字面量，无逻辑变更，无依赖影响。

### 7. 修复收益

**高。** 修正了每次科技升级到 level 2 时的奖励信号，直接影响模型对科技升级策略的学习质量。generation_speed 50% 偏差、ant 200% 偏差被彻底消除。

---

## Issue 2: [HIGH] step() 在 terminated 时返回空 observations，违反 Gymnasium 接口契约

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证（`antwar_env.py` 行 132-139）：
```python
if not terminated:
    state = self._runtime.state
    observations = {
        "self": self.observation_encoder.encode(state, self_player),
        "opponent": self.observation_encoder.encode(state, 1 - self_player),
    }
else:
    observations = {}
```

当 `terminated=True` 时，`observations = {}`，缺少 `"self"` 和 `"opponent"` 键。

**但需补充说明**：当前训练代码（`episode_collector.py` 行 107）在 `done=True` 时不再访问 `observations_new`，循环直接退出。因此当前训练流程不会因空 observations 而崩溃。但 PPO 训练中通常需要最后一步的 observation 来计算 terminal value bootstrap（GAE λ 的最后一步），如果后续实现这一标准逻辑，此 bug 将立即触发 KeyError。

### 2. 问题级别是否准确

**✅ HIGH 准确。**

虽然当前训练代码未受影响，但这是对 Gymnasium 接口契约的违反。任何依赖 final observation 的标准 PPO 实现（GAE value bootstrap）都会崩溃。作为环境封装层，接口合规性是基本要求。

### 3. 问题影响是否准确

**⚠️ 影响描述部分准确，需补充说明。**

原报告称"训练框架在处理 episode 最后一步时会 KeyError"，但当前训练代码实际不访问 terminated 时的 observation。影响更准确的描述是：**违反 Gymnasium 接口契约，阻碍后续引入标准 GAE value bootstrap**。

### 4. 建议是否科学合理

**⚠️ 建议方向正确，但需修正实现细节。**

原建议代码：
```python
else:
    observations = {
        "self": self.observation_encoder.encode(state, self_player),
        "opponent": self.observation_encoder.encode(state, 1 - self_player),
    }
```

原报告提到 `state` 变量在 terminated 分支中需要可用。验证源码：
- 行 91：`state = self._runtime.state`（在 step() 开头赋值）
- 行 115-130：try/except 块，如果 `_resolve_turn_from_ops` 成功，进入行 132 的 `if not terminated` 分支
- 行 132：`state = self._runtime.state`（再次从 runtime 获取）
- 行 138-139：else 分支中 `state` 变量仍然可用（行 91 赋值的那个）

**关键问题**：行 91 的 `state` 是回合推进前的状态，行 133 的 `state` 是回合推进后的状态。terminated 时应该返回回合推进后的观测（即 `_resolve_turn_from_ops` 中已调用 `state.advance_round()` 后的状态），所以应使用 `self._runtime.state` 而非行 91 的局部变量 `state`。

### 5. 具体修复方案

```python
# antwar_env.py 行 132-139，将：
if not terminated:
    state = self._runtime.state
    observations = {
        "self": self.observation_encoder.encode(state, self_player),
        "opponent": self.observation_encoder.encode(state, 1 - self_player),
    }
else:
    observations = {}

# 修改为：
state = self._runtime.state
observations = {
    "self": self.observation_encoder.encode(state, self_player),
    "opponent": self.observation_encoder.encode(state, 1 - self_player),
}
```

移除 `if not terminated` 分支，无论是否 terminated 都返回完整观测。`state = self._runtime.state` 在 terminated 后仍然可用（runtime 未被销毁）。

### 6. 修复代价

**低。** 删除 3 行、保留 3 行，逻辑简化，无副作用。

### 7. 修复收益

**高。** 恢复 Gymnasium 接口合规性，为后续引入 GAE value bootstrap 扫清障碍，同时使代码更简洁。

---

## Issue 3: [MEDIUM] reset() 中 "self" 观测始终映射为 player 0，与 step() 不一致

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证：

- **reset()**（行 62-68）：硬编码 `"self"` 为 `obs_0`（player 0 视角）
  ```python
  obs_0 = self.observation_encoder.encode(state, 0)
  obs_1 = self.observation_encoder.encode(state, 1)
  observations = {
      "self": obs_0,      # ← 始终为 player 0
      "opponent": obs_1,  # ← 始终为 player 1
  }
  ```

- **step()**（行 92）：动态映射
  ```python
  self_player = 0 if self_first else 1
  ```

- **self_first 可以为 False**：`selfplay.py` 行 571：
  ```python
  self_first = self._episode_count % 2 == 1
  ```
  偶数 episode 时 `self_first=False`，`self_player=1`。

- **观测编码器确实依赖 player 参数**：`observation.py` 行 47-56 中 `encode(state, player)` 会根据 player 生成不同的 board 特征、global 特征和 action_mask。player 0 和 player 1 的观测是不同的。

- **训练代码受影响**：`episode_collector.py` 行 76-77：
  ```python
  observations, _ = env.reset()
  obs_self, obs_opponent = self._setup_observations(observations)
  ```
  当 `self_first=False` 时，`obs_self` 是 player 0 视角，但后续 step() 返回的 `obs_self` 是 player 1 视角。第一个 step 的输入分布会突变。

### 2. 问题级别是否准确

**⚠️ MEDIUM 基本准确，但应提升为 HIGH。**

理由：
- 50% 的 episode（偶数 episode）受影响
- 观测视角不一致意味着神经网络在第一步收到完全不同的输入分布
- 这是环境封装的接口一致性问题，应属于环境本身的正确性缺陷
- 不过，游戏初始状态下 player 0 和 player 1 的对称性较高，实际观测差异可能小于中后期，影响有所缓解

维持 MEDIUM 也可接受，但建议团队关注。

### 3. 问题影响是否准确

**✅ 影响描述准确。**

视角不一致导致输入分布突变，影响策略学习。特别是不对称开局时影响更大。

### 4. 建议是否科学合理

**⚠️ 建议方向正确，但接口设计需调整。**

原建议给 `reset()` 添加 `self_first` 参数。但 `reset()` 是 Gymnasium 标准接口方法，不应添加自定义参数。

**更合理的方案**：在 `__init__` 或类属性中存储 `self_first` 配置，`reset()` 自动使用。或让环境在内部记住上一次 `self_first` 的值。

但考虑到项目原则"不做向前/向后兼容"，直接修改 `reset()` 签名也是可接受的，只要训练代码同步更新。但更简洁的方案是在 `reset()` 中接收参数，因为 self_first 是每局可变的。

### 5. 具体修复方案

```python
# antwar_env.py 修改 reset() 签名和实现：

def reset(self, self_first: bool = True) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """重置环境到初始状态，返回 (observations, info)

    Args:
        self_first: 己方是否映射为 player 0，默认 True
    """
    self._runtime = BackendAdapter.create_runtime(player=self._player_id)
    self._tower_build_slot_counts = {0: {}, 1: {}}
    self._noop_streak.clear()
    self._last_reward_detail = {}

    state = self._runtime.state
    self_player = 0 if self_first else 1
    observations = {
        "self": self.observation_encoder.encode(state, self_player),
        "opponent": self.observation_encoder.encode(state, 1 - self_player),
    }
    info = {"round_index": 0}
    return observations, info
```

训练代码同步修改（`episode_collector.py` 行 76）：
```python
# 原：
observations, _ = env.reset()

# 改为：
observations, _ = env.reset(self_first=self_first)
```

### 6. 修复代价

**低。** 修改 `reset()` 方法 3 行 + 训练代码 1 行，逻辑清晰无歧义。

### 7. 修复收益

**中。** 修复 50% episode 的观测不一致问题，保证训练数据的视角一致性。初始状态对称性较高，实际影响可能中等，但接口一致性是必要条件。

---

## Issue 4: [MEDIUM] step() 异常处理路径返回空 observations

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证（`antwar_env.py` 行 122-130）：
```python
except Exception as e:
    logger.error(f"[AntWarEnv] _resolve_turn_from_ops failed: {e}")
    # 返回安全默认值，防止训练崩溃
    observations = {}
    reward_self = 0.0
    reward_opponent = 0.0
    terminated = True
    truncated = False
    info = {"round_index": 0, "error": str(e)[:200]}
```

之后行 132-139 中 `terminated=True` 导致 `observations = {}` 再次覆盖。

### 2. 问题级别是否准确

**⚠️ MEDIUM 准确，但根据项目原则，此问题的本质不是"空 observations"，而是"吞掉异常"。**

项目原则明确指出："不做容错，让错误尽早暴露尽早解决"和"异常仍应合理捕获并记录，然后抛向上层"。当前代码在 except 中记录日志后返回默认值，实际上是在**吞掉异常**，违反了项目原则。

### 3. 问题影响是否准确

**⚠️ 影响描述需修正。**

原报告关注"空 observations 导致 KeyError"，但根据项目原则，更严重的问题是**异常被吞掉**——运行时错误被静默转换为 terminated=True + 零奖励 + 空 observations，训练代码会误以为游戏正常结束，但实际上环境状态已不一致。这比 KeyError 更危险，因为错误不会被暴露。

### 4. 建议是否科学合理

**❌ 原建议不符合项目原则。**

原建议在异常路径中尝试生成观测，甚至用零张量兜底。这是典型的容错设计，违反项目原则"不做容错"。

**根据项目原则，正确做法是**：记录异常日志后，将异常抛向上层，让调用方决定如何处理。`_resolve_turn_from_ops` 失败意味着环境状态不可恢复，不应假装一切正常。

### 5. 具体修复方案

```python
# antwar_env.py 行 115-130，将整个 try/except 替换为：
reward_self, reward_opponent, terminated, truncated, info = (
    self._resolve_turn_from_ops(self_ops, opponent_ops, self_first)
)
reward_self += reward_self_action
if "reward_detail" in info:
    info["reward_detail"]["action_reward"] = round(reward_self_action, 4)
```

移除 try/except，让 `_resolve_turn_from_ops` 的异常自然向上传播。如果确实需要对特定异常做日志记录后重抛，可以保留 except 但必须 re-raise：

```python
# 如需保留日志：
try:
    reward_self, reward_opponent, terminated, truncated, info = (
        self._resolve_turn_from_ops(self_ops, opponent_ops, self_first)
    )
except Exception as e:
    logger.error(f"[AntWarEnv] _resolve_turn_from_ops failed: {e}")
    raise
reward_self += reward_self_action
if "reward_detail" in info:
    info["reward_detail"]["action_reward"] = round(reward_self_action, 4)
```

### 6. 修复代价

**低。** 删除 try/except 块（或改为 re-raise），逻辑更简洁。

### 7. 修复收益

**高。** 遵循项目原则，让异常尽早暴露。避免环境状态不一致导致的静默错误传播，比返回空 observations 更安全。

---

## Issue 5: [MEDIUM] 战斗奖励中塔伤害计算只遍历旧状态的塔 ID

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证（`antwar_env.py` 行 228-242）：

```python
tower_dmg = sum(
    max(
        0.0,
        old.opponent_state.tower_hp.get(k, 0)
        - new.opponent_state.tower_hp.get(k, 0),
    )
    for k in old.opponent_state.tower_hp  # ← 只遍历旧塔 ID
)

tower_dmg_received = sum(
    max(
        0.0,
        old.self_state.tower_hp.get(k, 0) - new.self_state.tower_hp.get(k, 0),
    )
    for k in old.self_state.tower_hp  # ← 只遍历旧塔 ID
)
```

如果对手在当前回合建造了新塔，且该塔在同一回合的战斗阶段受到伤害：
- `old.opponent_state.tower_hp` 不含新塔 ID → 新塔的伤害被跳过
- 具体而言：新塔从满血 HP 被打到 lower_hp，实际伤害 = max_hp - lower_hp > 0，但由于 `k` 不在 old 中，此项未被遍历

**但需评估实际发生概率**：新塔在同一回合被攻击需要：1) 建造操作在本回合执行，2) 对方蚂蚁/武器恰好已在该位置附近。在游戏机制中，建塔和战斗在同一回合内顺序执行（先操作后战斗），因此新建塔确实可能被攻击。

### 2. 问题级别是否准确

**✅ MEDIUM 准确。**

新建塔立即被攻击的场景不常见但确实可能发生（尤其在对抗激烈的中后期），且奖励信号遗漏会导致策略偏差。

### 3. 问题影响是否准确

**✅ 影响描述准确。**

新建塔当回合受到的伤害不计入 `tower_dmg` 或 `tower_dmg_received`。不过原报告提到"`_resolve_turn_from_ops` 中已计算了 mid→new 的塔伤害（第 376-383 行），但仅用于日志"，这一点是正确的——行 376-383 的 `self_tower_dmg`/`opponent_tower_dmg` 确实只写入了 `_last_reward_detail`，未参与奖励计算。

### 4. 建议是否科学合理

**⚠️ 部分科学，但原建议的"遍历并集"方案有缺陷。**

原建议：
```python
all_tower_ids = set(old.opponent_state.tower_hp) | set(new.opponent_state.tower_hp)
tower_dmg = sum(
    max(0.0, old.opponent_state.tower_hp.get(k, 0) - new.opponent_state.tower_hp.get(k, 0))
    for k in all_tower_ids
)
```

**问题**：新建塔在 old 中 HP=0（`get(k, 0)`），new 中 HP=满血或受损。如果新塔未受损：`max(0, 0 - max_hp) = 0`，正确。如果新塔受损：`max(0, 0 - lower_hp) = 0`，**仍然为 0**，伤害未计入。因为旧值 0 比新值小，差值为负，被 `max(0, ...)` 截断。

**正确思路**：原报告自身也指出了这一点——"更精确的做法是结合 mid 快照的塔 HP 来计算战斗阶段造成的伤害"。mid 快照在操作执行后、回合推进前捕获，新建塔在 mid 中已存在（满血），因此 mid→new 的差值才能正确反映战斗伤害。

### 5. 具体修复方案

利用已有的 mid 快照来计算塔伤害，替换 old→new 的计算：

```python
# antwar_env.py _compute_battle_rewards 方法签名中添加 mid 参数：

def _compute_battle_rewards(
    self,
    old: StateSnapshot,
    new: StateSnapshot,
    terminated: bool,
    self_player: int,
    winner: Optional[int] = None,
    coins_self: Optional[float] = None,
    coins_opponent: Optional[float] = None,
    mid: Optional[StateSnapshot] = None,  # ← 新增参数
) -> Tuple[float, Dict[str, float]]:

# 行 228-242 替换为：

# 塔伤害使用 mid 快照（操作后、战斗前）作为基准
tower_base = mid if mid is not None else old
tower_dmg = sum(
    max(
        0.0,
        tower_base.opponent_state.tower_hp.get(k, 0)
        - new.opponent_state.tower_hp.get(k, 0),
    )
    for k in tower_base.opponent_state.tower_hp
)
tower_dmg_received = sum(
    max(
        0.0,
        tower_base.self_state.tower_hp.get(k, 0) - new.self_state.tower_hp.get(k, 0),
    )
    for k in tower_base.self_state.tower_hp
)
```

在 `_resolve_turn_from_ops` 中调用时传入 mid：
```python
# antwar_env.py 行 390-398 修改：
reward_self, components = self._compute_battle_rewards(
    old,
    new,
    terminated,
    self_player=self_player,
    winner=winner,
    coins_self=coins_self,
    coins_opponent=coins_opponent,
    mid=mid,  # ← 传入 mid 快照
)
```

### 6. 修复代价

**中。** 需修改 `_compute_battle_rewards` 方法签名、内部逻辑、调用方，以及所有直接调用此方法的位置（需检查是否还有其他调用方）。逻辑变复杂度不高但涉及接口变更。

### 7. 修复收益

**中。** 修复新建塔当回合伤害的遗漏问题，但此场景频率较低。更重要地，此修复使塔伤害计算逻辑与游戏阶段（操作→战斗）对齐，语义更清晰。

---

## Issue 6: [LOW] 降级禁用回合数（1000）超过最大回合数（512）

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证：

- `action_mask.py` 行 17：`_DOWNGRADE_BAN_ROUNDS = 1000`
- `action_constants.py` 行 132：`MAX_ROUND = 512`
- `action_mask.py` 行 38-42：`if round_idx < _DOWNGRADE_BAN_ROUNDS:` → 所有合法回合（0-511）中降级都被禁用
- 行 33-36 注释："1000回合后降级操作恢复正常可用" — 误导性注释

### 2. 问题级别是否准确

**⚠️ LOW 基本准确，但注释误导性应提升为 MEDIUM。**

降级永远禁用是**设计意图**（奖励设计文档也确认了这一点）。真正的风险是：注释暗示降级后期可用，可能误导后续开发者。但实际影响有限，因为降级惩罚（-9.0）已经足够强，即使后期开放也不会有模型使用。

### 3. 问题影响是否准确

**✅ 影响描述准确，但可补充。**

- 降级操作永远被禁用
- 动作空间中 16 个降级维度（action_id 81-96）始终为 0，浪费了动作空间
- 注释有误导性

### 4. 建议是否科学合理

**❌ 原建议"从 TYPE_CONFIG 中移除 downgrade_tower 并调整 ACTION_DIM"过于激进。**

移除降级动作维度会导致：
1. ACTION_DIM 变更，所有依赖此常量的代码需更新
2. 已训练模型的动作空间不兼容（违反项目原则"不做向前/向后兼容"，但如果确实不需要降级则可以接受）
3. 如果未来需要重新启用降级，需要再次修改 ACTION_DIM

**更简单的方案**：修正常量和注释即可，保持降级在动作空间中但永远禁用。这与当前行为一致，只修正了文档错误。

### 5. 具体修复方案

```python
# action_mask.py 行 17 和行 33-36，将：
_DOWNGRADE_BAN_ROUNDS = 1000

# 注释部分：
# ======================================================================
# [INTENTIONAL DESIGN] 前1000回合禁止降级防御塔
# 这是故意设计的行为，不是bug！
# 原因：防止模型在早期通过频繁降级防御塔来获取短期奖励，
# 从而避免模型学会"先建后拆"的投机策略。
# 1000回合后降级操作恢复正常可用。
# ======================================================================

# 修改为：
_DOWNGRADE_BAN_ROUNDS = 999  # 永远禁用降级（MAX_ROUND=512）

# 注释部分：
# ======================================================================
# [INTENTIONAL DESIGN] 永久禁用降级防御塔
# 这是故意设计的行为，不是bug！
# 原因：防止模型通过降级防御塔来获取短期奖励，
# 从而避免模型学会"先建后拆"的投机策略。
# 降级动作仍保留在动作空间中（action_mask 始终为 0），
# 如需启用降级，将 _DOWNGRADE_BAN_ROUNDS 设为小于 MAX_ROUND 的值。
# ======================================================================
```

### 6. 修复代价

**低。** 修改 1 个常量 + 1 段注释，无逻辑变更。

### 7. 修复收益

**低。** 消除误导性注释，避免后续开发者误解。功能行为不变。

---

## Issue 7: [LOW] close() 清理 _tower_build_slot_counts 的方式与 reset() 不一致

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证：

- `reset()` 行 57：`self._tower_build_slot_counts = {0: {}, 1: {}}`
- `close()` 行 568：`self._tower_build_slot_counts.clear()` → 结果为 `{}`

### 2. 问题级别是否准确

**✅ LOW 准确。**

根据项目原则"不做容错"，`close()` 后不应再调用 `step()`。如果 `close()` 后调用 `step()`，本就应抛出异常（`self._runtime` 为 None 时会报错）。因此 `_tower_build_slot_counts` 的状态不一致不会在正常流程中被触发。

### 3. 问题影响是否准确

**⚠️ 影响描述需修正。**

原报告称"如果存在 close 后未 reset 就调用 step 的异常路径，会导致 KeyError"。但根据项目原则，这种异常路径本就不该存在——调用方应保证先 reset 再 step。`close()` 后调用任何方法都应被视为调用方错误。

### 4. 建议是否科学合理

**⚠️ 建议可行但不必要。**

根据项目原则"不做容错"，`close()` 的语义是销毁环境，调用方不应在 `close()` 后继续使用环境。将 `.clear()` 改为 `= {0: {}, 1: {}}` 是一种防御性编程，与项目原则不完全一致。

但此修改代价极低，且保持 `close()` 和 `reset()` 的一致性有代码可读性价值（符合项目原则"高度关注代码可读性"），因此可以接受。

### 5. 具体修复方案

```python
# antwar_env.py 行 568，将：
self._tower_build_slot_counts.clear()

# 修改为：
self._tower_build_slot_counts = {0: {}, 1: {}}
```

### 6. 修复代价

**低。** 修改 1 行，无逻辑变更。

### 7. 修复收益

**低。** 提升代码可读性（close 与 reset 行为一致），但功能影响为零。

---

## Issue 8: [LOW] reward_opponent 未包含对手动作奖励，零和假设被破坏

### 1. 问题是否真实存在

**✅ 确认存在。**

源代码验证：

- 行 113：`reward_self_action = self._compute_action_reward(action_self, self_player)` — 只计算己方动作奖励
- 行 116-119：
  ```python
  reward_self, reward_opponent, terminated, truncated, info = (
      self._resolve_turn_from_ops(self_ops, opponent_ops, self_first)
  )
  reward_self += reward_self_action  # 己方加了动作奖励
  ```
- 行 400-401：
  ```python
  # reward_opponent 仅用于日志（零和假设）
  reward_opponent = -reward_self
  ```

数学验证：
- `reward_self` (after +=) = `reward_self_resolve` + `reward_self_action`
- `reward_opponent` = `-reward_self` = `-(reward_self_resolve + reward_self_action)`
- `reward_self + reward_opponent` = `reward_self_resolve + reward_self_action - reward_self_resolve - reward_self_action` = 0

**等一下！** 实际上 `reward_self + reward_opponent = 0` 是成立的！让我重新分析：
- `_resolve_turn_from_ops` 返回的 `reward_self_resolve` 是 `_compute_battle_rewards` 计算的值
- `_resolve_turn_from_ops` 返回的 `reward_opponent = -reward_self_resolve`（行 401）
- 但回到 `step()` 后，`reward_self += reward_self_action`（行 119），此时 `reward_self` 变了
- `reward_opponent` 仍然是 `-reward_self_resolve`（来自 `_resolve_turn_from_ops` 的返回值）
- 所以 `reward_self + reward_opponent = (reward_self_resolve + reward_self_action) + (-reward_self_resolve) = reward_self_action`

**结论：零和确实被破坏**，但偏差仅为 `reward_self_action`（动作奖励），不是整个 `reward_self`。原报告描述有误——原报告称 `reward_opponent = -reward_self`（行 401），但实际上行 401 的 `reward_self` 是 `_resolve_turn_from_ops` 内部的局部变量，不是 `step()` 中加了动作奖励后的 `reward_self`。

### 2. 问题级别是否准确

**✅ LOW 准确。**

`reward_opponent` 在当前训练代码中仅用于日志（`episode_collector.py` 行 167：`f"r_opp={reward_opponent:.4f}"`），不参与任何训练计算。零和偏差仅影响日志准确性。

### 3. 问题影响是否准确

**⚠️ 影响描述需修正。**

原报告称 `reward_self + reward_opponent ≠ 0`，这是正确的，但偏差量仅为 `reward_self_action`（动作奖励，通常 0-15.0），而非整个 `reward_self`。原报告的措辞暗示偏差很大，实际上偏差相对较小。

### 4. 建议是否科学合理

**⚠️ 原建议的两种方案需重新评估。**

方案一"修正注释去掉零和说法"——更符合当前实际情况，代价最低。`reward_opponent` 本就是日志用途，不需要精确的零和。

方案二"补充对手动作奖励计算"——不符合项目原则。如果 `reward_opponent` 仅用于日志，增加额外计算没有收益。如果未来确实需要零和（如对手策略更新），则应重新设计 reward 体系，而非修补当前实现。

### 5. 具体修复方案

修正注释即可，保持代码行为不变：

```python
# antwar_env.py 行 400-401，将：
# reward_opponent 仅用于日志（零和假设）
reward_opponent = -reward_self

# 修改为：
# reward_opponent 仅用于日志（近似零和，不含对手动作奖励）
reward_opponent = -reward_self
```

同时修正 `step()` 中行 119 之后的注释，说明 `reward_opponent` 此时已不再精确零和：

```python
reward_self += reward_self_action
# 注意：reward_opponent 仍为 _resolve_turn_from_ops 返回的值，
# 未包含对手动作奖励，reward_self + reward_opponent = reward_self_action
```

### 6. 修复代价

**低。** 仅修改注释，无代码逻辑变更。

### 7. 修复收益

**低。** 消除注释误导，但无功能影响。

---

## 总体评估结论

### 问题真实性汇总

| # | 问题 | 真实性 | 原级别 | 二次评估级别 |
|---|------|--------|--------|-------------|
| 1 | 科技升级奖励配置键错误 | ✅ 确认 | CRITICAL | CRITICAL ✓ |
| 2 | terminated 时空 observations | ✅ 确认 | HIGH | HIGH ✓ |
| 3 | reset() 观测映射不一致 | ✅ 确认 | MEDIUM | MEDIUM-HIGH |
| 4 | 异常处理路径空 observations | ✅ 确认 | MEDIUM | MEDIUM（但问题本质是吞异常） |
| 5 | 塔伤害只遍历旧塔 ID | ✅ 确认 | MEDIUM | MEDIUM ✓ |
| 6 | 降级禁用回合数 > MAX_ROUND | ✅ 确认 | LOW | LOW ✓ |
| 7 | close() 清理不一致 | ✅ 确认 | LOW | LOW ✓ |
| 8 | reward_opponent 非零和 | ✅ 确认 | LOW | LOW ✓ |

### 建议修正汇总

| # | 原建议是否科学 | 二次审查修正 |
|---|---------------|-------------|
| 1 | ✅ 科学 | 无修正 |
| 2 | ⚠️ 方向正确，实现有误 | 应使用 `self._runtime.state` 而非局部变量；直接移除 if/else 分支更简洁 |
| 3 | ⚠️ 方向正确，接口需调整 | reset() 增加 self_first 参数可行，训练代码同步更新 |
| 4 | ❌ 不符合项目原则 | 应移除 try/except 或 re-raise，遵循"不做容错"原则 |
| 5 | ⚠️ 部分科学 | 遍历并集方案无效，应使用 mid 快照作为基准 |
| 6 | ❌ 过于激进 | 修正常量和注释即可，不改变 ACTION_DIM |
| 7 | ⚠️ 可行但不必要 | 代价极低可接受，主要提升可读性 |
| 8 | ⚠️ 方案二不符合原则 | 修正注释即可，无需增加对手动作奖励计算 |

### 修复优先级建议

1. **Issue 1**（CRITICAL）：立即修复，2 行代码变更，收益极高
2. **Issue 2**（HIGH）：尽快修复，3 行代码变更，恢复接口合规性
3. **Issue 4**（异常处理）：尽快修复，移除 try/except，遵循项目原则
4. **Issue 3**（MEDIUM-HIGH）：近期修复，4 行代码变更，保证训练数据一致性
5. **Issue 5**（MEDIUM）：计划修复，需接口变更，使用 mid 快照
6. **Issue 6-8**（LOW）：可择机修复，均为注释/常量级别的修正
