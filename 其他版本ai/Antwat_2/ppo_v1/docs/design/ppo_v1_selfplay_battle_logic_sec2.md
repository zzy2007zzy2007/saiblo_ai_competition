# 2. 游戏基础：环境与回合机制

本章详细阐述 AntWar 回合制游戏的核心机制——先手/后手、两种环境交互模式（双 step 与单刀 resolve_turn），以及兼容适配层的设计。理解这些基础概念是掌握后续 SelfPlay 训练流程和 Baseline Battle 对战流程的前提。

---

## 2.1 先手/后手机制

### 2.1.1 机制概述

AntWar 是一个**回合制（sequential）游戏**，存在明确的先手/后手之分。游戏通过一个 `AntWarSequentialEnv` 环境来驱动，该环境将每个玩家的行动建模为一个独立的 `step()` 调用。因此，**每两个 `step` 调用才构成一个完整的游戏回合**。

这一机制在 [antwar_rules_review.md](file:///Users/raymondzeng/Documents/trae_projects/AntWar/docs/design/antwar_rules_review.md) §1.3 中有详细分析，其核心原理如下：

- **玩家 0 是先手方**（first player），**玩家 1 是后手方**（second player）
- 一回合内，先手方先行动，后手方再行动
- 只有后手方行动完成后，游戏回合计数器 `round_index` 才推进

### 2.1.2 step 序号与 round_index 的变化关系

下表展示 step 序号、玩家行动、round_index 变化的对应关系：

| 环境 step 序号 | 行动玩家 | 内部操作 | round_index 变化 |
|:---:|:---:|------|:---:|
| step 1 | 玩家 0（先手） | `apply_self_operations` | 不变，保持 0 |
| step 2 | 玩家 1（后手） | `apply_opponent_operations` → `advance_round` | **0 → 1** |
| step 3 | 玩家 0（先手） | `apply_self_operations` | 不变，保持 1 |
| step 4 | 玩家 1（后手） | `apply_opponent_operations` → `advance_round` | **1 → 2** |
| step 5 | 玩家 0（先手） | `apply_self_operations` | 不变，保持 2 |
| step 6 | 玩家 1（后手） | `apply_opponent_operations` → `advance_round` | **2 → 3** |
| ... | ... | ... | ... |

规律总结：

- 奇数 step（1, 3, 5, ...）：先手方行动，`round_index` 不变化
- 偶数 step（2, 4, 6, ...）：后手方行动，行动完成后 `round_index += 1`
- 第 N 个完整回合对应 step 序号区间 `[2N-1, 2N]`

### 2.1.3 MAX_ROUND 与实际最大回合数

`action_constants.py` 中定义了回合上限常量（见 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L120)）：

```python
MAX_ROUND = 512
```

由于每个完整游戏回合需要 **2 个环境 step**（先手 1 步 + 后手 1 步），在环境 step 级别的截断下：

- 512 个环境 step = 最多 **256 个完整游戏回合**
- 这便是 MAX_ROUND = 512 在实际对战中的真实含义

需要注意的是，SelfPlay 训练中使用的是 `max_steps_per_episode` 配置项（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L478)），其默认值为 512，但该变量统计的是**循环迭代次数**（每次迭代内含 2 个 `env.step()` 调用），因此实际环境 step 数量是其 2 倍，但最终仍受 SDK 内部 `MAX_ROUND` 限制。

---

## 2.2 SelfPlay 使用的环境方式：双 step 模式

### 2.2.1 环境来源

SelfPlay 训练通路使用**外部注入的 env_factory** 创建环境。环境实例在训练循环外部创建后传入 `_collect_episode_with_swap()` 方法。环境遵循标准的 Gymnasium 风格接口（`reset()` / `step()`），但内部封装了 `AntWarSequentialEnv` 的回合制逻辑。

### 2.2.2 双 step 模式

SelfPlay 的数据收集每轮循环执行**两次 `env.step()` 调用**（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L581-L582)）：

```python
obs, _, _, _, _ = env.step(player_0_action)           # 第一步：玩家0行动
obs, rewards, terminated, truncated, info = env.step(player_1_action)  # 第二步：玩家1行动
```

这就是**双 step 模式**的核心：两个 step 构成一个完整的对战循环迭代。

### 2.2.3 双 step 的完整执行流程

在 `_collect_episode_with_swap()` 方法中（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L484-L609)），一轮循环的完整流程如下：

**步骤 1 — 根据 swap_positions 确定玩家角色**：

```python
player = 1 if swap_positions else 0  # L447
```

- `swap_positions = False`：Agent 是玩家 0（先手）
- `swap_positions = True`：Agent 是玩家 1（后手）

**步骤 2 — 获取双方观察**：

```python
# 根据 swap 取对应玩家的观察
if not swap_positions:
    player_obs = obs["player_0"]    # L539
else:
    player_obs = obs["player_1"]    # L541
```

对手的观察取另一方，逻辑对称。

**步骤 3 — Agent 动作选择**（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L543)）：

```python
action, log_prob, value = self.trainer._select_action(player_obs)
```

**步骤 4 — 对手动作选择**（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L545-L562)）：

`opponent_agent` 可能是：
- 一个 PPO 对手 Agent：调用 `opponent_agent.act(opponent_obs)`
- 随机合法动作（无对手 Agent 时）：通过 `action_mask` 过滤合法动作后随机采样
- Baseline agent：同样通过 `act()` 获取动作

**步骤 5 — 组装 player_0_action 和 player_1_action**（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L574-L579)）：

```python
if not swap_positions:
    player_0_action = action          # Agent = 玩家0
    player_1_action = opponent_action # 对手 = 玩家1
else:
    player_0_action = opponent_action # 对手 = 玩家0
    player_1_action = action          # Agent = 玩家1
```

**步骤 6 — 执行双 step**（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L581-L582)）：

```python
obs, _, _, _, _ = env.step(player_0_action)                   # 先手行动
obs, rewards, terminated, truncated, info = env.step(player_1_action)  # 后手行动
```

始终是先手（player_0）先 `step`，后手（player_1）后 `step`，与 `swap_positions` 无关——swap 只决定谁扮演哪个角色。

### 2.2.4 奖励只在后手 step 后发放

这是双 step 模式的一个关键设计：**奖励信息只从第二个 `env.step()` 的返回值中获取**（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L584-L587)）：

```python
if not swap_positions:
    reward = rewards.get("player_0", 0.0)  # Agent 在先手时取 player_0 的奖励
else:
    reward = rewards.get("player_1", 0.0)  # Agent 在后手时取 player_1 的奖励
```

原因在于：`AntWarSequentialEnv` 中，第一个 step（玩家 0）只应用操作，不推进回合；第二个 step（玩家 1）才会触发回合结算（`advance_round`），此时奖励才被计算出来。因此第一个 `env.step()` 返回的 rewards 信息被忽略（用 `_` 接收），只取第二个 step 中的 `rewards`。

### 2.2.5 结束判定与 round_index 跟踪

```python
done = terminated or truncated       # L600
if done:
    batch.dones[-1] = 1.0            # L602

if info and 'round_index' in info:
    current_round = info['round_index']             # L604-L605
elif hasattr(env, '_runtime') and env._runtime and hasattr(env._runtime, 'state'):
    current_round = env._runtime.state.round_index  # L606-L607

steps += 1  # L609：循环计数器，每轮迭代 +1（含2个 env step）
```

- `terminated`：游戏正常结束（一方基地血量归零）
- `truncated`：达到最大步数限制（超时截断）
- `round_index` 从 `info` 或底层 runtime state 中获取
- `steps` 变量统计的是循环迭代次数（不是 env step 次数），每轮循环含 2 个 env step

### 2.2.6 结果判定

对局结束后，基于最终基地血量判定胜负（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L626-L631)）：

```python
if battle_details['final_hp']['enemy_hp'] <= 0 and battle_details['final_hp']['our_hp'] > 0:
    battle_details['result'] = 'win'
elif battle_details['final_hp']['our_hp'] <= 0 and battle_details['final_hp']['enemy_hp'] > 0:
    battle_details['result'] = 'loss'
else:
    battle_details['result'] = 'draw'
```

并通过 swap 方向区分先手胜/后手胜（见 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L633-L642)）。

---

## 2.3 Baseline Battle 使用的环境方式：单刀 resolve_turn 模式

### 2.3.1 环境来源

Baseline Battle 通路**不使用 Gymnasium 风格的 env**，而是直接操作 `PythonBackendState` 对象。在 `run_single_battle_process()` 中加载 SDK 后端（见 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L60-L64)）：

```python
from SDK.backend.core import load_backend
from SDK.backend.state import PythonBackendState

backend = load_backend(prefer_native=False)
python_state_class = PythonBackendState
```

状态初始化在 `_execute_battle()` 中完成（见 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L168-L169)）：

```python
game_state = backend.initial_state(seed=seed if seed is not None else episode)
state = python_state_class(game_state)
```

### 2.3.2 单刀 resolve_turn 模式

Baseline Battle 的回合循环**每轮只调用一次** `state.resolve_turn(player0_ops, player1_ops)`（见 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L197)）：

```python
state.resolve_turn(player0_ops, player1_ops)
```

这就是**单刀模式**的核心特征：在一个调用中同时提交双方的操作，由 `PythonBackendState` 内部一次性完成整个回合的结算（包括双方操作应用、蚂蚁移动、防御塔攻击、回合推进等）。

### 2.3.3 _execute_battle() 的完整回合循环

`_execute_battle()` 的核心循环逻辑（见 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L167-L254)）如下：

```python
round_count = 0

while not state.terminal and round_count < max_rounds:
    round_count += 1

    # 1. 根据 first_player 获取双方操作
    if first_player == 0:
        player0_ops = _get_operations(agent1, state, 0)   # agent1 作为玩家0
        player1_ops = _get_operations(agent2, state, 1)   # agent2 作为玩家1
    else:
        player0_ops = _get_operations(agent2, state, 0)   # agent2 作为玩家0
        player1_ops = _get_operations(agent1, state, 1)   # agent1 作为玩家1

    # 2. 单次调用完成整个回合
    state.resolve_turn(player0_ops, player1_ops)

    # 3. 记录回合数据（HP、金币、操作）
    # ...
```

与双 step 模式的关键区别：

| 维度 | SelfPlay 双 step 模式 | Baseline Battle 单刀模式 |
|------|----------------------|-------------------------|
| 环境接口 | Gymnasium `env.step()` × 2 | `state.resolve_turn()` × 1 |
| 回合粒度 | 2 次 env 调用 = 1 回合 | 1 次方法调用 = 1 回合 |
| 奖励获取 | 从第 2 个 env.step() 的返回值 | 不涉及奖励（仅评估，不训练） |
| 观察获取 | `env.step()` 返回的 obs dict | 直接从 `state` 对象读取 |
| 回合推进 | env 内部控制 `advance_round` | `resolve_turn` 内部完成 |

### 2.3.4 操作获取方式

Baseline Battle 中通过 `_get_operations()` 函数获取 agent 的操作（见 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L257-L281)）：

```python
def _get_operations(agent, state, player, logger_obj=None):
    if hasattr(agent, 'choose_operations'):
        ops = agent.choose_operations(state, player)
        if ops and hasattr(ops, 'operations'):
            return list(ops.operations)
        return ops
    elif hasattr(agent, 'choose_bundle'):
        result = agent.choose_bundle(state, player)
        if hasattr(result, 'operations'):
            return list(result.operations)
        return result
    # ...
    return []
```

支持两种 agent 接口：
- `choose_operations(state, player)` — 旧版接口，直接接收 `PythonBackendState`
- `choose_bundle(state, player)` — 新版接口，返回包含 `operations` 的结果对象

### 2.3.5 子进程隔离机制

`run_single_battle_process()` 为每局对战创建独立进程（见 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L18-L132)），关键隔离措施：

1. **sys.modules 清理**：移除 `ai`、`common`、`SDK`、`AI` 等前缀的模块，避免与主进程的模块状态冲突
2. **sys.path 恢复**：保存原始 `sys.path`，在 finally 块中恢复
3. **工作目录恢复**：保存原始 `os.getcwd()`，在 finally 块中恢复
4. **cloudpickle 序列化**：Agent 通过 `cloudpickle.dumps()` 序列化后传入子进程，再通过 `cloudpickle.loads()` 反序列化

---

## 2.4 BackendAdapter — 兼容适配层

### 2.4.1 设计动机

Ant-Game SDK 的新版 `MatchRuntime` 不再暴露 `resolve_turn()` 方法，取而代之的是更细粒度的 API：

- `apply_self_operations(ops)` — 应用己方操作
- `apply_opponent_operations(ops)` — 应用对手操作
- `finish_round(public_state)` — 完成回合结算

但 Baseline Battle 的 `_execute_battle()` 仍然期望调用 `state.resolve_turn(player0_ops, player1_ops)`。为了兼容这种旧版 API 风格，引入了 `BackendAdapter` 和 `MatchRuntimeWrapper` 两个类（见 [adapters.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/compat/adapters.py)）。

### 2.4.2 MatchRuntimeWrapper

`MatchRuntimeWrapper` 是一个代理包装类，它将新版 API 重新组装为旧版的 `resolve_turn()` 方法（见 [adapters.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/compat/adapters.py#L16-L76)）。

**核心逻辑**：

```python
class MatchRuntimeWrapper:
    def __init__(self, original_runtime):
        self._original = original_runtime

    def __getattr__(self, name):
        return getattr(self._original, name)  # 代理未重写的方法

    def resolve_turn(self, player_0_operations, player_1_operations):
        # 1. 标准化输入格式
        ops_0 = player_0_operations if isinstance(player_0_operations, list) else [...]
        ops_1 = player_1_operations if isinstance(player_1_operations, list) else [...]

        # 2. 根据 runtime 的 player 身份，区分 self/opponent 操作
        if self._original.player == 0:
            if ops_0:
                self._original.apply_self_operations(ops_0)
            if ops_1:
                self._original.apply_opponent_operations(ops_1)
        else:
            if ops_1:
                self._original.apply_self_operations(ops_1)
            if ops_0:
                self._original.apply_opponent_operations(ops_0)

        # 3. 完成回合
        public_state = self._original.state.to_public_round_state()
        self._original.finish_round(public_state)

        # 4. 返回模拟的 TurnResolution 对象
        return SimulatedTurnResolution(self._original.state)
```

关键设计点：

- **身份感知**：根据 `self._original.player` 判断当前包装的 runtime 代表哪个玩家，从而正确地将 `player_0_operations` / `player_1_operations` 映射到 `apply_self_operations` / `apply_opponent_operations`
- **输入兼容**：支持列表和单个操作两种输入格式
- **返回值兼容**：构造一个 `SimulatedTurnResolution` 对象，提供 `state`、`terminal`、`round_index`、`players` 等属性访问，模拟旧版 API 的返回类型

### 2.4.3 BackendAdapter

`BackendAdapter` 提供统一的 runtime 创建接口（见 [adapters.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/compat/adapters.py#L79-L144)）：

```python
class BackendAdapter:
    @staticmethod
    def create_runtime(player, seed=0, prefer_native=False,
                       movement_policy=None, cold_handle_rule_illegal=False):
        from SDK.backend.core import load_backend
        from SDK.backend.runtime import MatchRuntime

        backend = load_backend(prefer_native=prefer_native)

        try:
            runtime = MatchRuntime.create(
                player=player, seed=seed,
                prefer_native=prefer_native, backend=backend,
                movement_policy=movement_policy or DEFAULT_MOVEMENT_POLICY,
                cold_handle_rule_illegal=cold_handle_rule_illegal,
            )
        except TypeError:
            # 降级：旧版 SDK 不支持新参数
            runtime = MatchRuntime.create(
                player=player, seed=seed,
                prefer_native=prefer_native, backend=backend,
            )

        return MatchRuntimeWrapper(runtime)
```

关键设计点：

- **API 版本检测**：通过 `try/except TypeError` 自动适配 SDK 是否支持 `movement_policy` 和 `cold_handle_rule_illegal` 等新参数
- **统一返回类型**：始终返回 `MatchRuntimeWrapper`，调用方无需关心底层是旧版还是新版 SDK

### 2.4.4 在 Baseline Battle 中的使用

在 Baseline Battle 中，`BackendAdapter` 并非直接使用——子进程直接调用 `load_backend()` 和 `PythonBackendState`（见 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L60-L64)）。`PythonBackendState` 本身就支持 `resolve_turn()` 方法，因此不需要包装。

`BackendAdapter` 的主要使用场景是在需要以特定玩家视角创建 `MatchRuntime` 并调用其 `resolve_turn()` 的场景（例如某些单边评估或调试场景）。

### 2.4.5 两种模式与适配层的关系总结

```
SelfPlay 训练通路:
  env_factory → Gymnasium Env (AntWarSequentialEnv)
  └── env.step(p0) → env.step(p1)  [双 step 模式]
      不需要 BackendAdapter

Baseline Battle 通路:
  load_backend() → PythonBackendState
  └── state.resolve_turn(p0_ops, p1_ops)  [单刀模式]
      不需要 BackendAdapter

其他可能需要 BackendAdapter 的场景:
  load_backend() → MatchRuntime.create()
  └── MatchRuntimeWrapper.resolve_turn(p0, p1)  [兼容模式]
```

---

## 2.5 小结

| 维度 | SelfPlay（双 step） | Baseline Battle（单刀） |
|------|---------------------|------------------------|
| 环境接口 | `env.step()` × 2 / 轮 | `state.resolve_turn()` × 1 / 轮 |
| 回合粒度 | 2 次 env 调用 → 1 回合 | 1 次调用 → 1 回合 |
| 奖励机制 | 第 2 个 step 后发放 | 无（仅评估） |
| 观察来源 | env.step() 返回的 dict | state 对象直接读取 |
| 玩家身份 | 通过 swap_positions 动态决定 | 通过 first_player 参数指定 |
| SDK 交互 | 通过 env 封装，不直接操作 SDK | 直接操作 PythonBackendState |
| 最大回合 | max_steps_per_episode（循环次数，默认 512） | max_rounds（round 次数，默认 512） |
| MAX_ROUND 含义 | 512 env steps = 256 完整回合（环境级截断） | 512 rounds 直接对应 |
