# PPO v1 — 第4章 Baseline Battle 流程

> 基于 `ppo/src/ppo_antwar/battle/` 源码分析。Baseline Battle 是独立于 SelfPlay 训练之外的评估通道，用于衡量 PPO Agent 相对于固定基线 AI 的强度变化。

---

## 4.1 触发时机

Baseline Battle 在**两个位置**被触发，最终都调用同一个核心方法 `BattleCoordinator.run_ppo_evaluation_with_agent()`：

### 4.1.1 SelfPlay 训练主循环中触发

在 `SelfPlayTrainer.train()` 的主循环中，每 `battle_interval`（默认 1000）个 episode 触发一次。

参考 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py)：

```python
# selfplay.py:L767-L796
if self.battle_coordinator is not None and episode > 0 and episode % self.battle_interval == 0:
    results = self.battle_coordinator.run_ppo_evaluation_with_agent(
        ppo_agent=self.trainer.get_ppo_agent_for_battle(),
        baseline_agents=self.baseline_agents,
        num_episodes=self.n_battles // 2
    )
```

触发条件：
- `self.battle_coordinator is not None` — BattleCoordinator 已初始化
- `episode > 0` — 不在首个 episode 触发
- `episode % self.battle_interval == 0` — 每隔 `battle_interval` 个 episode

PPO Agent 由 `trainer.get_ppo_agent_for_battle()` 创建，该方法返回一个包装了当前训练 policy 网络的 `PPOAgent` 实例：

```python
# ppo_trainer.py:L882-L895
def get_ppo_agent_for_battle(self) -> "PPOAgent":
    return PPOAgent(self.policy, self.device)
```

### 4.1.2 BattleEvalCallback 中触发

`BattleEvalCallback` 是一个 PPO 训练回调，在 `PPOTrainer` 的每一步中检查是否需要触发对战评估。

参考 [battle_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_callback.py)：

```python
# battle_callback.py:L78-L107
def on_step(self) -> bool:
    episode = getattr(self.trainer, 'episode_count', 0)
    if episode <= 0:
        return True
    if episode % self.battle_interval != 0:
        return True
    if episode == self._last_evaluated_episode:
        return True

    self._last_evaluated_episode = episode
    self._evaluate_battles(episode)
    return True
```

`_evaluate_battles()` 内部同样调用 `BattleCoordinator.run_ppo_evaluation_with_agent()`：

```python
# battle_callback.py:L109-L141
def _evaluate_battles(self, episode: int):
    ppo_agent = self.trainer.get_ppo_agent_for_battle()
    results = self._battle_coordinator.run_ppo_evaluation_with_agent(
        ppo_agent=ppo_agent,
        baseline_agents=self.baseline_agents,
        num_episodes=self.n_battles // 2
    )
```

`BattleEvalCallback` 与 SelfPlay 中内联的对战评估是**互斥的替代方案**，使用哪个取决于训练入口 `train_ppo()` 的配置方式：
- 如果通过 `Callbacks` 系统（`create_ppo_callbacks()`）创建 `BattleEvalCallback`，则在回调中触发
- 如果 `SelfPlayTrainer` 直接持有 `battle_coordinator` 且 `battle_interval` 配置为非零，则在 SelfPlay 循环中内联触发

### 4.1.3 on_checkpoint_saved() 事件触发

此外，`BattleCoordinator` 还提供了 `on_checkpoint_saved()` 接口，可在 checkpoint 保存事件时触发对战评估：

```python
# battle_coordinator.py:L318-L361
def on_checkpoint_saved(self, episode: int, checkpoint_path: str) -> None:
    results = self.run_ppo_evaluation(
        ppo_checkpoint_path=checkpoint_path,
        baseline_agents=self.config.baseline_agents,
        num_episodes=self.config.n_battles // 2
    )
```

这个接口通过 checkpoint 路径加载 PPO Agent（而非直接传入实例），适用于异步或事件驱动的评估场景。

---

## 4.2 对战次数计算

Baseline Battle 中的对战次数计算公式为：

```
num_episodes = n_battles // 2
```

相关配置参数来源于 [battle_config.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_config.py) 中的 `BaselineBattleConfig`：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `n_battles` | 4 | 并行工作进程数 / 实际对局数 |
| `battle_interval` | 1000 | 每隔多少个训练 episode 触发一次评估 |

计算示例（默认值）：

- `n_battles = 4` → `num_episodes = 4 // 2 = 2`
- 每个 episode 产生 **2 场实际对局**（先手 1 场 + 后手 1 场）
- 总计：**2 episode × 2 场 = 4 场实际对局**

如果 `n_battles` 被设置为 5：

- `num_episodes = 5 // 2 = 2`（整数除法，向下取整）
- 总计：**2 × 2 = 4 场实际对局**（仍然是 4 场）

---

## 4.3 BattleCoordinator 协调流程

`BattleCoordinator` 是 Baseline Battle 的**顶层协调类**，负责编排配置管理、日志初始化、Agent 加载、对战执行、结果聚合和报告生成。

参考 [battle_coordinator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_coordinator.py)。

### 4.3.1 初始化

```python
# battle_coordinator.py:L24-L31
def __init__(self, config: BaselineBattleConfig, path_config=None):
    self.config = config
    self.config.validate()
    self._path_config = path_config
    self.logger: Optional[BattleLogger] = None
    self.agent_loader: Optional[AgentLoader] = None
    self.simulator: Optional[BattleSimulator] = None
```

初始化时调用 `config.validate()` 进行配置验证（检查 `n_battles >= 1`、`max_rounds >= 1`、checkpoint 路径有效性等）。

### 4.3.2 run_ppo_evaluation_with_agent() 完整流程

这是 Baseline Battle 的**核心入口方法**。流程如下：

```
run_ppo_evaluation_with_agent(ppo_agent, baseline_agents, num_episodes)
│
├── 1. 验证 baseline_agents 列表不为空
│
├── 2. 将 PPO Agent 设为 eval() 模式
│   │  - 查找 agent 或其子模块 (.network / .model / .nn_model / .policy) 上的 .eval() 方法
│   │  - 避免对 batch norm 或 dropout 的随机性影响评估
│
├── 3. for each baseline_agent_name in baseline_agents:
│   │
│   ├── 3a. 创建独立的 BattleLogger（为该 baseline 单独记录）
│   ├── 3b. 创建独立的 AgentLoader
│   ├── 3c. 加载 baseline_agent
│   │      - 如果加载失败 (返回 None) → 跳过该 baseline
│   │
│   ├── 3d. 创建独立的 BattleSimulator(logger, max_rounds)
│   │
│   ├── 3e. 调用 simulator.run_battles_parallel()
│   │      - ppo_agent vs baseline_agent
│   │      - 对局数: num_episodes（每 episode 含先手+后手两场）
│   │      - 并行工作数: config.n_battles
│   │      - 返回 List[Dict] — 所有对局结果的列表
│   │
│   ├── 3f. aggregate_results(results) → 聚合统计
│   │
│   ├── 3g. generate_report(aggregated, "PPO_Agent", baseline_agent_name)
│   │      - 以字符串报告格式输出到日志
│   │
│   ├── 3h. logger.write_summary() — 写入汇总日志文件
│   │
│   └── 3i. 记录此 baseline 的聚合结果到 all_results[baseline_agent_name]
│
└── 4. return all_results: Dict[str, Dict]
```

完整源码：

```python
# battle_coordinator.py:L232-L316
def run_ppo_evaluation_with_agent(
    self,
    ppo_agent: Any,
    baseline_agents: Optional[List[str]] = None,
    num_episodes: int = 1
) -> Dict[str, Dict]:
    if baseline_agents is None:
        baseline_agents = self.config.baseline_agents

    if not baseline_agents:
        raise ValueError("No baseline agents specified")

    if self.logger is None:
        self._init_logger("PPO_Agent", baseline_agents[0])
    if self.simulator is None:
        self._init_simulator()

    # 将 PPO Agent 设为 eval 模式
    if hasattr(ppo_agent, 'eval') and callable(getattr(ppo_agent, 'eval')):
        ppo_agent.eval()
    else:
        for attr_name in ['network', 'model', 'nn_model', 'neural_agent', 'policy']:
            if hasattr(ppo_agent, attr_name):
                model_obj = getattr(ppo_agent, attr_name)
                if hasattr(model_obj, 'eval') and callable(getattr(model_obj, 'eval')):
                    model_obj.eval()
                    break

    all_results = {}

    for baseline_agent_name in baseline_agents:
        current_logger = BattleLogger(...)
        self.logger = current_logger
        current_agent_loader = AgentLoader(current_logger, path_config=self._path_config)

        try:
            baseline_agent = current_agent_loader.load_agent(baseline_agent_name)
            if baseline_agent is None:
                current_logger.error('coordinator', f"Skipping {baseline_agent_name}: load failed")
                continue

            simulator = BattleSimulator(current_logger, max_rounds=self.config.max_rounds)

            results = simulator.run_battles_parallel(
                ppo_agent, baseline_agent,
                "PPO_Agent", baseline_agent_name,
                num_episodes,
                parallel_workers=self.config.n_battles
            )

            aggregated = aggregate_results(results)

            report = generate_report(aggregated, "PPO_Agent", baseline_agent_name)
            for line in report.split('\n'):
                current_logger.info('report', line)

            current_logger.write_summary()
            all_results[baseline_agent_name] = aggregated

        except Exception as e:
            current_logger.error('coordinator', f"Evaluation against {baseline_agent_name} failed: {e}")
        finally:
            current_logger.close()

    return all_results
```

**设计要点**：
- **每个 baseline agent 独立使用一组 logger/simulator/agent_loader**，确保不同 baseline 之间不会互相污染
- **PPO Agent 进入 eval() 模式**，关闭 dropout 和 batch normalization 的随机性
- **异常隔离**：单个 baseline 的评估失败不会影响其他 baseline 的评估

### 4.3.3 其他评估入口

除了 `run_ppo_evaluation_with_agent()`，`BattleCoordinator` 还提供两个变体入口：

| 方法 | 输入方式 | 适用场景 |
|------|----------|----------|
| `run_ppo_evaluation()` | checkpoint 路径 | 从磁盘加载 PPO Agent |
| `run_ppo_evaluation_with_agent()` | PPO Agent 实例（直接传入） | 训练中直接从 trainer 获取 agent |
| `run_evaluation()` | Agent 实例（可选传入） | 通用评估，不限定 PPO vs baseline |

---

## 4.4 BattleSimulator — 并行对战引擎

`BattleSimulator` 负责实际执行多局对战。核心特性是**多进程并行架构**，通过 `cloudpickle` 序列化 agent 后在子进程中执行。

参考 [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py)。

### 4.4.1 初始化

```python
# battle_simulator.py:L413-L427
class BattleSimulator:
    def __init__(self, logger: BattleLogger, max_rounds: int = DEFAULT_MAX_ROUNDS):
        self.logger = logger
        self.max_rounds = max_rounds
        self._sdk_loaded = False
        self._backend = None
        self._python_state_class = None
        self._lock = threading.Lock()
```

- `max_rounds` 默认值为 `DEFAULT_MAX_ROUNDS = 512`
- SDK 采用懒加载模式（`_load_sdk()` 首次使用时触发）

### 4.4.2 run_battles_parallel() — 多进程并行执行

```python
# battle_simulator.py:L451-L526
def run_battles_parallel(self, agent1, agent2, agent1_name, agent2_name,
                         num_episodes, parallel_workers=4):
```

**步骤 1：SDK 加载**

在主进程中加载 SDK 后端：

```python
# battle_simulator.py:L429-L449
def _load_sdk(self):
    from ppo_antwar.config.path_config import PathConfig
    path_config = PathConfig()
    sdk_path = str(path_config.sdk_dir)
    sdk_parent_path = os.path.dirname(sdk_path)
    sys.path.insert(0, sdk_parent_path)
    sys.path.insert(0, sdk_path)

    from SDK.backend.core import load_backend
    from SDK.backend.state import PythonBackendState
    self._backend = load_backend(prefer_native=False)
    self._python_state_class = PythonBackendState
    self._sdk_loaded = True
```

使用 `prefer_native=False` 加载 Python 后端（而非 native 编译版本），确保跨平台兼容性。每个子进程内部会**再次独立**加载 SDK。

**步骤 2：cloudpickle 序列化 Agent**

```python
import cloudpickle
serialized_agent1 = cloudpickle.dumps(agent1)
serialized_agent2 = cloudpickle.dumps(agent2)
```

`cloudpickle` 可以序列化比标准 `pickle` 更广泛的 Python 对象（包括 lambda、动态类等），这对于包含 PyTorch 模型的 agent 对象至关重要。

**步骤 3：生成任务列表**

```python
# battle_simulator.py:L475-L480
tasks = []
log_level_str = self.logger._get_log_level_str()
for episode in range(num_episodes):
    seed = episode
    # agent1 先手
    tasks.append((episode, seed, 0, agent1_name, agent2_name,
                  self.max_rounds, self.logger.get_log_dir(),
                  log_level_str, serialized_agent1, serialized_agent2))
    # agent1 后手
    tasks.append((episode + 1000, seed + 1000, 1, agent1_name, agent2_name,
                  self.max_rounds, self.logger.get_log_dir(),
                  log_level_str, serialized_agent1, serialized_agent2))
```

每个任务是一个 10 元组：
```
(episode, seed, first_player, agent1_name, agent2_name,
 max_rounds, log_dir, log_level_str, serialized_agent1, serialized_agent2)
```

**步骤 4：ProcessPoolExecutor 并行执行**

```python
# battle_simulator.py:L488-L525
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
    futures = {executor.submit(run_single_battle_process, task): task for task in tasks}

    for future in __import__('concurrent.futures').futures.as_completed(futures):
        try:
            result = future.result()
            results.append(result)
            self.logger.log_result(result)
        except Exception as e:
            task = futures[future]
            self.logger.error('battle', f"Battle execution failed: {task}, error: {e}")
            # 构造 error 结果并添加到结果列表
            results.append({...})

        completed += 1
        if completed % 10 == 0 or completed == total:
            self.logger.info('progress', f"{completed}/{total} battles completed")
```

**步骤 5：结果排序**

```python
results.sort(key=lambda x: x['episode'])
```

结果按 episode 编号排序，确保最终结果集中先手对局和对应后手对局按顺序排列。

### 4.4.3 对局编号规则

对局编号通过 `episode` 字段区分先后手：

| episode 值 | first_player | 含义 |
|:----------:|:------------:|------|
| 0 | 0 | Agent1（PPO）先手，Agent2（baseline）后手 |
| 1000 | 1 | Agent1（PPO）后手，Agent2（baseline）先手 |
| 1 | 0 | Agent1 先手 |
| 1001 | 1 | Agent1 后手 |
| ... | ... | ... |

- `episode` 范围 `[0, num_episodes-1]`：agent1 先手
- `episode` 范围 `[1000, 1000 + num_episodes - 1]`：agent1 后手
- `seed = episode`：使种子与 episode 相同，保证可复现性
- `seed + 1000`：后手对局的种子与先手对局不同，避免完全相同的随机序列

---

## 4.5 子进程对战逻辑（run_single_battle_process）

`run_single_battle_process()` 是每个子进程的**入口函数**，负责在独立进程中执行一局对战。

参考 [battle_simulator.py:L18-L133](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L18-L133)。

### 4.5.1 进程隔离

子进程启动后需要**清理可能冲突的模块和路径**：

```python
# battle_simulator.py:L30-L45
original_sys_path = sys.path.copy()
original_modules = set(sys.modules.keys())
original_cwd = os.getcwd()

# 移除可能冲突的模块
modules_to_remove = []
for key in list(sys.modules.keys()):
    if key in ['ai', 'common', 'SDK', 'AI'] or \
       key.startswith('ai.') or key.startswith('common.') or \
       key.startswith('SDK.') or key.startswith('AI.'):
        modules_to_remove.append(key)

removed_modules = {}
for mod in modules_to_remove:
    if mod in sys.modules:
        removed_modules[mod] = sys.modules[mod]
        del sys.modules[mod]
```

清理的模块类型：
- `ai` / `ai.*` — Baseline Agent 的 `ai` 模块（不同 baseline 使用不同的 ai.py，可能冲突）
- `common` / `common.*` — Agent 使用的工具模块
- `SDK` / `SDK.*` — SDK 模块（子进程中独立加载）
- `AI` / `AI.*` — 旧版 AI 模块命名空间

### 4.5.2 SDK 加载

```python
# battle_simulator.py:L51-L64
from ppo_antwar.config.path_config import PathConfig
path_config = PathConfig()
sdk_path = str(path_config.sdk_dir)
sdk_parent = os.path.dirname(sdk_path)

if sdk_parent not in sys.path:
    sys.path.insert(0, sdk_parent)

from SDK.backend.core import load_backend
from SDK.backend.state import PythonBackendState

backend = load_backend(prefer_native=False)
python_state_class = PythonBackendState
```

每个子进程独立加载 `PythonBackendState`，避免共享状态导致的并发问题。

### 4.5.3 cloudpickle 反序列化 Agent

```python
# battle_simulator.py:L66-L85
import cloudpickle
agent1 = cloudpickle.loads(serialized_agent1)
agent2 = cloudpickle.loads(serialized_agent2)

if not agent1 or not agent2:
    result = {
        'episode': episode,
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'first_player': first_player,
        'result': 'error',
        'error': 'Failed to deserialize agents',
        ...
    }
    return result
```

### 4.5.4 调用 _execute_battle()

```python
# battle_simulator.py:L87-L94
log_level = LogLevel.from_string(log_level_str)
sub_logger = SubProcessLogger(agent1_name, agent2_name, log_level=log_level)

result = _execute_battle(
    agent1, agent2, agent1_name, agent2_name,
    episode, seed, first_player, max_rounds, backend, python_state_class,
    sub_logger=sub_logger
)
```

### 4.5.5 异常处理

```python
# battle_simulator.py:L98-L116
except Exception as e:
    import traceback
    error_msg = str(e) + "\n" + traceback.format_exc()
    result = {
        'episode': episode,
        ...
        'result': 'error',
        'error': error_msg,
        ...
    }
```

### 4.5.6 资源清理

```python
# battle_simulator.py:L117-L132
finally:
    os.chdir(original_cwd)
    sys.path = original_sys_path

    # 清理子进程加载的新模块，避免污染
    current_modules = set(sys.modules.keys())
    new_modules = current_modules - original_modules
    for mod in new_modules:
        if mod in sys.modules:
            del sys.modules[mod]

    # 恢复之前移除的模块
    for mod in removed_modules:
        if mod not in sys.modules:
            sys.modules[mod] = removed_modules[mod]

    logger.complete()
```

---

## 4.6 _execute_battle() — 单局对战核心

`_execute_battle()` 是执行单局对战的**核心函数**，被 `run_single_battle_process()` 和主进程中的 `BattleSimulator.run_battle()` 共享使用。

参考 [battle_simulator.py:L135-L254](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L135-L254)。

### 4.6.1 状态初始化

```python
# battle_simulator.py:L167-L172
game_state = backend.initial_state(seed=seed if seed is not None else episode)
state = python_state_class(game_state)
```

- `backend.initial_state(seed)` — 创建初始游戏状态（棋盘、基地 HP、初始金币等），seed 确保随机性可复现
- `python_state_class(game_state)` — 将原生状态包装为 `PythonBackendState`，提供统一的 Python API

### 4.6.2 回合循环

```python
# battle_simulator.py:L176-L233
round_count = 0

while not state.terminal and round_count < max_rounds:
    round_count += 1

    if first_player == 0:
        # PPO Agent (agent1) 先手
        player0_ops = _get_operations(agent1, state, 0)
        player1_ops = _get_operations(agent2, state, 1)
    else:
        # PPO Agent (agent1) 后手
        player0_ops = _get_operations(agent2, state, 0)
        player1_ops = _get_operations(agent1, state, 1)

    # 一刀执行回合：双方操作同时生效 + 回合推进
    state.resolve_turn(player0_ops, player1_ops)
```

**循环终止条件**：
- `state.terminal == True` — 一方基地被摧毁（游戏正常结束）
- `round_count >= max_rounds` — 达到最大回合数限制（默认 512），防止无限拉锯

**回合执行方式**：

Baseline Battle 使用 `state.resolve_turn()` **一次性**执行双方操作：
- 双方在各自回合内的操作**同时提交、同时生效**
- `resolve_turn()` 内部处理：操作验证 → 操作应用 → 回合推进 → 终端判定

这与 SelfPlay 中使用的 `env.step()` 逐步执行方式形成对比（详见主文档 §2.3）。

### 4.6.3 操作获取：_get_operations()

```python
# battle_simulator.py:L257-L280
def _get_operations(agent, state, player, logger_obj=None):
    try:
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
    except Exception as e:
        error_context = get_error_context(
            player=player,
            agent_type=type(agent).__name__,
            ...
        )
        log_exception(logger_obj, "_get_operations", error_context, e)
        raise
    return []
```

**操作获取的优先级顺序**：

1. `agent.choose_operations(state, player)` — 直接操作 `PythonBackendState` 对象，返回 `Operation` 列表。适用于 Baseline Agent（如 `BasicTowerAI`）
2. `agent.choose_bundle(state, player)` — 备选接口，同样直接操作 state
3. 以上都不支持 → 返回 `[]`（空列表，即 NO_OP）

**关于 PPO Agent 在 Baseline Battle 中的操作获取**：

PPO Agent（`PPOAgent` 类）只有 `act(observation)` 方法，该方法要求输入是经过 `ObservationEncoder` 编码的 dict（board + global + action_mask），**不直接支持 `choose_operations(state, player)` 接口**。

因此，PPO Agent 在 `_get_operations()` 中会走到 fallback 路径返回 `[]`。这意味着**当前实现中 PPO Agent 在 Baseline Battle 中实际执行的是 NO_OP**。这是 Baseline Battle 的一个已知限制——PPO Agent 依赖 `ObservationEncoder` 将 `BackendState` 转为模型输入，但这部分桥接逻辑尚未在 `_get_operations` 中实现。

如需修正此行为，需在 PPO Agent 上添加 `choose_operations(state, player)` 方法，内部调用 `ObservationEncoder.encode(state, player)` 将 state 转为 observation，再通过 `agent.act(observation)` 获取 action，最终通过 `action_id_to_operation()` 转换为 `Operation` 列表。

### 4.6.4 每回合数据记录

```python
# battle_simulator.py:L207-L229
# 记录 HP
hp1 = state.bases[0].hp if hasattr(state, 'bases') and len(state.bases) > 0 else 50
hp2 = state.bases[1].hp if hasattr(state, 'bases') and len(state.bases) > 1 else 50
# 记录金币
coins1 = state.coins[0] if hasattr(state, 'coins') and len(state.coins) > 0 else 0
coins2 = state.coins[1] if hasattr(state, 'coins') and len(state.coins) > 1 else 0
# 记录操作字符串
ops1_str = _operations_to_string(player0_ops)
ops2_str = _operations_to_string(player1_ops)

result['round_data'].append({
    'round': round_count,
    'hp1': hp1, 'hp2': hp2,
    'coins1': coins1, 'coins2': coins2,
    'ops1': ops1_str, 'ops2': ops2_str
})

# 计时数据
result['timing_data'].append({
    'round': round_count,
    'agent1_ops_time': t0_end - t0_start,
    'agent2_ops_time': t1_end - t1_start,
    'resolve_time': resolve_end - resolve_start
})
```

### 4.6.5 终局数据收集

```python
# battle_simulator.py:L234-L237
result['total_rounds'] = round_count
result['final_hp'] = _get_final_hp(state, first_player)
result['final_coins'] = _get_final_coins(state)
result['result'] = _determine_result(state, first_player)
```

### 4.6.6 结果字典结构

每局对战产生如下结构的结果字典：

```python
{
    'episode': int,          # 对局编号
    'agent1_name': str,      # Agent1 名称（通常为 "PPO_Agent"）
    'agent2_name': str,      # Agent2 名称（baseline agent 名）
    'first_player': int,     # 先手玩家：0 = agent1 先手, 1 = agent2 先手
    'result': str,           # 结果：'agent1_win' / 'agent2_win' / 'draw' / 'error'
    'error': str,            # 错误信息（仅 result='error' 时有值）
    'start_time': str,       # 对战开始时间
    'end_time': str,         # 对战结束时间
    'duration': float,       # 对战耗时（秒）
    'total_rounds': int,     # 总回合数
    'final_hp': {            # 最终 HP
        'agent1': int,
        'agent2': int
    },
    'final_coins': {         # 最终金币
        'agent1': int,
        'agent2': int
    },
    'timing_data': List[Dict],  # 每回合计时数据
    'round_data': List[Dict],   # 每回合详细数据 (HP, coins, operations)
}
```

---

## 4.7 结果判定（_determine_result）

`_determine_result()` 使用**三级降级策略**判定对局结果。

参考 [battle_simulator.py:L326-L393](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L326-L393)。

### 4.7.1 判定流程

```
_determine_result(state, first_player)
│
├── 第一优先级：state.winner
│   ├── winner == 0
│   │   └── first_player == 0 → 'agent1_win'
│   │   └── first_player == 1 → 'agent2_win'
│   ├── winner == 1
│   │   └── first_player == 0 → 'agent2_win'
│   │   └── first_player == 1 → 'agent1_win'
│   └── winner == 其他值（如 -1、None）
│       └── 降级到 HP 比较
│
├── 第二优先级：HP 比较（降级方案）
│   ├── hp1 > hp2
│   │   └── first_player == 0 → 'agent1_win'
│   │   └── first_player == 1 → 'agent2_win'
│   ├── hp2 > hp1
│   │   └── first_player == 0 → 'agent2_win'
│   │   └── first_player == 1 → 'agent1_win'
│   └── hp1 == hp2
│       └── 降级到 draw
│
└── 第三优先级：draw
    └── HP 相等或无法获取 HP → 'draw'
```

### 4.7.2 完整源码

```python
# battle_simulator.py:L326-L393
def _determine_result(state, first_player):
    # 第一优先级：优先使用 state.winner
    try:
        winner = state.winner

        if winner == 0:
            return 'agent1_win' if first_player == 0 else 'agent2_win'
        elif winner == 1:
            return 'agent2_win' if first_player == 0 else 'agent1_win'
    except Exception as e:
        log_exception(None, "_determine_result.winner_check", error_context, e)

    # 第二优先级：降级方案 — 比较 HP
    try:
        hp1 = state.bases[0].hp if hasattr(state, 'bases') and len(state.bases) > 0 else 0
        hp2 = state.bases[1].hp if hasattr(state, 'bases') and len(state.bases) > 1 else 0

        if hp1 > hp2:
            return 'agent1_win' if first_player == 0 else 'agent2_win'
        elif hp2 > hp1:
            return 'agent2_win' if first_player == 0 else 'agent1_win'
    except Exception as e:
        log_exception(None, "_determine_result.hp_check", error_context, e)

    # 第三优先级：HP 相等 → draw
    return 'draw'
```

### 4.7.3 first_player 映射规则

`first_player` 指示 agent1 在游戏中的位置：

| first_player | player_0 是谁 | player_1 是谁 | winner==0 的含义 |
|:---:|:---:|:---:|:---:|
| 0 | agent1 | agent2 | agent1 胜 |
| 1 | agent2 | agent1 | agent2 胜 |

映射逻辑：

```
winner == 0 且 first_player == 0 → agent1 作为 player_0 获胜 → 'agent1_win'
winner == 0 且 first_player == 1 → agent2 作为 player_0 获胜 → 'agent2_win'
winner == 1 且 first_player == 0 → agent2 作为 player_1 获胜 → 'agent2_win'
winner == 1 且 first_player == 1 → agent1 作为 player_1 获胜 → 'agent1_win'
```

### 4.7.4 结果类型对比

| 结果字符串 | 含义 | 判定来源 |
|-----------|------|----------|
| `'agent1_win'` | Agent1（PPO）获胜 | state.winner 或 HP 比较 |
| `'agent2_win'` | Agent2（baseline）获胜 | state.winner 或 HP 比较 |
| `'draw'` | 平局 | HP 相等或无法判定 |
| `'error'` | 对局异常（agent 序列化失败 / 操作异常） | try-except 捕获 |

注意：Baseline Battle 使用 `'agent1_win'` / `'agent2_win'` 命名，而 SelfPlay 使用 `'win'` / `'loss'`，两者命名体系不同。

---

## 4.8 降级策略

### 4.8.1 多进程失败 → 顺序执行

当 `ProcessPoolExecutor` 整体执行异常时（例如系统资源不足、fork 失败等），自动降级为顺序执行：

```python
# battle_simulator.py:L521-L523
except Exception as e:
    self.logger.warning('parallel', f"多进程执行失败，降级为顺序执行: {e}")
    results = self._run_battles_sequential(agent1, agent2, agent1_name, agent2_name, num_episodes)
```

### 4.8.2 _run_battles_sequential() — 顺序执行降级方案

顺序执行直接在主进程中循环执行，不使用 `ProcessPoolExecutor`：

```python
# battle_simulator.py:L528-L588
def _run_battles_sequential(self, agent1, agent2, agent1_name, agent2_name, num_episodes):
    tasks = []
    for episode in range(num_episodes):
        seed = episode
        tasks.append((episode, seed, 0))      # agent1 先手
        tasks.append((episode + 1000, seed + 1000, 1))  # agent1 后手

    results = []
    for task in tasks:
        episode, seed, first_player = task
        try:
            result = self.run_battle(
                agent1, agent2, agent1_name, agent2_name,
                episode, seed=seed, first_player=first_player
            )
            results.append(result)
        except Exception as e:
            # 构造 error 结果
            ...

    results.sort(key=lambda x: x['episode'])
    return results
```

顺序执行调用 `BattleSimulator.run_battle()` 方法，该方法与 `_execute_battle()` 逻辑等同，但在主进程中同步执行，使用 `BattleLogger` 进行计时和日志记录（而非子进程的 `SubProcessLogger`）。

### 4.8.3 单局失败 → 记录 error

无论是并行模式还是顺序模式，单局对战的异常都会被捕获并记录为 `result='error'` 的结果：

```python
# battle_simulator.py:L239-L244
except Exception as e:
    result['error'] = str(e)
    result['result'] = 'error'
```

**不会因单局失败而中断整个评估流程**。错误局数会在结果聚合中被单独统计（`error_battles` 字段）。

### 4.8.4 结果判定降级

在 `_determine_result()` 内部的降级也遵循同一理念：

```
state.winner 可用 → 使用 state.winner
state.winner 不可用 / 异常 → HP 比较
HP 无法获取 / 异常 → draw
```

---

## 4.9 结果聚合（ResultAggregator）

`ResultAggregator` 将多局对战的原始结果列表聚合为统计摘要。

参考 [result_aggregator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/result_aggregator.py)。

### 4.9.1 聚合字段

```python
# result_aggregator.py:L8-L24
aggregated = {
    'total_battles': len(results),          # 总对局数
    'completed_battles': 0,                  # 正常完成的对局数
    'error_battles': 0,                      # 异常对局数
    'agent1_wins': 0,                        # Agent1 (PPO) 胜场数
    'agent2_wins': 0,                        # Agent2 (baseline) 胜场数
    'draws': 0,                              # 平局数
    'agent1_first_move_wins': 0,             # Agent1 作为先手的胜场数
    'agent1_first_move_battles': 0,          # Agent1 作为先手的对局数
    'agent1_second_move_wins': 0,            # Agent1 作为后手的胜场数
    'agent1_second_move_battles': 0,         # Agent1 作为后手的对局数
    'total_duration': 0.0,                   # 总耗时（秒）
    'avg_rounds': 0.0,                       # 平均回合数
    'start_time': None,                      # 最早开始时间
    'end_time': None,                        # 最晚结束时间
}
```

### 4.9.2 聚合逻辑

```python
# result_aggregator.py:L34-L62
for result in results:
    aggregated['total_duration'] += result['duration']

    if result['result'] == 'error':
        aggregated['error_battles'] += 1
    else:
        aggregated['completed_battles'] += 1
        total_rounds += result.get('total_rounds', 0)

        # 胜负统计
        if result['result'] == 'agent1_win':
            aggregated['agent1_wins'] += 1
        elif result['result'] == 'agent2_win':
            aggregated['agent2_wins'] += 1
        else:
            aggregated['draws'] += 1

        # 先后手分别统计
        if result['first_player'] == 0:
            aggregated['agent1_first_move_battles'] += 1
            if result['result'] == 'agent1_win':
                aggregated['agent1_first_move_wins'] += 1
        else:
            aggregated['agent1_second_move_battles'] += 1
            if result['result'] == 'agent1_win':
                aggregated['agent1_second_move_wins'] += 1

if aggregated['completed_battles'] > 0:
    aggregated['avg_rounds'] = total_rounds / aggregated['completed_battles']
```

### 4.9.3 先后手胜率计算

通过 `first_player` 字段区分，可计算先手胜率和后手胜率：

```
先手胜率 = agent1_first_move_wins / agent1_first_move_battles
后手胜率 = agent1_second_move_wins / agent1_second_move_battles
```

### 4.9.4 便捷函数

```python
def aggregate_results(results: List[Dict]) -> Dict:
    aggregator = ResultAggregator()
    return aggregator.aggregate(results)
```

---

## 4.10 报告生成（ReportGenerator）

`ReportGenerator` 将聚合结果格式化为可读的文本报告。

参考 [report_generator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/report_generator.py)。

### 4.10.1 报告结构

报告包含以下部分：

**1) 头部信息**

```
============================================================
        Baseline 对战评测报告
============================================================
执行时间: 2025-01-01 12:00:00
结束时间: 2025-01-01 12:01:30
对战双方: PPO_Agent vs BasicTowerAI
============================================================
```

**2) 总体统计**

```
【总体统计】
总对战数: 4
完成对战: 4
异常对战: 0
总耗时: 90.50秒
平均回合数: 128.0
```

**3) 对战结果表**

```
【对战结果】
┌────────────┬────────────┬───────┬───────┬──────┬────────┐
│   Agent1   │   Agent2   │  胜   │  负  │  平   │  胜率   │
├────────────┼────────────┼───────┼───────┼──────┼────────┤
│ PPO_Agent  │ BasicTowerAI│   3   │  0   │   1   │  75.0%  │
└────────────┴────────────┴───────┴───────┴──────┴────────┘
```

**4) 详细对战记录**

```
【详细对战记录】
PPO_Agent vs BasicTowerAI: 3胜 0负 1平 (胜率 75.0%)
```

**5) 先后手胜率**

```
【先后手胜率】
PPO_Agent - 先手: 2胜 0负 (100.0%)
PPO_Agent - 后手: 1胜 0负 (100.0%)
```

### 4.10.2 胜率计算公式

```python
# report_generator.py:L36
win_rate = (aggregated['agent1_wins'] / total * 100) if total > 0 else 0.0
```

### 4.10.3 便捷函数

```python
def generate_report(aggregated: Dict, agent1_name: str, agent2_name: str) -> str:
    generator = ReportGenerator()
    return generator.generate(aggregated, agent1_name, agent2_name)
```

报告以字符串形式返回，由 `BattleCoordinator` 按行输出到日志：

```python
# battle_coordinator.py:L302-L304
report = generate_report(aggregated, "PPO_Agent", baseline_agent_name)
for line in report.split('\