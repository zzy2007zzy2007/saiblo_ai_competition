# 对战执行模块 (Battle Execution)

## 1. 背景与定位

### 1.1 业务问题

在强化学习系统中，除了训练时的自对弈，还需要**独立的对战评估引擎**来评估智能体的水平。这个引擎需要支持：
- 加载不同类型的智能体（PPO 策略、随机 AI、规则 AI 等）
- 在独立的进程中执行对战（防止单场崩溃影响整体）
- 多场次对战并计算统计结果
- 生成可读的评估报告

**对战执行模块**的核心业务目标是：**提供一个独立于训练流程的对战引擎**，可以加载任意两个智能体进行多轮对战，聚合结果并生成报告。该模块可以在无训练的情况下独立使用（如仅评估基线 AI），也可以在训练过程中被自对弈编排调用（进行周期性基线评估）。

### 1.2 在整个系统中的位置

```
自对弈编排 (Self-Play Orchestration)
        │
        │  调用基线评估
        ▼
┌──────────────────────────────────────────────────┐
│              对战执行模块                          │
│                                                    │
│  BattleCoordinator (编排入口)                       │
│    ├─ BattleSimulator (对战核心引擎)               │
│    │   ├─ 多进程并行执行                           │
│    │   └─ 顺序执行降级                             │
│    ├─ AgentLoader (智能体加载)                     │
│    │   ├─ 内置 Agent: Random/TowerAI/RuleAI       │
│    │   ├─ 外部 Agent: 动态加载外部模块             │
│    │   └─ PPO Agent: 从训练检查点加载              │
│    ├─ ResultAggregator (结果聚合)                  │
│    ├─ BattleReporter (报告生成)                    │
│    ├─ BattleLogger / EvalBattleWriter (日志)       │
│    └─ 工具集                                     │
│        ├─ process_isolation (进程隔离)             │
│        ├─ exception_logging (异常记录)             │
│        └─ timing (计时器)                         │
└──────────────────────────────────────────────────┘
        │
        ▼
    游戏后端 (BackendAdapter → Ant-Game SDK)
```

- **上层**：自对弈编排模块通过 `BattleCoordinator.run_ppo_evaluation_with_agent()` 进行基线评估
- **独立使用**：也可被外部脚本直接调用来评估任意两个 agent 的对战
- **下层**：使用 `BackendAdapter` 创建游戏运行时，执行对战逻辑

### 1.3 与其他模块的关系

| 外部模块 | 关系 | 交互方式 |
|---------|------|---------|
| 游戏环境 (ActionMaskHandler) | 依赖 | 用于 `PPOWarAgent` 的动作转换 |
| 游戏环境 (BackendAdapter) | 依赖 | 创建游戏运行时 |
| 策略网络 (AntWarPolicyValueNetwork) | 依赖 | 用于加载 PPO 检查点 |
| PPO 训练引擎 (PPOAgent) | 依赖 | `OpponentAgent.from_checkpoint()` 需要 |
| 基础设施 (PathConfig, action_constants) | 依赖 | 路径管理和动作常量 |
| 自对弈编排 (`selfplay.py`) | 被依赖 | 调用基线评估 |

---

## 2. 核心概念

### 2.1 关键术语

| 术语 | 说明 |
|------|------|
| **Baseline AI** | 固定的基线智能体（如随机选择、简单规则、中级规则），用于评估训练算法的绝对进步 |
| **内置 Agent** | 代码内直接定义的智能体类（`BasicRandomAI`、`BasicTowerAI`、`MediumRuleAI`） |
| **外部 Agent** | 通过动态导入加载的第三方智能体（扫描 `baselines/` 目录） |
| **PPO Agent** | 用训练好的策略网络驱动的智能体 |
| **进程隔离** | 每场对战在独立子进程中执行，防止崩溃传播 |
| **先手/后手** | 每局对战双方各一次先手机会，消除先后手偏差 |

### 2.2 对战执行流程

```
1. BattleCoordinator.run_ppo_evaluation_with_agent(ppo_agent, baseline_list)
   │
   ├─ 对每个基线 AI:
   │   ├─ AgentLoader.load(baseline_name) → baseline_agent
   │   ├─ BattleSimulator.run_battles(ppo_agent, baseline_agent, num_episodes)
   │   │   ├─ 每局执行 2 场对战（轮流先手）
   │   │   ├─ 并行多进程执行
   │   │   └─ 收集结果列表
   │   ├─ aggregate_results(results) → 统计汇总
   │   ├─ add_battle_result() → 加入报告
   │   └─ EvalBattleWriter.write() → 日志
   │
   └─ BattleReporter.generate_summary() → 汇总报告
```

---

## 3. 架构总览

### 3.1 模块组成

```
battle-execution/
│
├── battle/battle_coordinator.py     # BattleCoordinator - 对战编排入口
├── battle/battle_simulator.py       # BattleSimulator - 对战执行引擎
├── battle/agent_loader.py           # AgentLoader + Agent 继承体系
├── battle/opponent_agent.py         # OpponentAgent - PPO 对手封装
├── battle/battle_config.py          # BaselineBattleConfig - 对战配置
├── battle/battle_logger.py          # BattleLogger - 对战日志
├── battle/report_generator.py       # BattleReporter - 报告生成
├── battle/result_aggregator.py      # aggregate_results - 结果聚合
├── battle/utils/exception_logging.py # 异常日志
├── battle/utils/process_isolation.py # 进程隔离
├── battle/utils/timing.py           # 计时器
└── monitor/eval_battle_writer.py    # EvalBattleWriter - 评估日志写入
```

### 3.2 组件关系

```
BattleCoordinator (编排入口)
│
├── BaselineBattleConfig (配置)
│
├── AgentLoader (智能体加载)
│   ├── PPOWarAgent (PPO 策略驱动)
│   ├── BasicRandomAI (随机策略)
│   ├── BasicTowerAI (建塔+规则)
│   ├── MediumRuleAI (中级规则)
│   └── 外部 Agent (动态导入)
│
├── BattleSimulator (对战引擎)
│   ├── ProcessPoolExecutor (多进程)
│   ├── run_single_battle_process (子进程入口)
│   └── _execute_battle (单场核心循环)
│
├── ResultAggregator / aggregate_results
├── BattleReporter (报告生成)
├── BattleLogger (日志)
├── EvalBattleWriter (评估日志)
└── 工具集 (process_isolation, timing, exception_logging)
```

---

## 4. 核心组件详解

### 4.1 BattleCoordinator (`battle/battle_coordinator.py`)

#### 4.1.1 职责

对战编排的入口类，协调智能体加载、对战执行、结果聚合和报告生成的完整流程。

#### 4.1.2 构造函数

```python
def __init__(self, config: BaselineBattleConfig, path_config: PathConfig)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | BaselineBattleConfig | 对战配置（最大回合数、日志级别等） |
| `path_config` | PathConfig | 全局路径管理器 |

#### 4.1.3 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `run_ppo_evaluation_with_agent` | `(ppo_agent, baseline_agents, num_episodes) -> Dict` | PPO 智能体 vs 多个基线 AI 的评估 |
| `run_batch` | `(agent1, agent2, num_episodes) -> Dict` | 两个指定智能体之间的对战 |

#### 4.1.4 内部逻辑

```python
def run_ppo_evaluation_with_agent(self, ppo_agent, baseline_agents, num_episodes):
    # 1. 验证所有基线名称
    AgentLoader.validate_baselines(baseline_agents)

    results = {}
    for baseline_name in baseline_agents:
        # 2. 加载基线 AI
        baseline_agent = AgentLoader.load(baseline_name, player_id=1)

        # 3. 多轮对战
        battle_results = self._simulator.run_battles(
            ppo_agent, baseline_agent, num_episodes)

        # 4. 聚合结果
        aggregated = aggregate_results(battle_results)

        # 5. 生成报告
        report = self._reporter.generate_report(
            aggregated, "PPO_Agent", baseline_name)

        # 6. 记录日志
        self._reporter.add_battle_result(baseline_name, aggregated)
        if self._active_logger:
            self._active_logger.write_summary()

        results[baseline_name] = {"aggregated": aggregated, "report": report}

    # 7. 生成汇总
    results['summary'] = self._reporter.generate_summary()
    return results
```

### 4.2 BattleSimulator (`battle/battle_simulator.py`)

#### 4.2.1 职责

对战执行的核心引擎，负责多场对战的多进程并行执行和异常降级。

#### 4.2.2 构造函数

```python
def __init__(self, config: BaselineBattleConfig)
```

默认参数：`_DEFAULT_MAX_WORKERS = 8`，`_BATTLE_TIMEOUT_SECONDS = 300`

#### 4.2.3 公开方法

| 方法 | 功能 |
|------|------|
| `run_battles(agent1, agent2, num_episodes) -> List[Dict]` | 执行 `num_episodes * 2` 场对局（含先手后手），返回结果列表 |

#### 4.2.4 多进程执行流程

```python
def run_battles(self, agent1, agent2, num_episodes):
    # 1. 序列化 agent（通过 cloudpickle 跨进程传递）
    agent1_bytes = AgentLoader.serialize(agent1)
    agent2_bytes = AgentLoader.serialize(agent2)

    # 2. 构建任务列表（每 episode 两场：先手/后手各一）
    tasks = []
    for i in range(num_episodes):
        tasks.append((agent1_bytes, agent2_bytes, 0, max_rounds))  # agent1 先手
        tasks.append((agent1_bytes, agent2_bytes, 1, max_rounds))  # agent2 先手

    # 3. 多进程并行执行
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_single_battle_process, *t)
                       for t in tasks]
            results = [f.result(timeout=300) for f in futures]
    except Exception:
        # 4. 降级：如果多进程失败（如 CUDA 环境），改为顺序执行
        for t in tasks:
            results.append(run_single_battle_process(*t))

    return results
```

#### 4.2.5 单场对战核心 (`_execute_battle`)

```python
def _execute_battle(agent1, agent2, first_player, max_rounds):
    # 返回格式: {result, rounds, duration, first_player, ...}

    runtime = BackendAdapter.create_runtime(player=0)  # 视角0
    state = runtime.advance_round()  # 初始状态

    for round_num in range(max_rounds):
        if first_player == 0:
            ops_0 = agent1.choose_operations(state)
            ops_1 = agent2.choose_operations(state)
        else:
            ops_1 = agent1.choose_operations(state)
            ops_0 = agent2.choose_operations(state)

        result = runtime.resolve_turn(ops_0, ops_1)
        state = result.state

        if result.terminal:
            break

    # 判定胜负
    winner = compare_hp(state)
    return {
        'result': winner,
        'rounds': round_num,
        'duration': timer.elapsed(),
        'first_player': first_player,
    }
```

### 4.3 Agent 体系 (`battle/agent_loader.py`)

#### 4.3.1 职责

提供统一的智能体加载接口，支持内置 agent、外部 agent 和 PPO agent 三种来源。

#### 4.3.2 Agent 继承体系

```
AntWarAgent (抽象基类)
  ├─ choose_operations(state) → List[Operation]  (抽象)
  │
  ├── PPOWarAgent: PPO 策略驱动
  │   ├─ 内部持有 PPOAgent + ObservationEncoder + ActionMaskHandler
  │   ├─ choose_operations: encode → act → action_id_to_op
  │   └─ 评估模式 (deterministic=True)
  │
  ├── BasicRandomAI: 随机选择合法动作
  │   ├─ 内部持有 ActionMaskHandler
  │   └─ choose_operations: get_action_mask → 随机选合法动作 → action_id_to_op
  │
  ├── BasicTowerAI: 简单规则 AI
  │   ├─ 优先级: 闪电风暴 > 电磁脉冲 > 建塔 > 升级 > 科技升级
  │   └─ choose_operations: 遍历优先级，选择第一个合法操作
  │
  └── MediumRuleAI (继承 BasicTowerAI): 中级规则 AI
      ├─ 优先级: 建塔 > 升级 > 超级武器 > 科技升级(生产>血)
      └─ 更细致的规则判断
```

#### 4.3.3 AgentLoader 静态方法

| 方法 | 功能 |
|------|------|
| `load(name, player_id) -> AntWarAgent` | 统一加载：内置优先，外部回退 |
| `load_builtin(name, player_id) -> AntWarAgent` | 从 `KNOWN_BUILTIN_AGENTS` 字典加载 |
| `load_external(name, player_id) -> AntWarAgent` | 动态导入外部模块，优先调用 `create_agent()` |
| `scan_external() -> List[str]` | 扫描 baselines 目录下含 `ai.py` 的子目录 |
| `validate_baselines(names) -> bool` | 验证基线名称是否有效 |
| `serialize(agent) -> bytes` | 使用 `cloudpickle` 序列化 |
| `deserialize(data) -> AntWarAgent` | 反序列化 |

`KNOWN_BUILTIN_AGENTS` 映射：

| 名称 | 类 | 说明 |
|------|-----|------|
| `random` | `BasicRandomAI` | 随机策略 |
| `tower_first` | `BasicTowerAI` | 简单建塔策略 |
| `medium` | `MediumRuleAI` | 中级规则策略 |

### 4.4 OpponentAgent (`battle/opponent_agent.py`)

#### 4.4.1 职责

从训练检查点加载 PPO 策略网络，包装为可供对战引擎调用的 `AntWarAgent` 接口。

#### 4.4.2 核心方法

```python
@classmethod
def from_checkpoint(cls, checkpoint_path, device, hidden_dim=256, enable_auxiliary=True):
    # 1. 创建策略网络
    network = AntWarPolicyValueNetwork(hidden_dim, enable_auxiliary)
    # 2. 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['policy_state_dict'])
    network.eval()
    # 3. 创建 PPOAgent（确定性模式）
    return cls(policy=network, device=device)

def act(self, observation, deterministic=True):
    return self._ppo_agent.act(observation, deterministic=deterministic)
```

### 4.5 BaselineBattleConfig (`battle/battle_config.py`)

#### 4.5.1 职责

对战配置的数据类，集中管理所有对战相关参数。

| 字段 | 类型 | 说明 |
|------|------|------|
| `max_rounds` | int | 每局最大回合数 |
| `n_battles` | int | 对战局数 |
| `log_level` | str | 日志级别，默认 `"INFO"` |
| `log_dir` | Optional[str] | 日志目录 |
| `log_to_console` | bool | 是否输出到控制台 |
| `enable_timing` | bool | 是否启用计时 |
| `timing_warning_threshold` | float | 超时阈值（秒），默认 2.0 |
| `device` | str | 设备，默认 `"auto"` |

### 4.6 结果聚合与报告

#### 4.6.1 aggregate_results (`battle/result_aggregator.py`)

将多场对战的结果列表聚合为统计摘要：

```python
def aggregate_results(results: List[Dict]) -> Dict:
    # 统计字段：
    # total_battles, completed_battles, error_battles
    # agent1_wins, agent2_wins, draws
    # agent1_first_move_wins, agent1_second_move_wins
    # avg_rounds, total_duration
```

#### 4.6.2 BattleReporter (`battle/report_generator.py`)

生成可读的对战报告：

```python
report = generate_report(results, "PPO_Agent", "random")
# 输出示例：
# 对战报告: PPO_Agent vs random
# 总场次: 20 | 完成: 18 | 错误: 2
# 胜: 15 | 负: 3 | 平: 0
# 胜率: 83.33%
# 平均回合: 285.3
# 先手胜率: 88.89% | 后手胜率: 77.78%
```

### 4.7 对战日志

#### 4.7.1 BattleLogger (`battle/battle_logger.py`)

详细的回合级日志记录器。创建独立的日志目录 `{agent1}_vs_{agent2}_{timestamp}/`，注册 5 个 loguru handler（debug/info/warning/error/all），记录每局开始/结束、每步动作、奖励、计时等信息。

#### 4.7.2 EvalBattleWriter (`monitor/eval_battle_writer.py`)

结构化的对战评估日志写入器。以 JSONL 格式写入 `eval_battle_log.jsonl`，并在 EpisodeBatch 窗口满时聚合基线胜率。

### 4.8 工具类

#### 4.8.1 process_isolation (`battle/utils/process_isolation.py`)

上下文管理器，在子进程启动前从 `sys.modules` 中清理指定模块，确保干净的模块环境。

```python
with process_isolation(module_prefixes=["ai", "common", "SDK"]):
    # 在此上下文中，指定模块已被清除
    # 退出后自动恢复
```

#### 4.8.2 timing (`battle/utils/timing.py`)

`BattleTimer` 类，测量对战执行时间，超过 `warning_threshold` 时自动记录 WARNING 日志。

#### 4.8.3 exception_logging (`battle/utils/exception_logging.py`)

`log_exception(e, context)` 函数，记录异常详细信息后重新抛出。

---

## 5. 外部交互

### 5.1 本模块依赖的外部模块

| 模块 | 依赖内容 | 使用方式 |
|------|---------|---------|
| 游戏环境 (ActionMaskHandler) | 动作 ID 转 Operation | `PPOWarAgent` 内部使用 |
| 游戏环境 (BackendAdapter) | 创建游戏运行时 | `_execute_battle()` 中调用 |
| 策略网络 (AntWarPolicyValueNetwork) | 加载 PPO 检查点 | `OpponentAgent.from_checkpoint()` |
| PPO 训练引擎 (PPOAgent) | 推理封装 | `OpponentAgent` 内部使用 |
| 基础设施 (action_constants) | 塔位置等常量 | `PPOWarAgent` 内部使用 |

### 5.2 依赖本模块的外部模块

| 模块 | 使用内容 | 使用方式 |
|------|---------|---------|
| 自对弈编排 (`selfplay.py`) | `BattleCoordinator` | 在 `_run_battle_evaluation()` 中创建和使用 |
| 外部脚本（独立使用） | `BattleCoordinator`, `AgentLoader` | 直接创建实例进行对战评估 |

### 5.3 典型交互场景

#### 场景 1：自对弈训练中的周期性基线评估

1. `SelfPlayTrainer._run_battle_evaluation(episode=5000)` 被调用
2. 从 `PPOTrainer` 获取当前策略的 `PPOAgent`（确定性模式）
3. 创建 `BattleCoordinator` 实例
4. 调用 `coordinator.run_ppo_evaluation_with_agent(agent, ['random', 'tower_first', 'medium'], num_episodes=10)`
5. 内部对每个基线执行 10 × 2 = 20 场对战
6. 结果聚合后写入 `EvalBattleWriter`
7. 评估完成后主循环继续

#### 场景 2：独立评估两个 PPO 检查点

1. 外部脚本创建 `BaselineBattleConfig`
2. 通过 `OpponentAgent.from_checkpoint()` 加载两个检查点
3. 创建 `BattleSimulator` 并调用 `run_battles(agent1, agent2, num_episodes=50)`
4. 多进程并行执行 50 × 2 = 100 场对战
5. 调用 `aggregate_results()` 聚合结果
6. 调用 `BattleReporter.generate_report()` 生成报告

#### 场景 3：添加新的基线 AI

1. 在 `agent_loader.py` 的 `KNOWN_BUILTIN_AGENTS` 中注册新的 Agent 类
2. 新类必须继承 `AntWarAgent` 并实现 `choose_operations(state)`
3. 或者，在 `baselines/` 目录下创建子目录，包含 `ai.py` 文件
4. `ai.py` 中需要定义 `create_agent(player_id)` 函数或 `AI` 类
5. `AgentLoader.scan_external()` 会自动发现新的外部基线

---

## 6. 关键设计决策

### 6.1 多进程并行执行 + 异常降级

`BattleSimulator` 的对战执行策略：

1. **首选**：使用 `ProcessPoolExecutor` + `spawn` 上下文启动多进程
2. **降级**：如果多进程失败（如 CUDA 环境不支持多进程），降级为顺序执行

这种设计兼顾了性能和兼容性。`cloudpickle` 序列化确保 agent 对象可以跨进程传递。

### 6.2 AntWarAgent 统一抽象

所有对战相关的智能体都继承自 `AntWarAgent` 抽象基类：

```
AntWarAgent
  ├─ PPOWarAgent (训练策略)
  ├─ BasicRandomAI (内置基线)
  ├─ BasicTowerAI (内置基线)
  ├─ MediumRuleAI (内置基线)
  └─ 外部 Agent (动态加载)
```

这种设计使得：
- `BattleSimulator` 无需关心 agent 的具体类型
- 新的 agent 类型只需实现 `choose_operations()` 方法即可接入
- 外部基线 AI 无需改动核心代码

### 6.3 每 episode 两场对局（先手/后手）

与自对弈训练的 swap 设计一致，对战评估也执行先手和后手各一场：
- 每 `num_episodes` 生成 `num_episodes * 2` 场对局
- 结果聚合时区分先手胜率和后手胜率
- 消除先手优势对评估结果的影响

### 6.4 序列化 + 多进程的设计

Agent 对象通过 `cloudpickle` 序列化后传递给子进程，而非共享内存：
- **避免锁竞争**：每个子进程拥有独立的 agent 副本
- **隔离性**：子进程崩溃不影响主进程
- **兼容 CUDA**：但 CUDA 环境下多进程创建存在限制，因此有顺序执行降级

### 6.5 内置基线 AI 的设计意图

| 基线 AI | 意图 | 预期胜率 |
|---------|------|---------|
| `BasicRandomAI` | 验证模型是否学到了基本能力（远高于随机） | > 95% |
| `BasicTowerAI` | 验证模型是否超越了简单规则 | > 70% |
| `MediumRuleAI` | 验证模型是否达到了中级规则水平 | > 50% |

这些基线 AI 提供了从易到难的评估梯度，帮助判断训练的进度。
