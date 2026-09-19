# 模块5：对战评估系统

## 1. 概述

### 1.1 模块定位

对战评估系统负责 **在训练过程中衡量模型的实际强度**。它通过与内置规则 AI 和外部 AI 进行对战，收集胜负统计数据，生成可读的报告，从而帮助判断训练是否有效推进。

### 1.2 在整体架构中的位置

```
模块4: 自对弈训练编排
    │
    │ 周期性调用: _run_battle_evaluation()
    ▼
┌───────────────────────────────────────────────┐
│              模块5: 对战评估系统                 │
│                                                 │
│  BattleCoordinator (一站式对战入口)              │
│      │                                          │
│      ├── AgentLoader (加载对战 AI)               │
│      │     ├── PPOWarAgent (PPO 驱动)           │
│      │     ├── BasicRandomAI                    │
│      │     ├── BasicTowerAI                     │
│      │     ├── MediumRuleAI                     │
│      │     └── 外部 AI (动态加载)                │
│      │                                          │
│      ├── BattleSimulator (多进程并行对战)        │
│      │     └── _execute_battle()               │
│      │                                          │
│      ├── ResultAggregator (结果聚合)             │
│      └── BattleReporter (报告生成)               │
│                                                 │
│  辅助工具:                                       │
│  ├── BattleLogger (对战日志)                     │
│  ├── SubProcessLogger (子进程日志)               │
│  ├── BattleTimer (计时)                          │
│  ├── exception_logging (异常记录)                │
│  └── process_isolation (进程隔离)               │
└───────────────────────────────────────────────┘
    │
    │ 输出: 胜率、对局统计、报告
    ▼
模块7: 监控与可观测性 (EvalBattleWriter)
```

### 1.3 业务目标

- 提供标准化的 PPO vs 基线 AI 对战评估能力
- 支持多进程并行对战以提高评估效率
- 支持多种 AI 对手（内置规则 + 外部动态加载）
- 生成详细的对战统计和可读报告
- 通过 OpponentAgent 从训练检查点加载对手进行对战

---

## 2. 背景与概念

### 2.1 为什么需要基线评估？

自对弈训练中，模型只与自身历史版本对战，评分体系是相对性的。基线评估通过与外部固定对手（规则 AI）对战，提供 **绝对强度** 的参考点：

- 如果模型在 TrueSkill 评分上持续提升，但对基线 AI 胜率没有提高，可能存在过拟合
- 基线 AI 的强度是稳定的参照物，便于跨训练阶段比较

### 2.2 内置 AI 类型

系统内置了三种规则 AI，强度递增：

| AI | 策略描述 | 难度 |
|----|---------|------|
| `BasicRandomAI` | 从所有合法动作中随机选择 | 极低 |
| `BasicTowerAI` | 优先使用超级武器 → 建塔 → 升级塔 → 科技升级 | 低 |
| `MediumRuleAI` | 建塔 → 升级塔 → 超级武器 → 加速产蚁 → 升级蚁 | 中 |

### 2.3 外部 AI 加载

系统支持从 `baselines/` 目录动态加载外部 AI。外部 AI 通过 `ai.py` 文件提供 `create_agent()` 工厂函数或 `AI` 类，实现 `choose_operations(state)` 接口。

### 2.4 多进程对战

Python GIL 限制了单进程的多线程并行，因此对战评估使用 `ProcessPoolExecutor` + `spawn` 上下文实现真正的多进程并行：

- 每场独立对战在单独的子进程中运行
- 通过 `cloudpickle` 序列化 Agent 传递给子进程
- `process_isolation` 确保子进程的模块环境干净

---

## 3. 架构设计

### 3.1 模块内部结构

```
┌──────────────────────────────────────────────────────┐
│                    对战评估系统                         │
│                                                        │
│  ┌──────────────────────────────────────────────┐     │
│  │           BattleCoordinator                   │     │
│  │   (协调器: 整合加载、执行、聚合、报告)          │     │
│  └──────┬──────────────┬──────────────┬─────────┘     │
│         │              │              │                │
│         ▼              ▼              ▼                │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐      │
│  │AgentLoader │ │Battle      │ │Result        │      │
│  │(Agent加载) │ │Simulator   │ │Aggregator    │      │
│  └─────┬──────┘ │(对战执行)   │ │(结果聚合)     │      │
│        │        └─────┬──────┘ └──────┬───────┘      │
│        │              │               │               │
│   ┌────┴────┐    ┌────┴────┐    ┌─────┴──────┐       │
│   │内置 AI  │    │多进程   │    │Battle      │       │
│   │外部 AI  │    │对战引擎 │    │Reporter    │       │
│   └─────────┘    │(并行/串行)│   │(报告生成)   │       │
│                  └─────────┘    └────────────┘       │
│                                                        │
│  ┌─────────────────────────────────────────────┐      │
│  │  对战辅助工具                                 │      │
│  │  ├── BattleLogger / SubProcessLogger         │      │
│  │  ├── BattleTimer                             │      │
│  │  ├── process_isolation                       │      │
│  │  └── exception_logging                       │      │
│  └─────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

### 3.2 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| `BattleCoordinator` | [battle/battle_coordinator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_coordinator.py) | 一站式评估入口：加载 → 执行 → 聚合 → 报告 |
| `BaselineBattleConfig` | [battle/battle_config.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_config.py) | 评估配置 dataclass |
| `BattleSimulator` | [battle/battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_simulator.py) | 多进程并行对战执行器 |
| `AgentLoader` | [battle/agent_loader.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/agent_loader.py) | Agent 加载（内置 + 外部 + 序列化） |
| `OpponentAgent` | [battle/opponent_agent.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/opponent_agent.py) | 从 checkpoint 加载的 PPO 对手 Agent |
| `BattleReporter` | [battle/report_generator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/report_generator.py) | 对战报告生成 |
| `ResultAggregator` | [battle/result_aggregator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/result_aggregator.py) | 对战结果聚合统计 |
| `BattleLogger` | [battle/battle_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_logger.py) | 对战会话日志（分级别/分文件） |

---

## 4. 核心组件详解

### 4.1 BattleCoordinator — 对战协调器

位置：[battle/battle_coordinator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_coordinator.py)

```
BattleCoordinator(config: BaselineBattleConfig, path_config: PathConfig)
```

一站式对战评估入口，整合了加载、执行、聚合、报告的全流程。

#### 关键方法

| 方法 | 说明 |
|------|------|
| `run_ppo_evaluation_with_agent(ppo_agent, baseline_agents, num_episodes)` | PPO vs 各基线 AI 的完整评估 |
| `run_batch(agent1, agent2, num_episodes)` | 两个指定 Agent 之间的对战 |

`run_ppo_evaluation_with_agent()` 返回：
```python
{
    "BasicTowerAI": {
        "wins": 42, "losses": 8, "draws": 0,
        "win_rate": 0.84, "total_battles": 50, ...
    },
    "MediumRuleAI": { ... }
}
```

---

### 4.2 BaselineBattleConfig — 评估配置

位置：[battle/battle_config.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_config.py)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_rounds` | `int` | 必填 | 单局最大回合数 |
| `n_battles` | `int` | 必填 | 总对局数 |
| `log_level` | `str` | `"INFO"` | 日志级别 |
| `log_dir` | `Optional[str]` | `None` | 日志目录 |
| `log_to_console` | `bool` | `True` | 是否控制台输出 |
| `enable_timing` | `bool` | `True` | 是否启用计时 |
| `timing_warning_threshold` | `float` | `2.0` | 超时警告阈值（秒） |
| `ppo_checkpoint_path` | `Optional[str]` | `None` | PPO checkpoint 路径 |
| `baseline_agents` | `List[str]` | `[]` | 基线 AI 名称列表 |
| `device` | `str` | `"auto"` | 计算设备 |

---

### 4.3 BattleSimulator — 多进程对战执行器

位置：[battle/battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_simulator.py)

```
BattleSimulator(config: BaselineBattleConfig)
```

#### 核心方法

```
run_battles(agent1, agent2, num_episodes) -> List[Dict]
```

**执行策略**：

1. **优先并行**：使用 `ProcessPoolExecutor` + `spawn` 上下文，最大 8 个 worker，每场对战超时 300 秒
2. **串行降级**：如果多进程全部失败，降级为单进程顺序执行

每个 episode 生成 2 场对战（先手/后手交换），以消除先后手偏差。

#### _execute_battle() — 单场对战逻辑

```python
_execute_battle(agent1, agent2, first_player, max_rounds) -> Dict:
    runtime = BackendAdapter.create_runtime(player=0)
    
    while not terminal and round_index < max_rounds:
        # 获取双方操作
        op1 = agent1.choose_operations(state)
        op2 = agent2.choose_operations(state)
        
        # 根据 first_player 分配操作
        if first_player == 0:
            play0_ops, play1_ops = op1, op2
        else:
            play0_ops, play1_ops = op2, op1
        
        # 解析回合
        resolution = runtime.resolve_turn(play0_ops, play1_ops)
        state = resolution.state
        terminal = resolution.terminal
        round_index += 1
    
    return {
        'result': 'agent1_win' | 'agent2_win' | 'draw' | 'error',
        'first_player': 0 | 1,
        'total_rounds': int,
        'final_hp': {'hp_0': agent1_hp, 'hp_1': agent2_hp},
        'duration': float,
    }
```

---

### 4.4 AgentLoader — Agent 加载器

位置：[battle/agent_loader.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/agent_loader.py)

#### Agent 继承体系

```
AntWarAgent (基类)
  ├── PPOWarAgent (PPO 驱动的智能体)
  ├── BasicRandomAI (随机合法动作)
  ├── BasicTowerAI (规则策略：建塔/升级/超级武器)
  ├── MediumRuleAI (中级规则策略)
  └── OpponentAgent (从 checkpoint 加载的对手)
```

#### AgentLoader 核心方法

| 方法 | 说明 |
|------|------|
| `load_builtin(name, player_id=0)` | 从 `KNOWN_BUILTIN_AGENTS` 加载内置 AI |
| `scan_external()` | 扫描 `baselines/` 目录下的外部 AI |
| `load_external(name, player_id=0)` | 动态导入外部 AI 模块（支持 `create_agent()` 工厂或 `AI` 类） |
| `load(name, player_id=0)` | 统一加载接口（内置优先，外部回退） |
| `validate_baselines(names)` | 验证基线名称的存在性 |
| `serialize(agent)` | `cloudpickle` 序列化（用于多进程传递） |
| `deserialize(data)` | `cloudpickle` 反序列化 |

#### PPOWarAgent

将 PPO 策略网络适配为标准对战接口：

```python
PPOWarAgent(player_id, ppo_agent, observation_encoder, action_mask_handler)
    choose_operations(state):
        obs = observation_encoder.encode(state, player_id)
        action_id = ppo_agent.act(obs, deterministic=True)
        op = action_mask_handler.action_id_to_op(action_id)
        return [op] if op else []
```

---

### 4.5 OpponentAgent — 对手 Agent

位置：[battle/opponent_agent.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/opponent_agent.py)

从训练检查点加载的 PPO 模型，用于评估对战中的对手角色。

```python
OpponentAgent.from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    hidden_dim: int = 256,
    enable_auxiliary: bool = True
) -> OpponentAgent
```

- 从 `.pt` 文件加载 `policy_state_dict`
- 以 `eval()` 模式运行（无 dropout/batchnorm 随机性）
- 确定性推理（`deterministic=True`）
- `exploration_epsilon=0.0`（不使用探索噪声）

---

### 4.6 ResultAggregator — 结果聚合器

位置：[battle/result_aggregator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/result_aggregator.py)

```
aggregate_results(results: List[Dict]) -> Dict:
    return {
        'total_battles': int,       # 总对局数
        'completed_battles': int,   # 正常完成数
        'error_battles': int,       # 出错数
        'agent1_wins': int,         # agent1 胜
        'agent2_wins': int,         # agent2 胜
        'draws': int,               # 平局
        'agent1_first_move_wins': int,  # agent1 先手胜
        'agent1_first_move_battles': int, # agent1 先手局数
        'agent1_second_move_wins': int,   # agent1 后手胜
        'agent1_second_move_battles': int, # agent1 后手局数
        'avg_rounds': float,        # 平均回合数
        'total_duration': float,    # 总耗时
    }
```

`ResultAggregator` 是有状态的聚合器包装，支持逐步 `add(result)` 然后 `aggregate()`。

---

### 4.7 BattleLogger — 对战会话日志

位置：[battle/battle_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/battle_logger.py)

为每场评估会话创建独立的日志目录：`{agent1}_vs_{agent2}_{timestamp}/`

日志分 4 个级别（DEBUG/INFO/WARNING/ERROR）各写独立文件，外加一个 `all.log` 汇总。

| 方法 | 说明 |
|------|------|
| `start_battle_session(agent1, agent2)` | 创建日志目录和 handler |
| `close_battle_session()` | 关闭所有 handler |
| `log_battle_start(episode)` | 记录对局开始 |
| `log_battle_end(episode, result, rounds, duration)` | 记录对局结束 |
| `log_round(episode, round_num, state)` | 记录每回合状态 |
| `log_action(episode, step, action, reward, value)` | 记录动作详情 |
| `write_summary()` | 写入汇总报告 |

`SubProcessLogger` 是轻量级版本，用于子进程（不创建目录/handler，但提供相同接口）。

---

### 4.8 辅助工具

位置：[battle/utils/](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/battle/utils/)

| 文件 | 功能 |
|------|------|
| `exception_logging.py` | `log_exception(e, context)` — 记录异常详情并重新抛出 |
| `process_isolation.py` | `process_isolation(module_prefixes)` — 上下文管理器，子进程启动前清理指定前缀的模块 |
| `timing.py` | `BattleTimer(warning_threshold)` — `start()` / `stop()` / `duration`，超阈值时记录警告 |

---

## 5. 数据流与时序

### 5.1 完整评估流程

```
BattleCoordinator.run_ppo_evaluation_with_agent(ppo_agent, baseline_agents)
  │
  ├── for baseline in baseline_agents:
  │     │
  │     ├── 1. AgentLoader.load(baseline) → baseline_agent
  │     │
  │     ├── 2. BattleSimulator.run_battles(ppo_agent, baseline_agent, n_battles)
  │     │       │
  │     │       ├── (并行) ProcessPoolExecutor.map()
  │     │       │    └── n_battles 个任务 → cloudpickle 序列化 Agent
  │     │       │         └── run_single_battle_process() → _execute_battle()
  │     │       │
  │     │       └── (降级) 串行 _execute_battle()
  │     │
  │     ├── 3. aggregate_results(results) → stats
  │     │       └── {wins, losses, draws, win_rate, first_move_win_rate, avg_rounds, ...}
  │     │
  │     ├── 4. BattleReporter.add_battle_result(baseline, stats)
  │     │
  │     └── 5. 保存 stats 到 eval_battle_log.jsonl (通过 EvalBattleWriter)
  │
  └── BattleReporter.generate_summary() → 汇总报告字符串
```

### 5.2 单场对战时序

```
_execute_battle(agent1, agent2, first_player, max_rounds)
  │
  ├── BackendAdapter.create_runtime(player=0)
  ├── state = runtime.reset()
  │
  ├── while not terminal and round < max_rounds:
  │     │
  │     ├── agent1.choose_operations(state) → ops1
  │     ├── agent2.choose_operations(state) → ops2
  │     │
  │     ├── if first_player == 0: p0_ops, p1_ops = ops1, ops2
  │     │   else:                 p0_ops, p1_ops = ops2, ops1
  │     │
  │     ├── resolution = runtime.resolve_turn(p0_ops, p1_ops)
  │     ├── state = resolution.state
  │     ├── terminal = resolution.terminal
  │     └── round += 1
  │
  └── return {result, first_player, total_rounds, final_hp, duration}
```

---

## 6. 对外交互

### 6.1 被依赖关系

| 调用方 | 场景 | 使用方法 |
|--------|------|---------|
| **SelfPlayTrainer** (模块4) | 周期性基线评估 | `battle_coordinator.run_ppo_evaluation_with_agent()` |

### 6.2 对下游模块的依赖

| 依赖模块 | 使用方式 |
|---------|---------|
| **模块1（游戏环境）** | 通过 SDK backend 创建运行时驱动游戏（非 Gymnasium 接口，直接使用 SDK API） |
| **模块2（神经网络）** | PPOWarAgent 和 OpponentAgent 内部持有 `AntWarPolicyValueNetwork` |
| **模块7（监控）** | EvalBattleWriter 记录评估结果到 JSONL 文件 |
| **模块8（基础设施）** | BackendAdapter 创建游戏运行时，PathConfig 管理日志输出路径 |

---

## 7. 配置参数

### 7.1 BaselineBattleConfig 参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `max_rounds` | 单局最大回合数 | 512 |
| `n_battles` | 总对局数 | 50-200 |
| `log_level` | 日志级别 | `"INFO"` |
| `enable_timing` | 是否启用计时 | `True` |
| `timing_warning_threshold` | 超时警告阈值（秒） | 2.0 |
| `baseline_agents` | 基线 AI 列表 | `["BasicTowerAI", "MediumRuleAI"]` |
| `device` | 计算设备 | `"auto"` (自动选择 CPU/CUDA) |

### 7.2 并行对战参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `max_workers` | 最大并行进程数 | 8 |
| `timeout_per_battle` | 单场对战超时（秒） | 300 |

---

## 8. 常见问题与注意事项

### 8.1 多进程的 spawn 上下文

子进程必须使用 `spawn` 而非 `fork` 启动，因为：
- `fork` 会导致 CUDA 上下文在子进程中不可用
- `fork` 可能继承主进程的文件描述符和锁状态
- `spawn` 创建干净的新 Python 解释器

### 8.2 cloudpickle 序列化

Python 标准 `pickle` 无法序列化 PyTorch 模型和闭包。`cloudpickle` 是第三方库，支持更广泛的对象序列化，解决了 Agent 在进程间传递的问题。

### 8.3 进程隔离（process_isolation）

子进程启动时会清理主进程中加载的模块缓存（`ai`, `common`, `SDK`, `AI` 等前缀），防止模块状态污染导致子进程行为异常。

### 8.4 先手/后手平衡

每个 episode 生成 2 场对战（先手交换），确保先手优势不会歪曲评估结果。聚合统计时会分开计算先手胜率和后手胜率。

### 8.5 对战超时与降级

如果多进程对战全部失败（如 CUDA 不可用在子进程），系统自动降级为串行执行，保证评估流程不被阻塞。

### 8.6 OpponentAgent vs PPOWarAgent

两者都封装 PPO 网络用于对战，但用途不同：
- **PPOWarAgent**：在 `BattleCoordinator` 中作为 PPO 方，依赖 `ObservationEncoder` + `ActionMaskHandler` 做观测编码和动作解码
- **OpponentAgent**：在 `SelfPlayTrainer` 中作为对手方，直接从 checkpoint 加载，仅需要 `get_action()` 输出 action_id
