# 第 1 章 总体架构概览

> 基于 `ppo/src/ppo_antwar` 源码分析。

本章从宏观层面描述 PPO v1 中 SelfPlay 训练与 Baseline Battle 两大对战通路的整体架构、关键代码位置以及核心类之间的协作关系。

---

## 1.1 两大对战通路

PPO v1 的训练与评估体系包含两条独立的对战通路，它们在目的、频率、核心类和环境使用方式上有显著差异。

### 1.1.1 SelfPlay 通路

SelfPlay 通路是 PPO 训练的核心驱动引擎。它通过 Agent 与自身历史版本（对手池中的对手）持续对战，收集经验数据供 PPO 更新使用。SelfPlay 在**每一个 training step** 都会执行，负责：

- 从对手池中选择对手（exploit/explore 权衡）
- 执行 Agent vs Opponent 的对局（含先后手交替）
- 收集对局中的 (state, action, reward, log_prob, value) 轨迹
- 将轨迹提交给 PPO 训练循环进行策略更新

核心类包括 [SelfPlayTrainer](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py)（训练循环编排）、[SelfPlayManager](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py)（对手管理与数据收集）以及 [PPOTrainer](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py)（PPO 算法更新）。

### 1.1.2 Baseline Battle 通路

Baseline Battle 通路是评估通路，用于**定期**检测当前 PPO Agent 的战斗力水平。它将当前 Agent 与一组固定的 Baseline Agent（如 BasicRandomAI、BasicTowerAI 等内置 AI 或历史 checkpoint）进行对战，生成统计报告和胜率数据。

Baseline Battle 的触发是**间歇性**的，由 `battle_interval` 参数控制（默认每 1000 个 episode 触发一次），也可通过 [BattleEvalCallback](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_callback.py) 在 checkpoint 保存时触发。

核心类包括 [BattleCoordinator](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_coordinator.py)（协调编排）、[BattleSimulator](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py)（并行对战引擎）以及 [run_single_battle_process](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py)（子进程单局对战）。

### 1.1.3 对比表格

| 维度 | SelfPlay 通路 | Baseline Battle 通路 |
|---|---|---|
| **对战类型** | Agent vs 历史对手（SelfPlay） | Agent vs 内置/基线 AI |
| **目的** | 收集训练数据，驱动 PPO 策略更新 | 评估当前 Agent 战斗力，生成报告 |
| **频率** | 每个 training step | 间歇性（默认每 1000 episode） |
| **核心类** | `SelfPlayTrainer` → `SelfPlayManager` → `PPOTrainer` | `BattleCoordinator` → `BattleSimulator` → `run_single_battle_process` |
| **环境方式** | `env.step()` 双 step 模式（通过 env factory 注入） | `PythonBackendState.resolve_turn()` 单刀模式（直接操作 SDK 后端） |
| **先后手** | 每局交替（`swap_positions`） | 每个 episode 含先手 + 后手两局（episode 编号区分） |
| **并行方式** | 单进程顺序执行 | 多进程并行（`ProcessPoolExecutor`），失败自动降级为顺序执行 |
| **输出** | 经验轨迹（EpisodeBatch） | 对战结果报告 + 胜率统计 |

---

## 1.2 关键代码位置

### 1.2.1 `ppo/src/ppo_antwar/` 完整目录树

```
ppo/src/ppo_antwar/
├── __init__.py                             # 顶层包导出
├── battle/                                 # Baseline Battle 模块
│   ├── __init__.py                         # battle 模块导出
│   ├── utils/                              # battle 工具函数
│   │   ├── __init__.py
│   │   ├── exception_logging.py            # 子进程异常日志记录
│   │   ├── process_isolation.py            # 进程隔离装饰器 @ensure_process_isolation
│   │   └── timing.py                       # 计时与性能警告工具
│   ├── agent_loader.py                     # Agent 加载器（PPO checkpoint / 内置 AI）
│   ├── battle_callback.py                  # BattleEvalCallback — 训练回调触发评估
│   ├── battle_config.py                    # BaselineBattleConfig — 对战配置模型
│   ├── battle_coordinator.py               # BattleCoordinator — 对战协调器
│   ├── battle_logger.py                    # BattleLogger / SubProcessLogger — 日志系统
│   ├── battle_simulator.py                 # BattleSimulator — 并行对战引擎
│   ├── report_generator.py                 # ReportGenerator — 评估报告生成
│   ├── result_aggregator.py                # ResultAggregator — 结果聚合
│   └── selfplay_logger.py                  # SelfPlay 日志记录（battle 子包内副本）
├── callbacks/                              # 训练回调模块
│   ├── __init__.py
│   ├── base_ppo_callback.py                # 基础回调接口
│   ├── battle_eval_callback.py             # Battle 评估回调
│   ├── callback_factory.py                 # 回调工厂
│   ├── checkpoint_callback.py              # Checkpoint 保存回调
│   └── metrics_callback.py                 # 训练指标回调
├── compat/                                 # 兼容适配层
│   ├── __init__.py
│   └── adapters.py                         # BackendAdapter / MatchRuntimeWrapper
├── config/                                 # 配置模块
│   ├── __init__.py
│   ├── config_parser.py                    # 配置解析（YAML → EasyDict）
│   └── path_config.py                      # 路径配置管理
├── configs/                                # 配置文件目录
│   └── ppo_antwar.yaml                     # 默认 PPO 训练配置
├── league/                                 # 联盟 / 对手管理模块
│   ├── __init__.py
│   ├── opponent_selector.py                # OpponentSelector / OpponentPool
│   └── payoff.py                           # TrueSkill 评分 / BattleSharedPayoff
├── monitor/                                # 监控模块
│   ├── __init__.py
│   ├── constants.py                        # 监控相关常量
│   ├── logger.py                           # Logger — 训练日志记录
│   ├── system_metrics_sampler.py           # 系统指标采样器
│   ├── time_tracker.py                     # TimeTracker — 时间追踪
│   └── training_logger.py                  # 训练日志输出
├── network/                                # 神经网络模块
│   ├── __init__.py
│   └── antwar_net.py                       # AntWarPolicyValueNetwork + count_parameters
├── trainer/                                # 训练核心模块
│   ├── __init__.py
│   ├── ppo_trainer.py                      # PPOTrainer / PPOAgent / EpisodeBatch / train_ppo
│   ├── selfplay.py                         # SelfPlayManager / SelfPlayTrainer
│   └── selfplay_logger.py                  # SelfPlayLogger — SelfPlay 专用日志
└── utils/                                  # 通用工具模块
    ├── __init__.py
    ├── action_constants.py                 # 动作空间常量（96 维动作 ID 映射）
    └── config.py                           # 配置工具（get_config / merge_config）
```

### 1.2.2 核心源文件说明

| 文件 | 核心内容 | 所属通路 |
|---|---|---|
| [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) | `PPOTrainer`（L60）、`PPOAgent`（L899）、`EpisodeBatch`（L34）、`train_ppo()` | SelfPlay |
| [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py) | `SelfPlayManager`（L22）、`SelfPlayTrainer`（L140） | SelfPlay |
| [battle_coordinator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_coordinator.py) | `BattleCoordinator`（L15） | Baseline Battle |
| [battle_simulator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py) | `BattleSimulator`（L413）、`run_single_battle_process()`（L18） | Baseline Battle |
| [battle_config.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_config.py) | `BaselineBattleConfig` | Baseline Battle |
| [agent_loader.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/agent_loader.py) | `AgentLoader`、内置 AI 类（`BasicRandomAI`、`BasicTowerAI` 等） | Baseline Battle |
| [battle_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_logger.py) | `BattleLogger`、`SubProcessLogger`、`LogLevel` | Baseline Battle |
| [report_generator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/report_generator.py) | `ReportGenerator`、`generate_report()` | Baseline Battle |
| [result_aggregator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/result_aggregator.py) | `ResultAggregator`、`aggregate_results()` | Baseline Battle |
| [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py) | `ACTION_DIM`、动作 ID 映射常量 | 共享（两者） |
| [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py) | `AntWarPolicyValueNetwork`、`count_parameters()` | 共享（两者） |
| [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py) | `OpponentSelector`、`OpponentPool` | SelfPlay |
| [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py) | `BattleSharedPayoff`、TrueSkill 评分 | SelfPlay |
| [features.py (SDK)](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/features.py) | `FeatureExtractor` — 观察编码（28 channel board + 33 dim global） | 共享（两者） |
| [actions.py (SDK)](file:///Users/raymondzeng/Documents/trae_projects/AntWar/Ant-Game/SDK/utils/actions.py) | `ActionCatalog` — 动作掩码生成 | 共享（两者） |

> **说明**：`env/` 目录在 `ppo/src/ppo_antwar/` 中不存在。游戏环境通过 `env_factory` 回调从外部注入，观察编码、动作掩码、奖励计算等底层逻辑位于 Ant-Game SDK 中（`Ant-Game/SDK/utils/features.py`、`Ant-Game/SDK/utils/actions.py`、`Ant-Game/SDK/training/env.py`）。

---

## 1.3 核心类关系图

PPO v1 的架构由两条核心类协作链组成：一条对应 SelfPlay 训练，一条对应 Baseline Battle 评估。

### 1.3.1 SelfPlay 训练链

```
PPOTrainer ──→ SelfPlayTrainer ──→ SelfPlayManager
   (PPO 算法)     (训练循环编排)        (对手池 + 数据收集)
```

**类职责**：

- **[PPOTrainer](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L60-L898)**：PPO 算法的核心实现。负责 GAE 计算、多 epoch 小批量更新、Policy Loss + Value Loss + Entropy Bonus 计算、梯度裁剪、NaN 鲁棒性处理。不直接参与对战逻辑，只消费 SelfPlay 产生的经验数据。

- **[PPOAgent](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L899-L1045)**：Agent 的神经网络策略包装类。封装了 [AntWarPolicyValueNetwork](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py)，提供 `act()`（训练用，带采样）、`select_action()`（评估用，deterministic）、`evaluate()`（返回动作分布 + value）三个核心接口。

- **[SelfPlayTrainer](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L140-L576)**：SelfPlay 训练循环的总编排者。内部持有 `PPOTrainer` 和 `SelfPlayManager`，在 `train()` 方法中循环执行：选择对手 → 收集 episode 数据 → PPO 更新 → 更新 payoff → 触发 Baseline Battle 评估。

- **[SelfPlayManager](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L22-L139)**：对手池与数据收集的管理器。持有 `OpponentSelector`（对手选择器）、`OpponentPool`（对手池）、`BattleSharedPayoff`（战绩/评分），负责选择对手、执行对局数据收集（`_collect_episode_with_swap`）、更新 payoff 和对手池。

### 1.3.2 Baseline Battle 评估链

```
BattleCoordinator ──→ BattleSimulator ──→ run_single_battle_process()
   (协调编排 + 报告)     (并行对战引擎)         (子进程单局对战)
```

**类职责**：

- **[BattleCoordinator](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_coordinator.py#L15-L373)**：Baseline Battle 的总协调器。负责初始化 Logger、AgentLoader 和 BattleSimulator，编排完整的评估流程：加载 Agent → 创建 Simulator → 执行并行对战 → 聚合结果 → 生成报告。提供 `run_evaluation()`、`run_ppo_evaluation()`、`run_ppo_evaluation_with_agent()` 和 `on_checkpoint_saved()` 等接口。别名 `BattleManager`。

- **[BattleSimulator](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L413-L576)**：并行对战引擎。负责加载 SDK 后端（`_load_sdk` → `PythonBackendState`），通过 `run_battles_parallel()` 方法使用 `ProcessPoolExecutor` 多进程并行执行子进程对战，失败时自动降级为 `_run_battles_sequential()` 顺序执行。每局对战通过 cloudpickle 序列化 agent 传入子进程。

- **[run_single_battle_process](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_simulator.py#L18-L410)**：子进程中执行的单局对战函数。在进程隔离环境中：清理 `sys.modules` → 加载 SDK 后端 → 反序列化 agent → 调用 `_execute_battle()` 执行回合制对战 → 返回对战结果。

### 1.3.3 全貌关系图

```
┌──────────────────────────────────────────────────────────────────┐
│                        PPO v1 训练/评估体系                         │
│                                                                  │
│  ┌── SelfPlay 训练通路 ─────────────────────────────────────┐    │
│  │                                                          │    │
│  │  SelfPlayTrainer.train()                                 │    │
│  │    │                                                     │    │
│  │    ├─ SelfPlayManager.select_opponent()                  │    │
│  │    │    └─ OpponentSelector (exploit/explore)            │    │
│  │    │    └─ OpponentPool (历史 Agent 快照)                 │    │
│  │    │                                                     │    │
│  │    ├─ SelfPlayManager._collect_episode_with_swap()       │    │
│  │    │    └─ env.step() × 2 (双 step 模式)                 │    │
│  │    │    └─ PPOAgent.act() (采样动作)                     │    │
│  │    │                                                     │    │
│  │    ├─ PPOTrainer.train_ppo()  ← EpisodeBatch             │    │
│  │    │    └─ GAE + Clipped PPO + Entropy                   │    │
│  │    │                                                     │    │
│  │    ├─ SelfPlayManager.update_payoff()                    │    │
│  │    │    └─ BattleSharedPayoff + TrueSkill rate_1vs1()    │    │
│  │    │                                                     │    │
│  │    └─ (每隔 battle_interval)　┐                           │    │
│  │                              ▼                           │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                │                                  │
│  ┌── Baseline Battle 评估通路 ──────────────────────────────┐    │
│  │                                                          │    │
│  │  BattleCoordinator.run_ppo_evaluation_with_agent()       │    │
│  │    │                                                     │    │
│  │    ├─ 遍历 baseline_agents[]                             │    │
│  │    │    ├─ AgentLoader.load_agent(name)                  │    │
│  │    │    │                                              │    │
│  │    │    └─ BattleSimulator.run_battles_parallel()        │    │
│  │    │         ├─ ProcessPoolExecutor                      │    │
│  │    │         │    └─ run_single_battle_process(task)     │    │
│  │    │         │         ├─ load_backend()                 │    │
│  │    │         │         ├─ _execute_battle()              │    │
│  │    │         │         └─ _determine_result()            │    │
│  │    │         │                                          │    │
│  │    │         └─ 失败降级 → _run_battles_sequential()     │    │
│  │    │                                                   │    │
│  │    ├─ aggregate_results()                               │    │
│  │    └─ generate_report()                                 │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
