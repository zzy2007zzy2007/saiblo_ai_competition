# PPO-AntWar 全局部署结构图与日志结构图

## 1. 项目整体架构

### 1.1 模块划分概览

PPO-AntWar 是一个基于 Proximal Policy Optimization (PPO) 的 AI 对战训练框架，主要包含以下核心模块：

| 模块 | 职责 | 核心文件 |
|------|------|----------|
| **trainer** | PPO训练核心逻辑 | `ppo_trainer.py`, `selfplay.py` |
| **battle** | 对战协调与执行 | `battle_coordinator.py`, `battle_simulator.py` |
| **league** | 对手选择与排名 | `opponent_selector.py`, `payoff.py` |
| **network** | 策略网络定义 | `antwar_net.py` |
| **monitor** | 日志与监控系统 | `logger.py`, `training_logger.py` |
| **callbacks** | 训练回调机制 | `callback_factory.py`, `checkpoint_callback.py` |
| **config** | 配置管理 | `path_config.py`, `config_parser.py` |

---

## 2. 文件路径结构图

### 2.1 源码目录结构

```
ppo/src/ppo_antwar/
├── battle/                    # 对战模块
│   ├── utils/                 # 对战工具函数
│   │   ├── __init__.py
│   │   ├── exception_logging.py   # 异常日志处理
│   │   ├── process_isolation.py   # 进程隔离
│   │   └── timing.py              # 计时工具
│   ├── __init__.py
│   ├── agent_loader.py       # Agent加载器
│   ├── battle_callback.py    # 对战回调
│   ├── battle_config.py      # 对战配置
│   ├── battle_coordinator.py # 对战协调器（核心）
│   ├── battle_logger.py      # 对战日志记录器
│   ├── battle_simulator.py   # 对战模拟器
│   ├── report_generator.py   # 报告生成器
│   ├── result_aggregator.py  # 结果聚合器
│   └── selfplay_logger.py    # SelfPlay日志
├── callbacks/                 # 回调模块
│   ├── __init__.py
│   ├── base_ppo_callback.py  # 回调基类
│   ├── battle_eval_callback.py # 对战评估回调
│   ├── callback_factory.py   # 回调工厂
│   ├── checkpoint_callback.py # 检查点回调
│   └── metrics_callback.py   # 指标回调
├── compat/                   # 兼容性模块
│   ├── __init__.py
│   └── adapters.py           # 适配器
├── config/                   # 配置模块
│   ├── __init__.py
│   ├── config_parser.py      # 配置解析器
│   └── path_config.py        # 路径配置（核心）
├── configs/                  # 配置文件目录
│   └── ppo_antwar.yaml       # PPO配置文件
├── league/                   # 联赛模块
│   ├── __init__.py
│   ├── opponent_selector.py  # 对手选择器
│   └── payoff.py             # 对战结果记录
├── monitor/                  # 监控模块
│   ├── __init__.py
│   ├── constants.py          # 常量定义
│   ├── logger.py             # 日志管理（核心）
│   ├── system_metrics_sampler.py # 系统指标采样
│   ├── time_tracker.py       # 时间追踪器
│   └── training_logger.py    # 训练日志记录器
├── network/                  # 网络模块
│   ├── __init__.py
│   └── antwar_net.py         # AntWar策略网络
├── trainer/                  # 训练模块
│   ├── __init__.py
│   ├── ppo_trainer.py        # PPO训练器（核心）
│   ├── selfplay.py           # SelfPlay训练器
│   └── selfplay_logger.py    # SelfPlay日志记录
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── action_constants.py   # 动作常量
│   └── config.py             # 工具配置
└── __init__.py               # 模块入口
```

### 2.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPOTrainer                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 训练循环 ─→ PPO更新 ─→ 指标记录 ─→ Checkpoint保存         │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Network     │  │   Monitor     │  │  SelfPlay     │
│ antwar_net.py │  │   logger.py   │  │  selfplay.py  │
│  策略网络     │  │   日志系统    │  │  自我对弈     │
└───────────────┘  └───────┬───────┘  └───────┬───────┘
                           │                  │
                           │                  ▼
                           │         ┌───────────────┐
                           │         │    League     │
                           │         │opponent_      │
                           │         │ selector.py   │
                           │         │   对手选择    │
                           │         └───────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │     Battle        │
                 │ battle_           │
                 │ coordinator.py    │
                 │   对战协调        │
                 └───────────────────┘
```

---

## 3. 日志结构图

### 3.1 日志目录结构（运行时生成）

根据 `PathConfig` 的定义，日志和输出文件存储在项目根目录的 `outputs/` 目录下：

```
outputs/                        # 根输出目录（由PathConfig管理）
├── training/                   # 训练日志
│   ├── training.log            # 训练主日志
│   ├── training_start.log      # 训练启动信息
│   ├── training_complete.log   # 训练完成信息
│   ├── training_details.log    # 每轮训练详情
│   ├── training_metrics.json   # 训练指标（含历史）
│   ├── training_status.json    # 当前训练状态
│   ├── error.log               # 错误日志（JSON格式）
│   ├── warning.log             # 警告日志（JSON格式）
│   ├── model_saves.log         # 模型保存记录
│   ├── battle_results.log      # 对战结果记录
│   └── opponent_selection.log  # 对手选择记录
├── checkpoint/                 # 模型检查点
│   ├── final_model.pt          # 最终模型
│   ├── checkpoint_xxx.pt       # 训练过程检查点
│   └── league_state/           # 联赛状态持久化
│       ├── opponent_pool.json  # 对手池状态
│       ├── payoff.json         # Payoff矩阵
│       └── checkpoint_map.json # Checkpoint映射
├── battle/                     # 对战日志
│   └── {agent1}_vs_{agent2}_{timestamp}/
│       ├── battle_all.log      # 全级别日志
│       ├── battle_debug.log    # DEBUG级别
│       ├── battle_info.log     # INFO级别
│       ├── battle_warning.log  # WARNING级别
│       ├── battle_error.log    # ERROR级别
│       └── battle_results.json # 对战结果JSON
├── selfplay/                   # SelfPlay日志
│   ├── selfplay_battles/
│   │   ├── selfplay_battle_stats.json # 统计汇总
│   │   └── detailed_battles/   # 详细对战记录
│   │       └── selfplay_xxx.json
│   └── selfplay.log            # SelfPlay主日志
└── system/                     # 系统监控
    ├── time_statistics.json    # 时间统计
    └── system_metrics.json     # 系统指标（CPU/GPU等）
```

### 3.2 日志类型说明

| 日志文件 | 生成位置 | 内容说明 |
|----------|----------|----------|
| `training.log` | `training/` | 训练过程主日志，包含所有级别日志 |
| `training_metrics.json` | `training/` | 训练指标（episode、reward、loss等） |
| `training_status.json` | `training/` | 当前训练状态（胜率、速度、稳定性等） |
| `error.log` | `training/` | JSON格式错误日志，含上下文 |
| `warning.log` | `training/` | JSON格式警告日志 |
| `battle_results.log` | `training/` | 对战结果汇总 |
| `battle_*.log` | `battle/{agent1}_vs_{agent2}_*/` | 按级别分离的对战日志 |
| `selfplay_battle_stats.json` | `selfplay/` | SelfPlay对战统计 |
| `system_metrics.json` | `system/` | 系统资源使用指标 |

### 3.3 日志流向架构

```
┌───────────────────────────────────────────────────────────────┐
│                        日志系统架构                           │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│   │   Trainer   │    │   Battle    │    │   SelfPlay      │  │
│   │   训练器    │    │   对战模块  │    │   自我对弈      │  │
│   └──────┬──────┘    └──────┬──────┘    └────────┬────────┘  │
│          │                  │                     │           │
│          ▼                  ▼                     ▼           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│   │ Training    │    │ Battle      │    │ SelfPlay        │  │
│   │ Logger      │    │ Logger      │    │ Logger          │  │
│   │ 训练日志器  │    │ 对战日志器  │    │ SelfPlay日志器  │  │
│   └──────┬──────┘    └──────┬──────┘    └────────┬────────┘  │
│          │                  │                     │           │
│          └────────┬─────────┴─────────────────────┘           │
│                   ▼                                           │
│          ┌─────────────────┐                                  │
│          │     Logger      │                                  │
│          │   (loguru)      │                                  │
│          │   日志管理器    │                                  │
│          └────────┬────────┘                                  │
│                   │                                           │
│     ┌─────────────┼─────────────┬─────────────────┐           │
│     ▼             ▼             ▼                 ▼           │
│  ┌───────┐   ┌─────────┐   ┌───────────┐   ┌───────────┐    │
│  │ stdout│   │ .log    │   │ .json     │   │ .json.log │    │
│  │ 控制台 │   │ 文本日志│   │ JSON状态  │   │ JSON日志  │    │
│  └───────┘   └─────────┘   └───────────┘   └───────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 3.4 日志记录层次

```
日志级别层次：
├── DEBUG    → battle_debug.log, training.log
├── INFO     → battle_info.log, training.log, 控制台
├── WARNING  → battle_warning.log, warning.log, training.log
└── ERROR    → battle_error.log, error.log, training.log
```

### 3.5 关键日志记录点

| 记录点 | 触发时机 | 记录内容 |
|--------|----------|----------|
| `log_training_start` | 训练启动 | 配置、设备、系统信息 |
| `save_training_metrics` | 每轮训练 | episode、reward、loss、entropy |
| `log_training_details` | 每轮训练 | 详细指标文本日志 |
| `save_training_status` | 定期/事件 | 完整训练状态JSON |
| `log_training_complete` | 训练完成 | 总览统计 |
| `log_model_save` | 模型保存 | 保存路径 |
| `log_battle_result` | 对战结束 | 对手、结果、奖励 |
| `log_opponent_selection` | 选择对手 | episode、对手名称 |
| `log_exception` | 异常发生 | 异常类型、消息、上下文 |

---

## 4. 部署结构图

### 4.1 组件部署关系

```
┌───────────────────────────────────────────────────────────────────┐
│                        部署架构                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                     训练节点                                │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │   │
│  │  │   Trainer    │   │   Network    │   │   Monitor    │   │   │
│  │  │  PPO训练器   │   │  策略网络    │   │  日志监控    │   │   │
│  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │   │
│  │         │                  │                  │           │   │
│  │         └────────┬─────────┴──────────────────┘           │   │
│  │                  ▼                                       │   │
│  │         ┌──────────────┐                                 │   │
│  │         │   League     │                                 │   │
│  │         │  对手选择器  │                                 │   │
│  │         └──────┬───────┘                                 │   │
│  └────────────────┼─────────────────────────────────────────┘   │
│                   │                                             │
│                   ▼                                             │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                     对战节点                                │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │   │
│  │  │ Coordinator  │   │  Simulator   │   │  AgentLoader │   │   │
│  │  │   协调器     │   │   模拟器     │   │    加载器    │   │   │
│  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │   │
│  │         │                  │                  │           │   │
│  │         └────────┬─────────┴──────────────────┘           │   │
│  │                  ▼                                       │   │
│  │         ┌──────────────┐                                 │   │
│  │         │  BattleLogger│                                 │   │
│  │         │   对战日志器 │                                 │   │
│  │         └──────────────┘                                 │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 数据流向

```
训练数据流：
  环境交互 → 数据收集 → PPO更新 → 指标记录 → 模型保存
     │              │           │           │
     ▼              ▼           ▼           ▼
  EpisodeBatch   Tensor转换   梯度更新    Logger

对战数据流：
  Agent加载 → 对战执行 → 结果聚合 → 报告生成 → 日志记录
     │           │           │           │
     ▼           ▼           ▼           ▼
  AgentLoader  Simulator  Aggregator  BattleLogger
```

### 4.3 关键路径说明

| 路径类型 | 路径 | 说明 |
|----------|------|------|
| **输出根目录** | `outputs/` | 所有运行时输出 |
| **训练日志** | `outputs/training/` | 训练过程日志和指标 |
| **检查点** | `outputs/checkpoint/` | 模型权重和联赛状态 |
| **对战日志** | `outputs/battle/` | 对战记录 |
| **SelfPlay日志** | `outputs/selfplay/` | SelfPlay对战记录 |
| **系统监控** | `outputs/system/` | 系统资源指标 |

---

## 5. 核心类关系图

### 5.1 训练核心类

```
PPOTrainer
  ├── policy: AntWarPolicyValueNetwork  # 策略网络
  ├── optimizer: Adam                    # 优化器
  ├── logger: Logger                     # 日志管理器
  ├── time_tracker: TimeTracker          # 时间追踪
  ├── system_metrics_sampler             # 系统指标采样
  └── battle_coordinator: BattleCoordinator  # 对战协调

SelfPlayTrainer
  ├── trainer: PPOTrainer                # 底层训练器
  ├── selfplay_manager: SelfPlayManager  # SelfPlay管理器
  ├── opponent_pool: OpponentPool        # 对手池
  ├── payoff: BattleSharedPayoff         # 对战矩阵
  └── selfplay_logger: SelfPlayLogger    # SelfPlay日志
```

### 5.2 对战核心类

```
BattleCoordinator
  ├── config: BaselineBattleConfig       # 对战配置
  ├── logger: BattleLogger               # 对战日志
  ├── agent_loader: AgentLoader          # Agent加载
  └── simulator: BattleSimulator         # 对战模拟

BattleLogger
  ├── handlers: List[loguru.Handler]     # 多级别日志处理器
  ├── _total_battles: int                # 总对战场数
  ├── _agent1_wins/losses/draws          # 统计数据
  └── _timings: Dict[str, float]         # 计时数据
```

---

## 6. 总结

### 6.1 架构特点

1. **模块化设计**：功能清晰分离，各模块职责明确
2. **日志分层**：训练日志、对战日志、系统日志独立管理
3. **持久化策略**：检查点、联赛状态、训练状态均可恢复
4. **可扩展性**：回调机制支持灵活扩展功能
5. **多进程安全**：通过 loguru enqueue 机制支持多进程日志

### 6.2 关键设计决策

| 设计点 | 决策 | 原因 |
|--------|------|------|
| 日志框架 | loguru | 高性能、多进程安全、配置灵活 |
| 路径管理 | PathConfig | 统一管理、自动创建目录、验证依赖路径 |
| 指标缓存 | MetricsCache | 环形缓冲区，内存友好 |
| 对战协调 | BattleCoordinator | 集中管理对战流程 |
| 对手选择 | OpponentSelector | 平衡探索与利用 |