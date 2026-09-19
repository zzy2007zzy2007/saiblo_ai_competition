# 自对弈编排模块 (Self-Play Orchestration)

## 1. 背景与定位

### 1.1 业务问题

训练一个游戏 AI 智能体需要一个完整的训练循环：创建环境、选择对手、采集数据、更新模型、评估进度、记录日志。如果所有这些逻辑都混杂在 PPO 算法实现中，代码将难以维护和扩展。**自对弈编排模块**的核心业务目标是：**编排整个训练流程**，将数据收集、模型更新、对手管理、日志记录等环节组织为一个可配置的训练循环，使得上层只需要调用一个 `train()` 方法即可启动完整的自对弈训练。

"自对弈 (Self-Play)" 是指智能体与自身的历史版本对战：随着训练的进行，当前策略会定期保存为对手快照，这些快照构成对手池。智能体与这些对手对战，从对战结果中学习。这种训练方式在 AlphaGo/AlphaZero 等项目中证明了其有效性。

### 1.2 在整个系统中的位置

```
┌──────────────────────────────────────────────────────────────────┐
│                    自对弈编排 (Self-Play Orchestration)           │
│                                                                  │
│  SelfPlayTrainer.train()                                         │
│    ├─ 创建日志子系统                                             │
│    ├─ 主循环: 每局                                              │
│    │   ├─ 通过联赛模块选择对手                                   │
│    │   ├─ 通过游戏环境采集数据                                    │
│    │   ├─ 通过 PPO 训练引擎更新模型                               │
│    │   ├─ 记录对战和训练日志                                     │
│    │   └─ 周期性: 保存对手快照 / 基线评估 / 更新联赛状态         │
│    └─ 关闭日志子系统                                             │
│                                                                  │
│  使用的其他模块:                                                  │
│    ├─ 游戏环境 (AntWarEnv) ─── 创建环境                          │
│    ├─ PPO 训练引擎 (PPOTrainer) ─ 采集数据 + 更新                │
│    ├─ 联赛管理 (SelfPlayManager) ─ 选择对手                      │
│    ├─ 对战执行 (BattleCoordinator) ─ 基线评估                    │
│    └─ 基础设施 (PathConfig, config) ─ 配置和路径                 │
└──────────────────────────────────────────────────────────────────┘
```

- **最上层**：`SelfPlayTrainer` 是整个训练系统的入口，调用 `train()` 启动训练
- **编排者**：负责协调所有其他模块，自身不实现具体算法
- **集成者**：日志子系统（logger, time_tracker, system_metrics_sampler 等）统一在此创建和管理

### 1.3 与其他模块的关系

| 外部模块 | 关系 | 交互方式 |
|---------|------|---------|
| PPO 训练引擎 | 依赖 | 创建 `PPOTrainer`，调用 `collect_episode()` 和 PPO 更新 |
| 游戏环境 | 依赖 | 工厂函数创建 `AntWarEnv` 实例 |
| 联赛管理 | 依赖 | 通过 `SelfPlayManager` 选择对手和更新评分 |
| 对战执行 | 依赖 | 通过 `BattleCoordinator` 执行基线对战评估 |
| 策略网络 | 依赖 | 用于创建初始随机对手 |
| 基础设施 (PathConfig/config) | 依赖 | 路径管理和配置加载 |

---

## 2. 核心概念

### 2.1 关键术语

| 术语 | 说明 |
|------|------|
| **Self-Play (自对弈)** | 智能体与自己的历史版本（保存为对手快照）对战来学习 |
| **Episode** | 一局完整的对战（从 reset 到 terminal） |
| **Swap Positions** | 轮流先手/后手，消除先手优势带来的偏差 |
| **Opponent Snapshot** | 定期保存的策略网络检查点，构成对手池 |
| **Baseline Evaluation** | 与固定的基线 AI（随机/规则 AI）对战，评估当前策略的绝对水平 |
| **Payoff** | 对战评分矩阵，记录每对对手的 TrueSkill 评分 |
| **EPM** | Episodes Per Minute，训练速度指标 |
| **Auxiliary Labels** | 从对局轨迹中计算的辅助任务标签（塔伤害、金币收入的未来视界） |

### 2.2 训练主循环概要

```
SelfPlayTrainer.train()
│
├─ 初始化: 日志子系统、PPO trainer、联赛管理器、环境工厂
│
├─ 主循环 (episode = start_episode ... total_episodes):
│   │
│   ├─ 选择对手: manager.select_opponent(episode)
│   │
│   ├─ 采集一局数据: _collect_episode_with_swap()
│   │   ├─ 先手半局: 己方先手，对手后手
│   │   ├─ 后手半局: 己方后手，对手先手
│   │   └─ 合并两条轨迹为一个 EpisodeBatch
│   │
│   ├─ 记录对局日志: selfplay_battle_writer.write()
│   │
│   ├─ 更新评分: manager.update_payoff(opponent_id, result)
│   │
│   ├─ 累积批次: 将 batch 加入累积队列
│   │
│   ├─ 如果累积步数 > update_timesteps:
│   │   ├─ 合并累积的 batches
│   │   ├─ 调用 PPOTrainer._ppo_update()
│   │   ├─ 更新学习率: trainer.update_lr_schedule()
│   │   └─ 记录训练指标
│   │
│   ├─ 周期性任务 (每 N 局):
│   │   ├─ 保存对手快照: _update_opponent()
│   │   ├─ 基线评估: _run_battle_evaluation()
│   │   ├─ 保存联赛状态: manager.save_state()
│   │   └─ 保存训练状态: compute_and_save_training_status()
│   │
│   └─ 更新训练进度: manager.update_progress()
│
└─ 关闭: 所有 logger、writer
```

---

## 3. 架构总览

### 3.1 模块组成

```
self-play-orchestration/
│
├── trainer/selfplay.py                  # SelfPlayTrainer - 训练主循环
├── trainer/selfplay_logger.py           # SelfPlayLogger - 自对弈日志
├── trainer/training_status.py           # compute_and_save_training_status
├── utils/aux_labels.py                  # 辅助标签计算
│
├── monitor/logger.py                    # Logger - 训练主日志
├── monitor/training_logger.py           # TrainingLogger - 训练日志封装
├── monitor/time_tracker.py              # TimeTracker - 时间追踪
├── monitor/system_metrics_sampler.py    # SystemMetricsSampler - 系统指标
└── monitor/selfplay_battle_writer.py    # SelfPlayBattleWriter - 对战记录
```

### 3.2 组件关系

```
SelfPlayTrainer
│
├── 训练核心
│   ├── PPOTrainer (PPO训练引擎)
│   ├── SelfPlayManager (联赛管理)
│   ├── env_factory → AntWarEnv (游戏环境)
│   └── BattleCoordinator (对战执行)
│
├── 日志系统
│   ├── Logger / TrainingLogger → 训练事件日志
│   ├── SelfPlayLogger → 自对弈事件日志（debug/info/warning/error）
│   ├── SelfPlayBattleWriter → 每局对战结构化日志
│   ├── TimeTracker → 时间统计
│   └── SystemMetricsSampler → CPU/GPU/内存采样
│
├── 数据处理
│   ├── EpisodeBatch → 轨迹数据
│   └── compute_aux_labels_from_trajectory → 辅助标签
│
└── 状态管理
    ├── training_status → JSON 训练状态
    └── opponent checkpoints → 对手快照
```

---

## 4. 核心组件详解

### 4.1 SelfPlayTrainer (`trainer/selfplay.py`)

#### 4.1.1 职责

自对弈训练循环的最高层编排器。整个训练过程的"大脑"，协调所有子模块完成训练。

#### 4.1.2 构造函数

```python
def __init__(self, config: Dict[str, Any])
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | Dict | 完整训练配置，包含 ppo/training/network/selfplay 等所有子节 |

**初始化流程**：
1. 解析配置（设备、路径、PPO 参数、自对弈参数等）
2. 创建 `PathConfig` 实例管理路径
3. 创建 `PPOTrainer` 实例
4. 创建 `SelfPlayManager` 实例（联赛管理器）
5. 创建日志子系统（`_create_logging_subsystem`）
6. 创建初始随机对手
7. 加载已有的联赛状态（如果指定了 `load_path`）

#### 4.1.3 公开方法

| 方法 | 功能 |
|------|------|
| `train()` | 启动自对弈训练主循环 |

#### 4.1.4 关键算法：训练主循环 (`train`)

```
train()
│
└─ try:
    │
    1. 设置环境工厂 env_factory()
    │
    2. 创建日志子系统:
       ├─ TimeTracker (时间追踪)
       ├─ Logger / TrainingLogger (训练日志)
       ├─ SelfPlayLogger (自对弈日志)
       ├─ SelfPlayBattleWriter (对战写入)
       ├─ BatchMetricsWriter (训练指标写入)
       ├─ EpisodeBatchWriter (窗口聚合写入)
       ├─ EvalBattleWriter (评估对战写入)
       └─ SystemMetricsSampler (系统指标采样)
    │
    3. for episode in range(start_episode, total_episodes):
       │
       a. swap = episode % 2 == 0  (交替先手)
       b. opponent_id, opponent_agent = manager.select_opponent(episode)
       │
       c. batch = _collect_episode_with_swap(env_0, env_1, opponent_agent, swap)
       │
       d. selfplay_battle_writer.write_battle(result_dict)
       │
       e. manager.update_payoff(opponent_id, battle_result)
       │
       f. accumulated_batches.append(batch)
       │
       g. 如果 accumulated_steps >= update_timesteps:
          ├─ merged = EpisodeBatch.merge(accumulated_batches)
          ├─ metrics = trainer._ppo_update(merged, episode)
          ├─ trainer.update_lr_schedule(episode, total_episodes)
          ├─ batch_metrics_writer.write(metrics)
          ├─ accumulated_batches = []
          └─ time_tracker.record_episode(...)
       │
       h. 周期性任务 (每 opponent_save_interval / eval_interval):
          ├─ _update_opponent(episode)  → 保存对手快照
          ├─ _run_battle_evaluation(episode)  → 基线评估
          ├─ manager.save_state()  → 保存联赛状态
          ├─ time_tracker.save_time_statistics()
          └─ compute_and_save_training_status(...)
       │
       i. manager.update_progress(episode / total_episodes)
       │
       j. 回调触发: trainer callbacks on_step()
    │
    └─ finally:
        ├─ 关闭所有 logger/writer
        └─ 最终保存：联赛状态、时间统计、训练状态
```

#### 4.1.5 关键算法：单局数据收集 (`_collect_episode_with_swap`)

```
_collect_episode_with_swap(env_0, env_1, opponent_agent, swap_positions)
│
├─ 创建两个环境的观测
│
├─ 先手半局:
│   ├─ 己方 = 先手 (player=swap ? 1 : 0)
│   ├─ 己方使用 _trainer._select_action()
│   ├─ 对手使用 opponent_agent.act()
│   ├─ 调用 env.step({player_0_action, player_1_action})
│   ├─ 收集轨迹到 batch_0
│   └─ 记录每步快照
│
├─ 后手半局:
│   ├─ 己方 = 后手 (player=swap ? 0 : 1)
│   ├─ 同上逻辑
│   └─ 收集轨迹到 batch_1
│
├─ 计算辅助标签:
│   ├─ tower_labels, gold_labels = compute_aux_labels_from_trajectory(snapshots)
│   ├─ batch_0.aux_tower_damage = tower_labels
│   └─ batch_0.aux_gold_income = gold_labels
│
├─ 合并两条轨迹: merged = EpisodeBatch.merge([batch_0, batch_1])
│
└─ 返回 merged + 对战详情字典
```

**关键细节**：通过 `swap_positions` 交替先后手，每条 episode 包含两个半局（先手视角 + 后手视角），两条轨迹合并后用于 PPO 更新。这消除了先手优势带来的数据偏差。

#### 4.1.6 基线对评估 (`_run_battle_evaluation`)

```
_run_battle_evaluation(episode)
│
├─ agent = trainer.get_agent(deterministic=True)
│
├─ for baseline_name in baseline_agents:
│   ├─ result = battle_coordinator.run_ppo_evaluation_with_agent(
│   │       agent, [baseline_name], num_eval_episodes)
│   ├─ eval_battle_writer.write_battle(result)
│   └─ 记录胜率
│
├─ 聚合结果 → 写入 EpisodeBatch 窗口
│
└─ 返回评估汇总
```

### 4.2 SelfPlayLogger (`trainer/selfplay_logger.py`)

#### 4.2.1 职责

为自对弈过程提供独立的日志文件系统。创建 5 个级别的日志文件（debug/info/warning/error/all），使用 `loguru` 的 `bind(category="selfplay")` 区分上下文。

#### 4.2.2 公开方法

| 方法 | 功能 |
|------|------|
| `log_battle_start(episode, opponent_id, swap)` | 记录对局开始事件 |
| `log_battle_end(episode, result, rounds, reward)` | 记录对局结束事件 |
| `log_opponent_added(opponent_id, checkpoint_path)` | 记录新对手添加事件 |
| `log_opponent_removed(opponent_id, reason)` | 记录对手淘汰事件 |

### 4.3 日志子系统组件

#### 4.3.1 Logger (`monitor/logger.py`)

训练主日志管理器，是所有结构化日志文件的统一入口。提供以下核心方法：

| 方法 | 功能 |
|------|------|
| `log_training_start(config_dict)` | 写入 `training_start.log`，记录完整训练配置 |
| `log_training_complete(summary_dict)` | 写入 `training_complete.log`，记录训练汇总 |
| `save_battle_stats(stats_dict)` | 写入 `battle_stats.json` |
| `log_json_state(filepath, state_dict)` | 追加写入任意 JSON 状态变化 |

#### 4.3.2 TrainingLogger (`monitor/training_logger.py`)

`Logger` 的简易封装，为训练循环提供 `log_training_start()` 和 `log_training_complete()` 两个接口，内部委托给 `Logger` 实例。

#### 4.3.3 TimeTracker (`monitor/time_tracker.py`)

追踪训练和对战中的时间消耗指标：

| 方法 | 功能 |
|------|------|
| `record_episode(duration)` | 记录一局时长 |
| `start_training()` / `end_training()` | 训练起止 |
| `start_battle()` / `end_battle()` | 对局起止 |
| `save_time_statistics()` | 保存时间统计到 JSON |
| `get_epm()` | 获取当前 EPM（Episodes Per Minute） |

#### 4.3.4 SystemMetricsSampler (`monitor/system_metrics_sampler.py`)

在后台守护线程中周期性采样系统资源：

- CPU 使用率（通过 `psutil`）
- 内存使用量（通过 `psutil`）
- GPU 使用率、显存、温度（通过 `pynvml`，可选）
- 按阶段（training / battle）分别统计
- 每 `SAMPLES_PER_WRITE` 次采样刷新一次 JSON 文件

#### 4.3.5 SelfPlayBattleWriter (`monitor/selfplay_battle_writer.py`)

每局自对弈结束后，将结构化对战数据以 JSONL 格式写入 `selfplay_battle_log.jsonl`。记录字段：

| 字段 | 说明 |
|------|------|
| `episode` | 局号 |
| `swap` | 是否交换先后手 |
| `opponent_id` | 对手 ID |
| `result` | 胜/负/平 |
| `rounds` | 总回合数 |
| `reward` | 总奖励 |
| `hp_damage`, `tower_damage` | 伤害统计 |
| `coins` | 金币统计 |
| `action_counts` | 各类型动作计数 |
| `action_rewards` | 各类型动作奖励 |
| `reward_sources` | 8 种奖励来源的分量 |

### 4.4 Auxiliary Labels (`utils/aux_labels.py`)

#### 4.4.1 职责

从对局轨迹的状态快照中，计算辅助任务的标签数据。这些标签在训练时由策略网络的辅助头预测，用于计算辅助 MSE 损失。

#### 4.4.2 核心函数

```python
def compute_aux_labels_from_trajectory(snapshots, num_horizons=5):
```

对每个时间步 t，计算 5 个未来视界 `[1, 2, 4, 8, 16]` 的：
- 己方塔 HP 减少量
- 敌方塔 HP 减少量
- 己方金币增加量
- 敌方金币增加量

返回两个 `(T, 10)` 的 ndarray：`tower_damage_labels` 和 `gold_income_labels`。

---

## 5. 外部交互

### 5.1 本模块依赖的外部模块

| 模块 | 依赖内容 | 使用方式 |
|------|---------|---------|
| PPO 训练引擎 | `PPOTrainer`, `EpisodeBatch` | 创建实例，收集轨迹，触发 PPO 更新 |
| 游戏环境 | `AntWarEnv` | 通过工厂函数创建环境 |
| 联赛管理 | `SelfPlayManager` | 选择对手、更新评分、保存/加载状态 |
| 对战执行 | `BattleCoordinator`, `AgentLoader` | 基线对战评估 |
| 策略网络 | `AntWarPolicyValueNetwork` | 创建初始随机对手 |
| 基础设施 | `PathConfig`, `config_parser`, `callback_factory` | 路径管理、配置、回调 |

### 5.2 依赖本模块的外部模块

本模块是整个系统的**顶层入口**，没有其他业务模块依赖它。`__init__.py` 导出了 `SelfPlayTrainer` 供外部脚本使用。

### 5.3 典型交互场景

#### 场景 1：启动一次完整的自对弈训练

1. 外部脚本加载 YAML 配置 → `create_ppo_config()`
2. 创建 `SelfPlayTrainer(config)`
3. 初始化过程：
   - `PPOTrainer` 创建策略网络和优化器
   - `SelfPlayManager` 创建初始对手池
   - 日志子系统全部初始化
4. 调用 `trainer.train()` 启动主循环
5. 训练完成后，所有 logger 自动关闭

#### 场景 2：周期性基线评估

1. 每 `eval_interval` 局，`SelfPlayTrainer` 暂停主循环
2. 从 `PPOTrainer` 获取当前策略的 `PPOAgent`（确定性模式）
3. 创建 `BattleCoordinator`，传入 agent 和基线 AI 列表
4. `BattleCoordinator.run_ppo_evaluation_with_agent()` 执行多轮对战
5. 结果写入 `EvalBattleWriter`
6. 评估完成后主循环继续

#### 场景 3：对手快照管理

1. 每 `opponent_save_interval` 局：
   - 调用 `PPOTrainer.save_checkpoint()` 保存当前策略到 `opponent_checkpoints/` 目录
   - 调用 `SelfPlayManager.add_opponent(episode, checkpoint_path)` 注册新对手
2. 如果对手池已满，`SelfPlayManager` 内部触发淘汰逻辑：
   - 计算每个对手的评分 = TrueSkill mu + 保护加成
   - 淘汰评分最低的对手
   - 删除对应的检查点文件
   - 记录日志（`log_opponent_removed`）

---

## 6. 关键设计决策

### 6.1 自对弈编排 vs PPO 训练引擎的分离

这是整个系统最重要的架构决策。两者分离的理由：

- **关注点不同**：PPO 训练引擎关心"如何从轨迹学习"（算法），自对弈编排关心"如何组织训练"（流程）
- **可测试性**：PPO 引擎可以单独用合成数据测试；编排可以单独用 mock 训练器测试
- **可替换性**：如果想换用其他算法（如 SAC、DQN），只需替换 PPO 训练引擎，编排层几乎不需要修改

### 6.2 交替先后手（Swap Positions）

每局训练数据包含先后手各一个半局：
- 消除先手优势带来的数据偏差
- 让模型学会"无论先手后手都能打好"
- 轨迹合并后，PPO 更新时无法区分先后手，迫使策略学会通用的决策能力

### 6.3 日志子系统的集中管理

所有日志组件（Logger、Writer、TimeTracker、Sampler）都在 `SelfPlayTrainer` 的 `_create_logging_subsystem` 中统一创建，并在 `finally` 块中统一关闭：

- **生命周期管理**：确保资源不会泄漏
- **目录一致性**：所有日志输出到由 `PathConfig` 统一管理的目录结构
- **可扩展性**：添加新的日志组件只需在创建和关闭两个位置修改

### 6.4 对手快照的"保护期"

在 `SelfPlayManager` 管理对手池时，新加入的对手有一个保护期：
- `保护加成 = max(0, min_games_threshold - games) * 0.5`
- 这使得新对手不会因为刚开始成绩差就被立即淘汰
- 保证对手池有足够的多样性

### 6.5 辅助标签的在线计算

辅助标签（塔伤害、金币收入）在数据收集过程中**在线计算**，而非训练时计算：

- 数据收集时：`_collect_episode_with_swap()` 中捕获每步状态快照
- 对局结束时：`compute_aux_labels_from_trajectory()` 从快照序列计算标签
- 标签存入 `EpisodeBatch.aux_*` 字段
- PPO 更新时：标签随 batch 一起传到 `compute_auxiliary_loss()`

这种设计避免了训练时的重复计算，但要求轨迹中保留完整的快照序列。
