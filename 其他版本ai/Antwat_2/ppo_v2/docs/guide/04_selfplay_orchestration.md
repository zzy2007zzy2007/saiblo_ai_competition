# 模块4：自对弈训练编排

## 1. 概述

### 1.1 模块定位

自对弈训练编排模块是训练系统的 **顶层调度器**。它驱动完整的训练主循环，协调数据收集、PPO 参数更新、对手池管理、TrueSkill 评分维护、基线评估和训练状态持久化等所有环节。

### 1.2 在整体架构中的位置

```
                    ┌──────────────────────────────┐
                    │  模块4: 自对弈训练编排         │
                    │                              │
                    │  SelfPlayTrainer.train()      │
                    │     │                        │
                    │     ├── SelfPlayManager       │
                    │     │   ├── OpponentPool      │
                    │     │   ├── OpponentSelector  │
                    │     │   └── BattleSharedPayoff│
                    │     │                        │
                    │     ├── EpisodeCollector ─────┤──► 模块1 (游戏环境)
                    │     ├── PPOTrainer.ppo_update─┤──► 模块3 (PPO训练引擎)
                    │     ├── BattleCoordinator ────┤──► 模块5 (对战评估)
                    │     ├── LoggingSubsystem ─────┤──► 模块7 (监控)
                    │     └── Callbacks ────────────┤──► 模块6 (回调)
                    └──────────────────────────────┘
```

### 1.3 业务目标

- 编排自对弈强化学习的完整训练循环
- 管理对手池（基于 TrueSkill 的添加/淘汰机制）
- 通过探索/利用策略选择合适的对手
- 周期性评估模型强度（vs 基线 AI）
- 持久化联赛状态以确保训练可恢复

---

## 2. 背景与概念

### 2.1 自对弈训练（Self-Play）

传统的强化学习训练通常与固定对手对战，容易过拟合。自对弈训练中，智能体的对手是其自身的 **历史版本**：

- 训练过程中定期将当前策略保存为新的对手
- 对手池中保存了多个不同时期的历史策略
- 当前策略与池中对手对战，持续挑战更强的对手

这种机制促使策略不断进化，类似 AlphaGo / AlphaZero 的训练方式。

### 2.2 TrueSkill 评分系统

TrueSkill 是微软研究院开发的贝叶斯评分系统，用 `(μ, σ)` 二元组表示玩家的技能水平：

- `μ`（均值）：估计的技能水平（默认 25.0）
- `σ`（标准差）：估计的不确定性（默认 8.33，σ 越小越确定）
- 置信区间：`μ ± 3σ`
- "已确定"（confident）：`σ < 3.0`

每场比赛后根据结果（胜负平）更新双方的 `(μ, σ)`。

### 2.3 探索与利用（Exploration vs Exploitation）

对手选择在两种策略间平衡：

- **利用（Exploitation）**：选择当前最强的对手（μ 最高且 σ 足够小的对手），最大化学习效率
- **探索（Exploration）**：选择不确定性高的对手（σ 大），探索多样化的对局体验

利用概率 `exploit_prob` 在训练过程中可以自适应调整（初始偏向探索，后期偏向利用）。

### 2.4 可利用性（Exploitability）

可利用性衡量当前策略相对于整个对手池的强度：

- `0.0`：当前策略碾压对手池（强大）
- `0.5`：与对手池势均力敌
- `1.0`：被对手池碾压（弱小）

通过计算当前策略对每个对手的胜率并取平均得到。

---

## 3. 架构设计

### 3.1 模块内部结构

```
┌───────────────────────────────────────────────────┐
│                自对弈训练编排模块                     │
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │      SelfPlayTrainer                │           │
│  │  (训练主循环驱动)                     │           │
│  └──────────────┬──────────────────────┘           │
│                 │                                   │
│     ┌───────────┼───────────┐                      │
│     │           │           │                       │
│     ▼           ▼           ▼                       │
│  ┌──────────┐ ┌────────┐ ┌──────────────┐         │
│  │SelfPlay  │ │Episode │ │Logging       │         │
│  │Manager   │ │Collector│ │Subsystem     │         │
│  └────┬─────┘ └────────┘ └──────────────┘         │
│       │                                             │
│  ┌────┴────────────────┐                           │
│  │                     │                            │
│  ▼                     ▼                            │
│  ┌────────────────┐ ┌──────────────────┐           │
│  │OpponentSelector│ │BattleSharedPayoff│           │
│  │ (对手选择)      │ │(TrueSkill 评分)   │           │
│  └────────┬───────┘ └────────┬─────────┘           │
│           │                  │                      │
│           ▼                  ▼                      │
│       ┌──────────────────────────┐                 │
│       │     OpponentPool         │                 │
│       │  (对手池: 添加/淘汰/管理)  │                 │
│       └──────────────────────────┘                 │
│                                                     │
│  ┌──────────────────────────────┐                  │
│  │  SelfPlayLogger              │                  │
│  │  training_status.py          │                  │
│  └──────────────────────────────┘                  │
└───────────────────────────────────────────────────┘
```

### 3.2 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| `SelfPlayTrainer` | [trainer/selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py) | 训练主循环驱动，编排所有子任务 |
| `LoggingSubsystem` | [trainer/selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py) | 统一管理所有 logger/writer 实例 |
| `SelfPlayManager` | [league/opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_selector.py) | 自对弈核心管理器，整合对手选择、评分更新 |
| `OpponentSelector` | [league/opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_selector.py) | 探索/利用策略的对手选择 |
| `BattleSharedPayoff` | [league/payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/payoff.py) | TrueSkill 评分矩阵管理 |
| `OpponentPool` | [league/payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/payoff.py) | 对手池管理（添加/淘汰） |
| `SelfPlayLogger` | [trainer/selfplay_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay_logger.py) | 自对弈事件日志 |

---

## 4. 核心组件详解

### 4.1 SelfPlayTrainer — 训练主循环

位置：[trainer/selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py)

#### 初始化

```
SelfPlayTrainer(trainer: PPOTrainer, config: Dict, path_config: PathConfig,
                battle_coordinator: BattleCoordinator, run_id: str = "default")
```

初始化时创建：
- `PPOTrainer`（模块3）— 算法核心
- `SelfPlayManager` — 对手选择与评分管理
- `EpisodeCollector` — 数据收集
- `LoggingSubsystem` — 日志/监控系统
- 3 个随机初始化的初始对手（`_create_initial_opponents()`）

#### 核心方法：train()

```
train(num_episodes: int, env_factory: Callable[[], AntWarEnv], callbacks=None) -> None
```

**主循环流程（每个 episode）**：

```
for episode in range(num_episodes):
    │
    ├── 1. 交替先手/后手位置（swap_positions）
    ├── 2. _select_opponent_agent() → 选择对手并加载
    ├── 3. env = env_factory() 创建新环境
    ├── 4. _collect_and_update_payoff()
    │       ├── EpisodeCollector.collect() → EpisodeBatch
    │       ├── 更新 BattleSharedPayoff
    │       └── 写入 SelfPlayBattleWriter
    ├── 5. 累积 batch 到 _batch_pool
    │
    ├── 6. 当 batch_pool 满足条件时：
    │       └── _flush_batch_to_update()
    │           ├── merge batches → 质量检查
    │           ├── trainer.ppo_update(batch, episode)
    │           └── 记录训练指标
    │
    ├── 7. 周期性任务：
    │       ├── 每 opponent_update_interval 局：
    │       │   └── _update_opponent() → 保存当前策略到对手池
    │       ├── 每 eval_interval 局：
    │       │   └── _run_battle_evaluation() → 基线评估
    │       ├── 每 checkpoint_interval 局：
    │       │   └── callbacks.on_step() → 保存检查点
    │       ├── 每 save_state_interval 局：
    │       │   └── _save_league_state() → 持久化联赛状态
    │       └── 每 training_status_interval 局：
    │           └── compute_and_save_training_status()
    │
    └── 8. 训练结束 → _finalize_training()
            ├── flush 剩余 batch
            ├── 保存最终检查点
            └── 输出训练完成日志
```

#### 对手选择逻辑

```
_select_opponent_agent() -> (opponent_id, agent):
    │
    ├── selfplay_manager.select_opponent() → opponent_id (或 None)
    │
    ├── if opponent_id is None:
    │       └── return None, None  # 无对手（自我对战）
    │
    └── else:
            ├── checkpoint_path = pool.get_checkpoint_path(opponent_id)
            └── OpponentAgent.from_checkpoint(checkpoint_path) → agent
```

---

### 4.2 SelfPlayManager — 自对弈管理器

位置：[league/opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_selector.py)

```
SelfPlayManager(opponent_pool_size: int, min_opponent_games: int,
                exploit_prob: float, total_episodes: int = 2000)
```

整合对手选择、评分更新和可利用性评估的统一入口。

| 方法 | 说明 |
|------|------|
| `select_opponent()` | 委托 OpponentSelector 选择对手 |
| `add_opponent(checkpoint_path, episode)` | 将当前策略加入对手池 |
| `update_payoff(opponent_id, result, rounds)` | 更新评分矩阵 |
| `update_progress(episode)` | 更新训练进度（影响自适应调度） |
| `get_opponent_checkpoint_path(opponent_id)` | 获取对手检查点路径 |
| `get_exploitability()` | 计算当前策略的可利用性 |
| `get_statistics()` | 获取对手池统计信息 |
| `save_state(pool_path, payoff_path)` | 持久化联赛状态 |
| `load_state(pool_path, payoff_path)` | 恢复联赛状态 |

---

### 4.3 OpponentSelector — 对手选择器

位置：[league/opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_selector.py)

```
OpponentSelector(payoff: BattleSharedPayoff, opponent_pool: OpponentPool,
                 exploit_prob: float, adaptive: bool = True,
                 schedule: str = "linear")
```

#### 选择策略

```
select(player_id: str) -> Optional[str]:
    │
    ├── if random() < exploit_prob:     # 利用模式
    │       └── 选择 μ 最高且 "已确定"（σ < 3.0）的对手
    │
    └── else:                            # 探索模式
            └── 选择 σ 最高的对手（不确定性大 → 收集更多对局数据）
```

#### 自适应调度

`exploit_prob` 在训练过程中动态调整：

| 调度策略 | 行为 |
|---------|------|
| `linear` | exploit_prob 从 0.1 线性增长到 0.9 |
| `sigmoid` | S 型曲线，后期加速增长 |
| `exp` | 指数增长 |

通过 `update_progress(episode, total_episodes)` 更新进度。

---

### 4.4 BattleSharedPayoff — TrueSkill 评分矩阵

位置：[league/payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/payoff.py)

```
BattleSharedPayoff(decay: float, min_win_rate_games: int)
```

维护所有玩家间对战的 TrueSkill 评分和胜率记录。

#### 关键方法

| 方法 | 说明 |
|------|------|
| `update(home, away, result)` | 更新比赛记录和 TrueSkill 评分。result: 1=home胜, -1=home负, 0=平局 |
| `get_win_rate(home, away)` | 贝叶斯胜率估计（含信度加权收缩） |
| `decay_all()` | 对所有历史记录应用衰减因子（淡化早期对局的影响） |
| `get_exploitability(current_player_id)` | 计算可利用性指标 |
| `get_all_win_rates(player_id)` | 获取对池中所有对手的胜率 |
| `get_payoff_matrix()` | N×N 胜率矩阵 |
| `save_state(filepath)` / `load_state(filepath)` | JSON 持久化 |

---

### 4.5 OpponentPool — 对手池管理

位置：[league/payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/payoff.py)

```
OpponentPool(max_size: int, min_games_threshold: int,
             payoff: Optional[BattleSharedPayoff] = None)
```

#### 添加对手

```
add(opponent_id: str, checkpoint_path: str, episode: int):
    ├── if pool is full:
    │       └── _evict_worst() → 淘汰最弱对手
    └── 加入新对手
```

#### 淘汰策略

```
_evict_worst() -> str:
    淘汰 μ + 保护加分 最低的对手
    - 保护分：最近添加的对手有保护期，期间不会被淘汰
    - 未达到 min_games_threshold 的对手不会被淘汰
```

---

### 4.6 LoggingSubsystem — 日志子系统容器

位置：[trainer/selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py)

统一管理所有日志/监控组件的生命周期。作为 dataclass，包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `loguru_handler_id` | `int` | loguru handler ID |
| `time_tracker` | `TimeTracker` | 训练时间追踪 |
| `logger` | `Logger` | 训练主日志 |
| `selfplay_logger` | `SelfPlayLogger` | 自对弈事件日志 |
| `battle_logger` | `SelfPlayBattleWriter` | 每局对战数据 |
| `batch_logger` | `BatchMetricsWriter` | 批次训练指标 |
| `ep_batch_logger` | `EpisodeBatchWriter` | Episode 聚合数据 |
| `eval_logger` | `EvalBattleWriter` | 基线评估记录 |
| `system_sampler` | `SystemMetricsSampler` | 系统指标采样 |

`close_all()` 按依赖顺序关闭所有组件。

---

## 5. 数据流与时序

### 5.1 完整训练周期时序图

```
Episode 1    Episode 2    ...    Episode N    Episode N+1   ...
  │            │                    │             │
  ├── 选对手    ├── 选对手          ├── 选对手     ├── 选对手
  ├── 对战      ├── 对战            ├── 对战       ├── 对战
  ├── 更新payoff├── 更新payoff      ├── 更新payoff ├── 更新payoff
  ├── 累积batch ├── 累积batch       ├── 累积batch  ├── 累积batch
  │            │                    │             │
  │            │              ┌─────┘             │
  │            │              │ batch_pool 满 ────┤
  │            │              ▼                   ▼
  │            │        ┌──────────────┐    ┌──────────────┐
  │            │        │ PPO Update   │    │ PPO Update   │
  │            │        └──────────────┘    └──────────────┘
  │            │                                │
  │            │                          ┌─────┘
  │            │                          │ 周期性任务触发
  │            │                          ▼
  │            │                    ┌──────────────┐
  │            │                    │ 添加对手      │
  │            │                    │ 基线评估      │
  │            │                    │ 保存检查点    │
  │            │                    └──────────────┘
```

### 5.2 对手生命周期

```
随机初始化 (3个) ──► 加入对手池
                        │
训练进行中...            │
                        ▼
当前策略 ──► 保存 checkpoint ──► 加入对手池 ──► 如果池满: 淘汰最弱
                        │
训练进行中...            │
                        ▼
                    每个对手与当前策略对战
                        │
                        ▼
                    TrueSkill 评分更新
                        │
                        ▼
                    评分低的对手被淘汰
```

---

## 6. 对外交互

### 6.1 与各模块的交互

| 交互对象 | 场景 | 具体调用 |
|---------|------|---------|
| **模块3（PPOTrainer）** | PPO 更新 | `trainer.ppo_update()`, `trainer.select_action()`, `trainer.save_checkpoint()` |
| **模块5（BattleCoordinator）** | 基线评估 | `battle_coordinator.run_ppo_evaluation_with_agent()` |
| **模块6（Callbacks）** | 训练钩子 | `callbacks.on_step()`, `callbacks.on_training_start()`, `callbacks.on_training_end()` |
| **模块7（监控）** | 日志/指标 | `LoggingSubsystem` 中的所有 writer/logger |
| **模块8（基础设施）** | 配置/路径 | `PathConfig`, `PPORLConfigParser` |
| **模块1（游戏环境）** | 环境工厂 | `env_factory()` 创建 `AntWarEnv` |

### 6.2 状态持久化

| 文件 | 内容 | 用途 |
|------|------|------|
| `pool.json` | 对手池状态（所有对手的 ID、checkpoint 路径、games 计数） | 训练恢复时重建对手池 |
| `payoff.json` | TrueSkill 评分矩阵（所有对手间的评分和胜率） | 训练恢复时重建评分 |
| `training_status_{run_id}.json` | 训练进度快照（奖励、损失、胜率、速度） | 训练状态监控 |

---

## 7. 配置参数

### 7.1 自对弈参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `opponent_pool_size` | 对手池最大容量 | 10-20 |
| `min_opponent_games` | 对手淘汰前最小对战场次 | 5-10 |
| `exploit_prob` | 初始利用概率 | 0.3-0.5 |
| `opponent_update_interval` | 添加新对手的间隔（episode 数） | 50-200 |
| `payoff_decay` | 历史评分衰减因子 | 0.99 |

### 7.2 评估参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `eval_interval` | 基线评估间隔 | 100-500 |
| `baseline_agents` | 评估使用的基线 AI 列表 | `["BasicTowerAI", "MediumRuleAI"]` |

### 7.3 持久化参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `save_state_interval` | 联赛状态保存间隔 | 50-100 |
| `training_status_interval` | 训练状态保存间隔 | 10-50 |
| `checkpoint_interval` | 检查点保存间隔 | 50-200 |

---

## 8. 常见问题与注意事项

### 8.1 为什么使用 TrueSkill 而非 ELO？

TrueSkill 提供不确定性估计（σ），这在自对弈中非常有价值：
- 可以用 σ 区分 "已确定" 和 "不确定" 的对手
- 对手选择可以利用 σ 进行探索/利用平衡
- 贝叶斯框架天然适合多玩家场景

### 8.2 初始对手的重要性

训练开始时，对手池中没有历史版本。系统创建 3 个随机初始化的网络作为初始对手。这些初始对手的强度随机，为策略提供了初始的、多样的学习信号。

### 8.3 对手池满时的淘汰策略

淘汰策略优先保护：
1. 最近添加的对手（保护期）：防止刚加入就被淘汰
2. 对局数不足的对手：需要足够数据才能准确评估强度
3. 强度最低的对手（`μ + 保护加分` 最小）：被淘汰

### 8.4 自适应调度的意义

训练初期应偏向探索（与各类对手对战，收集评分数据），后期偏向利用（与最强对手对战，提升策略强度）。自适应调度可根据总训练步数和当前进度动态调整 `exploit_prob`。

### 8.5 训练可恢复性

通过 `pool.json` + `payoff.json` + checkpoint 文件，训练可以在任意中断点恢复。恢复时：
1. 加载对手池状态
2. 加载评分矩阵
3. 加载最新的模型检查点
4. 从对应的 episode 编号继续训练
