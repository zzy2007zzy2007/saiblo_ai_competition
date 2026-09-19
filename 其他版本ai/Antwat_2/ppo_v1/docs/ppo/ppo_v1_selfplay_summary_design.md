# PPO v1 SelfPlay 系统概要设计

> 版本: v1.0
>
> 基于 `ppo/src/ppo_antwar` 源码分析，结合 `docs/design/antwar_rules.md`、`docs/design/antwar_rules_review.md`

---

## 1. 概述

### 1.1 什么是 SelfPlay

SelfPlay（自我对弈）是强化学习中一种重要的训练范式：智能体**通过与自身历史版本的副本（或此前保存的对手模型）进行对战**来持续学习和进步。在 AntWar 中，SelfPlay 是 PPO 训练的核心机制——每一轮训练，当前策略都会与对手池中的某个历史版本进行一局对战，收集轨迹数据用于 PPO 更新。

### 1.2 核心要解决的问题

| 问题 | SelfPlay 的解法 |
|------|----------------|
| 对手强度需要与当前策略匹配 | 对手池保存多个历史版本，选择器动态匹配 |
| 避免过拟合单一对手 | 对手池含多个不同实力的对手，explore/exploit 平衡 |
| 训练需要稳定可比的基线 | 定期用 Baseline Battle 评估绝对实力 |
| 对手实力评估需要度量 | TrueSkill 评分系统，量化 mu/sigma |

### 1.3 设计目标

1. **持续进步**：当前智能体通过与历史版本对战不断提升
2. **对手多样性**：对手池保持多个不同实力的对手，防止过拟合
3. **公平评估**：先后手交替机制确保立场公平
4. **可恢复训练**：完整的状态持久化机制，支持断点续训
5. **可观测性**：完整日志记录训练过程、对战结果、评分变化

---

## 2. 总体架构

### 2.1 核心组件关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SelfPlayTrainer                                │
│                                                                         │
│  ┌─────────────────┐     ┌─────────────────────┐     ┌──────────────┐  │
│  │   PPOTrainer    │◄───►│   SelfPlayManager   │◄───►│BattleCoordi- │  │
│  │  (策略更新引擎)  │     │  (对战协调+评分)     │     │  nator       │  │
│  │                 │     │                     │     │ (Baseline    │  │
│  │  - policy net   │     │  - select_opponent  │     │  评估)       │  │
│  │  - optimizer    │     │  - update_payoff    │     │              │  │
│  │  - _ppo_update  │     │  - get_statistics   │     │  - run_ppo_  │  │
│  │  - _select_action│    │                     │     │    evaluat-  │  │
│  └────────┬────────┘     └──────────┬──────────┘     │    ion       │  │
│           │                         │                └──────────────┘  │
│           │                         │                                  │
│           ▼                         ▼                                  │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                        League 子系统                          │      │
│  │  ┌──────────────┐    ┌────────────────┐    ┌──────────────┐  │      │
│  │  │OpponentPool  │◄──►│OpponentSelector│◄──►│BattleShared  │  │      │
│  │  │ 对手池       │    │  选择器        │    │  Payoff      │  │      │
│  │  │              │    │                │    │  评分系统    │  │      │
│  │  │ - add/evict  │    │ - exploit      │    │ - TrueSkill  │  │      │
│  │  │ - persist    │    │ - explore      │    │ - win_rate   │  │      │
│  │  └──────────────┘    └────────────────┘    └──────────────┘  │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                     日志子系统                                  │      │
│  │  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐  │      │
│  │  │  SelfPlayLogger│    │TrainingLogger  │    │SelfPlayBattle│  │      │
│  │  │  (对战记录)     │    │(训练指标)       │    │  数据缓存    │  │      │
│  │  └────────────────┘    └────────────────┘    └──────────────┘  │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心类职责

| 类名 | 模块 | 核心职责 |
|------|------|----------|
| `SelfPlayTrainer` | `trainer/selfplay.py` | SelfPlay 训练循环的编排器 |
| `SelfPlayManager` | `trainer/selfplay.py` | 对战、评分、对手管理的中央协调器 |
| `OpponentPool` | `league/opponent_selector.py` | 对手的生命周期管理（添加、淘汰、持久化） |
| `OpponentSelector` | `league/opponent_selector.py` | 对手选择策略（exploit vs explore） |
| `BattleSharedPayoff` | `league/payoff.py` | TrueSkill 评分 + 对战记录 |
| `SelfPlayLogger` | `trainer/selfplay_logger.py` | SelfPlay 对战日志记录 |

---

## 3. 训练循环

### 3.1 单 Episode 流程

```
对手选择
    │
    ├─ exploit 模式（默认概率 0.3→0.9，随训练进度递增）
    │    └─ 选择 TrueSkill mu 最高的 confident 对手
    │
    └─ explore 模式（默认概率 0.7→0.1，随训练进度递减）
         └─ 随机选择低置信度对手
    │
    ▼
先后手判定: swap_positions = (battle_count % 2 == 1)
    │
    ▼
单局对战数据收集 (_collect_episode_with_swap)
    │
    ├─ step 循环 (最大 512 steps = 256 个完整回合)
    │    ├─ 获取当前智能体动作 (action, log_prob, value)
    │    ├─ 获取对手动作 (agent.act 或 随机合法动作)
    │    ├─ 执行 env._resolve_turn_mixed()
    │    ├─ 收集 reward / obs / action_mask
    │    └─ 记录到 EpisodeBatch
    │
    ├─ 结果判定: 基于 final HP
    │    (win: 敌方HP≤0 && 己方HP>0)
    │    (loss: 己方HP≤0 && 敌方HP>0)
    │    (draw: 其他)
    │
    └─ 返回: (EpisodeBatch, battle_details)
    │
    ▼
更新评分 (SelfPlayManager.update_payoff)
    │
    ├─ TrueSkill rate_1vs1() 更新双方 mu/sigma
    ├─ 更新正向/反向战绩记录
    └─ 更新 opponents 对战场次计数
    │
    ▼
PPO 更新 (_ppo_update)
    │
    ├─ GAE 计算 advantages / returns
    ├─ 多 epoch 小批量 SGD
    ├─ Policy Loss (clipped surrogate)
    ├─ Value Loss (clipped MSE)
    ├─ Entropy Bonus (退火)
    └─ 梯度裁剪
```

### 3.2 先后手交替策略

先后手对 AntWar 的影响显著（先手有第一轮建造优势）。SelfPlay 通过 **偶数局 Agent 先手、奇数局 Agent 后手** 的交替策略来抵消除此影响：

```python
swap_positions = (battle_count % 2 == 1)

# swap_positions = False: Agent 执玩家 0（先手）
# swap_positions = True : Agent 执玩家 1（后手）
```

在 `_collect_episode_with_swap` 中：
- `player = 1 if swap_positions else 0`
- 观察从对应的 `obs["player_0"]` 或 `obs["player_1"]` 提取
- 奖励从对应的 `rewards["player_0"]` 或 `rewards["player_1"]` 提取

### 3.3 训练配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `total_episodes` | 150000 | 总训练轮数 |
| `battle_interval` | 1000 | Baseline Battle 评估间隔（episode） |
| `opponent_update_interval` | 500 | 将当前策略保存为对手的间隔（episode） |
| `max_steps_per_episode` | 512 | 每局最大 step 数 |
| `num_envs` | 1 | 环境并行数 |

---

## 4. SelfPlayManager — 中央协调器

### 4.1 职责边界

SelfPlayManager 是 SelfPlay 系统的**协调中心**，但不直接参与 PPO 更新和环境交互。它的职责包括：

```
SelfPlayManager
    │
    ├── update_progress(episode)
    │       └─ 将当前训练进度通知 OpponentSelector（用于自适应 exploit_prob）
    │
    ├── select_opponent()
    │       └─ 委托给 OpponentSelector + OpponentPool
    │
    ├── add_opponent(checkpoint_path)
    │       └─ 向对手池注册新对手
    │
    ├── update_payoff(opponent_id, result, rounds)
    │       └─ 委托给 BattleSharedPayoff 更新评分 + 记录
    │
    └── get_statistics()
            └─ 聚合对手池 + 评分 + 对战统计信息
```

### 4.2 依赖关系

```
SelfPlayManager
    ├── agent: PPOAgent            (对当前策略的引用，不直接使用)
    ├── opponent_pool: OpponentPool
    ├── payoff: BattleSharedPayoff
    └── selector: OpponentSelector
```

---

## 5. League 子系统

### 5.1 对手生命周期

```
       对手生命周期
       ────────────────────────
       创建:
           ├─ 预热阶段：创建 N 个随机初始化策略作为初始对手
           └─ 训练中：每 opponent_update_interval episode
                        将当前模型保存为 checkpoint → 添加至对手池
       生存:
           ├─ 被选择器选中 → 对战 → 更新评分
           └─ 积累对战场次
       淘汰:
           └─ 对手池满时，触发淘汰策略
               ├─ 优先保护：对战场次 < min_games_threshold 的对手免于淘汰
               └─ 评分判决：选择 mu 最低的对手淘汰
```

### 5.2 评分系统 (TrueSkill)

TrueSkill 为每个对手维护两个核心参数：

| 参数 | 含义 | 初始值 | 作用 |
|------|------|--------|------|
| **mu** | 估计实力 | 25.0 | 越强越高，用于 exploit 选择 |
| **sigma** | 不确定度 | 8.33 | 对战越多越低，sigma < 5.0 视为 confident |

**评分更新规则：**

```
对战结果 → TrueSkill rate_1vs1()
    ├─ win  : winner.mu ↑, loser.mu ↓ (双方 sigma ↓)
    ├─ loss : loser.mu ↓, winner.mu ↑ (双方 sigma ↓)
    └─ draw : 双方 mu 小幅变化 (双方 sigma ↓)
```

**可利用性 (exploitability)：**

量化当前智能体相对于对手池的实力差距：
- **1.0**：智能体远弱于对手池平均值（需要快速学习）
- **0.5**：与对手池平均水平相当
- **0.0**：远强于对手池（需要更强的对手）

### 5.3 对手选择器

采用 **exploit vs explore** 二分选择策略：

```
选择流程：

    exploit_prob = f(progress)  // 自适应计算
        │
        ├─ 0.3 → 0.9 (线性增长)
        │
        ├─ exploit (随机数 < exploit_prob)
        │    └─ 筛选 sigma < 5.0 的 confident 对手
        │         └─ 取 mu 最高的前 3 名 → 随机选 1
        │
        └─ explore (随机数 >= exploit_prob)
             └─ 随机选择低置信度对手
```

**自适应策略比较：**

| 策略 | 公式 | 特点 |
|------|------|------|
| linear | `0.3 + 0.6 * min(1.0, episode/10000)` | 线性增长，平稳过渡 |
| sigmoid | `0.3 + 0.6 * sigmoid(10*(p-0.5))` | S 形曲线，中期加速 |
| exp | `0.3 + 0.6 * (1 - exp(-5*p))` | 指数增长，前期快速上升 |

### 5.4 对手池管理

**数据结构：**

```
OpponentPool
    ├── max_size: 10                    // 最大容量
    ├── min_games_threshold: 8          // 淘汰保护阈值
    ├── _opponents: List[str]           // 对手 ID 列表（有序）
    ├── _checkpoint_paths: Dict         // 对手 ID → checkpoint 路径
    ├── _games_played: Dict             // 对手 ID → 对战场次
    └── _payoff: BattleSharedPayoff     // 用于淘汰决策
```

**淘汰逻辑：**

```
当 len(_opponents) > max_size 时触发:
    ├─ 筛选 candidates: games_played >= min_games_threshold
    ├─ 无 candidates → 淘汰列表第一个
    └─ 有 candidates → 计算淘汰分数
         ├─ eviction_score = mu - protection_bonus
         ├─ protection_bonus = (min_games_threshold - games) * 0.5 (仅对 games < threshold)
         └─ 选择 score 最低的淘汰
```

### 5.5 状态持久化

```
checkpoint_dir/league_state/
    ├── opponent_pool.json    // 对手池状态 (对手ID列表 + checkpoint路径 + 对战场次)
    └── payoff.json           // 评分状态 (TrueSkill ratings + 对战记录)
```

**保存时机：**
- 每 `opponent_update_interval` 添加新对手后
- 持久化文件位于 checkpoint 目录下的 `league_state/` 子目录

**加载时机：**
- `SelfPlayTrainer.__init__` 初始化时自动加载
- 加载成功后跳过预热创建初始对手的步骤

---

## 6. 预热机制

### 6.1 启动流程

```
SelfPlayTrainer.__init__()
    │
    ├─ 初始化组件 (OpponentPool, Payoff, SelfPlayManager...)
    │
    ├─ _load_league_state()
    │    ├─ 加载 opponent_pool.json
    │    ├─ 加载 payoff.json
    │    └─ 成功 → 恢复历史状态
    │
    ├─ 对手池为空？
    │    ├─ 是 → _create_initial_opponents()
    │    │         └─ 创建 3 个随机初始化策略的 checkpoint
    │    │              (AntWarPolicyValueNetwork, 随机权重)
    │    │
    │    └─ 否 → 跳过预热（恢复训练）
    │
    └─ 初始化 BattleCoordinator
```

### 6.2 初始对手创建

```python
def _create_random_agent_checkpoint(self):
    policy = AntWarPolicyValueNetwork(
        board_shape=(28, 19, 19),   # 28 channel × 19 × 19
        global_dim=33,               # 33 维全局特征
        action_dim=96,               # 96 维动作空间
        hidden_dim=256,              # 隐藏层维度
    )
    # 使用随机初始化权重（未训练）
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'config': {'network': ...}
    }
    torch.save(checkpoint, path)
```

核心要点：
- 初始对手使用**完全随机权重**，提供最基础的对抗
- 每个初始对手保存为独立 `.pt` 文件
- 对手池从 3 个随机对手开始，随着训练逐步替换为真实训练版本

---

## 7. 单局对战数据收集

### 7.1 数据收集流程

`_collect_episode_with_swap` 是 SelfPlay 训练的核心数据收集方法：

```
EpisodeBatch 数据结构:
    ├── observations_board: List[np.ndarray(28, 19, 19)]   // Board 特征
    ├── observations_global: List[np.ndarray(33,)]          // Global 特征
    ├── observations_mask: List[np.ndarray(96,)]            // Action mask
    ├── actions: List[int]                                  // 动作 ID
    ├── rewards: List[float]                                // 奖励值
    ├── values: List[float]                                 // 价值估计
    ├── log_probs: List[float]                              // 动作对数概率
    └── dones: List[float]                                  // 终止标记
```

### 7.2 Step 执行逻辑

```python
# 每步执行 env._resolve_turn_mixed()
# 该调用在一个 API 调用中完成双方的操作解析和回合推进
obs, rewards, terminated, truncated, info = env._resolve_turn_mixed(
    player_0_action_id=...,     # 玩家 0 的动作 ID
    player_0_ops=...,           # 玩家 0 的内部操作列表
    player_1_action_id=...,     # 玩家 1 的动作 ID
    player_1_ops=...,           # 玩家 1 的内部操作列表
)
```

**先后手对调时，agent 的 `action` 填入对应玩家的参数位置：**

```python
if not swap_positions:  # Agent 执先手 (player_0)
    obs, rewards, ... = env._resolve_turn_mixed(
        player_0_action_id=agent_action,
        player_1_ops=opponent_ops,
        ...
    )
    reward = rewards["player_0"]
else:                   # Agent 执后手 (player_1)
    obs, rewards, ... = env._resolve_turn_mixed(
        player_0_ops=opponent_ops,
        player_1_action_id=agent_action,
        ...
    )
    reward = rewards["player_1"]
```

### 7.3 对手动作获取

`_get_opponent_move` 支持多种对手类型：

| 对手类型 | 获取方式 | 场景 |
|----------|----------|------|
| PPOAgent (有 `act` 方法) | `opponent_agent.act(obs)` | SelfPlay 对手池中的模型 |
| Baseline AI (有 `choose_operations`) | `agent.choose_operations(state, player)` | Baseline 评估 |
| Baseline AI (有 `choose_bundle`) | `agent.choose_bundle(state, player).operations` | Baseline 评估 |
| None | `_random_action_from_mask()` | 随机对手（降级方案） |

### 7.4 结果判定

```python
# 基于最终 HP 判定
if enemy_hp <= 0 and our_hp > 0:
    result = 'win'
elif our_hp <= 0 and enemy_hp > 0:
    result = 'loss'
else:
    result = 'draw'
```

---

## 8. 日志与监控

### 8.1 SelfPlayLogger

SelfPlayLogger 记录每局对战的关键信息：

```python
# 对战开始
log_battle_start(episode, opponent_id, player_position)
# → "START #1234: Agent(先手) vs opp_abc12345"

# 回合状态（DEBUG 级别）
log_round(episode, round_num, our_hp, enemy_hp, our_coins, enemy_coins)
# → "#1234-50: HP=35/20, Coins=120/80"

# 对战结束
log_battle_end(episode, result, total_rounds, duration, final_hp)
# → "END #1234: Result: win, Rounds: 256, Time: 15.23s"
```

### 8.2 SelfPlay 对战数据缓存

为了减少 IO 频率，SelfPlay 使用内存缓存 + 批量写入策略：

```python
selfplay_battle_data_cache: Dict[str, Dict]   # 对手级别的统计汇总
selfplay_battle_details_cache: List[Dict]      # 详细对战记录列表
```

**缓存刷新时机：**
- `_flush_selfplay_battle_cache()` 每 episode 结束后调用
- 写入文件：`outputs/selfplay/selfplay_battles/`
  - `selfplay_battle_stats.json`：统计汇总
  - `detailed_battles/selfplay_{episode}_{opponent}_{timestamp}.json`：详细记录

### 8.3 训练状态保存

`_save_training_status(episode, battle_result)` 每 `training_status_save_interval=100` episode 保存：

| 指标 | 计算方法 |
|------|----------|
| 近 100/10 轮平均奖励 | `np.mean(recent_100 avg_reward)` / `np.mean(recent_10 avg_reward)` |
| 近 100 轮平均损失 | `np.mean(recent_100 avg_loss)` |
| 平均回合数 | `np.mean(episode_lengths[-100:])` |
| 胜率 | `wins / (wins + losses + draws)` |
| 训练速度 | `episodes / (training_time_minutes)` |
| 稳定性 | `reward_std`, `loss_std` |

---

## 9. Baseline Battle 评估

### 9.1 评估目的

SelfPlay 提供的是**相对进步**（基于对手池的胜率变化），Baseline Battle 提供的是**绝对进步**（基于固定基线 AI 的胜率变化）。

### 9.2 触发机制

```
每 battle_interval (默认 1000) episode:
    ├─ BattleCoordinator.run_ppo_evaluation_with_agent()
    │    ├─ 对每个 baseline_agent (默认 ['BasicTowerAI'])
    │    ├─ 先手 + 后手各 n_battles/2 局
    │    └─ 子进程隔离执行
    │
    └─ 结果聚合 → 日志记录
```

### 9.3 与 SelfPlay 的关键区别

| 维度 | SelfPlay 对战 | Baseline Battle |
|------|---------------|-----------------|
| 目的 | 收集训练数据 | 评估绝对实力 |
| 对手 | 对手池中的历史版本 | 固定的 Baseline AI 策略 |
| 环境 | env_factory 创建的完整环境 | 直接操作 PythonBackendState |
| 执行方式 | 主进程内执行 | 子进程隔离执行 |
| 先后手交替 | battle_count 偶数/奇数 | episode 内的先后手 |
| 结果用途 | PPO 更新 + 评分更新 | 监控 + 日志 |

---

## 10. 关键设计决策

### 10.1 为什么用 OpponentPool 而非直接从文件加载

**决策**：引入 OpponentPool 抽象层，管理对手的添加、淘汰、查询。

**原因**：
1. 对手数量有上限（10 个），需要淘汰机制
2. 淘汰需要综合考虑评分 + 对战场次（保护新入池对手）
3. 对手池需要持久化/恢复，抽象层封装了状态管理

### 10.2 为什么使用 TrueSkill 而非 Elo

**决策**：使用 TrueSkill 评分系统。

**原因**：
1. TrueSkill 提供 mu(实力) + sigma(不确定度) 双参数
2. sigma 可直接用于对手选择的置信度判断
3. 无需全局统一的 rating 基准（重点是比较而非绝对排名）
4. 对战场次较少时也能给出有意义的评分

### 10.3 为什么先后手交替基于 battle_count 而非 episode

**决策**：使用 `battle_count % 2 == 1` 判断先后手。

**原因**：
1. 训练过程中可能会中断/恢复，battle_count 从持久化状态恢复
2. 保证每次恢复后，先后手仍然交替
3. 与 episode 解耦，更健壮

### 10.4 为什么对手选择基于 episode 而非 step

**决策**：对手选择器使用 `_total_episodes` 作为自适应 exploit_prob 的输入。

**原因**：
1. 每个 episode 产生一个完整的对战轨迹
2. step 的数量波动大（早 50steps vs 晚 512steps），不适合作为进度指标
3. episode 数量直接对应 PPO 更新次数，与学习进度更相关

### 10.5 为什么使用缓存写入而非实时写入

**决策**：SelfPlay 对战数据使用内存缓存 + 批量写入。

**原因**：
1. SelfPlay 每 episode 都产生对战数据（vs Baseline Battle 每 1000 episode）
2. 实时写入磁盘会显著拖慢训练速度（IO 瓶颈）
3. episode 间数据独立，缓存写入不会丢失关键信息

---

## 11. 状态管理

### 11.1 需要持久化的状态

| 状态 | 存储位置 | 存储格式 | 恢复方式 |
|------|----------|----------|----------|
| 对手池 | `league_state/opponent_pool.json` | JSON | `OpponentPool.load_state()` |
| 评分 | `league_state/payoff.json` | JSON | `BattleSharedPayoff.load_state()` |
| 模型权重 | `checkpoint/model_{episode}.pt` | torch.save | `torch.load()` |
| 训练指标 | `training/training_metrics.json` | JSON | 重新加载 |

### 11.2 状态恢复流程

```
训练启动
    │
    ├─ 加载 league_state/payoff.json → BattleSharedPayoff
    ├─ 加载 league_state/opponent_pool.json → OpponentPool
    │
    ├─ payoff 和 opponent_pool 均恢复成功？
    │    ├─ 是 → 跳过预热，继续训练
    │    └─ 否 → 创建初始随机对手
    │
    ├─ 加载最新的 checkpoint → 恢复 policy weights
    │
    └─ 重置环境 → 开始训练
```

### 11.3 数据流汇总

```
训练阶段数据流：
    Agent → env交互 → EpisodeBatch → PPOUpdate → Policy更新
                                                        │
                                                        ▼
                                              checkpoint保存 → 对手池添加新对手
                                                                     │
                                                                     ▼
                                                           League状态持久化

评估阶段数据流：
    Policy → BattleCoordinator → 子进程对战 → 结果聚合 → 日志记录
```

---

## 12. 配置速查

### 12.1 SelfPlay 配置

```yaml
selfplay:
  opponent_pool_size: 10      # 对手池最大容量
  min_opponent_games: 8       # 淘汰保护所需最小对战场次
  exploit_prob: 0.7           # 初始利用概率
  explore_prob: 0.3           # 初始探索概率

league:
  decay: 0.99                 # 历史记录衰减因子
  min_win_rate_games: 8       # 胜率计算最小场次

training:
  opponent_update_interval: 500  # 对手更新间隔 (episode)
  battle_interval: 1000          # Baseline 评估间隔 (episode)
  n_battles: 5                   # Baseline 评估对局数
  total_episodes: 150000         # 总训练轮数
  initial_opponents: 3          # 初始随机对手数量
```

### 12.2 PPO 训练配置

```yaml
ppo:
  lr: 0.0001
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  ent_coef: 0.2
  ent_coef_final: 0.05
  vf_coef: 0.05
  max_grad_norm: 2.0
  ppo_epochs: 4
  batch_size: 64
```

---

## 13. 总结

### 13.1 系统架构要点

1. **SelfPlayTrainer** 是整个 SelfPlay 系统的编排器，协调训练循环、对手管理、评估和日志
2. **SelfPlayManager** 作为中央协调器，封装了对手选择、评分更新和统计分析
3. **League 子系统**（OpponentPool + OpponentSelector + BattleSharedPayoff）实现了对手的全生命周期管理
4. **先后手交替**通过 `battle_count % 2` 实现，确保公平性
5. **TrueSkill 评分**提供 mu + sigma 双维度的实力评估
6. **Exploit / Explore 自适应**根据训练进度动态调整选择策略

### 13.2 设计原则

| 原则 | 体现 |
|------|------|
| **模块化** | SelfPlayManager、League 子系统、日志子系统职责清晰分离 |
| **可恢复** | 完整的 League 状态持久化 + 模型 checkpoint + 训练指标记录 |
| **公平性** | 先后手交替、对手池多样性、exploit/explore 平衡 |
| **可观测** | 多级别日志、对战详情缓存、训练状态定期保存 |
| **鲁棒性** | 对手加载失败降级为随机策略、子进程失败降级为顺序执行 |
