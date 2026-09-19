# 联赛管理模块 (League Management)

## 1. 背景与定位

### 1.1 业务问题

在自对弈训练中，智能体需要与各种水平的对手对战才能持续进步。如果只与当前策略自身对战（纯粹的自对弈），模型可能陷入"自我强化"的循环——只擅长对付特定风格的对手。如果始终与最强的对手对战，模型可能被压制而无法学习。更关键的是，我们需要一个**评分系统**来衡量每个对手的实力，以及一个**选择策略**来决定每局该与谁对战。

**联赛管理模块**的核心业务目标是：**维护一个多样化的对手池，通过合理的评分系统和选择策略，为自对弈训练提供"最佳对手"**，使得训练过程可以持续、高效地进步。

### 1.2 在整个系统中的位置

```
自对弈编排 (Self-Play Orchestration)
        │
        │  选择对手、更新评分、管理对手池
        ▼
┌─────────────────────────────────────┐
│        联赛管理模块                   │
│                                     │
│  SelfPlayManager (统一入口)           │
│    ├─ OpponentSelector (选择策略)    │
│    │   ├─ 利用 (Exploit) / 探索     │
│    │   └─ 自适应调度                │
│    ├─ BattleSharedPayoff (评分矩阵)  │
│    │   ├─ TrueSkill 评分            │
│    │   ├─ 胜率计算（贝叶斯收缩）     │
│    │   └─ 可利用性评估              │
│    └─ OpponentPool (对手池)         │
│        ├─ 添加/淘汰                 │
│        ├─ 保护期机制                │
│        └─ 状态持久化                │
└─────────────────────────────────────┘
        │
        │  无需依赖其他业务模块
        ▼
   (纯内部逻辑，仅依赖基础设施)
```

- **调用者**：自对弈编排模块（`SelfPlayTrainer`）在每局训练前调用 `select_opponent()`，训练后调用 `update_payoff()`
- **独立性**：联赛管理模块是唯一不依赖其他业务模块的纯独立模块（仅依赖基础设施的常量）
- **可复用性**：联赛逻辑设计为与具体算法无关，可被任何需要对手选择机制的训练系统使用

### 1.3 与其他模块的关系

| 外部模块 | 关系 | 交互方式 |
|---------|------|---------|
| 基础设施（无具体依赖） | — | 仅使用 Python 标准库和 `trueskill` 第三方包 |
| 自对弈编排 (`selfplay.py`) | 被依赖 | 创建 `SelfPlayManager`，调用 `select_opponent()` / `update_payoff()` / `add_opponent()` / `save_state()` |

---

## 2. 核心概念

### 2.1 关键术语

| 术语 | 说明 |
|------|------|
| **TrueSkill** | Microsoft 开发的评分系统，通过 `μ` (mu, 技能均值) 和 `σ` (sigma, 不确定性) 两个参数衡量玩家水平 |
| **μ (Mu)** | 玩家技能水平的估计值，越高代表越强 |
| **σ (Sigma)** | 技能估计的不确定性，越低代表评估越确信 |
| **利用 (Exploitation)** | 选择当前最强的对手对战，挑战自己的极限 |
| **探索 (Exploration)** | 选择不确定性高的对手对战，获取更多关于对手池的信息 |
| **可利用性 (Exploitability)** | 当前策略相对对手池的劣势程度，用于衡量策略弱点 |
| **贝叶斯胜率** | 使用贝叶斯方法估计的胜率，对战场次少时会向 50% 收缩 |
| **保护期** | 新加入对手的评分保护机制，防止被立即淘汰 |

### 2.2 对手选择流程

```
每局训练前：
1. SelfPlayManager.select_opponent(episode_progress)
   │
   ├─ 从 OpponentPool 获取所有候选对手
   │
   ├─ OpponentSelector._compute_exploit_prob(progress) → exploit_prob
   │   (根据训练进度动态计算利用概率)
   │
   ├─ if random() < exploit_prob → 利用策略
   │   └─ 从 sigma < 3.0 的可信对手中，选 μ 最高的 EXPLOIT_TOP_K 个之一
   │
   └─ else → 探索策略
       └─ 从 sigma 较高的不确定性对手中随机选取
```

---

## 3. 架构总览

### 3.1 模块组成

```
league-management/
├── opponent_selector.py    # SelfPlayManager + OpponentSelector
└── payoff.py               # BattleSharedPayoff + OpponentPool + TrueSkillRating
```

### 3.2 组件关系

```
SelfPlayManager (对外统一入口)
│
├── OpponentSelector
│   ├── _exploit_select()   # 利用策略：选最强对手
│   ├── _explore_select()   # 探索策略：选不确定对手
│   └── _compute_exploit_prob()  # 自适应调度
│
├── BattleSharedPayoff
│   ├── update()            # 更新 TrueSkill 评分
│   ├── get_win_rate()      # 贝叶斯胜率
│   ├── get_exploitability() # 可利用性
│   ├── get_battle_stats()  # 评分统计
│   └── save_state/load_state
│
└── OpponentPool
    ├── add_opponent()      # 添加对手（含保护期）
    ├── _evict_worst()      # 淘汰最差对手
    ├── save_state/load_state
    └── get_stats()
```

---

## 4. 核心组件详解

### 4.1 SelfPlayManager (`league/opponent_selector.py`)

#### 4.1.1 职责

自对弈管理器的统一入口，整合对手选择、评分更新、对手池管理。上层训练循环只需与这一个类交互。

#### 4.1.2 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `select_opponent` | `(episode_progress: float) -> (str, OpponentAgent)` | 根据进度选择对手，返回 (opponent_id, agent) |
| `add_opponent` | `(episode: int, checkpoint_path: str) -> str` | 添加新对手到池，满时触发淘汰 |
| `update_payoff` | `(opponent_id: str, result: str)` | 更新与对手的对战评分 |
| `update_progress` | `(progress: float)` | 更新训练进度，传递给选择器 |
| `get_exploitability` | `() -> float` | 返回当前策略的可利用性评分 |
| `save_state` | `(save_dir: str)` | 保存联赛状态（评分矩阵 + 对手池）到 JSON |
| `load_state` | `(load_dir: str)` | 从 JSON 恢复联赛状态 |
| `get_opponent_stats` | `() -> List[Dict]` | 获取所有对手的统计信息 |

#### 4.1.3 构造函数

```python
def __init__(self, policy_network, device, checkpoint_dir, opponent_checkpoint_dir,
             max_opponents=20, opponent_save_interval=100, ...)
```

初始化时创建 `BattleSharedPayoff`、`OpponentPool` 和 `OpponentSelector` 实例。

### 4.2 OpponentSelector (`league/opponent_selector.py`)

#### 4.2.1 职责

实现探索/利用平衡的对手选择策略。核心思想是：训练早期多探索（与不同对手对战），训练后期多利用（与最强对手对战）。

#### 4.2.2 公开方法

| 方法 | 功能 |
|------|------|
| `select_opponent(payoff_matrix, opponent_pool, progress)` | 选择对手，返回 opponent_id |
| `_exploit_select(payoff_matrix, opponent_pool)` | 利用策略 |

#### 4.2.3 利用策略 (`_exploit_select`)

```python
def _exploit_select(self, payoff, pool):
    # 1. 筛选可信对手 (sigma < 3.0)
    confident = [id for id in pool.get_all()
                 if payoff.get_rating(id).sigma < 3.0]

    if not confident:
        # 没有可信对手 → 从所有对手中随机选
        return random.choice(pool.get_all())

    # 2. 按 μ 降序排列，取前 K 个
    confident.sort(key=lambda id: payoff.get_rating(id).mu, reverse=True)
    top_k = confident[:self.EXPLOIT_TOP_K]  # EXPLOIT_TOP_K = 3

    # 3. 从前 K 强中随机选一个
    return random.choice(top_k)
```

#### 4.2.4 探索策略 (`_explore_select`)

```python
def _explore_select(self, payoff, pool):
    # 从 sigma 较大（不确定性高）的对手中随机选
    candidates = [id for id in pool.get_all()
                  if payoff.get_rating(id).sigma >= EXPLORE_SIGMA_THRESHOLD]
    if not candidates:
        candidates = pool.get_all()
    return random.choice(candidates)
```

#### 4.2.5 自适应调度 (`_compute_exploit_prob`)

利用概率随训练进度动态调整，支持三种增长模式：

```python
def _compute_exploit_prob(self, progress):
    # progress ∈ [0, 1]
    # 初始概率 = 0.3，最大概率 = 0.9，增长范围 = 0.6

    if self.mode == 'linear':
        return min_prob + growth_range * progress
    elif self.mode == 'sigmoid':
        return min_prob + growth_range / (1 + exp(-10 * (progress - 0.5)))
    elif self.mode == 'exp':
        return min_prob + growth_range * (1 - exp(-5 * progress))
```

参数默认值：`min_prob=0.3`，`growth_range=0.6`，`max_prob=0.9`。

### 4.3 BattleSharedPayoff (`league/payoff.py`)

#### 4.3.1 职责

管理所有玩家间的对战记录和 TrueSkill 评分矩阵。这是联赛系统的数据核心。

#### 4.3.2 公开方法

| 方法 | 功能 |
|------|------|
| `update(home, away, result)` | 更新对战结果（胜/负/平），使用 `trueskill.rate_1vs1` 更新双方评分 |
| `get_win_rate(home, away) -> float` | 计算贝叶斯胜率（带信度加权收缩） |
| `get_exploitability(player_id) -> float` | 计算玩家的可利用性 |
| `decay_all(decay_factor)` | 对所有评分施加衰减（旧记录影响递减） |
| `get_rating(player_id) -> TrueSkillRating` | 获取玩家的当前评分 |
| `get_payoff_matrix() -> Dict` | 生成完整评分矩阵 |
| `get_battle_stats() -> List[Dict]` | 获取所有玩家评分统计 |
| `save_state() / load_state()` | JSON 序列化/反序列化 |

#### 4.3.3 TrueSkill 评分更新

```python
def update(self, home, away, result):
    home_rating = trueskill.Rating(mu=self._ratings[home].mu,
                                    sigma=self._ratings[home].sigma)
    away_rating = trueskill.Rating(mu=self._ratings[away].mu,
                                    sigma=self._ratings[away].sigma)

    if result == 'win':       # home 胜
        new_home, new_away = trueskill.rate_1vs1(home_rating, away_rating)
    elif result == 'lose':    # home 负
        new_away, new_home = trueskill.rate_1vs1(away_rating, home_rating)
    else:                     # 平局 (draw)
        new_home, new_away = trueskill.rate_1vs1(home_rating, away_rating, drawn=True)

    self._ratings[home] = TrueSkillRating(new_home.mu, new_home.sigma)
    self._ratings[away] = TrueSkillRating(new_away.mu, new_away.sigma)
    self._battle_counts[(home, away)] += 1
    self._battle_counts[(away, home)] += 1
```

#### 4.3.4 贝叶斯胜率计算

```python
PSEUDO_COUNT = 1    # 伪计数（向 0.5 收缩）
DRAW_WEIGHT = 0.5   # 平局权重

def get_win_rate(self, home, away):
    battles = self._battle_counts.get((home, away), 0)
    if battles == 0:
        return 0.5  # 无对战记录时返回 0.5

    home_wins = self._results[(home, away)]['wins']
    away_wins = self._results[(home, away)]['losses']
    draws = self._results[(home, away)]['draws']

    # 贝叶斯胜率：向 0.5 收缩
    win_rate = (home_wins + DRAW_WEIGHT * draws + PSEUDO_COUNT * 0.5) \
               / (battles + PSEUDO_COUNT)
    return win_rate
```

**信度收缩**：对战场次少时，胜率会向 50% 收缩（因为伪计数 `1 * 0.5 = 0.5` 场的影响更大），避免小样本的虚假高胜率。

### 4.4 OpponentPool (`league/payoff.py`)

#### 4.4.1 职责

管理对手的添加、淘汰和状态持久化。确保对手池保持稳定的规模和多样性。

#### 4.4.2 公开方法

| 方法 | 功能 |
|------|------|
| `add_opponent(opponent_id, checkpoint_path, episode)` | 添加对手。池满时触发 `_evict_worst()` |
| `remove_opponent(opponent_id)` | 移除指定对手 |
| `get_all() -> List[str]` | 获取所有对手 ID |
| `get_opponent(opponent_id) -> OpponentInfo` | 获取对手信息 |
| `save_state() / load_state()` | JSON 持久化 |

#### 4.4.3 淘汰策略 (`_evict_worst`)

```python
def _evict_worst(self, new_opponent_id):
    # 计算每个对手的评分 = mu + 保护加成
    scores = {}
    for opp_id in self._opponents:
        mu = self._payoff.get_rating(opp_id).mu
        games = self._opponents[opp_id].games
        # 保护加成：新对手有保护期，不会被立即淘汰
        bonus = max(0, self._min_games_threshold - games) * 0.5
        scores[opp_id] = mu + bonus

    # 淘汰评分最低的对手
    worst_id = min(scores, key=scores.get)
    self.remove_opponent(worst_id)
    return worst_id
```

**保护期机制**：`bonus = max(0, min_games_threshold - games) * 0.5`，新加入的对手在对战场次未达到阈值前，评分会获得加成，防止被立即淘汰。

### 4.5 TrueSkillRating (`league/payoff.py`)

#### 4.5.1 职责

封装单个玩家的 TrueSkill 评分数据类，提供评分相关的便捷属性。

#### 4.5.2 核心属性

| 属性 | 计算方式 | 说明 |
|------|---------|------|
| `mu` | 构造函数传入 | 技能均值 |
| `sigma` | 构造函数传入 | 技能不确定性 |
| `is_confident` | `sigma < 3.0` | 是否为可信评分 |
| `confidence_interval` | `mu - 3*sigma, mu + 3*sigma` | 99.7% 置信区间 |

---

## 5. 外部交互

### 5.1 本模块依赖的外部模块

| 依赖 | 依赖内容 | 依赖方式 |
|------|---------|---------|
| 第三方库 `trueskill` | `trueskill.Rating`, `trueskill.rate_1vs1` | import |
| Python 标准库 | `json`, `random`, `math`, `dataclasses` | import |

本模块**不依赖** `ppo_antwar` 内部的任何其他业务模块，是最独立的模块。

### 5.2 依赖本模块的外部模块

| 模块 | 使用内容 | 使用方式 |
|------|---------|---------|
| 自对弈编排 (`selfplay.py`) | `SelfPlayManager` | 创建实例，在训练循环中每局调用 |
| 对战执行 (`agent_loader.py`) | `OpponentAgent` | `from_checkpoint()` 加载对手网络 |

### 5.3 典型交互场景

#### 场景 1：完整的对手选择-对战-评分周期

1. 自对弈编排调用 `manager.select_opponent(progress=0.5)`
2. `OpponentSelector` 根据进度计算利用概率（假设 60% 利用 / 40% 探索）
3. 利用策略：从可信对手中选前 3 强的随机一个
4. 返回 `opponent_id` 和对应的 `OpponentAgent`（已从检查点加载网络）
5. 自对弈编排与对手完成一局对战
6. 调用 `manager.update_payoff(opponent_id, result='win')`
7. `BattleSharedPayoff.update()` 使用 TrueSkill 更新双方评分
8. 如果评分变化超出预期，记录日志

#### 场景 2：对手池管理

1. 训练进行到第 1000 局，触发周期性对手快照保存
2. 自对弈编排调用 `manager.add_opponent(episode=1000, checkpoint_path='...')`
3. `OpponentPool.add_opponent()` 检查池容量
4. 如果池已满（如 `max_opponents=20`），触发 `_evict_worst()`
5. 计算每个对手的评分（mu + 保护加成）
6. 淘汰评分最低的对手，删除其检查点文件
7. 新对手加入池中
8. 记录 `log_opponent_added` 和 `log_opponent_removed` 事件

#### 场景 3：训练中断后恢复联赛状态

1. 训练脚本重启，指定 `load_path` 参数
2. `SelfPlayManager.load_state(load_dir)` 被调用
3. 从 JSON 文件恢复 `BattleSharedPayoff` 的评分矩阵
4. 从 JSON 文件恢复 `OpponentPool` 的对手列表
5. 对所有已注册的对手，验证检查点文件是否还存在
6. 训练从上次的进度继续，联赛状态无缝恢复

---

## 6. 关键设计决策

### 6.1 选择 TrueSkill 而非 Elo

| 指标 | TrueSkill | Elo |
|------|-----------|-----|
| 不确定性表示 | σ (sigma) 显式建模 | 无 |
| 对战场次自适应 | σ 随对战场次减少 | 无 |
| 评分收敛速度 | 快（σ 大时更新幅度大） | 慢（固定 K 因子） |
| 平局处理 | 原生支持 | 需扩展 |

TrueSkill 的 σ 参数对本项目的"探索/利用"策略至关重要——我们利用 σ 大（不确定性高）作为"值得探索"的信号。

### 6.2 贝叶斯胜率收缩

`get_win_rate()` 使用伪计数（`PSEUDO_COUNT=1`）进行贝叶斯收缩：

- **对战场次少**：胜率向 50% 收缩，避免小样本偏差
- **对战场次多**：伪计数的影响可以忽略，胜率趋近真实值

这种设计确保了对手选择不会因为偶然的小样本高胜率而做出错误决策。

### 6.3 探索/利用的自适应调度

利用概率不是固定值，而是随训练进度动态调整：
- **训练早期**（progress ≈ 0）：利用概率 ≈ 30%，以探索为主，积累对手信息
- **训练中期**（progress ≈ 0.5）：利用概率 ≈ 60%，探索和利用平衡
- **训练后期**（progress ≈ 1）：利用概率 ≈ 90%，以利用为主，专注挑战最强对手

支持 linear/sigmoid/exp 三种增长曲线，可通过配置选择。

### 6.4 对手池的"保护期"机制

新加入的对手不会被立即淘汰，即使其初始评分很低。保护期通过评分加成实现：

- `bonus = max(0, min_games_threshold - games) * 0.5`
- 新对手 `games=0` 时获得最大加成 `min_games_threshold * 0.5`
- 随着对战场次增加，加成线性减少到 0

这保证了：
- 新对手有足够机会证明自己
- 对手池不会因为单次表现好坏而剧烈波动
- 保持对手池的多样性

### 6.5 模块的完全独立性

联赛管理模块不依赖 `ppo_antwar` 的任何内部模块，仅依赖 Python 标准库和 `trueskill` 包。这种设计的用意是：
- **可测试**：可以独立于整个训练系统测试评分逻辑
- **可复用**：如果将来需要做不同游戏的联赛系统，可以直接复用
- **可替换**：如果想换用不同的评分系统（如 Elo、Glicko），只需替换 `payoff.py` 中的实现
