# PPO V1 SelfPlay 对手池与评分系统设计文档

## 概述

本文档详细描述了 PPO V1 训练框架中 SelfPlay 预热、对手池管理、对手评分等核心逻辑的实现细节。

---

## 1. SelfPlay 预热机制

### 1.1 预热流程概述

SelfPlay 训练开始前需要构建初始对手池，以便智能体有对手可以对战。预热机制确保训练启动时对手池非空。

### 1.2 初始化流程

```
训练启动
    ↓
尝试加载持久化的league状态
    ↓
对手池为空？
    ├─ 是 → 创建初始随机对手
    └─ 否 → 继续正常训练
```

### 1.3 核心实现

#### 1.3.1 初始对手创建触发

在 `SelfPlayTrainer.__init__` 中检查并创建初始对手：

```python
# 只有当没有加载到任何对手时，才创建初始对手
if len(self.opponent_pool) == 0:
    self._create_initial_opponents()
```

**文件位置**: [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L204-L205)

#### 1.3.2 创建初始对手

`_create_initial_opponents()` 方法根据配置的初始对手数量创建随机策略对手：

```python
def _create_initial_opponents(self) -> None:
    if self.initial_opponents <= 0:
        return

    logger.info(f"Creating {self.initial_opponents} initial opponents...")
    for i in range(self.initial_opponents):
        checkpoint_path = self._create_random_agent_checkpoint()
        if checkpoint_path:
            new_opponent_id = self.selfplay_manager.add_opponent(checkpoint_path)
            logger.info(f"Created initial opponent: {new_opponent_id}")
```

**文件位置**: [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L806-L816)

#### 1.3.3 创建随机策略 Checkpoint

`_create_random_agent_checkpoint()` 方法创建一个使用随机初始化参数的策略网络作为初始对手：

```python
def _create_random_agent_checkpoint(self) -> Optional[str]:
    try:
        from ppo_antwar.network.antwar_net import AntWarPolicy

        policy = AntWarPolicy(
            board_shape=self.trainer.network_config.get('board_shape', (27, 15, 15)),
            global_dim=self.trainer.network_config.get('global_dim', 64),
            action_dim=self.trainer.network_config.get('action_dim', 43),
            hidden_dim=self.trainer.network_config.get('hidden_dim', 256),
        )

        checkpoint = {
            'policy_state_dict': policy.state_dict(),
            'config': {'network': self.trainer.network_config}
        }

        checkpoint_path = str(self.checkpoint_dir / f"initial_opponent_{uuid.uuid4().hex[:8]}.pt")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    except Exception as e:
        logger.error(f"Failed to create random agent checkpoint: {e}")
        return None
```

**文件位置**: [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L818-L840)

### 1.4 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_opponents` | 3 | 初始随机对手数量 |

---

## 2. 对手池管理（OpponentPool）

### 2.1 对手池数据结构

```python
class OpponentPool:
    def __init__(self, max_size: int = 10, min_games_threshold: int = 10) -> None:
        self.max_size = max_size                    # 对手池最大容量
        self.min_games_threshold = min_games_threshold  # 淘汰前最小对战场次
        self._opponents: List[str] = []             # 对手ID列表
        self._checkpoint_paths: Dict[str, str] = {} # 对手checkpoint路径映射
        self._added_timestamps: Dict[str, float] = {} # 添加时间戳
        self._games_played: Dict[str, int] = {}     # 对战场次计数
        self._payoff = None                         # payoff引用（用于淘汰决策）
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L142-L151)

### 2.2 核心操作

#### 2.2.1 添加对手

`add()` 方法将新对手加入池，并在超过容量时触发淘汰：

```python
def add(self, opponent_id: str, checkpoint_path: str = None) -> None:
    if opponent_id not in self._opponents:
        self._opponents.append(opponent_id)
        self._added_timestamps[opponent_id] = 0
        self._games_played[opponent_id] = 0
        if checkpoint_path:
            self._checkpoint_paths[opponent_id] = checkpoint_path
        if len(self._opponents) > self.max_size:
            self._evict_worst()  # 触发淘汰
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L156-L165)

#### 2.2.2 淘汰策略

`_evict_worst()` 方法实现淘汰逻辑：

```python
def _evict_worst(self) -> None:
    # 筛选满足最小对战场次要求的候选对手
    candidates = [
        opp_id for opp_id in self._opponents
        if self._games_played.get(opp_id, 0) >= self.min_games_threshold
    ]

    if not candidates:
        # 没有满足条件的对手，移除第一个
        victim = self._opponents[0]
    else:
        # 计算淘汰分数，选择分数最低的
        def eviction_score(opp_id: str) -> float:
            games = self._games_played.get(opp_id, 0)
            if self._payoff is not None:
                rating = self._payoff.get_trueskill_rating(opp_id)
                mu = rating.mu if rating else 25.0
            else:
                mu = 25.0

            # 对未满足最小场次的对手给予保护加成
            if games < self.min_games_threshold:
                protection_bonus = (self.min_games_threshold - games) * 0.5
            else:
                protection_bonus = 0

            return mu - protection_bonus  # 评分越低越可能被淘汰

        victim = min(candidates, key=eviction_score)

    self._remove_opponent(victim)
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L166-L193)

**淘汰策略要点**:
1. **优先保护**：未达到最小对战场次的对手会获得保护加成
2. **评分优先**：基于 TrueSkill 的 mu 值作为主要淘汰依据
3. **公平性**：新对手有机会积累足够对战数据后才参与淘汰竞争

#### 2.2.3 对战场次计数

每次对战后调用 `increment_games()` 更新计数：

```python
def increment_games(self, opponent_id: str) -> None:
    if opponent_id in self._games_played:
        self._games_played[opponent_id] += 1
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L209-L212)

### 2.3 持久化机制

#### 2.3.1 保存状态

```python
def save_state(self, filepath: str) -> None:
    import json
    state = {
        'max_size': self.max_size,
        'min_games_threshold': self.min_games_threshold,
        'opponent_ids': self._opponents,
        'checkpoint_paths': self._checkpoint_paths,
        'added_timestamps': self._added_timestamps,
        'games_played': self._games_played,
    }
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L238-L250)

#### 2.3.2 加载状态

```python
@classmethod
def load_state(cls, filepath: str) -> "OpponentPool":
    import json
    with open(filepath, 'r') as f:
        state = json.load(f)
    
    pool = cls(
        max_size=state.get('max_size', 10),
        min_games_threshold=state.get('min_games_threshold', 10)
    )
    pool._opponents = state.get('opponent_ids', [])
    pool._checkpoint_paths = state.get('checkpoint_paths', {})
    pool._added_timestamps = state.get('added_timestamps', {})
    pool._games_played = state.get('games_played', {})
    return pool
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L252-L267)

### 2.4 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `opponent_pool_size` | 10 | 对手池最大容量 |
| `min_opponent_games` | 8 | 淘汰前需要的最小对战场次 |

---

## 3. 对手选择器（OpponentSelector）

### 3.1 选择器架构

```
┌─────────────────────────────────────────────────────────────┐
│                    OpponentSelector                         │
├─────────────────────────────────────────────────────────────┤
│  exploit_prob: 0.7  ← 动态调整（基于训练进度）              │
│  explore_prob: 0.3  ← 动态调整                            │
│  adaptive: True     ← 是否启用自适应                      │
│  schedule: "linear" ← 调整策略（linear/sigmoid/exp）       │
├─────────────────────────────────────────────────────────────┤
│  select()                                                  │
│    ├─ random < exploit_prob → _exploit_select()           │
│    └─ random ≥ exploit_prob → _explore_select()           │
└─────────────────────────────────────────────────────────────┘
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L9-L49)

### 3.2 自适应探索/利用调整

`_get_adaptive_exploit_prob()` 根据训练进度动态调整 exploit 概率：

```python
def _get_adaptive_exploit_prob(self) -> float:
    if not self.adaptive:
        return self.base_exploit_prob

    progress = min(1.0, self._total_episodes / 10000.0)  # 归一化进度

    if self.schedule == "linear":
        return 0.3 + 0.6 * progress          # 0.3 → 0.9
    elif self.schedule == "sigmoid":
        return 0.3 + 0.6 * (1 / (1 + math.exp(-10 * (progress - 0.5))))
    elif self.schedule == "exp":
        return 0.3 + 0.6 * (1 - math.exp(-5 * progress))
    else:
        return self.base_exploit_prob
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L33-L48)

**策略对比**:

| 策略 | 公式 | 特点 |
|------|------|------|
| linear | `0.3 + 0.6 * progress` | 线性增长，平稳过渡 |
| sigmoid | `0.3 + 0.6 / (1 + exp(-10*(p-0.5)))` | S形曲线，中期加速 |
| exp | `0.3 + 0.6 * (1 - exp(-5*p))` | 指数增长，前期快速上升 |

### 3.3 利用选择（Exploit）

`_exploit_select()` 选择高置信度的顶级对手进行对战：

```python
def _exploit_select(self, player_id: str, opponent_candidates: List[str]) -> Optional[str]:
    if not opponent_candidates:
        return None

    # 筛选高置信度对手（sigma < 5.0）
    confident_opponents = [
        opp_id for opp_id in opponent_candidates
        if self._is_confident_opponent(opp_id)
    ]

    if confident_opponents:
        # 根据TrueSkill mu值排序，取前3名随机选择
        rated_opponents = [
            (opp_id, self.payoff.get_trueskill_rating(opp_id).mu)
            for opp_id in confident_opponents
        ]
        rated_opponents.sort(key=lambda x: x[1], reverse=True)
        top_opponents = [opp_id for opp_id, _ in rated_opponents[:min(3, len(rated_opponents))]]
        return random.choice(top_opponents) if top_opponents else None

    # 没有高置信度对手时，选择最确定的（sigma最小）
    all_ratings = [
        (opp_id, self.payoff.get_trueskill_rating(opp_id).sigma)
        for opp_id in opponent_candidates
        if self.payoff.get_trueskill_rating(opp_id) is not None
    ]
    if not all_ratings:
        return None
    all_ratings.sort(key=lambda x: x[1])
    return all_ratings[0][0]
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L71-L97)

**置信度判断标准**:

```python
def _is_confident_opponent(self, opponent_id: str) -> bool:
    rating = self.payoff.get_trueskill_rating(opponent_id)
    if rating is None:
        return False
    return rating.sigma < self._min_confidence_sigma  # 默认 sigma < 5.0
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L50-L54)

### 3.4 探索选择（Explore）

`_explore_select()` 选择低置信度或未评估的对手进行对战：

```python
def _explore_select(self, opponent_candidates: List[str]) -> str:
    if not opponent_candidates:
        raise ValueError("No opponent candidates available")

    # 筛选低置信度或未评估的对手
    filtered = [
        opp_id for opp_id in opponent_candidates
        if not self.payoff.get_trueskill_rating(opp_id) or
           self.payoff.get_trueskill_rating(opp_id).sigma < self._min_confidence_sigma * 1.5
    ]

    if not filtered:
        filtered = opponent_candidates

    return random.choice(filtered)
```

**文件位置**: [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L99-L112)

### 3.5 选择流程总结

```
select(player_id, candidates)
    │
    ├─ 获取自适应 exploit_prob
    │
    ├─ random() < exploit_prob?
    │    ├─ 是 → _exploit_select()
    │    │       ├─ 有高置信度对手？
    │    │       │    ├─ 是 → 取前3名mu最高的随机选
    │    │       │    └─ 否 → 取sigma最小的
    │    │       └─ 返回对手ID
    │    │
    │    └─ 否 → _explore_select()
    │            ├─ 筛选低置信度/未评估对手
    │            └─ 随机选择
    │
    └─ 返回选中的对手ID
```

---

## 4. 对手评分系统（BattleSharedPayoff）

### 4.1 TrueSkill 评分模型

系统采用 TrueSkill 算法进行对手实力评估：

```python
@dataclass
class TrueSkillRating:
    mu: float = 25.0      # 平均实力（mean）
    sigma: float = 8.33   # 不确定度（standard deviation）
    games_played: int = 0 # 对战场次

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return (self.mu - 3 * self.sigma, self.mu + 3 * self.sigma)

    @property
    def is_confident(self) -> bool:
        return self.sigma < 3.0  # sigma < 3.0 视为高置信度
```

**文件位置**: [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L12-L31)

### 4.2 评分更新机制

`update()` 方法在每次对战后更新评分：

```python
def update(self, home: str, away: str, result: int) -> None:
    self.add_player(home)
    self.add_player(away)

    # 更新对战记录
    key = self.get_key(home, away)
    record = self._data[key]
    record['games'] += 1
    if result == 0:
        record['draws'] += 1
    elif result == 1:
        record['wins'] += 1
    else:
        record['losses'] += 1

    # 更新反向记录（away vs home）
    reverse_key = self.get_key(away, home)
    reverse_record = self._data[reverse_key]
    reverse_record['games'] += 1
    if result == 0:
        reverse_record['draws'] += 1
    elif result == 1:
        reverse_record['losses'] += 1
    else:
        reverse_record['wins'] += 1

    # 更新 TrueSkill 评分
    home_rating = self._get_or_create_rating(home)
    away_rating = self._get_or_create_rating(away)

    ts_home = home_rating.to_trueskill()
    ts_away = away_rating.to_trueskill()

    if result == 1:
        new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away)
    elif result == -1:
        new_away, new_home = trueskill.rate_1vs1(ts_away, ts_home)
    else:
        new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away, drawn=True)

    self._trueskill_ratings[home] = TrueSkillRating.from_trueskill(
        new_home, games_played=home_rating.games_played + 1
    )
    self._trueskill_ratings[away] = TrueSkillRating.from_trueskill(
        new_away, games_played=away_rating.games_played + 1
    )
```

**文件位置**: [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L76-L119)

**评分更新流程**:

```
对战结果 → 更新双向对战记录 → TrueSkill计算 → 更新双方mu/sigma
    │
    ├─ result=1 (home胜) → home.mu↑, away.mu↓
    ├─ result=-1 (away胜) → home.mu↓, away.mu↑
    └─ result=0 (平局) → 双方mu变化较小
```

### 4.3 胜率计算

`get_win_rate()` 计算平滑后的胜率估计：

```python
def get_win_rate(self, home: str, away: str) -> float:
    key = self.get_key(home, away)
    if key not in self._data:
        return 0.5  # 默认0.5（未对战过）

    record = self._data[key]
    games = record['games']

    if games == 0:
        return 0.5

    # 使用贝叶斯平滑
    alpha = 1 + record['wins'] + record['draws'] * 0.5
    beta = 1 + record['losses'] + record['draws'] * 0.5
    win_rate = alpha / (alpha + beta)

    # 置信度加权：对战越少越接近先验0.5
    confidence_weight = min(1.0, games / self._min_win_rate_games)
    win_rate = 0.5 + confidence_weight * (win_rate - 0.5)

    return win_rate
```

**文件位置**: [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L124-L143)

**贝叶斯平滑原理**:
- 先验假设：胜率为 0.5
- 随着对战次数增加，逐渐向实际胜率收敛
- `min_win_rate_games` 控制收敛速度

### 4.4 可利用性计算

`get_exploitability()` 衡量当前智能体相对于对手池的整体实力：

```python
def get_exploitability(self) -> float:
    if not self._trueskill_ratings:
        return 1.0

    current = self._trueskill_ratings.get("current_agent")
    if current is None:
        return 1.0

    opponent_ratings = [
        r for pid, r in self._trueskill_ratings.items()
        if pid != "current_agent" and r.games_played > 0
    ]

    if not opponent_ratings:
        return 1.0

    avg_mu = sum(r.mu for r in opponent_ratings) / len(opponent_ratings)
    avg_sigma = sum(r.sigma for r in opponent_ratings) / len(opponent_ratings)

    if current.mu <= avg_mu:
        advantage = (avg_mu - current.mu) / (current.sigma + avg_sigma + 1e-6)
        return min(1.0, 0.5 + advantage * 0.5)
    else:
        advantage = (current.mu - avg_mu) / (current.sigma + avg_sigma + 1e-6)
        return max(0.0, 0.5 - advantage * 0.5)
```

**文件位置**: [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L203-L227)

**可利用性解读**:

| 值 | 含义 |
|----|------|
| 1.0 | 当前智能体远弱于对手池平均水平 |
| 0.5 | 当前智能体与对手池平均水平相当 |
| 0.0 | 当前智能体远强于对手池平均水平 |

### 4.5 历史记录衰减

`decay_all()` 定期衰减历史对战记录的权重：

```python
def decay_all(self) -> None:
    for key in self._data:
        self._data[key] = self._data[key] * self._decay  # 默认decay=0.99
```

**文件位置**: [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L145-L147)

**衰减机制作用**:
- 降低陈旧数据的权重
- 使评分更反映近期表现
- 默认每 1000 轮训练衰减一次（`decay_interval=1000`）

### 4.6 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `decay` | 0.99 | 历史记录衰减因子 |
| `min_win_rate_games` | 8 | 胜率计算的最小对战场次 |

---

## 5. SelfPlay 管理器（SelfPlayManager）

### 5.1 核心职责

`SelfPlayManager` 作为 SelfPlay 训练的核心协调器：

```python
class SelfPlayManager:
    def __init__(
        self,
        agent: PPOAgent,
        opponent_pool: OpponentPool,
        payoff: BattleSharedPayoff,
        exploit_prob: float = 0.7,
        explore_prob: float = 0.3,
        adaptive_explore: bool = True,
        explore_schedule: str = "linear",
    ) -> None:
        self.agent = agent
        self.opponent_pool = opponent_pool
        self.payoff = payoff
        self.selector = OpponentSelector(
            payoff=payoff,
            exploit_prob=exploit_prob,
            explore_prob=explore_prob,
            adaptive=adaptive_explore,
            schedule=explore_schedule,
        )
        self.current_player_id = "current_agent"
        self.games_against_opponents = {}  # 内部统计
```

**文件位置**: [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L22-L48)

### 5.2 主要方法

| 方法 | 功能 |
|------|------|
| `update_progress(episode)` | 更新训练进度，用于动态调整探索/利用 |
| `select_opponent()` | 从对手池中选择下一个对战对手 |
| `add_opponent(checkpoint_path)` | 添加新对手到池 |
| `update_payoff(opponent_id, result, rounds)` | 更新对战结果和评分 |
| `get_exploitability()` | 获取当前智能体的可利用性评分 |
| `get_statistics()` | 获取完整的统计信息 |

### 5.3 统计信息结构

```python
def get_statistics(self) -> dict:
    stats = {
        'num_opponents': len(self.opponent_pool),
        'games_against_opponents': dict(self.games_against_opponents),
        'exploitability': self.get_exploitability(),
        'avg_rounds': (self._total_rounds / self._total_battles) if self._total_battles > 0 else 0.0,
    }

    if len(self.opponent_pool) > 0:
        win_rates = self.payoff.get_all_win_rates(self.current_player_id)
        stats['win_rates'] = win_rates

    return stats
```

**文件位置**: [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L125-L137)

---

## 6. 整体架构图

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SelfPlayTrainer                                │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │   PPOTrainer     │    │   BattleCoordinator│  │   SelfPlayLogger │ │
│  │   (策略训练)      │    │   (Baseline对战)   │  │   (日志记录)      │ │
│  └────────┬─────────┘    └──────────────────┘    └──────────────────┘ │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      SelfPlayManager                             │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │  │
│  │  │OpponentPool  │←──│OpponentSelector│←──│BattleShared  │       │  │
│  │  │ (对手池)      │    │   (选择器)      │    │   Payoff     │       │  │
│  │  │              │    │              │    │   (评分系统)   │       │  │
│  │  │ - add()      │    │ - select()   │    │ - update()   │       │  │
│  │  │ - evict()    │    │ - exploit   │    │ - win_rate() │       │  │
│  │  │ - persist()  │    │ - explore   │    │ - trueskill  │       │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 训练流程时序

```
训练开始
    │
    ├─ 加载league状态（对手池+payoff）
    │       ↓
    │   对手池为空？→ 创建初始随机对手
    │
    ├─ 进入训练循环（每episode）
    │       │
    │       ├─ update_progress(episode) → 动态调整exploit_prob
    │       │
    │       ├─ select_opponent() → 选择对手
    │       │       │
    │       │       └─ exploit/explore 选择策略
    │       │
    │       ├─ 加载对手模型 → 对战
    │       │       │
    │       │       └─ 收集训练数据（EpisodeBatch）
    │       │
    │       ├─ update_payoff() → 更新评分
    │       │
    │       ├─ PPO更新 → 更新当前策略
    │       │
    │       └─ 检查对手更新间隔 → 添加新对手到池
    │
    ├─ 定期执行
    │       │
    │       ├─ decay_interval → 历史记录衰减
    │       │
    │       └─ battle_interval → Baseline对战评估
    │
    └─ 训练结束
```

---

## 8. 关键配置汇总

| 配置项 | 默认值 | 所属模块 | 说明 |
|--------|--------|----------|------|
| `opponent_pool_size` | 10 | SelfPlayTrainer | 对手池最大容量 |
| `min_opponent_games` | 8 | SelfPlayTrainer | 淘汰前最小对战场次 |
| `initial_opponents` | 3 | SelfPlayTrainer | 初始随机对手数量 |
| `exploit_prob` | 0.7 | SelfPlayManager | 初始利用概率 |
| `explore_prob` | 0.3 | SelfPlayManager | 初始探索概率 |
| `adaptive_explore` | True | OpponentSelector | 是否启用自适应 |
| `explore_schedule` | "linear" | OpponentSelector | 自适应策略 |
| `decay` | 0.99 | BattleSharedPayoff | 历史记录衰减因子 |
| `min_win_rate_games` | 8 | BattleSharedPayoff | 胜率计算最小场次 |
| `opponent_update_interval` | 500 | training | 对手更新间隔（episode） |
| `decay_interval` | 1000 | training | 历史衰减间隔（episode） |

---

## 9. 持久化机制

### 9.1 状态保存位置

```
checkpoint_dir/
    └── league_state/
            ├── opponent_pool.json    # 对手池状态
            └── payoff.json           # 对战记录和评分
```

### 9.2 保存时机

1. **添加新对手后**：调用 `_save_league_state()`
2. **训练结束时**：可手动保存

### 9.3 加载时机

`SelfPlayTrainer.__init__` 初始化时自动加载

---

## 10. 扩展与优化建议

### 10.1 潜在改进方向

1. **动态池大小**：根据训练进度自动调整对手池容量
2. **分层对手池**：将对手按实力等级分层管理
3. **多样性激励**：鼓励选择不同风格的对手
4. **在线学习率调整**：根据可利用性动态调整学习率
5. **对手归档**：将被淘汰的强对手保存到归档池

### 10.2 监控指标建议

| 指标 | 说明 |
|------|------|
| 对手池大小 | 监控池容量变化 |
| 平均 sigma | 评估评分置信度 |
| 可利用性 | 评估当前智能体实力 |
| exploit/explore 比例 | 监控选择策略效果 |
| 淘汰频率 | 评估淘汰策略合理性 |