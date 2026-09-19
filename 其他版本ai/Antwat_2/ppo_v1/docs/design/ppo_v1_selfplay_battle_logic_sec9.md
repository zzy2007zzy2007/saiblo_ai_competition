# 9. 对手管理与 League 机制

对手管理与 League 机制是 SelfPlay 训练的核心模块，负责维护一个动态演化的对手池，通过 TrueSkill 评分系统评估每个对手（以及当前智能体）的实力，并基于 exploit/explore 策略为每轮训练选择合适的对手。三个核心类——`OpponentSelector`、`OpponentPool`、`BattleSharedPayoff`——由 `SelfPlayManager` 统一编排。

> 关键源码位置：
> - [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py) — `OpponentSelector` 和 `OpponentPool`
> - [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py) — `BattleSharedPayoff` 和 `TrueSkillRating`
> - [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py) L22-L138 — `SelfPlayManager`

---

## 9.1 OpponentSelector — 对手选择

### 9.1.1 概述

`OpponentSelector` 负责为当前训练智能体从对手池中选择一个合适的对手。它采用 exploit/explore 混合策略：以一定的概率选择最强对手进行“利用”（挑战高难度），以剩余概率随机探索（增加多样性）。

构造参数（见 [opponent_selector.py:L10-L26](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L10-L26)）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `payoff` | `BattleSharedPayoff` | — | 战绩和评分数据源 |
| `exploit_prob` | `float` | `0.7` | 基础 exploit 概率 |
| `explore_prob` | `float` | `0.3` | 基础 explore 概率（实际 unused，仅记录） |
| `adaptive` | `bool` | `True` | 是否开启自适应调度 |
| `schedule` | `str` | `"linear"` | 自适应调度类型（`"linear"` / `"sigmoid"` / `"exp"`） |

内部状态：

- `_min_confidence_sigma`：固定为 `5.0`，用于判定一个对手的 TrueSkill 评分是否“置信”。

### 9.1.2 选择入口：`select()`

核心方法 `select(player_id, opponent_candidates)`（[opponent_selector.py:L56-L69](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L56-L69)）的逻辑如下：

```python
def select(self, player_id: str, opponent_candidates: List[str]) -> str:
    exploit_prob = self._get_adaptive_exploit_prob()

    if random.random() < exploit_prob:
        selected = self._exploit_select(player_id, opponent_candidates)
        if selected is not None:
            return selected

    return self._explore_select(opponent_candidates)
```

1. 调用 `_get_adaptive_exploit_prob()` 获取当前阶段的 exploit 概率；
2. 以 `exploit_prob` 的概率尝试 exploit 选择；
3. 如果 exploit 选择失败（返回 `None`，例如没有符合置信条件的对手），则降级为 explore 选择。

批量选择 `select_batch(player_id, opponent_candidates, num_selections)` 则是多次调用 `select()`，且保证同一批次内不会重复选择同一个对手。

### 9.1.3 Exploit 策略：`_exploit_select()`

Exploit 策略的目标是选出 TrueSkill 评分最高且已“置信”的对手，让智能体挑战强者。具体步骤（[opponent_selector.py:L71-L97](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L71-L97)）：

**第一步：筛选 confident 对手**

```python
confident_opponents = [
    opp_id for opp_id in opponent_candidates
    if self._is_confident_opponent(opp_id)
]
```

`_is_confident_opponent()` 的判断条件为 `sigma < 5.0`（[opponent_selector.py:L50-L54](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L50-L54)）：

```python
def _is_confident_opponent(self, opponent_id: str) -> bool:
    rating = self.payoff.get_trueskill_rating(opponent_id)
    if rating is None:
        return False
    return rating.sigma < self._min_confidence_sigma  # sigma < 5.0
```

sigma 越低，说明该对手的 TrueSkill 评分越稳定、越可信。只有 sigma < 5.0 的对手才被认为是“已充分评估”的强者，值得 exploit。

**第二步：在 confident 对手中选 mu 最高的**

```python
if confident_opponents:
    rated_opponents = [
        (opp_id, self.payoff.get_trueskill_rating(opp_id).mu)
        for opp_id in confident_opponents
    ]
    rated_opponents.sort(key=lambda x: x[1], reverse=True)
    top_opponents = [opp_id for opp_id, _ in rated_opponents[:min(3, len(rated_opponents))]]
    return random.choice(top_opponents)
```

将所有 confident 对手按 mu（TrueSkill 实力均值）降序排列，取前 3 名，再从中随机选一个。这既保证对手足够强，又引入一定的多样性避免每次都选同一个对手。

**第三步：降级策略**

如果没有 confident 对手，则直接选 sigma 最低的对手（评分最稳定的）：

```python
all_ratings = [
    (opp_id, self.payoff.get_trueskill_rating(opp_id).sigma)
    for opp_id in opponent_candidates
    if self.payoff.get_trueskill_rating(opp_id) is not None
]
all_ratings.sort(key=lambda x: x[1])
return all_ratings[0][0]
```

### 9.1.4 Explore 策略：`_explore_select()`

Explore 策略的目标是随机探索，避免过度拟合少数对手。它将候选对手中 sigma 过高的（即评分极不稳定的）过滤掉，再从剩余对手中随机抽取（[opponent_selector.py:L99-L112](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L99-L112)）：

```python
def _explore_select(self, opponent_candidates: List[str]) -> str:
    filtered = [
        opp_id for opp_id in opponent_candidates
        if not self.payoff.get_trueskill_rating(opp_id) or
           self.payoff.get_trueskill_rating(opp_id).sigma < self._min_confidence_sigma * 1.5
    ]
    if not filtered:
        filtered = opponent_candidates
    return random.choice(filtered)
```

过滤条件：`sigma < 5.0 * 1.5 = 7.5`，或该对手尚无 TrueSkill 记录。sigma >= 7.5 的对手意味着其评分极为不确定，explore 阶段也会将其排除。如果过滤后为空，则回退到从全部候选中随机选。

> **注意**：`TrueSkillRating` 的默认初始 sigma 为 `8.33`（见 §9.3.1），略高于 7.5 的阈值。这意味着刚加入的对手默认满足 `sigma >= 7.5`，在第一次对战并更新评分前不会被 explore 选中。

### 9.1.5 自适应 Exploit Prob 调度

当 `adaptive=True` 时，`_get_adaptive_exploit_prob()` 方法会根据训练进度动态调整 exploit 概率，使训练早期偏向探索，后期偏向利用（[opponent_selector.py:L32-L48](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L32-L48)）：

```python
def _get_adaptive_exploit_prob(self) -> float:
    if not self.adaptive:
        return self.base_exploit_prob

    progress = min(1.0, self._total_episodes / 10000.0)

    if self.schedule == "linear":
        return 0.3 + 0.6 * progress
    ...
```

核心公式：`progress = min(1.0, total_episodes / 10000)`，即将前 10000 个 episode 映射到 [0, 1] 区间。

三种调度曲线：

| 调度 | 公式 | 特点 |
|------|------|------|
| `linear` | `0.3 + 0.6 × progress` | exploit_prob 从 0.3 线性增长到 0.9 |
| `sigmoid` | `0.3 + 0.6 × sigmoid(10×(progress-0.5))` | S 形，在 progress=0.5 附近快速切换 |
| `exp` | `0.3 + 0.6 × (1 - e^(-5×progress))` | 指数衰减型，前期增长快、后期趋缓 |

所有调度的 exploit_prob 范围均为 **[0.3, 0.9]**。

训练进度通过 `update_progress(episodes)` 方法更新（[opponent_selector.py:L28-L30](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L28-L30)），由 `SelfPlayManager.update_progress()` 在每轮训练循环中调用。

---

## 9.2 OpponentPool — 对手池

### 9.2.1 概述

`OpponentPool` 管理所有历史对手的集合，包括添加、淘汰和持久化。它维护一个有最大容量限制的对手列表，当池满时自动淘汰弱者。

构造参数（[opponent_selector.py:L143](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L143)）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_size` | `int` | `10` | 池最大容量 |
| `min_games_threshold` | `int` | `10` | 对手需要累积多少场对战才能被纳入淘汰候选 |

内部状态（[opponent_selector.py:L146-L150](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L146-L150)）：

- `_opponents`：对手 ID 列表（按添加顺序）
- `_checkpoint_paths`：对手 ID → 模型 checkpoint 路径
- `_added_timestamps`：对手 ID → 添加时间戳
- `_games_played`：对手 ID → 累计对战场次
- `_payoff`：可选，指向 `BattleSharedPayoff` 实例，用于淘汰时获取 TrueSkill 评分

### 9.2.2 添加对手

```python
def add(self, opponent_id: str, checkpoint_path: str = None) -> None:
    if opponent_id not in self._opponents:
        self._opponents.append(opponent_id)
        self._games_played[opponent_id] = 0
        if checkpoint_path:
            self._checkpoint_paths[opponent_id] = checkpoint_path
        if len(self._opponents) > self.max_size:
            self._evict_worst()
```

（[opponent_selector.py:L156-L164](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L156-L164)）

流程：
1. 检查重复（幂等性保证）；
2. 加入 `_opponents` 列表，初始化对战场次为 0，记录 checkpoint 路径；
3. 如果池大小超过 `max_size`（10），触发淘汰。

### 9.2.3 淘汰策略：`_evict_worst()`

淘汰逻辑的优先级是：**保护对战场次不足的对手 → 在满足条件的对手中淘汰评分最低的**。

```python
def _evict_worst(self) -> None:
    candidates = [
        opp_id for opp_id in self._opponents
        if self._games_played.get(opp_id, 0) >= self.min_games_threshold
    ]

    if not candidates:
        victim = self._opponents[0]
    else:
        def eviction_score(opp_id: str) -> float:
            ...
            return mu - protection_bonus

        victim = min(candidates, key=eviction_score)

    self._remove_opponent(victim)
```

（[opponent_selector.py:L166-L193](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L166-L193)）

**淘汰得分（eviction_score）** 的计算方式：

```
eviction_score = mu - protection_bonus
```

其中：

- `mu` 来自 `payoff.get_trueskill_rating(opp_id).mu`（若无则默认 25.0）
- `protection_bonus`：
  - 若 `games < min_games_threshold`（即 10 场）：`protection_bonus = (10 - games) × 0.5`
  - 若 `games >= 10`：`protection_bonus = 0`

保护加成的含义是：场次越少的对手获得的保护分越多。例如，一个只打了 2 场的对手获得 `(10-2) × 0.5 = 4.0` 的保护分，其有效淘汰得分 = `mu - 4.0`，这使得它比其他 mu 低但已打满 10 场的对手更难被淘汰。

**如果所有对手都不满 10 场**（`candidates` 为空），则直接淘汰最先加入的对手（`_opponents[0]`）。

淘汰后调用 `_remove_opponent()` 清理所有相关数据（`_opponents`、`_checkpoint_paths`、`_added_timestamps`、`_games_played`）。

### 9.2.4 对战场次管理

每次对战结束后，由 `SelfPlayManager.update_payoff()` 调用 `opponent_pool.increment_games(opponent_id)`（[opponent_selector.py:L209-L212](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L209-L212)）：

```python
def increment_games(self, opponent_id: str) -> None:
    if opponent_id in self._games_played:
        self._games_played[opponent_id] += 1
```

### 9.2.5 持久化

`save_state(filepath)` 将对手池的完整状态序列化为 JSON（[opponent_selector.py:L238-L250](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L238-L250)），包含字段：

- `max_size`、`min_games_threshold`
- `opponent_ids`、`checkpoint_paths`、`added_timestamps`、`games_played`

对应的 `load_state(filepath)` 类方法从 JSON 恢复 `OpponentPool` 实例（[opponent_selector.py:L252-L267](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L252-L267)）。

### 9.2.6 其他接口

| 方法 | 说明 |
|------|------|
| `get_all()` | 返回所有对手 ID 列表的副本 |
| `get_random()` | 随机返回一个对手 ID |
| `get_checkpoint_path(opponent_id)` | 获取对手的模型 checkpoint 路径 |
| `remove(opponent_id)` | 手动移除指定对手 |
| `set_payoff_reference(payoff)` | 设置 payoff 引用，供淘汰时查询评分 |
| `__len__()` | 返回当前池中的对手数量 |
| `__contains__(opponent_id)` | 检查对手是否在池中 |

---

## 9.3 TrueSkill 评分系统

TrueSkill 评分系统由 `TrueSkillRating` 数据类和 `BattleSharedPayoff` 中的 `rate_1vs1()` 更新逻辑组成。

### 9.3.1 TrueSkillRating

`TrueSkillRating` 是封装单个玩家 TrueSkill 评分的轻量数据类（[payoff.py:L12-L31](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L12-L31)）：

```python
@dataclass
class TrueSkillRating:
    mu: float = 25.0        # 实力均值（默认 25.0）
    sigma: float = 8.33     # 实力不确定性（默认 8.33）
    games_played: int = 0   # 累计对战场次
```

**字段含义**：

| 字段 | 含义 | 默认值 |
|------|------|--------|
| `mu` | 玩家实力的估计均值，越高越强 | `25.0` |
| `sigma` | 实力估计的不确定性，越低越置信 | `8.33` |
| `games_played` | 玩家参与的对战总场次 | `0` |

**辅助属性和方法**：

- `confidence_interval`：返回 `(mu - 3σ, mu + 3σ)`，99.7% 置信区间
- `is_confident`：返回 `sigma < 3.0`（注意：这与 `OpponentSelector` 中的 `sigma < 5.0` 是不同的阈值，`TrueSkillRating.is_confident` 更严格）
- `to_trueskill()`：转换为 `trueskill.Rating` 对象，供 `trueskill.rate_1vs1()` 使用
- `from_trueskill(rating, games_played)`：从 `trueskill.Rating` 和场次创建 `TrueSkillRating`

### 9.3.2 `rate_1vs1()` 更新

每次对战后，`BattleSharedPayoff.update()` 使用 `trueskill.rate_1vs1()` 更新双方评分（[payoff.py:L101-L119](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L101-L119)）：

```python
ts_home = home_rating.to_trueskill()
ts_away = away_rating.to_trueskill()

if result == 1:                                          # home 胜
    new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away)
elif result == -1:                                       # away 胜
    new_away, new_home = trueskill.rate_1vs1(ts_away, ts_home)
else:                                                    # 平局
    new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away, drawn=True)

self._trueskill_ratings[home] = TrueSkillRating.from_trueskill(
    new_home, games_played=home_rating.games_played + 1
)
self._trueskill_ratings[away] = TrueSkillRating.from_trueskill(
    new_away, games_played=away_rating.games_played + 1
)
```

关键点：
- `trueskill.rate_1vs1()` 是标准 TrueSkill 算法，根据比赛结果更新 mu 和 sigma；
- 胜者 mu 上升、败者 mu 下降；sigma 在双方都下降（随着对战增多，评分越来越确定）；
- `games_played` 字段仅做累加计数，不参与 TrueSkill 算法（TrueSkill 通过 sigma 的下降隐含地反映了经验的积累）。

### 9.3.3 Explitability 计算

`BattleSharedPayoff.get_exploitability()` 用于衡量当前智能体相对于对手池的实力强弱（[payoff.py:L203-L227](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L203-L227)）：

1. 获取 `"current_agent"` 的 TrueSkill 评分；
2. 计算所有对手的 mu 均值和 sigma 均值；
3. 比较当前 agent 的 mu 与对手平均 mu：
   - 若 `current.mu <= avg_mu`（agent 弱于对手）：`exploitability = min(1.0, 0.5 + (avg_mu - current.mu) / (current.sigma + avg_sigma) × 0.5)`
   - 若 `current.mu > avg_mu`（agent 强于对手）：`exploitability = max(0.0, 0.5 - (current.mu - avg_mu) / (current.sigma + avg_sigma) × 0.5)`

返回值范围 [0, 1]：
- `≈ 1.0`：当前智能体远弱于对手池
- `≈ 0.5`：当前智能体与对手池平均实力相当
- `≈ 0.0`：当前智能体远强于对手池

---

## 9.4 BattleSharedPayoff — 战绩记录

### 9.4.1 概述

`BattleSharedPayoff` 是 League 系统的核心数据层，负责：

1. **维护所有玩家间的双向对战记录**（home vs away）
2. **基于贝叶斯方法计算胜率**（带 confidence weight）
3. **维护 TrueSkill 评分**
4. **支持历史衰减（decay）**

构造参数（[payoff.py:L50-L60](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L50-L60)）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `decay` | `float` | `0.99` | 历史衰减因子 |
| `min_win_rate_games` | `int` | `8` | 胜率置信权重所需的场次阈值 |

内部状态：
- `_players` / `_players_ids`：所有已注册的玩家 ID
- `_data`：`defaultdict(BattleRecordDict)`，key 为 `"home-away"` 格式
- `_trueskill_ratings`：玩家 ID → `TrueSkillRating`

### 9.4.2 双向战绩记录

`update(home, away, result)` 的核心特点是**双向记录**（[payoff.py:L76-L99](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L76-L99)）：

```python
key = self.get_key(home, away)          # "home-away"
reverse_key = self.get_key(away, home)  # "away-home"

# home 视角
record['wins'] += 1 if result == 1 else 0
record['losses'] += 1 if result == -1 else 0
record['draws'] += 1 if result == 0 else 0

# away 视角（镜像反转）
reverse_record['losses'] += 1 if result == 1 else 0
reverse_record['wins'] += 1 if result == -1 else 0
reverse_record['draws'] += 1 if result == 0 else 0
```

这意味着每场对战同时在 `"home-away"` 和 `"away-home"` 两条记录中各记一次，确保从任一玩家视角查询胜率时都能获得正确的数据。

`BattleRecordDict` 是 `dict` 的子类，包含四个字段：`wins`、`draws`、`losses`、`games`（[payoff.py:L34-L46](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L34-L46)）。

### 9.4.3 胜率计算（带 Confidence Weight）

`get_win_rate(home, away)` 采用贝叶斯方法计算胜率（[payoff.py:L124-L143](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L124-L143)）：

```python
alpha = 1 + record['wins'] + record['draws'] * 0.5   # Beta 先验 alpha
beta  = 1 + record['losses'] + record['draws'] * 0.5  # Beta 先验 beta

win_rate = alpha / (alpha + beta)

confidence_weight = min(1.0, games / self._min_win_rate_games)  # games / 8
win_rate = 0.5 + confidence_weight * (win_rate - 0.5)
```

**计算逻辑**：

1. 使用 **Beta(α, β)** 分布建模胜率，α = 1 + wins + 0.5×draws，β = 1 + losses + 0.5×draws（先验为 Beta(1,1)，即均匀分布，均值为 0.5）；
2. 后验均值 `win_rate = α / (α + β)` 即为贝叶斯胜率估计；
3. 引入 **confidence weight** 对胜率进行回归收缩（shrinkage）：
   - 当 `games < 8` 时，`confidence_weight = games / 8 < 1`，胜率向 0.5 收缩；
   - 当 `games >= 8` 时，`confidence_weight = 1`，直接使用贝叶斯估计。

这种设计确保在数据量不足时（对战场次少）胜率估计保守地偏向 0.5，避免被偶然结果误导。

### 9.4.4 历史衰减机制

`decay_all()` 方法对所有战绩记录乘以衰减因子 0.99（[payoff.py:L145-L147](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L145-L147)）：

```python
def decay_all(self) -> None:
    for key in self._data:
        self._data[key] = self._data[key] * self._decay
```

每次调用（通常在每个训练 episode 后），所有 `wins`、`draws`、`losses`、`games` 计数都会乘以 0.99。这确保了**近期的对战记录比早期的有更高的权重**，使 League 系统能够适应智能体实力的持续变化。

> **注意**：`BattleRecordDict.__mul__` 对所有字段施加相同的衰减因子（[payoff.py:L42-L46](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L42-L46)），包括 `games`。这意味着 `games` 字段不再是精确的总场次，而是“有效场次”（effective games）。这与 §9.4.3 的 confidence weight 设计是兼容的——衰减后的小数 games 通过 `min(1.0, games/8)` 平滑过渡。

### 9.4.5 持久化

`save_state(filepath)` 将完整的 payoff 状态序列化为 JSON（[payoff.py:L229-L248](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L229-L248)），包含：

- `players`、`players_ids`
- `trueskill_ratings`：每个玩家的 `mu`、`sigma`、`games_played`
- `battle_records`：完整双向战绩记录
- `decay`、`min_win_rate_games`

`load_state(filepath)` 类方法可恢复完整状态（[payoff.py:L250-L268](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L250-L268)）。

### 9.4.6 其他接口

| 方法 | 说明 |
|------|------|
| `add_player(player_id)` | 注册新玩家 |
| `get_win_rate(home, away)` | 查询 home 对 away 的贝叶斯胜率 |
| `get_all_win_rates(player_id)` | 查询某玩家对所有对手的胜率字典 |
| `get_payoff_matrix()` | 返回 n×n 的胜率矩阵 |
| `get_trueskill_rating(player_id)` | 获取某玩家的 TrueSkill 评分 |
| `get_all_trueskill_ratings()` | 获取所有玩家的 TrueSkill 评分 |
| `get_battle_stats()` | 返回各玩家聚合统计数据 |
| `get_exploitability()` | 计算当前 agent 的可利用性分数 |

---

## 9.5 SelfPlayManager — 整合管理

### 9.5.1 概述

`SelfPlayManager` 是将对手管理的三大模块（`OpponentPool`、`BattleSharedPayoff`、`OpponentSelector`）整合在一起的门面类。它不直接操作环境或执行对战，而是作为对手生命周期管理的编排层。

构造参数（[selfplay.py:L23-L47](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L23-L47)）：

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
    ):
```

| 参数 | 说明 |
|------|------|
| `agent` | 当前训练的 PPO 智能体 |
| `opponent_pool` | 外部传入的 `OpponentPool` 实例 |
| `payoff` | 外部传入的 `BattleSharedPayoff` 实例 |
| `exploit_prob` | 基础 exploit 概率（默认 0.7） |
| `explore_prob` | 基础 explore 概率（默认 0.3） |
| `adaptive_explore` | 是否开启自适应调度（默认 True） |
| `explore_schedule` | 调度类型（默认 `"linear"`） |

内部状态：
- `self.current_player_id = "current_agent"`：固定标识符
- `self.games_against_opponents`：内部统计字典（用于监控）
- `self._current_episode`、`self._total_rounds`、`self._total_battles`：进度和统计计数器

### 9.5.2 组合关系

```
SelfPlayManager
 ├── OpponentPool          (self.opponent_pool)
 ├── BattleSharedPayoff    (self.payoff)
 └── OpponentSelector      (self.selector, 内部创建)
      └── 引用 self.payoff
```

`OpponentSelector` 由 `SelfPlayManager.__init__()` 内部创建，将 `payoff` 引用传给 selector，确保选择策略基于实时评分数据。

### 9.5.3 对手生命周期

完整的对手管理流程通过以下方法串联：

**1. 进度同步**

```python
def update_progress(self, episode: int) -> None:
    self._current_episode = episode
    self.selector.update_progress(episode)
```

（[selfplay.py:L49-L52](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L49-L52)）

每轮训练循环开始时调用，将当前 episode 传递给 `OpponentSelector`，触发自适应 exploit_prob 更新。

**2. 选择对手**

```python
def select_opponent(self) -> Optional[str]:
    if len(self.opponent_pool) == 0:
        return None
    opponent_id = self.selector.select(
        player_id=self.current_player_id,
        opponent_candidates=self.opponent_pool.get_all(),
    )
    return opponent_id
```

（[selfplay.py:L54-L62](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L54-L62)）

从 `OpponentPool` 获取所有候选，委托 `OpponentSelector.select()` 做出选择。如果池为空则返回 `None`（此时训练循环应跳过本局）。

**3. 添加新对手**

```python
def add_opponent(self, checkpoint_path: str) -> str:
    opponent_id = f"opp_{uuid.uuid4().hex[:8]}"
    self.opponent_pool.add(opponent_id, checkpoint_path)
    self.payoff.add_player(opponent_id)
    return opponent_id
```

（[selfplay.py:L64-L77](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L64-L77)）

每次将当前 agent 的快照（checkpoint）添加为新对手时：
1. 生成唯一 ID（`opp_` + 8 位 UUID 十六进制）；
2. 加入 `OpponentPool`（可能触发淘汰）；
3. 在 `BattleSharedPayoff` 中注册。

**4. 更新对战记录和评分**

```python
def update_payoff(self, opponent_id: str, result: int, rounds: int = 0) -> None:
    self.payoff.update(home=self.current_player_id, away=opponent_id, result=result)
    self.opponent_pool.increment_games(opponent_id)
    # 更新内部统计...
```

（[selfplay.py:L79-L109](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L79-L109)）

每局对战结束后调用：
- `payoff.update()`：更新双向战绩 + TrueSkill 评分
- `opponent_pool.increment_games()`：增加对手的对战场次计数
- 内部统计字典 `games_against_opponents` 同步更新

### 9.5.4 统计信息聚合

`get_statistics()` 返回一个综合统计字典（[selfplay.py:L125-L137](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L125-L137)）：

```python
{
    'num_opponents': len(self.opponent_pool),
    'games_against_opponents': {...},     # 各对手的胜/负/平/回合统计
    'exploitability': float,              # 可利用性分数
    'avg_rounds': float,                  # 平均每局回合数
    'win_rates': {opp_id: win_rate, ...}  # 对各对手的贝叶斯胜率
}
```

### 9.5.5 完整对战循环流程

在 `SelfPlayTrainer.train()` 中的典型调用顺序：

```
for episode in range(total_episodes):
    manager.update_progress(episode)           # 1. 同步进度

    opponent_id = manager.select_opponent()    # 2. 选择对手
    result, rounds = collect_episode(...)       # 3. 执行对战
    manager.update_payoff(opponent_id, result) # 4. 更新记录和评分

    if episode % opponent_update_interval == 0:
        manager.add_opponent(checkpoint)       # 5. 定期添加快照为对手

    payoff.decay_all()                         # 6. 历史衰减
```

---

## 9.6 核心参数速查

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `exploit_prob` | `0.7` | `OpponentSelector` | 基础 exploit 概率 |
| `explore_prob` | `0.3` | `OpponentSelector` | 基础 explore 概率 |
| `adaptive` | `True` | `OpponentSelector` | 是否开启自适应调度 |
| `adaptive exploit_prob` 范围 | `[0.3, 0.9]` | `OpponentSelector._get_adaptive_exploit_prob()` | 线性增长 |
| `progress` 参考值 | `10000 episodes` | `OpponentSelector._get_adaptive_exploit_prob()` | episode/10000 映射到 [0,1] |
| `_min_confidence_sigma` | `5.0` | `OpponentSelector` | confident 判定阈值 |
| `max_size` | `10` | `OpponentPool` | 池最大容量 |
| `min_games_threshold` | `10` | `OpponentPool` | 淘汰候选的最小场次 |
| `protection_bonus` | `(10-games)×0.5` | `OpponentPool._evict_worst()` | 低场次保护分 |
| `decay` | `0.99` | `BattleSharedPayoff` | 历史衰减因子 |
| `min_win_rate_games` | `8` | `BattleSharedPayoff` | 胜率置信权重阈值 |
| `TrueSkillRating.mu` | `25.0` | `TrueSkillRating` | 实力均值初始值 |
| `TrueSkillRating.sigma` | `8.33` | `TrueSkillRating` | 不确定性初始值 |
| `current_player_id` | `"current_agent"` | `SelfPlayManager` | 当前 agent 在 payoff 中的标识 |
| `TrueSkillRating.is_confident` | `sigma < 3.0` | `TrueSkillRating` | 该数据类的自信阈值（比 selector 的 5.0 更严格） |

---

> **第 9 章结束。** 第 10 章将介绍日志与报告系统。
