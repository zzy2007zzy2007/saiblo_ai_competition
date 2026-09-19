# SelfPlay对手池管理设计文档

## 一、概述

SelfPlay对手池管理是ppo_v1中用于训练AI对战能力的关键组件。通过让AI与自身的历史版本对战，不断提升对战水平。

**核心设计目标：**
1. 积累多样化的对手历史版本
2. 通过TrueSkill评分系统量化对手水平
3. 智能选择对手，平衡探索与利用
4. 支持训练中断恢复

---

## 二、核心组件

### 2.1 TrueSkillRating（评分系统）

位置：`league/payoff.py`

**数据结构：**
```python
@dataclass
class TrueSkillRating:
    mu: float = 25.0          # 技能水平，初始值25
    sigma: float = 8.33       # 不确定性，初始值8.33
    games_played: int = 0     # 对战场次

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return (self.mu - 3 * self.sigma, self.mu + 3 * self.sigma)

    @property
    def is_confident(self) -> bool:
        return self.sigma < 3.0  # sigma < 3表示评分可信
```

**说明：**
- mu值越高表示对手越强
- sigma值越低表示评分越确定
- 每次对战后，根据结果使用TrueSkill算法更新评分

---

### 2.2 BattleSharedPayoff（Payoff记录）

位置：`league/payoff.py`

**功能：**
- 双向记录对战结果（home-away和away-home）
- 管理所有玩家的TrueSkill评分
- 提供胜率计算接口

**关键方法：**

| 方法 | 说明 |
|------|------|
| `update(home, away, result)` | 更新对战结果，同时更新双向记录和TrueSkill评分 |
| `get_win_rate(home, away)` | 获取贝叶斯胜率估计 |
| `decay_all()` | 对所有历史记录执行衰减（乘以decay系数） |
| `get_exploitability()` | 计算可利用性指标 |
| `save_state()` | 保存状态到JSON文件 |
| `load_state()` | 从JSON文件加载状态 |

**胜率计算（贝叶斯估计）：**
```python
def get_win_rate(self, home: str, away: str) -> float:
    # 使用Beta先验的贝叶斯估计
    alpha = 1 + record['wins'] + record['draws'] * 0.5
    beta = 1 + record['losses'] + record['draws'] * 0.5
    win_rate = alpha / (alpha + beta)

    # 平滑过渡：随样本增加从0.5趋近真实胜率
    confidence_weight = min(1.0, games / self._min_win_rate_games)
    win_rate = 0.5 + confidence_weight * (win_rate - 0.5)
    return win_rate
```

---

### 2.3 OpponentPool（对手池）

位置：`league/opponent_selector.py`

**功能：**
- 管理固定大小的对手列表（默认max_size=10）
- 使用UUID生成唯一对手ID
- 智能淘汰策略：保护有潜力的新手，淘汰评分低的老手
- 支持对手checkpoint路径管理
- 支持持久化

**智能淘汰策略：**
```python
def _evict_worst(self) -> None:
    # 候选对手：满足最小对战场次要求
    candidates = [
        opp_id for opp_id in self._opponents
        if self._games_played.get(opp_id, 0) >= self.min_games_threshold
    ]

    if not candidates:
        victim = self._opponents[0]  # FIFO兜底
    else:
        # 淘汰分数 = mu - protection_bonus
        # protection_bonus保护对战场次少的对手
        victim = min(candidates, key=eviction_score)

    self._remove_opponent(victim)
```

**UUID对手ID：**
```python
opponent_id = f"opp_{uuid.uuid4().hex[:8]}"
```

---

### 2.4 OpponentSelector（对手选择器）

位置：`league/opponent_selector.py`

**功能：**
- 平衡探索（explore）与利用（exploit）
- 支持动态调整探索/利用比例
- 选择评分可信且最强的对手进行"利用"

**探索/利用策略：**

| 模式 | 说明 |
|------|------|
| **利用（Exploit）** | 选择TrueSkill评分最高且可信的对手 |
| **探索（Explore）** | 选择评分不确定的对手，增加多样性 |

**动态探索/利用比例（adaptive=True）：**

```python
def _get_adaptive_exploit_prob(self) -> float:
    progress = min(1.0, self._total_episodes / 10000.0)

    if self.schedule == "linear":
        # 线性增长：0.3 -> 0.9
        return 0.3 + 0.6 * progress
    elif self.schedule == "sigmoid":
        # S型增长：初期慢，中期快，后期饱和
        return 0.3 + 0.6 * (1 / (1 + math.exp(-10 * (progress - 0.5))))
    elif self.schedule == "exp":
        # 指数增长：快速趋近，后期稳定
        return 0.3 + 0.6 * (1 - math.exp(-5 * progress))
```

**调度曲线：**
```
linear:     0.3 ──────────────────────────► 0.9
sigmoid:    0.3 ┐                          ┌ 0.9
               ┌┘                          └┐
               └────────────────────────────┘
exp:       0.3 ┌────┐
               │    └────────────────────────► 0.9
```

---

## 三、SelfPlayManager（SelfPlay管理器）

位置：`trainer/selfplay.py`

**功能：**
- 协调对手池、Payoff和选择器
- 管理对战流程
- 收集对战统计信息

**关键方法：**

| 方法 | 说明 |
|------|------|
| `add_opponent(checkpoint_path)` | 添加新对手，使用UUID生成ID |
| `update_payoff(opponent_id, result)` | 更新对战结果 |
| `select_opponent()` | 选择对手 |
| `get_statistics()` | 获取统计信息 |
| `update_progress(episode)` | 更新训练进度，用于动态比例调整 |

---

## 四、SelfPlayTrainer（训练器集成）

位置：`trainer/selfplay.py`

**功能：**
- 集成对手池管理与训练流程
- 支持初始对手生成
- 支持状态持久化
- 支持历史衰减

### 4.1 训练流程

```python
def train(self, num_episodes, opponent_update_interval=500):
    for episode in range(num_episodes):
        # 1. 更新训练进度（用于动态探索/利用调整）
        self.selfplay_manager.update_progress(episode)

        # 2. 选择对手
        opponent_id = self.selfplay_manager.select_opponent()

        # 3. 收集对战数据
        batch, battle_details = self._collect_episode_with_swap(...)

        # 4. 更新Payoff
        self.selfplay_manager.update_payoff(opponent_id, result)

        # 5. 执行PPO更新
        self.trainer._ppo_update(batch)

        # 6. 定期保存checkpoint和league状态
        if episode % opponent_update_interval == 0 and episode > 0:
            checkpoint_path = self.trainer.save_checkpoint(...)
            self.selfplay_manager.add_opponent(checkpoint_path)
            self._save_league_state()

        # 7. 定期执行历史衰减
        self._maybe_decay(episode)
```

### 4.2 初始对手生成

```python
def _create_initial_opponents(self) -> None:
    """训练开始时自动创建3个随机策略的初始对手"""
    if self.initial_opponents <= 0:
        return

    for i in range(self.initial_opponents):
        checkpoint_path = self._create_random_agent_checkpoint()
        new_opponent_id = self.selfplay_manager.add_opponent(checkpoint_path)
        self._opponent_checkpoints[new_opponent_id] = checkpoint_path
```

### 4.3 持久化机制

**保存时机：** 与checkpoint保存同步（`episode % opponent_update_interval == 0`）

**文件结构：**
```
checkpoints/
├── model_1000.pt
├── model_2000.pt
└── league_state/
    ├── opponent_pool.json   # 对手池状态
    ├── payoff.json          # Payoff记录
    └── checkpoint_map.json   # checkpoint路径映射
```

**加载逻辑：**
```python
def __init__(self, ...):
    # ...
    self._load_league_state()

    # 只有当没有加载到任何对手时，才创建初始对手
    if len(self.opponent_pool) == 0:
        self._create_initial_opponents()
```

### 4.4 历史衰减

```python
def _maybe_decay(self, episode: int) -> None:
    """每decay_interval个episode执行一次衰减"""
    if episode > 0 and episode % self.decay_interval == 0:
        self.payoff.decay_all()
```

---

## 五、配置参数汇总

### 5.1 SelfPlayTrainer

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `opponent_pool_size` | 10 | 对手池最大容量 |
| `min_opponent_games` | 8 | 最小对战场次（用于胜率门槛） |
| `exploit_prob` | 0.7 | 基础利用概率 |
| `explore_prob` | 0.3 | 基础探索概率 |
| `decay_interval` | 1000 | 历史衰减间隔 |
| `initial_opponents` | 3 | 初始对手数量 |
| `checkpoint_dir` | "./checkpoints" | checkpoint保存目录 |

### 5.2 OpponentSelector

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adaptive` | True | 是否启用动态调整 |
| `schedule` | "linear" | 调度策略：linear/sigmoid/exp |

### 5.3 OpponentPool

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_size` | 10 | 对手池最大容量 |
| `min_games_threshold` | 10 | 淘汰保护最小对战场次 |

---

## 六、已解决的问题

| 问题 | 解决方案 |
|------|----------|
| 历史衰减未使用 | 在训练循环中定期调用`decay_all()` |
| FIFO淘汰不合理 | 实现智能淘汰策略，保护新手 |
| 探索/利用比例固定 | 支持动态调整（linear/sigmoid/exp） |
| 对手ID可能重复 | 使用UUID生成唯一ID |
| 胜率门槛矛盾 | 使用贝叶斯估计，平滑过渡 |
| 对手池初始为空 | 自动创建3个随机初始对手 |
| 持久化机制缺失 | 保存/加载league状态（JSON格式） |

---

## 七、数据流图

```
训练开始
    │
    ▼
加载持久化状态（如有）
    │
    ▼
创建初始对手（如果池为空）
    │
    ▼
┌─────────────────────────────────┐
│        训练循环 (episode)         │
├─────────────────────────────────┤
│ 1. update_progress(episode)     │
│ 2. select_opponent()             │
│ 3. 收集对战数据                  │
│ 4. update_payoff()               │
│ 5. PPO更新                      │
│ 6. 定期：保存checkpoint+状态     │
│ 7. 定期：decay_all()            │
└─────────────────────────────────┘
    │
    ▼
训练结束，保存最终状态
```

---

## 八、未来改进方向

1. **对手风格多样性**：根据对手风格特征进行多样化选择
2. **分布式训练支持**：支持多进程/多机器分布式训练
3. **更复杂的淘汰策略**：考虑对手的"潜力"和"多样性"
4. **对手分组**：按风格/水平分组，选择时考虑分组平衡