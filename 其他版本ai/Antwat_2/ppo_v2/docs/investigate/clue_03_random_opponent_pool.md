# 线索 03：对手池初始全为随机策略，早期训练信号噪声大

## 结论：**确认**

本问题已通过代码审查确认。对手池初始化创建 3 个随机初始化策略，导致早期训练信号质量极差，且对手替换时强度突变。

---

## 1. 初始对手数量和类型

### 代码位置：`selfplay.py` 第 92 行、第 719-762 行

```python
# selfplay.py:92
INITIAL_OPPONENT_COUNT = 3

# selfplay.py:719-762
def _create_initial_opponents(self) -> None:
    """创建初始对手池，避免前 100 个 episode 无有效对手训练。"""
    if self._selfplay_manager is None:
        return

    opponent_count = self._selfplay_manager.get_opponent_count()
    if opponent_count > 0:
        logger.info(
            f"Opponent pool already has {opponent_count} opponents, skipping initial creation"
        )
        return

    logger.info(
        f"Creating {self.INITIAL_OPPONENT_COUNT} initial random opponents..."
    )
    checkpoint_dir = (
        self._path_config.get_selfplay_dir(self._run_id) / "opponent_checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for i in range(self.INITIAL_OPPONENT_COUNT):
        try:
            random_policy = AntWarPolicyValueNetwork(
                hidden_dim=self._config.get("network", {}).get("hidden_dim"),
                enable_auxiliary=self._config.get("ppo", {}).get("enable_auxiliary"),
            )
            random_policy.to(self._trainer.device)

            checkpoint = {
                "policy_state_dict": random_policy.state_dict(),
                "config": self._config,
            }
            filename = f"initial_opponent_{uuid.uuid4().hex[:8]}.pt"
            checkpoint_path = str(checkpoint_dir / filename)
            torch.save(checkpoint, checkpoint_path)

            self._selfplay_manager.add_opponent(checkpoint_path, -1)
```

**关键发现**：
- 初始对手数量为 **3 个**（`INITIAL_OPPONENT_COUNT = 3`）
- 每个初始对手都是 `AntWarPolicyValueNetwork()` 的 **随机初始化实例**，未经任何训练
- 初始对手的 episode 标记为 **-1**，表示不属于任何训练阶段
- 只有当对手池为空时才创建（`opponent_count > 0` 时跳过）

---

## 2. opponent_update_interval 的值

### 代码位置：`ppo_antwar.yaml` 第 61 行

```yaml
# ppo_antwar.yaml:61
training:
  opponent_update_interval: 100
```

### 代码位置：`selfplay.py` 第 626-635 行

```python
# selfplay.py:626-635
def _run_periodic_tasks(self) -> None:
    save_frequency = self._config.get("training", {}).get("save_interval")
    opponent_update_frequency = self._config.get("training", {}).get(
        "opponent_update_interval"
    )
    # ...
    if self._episode_count % opponent_update_frequency == 0:
        self._update_opponent(self._episode_count)
```

**关键发现**：
- `opponent_update_interval = 100`，即每 **100 个 episode** 添加一个新对手
- 新对手是当前训练策略的快照（`_update_opponent` 保存当前 trainer 的 checkpoint）
- 总训练 episode 数为 2000，因此整个训练过程中会添加约 **20 个对手快照**

---

## 3. 对手选择策略（exploit vs explore）

### 代码位置：`opponent_selector.py` 第 45-58 行

```python
# opponent_selector.py:45-58
def select(self) -> Optional[str]:
    candidates = self._pool.get_all()
    if not candidates:
        return None

    if random.random() < self._compute_exploit_prob():
        return self._exploit_select(candidates)
    else:
        return self._explore_select(candidates)
```

### 代码位置：`opponent_selector.py` 第 60-97 行

```python
# opponent_selector.py:60-83
def _exploit_select(self, candidates: List[str]) -> str:
    """利用策略：筛选可信对手，按 mu 排序取最强"""
    confident = []
    for cid in candidates:
        rating = self._payoff._trueskill_ratings.get(cid, None)
        if rating is not None and rating.is_confident:
            confident.append(cid)

    if not confident:
        best = min(
            candidates,
            key=lambda cid: (
                self._payoff._trueskill_ratings.get(cid, TrueSkillRating()).sigma
            ),
        )
        return best

    confident_sorted = sorted(
        confident,
        key=lambda cid: self._payoff._trueskill_ratings[cid].mu,
        reverse=True,
    )
    top_k = min(self.EXPLOIT_TOP_K, len(confident_sorted))
    return confident_sorted[random.randint(0, top_k - 1)]

# opponent_selector.py:85-97
def _explore_select(self, candidates: List[str]) -> str:
    """探索策略：选择 sigma 较大（不确定性高）的对手"""
    high_sigma = [
        cid
        for cid in candidates
        if self._payoff._trueskill_ratings.get(cid, TrueSkillRating()).sigma
        > self._min_confidence_sigma
    ]

    if not high_sigma:
        high_sigma = candidates

    return random.choice(high_sigma)
```

### 代码位置：`opponent_selector.py` 第 122-138 行

```python
# opponent_selector.py:122-138
def _compute_exploit_prob(self) -> float:
    """根据当前进度和调度策略计算利用概率"""
    if not self._adaptive:
        return self._exploit_prob

    p = self._progress
    if self._schedule == "sigmoid":
        x = (p - 0.5) * 10
        p_effective = 1.0 / (1.0 + math.exp(-x))
    elif self._schedule == "exp":
        p_effective = EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE * (
            1.0 - math.exp(-EXP_DECAY_RATE * p)
        )
    else:
        p_effective = EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE * p

    return p_effective
```

### 代码位置：`ppo_antwar.yaml` 第 71 行

```yaml
selfplay:
  exploit_prob: 0.7
```

**关键发现**：
- 配置 `exploit_prob: 0.7`，但由于 `adaptive=True`（默认），实际利用概率由 `_compute_exploit_prob()` 动态计算
- 默认 schedule 为 `"linear"`，公式为 `0.3 + 0.6 * progress`
- 训练初期（progress ≈ 0），exploit_prob ≈ 0.3，即 **70% 的时间走探索路径**
- 探索策略优先选择 sigma 大（不确定性高）的对手，**初始随机对手 sigma=8.33（默认值），恰好是最不确定的**
- 因此训练初期，**初始随机对手被选中的概率极高**

---

## 4. 初始对手何时会被淘汰

### 代码位置：`opponent_pool.py` 第 48-85 行

```python
# opponent_pool.py:48-85
def _evict_worst(self) -> str:
    """淘汰最差对手

    淘汰评分 = mu + protection_bonus
    protection_bonus = max(0, min_games_threshold - games) * 0.5
    """
    if not self._opponents:
        return ""

    evictable = []
    for opp_id in self._opponents:
        games = self._games_played.get(opp_id, 0)
        if games < self.min_games_threshold:
            continue
        evictable.append(opp_id)

    if not evictable:
        evictable = self._opponents

    def eviction_score(opp_id):
        mu = 25.0
        if self._payoff is not None:
            rating = self._payoff._trueskill_ratings.get(opp_id, TrueSkillRating())
            mu = rating.mu
        games = self._games_played.get(opp_id, 0)
        protection = (
            max(0, self.min_games_threshold - games)
            * self._EVICTION_PROTECTION_FACTOR
        )
        return mu + protection

    victim = min(evictable, key=eviction_score)
    # ... remove victim
```

### 代码位置：`ppo_antwar.yaml` 第 69-70 行

```yaml
selfplay:
  opponent_pool_size: 10
  min_opponent_games: 8
```

**关键发现**：
- 对手池最大容量为 **10**，初始有 3 个随机对手
- 池满时（达到 10 个）触发淘汰，淘汰评分最低的对手
- `min_opponent_games = 8`：对战场次 < 8 的对手受保护（不会被优先淘汰）
- 初始对手的 episode=-1，它们会持续被选中参与对战

**淘汰时间线推算**：
- 初始 3 个对手，池满需 10 个，即需添加 7 个新对手
- 每 100 episode 添加 1 个，第 700 episode 时池满
- 但在此之前，初始对手的对战场次会逐渐达到 8 场以上，失去保护
- 初始随机对手的 TrueSkill mu 会很低（因为总是输），eviction_score 最低
- **预计在第 700-1000 episode 区间，初始随机对手会被逐步淘汰**

**问题**：前 700 个 episode（占总训练的 35%）中，对手池始终包含 1-3 个随机策略对手。

---

## 5. 早期训练信号的噪声程度评估

### 5.1 随机策略的行为特征

`AntWarPolicyValueNetwork` 随机初始化后，其策略输出为近似均匀分布。结合 `get_action()` 的实现：

```python
# ant_war_policy_value_network.py:154-158
explore = (
    not deterministic
    and exploration_epsilon > 0
    and torch.rand(1).item() < exploration_epsilon
)
```

`OpponentAgent` 创建时 `exploration_epsilon=0.0`（见 `opponent_agent.py:30`），因此不走 epsilon-greedy 探索路径，而是完全依赖 softmax 采样。

随机初始化网络的 softmax 输出近似均匀分布，9 种动作类型的概率约为 1/9 ≈ 11.1%。而 `TYPE_CONFIG` 显示：

```
noop:             flat_start=0,   flat_end=1,   max_targets=1
build_tower:      flat_start=1,   flat_end=17,  max_targets=16
upgrade_tower:    flat_start=17,  flat_end=81,  max_targets=64
downgrade_tower:  flat_start=81,  flat_end=97,  max_targets=16
lightning_storm:  flat_start=97,  flat_end=102, max_targets=5
emp_blaster:      flat_start=102, flat_end=107, max_targets=5
deflector:        flat_start=107, flat_end=112, max_targets=5
evasion:          flat_start=112, flat_end=117, max_targets=5
tech_upgrade:     flat_start=117, flat_end=119, max_targets=2
```

随机策略在均匀分布下，noop 的概率约 1/9 ≈ 11%，但 **build_tower、upgrade_tower、downgrade_tower 等动作占比更高**（因为 target 维度更大），且大量动作会因 action_mask 被屏蔽，实际 noop 概率可能更高。

### 5.2 噪声信号分析

当对手为随机策略时：

1. **对手几乎不执行有效操作**：随机初始化的网络不会产生有意义的建造/升级/技能释放序列，大量动作被 action_mask 屏蔽后回退为 noop
2. **reward 信号由环境固有属性主导**：
   - `noop_base_penalty: -0.0025`、`noop_max_penalty: -0.005`：对手 noop 会导致其持续受到小惩罚
   - `tower_survival_per_tower: 0.02`：双方都不建塔时此项为 0
   - `win_reward: 500.0`、`loss_reward: -500.0`：终局奖励占绝对主导
3. **虚假的高胜率**：当前策略（即使是早期未训练好的策略）对随机对手几乎必胜，产生大量 +500 的 reward，**掩盖了策略本身的弱点**
4. **value function 被误导**：value head 学习到"对随机对手 = 高价值"的错误映射，当对手变为真实策略时，value 预测严重偏离

### 5.3 对手替换时的强度突变

- 第 100 episode 时，第一个真实策略快照（`opponent_ep100`）加入对手池
- 此时当前策略已经过 100 episode 对随机对手的训练，可能已经过拟合到"打随机对手"的模式
- 真实策略对手的出现导致 **WinRate 从接近 100% 突然下降**
- 由于 `opponent_update_interval=100`，每 100 episode 才添加一个新对手，对手池的强度变化是 **阶梯式** 的，不是渐进的

### 5.4 定量估算

| 阶段 | Episode 范围 | 对手池构成 | 随机对手占比 | 信号质量 |
|------|-------------|-----------|------------|---------|
| 初始化 | 1-100 | 3 个随机对手 | 100% | 极差 |
| 早期 | 101-300 | 3 随机 + 1-2 真实 | 60%-100% | 差 |
| 中期 | 301-700 | 3 随机 + 3-7 真实 | 30%-43% | 中等 |
| 后期 | 701-2000 | 0-1 随机 + 9-10 真实 | 0%-10% | 较好 |

**前 300 个 episode（占总训练 15%）中，随机对手被选中概率超过 60%**，这段时间的训练信号几乎无效。

---

## 6. 修复建议

### 方案 A：使用规则 AI 作为初始对手（推荐）

用已有的 `BasicRandomAI`、`BasicTowerAI`、`MediumRuleAI` 替代随机初始化策略作为初始对手。规则 AI 至少能执行有意义的操作（建塔、升级等），提供更有信息量的训练信号。

```python
def _create_initial_opponents(self) -> None:
    """使用规则 AI 创建初始对手，而非随机策略。"""
    # 选项 1：将规则 AI 的对战数据作为初始对手
    # 选项 2：用预训练几个 epoch 的策略作为初始对手
    pass
```

### 方案 B：预训练初始对手

在正式自对弈训练前，用规则 AI 作为对手进行少量预训练（如 50-100 episode），然后将预训练后的策略快照作为初始对手。这样初始对手至少具备基本的游戏理解。

### 方案 C：缩短 opponent_update_interval

将 `opponent_update_interval` 从 100 减小到 20-50，使真实策略更快进入对手池，缩短随机对手主导期。

### 方案 D：初始对手快速淘汰机制

为初始随机对手设置特殊标记（如 episode=-1），在对手池达到一定数量时（如 3 个真实对手后）优先淘汰它们，而不依赖正常的 eviction 逻辑。

### 方案 E：混合对手池初始化（推荐，结合 A+C）

1. 初始对手池使用 1 个规则 AI + 2 个随机策略（而非 3 个纯随机）
2. 将 `opponent_update_interval` 从 100 降至 50
3. 为初始随机对手添加 `is_initial=True` 标记，当真实对手数量 ≥ 3 时优先淘汰

---

## 7. 总结

| 维度 | 现状 | 影响 |
|------|------|------|
| 初始对手数量 | 3 个 | 对手池小，多样性不足 |
| 初始对手类型 | 随机初始化策略 | 行为近似均匀随机，noop 占比高 |
| 对手更新间隔 | 100 episode | 真实策略进入池的速度慢 |
| 对手选择策略 | 初期 70% 探索，优先选 sigma 高的对手 | 随机对手 sigma=8.33，被优先选中 |
| 初始对手淘汰 | 约 700-1000 episode | 前 35% 训练时间受随机对手影响 |
| 早期信号质量 | 极差 | reward 主要由环境属性决定，非策略行为 |

**问题确认**：对手池初始全为随机策略，导致早期训练信号噪声大，且对手替换时强度突变，WinRate 大幅波动。建议采用方案 E（混合初始化 + 缩短更新间隔 + 快速淘汰）进行修复。
