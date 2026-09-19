# 对手池管理模块代码审查报告

## 审查概要
- 审查文件数: 5
- 发现问题数: 9
- CRITICAL: 1 | HIGH: 1 | MEDIUM: 4 | LOW: 3

## 问题清单

### Issue 1: load_state 加载 TrueSkillRating 时构造函数参数不匹配导致崩溃
- **严重程度**: CRITICAL
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/battle_shared_payoff.py
- **行号**: 215
- **问题描述**: `load_state` 中使用 `TrueSkillRating(**r)` 恢复评分对象，其中 `r = {"mu": ..., "sigma": ..., "games_played": ...}`。但 `TrueSkillRating.__init__` 的签名是 `def __init__(self, mu: float = 25.0, sigma: float = 8.33)`，**不接受 `games_played` 关键字参数**。调用时会抛出 `TypeError: __init__() got an unexpected keyword argument 'games_played'`，导致状态加载完全失败。
- **修复建议**: 在 `TrueSkillRating.__init__` 中增加 `games_played` 参数：`def __init__(self, mu: float = 25.0, sigma: float = 8.33, games_played: int = 0)`，并在方法体中赋值 `self.games_played = games_played`。或者在 `load_state` 中改为手动构造：
  ```python
  self._trueskill_ratings = {}
  for pid, r in state["trueskill_ratings"].items():
      rating = TrueSkillRating(mu=r["mu"], sigma=r["sigma"])
      rating.games_played = r["games_played"]
      self._trueskill_ratings[pid] = rating
  ```

---

### Issue 2: 淘汰保护奖励在正常淘汰路径中是死代码，保护机制名存实亡
- **严重程度**: HIGH
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_pool.py
- **行号**: 60-77
- **问题描述**: `_evict_worst` 方法的逻辑分两步：(1) 先将 `games < min_games_threshold` 的对手从 `evictable` 列表中排除（第60-62行）；(2) 如果 `evictable` 为空，才回退到全部对手。而 `eviction_score` 中的 `protection_bonus`（第73-76行）是为 `games < threshold` 的对手提供额外保护分，使其更难被淘汰。然而在正常路径下，`evictable` 列表中的对手都满足 `games >= threshold`，因此 `protection = max(0, threshold - games) * 0.5` 恒为 0。保护奖励只在全部对手都低于阈值（退化为 `evictable = self._opponents`）的罕见情况下才生效。代码注释和设计意图与实际行为严重不符——预期的"渐进式保护"实际上变成了"硬截断式排除"，`_EVICTION_PROTECTION_FACTOR` 在正常路径下完全无意义。
- **修复建议**: 修改淘汰逻辑，不将低场次对手排除出 `evictable`，而是让保护奖励在所有对手中生效：
  ```python
  evictable = list(self._opponents)  # 不排除任何对手，让 protection_bonus 自然保护
  ```

---

### Issue 3: decay_all 使用 int() 截断破坏 games == wins + draws + losses 不变量
- **严重程度**: MEDIUM
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/battle_shared_payoff.py
- **行号**: 119-124
- **问题描述**: `decay_all` 对 `wins`、`draws`、`losses`、`games` 分别独立应用 `int(value * decay)` 截断。由于各字段截断误差不同，衰减后 `games` 可能大于 `wins + draws + losses`，破坏数据一致性。例如：`wins=3, draws=1, losses=1, games=5` 经 decay=0.99 后变为 `wins=int(2.97)=2, draws=int(0.99)=0, losses=int(0.99)=0, games=int(4.95)=4`，此时 `wins+draws+losses=2 ≠ games=4`。这会导致 `get_win_rate` 中的 `confidence_weight` 基于虚高的 `games` 值，使胜率估计过度自信（过度偏离0.5）。
- **修复建议**: 先衰减 `wins`、`draws`、`losses`，然后令 `games = wins + draws + losses`，确保不变量成立：
  ```python
  d["wins"] = int(d["wins"] * self._decay)
  d["draws"] = int(d["draws"] * self._decay)
  d["losses"] = int(d["losses"] * self._decay)
  d["games"] = d["wins"] + d["draws"] + d["losses"]
  ```
  或使用 `round()` 替代 `int()` 减少截断误差。

---

### Issue 4: 被淘汰对手的 payoff 数据未清理，导致内存无限增长
- **严重程度**: MEDIUM
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_pool.py
- **行号**: 80-84
- **问题描述**: `_evict_worst` 淘汰对手时，只清理了 `_opponents`、`_checkpoint_paths`、`_games_played`、`_added_episodes` 中的数据，但该对手在 `BattleSharedPayoff` 中的 TrueSkill 评分（`_trueskill_ratings`）和对战记录（`_data`）完全未清理。长期训练中，对手池不断淘汰旧对手并添加新对手，payoff 中的历史数据会持续积累，导致内存无限增长。此外，`get_exploitability` 和 `get_payoff_matrix` 等方法遍历 `_players` 列表时也会包含已淘汰的对手，可能导致计算结果偏差。
- **修复建议**: 在 `OpponentPool._evict_worst` 中淘汰对手后，同步调用 payoff 清理接口（需在 `BattleSharedPayoff` 中新增 `remove_player` 方法），或在 `_evict_worst` 中提供回调机制通知 payoff 清理。

---

### Issue 5: sigmoid 调度策略未使用调度常量，与其他策略行为不一致
- **严重程度**: MEDIUM
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_selector.py
- **行号**: 128-130
- **问题描述**: `_compute_exploit_prob` 中，`linear` 和 `exp` 调度策略都使用 `EXPLORE_SCHEDULE_MIN=0.3` 和 `EXPLORE_SCHEDULE_RANGE=0.6` 常量，使利用概率范围约为 [0.3, 0.9]。但 `sigmoid` 策略直接使用 `1/(1+exp(-x))`，其范围约为 [0.007, 0.993]，完全忽略了这两个常量。这导致：(1) sigmoid 策略在训练初期几乎完全探索（p_effective≈0.007），训练后期几乎完全利用（p_effective≈0.993），与 linear/exp 策略的 30%-90% 范围截然不同；(2) 定义常量却不在所有策略中使用，代码意图不明确。
- **修复建议**: 将 sigmoid 映射到相同的 [MIN, MIN+RANGE] 区间：
  ```python
  elif self._schedule == "sigmoid":
      x = (p - 0.5) * 10
      raw = 1.0 / (1.0 + math.exp(-x))
      p_effective = EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE * raw
  ```

---

### Issue 6: decay_all 仅衰减原始胜负计数，不衰减 TrueSkill 评分，导致两种评估指标随时间发散
- **严重程度**: MEDIUM
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/battle_shared_payoff.py
- **行号**: 117-124
- **问题描述**: `decay_all` 仅对 `_data` 中的原始胜/负/平/场次计数应用衰减，但 `_trueskill_ratings` 中的 TrueSkill 评分（mu 和 sigma）完全不受影响。随着训练进行，被衰减的原始数据计算的 `get_win_rate` 会逐渐偏向0.5（因计数趋零），而 TrueSkill 的 mu 值保持不变。这意味着：(1) 对手选择器的 `_exploit_select`（基于 mu）和 `_explore_select`（基于 sigma）使用的评估标准与 `get_win_rate` 的评估标准随时间严重脱节；(2) `get_exploitability` 混合使用了 TrueSkill mu（未衰减）和 count-based confidence weight（已衰减），语义不一致。
- **修复建议**: 考虑同步对 TrueSkill 评分应用衰减（如将 sigma 适当增大以反映不确定性增加），或重新评估是否需要 `decay_all` 机制——TrueSkill 本身通过新对局自然更新评分，可能不需要外部衰减。

---

### Issue 7: 类文档注释称 exploit 策略选择"胜率最低"的对手，但实际实现按 TrueSkill mu 最高选择
- **严重程度**: LOW
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/opponent_selector.py
- **行号**: 21-22
- **问题描述**: 类文档注释写道"利用 (Exploit): 选择胜率最低（最有挑战性）的可信对手"，但 `_exploit_select` 方法实际按 TrueSkill `mu` 降序排序取最强对手。TrueSkill mu 是绝对实力评估，而"胜率最低"是相对概念——一个高 mu 的对手不一定是当前玩家胜率最低的对手（例如当前玩家可能已找到针对该对手的策略）。两者语义不同，文档与实现不一致。
- **修复建议**: 修改文档注释使其与实现一致，如改为"利用 (Exploit): 选择 TrueSkill 评分最高（绝对实力最强）的可信对手"。

---

### Issue 8: update 方法未防止 home == away 导致同一条记录被重复更新
- **严重程度**: LOW
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/battle_shared_payoff.py
- **行号**: 36
- **问题描述**: `update(home, away, result)` 方法用 `f"{home}-{away}"` 和 `f"{away}-{home}"` 分别更新正反两条记录。当 `home == away` 时，两个 key 相同，同一条记录被更新两次。例如 `update("A", "A", 1)` 会导致 wins+1、losses+1、games+2，而不是预期的逻辑。虽然在自对弈场景中 `home == away` 不太可能出现，但缺少防御性检查，可能在误用时产生难以排查的数据污染。
- **修复建议**: 在 `update` 开头添加断言或早期返回：
  ```python
  if home == away:
      logger.warning(f"update called with home == away ({home}), skipping")
      return
  ```

---

### Issue 9: _players_ids 字典保存/加载了索引值但从未被实际查询使用
- **严重程度**: LOW
- **文件**: /Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/league/battle_shared_payoff.py
- **行号**: 23, 31-33, 42-49, 191, 211
- **问题描述**: `_players_ids` 字典将玩家 ID 映射到其在 `_players` 列表中的索引位置，但在整个类中只用于 `if player_id not in self._players_ids` 的成员检查，从未通过索引值进行数据查找。该字典完全可以简化为 `set`。此外，`save_state`/`load_state` 保存和恢复了索引映射，但如果 `_players` 列表顺序发生变化，索引值会失效。当前因为 `load_state` 同时恢复两者所以不会出错，但存在潜在的一致性风险。
- **修复建议**: 将 `_players_ids` 改为 `set[str]` 类型，仅用于成员检查；在 `save_state`/`load_state` 中移除对该字段的序列化，或保留为 `Dict[str, int]` 但在加载后重建索引以确保一致性。
