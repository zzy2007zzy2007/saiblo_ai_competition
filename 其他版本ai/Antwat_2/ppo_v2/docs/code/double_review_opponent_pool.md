# 对手池管理模块二次审查评估报告

## 评估概要
- 评估问题数: 9
- 确认存在: 9 | 级别调整: 3 | 建议修正: 3

## 逐项评估

### Issue 1: load_state 加载 TrueSkillRating 时构造函数参数不匹配导致崩溃
- **真实性**: 确认存在
- **源码验证**: `TrueSkillRating.__init__` 签名为 `def __init__(self, mu: float = 25.0, sigma: float = 8.33)`（trueskill_rating.py 第14行），不接受 `games_played` 参数。`save_state` 保存 `{"mu": r.mu, "sigma": r.sigma, "games_played": r.games_played}`（battle_shared_payoff.py 第195行），`load_state` 使用 `TrueSkillRating(**r)` 恢复（第215行），会传入 `games_played` 关键字参数，必定触发 `TypeError`。`load_state` 完全不可用。
- **级别评估**: CRITICAL → CRITICAL（级别准确，状态加载完全失败是致命缺陷）
- **影响评估**: 准确。训练恢复/断点续训完全不可用。
- **建议评估**: 原建议给出了两个方案，第一个（修改 `__init__` 签名增加 `games_played` 参数）更简洁直接，符合"从根源解决"原则。第二个方案（手动构造）是 workaround，不符合项目原则。
- **详细修复方案**: 修改 `TrueSkillRating.__init__` 签名，增加 `games_played` 参数：

  文件 `trueskill_rating.py` 第14行：
  ```python
  # 修改前
  def __init__(self, mu: float = 25.0, sigma: float = 8.33):
      self.mu = mu
      self.sigma = sigma
      self.games_played: int = 0

  # 修改后
  def __init__(self, mu: float = 25.0, sigma: float = 8.33, games_played: int = 0):
      self.mu = mu
      self.sigma = sigma
      self.games_played = games_played
  ```

  此修改后，`TrueSkillRating(**r)` 即可正常工作，因为 `r = {"mu": ..., "sigma": ..., "games_played": ...}` 中的三个参数都有了对应的构造函数参数。`from_trueskill` 类方法（第31行）也无需修改，因为它通过 `cls(mu=..., sigma=...)` 构造后手动赋值 `games_played`，行为不受影响。

- **修复代价**: 低
- **修复收益**: 高

---

### Issue 2: 淘汰保护奖励在正常淘汰路径中是死代码
- **真实性**: 确认存在
- **源码验证**: `_evict_worst` 方法（opponent_pool.py 第57-62行）先过滤掉 `games < min_games_threshold` 的对手，只保留 `games >= threshold` 的进入 `evictable` 列表。然后在 `eviction_score` 函数（第73-76行）中计算 `protection = max(0, self.min_games_threshold - games) * self._EVICTION_PROTECTION_FACTOR`。在正常路径下，`evictable` 中的对手 `games >= threshold`，所以 `max(0, threshold - games) = 0`，`protection` 恒为 0。`_EVICTION_PROTECTION_FACTOR = 0.5` 在正常路径下毫无作用。保护机制仅在所有对手都低于阈值（第64-65行回退到 `evictable = self._opponents`）时才生效。
- **级别评估**: HIGH → MEDIUM（降级。这不是崩溃或数据错误，而是逻辑设计上保护粒度不够细——硬排除代替了渐进保护。当前行为（硬排除低场次对手）实际上比渐进保护更保守，不会导致错误淘汰。但代码意图与实际行为不一致，`_EVICTION_PROTECTION_FACTOR` 常量在正常路径下名存实亡，确实需要修正。）
- **影响评估**: 部分准确。原报告称"保护机制名存实亡"过于绝对——硬排除本身就是一种保护，且在回退路径下保护奖励仍然生效。真正的问题是：保护只有"全有或全无"两种状态，缺乏中间过渡。
- **建议评估**: 原建议"移除硬排除，让 protection_bonus 在所有对手中生效"改变了核心语义：移除硬排除后，低场次对手可以被淘汰（只是更难被淘汰），可能导致新加入的对手尚未充分评估就被淘汰。更合理的做法是删除无用的 `protection_bonus` 计算代码，保留硬排除逻辑（因为硬排除本身就是更强的保护），使代码意图与行为一致。
- **详细修复方案**: 删除 `eviction_score` 中的 `protection_bonus` 计算，简化为仅按 `mu` 排序：

  文件 `opponent_pool.py` 第67-77行：
  ```python
  # 修改前
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

  # 修改后
  def eviction_score(opp_id):
      if self._payoff is not None:
          rating = self._payoff._trueskill_ratings.get(opp_id, TrueSkillRating())
          return rating.mu
      return 25.0
  ```

  同时删除类属性 `_EVICTION_PROTECTION_FACTOR = 0.5`（第15行），以及修改方法文档注释（第49-52行）移除 `protection_bonus` 的描述。保留第57-65行的硬排除逻辑不变。

- **修复代价**: 低
- **修复收益**: 中

---

### Issue 3: decay_all 使用 int() 截断破坏 games == wins + draws + losses 不变量
- **真实性**: 确认存在
- **源码验证**: `decay_all`（battle_shared_payoff.py 第117-124行）对四个字段独立执行 `int(value * decay)` 截断。以 decay=0.99 为例：`wins=3, draws=1, losses=1, games=5` → `wins=int(2.97)=2, draws=int(0.99)=0, losses=int(0.99)=0, games=int(4.95)=4`，此时 `wins+draws+losses=2 ≠ games=4`。不变量确实被破坏。`get_win_rate` 中 `confidence_weight = min(1.0, games / max(self._min_win_rate_games, 1))` 使用了虚高的 `games` 值，使胜率估计过度自信。
- **级别评估**: MEDIUM → MEDIUM（级别准确）
- **影响评估**: 准确。games 虚高导致 confidence_weight 偏大，win_rate 过度偏离 0.5。
- **建议评估**: 第一个建议（先衰减 wins/draws/losses，再令 games = 三者之和）完全正确，符合"从根源解决"原则。第二个建议（用 `round()` 替代 `int()`）只是减少误差而非消除，不彻底。
- **详细修复方案**:

  文件 `battle_shared_payoff.py` 第119-124行：
  ```python
  # 修改前
  for key in self._data:
      d = self._data[key]
      d["wins"] = int(d["wins"] * self._decay)
      d["draws"] = int(d["draws"] * self._decay)
      d["losses"] = int(d["losses"] * self._decay)
      d["games"] = int(d["games"] * self._decay)

  # 修改后
  for key in self._data:
      d = self._data[key]
      d["wins"] = int(d["wins"] * self._decay)
      d["draws"] = int(d["draws"] * self._decay)
      d["losses"] = int(d["losses"] * self._decay)
      d["games"] = d["wins"] + d["draws"] + d["losses"]
  ```

- **修复代价**: 低
- **修复收益**: 中

---

### Issue 4: 被淘汰对手的 payoff 数据未清理，导致内存无限增长
- **真实性**: 确认存在
- **源码验证**: `_evict_worst`（opponent_pool.py 第80-83行）仅清理了 `_opponents`、`_checkpoint_paths`、`_games_played`、`_added_episodes`。`BattleSharedPayoff` 中的 `_trueskill_ratings`、`_data`、`_players`、`_players_ids` 均未清理。更关键的是，`get_exploitability`（battle_shared_payoff.py 第142-146行）遍历 `self._players` 包含已淘汰对手，`get_payoff_matrix`（第170-173行）同样如此。这意味着可利用性计算包含已不存在的对手，导致计算结果失真。
- **级别评估**: MEDIUM → HIGH（升级。原报告侧重内存增长，但更严重的影响是 `get_exploitability` 和 `get_payoff_matrix` 包含已淘汰对手，导致可利用性评估失真，直接影响对手选择和训练决策。这是逻辑错误而非仅内存问题。）
- **影响评估**: 不够准确。原报告强调"内存无限增长"是次要问题（单个对手数据量很小），主要影响是计算准确性——`get_exploitability` 对已淘汰对手的 mu 求均值，使得可利用性指标无法反映当前对手池的真实难度。
- **建议评估**: 建议方向正确，但需要具体化。在 `BattleSharedPayoff` 中新增 `remove_player` 方法是最直接的方案。
- **详细修复方案**:

  1. 在 `BattleSharedPayoff` 中新增 `remove_player` 方法：

  文件 `battle_shared_payoff.py`，在 `ensure_player` 方法后添加：
  ```python
  def remove_player(self, player_id: str) -> None:
      """移除玩家的对战记录和评分"""
      self._players.remove(player_id)
      self._players_ids.pop(player_id, None)
      self._trueskill_ratings.pop(player_id, None)
      keys_to_remove = [k for k in self._data if k.startswith(f"{player_id}-") or k.endswith(f"-{player_id}")]
      for k in keys_to_remove:
          del self._data[k]
  ```

  2. 在 `OpponentPool._evict_worst` 中调用清理：

  文件 `opponent_pool.py` 第80-84行：
  ```python
  # 修改前
  victim = min(evictable, key=eviction_score)
  self._opponents.remove(victim)
  self._checkpoint_paths.pop(victim, None)
  self._games_played.pop(victim, None)
  self._added_episodes.pop(victim, None)
  logger.info(f"Opponent {victim} evicted from pool")

  # 修改后
  victim = min(evictable, key=eviction_score)
  self._opponents.remove(victim)
  self._checkpoint_paths.pop(victim, None)
  self._games_played.pop(victim, None)
  self._added_episodes.pop(victim, None)
  if self._payoff is not None:
      self._payoff.remove_player(victim)
  logger.info(f"Opponent {victim} evicted from pool")
  ```

- **修复代价**: 中
- **修复收益**: 高

---

### Issue 5: sigmoid 调度策略未使用调度常量
- **真实性**: 确认存在
- **源码验证**: `_compute_exploit_prob`（opponent_selector.py 第128-130行）中 sigmoid 策略使用 `1.0 / (1.0 + math.exp(-x))`，值域约 [0.007, 0.993]。而 linear（第136行）和 exp（第132-133行）策略都使用 `EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE * ...`，值域 [0.3, 0.9]。三种策略值域截然不同。
- **级别评估**: MEDIUM → MEDIUM（级别准确）
- **影响评估**: 准确。sigmoid 在训练初期几乎完全探索（~0.7% 利用率），训练后期几乎完全利用（~99.3% 利用率），与 linear/exp 的 30%-90% 范围差异巨大，用户选择不同调度策略时会得到完全不同的行为。
- **建议评估**: 建议科学合理。将 sigmoid 原始输出映射到 `[EXPLORE_SCHEDULE_MIN, EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE]` 区间，保持三种策略值域一致。
- **详细修复方案**:

  文件 `opponent_selector.py` 第128-130行：
  ```python
  # 修改前
  if self._schedule == "sigmoid":
      x = (p - 0.5) * 10
      p_effective = 1.0 / (1.0 + math.exp(-x))

  # 修改后
  if self._schedule == "sigmoid":
      x = (p - 0.5) * 10
      raw = 1.0 / (1.0 + math.exp(-x))
      p_effective = EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE * raw
  ```

- **修复代价**: 低
- **修复收益**: 中

---

### Issue 6: decay_all 仅衰减原始胜负计数，不衰减 TrueSkill 评分
- **真实性**: 确认存在
- **源码验证**: `decay_all`（battle_shared_payoff.py 第117-124行）仅对 `_data` 中的 wins/draws/losses/games 应用衰减，`_trueskill_ratings` 中的 mu 和 sigma 完全不受影响。随着衰减进行，`get_win_rate` 因计数趋零而向 0.5 收缩，而 TrueSkill mu 保持不变。`get_exploitability`（第126-156行）混合使用了未衰减的 TrueSkill mu 和已衰减的计数（通过 confidence_weight 间接影响），语义不一致。
- **级别评估**: MEDIUM → LOW（降级。这是一个设计层面的问题，而非 bug。TrueSkill 本身就是通过新对局自然更新的评分系统，衰减 TrueSkill mu 在理论上并不合理——mu 反映的是绝对实力估计，不应该随时间"遗忘"。真正的根源问题是：既然已经有 TrueSkill 评分系统，是否还需要 decay_all？但这是设计决策，不是代码缺陷。）
- **影响评估**: 部分准确。两种指标"发散"的说法过于严重——在实际使用中，decay=0.99 衰减非常缓慢，且 `get_win_rate` 仅用于显示/调试，对手选择完全基于 TrueSkill，所以影响有限。且 `get_exploitability` 并未直接使用 `_data` 中的衰减数据，原报告中"confidence_weight 间接影响"的说法不准确——`get_exploitability` 完全基于 TrueSkill，不调用 `get_win_rate`。
- **建议评估**: 原建议中"重新评估是否需要 decay_all"的方向正确。根据项目原则"解决问题要从根源触发"，应该评估 decay_all 的存在必要性，而非给 TrueSkill 也加衰减（那是 workaround）。
- **详细修复方案**: 需要先确认 `decay_all` 的实际调用场景和目的。如果 `decay_all` 的目的是让旧的胜负记录逐渐"过期"以免影响胜率计算，而 TrueSkill 已经通过自身机制处理了这个问题（新对局会自然覆盖旧评分），那么 `decay_all` 本身可能是多余的。建议先保留现状，在确定 `decay_all` 的业务必要性后再决定是否移除。当前不执行代码修改。
- **修复代价**: 低（如确认移除） / 中（如需重新设计）
- **修复收益**: 低

---

### Issue 7: 类文档注释称 exploit 策略选择"胜率最低"的对手，但实际按 TrueSkill mu 最高选择
- **真实性**: 确认存在
- **源码验证**: 类文档注释（opponent_selector.py 第21行）称"利用 (Exploit): 选择胜率最低（最有挑战性）的可信对手"。但 `_exploit_select` 方法（第77-83行）按 `mu` 降序排序取最强对手：`sorted(confident, key=lambda cid: self._payoff._trueskill_ratings[cid].mu, reverse=True)`。TrueSkill mu 是绝对实力评估，"胜率最低"是相对概念，两者语义不同。
- **级别评估**: LOW → LOW（级别准确）
- **影响评估**: 准确。文档与实现不一致，可能误导维护者。
- **建议评估**: 建议科学合理。修改文档使其与实现一致。
- **详细修复方案**:

  文件 `opponent_selector.py` 第21行：
  ```python
  # 修改前
  - 利用 (Exploit): 选择胜率最低（最有挑战性）的可信对手

  # 修改后
  - 利用 (Exploit): 选择 TrueSkill 评分最高（绝对实力最强）的可信对手
  ```

- **修复代价**: 低
- **修复收益**: 低

---

### Issue 8: update 方法未防止 home == away 导致同一条记录被重复更新
- **真实性**: 确认存在
- **源码验证**: `update` 方法（battle_shared_payoff.py 第36行）中 `key = f"{home}-{away}"` 和 `rev_key = f"{away}-{home}"`，当 `home == away` 时两个 key 相同，同一条记录被更新两次，导致 wins/losses 各加1、games 加2。当前使用场景中 `home = "current"`, `away = "opponent_ep{episode}"`，不会出现 `home == away`。
- **级别评估**: LOW → LOW（级别准确。当前使用场景下不会触发，且按项目原则"不做容错"，防御性检查不是必须的。）
- **影响评估**: 准确但需补充。在当前使用方式下不会触发，但作为公开 API 缺少约束。
- **建议评估**: 原建议使用 `logger.warning` + `return` 属于容错处理，不符合项目原则"不做容错，让错误尽早暴露尽早解决"。应使用 `assert` 或 `ValueError` 让错误尽早暴露。
- **详细修复方案**:

  文件 `battle_shared_payoff.py` 第36行，在 `update` 方法开头添加断言：
  ```python
  def update(self, home: str, away: str, result: int) -> None:
      """更新对战记录和 TrueSkill 评分

      Args:
          result: 1 = home 胜, -1 = away 胜, 0 = 平局
      """
      assert home != away, f"home and away must be different players, got home=away={home}"
      # ... 原有逻辑不变
  ```

- **修复代价**: 低
- **修复收益**: 低

---

### Issue 9: _players_ids 字典保存/加载了索引值但从未被实际查询使用
- **真实性**: 确认存在
- **源码验证**: `_players_ids`（battle_shared_payoff.py 第23行）类型为 `Dict[str, int]`，将玩家 ID 映射到列表索引。但全类中仅用于 `if player_id not in self._players_ids` 成员检查（第31、42、46行），从未通过索引值查找数据。`get_payoff_matrix` 使用 `enumerate(self._players)` 而非 `_players_ids` 的索引。`save_state`（第192行）和 `load_state`（第212行）保存/恢复该字典。该字典完全可以简化为 `set`。
- **级别评估**: LOW → LOW（级别准确）
- **影响评估**: 准确。`_players_ids` 的索引值从未被使用，增加了不必要的序列化开销和一致性维护负担。
- **建议评估**: 建议合理。将 `_players_ids` 改为 `set` 是最直接的简化方案。同时应在 `save_state`/`load_state` 中移除对该字段的序列化。
- **详细修复方案**:

  文件 `battle_shared_payoff.py`：

  1. 第23行，类型声明修改：
  ```python
  # 修改前
  self._players_ids: Dict[str, int] = {}

  # 修改后
  self._players_ids: set = set()
  ```

  2. 第31-33行，`ensure_player` 方法修改：
  ```python
  # 修改前
  if player_id not in self._players_ids:
      self._players.append(player_id)
      self._players_ids[player_id] = len(self._players) - 1
      self._trueskill_ratings[player_id] = TrueSkillRating()

  # 修改后
  if player_id not in self._players_ids:
      self._players.append(player_id)
      self._players_ids.add(player_id)
      self._trueskill_ratings[player_id] = TrueSkillRating()
  ```

  3. 第42-49行，`update` 方法修改（两处）：
  ```python
  # 修改前
  if home not in self._players_ids:
      self._players.append(home)
      self._players_ids[home] = len(self._players) - 1
      self._trueskill_ratings[home] = TrueSkillRating()
  if away not in self._players_ids:
      self._players.append(away)
      self._players_ids[away] = len(self._players) - 1
      self._trueskill_ratings[away] = TrueSkillRating()

  # 修改后
  if home not in self._players_ids:
      self._players.append(home)
      self._players_ids.add(home)
      self._trueskill_ratings[home] = TrueSkillRating()
  if away not in self._players_ids:
      self._players.append(away)
      self._players_ids.add(away)
      self._trueskill_ratings[away] = TrueSkillRating()
  ```

  4. 第190-192行，`save_state` 方法移除 `players_ids`：
  ```python
  # 修改前
  state = {
      "players": self._players,
      "players_ids": self._players_ids,
      "data": self._data,
      ...
  }

  # 修改后
  state = {
      "players": self._players,
      "data": self._data,
      ...
  }
  ```

  5. 第207-212行，`load_state` 方法修改：
  ```python
  # 修改前
  required_fields = ["players", "players_ids", "data", "trueskill_ratings"]
  ...
  self._players = state["players"]
  self._players_ids = state["players_ids"]
  self._data = state["data"]

  # 修改后
  required_fields = ["players", "data", "trueskill_ratings"]
  ...
  self._players = state["players"]
  self._players_ids = set(self._players)
  self._data = state["data"]
  ```

  注意：`load_state` 中从 `self._players` 重建 `set` 而非从文件加载，既简化了序列化也消除了一致性风险。

- **修复代价**: 低
- **修复收益**: 低

## 总结

### 确认问题统计
| 编号 | 原级别 | 建议级别 | 真实性 | 修复代价 | 修复收益 |
|------|--------|----------|--------|----------|----------|
| 1 | CRITICAL | CRITICAL | 确认 | 低 | 高 |
| 2 | HIGH | MEDIUM | 确认 | 低 | 中 |
| 3 | MEDIUM | MEDIUM | 确认 | 低 | 中 |
| 4 | MEDIUM | HIGH | 确认 | 中 | 高 |
| 5 | MEDIUM | MEDIUM | 确认 | 低 | 中 |
| 6 | MEDIUM | LOW | 确认 | 低/中 | 低 |
| 7 | LOW | LOW | 确认 | 低 | 低 |
| 8 | LOW | LOW | 确认 | 低 | 低 |
| 9 | LOW | LOW | 确认 | 低 | 低 |

### 级别调整说明
- **Issue 2** HIGH → MEDIUM：硬排除本身是有效的保护机制，保护奖励只是"粒度更细"的优化而非必要功能。当前行为（硬排除）比预期行为（渐进保护）更保守，不会导致错误淘汰。
- **Issue 4** MEDIUM → HIGH：原报告低估了影响。核心问题不是内存增长，而是 `get_exploitability` 包含已淘汰对手导致计算失真，直接影响训练决策。
- **Issue 6** MEDIUM → LOW：这是设计层面问题而非代码缺陷。TrueSkill 不应被衰减，根源问题在于 decay_all 的必要性评估，当前不会造成实际错误。

### 优先修复建议
1. **Issue 1**（CRITICAL）：立即修复，`load_state` 完全不可用，阻断断点续训
2. **Issue 4**（HIGH）：尽快修复，可利用性计算包含已淘汰对手导致训练决策失真
3. **Issue 3**（MEDIUM）：推荐修复，不变量破坏可能导致胜率估计偏差
4. **Issue 5**（MEDIUM）：推荐修复，sigmoid 策略行为与其他策略不一致
5. **Issue 2**（MEDIUM）：建议修复，删除死代码使逻辑清晰
6. 其余 LOW 级别问题可按计划排期修复
