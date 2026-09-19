# Clue 07：胜率判定基于 reward 符号而非实际胜负，截断 episode 判定不准确

## 结论：**确认** — 问题存在，影响范围明确

---

## 一、问题描述

`selfplay.py` 中使用 episode reward 的正负来判定胜负：

```python
# selfplay.py 第 343 行
result_str = "win" if reward > 0 else ("loss" if reward < 0 else "draw")
```

但 episode reward 是所有步奖励之和。当 episode 被截断（达到 `max_steps`）时，没有终局奖励（`end_reward=0`），此时 reward 的正负可能不代表实际胜负。

---

## 二、代码追踪

### 2.1 胜率判定的三处使用

在 `selfplay.py` 中，**完全相同的 reward 符号判定逻辑** 出现了 3 次：

```python
# 第 343 行 — _collect_and_update_payoff 中，写入 battle_logger 和 ep_batch_logger
result_str = "win" if reward > 0 else ("loss" if reward < 0 else "draw")

# 第 424 行 — _collect_and_update_payoff 中，更新 payoff 矩阵
result = 1 if reward > 0 else (-1 if reward < 0 else 0)
self._selfplay_manager.update_payoff(opponent_id, result)

# 第 593 行 — _run_single_episode 中，记录 selfplay_logger
battle_result = "win" if reward > 0 else ("loss" if reward < 0 else "draw")
```

这三处全部依赖 `reward`（即 `episode_reward`，所有步奖励之和）的符号。

### 2.2 环境的终局奖励机制

`antwar_env.py` 第 293-299 行：

```python
# ── 终局奖励 ──
end_reward = 0.0
if terminated and winner is not None:
    if winner == self_player:
        end_reward = REWARD_CONFIG["win_reward"]    # 500.0
    else:
        end_reward = REWARD_CONFIG["loss_reward"]   # -500.0
r += end_reward
```

关键条件：**`terminated and winner is not None`** 才会发放终局奖励。

- `terminated=True, winner=0或1` → 发放 ±500 终局奖励
- `terminated=True, winner=None` → `end_reward=0`（平局但无 winner）
- `terminated=False`（截断）→ `end_reward=0`

### 2.3 截断判定

`antwar_env.py` 第 439 行：

```python
truncated = round_idx >= MAX_ROUND and not terminated
```

`MAX_ROUND = 512`，`max_steps_per_episode = 512`（默认配置）。

`episode_collector.py` 第 185-192 行还有一层截断：

```python
if episode_length >= self._max_steps:
    done = True

# 检测截断：达到 max_steps 但 env 未返回 terminated/truncated
was_truncated = (episode_length >= self._max_steps) and not (terminated or truncated)
if was_truncated:
    batch.final_value = self._self_agent.get_value(obs_self)
    batch.dones[-1] = True
```

### 2.4 winner 信息未传递到上层

`antwar_env.py` 第 440-448 行，`info` 字典的构成：

```python
return (
    reward_self,
    reward_opponent,
    terminated,
    truncated,
    {
        "round_index": round_idx,
        "reward_detail": self._last_reward_detail,
    },
)
```

**`winner` 信息未包含在 `info` 字典中**。虽然 `winner` 在 `_resolve_turn_from_ops` 中被获取（第 388 行），但仅用于内部计算 `end_reward`，不对外暴露。

### 2.5 battle_details 未包含胜负判定信息

`episode_collector.py` 第 354-373 行：

```python
@staticmethod
def _build_battle_details(
    episode_reward, episode_length, env, action_counts,
    action_rewards, reward_sources, step_snapshots,
) -> Dict:
    return {
        "reward": episode_reward,
        "length": episode_length,
        "rounds": getattr(env, "round_count", episode_length),
        "action_counts": action_counts,
        "action_rewards": action_rewards,
        "reward_sources": reward_sources,
        "snapshots": step_snapshots,
    }
```

`battle_details` 中没有 `winner`、`terminated`、`truncated`、`was_truncated` 等字段。

---

## 三、具体分析

### 3.1 截断 episode 的比例

- `MAX_ROUND = 512`，`max_steps_per_episode = 512`
- 根据项目文档 `clue_02_end_reward_too_large.md` 的估算："实际对局通常 150-300 步结束"
- 因此**正常对局中截断比例较低**（估计 < 5%）
- 但在训练早期或对手实力接近时，可能出现更多持久战，截断比例上升

### 3.2 截断时 reward 正负是否准确反映胜负

**不准确。** 分析如下：

| 场景 | 终局奖励 | 累积 reward 可能 | 实际胜负 | 判定结果 | 是否正确 |
|------|---------|-----------------|---------|---------|---------|
| 正常胜出 | +500 | 正 | 胜 | win | ✅ |
| 正常败北 | -500 | 负 | 负 | loss | ✅ |
| 截断，己方优势 | 0 | 正 | 应为胜/优势 | win | ⚠️ 偶然正确 |
| 截断，己方劣势 | 0 | 负 | 应为负/劣势 | loss | ⚠️ 偶然正确 |
| 截断，己方劣势但累积 reward 为正 | 0 | 正 | 应为负 | win | ❌ **误判** |
| 截断，己方优势但累积 reward 为负 | 0 | 负 | 应为胜 | loss | ❌ **误判** |
| 截断，双方均势 | 0 | ≈0 | 平局 | draw | ⚠️ 偶然正确 |

**误判场景举例**：
- 己方 HP 低但塔多、科技高 → `tower_survival` 和 `tech_bonus` 为正 → reward > 0 → 判定为 "win"，但实际可能即将输掉
- 己方 HP 高但频繁死亡蚂蚁 → `die_penalty` 为负 → reward < 0 → 判定为 "loss"，但实际可能占优

### 3.3 环境是否提供 winner 信息

**环境内部有 winner 信息，但未对外暴露。**

- `antwar_env.py` 第 388 行：`winner = getattr(new_state, "winner", None)` — 从 SDK 获取
- 第 395 行：传入 `_compute_battle_rewards` 用于计算终局奖励
- **但 `info` 字典中不包含 `winner`**，也不包含 `terminated`/`truncated` 标志

### 3.4 对 payoff 更新和 TrueSkill 评分的影响

`selfplay.py` 第 424-428 行：

```python
if opponent_id is not None and self._selfplay_manager is not None:
    result = 1 if reward > 0 else (-1 if reward < 0 else 0)
    self._selfplay_manager.update_payoff(opponent_id, result)
```

`BattleSharedPayoff.update()` 方法（`battle_shared_payoff.py` 第 36-94 行）接收 `result` 参数：
- `result=1` → 记录为 home 胜，调用 `trueskill.rate_1vs1(home, away)`
- `result=-1` → 记录为 away 胜，调用 `trueskill.rate_1vs1(away, home)`
- `result=0` → 记录为平局，调用 `trueskill.rate_1vs1(home, away, drawn=True)`

**影响链条**：

1. **截断误判 → payoff 矩阵偏差**：将劣势方误判为胜方，`wins/losses` 计数错误
2. **payoff 偏差 → 贝叶斯胜率偏差**：`get_win_rate()` 使用 `wins + draws * 0.5` 计算，误判直接影响胜率
3. **TrueSkill 评分漂移**：`rate_1vs1` 根据胜负结果更新 mu/sigma，误判导致评分向错误方向移动
4. **对手选择偏差**：`OpponentSelector` 基于 TrueSkill 评分和胜率选择对手，评分漂移导致对手选择策略偏移
5. **可利用性评估失真**：`get_exploitability()` 基于 mu/sigma 计算，评分漂移导致可利用性评估不准确

**影响程度**：由于截断比例较低（估计 < 5%），对整体 TrueSkill 评分的影响有限，但会在边界情况下产生系统性偏差。

---

## 四、修复建议

### 方案 A：将 winner 信息从环境传递到上层（推荐）

**步骤 1**：在 `antwar_env.py` 的 `info` 字典中暴露 `winner` 和 `terminated`/`truncated`

```python
# antwar_env.py _resolve_turn_from_ops 返回值修改
return (
    reward_self,
    reward_opponent,
    terminated,
    truncated,
    {
        "round_index": round_idx,
        "reward_detail": self._last_reward_detail,
        "winner": winner,           # 新增
        "terminated": terminated,   # 新增
        "truncated": truncated,     # 新增
    },
)
```

**步骤 2**：在 `episode_collector.py` 的 `battle_details` 中传递胜负信息

```python
# episode_collector.py collect() 方法中，循环结束后
# 从最后一步的 info 中提取终局信息
final_info = info  # 最后一步的 info
battle_details = self._build_battle_details(
    episode_reward, episode_length, env,
    action_counts, action_rewards, reward_sources, step_snapshots,
)
# 新增字段
battle_details["winner"] = final_info.get("winner")
battle_details["terminated"] = final_info.get("terminated", False)
battle_details["truncated"] = final_info.get("truncated", False)
```

**步骤 3**：在 `selfplay.py` 中使用 winner 信息判定胜负

```python
# selfplay.py _collect_and_update_payoff 中
winner = battle_details.get("winner")
terminated = battle_details.get("terminated", False)
truncated = battle_details.get("truncated", False)

if terminated and winner is not None:
    self_player = 0 if self_first else 1
    if winner == self_player:
        result_str = "win"
        result = 1
    else:
        result_str = "loss"
        result = -1
elif truncated:
    # 截断时基于 HP 判定或标记为 draw
    result_str = "draw"
    result = 0
else:
    # 兜底：使用 reward 符号
    result_str = "win" if reward > 0 else ("loss" if reward < 0 else "draw")
    result = 1 if reward > 0 else (-1 if reward < 0 else 0)
```

### 方案 B：利用已有的 reward_sources 判定（最小改动）

`reward_sources["rw_end_reward"]` 已经累积了终局奖励：
- `rw_end_reward = 500.0` → 胜
- `rw_end_reward = -500.0` → 负
- `rw_end_reward = 0.0` → 截断或平局

```python
# selfplay.py 中
end_reward = battle_details.get("reward_sources", {}).get("rw_end_reward", 0.0)
if end_reward > 0:
    result_str = "win"
    result = 1
elif end_reward < 0:
    result_str = "loss"
    result = -1
else:
    # rw_end_reward=0：截断或平局，标记为 draw
    result_str = "draw"
    result = 0
```

**方案 B 的局限**：截断时全部标记为 draw，丢失了 HP 优势方信息。但比当前基于 reward 符号的判定更准确，且改动最小。

### 方案 C：截断时基于 HP 判定胜负

在方案 A 基础上，截断时比较双方 HP：

```python
if truncated:
    # 从 snapshots 获取终局 HP
    last_snapshot = battle_details.get("snapshots", [])[-1] if battle_details.get("snapshots") else {}
    own_hp = last_snapshot.get("own_tower_hp", {})
    enemy_hp = last_snapshot.get("enemy_tower_hp", {})
    own_total = sum(own_hp.values()) if own_hp else 0
    enemy_total = sum(enemy_hp.values()) if enemy_hp else 0
    if own_total > enemy_total:
        result_str = "win"
        result = 1
    elif own_total < enemy_total:
        result_str = "loss"
        result = -1
    else:
        result_str = "draw"
        result = 0
```

**推荐**：方案 A + 方案 C 组合，既利用环境提供的 winner 信息，又为截断场景提供基于 HP 的合理判定。

---

## 五、总结

| 维度 | 分析结果 |
|------|---------|
| 问题是否确认 | **确认** — reward 符号判定胜负在截断时不准确 |
| 截断比例 | 估计 < 5%（正常对局 150-300 步，max_steps=512） |
| 环境是否提供 winner | **内部有，但未暴露到 info 字典** |
| 对 TrueSkill 的影响 | 截断误判导致评分漂移，影响有限但存在系统性偏差 |
| 对 payoff 的影响 | wins/losses 计数错误，贝叶斯胜率偏差 |
| 修复优先级 | 中 — 截断比例低但影响评分系统正确性 |
| 推荐方案 | 方案 A + C：暴露 winner 信息 + 截断时 HP 判定 |
