# TowerDamageHead / GoldIncomeHead 标签计算修正方案 v2

> **版本说明**：基于已验证的 `twr_hp_dmg = mid − after` 和 `coins_earned = after − mid` 公式重写。

---

## 一、当前问题

### 1.1 标签计算现状（`aux_labels.py`）

```
# TowerDamageHead（第 58–61 行）
tower_labels[t, h] = max(0, snapshot[t].tower_hp_sum − snapshot[t+h].tower_hp_sum)

# GoldIncomeHead（第 63–66 行）
gold_labels[t, h] = snapshot[t+h].coins − snapshot[t].coins
```

### 1.2 根因

`snapshot` 仅捕获回合开始前（操作前）的状态。相邻两帧快照间的 HP/Coin 差值同时混入了操作变化和回合推进变化，导致：

| 标签 | 混入的噪声 |
|---|---|
| TowerDamageHead | 新建/升级塔增加 HP，掩盖同期蚂蚁伤害（`max(0, old−new)` 可能为 0） |
| GoldIncomeHead | 操作花费扣减金币，标签反映净余额变化而非毛收入 |

### 1.3 正确公式（已验证）

两个标签的每回合原始值应使用 `mid → after` 差值：

| 标签 | 每回合原始值 | 含义 | 当前代码位置 |
|---|---|---|---|
| 己方塔伤害 | `self_tower_dmg = mid.tower_hp_sum − new.tower_hp_sum` | 纯蚂蚁+武器伤害 | `antwar_env.py:377-380` |
| 敌方塔伤害 | `opponent_tower_dmg = mid.opponent_tower_hp_sum − new.opponent_tower_hp_sum` | 同上 | `antwar_env.py:381-384` |
| 己方金币收入 | `coin_gain_self = new.coins − mid.coins` | advance_round 纯收入 | `antwar_env.py:247` |
| 敌方金币收入 | `coin_gain_opponent = new.opponent_coins − mid.opponent_coins` | 同上 | `antwar_env.py:248` |

以上原始值**已在 `reward_detail` 中可用**：
- `reward_detail["self_tower_dmg"]`（`antwar_env.py:435`）
- `reward_detail["opponent_tower_dmg"]`（`antwar_env.py:436`）
- `reward_detail["coin_gain_raw"] = (coin_gain_self, coin_gain_opponent)`（`antwar_env.py:325`）

### 1.4 聚合方式

对时间步 t 和视界 h ∈ {1, 2, 4, 8, 16}，将接下来 h 个回合的每回合原始值累加：

```
tower_labels[t][h] = Σ_{i=0}^{h-1} step_aux_data[t+i].twr_hp_dmg
gold_labels[t][h]  = Σ_{i=0}^{h-1} step_aux_data[t+i].coin_earned
```

不再使用 `max(0, Δ)` 或 `Δbalance` 差值法。

---

## 二、修正方案

### 2.1 涉及文件（3 个）

| 文件 | 修改内容 | 改动量 |
|---|---|---|
| `trainer/episode_collector.py` | `collect()`：新增 `step_aux_data` 采集；`_attach_aux_labels()`：改为接收 `step_aux_data` | ~15 行 |
| `utils/aux_labels.py` | `compute_aux_labels_from_trajectory()`：改为接受 per-round 原始值，按视界累加 | 重写 ~40 行 |
| 网络/损失/Trainer | **无需修改** | 0 行 |

### 2.2 `episode_collector.py` — `collect()` 修改

**位置**：`ppo_v2/src/ppo_antwar/trainer/episode_collector.py` 第 70–194 行

**修改 1**：在 `step_snapshots = []` 之后新增 `step_aux_data` 列表（约第 85 行后）：

```python
step_snapshots = []
step_aux_data = []  # 新增：每回合 {own_twr_dmg, enemy_twr_dmg, own_gold, enemy_gold}
```

**修改 2**：在每回合 `episode_length += 1` 之后（约第 108 行后），从 `info["reward_detail"]` 提取：

```python
# 采集 per-round 辅助标签原始值
rd = info.get("reward_detail", {})
coin_gain = rd.get("coin_gain_raw", (0.0, 0.0))
step_aux_data.append({
    "own_twr_dmg": rd.get("self_tower_dmg", 0.0),
    "enemy_twr_dmg": rd.get("opponent_tower_dmg", 0.0),
    "own_gold": coin_gain[0],
    "enemy_gold": coin_gain[1],
})
```

**修改 3**：`_attach_aux_labels` 调用（约第 174 行）改为传入 `step_aux_data`：

```python
# 修改前
self._attach_aux_labels(batch, step_snapshots)

# 修改后
self._attach_aux_labels(batch, step_aux_data)
```

### 2.3 `episode_collector.py` — `_attach_aux_labels()` 修改

**位置**：第 306–325 行

签名从 `step_snapshots` 改为 `step_aux_data`，内部调用不变：

```python
@staticmethod
def _attach_aux_labels(batch: EpisodeBatch, step_aux_data: List[Dict]):
    """从 per-round 辅助数据计算累积标签并附加到 batch。"""
    tower_labels, gold_labels = compute_aux_labels_from_trajectory(step_aux_data)
    if tower_labels.shape[0] > 0:
        num_steps = len(batch)
        valid_count = tower_labels.shape[0]
        if valid_count < num_steps:
            padded_tower = np.zeros((num_steps, 10), dtype=np.float32)
            padded_gold = np.zeros((num_steps, 10), dtype=np.float32)
            padded_tower[:valid_count] = tower_labels
            padded_gold[:valid_count] = gold_labels
            batch.aux_tower_damage = padded_tower
            batch.aux_gold_income = padded_gold
        else:
            batch.aux_tower_damage = tower_labels[:num_steps]
            batch.aux_gold_income = gold_labels[:num_steps]
        mask = np.zeros(num_steps, dtype=bool)
        mask[:valid_count] = True
        batch.aux_valid_mask = mask
```

### 2.4 `utils/aux_labels.py` — `compute_aux_labels_from_trajectory()` 重写

**位置**：`ppo_v2/src/ppo_antwar/utils/aux_labels.py`

从旧逻辑（snapshots 差值）改为新逻辑（per-round 原始值累加）：

```python
from __future__ import annotations
from typing import List, Dict, Tuple

import numpy as np

from .action_constants import NUM_HORIZONS, AUX_HORIZON_STEPS


def compute_aux_labels_from_trajectory(
    step_aux_data: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """从 per-round 辅助原始值计算多视界累积标签。

    输入 `step_aux_data` 的每个元素是一个 dict:
        own_twr_dmg  : float  # 本回合己方塔受到的伤害（= mid.hp − new.hp）
        enemy_twr_dmg: float  # 本回合敌方塔受到的伤害
        own_gold     : float  # 本回合己方金币毛收入（= new.coins − mid.coins）
        enemy_gold   : float  # 本回合敌方金币毛收入

    输出:
        tower_damage_labels: (T, 10) float32  [own_5h, enemy_5h]
        gold_income_labels:  (T, 10) float32  [own_5h, enemy_5h]
        其中 T = len(step_aux_data) − max(AUX_HORIZON_STEPS)
    """
    max_horizon = max(AUX_HORIZON_STEPS)
    n = len(step_aux_data)
    T = n - max_horizon
    if T <= 0:
        return (
            np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
            np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
        )

    # 前缀和优化：O(T × |H|) → O(n + T × |H|)
    pref_own_twr   = np.zeros(n + 1, dtype=np.float32)
    pref_enemy_twr = np.zeros(n + 1, dtype=np.float32)
    pref_own_gold  = np.zeros(n + 1, dtype=np.float32)
    pref_enemy_gold = np.zeros(n + 1, dtype=np.float32)

    for i, d in enumerate(step_aux_data):
        pref_own_twr[i + 1]   = pref_own_twr[i]   + d["own_twr_dmg"]
        pref_enemy_twr[i + 1] = pref_enemy_twr[i] + d["enemy_twr_dmg"]
        pref_own_gold[i + 1]  = pref_own_gold[i]  + d["own_gold"]
        pref_enemy_gold[i + 1] = pref_enemy_gold[i] + d["enemy_gold"]

    tower_labels = np.zeros((T, NUM_HORIZONS * 2), dtype=np.float32)
    gold_labels  = np.zeros((T, NUM_HORIZONS * 2), dtype=np.float32)

    for t in range(T):
        for h_idx, h in enumerate(AUX_HORIZON_STEPS):
            # own 部分
            tower_labels[t, h_idx] = pref_own_twr[t + h] - pref_own_twr[t]
            gold_labels[t, h_idx]  = pref_own_gold[t + h] - pref_own_gold[t]
            # enemy 部分（偏移 len(AUX_HORIZON_STEPS) = 5）
            enemy_idx = h_idx + len(AUX_HORIZON_STEPS)
            tower_labels[t, enemy_idx] = pref_enemy_twr[t + h] - pref_enemy_twr[t]
            gold_labels[t, enemy_idx]  = pref_enemy_gold[t + h] - pref_enemy_gold[t]

    return tower_labels, gold_labels
```

**性能**：前缀和预处理 O(n)，标签生成 O(T × |H|)。由于 |H| = 5 ≤ 16 且 T 通常 ≤ 500，对训练无影响。

### 2.5 `episode_collector.py` — `compute_total_coin_income()` 附带修复

**位置**：`ppo_v2/src/ppo_antwar/trainer/episode_collector.py` 第 365–381 行

当前逻辑使用旧启发式（`delta > 0` 累加），不精确。在新 `_build_battle_details` 中附带 `step_aux_data` 后，可改为直接累加：

```python
@staticmethod
def compute_total_coin_income(
    step_snapshots: List[Dict],
    player: str = "self",
    step_aux_data: Optional[List[Dict]] = None,
) -> float:
    """总金币毛收入。

    有 step_aux_data 时直接累加 per-round 值；
    否则回退到旧启发式。
    """
    key = "own_gold" if player == "self" else "enemy_gold"
    if step_aux_data:
        return sum(max(0.0, d.get(key, 0.0)) for d in step_aux_data)
    # Fallback
    coin_key = "own_coins" if player == "self" else "enemy_coins"
    if len(step_snapshots) < 2:
        return 0.0
    total = 0.0
    prev = step_snapshots[0].get(coin_key, 0.0)
    for s in step_snapshots[1:]:
        curr = s.get(coin_key, 0.0)
        delta = curr - prev
        if delta > 0:
            total += delta
        prev = curr
    return total
```

并在 `selfplay.py` 第 347 行附近，传入 `battle_details["step_aux_data"]`。

---

## 三、数据流对比

### 修改前

```
_collect()
  └→ step_snapshots（回合开始前 HP/Coin）
       └→ _attach_aux_labels
            └→ compute_aux_labels_from_trajectory(snapshots)
                 ├ TowerDmg = max(0, HP_start[t] − HP_start[t+h])  ← 混入 opΔ
                 └ GoldInc  = Coin_start[t+h] − Coin_start[t]       ← 混入花费
```

### 修改后

```
_collect()
  ├→ step_snapshots（回合开始前 HP/Coin，日志保留）
  └→ step_aux_data（每回合 mid→after 纯值，从 info["reward_detail"] 提取）
       └→ _attach_aux_labels
            └→ compute_aux_labels_from_trajectory(step_aux_data)
                 ├ TowerDmg[t][h] = Σ_{i=0}^{h-1} own_twr_dmg[t+i]    ← 纯伤害
                 └ GoldInc[t][h]  = Σ_{i=0}^{h-1} own_gold[t+i]       ← 纯收入
```

---

## 四、不变部分

| 组件 | 原因 |
|---|---|
| `TowerDamageHead` / `GoldIncomeHead`（`heads.py`） | 输入输出维度不变：(batch, 256) → (batch, 10) |
| `compute_auxiliary_loss()`（`ant_war_policy_value_network.py`） | 仍用 MSE，标签 shape 不变 |
| `EpisodeBatch`（`batch.py`） | `aux_tower_damage` / `aux_gold_income` shape 不变 |
| PPO Trainer（`ppo_trainer.py`） | 只消费 batch tensor，不感知来源变化 |
| `_last_reward_detail`（`antwar_env.py`） | `self_tower_dmg` / `coin_gain_raw` 已存在，无需新增字段 |

---

## 五、验证方案

1. **值域验证**：跑一局 battle，采集 `step_aux_data`，确认 `own_twr_dmg >= 0` 始终成立
2. **恒等式验证**：对每个回合，`coin_after = coin_before + own_gold - ops_cost` 成立
3. **回归对比**：新旧两种方式分别计算标签，确认：
   - TowerDamageHead：新值 ≥ 旧值（因建塔/升级回合旧值可能为 0）
   - GoldIncomeHead：新值 ≈ 旧值（因 `coin_gain_self` 本就正确，只是累加方式不同）
4. **训练验证**：跑 10 个 episode，确认 aux_tower_loss 和 aux_gold_loss 正常下降
