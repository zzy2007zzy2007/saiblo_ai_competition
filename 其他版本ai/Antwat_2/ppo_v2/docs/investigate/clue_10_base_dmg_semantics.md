# Clue 10: own_base_dmg 语义调查报告

## 问题描述

`episode_collector.py` 中 `"own_base_dmg": hp_dmg_raw[1]`，而 `hp_dmg_raw` 来自 `antwar_env.py` 的 `(hp_dmg, hp_dmg_received)`。即 `hp_dmg_raw[0]` 是己方对敌方造成的伤害，`hp_dmg_raw[1]` 是己方受到的伤害。疑问：`own_base_dmg` 使用 `hp_dmg_raw[1]`（己方受到的伤害），语义是否与预测头名称 "own" 不匹配？

## 结论：排除 — 标签赋值正确，语义匹配

---

## 详细分析

### 1. hp_dmg_raw 的确切语义

**文件**: `antwar_env.py` 第 224-226 行

```python
# 伤害计算（self 视角：hp_dmg 是对 opponent 造成的伤害）
hp_dmg = max(0.0, old.opponent_state.hp - new.opponent_state.hp)
hp_dmg_received = max(0.0, old.self_state.hp - new.self_state.hp)
```

- `hp_dmg` = 己方对敌方造成的基地伤害（敌方 HP 减少）
- `hp_dmg_received` = 己方受到的基地伤害（己方 HP 减少）

**文件**: `antwar_env.py` 第 322 行

```python
"hp_dmg_raw": (round(hp_dmg, 2), round(hp_dmg_received, 2)),
```

因此：
- `hp_dmg_raw[0]` = 己方对敌方造成的伤害（= 敌方基地受到的伤害）
- `hp_dmg_raw[1]` = 己方受到的伤害（= 己方基地受到的伤害）

### 2. own_base_dmg 和 enemy_base_dmg 的标签赋值

**文件**: `episode_collector.py` 第 115-122 行

```python
hp_dmg_raw = rd.get("hp_dmg_raw", (0.0, 0.0))
step_aux_data.append({
    "own_twr_dmg": rd.get("self_tower_dmg", 0.0),
    "enemy_twr_dmg": rd.get("opponent_tower_dmg", 0.0),
    "own_gold": coin_gain[0],
    "enemy_gold": coin_gain[1],
    "own_base_dmg": hp_dmg_raw[1],
    "enemy_base_dmg": hp_dmg_raw[0],
})
```

赋值关系：
- `own_base_dmg` = `hp_dmg_raw[1]` = 己方受到的伤害 = **己方基地受到的伤害**
- `enemy_base_dmg` = `hp_dmg_raw[0]` = 己方对敌方造成的伤害 = **敌方基地受到的伤害**

### 3. 与 tower/gold 标签的横向对比

这是判断语义是否一致的关键。代码库中 "own" 和 "enemy" 的命名约定是一致的：

| 标签名 | 来源 | 语义 |
|--------|------|------|
| `own_twr_dmg` | `self_tower_dmg` | 己方塔受到的伤害 |
| `enemy_twr_dmg` | `opponent_tower_dmg` | 敌方塔受到的伤害 |
| `own_gold` | `coin_gain_raw[0]` = `coin_gain_self` | 己方获得的金币 |
| `enemy_gold` | `coin_gain_raw[1]` = `coin_gain_opponent` | 敌方获得的金币 |
| `own_base_dmg` | `hp_dmg_raw[1]` = `hp_dmg_received` | 己方基地受到的伤害 |
| `enemy_base_dmg` | `hp_dmg_raw[0]` = `hp_dmg` | 敌方基地受到的伤害 |

**命名约定**：`own_X` = "己方相关事件"，`enemy_X` = "敌方相关事件"。对于伤害类标签，"相关" 指的是 "谁受到了伤害"。

### 4. BaseDamageHead 输出 own/enemy 部分的语义定义

**文件**: `aux_labels.py` 第 25 行注释

```python
base_damage_labels:  (T, 10) float32  [own_5h, enemy_5h]
```

**文件**: `aux_labels.py` 第 51-52 行

```python
pref_own_base[i + 1] = pref_own_base[i] + d.get("own_base_dmg", 0.0)
pref_enemy_base[i + 1] = pref_enemy_base[i] + d.get("enemy_base_dmg", 0.0)
```

**文件**: `aux_labels.py` 第 61-69 行

```python
for t in range(T):
    for h_idx, h in enumerate(AUX_HORIZON_STEPS):
        # own 部分 (前 5 维)
        base_labels[t, h_idx] = pref_own_base[t + h] - pref_own_base[t]
        # enemy 部分 (后 5 维)
        e = h_idx + H
        base_labels[t, e] = pref_enemy_base[t + h] - pref_enemy_base[t]
```

因此 `base_labels` 的布局为：
- 前 5 维（`[:mid]`）= `own_base_dmg` 的累积 = 己方基地受到的伤害
- 后 5 维（`[mid:]`）= `enemy_base_dmg` 的累积 = 敌方基地受到的伤害

**文件**: `ant_war_policy_value_network.py` 第 379-386 行

```python
if aux_base_damage is not None:
    base_pred = self.base_damage_head(features)  # (batch, 10)
    result["aux_base_loss"] = F.mse_loss(
        base_pred[:, :mid], aux_base_damage[:, :mid]
    )
    result["aux_enemy_base_loss"] = F.mse_loss(
        base_pred[:, mid:], aux_base_damage[:, mid:]
    )
```

- `aux_base_loss` = 前 5 维预测 vs 前 5 维标签（己方基地受到的伤害）
- `aux_enemy_base_loss` = 后 5 维预测 vs 后 5 维标签（敌方基地受到的伤害）

### 5. 标签和预测头语义匹配性判断

| 维度 | 预测头输出 | 标签 | 语义 | 是否匹配 |
|------|-----------|------|------|---------|
| 前 5 维 | `base_pred[:, :mid]` | `aux_base_damage[:, :mid]` | 己方基地受到的伤害 | ✅ |
| 后 5 维 | `base_pred[:, mid:]` | `aux_base_damage[:, mid:]` | 敌方基地受到的伤害 | ✅ |

**标签和预测头语义完全匹配。**

---

## 补充说明：潜在的可读性风险

虽然标签赋值是正确的，但 `hp_dmg_raw` 的元组顺序 `(caused, received)` 与 `coin_gain_raw` 的元组顺序 `(self, opponent)` 在语义组织上不一致：

- `hp_dmg_raw[0]` = 己方造成的伤害（主动视角）
- `hp_dmg_raw[1]` = 己方受到的伤害（被动视角）
- `coin_gain_raw[0]` = 己方获得的金币（self 视角）
- `coin_gain_raw[1]` = 敌方获得的金币（opponent 视角）

这种不一致容易导致后续开发者误解 `hp_dmg_raw` 的索引含义。但这是**可读性问题**，不影响当前逻辑正确性。

---

## 修复建议

**无需修复逻辑**。标签赋值 `own_base_dmg = hp_dmg_raw[1]` 和 `enemy_base_dmg = hp_dmg_raw[0]` 是正确的。

可选的代码可读性改进（非必须）：
1. 在 `episode_collector.py` 中添加注释说明 `hp_dmg_raw` 的索引含义
2. 或将 `hp_dmg_raw` 改为具名字段（如 `NamedTuple`），避免索引混淆
