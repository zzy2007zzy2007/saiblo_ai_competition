# Clue 05：辅助损失耦合问题调查报告

## 问题描述

`_add_auxiliary_loss()` 将所有辅助损失（包括 policy encoder 和 value encoder 的辅助头损失）加到同一个 `total_loss` 上。然后 `total_loss.backward()` 一次性反传，梯度同时流向策略参数和价值参数。但辅助损失的系数是针对 policy encoder 设计的，对 value encoder 可能过大或过小。

## 结论：**确认** — 问题存在，且比预期更严重

---

## 1. 代码证据

### 1.1 辅助损失系数配置（ppo_antwar.yaml）

```yaml
# ppo_antwar.yaml 第 35-40 行
aux_tower_coef: 0.005
aux_gold_coef: 0.001
aux_enemy_tower_coef: 0.005
aux_enemy_gold_coef: 0.001
aux_base_dmg_coef: 0.0003
aux_enemy_base_dmg_coef: 0.01
```

这些系数只有一组，policy encoder 和 value encoder 的辅助头**使用完全相同的系数**。

### 1.2 _add_auxiliary_loss() 方法（ppo_trainer.py 第 631-696 行）

```python
# 第 658-671 行：policy encoder 辅助损失
total_loss = total_loss + aux_tower_coef * aux_losses["aux_tower_loss"]
total_loss = total_loss + aux_gold_coef * aux_losses["aux_gold_loss"]
total_loss = total_loss + aux_enemy_tower_coef * aux_losses["aux_enemy_tower_loss"]
total_loss = total_loss + aux_enemy_gold_coef * aux_losses["aux_enemy_gold_loss"]
if "aux_base_loss" in aux_losses:
    total_loss = total_loss + aux_base_coef * aux_losses["aux_base_loss"]
if "aux_enemy_base_loss" in aux_losses:
    total_loss = total_loss + aux_enemy_base_coef * aux_losses["aux_enemy_base_loss"]

# ── value encoder 辅助损失（与 policy aux 使用相同系数）──  ← 注释明确承认
# 第 674-693 行：value encoder 辅助损失，系数完全相同
if value_features is not None:
    total_loss = total_loss + aux_tower_coef * aux_losses["vf_aux_tower_loss"]
    total_loss = total_loss + aux_gold_coef * aux_losses["vf_aux_gold_loss"]
    total_loss = total_loss + aux_enemy_tower_coef * aux_losses["vf_aux_enemy_tower_loss"]
    total_loss = total_loss + aux_enemy_gold_coef * aux_losses["vf_enemy_gold_loss"]
    if "vf_aux_base_loss" in aux_losses:
        total_loss = total_loss + aux_base_coef * aux_losses["vf_aux_base_loss"]
    if "vf_aux_enemy_base_loss" in aux_losses:
        total_loss = total_loss + aux_enemy_base_coef * aux_losses["vf_aux_enemy_base_loss"]
```

### 1.3 total_loss 构建与反传（ppo_trainer.py 第 459-500 行）

```python
# 第 459 行：total_loss = policy_loss + vf_coef * value_loss + entropy_loss
# 第 461 行：total_loss = self._add_auxiliary_loss(total_loss, ...)

# 第 488-500 行：一次性反传
self.optimizer.zero_grad()
self.value_optimizer.zero_grad()
total_loss.backward()          # ← 一次 backward，所有辅助损失梯度同时流向两组参数

self.optimizer.step()          # ← 策略优化器更新
self.value_optimizer.step()    # ← 价值优化器更新
```

### 1.4 双优化器参数分组（ppo_trainer.py 第 36-42 行、92-107 行）

```python
# 第 36-42 行：VALUE_PREFIXES 定义
VALUE_PREFIXES = (
    "value_cnn.",
    "value_mlp.",
    "value_proj.",
    "value_layer_norm.",
    "value_head.",
)

# 第 98-107 行：参数分组
for name, param in self.policy.named_parameters():
    if name.startswith(self.VALUE_PREFIXES):
        value_params.append(param)
    else:
        policy_params.append(param)

self.optimizer = optim.Adam(policy_params, lr=lr)           # lr=0.0005
self.value_optimizer = optim.Adam(value_params, lr=lr_vf)   # lr=0.00025
```

### 1.5 compute_auxiliary_loss() 方法（ant_war_policy_value_network.py 第 335-415 行）

```python
# 第 360-386 行：policy encoder 辅助损失
tower_pred = self.tower_damage_head(features)       # features = policy_features
gold_pred = self.gold_income_head(features)
# ... MSE loss 计算

# 第 388-413 行：value encoder 辅助损失
if value_features is not None:
    vf_tower_pred = self.value_tower_damage_head(value_features)   # value_features
    vf_gold_pred = self.value_gold_income_head(value_features)
    # ... MSE loss 计算
```

---

## 2. 深度分析

### 2.1 梯度流向图

```
total_loss = policy_loss + vf_coef * value_loss + entropy_loss
           + aux_tower_coef * aux_tower_loss           → policy_cnn, policy_mlp, tower_damage_head
           + aux_gold_coef * aux_gold_loss             → policy_cnn, policy_mlp, gold_income_head
           + aux_enemy_tower_coef * aux_enemy_tower_loss → policy_cnn, policy_mlp, tower_damage_head
           + aux_enemy_gold_coef * aux_enemy_gold_loss → policy_cnn, policy_mlp, gold_income_head
           + aux_base_coef * aux_base_loss             → policy_cnn, policy_mlp, base_damage_head
           + aux_enemy_base_coef * aux_enemy_base_loss → policy_cnn, policy_mlp, base_damage_head
           + aux_tower_coef * vf_aux_tower_loss        → value_cnn, value_mlp, value_tower_damage_head
           + aux_gold_coef * vf_aux_gold_loss          → value_cnn, value_mlp, value_gold_income_head
           + aux_enemy_tower_coef * vf_aux_enemy_tower_loss → value_cnn, value_mlp, value_tower_damage_head
           + aux_enemy_gold_coef * vf_aux_enemy_gold_loss   → value_cnn, value_mlp, value_gold_income_head
           + aux_base_coef * vf_aux_base_loss          → value_cnn, value_mlp, value_base_damage_head
           + aux_enemy_base_coef * vf_aux_enemy_base_loss   → value_cnn, value_mlp, value_base_damage_head
```

**关键发现**：`total_loss.backward()` 一次调用，辅助损失的梯度同时流向：
- **策略参数**：`policy_cnn`、`policy_mlp`、`policy_proj`、`policy_layer_norm` + policy 辅助头
- **价值参数**：`value_cnn`、`value_mlp`、`value_proj`、`value_layer_norm` + value 辅助头

### 2.2 系数不匹配分析

辅助损失系数是为 policy encoder 设计的，但被原封不动地用于 value encoder。问题在于：

| 维度 | Policy Encoder | Value Encoder |
|------|---------------|---------------|
| 学习率 | 0.0005 | 0.00025（1/2） |
| 梯度裁剪阈值 | max_grad_norm=2.5 | max_grad_norm_vf=50.0（20×） |
| 主损失系数 | policy_loss 系数=1 | value_loss 系数=vf_coef=0.05 |
| 辅助损失系数 | 原始设计值 | **相同的系数** |

**核心矛盾**：

1. **value_loss 的系数只有 0.05**（`vf_coef=0.05`），但 value encoder 的辅助损失系数与 policy encoder 完全相同。以 `aux_tower_coef=0.005` 为例，对 value encoder 而言，辅助损失相对于 value_loss 的比例是 `0.005 / 0.05 = 0.1`，即辅助损失占价值学习信号的 **10%**。而 policy_loss 系数为 1，辅助损失占比仅为 `0.005 / 1 = 0.5%`。

2. **梯度裁剪阈值差异巨大**：policy 参数裁剪阈值 2.5，value 参数裁剪阈值 50.0。value encoder 的辅助损失梯度几乎不会被裁剪，但 policy encoder 的辅助损失梯度可能被裁剪。这意味着相同的辅助损失系数，对 value encoder 的实际梯度影响更大。

3. **学习率差异**：value optimizer 学习率是 policy optimizer 的一半（0.00025 vs 0.0005），但辅助损失系数相同，进一步放大了辅助损失对 value encoder 的相对影响。

### 2.3 辅助损失对 value 参数的梯度量级估算

假设 MSE loss 的梯度量级约为 1（归一化后），则：

- **policy encoder 辅助损失对 policy 参数的梯度**：系数 × 1 = 0.005（tower），0.001（gold）等
  - 相对于 policy_loss（系数=1）的占比：0.5%（tower），0.1%（gold）
- **value encoder 辅助损失对 value 参数的梯度**：系数 × 1 = 0.005（tower），0.001（gold）等
  - 相对于 value_loss（系数=0.05）的占比：10%（tower），2%（gold）

**辅助损失对 value encoder 的相对影响是 policy encoder 的 20 倍**（因为 vf_coef=0.05 vs policy_loss 系数=1）。

### 2.4 value 辅助头参数归属问题（额外发现）

`VALUE_PREFIXES` 定义为：
```python
VALUE_PREFIXES = (
    "value_cnn.",
    "value_mlp.",
    "value_proj.",
    "value_layer_norm.",
    "value_head.",
)
```

**不包含** `value_tower_damage_head.`、`value_gold_income_head.`、`value_base_damage_head.`。

这意味着 value encoder 的辅助头参数（`value_tower_damage_head`、`value_gold_income_head`、`value_base_damage_head`）被错误地归入了 `policy_params`，由 `self.optimizer`（策略优化器，lr=0.0005）管理，而非 `self.value_optimizer`（价值优化器，lr=0.00025）。

这是一个**参数归属错误**，导致：
1. value 辅助头使用更高的学习率（0.0005 vs 0.00025）
2. value 辅助头受 policy 参数的梯度裁剪阈值约束（2.5 vs 50.0）
3. value 辅助头的梯度被计入 `grad_norm_policy_sq` 而非 `grad_norm_value_sq`

### 2.5 辅助损失是否会干扰价值函数学习

**会，且干扰程度显著**：

1. **信号比例失衡**：辅助损失对 value encoder 的相对梯度占比远高于 policy encoder，可能使 value encoder 过度关注辅助任务而偏离价值预测主任务。

2. **梯度方向冲突**：辅助损失（MSE on tower_damage/gold_income）的梯度方向与 value_loss（MSE on returns）的梯度方向可能不一致。当辅助损失的梯度量级相对于 value_loss 过大时，会导致 value encoder 的参数更新方向被辅助损失主导。

3. **训练不稳定**：value encoder 的辅助损失梯度不受 `max_grad_norm_vf=50.0` 的保护（因为 value 辅助头参数归入了 policy_params），而是受 `max_grad_norm=2.5` 约束。但由于辅助损失系数较小，实际梯度可能远小于 2.5，此约束形同虚设。真正的问题是辅助损失梯度与 value_loss 梯度的方向冲突。

---

## 3. 问题严重程度评估

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| 辅助损失系数不区分 policy/value | **高** | 对 value encoder 的相对影响是 policy encoder 的 20 倍 |
| value 辅助头参数归属错误 | **高** | 归入策略优化器，学习率和裁剪阈值都不匹配 |
| 辅助损失与 value_loss 梯度方向冲突 | **中** | 可能导致价值函数学习不稳定 |
| total_loss 一次性反传 | **中** | 设计上可接受，但需要正确的系数和参数归属 |

---

## 4. 修复建议

### 方案 A：为 value encoder 辅助损失设置独立系数（推荐，最小改动）

在 `ppo_antwar.yaml` 中增加 value encoder 专用的辅助损失系数：

```yaml
# 新增 value encoder 专用辅助损失系数
vf_aux_tower_coef: 0.00025       # = aux_tower_coef × vf_coef ≈ 0.005 × 0.05
vf_aux_gold_coef: 0.00005        # = aux_gold_coef × vf_coef ≈ 0.001 × 0.05
vf_aux_enemy_tower_coef: 0.00025
vf_aux_enemy_gold_coef: 0.00005
vf_aux_base_dmg_coef: 0.000015
vf_aux_enemy_base_dmg_coef: 0.0005
```

在 `_add_auxiliary_loss()` 中使用这些独立系数。

### 方案 B：修复参数归属 + 独立系数（推荐，彻底修复）

1. **修复 `VALUE_PREFIXES`**，将 value 辅助头参数归入 value optimizer：

```python
VALUE_PREFIXES = (
    "value_cnn.",
    "value_mlp.",
    "value_proj.",
    "value_layer_norm.",
    "value_head.",
    "value_tower_damage_head.",    # 新增
    "value_gold_income_head.",     # 新增
    "value_base_damage_head.",     # 新增
)
```

2. **为 value encoder 辅助损失设置独立系数**（同方案 A）。

### 方案 C：分离辅助损失反传（最彻底，改动最大）

将 policy encoder 和 value encoder 的辅助损失分开反传：

```python
# policy 辅助损失 → 只反传到 policy 参数
policy_aux_loss = aux_tower_coef * aux_losses["aux_tower_loss"] + ...
policy_aux_loss.backward(retain_graph=True)

# value 辅助损失 → 只反传到 value 参数
value_aux_loss = vf_aux_tower_coef * aux_losses["vf_aux_tower_loss"] + ...
value_aux_loss.backward()
```

此方案改动最大，但能完全隔离两组参数的辅助损失梯度。

---

## 5. 总结

| 项目 | 结论 |
|------|------|
| policy encoder 和 value encoder 的辅助头是否使用相同系数 | **是**，完全相同，代码注释也明确承认 |
| 辅助损失对 value 参数的梯度量级 | 相对影响是 policy encoder 的 **20 倍**（因 vf_coef=0.05） |
| 辅助损失是否会干扰价值函数学习 | **会**，辅助损失梯度相对 value_loss 占比过高 |
| 双优化器架构下辅助损失的梯度流向 | value 辅助头参数**错误地归入策略优化器**，存在参数归属错误 |
| 整体结论 | **确认** — 问题存在，且伴随参数归属错误，比预期更严重 |

**建议优先级**：方案 B（修复参数归属 + 独立系数）> 方案 A（仅独立系数）> 方案 C（分离反传）
