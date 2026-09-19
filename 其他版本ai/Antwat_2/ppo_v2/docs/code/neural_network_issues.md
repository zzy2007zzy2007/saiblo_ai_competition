# PPO V2 神经网络模块代码审查报告

审查日期：2026-06-10

审查范围：
- `ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py`
- `ppo_v2/src/ppo_antwar/network/heads.py`
- `ppo_v2/src/ppo_antwar/network/hex_cnn_encoder.py`
- `ppo_v2/src/ppo_antwar/network/hex_conv.py`
- `ppo_v2/src/ppo_antwar/network/mlp_encoder.py`
- `ppo_v2/src/ppo_antwar/utils/action_constants.py`
- `ppo_v2/src/ppo_antwar/utils/obs_utils.py`

---

## 问题 1：`evaluate_actions` 中 `gather` 索引维度不匹配

**严重程度：CRITICAL**

**所在文件：** `ant_war_policy_value_network.py`，第 293-295 行

**问题描述：**

在 `evaluate_actions` 方法的目标 log_prob 计算循环中：

```python
for type_id_val, t_logits in enumerate(all_target_logits):
    sample_mask = type_ids == type_id_val
    ...
    if sample_mask.any():
        target_logp_selected[sample_mask] = t_logp.gather(
            1, target_ks[sample_mask].unsqueeze(-1)
        ).squeeze(-1)
```

`t_logp` 的形状为 `(batch, n_valid)`，而 `target_ks[sample_mask].unsqueeze(-1)` 的形状为 `(n_samples, 1)`，其中 `n_samples` 是当前 `type_id_val` 对应的样本数量。

`torch.gather(dim=1, index)` 要求 `index` 与 `input` 在除 `dim` 之外的维度上大小一致，即 `index` 的形状必须为 `(batch, K)`。当 batch 中存在多种不同 type 的样本时（`n_samples < batch`），会导致 `RuntimeError: shape mismatch`。

即使碰巧 `n_samples == batch`（所有样本属于同一 type），`gather` 的语义是 `out[i][j] = input[i][index[i][j]]`，会从第 `i` 行取值，而不是从 `sample_mask` 指定的行取值，导致取到错误 batch 元素的 log_prob。

**修复建议：**

先用 `sample_mask` 筛选 `t_logp` 的对应行，再执行 `gather`：

```python
target_logp_selected[sample_mask] = t_logp[sample_mask].gather(
    1, target_ks[sample_mask].unsqueeze(-1)
).squeeze(-1)
```

---

## 问题 2：`compute_auxiliary_loss` 在 `enable_auxiliary=False` 时崩溃

**严重程度：HIGH**

**所在文件：** `ant_war_policy_value_network.py`，第 360-413 行

**问题描述：**

`compute_auxiliary_loss` 方法直接访问 `self.tower_damage_head`、`self.gold_income_head`、`self.base_damage_head` 以及 `self.value_tower_damage_head` 等属性，但这些辅助头仅在 `__init__` 中 `enable_auxiliary=True` 时才会创建（第 66-74 行）。

当 `enable_auxiliary=False` 时，这些属性不存在，调用 `compute_auxiliary_loss` 会抛出 `AttributeError`。

**修复建议：**

在方法开头添加 `enable_auxiliary` 守卫检查：

```python
def compute_auxiliary_loss(self, features, aux_tower_damage, aux_gold_income,
                           aux_base_damage=None, value_features=None):
    if not self.enable_auxiliary:
        return {}
    # ... 原有逻辑
```

---

## 问题 3：`action_id_to_type_and_target` 使用 `torch.zeros_like` 保留了 float 类型

**严重程度：MEDIUM**

**所在文件：** `action_constants.py`，第 320-321 行

**问题描述：**

```python
type_ids = torch.zeros_like(flat)
target_ks = torch.zeros_like(flat)
```

`torch.zeros_like` 会保留输入张量的 `dtype`。如果传入的 `action_ids` 是 `torch.float32` 类型（Python 类型注解不强制执行），则 `type_ids` 和 `target_ks` 也会是 `float32`。后续使用这些值作为 `gather` 的索引时，`torch.gather` 要求索引为 `torch.long` 类型，会导致运行时错误或静默的取整问题。

在 `get_action` 中（第 215-217 行），`action_t` 显式使用了 `dtype=torch.long`，所以该调用路径安全。但 `evaluate_actions` 中 `action` 参数来自外部调用者，如果传入 float 类型则会触发此 bug。

**修复建议：**

显式指定 `dtype=torch.long`：

```python
type_ids = torch.zeros_like(flat, dtype=torch.long)
target_ks = torch.zeros_like(flat, dtype=torch.long)
```

或者更防御性地先转换输入：

```python
flat = action_ids.long().flatten()
type_ids = torch.zeros_like(flat)
target_ks = torch.zeros_like(flat)
```

---

## 问题 4：`get_action` 中 log_prob 包含 logit 噪声，与 `evaluate_actions` 不一致

**严重程度：MEDIUM**

**所在文件：** `ant_war_policy_value_network.py`，第 145-148 行、第 222 行

**问题描述：**

在 `get_action` 方法中，当 `logit_noise_std > 0` 时，高斯噪声被添加到 `type_logits`（第 146-148 行）。后续的 `log_prob` 计算基于这个含噪声的 `type_logits`（第 222 行 `F.log_softmax(type_logits, dim=-1)`），因此存储的 `old_log_prob` 包含了噪声的影响。

然而在 `evaluate_actions` 中，`type_logits` 是不带噪声重新计算的。这导致 PPO 的重要性比率 `exp(new_log_prob - old_log_prob)` 不仅反映策略参数的更新，还包含随机噪声的扰动，引入额外的方差，可能导致训练不稳定。

**修复建议：**

方案 A（推荐）：在添加噪声之前保存 clean logits，用于 log_prob 计算：

```python
# 在 get_action 中，噪声添加之前
clean_type_logits = type_logits.clone()

if not deterministic and self.logit_noise_std > 0:
    type_logits = type_logits + torch.randn_like(type_logits) * self.logit_noise_std

# 用 clean logits 计算 log_prob
type_logp = F.log_softmax(clean_type_logits, dim=-1)
```

方案 B：在 `evaluate_actions` 中也添加相同的噪声（不推荐，增加实现复杂度）。

---

## 审查总结

| # | 严重程度 | 文件 | 行号 | 问题简述 |
|---|---------|------|------|---------|
| 1 | CRITICAL | ant_war_policy_value_network.py | 293-295 | `evaluate_actions` 中 `gather` 索引维度不匹配，多类型 batch 会崩溃 |
| 2 | HIGH | ant_war_policy_value_network.py | 360-413 | `compute_auxiliary_loss` 在 `enable_auxiliary=False` 时 AttributeError 崩溃 |
| 3 | MEDIUM | action_constants.py | 320-321 | `torch.zeros_like` 保留 float 类型，用作索引时可能出错 |
| 4 | MEDIUM | ant_war_policy_value_network.py | 145-148, 222 | `get_action` log_prob 含噪声，与 `evaluate_actions` 不一致，影响 PPO 训练 |

**最紧急修复：** 问题 1 是 CRITICAL 级别，在标准 PPO 训练（batch > 1 且含多种动作类型）时必定触发 `RuntimeError`，需要立即修复。
