# 调查报告：logit_noise_std=0.2 在训练时注入噪声，干扰 old_log_prob 一致性

**调查日期**：2026-06-11
**问题编号**：clue_11
**状态**：✅ 确认

---

## 1. 问题描述

当前 PPO V2 训练中，`logit_noise_std=0.2` 在 `get_action()` 采集阶段向 `type_logits` 注入高斯噪声。采集时的 `log_prob` 基于加噪后的 logits 计算（作为 `old_log_prob` 存储），但 PPO 更新时 `evaluate_actions()` 计算的 `new_log_prob` 基于无噪 logits。这种 on-policy 一致性违反导致：
1. 采集策略与更新策略不一致
2. `approx_kl` 估计偏高，可能触发早停
3. `clip_fraction` 虚高

---

## 2. 代码审查

### 2.1 噪声注入位置和方式

**文件**：`ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py:144-148`

```python
# logit noise：训练时添加高斯噪声，增强探索，防止过早确定性收敛
if not deterministic and self.logit_noise_std > 0:
    type_logits = (
        type_logits + torch.randn_like(type_logits) * self.logit_noise_std
    )
```

**关键细节**：
- 噪声仅在 `get_action()` 中注入，`evaluate_actions()` 中**不注入**
- 噪声加在 `type_logits` 上（9维），**不影响** `target_logits`
- 噪声在 `torch.clamp(type_logits, min=-20.0, max=20.0)` 之后、`masked_fill` 之前注入
- `torch.randn_like` 生成标准正态噪声，乘以 `self.logit_noise_std=0.2`

### 2.2 采集时 log_prob 计算（含噪声）

**文件**：`ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py:129-234`

`get_action()` 的完整流程：

```python
@torch.no_grad()
def get_action(self, board, global_vec, action_mask, deterministic=False, exploration_epsilon=0.05):
    policy_features = self._encode_policy(board, global_vec)
    value = self.value_head(self._encode_value(board, global_vec))

    type_logits = self.action_head.calc_type_logits(policy_features)
    type_logits = torch.clamp(type_logits, min=-20.0, max=20.0)

    # ★ 噪声注入点 ★
    if not deterministic and self.logit_noise_std > 0:
        type_logits = type_logits + torch.randn_like(type_logits) * self.logit_noise_std

    # mask 处理
    if action_mask is not None:
        type_mask = self.action_head.flat_mask_to_type_mask(action_mask)
        type_logits = type_logits.masked_fill(type_mask == 0, float("-inf"))

    # ... 采样动作 ...

    # ★ log_prob 基于加噪后的 type_logits 计算 ★
    type_logp = F.log_softmax(type_logits, dim=-1)           # 含噪声
    type_logp_selected = type_logp[:, type_id_0 : type_id_0 + 1]

    # target_logp 基于无噪的 target_logits 计算
    target_logits = self.action_head.calc_target_logits(policy_features, type_id_0)
    target_logp = F.log_softmax(target_logits, dim=-1)
    target_logp_selected = target_logp[:, target_k_0 : target_k_0 + 1]

    log_prob = (type_logp_selected + target_logp_selected).squeeze(-1)
```

**核心问题**：`type_logp = F.log_softmax(type_logits)` 中的 `type_logits` 已被加噪，因此 `old_log_prob` 包含噪声影响。

### 2.3 更新时 log_prob 计算（无噪声）

**文件**：`ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py:242-307`

```python
def evaluate_actions(self, board, global_vec, action_mask, action):
    policy_features = self._encode_policy(board, global_vec)
    value_features = self._encode_value(board, global_vec)
    values = self.value_head(value_features)

    type_ids, target_ks = action_id_to_type_and_target(action)

    type_logits = self.action_head.calc_type_logits(policy_features)
    # ★ 没有噪声注入 ★
    # ★ 没有 clamp ★

    if action_mask is not None:
        type_mask = self.action_head.flat_mask_to_type_mask(action_mask)
        type_logits = type_logits.masked_fill(type_mask == 0, float("-inf"))

    type_logp = F.log_softmax(type_logits, dim=-1)           # 无噪声
    type_logp_selected = type_logp.gather(1, type_ids.unsqueeze(-1)).squeeze(-1)

    # ... target_logp 同样无噪声 ...
    action_log_probs = type_logp_selected + target_logp_selected
```

### 2.4 logit_noise_std 配置

**文件**：`ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml:26`

```yaml
logit_noise_std: 0.2
```

**设置方式**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:89-90`

```python
ppo_cfg = self.config.get("ppo", {})
self.policy.logit_noise_std = ppo_cfg.get("logit_noise_std", 0.0)
```

### 2.5 PPO ratio 和 approx_kl 计算

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:429-445`

```python
log_ratio = new_log_probs - mb_data["old_log_probs"]
log_ratio = torch.clamp(log_ratio, math.log(1e-5), math.log(10.0))
ratio = torch.exp(log_ratio)

surr1 = ratio * mb_data["advantages_norm"]
surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_data["advantages_norm"]
policy_loss = -torch.min(surr1, surr2).mean()
clip_fraction = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean().item()
```

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:529-530`

```python
# 近似 KL 散度：KL(pi_old || pi_new) ≈ (exp(log_ratio) - 1 - log_ratio).mean()
approx_kl = (ratio - 1 - log_ratio).mean().item()
```

---

## 3. 具体分析

### 3.1 噪声对 log_prob 的直接影响

设无噪 logits 为 `l`，噪声为 `ε ~ N(0, 0.2²)`，加噪后 logits 为 `l' = l + ε`。

**采集时 old_log_prob**：
```
old_log_prob = log_softmax(l')_a = l'_a - log(Σ exp(l'_i))
```

**更新时 new_log_prob**（同一网络参数，无噪声）：
```
new_log_prob = log_softmax(l)_a = l_a - log(Σ exp(l_i))
```

**log_ratio**：
```
log_ratio = new_log_prob - old_log_prob
          = (l_a - log(Σ exp(l_i))) - (l'_a - log(Σ exp(l'_i)))
          = (l_a - l'_a) + (log(Σ exp(l'_i)) - log(Σ exp(l_i)))
          = -ε_a + (log(Σ exp(l_i + ε_i)) - log(Σ exp(l_i)))
```

由于 `ε_i` 是随机噪声，`log_ratio` 的期望不为零，且方差与 `logit_noise_std` 正相关。

### 3.2 噪声对 ratio 和 clip_fraction 的影响

当 `logit_noise_std=0.2` 时，9维 type_logits 上的高斯噪声会导致：

1. **ratio 方差增大**：噪声使 `old_log_prob` 偏离真实策略的 log_prob，导致 `ratio = exp(log_ratio)` 的方差增大。即使网络参数完全不变（new = old），ratio 也不会恒等于 1.0。

2. **clip_fraction 虚高**：由于 ratio 方差增大，更多样本的 ratio 会落在 `[1-clip_eps, 1+clip_eps]` 之外。当前 `clip_eps=0.15`，这意味着 ratio 在 `[0.85, 1.15]` 之外的样本被 clip。噪声导致的 ratio 偏移使得 clip_fraction 高于真实策略更新应有的水平。

3. **定量估计**：对于 9 类 type logits，softmax 对 logit 偏移的敏感度取决于 logit 的分布。当 logits 较为集中时（训练初期），0.2 的噪声可以显著改变概率分布，导致 ratio 偏移较大。假设 type 概率均匀（每类约 1/9），logit 偏移 0.2 对应概率比约 exp(0.2) ≈ 1.22，已超出 clip_eps=0.15 的范围。

### 3.3 噪声对 approx_kl 的影响

```python
approx_kl = (ratio - 1 - log_ratio).mean()
```

当网络参数不变时，理论上 `approx_kl = 0`。但由于噪声导致 `old_log_prob` ≠ `new_log_prob`，即使参数不变，`approx_kl` 也会有一个正的偏移量。

**这直接导致早停误触发**：

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:296-304`

```python
if epoch_kl_list:
    epoch_approx_kl = sum(epoch_kl_list) / len(epoch_kl_list)
    if epoch_approx_kl > target_kl:   # target_kl = 0.05
        logger.info(
            f"Early stopping at PPO epoch {epoch + 1}/{ppo_epochs}, "
            f"approx_kl={epoch_approx_kl:.6f} > target_kl={target_kl}"
        )
        break
```

噪声导致的 baseline KL 偏移使得 `approx_kl` 更容易超过 `target_kl=0.05`，从而在 epoch 0 结束后就触发早停，实际只训练 1 个 epoch（`ppo_epochs=2`），减少了策略更新的充分性。

### 3.4 与 exploration_epsilon 的叠加效应

**文件**：`ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py:154-158`

```python
explore = (
    not deterministic
    and exploration_epsilon > 0
    and torch.rand(1).item() < exploration_epsilon
)
```

当 `explore=True`（5% 概率）时，动作完全随机选择，但 log_prob 仍然基于加噪后的 logits 计算（第 222-234 行）。这造成了**双重不一致**：

1. **logit_noise 不一致**：采集时加噪，更新时无噪
2. **exploration_epsilon 不一致**：5% 的样本是随机动作，但其 log_prob 仍按策略分布计算，而非均匀分布

两种不一致叠加，进一步放大了 `old_log_prob` 与 `new_log_prob` 的偏差。

### 3.5 三重探索机制冗余分析

当前系统同时使用三种探索机制：

| 机制 | 参数 | 作用方式 | 是否影响 log_prob 一致性 |
|------|------|----------|------------------------|
| 熵正则化 | `ent_coef=0.1` | 在 loss 中加入 -H(π) 项，鼓励策略保持高熵 | ❌ 不影响 |
| ε-贪心 | `exploration_epsilon=0.05` | 5% 概率随机选择合法动作 | ⚠️ 间接影响（log_prob 不反映真实采样分布） |
| logit 噪声 | `logit_noise_std=0.2` | 在 type_logits 上加高斯噪声 | ✅ 直接影响（old/new log_prob 不一致） |

**冗余性分析**：

1. **熵正则化**是 PPO 的标准探索机制，通过梯度信号鼓励策略保持不确定性，是**最稳定**的探索方式。

2. **ε-贪心**在结构化动作空间中有其价值——确保所有动作类型都被尝试，避免某些类型被完全忽略。但它引入了 on-policy 一致性问题：采样分布 ≠ 策略分布。

3. **logit 噪声**与熵正则化功能高度重叠（都是通过增加策略随机性来探索），但 logit 噪声的实现方式**破坏了 PPO 的核心假设**——old_log_prob 和 new_log_prob 应基于同一策略计算。

**结论**：三种机制中，`ent_coef` 是必要且无害的；`exploration_epsilon` 有特定价值但需谨慎；`logit_noise_std` 与 `ent_coef` 功能重叠且破坏 PPO 一致性，应移除。

---

## 4. 结论

**状态：✅ 确认**

`logit_noise_std=0.2` 确实干扰了 `old_log_prob` 的一致性，是一个真实存在的问题。具体确认：

1. **采集/更新不一致**：`get_action()` 在 type_logits 上加噪后计算 log_prob，`evaluate_actions()` 用无噪 logits 计算 log_prob，两者基于不同的概率分布。

2. **ratio 和 clip_fraction 虚高**：即使网络参数不变，噪声也会导致 ratio ≠ 1.0，使 clip_fraction 高于实际策略更新应有的水平。

3. **approx_kl 偏高导致早停**：噪声引入的 baseline KL 偏移使 approx_kl 更容易超过 target_kl=0.05，导致 PPO 只训练 1 个 epoch（本应 2 个），降低学习效率。

4. **三重探索冗余**：`logit_noise_std` 与 `ent_coef` 功能重叠，且是唯一破坏 PPO on-policy 一致性的机制。

5. **与 exploration_epsilon 叠加**：5% 的随机动作样本的 log_prob 仍按加噪策略计算，双重不一致进一步放大偏差。

---

## 5. 修复建议

### 方案 A（推荐）：移除 logit_noise_std，保留 ent_coef + exploration_epsilon

```yaml
# ppo_antwar.yaml
ppo:
  ent_coef: 0.01              # 0.1 → 0.01，降低但仍保留
  exploration_epsilon: 0.02   # 0.05 → 0.02，降低但仍保留
  logit_noise_std: 0.0        # 0.2 → 0.0，完全移除
```

**理由**：
- 消除 on-policy 一致性违反，使 approx_kl 和 clip_fraction 恢复真实含义
- ent_coef 提供稳定的梯度级探索
- exploration_epsilon 以极低代价确保动作类型覆盖
- 最小化代码改动

### 方案 B（如果需要保留噪声探索）：在 evaluate_actions 中同步注入噪声

在 `evaluate_actions()` 中也加入相同的噪声逻辑，使 old/new log_prob 基于同一加噪策略计算。但这种方式**不推荐**，因为：
- 需要存储每步的噪声种子或噪声值，增加实现复杂度
- 噪声在每次 PPO epoch 重新采样，导致 new_log_prob 在不同 epoch 间不一致
- 本质上仍然不如直接移除噪声

### 方案 C（渐进式）：先降后删

```yaml
# 阶段 1：降低噪声
logit_noise_std: 0.05   # 0.2 → 0.05

# 阶段 2：确认指标改善后完全移除
logit_noise_std: 0.0
```

---

## 6. 验证建议

1. 设置 `logit_noise_std=0.0`，保持其他参数不变，跑 20 个 episode，对比：
   - `clip_fraction`：预期从偏高降至正常水平（< 0.1）
   - `approx_kl`：预期 baseline 偏移消失，更准确反映策略更新幅度
   - `ppo_epochs` 实际执行数：预期恢复到 2 个 epoch

2. 对比训练曲线的稳定性和收敛速度

3. 确认移除噪声后，`ent_coef` 和 `exploration_epsilon` 仍能提供足够的探索
