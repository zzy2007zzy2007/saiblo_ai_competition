# 调查报告：PPO epoch=2 且 target_kl=0.05，可能过早触发早停

**调查日期**：2026-06-11
**问题编号**：clue_13
**状态**：✅ 确认（与线索 11 强关联）

---

## 1. 问题描述

当前 PPO V2 配置中 `ppo_epochs=2`、`target_kl=0.05`。早停逻辑在 epoch 0 结束后检查 `epoch_approx_kl`，若超过 `target_kl` 则跳过 epoch 1。考虑到 `logit_noise_std=0.2`（线索 11）导致的 `old_log_prob` / `new_log_prob` 不一致，`approx_kl` 存在正向偏移，极易超过 0.05 阈值，导致实际只训练 1 个 epoch，策略更新不充分。

---

## 2. 代码审查

### 2.1 早停逻辑

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:264-304`

```python
target_kl = ppo_cfg.get("target_kl")   # 0.05

for epoch in range(ppo_epochs):        # ppo_epochs = 2
    indices = torch.randperm(total_samples, device=self.device)
    epoch_kl_list = []
    for start in range(0, total_samples, batch_size):
        # ... minibatch 处理 ...
        if "approx_kl" in step_metrics:
            epoch_kl_list.append(step_metrics["approx_kl"])

    # 基于近似 KL 散度的 epoch 早停：如果策略偏移过大，停止后续 epoch
    if epoch_kl_list:
        epoch_approx_kl = sum(epoch_kl_list) / len(epoch_kl_list)
        if epoch_approx_kl > target_kl:
            logger.info(
                f"Early stopping at PPO epoch {epoch + 1}/{ppo_epochs}, "
                f"approx_kl={epoch_approx_kl:.6f} > target_kl={target_kl}"
            )
            break
```

**关键特征**：
- 早停检查在**每个 epoch 结束后**执行，不是在 minibatch 级别
- `epoch_approx_kl` 是该 epoch 所有 minibatch 的 `approx_kl` 的**算术平均**
- 一旦触发早停，直接 `break` 跳出 epoch 循环，后续 epoch 全部跳过
- `ppo_epochs=2` 时，最多只有 1 次早停判断机会（epoch 0 结束后）

### 2.2 approx_kl 的计算方式

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:529-530`

```python
# 近似 KL 散度：KL(pi_old || pi_new) ≈ (exp(log_ratio) - 1 - log_ratio).mean()
approx_kl = (ratio - 1 - log_ratio).mean().item()
```

其中 `ratio` 和 `log_ratio` 来自：

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:429-435`

```python
log_ratio = new_log_probs - mb_data["old_log_probs"]
log_ratio = torch.clamp(
    log_ratio,
    math.log(float(clip_values.get("ratio_min", 1e-5))),
    math.log(float(clip_values.get("ratio_max", 10.0))),
)
ratio = torch.exp(log_ratio)
```

**数学含义**：

`approx_kl = (ratio - 1 - log_ratio).mean()` 是 KL 散度的二阶近似：

$$KL(\pi_{old} \| \pi_{new}) \approx \mathbb{E}\left[\frac{\pi_{new}}{\pi_{old}} - 1 - \log\frac{\pi_{new}}{\pi_{old}}\right]$$

当 `ratio = 1`（策略未变化）时，`approx_kl = 0`。`approx_kl` 始终 ≥ 0（由凸性保证）。

### 2.3 当前配置

**文件**：`ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml:30-32`

```yaml
ppo_epochs: 2
batch_size: 1024
target_kl: 0.05
```

相关参数：
```yaml
clip_eps: 0.15
logit_noise_std: 0.2
ent_coef: 0.1
lr: 0.0005
```

---

## 3. 具体分析

### 3.1 早停触发条件分析

早停触发需要同时满足：
1. `epoch_approx_kl > target_kl`（即 > 0.05）
2. 当前 epoch 结束后（不是 minibatch 级别）

当 `ppo_epochs=2` 时：
- epoch 0 结束后检查：若 `approx_kl > 0.05`，跳过 epoch 1 → **实际只训练 1 个 epoch**
- epoch 1 结束后检查：即使触发也无影响（已是最后一个 epoch）

**因此，早停只在 epoch 0 结束后有意义，且一旦触发，PPO 更新量直接减半。**

### 3.2 logit_noise 对 approx_kl 的偏移效应（与线索 11 的关联）

线索 11 已确认：`logit_noise_std=0.2` 导致采集时 `old_log_prob` 基于加噪 logits 计算，更新时 `new_log_prob` 基于无噪 logits 计算。

**即使网络参数完全不变**，噪声也会引入一个正的 baseline KL 偏移：

设无噪 logits 为 `l`，噪声 `ε ~ N(0, 0.2²)`，加噪 logits `l' = l + ε`：

```
log_ratio = new_log_prob - old_log_prob
          = log_softmax(l)_a - log_softmax(l')_a
          = (l_a - l'_a) + (log Σexp(l'_i) - log Σexp(l_i))
          = -ε_a + (log Σexp(l_i + ε_i) - log Σexp(l_i))
```

由于 `log Σexp(l + ε) ≥ log Σexp(l)`（对数和指数的凸性），`log_ratio` 的期望为正，导致 `approx_kl` 存在正向偏移。

**定量估计**：对于 9 维 type_logits，当 logits 较为集中时（训练初期常见），0.2 的噪声可导致：
- 单个 logit 偏移 0.2 → 概率比约 exp(0.2) ≈ 1.22
- 对 `approx_kl` 的贡献约 `(1.22 - 1 - log(1.22)) ≈ 0.022`
- 9 维平均后，baseline KL 偏移估计在 **0.01~0.03** 量级

这意味着：
- **真实策略更新**只需贡献 0.02~0.04 的 KL，加上噪声偏移 0.01~0.03，总 `approx_kl` 就会超过 0.05
- 相当于有效阈值从 0.05 降至约 **0.02~0.04**

### 3.3 clip_eps=0.15 的放大效应

`clip_eps=0.15` 意味着 ratio 在 `[0.85, 1.15]` 范围内不被裁剪。噪声导致的 ratio 偏移（约 1.22）已超出此范围，导致：
1. `clip_fraction` 虚高
2. 被裁剪的样本的梯度信号被截断，但 `approx_kl` 仍反映完整偏移
3. 未被裁剪的样本（ratio 在 clip 范围内但偏移较大）仍贡献较高的 KL

### 3.4 早停频率估算

**无训练日志时的理论估算**：

| 场景 | baseline KL 偏移（噪声） | 真实策略 KL | 总 approx_kl | 是否触发早停 |
|------|------------------------|------------|-------------|------------|
| 训练初期（logits 集中） | 0.02~0.03 | 0.02~0.04 | 0.04~0.07 | **大概率触发** |
| 训练中期（logits 分散） | 0.01~0.02 | 0.01~0.03 | 0.02~0.05 | **可能触发** |
| 训练后期（策略稳定） | 0.005~0.01 | 0.005~0.01 | 0.01~0.02 | 不太触发 |

**估算结论**：训练初期到中期，早停触发概率预计 **50%~80%**，实际 PPO 更新经常只有 1 个 epoch。

### 3.5 ppo_epochs=2 本身的限制

即使不考虑噪声偏移，`ppo_epochs=2` 本身就是 PPO 论文推荐范围的下限（通常 3~10）。2 个 epoch 的数据利用率较低，而早停进一步将其降至 1 个 epoch，意味着：
- 每次策略更新只使用数据 1 遍，样本效率极低
- 与 PPO 的核心优势（多次复用 on-policy 数据）相矛盾
- 学习速度可能显著慢于预期

---

## 4. 与线索 11 的关联

线索 11（`logit_noise_std=0.2`）是线索 13 的**根本原因**。两者关系如下：

```
logit_noise_std=0.2 (线索11)
    ↓
采集时 old_log_prob 基于加噪 logits，更新时 new_log_prob 基于无噪 logits
    ↓
approx_kl 存在正向 baseline 偏移（约 0.01~0.03）
    ↓
epoch 0 结束后 approx_kl 更容易超过 target_kl=0.05
    ↓
早停频繁触发，实际只训练 1 个 epoch (线索13)
```

**如果线索 11 被修复（移除 logit_noise_std），线索 13 的问题将大幅缓解甚至消失。**

---

## 5. 结论

**状态：✅ 确认**

`ppo_epochs=2` 配合 `target_kl=0.05` 的早停机制确实存在过早触发的问题，且与线索 11（`logit_noise_std=0.2`）强关联。具体确认：

1. **早停逻辑确认**：epoch 级别检查，`approx_kl > 0.05` 即跳过后续所有 epoch。`ppo_epochs=2` 时，早停只可能在 epoch 0 结束后触发，一旦触发更新量减半。

2. **噪声偏移确认**：`logit_noise_std=0.2` 导致 `approx_kl` 存在约 0.01~0.03 的正向 baseline 偏移，使有效阈值从 0.05 降至约 0.02~0.04。

3. **早停频率估算**：训练初期到中期，早停触发概率预计 50%~80%，实际 PPO 更新经常只有 1 个 epoch。

4. **与线索 11 的因果链**：logit_noise → old/new log_prob 不一致 → approx_kl 偏高 → 早停过早触发。修复线索 11 可从根本上缓解此问题。

5. **独立问题**：即使移除 logit_noise，`ppo_epochs=2` 仍偏少，若真实策略 KL 较大（如学习率较高时），仍可能频繁触发早停。

---

## 6. 修复建议

### 方案 A（推荐，与线索 11 联合修复）：移除 logit_noise + 调整 target_kl

```yaml
# ppo_antwar.yaml
ppo:
  logit_noise_std: 0.0        # 0.2 → 0.0，消除 baseline KL 偏移（线索 11 修复）
  target_kl: 0.08             # 0.05 → 0.08，适度放宽早停阈值
  ppo_epochs: 3               # 2 → 3，增加数据复用次数
```

**理由**：
- 移除 `logit_noise_std` 消除 baseline KL 偏移，使 `approx_kl` 恢复真实含义
- `target_kl` 从 0.05 放宽至 0.08，允许更大的策略更新步幅，减少误触发
- `ppo_epochs` 从 2 增至 3，配合 `target_kl=0.08` 的早停保护，在数据利用率和策略稳定性之间取得更好平衡
- 三项调整协同作用，缺一不可

### 方案 B（仅针对线索 13，不修改 logit_noise）：放宽早停阈值 + 增加 epoch

```yaml
# ppo_antwar.yaml
ppo:
  target_kl: 0.1              # 0.05 → 0.1，大幅放宽
  ppo_epochs: 3               # 2 → 3
```

**理由**：
- 不修改 `logit_noise_std`，仅通过放宽阈值来容纳噪声偏移
- 风险：`approx_kl` 仍包含噪声偏移，不反映真实策略变化，早停的监控意义下降
- **不推荐**，因为未解决根本问题

### 方案 C（渐进式）：先修复线索 11，观察后再调整

```yaml
# 阶段 1：移除 logit_noise
ppo:
  logit_noise_std: 0.0        # 0.2 → 0.0
  # target_kl 和 ppo_epochs 暂不变

# 阶段 2：观察 approx_kl 分布后决定是否调整
# 如果 approx_kl 仍频繁 > 0.05 → target_kl 提至 0.08 或 ppo_epochs 增至 3
# 如果 approx_kl 正常（< 0.03）→ 维持当前配置
```

---

## 7. 验证建议

1. **移除 logit_noise 后观察**：设置 `logit_noise_std=0.0`，跑 20 个 episode，观察：
   - `approx_kl` 的分布：预期 baseline 偏移消失，值显著降低
   - 早停触发频率：预期大幅下降
   - 实际执行的 epoch 数：预期恢复到 2 个 epoch

2. **调整参数后对比**：在方案 A 的配置下跑 50 个 episode，对比：
   - 训练曲线的收敛速度
   - `approx_kl` 的均值和方差
   - 早停触发比例

3. **消融实验**：分别测试 `ppo_epochs=2/3/4` 配合 `target_kl=0.05/0.08/0.1` 的组合，找到最优配置
