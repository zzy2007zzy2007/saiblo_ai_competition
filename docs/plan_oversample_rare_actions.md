# 方案：稀有动作数据过采样（f^(-α) 方法）

## 问题

梯度变异能产生行为多样的个体（建塔、闪电、EMP、闪避等），但 BC 训练把均值模型推回主流策略（纯闪电）。原因：top-K 数据里 90%+ 的帧是闪电，稀有动作（建塔、EMP 等）被梯度淹没了。gen_0014 的 H2=EMP 分化在 gen_0015 坍缩回纯 Lightning，正是因为这个。

## 方案：f^(-α) 过采样

### 核心思路

统计每个头 24 个类的出现频率 f_c，每个样本复制 f_c^(-α) 份。

这样，类 c 对 loss 的有效贡献正比于：

> f_c × f_c^(-α) = f_c^(1-α)

### α 的物理含义

| α | 有效贡献 | 效果 |
|---|---------|------|
| 0 | ∝ f_c | 原始分布，不变 |
| 0.5 | ∝ √f_c | 压缩差距，但不拉平 |
| 1.0 | ∝ 常数 | 所有类对 loss 贡献完全相等 |

推荐 α=0.5~0.8 起步。

### 具体实现

1. 在 `SSDataset.__init__` 末尾，对 `self.class_label` 统计每个头的类频率
2. 对每个样本，取三个头的 `max(频率^(-α))` 作为复制倍数
3. 倍数 clamp 到 [1, max_dup]（max_dup=20 防止极低频帧膨胀）
4. 复制后的数据直接拼到现有数组后面，`__len__` 自然更新

```
# 示例：1000 帧，Lightning 900 帧，EMP 50 帧，其余 50 帧
# α=0.5:
#   EMP 频率 0.05 → 复制倍数 0.05^(-0.5) ≈ 4.5
#   Lightning 频率 0.9 → 复制 0.9^(-0.5) ≈ 1.05 倍
#   EMP 帧中有一部分和 Lightning 共享同一帧（跨头污染）
#   但即使最坏情况，EMP 权重也能从 5% 提升到 ~15-20%
```

### 和"硬阈值"方案对比

| 维度 | 硬阈值（5%） | f^(-α) |
|------|------------|--------|
| 参数数 | 阈值 + max_copy | α + max_dup |
| 边界效应 | 4.9% 和 5.1% 差别 10x | 连续无跳变 |
| 直觉 | "低于阈值就翻倍" | "越稀有越要看" |
| 跨头污染处理 | 同帧的常见类被连带提升 | 同样存在，但 α 控制程度 |

### 关于跨头污染

一帧中 H1=Lightning(90%), H2=EMP(3%)，EMP 导致该帧被复制。Lightning 的 count 也随之增加。但 Lightning 已经占 90%+，多几个百分点无关紧要——我们要的是 EMP 从几乎为 0 变成有存在感，而不是精确控制分布。

## 具体参数

- α=0.5（起步值，可调）
- max_dup=20（防止 <0.1% 的帧膨胀到 1000 倍）
- 各头独立算频率，复制倍数取三个头的 max

## 代码改动

只改 `SSDataset.__init__` 末尾，添加过采样逻辑：

```python
# 在 HOLD 降采样之后，self.board/stats/class_label/action_map/head_logits 已就绪

def _compute_dup_rates(class_label, alpha=0.5, max_dup=20):
    """对每帧计算复制倍数。"""
    N, NH = class_label.shape
    rates = np.ones(N)
    for hi in range(NH):
        classes = class_label[:, hi]
        counts = np.bincount(classes, minlength=24) + 1  # +1 平滑，避免除以零
        freqs = counts / counts.sum()
        # 每个样本该头的复制倍数
        head_rates = freqs[classes] ** (-alpha)
        head_rates = np.clip(head_rates, 1, max_dup)
        rates = np.maximum(rates, head_rates)
    return rates

dup_rates = _compute_dup_rates(self.class_label, alpha=0.5, max_dup=20)
indices_to_dup = np.where(dup_rates > 1)[0]

new_indices = []
for idx in indices_to_dup:
    n = int(dup_rates[idx])
    new_indices.extend([idx] * (n - 1))  # -1 因为原样本已存在

if new_indices:
    self.board = np.concatenate([self.board, self.board[new_indices]])
    self.stats = np.concatenate([self.stats, self.stats[new_indices]])
    self.class_label = np.concatenate([self.class_label, self.class_label[new_indices]])
    self.action_map = np.concatenate([self.action_map, self.action_map[new_indices]])
    self.head_logits = np.concatenate([self.head_logits, self.head_logits[new_indices]])
```

## 实验计划

1. 先在当前暂停的训练（ss_20260708_155510）上改代码
2. 从 gen_0016 恢复训练，观察 gen_0017-0020 的头部分化情况
3. 如果有效：H2/H3 应该保留独立偏好而不是坍缩到 Lightning
4. 如果无效：尝试 α=0.8 或尝试方案 B（WeightedRandomSampler）

## 风险

1. **过拟合稀有动作** —— 如果某个稀有动作只出现 1 次，复制 20 倍让模型在"不可能复现的局面"上过拟合。但 gen_0014 的 EMP 出现了几十次（不同局面），风险不大。
2. **计算量增加** —— 每代 ~几千帧，复制后最多几万帧。SSDataset 本身很小，不影响训练速度。
