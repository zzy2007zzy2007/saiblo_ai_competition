# 调查报告：VLoss 值域与 vf_coef 严重失衡

**调查日期**：2026-06-11
**问题编号**：clue_01
**状态**：✅ 确认

---

## 1. 问题描述

当前 PPO V2 训练中，`vf_coef=0.05`，VLoss 约 4500，加权贡献约 225，远超 PLoss（~0.1）。VLoss 值域数千是因为 returns 未归一化（终局奖励 ±500 叠加 gamma 折扣后 returns 值域可达数百），value head 输出也在数百量级，MSE 自然为数千。当 vf_coef 极低时，价值函数几乎学不到有效梯度。

---

## 2. 代码审查

### 2.1 vf_coef 配置

**文件**：`ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml:27`

```yaml
vf_coef: 0.05
```

同时相关的关键配置：

```yaml
# ppo_antwar.yaml:18-29
lr: 0.0005           # 策略网络学习率
lr_vf: 0.00025       # 价值网络学习率
gamma: 0.99
gae_lambda: 0.95
max_grad_norm: 2.5   # 策略梯度裁剪
max_grad_norm_vf: 50.0  # 价值梯度裁剪

# ppo_antwar.yaml:42-48
clip_values:
  values_min: -500
  values_max: 500
  returns_min: -1e6
  returns_max: 1e6
```

### 2.2 Value Loss 计算

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:447-456`

```python
new_values = new_values.clamp(
    float(clip_values.get("values_min", -500)),
    float(clip_values.get("values_max", 500)),
)
value_pred_clipped = mb_data["old_values"] + torch.clamp(
    new_values - mb_data["old_values"], -clip_eps_vf, clip_eps_vf
)
value_loss_unclipped = (new_values - mb_data["returns"]).pow(2)
value_loss_clipped = (value_pred_clipped - mb_data["returns"]).pow(2)
value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

**关键发现**：
- `value_loss` 使用 MSE：`0.5 * (V_pred - returns)^2`，未做任何归一化
- `new_values` 被 clamp 到 [-500, 500]，但 `returns` 的 clamp 范围是 [-1e6, 1e6]
- 当 returns 在数百量级、value 预测也在数百量级时，MSE = 0.5 × (数百 - 数百)^2 ≈ 数千

### 2.3 Total Loss 组合

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:458-459`

```python
entropy_loss = -ent_coef * entropy.mean()
total_loss = policy_loss + vf_coef * value_loss + entropy_loss
```

**关键发现**：
- `total_loss = PLoss + 0.05 × VLoss + entropy_loss`
- 所有损失项在同一个 `total_loss` 上调用 `backward()`

### 2.4 Returns 计算

**文件**：`ppo_v2/src/ppo_antwar/utils/gae.py:80-84`

```python
delta = rewards_t[t] + gamma * next_value - values_t[t]
gae = delta + gamma * gae_lambda * (1.0 - dones_t[t]) * gae
advantages[t] = gae

returns = advantages + values_t
```

**关键发现**：
- `returns = advantages + values`，即标准的 GAE 回报计算
- returns **未做任何归一化**（仅后续在 `ppo_trainer.py:793-797` 做 clamp 到 [-1e6, 1e6]）
- 终局奖励 ±500（见 `action_constants.py:256-257` 的 `win_reward: 500.0` / `loss_reward: -500.0`）

### 2.5 奖励值域分析

**文件**：`ppo_v2/src/ppo_antwar/utils/action_constants.py:230-259`

```python
REWARD_CONFIG: Dict = {
    "win_reward": 500.0,
    "loss_reward": -500.0,
    "step_reward_clip": 2000.0,
    "base_hp_attack_weight": 2.0,
    "tower_hp_attack_weight": 0.2,
    "own_coin_gain_weight": 0.05,
    ...
    "upgrade_gen_speed_l1": 15.0,
    "upgrade_gen_speed_l2": 10.0,
    "upgrade_gen_ant_l1": 7.5,
    ...
}
```

**终局奖励 ±500 是单步奖励的 30-500 倍**，这是 returns 值域巨大的根本原因。

### 2.6 双优化器梯度分配逻辑

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:92-107`

```python
def _build_optimizers(self):
    policy_params = []
    value_params = []
    for name, param in self.policy.named_parameters():
        if name.startswith(self.VALUE_PREFIXES):
            value_params.append(param)
        else:
            policy_params.append(param)

    self.optimizer = optim.Adam(policy_params, lr=lr)
    self.value_optimizer = optim.Adam(value_params, lr=lr_vf)
```

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:488-500`

```python
def _backward_and_clip(self, total_loss, max_grad_norm, max_grad_norm_vf):
    self.optimizer.zero_grad()
    self.value_optimizer.zero_grad()
    total_loss.backward()

    # ... 梯度检测 ...

    nn.utils.clip_grad_norm_(policy_params, max_grad_norm)    # 2.5
    nn.utils.clip_grad_norm_(value_params, max_grad_norm_vf)  # 50.0

    self.optimizer.step()
    self.value_optimizer.step()
```

**关键发现**：
- 双优化器架构：`optimizer`（策略）和 `value_optimizer`（价值）各自管理不同参数组
- 但 `total_loss.backward()` 对**所有参数**计算梯度，包括 value 参数
- value 参数的梯度来源于 `total_loss` 对 value 参数的偏导，即 `∂(PLoss + vf_coef × VLoss + entropy_loss)/∂θ_v`
- 由于 PLoss 和 entropy_loss 不依赖 value 参数，所以 value 参数的实际梯度 = `vf_coef × ∂VLoss/∂θ_v`

---

## 3. 数值分析

### 3.1 VLoss 与 PLoss 量级对比

| 指标 | 典型值 | 来源 |
|------|--------|------|
| PLoss | ~0.1 | 策略损失（advantages 已归一化，值域小） |
| VLoss | ~4500 | `0.5 × (V - returns)^2`，returns 在数百量级 |
| vf_coef × VLoss | ~225 | `0.05 × 4500` |
| ent_coef × entropy | ~0.1-0.3 | `0.1 × 1-3` |
| **total_loss** | **~225** | PLoss + vf_coef×VLoss + entropy_loss |

**结论**：`vf_coef × VLoss ≈ 225`，占总 loss 的 **99.5%+**，PLoss 和 entropy_loss 几乎可忽略。

### 3.2 双优化器下 value 参数的实际梯度缩放

虽然 `total_loss` 中 VLoss 加权后仍占主导（~225），但对 value 参数而言：

```
∂total_loss/∂θ_v = vf_coef × ∂VLoss/∂θ_v
                  = 0.05 × ∂(0.5 × MSE)/∂θ_v
```

**VLoss ≈ 4500 时，∂VLoss/∂θ_v 的量级**：
- MSE 对 value_head 最后一层权重 W 的梯度：`∂VLoss/∂W = 0.5 × 2 × (V - returns) × ∂V/∂W = (V - returns) × x`
- 假设 (V - returns) ≈ 30（TD error），x 为输入特征（~1 量级），则单层梯度 ≈ 30
- 经过 `vf_coef = 0.05` 缩放后：`0.05 × 30 = 1.5`

**对比策略参数梯度**：
- PLoss ≈ 0.1，advantages 已归一化（~1 量级），策略参数梯度量级 ≈ 0.1
- vf_coef 缩放后 value 参数梯度 ≈ 1.5，看似比策略梯度大

**但关键在于梯度裁剪**：
- 策略梯度裁剪阈值：`max_grad_norm = 2.5`
- 价值梯度裁剪阈值：`max_grad_norm_vf = 50.0`（20 倍宽松）

这意味着：
1. value 参数梯度在裁剪前可能很大（因为 VLoss 大），但 `vf_coef=0.05` 已经将梯度缩小了 20 倍
2. 裁剪阈值 50.0 又非常宽松，几乎不会触发裁剪
3. **实际效果**：value 参数的有效学习率 = `lr_vf × vf_coef × ∂VLoss/∂θ_v 的缩放`

### 3.3 有效学习率分析

| 参数组 | 名义学习率 | vf_coef 缩放 | 梯度裁剪 | 有效更新幅度 |
|--------|-----------|-------------|---------|-------------|
| 策略参数 | 5e-4 | 无（直接） | 2.5 | 正常 |
| 价值参数 | 2.5e-4 | ×0.05 | 50.0 | **严重不足** |

价值参数的有效梯度被 `vf_coef=0.05` 缩小了 20 倍，即使 `max_grad_norm_vf=50.0` 很宽松也无法弥补。这导致价值网络的更新步长极小。

### 3.4 Returns 值域推导

假设一个 episode 有 200 步（`max_steps_per_episode=512`）：

- 终局奖励：±500
- 每步中间奖励：~1-10（塔存活、科技、金币等）
- 累积中间奖励（gamma=0.99）：约 `sum(1 × 0.99^t, t=0..200) ≈ 63`
- Returns = 终局奖励 + 累积中间奖励 ≈ ±500 + 63 ≈ **±563**
- Value head 输出被 clamp 到 [-500, 500]，与 returns 范围接近
- MSE = 0.5 × (563 - 500)^2 ≈ 0.5 × 3969 ≈ **1985**（仅终局一步的误差贡献）

**如果 value 预测偏差更大（如初期 value 接近 0），MSE 可达**：
- `0.5 × (500 - 0)^2 = 125,000`（极端情况）
- 训练初期 VLoss 在数千量级完全合理

---

## 4. 结论

**✅ 确认问题存在**：VLoss 值域与 vf_coef 严重失衡。

### 4.1 核心矛盾

1. **Returns 未归一化** → VLoss 值域在数千量级
2. **vf_coef = 0.05** → 将 VLoss 缩小 20 倍后加入 total_loss
3. **双优化器 + total_loss.backward()** → value 参数梯度 = `vf_coef × ∂VLoss/∂θ_v`，被 vf_coef 缩放
4. **vf_coef × VLoss ≈ 225** → 仍然主导 total_loss，但 value 参数的有效梯度被压缩

### 4.2 实际影响

- **表面现象**：total_loss 被 vf_coef × VLoss 主导（~225），PLoss（~0.1）几乎可忽略
- **深层问题**：value 参数的梯度 = `0.05 × ∂VLoss/∂θ_v`，虽然 VLoss 大但被 vf_coef 严重压缩
- **训练效果**：价值函数学习速度远慢于策略网络，导致 value 估计长期不准确，进而影响 GAE 优势估计质量

### 4.3 梯度裁剪的不匹配

- `max_grad_norm = 2.5`（策略）vs `max_grad_norm_vf = 50.0`（价值）
- 20 倍的裁剪差异本意是允许 value 参数有更大的梯度
- 但 `vf_coef=0.05` 已经把梯度缩小了 20 倍，两者相互抵消
- 实际效果：value 参数的梯度在裁剪前就已经很小了，50.0 的裁剪阈值形同虚设

---

## 5. 修复建议

### 方案 A：Returns 归一化（推荐）

在 GAE 计算后对 returns 做标准化：

```python
# ppo_trainer.py _compute_gae 方法中，在 clamp 之前添加：
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

**优点**：
- Returns 归一化后值域 ~[-3, 3]，VLoss 降至 ~1-10 量级
- vf_coef=0.05 时，vf_coef × VLoss ≈ 0.05-0.5，与 PLoss 量级匹配
- 不需要调整 vf_coef、学习率等其他超参
- 价值函数预测目标变为标准化后的 returns，训练更稳定

**注意事项**：
- Value head 的输出范围需要相应调整（从数百变为 ~[-3, 3]）
- 推理时需要反标准化：`value = value_pred × returns_std + returns_mean`
- 需要保存 returns 的均值和标准差用于推理
- `values_min/max` 的 clamp 范围需要调整

### 方案 B：提高 vf_coef

将 `vf_coef` 从 0.05 提高到 0.5-1.0：

```yaml
# ppo_antwar.yaml
vf_coef: 0.5  # 或 1.0
```

**优点**：
- 改动最小，仅修改一个配置值
- vf_coef × VLoss ≈ 2250-4500，value 参数梯度不再被压缩

**缺点**：
- total_loss 仍然被 VLoss 主导，可能影响策略学习
- 治标不治本，returns 值域仍然很大

### 方案 C：Value Loss 使用归一化的 MSE

将 value loss 改为基于标准化的误差：

```python
# 替代原始 MSE
returns_std = mb_data["returns"].std() + 1e-8
value_loss = 0.5 * ((new_values - mb_data["returns"]) / returns_std).pow(2).mean()
```

**优点**：
- VLoss 值域归一化到 ~1 量级
- 不影响 value head 的输出范围
- 推理时无需反标准化

**缺点**：
- 非标准做法，可能引入训练不稳定
- returns_std 的估计可能有方差

### 方案 D：分离 value loss 的 backward（推荐与 A 组合）

在双优化器架构下，将 value loss 的反向传播从 total_loss 中分离：

```python
# 策略参数：仅用 PLoss + entropy_loss
policy_loss_total = policy_loss + entropy_loss
policy_loss_total.backward(retain_graph=True)

# 价值参数：单独用 VLoss（不乘 vf_coef）
value_loss.backward()
```

**优点**：
- 彻底解耦策略和价值梯度
- value 参数不再受 vf_coef 缩放
- 每个优化器独立控制梯度

**缺点**：
- 需要两次 backward，计算开销略增
- 需要仔细处理共享参数（当前架构无共享参数，所以无此问题）

---

## 6. 推荐方案

**优先级排序**：A+D > A > C > B

**最佳方案**：方案 A（Returns 归一化）+ 方案 D（分离 backward）

理由：
1. Returns 归一化从根本上解决值域问题，使 VLoss 与 PLoss 量级匹配
2. 分离 backward 消除 vf_coef 对 value 梯度的缩放效应
3. 两者组合后，vf_coef 可以作为纯粹的超参调节策略-价值平衡，而非补偿值域差异的工具

**如果只选一个方案**：方案 A（Returns 归一化），因为它是根本性修复，且实现简单。

---

## 7. 代码引用索引

| 内容 | 文件 | 行号 |
|------|------|------|
| vf_coef 配置 | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 27 |
| lr / lr_vf 配置 | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 18-19 |
| max_grad_norm / max_grad_norm_vf | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 28-29 |
| values_min/max clamp | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 45-46 |
| returns_min/max clamp | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 47-48 |
| value_loss MSE 计算 | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 447-456 |
| total_loss 组合 | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 458-459 |
| 双优化器构建 | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 92-107 |
| total_loss.backward() | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 490 |
| 梯度裁剪 | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 621-622 |
| GAE returns 计算 | `ppo_v2/src/ppo_antwar/utils/gae.py` | 80-84 |
| returns clamp | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 793-797 |
| advantage 归一化 | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 131-133 |
| win/loss reward | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 256-257 |
| ValueHead 定义 | `ppo_v2/src/ppo_antwar/network/heads.py` | 120-129 |
| VALUE_PREFIXES | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 36-42 |
| LR 调度器 | `ppo_v2/src/ppo_antwar/trainer/lr_scheduler.py` | 33-53 |
