# 调查报告：终局奖励 ±500 过大导致 Returns 和 Advantages 方差极高

**调查日期**：2026-06-11
**问题编号**：clue_02
**状态**：✅ 确认

---

## 1. 问题描述

当前 PPO V2 训练中，`win_reward=500.0`、`loss_reward=-500.0`，而中间步奖励通常在 -10 到 +20 之间。终局奖励占总 return 的绝对主导地位，导致赢局和输局的 returns 分布严重双峰化，advantages 方差极大。

---

## 2. 代码审查

### 2.1 终局奖励配置

**文件**：`ppo_v2/src/ppo_antwar/utils/action_constants.py:256-257`

```python
REWARD_CONFIG: Dict = {
    ...
    "win_reward": 500.0,
    "loss_reward": -500.0,
    "step_reward_clip": 2000.0,
    ...
}
```

**关键发现**：
- `win_reward = 500.0`，`loss_reward = -500.0`
- `step_reward_clip = 2000.0`，终局奖励 500 远低于 clip 阈值，不会被裁剪
- 终局奖励 ±500 是典型中间步奖励（1-20）的 **25-500 倍**

### 2.2 终局奖励发放逻辑

**文件**：`ppo_v2/src/ppo_antwar/env/antwar_env.py:292-303`

```python
# ── 终局奖励 ──
end_reward = 0.0
if terminated and winner is not None:
    if winner == self_player:
        end_reward = REWARD_CONFIG["win_reward"]
    else:
        end_reward = REWARD_CONFIG["loss_reward"]
r += end_reward

# clip
clip_val = REWARD_CONFIG["step_reward_clip"]
r = max(-clip_val, min(clip_val, r))
```

**关键发现**：
- 终局奖励在 `terminated=True` 且 `winner is not None` 时发放
- 终局奖励与其他中间步奖励（伤害、金币、塔存活等）**叠加在同一步**
- `step_reward_clip = 2000.0`，终局步的 `r = 中间奖励 + 500`（赢局）或 `r = 中间奖励 - 500`（输局），远低于 clip 阈值
- **终局奖励以单步 spike 形式注入**，而非分散到多步

### 2.3 中间步奖励量级分析

**文件**：`ppo_v2/src/ppo_antwar/utils/action_constants.py:230-259`

```python
REWARD_CONFIG: Dict = {
    "base_hp_attack_weight": 2.0,       # 对敌方基地伤害 × 2.0
    "tower_hp_attack_weight": 0.2,      # 对敌方塔伤害 × 0.2
    "own_coin_gain_weight": 0.05,       # 己方金币收入 × 0.05
    "enemy_coin_gain_weight": 0.01,     # 敌方金币收入 × 0.01
    "tower_survival_per_tower": 0.02,   # 每塔存活 × 0.02
    "enemy_tower_survival_per_tower": 0.01,
    "balance_reward_weight": 0.02,      # 金币余额 × 0.02
    "own_die_penalty_per_ant": -0.05,   # 己方蚂蚁死亡 × -0.05
    "tech_speed_per_level": 1.5,        # 科技等级 × 1.5
    "tech_hp_per_level": 1.0,           # 蚂蚁等级 × 1.0
    ...
    "upgrade_gen_speed_l1": 15.0,       # 动作奖励：升级科技
    "upgrade_tower_l2": 3.0,            # 动作奖励：升级塔
    "downgrade_tower_penalty": -9.0,    # 动作奖励：降级塔惩罚
    ...
}
```

**典型中间步奖励估算**（非终局步）：

| 奖励来源 | 典型值 | 计算依据 |
|---------|--------|---------|
| 基地伤害 | 0-6 | 每回合造成 0-3 HP 伤害 × 2.0 |
| 塔伤害 | 0-4 | 每回合造成 0-20 塔HP × 0.2 |
| 金币收入 | 0.5-2 | 每回合收入 10-40 金币 × 0.05 |
| 塔存活 | 0.1-0.3 | 3-5 塔 × 0.02-0.05 |
| 科技加成 | 0-4.5 | gen_level 0-3 × 1.5 |
| 余额奖励 | 0.2-1 | 10-50 金币余额 × 0.02 |
| 动作奖励 | -9 ~ +15 | 建塔/升级/科技等 |
| **单步总计** | **-10 ~ +20** | 大部分步在 0-10 范围 |

### 2.4 GAE 计算逻辑

**文件**：`ppo_v2/src/ppo_antwar/utils/gae.py:66-84`

```python
for t in reversed(range(T)):
    if t == T - 1:
        next_value = 0.0 if dones_t[t] > 0.5 else final_value
    elif dones_t[t] > 0.5:
        if 0 <= ep_idx < len(episode_final_values):
            next_value = episode_final_values[ep_idx]
        else:
            next_value = 0.0
        ep_idx -= 1
    else:
        next_value = values_t[t + 1]

    delta = rewards_t[t] + gamma * next_value - values_t[t]
    gae = delta + gamma * gae_lambda * (1.0 - dones_t[t]) * gae
    advantages[t] = gae

returns = advantages + values_t
```

**关键发现**：
- 标准 GAE 计算：`delta_t = r_t + γ × V(s_{t+1}) - V(s_t)`
- 终局步（`dones[t]=True`）：`delta_T = r_T + γ × 0 - V(s_T) = r_T - V(s_T)`
- 如果 `r_T = 500`（赢局），`V(s_T)` 可能远小于 500，则 `delta_T ≈ 500 - V(s_T)` 极大
- GAE 逆向传播：`gae_t = delta_t + γ × λ × gae_{t+1}`，终局步的大 delta 会向前传播
- `returns = advantages + values`，终局步的 return ≈ `500 + 中间累积`

### 2.5 Advantages 归一化逻辑

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:128-133`

```python
# Per-batch advantage normalization（一次性，使用全批量统计量）
# 所有 PPO epoch 和 minibatch 共享同一组 (μ, σ)，避免 per-minibatch
# 归一化引入的尺度不一致和训练方差
adv_mean = advantages.mean()
adv_std = advantages.std() + 1e-8
advantages_norm = (advantages - adv_mean) / adv_std
```

**关键发现**：
- Advantages 做了 per-batch 归一化，归一化后值域约 [-3, 3]
- **但 returns 未做归一化**，直接用于 value loss 计算
- 归一化 advantages 缓解了策略梯度方向的问题，但**未解决 returns 双峰化对 value loss 的影响**

### 2.6 Returns 的 Clamp 保护

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:787-797`

```python
advantages = torch.clamp(
    advantages,
    float(clip_values.get("advantages_min", -1e4)),
    float(clip_values.get("advantages_max", 1e4)),
)
returns = torch.clamp(
    returns,
    float(clip_values.get("returns_min", -1e6)),
    float(clip_values.get("returns_max", 1e6)),
)
```

**关键发现**：
- Advantages clamp 到 [-1e4, 1e4]，returns clamp 到 [-1e6, 1e6]
- 这些 clamp 范围极宽，对 ±500 量级的终局奖励**完全不起约束作用**
- 仅作为极端值安全网，无法解决双峰化问题

---

## 3. 数值分析

### 3.1 一局 250 步 Episode 中终局奖励占总 Return 的比例

假设一个典型 episode 有 250 步（`max_steps_per_episode=512`，`MAX_ROUND=512`，实际对局通常 150-300 步结束）：

**赢局 Return 估算**：

```
中间步奖励累积（gamma=0.99 折扣）：
  假设每步平均奖励 r_step ≈ 3（含正负波动）
  折扣累积 ≈ Σ(r_step × γ^t) = r_step × (1 - γ^T) / (1 - γ)
           ≈ 3 × (1 - 0.99^250) / 0.01
           ≈ 3 × 0.918 / 0.01
           ≈ 275

终局奖励折扣值：
  win_reward 在最后一步发放，折扣 γ^249 ≈ 0.99^249 ≈ 0.082
  折扣终局奖励 ≈ 500 × 0.082 ≈ 41

Return（赢局）≈ 275 + 41 = 316
```

**但 GAE returns 的计算方式不同**——`returns = advantages + values`，不是简单折扣累积。更准确的分析应基于 TD 视角：

```
终局步（t=T）：
  delta_T = r_T + 0 - V(s_T) = 500 - V(s_T)
  若 V(s_T) ≈ 0（训练初期），delta_T ≈ 500
  gae_T = delta_T = 500
  return_T = gae_T + V(s_T) ≈ 500

倒数第二步（t=T-1）：
  delta_{T-1} = r_{T-1} + γ × V(s_T) - V(s_{T-1})
  gae_{T-1} = delta_{T-1} + γ × λ × gae_T
            ≈ delta_{T-1} + 0.99 × 0.95 × 500
            ≈ delta_{T-1} + 470
  return_{T-1} = gae_{T-1} + V(s_{T-1})
```

**终局奖励通过 GAE λ 传播的影响范围**：

| 距终局步数 | 终局奖励的传播系数 (γλ)^k | 传播的奖励量 |
|-----------|--------------------------|-------------|
| 0（终局步） | 1.0 | 500 |
| 1 | 0.9405 | 470 |
| 5 | 0.733 | 367 |
| 10 | 0.538 | 269 |
| 20 | 0.289 | 145 |
| 50 | 0.045 | 22 |
| 100 | 0.002 | 1 |

**关键结论**：终局奖励 ±500 通过 GAE 传播，在距终局 20 步内仍贡献 >145 的 advantage 增量，50 步内仍有显著影响。

### 3.2 双峰化 Returns 的定量分析

假设 batch 中有 20 个 episode，50% 赢局 50% 输局（自对弈场景）：

**赢局 episode 的 returns 分布**：
- 终局步 return ≈ 500
- 距终局 10 步的 return ≈ 269 + V(s_{T-10}) + 中间累积 ≈ 300-400
- 距终局 50 步的 return ≈ 22 + V(s_{T-50}) + 中间累积 ≈ 50-150
- 早期步 return ≈ V(s_0) ≈ 0-50（训练初期接近 0）

**输局 episode 的 returns 分布**：
- 终局步 return ≈ -500
- 距终局 10 步的 return ≈ -269 + V(s_{T-10}) + 中间累积 ≈ -300~-200
- 早期步 return ≈ -50~0

**全 batch returns 统计**：

| 统计量 | 估算值 |
|--------|--------|
| return_mean | ≈ 0（赢输抵消） |
| return_std | ≈ 250-350 |
| return_min | ≈ -500 |
| return_max | ≈ +500 |
| **赢局 return 均值** | **≈ 200-300** |
| **输局 return 均值** | **≈ -200~-300** |

**双峰间距**：赢局与输局的 return 均值差 ≈ 400-600，而同侧内的标准差 ≈ 50-100。双峰间距是同侧方差的 **4-12 倍**，形成严重的双峰分布。

### 3.3 双峰化 Returns 对 GAE Advantages 的影响

**Advantages 归一化前**：

赢局 episode 的 advantages 普遍为正（因为 return >> value），输局 episode 的 advantages 普遍为负。这导致：

```
advantages 跨 episode 方差 ≈ (赢局均值 - 输局均值)^2 / 4
                           ≈ (250 - (-250))^2 / 4
                           ≈ 62500
advantages_std ≈ 250
```

**Advantages 归一化后**：

```python
advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)
```

归一化后值域约 [-2, +2]，**看似解决了量级问题**，但引入以下隐患：

1. **信息丢失**：赢局/输局的信号被压缩到归一化后的 ±2 范围内，中间步的细微优势差异被终局奖励的巨大方差淹没
2. **跨 batch 不一致**：不同 batch 的 win_rate 不同，adv_mean 和 adv_std 差异很大，导致同一状态在不同 batch 中的归一化 advantage 值不同
3. **策略梯度方向偏移**：归一化后的 advantages 主要反映"这步属于赢局还是输局"，而非"这步比同局其他步好多少"

### 3.4 终局奖励对 VLoss 的影响（与线索1关联）

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py:454-456`

```python
value_loss_unclipped = (new_values - mb_data["returns"]).pow(2)
value_loss_clipped = (value_pred_clipped - mb_data["returns"]).pow(2)
value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

**终局奖励对 VLoss 的直接影响**：

1. **Returns 值域**：赢局 returns ≈ +200~+500，输局 returns ≈ -200~-500
2. **Value head 输出**：被 clamp 到 [-500, 500]
3. **训练初期**：V(s) ≈ 0，赢局步的 TD error ≈ 200-500，输局步的 TD error ≈ -200~-500
4. **VLoss**：`0.5 × (V - returns)^2`，当 TD error ≈ 300 时，单步贡献 = 0.5 × 90000 = 45000

**与线索1的关联**：
- 终局奖励 ±500 是 returns 值域巨大的**根本原因**
- Returns 值域大 → VLoss 值域大（数千）→ vf_coef=0.05 导致 value 梯度被压缩
- **线索1（vf_coef 失衡）是线索2（终局奖励过大）的直接后果**

### 3.5 终局奖励对 Value Function 学习的干扰

Value function 的目标是预测 returns。在双峰化 returns 下：

1. **赢局和输局的中间状态可能非常相似**（同样的塔布局、同样的金币），但 returns 差异巨大（+300 vs -300）
2. Value function 无法从当前观测区分赢输，导致 V(s) 在中间状态被拉向均值（≈0）
3. 这进一步增大了终局步的 TD error（V(s_T) ≈ 0 vs return_T ≈ ±500），形成**正反馈恶性循环**：
   - V(s) 不准确 → TD error 大 → GAE advantage 噪声大 → 策略更新方向差
   - 策略差 → 赢输随机 → returns 双峰化更严重 → V(s) 更难学

---

## 4. Reward Scaling 可行方案分析

### 方案 A：降低终局奖励绝对值（推荐）

将 `win_reward` 和 `loss_reward` 从 ±500 降低到 ±10~±50：

```python
REWARD_CONFIG: Dict = {
    ...
    "win_reward": 20.0,    # 原 500.0
    "loss_reward": -20.0,  # 原 -500.0
    ...
}
```

**优点**：
- 终局奖励与中间步奖励量级匹配（±20 vs 中间步 -10~+20）
- Returns 双峰化大幅缓解，returns 值域从 ±500 降至 ±50~±100
- VLoss 值域从数千降至数十，与 PLoss 量级接近
- 不需要修改 GAE、归一化、value head 等其他组件
- **最简单、最直接的修复**

**缺点**：
- 终局奖励过小可能导致策略不够重视胜负
- 需要实验确定最佳值（建议 10-50 范围搜索）

**推荐值**：`win_reward = 20.0`，`loss_reward = -20.0`

理由：中间步最大单步奖励约 15（upgrade_gen_speed_l1），终局奖励设为 20 约为最大中间步奖励的 1.3 倍，足以区分胜负，同时不会造成严重的双峰化。

### 方案 B：终局奖励分散到多步

将终局奖励从单步 spike 改为多步均匀发放：

```python
# 伪代码：将 ±500 分散到最后 N 步
if terminated:
    remaining_reward = win_reward  # 或 loss_reward
    steps_to_distribute = min(20, episode_length)
    per_step_bonus = remaining_reward / steps_to_distribute
    # 在最后 N 步每步额外加 per_step_bonus
```

**优点**：
- 避免单步 reward spike
- 终局信号仍然强烈

**缺点**：
- 实现复杂，需要修改环境逻辑追踪"最后 N 步"
- 需要提前知道 episode 何时结束（不可行，因为 terminated 是事后判断）
- **不可行**：无法在 episode 进行中预知何时终止

**变体方案**：在 GAE 计算时将终局奖励视为多步虚拟奖励的累积：

```python
# GAE 计算时，将终局步的 delta 分散
if dones_t[t] > 0.5:
    # 将 delta_T 分散到虚拟的 K 步
    K = 20
    delta_spread = delta / K
    # 这需要修改 GAE 算法，实现复杂
```

**结论**：此方案实现复杂且不标准，不推荐。

### 方案 C：Reward Scaling / Normalization

在 GAE 计算前对 rewards 做归一化：

```python
# 在 _ppo_update 中，GAE 计算前
rewards_tensor = torch.tensor(rewards)
rewards_mean = rewards_tensor.mean()
rewards_std = rewards_tensor.std() + 1e-8
rewards_normalized = ((rewards_tensor - rewards_mean) / rewards_std).tolist()

advantages, returns = compute_gae(
    rewards=rewards_normalized,  # 使用归一化后的奖励
    ...
)
```

**优点**：
- Returns 值域归一化到 ~[-3, 3]
- VLoss 值域降至 ~1-10
- 终局奖励的相对大小被保留（win > 中间步 > loss）

**缺点**：
- Per-batch 归一化导致不同 batch 的 reward scale 不同
- 训练过程中 reward 分布变化时，归一化统计量不稳定
- 需要同步调整 value head 的输出范围和 clamp 值
- 推理时无法使用 reward 归一化（因为不知道未来奖励）

**结论**：可行但需要较多配套修改，且引入训练-推理不一致的风险。

### 方案 D：Returns 归一化（与线索1方案A一致）

在 GAE 计算后对 returns 做标准化：

```python
# ppo_trainer.py _compute_gae 方法中
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

**优点**：
- Returns 值域归一化到 ~[-3, 3]，VLoss 降至 ~1-10
- 不改变 reward 信号，仅改变 value function 的预测目标

**缺点**：
- Value head 输出范围需要调整（从 [-500, 500] 到 [-3, 3]）
- 推理时需要反标准化
- Per-batch 归一化存在跨 batch 不一致问题
- **治标不治本**：终局奖励仍然造成 advantages 的双峰化（虽然归一化后看似解决）

### 方案 E：使用 Reward Shaping 替代终局奖励

用潜在函数（potential-based reward shaping）替代终局奖励，使奖励更平滑：

```python
# 基于胜负概率的潜在函数
def potential(state):
    # 用 HP 差、塔数差、金币差等估计胜负概率
    hp_diff = state.self_state.hp - state.opponent_state.hp
    tower_diff = state.self_state.tower_count - state.opponent_state.tower_count
    return some_function(hp_diff, tower_diff, ...)

# 每步奖励 = 原始奖励 + γ × Φ(s') - Φ(s)
shaped_reward = original_reward + gamma * potential(new_state) - potential(old_state)
```

**优点**：
- 理论上保证不改变最优策略（potential-based shaping 的理论保证）
- 终局信号被分散到每步的 potential 变化中
- 避免单步 reward spike

**缺点**：
- 需要设计合理的 potential 函数
- 实现复杂度高
- 如果 potential 函数设计不当，可能引入新的偏差

**结论**：理论最优但实现复杂，可作为长期优化方向。

---

## 5. 结论

**✅ 确认问题存在**：终局奖励 ±500 过大，导致 returns 和 advantages 方差极高。

### 5.1 核心问题链

```
终局奖励 ±500（单步 spike）
    ↓
Returns 双峰化（赢局 +200~+500，输局 -200~-500）
    ↓
┌─────────────────────────────────────────────┐
│ 1. Advantages 归一化前方差极大（std ≈ 250）    │
│    → 归一化后信息丢失，策略梯度方向偏移         │
│                                               │
│ 2. VLoss 值域数千（returns 在数百量级）         │
│    → vf_coef=0.05 导致 value 梯度被压缩        │
│    → 价值函数学习缓慢（线索1 的根本原因）       │
│                                               │
│ 3. Value function 难以从中间状态预测胜负        │
│    → V(s) 被拉向均值 → TD error 更大           │
│    → 正反馈恶性循环                            │
└─────────────────────────────────────────────┘
```

### 5.2 问题严重性评估

| 维度 | 严重性 | 说明 |
|------|--------|------|
| Returns 双峰化 | **高** | 赢输 return 差距 400-600，同侧方差仅 50-100 |
| Advantages 方差 | **中** | 归一化后缓解，但信息丢失和跨 batch 不一致仍存在 |
| VLoss 影响 | **高** | 终局奖励是 VLoss 值域大的根本原因（线索1 的根因） |
| 策略学习影响 | **高** | 策略梯度方向被赢/输信号主导，中间步优劣难以区分 |

### 5.3 与线索1的关系

**线索2 是线索1 的根本原因**：
- 线索1：vf_coef=0.05 与 VLoss 值域失衡
- 线索2：终局奖励 ±500 导致 returns 值域大
- **如果终局奖励合理（如 ±20），returns 值域在 ±50~±100，VLoss 在数十量级，vf_coef=0.05 就不会导致严重的梯度压缩**

---

## 6. 修复建议

### 推荐方案：降低终局奖励（方案 A）

**优先级**：最高

**具体修改**：

```python
# ppo_v2/src/ppo_antwar/utils/action_constants.py:256-257
REWARD_CONFIG: Dict = {
    ...
    "win_reward": 20.0,    # 原 500.0，降低 25 倍
    "loss_reward": -20.0,   # 原 -500.0，降低 25 倍
    ...
}
```

**配套调整**：

```yaml
# ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml
clip_values:
  values_min: -100    # 原 -500，相应缩小
  values_max: 100     # 原 500，相应缩小
```

**预期效果**：
- Returns 值域从 ±500 降至 ±50~±100
- VLoss 从 ~4500 降至 ~50-200
- vf_coef × VLoss 从 ~225 降至 ~2.5-10，与 PLoss（~0.1）量级接近
- Advantages 归一化前的 std 从 ~250 降至 ~25-50，信息保留更完整

### 辅助方案：Returns 归一化（方案 D，与线索1联动）

如果降低终局奖励后 VLoss 仍然偏大，可叠加 Returns 归一化：

```python
# ppo_trainer.py _compute_gae 方法中
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### 不推荐方案

- **方案 B（分散终局奖励）**：实现不可行（无法预知 episode 终止时间）
- **方案 C（Reward 归一化）**：训练-推理不一致风险
- **方案 E（Reward Shaping）**：实现复杂，可作为长期优化

---

## 7. 代码引用索引

| 内容 | 文件 | 行号 |
|------|------|------|
| win_reward / loss_reward | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 256-257 |
| step_reward_clip | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 258 |
| 终局奖励发放逻辑 | `ppo_v2/src/ppo_antwar/env/antwar_env.py` | 292-299 |
| step reward clip | `ppo_v2/src/ppo_antwar/env/antwar_env.py` | 301-303 |
| 中间步奖励权重 | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 230-254 |
| GAE delta 计算 | `ppo_v2/src/ppo_antwar/utils/gae.py` | 80-82 |
| GAE returns 计算 | `ppo_v2/src/ppo_antwar/utils/gae.py` | 84 |
| Advantages 归一化 | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 131-133 |
| Value loss MSE | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 454-456 |
| Returns clamp | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 793-797 |
| Advantages clamp | `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | 787-792 |
| values_min/max clamp | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 45-46 |
| returns_min/max clamp | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 47-48 |
| vf_coef 配置 | `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 27 |
| MAX_ROUND | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 132 |
