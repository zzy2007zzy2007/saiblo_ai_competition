# PPO 位置通道真正优化 — 设计文档

## 问题：当前位置不是被 PPO 优化的

当前位置通道（action_map）的损失有两条：

| 损失 | 公式 | 性质 |
|------|------|------|
| `pos_hard` | `-log softmax(action_map)[采样位置]` | **BC 式**：采样到就强化，不管好坏 |
| `pos_mse` | `MSE(action_map_new, action_map_old)` | 约束：防爆，不优化 |

**pos_hard 的问题**：它没有用 advantage。正优势的位置和负优势的位置被**一视同仁地强化**——只要被采样到。这不是 PPO 的"正优势推高、负优势压低"逻辑。

**用户要的**：位置像 class 一样被 PPO 优化——用 advantage 区分好坏，正优势位置概率升高、负优势位置概率降低。

## 目标

让位置通道**真正进入 PPO 的策略更新**（用 advantage 加权），同时保持训练稳定。

```
真正的位置 PPO 优化:
  ratio_pos = exp(new_logp_pos - old_logp_pos)
  pg_loss_pos = -min(ratio_pos · A, clip(ratio_pos) · A)
```

## 为什么之前"位置进 ratio"会爆炸

**历史实验**（无 MSE 约束时）：

| 方案 | 结果 |
|------|------|
| 位置进 ratio（raw）| KL 爆炸 |
| 位置进 ratio（z-score）| KL 爆炸 |
| 位置进 ratio（固定缩放 ÷100）| KL 爆炸 |
| 位置进 ratio（+entropy）| KL 爆炸 |

**爆炸机制**：位置是 361 选 1 的稀疏 one-hot。PPO 梯度推高正优势位置 → 该位置概率冲向 1、其他格冲向 0 → logp 极端跳变 → ratio = exp(new-old) 指数放大 → KL 爆炸。

**但所有爆炸实验都缺少 MSE trust-region**。而 MSE 方案验证了"约束能防爆"（pos_mse 收紧时 KL 稳定）。

## 核心设计：位置进 ratio + MSE trust-region

```
L = L_ppo(class, ratio)          # class 正常 PPO
  + L_ppo(position, ratio)       # 位置真正进 ratio（用 advantage）
  + λ_mse · MSE(pos_new, pos_old)  # 防爆（trust-region）
```

**关键问题**：MSE 能否防住位置进 ratio 的爆炸？

- 爆炸根源是"位置概率被优势推到 one-hot"
- MSE 的梯度方向是"拉回行为策略的值"
- 如果 λ_mse 足够大，能抵消优势的过度拉偏 → 位置图不会 collapse → logp 不极端 → ratio 不爆

**这是之前没试过的组合**：位置真进 ratio + MSE 约束。

## 参数权衡

| 参数 | 作用 | 初始建议 |
|------|------|---------|
| λ_mse | 防爆强度 | 需要比当前 1e-4 大（因为要对抗的是 PPO 优势梯度，不是弱 hard 梯度）|
| clip_epsilon | ratio 限幅 | 位置项可以考虑更小（如 0.1）|
| 位置温度 T_pos | 采样探索 | 保持 0.3 |

**λ_mse 是关键**：太小防不住爆，太大位置不优化。需要实验确定。候选：1e-3, 1e-2, 1e-1。

## 实现要点

### 1. 位置 logp 计算（复用已验证的固定缩放）

```python
# new_logp_pos: 训练时用当前模型算
ch = action_map[class_id] / 100.0            # 固定缩放
if T_pos > 0: ch = ch / T_pos
masked = where(pos_mask, ch, -inf)
new_logp_pos = log_softmax(masked)[x, y]      # x,y 是 rollout 采样的位置

# old_logp_pos: 直接用 rollout 存的（pos_record[:, :, 2]）
```

### 2. 训练循环

```python
# old: class + position
old_logp = old_logp_class + old_logp_pos      # 位置项从 pos_record 读

# new: class + position
new_logp = new_logp_class + new_logp_pos      # 位置项用当前模型算

# PPO ratio 同时含 class 和 position
pg_loss = -min(ratio · A, clip(ratio) · A)    # ratio = exp(new - old)

# 总损失
loss = pg_loss + c_v·vf - c_e·ent + λ_mse · pos_mse
```

### 3. 位置项独立 clip（可选）

如果位置进 ratio 仍不稳，可以给位置项**独立的 clip_epsilon**（更小），class 和位置分开 clip：

```python
pg_loss_class = -min(ratio_cls · A, clip(ratio_cls, 1-εc, 1+εc) · A)
pg_loss_pos   = -min(ratio_pos · A, clip(ratio_pos, 1-εp, 1+εp) · A)
```

## 验证方法

1. **稳定性**：KL 是否保持在 <10 且收敛（对比之前 184063）
2. **位置真的被优化**：训练前后，同一局面的位置 argmax 是否变化（区别于仅 backbone 漂移）
3. **行为合理性**：diagnose 看建塔位置是否随训练变好（不是乱建）
4. **整体胜率**：reward / 挑战胜率是否上升（对比 3 头训练卡在 -0.03）

## 位置通道必须对齐"实际执行的动作"（用户设计要点）

**核心原则**：位置 logp 应该基于**实际使用到的 action_map 通道**计算，不是意图 class。

**class 保持用意图没问题**——对神经网络来说 class=17 就是"想放闪电"这个意图，意图解码只是人工重定义了动作效果（拆塔凑钱）。但**位置图是按通道分离的**，优势梯度必须只作用在实际使用的通道。

### 完整场景清单（意图 vs 实际通道）

| 意图 class | 实际执行 | 位置该用的通道 | 当前代码 |
|-----------|---------|---------------|---------|
| 0-15（塔动作）| 建/升级塔 | 通道 = class_id | ✅ 正确（已采样记录）|
| 16（降级）| 降级/拆除 | 通道 = 16 | ✅ 正确（已采样记录）|
| 17-20（超武）+ 钱够 | 放超武 | 通道 = class_id | ✅ 正确（已采样记录）|
| **17-20 + 钱不够 + 有塔拆** | **拆塔** | **通道 16** | ❌ 用通道16但没采样没记录 |
| **17-20 + 钱不够 + 无塔拆** | **HOLD** | **不更新** | ✅ 返回 None 不记录 |

**问题集中在 decoder.py 343-352 行**：`if intent_decoding and state.coins[player] < cost:` 分支，对**所有四个超武**（17/18/19/20）都是同样处理——用 `action_map[16]` 拆塔但不采样不记录。四种超武的拆塔位置全部丢失。

### 当前代码的问题

拆塔分支（decoder.py 343-352）用 `action_map[16]` argmax 选塔，**但不采样、不记录到 sampled_pos_out**：
- 拆塔位置完全丢失，不被学习
- 意图闪电的位置可能被错误强化（如果采样了闪电通道但实际拆塔）

### 需要改的

1. **decode_head 拆塔分支也采样并记录位置**（用通道 16 的分布，像其他通道一样）
2. **pos_record 记录实际使用的通道**（拆塔时是 16，不是意图的 17）
3. **训练时用实际通道 gather action_map** 算位置 logp

### 实现方式

`sampled_pos_out` 需要额外携带"实际通道号"：

```python
# decoder: 拆塔分支
pos = _sample_position(action_map[16], position_mask[16], rng, pos_temperature)
if pos is not None:
    sampled_pos_out.append((16, x, y, logprob, mask))   # 记录实际通道 16

# 正常放武器分支
pos = _sample_position(action_map[class_id], pos_mask, rng, pos_temperature)
sampled_pos_out.append((class_id, x, y, logprob, mask))  # 记录实际通道 = class_id
```

训练时：
```python
# 用 pos_record 里的实际通道 gather action_map
ch = action_map.gather(1, pos_channel)   # 不是 action_classes
logp_pos = log_softmax(masked ch)[x, y]
```

## 风险与备选

**风险**：位置进 ratio 即使有 MSE 也可能爆（之前的爆炸可能不只是缺约束）。如果爆：
- 备选 A：位置独立 clip（εp 更小）
- 备选 B：位置用 advantage 但不用 ratio——`L = -logp_pos · A`（无 clip 的 REINFORCE 式），配合 MSE
- 备选 C：退回 pos_hard（BC 式）+ MSE（当前方案）

## 与 1 头实验的关系

当前也在测 `--num-heads 1`（1 头简化动作组）。1 头 + 位置真优化的组合应该一起考虑：
- 1 头减少多头冲突（已验证行为不打架）
- 位置真优化解决"位置不被 PPO 优化"的根本问题

建议：先确定 1 头 vs 3 头（跑 30-50 轮对比），再上位置真优化。
