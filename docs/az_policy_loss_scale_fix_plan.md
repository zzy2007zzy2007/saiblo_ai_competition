# 策略损失尺度不对称修复方案

> 日期：2026-08-12。状态：计划阶段。目标：修复 position CE 用未归一化 action_map 导致策略训练失效。

## 1. 背景：策略训练"学不动"的根因

batch71-85 训练后实测（消融实验，128 样本过拟合 30 epoch）：

| 配置 | 收敛 CE |
|------|--------|
| 标准 softmax CE（class 部分，单 head）| 0.029 |
| 当前 class CE（z-score + /0.5，单 head）| 0.064 |
| 当前完整实现（3 head class CE）| 0.216 |
| **当前完整实现（class + position CE）** | **1.393** |

结论：position CE 让策略训练从 0.216 恶化到 1.393——**完全学不动**。
（顺带确认：class CE 的 z-score+温度设计本身是好的，0.064 优于标准 CE 0.029 之外的其他配置，保留不动。）

## 2. 根因：position CE 用了未归一化的 action_map

同一个 `_sample_policy_loss` 函数里，两个部分处理方式不对称：

| 部分 | 输入 | 归一化 | 尺度 |
|------|------|--------|------|
| class CE | head_logits | z-score + /t_class | std≈1（正常）|
| **position CE** | action_map | **无归一化，只 /t_pos=0.3** | **-44~+80，std=23** |

- action_map 原始 logits 范围 -44.4~+79.5（std=23.2），`/0.3` 放大 3.3 倍更极端
- `logsumexp` 对 -95~-89 这种值计算，单 position 项 loss 贡献实测 **±5.5**（如 `pos_logit=-95.29, pos_lse=-89.74`）
- class CE 项只有 0.05~0.2 → **position 一项比 class 全部项大 27 倍**，梯度被 position 项主导，共享的 action_map_conv 被拉偏，整个策略头训练被破坏

## 3. 采样侧已有正确做法（对称基准）

decoder.py `_sample_position` 对 position 用了**固定尺度归一化**：

```python
POS_SCALE = 100.0
scaled = channel_map / POS_SCALE      # ÷100，尺度无关
logits = masked / pos_temperature
```

而训练侧 position CE 用的是 raw action_map / t_pos——**采样侧 ÷100，训练侧不归一化，两侧不对称**。
采样侧刻意用固定 ÷100 而不是 z-score（注释：不依赖数据统计），训练侧应与其对称。

## 4. 修改方案

**只改 `code/my_ai/az_intent/az_train.py` 的 `_sample_policy_loss` position CE 分支**，采样侧
（decoder.py / bundle_mcts.py）不动：

```python
# 修改前
am_c = out["action_map"][b, c] / t_pos

# 修改后：与采样侧 _sample_position 对称（固定尺度 ÷POS_SCALE）
POS_SCALE = 100.0
am_c = out["action_map"][b, c] / POS_SCALE / t_pos
```

- `POS_SCALE = 100.0` 与 decoder.py 常量保持一致（也可从 decoder 导入）
- `t_pos` 默认值从 0.3 调回 **1.0**（÷100 后 action_map 已是 ~0.2 量级的小值，/0.3 不再必要；1.0 是采样侧同款中性起点）
- class CE 分支完全不动（z-score + t_class 已验证是好的）

## 5. 其他待验证点

- **t_class 默认 0.5 是否保留**：class CE 单 head 收敛 0.064 已可用，但 3 head 冲突（0.216）仍在。
  本次不处理多 head 冲突（那是另一个问题：共享 policy_base 被 3 个不同 target 拉扯），
  先确认 position 修复后 3 head 能否降到 0.1 量级。
- **anchor 是否要调**：anchor 实测梯度只有 CE 的 2.3%，调 0 无影响，本次不动。

## 6. 验证计划

1. 128 样本过拟合 30 epoch：修复后完整实现（3 head class + position CE）应显著低于 1.393
2. 用 batch85 数据 + az_r84 init 跑 5 epoch 完整训练，确认 policy CE 随 epoch 下降
3. 若收敛正常，启动 1 个新 batch 训练（batch89），search-vs-search + vs rule_v4 对比 az_r85

## 7. 一句话总结

**position CE 用未归一化的 action_map（-44~+80）直接算 softmax，单项 loss 比 class CE 全部项大 27 倍，
主导并破坏梯度。修复：与采样侧 decoder 的固定 ÷100 归一化对称，action_map ÷POS_SCALE=100 再 /t_pos，t_pos 调回 1.0。**
