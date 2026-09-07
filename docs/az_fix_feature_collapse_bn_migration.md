# 修复特征提取器坍缩：no_bn → BatchNorm bias 迁移方案

> 日期：2026-08-18。状态：计划阶段。目标：让 az 训练线不再把特征提取器训坍缩。

## 1. 背景：最终根因

az 训练线（batch 30-35 起）把特征提取器训坍缩了——`spatial_feat`（resblocks 输出）
对任何棋盘局面输出相同特征图（跨样本 std=0），模型只能靠 stats（HP/金币）决策，
棋盘信息全丢。详见 `docs/az_training_validation_results.md` §20。

**数值机制**：`no_bn=True`（BatchNorm → Identity）下，az 训练把 resblocks[0-2] 的
权重推到"放大激活"方向（std 从健康的 0.5-0.7 涨到 6-8，max 达 143），resblocks[3]
的 ReLU 被大值压死（100% 零激活），特征从此与输入无关。gen0120 源头模型健康
（同样 no_bn 结构，std 稳定 0.5-0.7），证明结构本身可行，是**训练时激活失控**。

## 2. 方案：bias 迁移（无损初始等价）

把 no_bn 结构迁移到带 BatchNorm 结构，conv 的 bias 搬进 BN 的 β，BN 初始化为恒等。

**no_bn ResBlock（当前）**：
```
conv2d(bias=b) → ReLU → conv2d(bias=b) → +residual → ReLU
```

**带 BN ResBlock（目标）**：
```
conv2d(bias=False) → BatchNorm → ReLU → conv2d(bias=False) → BatchNorm → +residual → ReLU
```

**转换规则**（对每个 conv 对，initial_conv 同理）：
1. conv 权重原样复制
2. 原 conv 的 `bias` 存入对应 BN 的 `beta`（β）
3. BN 的 `gamma=1, running_mean=0, running_var=1`（恒等起点）

**数学等价**：初始 `BN(x) = γ·(x−μ)/σ + β = 1·(x−0)/1 + bias = x + bias`，
正好等于原 `conv(x)+bias` → **转换后 eval 输出与原模型逐位相同**。

**为何不直接改 no_bn=False 重训**：现有权重（含价值头、policy 头）是 no_bn 训练出的
数百万参数，直接换结构全部作废。bias 迁移保留所有权重价值，只补上缺失的归一化能力。

## 3. 需迁移的层

按 network.py 结构（no_bn=True 分支），所有 `bias=True` 的层：

| 层 | 原结构 | 迁移目标 |
|----|--------|---------|
| `initial_conv` | Sequential[Conv(bias=True), ReLU] | Sequential[Conv(bias=False), BN, ReLU] |
| `resblocks[i].conv1` | Conv(bias=True) | Conv(bias=False) + BN1 |
| `resblocks[i].conv2` | Conv(bias=True) | Conv(bias=False) + BN2 |

注意 ResBlock 的残差连接在 conv2+BN 之后、ReLU 之前，迁移后：
```
x = conv1(x); x = bn1(x); x = relu(x)
x = conv2(x); x = bn2(x); x += residual; x = relu(x)
```
与 no_bn 版（bn=Identity）逐层对齐，语义一致。

**不动**：stats_mlp、policy_base、policy_heads、action_map_conv、value_head
（这些不是 ResBlock 卷积，无激活爆炸问题）。

## 4. 实现

新脚本 `code/my_ai/az_intent/convert_no_bn_to_bn.py`：
- 加载 no_bn checkpoint（单模型 or split 都支持）
- 构建 `no_bn=False` 的同结构模型
- 逐层迁移：conv.weight 复制；conv.bias → 对应 BN.beta；BN gamma=1/mean=0/var=1
- 保存为同格式 checkpoint（model_state/value_state + no_bn=False 标记）

## 5. 验证计划

1. **逐位等价**：转换后模型 vs 原模型，对多个局面输入，action_map/head_logits/value
   输出差 < 1e-6
2. **特征坍缩测试**：转换后的模型（尚未训练）spatial_feat 跨样本 std 应保持
   gen0120 的量级（>1e-3，正常）
3. **短训验证**：从转换后的 gen0120_warm_cpp_bn 起，训 5-10 batch，验证 spatial_feat
   不再坍缩（std 保持 >1e-3），且 policy loss 能正常下降
4. 若通过，用它作为新训练线起点，跑正式 batch

## 6. 风险

- 训练模式下 BN 用 batch 统计，第一个 batch 行为即与 no_bn 不同（预期：归一化开始生效）
- BN 的 momentum/eps 用默认值（0.1 / 1e-5）；若激活仍爆炸可调小 lr 或加 grad clip
- conv 权重初始偏大（std 0.15-0.23）可能让前几个 batch 的 BN 统计震荡，可用小 lr 起步

## 7. 一句话总结

**把 no_bn 模型的 conv bias 迁移到等价的 BatchNorm β（BN 恒等初始化），得到结构兼容、
初始逐位等价的带 BN 模型——补上归一化能力但不丢任何已学权重，阻断 az 训练的激活爆炸**
**导致的特征坍缩。**
