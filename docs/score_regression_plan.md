# 基于 ActionCatalog 评分的回归训练方案

## 动机

当前 BC/蒸馏训练的监督信号：
- 硬标签 CE：24 类中哪一类被选中（1 bit）
- 软标签 KL：模型输出 logits（24 个值，但来自另一个可能也塌缩的模型）

ActionCatalog 评分提供了第三类信号：**游戏引擎的启发式评估**（24 个基于棋盘状态实时算出的分数）。

## ActionCatalog 评分是什么

每步调用 `ActionCatalog.build(state, player)` 后，引擎对**每个候选动作**算出一个分数，包含：

- 建塔得分：`lane_bonus + pressure * 2.5 - cost * 0.03`
- 升级得分：`tower_type_fit + level * 1.5 + slot_priority * 0.15`
- 武器得分：`storm_value / emp_value / deflector_value / evasion_value`
- 等等

这些分数编码了：敌人压力、兵线价值、位置优先级、成本收益比等**真实的棋盘特征**。

## 方案

### 数据收集（改 `_eval_worker`）

每步对弈时，额外调用一次 `ActionCatalog`：

```
# 当前每步已有的数据：
board, stats → 模型 → head_logits(24) + action_map(24×19×19)

# 新增：
catalog = ActionCatalog(...)
bundles = catalog.build(state, our_player)
class_scores = 聚合到的 24 维分数向量  ← 新增保存
```

聚合方式：每个动作类的候选 bundle 取最高分。没有候选的类得分为 0。

### 损失函数

在现有 BC 损失基础上加一项评分回归损失：

```
loss = λ_class * CE(硬标签) + λ_soft * KL(head_logits) + λ_map * KL(action_map) + λ_score * MSE(模型输出 vs ActionCatalog 评分)
```

其中 `λ_score * MSE` 项：

```
模型输出: softmax(head_logits) / temperature → 概率分布 P (24,)
目标: softmax(ActionCatalog_scores / temperature) → 概率分布 Q (24,)
score_loss = MSE(P, Q) 或 KL(P || Q)
```

### 对 backbone 的影响

ActionCatalog 评分依赖的棋盘特征量：

| 特征 | 涉及函数 | 依赖 |
|---|---|---|
| 敌人压力 | `_local_enemy_pressure` | 敌方蚂蚁位置、等级、距离 |
| 位置优先级 | `slot_priority` | 兵线、坐标 |
| 前线距离 | `frontline_distance` | 双方兵力分布 |
| 塔类型适配 | `_tower_type_fit` | 压力、到敌方距离 |
| 武器价值 | `_storm_value` 等 | 蚂蚁密度、塔密度、命中数 |

每个 `class_scores[i]` 都包含这些特征的不同组合。**backbone 必须学会提取这些特征才能准确预测分数**——这天然解决了 backbone 塌缩问题。

### 为什么不用硬标签

ExampleAI 水平有限，最优动作不一定是它选的那个。但 ActionCatalog 的评分连续且信息丰富——即使 ExampleAI 选了次优动作，分数的相对大小仍然反应了各类的优劣。

## 实现计划

### Step 1: 修改 `_eval_worker` 保存 ActionCatalog 评分

`code/my_ai/_eval_worker.py`：

```python
from SDK.utils.actions import ActionCatalog

# ...在游戏循环中每步...
catalog = ActionCatalog(...)
bundles = catalog.build(state, our_player)
# 聚合成 24 维分数向量
class_scores = np.zeros(24, dtype=np.float32)
for bundle in bundles:
    class_scores[bundle.operations[0].op_type] = max(
        class_scores[bundle.operations[0].op_type], bundle.score
    )
```

保存到 `.npz` 作为新的 field `class_scores`。

### Step 2: 训练加 score loss

`code/my_ai/ss_train.py` 的 `ss_supervised_update`：

```python
if lambda_score > 0 and "class_scores" in batch:
    score_target = batch["class_scores"].to(device)  # (B, 24)
    score_dist = F.softmax(score_target / score_temperature, dim=-1).detach()
    for i in range(model.num_heads):
        pred_dist = F.softmax(output[f"head{i+1}_logits"], dim=-1)
        score_loss += F.mse_loss(pred_dist, score_dist)
    score_loss /= model.num_heads
```

### Step 3: 命令行参数

```bash
--lambda-score 1.0    # MSE loss weight (0 = disabled)
```

## 与现有损失的关系

| 损失 | 信号 | 提供者 | 包含局面信息 |
|---|---|---|---|
| `--lambda-class` | 硬标签 | ExampleAI 的选择 | 极少（1 bit） |
| `--lambda-soft` | 模型 logits | 被蒸馏模型 | 中等（24 维分布） |
| **`--lambda-score`** | **启发式评分** | **ActionCatalog** | **丰富（编码了压力/距离/成本）** |
| `--lambda-map` | 位置图 | 模型的放置偏好 | 空间信息 |

推荐起步配置：
```bash
--lambda-class 0 --lambda-soft 0.5 --lambda-score 1.0 --lambda-map 1.0
```
