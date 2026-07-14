# 基于 ActionCatalog 评分的回归训练方案

## 动机

当前 BC/蒸馏训练的监督信号：
- 硬标签 CE：24 类中哪一类被选中（1 bit）
- 软标签 KL：模型输出 logits（24 个值，但来自另一个可能也塌缩的模型）

ActionCatalog 评分提供了第三类信号：**游戏引擎的启发式评估**（基于棋盘状态实时算出的分数），且评分天然从棋盘特征（压力、距离、成本等）计算而来，模型要拟合评分就必须学会提取这些特征，从而解决 backbone 塌缩。

## ActionCatalog 评分是什么

每步调用 `ActionCatalog.build(state, player)` 后，引擎对每个候选动作算出一个分数。每个分数都依赖棋盘特征量：

| 特征 | 涉及函数 | 依赖 |
|---|---|---|
| 敌人压力 | `_local_enemy_pressure` | 敌方蚂蚁位置、等级、距离 |
| 位置优先级 | `slot_priority` | 兵线、坐标 |
| 前线距离 | `frontline_distance` | 双方兵力分布 |
| 塔类型适配 | `_tower_type_fit` | 压力、到敌方距离 |
| 武器价值 | `_storm_value` 等 | 蚂蚁密度、塔密度、命中数 |

## 核心思路

对 action_map 的每个格子，用 decoder 解码出真实游戏动作 → 交给 ActionCatalog 评分 → 得分为该格子的标签。

```
for class_id in 0..23:
    for (x, y) in 全部 19×19 格子:
        # 1. decode: (class_id, x, y) → 具体游戏动作 或 HOLD
        op = decode_cell(class_id, x, y, state)

        # 2. ActionCatalog 评分
        if op 是 HOLD/非法:
            score = 0.0
        else:
            score = catalog.score_single_operation(op, state)

        # 3. 存入位置分图
        score_map[class_id, x, y] = score

    # 4. 类标签 = 该类 score_map 的最大值
    class_scores[class_id] = score_map[class_id].max()
```

该方案解决了 `ActionCatalog.build()` 的三个问题：

| 问题 | `build()` 的做法 | 本方案 |
|---|---|---|
| 截断 | 只返回 top 64 个 bundle | 24×19×19 全部格子都算 |
| Combo 歧义 | 多操作 bundle 分不清归哪个类 | 一个格子只解码成一个动作 |
| 缺失类 | 超武 CD 时无候选 | 所有类都有分，不合法就是 0 |

## 与模型输出的结构对齐

模型输出天然有两部分：

| 模型输出 | 形状 | 对应标签 |
|---|---|---|
| `head{i+1}_logits` | (24,) | `class_scores` (24,) — 每类最高位置分 |
| `action_map` | (24, 19, 19) | `score_map` (24, 19, 19) — ActionCatalog 位置评分 |

两个损失函数：

```
loss = λ_score * MSE(softmax(head_logits), softmax(class_scores))
     + λ_pos   * MSE(action_map, score_map)
```

## 数据收集实现（改 `_eval_worker`）

```python
from SDK.utils.actions import ActionCatalog, ActionBundle
from code.my_ai.decoder import make_position_masks

catalog = ActionCatalog()
position_mask = make_position_masks(state, our_player)

score_map = np.zeros((NUM_CLASSES, 19, 19), dtype=np.float32)

for class_id in range(NUM_CLASSES):
    for x in range(19):
        for y in range(19):
            if not position_mask[class_id, x, y]:
                continue
            # decode → 具体动作
            op = decode_cell(class_id, x, y, state, our_player)
            if op is None:
                score_map[class_id, x, y] = 0.0
            else:
                # 用 ActionCatalog 给这个动作打分
                bundle = ActionBundle(operations=(op,))
                score = catalog._score_bundle(bundle, state, our_player)
                score_map[class_id, x, y] = score

# 类分 = 位置分取 max
class_scores = score_map.reshape(NUM_CLASSES, -1).max(axis=1)
```

其中 `_score_bundle` 是根据 ActionCatalog 中各评分方法聚合出的单动作评分函数。

## 训练实现

`ss_supervised_update` 中新增：

```python
if lambda_score > 0 and "class_scores" in batch:
    score_target = batch["class_scores"].to(device)  # (B, 24)
    score_dist = F.softmax(score_target / temperature, dim=-1).detach()
    for i in range(model.num_heads):
        pred_dist = F.softmax(output[f"head{i+1}_logits"], dim=-1)
        score_loss += F.mse_loss(pred_dist, score_dist)
    score_loss /= model.num_heads

if lambda_pos > 0 and "score_map" in batch:
    pos_target = batch["score_map"].to(device)  # (B, 24, 19, 19)
    pos_dist = F.softmax(pos_target.reshape(B, 24, -1), dim=-1).reshape(B, 24, 19, 19).detach()
    pos_loss = F.mse_loss(output["action_map"], pos_dist)
```

## 推荐配置

```bash
# 纯评分回归（不用硬标签和蒸馏）
--lambda-class 0 --lambda-soft 0 --lambda-score 1.0 --lambda-pos 1.0 --lambda-map 0.5

# 混合模式
--lambda-class 0 --lambda-soft 0.3 --lambda-score 1.0 --lambda-pos 1.0 --lambda-map 0.5
```

## 实现步骤

1. `decoder.py` 添加 `decode_cell(class_id, x, y, state, player) → Operation | None`
2. `ActionCatalog` 添加 `score_operation(op, state, player) → float`
3. `_eval_worker` 收集 `score_map` + `class_scores` 存入 npz
4. `SSDataset` 加载新字段
5. `ss_supervised_update` 新增 score loss + position loss
6. CLI 参数 `--lambda-score`, `--lambda-pos`
