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

## 数据收集实现（独立程序）

不复用 `_eval_worker`，单独写一个数据收集脚本 `code/my_ai/collect_scores.py`。

**原因：**
1. 每步计算 24×19×19 个格子的 ActionCatalog 评分计算量大，不适合在 GA 评估循环中做
2. 可以用更强的 AI（如 ExampleAI）而不是当前种群来生成数据
3. 数据收集是一次性批处理任务，和 GA 训练解耦

**流程：**

```python
# collect_scores.py：用 ExampleAI 对弈，每步存 action_map + score_map
import numpy as np
from SDK.backend.engine import GameState
from SDK.utils.actions import ActionCatalog, ActionBundle
from AI.ai_example import AI as ExampleAI
from code.my_ai.decoder import make_position_masks, decode_single_cell
from SDK.utils.constants import MAX_ROUND

NUM_CLASSES = 24
MAP_SIZE = 19
catalog = ActionCatalog()

for seed in range(n_games):
    state = GameState.initial(seed=seed, ...)
    # 每步收集
    boards, statss, score_maps, action_maps = [], [], [], []
    for _ in range(MAX_ROUND):
        if state.terminal: break
        # 用 ExampleAI 走棋（产生数据）
        example_ops = example_ai.choose_operations(state, player)
        # 记录棋盘
        feat = feature_extractor.encode_observation(state, player, ...)
        boards.append(feat["board"])
        statss.append(feat["stats"])
        # 用 ExampleAI 的 action_map 作为模型输出（或留空让训练时算）
        action_maps.append(expert_action_map)

        # 计算 score_map
        pos_mask = make_position_masks(state, player)
        score_map = np.zeros((NUM_CLASSES, MAP_SIZE, MAP_SIZE))
        for class_id in range(NUM_CLASSES):
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    if not pos_mask[class_id, x, y]:
                        continue
                    op = decode_single_cell(class_id, x, y, state, player)
                    if op is not None:
                        score_map[class_id, x, y] = catalog.score_operation(...)
        score_maps.append(score_map)

        state.resolve_turn(example_ops, opp_ops)

    # 存 npz
    class_scores = np.array(score_maps).reshape(-1, NUM_CLASSES, MAP_SIZE**2).max(axis=-1)
    np.savez_compressed(path, board=..., stats=..., score_map=..., class_scores=...)
```

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

1. `decoder.py` 添加 `decode_single_cell(class_id, x, y, state, player) → Operation | None`
2. `ActionCatalog` 添加 `score_operation(op, state, player) → float`
3. 独立程序 `code/my_ai/collect_scores.py` 收集 `score_map` + `class_scores` 存入 npz
4. `SSDataset` 加载新字段 `score_map`, `class_scores`
5. `ss_supervised_update` 新增 score loss + position loss
6. ga_ss_train CLI 参数 `--lambda-score`, `--lambda-pos`
7. 用收集的数据跑蒸馏训练，替代当前的 BC 训练
