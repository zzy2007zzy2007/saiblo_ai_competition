# 蒸馏评分函数方案设计

## 背景

之前的 diagnose 日志显示，ss_train 训练出来的模型各 head 在整局游戏中输出的 class 几乎固定不变（H1 永远是 HOLD、H2/H3 永远是 LIGHTNING），完全不随棋盘变化。这说明模型没有学会利用局面信息做决策。

根本原因在于 ss_train 的 BC 变异是**自我指涉的**——用模型自己的 head_logits 作为标签训练自己，如果模型已经不看棋盘了，BC 标签也来自不看棋盘的状态，形成闭环。

**外部监督信号**是打破这个闭环的关键。最直接的外部信号来源是 SDK 自带的启发式动作评分系统——`ActionCatalog.build()` 能为每个合法操作计算一个 heuristic score，这个 score 是局面相关的，包含了棋盘信息。

## 方法总览

### BC 方案的局限

最初的设计是用行为克隆（BC）模仿 ExampleAI 的选择：
- 每回合记录 ExampleAI 选中的操作 → 1 个 hard label（one-hot class）
- 信息量极低：每回合只告诉模型"这个操作好"，没说"别的操作有多差"
- 而且 decoder 是 intent-level 的，同一个 class 在不同局面下产生不同操作，直接把操作→class 做硬映射会有歧义

### 新方案：评分函数拟合

核心思路：不用模仿 ExampleAI 的 argmax 选择，而是直接用 SDK 的评分函数为**所有**合法操作打分，然后用 decoder 把这些分数投影到模型的 (class, position) 输出空间。

```
ActionCatalog.build()  →  所有操作的评分  →  decoder 反向投影  →  (class目标, position目标)
```

这样每回合产生几十个监督信号（而不是 1 个），而且这些信号天然包含局面信息——评分函数会根据棋盘状态给不同操作打不同分数，模型必须学会看棋盘才能预测这些分数。

## 数据采集

### 采集流程

每回合执行：

```
1. state.encode(board, stats)  ← 局面编码（与训练时一致）
2. bundles = ActionCatalog.build(state, player, max_actions=9999, rerank=False)
   → 获得所有候选 bundle，每个 bundle 包含 (operation, score)
3. 构建操作→分数查表: {(op_type, arg0, arg1) → score}
4. 遍历 24 个 class × 可达位置，对每个 (class, x, y):
   运行 decoder 逻辑 (_decode_tower_action / 武器逻辑等)
   → 得到实际产生的 Operation
   → 查表得到该 Operation 的分数（查不到则为 0 或极小值）
5. 构建训练目标:
   class_target[class] = max_{x,y} score(class, x, y)      ← 24 维
   map_target[class, x, y] = score(class, x, y)             ← 24×19×19
6. 存储: board, stats, class_target, map_target
```

### `rerank=False` 的理由

`_rerank_with_one_step_rollout` 会 clone state 做 rollout 重新排名，计算成本高且只影响前 `max_actions` 个 bundle 的相对排序。数据采集阶段直接用原始启发式评分就够（信息量已经远超硬标签 BC）。

### `max_actions=9999` 的理由

默认 `max_actions` 值（通常 50~100）会导致 `build()` 在第 69 行截断 bundle 列表，丢失一部分候选操作的评分。设一个很大的值保证所有候选生成器的输出都被保留。

### 存储格式

```
每局一个 .npz 文件:
  board:        (T, 28, 19, 19) float16     ← 局面编码
  stats:        (T, 42) float16             ← 统计数据
  class_target: (T, 24) float16             ← 每个 class 的最佳分数（用于 class logits 训练）
  map_target:   (T, 24, 19, 19) float16    ← 每个 (class, x, y) 的分数（用于 action_map 训练）
```

### 采集量

建议 500-1000 局，每局约 200-300 回合 → 总计 100K-300K 样本。每回合有数十个带分数的 bundle → 每回合数十个监督信号。

## 目标构建详解

### 操作→分数的查表构建

```python
# bundles 来自 ActionCatalog.build()
score_lookup = {}
for b in bundles:
    if len(b.operations) == 1:  # 只取单操作 bundle
        op = b.operations[0]
        score_lookup[(int(op.op_type), op.arg0, op.arg1)] = b.score
```

忽略双操作组合 bundle（`paired_candidates`），因为它们在实际解码中不会出现（每个 head 只产生一个 Operation）。

### (class, x, y) → 操作的 decoder 模拟

对塔类操作 (class 0-15)：

```python
def simulated_decode(class_id, x, y, state, player):
    """返回 (class_id, x, y) 通过 decoder 实际产生的 operation key."""
    target_type = CHANNEL_TO_TOWER_TYPE[class_id]
    tower = state.tower_at(x, y)

    if tower is None:
        # 空地: 所有 class 0-15 都变成 BUILD_TOWER(x, y)
        return (OperationType.BUILD_TOWER, x, y)

    if tower.player != player:
        return None  # 敌方塔 -> 非法

    # 已有己方塔: 看是否能升级
    step = upgrade_step(tower.tower_type, target_type)
    if step is None:
        return None  # 已经是目标类型或无法升级

    return (OperationType.UPGRADE_TOWER, tower.tower_id, int(step))
```

对超级武器 (class 17-20)：

```python
# 直接映射到对应的 USE_* 操作
sw_type = CHANNEL_TO_SUPER_WEAPON[class_id]
op_type = SUPER_WEAPON_TO_OP_TYPE[sw_type]
return (op_type, x, y)
```

对基地升级 (class 21-22)：

```python
if class_id == 21: return (OperationType.UPGRADE_GENERATION_SPEED, 0, 0)
if class_id == 22: return (OperationType.UPGRADE_GENERATED_ANT, 0, 0)
```

### 多操作 bundle 的处理

`ActionCatalog.build()` 也会生成 `paired_candidates`（双操作组合 bundle）。这些 bundle 的评分不能直接拆成两个单操作评分之和（`score = first.score + second.score * 0.9` 不是一个严格可分解的评分函数）。

处理方式：忽略双操作 bundle，只使用单操作 bundle 的评分。单操作 bundle 已经覆盖了所有操作类型（建塔、升级、拆塔、武器、基地升级），数量也足够（每回合数十个）。

### 分数归一化

不同评分函数（建塔 vs 武器 vs 升级）的分数量纲不同，直接作为软标签训练可能需要归一化。

方案：对每个 class_target 向量做 softmax（带温度 T），得到一个概率分布：

```python
class_probs = softmax(class_target / T)
```

- 温度 T=1.0：标准 softmax
- T>1.0：分布更平滑（保留更多"次优操作"的信息）
- T<1.0：分布更尖锐（接近 argmax）

建议先试 T=1.0，看 loss 曲线再调。

## 训练

### 复用 ss_train 的基础设施

```python
from ss_train import SSDataset, ss_supervised_update

npz_paths = sorted(Path("distill_score_data/").glob("distill_*.npz"))
ds = SSDataset(npz_paths, p_hold=1.0)
# 修改 SSDataset 使其支持读取 class_target / map_target 作为软标签
```

### 损失函数

需要修改 `ss_supervised_update` 以支持软标签：

| 输出 | 目标 | 损失 |
|:----|:----|:-----|
| head_logits (3×24) | class_target (24) | CrossEntropy 或 KL 散度（配合 softmax 目标） |
| action_map (24×19×19) | map_target (24×19×19) | MSE 或 BCE |

class 损失使用 soft 版本的 CrossEntropy：

```python
# logits: 模型输出的 class logits (24,)
# target: 归一化后的 class 分数分布 (24,)
loss_class = -sum(target * log_softmax(logits))  # KL 散度等价形式
```

position 损失使用 MSE：

```python
# map_pred: 模型输出的 action_map (24, 19, 19)
# map_target: 分数图 (24, 19, 19) — 稀疏，大部分位置为 0
loss_map = MSE(map_pred, map_target)
```

### 超参数

| 参数 | 建议值 | 说明 |
|:----|:------|:-----|
| epochs | 10-20 | 软标签比硬标签收敛更快 |
| lr | 1e-3 | 标准学习率 |
| batch_size | 64 | 标准 |
| lambda_class | 1.0 | class 损失的权重 |
| lambda_map | 0.1~0.5 | position 损失的权重（action_map 比较稀疏） |
| T | 1.0 | softmax 温度 |

## 与 ss_train 的集成

训练好的蒸馏 checkpoint 作为 ss_train 的初始均值参数：

```
distill_score.pt → ss_train --checkpoint distill_score.pt --sigma 0.0002 ...
```

由于蒸馏阶段模型已经学会了"看棋盘做决策"，ss_train 的自我对弈 + 变异阶段是在此基础上微调，收敛速度预计比从随机初始化快 5-10 倍。

## 预期效果对比

| 指标 | 硬标签 BC | 评分函数拟合 |
|:----|:---------|:------------|
| 每回合监督信号数 | 1 | 数十个 |
| class 目标信息量 | one-hot（只有 argmax） | 全分布（知道每个 class 的好坏） |
| position 目标信息量 | 单峰（只有最优位置） | 多峰（知道多个位置的好坏） |
| 对 decoder 歧义的处理 | 依赖硬编码映射规则 | 通过 decoder 模拟自动处理 |
| 学到局面关联 | 可能学到（BC 标签有隐式关联） | 必然学到（评分依赖棋盘状态） |
| 数据采集速度 | 快（不调用评分函数） | 略慢（需调用 ActionCatalog） |

## 和蒸馏 rule_v4 的互补

评分函数蒸馏和 rule_v4 BC 蒸馏解决不同的问题：

| 蒸馏方案 | 学什么 | 适合场景 |
|:--------|:------|:---------|
| **评分函数蒸馏**（本方案） | SDK 全局启发式评分 | 覆盖全动作空间，学局面关联 |
| **rule_v4 BC** | rule_v4 的特定策略偏好 | 快速掌握建塔/升级基础策略 |

两个蒸馏模型可以 ensemble（参数平均）作为 ss_train 的初始化。
