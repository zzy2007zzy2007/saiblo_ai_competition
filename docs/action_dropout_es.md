# 动作 Dropout 训练方案

## 1. 动机

当前 BC + ES 训练中，模型快速收敛到"只放 Thunder"的单一策略。我们的分析认为：这不是 Thunder 本身太强，而是 Thunder 消耗 90 金币，任何其他行为（建塔、升级等）都会占用金币预算，降低 Thunder 频率，从而导致胜率下降。

要想学会"建塔→拆塔回收金币→Thunder"这样的多步策略，模型必须先经历"建了塔但不会拆 → 没钱放 Thunder → 胜率低"的中间态。但在 ES 训练中，这个中间态的 fitness 太低，模型根本走不过去。

## 2. 核心思路：动作 Dropout

在 ES 评估（打局）时，以概率 p 将当前动作替换为随机合法动作。这样：

- **只会 Thunder 的模型**：10-20% 的回合被随机动作干扰（建塔消耗金币），Thunder 频率下降 → 没钱放 Thunder → 胜率暴跌
- **学会恢复的模型**：被随机建塔后，能主动拆塔回收金币，再继续放 Thunder → 胜率回升
- **终极目标**：模型被迫学会"不管资源被怎么折腾，都能恢复并赢下比赛"的鲁棒策略

### 类比

这类似于神经网络的 Dropout——随机丢弃神经元防止过拟合。这里是在**动作层面做 dropout**，随机覆盖模型的部分决策，迫使策略不依赖"所有操作都在自己控制下"的假设。

## 3. 实现方案

### 3.1 改动范围

只需修改 `es_train.py` 中的 `_eval_worker` 函数（+ 新增一个 CLI 参数），约 30 行代码。

### 3.2 随机动作生成

在合法动作中完全均匀随机选一个：

```python
def random_legal_action(state: BackendState, player: int) -> ActionBundle:
    """从所有合法bundle中均匀随机选一个"""
    from SDK.utils.actions import ActionCatalog
    catalog = ActionCatalog(state, player)
    bundles = catalog.list_bundles()
    if not bundles:
        return ActionBundle(name="hold", score=0.0, tags=("noop",))
    return random.choice(bundles)
```

### 3.3 评估过程改动

```python
def _eval_worker(params, opp_params, seed, ...):
    ...
    for turn in range(max_turns):
        if random.random() < action_dropout_p:  # ← 新增
            my_bundle = random_legal_action(state, player)
        else:
            my_bundle = agent.choose_bundle(state, player)  # 正常决策
        ...
```

### 3.4 CLI 参数

```bash
python code/my_ai/es_train.py ... --action-dropout 0.15
```

- `--action-dropout`：随机替换概率，默认 0.0（不启用）
- 可选范围 0.05-0.30，建议初始用 0.10

### 3.5 关键细节

| 问题 | 方案 |
|------|------|
| 对手也要 dropout 吗？ | **不**，只有本方 dropout。对手保持正常策略 |
| 随机后仍合法吗？ | 是的，`ActionCatalog.list_bundles()` 只返回合法 bundle |
| 随机动作会选 HOLD 吗？ | 有可能，但如果 HOLD 在 bundle 中占比大，随机到 HOLD 概率也大 |
| 训练完成后推理时？ | **关闭** dropout（p=0），用模型正常决策 |

## 4. 预期效果

### 理想学习路径

1. **阶段 0**（纯 Thunder 模型 + dropout=0.15）：15% 回合被随机建塔消耗金币 → Thunder 频率降 ~15% → 胜率下降 → fitness 低
2. **阶段 1**：ES 压力下，模型逐渐学会"偶尔拆塔"来回收被随机消耗的金币 → 胜率部分恢复
3. **阶段 2**：模型学会主动"建塔建设 → 拆塔回收 → Thunder"的策略组合 → 资源分配更智能 → 胜率超过纯 Thunder

### 预期 fitness 曲线

```
胜率
 ↑
0.5 ┤         ┌─── 阶段2: 学会恢复 + 主动多策略
0.4 ┤    ┌────╯
0.3 ┼────┤    ← 纯Thunder基线（~30% vs rule_v4?）
0.2 ┤    │  ← 阶段0-1: dropout干扰导致胜率下降
0.1 ┤    └──────── 模型挣扎期
0.0 └─────────────────────────────→ 世代
```

## 5. 风险与缓解

| 风险 | 缓解 |
|------|------|
| dropout 太大，永远学不会 | 从 0.10 起步，观察 fitness 是否回升 |
| dropout 太小，没效果 | 0.05 是最低有效值，调大即可 |
| 模型学会"对抗随机"而非"更好策略" | 如果模型只能适应随机，但对正常对手反而变弱，就说明这个方向有问题 |
| 随机动作中 HOLD 占比高，实际干扰小 | 如果发现随机到 HOLD 概率超过 30%，可以考虑随机时排除 HOLD |

## 6. 在多专家方案中的定位

注意：这个方案**不是替代**多专家方案，而是和它**互补**的关系：

- **多专家方案**：架构解法——分而治之，消除金币竞争
- **动作 Dropout**：训练技巧——迫使模型学会鲁棒性

两者可以独立使用，也可以结合：
- 先用动作 dropout 在普通 ES 上试试，看能否突破全 Thunder 陷阱
- 如果有效，多专家的每个专家训练时也可以加上 dropout
- 最终门控网络组合各专家时，每个专家本身就更鲁棒

## 7. 实现步骤

1. 在 `es_train.py` 的 `EvalConfig` 中加 `action_dropout: float = 0.0`
2. 实现 `random_legal_action()` 辅助函数
3. 在 `_eval_worker` 中插入 dropout 逻辑
4. 在 `_eval_worker_for_pop` 的 config 传参 pipeline 中透传
5. 添加 `--action-dropout` CLI 参数
6. 先测试 p=0（不改变现有行为）
7. 再用 p=0.10 跑短训练（10-20 代）观察效果
