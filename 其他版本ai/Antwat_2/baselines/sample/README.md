# Sample 基线策略说明

## 策略简介

Sample策略是一个基础的参考策略，展示了如何使用SDK的后端能力来选择动作。它是一个简单但完整的智能体实现，可作为开发更复杂策略的起点。

## 策略特点

- **基础实现**：展示了SDK的基本使用方法
- **动作选择**：从候选动作包中选择分数最高的动作
- **简单有效**：逻辑清晰，易于理解和扩展

## 核心逻辑

```python
def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
    bundles = bundles or self.list_bundles(state, player)
    if len(bundles) <= 1:
        return bundles[0]

    # 选择分数最高且操作数量最少的动作包
    shortlist = bundles[1 : min(len(bundles), 8)]
    best = max(shortlist, key=lambda bundle: (bundle.score, -len(bundle.operations)), default=None)
    return best or bundles[0]
```

## 工作原理

1. **获取候选动作**：通过`list_bundles`方法获取所有可能的动作包
2. **动作筛选**：从候选动作中选择前8个动作（排除第一个noop动作）
3. **动作评估**：根据动作包的分数和操作数量进行排序
4. **选择最佳动作**：返回分数最高、操作最少的动作包

## 优势与局限性

### 优势
- 实现简单，易于理解
- 利用SDK提供的分数评估机制
- 可以作为开发更复杂策略的基础

### 局限性
- 策略过于简单，缺乏深度思考
- 没有考虑长期规划
- 仅基于当前状态选择动作，没有考虑未来影响

## 使用方法

```bash
# 运行本地对战
cd tools
python run_local_match.py

# 训练模型（如果需要）
cd SDK
bash train_mcts.sh
```

## 扩展建议

1. **添加长期规划**：考虑动作对未来状态的影响
2. **引入机器学习**：使用MCTS或神经网络来评估动作
3. **优化动作选择**：根据不同游戏阶段采用不同的策略
4. **添加防御和进攻策略**：根据游戏状态调整防御和进攻比例

## 注意事项

- 该策略是一个基础参考实现，实际比赛中需要根据具体情况进行优化
- SDK提供了丰富的工具和方法，可以根据需要使用更多功能
- 动作包的分数是由后端根据规则计算的，可以作为参考但不是绝对的最佳选择
