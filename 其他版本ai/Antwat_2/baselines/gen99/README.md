# gen99 基线策略说明

## 技术思路

gen99 是基于 ActionCatalog 的智能体，通过加载参数文件来配置动作选择策略。该策略使用 ActionCatalog 来评估和选择动作，基于动作的得分和操作数量来做出决策。

## 主要架构

### 核心组件

- **ActionCatalog**：用于评估和选择动作的目录，加载参数文件进行配置
- **ModifiedAgent**：继承自 BaseAgent，实现动作选择逻辑
- **参数文件**：params.json，存储策略配置参数

### 输入输出

- **输入**：游戏状态（BackendState）和玩家ID
- **输出**：选择的动作束（ActionBundle）

## 主要方法

1. **初始化**：加载参数文件并初始化 ActionCatalog
2. **动作选择**：评估所有可能的动作，选择得分最高且操作数量最少的动作
3. **参数配置**：通过 params.json 文件配置策略参数

## 关键文件

- **ai.py**：智能体主类，实现与游戏环境的交互
- **params.json**：策略配置参数文件
- **common.py**：基础组件

## 评估指标

- **动作选择效率**：基于动作得分快速选择最优动作
- **策略适应性**：通过参数文件配置适应不同游戏场景
- **决策稳定性**：基于明确的评分标准做出决策

## 使用方法

```python
from baselines.gen99.ai import create_agent

# 创建智能体
agent = create_agent()

# 在游戏循环中使用
bundle = agent.choose_bundle(state, player)
```