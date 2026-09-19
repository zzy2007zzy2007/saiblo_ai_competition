# 基于胜率的公平赋分策略设计文档

## 1. 设计背景

在强化学习中，奖励信号的设计直接影响模型的学习效果。当面对不同强度的对手时，使用统一的奖励分值会导致以下问题：

- **奖励信号扭曲**：无法区分击败强对手和弱对手的价值差异
- **学习目标不明确**：模型无法感知对手的相对强度，导致学习策略失衡
- **探索与利用失衡**：可能导致模型过度避免与强对手对战，影响整体性能

## 2. 设计目标

设计一个基于胜率的公平赋分策略，实现：

- **正分策略**：击败强对手获得高奖励，击败弱对手获得低奖励
- **负分策略**：输给强对手获得小惩罚，输给弱对手获得大惩罚
- **公平性**：基于客观胜率数据，避免主观因素影响
- **平滑性**：奖励值随对手强度平滑变化，避免突变

## 3. 设计原理

### 3.1 相对强度评估

1. **平均胜率计算**：每个策略对其他所有策略的胜率平均值
2. **强度归一化**：将平均胜率映射到[0,1]区间，形成相对强度得分
3. **强度排序**：根据相对强度对策略进行排序，确定强度等级

### 3.2 正分策略

- **基本原理**：击败越强的对手，获得的奖励应该越高
- **计算方法**：
  ```python
  positive_reward = k * f(opponent_strength)
  ```
  其中k是正分系数，f(opponent_strength)是强度函数

### 3.3 负分策略

- **基本原理**：输给越强的对手，获得的负分应该越小
- **计算方法**：
  ```python
  negative_reward = -k * g(opponent_strength)
  ```
  其中k是负分系数，g(opponent_strength)是强度函数

### 3.4 强度函数选择

采用对数缩放函数作为强度函数，原因：
- **平滑性**：对数函数提供平滑的非线性映射
- **边界控制**：避免极端值，确保奖励在合理范围内
- **数学性质**：符合直觉预期

### 3.5 课程学习设计

- **基本原理**：从简单对手开始训练，逐步过渡到复杂对手
- **训练阶段划分**：
  - **阶段1**（0-100轮）：仅与简单策略对战
  - **阶段2**（101-300轮）：与中等强度策略对战
  - **阶段3**（301-5300轮）：与所有策略对战
- **对手选择策略**：
  - 阶段1：100% 简单策略
  - 阶段2：40% 简单策略，30% 中等策略，30% 中等策略
  - 阶段3：均匀分布所有策略
- **优化理由**：
  - 第100轮时模型对BasicTowerAI的胜率已达100%
  - 继续训练BasicTowerAI可能导致过度训练
  - 提前进入复杂对手训练可以节省训练时间

### 3.6 自适应难度调整

- **基本原理**：根据训练性能动态调整对手强度
- **调整机制**：
  - 实时评估模型性能
  - 性能好则增加强对手比例
  - 性能差则增加弱对手比例
- **实现方法**：
  ```python
  def get_opponent_probabilities(episode, performance_history):
      # 根据最近性能调整对手分布
      recent_performance = np.mean(performance_history[-10:])
      if recent_performance > 0:
          # 增加强对手概率
          base_probs["BasicTowerAI"] *= 0.5
          base_probs["Gen99"] *= 1.2
          # 归一化概率
          total = sum(base_probs.values())
          return {k: v/total for k, v in base_probs.items()}
  ```

### 3.7 自博弈训练

- **基本原理**：让模型与自身的不同版本对战，促进策略进化
- **实现方法**：
  - 定期保存模型的不同版本
  - 让当前模型与历史最佳模型对战
  - 维护一个策略池，包含多个版本的模型
- **优势**：
  - 促进策略多样性
  - 避免过度拟合特定对手
  - 提高泛化能力

## 4. 实现步骤

### 4.1 胜率统计系统

1. **策略对战**：让所有baseline策略相互对战
2. **胜率记录**：记录每对策略之间的对战结果
3. **数据保存**：将胜率数据保存为JSON文件

### 4.2 奖励计算系统

1. **加载胜率数据**：从JSON文件加载胜率数据
2. **强度计算**：计算每个策略的相对强度
3. **奖励计算**：根据对手强度计算奖励值

### 4.3 课程学习实施

1. **训练阶段配置**：设置训练阶段的轮次范围和对手分布
2. **阶段切换**：在训练过程中根据当前轮次切换训练阶段
3. **对手选择**：根据当前训练阶段和自适应调整机制选择对手

### 4.4 自适应调整实现

1. **性能评估**：定期评估模型对不同强度对手的表现
2. **难度调整**：根据评估结果调整对手分布
3. **参数优化**：根据训练反馈调整学习率等超参数

## 5. 预期效果

- **更合理的奖励信号**：击败强对手获得更高奖励，击败弱对手获得较低奖励
- **更有效的学习**：模型会优先学习如何击败强对手，提高整体性能
- **更平衡的训练**：避免模型过度专注于击败弱对手
- **更好的泛化能力**：模型会学习到更通用的策略，而不是针对特定对手的策略
- **更稳定的训练曲线**：课程学习提供平滑的学习路径，避免训练崩溃
- **更快的收敛速度**：从简单到复杂的训练顺序加速学习过程
- **更强的策略适应性**：自适应难度调整使模型能够应对不同强度的对手
- **更高的策略多样性**：自博弈训练促进策略进化，避免过度拟合

## 6. 代码结构

- `tests/evaluation/compare_strategies.py`：统计baseline策略之间的胜率（现有文件）
- `tests/evaluation/reward_calculator.py`：基于胜率计算奖励值（新文件）
- `tests/evaluation/win_rates.json`：保存胜率数据
- `ppo_v1/reward_calculator.py`：PPO训练中的奖励计算模块（新文件）
- `ppo_v1/training_strategy.md`：PPO训练策略详细设计（新文件）
- `ppo_v1/reward_system.md`：奖励系统详细设计（新文件）

## 7. 参考资料

- CSDN博客：强化学习奖励函数设计经验
- arXiv:2502.18770v3：Reward Shaping to Mitigate Reward Hacking in RLHF
- EMNLP 2024：Optimizing Language Models with Fair and Stable Reward Composition in Reinforcement Learning
- arXiv:2506.07548：Curriculum Learning With Counterfactual Group Relative Policy Advantage For Multi-Agent Reinforcement Learning
- FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning
- The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
- Distributed Reinforcement Learning with Self-Play