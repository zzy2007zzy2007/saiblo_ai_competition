"""original_rule 实验包

实现 ppo_v10/docs/selection/original_rule.txt 中描述的二分种子选拔方案，
并提供完整的验证实验框架。

与 tools/bisection_experiment 的核心差异：
- 初始候选 = 种群第 1 个个体（而非随机 P 个）
- 升级条件：M 个对手后，mu ∈ Top33%-Top66% 区间
- 失败条件：M 个对手后，mu ∉ Top33%-Top66% 区间
- max_candidate_fails 为累计值（默认 12，非连续）
- M=16, n_battle=1
"""
