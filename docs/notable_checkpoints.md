# Notable Checkpoints

## ss_20260709_191050 / gen_0030

- **路径**: `training_history/ss_20260709_191050/gen_0030.pt`
- **训练参数**: action_dropout=0.1, lb_inject=3, 常规 ES+BC
- **头分工**: H1=Lightning, H2=Downgrade, H3=HOLD
- **行为**: BUILD→DOWNGRADE(退钱)→LIGHTNING 操作链，反复刷闪电
- **胜率 vs ExampleAI**: 90.6% (58/64) 无 dropout, 95.3% (61/64) dropout=0.15
  - 先手: 87.5% / 96.9%
  - 后手: 93.8% / 93.8%
  - 平均 HP: 11.0 / 0.5 (无 dropout)
- **胜率 vs rule_v4**: 1.6% (1/64)
- **说明**: 第一个突破 50% 的模型，证明 dropout+精英保留+足够世代数可以产生强模型。但策略窄，只针对不建防的对手。

## ss_20260710_155327 / gen_0101

- **路径**: `training_history/ss_20260710_155327/gen_0101.pt`
- **训练参数**: oversample_alpha=0.6, action_dropout=0.1, 常规 ES+BC
- **头分工**: H1=HOLD, H2=43Medic, H3=31Mortar+
- **行为**: 纯建塔策略，BUILD + UPGRADE，不依赖超级武器
- **胜率 vs ExampleAI (64局)**: **67.2%** (43/64)
  - 先手: 59.4% (19/32)
  - 后手: 75.0% (24/32)
  - 平均 HP: 6.6 / 2.1
- **训练 eval best**: 0.667 (gen_101)
- **说明**: 纯塔防策略，靠建 Medic 和 Mortar+ 塔守住基地。和 gen_0030 的闪电流不同，走的是完全不同的战术路径。有趣的是后手胜率更高。
