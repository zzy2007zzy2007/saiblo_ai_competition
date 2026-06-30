# 训练评估方法

## 评估脚本

`code/test_match/eval_checkpoint.py` — 加载 checkpoint 与 ExampleAI 对战 N 局，报告胜率、HP 统计。
注意：如果 checkpoint 是单头模型（`--single-head` 训练），仍可用此脚本评估，解码器自动适配。

## 用法

```bash
# 评估一个 checkpoint vs ExampleAI（100 局，12 进程并行）
python code/test_match/eval_checkpoint.py <checkpoint.pt> --games 100 --workers 12
```

## 当前训练状态

**路径**: `code/my_ai/training_history_20260630_115331/`

**配置** (train.log 第 1-15 行可见):
- pop_size=54, sigma=0.2, lr=0.01, momentum=0.9
- games_per_ind=6, workers=12, generations=500
- Mirrored Sampling, 自对弈

**已运行的测试点**（截至写文档时）：

| Generation | Win rate vs Example (100 局) | Avg HP (us/opp) |
|-----------|:---------------------------:|:---------------:|
| 9 | 3% | 0.2 / 8.5 |
| 15 | 4% | 0.2 / 8.5 |
| 21 | (待测) | |

**Baseline**: 零权重 / 随机初始化的网络 vs ExampleAI 胜率约 1%，均 HP ≈ 0/10。

## 测试计划

建议每隔 20-30 代测试一次，记录到上表。测试命令：

```bash
# 第 N 代
python code/test_match/eval_checkpoint.py code/my_ai/training_history_20260630_115331/gen_NNNN.pt --games 100 --workers 12
```

留意胜率是否出现持续上升趋势（比如稳定 >5% 且不断上升），以及 avg HP (us/opp) 中的我方 HP 是否在增加。
