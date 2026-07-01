# 训练评估方法

## 评估脚本

`code/test_match/eval_checkpoint.py` — 加载 checkpoint 与 ExampleAI 对战 N 局，报告胜率、HP 统计。
注意：如果 checkpoint 是单头模型（`--num-heads 1` 训练），仍可用此脚本评估，解码器自动适配。

### 选项

- `--games N`：总局数（默认 30）
- `--workers N`：并行进程数（默认 12，推荐 16）
- `--top1`：加载 `top2_params[0]`（本代表现最好的个体）而非 `mean`（种群均值）

## 用法

```bash
# 评估种群均值 vs ExampleAI（100 局，16 进程并行）
python code/test_match/eval_checkpoint.py <checkpoint.pt> --games 100 --workers 16

# 评估本代最佳个体
python code/test_match/eval_checkpoint.py <checkpoint.pt> --games 100 --workers 16 --top1
```

## 训练运行 1：`training_history_20260630_115331`

**配置**：
- pop_size=54, sigma=0.2, lr=0.01, momentum=0.9
- games_per_ind=6, workers=6, generations=500
- Mirrored Sampling, 自对弈（无 num-heads，即默认3头）

**已运行的测试点**：

| Generation | Win rate vs Example (100 局) | Avg HP (us/opp) |
|-----------|:---------------------------:|:---------------:|
| 9 | 3% | 0.2 / 8.5 |
| 15 | 4% | 0.2 / 8.5 |
| 21 | (待测) | |

测试命令：
```bash
python code/test_match/eval_checkpoint.py code/my_ai/training_history_20260630_115331/gen_NNNN.pt --games 100 --workers 16
```

---

## 训练运行 2：`training_history/20260630_152924`

**配置**（num_heads=3）：
- pop_size=54, sigma=0.2, lr=0.01, momentum=0.9
- games_per_ind=8, workers=24, generations=500
- Mirrored Sampling, 自对弈

**已运行的测试点**：

| Generation | Type | Win rate vs Example (100 局) | Avg HP (us/opp) |
|-----------|:----:|:---------------------------:|:---------------:|
| 8 | mean | 4% | 0.1 / 9.5 |
| 11 | top1 | 2% | 0.0 / 8.6 |
| 11 | mean | 0%* | 0.0 / 8.8 |

\*仅 10 局快速验证，非 100 局标准测试。

测试命令：
```bash
python code/test_match/eval_checkpoint.py training_history/20260630_152924/gen_NNNN.pt --games 100 --workers 16
# 加 --top1 可测本代最佳个体（而非种群均值）
```

---

**Baseline**: 零权重 / 随机初始化的网络 vs ExampleAI 胜率约 1%，均 HP ≈ 0/10。

## 测试计划

- 每隔 20-30 代从各训练目录中测试一次 checkpoint，记录到对应表格。
- 留意胜率是否出现持续上升趋势（比如稳定 >5% 且不断上升），以及 avg HP (us/opp) 中的我方 HP 是否在增加。
