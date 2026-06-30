# 性能分析

## 1. 引擎速度

纯引擎（无 AI 操作）在各种环境下的基准测试：

| 环境 | 每回合 | 每场 (250回合) | 测试条件 |
|------|-------|---------------|---------|
| 纯引擎 (base) | 2.7ms | 0.65s | `resolve_turn([], [])` |
| 纯引擎 (pytorch-gpu) | ~2.7ms | ~0.65s | 同上 |

引擎本身是纯 Python，使用 `__slots__` 优化，单场约 0.65s。**引擎不是瓶颈。**

## 2. AI 速度对比

完整对战（含 AI 决策 + 引擎）的每回合耗时：

| AI 组合 | 每回合 | 每场 | 相对纯引擎 |
|---------|-------|------|-----------|
| 纯引擎 | 2.7ms | 0.65s | 1x |
| **Neural vs Neural** (pytorch-gpu) | **29ms** | **~7.3s** | **11x** |
| Example vs Example (pytorch-gpu) | 22ms | ~5.5s | 8x |
| Example vs Neural (pytorch-gpu) | 77ms | ~19.3s | 29x |
| Greedy (base) | 231ms | ~58s | 86x |
| Example (base) | 80ms | ~20s | 30x |

## 3. 一回合耗时拆解

完整一回合（2 个玩家）耗时分项（pytorch-gpu 环境，NeuralAgent）：

```
FeatureExtractor × 2       5.6ms  ← 47%
网络前向 × 2                4.0ms  ← 33%
游戏引擎 (resolve_turn)     2.7ms  ← 22%
Decoder × 2                2.0ms  ← 17%
其他                         ~1ms  ←  8%
-------------------------------
总计                        ~12ms
```

FeatureExtractor 是最大的单项，网络推理占比不高（因为 MKL 加速）。

## 4. CPU vs GPU

测试条件：Ryzen 9 9955HX, RTX 5060 Laptop, pytorch-gpu 环境

| 场景 | 120回合 | 每回合 | 加速比 |
|------|--------|-------|-------|
| CPU 单进程 | 1.47s | 12ms | 1x |
| GPU 单进程 | 1.49s | 12ms | **1.0x** |

**结论：用不上 GPU。** 瓶颈在特征提取和引擎结算（都在 CPU），网络推理只占 2ms。GPU 加速不了。

⚠️ 注意：之前在 `base` 环境测出 CPU 769ms，是因为 base 环境没有 MKL 优化。切换到 `pytorch-gpu` 环境后 CPU 推理有 MKL 加速，性能大幅提升。

## 5. 多进程吞吐

12 workers 并行（pytorch-gpu 环境）：

| AI 组合 | 12场耗时 | 每场 | 每回合 | 每小时 |
|---------|---------|------|-------|-------|
| Example vs Example | 25.9s | 2.15s | 22ms | ~5000场 |
| Example vs Neural | 92.6s | 7.72s | 77ms | ~1400场 |
| **Neural vs Neural** | **35.2s** | **2.93s** | **29ms** | **~3600场** |

### 为什么 Example vs Neural 最慢？

这是一个反直觉的结果。通过消融实验验证了：

| 实验 | 每场 | 说明 |
|------|------|------|
| Ex+Ex, cold_handle=True | 2.08s | baseline |
| Ex+Ex, cold_handle=False | 2.83s | 反而更慢 |
| Ex+Ex + NN 构造开销 | 2.08s | 无影响 |
| Ex+NN, cold_handle=False | 1.12s | 游戏立即结束 |

原因分析：
- NeuralAgent 输出大量非法操作，必须开 `cold_handle_rule_illegal=True`
- 这使得游戏能持续进行（不像 False 时非法操作直接判负）
- Example AI 的 ActionCatalog 本身就重，加上混合负载的 CPU 调度开销

**结论：Example 不是好对手。** 用 `--opponent example` 训练既慢又不准。建议用自对弈或 random。

## 6. ES 训练吞吐估算

Neural vs Neural 自对弈，12 workers：

| 配置 | 每世代 | 每小时 |
|------|-------|-------|
| pop=16, games=6 | ~3 min | ~20 世代 |
| pop=32, games=4 | ~4 min | ~15 世代 |
| pop=64, games=2 | ~5 min | ~12 世代 |
