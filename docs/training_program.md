# ES 训练程序文档

## 概览

进化策略（Evolution Strategies, ES）训练器，用于训练 `AntWarNetwork` 神经网络。
算法为 OpenAI-ES 风格：在参数空间加高斯噪声生成种群 → 评估 → fitness shaping → 梯度估计 → 更新。

## 当前架构

### 网络结构（`code/my_ai/network.py`）

```
board(28,19,19) ── Conv7×7 ── ResBlock×6 ── spatial_feat(64,19,19)
                     ├── Conv1×1, 64→23 → action_map(23,19,19)
                     └── GAP → board_emb(64)
stats(42) ── MLP → stats_emb(64)
state_emb = concat(board_emb, stats_emb) = 128
  ├── Value Head → scalar [-1, 1]
  └── Policy Head → 3 × 23 class logits
```

- **参数量**：~553K
- **单次推理**（CPU）：~10ms（含解码）
- **一局完整对局**：~6 秒

### 输出解码（`code/my_ai/decoder.py`）

23 个动作通道：

| 通道 | 动作 | 说明 |
|------|------|------|
| 0-15 | 14 种塔类型 + 降级 | 意图驱动：空地→建 Basic，有塔→向目标升级一步 |
| 17-20 | 4 种超级武器 | 位置由 action_map 决定 |
| 21-22 | 基地升级 | 出兵速度 / 兵种血量 |

解码流程：class mask → argmax → position mask → argmax → intent decoder → Operation

### ES 训练器（`code/my_ai/es_train.py`）

```
每世代：
  1. 采样 noise: pop_size × param_count
  2. 生成扰动参数: θ + σ·ε
  3. 并行评估（multiprocessing）
  4. Rank-based fitness shaping
  5. 梯度估计 & 更新: θ ← θ + lr · g
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pop-size` | 16 | 每代个体数 |
| `--sigma` | 0.05 | 噪声标准差 |
| `--lr` | 0.01 | 学习率 |
| `--workers` | 4 | 并行进程数 |
| `--games` | 2 | 每个个体每代对局数 |
| `--generations` | 100 | 总世代数 |
| `--opponent` | example | 对手 AI（random / example） |
| `--seed` | 42 | 随机种子 |
| `--save-every` | 10 | 每 N 代保存一次模型 |
| `--out-dir` | auto | 输出目录（默认自动生成） |

## 输出结构

运行后自动创建时间戳目录：

```
training_history_20260629_224804/
├── config.txt       # 训练参数
├── train.log        # 终端日志（带时间戳）
├── history.csv      # 每代 fitness 数据
├── gen_0010.pt      # 模型检查点（每 save-every 代）
├── gen_0020.pt
└── final.pt         # 最终模型
```

### CSV 格式

```csv
generation,best_fitness,avg_fitness,eval_time_s,total_time_s
0,0.5000,0.4688,123.456,125.789
```

## Fitness 计算

- 每局：胜=1.0，平=0.5，负=0.0
- 个体 fitness = 所有对局的平均值
- 先后手按 seed 奇偶自动交替，消除 bias

## 关键设计

### cold_handle_rule_illegal=True
训练时非法操作被静默过滤而非判负，避免随机网络早期一出手就游戏结束。

### 并行评估
所有个体的所有对局一次性提交到进程池，统一取结果，最大化并行度。

### 意图解码
网络不输出塔 ID 等运行时数据，只输出"想要什么+在哪做"。
解码器根据场上状态查表确定具体操作。详见 `model_design.md`。

## 性能估算

| 配置 | 每代耗时 | 30 代耗时 |
|------|---------|----------|
| pop=8, games=2, workers=4 | ~2min | ~1h |
| pop=16, games=2, workers=4 | ~4min | ~2h |
| pop=32, games=2, workers=4 | ~8min | ~4h |

## 下一步方向

- [ ] 自对弈：对手池加入上一代精英，避免过拟合单一对手
- [ ] 行为克隆冷启动：先用 greedy AI 数据预训练
- [ ] MCTS 搜索集成（当前架构天然支持）
- [ ] 学习率 / sigma 自适应
- [ ] mirrored sampling 降方差
