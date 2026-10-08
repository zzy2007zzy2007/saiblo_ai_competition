# 价值网络数据合并 & 大规模训练方案

## 问题

5000 局价值数据约 1M 帧，展开约 20GB，超出 16GB 内存。当前 ValueDataset 全量加载到内存的做法不可行。

## 方案：多局合并 + 分块训练

### 数据合并

写一个 `merge_value_npz.py`，将数据目录中的 .npz 文件按每 N 局合并为一个文件：

```
输入: value_data/  (5000 个 .npz, 每个 1 局)
                ↓ merge_value_npz.py --games-per-file 100
输出: value_data_merged/  (50 个 .npz, 每个 ~100 局)
```

合并后的文件结构不变：

| key | shape | dtype |
|-----|-------|-------|
| board | (sum_T, 28, 19, 19) | float16 |
| stats | (sum_T, 42) | float16 |
| value | (sum_T, 2) | float32 |

合并的好处：
- 文件数从 5000 降为 50，`np.load` 次数减少 100 倍
- 加载 50 个文件的循环开销（open/read header）可忽略
- 单文件大小 ~100 局 × 250 帧 × 20KB ≈ 500MB，加载很快

### 训练加载策略

#### 方案 A: 全量加载（如果内存够）

50 个文件 × 500MB ≈ 25GB 展开。如果内存够 32GB 可以直接一次加载完。

#### 方案 B: 分块流式训练（推荐）

每个 epoch 按文件顺序流式训练：

```
for each epoch:
  shuffle file list
  for each merged_file:
    data = np.load(merged_file)         # 加载一个 ~500MB 块
    board, stats, target = data[...]    # 展开 ~2GB 临时内存
    for batch in random_batches(data):
      train_step(batch)
    del data                             # 释放内存，加载下一块
```

每块 ~2GB 临时内存，总内存占用可以控制在 4GB 以内（含模型 + 临时缓冲区）。

优点是：
- 内存稳定，不随数据量增长
- 不需要 memmap 或额外依赖
- 每个 epoch 文件顺序随机化，保证数据多样性

缺点是：
- 每个 epoch 要多读一遍磁盘（50 个文件顺序读 ≈ 2.5GB/epoch）
- 但 SSD 顺序读 ~500MB/s，50 个文件读一遍只要 5 秒，可忽略

### 训练脚本修改

`train_value_net.py` 改动：

1. `ValueDataset` 支持加载已合并的大文件
2. 可选 `--stream` 模式：分块流式训练
   - 不保留 all_data 的引用
   - 每个 epoch 遍历文件，每次加载一个文件 → 打散 → 训练 → 释放
3. 保留现有的 `--tau` 标签方案

## 和 memmap 方案的对比

| | 多局合并 + 流式 | memmap |
|--|---------------|--------|
| 实现复杂度 | 低（纯 numpy） | 中（memmap 随机访问） |
| 额外依赖 | 无 | 无 |
| 内存占用 | ~2GB（每块临时） | ~OS 缓存（可控） |
| 每个 epoch 磁盘读取 | 50 次顺序读 2.5GB | ~1M 次随机读 ~20GB |
| epoch 间数据顺序 | 文件内 shuffle + 文件间 shuffle | 全量 shuffle |
| 训练速度 | 略慢（文件 I/O 夹在 GPU 计算间） | 快（随机访问，缓存友好） |

## 实施步骤

1. `code/merge_value_npz.py` — 合并脚本
2. 修改 `train_value_net.py` — 支持流式训练
3. 收集 5000 局数据
4. 合并 → 训练

## 附录：内存估算

```
公式: 帧数 × (28×19×19 + 42) × 2 bytes (float16)

5000局 × 250帧/局 = 1,250,000 帧
1,250,000 × 20,300 bytes = ~25.4 GB  (总展开)
25.4 GB ÷ 50 文件 = ~500 MB/文件    (合并后，压缩)
25.4 GB ÷ 50 文件 = ~508 MB/文件    (展开，float16)
```
