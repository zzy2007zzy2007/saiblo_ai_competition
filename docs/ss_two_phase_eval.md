# ss_train 两阶段评估方案

## 问题

当前每代将所有 48 个个体的 864 局游戏数据全部写入磁盘（约 4.3GB .npz），但只有 top-K（默认 3，即 54 局）的数据被用于 BC 训练。剩余 90% 的数据写了就删。

导致**每代额外 I/O 约 240s**（占 eval 总时间 37%）。

## 两阶段方案

### 改动点一览

```
改动前:
  eval_all(48×18局, 全写盘) → 选 top-K → SSDataset → BC training

改动后:
  Phase 1 eval_all(48×18局, 不写盘) → 选 top-K → Phase 2 eval_topK(top-K×18局, 写盘) → SSDataset → BC training
```

### Phase 1：筛选（不写盘）

```
for idx in range(pop_size):
    for k_idx in range(k_per_ind):
        opp_params = opp_params_list[k_idx]
        seed = gen_seed + idx * games + k_idx * 2
        pool.starmap(_eval_worker, (params, opp, seed, heads, None, gen, idx))
                                    ^^^^ bc_dir=None → 不保存
```

- 和当前流程相同，但传给 `_eval_worker` 的 `bc_dir=None`
- `_eval_worker` 内部检查 `if bc_dir:` → 跳过 write_npz
- 返回结果只包含 score（当前已支持）

### Phase 2：top-K 写数据

Phase 1 结束后，从 fitness 排序取 top-K：

```
top_k_idx = argsort(fitness)[-args.k:]
for idx in top_k_idx:
    for k_idx in range(k_per_ind):
        opp_params = opp_params_list[k_idx]
        seed = new_seed  # Phase 1 的种子 + offset，保证数据不相同
        pool.starmap(_eval_worker, (params, opp, seed, heads, bc_dir, gen, idx))
                                    ^^^^ bc_dir=... → 保存 .npz
```

**关键问题**：种子策略。Phase 2 需要**不同的种子**，否则会生成和 Phase 1 完全相同的游戏数据，无法提供新信息。

```python
offset = 10_000  # 大偏移保证不重叠
new_seed = base_seed + offset + k_idx
```

### 参数控制

添加 `--save-data` 开关和 `--data-games` 参数：

```
--save-data       全保存（当前行为，I/O 最重）
--no-save-data    两阶段（推荐新代码）
--data-games      每个 top-K 个体记录多少局数据（默认 6，可热加载）
```

当 `--no-save-data` 时，自动启用两阶段评估。Phase 2 每个 top-K 个体打 `--data-games` 局（而不是 `--games` 局），写出 .npz 用于 BC 训练。

`--data-games` 和 `--games` 解耦后：

| 参数 | 作用 | 热加载 |
|:---|:----|:------|
| `--games` | 评估阶段每人的比赛局数（决定 fitness） | 是 |
| `--data-games` | 数据收集阶段每人的记录局数（决定数据量） | 是 |
| `--k` | 收集前几名个体的数据 | 是 |

这样可以在不改变评估精度的情况下，独立控制训练数据量：

```python
# Phase 1: 评估用 --games 局
games_per_ind = args.games

# Phase 2: 数据收集用 --data-games 局（默认 6）
data_games = args.data_games if not args.save_data else args.games
```

### 性能预估

| | 单阶段全写（当前） | 两阶段（data-games=6） |
|:--|:-----------------|:----------------------|
| Phase 1 游戏 | 864 局 × 0.42s = 366s | 864 局 × 0.42s = 366s |
| Phase 1 I/O | 243s | **0s** |
| Phase 2 游戏 | — | 18 局 × 0.42s = 8s |
| Phase 2 I/O | — | 18/864 × 243 = 5s |
| BC 训练 | 15s | 15s |
| **总计** | **624s** | **394s** |

**节省约 230s/代（37%），每代 ≈6.5min**

### 风险

**种子差异**：Phase 2 的种子和 Phase 1 不同，top-K 个体在 Phase 2 中的实际表现和 Phase 1 的 fitness 可能有微小差异。这不影响 BC 训练——收集到的数据仍然是精英个体的游戏记录，只是具体局面略有不同。

### 实现步骤

1. 修改 `_eval_worker`：当 `bc_dir=None` 时不写 .npz（当前已是这样，但需要确认返回值格式一致）
2. 添加 `--save-data` 参数（默认 True，向后兼容）
3. 修改主循环：
   ```python
   if args.save_data:
       # 单阶段：全写盘（当前行为）
       all_args = build_eval_tasks(...)
       results = pool.starmap_async(...).get()
   else:
       # Phase 1：不写盘
       all_args = build_eval_tasks(bc_dir=None, ...)
       results1 = pool.starmap_async(...).get()
       fitness1 = parse_fitness(results1)
       # 选 top-K
       top_k_idx = sorted(...)[:args.k]
       # Phase 2：top-K 写盘
       all_args2 = build_eval_tasks(bc_dir=bc_dir, only_idx=top_k_idx, seed_offset=10000, ...)
       results2 = pool.starmap_async(...).get()
       # 合并 fitness（Phase 2 的 score 覆盖 Phase 1 的）
       fitness = merge_scores(fitness1, results2, top_k_idx)
   ```
4. 提取 `build_eval_tasks()` 和 `parse_fitness()` 为辅助函数，避免主循环重复代码

### 可选的进一步优化

如果 Phase 2 的游戏种子偏移不够大，Phase 2 的游戏序列会和 Phase 1 高度相似。可以用大偏移 + 打乱对手顺序来确保多样性：

```python
# Phase 2 打乱对手顺序
import random
opp_indices = list(range(k_per_ind))
random.shuffle(opp_indices)
for k_idx in opp_indices:
    ...
```
