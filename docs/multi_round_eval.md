# 多轮淘汰评估方案

## 问题

当前评估所有个体打固定局数：

```
pop=96 × opp=12 × 2局 = 2304局/代
```

95% 置信区间 ±0.20，超过了群体真实强度差距（~0.15），排序基本靠随机。

暴力增加局数收益递减（√N），96 局才能到 ±0.10，评估量 ×4。

## 方案

多轮淘汰：

```
Round 1: 96人 × 24局 → 淘汰后48名, 幸存48人
Round 2: 48人 × 24局 → 淘汰后24名, 幸存24人  
Round 3: 24人 × 24局 → 淘汰后8名, 幸存16人
```

**幸存人数 = top_k（需要评估到能够选出 top_k 有意义）**

累计评估量：
```
R1: 96 × 24 = 2304
R2: 48 × 24 = 1152
R3: 24 × 24 = 576
合计: 4032局/代，是单轮(2304)的1.75倍
```

冠军精度：
```
冠军在三轮中各打24局 = 72局 → CI ≈ ±0.12
```

收益：1.75 倍评估量换来 2 倍冠军精度，比暴力 4 倍评估量换 2 倍精度高效。

## 实现细节

### 每轮对手重采样

每轮独立调用 `_select_opponents`，对手池不变但采样不同，避免某一轮被特别强/弱的对手偏置结果。

### 多轮分数累积

每轮对各幸存者的分数做累加，最终 fitness = 总分 / 总局数：

```python
total_scores = np.zeros(pop_size)   # 累加所有轮的得分
total_games = np.zeros(pop_size)    # 累加所有轮的局数
survivors = list(range(pop_size))

while len(survivors) > k:
    # 评估本轮幸存者
    n_games = len(opp_params) * 2
    scores = ...  # (len(survivors), n_games)
    fitness = scores.mean(axis=1)
    
    # 累加
    total_scores[survivors] += scores.sum(axis=1)
    total_games[survivors] += n_games
    
    # 轮次间排序使用本轮 fitness（短期精度已够区分谁该淘汰）
    # 最终 fitness 使用全部累加数据（更高精度）
    order = np.argsort(fitness)[::-1]
    survivors = [survivors[i] for i in order[:max(k, len(survivors)//2)]]

# 最终适应度：全部轮次平均
final_fitness = total_scores[survivors] / total_games[survivors]
```

三轮累积后幸存者每人打 72 局，最终排序精度 ±0.12 而不是 ±0.20。评估量不增加。

### 幸存者索引

维护 `survivors = list(range(pop_size))`（初始）。每轮只评估 `[ga_pop[i] for i in survivors]`，淘汰后更新 `survivors`。最终 `survivors[:k]` 即为 top_k。

### 伪代码

```python
survivors = list(range(pop_size))
opp_list = _select_opponents(...)
opp_params = [p for p, _ in opp_list]
opp_gens = [g for _, g in opp_list]

while len(survivors) > args.k:
    subset = [ga_pop[i] for i in survivors]
    all_args = build_eval_args(subset, opp_params, ...)
    # 附加 no_bn, bn_stats, eval_temperature
    results = run_eval(pool, all_args)
    
    n_games = len(opp_params) * 2
    scores = np.array([r["score"] for r in results]).reshape(len(survivors), n_games)
    fitness = scores.mean(axis=1)
    
    # 淘汰后半
    order = np.argsort(fitness)[::-1]
    survivors = [survivors[i] for i in order[:max(args.k, len(survivors) // 2)]]

top_k_idx = survivors[:args.k]
```

### 对阵型数据的影响

对局数据（npz）只在最后一轮保存，或者全部保存但标记轮次。推荐只在最后一轮保存——前面轮的个体没进 top_k，数据质量不高。

### 命令行参数

```bash
# 可选，默认单轮
--multi-round    # 启用多轮淘汰评估
```

## 预期效果

| 指标 | 单轮 | 多轮 |
|---|---|---|
| 评估局数/代 | 2304 | 4032 |
| 冠军精度 (CI) | ±0.20 | ±0.12 |
| top_k 重叠率 | ~18% | 预期 >50% |
| 耗时 | 1× | ~1.75× |

## 注意事项

1. 每轮的 `bn_stats` 和 `no_bn` 保持不变
2. LB 挑战的 `_vs_lb` 不受影响（仍然用单轮）
3. 每轮之间的种子偏移确保对手采样独立
4. 对手池的 `wr_by_gen` 在最后一轮计算，使用最后评估的 fitness 结果
