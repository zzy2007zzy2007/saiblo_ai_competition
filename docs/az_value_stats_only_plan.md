# 价值头"只看 42 维 state"对照实验（2026-09-12）

## 动机

前面所有价值头实验都隐含一个假设：**棋盘的空间特征对价值有用**。证据链却是相反的：

1. **池化消融全灭**（`docs/az_value_head_spatial_pooling_plan.md`）：5 种空间池化
   （gapmask/gapmax/region/grid/attn）没有一个可靠优于 GAP，grid(1088 维) 最差。
2. **tau-abs 标签有"抄输入"捷径**（corr 0.937）：价值头可以用 `stats[1]`（当前血量差）
   拟合大部分标签，不需要理解棋盘。
3. 42 维 state 里本身就有大量"结果性"信息：`hp_delta`(1)、`hp/50`(24/25)、
   `coin_ratio`(2)、`tower_level_sum`(11/12)、`kill_delta`(13)、`base_arc_coverage`(20)……

**所以最直接的对照是**：把价值头的输入砍到**只有 42 维 state**（不给任何空间信息），
如果它训出来和"GAP 池化 + 全空间特征"差不多，那就说明**当前价值头的瓶颈根本不是
空间表达**，池化那条线可以彻底关掉，算力应该转向**改变数据/对手**。

## 设计

利用现成的 `value_pool` 机制加一个 `"stats"`：

- `value_mult = 0`，`value_spatial = []` → `state_emb = stats_emb`（64 维）
- 价值头输入 = `LATENT_DIM * (0 + 1)` = 64 维，**只由 42 维 stats 经 MLP 得来**
- 策略路径完全不动（`board_emb` 仍走 GAP 给策略头；骨干照常 forward）

即价值头**结构上无法看到棋盘**，而不是"把棋盘置零"这种近似。

### 已知影响

- `state_emb` 从 128 维变 64 维。消费者只有：老 ES 路径的 `agent.py`/`gating_network.py`
  （本实验不经过），以及 `az_train.py` 的 `--lambda-anchor-v` 价值骨干锚定（`value_warmup`
  路径不经过）。元数据写 `value_pool`，加载端按同一配置建网 → 自洽。
- **骨干对价值头不再有梯度**（没有空间通路），与"冻结骨干"叠加后骨干完全惰性。
  这正是对照的目的：测量"纯 state 价值头"的天花板。

## 实验设计

2 臂，**唯一变量 = 价值头输入**，其余照抄已有臂：

| arm | `--value-pool` | 标签 | 直接对照 |
|-----|---------------|------|---------|
| `stats_term` | `stats` | terminal | `gap`+terminal（29.7%）|
| `stats_tau50` | `stats` | tau50/abs/scale2 | `gap`+tau50（15.6%，已测）|

配方：`gen0120_bn_init` + 冻结骨干 + no_bn + `warm_polonly`(336 局) + 10 epochs，
策略固定 `az_r10` 合成。

## 判读

- **`stats_term` ≈ `gap`+terminal（29.7%）** → 空间特征对价值**没有贡献**，池化线关闭，
  瓶颈在数据/对手 → 转对手池
- **`stats_term` 明显更低** → 空间特征确实有用，池化全灭是**池化方式**的问题
  （例如 GAP 恰好够用，更细的池化引入噪声）→ 池化还值得再挖
- `stats_tau50` 同理对照 `gap`+tau50

## 风险

- 价值头只看 stats 时**过拟合风险更低但表达力更弱**；若 label 里有纯空间依赖的成分
  （如"塔摆在哪决定防守成败"），它会学不到 → 这正是我们要测的
- 只用 2 臂、32 局粗筛**不作结论**；最终判据仍是 search-vs-search 64 局
