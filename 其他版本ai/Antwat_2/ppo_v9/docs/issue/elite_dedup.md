# 种子精英去重方案

> 日期: 2026-06-12
> 状态: 待实施
> 关联: 种子池家族垄断问题分析（gen4 实测数据）

## 1. 问题背景

### 1.1 实测发现的家族垄断

server10 gen4 的 8 个种子权重余弦相似度矩阵（节选）：

```
            seed_1  seed_2  seed_3  seed_4  seed_5
seed_1      1.000   0.903   0.500   0.483   0.430
seed_2      0.903   1.000   0.546   0.499   0.462
seed_3      0.500   0.546   1.000   0.917   0.430
seed_4      0.483   0.499   0.917   1.000   0.405
```

形成了两个明显家族：
- **家族 A**：seed_1 ↔ seed_2（cos=0.903）
- **家族 B**：seed_3 ↔ seed_4（cos=0.917）
- 其余 4 个种子中，3 个对家族 A 的 cos≥0.66，呈现明显偏向

更严重的是，跨代存在逐位相同的拷贝：
- gen4 seed_2 ≡ gen3 seed_2（cos=1.000, L2=0.00，精英拷贝）
- gen4 seed_3 ≡ gen3 seed_1（cos=1.000, L2=0.00，精英拷贝）

### 1.2 退化路径推演

当前家族 A 对家族 B 有 100~200 ELO 优势。在纯 ELO 选拔 + `elitism_count=2` 下：

```
gen5: 家族 A 占 5~6 个种子位
gen6: 家族 B 被完全挤出
gen7+: 8 个种子全部来自家族 A，交叉失去重组意义，进化停滞
```

### 1.3 去重的必要性

用户判断：**初始阶段两个家族可以接受，但不能容忍后续退化为单一家族。**

去重方案需要做到：
- 阻止逐位相同的拷贝（cos>0.95）占用种子位
- 限制同一家族（cos>0.85）在种子池中的最大比例
- 为不同策略方向的个体保留进入亲本池的通道

---

## 2. 方案设计

### 2.1 两档去重策略

| 档次 | cos 阈值 | 含义 | 限制 |
|------|---------|------|------|
| **硬去重** | > 0.95 | 几乎完全相同（含逐位拷贝） | 只保留 ELO 最高的 1 个 |
| **软限流** | 0.85 ~ 0.95 | 同一家族（近亲） | 最多保留 2 个（可配） |

### 2.2 选择算法

```
输入: ranked (按 ELO 降序排列的 80 个个体), n_seeds=8
输出: 8 个种子

已选种子列表 = []
硬重复阈值 = 0.95
软限制阈值 = 0.85
软限制最大数 = 2

for ind in ranked:
    计算 ind 与已选列表中每个种子的最大余弦相似度 max_cos

    if max_cos > 0.95:
        # 硬去重：几乎是同一个模型，跳过
        跳过
        continue

    同一家族计数 = 统计已选列表中与 ind 的 cos > 0.85 的数量

    if 同一家族计数 >= 2:
        # 软限流：该家族已满 2 个名额，跳过
        跳过
        continue

    # 通过筛选，加入种子列表
    已选列表.append(ind)
    记录 ind 的 ELO

    if len(已选列表) >= 8:
        break

if len(已选列表) < 8:
    # 兜底：放宽软限制到 3，重扫
    放宽软限制 = 3，重新扫描剩余的个体
```

### 2.3 兜底机制

如果通过筛选的个体不足 8 个（例如极端近亲情况下），逐级放宽：

1. 放宽软限制阈值从 0.85→0.90（缩小"家族"判定范围）
2. 增加软限制最大数从 2→3
3. 最后直接降级到无去重：按 ELO 取满 8 个，但在日志中标记警告

### 2.4 对精英保留的联动调整

当前 `elitism_count=2` 使得 gen4 的 top-2 精英被原样拷贝到下一代，且精英的 `parent_ids` 只有一个父本。去重方案对精英保留的影响：

- **精英也在种子池中**：如果 gen4 seed_2 被种子去重排除（与被选种子 cos>0.95），那么它不会进入 gen5 的亲本池，也就不会被精英保留
- **精英 ID 变化**：gen5 的精英不再是 gen4 某个个体的拷贝，而是 gen5 交叉变异产生子代中 ELO 最高的
- **建议**：将 `elitism_count` 从 2 降为 1，或将精英也纳入去重逻辑（只保留去重后的第一个）

### 2.5 阈值选择实验

当前阈值（hard=0.95, soft=0.85）基于经验设定。为了找到更优的阈值，可以基于已保存的 gen4 权重数据做一个离线网格搜索实验。

#### 2.5.1 实验设计

**输入数据**：已缓存的 gen4 全部 80 个个体的扁平权重向量（如 `gen4_weights.npy`，shape=(80, param_count)）及对应 ELO 排名。

**搜索空间**：

| 参数 | 候选值 |
|------|--------|
| `hard_threshold` | 0.90, 0.92, 0.95, 0.97, 0.99 |
| `soft_threshold` | 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90 |
| `soft_max_per_family` | 1, 2, 3 |

共 5×7×3 = 105 组参数组合，对 80 个个体运行选择算法，每组耗时 < 0.01s，总计 < 2s，完全可行。

**评估指标**（对每组参数组合运行后统计选出的 8 个种子）：

| 指标 | 含义 | 目标 |
|------|------|------|
| `family_count` | 种子间 cos>soft_threshold 形成的连通分量数 | 越大越好（≥4 为优） |
| `cos_max` | 任意两个种子间的最大余弦相似度 | 越小越好（<0.90 为优） |
| `cos_mean` | 种子间两两余弦相似度的均值 | 越小越好 |
| `elo_drop` | (原 top-8 ELO 均值) − (去重后 8 种子 ELO 均值) | 越小越好（<50 可接受） |

**最优参数选择逻辑**：

```
1. 筛选: family_count ≥ 4 AND cos_max < 0.90
2. 在筛选结果中, 按 elo_drop 升序排列
3. 取 elo_drop 最小的 top-3 参数组合
4. 人工审查这 3 组，选择直觉上最合理的一组
```

如果没有任何组合满足 `family_count ≥ 4`，则放宽到 `family_count ≥ 3`，同时发出警告说明 gen4 本身的多样性不足以支撑更严格的去重。

#### 2.5.2 实验脚本骨架

```python
# tools/elite_dedup_threshold_search.py
import numpy as np
import itertools
from collections import defaultdict

def select_seeds_sim(weights, elos, hard_th, soft_th, soft_max, n=8):
    """离线模拟 select_seeds_diverse，返回选中的种子索引列表"""
    order = np.argsort(elos)[::-1]  # ELO 降序
    selected = []
    for idx in order:
        w = weights[idx]
        hard_dup = False
        family_count = 0
        for s_idx in selected:
            cos = cosine_similarity(w, weights[s_idx])
            if cos > hard_th:
                hard_dup = True
                break
            if cos > soft_th:
                family_count += 1
        if hard_dup or family_count >= soft_max:
            continue
        selected.append(idx)
        if len(selected) >= n:
            break
    return selected

def compute_family_count(weights, indices, soft_th):
    """用并查集计算连通分量数（家族数）"""
    n = len(indices)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in range(i+1, n):
            if cosine_similarity(weights[indices[i]], weights[indices[j]]) > soft_th:
                union(i, j)
    return len(set(find(i) for i in range(n)))

# 加载数据
weights = np.load("gen4_weights.npy")  # (80, param_count)
elos = np.load("gen4_elos.npy")         # (80,)
orig_top8_elo_mean = np.mean(sorted(elos, reverse=True)[:8])

# 网格搜索
results = []
for hard_th in [0.90, 0.92, 0.95, 0.97, 0.99]:
    for soft_th in [0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]:
        for soft_max in [1, 2, 3]:
            sel = select_seeds_sim(weights, elos, hard_th, soft_th, soft_max)
            cos_vals = [cosine_similarity(weights[sel[i]], weights[sel[j]])
                        for i in range(len(sel)) for j in range(i+1, len(sel))]
            results.append({
                "hard_th": hard_th, "soft_th": soft_th, "soft_max": soft_max,
                "n_selected": len(sel),
                "family_count": compute_family_count(weights, sel, soft_th),
                "cos_max": max(cos_vals) if cos_vals else 0,
                "cos_mean": np.mean(cos_vals) if cos_vals else 0,
                "elo_drop": orig_top8_elo_mean - np.mean(elos[sel]),
            })

# 筛选并输出最优组合
viable = [r for r in results
          if r["family_count"] >= 4 and r["cos_max"] < 0.90]
viable.sort(key=lambda r: r["elo_drop"])
for r in viable[:5]:
    print(f"hard={r['hard_th']:.2f} soft={r['soft_th']:.2f} max={r['soft_max']} "
          f"families={r['family_count']} cos_max={r['cos_max']:.3f} "
          f"elo_drop={r['elo_drop']:.1f}")
```

#### 2.5.3 实验结果的解读与落地

- 如果多组参数结果接近（elo_drop 差距 < 10），优先选 `soft_threshold` 更严格的（值更大），因为更严格的阈值更稳妥
- 如果 `soft_max=1` 和 `soft_max=2` 效果接近，选 2，避免过度限制
- 实验只需在引入去重前跑一次，不需要每次进化都跑
- 阈值确定后写入配置文件，后续如需调整（如网络结构变化导致 cos 分布漂移），重新跑实验即可

---

## 3. 代码修改

### 3.1 新增工具函数：基因组余弦相似度

在 `src/ppo_ga/selection/` 或 `src/ppo_ga/genome/` 下新增：

```python
# src/ppo_ga/genome/genome.py 或新文件 genome_utils.py

def cosine_similarity(weights_a: np.ndarray, weights_b: np.ndarray) -> float:
    """计算两个扁平权重向量的余弦相似度。

    Args:
        weights_a: shape (N,)
        weights_b: shape (N,)

    Returns:
        cos(a, b) = (a·b) / (|a|·|b|)，范围 [-1, 1]
    """
    dot = np.dot(weights_a, weights_b)
    norm_a = np.linalg.norm(weights_a)
    norm_b = np.linalg.norm(weights_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
```

### 3.2 修改：`select_seeds()` → `select_seeds_diverse()`

在 `battle_selection.py` 中，替换或新增：

```python
def select_seeds_diverse(
    self,
    ranked_individuals: List[Individual],
    hard_threshold: float = 0.95,
    soft_threshold: float = 0.85,
    soft_max_per_family: int = 2,
) -> List[Individual]:
    """基于 ELO 排名进行多样性感知的种子选拔。

    在按 ELO 降序遍历时：
    1. 硬去重：cos > hard_threshold → 视为相同模型，跳过
    2. 软限流：已选同类 ≥ soft_max_per_family 时跳过
    3. 兜底：不足 n_seeds 时逐级放宽限制

    Args:
        ranked_individuals: 按 ELO 降序排列的个体列表
        hard_threshold: 硬去重的余弦相似度阈值（默认 0.95）
        soft_threshold: 同一家族的余弦阈值（默认 0.85）
        soft_max_per_family: 每家族最大名额（默认 2）

    Returns:
        去重后的种子列表（前 num_seeds 个，去重后可能少于 num_seeds）
    """
    from ..genome.genome import cosine_similarity

    # 设计说明：同时存储 Individual 和其 weights 的元组，避免后续通过
    # ranked_individuals.index(sel) 回查索引（O(n) 且依赖 dataclass 自动生成
    # 的 __eq__，后者会对 Genome.weights（np.ndarray）做元素级比较，语义脆弱）。
    selected: List[Individual] = []
    selected_weights: List[np.ndarray] = []  # 与 selected 一一对应

    for i, candidate in enumerate(ranked_individuals):
        w_candidate = candidate.genome.weights

        # 检查是否与已选种子硬重复
        is_hard_dup = False
        family_count = 0
        for sel_w in selected_weights:
            cos = cosine_similarity(w_candidate, sel_w)

            if cos > hard_threshold:
                is_hard_dup = True
                logger.debug(
                    f"Hard duplicate: {candidate.id} (ELO={candidate.elo_rating:.0f}) "
                    f"cos={cos:.4f}"
                )
                break

            if cos > soft_threshold:
                family_count += 1

        if is_hard_dup:
            continue

        if family_count >= soft_max_per_family:
            logger.debug(
                f"Soft throttled: {candidate.id} (ELO={candidate.elo_rating:.0f}) "
                f"family already has {family_count} seeds"
            )
            continue

        # 通过筛选：同时存储 Individual 和 weights
        selected.append(candidate)
        selected_weights.append(w_candidate)
        if len(selected) >= self.num_seeds:
            break

    # ── 兜底：不足 num_seeds，逐级放宽 ──
    if len(selected) < self.num_seeds:
        # 用 ID set 做 O(1) 排除已选个体，避免 candidate in selected 依赖 __eq__
        selected_ids = {ind.id for ind in selected}
        remaining = [
            ind for ind in ranked_individuals if ind.id not in selected_ids
        ]

        for fallback_level, (st, sm) in enumerate([
            (soft_threshold + 0.05, 3),   # Level 1: 放宽家族阈值，增加名额
            (soft_threshold + 0.05, 4),   # Level 2: 继续放宽
            (1.0, self.num_seeds),         # Level 3: 完全降级
        ]):
            for candidate in remaining:
                if candidate.id in selected_ids:
                    continue
                if len(selected) >= self.num_seeds:
                    break

                w_candidate = candidate.genome.weights
                family_count = 0
                is_hard = False
                for sel_w in selected_weights:
                    cos = cosine_similarity(w_candidate, sel_w)
                    if cos > hard_threshold:
                        is_hard = True
                        break
                    if cos > st:
                        family_count += 1
                if is_hard:
                    continue
                if family_count >= sm:
                    continue

                selected.append(candidate)
                selected_weights.append(w_candidate)
                selected_ids.add(candidate.id)

            if len(selected) >= self.num_seeds:
                logger.warning(
                    f"Seed dedup fallback level {fallback_level}: "
                    f"got {len(selected)} seeds"
                )
                break

    # 最终兜底：仍然不够，直接按 ELO 取
    if len(selected) < self.num_seeds:
        logger.error(
            f"Seed dedup exhausted: only {len(selected)}/{self.num_seeds} seeds selected. "
            f"Using ELO-only fallback."
        )
        selected = ranked_individuals[:self.num_seeds]

    # 设置种子排名
    for i, seed in enumerate(selected):
        seed.seed_rank = i + 1

    # 日志：输出种子间的多样性概览
    if len(selected) >= 2:
        cos_vals = []
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                cos_vals.append(cosine_similarity(
                    selected_weights[i],
                    selected_weights[j],
                ))
        logger.info(
            f"Seeds diversity: {len(selected)} seeds, "
            f"cos range=[{min(cos_vals):.3f}, {max(cos_vals):.3f}], "
            f"cos mean={sum(cos_vals)/len(cos_vals):.3f}"
        )

    return selected[:self.num_seeds]
```

### 3.3 Trainer 集成

在 `genetic_evolution_trainer.py` 的 `train()` 中，阶段三改为调用新方法：

```python
# 原代码：
# seeds = self.battle_selection.select_seeds(ranked)

# 修改为：
dedup_cfg = self.config.get("elite_dedup", {})
if dedup_cfg.get("enabled", True):
    seeds = self.battle_selection.select_seeds_diverse(
        ranked,
        hard_threshold=dedup_cfg.get("hard_threshold", 0.95),
        soft_threshold=dedup_cfg.get("soft_threshold", 0.85),
        soft_max_per_family=dedup_cfg.get("soft_max_per_family", 2),
    )
else:
    seeds = self.battle_selection.select_seeds(ranked)
```

### 3.4 配置文件新增

```yaml
# ga_evo.yaml 新增
elite_dedup:
  enabled: true
  hard_threshold: 0.95       # cos>0.95 视为相同模型
  soft_threshold: 0.85       # cos>0.85 视为同一家族
  soft_max_per_family: 2     # 每家族最多 2 个种子位
```

---

## 4. 对 gen4 实际数据的效果推演

以 gen4 的 ELO 排序，套用去重算法（hard=0.95, soft=0.85, max_per_family=2）：

```
候选遍历:
  #1 ind_012 (1777) → 通过，种子=[ind_012]
  #2 ind_001 (1631) → cos(ind_001, ind_012)=0.903 → 软限流计数=1 < 2 → 通过
  #3 ind_000 (1616) → cos(ind_000, ind_001)≈0.546, cos(ind_000, ind_012)≈0.500
                       → 软限流计数=0 → 通过
  #4 ind_019 (1408) → cos(ind_019, ind_003)≈0.917 → 软限流计数=1 < 2 → 通过
  #5 ind_062 (1333) → cos vs ind_012=?, 分析中... → 假设通过
  #6 ind_018 (1311) → cos vs ind_001≈0.659 → 软限流家族A计数=2，如果通过则A=3→跳过

  继续扫描 #7, #8...
```

**预期效果**：
- 家族 A 最多占 2 个种子位（而非 5-6 个）
- 家族 B 最多占 2 个
- 剩余 4 个位给独立方向的个体
- **有效策略数从 ~3-4 提升到 ≥ 5**

### 4.1 潜在风险

| 风险 | 缓解 |
|------|------|
| 排除了 ELO 很高的个体只因它们是近亲 | 软限流允许每家族 2 个，不会完全排除优势家族 |
| gen=1 时所有个体都是随机的，cos 普遍 ~0.5 | 正常，去重不会触发 |
| 个别代次因去重导致亲本池质量下降 | 短期（1-2 代）的 ELO 损失会被长期多样性收益弥补 |

---

## 5. 与现有 diversity 指标的关系

当前 `compute_diversity()` 计算的是权重欧氏距离。去重方案不会替换它，而是补充：

| 指标 | 衡量什么 | 当前有？ | 去重方案 |
|------|---------|---------|---------|
| 权重欧氏距离 | 参数空间分散度 | ✅ | 不改变 |
| 种子间 cos 均值 | 策略方向多样性 | ❌ | ✅ 新增 |
| 家族数量 | 有效策略数 | ❌ | ✅ 隐式保证 ≥ 4 |

建议在日志中同时输出种子间 cos 均值（已在 `select_seeds_diverse()` 中实现），与 diversity 一起记录到 TensorBoard。

---

## 6. 实施步骤

| 步骤 | 内容 | 文件 | 预估 |
|------|------|------|------|
| 1 | 在 `genome.py` 或 `genome_utils.py` 新增 `cosine_similarity()` | genome/ | 0.2h |
| 2 | 在 `BattleSelection` 中新增 `select_seeds_diverse()` | selection/battle_selection.py | 1h |
| 3 | 修改 `trainer.train()` 调用新方法 | trainer/genetic_evolution_trainer.py | 0.3h |
| 4 | 新增 yaml 配置字段 | configs/ga_evo.yaml | 0.2h |
| 5 | 单元验证 + gen4 数据回测 | — | 0.5h |
| **合计** | | | **~2.2h** |
