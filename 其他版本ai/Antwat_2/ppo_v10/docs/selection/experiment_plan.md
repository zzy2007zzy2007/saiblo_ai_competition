# 二分种子方案验证实验

## 1. 实验目标

验证 `original_rule.txt` 方案的以下两个维度：

| 维度 | 核心问题 |
|------|---------|
| **计算规模** | 实际对战局数相比全量循环赛节省了多少？各阶段分别贡献多少？ |
| **覆盖率** | 二分种子选出的 Top16 与 Ground Truth Top16 的重叠率如何？在不同分布下的鲁棒性如何？ |

---

## 2. 方案参数

基于 `original_rule.txt` 的完整参数：

| 参数 | 值 | 含义 |
|------|:---:|------|
| n | 80 | 种群规模 |
| M | 16 | 候选升级所需最少对手数 |
| max_seeds | 3 | 二分种子目标数 |
| n_battle | 1 | 批量阶段每对个体对战次数（实际 2×n_battle=2 局，先后手各一） |
| max_candidate_fails | 12 | 累计候选失败上限（非连续） |
| max_workers | 25 | 并行执行 workers 数量 |

关键规则要点：

- **初始候选**：种群第 1 个个体
- **候选升级条件**：与 M 个以上对手对战后，TrueSkill mu 处于有对战记录个体的 Top33%-Top66% 区间
- **候选失败条件**：与 M 个以上对手对战后，TrueSkill mu 不在 Top33%-Top66% 区间
- **新候选选取**：有对战记录个体按 mu 升序取中位数
- **新候选对手优先**：优先与 battle_count > 0 的个体对战
- **批量阶段**：3 个种子确定后，补齐与全体的对战（n_battle×2 局/对）
- **Fallback**：累计 12 个候选失败后，直接用累积 TrueSkill 排序取 Top16

---

## 3. 实验矩阵

### 3.1 固定参数

```
n=80, M=16, max_seeds=3, n_battle=1, max_candidate_fails=12
```

### 3.2 变量

| 维度 | 取值 | 说明 |
|------|------|------|
| p 分布 | uniform / normal / bimodal | 个体真实强度的分布形态 |
| q 模式 | constant(8.33) / proportional / random(2~8.33) | 个体表现方差的模式 |
| 随机种子 | 42, 142, 242, 342, 442 | 控制个体生成和随机性的多轮重复 |

### 3.3 场景矩阵（3×3×5 = 45 组）

| # | p 分布 | q 模式 | 说明 |
|:--:|--------|--------|------|
| 1-5 | uniform | constant(8.33) | 基线：均匀强度 + 固定方差 |
| 6-10 | uniform | proportional | 越强的个体越稳定 |
| 11-15 | uniform | random(2~8.33) | 方差随机，模拟真实不确定性 |
| 16-20 | normal | constant(8.33) | 正态强度 + 固定方差 |
| 21-25 | normal | proportional | 正态强度 + 强度相关方差 |
| 26-30 | normal | random(2~8.33) | 正态强度 + 随机方差 |
| 31-35 | bimodal | constant(8.33) | 双峰强度（强弱分明）+ 固定方差 |
| 36-40 | bimodal | proportional | 双峰强度 + 强度相关方差 |
| 41-45 | bimodal | random(2~8.33) | 双峰强度 + 随机方差 |

每个 seed 组内保持个体生成一致，仅改变选拔算法。

---

## 4. 核心指标

### 4.1 计算规模指标

| 指标 | 公式 / 含义 |
|------|------------|
| **总对战局数** | 二分种子方案实际执行的对战总局数 |
| **节省率** | `1 - (二分总局数 / 全量RR总局数)` |
| **甄别阶段局数** | 候选甄别过程中的对局数 |
| **批量阶段局数** | 种子批量对战的局数 |
| **有效对局数** | 至少一方是种子或候选的对局 |
| **无效对局数** | 双方都不是种子的对局（含已取消） |
| **候选统计** | 启动数 / 升级数 / 失败数 |
| **Fallback 率** | 触发 fallback 的实验占比 |
| **提交后被取消的 task 数** | 甄别 + 最后清理阶段取消的总 task |

### 4.2 覆盖率指标

| 指标 | 含义 |
|------|------|
| **TopK 重叠率** | `|二分TopK ∩ 真值TopK| / K`，K ∈ {3, 5, 8, 16} |
| **Spearman ρ vs GT** | 二分排名与 Ground Truth 排名的相关性 |
| **Spearman ρ vs RR** | 二分排名与全量 RoundRobin 排名的相关性 |
| **Recall@K within Top16** | 二分 Top16 候选池中包含了多少个真值 TopK（K ∈ {3, 5, 8, 16}） |
| **平均排名偏差** | 二分 Top16 个体在真值排名中的位置均值（越小越好，最优=8.5） |
| **RR 自身 ρ vs GT** | RoundRobin 排名与 Ground Truth 的相关性（作为对照基线） |

### 4.3 过程指标（辅助分析）

| 指标 | 含义 |
|------|------|
| 甄别阶段已对战个体数 | 甄别结束时 battle_count > 0 的个体数 |
| 找到 3 种子所需候选数 | 累计启动了多少候选才凑齐 3 个种子 |
| 种子 mu 在种群中的百分位 | 确认的种子在全部个体 mu 排序中的位置 |

---

## 5. 预期分析（理论推算）

### 5.1 全量 RoundRobin 计算量

```
C(80, 2) × 2 × n_battle = 3160 × 2 = 6320 局
```

### 5.2 二分种子方案计算量

**最优情况**（前 3 个候选依次成功）：
```
甄别:  3 × 16 = 48 局
批量:  3 × 79 × 2 = 474 局
合计:  522 局  →  节省率 ≈ 91.7%
```

**典型情况**（4-6 个候选，部分失败）：
```
甄别:  5 × 16 = 80 局
批量:  3 × 79 × 2 = 474 局
合计:  554 局  →  节省率 ≈ 91.2%
```

**最坏情况**（12 候选全部失败 → fallback）：
```
甄别:  12 × 16 = 192 局
批量:  0 局（fallback 跳过）
合计:  192 局  →  节省率 ≈ 97.0%（但覆盖率可能下降）
```

### 5.3 关键风险点

1. **初始候选质量**：首个个体如果太强或太弱，败后需多次迭代
2. **M=16 是否足够**：16 局后 TrueSkill mu 的估计精度是否足以正确判断中位区间
3. **方差高的分布**：q 较大时，16 局的对战结果噪声大，可能误判
4. **双峰分布的种子困境**：强弱分明时，"中位种子"可能不存在或极难找到

---

## 6. 与现有实验框架的差异

现有 `tools/bisection_experiment/bisection_coordinator.py` 实现的是 v4 版本，与 `original_rule.txt` 存在以下关键差异：

| 规则项 | original_rule.txt | 现有 v4 实现 | 影响 |
|--------|:---|:---|------|
| 初始候选 | 第 1 个个体 | 随机 P=5 个 | 影响发现效率 |
| 升级条件 | M+对手后 mu ∈ Top33%-Top66% | M 对手且全程 streak=M | original 更宽松 |
| 失败条件 | M+对手后 mu ∉ Top33%-Top66% | N=10 对手且 streak<1 | original 判定更早 |
| max_candidate_fails | 累计 12 | 连续 3 | original 更宽容 |
| M 值 | 16 | 20 | 原始方案对战更少 |
| n_battle | 1 | 2 | 原始方案批量局数减半 |

实验需要对 `BisectionCoordinator` 做适配以对齐 original 规则。

---

## 7. 实施步骤

### Step 1: 适配 BisectionCoordinator

修改 `tools/bisection_experiment/bisection_coordinator.py`（或创建新文件），对齐 original 规则：
- 初始候选改为单个（首个个体）
- 升级/失败判定改为基于 Top33%-Top66% 区间
- M 改为 16
- max_candidate_fails 改为累计 12
- n_battle 改为 1

### Step 2: 扩展实验矩阵

修改 `tools/bisection_experiment/experiment.py` 的 `run.py`：
- 增加 `p_dist × q_mode × seed` 三维交叉场景
- 支持多 seed 重复实验

### Step 3: 增强指标

修改 `tools/bisection_experiment/metrics.py`：
- 增加节省率、对局分解（甄别/批量/有效/无效）
- 增加平均排名偏差

### Step 4: 运行实验

```bash
python -m ppo_v10.tools.bisection_experiment.run \
    --preset full \
    --output-dir ppo_v10/outputs/bisection_experiment_original_rule
```

### Step 5: 分析 & 可视化

- 汇总 45 组指标，分 p 分布和 q 模式做对比
- 输出 Top16 重叠率的均值 ± 标准差
- 输出节省率的均值 ± 标准差
- 分析 fallback 触发场景的共性

---

## 8. 实验环境

| 项目 | 说明 |
|------|------|
| Python | 3.10+ |
| 依赖 | trueskill, numpy, scipy, loguru |
| 对战模拟 | 纯随机抽样模拟（不启动真实 SDK），单组实验秒级完成 |
| 并行 | max_workers=25（影响 task 处理的并发度，不影响对战语义） |
