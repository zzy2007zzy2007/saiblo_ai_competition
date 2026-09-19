# PPO_V10 需求说明书

> 基于 ppo_v9 代码库，重新设计的遗传进化训练系统 v10。

---

## 1. 概述

PPO_V10 在 PPO_V9 的遗传进化框架基础上，引入**双池双阶段选拔机制**，用 TrueSkill 替代 ELO 评分体系，并通过 Kendall τ 区分度指标和余弦基因相似度过滤来提升选拔质量。

---

## 2. 核心架构变更总览

| 维度 | PPO_V9 | PPO_V10 |
|------|--------|---------|
| 选拔流程 | 单阶段：个体 vs 参照物 → ELO 排名 | 双阶段：基准测试 → Top N 循环赛 |
| 评分体系 | ELO（代内排名）+ TrueSkill（跨代对手池） | 纯 TrueSkill |
| 对手池 | 单一 OpponentPool（动态淘汰） | 基准对手池 + 精英种子池（独立维护） |
| 参照物选择 | 锚定参照物 + 动态参照物 | 基准对手池有序遍历，按区分度早停 |
| 区分度指标 | 无 | Kendall τ |
| 基因去重 | 仅种子选拔 | 种子选拔 + 基准池入池 |
| 诊断环节 | BaselineEvaluator（种子 vs 固定基线） | 同 v9，保持不变 |

---

## 3. 系统组件

### 3.1 基准对手池（Benchmark Opponent Pool）

**定义**：为基准测试阶段提供对战对手的 agent 集合。

**特性**：
- 由**内置 agent**（如 NoviceAI、MediumRuleAI）和**特定个体**（进化中选拔出的高区分度个体）构成
- 初始默认值：`["NoviceAI", "MediumRuleAI"]`（可配置）
- 每个 agent 带有**加入次序**（order/index），按次序递增
- 数量有限，**不需要淘汰机制**（区别于 v9 的 OpponentPool）
- 独立于精英种子池维护

**入池条件**（每代最多 1 个）：
1. 通过循环赛阶段，其 Kendall τ 区分度 > `MediumRuleAI` 的区分度
2. 通过基因相似度过滤：与池中已有个体的 cos 相似度 ≤ 配置阈值
3. 每代只取区分度最高的 1 个（如有多个满足条件）

**持久化**：状态保存到 `league/benchmark_pool.json`

### 3.2 精英种子池（Elite Seed Pool）

**定义**：记录每一代选拔出的种子和精英，用于分析模型迭代效果。

**特性**：
- 每代结束后，将当轮种子和精英信息追加到池中
- 记录内容：个体 ID、代数、TrueSkill 评分、种子排名、模型检查点路径
- 数量有限，**不需要淘汰机制**
- 独立于基准对手池维护
- 主要用于**离线分析**（不参与主流对战）

**持久化**：状态保存到 `league/elite_seed_pool.json`

### 3.3 内置 Agent（Built-in Agents）

沿用 v9 的规则基 agent，强度从低到高：

| Agent | 强度定位 | 文件位置 |
|-------|---------|---------|
| `BasicRandomAI` | 随机选择合法动作 | `rule_based_agents.py` |
| `NoviceAI` | 仅按优先级建塔（最多 4 座） | `rule_based_agents.py` |
| `BasicTowerAI` | 建塔 + 升级 + 超级武器 + 科技 | `rule_based_agents.py` |
| `MediumRuleAI` | 多阶段决策链（11 步优先级） | `rule_based_agents.py` |

---

## 4. 主流程（每代）

```
一代流程:

1. 种群生成
   ├── 第 1 代: 随机初始化 population_size 个个体
   └── 后续代: 精英保留 + 种子杂交 + 变异

2. 个体选择阶段 ── 步骤一：基准测试
   ├── 获取基准对手池（按加入次序排序）
   ├── 依次取池中 agent 与全部个体对战 (n_battle=2)
   ├── 每次对战后计算该 agent 的 Kendall τ 区分度
   ├── 区分度 > 阈值 → 停止（不再使用后续池中 agent）
   └── 池用尽仍未达标 → 使用所有已完成的战斗结果
   输出: 全部个体的 TrueSkill 排名

3. 个体选择阶段 ── 步骤二：循环赛
   ├── 取基准测试 Top N（可配置，如 N=24）进入循环赛
   ├── 全圆桌对战（每对 agent 对战 n_battle 局）
   ├── TrueSkill 排名
   ├── 基因相似度过滤 → 选出 seeds（num_seeds 个）
   ├── 前 elitism_count 个为 elites（直接进入下一代）
   └── 输出: seeds, elites, 循环赛 TrueSkill 排名

4. 基准对手池更新
   ├── 评估循环赛中个体的 Kendall τ 区分度
   ├── 找出区分度 > MediumRuleAI 区分度 的个体
   ├── 基因相似度过滤（vs 池中已有个体）
   └── 选出区分度最高的 1 个加入基准池（如有）

5. 精英种子池更新
   └── 将本轮 seeds/elites 追加到精英种子池

6. 诊断环节（基线验证）
   ├── 用内置 agent 对 seeds 做胜率测试
   └── 与 v9 BaselineEvaluator 完全相同

7. 日志与持久化
   └── 保存 checkpoint、pool 状态、TensorBoard 日志
```

---

## 5. 各阶段详细设计

### 5.1 基准测试（Benchmark Test）

#### 5.1.1 流程

```
输入: 当前代全部个体 (population_size 个), 基准对手池
输出: 全部个体的 TrueSkill 排名

for (benchmark_agent, order) in 基准对手池(按 order 升序):
    1. 该 benchmark_agent 与全部个体各对战 n_battle=2 局（先后手各 1）
    2. 更新 TrueSkill 评分（所有对战累计）
    3. 计算该 benchmark_agent 的 Kendall τ 区分度:
       - 基准排名 R_base: 仅根据当前 benchmark_agent 的对战结果排序
       - 参考排名 R_ref: 根据累计所有对战结果（含之前轮次）的 TrueSkill mu 排序
       - τ = Kendall(R_base, R_ref)
    4. if τ > kendall_tau_threshold:
        停止，不再使用后续 benchmark_agent
    5. else:
        继续下一个 benchmark_agent

如果池用尽仍未达标: 使用所有已完成的对战结果
```

#### 5.1.2 Kendall τ 计算

- 对两个排名列表（长度 N = population_size），计算 Kendall τ 相关系数
- τ = (一致对数 - 不一致对数) / (N*(N-1)/2)
- 范围 [-1, 1]，越接近 1 表示两个排名越一致
- 阈值可配置（如 0.7）

#### 5.1.3 TrueSkill 更新

- 每局对战（先后手各视为独立对局），用 trueskill 库更新双方评分
- 使用与 v9 相同的 TrueSkillRating 封装（mu=25.0, sigma=8.33 初始化）
- 全部个体和基准池 agent 的 TrueSkill 评分在跨代间持久化

#### 5.1.4 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `benchmark.n_battles` | 每个 (个体, benchmark_agent) 对战局数 | 2 |
| `benchmark.kendall_tau_threshold` | 区分度阈值 | 0.7 |
| `benchmark.max_workers` | 并行对战工作进程数 | 同 v9 |
| `benchmark.initial_pool` | 初始基准对手池 | ["NoviceAI", "MediumRuleAI"] |

### 5.2 循环赛（Round-Robin Tournament）

#### 5.2.1 流程

```
输入: 基准测试排名 Top N 的个体
输出: seeds 列表, elites 列表

1. Top N 个体进行全圆桌循环赛:
   每对 (个体_i, 个体_j) 进行 n_battle 局对战（先后手各半）
   
2. 基于所有对战结果，用 TrueSkill 更新评分

3. 按 TrueSkill mu 降序排名

4. 基因相似度过滤选出 seeds:
   - 方法与 v9 select_seeds_diverse() 相同
   - 硬阈值 hard_threshold（cos > 此值 → 跳过）
   - 软阈值 soft_threshold（cos > 此值 → 同家族限流）
   - 软限流名额 soft_max_per_family
   - 兜底逻辑：不足时逐级放宽

5. 前 elitism_count 个 seeds 标记为 elites

6. 设置 seed_rank
```

#### 5.2.2 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `round_robin.top_n` | 进入循环赛的个体数 | 24 |
| `round_robin.n_battles` | 每对对战局数 | 待定 |
| `round_robin.max_workers` | 并行对战工作进程数 | 同 v9 |

### 5.3 基准对手池更新

#### 5.3.1 流程

```
输入: 循环赛结果, 当前基准对手池
输出: 更新后的基准对手池（最多新增 1 个）

1. 计算 MediumRuleAI 的 Kendall τ 区分度（基准值）:
   - 使用循环赛的 TrueSkill 排名作为参考排名 R_ref
   - 使用 MediumRuleAI 在基准测试阶段的对战结果排名作为 R_med
   - τ_med = Kendall(R_med, R_ref)

2. 对于循环赛中的每个个体 i:
   - 使用个体 i 在循环赛中的对战结果排名作为 R_i
   - τ_i = Kendall(R_i, R_ref)
   - 如果 τ_i > τ_med: 候选加入

3. 基因相似度过滤:
   - 计算候选个体与基准池中每个已有个体的 cos 相似度
   - 如果 max(cos) > pool_similarity_threshold: 排除

4. 从通过过滤的候选中，选 τ 最高的 1 个加入基准池

5. 记录加入次序（order = 当前池大小）
```

#### 5.3.2 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `benchmark_pool.max_add_per_gen` | 每代最多入池数 | 1 |
| `benchmark_pool.similarity_threshold` | 入池基因相似度阈值 | 0.85 |

### 5.4 基因相似度过滤

- 使用与 v9 完全相同的 `cosine_similarity()` 函数（`genome/genome.py`）
- 对两个 Genome 的扁平 weights 向量计算 cos 相似度
- 两处使用：
  1. **种子选拔**：与 v9 的 `select_seeds_diverse()` 相同
  2. **基准池入池**：候选个体 vs 池中已有个体的 max cos ≤ 阈值

### 5.5 诊断环节

- 每代结束后，使用内置 agent 对 seeds 做胜率测试
- 与 v9 的 `BaselineEvaluator` 完全相同
- 输出胜率报告到 `evaluation/` 目录

---

## 6. 配置结构

### 6.1 完整配置项（YAML）

```yaml
# 系统配置
system:
  device: "cuda"
  seed: 42

# 网络结构配置
network:
  hidden_dim: 256
  enable_auxiliary: true

# 进化参数
evolution:
  max_generations: 200
  population_size: 80
  num_seeds: 8
  elitism_count: 2
  convergence_window: 20
  convergence_threshold: 5.0        # 注意：改用 TrueSkill mu 变动阈值

# 杂交参数
crossover:
  method: "layer_wise"
  rate: 0.8

# 变异参数
mutation:
  method: "gaussian"
  rate: 0.1
  scale: 0.03
  decay: 0.99

# 基准测试参数（新增）
benchmark:
  n_battles: 2
  kendall_tau_threshold: 0.7
  max_workers: 25
  initial_pool:
    - "NoviceAI"
    - "MediumRuleAI"

# 循环赛参数（新增）
round_robin:
  top_n: 24
  n_battles: 4                     # 待定
  max_workers: 25

# 基因相似度过滤（复用 + 扩展）
elite_dedup:
  enabled: true
  hard_threshold: 0.95
  soft_threshold: 0.85
  soft_max_per_family: 2

# 基准对手池参数（新增）
benchmark_pool:
  max_add_per_gen: 1
  similarity_threshold: 0.85

# 基线验证参数（保持不变）
baseline_battle:
  enabled: true
  interval: 1
  n_battles: 16
  max_workers: 25
  agents:
    - "MediumRuleAI"
    - "BasicTowerAI"
    - "nn_gen_32_cuda"
    - "rule_zzy25"

# 环境参数
env:
  player_id: 0
  backend_type: "python"
  max_steps: 512

# 日志参数
logging:
  log_level: "INFO"
  save_interval: 10
  tensorboard: true
```

### 6.2 删除的配置项

以下 v9 配置项在 v10 中**不再需要**：

- `selection.method` — 不再有 ELO 选拔方法选择
- `selection.anchor_references` — 替换为 `benchmark.initial_pool`
- `selection.dynamic_references` — 不再有动态参照物概念
- `selection.include_previous_champion` — 上代冠军不再特殊处理
- `selection.elo_k_factor` — ELO 参数移除
- `selection.elo_initial` — ELO 参数移除
- `opponent_pool.*` — 旧的对手池被基准对手池替代

---

## 7. 目录结构与日志（完全沿用 PPO_V9）

日志目录、目录结构、对战日志、回合日志等**完全保持与 PPO_V9 相同**。仅新增 league/ 下的两个池状态文件：

```
outputs/<run_id>/
  training/                    # 训练日志（ga_training_*.log，与 v9 相同）
  checkpoint/                  # 检查点（checkpoint_gen_XXXX.pt，与 v9 相同）
  battle/                      # 对战日志 & 回合日志（与 v9 完全相同）
  selfplay/                    # 保留（与 v9 相同）
  system/                      # 系统指标（system_metrics.json，与 v9 相同）
  evaluation/                  # 基线验证结果（与 v9 相同）
  generations/
    gen_XXXX/
      population_stats.json    # 种群统计（与 v9 相同）
      battle_results.json      # 对战结果（与 v9 相同）
      evaluation.json          # 基线验证（与 v9 相同）
      seed_models/
        seed_1.pt              # 种子模型（与 v9 相同）
        ...
  league/
    benchmark_pool.json        # 新增：基准对手池状态
    elite_seed_pool.json       # 新增：精英种子池状态
    payoff.json                # TrueSkill payoff 矩阵（与 v9 相同）
  tensorboard/                 # TensorBoard 日志（与 v9 相同）
```

---

## 8. 代码结构

沿用 v9 的包结构，新增/修改以下模块：

| 模块 | 变更 |
|------|------|
| `selection/benchmark_test.py` | **新增** — 基准测试阶段 |
| `selection/round_robin.py` | **新增** — 循环赛阶段 |
| `selection/kendall_tau.py` | **新增** — Kendall τ 计算 |
| `league/benchmark_pool.py` | **新增** — 基准对手池管理 |
| `league/elite_seed_pool.py` | **新增** — 精英种子池管理 |
| `selection/battle_selection.py` | 重构为调用 benchmark + round_robin |
| `selection/elo_rating.py` | **移除** — 不再使用 ELO |
| `league/opponent_pool.py` | **移除** — 被 benchmark_pool 替代 |
| `league/opponent_selector.py` | **移除** — 不再需要对手选择器 |
| `trainer/genetic_evolution_trainer.py` | 重写主循环编排 |
| `config/config_parser.py` | 更新配置项和默认值 |
| `configs/ga_evo.yaml` | 更新配置文件 |

---

## 9. 重要未明确细节（Open Questions）

以下细节在需求中尚未明确，需要确认：

### 9.1 Kendall τ 计算的参考排名问题

**问题**：基准测试中，Kendall τ 需要"参考排名"来与 benchmark_agent 的排名比较。参考排名如何定义？

**可能方案 A**：使用**累计 TrueSkill 排名**（包含当前及之前所有 benchmark_agent 的对战结果）作为参考。这确保了随着更多 agent 加入对战，参考排名越来越准确。

**可能方案 B**：使用**上一代循环赛排名**作为参考。但这导致第 1 代无参考可用的困境。

**建议**：方案 A 更自然，也与"依次添加、早停"的流程吻合。

### 9.2 循环赛 n_battles 的值

**问题**：循环赛中每对个体对战多少局？

**分析**：n_battles 影响 TrueSkill 收敛速度。n_battles=2（先后手各 1）可能不够稳定；n_battles=4 或更多可以产生更可靠的评分，但计算成本高。

**建议**：默认 n_battles=4，可配置。

### 9.3 基准测试中 benchmark_agent 的排名如何计算

**问题**：每个 benchmark_agent 与所有个体对战后，如何生成"仅根据该 agent 的对战结果"的个体排名？

**分析**：可以基于该 agent 的 n_battle=2 局对战结果：
- 方案 A：直接用胜率排序（2 局太少，噪音大）
- 方案 B：用该 agent 的对战结果单独做 TrueSkill 排名
- 方案 C：用平均剩余 HP / 金币差值等连续指标排序

**建议**：方案 B — 用该 benchmark_agent 的对战结果独立计算 TrueSkill 排名，与累计排名比较。

### 9.4 收敛判断指标

**问题**：v9 用 "best ELO 在窗口内变动 < 阈值" 判断收敛。v10 改用 TrueSkill，收敛判断用什么指标？

**可能方案**：
- 方案 A：best TrueSkill mu 在窗口内变动 < 阈值
- 方案 B：best TrueSkill mu - 2*sigma（保守估计）变动 < 阈值
- 方案 C：种群平均 TrueSkill mu 变动 < 阈值

**建议**：方案 A，沿用 v9 逻辑但替换指标为 TrueSkill mu。

### 9.5 基准池入池时的区分度计算口径

**问题**：第 5.3.1 节中，"MediumRuleAI 的区分度"和"个体的区分度"基于什么数据计算？

**分析**：
- 基准测试阶段已有每个 benchmark_agent 的 τ，包括 MediumRuleAI
- 循环赛阶段有所有 Top N 个体的 TrueSkill 排名
- 最自然的做法：用循环赛的 TrueSkill 排名作为 R_ref，用个体在循环赛中的对战结果排名作为 R_i

但还有一种可能：直接用基准测试阶段计算的 τ 值来判断。

**建议**：使用循环赛排名作为参考，因为循环赛排名更可靠。个体的循环赛排名 = 仅根据该个体在循环赛中的对局结果排出的其他个体排名。

### 9.6 精英种子池的具体用途

**问题**：精英种子池"用于分析模型迭代效果"——这个分析是离线进行的还是系统内置的？

**分析**：
- 如果仅用于离线分析，则只需持久化数据即可
- 如果是系统内置的分析功能，需要定义分析指标和输出格式

**建议**：精英种子池仅做数据记录和持久化，分析和可视化由外部工具完成。池中记录内容：`[{generation, individual_id, seed_rank, trueskill_mu, trueskill_sigma, checkpoint_path, genome_hash}]`。

### 9.7 循环赛 Top N 的 TrueSkill 初始化

**问题**：进入循环赛的 Top N 个体，其 TrueSkill 评分是从基准测试继承，还是重新初始化？

**建议**：从基准测试继承。这样可以利用基准测试阶段积累的信息。

### 9.8 基准测试早停后的剩余 benchmark_agent

**问题**：如果第 k 个 benchmark_agent 的对战使得 τ > threshold 后停止，那么第 k+1 个及之后的 agent 不在本轮使用。它们是否需要在跨代时保持"未使用"状态以便将来使用？

**建议**：不区分"已使用/未使用"状态。每代都从第一个 benchmark_agent 开始依次使用。这样设计最简单，也确保每代总是优先使用高优先级（早加入）的 agent。

### 9.9 初始代（第 1 代）的基准测试

**问题**：第 1 代没有上一代的循环赛排名作为参考。基准测试的 Kendall τ 计算和第 1 个 benchmark_agent 的 τ 如何计算？

**建议**：
- 第 1 个 benchmark_agent 的 τ：仅用该 agent 的对战结果排名 vs 随机排名 → τ ≈ 0（即不早停）
- 从第 2 个 benchmark_agent 开始：用累计 TrueSkill 排名作为参考
- 确保第 1 代至少使用 2 个 benchmark_agent

### 9.10 基因相似度过滤在基准池入池时的处理

**问题**：如果候选个体的 cos 相似度刚好超过阈值一点点（如 0.86 vs threshold 0.85），是严格拒绝还是需要兜底？

**建议**：严格拒绝。基准池不需要兜底逻辑（与种子选拔不同），因为基准池不要求填满固定数量。

---

## 10. 与 PPO_V9 保持不变的部分

以下部分与 v9 完全一致，无变更：

- 网络结构（`AntWarPolicyValueNetwork`）
- 基因组表示（`Genome`、`extract_genome`、`genome_to_state_dict`、`cosine_similarity`）
- 杂交算法（`layer_wise_crossover`）
- 变异算法（`gaussian_mutation` + decay）
- 种群管理（`PopulationManager`、`Individual`、`Population`）
- 对战模拟器（`BattleSimulator`、`_execute_battle`）
- Agent 加载器（`AgentLoader`、`GAWarAgent`、`PPOWarAgent`）
- 基线评估器（`BaselineEvaluator`）— 诊断环节
- 系统监控（`SystemMetricsSampler`）
- 日志子系统（`GALoggingSubsystem`）
- 检查点管理（`GACheckpointManager`）
- TrueSkill 评分（`TrueSkillRating`、`BattleSharedPayoff`）
- 所有内置 agent（`BasicRandomAI`、`NoviceAI`、`BasicTowerAI`、`MediumRuleAI`）
- 环境接口（`ObservationEncoder`、`ActionMaskHandler`）
- 目录结构和日志格式
