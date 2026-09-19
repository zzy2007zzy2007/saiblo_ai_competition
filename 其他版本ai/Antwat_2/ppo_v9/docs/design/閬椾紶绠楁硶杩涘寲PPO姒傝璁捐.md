# 基于遗传算法进化策略网络的训练框架 —— 概要设计

## 1. 概述

### 1.1 背景

> **说明**：本方案与 PPO（Proximal Policy Optimization）算法完全无关。PPO 是一种基于梯度下降的策略梯度算法，而本方案使用遗传算法搜索网络参数，两者在优化原理上没有任何共同之处。下文中所涉及的"策略网络"仅指其**网络架构设计**（CNN 编码器 + MLP 投影 + 动作头）复用了此前 PPO 项目中的网络结构；至于如何优化该网络的权重，本方案不涉及任何 PPO 的训练方法，完全采用遗传算法。

本框架使用遗传算法（Genetic Algorithm, GA）搜索策略网络的参数空间，整个过程不依赖梯度计算。

核心思想：将策略网络视为一个"基因组"，通过**随机初始化 → 对战评估 → 种子选拔 → 杂交变异 → 下一代**的进化循环，迭代优化网络权重。

### 1.2 目标

- 实现基于 GA 进化策略网络权重的完整训练框架
- 支持高度可配置的种群管理、对战选拔、杂交变异机制
- 通过固定参照物对战，实现跨代可比的模型评分
- 通过 Baseline battle 验证每一代种子模型的绝对质量

### 1.3 关键设计决策

| 决策点 | 选型 | 理由 |
|--------|------|------|
| 网络权重初始化 | 完全随机初始化（均匀分布） | 最大化初始种群多样性，不受先验约束 |
| 杂交操作 | 层级杂交（layer-wise） | 保留网络层结构完整性，破坏性较小 |
| 变异操作 | 全局高斯变异 | 实现简单，参数少，适用于~1000万参数的搜索空间 |
| 代内评分 | ELO（个体 vs 固定参照物） | 参照物固定 = 标尺固定 = 分数跨代可比 |
| 跨代对手池 | TrueSkill 双值评分 | sigma 天然提供不确定性信号，驱动探索/利用平衡 |
| 对战对手 | TrueSkill 对手池 + 固定 Baseline agents | 兼顾绝对标尺和精细区分度 |

---

## 2. 总体架构

### 2.1 系统模块图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GeneticEvolutionTrainer                          │
│  （主控制器 — 编排整个进化流程）                                       │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                        │
         ▼                    ▼                        ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ PopulationManager │  │ BattleSelection  │  │  BaselineEvaluator  │
│  （种群管理）      │  │  （对战选拔）     │  │  （基线验证）        │
├─────────────────┤  ├─────────────────┤  ├─────────────────────┤
│ • 个体基因组管理  │  │ • 参照物管理     │  │ • vs baseline AI    │
│ • 种群初始化     │  │ • ELO 评分       │  │ • 可配置开关        │
│ • 杂交操作      │  │ • 种子选拔       │  │ • 报告生成          │
│ • 变异操作      │  │ • 对战日志       │  │                     │
│ • 精英保留      │  │                 │  │                     │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │  OpponentPool    │
                      │  （对手池管理）    │
                      ├─────────────────┤
                      │ • TrueSkill 评分  │
                      │ • 淘汰策略       │
                      │ • 探索/利用选择   │
                      │ • 状态持久化     │
                      └─────────────────┘
```

### 2.2 各模块职责

| 模块 | 职责 | 不负责 |
|------|------|--------|
| `GeneticEvolutionTrainer` | 编排主循环：评估→选拔→验证→进化→日志 | 不涉及对战执行、权重计算 |
| `PopulationManager` | 个体基因组存储、初始化、杂交、变异 | 不关心个体如何被评估 |
| `BattleSelection` | 编排个体与参照物的对战、ELO 评分、种子选拔 | 不关心种子后续用途 |
| `BaselineEvaluator` | 种子与固定 baseline AI 对战，验证绝对质量 | 不知道种子从何而来 |
| `OpponentPool` | TrueSkill 评分维护、对手池淘汰、探索/利用选择 | 不参与代内排名 |

---

## 3. 网络权重初始化（完全随机初始化）

### 3.1 初始化对象

复用策略价值网络（Policy-Value Network），包含：
- **Policy 分支**：策略 CNN 编码器（六边形卷积）+ 策略 MLP 编码器 + 投影层 + 动作头（StructuredActionHead）
- **Value 分支**：价值 CNN 编码器 + 价值 MLP 编码器 + 投影层 + 价值头（ValueHead）
- **辅助头**（可选）：塔伤害预测、金币收入预测、基地伤害预测（Policy 和 Value 各一套）

### 3.2 初始化策略

所有可训练参数使用**均匀分布** `U(-k, k)` 初始化，其中 `k = 1 / sqrt(fan_in)`：

| 层类型 | 初始化范围 | 理由 |
|--------|-----------|------|
| Conv2d 权重 | U(-1/√fan_in, 1/√fan_in) | 保持输入方差稳定 |
| Conv2d 偏置 | 0 | 默认零偏置 |
| Linear 权重 | U(-1/√fan_in, 1/√fan_in) | 保持输入方差稳定 |
| Linear 偏置 | 0 | 默认零偏置 |
| LayerNorm weight | 1 | 归一化层初始为单位缩放 |
| LayerNorm bias | 0 | 归一化层初始为零偏移 |
| BatchNorm weight | 1 | 同上 |
| BatchNorm bias | 0 | 同上 |

注：不使用正交初始化（orthogonal_init），因为正交初始化引入的先验偏向于特定初始化分布，与"完全随机"的初衷不符。

### 3.3 基因组表示

每个个体的基因组是一个扁平化的一维权重向量：

```python
@dataclass
class Genome:
    weights: np.ndarray              # 一维权重向量（float32），长度 ~1000 万
    shapes: List[Tuple[int, ...]]    # 各层的原始形状，用于重建网络
    param_count: int                 # 总参数量
    layer_boundaries: List[LayerBoundary]  # 层边界索引（用于层级杂交）

@dataclass
class LayerBoundary:
    start_idx: int                   # 在扁平向量中的起始位置
    end_idx: int                     # 在扁平向量中的结束位置
    layer_type: str                  # conv | linear | norm | bias | head
    layer_name: str                  # 层名，如 policy_cnn.stage1.0
```

### 3.4 参数规模

| 组件 | 参数量（约） | 占比 |
|------|------------|------|
| Policy CNN（三阶段六边形卷积） | ~390 万 | ~39% |
| Policy MLP + 投影 | ~87 万 | ~8.7% |
| Value CNN（与 Policy 对称） | ~390 万 | ~39% |
| Value MLP + 投影 | ~87 万 | ~8.7% |
| Head 层（动作/价值/辅助） | ~13 万 | ~1.3% |
| 归一化层（BatchNorm/LayerNorm） | ~3 万 | ~0.3% |
| **总计** | **~1000 万** | **100%** |

---

## 4. 种群管理（PopulationManager）

### 4.1 数据结构

```python
@dataclass
class Individual:
    id: str                          # 唯一标识，如 "gen_001_ind_042"
    generation: int                  # 所属代次
    genome: Genome                   # 权重向量 + 形状/边界信息
    elo_rating: float = 1200.0       # 代内 ELO 评分
    seed_rank: int = -1              # 种子排名（-1 表示非种子）
    parent_ids: List[str] = field(default_factory=list)  # 亲本 ID
    mutation_desc: str = ""           # 变异描述

@dataclass
class Population:
    generation: int                  # 当前代次
    individuals: List[Individual]    # 所有个体
    best_elo: float                  # 本代最高 ELO
    mean_elo: float                  # 本代平均 ELO
    diversity: float                 # 种群多样性指标
```

### 4.2 第一代初始化

```python
def init_population(population_size: int = 80) -> Population:
    individuals = []
    for i in range(population_size):
        # 1. 创建策略网络实例
        network = AntWarPolicyValueNetwork(hidden_dim=256)
        
        # 2. 完全随机初始化（见第 3 章）
        random_init_network(network)
        
        # 3. 提取扁平权重向量
        genome = extract_genome(network)
        
        # 4. 生成个体
        ind = Individual(
            id=f"gen_000_ind_{i:03d}",
            generation=0,
            genome=genome,
        )
        individuals.append(ind)
    
    return Population(generation=0, individuals=individuals, ...)
```

### 4.3 精英保留

每一代评分最高的 `elitism_count`（默认 2）个个体，**不经过杂交和变异**，直接复制到下一代。确保最优解不会因随机操作丢失。

### 4.4 层级杂交（Crossover）

#### 4.4.1 原理

以网络的**层**（Layer）为基本单元进行杂交。两个亲本的对应层交换权重，而非在参数级别混合。

#### 4.4.2 算法

```
输入：亲本 A (Individual), 亲本 B (Individual)
输出：子代 C, 子代 D

1. 获取网络的层边界列表 layer_boundaries（在初始化 Genome 时已计算并保存）
2. 在层序列中随机选择一个切点（从全部 N 层中随机选一个切点索引 k）
3. 生成子代 C：
   - 前 k 层的权重 = 亲本 A 的对应层权重
   - 后 N-k 层的权重 = 亲本 B 的对应层权重
4. 生成子代 D：
   - 前 k 层的权重 = 亲本 B 的对应层权重
   - 后 N-k 层的权重 = 亲本 A 的对应层权重
5. 子代 C 和 D 的 parent_ids = [A.id, B.id]
```

```
层切点示意图（切点 k=3）：

亲本 A:  [层0][层1][层2]│[层3][层4]...[层N]
亲本 B:  [层0][层1][层2]│[层3][层4]...[层N]
                      ↑ 切点
子代 C:  [A层0][A层1][A层2]│[B层3][B层4]...[B层N]
子代 D:  [B层0][B层1][B层2]│[A层3][A层4]...[A层N]
```

#### 4.4.3 层分组说明

网络中的层按以下逻辑分组（每组的序号即杂交切点位置）：

| 层序号 | 层名 | 层类型 | 参数形状示例 |
|--------|------|--------|-------------|
| 0 | policy_cnn.stage1.0.weight | conv | (128, 28, 1, 1) |
| 1 | policy_cnn.stage1.0.bias | bias | (128,) |
| 2 | policy_cnn.stage1.1.weight | norm | (128,) |
| 3 | policy_cnn.stage1.1.bias | norm | (128,) |
| 4 | policy_cnn.hex_conv1.weight | hex_conv | (128, 128, 6) |
| 5 | policy_cnn.hex_conv1.bias | bias | (128,) |
| ... | ... | ... | ... |
| 34 | policy_proj.weight | linear | (256, 3328) |
| 35 | policy_proj.bias | bias | (256,) |
| ... | ... | ... | ... |
| M | action_head.type_head | head | (9, 256) |
| ... | ... | ... | ... |

注：实际层数约 80-120 层（含 Policy/Value 分支所有子层）。每一层（如 `Conv2d.weight` + `Conv2d.bias` 算两个独立层）都可以作为杂交切点。

#### 4.4.4 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 杂交概率 crossover_rate | 0.8 | 配对后执行杂交的概率，否则直接复制亲本 |
| 杂交方法 | layer_wise | 层级单点杂交（固定，本文档锁定此方案） |

### 4.5 全局高斯变异（Mutation）

#### 4.5.1 原理

对扁平权重向量的每个元素，以概率 `mutation_rate` 决定是否变异，被选中的元素加上高斯噪声 N(0, mutation_scale²)。

#### 4.5.2 算法

```python
def mutate(genome: Genome, rate: float = 0.1, scale: float = 0.03) -> Genome:
    # 1. 生成变异掩码：每个位置独立判定
    mask = torch.rand_like(genome.weights) < rate
    
    # 2. 生成高斯噪声
    noise = torch.randn_like(genome.weights) * scale
    
    # 3. 施加变异（仅 mask=True 的位置）
    genome.weights = genome.weights + noise * mask
    
    return genome
```

#### 4.5.3 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 变异率 mutation_rate | 0.1 | 每代约 10% 的参数发生变异 |
| 变异幅度 mutation_scale | 0.03 | 高斯噪声标准差的倍数 |
| 衰减速 mutation_decay | 0.99 | 每代衰减至 0.99 倍，100 代后约为初始的 37% |

#### 4.5.4 与层级杂交的交互

```
1. 种子选拔完成后，从种子集合中随机配对
2. 每对亲本执行层级杂交（crossover_rate 概率），生成 2 个子代
3. 对子代施加全局高斯变异
4. 重复直到新种群填满（population_size - elitism_count）
```

---

## 5. 对战选拔系统（BattleSelection）

### 5.1 评估方式：个体 vs 固定参照物

每个个体不与种群内其他个体对战，而是与一组**固定强度的参照物**对战。参照物分为两类：

#### 5.1.1 锚定参照（Anchor References）

永不更换，提供**绝对标尺**，确保 ELO 评分跨代可比：

| 参照物 | 初始 ELO | 策略描述 |
|--------|---------|---------|
| BasicRandomAI | 400 | 随机选择合法动作，最弱基线 |
| BasicTowerAI | 1000 | 优先建造和升级塔，辅以科技升级 |
| MediumRuleAI | 1600 | 11 阶段决策链，包含建造、科技、超级武器 |

#### 5.1.2 动态参照（Dynamic References）

每代更新，从对手池中选取，提供**精细区分度**：
- 对手池中 TrueSkill mu 最高的 2 个历史种子
- 上一代的冠军个体（Top-1 种子）

### 5.2 评估流程

```python
def evaluate_population(population: Population, references: List[Reference]) -> List[ELORating]:
    for individual in population.individuals:
        for ref in references:
            # 每个参照物战 2 局（先手/后手）
            for first_player in [0, 1]:  # 0=个体先手, 1=参照物先手
                result = battle(individual.genome, ref, first_player)
                update_elo(individual, ref, result)
    
    # 所有个体完成对战，按最终 ELO 排序
    return sorted(individuals, key=lambda ind: ind.elo_rating, reverse=True)
```

对战局数计算：
```
80 个体 × (3 锚定 + 2 动态 + 1 上代冠军 = 6 参照物) × 2 局（先手+后手）= 960 局/代
```

### 5.3 ELO 评分系统

#### 5.3.1 公式

```
期望得分：E_A = 1 / (1 + 10^((R_B - R_A) / 400))
新评分：  R_A' = R_A + K × (S_A - E_A)
  S_A = 1（胜）、0.5（平）、0（负）
```

#### 5.3.2 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 个体初始 ELO | 1200 | 低于最强参照物 MediumRuleAI（1600），有向上空间 |
| K 因子 | 64 | 相对参照物对战次数少，大 K 快速收敛 |
| 先手/后手补偿 | 无 | 对战数量对称，自然抵消 |

#### 5.3.3 简化方案（可选）

由于所有个体 vs 所有参照物的对战是**一次性批量完成**的，无需逐场更新 ELO。可以直接统计个体对参照物的**加权胜率**：

```python
weighted_win_rate = Σ(w_i × win_rate_vs_ref_i)

# 权重 w_i 根据参照物强度设定：
#   BasicRandomAI: 0.1（弱参照，权重低）
#   BasicTowerAI:  0.2
#   MediumRuleAI:  0.3（分水岭，权重最高）
#   历史 Top-1:    0.2
#   上代冠军:      0.2
```

直接用加权胜率排序选拔种子，效果等价于 ELO 排序。

### 5.4 种子选拔规则

- 按 ELO 评分降序排列，取前 `num_seeds` 个作为种子
- 默认 `num_seeds = max(5, population_size // 10)`（80 个体 → 8 个种子）
- 精英保留：`elitism_count`（默认 2）个最优个体直接进入下一代
- 收敛判断：若连续 N 代最高 ELO 无显著提升，可提前终止进化

### 5.5 对战执行

对战使用多进程并行执行，框架如下：

```
BattleSimulator:
  - 使用 ProcessPoolExecutor，默认 12 个 worker
  - 每个 worker 运行完整的游戏对局
  - 每局对战包括：
    1. 初始化游戏引擎（PythonBackendState）
    2. 加载双方 agent（个体网络 + 参照物 agent）
    3. 逐回合执行（每个回合双方各执行一个动作）
    4. 游戏结束或达到最大回合数（默认 512）时终止
    5. 返回对战结果（胜/负/平 + 统计数据）
  - 所有 worker 完成后汇总结果
```

---

## 6. 对手池管理（OpponentPool）

### 6.1 TrueSkill 评分系统

#### 6.1.1 原理

TrueSkill 使用双值 `(mu, sigma)` 描述玩家技能：

- **mu**：技能均值，初始 25.0。mu 越高 = 技能越强
- **sigma**：不确定性，初始 8.33。sigma 越高 = 对这个玩家的了解越少

对战结果后，通过**贝叶斯推断**同时更新 `(mu, sigma)`：

| 结果 | mu 变化 | sigma 变化 |
|------|---------|-----------|
| 胜 | mu ↑ | sigma ↓（更确信） |
| 负 | mu ↓ | sigma ↓（更确信） |
| 平 | mu 微调 | sigma ↓（更确信） |

#### 6.1.2 关键属性

```
置信区间: (mu - 3×sigma, mu + 3×sigma)  # 约 99.7% 置信区间
is_confident: sigma < 3.0               # 是否已充分了解该玩家
```

### 6.2 对手池结构

```python
@dataclass
class OpponentPool:
    max_size: int = 20                   # 对手池最大容量
    min_games_threshold: int = 8         # 淘汰保护门槛
    
    # 运行时状态
    opponents: List[str]                 # 对手 ID 有序列表
    checkpoint_paths: Dict[str, str]     # 每个对手的模型权重路径
    games_played: Dict[str, int]         # 每个对手的累计对局数
    trueskill_ratings: Dict[str, TrueSkillRating]  # TrueSkill 评分
```

### 6.3 添加与淘汰策略

**添加**：
1. 每一代选拔出的种子选手（Top-K）加入对手池
2. 如果池已满（`>= max_size`），触发淘汰

**淘汰策略**（评分 = mu + protection_bonus，最低者被淘汰）：

```
eviction_score = mu + protection_bonus
protection_bonus = max(0, min_games_threshold - games_played) × 0.5
```

保护机制确保新加入的对手（对局次数少）不会立即被淘汰。

### 6.4 探索/利用选择

从对手池中选取动态参照物时，使用探索/利用平衡策略：

```
exploit_prob = 0.3 + 0.6 × progress    # 从 0.3 线性增长到 0.9

利用（Exploit）- 选最强对手：
  1. 筛选 sigma < 3.0 的可信对手
  2. 按 mu 降序排列
  3. 从前 EXPLOIT_TOP_K（默认 3）中随机选一个

探索（Explore）- 选"最陌生"对手：
  1. 筛选 sigma > 5.0 的高不确定性对手
  2. 如果筛选结果为空，从全部对手中随机选
```

### 6.5 可利用性（Exploitability）

计算当前种群相对于对手池的综合水平：

```
exploitability = 0.5 + (avg_opponent_mu - current_mu) / (current_sigma × 2) × 0.5

  0.0：远强于对手池
  0.5：与对手池相当
  1.0：远弱于对手池
```

当 exploitability 持续偏低（种群远强于对手池），说明对手池需要更新更强的参照物，以避免过拟合到弱对手。

### 6.6 状态持久化

对手池状态持久化为 JSON 文件：

```json
// pool.json
{
  "opponents": ["opponent_ep001", "opponent_ep002", ...],
  "checkpoint_paths": {
    "opponent_ep001": "outputs/run_001/generations/gen_001/seed_models/seed_0.pt"
  },
  "games_played": {"opponent_ep001": 24},
  "added_episodes": {"opponent_ep001": 1}
}

// payoff.json
{
  "players": ["opponent_ep001", "opponent_ep002"],
  "trueskill_ratings": {
    "opponent_ep001": {"mu": 28.5, "sigma": 2.3, "games_played": 24}
  },
  "data": {
    "opponent_ep001-opponent_ep002": {"wins": 3, "draws": 1, "losses": 2, "games": 6}
  }
}
```

---

## 7. 基线验证（BaselineEvaluator）

### 7.1 功能定位

每一代种子选拔完成后，对种子选手执行与 baseline agents 的对战评估，验证其**绝对质量**（相对于 ELO 评分的独立验证）。

### 7.2 Baseline Agents 列表

| Agent 名称 | 强度级别 | 策略描述 |
|-----------|---------|---------|
| BasicRandomAI | 弱 | 随机选择合法动作 |
| BasicTowerAI | 中弱 | 预定义建造顺序（优先造塔 → 升级塔 → 科技升级） |
| MediumRuleAI | 中强 | 11 阶段决策链，包含多阶段建造、超级武器、防御策略 |

### 7.3 配置开关

```yaml
baseline_battle:
  enabled: true              # 是否启用
  interval: 1                # 每 N 代执行一次（1 = 每代执行）
  n_battles: 12              # 每个 baseline agent 的对战场次（每场含先手+后手）
```

### 7.4 与选拔系统的隔离

- `BaselineEvaluator` 不关心种子是如何选拔出来的
- `BattleSelection` 不知道其后是否有基线验证
- `GeneticEvolutionTrainer` 负责串联两者：先选拔 → 再验证

---

## 8. 主训练循环（GeneticEvolutionTrainer）

### 8.1 完整流程

```
主循环（for generation in 1..max_generations）:

  阶段一：种群生成
    if generation == 1:
        完全随机初始化生成 population_size 个个体
    else:
        1. 精英保留（top elitism_count 直接进入下一代）
        2. 从种子池中随机配对
        3. 每对执行层级杂交，生成子代
        4. 对子代施加全局高斯变异
        5. 重复直到填满 population_size
        6. 新种群出生代 = generation
    
  阶段二：对战评估
    1. 更新动态参照物列表（从对手池中选取）
    2. 遍历每个个体 vs 每个参照物 × 2 局（先手/后手）
    3. 每局对战通过 BattleSimulator（多进程）执行
    4. 收集对战结果，更新 ELO 评分
    
  阶段三：种子选拔
    1. 按 ELO 降序排列所有个体
    2. 取前 num_seeds 个作为种子
    3. 打印本代统计（最佳/平均 ELO、多样性指标）
    
  阶段四：基线验证（条件执行）
    if baseline_battle.enabled and generation % interval == 0:
        对每个种子执行 vs baseline agent 对战
        记录种子 vs baseline 的胜率到日志
    
  阶段五：对手池更新
    1. 种子加入对手池（保存 .pt 权重文件）
    2. 更新 TrueSkill 评分（与对手池现有对手对战记录更新）
    3. 池满时淘汰最弱对手
    
  阶段六：日志与持久化
    1. 记录本代统计到 JSON/TensorBoard
    2. 保存对战日志
    3. 每 save_interval 代保存完整 checkpoint
```

### 8.2 伪代码

```python
class GeneticEvolutionTrainer:
    def __init__(self, config):
        self.population_mgr = PopulationManager(config.population_size)
        self.battle_selection = BattleSelection(config.selection)
        self.baseline_evaluator = BaselineEvaluator(config.baseline_battle)
        self.opponent_pool = OpponentPool(
            max_size=20, min_games_threshold=8
        )
        self.logger = GALogger(config.logging)
    
    def train(self):
        for gen in range(1, self.config.max_generations + 1):
            # 阶段一
            if gen == 1:
                population = self.population_mgr.init_random()
            else:
                population = self.population_mgr.evolve(seeds)
            
            # 阶段二
            references = self._build_references()
            results = self.battle_selection.evaluate(population, references)
            
            # 阶段三
            seeds = self.battle_selection.select_seeds(results, top_k=8)
            
            # 阶段四
            if self.config.baseline_battle.enabled and gen % self.config.baseline_battle.interval == 0:
                for seed in seeds:
                    baseline_results = self.baseline_evaluator.evaluate(seed)
            
            # 阶段五
            for seed in seeds:
                self.opponent_pool.add(seed.genome, seed.id, gen)
            
            # 阶段六
            self.logger.log_generation(gen, population, seeds, baseline_results)
            if gen % self.config.save_interval == 0:
                self.save_checkpoint(gen, population, self.opponent_pool)
```

### 8.3 收敛判断

```python
# 连续 N 代最优 ELO 提升 < threshold 时提前终止
if gen >= WINDOW_SIZE:
    recent_best = [gen_history[g].best_elo for g in range(gen-WINDOW_SIZE, gen)]
    if max(recent_best) - min(recent_best) < CONVERGENCE_THRESHOLD:
        logger.info(f"Converged at generation {gen}")
        break
```

---

## 9. 日志系统

### 9.1 日志目录结构

```
outputs/{run_id}/
├── training/
│   ├── training_start.log          # 训练启动配置快照
│   ├── training_complete.log       # 训练完成汇总
│   ├── ga_training_{time}.log      # loguru 主日志
│   └── ga_training_error_{time}.log # loguru 错误日志
├── generations/
│   ├── gen_0001/
│   │   ├── population_stats.json   # 种群统计（最佳/平均/中位数 ELO，标准差）
│   │   ├── battle_results.json     # ELO 排名、对战矩阵
│   │   ├── battle_log.jsonl        # 每场对战详细日志
│   │   ├── evaluation.json         # Baseline 验证结果
│   │   └── seed_models/
│   │       ├── seed_0.pt
│   │       └── seed_1.pt ...
│   ├── gen_0002/
│   └── ...
├── checkpoints/
│   ├── checkpoint_gen_0010.pt      # 完整训练状态（种群 + 对手池 + 配置）
│   └── checkpoint_gen_0020.pt ...
├── tensorboard/
│   └── events.out.tfevents.xxx
└── league/
    ├── pool.json                   # 对手池状态
    └── payoff.json                 # TrueSkill 评分矩阵
```

### 9.2 日志内容

| 日志文件 | 内容 | 格式 |
|---------|------|------|
| `population_stats.json` | 最佳/平均/中位数/最差 ELO，多样性指标（权重向量距离矩阵均值） | JSON |
| `battle_results.json` | ELO 排名表（个体 ID → ELO 评分），Top-K 种子名单 | JSON |
| `battle_log.jsonl` | 每场对战的参与者、胜负、回合数、用时 | JSONL |
| `evaluation.json` | 每个种子 vs 各 baseline 的胜率（含先手/后手分离） | JSON |
| TensorBoard | ELO 曲线、baseline 胜率曲线、多样性趋势、exploitability 趋势 | events |

---

## 10. 配置系统

### 10.1 默认配置文件

```yaml
# configs/ga_evo.yaml
system:
  device: "cuda"
  seed: 42

evolution:
  max_generations: 200
  population_size: 80
  num_seeds: 8
  elitism_count: 2
  convergence_window: 20           # 连续 N 代无提升
  convergence_threshold: 5.0       # ELO 提升 < 此值时认为收敛

crossover:
  method: "layer_wise"             # 锁定层级杂交
  rate: 0.8

mutation:
  method: "gaussian"               # 锁定全局高斯变异
  rate: 0.1
  scale: 0.03
  decay: 0.99

selection:
  method: "fixed_reference_elo"
  anchor_references:
    - "BasicRandomAI"
    - "BasicTowerAI"
    - "MediumRuleAI"
  dynamic_references: 2
  include_previous_champion: true
  elo_k_factor: 64
  elo_initial: 1200

opponent_pool:
  max_size: 20
  min_games_threshold: 8
  exploit_prob: 0.7
  decay: 0.99

baseline_battle:
  enabled: true
  interval: 1
  n_battles: 12

logging:
  log_level: "INFO"
  save_interval: 10
  tensorboard: true

env:
  player_id: 0
  backend_type: "python"
  max_steps: 512
  n_envs: 20
```

### 10.2 配置加载

- 支持 YAML 配置文件加载
- 支持 CLI 参数覆盖（如 `--population_size 100 --max_generations 300`）
- 支持类型验证和深层配置合并

---

## 11. 错误处理与恢复

### 11.1 断点续训

- 每 `save_interval` 代保存完整训练状态到 checkpoint 文件
- Checkpoint 包含：当前代次、所有个体基因组、ELO 评分、对手池状态、TrueSkill 评分矩阵
- 重启时指定 `--resume` 参数，自动从最新的 checkpoint 恢复

### 11.2 异常处理

- 个体的对战过程崩溃 → 该局判负，不影响其他个体对战
- 支持 `skip_broken_individuals` 配置项
- 对战超时（默认 300s） → 该局判平或根据基地 HP 判定
- 所有异常记录到 ga_training_error.log

---

## 12. 参考索引

以下是被借鉴的技术特征描述及其对应的来源代码位置索引。

| 序号 | 技术特征 | 本设计中的用途 | 来源位置索引 |
|------|---------|---------------|-------------|
| 1 | **AntWarPolicyValueNetwork**：独立 Policy/Value 编码器架构。Policy 使用 CNN+MLP 编码后拼接，经投影层到 hidden_dim，再输出动作 logits 和价值。Value 编码器结构相同但独立参数。 | 个体的策略网络，权重被提取为 Genome | `ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py` |
| 2 | **HexCNNEncoder**：三阶段六边形卷积编码器。每阶段包含 Conv2d + BN + ReLU + HexConv（六方向卷积 + einsum）+ 残差连接，最终通道压缩 + AdaptiveAvgPool2d 展平。 | 棋盘特征提取（编码器），是 Genome 参数的主要组成部分 | `ppo_v2/src/ppo_antwar/network/hex_cnn_encoder.py` |
| 3 | **StructuredActionHead**：分层动作头。先将 119 维动作空间分为 9 种类型（type_head: Linear(256→9)），再为每类分配独立的 target_head（9 个 Linear(256→n_i)）。 | 个体的动作输出头，参与基因组 | `ppo_v2/src/ppo_antwar/network/heads.py` |
| 4 | **OpponentPool**：对手池管理。支持 max_size 容量限制、淘汰保护（games < threshold 的对手享有保护）、状态 JSON 持久化。 | 管理跨代种子模型，作为动态参照物来源 | `ppo_v2/src/ppo_antwar/league/opponent_pool.py` |
| 5 | **BattleSharedPayoff**：TrueSkill 评分矩阵。使用 `trueskill.rate_1vs1` 更新双方评分，支持 `get_win_rate`（贝叶斯胜率收缩）、`get_exploitability`（相对强度评估）、`decay_all`（历史衰减）。 | 跨代种子模型的技能评分管理 | `ppo_v2/src/ppo_antwar/league/battle_shared_payoff.py` |
| 6 | **TrueSkillRating**：双值评分封装。初始 (mu=25.0, sigma=8.33)，提供 `is_confident`（sigma<3.0）、`confidence_interval`（mu±3sigma）属性。 | 技能评分的核心数据结构 | `ppo_v2/src/ppo_antwar/league/trueskill_rating.py` |
| 7 | **OpponentSelector**：探索/利用对手选择。利用模式选 sigma 最小的可信对手中 mu 最高的，探索模式选 sigma > 5.0 的对手。exploit_prob 支持 linear/sigmoid/exp 三种调度。 | 从对手池中选取动态参照物 | `ppo_v2/src/ppo_antwar/league/opponent_selector.py` |
| 8 | **BattleSimulator**：多进程对战引擎。使用 ProcessPoolExecutor（默认 12 workers）并行执行对战，每个 worker 运行完整游戏引擎，使用 cloudpickle 序列化 agent 传递。 | 执行个体 vs 参照物的每场对战 | `ppo_v2/src/ppo_antwar/battle/battle_simulator.py` |
| 9 | **BaselineBattleConfig**：对战评估配置。管理 max_rounds、n_battles、log_level、baseline_agents 等可调参数。 | 配置基线验证的行为 | `ppo_v2/src/ppo_antwar/battle/battle_config.py` |
| 10 | **LoggingSubsystem** 日志风格：loguru 主日志 + JSONL 逐行日志 + JSON 统计 + TensorBoard。目录按 run_id 隔离，不同日志类型分目录存放。 | 日志系统的结构和风格参考 | `ppo_v2/src/ppo_antwar/trainer/logging_subsystem.py` |
| 11 | **PPORLConfigParser**：YAML + CLI 覆盖 + 类型验证 + 深层合并的配置加载方案。 | 配置文件的加载和解析方式 | `ppo_v2/src/ppo_antwar/config/config_parser.py` |
