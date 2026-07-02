# Elite Behavior Cloning — ES + 监督学习混合训练方案

## 背景

### 当前 ES 的问题

ES 直接在 553K 维参数空间上做梯度更新：

```
mean += lr * Σ rank(f) * ε
```

问题在于：

1. **参数空间脆弱**：好的策略（如 gen_5 的 BUILD/DOWNGRADE/闪电 精妙平衡）在参数空间中是一个极窄的峰。ES 噪声 ε 很容易一碰就碎，导致策略在 1-2 代内退化（gen_25→gen_26 的 BUILD/DOWNGRADE 循环在单次更新后坍缩为 86% HOLD）。

2. **信号稀释**：典型场景中 64/100 个体的 fitness 接近 0。ES 的 rank 梯度被大量无效个体稀释，信噪比极低。

3. **动量不足以解决问题**：momentum 累积的是参数空间的方向，无法处理"协调多个头的输出动作"这种结构性问题。

### 关键观察

- gen_5 的 BUILD/DOWNGRADE/闪电策略（3 个头的协调）是一个难得的优势策略，但在 ES 下无法稳定保持。
- 自对弈的适应度信号奖励"不犯错"多于"打得好"，导致 HOLD 策略在长期训练中胜出。
- 多个头之间的协调本质上是**决策层面的问题**，不是参数层面的问题。

## 核心思路

不是直接在参数空间做梯度更新，而是：

1. ES 采样的 100 个体照常打比赛、算 fitness
2. 选出 top-K 的（K=5），**从评估对战中**提取它们的 (state → action) 数据
3. 用这些数据做监督学习：`loss = CrossEntropy(mean_pred, top-K_action)`
4. 训练好的 mean 进入下一代

```
ES 传统方式:
  mean → 采样 100 个体 → 打比赛 → 加权平均参数 → 新 mean

Elite BC:
  mean → 采样 100 个体 → 打比赛（记录 top-K 的对战数据）
        └→ 提取 (state, action) → 监督训练 mean → 新 mean
```

### 关键优化：复用评估对战数据

`_eval_worker` 中每个个体已经打了 games 局比赛。当前只返回 fitness（胜率），但每回合的 board/stats/action 都在 `state.resolve_turn` 之前可用。改动量极小：

```
修改前（每代）:
  100 个体 × games 局 = 100 场比赛 → 返回 100 个 fitness

修改后（每代）:
  100 个体 × games 局 = 100 场比赛 → 返回 100 个 fitness
  + 其中 top-5 个体的对战数据（额外返回 boards × stats × actions 列表）
```

**额外时间开销 = 0**。数据已经在内存里了，只是多一个 return 字段。不需要让 top-K 额外打比赛。

## 期望优势

| 方面 | ES 传统 | Elite BC |
|------|---------|----------|
| 梯度源 | 62/100 零分个体的噪声 | top-5 个体的真实决策 |
| 优化空间 | 553K 参数向量 | 状态→动作映射 |
| 策略稳定性 | 窄峰，一碰就碎 | 决策边界相对平滑 |
| 头协调保持 | 参数级叠加，易冲突 | 行为级学习，天然协调 |
| 从失败中学习 | 不直接（零分个体贡献噪声） | 不直接（但至少不引入噪声） |

## 风险和挑战

1. **Distribution Shift**：top-K 个体打出的局面和 mean 个体打出的局面分布不同，监督学习可能学偏。这是最大的风险。

2. **监督 Loss ≠ 胜率**：最小化 CrossEntropy 不一定等于最大化胜率。可能学到"top-5 的平均行为"，而不是"比 top-5 更好的行为"。

3. **数据量**：复用评估对战数据后，每代约 45K 样本（games=18, top-5），对监督学习较充足。跨代累积可达数十万。

4. **速度**：复用评估对战数据后额外时间开销 = 0。worker 写 .npz 的 I/O 开销可以忽略（~几毫秒/局）。

5. **完整输出监督**：位置图是 23×19×19 = 8303 维输出，argmax 后为单标签，CrossEntropy 开销可控（约等于 8303 类分类）。

## 最小可行版本设计

### 数据收集

```
每代结束后，从评估阶段已打的对战中提取：
1. 按 fitness 选出 top-5 个体
2. 从它们已完成的 games 局中，提取所有回合的：
   - board(28,19,19) + stats(42)         → 输入特征
   - 每个头的动作类 argmax（24 类）      → 标签 1
   - 每个头的位置图 argmax（23×19×19）   → 标签 2
3. 产出的数据量 ≈ 5 × games × ~250 回合 × ~2 动作/回合
   例如 games=18 → 约 45,000 样本/代
```

**不额外打比赛**，仅从评估对战中多返回一个字段。

### 数据选择策略：可替换接口

当前方案是固定选 top-5，但未来可能需要不同的选择策略。抽象为 `EliteSelector` 接口：

```python
class EliteSelector(ABC):
    """从种群中选择哪些个体的对战数据用于监督学习。"""
    @abstractmethod
    def select(self, fitness: list[float]) -> list[int]:
        """返回被选中的个体 index 列表。"""
        ...

# 实现 1：固定 top-K
class TopKSelector(EliteSelector):
    def __init__(self, k: int = 5): self.k = k
    def select(self, fitness):
        return sorted(range(len(fitness)), key=lambda i: -fitness[i])[:self.k]

# 实现 2：概率抽样（按排名加权）
class RankWeightedSelector(EliteSelector):
    def __init__(self, k: int = 5, temperature: float = 1.0): ...
    def select(self, fitness):
        probs = softmax(fitness / temperature)
        return np.random.choice(len(fitness), size=k, replace=False, p=probs)

# 实现 3：阈值筛选
class ThresholdSelector(EliteSelector):
    def __init__(self, threshold: float = 0.6):
    def select(self, fitness):
        return [i for i, f in enumerate(fitness) if f >= threshold]
```

主训练循环只依赖 `EliteSelector` 接口，不关心具体实现：

```python
selector = TopKSelector(k=5)
for gen in range(generations):
    results = evaluate_population(...)         # 100 个体打比赛 + 写 .npz
    selected = selector.select(results.fitness)  # 选 index
    train_data = load_bc_data(selected)          # 读对应 .npz
    supervised_update(mean, train_data)          # 监督训练
```

### 数据存储：文件方案

**问题**：45K 样本/代 × (board 28×19×19 + stats 42 + labels) ≈ 3.6GB/代，全存内存 + IPC pickle 不可行。

**方案**：在训练历史目录下建 `bc_data/` 子文件夹，每个 worker 进程直接写 .npz 文件：

```
training_history/20260702_005921/
  ├── bc_data/
  │   ├── gen_0010_ind042_seed0.npz    # worker index=42, seed=0
  │   ├── gen_0010_ind042_seed1.npz    # worker index=42, seed=1
  │   ├── gen_0010_ind073_seed0.npz    # worker index=73, seed=0
  │   └── ...
  ├── gen_0010.pt
  └── train.log
```

**流程**：
1. 每个 worker 写入数据时用 `gen_NNNN_ind{index}_seed{seed}.npz` 命名，不知道自己排名
2. 主进程收齐 100 个 fitness 后排序，确定 top-K 的 index
3. 主进程根据 index 读取对应的 .npz 文件组装 DataLoader

每个 .npz 文件包含一局数据：`{ "board": ndarray, "stats": ndarray, "class": ndarray, "map": ndarray }`

主进程在监督训练阶段读取该目录下所有 .npz 组装成 DataLoader。

**优点**：
- 无 IPC pickle 瓶颈（worker 只写文件，不传数据）
- 数据跨代持久化，可滑动窗口累积多代
- 断点恢复时数据不丢失
- 可随时 inspect / debug 收集到的数据

### 训练流程

```
1. 用累积数据训练 mean 模型 1-2 个 epoch
2. Loss = λ_class × CrossEntropy(head_class, elite_class)
        + λ_map   × CrossEntropy(head_map, elite_map_argmax)
   其中 λ_class=1.0, λ_map=1.0（等权重）
3. 更新所有可训练参数（class heads + map head + encoder）
4. 训练后的 mean 进入下一代的 ES 采样
```

**GPU 注意事项**：
- ES 评估阶段在 CPU 上多进程并行（worker 内 `torch.set_num_threads(1)`）
- 监督学习在主进程运行，可独立使用 GPU：`.cuda()`
- .npz 数据加载后转为 tensor：`torch.from_numpy(arr).cuda()`
- CPU worker 和 GPU 训练互不干扰——评估结束后主进程才读数据做训练

使用完整的动作类 + 位置图标签的理由：
- **动作类只是"做什么"**，位置图是"在哪做"——两者共同构成完整策略
- 只训练 class head 而冻结位置头 → 位置头输出与 class 不匹配 → 策略变形
- gen_5 的 BUILD/DOWNGRADE 精妙平衡既依赖类选择也依赖位置选择
- 位置图是 23×19×19 = 8303 维输出，argmax 后为单标签，CrossEntropy 开销可控

### 何时停止

- 达到 `--generations` 上限

## 和其他方案的关系

| 方案 | 关系 |
|------|------|
| 减小 sigma | 能延缓坍缩但无法根治，BC 方案从行为层面解决 |
| 固定对手训练 | 可并行使用，BC 方案本身也建议用 ExampleAI 做数据收集对手 |
| 多进程提速 | BC 方案增加的数据收集成本需要多进程支持 |
| WinGraph | 对手选择优化和 BC 方案是正交的，可同时启用 |

## 未解决的问题

1. distribution shift 是否有实证？是否需要 DAgger 风格的 online 数据收集？
2. 跨代数据如何管理？滑动窗口还是全累积？
