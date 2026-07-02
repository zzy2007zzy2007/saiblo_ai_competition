# 策略空间变异 — 从行为空间进行探索

## 动机

### 参数空间变异的问题

传统 ES 在参数空间做变异：

```
θ' = θ + σ · ε      ε ~ N(0, I),  σ ≈ 0.05
```

在 553K 维参数空间中，这等价于"给管弦乐队的每个乐手微调 0.05 分贝"。问题：

1. **全局耦合**：任一权重的微小变化影响所有状态下所有头的所有决策。553K 个微小变化的累计效应完全不可控。
2. **窄峰脆弱**：好策略（如 gen_5 的 BUILD/DOWNGRADE 平衡）在参数空间中是极窄的山峰。σ=0.05 的噪声已经足以让策略在 1 代内崩坍（gen_25→gen_26 的 0%→86% HOLD 坍缩）。
3. **σ 的两难**：σ 再小 → 没有探索效果；σ 稍大 → 策略崩坍。

### 核心洞察

参数空间变异的所有问题都源于一个事实：**参数是全局耦合的**。一个权重影响所有决策。离散策略空间则天然是**局部独立的**。

## 从参数进化到策略进化

### 第一步：用监督学习替代参数梯度（Elite BC）

在实现策略空间变异之前，我们先做了一个更基础的改变：**用监督学习替代 ES 参数梯度更新**。

```
传统 ES:
  mean → 采样 N 个体 → 评估 → 参数空间梯度更新 → 新 mean

Elite BC (CEM-style):
  mean → 采样 N 个体 → 评估 → 选 top-K → 从精英数据监督学习 → 新 mean
```

这一步把优化从参数空间搬到了策略空间的门槛上：

| 维度 | ES 参数梯度 | Elite BC（监督学习） |
|------|------------|-------------------|
| 优化对象 | 553K 维参数向量 | 状态→动作映射 |
| 学习信号 | rank(fitness) × 噪声 | top-K 个体的决策标签 |
| 策略稳定性 | 窄峰，一碰就碎 | 决策边界相对平滑 |
| 头协调保持 | 参数级叠加，易冲突 | 行为级学习，天然协调 |

具体实现细节见 [elite_bc_implementation.md](elite_bc_implementation.md)。

### 第二步：在策略空间中做变异（本文主题）

Elite BC 解决了"用策略信号更新策略"的问题，但没有解决"策略空间中的探索"——BC 只是从 top-K 行为中做最大似然估计，不产生新行为。策略空间变异正是在这个基础上加入探索机制。

```
完整框架:

mean → 采样 N 个体 → 评估（全量写 .npz）→ 选 top-K
     → 读取精英数据 → 以 p_mutate 概率随机替换 action label → 监督训练 mean  → 下一代
                                           ↑
                                    策略空间变异
```

两步串联的效果：

```
参数空间:  采样(θ + σε) + 评估        ← 唯一保留的参数级操作
行为空间:  监督学习(top-K action)      ← 替代参数梯度
行为空间:  突变(action label)          ← 替代参数噪声
```

参数只用来生成初始的变异个体，之后的所有操作（选择、学习、变异）都在行为空间中进行。

## 策略空间变异

### 思路

不修改参数，而是修改监督学习的数据标签（elite 个体的动作选择），用带有受控噪声的标签训练模型。

```
传统 BC:
  label = argmax elite_head_logits
  loss = CrossEntropy(mean_output, label)

策略变异 BC:
  label' = mutate(label)    ← 以概率 p_mutate 替换某个头的动作
  loss = CrossEntropy(mean_output, label')
```

### 关键性质：局部独立

改变单个回合、单个头的动作选择，只影响"在特定局面下选什么"。其他所有局面下的策略完全不受影响。

```
参数变异:
  改权值 w_{i,j} → 所有 forward 路径 → 所有局面下的输出都变了
  效果: 全局不可控

策略变异:
  改动作 class 5→16 → 只有这一个回合的这一个头受影响
  效果: 局部独立，副作用为零
```

### 噪声特性

参数空间 σ 必须小（~0.05）是因为全局耦合放大了噪声效果。策略空间没有耦合，所以可以用完全不同的噪声策略：

| 维度 | 参数变异 | 策略变异 |
|------|---------|---------|
| 噪声强度 | σ=0.05（必须小） | p_mutate=0.1~0.3（可以大） |
| 噪声稀疏度 | 全维度修改 | 稀疏（只改少数 label） |
| 每步影响 | 所有决策漂移 | 少数决策突变 |
| 副作用 | 精妙平衡崩坍 | 无副作用 |

**"稀疏但强"的噪声**：不是所有 label 都加微弱噪声，而是少数 label 做完全替换。效果相当于"在 90% 的回合保持精英策略，10% 的回合试试别的动作"。

### 实现方式

#### 方式 A：硬替换（均匀随机）

```python
if random() < p_mutate:
    label = random.choice(all_classes)  # 包括 HOLD
```

简单直接，探索范围最大。风险是可能产生大量无意义动作（如 HOLD）。

#### 方式 B：温度采样（推荐）

```python
if random() < p_mutate:
    # 用精英头的 logits 做 softmax 采样（高温度）
    probs = softmax(elite_logits / temperature)
    label = categorical_sample(probs)
```

温度 T=1 时和精英一致，T=3 时接近均匀。这种方式保留了精英的偏好结构——第二偏好的动作比完全随机的动作更有信息量。

#### 方式 C：只替换 class，保留 map

保留 position argmax 不变，只替换 class label。这样变异的是"做什么"而不是"在哪做"，减少不必要的探索维度。

### 和传统变异的关系

```
参数变异:
  θ + noise → 影响 forward → 改变 (state → action) 映射
  间接、全局、不可控

策略变异:
  直接改变 (state → action) 映射中的少数样本点
  直接、局部、可控
```

策略变异等价于在**行为流形**上做探索，而不是在参数流形上。参数空间中的任何有意义的行为改变，都可以映射为一组 (state → action) 对的修改。策略变异直接做这种修改，跳过了中间的参数变换步骤。

## 风险

1. **精英数据本身已经很少**：每代 top-5 个体约数千个回合。再在其中随机替换一部分，有效数据量进一步减少。这个风险不大——因为 HOLD 降采样后数据量本来就少，但足够训练（实验验证过）。

2. **变异过强导致策略退化**：如果 p_mutate 太大（>0.5），BC 几乎是在拟合随机数据，精英信号被淹没。建议从 p_mutate=0.1 开始。

3. **可能产生非法动作**：无约束的随机替换可能产生精英从没做过的动作。不过 CrossEntropy 会让模型学习到"在某些局面下可以选这个动作"——即使精英没选过，只要不是完全非法，这就是探索的价值。

## 实验建议

1. 先用 p_mutate=0 跑几代确认 BC 正常工作
2. 逐步提高 p_mutate: 0.05 → 0.1 → 0.2
3. 对比指标：top-K fitness、class loss、HOLD 率变化
4. 如果策略多样化明显提升，说明策略空间变异起了作用

## 代码改动

极小。在 BCDataset 或 supervised_update 中加几行：

```python
# 在 supervised_update 的 DataLoader 迭代中
if p_mutate > 0 and self.training:
    mask = torch.rand(len(cls_label), device=device) < p_mutate  # (B, N_heads)
    if mask.any():
        # 替换被选中的标签为温度采样或随机
        noise = torch.randint(0, NUM_CLASSES, cls_label.shape, device=device)
        cls_label[mask] = noise[mask]
```

## 相关文献

### 参数空间 vs 动作空间噪声

**Parameter Space Noise for Exploration** (Plappert et al., OpenAI 2018)
- 系统比较了 RL 中参数噪声和动作噪声的探索效果
- 结论：参数噪声产生更一致、更丰富的探索行为
- 区别：该文做的是 RL（连续动作，π(a|s) 执行时加噪声），和我们的场景（ES+BC，训练时改离散标签）不同
- 链接：`Parameter Space Noise for Exploration` (arXiv)

### 行为克隆中的数据增强

**BC with Data Augmentation** — 机器人行为克隆的综述类工作
- 提到 action noise injection 作为数据增强技术，用于缓解分布偏移
- 方向：添加噪声使策略更鲁棒，而非用于探索
- 链接：fr.roboticscenter.ai/glossary/behavior-cloning

### 反事实行为克隆

**Counterfactual Behavior Cloning** (Sagheb & Losey, 2025)
- 对 imperfect human demo 的 action 做 counterfactual 扩充
- 共同点：修改 BC 标签来改善策略
- 不同点：他们用于降噪（从有噪声的人体演示中恢复真实意图），我们用于变异（在精英数据中引入受控噪声以探索）
- 链接：arXiv 2505.10760

### 交叉熵方法（CEM）

CEM 是和我们做法最接近的形式化框架：
- 标准形式：elite episode → (s, a) pairs → 监督学习策略
- 扩展方向之一：在 elite 策略上做"动作扰动"增加多样性
- CEM 的扰动通常通过 softmax 温度采样实现，而非显式的 label replacement

### ES 全参微调

**Evolution Strategies at Scale: LLM Fine-Tuning Beyond RL** (arXiv 2509.24372)
- 将 ES 扩展到十亿级 LLM 全参微调
- 核心思想：ES 从参数空间（而非动作空间）探索
- 验证了 ES 在长视野、仅结果可观测场景中的优势
- 链接：arXiv 2509.24372

### 总结

"策略空间变异"没有标准的学术名称——它是**离散动作 + ES + BC** 三个方向的交叉领域，每个方向本身不冷门但三者的交集很小。最相关的现有工作是 CEM + 动作扰动，但它通常用温度采样而不是稀疏强替换。我们的方案（稀疏但强的 label 替换）在这个特定问题设置下是合理的，但没有现成论文直接支持或反驳——更多是一个工程直觉。
