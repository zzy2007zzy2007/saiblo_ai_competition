# 策略空间进化训练程序 — 设计文档

## 概要

将 Elite BC（监督学习替代 ES 梯度）与策略空间变异（标签噪声探索）统一为完整的策略空间进化算法。参数空间只保留采样操作，选择、学习、变异全部在行为空间进行。

新建 `code/my_ai/ss_train.py`，复用 `elite_bc.py` 的组件，独立于现有的 `es_train.py`。

```
参数空间:  采样(θ + σε) + 评估              ← 唯一步骤
行为空间:  从 elite 数据监督学习 mean        ← 替代 ES 梯度
行为空间:  以 p_mutate 替换动作标签          ← 替代 ES 噪声
```

## 和现有方案的关系

| 方案 | 选择 | 学习/更新 | 变异/探索 |
|:----|:----|:---------|:---------|
| 传统 ES | rank(fitness) | 参数空间梯度 | 参数噪声 σ |
| Elite BC (`es_train.py --bc`) | top-K | BC 监督学习 | **无** |
| 策略空间进化 (`ss_train.py`) | top-K | BC 监督学习 | 标签突变 p_mutate |

`ss_train.py` 从 `es_train.py` 中提取 ES 采样/评估逻辑，去掉 ES 梯度更新，加入策略空间变异。`es_train.py` 保持不变。

## 文件结构

```
code/my_ai/
├── es_train.py        ← 不变，传统 ES / 已有 Elite BC 模式
├── elite_bc.py        ← 不变，被两个训练程序共享
├── ss_train.py        ← 新增：策略空间进化
└── network.py / agent.py / decoder.py  ← 不变
```

`ss_train.py` 依赖于：
- `elite_bc.py` — BCDataset, supervised_update, BCConfig, 数据 I/O
- `agent.py` — NeuralAgent（含 last_output）
- `network.py` — 模型定义
- `win_graph.py`（可选）— 对手选择

## 核心改动

### 1. 策略空间变异实现原理

在 BC 训练时，以概率 `p_mutate` 随机替换精英数据的动作标签。替换后的标签告诉模型："在这个局面下，大部分情况按精英的决策做，小部分情况试试别的动作"。

```
变异前:  cls_label = [5, 16, 5, 5, 16, 23, 5, ...]  ← 精英决策
变异后:  cls_label = [5, 16, 5, 9, 16, 23, 5, ...]  ← 第 3 个动作被替换
                                                         ↑ p_mutate
```

### 2. 训练循环

```
ss_train.py 每代流程:

1. mean → 采样 N 个体（mirrored sampling，同 ES）
2. 所有个体打比赛 + 全量写 .npz（含 elite 原始 logits）
3. 按 fitness 排序 → 选 top-K
4. 读取 top-K 的 .npz 文件
5. BCDataset 加载 + HOLD 降采样（p_hold=0.1）
6. 以 p_mutate 概率、temperature 采样替换 class label
7. GPU 上监督训练 mean 模型
8. 更新 mean，进入下一代

（无 ES 梯度、无 velocity/momentum、无精英池、无 WinGraph 依赖）
```

### 3. 变异方式

在 `.npz` 中额外保存精英的原始 logits，用温度采样做变异。不需要额外 forward。

```python
# 数据收集阶段（_eval_worker）
npz_data["head_logits"] = torch.stack([
    agent.last_output["head1_logits"],
    agent.last_output["head2_logits"],
    agent.last_output["head3_logits"],
], dim=1).numpy()  # (T, N_heads, 24)

# 训练阶段（supervised_update）
if p_mutate > 0:
    for hi in range(num_heads):
        mask = torch.rand(B) < p_mutate  # (B,)
        if mask.any():
            probs = F.softmax(logits[:, hi, :] / temperature, dim=-1)
            new_label = torch.multinomial(probs, 1).squeeze(-1)
            cls_label[mask, hi] = new_label[mask]
```

## 新建 `ss_train.py`

### 从 `es_train.py` 复用的部分

| 模块 | 复用方式 |
|:----|:--------|
| 模型创建 / 参数 flatten | 直接调用 `network.create_model()` / `model.get_parameters_as_vector()` |
| Mirrored sampling | 复制 `noise = rng.randn(n_noise, n_params)` + `mean ± sigma*noise` |
| `_eval_worker` | 复制并修改（去掉 `collect_data` 条件判断，全量写 .npz + logits） |
| 对手选择 | 简化版：随机从种群中选对手（去掉 WinGraph、精英池） |
| Checkpoint | 简化版：只保存 mean、model_state、generation、config |

### 从 `elite_bc.py` 直接使用的

| 组件 | 用途 |
|:----|:----|
| `TopKSelector` | 按 fitness 排序选 top-K |
| `BCDataset` | 加载精英数据 + HOLD 降采样 + 预计算变异 |
| `supervised_update` | GPU 监督训练（含变异步骤） |
| `write_bc_npz` / `collect_bc_data` / `cleanup_gen_npz` | 数据 I/O |
| `BCConfig` | 配置容器 |

### 和 `es_train.py` 的关键差异

| 方面 | `es_train.py` | `ss_train.py` |
|:----|:-------------|:--------------|
| 梯度更新 | ES 参数梯度（或 BC 跳过） | **无**，只用 BC + 变异 |
| Momentum / velocity | 有 | **无** |
| WinGraph | 有 | **无**（简化对手选择） |
| 精英池 | 有 | **无** |
| 精英保留 | 有 | **无**（变异自动产生多样性） |
| Synthetic test | 有 | **无** |
| Hot-reload config | 有 | **无** |
| Mirrored sampling | 有 | 有（复用） |
| .npz 数据收集 | `--bc` 时全量写 | **始终全量写** |
| 策略空间变异 | 无 | `p_mutate` 控制 |
| Genetic crossover | 无 | **无**（未来可加） |

### 命令行接口

```bash
python code/my_ai/ss_train.py \
    --pop-size 64 \
    --sigma 0.2 \
    --games 18 \
    --workers 24 \
    --generations 500 \
    --num-heads 3 \
    --k 5 \
    --epochs 3 \
    --lr 1e-3 \
    --batch-size 64 \
    --p-mutate 0.1 \
    --temperature 3.0 \
    --p-hold 0.1
```

### 训练主循环伪代码

```python
def main():
    args = parse_args()
    trainer = SSTrainer(args)
    pool = mp.Pool(args.workers)

    for gen in range(generations):
        # 1. Sample
        noise = rng.randn(pop_size // 2, param_count)
        params_list = [mean + sigma * n for n in noise] + [mean - sigma * n for n in noise]

        # 2. Evaluate + collect data
        all_args = build_eval_args(params_list, gen, bc_dir)
        results = pool.starmap_async(_eval_worker, all_args).get()
        fitness = compute_fitness(results)

        # 3. Select top-K
        selected = TopKSelector(k=args.k).select(fitness)

        # 4. Load BC data + mutate + train
        npz_paths = collect_bc_data(bc_dir, selected, gen)
        dataset = BCDataset(npz_paths, p_hold=args.p_hold, p_mutate=args.p_mutate, temperature=args.temperature)
        supervised_update(model, dataset, device=device, epochs=args.epochs, lr=args.lr)

        # 5. Update
        mean = model.get_parameters_as_vector()
        save_checkpoint(...)
        cleanup_gen_npz(bc_dir, gen)
```

## 参数设计方案

### 标签变异

```python
if random() < p_mutate:
    probs = softmax(saved_logits / temperature)
    new_label = categorical_sample(probs)
```

### 位置变异

MVP 阶段不做位置变异。理由：
1. 位置是 23×19×19 = 8303 类，变异空间太大
2. 位置选择更依赖具体局面，无约束变异容易产生非法位置
3. 只探索"做什么"不探索"在哪做"

## 实验计划

1. **基线**：p_mutate=0（纯 Elite BC），跑 20 代
2. **策略变异**：p_mutate=0.1，temperature=3.0，跑 20 代
3. **对比指标**：top-K fitness、策略多样性（HOLD 率、BUILD/DOWNGRADE 比例）、class loss、头活跃度

## 风险和未解决的问题

1. **p_mutate 和 temperature 的耦合**：p_mutate 控制"是否变异"，temperature 控制"变异到什么程度"。当前固定 p_mutate=0.1，调 temperature。
2. **保存 logits 的存储开销**：每回合多 3×24×4=288 字节，45K 样本/代 ≈ 13MB。可接受。
3. **变异积累**：多代变异后策略可能偏离精英太远。可能需要 p_mutate 退火。
4. **无精英保留**：相比 `es_train.py` 的精英保留机制，`ss_train.py` 没有显式的精英保护。如果变异导致策略退化，可能比 ES 更难恢复。
