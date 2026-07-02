# 策略空间进化训练程序 — 设计文档

## 概要

将 Elite BC（监督学习替代 ES 梯度）与策略空间变异（标签噪声探索）统一为完整的策略空间进化算法。参数空间只保留采样操作，选择、学习、变异全部在行为空间进行。

新建 `code/my_ai/ss_train.py`，**完全独立于 `es_train.py` 和 `elite_bc.py`**，不复用后两者的代码。

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

`ss_train.py` **不引用** `es_train.py` 或 `elite_bc.py` 中的任何代码。`es_train.py` 和 `elite_bc.py` 保持不变。

## 文件结构

```
code/my_ai/
├── es_train.py        ← 不变
├── elite_bc.py        ← 不变
├── ss_train.py        ← 新增：策略空间进化，完全独立
└── network.py / agent.py / decoder.py  ← 不变（模型/智能体定义）
```

`ss_train.py` 只依赖 `network.py` / `agent.py` / `decoder.py`，**不依赖** `elite_bc.py` 的任何组件。虽然 BCDataset 和 supervised_update 的代码逻辑上有重叠，但为了独立性和稳定性，直接复制到 `ss_train.py` 中做定制修改，不共享。

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

1. mean → 采样 N 个体（mirrored sampling）
2. 所有个体打比赛 + 每局写一个 .npz（含精英原始 logits）
3. 按 fitness 排序 → 选 top-K
4. 读取 top-K 的 .npz 文件
5. 加载数据 + HOLD 降采样（p_hold=0.1）
6. 以 p_mutate 概率、temperature 采样替换 class label
7. GPU 上监督训练 mean 模型
8. 更新 mean，保存 checkpoint
9. 清理本代 .npz，进入下一代

（无 ES 梯度、无 velocity/momentum、无精英池、无 WinGraph）
```

### 3. 重要设计决策

#### 3a. 数据收集粒度

`_eval_worker` 每局一个进程。每局结束时写一个 `.npz` 文件，文件名格式：

```
gen_NNNN_ind{idx}_seed{seed}.npz
```

每个 `.npz` 包含该局全部回合的数据。和 `es_train.py` 模式一致。

#### 3b. .npz 存储字段

```
.npz 内容:
  board:        (T, 28, 19, 19)      float16
  stats:        (T, 42)               float16
  class_:       (T, N_heads)          int64     ← 每个头的 argmax class
  action_map:   (T, 23, 19, 19)      float16   ← 完整位置图（非 argmax）
  head_logits:  (T, N_heads, 24)      float16   ← 用于温度采样变异
```

位置标签使用完整的 action_map 而不是 argmax，原因：
- class 输出用于离散动作选择（24 选 1）
- map 输出是稠密的 23×19×19 位置偏好分布，argmax 后只剩峰值位置，损失了副峰和渐变信息
- 损失函数改为 KL 散度：`KL(predicted_action_map || elite_action_map)`

存储开销（每回合）：
- action_map: 23×19×19×2 = 16.6KB（float16），45K 样本/代 ≈ 747MB
- head_logits: 3×24×2 = 144B，45K 样本/代 ≈ 6.5MB
- 总代 ≈ 800MB（压缩后 ~100MB），读完即删，峰值 < 1GB

#### 3c. 对手选择

保留 `es_train.py` 的 `select_opponents` 接口。暂时简化成**在当前种群中随机选对手**（每个对手打两局，一先一后，保证公平性）。

后续需要优化时再引入 WinGraph 或精英池。

#### 3d. Checkpoint

和 `es_train.py` 保持一致：`--save-every N` 每隔 N 代保存一次 checkpoint，中断时保存 `interrupt_gen_*.pt`，完成后保存 `final.pt`。

checkpoint 内容：
- `mean` — 参数向量
- `model_state` — 模型 state dict
- `generation` — 当前代数
- `config` — 命令行参数

#### 3e. 和 `elite_bc.py` 的关系

`ss_train.py` **不引用** `elite_bc.py`。虽然 BCDataset 和 supervised_update 的逻辑可以复用，但为了独立性和避免耦合：

- BCDataset 的 HOLD 降采样逻辑 → 直接复制到 ss_train.py 的 `SSDataset`
- supervised_update 的训练逻辑（Adam + CE loss）→ 直接复制到 ss_train.py 的 `ss_supervised_update`
- 写 .npz 的 write_bc_npz → 直接复制到 ss_train.py

理由：`ss_train.py` 需要额外加载 `head_logits`、需要做标签变异、训练逻辑和 `elite_bc.py` 有差异。与其在 `elite_bc.py` 里加 if-else 兼容两个模式，不如直接复制一份做定制。虽然代码上有一份冗余，但保证了两个模块互不影响。

## 训练主循环伪代码

```python
def main():
    args = parse_args()
    model = create_model(num_heads=args.num_heads)
    mean = model.get_parameters_as_vector()
    pool = mp.Pool(args.workers)

    for gen in range(generations):
        # 1. Mirrored sampling
        noise = rng.randn(pop_size // 2, param_count)
        params_list = [mean + sigma * n for n in noise] + [mean - sigma * n for n in noise]

        # 2. Evaluate + collect .npz
        all_args = [(p, opp, seed, num_heads, bc_dir, gen, idx)
                    for idx, p in enumerate(params_list)
                    for opp, seed in zip(opponents, seeds)]
        results = pool.starmap_async(_eval_worker, all_args).get()
        fitness = average_fitness(results)

        # 3. Select top-K
        selected = sorted(range(pop_size), key=lambda i: -fitness[i])[:k]

        # 4. Load + HOLD filter + mutate + train
        npz_paths = collect_npz(bc_dir, selected, gen)
        dataset = SSDataset(npz_paths, p_hold=args.p_hold, p_mutate=args.p_mutate,
                              temperature=args.temperature, pos_noise_std=args.pos_noise_std)
        ss_supervised_update(model, dataset, device, epochs=args.epochs, lr=args.lr)

        # 5. Update mean + save
        mean = model.get_parameters_as_vector()
        if (gen + 1) % save_every == 0:
            save_checkpoint(out_dir, gen, mean, model)
        cleanup_gen_npz(bc_dir, gen)
```

## 命令行接口

```bash
python code/my_ai/ss_train.py \
    --pop-size 64 \
    --sigma 0.2 \
    --games 18 \
    --workers 24 \
    --generations 500 \
    --save-every 10 \
    --num-heads 3 \
    --k 5 \
    --epochs 3 \
    --lr 1e-3 \
    --batch-size 64 \
    --p-mutate 0.1 \
    --temperature 3.0 \
    --p-hold 0.1 \
    --pos-noise-std 0.1
```

## 参数设计方案

### 标签变异

在 SSDataset 的 `__getitem__` 中实现：

```python
class SSDataset(Dataset):
    def __init__(self, npz_paths, p_hold=0.1, p_mutate=0.1, temperature=3.0,
                 pos_noise_std=0.1):
        # 加载数据 + HOLD 降采样（同 BCDataset）
        # 额外加载 head_logits 和 action_map

    def __getitem__(self, idx):
        board = self.board[idx]
        stats = self.stats[idx]
        cls_label = self.class_label[idx].clone()
        action_map = self.action_map[idx]  # (23, 19, 19)

        # 策略空间变异：class（温度采样）+ 位置（高斯噪声）
        if torch.rand(1).item() < self.p_mutate:
            logits = self.head_logits[idx]  # (N_heads, 24)
            for hi in range(num_heads):
                probs = F.softmax(torch.from_numpy(logits[hi]) / self.temperature, dim=0)
                cls_label[hi] = torch.multinomial(probs, 1).item()

        if torch.rand(1).item() < self.p_mutate:
            noise = torch.randn_like(action_map) * self.pos_noise_std
            action_map = action_map + noise  # 在精英偏好上叠加噪声

        return {"board": board, "stats": stats,
                "class_label": cls_label, "action_map": action_map}
```

位置变异使用高斯噪声叠加在 `action_map` 上，不破坏原有的偏好结构——精英不喜欢的区域（极低值）加了噪声后仍然很低，精英偏好的峰值加了噪声后仍然是峰值，只是周边副峰被抬高了一些。

class 和位置变异使用**同一个 p_mutate 但独立触发**，每回合的效果分布：

```
p_mutate=0.1 时：
  81% 两者都不变异
   9% 只有 class 变异
   9% 只有位置变异
   1% 两者都变异
```

训练时的 map loss：

```python
# Map loss: KL 散度（完整分布）
pred_map = output["action_map"]  # (B, 23, 19, 19)
target_map = batch["action_map"].to(device)  # (B, 23, 19, 19)

pred_log = F.log_softmax(pred_map.view(B, -1), dim=1)
target_prob = F.softmax(target_map.view(B, -1), dim=1).detach()  # ← detach，不传播梯度到精英数据
map_loss = F.kl_div(pred_log, target_prob, reduction="batchmean")
```

### 位置变异

MVP 阶段也做位置变异（高斯噪声），但独立于 class 变异控制。

| 参数 | 默认值 | 说明 |
|:---|:------|:----|
| `pos_noise_std` | 0.1 | 位置噪声标准差，相对于 action_map 的典型值范围（~±5） |
| `p_mutate` | 0.1 | class 和位置共享同一个变异概率 |

`pos_noise_std=0.1` 相对于 action_map 的典型 logit 值（-3 ~ +5）是一个温和的扰动，不会把偏好分布彻底打乱。

## 实验计划

1. **基线**：p_mutate=0（纯 BC），跑 20 代
2. **策略变异**：p_mutate=0.1, temperature=3.0, 跑 20 代
3. **对比指标**：top-K fitness、HOLD 率、BUILD/DOWNGRADE 比例、头活跃度

## 风险和未解决的问题

1. **代码冗余**：ss_train.py 复制了 elite_bc.py 的部分代码。后续如果需要同步修改两处的相同逻辑，需要手动操作。
2. **变异积累**：多代后策略可能偏离精英太远，可能需要 p_mutate 退火。
3. **无精英保留**：没有显式保护，变异导致策略退化后恢复较慢。
4. **磁盘 I/O 瓶颈**：每代写入 ~800MB（压缩前），24 个 worker 并发写同一磁盘。若使用 HDD，写入可能成为瓶颈（smoke test 测得写 0.16MB 需 118ms，推算写入 800MB 需 ~90s 串行时间）。SSD 无问题。读完即删缓解了空间压力但不缓解写入压力。
