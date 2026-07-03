# ss_train.py 重构方案 — 完全用行为空间变异替代参数采样

## 当前实现的问题

ss_train.py 当前的种群生成方式：

```
mean → σε 采样 100 个体 → 评估 → 选 top-K → BC 训练(含 10% 噪声) → 新 mean
           ↑                                   ↑
       参数空间噪声                        行为空间噪声（混入）
```

两处噪声是**冗余且矛盾的**：
- 参数噪声 σε 在 553K 维权重空间做无差别扰动
- 标签噪声 p_mutate 在策略空间做局部探索
- 两者叠加的效果难以预测，BC 训练时标签变异可能被参数噪声抵消

## 目标架构

种群生成以行为空间变异为主，极小参数噪声保底：

```
mean → BC 变异 × 100 → 100 个变种 → +σ'·ε → 评估 → 选 top-K → BC 训练(clean) → 新 mean
       ↑                                         ↑
    行为空间变异                               σ'=0.0002
```

| 步骤 | 当前实现 | 目标实现 |
|:----|:--------|:--------|
| 生成种群 | mean + σε | BC 变异 × 100（+ 极小 σε 保底） |
| 评估 | 100 个体对多局 | 不变 |
| 更新 | BC 训练(含 10% 噪声标签) | BC 训练(纯精英数据) |
| 剩余参数 | sigma, p_mutate, temperature | sigma(极小), p_mutate, temperature |

### 关于 sigma

BC 变异是主要的多样性来源。但保留一个极小的 sigma（默认 0.02，原默认 0.2）作为**保底探索**：

```
BC 变异后:
  variant_params[i] = bc_mutate(mean, seed=i) + σ'·ε
                                                  ↑
                                              σ'=0.02
```

在行为空间变异生成的个体上叠加一个极小的参数噪声，相当于"行为层面的不同 + 参数层面的微扰"。BC 变异已经把个体分开了，tiny sigma 不会抵消 BE 变异的效果；但如果 BC 变异偶尔产生了两个近乎一样的个体，tiny sigma 至少能保证它们不完全重合。

最坏情况（σ'=0）就是纯行为空间变异，不会更差。

## 关键设计：轻量 BC 变异

100 个独立 BC 训练太慢。改为**只对最后的分类头做微调**：

```
完整 BC 训练:     3 epoch × 全参数更新 → mean           ~8s
轻量 BC 变异:     1 epoch × 只更新 policy_heads(i) → 变种  ~0.2s × 100 = 20s
```

每代的完整时间线：

```
1. 评估 100 个变种（和多局对手打比赛）            ← 5min（瓶颈，不变）
2. 写 .npz（所有个体）                           ← 1-2s
3. 选 top-K                                      ← 微秒级
4. BC 训练 mean（3 epoch，全参数）                ← 8s
5. BC 变异 × 100（1 epoch，只更新 policy_heads）  ← 20s
6. 清理 .npz                                     ← 微秒级
```

**变异微调**的实现：

### BC 变异的具体流程

核心思路：所有个体用**同一批精英数据**，但每个个体使用不同的随机种子 → 不同的标签变异 → 不同的梯度方向 → 不同的权重。

```
精英数据:   [(s1, a1), (s2, a2), ..., (sN, aN)]  ← 同一批

个体 0:  seed=0 → 变异掩码 [0,1,0,0,1,...] → 替换部分标签 → 梯度方向 A → 权重 w₀
个体 1:  seed=1 → 变异掩码 [0,0,1,0,0,...] → 替换部分标签 → 梯度方向 B → 权重 w₁
个体 2:  seed=2 → 变异掩码 [1,0,0,0,1,...] → 替换部分标签 → 梯度方向 C → 权重 w₂
...
个体 99: seed=99 → 变异掩码 [...] → 替换部分标签 → 梯度方向 → 权重 w₉₉

结果: 100 个不同权重的个体 ≈ 种群多样性
```

### 标签变异的具体实现

每个 batch 中，每个样本的每个头**独立**判断是否变异：

```python
def mutate_labels_for_individual(cls_label, head_logits, p_mutate, temperature, seed):
    """In-place class label mutation per individual.
    
    Args:
        cls_label:  (B, N_heads) — 精英原始 argmax 标签
        head_logits: (B, N_heads, 24) — 精英原始 logits（从 .npz 加载）
        p_mutate: 每个标签变异概率（默认 0.1）
        temperature: 温度采样参数（默认 3.0）
        seed: 个体编号，确保不同个体产生不同的变异掩码
    Returns:
        mutated cls_label (same tensor, modified in-place)
    """
    torch.manual_seed(seed)
    B, N = cls_label.shape
    for hi in range(N):
        # 每个头独立生成变异掩码
        mask = torch.rand(B) < p_mutate
        if not mask.any():
            continue
        # 用精英的 logits 做温度采样（保留偏好结构）
        logits = head_logits[mask, hi, :]  # (n_mutated, 24)
        probs = F.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        cls_label[mask, hi] = sampled
```

每个个体：

```python
for i in range(population_size):
    model_copy = deepcopy(mean_model)
    
    # 1 batch 数据
    batch = next(iter(loader))
    board = batch["board"].to(device)
    stats = batch["stats"].to(device)
    cls_label = batch["class_label"].to(device).clone()  # 精英原始标签
    head_logits = batch["head_logits"].to(device)         # 从 .npz 加载的精英 logits
    
    # 用个体编号 i 作为种子对标签变异
    mutate_labels_for_individual(cls_label, head_logits, 
                                  p_mutate=args.p_mutate,
                                  temperature=args.temperature,
                                  seed=i)
    
    # 冻结 encoder，只更新 policy_heads
    for name, param in model_copy.named_parameters():
        param.requires_grad = name.startswith("policy_heads")
    
    optimizer = Adam(model_copy.policy_heads.parameters(), lr=args.lr)
    optimizer.zero_grad()
    output = model_copy(board, stats)
    loss = CE(output["head_logits"], cls_label)  # 只算 class loss
    loss.backward()
    optimizer.step()
    
    # 还原 + tiny parameter noise
    params = model_copy.get_parameters_as_vector()
    noise = np.random.randn(*params.shape).astype(np.float32) * args.sigma
    population_params[i] = params + noise
```

注意：变异步骤同时更新 class 和 map。class 变异通过温度采样替换标签（改变"选什么动作"），map 变异通过在目标 action_map 上叠加高斯噪声（改变"位置偏好"）。两者都只更新最后一层：

```
class 变异: 更新 policy_heads     (~1.5K 参数, 占 0.3%)
map 变异:   更新 action_map_conv  (~25K 参数, 占 4.5%)
          合计 ~26K 参数（全模型 5%），benchmark 实测 12s/100 个体
```

```
# BC 变异时同时训练 class 和 map
for name, param in model_copy.named_parameters():
    param.requires_grad = name.startswith("policy_heads") or "action_map_conv" in name

optimizer = Adam([
    {"params": model_copy.policy_heads.parameters(), "lr": args.lr},
    {"params": model_copy.action_map_conv.parameters(), "lr": args.lr * 0.3},
])
```

map 的 lr 降低到 class 的 30%，因为 action_map 是共享的，变异幅度需要温和一些，避免破坏多次代际积累的位置知识。

每个个体的 map 变异：

```python
# 在 mutate_labels_for_individual 之后加入
pos_noise = torch.randn_like(action_map) * args.pos_noise_std  # 高斯噪声
action_map_mutated = action_map + pos_noise

# Loss = CE(class) + KL(mutated_map || noisy_map)
map_flat = output["action_map"].view(B, -1)
target_flat = action_map_mutated.view(B, -1)
map_loss = F.kl_div(F.log_softmax(map_flat, dim=-1),
                    F.softmax(target_flat.detach(), dim=-1),
                    reduction="batchmean")
loss = class_loss + lambda_map * map_loss
```

这样 100 个变种 ≈ 100 × 1 batch × 只更新分类头 = 约等于 2-3 epoch 的全参数训练。预估 20-30s，可接受。

## 日志输出

每代新增 BC 变异阶段的日志：

```
gen=0  best=1.0000  avg=0.9375  eval_s=6.9  total_s=13.3  ss=194.6286
  [BC] mutate 1/100  seed=0  cls=0.4231  map=12.3456
  [BC] mutate 2/100  seed=1  cls=0.5102  map=13.2045
  ...
  [BC] mutate 100/100  seed=99  cls=0.3987  map=11.8923
  [LB] challenging (pool=5)
  [LB] added
```

变异日志每 10 或 20 个个体输出一次（而不是每 1 个），避免日志刷屏：

```python
if (i + 1) % 20 == 0:
    log.print(key="bc_mut", value=f"{i+1}/{n} seed={i} cls={cls_loss:.4f}")
```

## 移出和保留的参数

| 参数 | 变化 | 理由 |
|:----|:----|:-----|
| `--sigma` | 默认 0.2 → **0.0002** | 极小参数噪声保底 |
| `--p-mutate` | 保留 | 变异强度控制 |
| `--temperature` | 保留 | 温度采样 |
| `--pos-noise-std` | 保留 | 位置变异 |
| `--epochs` | 保留 | BC 训练 epoch |
| `--lr` | 保留 | BC 学习率 |
| `--k` | 保留 | top-K 数 |

## 文件改动

`ss_train.py` 中改动点：

1. **修改** `sigma` 默认值 0.2 → 0.0002，mirrored sampling 保留但噪声极小
2. **新增** `bc_mutate()` 函数：对模型做轻量 BC 变异（1 epoch，只更新 policy_heads）
3. **修改** main 循环：
   - step 1: `bc_mutate(mean) × 100 → population`
   - step 2: 评估（不变）
   - step 3: BC 训练 mean（不变，纯精英数据，无噪声）
4. **移除** 数据收集中的标签变异逻辑（SSDataset 的 p_mutate 改为 0）

## 风险

1. **计算量**：100 个变种 × 1 epoch 的 forward + backward ≈ 20s。如果这个数字对 I/O 敏感可能更慢。最坏情况 60s，仍然可接受。
2. **变异不够多样**：只更新分类头可能产生不够多样化的种群。如果 100 个变种的策略相近，评估就失去了筛选意义。解决：必要时增加 `temperature` 或 `p_mutate`。
3. **BC 训练后的 mean 参数变化**：clean BC 训练完成后，mean 已被更新。复制 mean 的 state dict 再做变异，不干扰训练好的 mean。
