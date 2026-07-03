# bc_mutate_population 改造方案 — 梯度导向变异

## 当前实现的问题

当前每个个体的变异是独立执行多 batch 梯度下降：

```
for 每个个体 i:
  copy mean → model_i                 ← 深拷贝 561K 参数
  for 5 batches:
    forward(mutated_labels) → loss    ← 前向传播
    backward()                         ← 反向传播
    optimizer.step()                   ← 更新权重
  collect params_i                     ← 收集参数
```

**慢**：100 个体 × 5 batch × deepcopy = 单代约 45s

**个体间差异小**：每个个体都从同一个 mean 起点出发、在类似的标签上反复更新几次，最终收敛方向趋同。

## 梯度导向变异

核心思路：**不更新权重，只提取梯度方向作为扰动加到 mean 上**。

```
for 每个个体 i:
  取 1 batch 数据
  变异标签（temperature 采样，seed=i）
  model.zero_grad()
  forward(变异标签) → loss              ← 只算 class loss
  backward()                             ← 算梯度
  grad_vec = model.get_gradients_as_vector()  ← 提取梯度
  model.zero_grad()                      ← 清除梯度
  
  # 梯度是"指向变异标签的方向"
  # 归一化后作为扰动加到 mean 上
  grad_vec = normalize(grad_vec)
  params_i = mean + η · grad_vec + σ' · ε  ← η=步长, σ'=极小参数噪声
```

### 为什么有效

梯度 $\nabla_\theta \mathcal{L}(f_\theta(x), y_{mutated})$ 指向**"让模型输出变异标签"**的方向。和随机噪声对比：

| | 随机噪声 σε | 梯度 η·∇ |
|:---|:----------|:--------|
| 方向 | 561K 维球面上均匀分布 | 指向改变策略的方向 |
| 对于为的影响 | 大部分方向不改变输出 | 精准改变标签偏好 |
| 和 p_mutate 配合 | 无关 | 天然耦合（变异什么标签就朝哪个方向走） |

**不同个体有不同的 seed → 不同的变异标签 → 不同的梯度方向**。100 个梯度方向在参数空间中拉开 100 个不同的扰动方向，比 100 个随机方向更高效地覆盖策略空间。

### 和当前方法的对比

| | 当前（多 batch GD） | 梯度导向（新） |
|:---|:-----------------|:-------------|
| deepcopy | 100 次 | **0 次** |
| forward | 100 × 5 = 500 次 | **100 次** |
| backward | 500 次 | **100 次** |
| 预计耗时 | ~45s | **~2s** |
| 个体多样性 | 收敛到相似方向 | 每个个体独立梯度方向 |
| 正交性 | 和随机噪声无协同 | **参数 σ'·ε 叠加在已定向的面上** |

### 归一化和步长

梯度范数不稳定（CNN 的梯度可能很大或很小），必须归一化：

```python
grad_vec = model.get_gradients_as_vector()
grad_norm = np.linalg.norm(grad_vec) + 1e-8
grad_vec = grad_vec / grad_norm  # 单位方向

params_i = mean + mutation_step * grad_vec + sigma * noise
```

- `mutation_step`：梯度方向上的步长（新参数，默认 0.01，需要调试）
- `sigma`：叠加的随机噪声（默认 0.0002，不动）

归一化后的梯度方向保证每个个体都以**相同的影响力**扰动策略空间，个体差异完全由 seed 差异导致的不同变异标签决定。

### 新增方法：get_gradients_as_vector

```python
def get_gradients_as_vector(self) -> np.ndarray:
    """Flatten all parameter gradients into a single vector."""
    grads = []
    for p in self.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1).cpu().numpy())
        else:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(grads)
```

### bc_mutate_population 新实现

两阶段设计。Phase 1 预计算 N_DIRS 个梯度基方向（expensive），Phase 2 组合出种群（cheap）：

```
Phase 1：预计算梯度基方向（20 × FWD/BWD, ~0.4s）
  for j in range(N_DIRS):
    random batch → mutate_labels(seed=j) → FWD → BWD
    grad_basis[j] = normalize(grad)

Phase 2：生成 100 个个体（线性组合, ~0.01s）
  for i in range(pop_size):
    w = randn(N_DIRS) / sqrt(N_DIRS)    ← 权重期望范数 = 1
    delta = sum(w[j] · grad_basis[j])     ← 在梯度子空间中组合
    params_i = mean + step · delta + σ · ε
```

为什么权重缩放为 `1/sqrt(N_DIRS)`：

```python
期望的 ||delta||² = E[||Σ w_j·g_j||²]
  = Σ E[w_j²] + 2·Σ E[w_j w_k] · g_j·g_k
  = N_DIRS · (1/N_DIRS) + 0
  = 1
```

确保 delta 的期望范数为 1，不随 N_DIRS 增加而膨胀。

```python
def bc_mutate_population(mean_model, dataset, device, n_individuals,
                          p_mutate=0.1, temperature=3.0,
                          mutation_step=0.01, sigma=0.0002,
                          n_dirs=20, log=None):
    """Generate N individuals via linear combination of gradient directions.

    Phase 1: Pre-compute ``n_dirs`` gradient basis directions (each from a
    different batch × mutation seed). Phase 2: For each individual, sample
    random weights and linearly combine basis directions as the perturbation.
    """
    if len(dataset) == 0:
        return []

    N = mean_model.num_heads
    mean_params = mean_model.get_parameters_as_vector()
    model_template = create_model(num_heads=N)

    # Phase 1: Pre-compute gradient basis
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    all_batches = list(loader)
    grad_basis = []

    for j in range(n_dirs):
        batch = all_batches[j % len(all_batches)]
        board = batch["board"].to(device)
        stats = batch["stats"].to(device)
        cls_label_ref = batch["class_label"].to(device)
        head_logits = batch["head_logits"].to(device)

        model_template.load_state_dict(mean_model.state_dict())
        model_template.train()
        model_template.to(device)

        # Mutate labels
        cls_label = cls_label_ref.clone()
        torch.manual_seed(j)
        B, NH = cls_label.shape
        for hi in range(NH):
            mask = torch.rand(B) < p_mutate
            if not mask.any():
                continue
            logits_i = head_logits[mask, hi, :]
            logits_i = torch.clamp(logits_i, -50.0, 50.0)
            probs = F.softmax(logits_i / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            cls_label[mask, hi] = sampled

        # Forward + backward
        model_template.zero_grad()
        output = model_template(board, stats)
        cls_loss = 0.0
        for hi in range(NH):
            cls_loss += F.cross_entropy(output[f"head{hi+1}_logits"], cls_label[:, hi])
        cls_loss /= NH
        cls_loss.backward()

        # Extract + normalize gradient direction
        grad = _extract_grad_vector(model_template)
        grad_norm = np.linalg.norm(grad) + 1e-8
        grad_basis.append(grad / grad_norm)

        model_template.zero_grad()

    # Phase 2: Generate individuals
    params_list = []
    rng = np.random.RandomState()
    rng.seed(42)  # deterministic reproducibility

    for i in range(n_individuals):
        # Random weights: unit expected norm
        w = rng.randn(n_dirs).astype(np.float32) / np.sqrt(n_dirs)

        # Linear combination of gradient basis
        delta = np.zeros_like(mean_params, dtype=np.float32)
        for j in range(n_dirs):
            delta += w[j] * grad_basis[j]

        # Apply: mean + step · delta + σ · ε
        noise = rng.randn(*mean_params.shape).astype(np.float32) * sigma
        params_list.append(mean_params + mutation_step * delta + noise)

        if log and (i + 1) % max(1, n_individuals // 5) == 0:
            log.print(key="gmut", value=f"{i+1}/{n_individuals}")

    model_template.cpu()
    return params_list


def _extract_grad_vector(model):
    """Flatten all parameter gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1).cpu().numpy())
        else:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(grads)
```

### 参数变更

| 参数 | 变化 | 理由 |
|:---|:----|:-----|
| `--mutation-step` | **新增**，默认 0.01 | 梯度方向上的步长 |
| `--n-grad-dirs` | **新增**，默认 20 | 预计算的梯度基方向数 |
| `--sigma` | 不变，0.0002 | 叠加的随机噪声 |
| `--p-mutate` | 不变 | 控制变异 mask |
| `--temperature` | 不变 | 温度采样 |
| `--lr` | **不再用于变异** | 变异不需要 optimizer |

### 风险和注意事项

1. **梯度只针对 class loss**——不包含 map loss。梯度方向只反映"改变分类偏好"的信息，不包含位置信息。但如果 class 行为多样化了，位置也会跟着变化（不同动作落点不同）。

2. **单 batch 的梯度可能不稳定**——64 条数据的梯度对 561K 参数来说是稀疏的。但不同个体用不同 batch（循环取），天然实现了 batch 层面的多样性。

3. **归一化抹消了幅度差异**——所有个体的扰动幅度相同（mutation_step），区别只在于方向。这可能丢失一些有用信号（有的个体应该扰动大一些）。

4. **mutation_step 需要调试**——太大 → 个体偏离 mean 太远，可能退化。太小 → 个体差异不够。建议先 0.01 试跑，观察 ind_dist 分布。
