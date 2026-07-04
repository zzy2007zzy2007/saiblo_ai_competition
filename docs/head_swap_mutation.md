# 头交换变异方案

## 动机

梯度变异只能微调权重，产生不了梯度空间以外的组合。头交换（Head Swap）直接在离散的权重空间重组现有策略——把不同个体的 head 拆开再组合，零计算量。

## 设计方案

### 整体流程

```
种群 = 梯度变异个体(N_grad) + 头交换个体(N_swap) + 精英保留(N_elite)
      N_grad + N_swap + N_elite = pop_size
```

`N_swap = int(pop_size * swap_ratio)`（受 `--swap-ratio` 控制），剩余名额由梯度变异填补。

精英保留总是替换末尾，不受影响。

### 头交换的具体实现

从梯度变异生成的个体中随机抽选，对每个选中的个体，随机选两个头互换参数：

```
for i in range(N_swap):
    idx = randint(N_grad)     # 从梯度变异个体中随机选
    ha, hb = randint(N_heads), randint(N_heads)  # 随机两个不同的头
    
    # 互换这两个头的 weight 和 bias
    swap(params_list[idx].policy_heads[ha], params_list[idx].policy_heads[hb])
    
    # 被修改的个体作为新个体加入种群
    swap_child = params_list[idx].copy()
    params_list.append(swap_child)
```

每次互换产生 1 个新个体，零计算量。

### `--swap-ratio` 参数

控制头交换个体数占种群的比例，默认 0.3（pop=48 → 14 个头交换个体）。

```
默认: pop_size=48, swap_ratio=0.3
  → N_swap = 14, N_grad = 48 - 14 - N_elite ≈ 31
  → 31 个梯度变异 + 14 个头交换 + 3 个精英保留 = 48
```

### `--swap-p` 参数

控制每个梯度变异个体被选中做头交换的概率。默认 0.5（50% 的梯度个体会被做一次头互换）。

```
swap_p=0.3:
  梯度个体 A: H1↔H3　　→ 产生头交换体 A'
  梯度个体 B: 没选中
  梯度个体 C: H0↔H5　　→ 产生头交换体 C'
  ...

  最终个体池: [A, B, C, ..., A', C', ...]
  原始梯度个体保留 + 头交换个体作为新增
```

注意：**原始梯度个体保留不变**，头交换产生的是新的个体副本追加到池中。这样梯度变异和头交换不互相取代，而是共同存在。

`swap_p=0.3` 时，从 48 个梯度个体中选约 14 个做头交换，产生 14 个头交换体，池子变成 48 + 14 - N_elite(3) = 59... 不对，需要调整——应该是先计算 N_swap，然后从梯度池中随机抽 N_swap 个个体做头交换，结果替换掉末尾的 N_swap 个梯度个体，使总池大小保持 pop_size。

```
n_swap = int(pop_size * swap_ratio)
# 生成 N_grad = pop_size 个梯度个体
# 从 N_grad 中随机抽 n_swap 个做头交换
# 头交换体替换末尾 n_swap 个梯度个体
# 总池大小 = pop_size
```

### 实现步骤

1. 在 `bc_mutate_population` 返回值的基础上，在 `main()` 中执行头交换
2. 添加 `--swap-ratio` 和 `--swap-p-mutate` CLI 参数
3. 在生成种群后、精英保留前插入头交换
4. `swap_ratio` 和 `swap_p_mutate` 在日志中打印配置

### 代码变化位置

```python
# main() 中，BC 变异之后、精英保留之前：

if args.swap_ratio > 0 and len(params_list) >= 2:
    n_swap = max(1, int(args.pop_size * args.swap_ratio))
    n_grad = len(params_list)
    swap_childs = []
    
    for i in range(n_swap):
        # 从梯度变异个体中随机选两个 parent
        a = rng.randint(n_grad)
        b = rng.randint(n_grad)
        child = params_list[a].copy()  # numpy 向量
        
        # 逐个头决定是否交换
        for hi in range(num_heads):
            if torch.rand(1).item() < args.swap_p_mutate:
                # 交换 head hi 的权重
                swap_head_params(child, params_list[b], hi, model)
        
        swap_childs.append(child)
    
    # 用头交换个体替换梯度变异的末尾
    params_list = params_list[:-n_swap] + swap_childs
```

需要工具函数 `swap_head_params(vec_a, vec_b, head_idx, model)` —— 通过 `model.named_parameters()` 定位 head 参数在参数向量中的偏移量，然后交换对应位置的数据。

### 风险

- **头交换不一定是好策略**——H1 偏好防守、H2 偏好进攻，互换后可能互不协调。但种群评估会自然淘汰这种组合
- **需要实现参数向量的 head 定位**——`get_parameters_as_vector()` 是扁平的，需要按参数名定位每个 head 的偏移量，稍微复杂
