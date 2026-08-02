# PPO 实验方案

## 目标

验证 PPO 能否在这类回合制策略游戏上学到有效策略。先用固定对手（ExampleAI）做实验，排除自对弈的不稳定因素。

## 核心设计决策

### 奖励函数

只用基地血量变化作为奖励，避免人工 shaping：

```
r_t = (hp_opp_{t-1} - hp_opp_t) - (hp_us_{t-1} - hp_us_t)
    = Δhp_opp - Δhp_us
```

- 对方基地扣血 = +1（正奖励）
- 己方基地扣血 = -1（负奖励）
- 平局或未发生 = 0

这个信号和最终胜负高度一致，几乎没有"钻空子"的空间。

### 动作空间

复用现有 3 头架构：
- 每头输出 24 类 logits + 19×19 位置图
- 训练时从 class logits 做**温度采样**（不是 argmax），保持探索
- 评估时用 argmax（和现有 eval 一致）

### 网络结构

现有 `AntWarNetwork` **已经包含 value head**（`self.value_head`），输出经过 Tanh 归一化到 [-1, 1]。可以直接复用。

PPO 需要的改动很少：
- forward 不变
- 训练时 policy head 输出 logits，value head 输出 V(s)
- 不需要额外的网络结构

## 数据流

PPO 是 on-policy 的——同一批数据不能用很多轮（ratio 会崩）。但"传数据"的方式无所谓，用 npz 还是直接 return 都可以。

### 复用现有 npz 格式

现有 `_eval_worker` 每局存 npz 的字段：

| 字段 | 已有 | PPO 需要？ | 说明 |
|------|------|-----------|------|
| `board` (28,19,19) | ✅ | ✅ 状态 s | — |
| `stats` (42,) | ✅ | ✅ 状态 s | — |
| `head_logits` (N_heads, 24) | ✅ | ✅ 算 logπ_old | rollout 时存的就是 π_old |
| `action_map` (24, 19, 19) | ✅ | ✅ 算 logπ 位置 | — |
| `value` (T, 2) 每帧血量 | ✅ | ✅ 可算 reward | `r_t = Δhp_opp - Δhp_us` |
| `class_` (N_heads,) argmax 标签 | ✅ | ❌ 需要**采样后的** class | 现有存 argmax，PPO 需要采样的 |
| V(s) value head 输出 | ❌ 缺 | ✅ 需要存 | GAE 需要 V(s) 序列 |

所以实际新增的字段只有两个：
1. **采样后的 class**（代替 argmax 标签）
2. **value head 输出** V(s)（标量，每帧一个）

位置目前用 argmax（原因见下文），所以位置不需要采样。

### Rollout 收集流程

```
每轮 PPO 更新:
  1. 用当前策略跑 N 局 vs ExampleAI（固定对手）
  2. 每局存 npz，比现有多两个字段：sampled_class, value_pred
  3. 加载 npz 计算 GAE → PPO 更新
  4. 数据用完即可丢弃（on-policy，下一轮用新数据）
```

存 npz 相比直接 return 的好处：
- 和现有框架一致（_eval_worker 已经在做同样的事）
- 方便调试：可以保留几轮的数据做对比分析
- 可以不删，用来做 BC 对照实验（虽然 PPO 不能复用）

### 和现有 _eval_worker 的改动

现有 `_eval_worker` 在存 npz 时稍微改一下：

```python
# 现有：存 argmax 标签
cls_labels.append(logits.argmax().item())

# PPO 版：存采样后的标签 + V(s)
sampled = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item()
cls_labels.append(sampled)          # ✅ 采样后的 class
value_preds.append(value.item())    # ✅ 新增 V(s)
reward = hp_opp_before - hp_opp + hp_us - hp_us_before
rewards.append(reward)              # ✅ 新增 reward
```

### 数据被"用完即弃"的原因

PPO 的损失函数中有 ratio = π_new(a|s) / π_old(a|s)。当策略更新了多轮后，π_old（收集数据时的策略）和 π_new 差距变大，ratio 要么爆炸要么归零，梯度不可靠。所以每轮用新数据更新后，旧数据就不能再用。

这和用什么格式存数据（npz / pickle / 直接 return）无关，是算法本身的约束。

## GAE（泛化优势估计）

对每个轨迹计算 Advantage：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)    # TD 误差
A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...  # 反向累积
```

参数初始推荐：
- γ = 0.99（折扣因子，512 回合足够覆盖远期）
- λ = 0.95（GAE 参数，介于 TD 和 Monte Carlo 之间）

## PPO 更新

### 损失函数

```
L_clip = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)   # 裁剪的策略梯度
L_value = MSE(V(s), R_t)                                  # 价值网络回归
L_entropy = -Σ π(a|s) · log π(a|s)                       # 熵奖励，鼓励探索

L_total = L_clip + c_v · L_value - c_e · L_entropy
```

### 对 3 头架构的处理

每个 head 独立算 logπ(a|s)：

```
logπ(action|s) = Σ_i logπ_i(class_i|s) + Σ_i logπ_i(x_i, y_i|s)
```

其中 logπ_i(class_i|s) = log_softmax(class_logits_i)[class_i]
位置部分：对 action_map 做 softmax，取 (x_i, y_i) 处的 log 概率

但由于位置图是 argmax 解码的（确定性），采样位置需要特殊处理：
- 方案 A：对 action_map 做 softmax 展开成 19×19 的分布，采样位置
- 方案 B：训练时位置用 argmax，只对 class 做采样
- 方案 C：class 和位置都采样

推荐方案 B——位置采样对策略影响太大（放错位置 = 完全不同的操作），初期先降低复杂度。class 做采样已经足够探索。

### 更新参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| ε (clip) | 0.2 | 标准值 |
| c_v | 0.5 | 价值损失权重 |
| c_e | 0.01 | 熵奖励权重（初始） |
| γ | 0.99 | 折扣因子 |
| λ (GAE) | 0.95 | GAE 参数 |
| lr | 3e-4 | 学习率（标准 PPO 值）|
| PPO epochs | 4 | 每批数据重复训练次数 |
| batch_size | 64 | mini-batch 大小 |

## 训练流程

```
初始化:
  加载 BC 预训练模型
  复制：policy_net = model（可训练）
        value_net 复用 model.value_head（可训练）
  
循环 (每轮):
  # 1. Rollout 收集
  trajectories = []
  for i in range(N_workers):
    并行: worker 跑 1 局 vs ExampleAI
    返回 [(s_1, a_1, logπ_1, V_1, r_1), ..., (s_T, ...)]
  
  # 2. GAE 计算
  for each trajectory:
    反向计算 A_t, R_t
  
  # 3. PPO 更新（多 epoch）
  for _ in range(K_epochs):
    for mini_batch in DataLoader(all_data):
      - 用当前 policy_net 算 logπ_new(a|s)
      - ratio = exp(logπ_new - logπ_old)
      - L_clip = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
      - L_value = MSE(V(s), R)
      - L_entropy = entropy(policy)
      - 反向传播
  
  # 4. 日志
  - 平均 reward / 胜率 / entropy / 价值损失
  
每 10 轮: 跑一次完整评估（多局 vs ExampleAI，argmax 解码）
每 50 轮: 保存 checkpoint
```

## 实验方案（分阶段）

### Phase 1：功能验证（1-2 天）

目标：确认 PPO 代码正确，能跑通

- 用随机对手做对照
- 看 PPO 是否能让胜率 > 随机基线
- 验证 GAE 计算正确（DOTA 测试）
- N = 20 局/轮，跑 50 轮

### Phase 2：固定对手学习（3-5 天）

目标：确认 PPO 能学会打败固定对手

- 对手 = ExampleAI（固定）
- 奖励 = 基地血量差
- N = 50-200 局/轮
- 跑 200-500 轮
- 学习曲线：胜率从基线 → 提升

### Phase 3：自对弈（可选）

目标：尝试 PPO 自对弈超越 ExampleAI

- 对手 = 历史策略（类似 GA 的精英池）
- 或对手 = 当前策略的快照
- 需要处理自对弈的稳定性问题

### Phase 4：PPO vs GA 对比（可选）

目标：确认哪种方法更适合这类游戏

- 相同 BC 预训练起点
- 相同计算量预算
- 对比最终胜率

## 需要新增/修改的文件

| 文件 | 改动 | 代码量 |
|------|------|--------|
| `code/my_ai/ppo_train.py` | **新增**。PPO 训练主脚本（GAE + 更新 + 主循环） | ~250 行 |
| `code/my_ai/_eval_worker.py` | **修改**。新增 PPO rolloute 模式（采样动作、存 V(s)、奖励） | ~50 行 |

### 完全复用的基础设施

- `network.py` — 不动。value head 已经存在
- `agent.py` / `decoder.py` — 不动。rollout 中直接使用
- `ss_train.py` / `ga_ss_train.py` — 不动。独立训练脚本
- `leaderboard.py` — 不动。暂时不需要（Phase 1 固定对手）

## 关键风险

1. **价值网络初始不准** — 虽然有 BC 预训练的价值头，但它是用最终胜负训练的，可能和 PPO 需要的 V(s) 分布不同。方案：前几轮用高学习率快速适应。
2. **采样效率** — 1 局 512 步，200 局/轮 = 10 万步/轮。PPO 通常需要 100 万步以上才能看到明显学习信号。可能前 10 轮（100 万步，约 1.5 天）看不到效果。
3. **自对弈升级** — 固定对手能学到上限（ExampleAI 的水平），要超越它必须过渡到自对弈。自对弈的奖励信号更吵（对手也在变），稳定性需要验证。
