# Clue 08: GAE Bootstrap Value 在截断时被忽略，引入系统性偏差

## 结论：**确认** — 存在两个 Bug

1. **Bug A（严重）**：截断 episode 的 bootstrap value (`final_value`) 被 `dones[-1]=True` 覆盖，导致 GAE 将截断视为自然终止，`next_value` 被错误设为 0 而非 `V(s_{T+1})`
2. **Bug B（中等）**：多 episode 合并 batch 中，`ep_idx` 映射存在 off-by-one 错误，导致 bootstrap value 被应用到错误的 episode

原始问题描述的"value head 估值不准引入偏差"是 PPO 的已知权衡，但实际代码中的问题远比这严重——bootstrap value **完全未被使用**。

---

## 1. 截断时 final_value 的计算方式

### 1.1 截断检测（episode_collector.py 第 185-200 行）

```python
# episode_collector.py 第 185-200 行
if episode_length >= self._max_steps:
    done = True

# ...循环结束后...

# 检测截断：达到 max_steps 但 env 未返回 terminated/truncated
was_truncated = (episode_length >= self._max_steps) and not (terminated or truncated)
if was_truncated:
    batch.final_value = self._self_agent.get_value(obs_self)   # ← 使用当前策略 value head
    # 修正末步 done=True，确保 GAE 在 episode 边界正确重置
    batch.dones[-1] = True                                      # ← BUG A 的根源
else:
    batch.final_value = 0.0
batch.final_values = [batch.final_value]
```

**关键逻辑**：
- 截断条件：`episode_length >= max_steps` 且环境未返回 `terminated/truncated`
- 截断时：调用 `self._self_agent.get_value(obs_self)` 获取当前策略对截断后状态的估值
- **同时**：将 `batch.dones[-1]` 强制设为 `True`

### 1.2 get_value 的实现（neural_agent.py 第 46-55 行）

```python
# neural_agent.py 第 46-55 行
def get_value(self, observation: Dict[str, np.ndarray]) -> float:
    """获取状态价值（仅运行 value encoder，跳过 policy 前向）。"""
    from ..utils.obs_utils import observation_to_tensors

    tensors = observation_to_tensors(observation, self._device)
    with torch.no_grad():
        value = self._policy.get_value_only(
            tensors["board"], tensors["global"]
        )
    return value.item()
```

```python
# ant_war_policy_value_network.py 第 121-124 行
@torch.no_grad()
def get_value_only(self, board, global_vec):
    """仅计算状态价值，跳过策略编码"""
    return self.value_head(self._encode_value(board, global_vec))
```

**确认**：`final_value` 来自当前策略的 value head，这是 PPO 的标准做法。

---

## 2. GAE 中如何使用 final_value

### 2.1 GAE 计算入口（ppo_trainer.py 第 116-122 行）

```python
# ppo_trainer.py 第 116-122 行
advantages, returns = self._compute_gae(
    tensors["rewards"].tolist(),
    tensors["values"].tolist(),
    tensors["dones"].tolist(),
    final_value=batch.final_value,
    final_values=batch.final_values if batch.final_values else None,
)
```

### 2.2 GAE 核心计算（gae.py 第 46-82 行）

```python
# gae.py 第 46-82 行
# 构建 episode 边界到 final_value 的映射
if final_values is not None and len(final_values) > 0:
    episode_final_values = list(final_values)
else:
    # 兼容旧接口...
    ...

# 从最后一个 episode 开始倒推匹配 final_value
ep_idx = len(episode_final_values) - 1          # ← BUG B: 应为 len-2

for t in reversed(range(T)):
    if t == T - 1:
        # 末步：自然终止时 next_value=0；截断时使用最后一个 episode 的 final_value
        next_value = 0.0 if dones_t[t] > 0.5 else final_value   # ← BUG A: 截断时 dones[-1]=True，走 0.0 分支
    elif dones_t[t] > 0.5:
        # episode 边界：使用该 episode 的 bootstrap value
        if 0 <= ep_idx < len(episode_final_values):
            next_value = episode_final_values[ep_idx]            # ← BUG B: ep_idx 映射错位
        else:
            next_value = 0.0
        ep_idx -= 1
    else:
        next_value = values_t[t + 1]

    delta = rewards_t[t] + gamma * next_value - values_t[t]
    gae = delta + gamma * gae_lambda * (1.0 - dones_t[t]) * gae
    advantages[t] = gae
```

---

## 3. Bug A 详细分析：dones[-1]=True 导致 bootstrap value 被忽略

### 3.1 问题描述

截断 episode 设置 `batch.dones[-1] = True`（episode_collector.py 第 196 行），目的是让 GAE 在 episode 边界正确重置传播（`gae = delta + gamma * lambda * (1 - done) * gae`）。

但这同时影响了 `next_value` 的计算：GAE 在 `t == T-1` 时，判断 `dones_t[t] > 0.5` 为 True，选择 `next_value = 0.0`，**完全忽略** `final_value`。

### 3.2 代码注释与实际行为矛盾

```python
# gae.py 第 68-69 行的注释：
# 末步：自然终止时 next_value=0；截断时使用最后一个 episode 的 final_value
next_value = 0.0 if dones_t[t] > 0.5 else final_value
```

注释说"截断时使用 final_value"，但代码逻辑恰恰相反——截断时 `dones[-1]=True`，走 `0.0` 分支。**注释与代码行为矛盾**。

### 3.3 单 episode 场景推演

假设一个截断 episode，10 步，`dones = [F,F,...,F,T]`，`final_value = V(s_{11}) = 5.0`：

| 步骤 t | dones[t] | 代码计算的 next_value | 正确的 next_value |
|--------|----------|----------------------|-------------------|
| 9 (T-1) | True | 0.0 | 5.0 (V(s_{11})) |
| 8 | False | values[9] | values[9] |
| ... | ... | ... | ... |

**影响**：最后一步的 TD error 为 `delta_9 = r_9 + gamma * 0 - V(s_9)`，正确应为 `delta_9 = r_9 + gamma * 5.0 - V(s_9)`。若 `gamma=0.99`，偏差为 `0.99 * 5.0 = 4.95`，通过 GAE 向前传播。

### 3.4 影响范围

- **截断 episode 的所有步**的 advantage 都受影响，因为 GAE 是反向传播的
- 偏差方向：`next_value` 被低估（0 vs V(s_{T+1})），导致 advantage 偏负
- 偏差幅度：`gamma^k * V(s_{T+1})`，随距截断点的距离 k 衰减

---

## 4. Bug B 详细分析：多 episode 合并时 ep_idx 映射错位

### 4.1 问题描述

`ep_idx` 初始化为 `len(episode_final_values) - 1`，但 `t == T-1` 时未递减 `ep_idx`。由于最后一个 episode 的 bootstrap value 已通过标量 `final_value` 处理，后续遇到的 episode 边界应使用 `episode_final_values[len-2]`，但实际使用 `episode_final_values[len-1]`，导致 bootstrap value 错位一个 episode。

### 4.2 多 episode 场景推演

假设合并 3 个 episode：
- Episode 1: 步 0-4，自然终止，`final_values[0] = 0.0`
- Episode 2: 步 5-9，**截断**，`final_values[1] = V(obs) = 5.0`
- Episode 3: 步 10-14，自然终止，`final_values[2] = 0.0`

`episode_final_values = [0.0, 5.0, 0.0]`，`ep_idx = 2`

| 步骤 t | dones[t] | 代码使用的 next_value | 正确的 next_value | 偏差 |
|--------|----------|----------------------|-------------------|------|
| 14 (T-1) | True | 0.0 (标量 final_value) | 0.0 | 无 |
| 9 | True | `episode_final_values[2]` = **0.0** | `episode_final_values[1]` = **5.0** | **-5.0** |
| 4 | True | `episode_final_values[1]` = **5.0** | `episode_final_values[0]` = **0.0** | **+5.0** |

**结果**：
- Episode 2（截断）的 bootstrap value 被忽略，使用了 Episode 3 的 0.0
- Episode 1（自然终止）错误地使用了 Episode 2 的 5.0 作为 bootstrap value

### 4.3 影响范围

- 仅在多 episode 合并 batch 时触发
- 每个 episode 的 bootstrap value 被错位到下一个 episode
- 截断 episode 的 bootstrap value 被忽略，自然终止 episode 被错误地赋予非零 bootstrap

---

## 5. 截断 episode 的比例和影响

### 5.1 截断频率

- `max_steps_per_episode = 512`（默认配置，ppo_antwar.yaml 第 65 行）
- 根据已有调查（clue_01、clue_02），典型 episode 长度为 150-300 步
- 截断发生在 episode 达到 512 步但环境未返回 terminated/truncated 时
- **截断比例较低**（估计 < 5%），但并非零

### 5.2 影响程度

虽然截断比例低，但影响是系统性的：

1. **Bug A**：截断 episode 的所有 advantage 偏负，等效于告诉策略"这些状态比实际更差"，可能导致策略回避接近截断边界的状态
2. **Bug B**：在多 episode batch 中，即使只有一个截断 episode，其 bootstrap value 也会被错位到相邻 episode，污染正常 episode 的 advantage 计算

---

## 6. final_value 不准对 GAE advantages 的传播影响

### 6.1 原始问题的分析

原始问题关注"value head 估值不准引入偏差"。这是 PPO 的已知权衡：

- **标准做法**：使用当前策略的 value head 做 bootstrap 是 PPO/SAC 等算法的标配
- **替代方案**：不使用 bootstrap（`final_value=0`）会引入更大偏差，因为截断不是终止
- **Value head 不准的影响**：如果 V(s) 高估，advantage 偏正；如果低估，advantage 偏负。但这是自校正的——value loss 会逐步修正 V(s)

### 6.2 实际代码中的问题更严重

代码中的问题不是"value head 不准"，而是 **value head 的估值完全未被使用**：

- Bug A 导致截断 episode 的 `final_value` 被 `dones[-1]=True` 覆盖为 0
- Bug B 导致多 episode batch 中 bootstrap value 错位

这比"value head 不准"严重得多——不是估值有偏差，而是估值被丢弃。

---

## 7. 修复建议

### 7.1 Bug A 修复：分离 done 标记与 bootstrap value 的逻辑

**方案**：在 `episode_collector.py` 中引入独立的 `truncated` 标记，不依赖 `dones` 来判断是否需要 bootstrap。

```python
# episode_collector.py 修改
if was_truncated:
    batch.final_value = self._self_agent.get_value(obs_self)
    batch.dones[-1] = True  # 保留：GAE 传播重置仍需要
    batch.truncated = True  # 新增：标记截断，供 GAE 区分
else:
    batch.final_value = 0.0
    batch.truncated = False
```

```python
# gae.py 修改：t == T-1 分支
if t == T - 1:
    if dones_t[t] > 0.5:
        # episode 边界：使用该 episode 的 bootstrap value
        if 0 <= ep_idx < len(episode_final_values):
            next_value = episode_final_values[ep_idx]
            ep_idx -= 1
        else:
            next_value = 0.0
    else:
        # 非边界：使用标量 final_value
        next_value = final_value
```

**核心思路**：`t == T-1` 也走 `episode_final_values` 查表逻辑，不再用 `dones` 判断 `next_value`。`dones` 仅用于 GAE 传播重置 `(1 - done) * gae`。

### 7.2 Bug B 修复：修正 ep_idx 初始值

**方案 A**：将 `ep_idx` 初始化为 `len(episode_final_values) - 1`，并在 `t == T-1` 时也递减：

```python
ep_idx = len(episode_final_values) - 1

for t in reversed(range(T)):
    if t == T - 1:
        if dones_t[t] > 0.5:
            next_value = episode_final_values[ep_idx] if 0 <= ep_idx < len(episode_final_values) else 0.0
            ep_idx -= 1
        else:
            next_value = final_value
    elif dones_t[t] > 0.5:
        next_value = episode_final_values[ep_idx] if 0 <= ep_idx < len(episode_final_values) else 0.0
        ep_idx -= 1
    else:
        next_value = values_t[t + 1]
```

**方案 B**（更简洁）：统一 `t == T-1` 和 episode 边界的处理：

```python
ep_idx = len(episode_final_values) - 1

for t in reversed(range(T)):
    if dones_t[t] > 0.5:
        # episode 边界（包括 t == T-1）：使用该 episode 的 bootstrap value
        if 0 <= ep_idx < len(episode_final_values):
            next_value = episode_final_values[ep_idx]
            ep_idx -= 1
        else:
            next_value = 0.0
    else:
        next_value = values_t[t + 1] if t < T - 1 else final_value

    delta = rewards_t[t] + gamma * next_value - values_t[t]
    gae = delta + gamma * gae_lambda * (1.0 - dones_t[t]) * gae
    advantages[t] = gae
```

### 7.3 推荐的完整修复

推荐方案 B，因为它：
1. 同时修复 Bug A 和 Bug B
2. 逻辑更简洁，消除 `t == T-1` 的特殊分支
3. `dones` 仅用于 GAE 传播重置，不再影响 `next_value` 的选择
4. 移除了标量 `final_value` 参数的依赖（可保留用于向后兼容）

---

## 8. 验证建议

1. **单元测试**：构造已知 reward/value/dones 的截断 episode，验证 GAE 输出与手算一致
2. **多 episode 测试**：构造含截断和自然终止的混合 batch，验证每个 episode 边界的 `next_value` 正确
3. **训练对比**：修复前后跑相同 seed 的训练，观察 advantage 分布和 value loss 变化
