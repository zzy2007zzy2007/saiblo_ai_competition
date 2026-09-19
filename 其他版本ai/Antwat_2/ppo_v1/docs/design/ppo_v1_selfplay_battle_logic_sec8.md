# 8. PPO Agent 决策流程

> 参考源码：
> - [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) L899–952（PPOAgent 类）、L628–660（`_select_action()`）、L662–823（`_ppo_update()`）、L825–856（`_compute_gae()`）
> - [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py)（完整网络结构）
> - [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml)（PPO 配置参数）

---

## 8.1 PPOAgent 类

`PPOAgent` 是对策略网络 `AntWarPolicyValueNetwork` 的轻量封装，提供三个用于不同场景的推理接口。定义在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L899-L952)。

### 8.1.1 `act(observation, deterministic=False)`

**用途**：训练过程中使用（SelfPlay 数据收集阶段）。将 NumPy 格式的 observation 转换为 tensor，调用 `policy.get_action()`，返回 `(action, log_prob, value)` 三元组。

```python
def act(
    self,
    observation: Dict[str, np.ndarray],
    deterministic: bool = False,
) -> tuple:
    board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
    global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
    action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

    with torch.no_grad():
        action, log_prob, value = self.policy.get_action(
            board, global_obs, action_mask, deterministic=deterministic
        )

    return (
        action.cpu().item(),
        log_prob.cpu().item(),
        value.cpu().item(),
    )
```

**流程**：

1. 从 `observation` 字典中取出 `board`（shape `(28, 19, 19)`）、`global`（shape `(33,)`）、`action_mask`（shape `(96,)`）
2. 分别转为 `FloatTensor`，通过 `unsqueeze(0)` 增加 batch 维度（`(1, ...)`），再 `.to(self.device)` 送到 GPU
3. 在 `torch.no_grad()` 上下文中调用 `policy.get_action()` 进行推理
4. 将返回的 `(action, log_prob, value)` 从 GPU tensor 转为 Python 标量返回

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `observation` | `Dict[str, np.ndarray]` | — | 包含 `board`、`global`、`action_mask` 的观察字典 |
| `deterministic` | `bool` | `False` | 是否确定性选动作；训练时为 `False`（sampling），评估时为 `True`（argmax） |

**返回值**：

| 返回 | 类型 | 说明 |
|:---|:---|:---|
| `action` | `int` | 选中的动作 ID（0–95） |
| `log_prob` | `float` | 该动作的对数概率 |
| `value` | `float` | 当前状态的价值估计 |

### 8.1.2 `select_action(obs)`

**用途**：评估中对战使用（Baseline Battle 中对手/Agent 的决策）。与 `act()` 的区别在于：
- 始终以 **`deterministic=True`** 模式调用，确保对战行为可复现
- 只返回 `action`，不需要 `log_prob` 和 `value`（因为评估不进行梯度更新）

```python
def select_action(self, obs: Dict[str, Any]) -> int:
    player_obs = obs.get('player_0', obs)
    board = torch.FloatTensor(player_obs['board']).unsqueeze(0).to(self.device)
    global_obs = torch.FloatTensor(player_obs['global']).unsqueeze(0).to(self.device)
    action_mask = torch.FloatTensor(player_obs['action_mask']).unsqueeze(0).to(self.device)

    with torch.no_grad():
        action, _, _ = self.policy.get_action(
            board, global_obs, action_mask, deterministic=True
        )

    return action.cpu().item()
```

**特殊处理**：通过 `obs.get('player_0', obs)` 兼容两种 obs 格式：
- 直接传入单个玩家的观测字典（`{'board': ..., 'global': ..., 'action_mask': ...}`）
- 传入嵌套格式（`{'player_0': {...}}`），自动提取 `player_0`

### 8.1.3 `evaluate(observation)`

**用途**：返回当前状态下的**完整动作概率分布**和**价值估计**。通常用于调试、分析或需要概率分布的场景（如计算 entropy）。

```python
def evaluate(
    self,
    observation: Dict[str, np.ndarray],
) -> tuple:
    board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
    global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
    action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

    with torch.no_grad():
        action_logits, value = self.policy(
            board, global_obs, action_mask
        )

    probs = torch.softmax(action_logits, dim=-1)
    return probs.cpu().numpy(), value.cpu().item()
```

**返回值**：

| 返回 | 类型 | 说明 |
|:---|:---|:---|
| `probs` | `np.ndarray` shape `(1, 96)` | 各动作的 softmax 概率（已受 action mask 影响） |
| `value` | `float` | 当前状态的价值估计 |

### 8.1.4 三个方法的对比总结

| 特性 | `act()` | `select_action()` | `evaluate()` |
|:---|:---|:---|:---|
| 使用场景 | 训练数据收集 | 评估/Baseline Battle | 调试/概率分析 |
| deterministic | 可配置（默认 `False`） | 始终 `True` | N/A（直接返回分布） |
| 返回 action | ✅ | ✅ | ❌ |
| 返回 log_prob | ✅ | ❌ | ❌ |
| 返回 value | ✅ | ❌ | ✅ |
| 返回概率分布 | ❌ | ❌ | ✅ |
| 调用网络方法 | `get_action()` | `get_action()` | `forward()` |

---

## 8.2 `_select_action()` 内部流程

`PPOTrainer._select_action()` 是训练数据收集中选动作的核心方法，定义在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L628-L660)。

### 8.2.1 完整流程图

```
observation (dict[str, np.ndarray])
  │
  ├─► board:        np→FloatTensor → unsqueeze(0) → .to(device)
  ├─► global_obs:   np→FloatTensor → unsqueeze(0) → .to(device)
  ├─► action_mask:  np→FloatTensor → unsqueeze(0) → .to(device)
  │
  ▼
[NaN 检测] board / global_obs 是否含 NaN？
  │ 是 → torch.nan_to_num(nan=0.0) 清理 + 日志警告
  │
  ▼
policy.get_action(board, global_obs, action_mask, deterministic=False)
  │
  ├─► cnn_encoder(board)          → board_encoded
  ├─► mlp_encoder(global_obs)     → global_encoded
  ├─► cat + projection            → merged
  ├─► policy_head(merged)         → action_logits
  ├─► value_head(merged)          → value
  ├─► action_logits.masked_fill(mask==0, -inf)
  ├─► action_logits - max         → 稳定化
  ├─► softmax → probs             → multinomial 采样  → action
  └─► log_softmax → gather        → action_log_prob
  │
  ▼
(action, log_prob, value)
  │
  ▼
[NaN 检测] action / log_prob / value 是否含 NaN？
  │ 是 → 替换为 (0, 0.0, 0.0) + 日志警告
  │
  ▼
转 CPU → .item()
  │
  ▼
返回 (action: int, log_prob: float, value: float)
```

### 8.2.2 详细步骤

**步骤 1：Observation → Tensor → Device**

```python
board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)
```

原始 observation 中的三个字段都是 `np.ndarray`，维度分别是：
- `board`: `(28, 19, 19)` → `FloatTensor` → `(1, 28, 19, 19)`
- `global_obs`: `(33,)` → `FloatTensor` → `(1, 33)`
- `action_mask`: `(96,)` → `FloatTensor` → `(1, 96)`

**步骤 2：输入 NaN 检测与清理**

```python
if torch.isnan(board).any() or torch.isnan(global_obs).any():
    logger.warning(...)
    board = torch.nan_to_num(board, nan=0.0)
    global_obs = torch.nan_to_num(global_obs, nan=0.0)
```

在将数据送入网络前，检查 `board` 和 `global_obs` 是否包含 NaN。若存在，用 `torch.nan_to_num()` 将 NaN 替换为 `0.0`。这确保网络不会因为输入异常而产生不可控的输出。

**步骤 3：网络前向 + 采样**

```python
with torch.no_grad():
    action, log_prob, value = self.policy.get_action(
        board, global_obs, action_mask, deterministic=False
    )
```

调用 `AntWarPolicyValueNetwork.get_action()` 进行完整的前向传播和动作采样（详见 §8.3）。

**步骤 4：输出 NaN 检测**

```python
if torch.isnan(action) or torch.isnan(log_prob) or torch.isnan(value):
    logger.warning(...)
    action = torch.tensor(0, device=self.device)
    log_prob = torch.tensor(0.0, device=self.device)
    value = torch.tensor(0.0, device=self.device)
```

如果网络输出的 `action`、`log_prob`、`value` 任一为 NaN，则将所有值替换为安全的默认值（action=0 即 NO_OP，log_prob=0，value=0）。

**步骤 5：转回 CPU 并返回**

```python
action = action.cpu().item()
log_prob = log_prob.cpu().item()
value = value.cpu().item()
return action, log_prob, value
```

### 8.2.3 调试日志

每 50 个 episode 输出一次调试日志：

```python
if self.episode_count % 50 == 0:
    logger.debug(f"[调试] 动作选择: action={raw_action}, log_prob={raw_log_prob:.4f}, value={raw_value:.4f}")
```

---

## 8.3 Deterministic vs Sampling 模式

在 `AntWarPolicyValueNetwork.get_action()` 中，通过 `deterministic` 参数控制动作选择策略。定义在 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L268-L294)。

```python
def get_action(
    self,
    board: torch.Tensor,
    global_features: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_logits, value = self.forward(board, global_features, action_mask)

    action_logits_stable = action_logits - action_logits.max(dim=-1, keepdim=True)[0]

    if deterministic:
        action = torch.argmax(action_logits, dim=-1)
    else:
        probs = F.softmax(action_logits_stable, dim=-1)
        probs = probs.clamp(min=1e-10)
        valid_mask = torch.isfinite(probs) & (probs > 0)
        if not valid_mask.any():
            probs = torch.ones_like(probs) / probs.numel()
        else:
            probs = probs / probs.sum(dim=-1, keepdim=True)
        action = torch.multinomial(probs, 1).squeeze(-1)

    log_prob = F.log_softmax(action_logits_stable, dim=-1)
    action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)

    return action, action_log_prob, value
```

### 8.3.1 数值稳定技巧：logits 减去最大值

无论是 deterministic 还是 sampling 模式，都会先做一步数值稳定化：

```python
action_logits_stable = action_logits - action_logits.max(dim=-1, keepdim=True)[0]
```

将 logits 减去最大值，使得 softmax 计算时不会因为指数爆炸而产生 NaN。这不改变 softmax 结果的相对大小，但保证了数值稳定性。

### 8.3.2 Deterministic 模式（`deterministic=True`）

```python
action = torch.argmax(action_logits, dim=-1)
```

直接在 logits 上取 argmax，选出概率最高的动作。不经过 softmax 转换，效率更高。

**使用场景**：
- Baseline Battle 评估中的对手决策（`select_action()`）
- 需要可复现行为的场景

### 8.3.3 Sampling 模式（`deterministic=False`）

```python
probs = F.softmax(action_logits_stable, dim=-1)
probs = probs.clamp(min=1e-10)
valid_mask = torch.isfinite(probs) & (probs > 0)
if not valid_mask.any():
    probs = torch.ones_like(probs) / probs.numel()
else:
    probs = probs / probs.sum(dim=-1, keepdim=True)
action = torch.multinomial(probs, 1).squeeze(-1)
```

**详细步骤**：

1. **Softmax**：将稳定化后的 logits 转为概率分布
2. **Clamp 最小概率**：`probs.clamp(min=1e-10)` 防止概率为零导致 multinomial 出错
3. **NaN 安全归一化**：
   - 构建 `valid_mask`：概率值必须是有限的且严格大于 0
   - 如果没有任何有效概率（极端情况），退化为均匀分布：`probs = torch.ones_like(probs) / probs.numel()`
   - 否则，只对有效概率进行重新归一化：`probs = probs / probs.sum(dim=-1, keepdim=True)`
4. **Multinomial 采样**：`torch.multinomial(probs, 1)` 从概率分布中随机采样一个动作

**使用场景**：
- 训练过程中的动作选择（`act()` 默认行为）
- 需要探索性行为的场景

### 8.3.4 Log 概率计算

两种模式最终都通过相同的路径计算 `log_prob`：

```python
log_prob = F.log_softmax(action_logits_stable, dim=-1)
action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
```

用 `log_softmax` 计算完整分布的对数概率，然后通过 `gather` 提取所选动作对应的值。这种方法比先 softmax 再 log 更数值稳定。

### 8.3.5 两种模式对比

| 特性 | Deterministic（`True`） | Sampling（`False`） |
|:---|:---|:---|
| 选动作方式 | `argmax(logits)` | `softmax → multinomial` |
| 探索性 | 无（始终选最高概率动作） | 有（按概率随机） |
| 可复现性 | 完全可复现 | 需要固定 seed |
| 使用场景 | 评估、Baseline Battle | 训练数据收集 |
| 熵 | 不使用 | 隐式通过采样实现探索 |

---

## 8.4 NaN 检测与清理机制

由于 PPO 训练涉及大量数值运算（指数、除法、梯度反传），NaN 问题在训练中较为常见。代码中建立了多层次的 NaN 检测与清理机制。

### 8.4.1 值边界裁剪（数值安全配置）

在 PPOTrainer 初始化时，从配置文件 `ppo_antwar.yaml` 的 `clip_values` 段读取数值安全边界：

```python
# ppo_antwar.yaml, clip_values 段
clip_values:
    logits_min: -20
    logits_max: 20
    values_min: -10
    values_max: 10
    returns_min: -1e6
    returns_max: 1e6
    advantages_min: -1e4
    advantages_max: 1e4
    prob_min: 1e-10
    ratio_min: 1e-5
    ratio_max: 1e5
    loss_max: 1e6
    nan_threshold: 5
```

在训练中对各张量进行 clamp 操作，防止极端值：

```python
# 值函数输出截断
old_values = torch.clamp(old_values, min=self.VALUES_MIN, max=self.VALUES_MAX)
values = torch.clamp(values, min=self.VALUES_MIN, max=self.VALUES_MAX)

# Returns 和 Advantages 截断
returns = torch.clamp(returns, min=self.RETURNS_MIN, max=self.RETURNS_MAX)
advantages = torch.clamp(advantages, min=self.ADVANTAGES_MIN, max=self.ADVANTAGES_MAX)

# Ratio 截断（防止 exp 爆炸）
ratio = torch.clamp(ratio, min=self.RATIO_MIN, max=self.RATIO_MAX)
```

### 8.4.2 `_check_tensor_nan()` — 检测异常值

```python
@staticmethod
def _check_tensor_nan(name: str, tensor: torch.Tensor) -> bool:
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        logger.warning(f"[警告] {name} 包含异常值: NaN={has_nan}, Inf={has_inf}")
        return True
    return False
```

同时检测 NaN 和 Inf，存在任一异常时返回 `True` 并打印警告日志。

### 8.4.3 `_clean_tensor()` — 清理异常值

```python
@staticmethod
def _clean_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    if tensor.numel() == 1:
        if torch.isnan(tensor).item() or torch.isinf(tensor).item():
            logger.warning(f"[警告] 清理 {name}: 异常值={tensor.item()}")
            return torch.tensor(0.0, device=tensor.device)
    else:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            logger.warning(f"[警告] 清理 {name}: NaN={nan_count}, Inf={inf_count}")
            return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
    return tensor
```

- 标量（`numel() == 1`）：直接替换为 `0.0`
- 向量/矩阵：使用 `torch.nan_to_num()` 将 NaN 替换为 `0.0`，`+inf` 替换为 `1e4`，`-inf` 替换为 `-1e4`

### 8.4.4 `_on_nan_detected()` — 阈值保护

```python
def _on_nan_detected(self) -> bool:
    self.nan_count += 1
    logger.warning(f"[警告] 检测到第 {self.nan_count} 次 NaN")

    if self.nan_count >= NAN_THRESHOLD:
        logger.error(f"[错误] 连续 {self.nan_count} 次检测到 NaN，训练异常")
        return False
    return True
```

当连续检测到 `nan_threshold`（默认 5）次 NaN 时，返回 `False`，触发训练跳过当前批次或终止更新。

### 8.4.5 NaN 检测的完整调用链

在 `_ppo_update()` 中的 NaN 检测发生位置：

```
对每个 mini-batch:
  1. old_values      → _clean_tensor + clamp
  2. returns          → _compute_returns → _check_tensor_nan → _clean_tensor + clamp
  3. advantages       → returns - old_values.detach() → _check_tensor_nan → _clean_tensor + clamp
  4. action_log_probs → evaluate_actions → _check_tensor_nan → _clean_tensor
  5. values           → evaluate_actions → _check_tensor_nan → _clean_tensor + clamp
  6. entropy          → evaluate_actions → _check_tensor_nan → _clean_tensor
  7. ratio            → clamp
  8. policy_loss      → _clean_tensor
  9. value_loss       → _clean_tensor
  10. entropy_loss    → _clean_tensor
  11. total_loss      → _clean_tensor → NaN/Inf 最终检查 → _on_nan_detected
```

如果最终 `loss` 仍为 NaN，则跳过该 mini-batch 的梯度更新，并调用 `_on_nan_detected()` 计数。当累积达到阈值时，终止当前 PPO 更新。

---

## 8.5 PPO 更新全流程

PPO 更新由 `PPOTrainer._ppo_update()` 执行，定义在 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L662-L823)。更新发生在每个 episode 结束后，使用 `EpisodeBatch` 中收集的经验数据。

### 8.5.1 整体流程概览

```
EpisodeBatch (一局完整轨迹)
  │
  ├─► 奖励统计：mean, std, min, max
  │
  ├─► 随机打乱 episode 内 steps 的顺序
  │
  ├─► for epoch in range(ppo_epochs):                    # 多 epoch
  │     for mini_batch in split(batch_size):              # 小批量
  │       │
  │       ├─► 数据准备：取出 board/global/action_mask/actions/log_probs/values/rewards/dones
  │       ├─► old_values 清理 + clamp
  │       ├─► _compute_returns() → returns (基于 _compute_gae)
  │       ├─► advantages = returns - old_values.detach()
  │       ├─► returns/advantages NaN 清理 + clamp
  │       │
  │       ├─► policy.evaluate_actions() → new_log_probs, new_values, entropy
  │       ├─► new_log_probs/new_values/entropy NaN 清理
  │       │
  │       ├─► ratio = exp(new_log_probs - old_log_probs), clamp
  │       ├─► Policy Loss (Clipped PPO):
  │       │     surr1 = ratio * advantages
  │       │     surr2 = clip(ratio, 1-ε, 1+ε) * advantages
  │       │     policy_loss = -min(surr1, surr2).mean()
  │       │
  │       ├─► Value Loss (Clipped):
  │       │     v_clipped = old_values + clip(new_values - old_values, -ε_vf, +ε_vf)
  │       │     vf_loss = max(MSE(new_values, returns), MSE(v_clipped, returns)).mean()
  │       │
  │       ├─► Entropy Bonus:
  │       │     entropy_loss = -entropy.mean()
  │       │
  │       ├─► total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
  │       │
  │       ├─► total_loss NaN 检查 → 异常跳过
  │       ├─► loss.backward()
  │       ├─► 梯度范数日志（分级告警）
  │       ├─► 分别裁剪:
  │       │     clip_grad_norm_(policy_params, max_grad_norm=0.5)
  │       │     clip_grad_norm_(value_params, max_grad_norm_vf=0.2)
  │       ├─► optimizer.step()
  │
  └─► 返回训练指标
```

### 8.5.2 GAE 计算（`_compute_gae`）

GAE（Generalized Advantage Estimation）在更新开始前对每个 mini-batch 独立计算。核心实现见 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L825-L856)。

```python
def _compute_gae(
    self,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    running_advantage = 0.0
    gamma = self.ppo_config.gamma
    gae_lambda = self.ppo_config.gae_lambda

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        running_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * running_advantage
        advantages[t] = running_advantage

    returns = advantages + values.detach()
    return advantages, returns
```

**配置参数**：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ 参数，平衡 bias 和 variance |

**计算流程**：

1. **反向遍历**：从最后一步到第一步计算 advantage
2. **TD 误差（δ）**：
   ```
   δ_t = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
   ```
   - 最后一步的 `next_value = 0.0`
   - `done=1` 时，下一状态的价值不参与计算（episode 结束）
3. **GAE 累积**：
   ```
   A_t = δ_t + γ·λ·(1 - done_t)·A_{t+1}
   ```
4. **Returns**：`R_t = A_t + V(s_t)`

### 8.5.3 数据准备与打乱

```python
indices = np.arange(batch_size)
np.random.shuffle(indices)
```

在每个 epoch 开始前，对整个 episode 的 step 索引进行随机打乱，以消除时间相关性。然后在外层 `ppo_epochs` 循环中，内层按 `batch_size` 切分 mini-batch：

```python
for _ in range(self.ppo_config.ppo_epochs):      # 默认 4
    for start in range(0, batch_size, self.ppo_config.batch_size):  # 默认 64
        end = min(start + self.ppo_config.batch_size, batch_size)
        batch_indices = indices[start:end]
```

**关键配置**：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `ppo_epochs` | 4 | 每个 episode 数据重复使用的 epoch 数 |
| `batch_size` | 64 | mini-batch 大小 |

### 8.5.4 Policy Loss（Clipped PPO Surrogate Objective）

```python
ratio = torch.exp(action_log_probs - old_log_probs)
ratio = torch.clamp(ratio, min=self.RATIO_MIN, max=self.RATIO_MAX)

surr1 = ratio * advantages.detach()
surr2 = torch.clamp(
    ratio, 1 - self.ppo_config.clip_eps, 1 + self.ppo_config.clip_eps
) * advantages.detach()

policy_loss = -torch.min(surr1, surr2).mean()
```

**数学公式**：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

$$
L^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \cdot A_t,\; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \cdot A_t \right) \right]
$$

**关键点**：
- `ratio`：新旧策略的概率比。若 `ratio > 1 + clip_eps`，表明新策略对该动作的偏好大幅增加，会被 `clip` 限制住
- 对正 advantage 的动作，鼓励概率增大但不超 `1 + clip_eps`；对负 advantage 的动作，鼓励概率减小但不低于 `1 - clip_eps`
- 最终 loss = `-min(surr1, surr2)`，取保守的估计

**配置参数**：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `clip_eps` | 0.2 | PPO 裁剪范围 ε |

### 8.5.5 Value Loss（Value Clipping）

```python
values_clipped = old_values + torch.clamp(
    values - old_values,
    -self.ppo_config.clip_eps_vf,
    self.ppo_config.clip_eps_vf
)
value_loss_unclipped = F.mse_loss(values, returns.detach(), reduction='none')
value_loss_clipped = F.mse_loss(values_clipped, returns.detach(), reduction='none')
value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

**数学公式**：

$$
V_{\text{clipped}} = V_{\text{old}} + \text{clip}(V_{\text{new}} - V_{\text{old}}, -\epsilon_{vf}, +\epsilon_{vf})
$$

$$
L^{\text{VF}} = \mathbb{E}\left[ \max\left( (V_{\text{new}} - R)^2,\; (V_{\text{clipped}} - R)^2 \right) \right]
$$

**关键点**：
- 与 Policy Loss 的裁剪思想一致：限制单次更新的价值估计变化幅度
- 取 `max` 确保保守更新——即选择更大的 loss 进行优化，防止价值估计剧烈波动

**配置参数**：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `clip_eps_vf` | 0.2 | Value Clipping 范围 |
| `vf_coef` | 0.02 | Value Loss 权重系数 |

### 8.5.6 Entropy Bonus

```python
entropy_loss = -entropy.mean()
```

Entropy 来自 `evaluate_actions()` 的计算结果，取负号后构成熵正则项。最大化 entropy（即最小化 `-entropy`）鼓励策略保持一定的随机性，避免过早收敛到次优的确定性策略。

**配置参数**：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `ent_coef` | 0.15 | 初始熵系数 |
| `ent_coef_final` | 0.15 | 最终熵系数（若不退火则始终为初始值） |
| `ent_anneal_power` | 1.0 | 熵退火幂次（线性退火） |

### 8.5.7 总 Loss

```python
loss = (
    policy_loss
    + self.ppo_config.vf_coef * value_loss
    + self.current_ent_coef * entropy_loss
)
```

**最终的目标函数**：

$$
L = L^{\text{CLIP}} + \text{vf\_coef} \cdot L^{\text{VF}} + \text{ent\_coef} \cdot (-H)
$$

### 8.5.8 梯度分别裁剪

```python
policy_params = []
value_params = []

for name, param in self.policy.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm ** 2

        if 'value_head' in name:
            value_params.append(param)
            if grad_norm > 2.0:
                logger.warning(f"[警告] 价值网络参数 {name} 梯度范数过大: {grad_norm:.2f}")
        else:
            policy_params.append(param)
            if grad_norm > 10.0:
                logger.warning(f"[警告] 策略网络参数 {name} 梯度范数过大: {grad_norm:.2f}")

nn.utils.clip_grad_norm_(policy_params, self.ppo_config.max_grad_norm)
nn.utils.clip_grad_norm_(value_params, self.ppo_config.max_grad_norm_vf)
```

**设计要点**：

1. **参数分组**：根据参数名是否包含 `'value_head'`，将网络参数分为策略网络参数（`policy_params`）和价值网络参数（`value_params`）两组
2. **分别裁剪**：
   - 策略网络：`max_grad_norm = 0.5`（较宽松，允许策略梯度有更大变化）
   - 价值网络：`max_grad_norm_vf = 0.2`（更严格，价值函数需要更稳定的学习）
3. **梯度监控**：逐参数检查梯度范数，策略参数超过 10.0 或价值参数超过 2.0 时输出警告
4. **总体监控**：计算 `total_grad_norm`（所有参数梯度的 L2 范数之和的平方根），超过 `max_grad_norm * 10` 时发出警告

**配置参数**：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `max_grad_norm` | 0.5 | 策略网络梯度裁剪阈值 |
| `max_grad_norm_vf` | 0.2 | 价值网络梯度裁剪阈值 |

---

## 8.6 网络结构

`AntWarPolicyValueNetwork` 是 PPO 训练中的核心神经网络，定义在 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L212-L318)。采用**双编码器 + 双头输出**的架构。

### 8.6.1 整体架构图

```
                       board (batch, 28, 19, 19)
                              │
                   ┌──────────▼──────────┐
                   │   HexCNNEncoder     │
                   │                     │
                   │  conv1(1×1): 28→64  │
                   │    + hex_conv1     │
                   │  conv2(3×3,s=2):   │    global (batch, 33)
                   │    64→128          │         │
                   │    + hex_conv2     │  ┌──────▼──────┐
                   │  conv3(3×3,s=2):   │  │ MLPEncoder  │
                   │    128→256        │  │              │
                   │    + hex_conv3     │  │ fc1: 33→128 │
                   │                     │  │ fc2: 128→128│
                   │  AdaptiveAvgPool2d │  └──────┬──────┘
                   │  (to 5×5)          │         │
                   │  Flatten → 6400    │         │ (batch, 128)
                   └──────────┬─────────┘         │
                              │                   │
                         (batch, 6400)    (batch, 128)
                              │                   │
                              └─────┬─────────────┘
                                    │
                              torch.cat(dim=1)
                                    │
                              (batch, 6528)
                                    │
                         ┌──────────▼──────────┐
                         │   projection (Linear)│
                         │   6528 → 256         │
                         └──────────┬──────────┘
                                    │
                              (batch, 256)
                                    │
                    ┌───────────────┼───────────────┐
                    │                               │
            ┌───────▼───────┐               ┌───────▼───────┐
            │  policy_head  │               │  value_head   │
            │  Linear       │               │  Linear       │
            │  256 → 96     │               │  256 → 1      │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
              action_logits                      value
              (batch, 96)                      (batch, 1)
                    │
            ┌───────▼───────┐
            │  action_mask  │
            │  masked_fill  │
            │  (0 → -inf)   │
            └───────┬───────┘
                    │
           action_logits (masked)
           (batch, 96)
```

### 8.6.2 HexCNNEncoder — 棋盘编码器

六边形卷积编码器，专为六边形网格地图设计。见 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L83-L183)。

**架构参数**：

| 参数 | 值 | 说明 |
|:---|:---|:---|
| `input_channels` | 28 | 输入特征通道数 |
| `hidden_channels` | 128（基准）/ 256（实际使用） | 隐藏层通道数基准 |
| `map_size` | 19 | 地图大小 |

**实际通道配置**（`hidden_channels=256` 时）：

| 层 | 输入通道 | 输出通道 | 说明 |
|:---|:---|:---|:---|
| `conv1` | 28 | 128 | 1×1 卷积，投影到隐藏空间 |
| `hex_conv1` | 128 | 128 | 六边形方向卷积 + 残差连接 |
| `conv2` | 128 | 256 | 3×3 卷积，stride=2（下采样到 10×10） |
| `hex_conv2` | 256 | 256 | 六边形方向卷积 + 残差连接 |
| `conv3` | 256 | 512 | 3×3 卷积，stride=2（下采样到 5×5） |
| `hex_conv3` | 512 | 512 | 六边形方向卷积 + 残差连接 |
| `pool` | — | — | AdaptiveAvgPool2d → (5, 5)，Flatten → 12800 |

每层卷积后跟 BatchNorm2d + ReLU 激活。`HexConv` 通过 `torch.einsum` 进行高效的方向权重计算。

**HexConv 六方向定义**：

```python
HEX_DIRECTIONS = [
    (+1, 0), (+1, -1), (0, -1),
    (-1, 0), (-1, +1), (0, +1),
]
```

**残差结构**：每个 `HexConv` 的输出与 `conv` 的输出做加法（`x = x + F.relu(self.hex_conv1(x, hex_neighbor))`），形成残差连接，增强梯度传播。

### 8.6.3 MLPEncoder — 全局编码器

两层全连接网络，处理 33 维全局标量特征。见 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L186-L195)。

```python
class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
```

**参数配置**：

| 层 | 输入 | 输出 | 激活函数 |
|:---|:---|:---|:---|
| `fc1` | 33 | 128 | ReLU |
| `fc2` | 128 | 128 | ReLU |

### 8.6.4 合并与投影

```python
merged = torch.cat([board_encoded, global_encoded], dim=1)
merged = self.projection(merged)
```

- **拼接**：CNN 输出（12800 维）与 MLP 输出（128 维）在 dim=1 上拼接，得到 12928 维向量
- **投影**：通过 `Linear(12928, 256)` 投影到 256 维共享表示空间

### 8.6.5 Policy Head

```python
self.policy_head = nn.Linear(hidden_dim, action_dim)  # 256 → 96
```

输出 96 维的 action logits，对应 96 个可选动作（详见第 5 章动作空间）。在前向传播中，若提供了 `action_mask`，则将 mask 值为 0 的动作 logits 设为 `-inf`：

```python
if action_mask is not None:
    action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))
```

### 8.6.6 Value Head

```python
self.value_head = nn.Linear(hidden_dim, 1)  # 256 → 1
```

输出一个标量值，表示当前状态的价值估计 $V(s)$。

### 8.6.7 初始化策略

网络使用**正交初始化（orthogonal initialization）**，见 [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L237-L246)：

```python
# 应用正交初始化
self.apply(orthogonal_init)

# 保持 value_head 的零初始化
nn.init.zeros_(self.value_head.weight)
nn.init.zeros_(self.value_head.bias)

