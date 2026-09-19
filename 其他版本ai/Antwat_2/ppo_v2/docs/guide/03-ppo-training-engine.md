# PPO 训练引擎模块 (PPO Training Engine)

## 1. 背景与定位

### 1.1 业务问题

Proximal Policy Optimization (PPO) 是目前最主流的深度强化学习算法之一。其核心思想是：在与环境交互采集一批轨迹数据后，使用这些数据对策略网络进行多次梯度更新，但通过**裁剪 (Clipping)** 机制限制每次更新的幅度，避免策略发生剧烈变化导致训练崩溃。

**PPO 训练引擎模块**的核心业务目标是：**实现完整的 PPO 算法**，包括轨迹数据收集、GAE 优势估计、多 epoch 小批量更新、损失计算、梯度裁剪、学习率调度、数值安全防护和检查点管理。该模块不关心"跟谁打"（对手选择）、"怎么组织循环"（训练编排），只关心"如何从一批轨迹中学习"。

### 1.2 在整个系统中的位置

```
自对弈编排 (Self-Play Orchestration)
        │
        │  收集轨迹 → 调用 PPO 更新
        ▼
┌───────────────────────────────────────────┐
│           PPO 训练引擎                     │
│                                           │
│  EpisodeBatch ──▶ GAE计算 ──▶ PPO更新     │
│  (轨迹数据)         │          │           │
│                     │          ├─ 损失计算 │
│                     │          ├─ 梯度裁剪 │
│                     │          └─ 参数更新 │
│                     │                      │
│  ┌──────────────────────────────────────┐ │
│  │  辅助组件                            │ │
│  │  - safety (NaN检测/恢复)            │ │
│  │  - checkpoint_manager (模型保存)    │ │
│  │  - agent (推理封装)                 │ │
│  │  - batch_metrics_writer (指标日志)  │ │
│  │  - callbacks (回调钩子)             │ │
│  └──────────────────────────────────────┘ │
└───────────────────────────────────────────┘
        │
        ▼
    策略网络 (Policy Network)
```

- **上层**：自对弈编排模块调用 `PPOTrainer` 的 `collect_episode()` 收集数据，累积到阈值后调用内部 `_ppo_update()` 执行学习
- **下层**：依赖策略网络模块进行前向和反向传播

### 1.3 与其他模块的关系

| 外部模块 | 关系 | 交互方式 |
|---------|------|---------|
| 策略网络 (`antwar_net.py`) | 依赖 | 创建网络实例，调用 `forward()`/`evaluate_actions()` |
| 基础设施 (`action_constants`) | 依赖 | 使用 `TYPE_CONFIG` 进行动作类型统计 |
| 基础设施 (`gae.py`) | 依赖 | 调用 `compute_gae()` 计算优势估计 |
| 基础设施 (`obs_utils.py`) | 依赖 | 在推理时调用 `observation_to_tensors()` 进行观测转换 |
| 自对弈编排 (`selfplay.py`) | 被依赖 | 创建 `PPOTrainer`，调用 `get_agent()`/`collect_episode()`，触发 PPO 更新 |

---

## 2. 核心概念

### 2.1 关键术语

| 术语 | 说明 |
|------|------|
| **EpisodeBatch** | 一局对战的完整轨迹数据容器，存储每个时间步的 obs/action/reward/value/log_prob/done |
| **GAE (Generalized Advantage Estimation)** | 广义优势估计，在 bias-variance tradeoff 中通过 λ 参数平衡 |
| **PPO-Clip** | PPO 的核心损失函数：$L^{CLIP}(\theta) = \mathbb{E}[\min(r(\theta)\hat{A}, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A})]$ |
| **Minibatch** | 从单条轨迹或多个合并轨迹中随机采样的训练子集 |
| **Epoch** | 对同一批数据多次遍历训练，每次打乱后分 minibatch 更新 |
| **KL Early Stopping** | 如果近似 KL 散度超过阈值，提前终止本轮 epochs，防止过度更新 |
| **Entropy Bonus** | 在损失中加入策略熵的负值，鼓励探索 |
| **NaN Recovery** | 检测训练中的 NaN/Inf 并回滚到最近一次健康状态的恢复机制 |

### 2.2 PPO 更新流程概要

```
1. 收集一条或多条轨迹 → EpisodeBatch
2. 将轨迹数据转为 PyTorch 张量
3. 计算 GAE 优势和折扣回报
4. 进入多 epoch 训练循环：
   a. 随机打乱数据
   b. 划分 minibatch
   c. 对每个 minibatch:
      - 前向传播 (evaluate_actions)
      - 计算 PPO-Clip 损失
      - 反向传播 + 梯度裁剪
   d. 检查 KL 散度 → 决定是否早停
5. 保存健康检查点（用于 NaN 恢复）
6. 聚合并返回训练指标
```

---

## 3. 架构总览

### 3.1 模块组成

```
ppo-training-engine/
│
├── trainer/ppo_trainer.py          # PPOTrainer - PPO 算法主实现
├── trainer/batch.py                # EpisodeBatch - 轨迹数据容器
├── trainer/agent.py                # PPOAgent - 推理封装
├── trainer/safety.py               # NanRecoveryHandler - 数值安全
├── trainer/checkpoint_manager.py   # CheckpointManager - 模型保存/加载
├── utils/gae.py                    # compute_gae - 优势估计
├── monitor/batch_metrics_writer.py # BatchMetricsWriter - 训练指标写入
├── monitor/metrics_cache.py        # MetricsCache - 指标缓存
├── callbacks/metrics_callback.py   # MetricsLoggingCallback - 终端指标输出
├── callbacks/checkpoint_callback.py# CheckpointCallback - 定期保存检查点
└── callbacks/tensorboard_callback.py # TensorBoardCallback - TensorBoard 记录
```

### 3.2 组件关系

```
                      PPOTrainer (核心调度器)
                     /    |    |    |    \
                    /     |    |    |     \
                   ▼      ▼    ▼    ▼      ▼
             EpisodeBatch  │  safety  checkpoint_manager
                   │       │                    │
                   │       │  PPOAgent ──── 策略网络
                   │       │
                   │  compute_gae(utils/gae.py)
                   │
              回调系统 (callbacks)
                   │
            ┌──────┼──────┐
            ▼      ▼      ▼
     metrics   checkpoint  tensorboard
     callback  callback    callback
            │      │
            ▼      ▼
     BatchMetrics  Checkpoint
     Writer        Manager
```

---

## 4. 核心组件详解

### 4.1 PPOTrainer (`trainer/ppo_trainer.py`)

#### 4.1.1 职责

PPO 算法的核心实现，是整个训练引擎的调度中心。职责包括：
- 构建策略网络和双优化器
- 单步动作选择（`_select_action`）
- 单条轨迹收集（`collect_episode`）
- PPO 多 epoch 更新（`_ppo_update`）
- 学习率调度（`update_lr_schedule`）
- 检查点管理（委托 `CheckpointManager`）
- Agent 推理接口（`get_agent()`）

#### 4.1.2 构造函数

```python
def __init__(self, env_factory, config: Dict, device: torch.device, base_dir: str, callbacks=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `env_factory` | Callable | 环境工厂函数，返回 `(env_0, env_1)` 用于双视角数据收集 |
| `config` | Dict | 完整配置字典，包含 `ppo`/`training`/`network`/`selfplay` 等子节 |
| `device` | torch.device | 计算设备（CPU/CUDA） |
| `base_dir` | str | 基础目录，用于保存检查点和日志 |
| `callbacks` | Optional | 回调对象，用于在训练步中插入自定义逻辑 |

**初始化流程**：
1. 解析配置参数（`_parse_config`）：ppo 超参、训练参数、网络参数
2. 构建网络（`_build_network`）：创建 `AntWarPolicyValueNetwork`
3. 构建双优化器（`_build_optimizers`）：将网络参数按名称前缀分为策略组和价值组
4. 创建 `NanRecoveryHandler` 和 `CheckpointManager`
5. 初始化回调系统

#### 4.1.3 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `collect_episode` | `(env, opponent) -> EpisodeBatch` | 收集单局对战的完整轨迹数据，返回 EpisodeBatch |
| `save_checkpoint` | `(filepath, episode, metrics)` | 保存模型、优化器、配置到 .pt 文件 |
| `load_checkpoint` | `(filepath) -> int` | 从 .pt 文件恢复模型和优化器，返回已训练 episode 数 |
| `get_agent` | `(deterministic=False) -> PPOAgent` | 创建推理用的 PPOAgent 实例 |
| `update_lr_schedule` | `(episode, total_episodes)` | 执行学习率调度（warmup + 余弦退火） |
| `log_config_summary` | `()` | 打印训练配置摘要到日志 |
| `check_batch_quality` | `(batch) -> bool` | 检查批次数据质量（空/NaN/全零奖励） |
| `get_learning_rate` | `() -> (float, float)` | 返回策略/价值网络的当前学习率 |

#### 4.1.4 关键算法：PPO 更新 (`_ppo_update`)

这是最核心的算法流程：

```
_ppo_update(batch, episode)
│
1. 将 batch 转为张量 → batch.to_tensors(device)
│
2. 计算 GAE 优势和折扣回报:
   advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)
   advantages = clamp(advantages, -advantage_clip, advantage_clip)
   returns = clamp(returns, -return_clip, return_clip)
│
3. 计算 TD 误差统计 (returns.mean(), returns.std())
│
4. 进入多 epoch 训练循环 _train_ppo_epochs():
   for epoch in range(ppo_epochs):
       a. 生成随机排列索引并打乱数据
       b. 按 batch_size 划分 minibatch:
           for indices in minibatches:
               _process_minibatch(indices, batch_data, advantages, returns)
│
5. _process_minibatch():
   a. 网络前向: evaluate_actions() → log_probs, values, entropy, aux_preds, features
   b. 计算比率 ratio = exp(new_log_probs - old_log_probs)
   c. 裁剪比率: ratio = clamp(ratio, ratio_min, ratio_max)
   d. _compute_losses():
      - policy_loss = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv).mean()
      - value_loss = 0.5 * max(unclipped_vf, clipped_vf).mean()
      - entropy_loss = -ent_coef * entropy.mean()
      - aux_loss = aux_coef * compute_auxiliary_loss(...)
      - total_loss = policy_loss + vf_coef * value_loss + entropy_loss + aux_loss
   e. 反向传播 + _check_and_clip_gradients()
   f. 更新策略优化器和价值优化器
   g. 检测当前 minibatch 的 KL 散度 → 如果 KL > target_kl，标记早停
   h. 记录指标（loss, clip_fraction, grad_norm 等）
│
6. on_nan_detected() 检查 NaN → 如果超过阈值则停止训练
│
7. 保存良好状态快照（用于后续 NaN 恢复）
│
8. 聚合所有 minibatch 指标并返回
```

**PPO-Clip 损失详解**：

```
policy_loss = E[ min( ratio * A, clip(ratio, 1-ε, 1+ε) * A ) ]

其中:
  ratio = π_θ(a|s) / π_θ_old(a|s)    # 新旧策略的概率比
  A = advantage estimate              # 优势估计
  ε = clip_epsilon                    # 裁剪阈值（如 0.2）

- 当 A > 0（好动作）：ratio 被限制在 ≤ 1+ε，避免过度利用
- 当 A < 0（差动作）：ratio 被限制在 ≥ 1-ε，避免过度惩罚
```

**双优化器设计**：

```python
VALUE_PREFIXES = ['value_cnn', 'value_mlp', 'value_proj', 'value_layer_norm',
                  'value_head', 'tower_damage_head', 'gold_income_head']

# 策略网络参数（从总参数中排除价值网络参数）
policy_params = [p for n, p in network.named_parameters()
                 if not any(n.startswith(prefix) for prefix in VALUE_PREFIXES)]
# 价值网络参数
value_params = [p for n, p in network.named_parameters()
                if any(n.startswith(prefix) for prefix in VALUE_PREFIXES)]

self.optimizer = Adam(policy_params, lr=policy_lr)
self.value_optimizer = Adam(value_params, lr=value_lr)
```

#### 4.1.5 关键算法：轨迹收集 (`collect_episode`)

```
collect_episode(env, opponent)
│
1. obs, info = env.reset()
│
2. 重复直到 terminated:
   a. own_action = _select_action(obs[own_player])  # 己方动作
   b. opp_action = opponent.act(obs[opp_player])     # 对手动作
   c. next_obs, reward, terminated, truncated, info = env.step({0: act_0, 1: act_1})
   d. batch.add_step(obs, action, reward, value, log_prob, done)
   e. obs = next_obs
│
3. 返回 EpisodeBatch
```

#### 4.1.6 关键算法：学习率调度 (`update_lr_schedule`)

支持 warmup + 余弦退火：

```
如果 episode < warmup_steps:
    lr_scale = episode / warmup_steps
否则:
    progress = (episode - warmup_steps) / (total_episodes - warmup_steps)
    lr_scale = 0.5 * (1 + cos(π * progress))

实际 lr = base_lr * lr_scale
```

#### 4.1.7 关键算法：梯度裁剪 (`_check_and_clip_gradients`)

对策略网络和价值网络分别进行梯度裁剪：

```
_check_and_clip_gradients()
│
1. 检测所有参数的梯度中是否有 NaN/Inf → 有则跳过此 minibatch
2. 分别裁剪策略网络和价值网络的梯度范数:
   clip_grad_norm_(policy_params, max_grad_norm)
   clip_grad_norm_(value_params, max_grad_norm_vf)
3. 记录大梯度警告（范数 > max_grad_norm × 2）
```

---

### 4.2 EpisodeBatch (`trainer/batch.py`)

#### 4.2.1 职责

轨迹数据容器，在训练循环中收集一局对战的完整经验数据。关键设计是**延迟转换**：数据先以 Python 原生列表存储，在需要训练时才转换为 PyTorch 张量，避免过早分配 GPU 内存。

#### 4.2.2 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `add_step` | `(obs, action, reward, value, log_prob, done)` | 追加一个时间步的数据，从 obs 字典中提取 board/global/action_mask |
| `to_tensors` | `(device) -> Dict[str, Tensor]` | 将所有列表数据转为 PyTorch 张量字典 |
| `merge` | `(batches: List[EpisodeBatch]) -> EpisodeBatch` | 类方法，合并多个 EpisodeBatch，严格检查 aux 数据一致性 |
| `__len__` | `() -> int` | 返回轨迹长度 |

#### 4.2.3 内部数据结构

| 属性 | 类型 | 说明 |
|------|------|------|
| `observations_board` | `List[np.ndarray]` | 棋盘特征 (28, 19, 19) |
| `observations_global` | `List[np.ndarray]` | 全局特征 (33,) |
| `observations_mask` | `List[np.ndarray]` | 动作掩码 (96,) |
| `actions` | `List[int]` | 动作 ID |
| `rewards` | `List[float]` | 奖励值 |
| `values` | `List[float]` | 状态价值估计 |
| `log_probs` | `List[float]` | 动作对数概率 |
| `dones` | `List[bool]` | 终止标记 |
| `aux_tower_damage` | `Optional[np.ndarray]` | 塔伤害辅助标签 (可选) |
| `aux_gold_income` | `Optional[np.ndarray]` | 金币收入辅助标签 (可选) |

---

### 4.3 PPOAgent (`trainer/agent.py`)

#### 4.3.1 职责

策略网络的轻量推理封装，为上层训练循环和评估代码提供简洁的接口。所有推理在 `torch.no_grad()` 上下文中执行。

#### 4.3.2 构造函数

```python
def __init__(self, policy: AntWarPolicyValueNetwork, device: torch.device,
             exploration_epsilon: float, observation_encoder=None)
```

#### 4.3.3 公开方法

| 方法 | 功能 |
|------|------|
| `act(observation, deterministic=False) -> int` | 选择动作，返回 action_id |
| `get_action_values(observation, deterministic=False) -> (action_id, log_prob, value)` | 完整动作选择，同时返回 log_prob 和 value |
| `get_value(observation) -> float` | 仅获取状态价值估计 |

---

### 4.4 NanRecoveryHandler (`trainer/safety.py`)

#### 4.4.1 职责

检测训练过程中的 NaN/Inf 异常，提供回滚恢复机制。这是保障长时间训练稳定性的关键组件。

#### 4.4.2 核心逻辑

```python
class NanRecoveryHandler:
    def __init__(self, nan_threshold: int):
        self._nan_count = 0
        self._nan_threshold = nan_threshold

    def save_good_state(self, policy_state, optimizer_state, value_optimizer_state):
        # 保存当前健康的网络和优化器状态（深拷贝）

    def on_nan_detected(self, episode, policy, optimizer, value_optimizer) -> bool:
        self._nan_count += 1
        if self._nan_count > self._nan_threshold:
            return False  # 停止训练
        # 回滚到最近一次健康状态
        policy.load_state_dict(self._last_good_policy_state)
        return True  # 继续训练
```

辅助函数：
- `check_tensor_nan(tensor, name) -> bool`: 检测 NaN/Inf
- `clean_tensor(tensor) -> Tensor`: 用 `torch.nan_to_num` 替换 NaN/Inf 为安全值

---

### 4.5 CheckpointManager (`trainer/checkpoint_manager.py`)

#### 4.5.1 职责

管理模型检查点的保存和加载，支持策略网络、双优化器的完整状态持久化。

#### 4.5.2 公开方法

| 方法 | 功能 |
|------|------|
| `save(filepath, episode, policy_state, optimizer_state, value_optimizer_state, metrics, config)` | 保存完整检查点 |
| `load(filepath, policy, optimizer, value_optimizer) -> int` | 加载检查点，返回已训练 episode 数 |

#### 4.5.3 检查点结构

```python
{
    'episode': int,
    'policy_state_dict': Dict,
    'optimizer_state_dict': Dict,
    'value_optimizer_state_dict': Dict,
    'metrics': Dict,
    'config': Dict,
}
```

---

### 4.6 compute_gae (`utils/gae.py`)

#### 4.6.1 职责

实现 Generalized Advantage Estimation (GAE) 算法，从轨迹的奖励和价值序列中计算优势估计和折扣回报。

#### 4.6.2 算法

```python
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t+1] * (1 - dones[t])

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
```

**公式**：
$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

$\lambda=0$ 时为 1-step TD（高偏差低方差），$\lambda=1$ 时为 Monte Carlo（低偏差高方差）。

---

### 4.7 训练指标组件

#### 4.7.1 BatchMetricsWriter (`monitor/batch_metrics_writer.py`)

每次 PPO update 后将训练技术指标以 JSONL 格式追加写入 `batch_metrics.jsonl`。记录的指标：
- `episode`, `loss`, `policy_loss`, `value_loss`, `entropy`
- `clip_fraction`, `approx_kl`
- `grad_norm`, `grad_norm_policy`, `grad_norm_value`
- `learning_rate`, `value_learning_rate`
- `reward_mean`, `reward_std`, `return_mean`, `return_std`
- `advantages_mean`, `advantages_std`

#### 4.7.2 MetricsCache (`monitor/metrics_cache.py`)

内存中的指标环形缓冲区，支持追加、最近 N 条查询和滑动窗口统计。用于回调组件获取近期指标。

#### 4.7.3 MetricsLoggingCallback (`callbacks/metrics_callback.py`)

继承 `BasePPOCallback`，每隔 `log_interval` 步从 `_trainer._last_metrics` 提取关键指标并输出到终端（通过 loguru）。

#### 4.7.4 CheckpointCallback (`callbacks/checkpoint_callback.py`)

继承 `BasePPOCallback`，每隔 `save_interval` 步调用 `trainer.save_checkpoint()` 保存检查点，并自动清理旧文件（仅保留最近 `keep_last_n` 个）。

#### 4.7.5 TensorBoardCallback (`callbacks/tensorboard_callback.py`)

继承 `BasePPOCallback`，每隔 `log_interval` 步将标量指标通过 `SummaryWriter` 写入 TensorBoard。在 `on_training_end()` 中关闭 writer。

---

## 5. 外部交互

### 5.1 本模块依赖的外部模块

| 模块 | 依赖内容 | 依赖方式 |
|------|---------|---------|
| 策略网络 (`antwar_net.py`) | `AntWarPolicyValueNetwork` | 创建实例，调用前向/评估方法 |
| 基础设施 (`action_constants`) | `TYPE_CONFIG` | 用于动作类型统计 |
| 基础设施 (`obs_utils.py`) | `observation_to_tensors` | 在 `_select_action` 中方法级导入 |
| 基础设施 (`base_ppo_callback.py`) | `BasePPOCallback` | 各 callback 继承自基类 |

### 5.2 依赖本模块的外部模块

| 模块 | 使用内容 | 使用方式 |
|------|---------|---------|
| 自对弈编排 (`selfplay.py`) | `PPOTrainer`, `EpisodeBatch` | 创建 `PPOTrainer`，收集轨迹，触发 PPO 更新 |
| 对战执行 (`opponent_agent.py`) | `PPOAgent` | `from_checkpoint()` 工厂方法创建推理 agent |

### 5.3 典型交互场景

#### 场景 1：完整的 PPO 更新周期

1. **数据收集阶段**（由自对弈编排驱动）：
   - 多次调用 `PPOTrainer.collect_episode(env, opponent)` 收集轨迹
   - 每次调用返回一个 `EpisodeBatch`
   - 累积到 `update_timesteps` 阈值

2. **PPO 更新阶段**：
   - 合并多个 `EpisodeBatch` → `EpisodeBatch.merge(batches)`
   - 调用 `batch.to_tensors(device)` 将数据迁移到 GPU
   - `_ppo_update()` 内部：
     - `compute_gae()` 计算优势
     - 多 epoch 循环，每 epoch 划分 minibatch
     - 对每个 minibatch：前向 → 损失 → 反向传播
     - KL 早停检查 → NaN 检测
   - `BatchMetricsWriter` 写入训练指标

3. **后处理**：
   - 回调的 `on_step()` 触发（终端输出/检查点保存/TensorBoard）
   - `update_lr_schedule()` 更新学习率

#### 场景 2：从检查点恢复训练

1. `PPOTrainer.load_checkpoint(checkpoint_path)` 被调用
2. `CheckpointManager.load()` 读取 .pt 文件
3. 恢复策略网络权重、双优化器状态
4. 返回已训练的 episode 数
5. 训练循环从该 episode 继续

#### 场景 3：NaN 检测与恢复

1. `_process_minibatch()` 中检测到梯度包含 NaN
2. `check_tensor_nan()` 记录警告日志
3. `_check_and_clip_gradients()` 跳过此 minibatch
4. `on_nan_detected()` 被调用：
   - 如果 NaN 计数 < 阈值：从最近一次健康检查点恢复网络权重
   - 如果 NaN 计数 ≥ 阈值：返回 `False`，触发训练停止

---

## 6. 关键设计决策

### 6.1 双优化器（策略/价值分离）

策略网络和价值网络的参数使用独立的 Adam 优化器，而非共享一个优化器：

- **原因**：策略损失和价值损失的梯度尺度差异很大。价值损失通常比策略损失大 1-2 个数量级，共享优化器会导致学习率无法同时适配两者
- **实现**：按参数名称前缀（`VALUE_PREFIXES`）区分策略参数和价值参数
- **好处**：可以为策略和价值设置不同的学习率与梯度裁剪阈值

### 6.2 延迟张量转换

`EpisodeBatch` 使用 Python 原生列表存储轨迹数据，仅在 PPO 更新前才调用 `to_tensors()` 转换为 GPU 张量：

- **内存效率**：避免在数据收集过程中占用 GPU 显存
- **批量转换**：多局轨迹可以合并后一次性转换，减少 PCIe 传输次数
- **灵活性**：合并、裁剪等操作在 Python 列表中执行更快

### 6.3 数值安全的全局设计

数值安全不是事后补救，而是贯穿训练引擎的各个层面：
- **推理层**：`_select_action()` 检测网络输出的 NaN
- **损失层**：`_compute_losses()` 对 ratio/values/advantages 做数值裁剪
- **梯度层**：`_check_and_clip_gradients()` 检测梯度 NaN，跳过异常 minibatch
- **状态层**：`NanRecoveryHandler` 在训练过程中持续保存健康检查点，提供回滚能力

### 6.4 回调（Callback）系统

训练引擎通过回调机制将非核心功能（指标记录、检查点保存、TensorBoard）解耦出去：

- **核心循环**：`PPOTrainer` 只关心 PPO 算法本身
- **扩展点**：`on_step()`、`on_training_start()`、`on_training_end()` 三个钩子
- **组合**：`CallbackList` 将多个回调组合为一条链，任一回调返回 `False` 即可停止训练
- **工厂**：`create_ppo_callbacks()` 根据配置按需创建回调组合

### 6.5 GAE 参数的设计

GAE 的两个参数 `γ` (gamma) 和 `λ` (gae_lambda) 有明确的语义分工：
- **γ** 决定价值函数的"远视程度"：较小的 γ 使模型更关注短期收益
- **λ** 在 bias-variance tradeoff 中做平衡：λ=0 时等价于 1-step TD（高偏差），λ=1 时等价于 Monte Carlo（高方差）
- 本项目默认 γ=0.99, λ=0.95，是 PPO 论文中推荐的典型值
