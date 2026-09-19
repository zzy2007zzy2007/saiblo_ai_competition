# 模块3：PPO 训练引擎

## 1. 概述

### 1.1 模块定位

PPO 训练引擎是强化学习算法的 **核心实现模块**。它负责从原始游戏交互数据中提取训练信号，通过 PPO-Clip 算法更新神经网络参数，使策略逐步优化。该模块涵盖了完整的数据收集、GAE 优势估计、多 epoch/minibatch 训练、辅助任务学习、学习率调度和 NaN 鲁棒性保障。

### 1.2 在整体架构中的位置

```
模块4: 自对弈编排
    │
    │ 调用 ppo_update(), select_action()
    ▼
┌─────────────────────────────────────────┐
│           模块3: PPO 训练引擎              │
│                                         │
│  EpisodeCollector ──► EpisodeBatch      │
│        │                    │           │
│        ▼                    ▼           │
│  AntWarEnv (模块1)    PPOTrainer        │
│                       │     │           │
│                       ▼     ▼           │
│                 Network (模块2)          │
│                 GAE / Aux Labels        │
│                 LR Scheduler            │
│                 NaN Recovery            │
│                 Checkpoint Manager      │
└─────────────────────────────────────────┘
    │
    │ 输出: 更新后的模型参数、训练指标
    ▼
模块7: 监控与可观测性
```

### 1.3 业务目标

- 实现 PPO-Clip 算法完成策略参数更新
- 通过 GAE 高效估计优势函数，降低策略梯度方差
- 管理完整的数据收集管线（游戏交互 → 批次构建 → 张量转换）
- 保障训练数值稳定性（NaN 检测/清理/恢复）
- 提供灵活的检查点管理和学习率调度

---

## 2. 背景与概念

### 2.1 PPO-Clip 算法简介

PPO（Proximal Policy Optimization）是一种策略梯度方法，其核心思想是限制每次更新的策略变化幅度。PPO-Clip 的目标函数为：

```
L(θ) = E[min(ratio(θ) × A, clip(ratio(θ), 1-ε, 1+ε) × A)]
```

其中：
- `ratio(θ) = π_θ(a|s) / π_old(a|s)` — 新旧策略的概率比
- `A` — 优势函数（Advantage）
- `ε` — 裁剪范围（clip_epsilon，通常为 0.2）
- `clip(ratio, 1-ε, 1+ε)` — 将 ratio 限制在 [1-ε, 1+ε] 范围内

### 2.2 GAE（广义优势估计）

GAE 通过指数加权平均多步 TD 误差来估计优势函数：

```
A_t = Σ (γλ)^k × δ_{t+k}
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
```

- `γ`（gamma）：折扣因子
- `λ`（lambda）：GAE 参数，控制偏差-方差权衡（λ=0 为 TD(0)，λ=1 为 Monte Carlo）

### 2.3 双优化器架构

策略网络和价值网络使用 **独立优化器** + **独立梯度裁剪阈值**：

- Policy Optimizer：管理 Policy 编码器 + ActionHead + AuxHeads 的参数
- Value Optimizer：管理 Value 编码器 + ValueHead 的参数

参数通过参数名前缀区分（`VALUE_PREFIXES` = `['value_']` 匹配以 `value_` 开头的参数）。

### 2.4 Mini-batch 训练

一次 PPO update 可以使用多个 episode 的数据。数据先 concat 为一个大 batch，然后：
1. 将数据切分成多个 minibatch（batch_size = ppo_batch_size）
2. 每个 epoch 随机打散 minibatch
3. 逐 minibatch 计算 loss 和梯度更新

---

## 3. 架构设计

### 3.1 模块内部结构

```
┌───────────────────────────────────────────────────────┐
│                    PPO 训练引擎                         │
│                                                       │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │EpisodeCollector│ │PPOAgent  │  │   PPOTrainer     │ │
│  │ (数据收集)    │  │(推理代理) │  │   (算法核心)      │ │
│  └──────┬───────┘  └────┬─────┘  └────────┬─────────┘ │
│         │               │                  │           │
│         │   ┌───────────┴──────┐           │           │
│         │   │                  │           │           │
│         ▼   ▼                  ▼           ▼           │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │EpisodeBatch │  │Network(模块2)│  │NaN Recovery    │ │
│  │(数据容器)   │  │              │  │(安全防护)       │ │
│  └────────────┘  └──────────────┘  └────────────────┘ │
│                                                       │
│  ┌──────────────────┐  ┌──────────────┐              │
│  │LR Scheduler      │  │Checkpoint Mgr│              │
│  │(学习率调度)       │  │(检查点管理)   │              │
│  └──────────────────┘  └──────────────┘              │
│                                                       │
│  工具函数:                                             │
│  ┌──────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │gae.py    │  │aux_labels.py  │  │obs_utils.py   │  │
│  │(GAE计算) │  │(辅助标签计算)  │  │(观测转张量)   │  │
│  └──────────┘  └───────────────┘  └───────────────┘  │
└───────────────────────────────────────────────────────┘
```

### 3.2 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| `PPOTrainer` | [trainer/ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py) | PPO 核心算法：update、loss 计算、梯度管理 |
| `EpisodeCollector` | [trainer/episode_collector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/episode_collector.py) | 单局对战数据收集，快照采集，辅助标签 |
| `PPOAgent` | [trainer/agent.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/agent.py) | 策略网络推理代理 |
| `EpisodeBatch` | [trainer/batch.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/batch.py) | 对战数据批次容器 |
| `NanRecoveryHandler` | [trainer/safety.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/safety.py) | NaN 检测/清理/恢复 |
| `LRScheduler` | [trainer/lr_scheduler.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/lr_scheduler.py) | 学习率调度 |
| `CheckpointManager` | [trainer/checkpoint_manager.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/checkpoint_manager.py) | 检查点保存/加载 |
| `MetricsSchema` | [trainer/metrics_schema.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/metrics_schema.py) | 训练指标定义 |
| `compute_gae` | [utils/gae.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/gae.py) | GAE 优势估计 |
| `compute_aux_labels_from_trajectory` | [utils/aux_labels.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/aux_labels.py) | 辅助任务标签计算 |
| `observation_to_tensors` | [utils/obs_utils.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/obs_utils.py) | 观测 numpy → PyTorch 张量 |

---

## 4. 核心组件详解

### 4.1 PPOTrainer — PPO 算法核心

位置：[trainer/ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py)

#### 初始化

```
PPOTrainer(env_factory, config, device, base_dir, callbacks=None)
```

初始化流程：
1. 根据配置构建 `AntWarPolicyValueNetwork`
2. 创建双优化器（按参数名前缀分组：`value_` 前缀 → Value Optimizer，其余 → Policy Optimizer）
3. 初始化日志、`NanRecoveryHandler`、`CheckpointManager`、`LRScheduler`

#### 关键公开方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `select_action(obs, deterministic=False)` | `-> (action_id, log_prob, value)` | 单步动作选择（含 NaN fallback） |
| `ppo_update(batch, episode)` | `-> Dict[str, Any]` | 主训练入口：执行完整 PPO 更新 |
| `save_checkpoint(filepath, episode, metrics)` | `-> None` | 保存检查点（委托 CheckpointManager） |
| `load_checkpoint(filepath)` | `-> int` | 加载检查点，返回 episode 编号 |
| `get_agent(deterministic=False)` | `-> PPOAgent` | 获取推理代理 |
| `get_learning_rate()` | `-> (float, float)` | 当前 (policy_lr, value_lr) |
| `handle_nan_recovery(episode)` | `-> bool` | NaN 恢复；False = 应停止训练 |
| `update_lr_schedule(episode, total_episodes)` | `-> None` | 更新学习率 |
| `check_batch_quality(batch)` | `-> (bool, List[str])` | 检查批次数据质量 |

#### ppo_update() 内部流程

```
ppo_update(batch, episode)
  │
  ├── 1. batch.to_tensors(device)          # numpy → torch
  ├── 2. _compute_gae()                     # GAE 优势估计
  │       └── compute_gae() → advantages, returns
  │       └── clip advantages
  ├── 3. _compute_reward_stats()            # 奖励统计（均值/方差/最值）
  ├── 4. _compute_type_metrics()            # 各动作类型统计
  ├── 5. _train_ppo_epochs()                # 多 epoch/minibatch 训练
  │       ├── 打散 → minibatch → evaluate_actions()
  │       ├── _compute_losses() → PPO-Clip + value clip + entropy
  │       ├── compute_auxiliary_loss() → aux losses
  │       ├── _backward_and_clip() → backward + grad clip + step
  │       └── 基于 approx_kl 的 epoch 级早停
  ├── 6. _nan_handler.save_good_state()     # 保存健康权重
  └── 7. _aggregate_metrics()               # 聚合所有指标
```

#### 损失计算（_compute_losses）

**策略损失（PPO-Clip）**：
```python
ratio = exp(log_ratio)                    # π_new / π_old
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
surr1 = ratio * advantages
surr2 = clipped_ratio * advantages
policy_loss = -min(surr1, surr2).mean()
```

**价值损失（Clipped Value Loss）**：
```python
value_pred_clipped = value_old + clamp(value_pred - value_old, -ε_vf, +ε_vf)
loss_unclipped = (value_pred - returns)^2
loss_clipped = (value_pred_clipped - returns)^2
value_loss = 0.5 * max(loss_unclipped, loss_clipped).mean()
```

**熵正则化**：
```python
entropy_bonus = entropy_coef * entropy.mean()
```

**总损失**：
```python
total_loss = policy_loss + value_loss_coef * value_loss - entropy_bonus + aux_loss_weight * aux_loss
```

---

### 4.2 EpisodeCollector — 单局数据收集器

位置：[trainer/episode_collector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/episode_collector.py)

#### 初始化

```
EpisodeCollector(config: Dict, episode_count_ref: Callable[[], int])
```

`episode_count_ref` 是获取当前 episode 编号的回调函数，用于日志记录。

#### 核心方法

```
collect(env, opponent_agent, opponent_id, swap_positions, select_action_fn)
    -> (EpisodeBatch, battle_details)
```

**内部流程**：

1. `env.reset()` 获取初始观测
2. 循环（直到 max_steps 或 terminated）：
   - 己方动作：`select_action_fn(obs_self)` → action_id
   - 对手动作：若有 opponent_agent 则 `opponent_agent.act(obs_opponent)`，否则从合法掩码随机
   - `env.step(action)` 执行动作
   - 记录观测/动作/奖励/价值/log_prob 到 EpisodeBatch
   - 采集状态快照（`_capture_snapshot`）
   - 更新奖励来源统计和动作统计
3. `_attach_aux_labels()` 计算辅助标签并填充到 batch
4. `_build_battle_details()` 构建对局详情字典

#### 辅助方法

| 方法 | 说明 |
|------|------|
| `compute_max_coin(snapshots, player)` | 对局中最大金币余额 |
| `compute_total_coin_income(snapshots, player)` | 总正向金币增量 |
| `_capture_snapshot(state, round_idx)` | 采集单步状态快照 |
| `_update_reward_sources()` | 追踪 8 种奖励来源的累计值 |
| `_update_action_stats()` | 统计各种动作类型的使用频率和奖励 |
| `_attach_aux_labels()` | 计算并填充辅助任务标签 |

---

### 4.3 EpisodeBatch — 数据批次容器

位置：[trainer/batch.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/batch.py)

#### 数据结构

内部维护时间步列表：
- `observations_board`, `observations_global`, `observations_mask` — 各步观测
- `actions` — 各步动作 ID
- `rewards`, `values`, `log_probs`, `dones` — 各步奖励/价值/log_prob/终止标志
- `aux_tower_damage`, `aux_gold_income`, `aux_valid_mask` — 可选辅助标签

#### 关键方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `add_step(obs, action, reward, value, log_prob, done)` | — | 追加一个时间步 |
| `to_tensors(device)` | `-> Dict[str, Tensor]` | numpy → PyTorch 张量转换 |
| `merge(batches)` | `-> EpisodeBatch` (静态) | 拼接多个 EpisodeBatch |

`to_tensors()` 输出格式：
- 观测三件套：`(T, 28, 19, 19)`, `(T, 33)`, `(T, 96)` — float32
- actions：`(T,)` — int64
- rewards / values / log_probs / dones：`(T,)` — float32
- aux 字段：`(T, 10)` — float32（可选）

---

### 4.4 PPOAgent — 推理代理

位置：[trainer/agent.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/agent.py)

封装策略网络的推理接口。

| 方法 | 返回 | 说明 |
|------|------|------|
| `act(observation, deterministic=False)` | `int` | 快速动作选择，只返回 action_id |
| `get_action_values(observation, deterministic=False)` | `(action_id, log_prob, value)` | 完整动作选择（训练数据收集用） |
| `get_value(observation)` | `float` | 获取状态价值（不选择动作） |

内部通过 `observation_to_tensors()` 将 numpy 观测转为 batch=1 张量，调用 `network.get_action()` 或 `network.forward()`。

---

### 4.5 NanRecoveryHandler — NaN 恢复处理

位置：[trainer/safety.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/safety.py)

#### 工具函数

| 函数 | 说明 |
|------|------|
| `check_tensor_nan(tensor, name="") -> bool` | 检查张量是否含 NaN/Inf |
| `clean_tensor(tensor) -> Tensor` | `nan_to_num(nan=0, posinf=1e4, neginf=-1e4)` |

#### 核心类

```
NanRecoveryHandler(nan_threshold: int)
```

| 方法 | 说明 |
|------|------|
| `save_good_state(policy, optimizer, value_optimizer)` | 保存最近一次健康权重快照 |
| `on_nan_detected(episode, policy, optimizer, value_optimizer) -> bool` | NaN 计数+1；超阈值返回 False（停止训练）；否则从快照恢复，返回 True |

**恢复机制**：每次 PPO update 成功后保存权重快照。当检测到 NaN 时，回滚到最近一次健康检查点并重新开始训练。

---

### 4.6 LRScheduler — 学习率调度器

位置：[trainer/lr_scheduler.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/lr_scheduler.py)

#### 配置

```
LRScheduleConfig:
  - lr_base: float          # 策略网络基础学习率
  - lr_vf_base: float       # 价值网络基础学习率
  - warmup_episodes: int    # 线性 warmup 的 episode 数
  - cosine_decay: bool      # 是否启用余弦衰减
```

#### 调度策略（优先级递减）

1. **Warmup**：episode < warmup_episodes 时，学习率从 0 线性上升到 base
2. **Cosine Decay**：cosine_decay=True 时，学习率从 base 余弦衰减到 base × min_lr_factor
3. **Fixed**：以上都不满足时，使用固定学习率

---

### 4.7 CheckpointManager — 检查点管理器

位置：[trainer/checkpoint_manager.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/checkpoint_manager.py)

| 方法 | 说明 |
|------|------|
| `save(filepath, episode, policy_state, optimizer_state, value_optimizer_state, metrics, config)` | 保存完整检查点 |
| `load(filepath, policy, optimizer, value_optimizer) -> int` | 加载检查点，返回 episode |

保存内容：`episode`、三个 `state_dict`（policy/optimizer/value_optimizer）、`metrics`、`config`。

---

### 4.8 工具函数

#### compute_gae（[utils/gae.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/gae.py)）

```
compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95, device)
    -> (advantages, returns)
```

从后向前计算：`δ = r + γ × V_next × (1-done) - V`，`GAE = δ + γλ × (1-done) × prev_GAE`。

#### compute_aux_labels_from_trajectory（[utils/aux_labels.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/aux_labels.py)）

```
compute_aux_labels_from_trajectory(snapshots)
    -> (tower_damage_labels: (T,10), gold_income_labels: (T,10))
```

从轨迹快照计算 5 个未来视界（1, 2, 4, 8, 16 步）的塔 HP 损伤和金币净变化。标签 = `initial - future`（非负）。

#### observation_to_tensors（[utils/obs_utils.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/obs_utils.py)）

```
observation_to_tensors(observation, device)
    -> Dict[str, Tensor]
```

numpy 观测 → `(1, C, H, W)` / `(1, D)` / `(1, A)` 张量。

---

## 5. 数据流与时序

### 5.1 完整的 PPO 训练周期

```
EpisodeCollector.collect()
  │
  ├── 多局对战 → EpisodeBatch × N
  │
  ▼
EpisodeBatch.merge([batch1, batch2, ...])
  │
  ▼
PPOTrainer.ppo_update(merged_batch, episode)
  │
  ├── batch.to_tensors(device)
  │     └── boards, globals, masks, actions, rewards, values, log_probs, dones
  │
  ├── compute_gae(rewards, values, dones)
  │     └── advantages (T,), returns (T,)
  │
  ├── advantages = clip(advantages, -adv_clip, +adv_clip)
  │     └── advantages = normalize(advantages)  # 零均值单位方差
  │
  ├── for epoch in range(ppo_epochs):
  │     │
  │     ├── shuffled_indices = randperm(T)
  │     │
  │     ├── for each minibatch (size = ppo_batch_size):
  │     │     │
  │     │     ├── 选取 minibatch 数据
  │     │     ├── evaluate_actions() → new_log_probs, values, entropy
  │     │     ├── _compute_losses() → policy_loss, value_loss, entropy_bonus
  │     │     ├── compute_auxiliary_loss() → aux losses
  │     │     ├── total_loss = policy_loss + vf_coef×value_loss - ent_coef×entropy + aux_weight×aux_loss
  │     │     ├── _backward_and_clip() → backward + grad_nan_check + clip_grad + step
  │     │     └── 收集 step 指标
  │     │
  │     └── 检查 approx_kl > target_kl → 早停
  │
  ├── _nan_handler.save_good_state()
  └── _aggregate_metrics() → 33+ 维指标字典
```

### 5.2 指标维度

每次 `ppo_update()` 返回 33+ 维的指标字典，包括：

| 类别 | 指标示例 |
|------|---------|
| **损失** | `policy_loss`, `value_loss`, `entropy_loss`, `total_loss` |
| **辅助损失** | `aux_tower_loss`, `aux_enemy_tower_loss`, `aux_gold_loss`, `aux_enemy_gold_loss` |
| **熵** | `entropy`, `type_entropy`, `target_entropy` |
| **KL** | `approx_kl`, `clip_fraction` |
| **梯度** | `policy_grad_norm`, `value_grad_norm` |
| **比率** | `ratio_mean`, `ratio_min`, `ratio_max` |
| **奖励** | `reward_mean`, `reward_std`, `reward_min`, `reward_max` |
| **学习率** | `policy_lr`, `value_lr` |
| **动作** | 各类动作类型的比例、合法比例、logit 均值等 |

---

## 6. 对外交互

### 6.1 被依赖关系

| 调用方 | 场景 | 使用方法 |
|--------|------|---------|
| **SelfPlayTrainer** (模块4) | 触发 PPO 更新 | `trainer.ppo_update(batch, episode)` |
| **SelfPlayTrainer** (模块4) | 数据收集动作选择 | `trainer.select_action(obs)` |
| **SelfPlayTrainer** (模块4) | 周期性任务 | `save_checkpoint()`, `load_checkpoint()`, `update_lr_schedule()`, `get_agent()` |
| **EpisodeCollector** (模块3内部) | 数据收集 | `trainer.select_action(obs)` |

### 6.2 对下游模块的依赖

| 依赖模块 | 使用方式 |
|---------|---------|
| **模块1（游戏环境）** | EpisodeCollector 通过 AntWarEnv 驱动游戏交互 |
| **模块2（神经网络）** | PPOTrainer 持有 `AntWarPolicyValueNetwork`，调用其推理和训练方法 |
| **模块8（基础设施）** | 通过配置和路径管理获取训练参数和输出路径 |

### 6.3 对模块7（监控）的间接调用

训练指标通过 `SelfPlayTrainer` 传递给监控模块的各类 Writer（BatchMetricsWriter, EpisodeBatchWriter 等），不直接耦合。

---

## 7. 配置参数

### 7.1 PPO 算法参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `clip_epsilon` | PPO 裁剪范围 ε | 0.2 |
| `clip_epsilon_vf` | 价值函数裁剪范围 | 0.2 |
| `ppo_epochs` | 每次 update 的训练轮数 | 4-10 |
| `ppo_batch_size` | minibatch 大小 | 512-2048 |
| `gamma` | 奖励折扣因子 | 0.99 |
| `gae_lambda` | GAE λ 参数 | 0.95 |
| `entropy_coef` | 熵正则化系数 | 0.01 |
| `value_loss_coef` | 价值损失系数 | 0.5 |
| `aux_loss_weight` | 辅助损失权重 | 0.1-1.0 |
| `max_grad_norm` | 梯度裁剪阈值 | 0.5 |
| `target_kl` | 近似 KL 早停阈值 | 0.01-0.03 |
| `adv_clip` | 优势值裁剪范围 | 10.0 |

### 7.2 学习率参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `lr_base` | 策略网络基础学习率 | 3e-4 |
| `lr_vf_base` | 价值网络基础学习率 | 1e-3 |
| `warmup_episodes` | 学习率 warmup 步数 | 100-500 |
| `cosine_decay` | 是否启用余弦衰减 | True |
| `lr_min_factor` | 余弦衰减的最低倍率 | 0.01-0.1 |

### 7.3 数据收集参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `max_steps` | 单局最大步数 | 512 |
| `n_steps` | 每次 update 的数据量 | 2048-4096 |
| `num_minibatches` | minibatch 数量 | 4-8 |

---

## 8. 常见问题与注意事项

### 8.1 NaN 问题的常见原因

1. **学习率过高**：梯度爆炸导致参数变为 NaN
2. **数值不稳定**：softmax 中 `exp(-inf)→0, log(0)→-inf` 的链式反应
3. **奖励范围过大**：未裁剪的极端奖励值导致优势值异常

**应对策略**：
- 使用 -1e9 替代 -inf 作为 mask fill value
- 奖励裁剪到 [-step_reward_clip, step_reward_clip]
- 梯度检测 + NaN 恢复机制
- 双优化器独立梯度裁剪

### 8.2 为什么需要双优化器？

策略网络和价值网络的最优学习率通常不同。共享优化器时，一个子网络的梯度可能主导更新。双优化器允许为 Policy 和 Value 设置独立的学习率和梯度裁剪阈值。

### 8.3 GAE 裁剪

`compute_gae()` 本身不包含裁剪。裁剪在 `PPOTrainer._compute_gae()` 中额外进行：`advantages = clip(advantages, -adv_clip, +adv_clip)` 然后标准化。

### 8.4 辅助标签有效性检查

`compute_aux_labels_from_trajectory()` 返回的标签数组长度 = `T - max_horizon`（因为未来 16 步的标签在末尾 T-16 步后无法计算）。当 `T <= max_horizon` 时返回空数组。EpisodeCollector 负责处理这种情况。

### 8.5 batch 质量检查

`check_batch_quality()` 在 PPO update 前检查：
- rewards 是否含 NaN/Inf
- values 是否含 NaN/Inf
- rewards/values 是否全零
- 数据长度是否足够

不通过质检的 batch 将被丢弃，跳过本次 update。
