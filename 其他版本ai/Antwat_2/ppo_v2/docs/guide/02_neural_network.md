# 模块2：神经网络模型

## 1. 概述

### 1.1 模块定位

神经网络模型模块定义了 PPO 训练所需的 **完整策略-价值网络架构**。它负责将游戏观测（棋盘图 + 全局向量 + 动作掩码）映射为动作概率分布和状态价值估计，同时提供动作采样、动作评估和辅助任务预测功能。

### 1.2 在整体架构中的位置

```
┌──────────────────────────────────────┐
│              模块2: 神经网络模型        │
│                                      │
│  ┌────────────────────────────┐     │
│  │ AntWarPolicyValueNetwork    │     │
│  │                             │     │
│  │  ┌─────────┐ ┌───────────┐ │     │
│  │  │Policy   │ │Value      │ │     │
│  │  │Encoder  │ │Encoder    │ │     │
│  │  └────┬────┘ └─────┬─────┘ │     │
│  │       │            │       │     │
│  │  ┌────▼────────────▼─────┐ │     │
│  │  │  StructuredActionHead │ │     │
│  │  │  ValueHead            │ │     │
│  │  │  TowerDamageHead      │ │     │
│  │  │  GoldIncomeHead       │ │     │
│  │  └───────────────────────┘ │     │
│  └────────────────────────────┘     │
└──────────────────────────────────────┘
```

该模块被以下模块依赖：
- **模块3（PPO 训练引擎）**：`PPOTrainer` 使用网络进行推理和前向/反向传播
- **模块4（自对弈编排）**：通过 `PPOAgent` 和 `OpponentAgent` 使用网络进行对战
- **模块5（对战评估）**：`OpponentAgent` 使用网络进行基线评估

### 1.3 业务目标

- 设计适合六边形棋盘的卷积神经网络编码器
- 支持 96 维结构化的分层动作空间（类型 → 目标）
- 提供独立双编码器架构，解耦策略学习与价值学习
- 通过辅助任务（未来塔伤害/金币预测）加速特征学习

---

## 2. 背景与概念

### 2.1 策略-价值网络（Actor-Critic）

PPO 是一种 Actor-Critic 方法，需要同时维护：

- **策略网络（Actor）**：输入观测，输出动作的概率分布 π(a|s)
- **价值网络（Critic）**：输入观测，输出状态价值 V(s) 的估计

本模块采用 **独立双编码器架构**：Policy 和 Value 各自拥有独立的 CNN + MLP 编码器，仅共享观测输入。这避免了价值梯度淹没策略梯度的问题（常见于共享编码器的架构中）。

### 2.2 六边形卷积（HexConv）

标准正方形卷积（Conv2d）假设相邻像素在上下左右方向，不适用于六边形网格。`HexConv` 通过 **6 个方向的邻域提取 + einsum 线性组合** 实现六边形邻域的卷积操作。

六方向定义：
```
  (-1, 1)  (0, 1)
         \   /
(-1, 0) — o — (1, 0)
         /   \
  (0, -1)  (1, -1)
```

### 2.3 结构化动作空间

96 维扁平动作空间是 9 种动作类型 × 各类型目标数的并集。为了避免 96 维 softmax 的稀疏性问题，网络采用 **分层架构**：

1. **类型头（Type Head）**：输出 9 维 logits，用于选择动作类型
2. **目标头（Target Heads）**：9 个独立子头，各输出其类型下的合法目标 logits

实际推理时先采样类型（9 维 → categorical），再在选中类型下采样目标（k 维 → categorical），等价于在整个 96 维空间上采样。

### 2.4 辅助任务学习

除了主任务（策略学习 + 价值学习），网络还训练预测未来 5 个时间视界（{1, 2, 4, 8, 16} 步后）的：

- **塔伤害**：己方和敌方塔的 HP 减少量
- **金币收入**：己方和敌方金币的净变化

这些辅助任务通过 MSE 损失联合训练，目的是让共享特征提取器学会更有远见的表示。

---

## 3. 架构设计

### 3.1 网络整体架构

```
Board (B, 28, 19, 19) ────────────────────┐
                                            │
Global Vec (B, 33) ────────────────────────┤
                                            │
    ┌───────────────────────────────────────┤
    │                                       │
    ▼ (Policy Encoder)                     ▼ (Value Encoder)
┌───────────┐                         ┌───────────┐
│HexCNN     │ 28→128→256→512         │HexCNN     │
│Encoder    │ → AdaptivePool(5,5)    │Encoder    │
│           │ → Flatten → 12800      │           │
└─────┬─────┘                         └─────┬─────┘
      │                                     │
┌─────▼─────┐                         ┌─────▼─────┐
│MLP        │ 33→128→128             │MLP        │
│Encoder    │                         │Encoder    │
└─────┬─────┘                         └─────┬─────┘
      │                                     │
      ▼                                     ▼
  12800 + 128 = 12928                  12800 + 128 = 12928
      │                                     │
      ▼ (Linear + LayerNorm)                ▼ (Linear + LayerNorm)
    (B, 256)  policy_features             (B, 256)  value_features
      │                                     │
      ├──► StructuredActionHead ──► (B, 96) │
      ├──► TowerDamageHead ───────► (B, 10) │
      └──► GoldIncomeHead ────────► (B, 10) │
                                            │
                                            └──► ValueHead ──► (B, 1)
```

### 3.2 组件树

| 组件 | 输入 | 输出 | 参数 |
|------|------|------|------|
| `HexCNNEncoder` | `(B, 28, 19, 19)` | `(B, 12800)` | 3阶段 CNN + HexConv + skip |
| `MLPEncoder` | `(B, 33)` | `(B, 128)` | 2层 Linear+ReLU (33→128→128) |
| `Policy Proj` | `(B, 12928)` | `(B, 256)` | Linear + LayerNorm |
| `Value Proj` | `(B, 12928)` | `(B, 256)` | Linear + LayerNorm |
| `StructuredActionHead` | `(B, 256)` | `(B, 96)` | 1×Linear(9) + 9×Linear(target_count) |
| `ValueHead` | `(B, 256)` | `(B, 1)` | 1×Linear |
| `TowerDamageHead` | `(B, 256)` | `(B, 10)` | 256→64→ReLU→10 |
| `GoldIncomeHead` | `(B, 256)` | `(B, 10)` | 256→64→ReLU→10 |

---

## 4. 核心组件详解

### 4.1 HexConv — 六边形卷积层

位置：[network/antwar_net.py#L38-L83](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/network/antwar_net.py#L38)

```
HexConv(in_channels: int, out_channels: int)
```

六方向邻域定义：`(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)`

**前向传播**：
1. `get_hex_neighbor_sum(x)` 对输入的每个像素沿六个方向做 `torch.roll` 提取邻域值，堆叠成 `(B, C, 6, H, W)`
2. 用可学习权重 `(out_c, in_c, 6)` 通过 `einsum("ocn,bcnhw->bohw")` 做线性组合
3. 加偏置

权重初始化：`kaiming_uniform_`，偏置初始化：`uniform_(-0.1, 0.1)`

### 4.2 HexCNNEncoder — 六边形卷积编码器

位置：[network/antwar_net.py#L100-L145](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/network/antwar_net.py#L100)

```
HexCNNEncoder(hidden_dim: int = 256)
```

**三个阶段**（每阶段 = Conv2d → BatchNorm → ReLU → HexConv + 残差连接）：

| 阶段 | 卷积 | 输出通道 | HexConv 通道 | stride |
|------|------|---------|-------------|--------|
| Stage 1 | Conv2d(28, 128, k1) | 128 | 128 | 1 |
| Stage 2 | Conv2d(128, 256, k3) | 256 | 256 | 2 |
| Stage 3 | Conv2d(256, 512, k3) | 512 | 512 | 2 |

最终通过 `AdaptiveAvgPool2d(5,5)` 池化到 5×5，展平得到 `(B, 12800)`。

关键设计：
- 每阶段 HexConv 输出以残差方式加到标准 CNN 输出上
- stride=2 进行逐步下采样（19×19 → 10×10 → 5×5）

### 4.3 MLPEncoder — 全局特征编码器

位置：[network/antwar_net.py#L148-L163](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/network/antwar_net.py#L148)

```
MLPEncoder(input_dim: int = 33, hidden_dim: int = 128)
```

简单的两层 MLP：`Linear(33, 128) → ReLU → Linear(128, 128) → ReLU`

输出 `(B, 128)` 维特征向量。

### 4.4 AntWarPolicyValueNetwork — 策略-价值网络

位置：[network/antwar_net.py#L273](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/network/antwar_net.py#L273)

#### 初始化

```
AntWarPolicyValueNetwork(hidden_dim: int = 256, enable_auxiliary: bool = True)
```

- `hidden_dim`：编码器融合后的隐藏维度（默认 256）
- `enable_auxiliary`：是否启用辅助预测头（默认 True）

初始化时创建：
- Policy 编码器（HexCNNEncoder + MLPEncoder + proj + LayerNorm）
- Value 编码器（HexCNNEncoder + MLPEncoder + proj + LayerNorm）
- 输出头（action_head, value_head, tower_damage_head, gold_income_head）

#### forward() — 标准前向传播

```
forward(board, global_vec, action_mask=None) -> (action_logits, value)
```

用于推理/评估。动作 logits 中非法动作被掩码为 `-1e9`（而非 `-inf`，避免下游计算 NaN）。

**数据流**：
```
board ──► policy_cnn ──┐
                       ├──► concat ──► policy_proj ──► LN ──► action_head ──► logits
global ──► policy_mlp ─┘

board ──► value_cnn ──┐
                      ├──► concat ──► value_proj ──► LN ──► value_head ──► value
global ──► value_mlp ─┘
```

#### get_action() — 动作采样

```
get_action(board, global_vec, action_mask, deterministic=False, exploration_epsilon=0.05)
    -> (action_id: int, log_prob: Tensor, value: Tensor, type_probs: Tensor)
```

**分层采样流程**：

1. **类型选择**：
   - 非确定性模式：计算 9 维 softmax 概率（NaN-safe），multinomial 采样
   - 确定性模式：argmax
   - 探索模式（概率 epsilon）：从全部合法动作中随机选择

2. **目标选择**：
   - 根据选中的类型，截取其目标 logits
   - 应用目标掩码，softmax → multinomial（非确定性）或 argmax（确定性）

3. **log_prob 计算**：
   - `log_prob = log P(type) + log P(target | type)`
   - 对 NaN/inf 进行安全处理（nan→0, inf 裁剪到 ±20）

#### evaluate_actions() — 动作评估（用于 PPO 更新）

```
evaluate_actions(board, global_vec, action_mask, action)
    -> (action_log_probs, values, entropy, type_entropy, target_entropy, aux_predictions, policy_features)
```

输入一批已执行的动作，重新计算其 `log_prob`、当前状态价值和熵分量。这是 PPO 更新循环中的关键方法——用更新后的网络参数重新评估旧动作。

**熵计算**：
- `type_entropy = -Σ P(type) log P(type)` — 类型级别的策略熵
- `target_entropy = Σ P(type) * (-Σ P(target|type) log P(target|type))` — 按类型概率加权的目标级熵
- `entropy = type_entropy + target_entropy`

#### compute_auxiliary_loss() — 辅助损失计算

```
compute_auxiliary_loss(features, aux_tower_damage, aux_gold_income)
    -> Dict[str, Tensor]
```

将 10 维预测向量拆分为两半（各 5 视界），分别计算己方/敌方的 MSE 损失。

| 损失项 | 含义 | 切片 |
|--------|------|------|
| `aux_tower_loss` | 己方塔伤害预测 MSE | `pred[:, :5]` vs `label[:, :5]` |
| `aux_enemy_tower_loss` | 敌方塔伤害预测 MSE | `pred[:, 5:]` vs `label[:, 5:]` |
| `aux_gold_loss` | 己方金币收入预测 MSE | `pred[:, :5]` vs `label[:, :5]` |
| `aux_enemy_gold_loss` | 敌方金币收入预测 MSE | `pred[:, 5:]` vs `label[:, 5:]` |

---

## 5. 数据流与时序

### 5.1 推理时数据流（训练数据收集）

```
ObservationEncoder.encode()            # 模块1
    │
    ├── board (28, 19, 19) numpy
    ├── global_vec (33,) numpy
    └── action_mask (96,) numpy
    │
    ▼
observation_to_tensors()              # obs_utils.py
    │
    ├── board_t (1, 28, 19, 19)
    ├── global_t (1, 33)
    └── mask_t (1, 96)
    │
    ▼
PPOAgent.get_action_values(obs)        # agent.py
    │
    └──► network.get_action(board, global, mask)
            │
            ├── _encode_policy() → policy_features (1, 256)
            ├── _encode_value() → value_features (1, 256)
            ├── type_head → type_logits (1, 9) → type selection
            ├── target_head → target_logits (1, k) → target selection
            ├── value_head → value (1, 1)
            └──► return (action_id, log_prob, value, type_probs)
```

### 5.2 PPO 更新时数据流

```
EpisodeBatch.to_tensors()             # batch.py
    │
    ├── boards (T, 28, 19, 19)
    ├── globals (T, 33)
    └── masks (T, 96)
    │
    ▼
network.evaluate_actions(boards, globals, masks, actions)
    │
    ├── _encode_policy() → policy_features (T, 256)
    ├── _encode_value() → value_features (T, 256)
    │
    ├── type_head → type_logits → log_softmax → type_log_probs
    ├── 9× target_heads → 各类型 target logits → log_softmax → target_log_probs
    │
    ├── value_head → values (T,)
    │
    ├── tower_damage_head → tower_pred (T, 10)
    └── gold_income_head → gold_pred (T, 10)
    │
    ▼
(loss计算在 PPOTrainer._process_minibatch 中进行)
```

---

## 6. 对外交互

### 6.1 被依赖关系

| 调用方 | 场景 | 使用方法 |
|--------|------|---------|
| **PPOTrainer** (模块3) | PPO 更新 | `evaluate_actions()`, `forward()`, `compute_auxiliary_loss()` |
| **PPOAgent** (模块3) | 数据收集推理 | `get_action()`, `forward()` |
| **OpponentAgent** (模块5) | 对战评估 | `forward()` → 通过 PPOAgent 间接使用 |
| **EpisodeCollector** (模块3) | 数据收集 | 通过 PPOAgent 间接使用 |

### 6.2 对模块1（游戏环境）的隐式依赖

网络输入格式与 [ObservationEncoder](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/env/observation.py) 输出严格一致：

| 网络输入 | 形状 | 来源 |
|---------|------|------|
| `board` | `(B, 28, 19, 19)` | ObservationEncoder.encode() → board |
| `global_vec` | `(B, 33)` | ObservationEncoder.encode() → global_vec |
| `action_mask` | `(B, 96)` | ActionMaskHandler.get_action_mask() |

### 6.3 对模块3（utils）的依赖

网络引用了 [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/utils/action_constants.py) 中的常量：
- `TYPE_CONFIG`：动作类型定义（`flat_start`, `flat_end`, `max_targets`）
- `ACTION_DIM`：总动作维度（96）
- `NUM_HORIZONS`：辅助任务视界数量（5）
- `action_id_to_type_and_target()`：action_id → (type_id, target_k) 转换

---

## 7. 配置参数

### 7.1 网络结构参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dim` | `__init__` | 256 | 隐藏层维度，影响所有子模块通道数 |
| `enable_auxiliary` | `__init__` | True | 是否启用辅助预测头 |
| `BOARD_CHANNELS` | 模块常量 | 28 | 棋盘特征图通道数 |
| `BOARD_SIZE` | 模块常量 | 19 | 棋盘网格尺寸 |
| `GLOBAL_FEATURE_DIM` | 模块常量 | 33 | 全局特征向量维度 |
| `MLP_OUTPUT_DIM` | 模块常量 | 128 | MLP 编码器输出维度 |
| `AUX_HEAD_HIDDEN` | 模块常量 | 64 | 辅助头隐藏层维度 |
| `MASK_FILL_VALUE` | 模块常量 | -1e9 | 非法动作 logits 填充值 |

### 7.2 采样参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `deterministic` | `get_action()` | False | 确定性采样（argmax） |
| `exploration_epsilon` | `get_action()` | 0.05 | 随机探索概率 |

---

## 8. 常见问题与注意事项

### 8.1 为什么 Policy 和 Value 使用独立的编码器？

共享编码器时，价值网络的梯度（通常比策略网络梯度更大）会通过共享参数反向传播到编码器，干扰策略学习。独立编码器彻底消除了这种梯度干扰，是分布式强化学习中的常见最佳实践。

### 8.2 masked_fill 使用 -1e9 而非 -inf 的原因

`float('-inf')` 在后续操作中（如 softmax 中的 `exp(-inf) = 0`，log 中的 `log(0) = -inf`）容易产生 NaN。使用 `-1e9` 作为替代，在数值上等价（softmax 后概率 ≈ 0）但不会导致 NaN。

### 8.3 NaN-safe 概率重归一化

在 `get_action()` 和 `evaluate_actions()` 中，softmax 后的概率可能因数值问题出现 NaN 或全零。代码中有多层保护：
1. `torch.nan_to_num(probs, nan=0.0)` — 将 NaN 替换为 0
2. `probs / probs.sum()` with clamp min=1e-8 — 重归一化，防止除零
3. `probs.sum() < 0.5` fallback — 概率质量不足时回退到 action 0

### 8.4 辅助任务的设计动机

强化学习中，奖励信号通常稀疏且延迟。辅助预测任务（预测未来塔伤害和金币收入）提供即时的稠密监督信号，帮助编码器学习有远见的特征表示。这是 "Auxiliary Tasks" 范式在 RL 中的应用，类似 UNREAL 和 Agent57 的设计。

### 8.5 模型参数统计

`count_parameters(model)` 可统计可训练参数总数。策略和价值编码器各占一份完整的 HexCNNEncoder + MLPEncoder，因此参数总量约为基础编码器的两倍。
