# 策略网络模块 (Policy Network)

## 1. 背景与定位

### 1.1 业务问题

在 PPO 强化学习算法中，核心需要两个函数近似器：**策略 (Policy)** 和 **价值函数 (Value Function)**。策略决定智能体在给定状态下选择各动作的概率；价值函数估计从给定状态开始的期望累积回报。此外，为了提高样本效率和加速学习，实践中常常添加**辅助任务 (Auxiliary Tasks)**——让网络同时预测与主任务相关的其他目标（如塔伤害、金币收入），迫使网络学习更丰富的状态表征。

**策略网络模块**的核心业务目标是：**设计并实现一个适合 AntWar 游戏特点的神经网络架构**，将游戏环境模块输出的观测（board 特征图 + global 特征向量）映射为动作概率分布和状态价值估计，同时支持辅助任务预测。

### 1.2 在整个系统中的位置

```
游戏环境 (Environment)              策略网络模块                    PPO 训练引擎
      │                                  │                             │
      │  观测 (board+global+mask)         │                             │
      ├─────────────────────────────────▶ │                             │
      │                                  │  ├─ 前向传播                │
      │                                  │  │  forward() ────────▶ logits + value
      │                                  │  │                             │
      │                                  │  ├─ 动作选择                │
      │                                  │  │  get_action() ──────▶ action_id
      │                                  │  │                             │
      │                                  │  ├─ 动作评估                │
      │                                  │  │  evaluate_actions() ──▶ log_probs
      │                                  │  │      + entropy + aux_preds
      │                                  │  │                             │
      │                                  │  └─ 辅助损失计算            │
      │                                  │     compute_aux_losses() ──▶ │
      │                                  │                             │
      │  推理 (agent.py)                 │                             │
      │  ◀──────────────────────────────│                             │
```

- **上层**：PPO 训练引擎（`ppo_trainer.py`）在训练过程中调用 `evaluate_actions()` 计算策略损失和价值损失；`PPOAgent`（`agent.py`）在推理时调用 `get_action()` 选择动作。
- **同层依赖**：依赖基础设施模块中的 `action_constants`（动作空间常量）和 `aux_labels`（辅助任务视界常量）。
- **下层**：网络架构的设计紧密配合游戏环境模块输出的观测结构（board shape 28×19×19，global dim 33）。

### 1.3 与其他模块的关系

| 外部模块 | 关系 | 交互方式 |
|---------|------|---------|
| 基础设施 (`action_constants`) | 依赖 | 使用 `TYPE_CONFIG`（动作类型映射）、`ACTION_DIM`（动作空间维度）、`action_id_to_type_and_target`（ID 分解工具） |
| 基础设施 (`aux_labels`) | 依赖 | 使用 `NUM_HORIZONS` 常量（辅助任务的未来视界数） |
| PPO 训练引擎 (`ppo_trainer.py`) | 被依赖 | 调用 `forward()`/`evaluate_actions()`/`compute_auxiliary_loss()` |
| PPO 训练引擎 (`agent.py`) | 被依赖 | 调用 `get_action()` 进行推理 |

---

## 2. 核心概念

### 2.1 关键术语

| 术语 | 说明 |
|------|------|
| **Policy（策略）** | 从状态到动作概率分布的映射 $\pi(a\|s)$ |
| **Value Function（价值函数）** | 从状态到期望累积回报的估计 $V(s)$ |
| **Logits** | 未经过 Softmax 的原始网络输出分数 |
| **Structured Action Head** | 分层动作头，将 96 维动作空间分解为"类型选择 → 目标选择"两步 |
| **Auxiliary Task（辅助任务）** | 辅助预测任务（塔伤害、金币收入），不直接影响策略但促进特征学习 |
| **Hex Convolution** | 六边形卷积，使用 6 方向邻域替代标准 3×3 方形的卷积操作 |
| **GAE** | Generalized Advantage Estimation，优势估计方法（本模块不实现，在训练引擎模块中） |

### 2.2 数据流概要

```
训练阶段：
  board (B,28,19,19) ──▶ PolicyCNN ──▶ policy_features (B,256)
  global (B,33) ──────▶ PolicyMLP ──▶ 
                                            │
  board (B,28,19,19) ──▶ ValueCNN ──▶ value_features (B,256)
  global (B,33) ──────▶ ValueMLP ──▶ 
                                            │
  policy_features + action_mask ──▶ StructuredActionHead ──▶ action_logits (B,96)
  value_features ──────────────────▶ ValueHead ──────────────▶ value (B,1)
  features ────────────────────────▶ TowerDamageHead ────────▶ (B,10)
  features ────────────────────────▶ GoldIncomeHead ─────────▶ (B,10)

推理阶段：
  board (B,28,19,19) + global (B,33) + action_mask (B,96)
      └─▶ get_action() ──▶ (action_id, log_prob, value, type_probs)
```

---

## 3. 架构总览

### 3.1 模块组成

策略网络模块仅包含一个文件：

```
network/
  └── antwar_net.py        # AntWarPolicyValueNetwork 及所有子模块
```

### 3.2 组件关系

```
AntWarPolicyValueNetwork
│
├── 策略编码器 (policy_cnn + policy_mlp)
│   ├── HexCNNEncoder          # 处理 board 输入 (28,19,19) → 12800维
│   │   ├── HexConv (3层残差)   # 六边形卷积层
│   │   └── AdaptiveAvgPool2d  # 自适应池化 → 12800
│   │
│   └── MLPEncoder             # 处理 global 输入 (33,) → 128维
│       └── Linear × 2         # 33→128→128
│
├── 价值编码器 (value_cnn + value_mlp)
│   └── 与策略编码器结构对称但参数独立
│
├── StructuredActionHead       # 分层动作头
│   ├── type_logits (9类)      # 9 种动作类型
│   └── target_logits (各类型独立维度)
│
├── ValueHead                  # 价值头
│   └── Linear(256, 1)
│
├── TowerDamageHead (可选)     # 辅助头：塔伤害预测
│   └── Linear(256, 10)
│
└── GoldIncomeHead (可选)      # 辅助头：金币收入预测
    └── Linear(256, 10)
```

---

## 4. 核心组件详解

### 4.1 AntWarPolicyValueNetwork (`network/antwar_net.py`)

#### 4.1.1 职责

这是整个模块的核心类，封装了完整的策略-价值神经网络。主要职责：
- 接收观测输入（board + global + action_mask），输出动作 logits 和状态价值
- 支持训练模式下的动作评估（计算 log_probs、entropy、辅助预测）
- 支持推理模式下的动作选择（确定性或随机采样）
- 计算辅助任务的损失值

#### 4.1.2 构造函数

```python
def __init__(self, hidden_dim: int = 256, enable_auxiliary: bool = True)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dim` | int | 256 | 网络隐层维度 |
| `enable_auxiliary` | bool | True | 是否启用辅助任务头（塔伤害、金币收入预测） |

构造时创建以下子模块：
- `policy_cnn`: `HexCNNEncoder`（策略网络棋盘编码器）
- `policy_mlp`: `MLPEncoder`（策略网络全局特征编码器）
- `policy_proj`: `Linear(12800 + 128, hidden_dim)` 拼接投影层
- `policy_layer_norm`: `LayerNorm(hidden_dim)`
- `value_cnn`/`value_mlp`/`value_proj`/`value_layer_norm`: 上述结构的完整镜像（参数独立）
- `action_head`: `StructuredActionHead(hidden_dim)`
- `value_head`: `ValueHead(hidden_dim)`
- `tower_damage_head`/`gold_income_head`: 在 `enable_auxiliary=True` 时创建

#### 4.1.3 公开方法

| 方法 | 签名 | 功能 |
|------|------|------|
| `forward` | `(board, global_vec, action_mask) -> (action_logits, value)` | 标准前向传播。board 通过 cnn，global 通过 mlp，拼接后分叉为 action_head 和 value_head。action_logits 中被 mask 的位置填充为 `-1e9` |
| `get_action` | `(board, global_vec, action_mask, deterministic, exploration_epsilon) -> (action_id, log_prob, value, type_probs)` | **推理模式**。先选 type（9 类），再选 target。支持确定性（argmax）和随机采样。`exploration_epsilon` 控制随机探索概率 |
| `evaluate_actions` | `(board, global_vec, action_mask, action) -> (log_probs, values, entropy, type_entropy, target_entropy, aux_preds, features)` | **训练模式**。为给定的 `action` 计算 log_prob、entropy 和辅助头预测。返回值用于 PPO 损失计算 |
| `compute_auxiliary_loss` | `(features, aux_tower_damage, aux_gold_income) -> (loss, detail_dict)` | 计算辅助任务的 MSE 损失。包含 4 个分量：己方/敌方 × 塔伤害/金币收入 |

#### 4.1.4 关键算法：前向传播与双编码器

```
forward(board, global_vec, action_mask)
│
1. policy_feat = _encode_policy(board, global_vec)
   ├─ board_feat = policy_cnn(board)          # (B, 12800)
   ├─ global_feat = policy_mlp(global_vec)    # (B, 128)
   ├─ cat = concat([board_feat, global_feat]) # (B, 12928)
   └─ policy_feat = LayerNorm(Linear(cat))    # (B, 256)
│
2. value_feat = _encode_value(board, global_vec)  # 同上，但使用独立参数
│
3. action_logits = action_head(policy_feat, action_mask)
   ├─ type_logits = Linear(policy_feat)        # (B, 9)
   ├─ 计算每个 type 的 target_logits
   └─ logits = type_logits[:, t] + target_logits  → 组合为 (B, 96)
   └─ logits[action_mask == 0] = -1e9  # 掩码非法动作
│
4. value = value_head(value_feat)  # (B, 1)
│
5. return (action_logits, value)
```

**关键设计**：策略网络和价值网络使用**独立的编码器**（`_encode_policy` 和 `_encode_value`），而不是共享编码器后分叉。原因：价值网络的梯度更稳定，策略网络的梯度更稀疏，共享编码器会导致价值梯度淹没策略梯度。

#### 4.1.5 关键算法：结构化动作选择 (`get_action`)

`get_action` 实现了分层动作选择，避免了对 96 维 logits 直接做 softmax：

```
get_action(board, global_vec, action_mask, deterministic, epsilon)
│
1. policy_feat = _encode_policy(board, global_vec)
│
2. 计算 type_logits (9,) 和 target_logits (各类型不同维度)
│
3. 类型选择:
   ├─ type_mask = 对每个类型，检查其所有 target 是否全部被 mask
   ├─ type_probs = softmax(type_logits + type_mask)
   ├─ 如果 deterministic: type_id = argmax(type_probs)
   ├─ 否则: 按 type_probs 随机采样
   └─ 如果随机值 < epsilon: 从合法类型中均匀采样（epsilon-贪心）
│
4. 目标选择:
   ├─ 取出该 type 的 target_logits
   ├─ 应用 target_mask
   ├─ target_probs = softmax(target_logits)
   ├─ 类似地 deterministic/随机/epsilon 采样
   └─ target_id = chosen_target
│
5. action_id = 根据 TYPE_CONFIG 将 (type_id, target_id) 编码为扁平 ID
│
6. 计算 log_prob 和 value
│
7. return (action_id, log_prob, value, type_probs)
```

### 4.2 HexCNNEncoder

#### 4.2.1 职责

处理棋盘输入 `(B, 28, 19, 19)`，提取空间特征。采用 3 阶段下采样结构，其中关键创新是使用**六边形卷积 (HexConv)** 替代标准 3×3 方形卷积，以更好地适配游戏地图的六边形网格特性。

#### 4.2.2 架构

```
输入 (B, 28, 19, 19)
  │
  ├─ Conv2d(28, 64, 1) + LayerNorm + ReLU
  ├─ HexConv(64, 64) + LayerNorm + ReLU + 残差连接
  │
  ├─ Conv2d(64, 128, 3, stride=2) + LayerNorm + ReLU
  ├─ HexConv(128, 128) + LayerNorm + ReLU + 残差连接
  │
  ├─ Conv2d(128, 256, 3, stride=2) + LayerNorm + ReLU
  ├─ HexConv(256, 256) + LayerNorm + ReLU + 残差连接
  │
  └─ AdaptiveAvgPool2d(5, 5) → Flatten → (B, 256*5*5=6400)
```

对于 value 编码器，展平输出为 `6400`；对于 policy 编码器，额外拼接 64 维可学习位置编码后输出 `12800` 维。

#### 4.2.3 HexConv 实现

```python
# 六边形卷积：沿 6 个方向（轴向坐标）提取邻域特征
directions = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]

# 对每个方向 d:
shifted = torch.roll(x, shifts=(d[0], d[1]), dims=(-2, -1))
shifted[..., boundary_mask] = 0  # 边界置零

# 使用 einsum 实现卷积
output = einsum("ocn,bcnhw->bohw", weight, stacked_features)
```

### 4.3 MLPEncoder

#### 4.3.1 职责

处理全局特征向量 `(B, 33)`，将其映射到 128 维隐空间。

#### 4.3.2 架构

```
输入 (B, 33)
  ├─ Linear(33, 128) + ReLU
  └─ Linear(128, 128) + ReLU
输出 (B, 128)
```

### 4.4 StructuredActionHead

#### 4.4.1 职责

将动作空间分解为分层结构：先选动作**类型**（9 类），再选**目标**（每类对应的目标索引）。最终将 `type_logits + target_logits` 组合为 96 维扁平 logits。

#### 4.4.2 架构

每种动作类型 i 都有独立的 `target_linear[i]` 网络层：

```
type_logits = Linear(256, 9)                         # (B, 9)
target_logits_i = target_linear[i](policy_feat)       # (B, dim_i)
logits = type_logits[:, i: i+1] + target_logits_i     # 广播相加
```

这种设计的好处是：
- **类型先验**：类型选择比目标选择更重要，模型可以先学会"该不该建塔"再学"建在哪里"
- **参数效率**：9 个小线性层比一个 256×96 的大线性层更容易优化
- **目标维度不对称**：不同类型的 target 数不同（noop=1, build=10, upgrade=40 等），分层头自然支持变长输出

### 4.5 辅助任务头

#### 4.5.1 职责

两个辅助任务头从价值编码器的特征中预测辅助标签：
- **TowerDamageHead**: `Linear(256, 10)` — 预测 5 个未来视界的己方和敌方塔伤害
- **GoldIncomeHead**: `Linear(256, 10)` — 预测 5 个未来视界的己方和敌方金币收入

#### 4.5.2 损失计算

`compute_auxiliary_loss` 返回 4 个 MSE 损失的和：

```
aux_loss = MSE(tower_pred[:, :5], tower_label[:, :5])    # 己方塔伤害
         + MSE(tower_pred[:, 5:], tower_label[:, 5:])    # 敌方塔伤害
         + MSE(gold_pred[:, :5], gold_label[:, :5])      # 己方金币
         + MSE(gold_pred[:, 5:], gold_label[:, 5:])      # 敌方金币
```

辅助损失会乘以一个较小的系数（`aux_loss_coef`）后加入总 PPO 损失中。

---

## 5. 外部交互

### 5.1 本模块依赖的外部模块

| 模块 | 依赖内容 | 依赖方式 |
|------|---------|---------|
| 基础设施 (`action_constants`) | `TYPE_CONFIG`, `ACTION_DIM`, `action_id_to_type_and_target` | 直接 import，用于动作头输出维度和 ID 编解码 |
| 基础设施 (`aux_labels`) | `NUM_HORIZONS` | 直接 import，用于辅助头输出维度定义 |

### 5.2 依赖本模块的外部模块

| 模块 | 使用内容 | 使用方式 |
|------|---------|---------|
| PPO 训练引擎 (`ppo_trainer.py`) | `AntWarPolicyValueNetwork` | 创建网络实例，在训练循环中调用 `forward()`、`evaluate_actions()`、`compute_auxiliary_loss()` |
| PPO 训练引擎 (`agent.py`) | `AntWarPolicyValueNetwork` | `PPOAgent` 构造时接收网络实例，在 `act()`/`get_action_values()` 中调用 `get_action()` |
| 自对弈编排 (`selfplay.py`) | `AntWarPolicyValueNetwork` | 用于创建初始随机对手的策略网络 |
| 对战执行 (`opponent_agent.py`) | `AntWarPolicyValueNetwork` | `OpponentAgent.from_checkpoint()` 从检查点恢复网络 |

### 5.3 典型交互场景

#### 场景 1：PPO 训练中的前向-反向传播

1. `PPOTrainer._train_ppo_epochs()` 从 `EpisodeBatch` 获取一批训练数据
2. 调用 `network.evaluate_actions(board, global, mask, actions)` 计算：
   - `log_probs`: 用于 PPO 策略损失计算
   - `values`: 用于价值损失计算
   - `entropy`: 用于熵正则化
   - `aux_preds`: 用于辅助损失计算
3. `PPOTrainer._compute_losses()` 组装总损失 = `policy_loss + value_loss * vf_coef - entropy * ent_coef + aux_loss * aux_coef`
4. 反向传播更新网络参数

#### 场景 2：推理时选择动作

1. `PPOAgent.act(observation)` 接收来自环境的观测字典
2. 调用 `observation_to_tensors()` 转为 PyTorch 张量
3. 调用 `network.get_action(board, global, mask, deterministic=False)`：
   - 双编码器提取特征
   - 分层动作头先选类型再选目标
   - 返回 `(action_id, log_prob, value)`
4. `PPOAgent` 返回 `action_id` 给环境执行

#### 场景 3：从检查点恢复网络权重

1. `CheckpointManager.load()` 读取 `.pt` 文件
2. 提取 `checkpoint['policy_state_dict']`
3. 调用 `network.load_state_dict(state_dict)` 恢复权重
4. 网络进入 `eval()` 模式用于评估或继续训练

---

## 6. 关键设计决策

### 6.1 独立 Policy/Value 双编码器

这是本模块最重要的架构决策。与常见的共享编码器 + 双头输出不同，本项目使用完全独立的策略网络和价值网络编码器：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **共享编码器** | 参数少，训练快 | 价值梯度会污染策略特征；二者优化目标不同 |
| **独立编码器 (本方案)** | 互不干扰，各自学习专用特征 | 参数量翻倍 |

选择独立编码器的核心动机是：价值网络的损失通常比策略网络大得多，共享编码器会导致价值梯度主导，使策略特征退化。

### 6.2 结构化分层动作头

将 96 维扁平动作空间分解为"类型 + 目标"的分层结构，而非直接输出 96 维 logits：

- **类型先验**：模型先学会"该不该建塔"再学"建在哪里"，更符合人类决策逻辑
- **梯度传播优化**：类型选择的梯度直接作用于类型 logits，不受目标维度差异影响
- **灵活的目标空间**：不同类型的目标数量不同（1~40），分层结构自然支持

### 6.3 六边形卷积 (HexConv)

标准 3×3 卷积假设像素在方形网格上排列，但 AntWar 的游戏地图使用六边形网格。六边形网格中，每个格子有 6 个邻域（而非 8 个），使用 HexConv 可以：
- 更准确地捕捉六边形棋盘上的空间关系
- 避免方形卷积在六边形网格上引入的偏差
- 在游戏 AI 领域（如 AlphaStar）中有成熟应用

### 6.4 辅助任务设计

添加塔伤害预测和金币收入预测两个辅助任务，其作用是：
- **表征学习**：迫使网络学习与游戏结果强相关的中间特征（塔的位置和状态、经济状况）
- **梯度正则化**：额外的监督信号在反向传播时起到正则化作用
- **零额外推理成本**：辅助头仅在训练时使用，推理时只需 `get_action()` 或 `forward()`
