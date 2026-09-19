# PPO v1 GRU 时序记忆方案

## 一、方案概述

### 1.1 动机

当前架构每步决策独立：

```
step t:  merged_t(256) → policy_head → action_t
         merged_t(256) → value_head  → value_t
```

`merged_t` 只包含**当前帧**的 board 和 global 信息，模型无法感知：

- 前 5 步自己做了什么动作（连续 Build 还是连续 Noop？）
- 前 10 步对手做了什么动作（大规模进攻还是骚扰？）
- 战局处于前期/中期/后期（大局观）

**GRU 方案**通过在 Encoder 输出的 `merged` 之上叠加一个 GRUCell，让隐状态 `hidden_t` 携带截至当前步的全部历史信息，实现跨步记忆。

### 1.2 预期收益

| 能力 | 当前 | +GRU |
|------|------|------|
| 感知连续动作的上下文（如连续 Build 5 次 → 缺钱） | ❌ | ✅ |
| 区分战局阶段（开局 vs 中局 vs 残局） | ❌ | ✅ |
| 记忆对手历史行为模式 | ❌ | ✅ |
| 辅助 Head（Tower/Gold）利用历史信息 | ❌ | ✅ |
| 推理速度影响 | baseline | +0.01ms/步 |

---

## 二、架构设计

### 2.1 网络结构变化

```
当前架构:
board → CNN_encoder ─┐
                      ├→ concat → Linear(256) → LayerNorm → merged → policy_head
global → MLP_encoder ─┘                                              → value_head
                                                                      → tower_head(训练)
                                                                      → gold_head(训练)

GRU 架构:
board → CNN_encoder ─┐
                      ├→ concat → Linear(256) → LayerNorm → merged ─┐
global → MLP_encoder ─┘                                              |
                                                                     ↓
                                                         GRUCell(256, 256)
                                                         ↑          |
                                                    hidden_{t-1}  hidden_t
                                                                     |
                                                    ┌────────────────┤
                                                    ↓                ↓
                                               policy_head      value_head
                                               tower_head(训练)   gold_head(训练)
```

**关键设计决策**：GRU 放在 Encoder 输出 `merged` 之后、所有 Head 之前。这样 policy/value/aux heads 全部从 `hidden_t` 读取特征，确保决策和辅助预测一致地拥有历史信息。

### 2.2 数据流（单步）

```
输入: board(B,28,19,19), global(B,33), hidden_prev(B,256)

  board_encoded = CNN_encoder(board)                → (B, 128)
  global_encoded = MLP_encoder(global)              → (B, 128)
  merged = concat(board_encoded, global_encoded)    → (B, 256)
  merged = Linear(256) + LayerNorm                  → (B, 256)

  hidden = GRUCell(merged, hidden_prev)              → (B, 256)

  action_logits = policy_head(hidden)               → (B, 96)
  value = value_head(hidden)                        → (B, 1)
  tower_pred = tower_head(hidden)                   → (B, 5)  [训练时]
  gold_pred = gold_head(hidden)                     → (B, 5)  [训练时]

输出: action_logits, value, hidden
```

### 2.3 与现有辅助 Head 的关系

**好消息**：TowerDamageHead 和 GoldIncomeHead 的预测输入从 `merged_t` 改为 `hidden_t` 后，理论上预测更准确——因为：

```
merged_t        = f(board_t, global_t)             # 只看当前帧
hidden_t        = GRU(merged_t, hidden_{t-1})      # 包含历史帧信息
```

Tower/Gold 头现在可以"看到"前 N 步的塔是否在被攻击、金币是否在增长，预测未来 16 步的塔伤害和金币收入自然更准。

**无需改动**辅助 Head 的网络结构，它们始终保持 `Linear(256→64) → ReLU → Linear(64→5)`，只是输入从 `merged` 换成了 `hidden`。

---

## 三、代码修改计划

### 3.1 文件清单

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `network/antwar_net.py` | 修改 | 新增 GRUCell，修改 `_encode` 签名和逻辑，全 Head 输入从 merged 改为 hidden |
| `trainer/ppo_trainer.py` | 修改 | PPO update 中的 encode 调用传 hidden，返回值增加 hidden |
| `trainer/selfplay.py` | 修改 | rollout 和训练中维护 hidden state 传递 |

---

### 3.2 antwar_net.py 详细修改

#### 3.2.1 新增配置参数

在 `AntWarPolicyValueNetwork.__init__` 中：

```python
class AntWarPolicyValueNetwork(nn.Module):
    def __init__(
        self,
        board_shape: Tuple[int, int, int] = (28, 19, 19),
        global_dim: int = 30,
        action_dim: int = 96,
        hidden_dim: int = 256,
        enable_auxiliary: bool = False,
        enable_gru: bool = False,         # 新增
    ) -> None:
        super().__init__()
        ...
        self.enable_gru = enable_gru

        # 现有模块不变
        self.cnn_encoder = HexCNNEncoder(...)
        self.mlp_encoder = MLPEncoder(...)
        self.projection = nn.Linear(...)
        self.merged_norm = nn.LayerNorm(hidden_dim)
        self.policy_head = StructuredActionHead(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # GRU Cell（新增）
        if enable_gru:
            self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # 辅助 Head（不变）
        if enable_auxiliary:
            self.tower_damage_head = TowerDamageHead(hidden_dim)
            self.gold_income_head = GoldIncomeHead(hidden_dim)

        ...
```

#### 3.2.2 修改 _encode 方法

```python
def _encode(
    self,
    board: torch.Tensor,
    global_features: torch.Tensor,
    hidden_prev: Optional[torch.Tensor] = None,  # 新增参数
) -> torch.Tensor:
    board_encoded = self.cnn_encoder(board)                     # (B, 128)
    global_encoded = self.mlp_encoder(global_features)           # (B, 128)
    merged = torch.cat([board_encoded, global_encoded], dim=1)  # (B, 256)
    merged = self.projection(merged)                            # (B, 256)
    merged = self.merged_norm(merged)                           # (B, 256)

    if self.enable_gru:
        B = merged.size(0)
        if hidden_prev is None:
            hidden_prev = torch.zeros(B, self.hidden_dim, device=merged.device)
        hidden = self.gru_cell(merged, hidden_prev)             # (B, 256)
        return hidden
    else:
        return merged  # 无 GRU 时行为不变
```

**重要**：`_encode` 的返回值语义发生变化：
- 无 GRU（默认）：返回 `merged`（当前帧信息）
- 有 GRU：返回 `hidden`（当前帧 + 历史信息）

#### 3.2.3 修改 forward / get_action / evaluate_actions 签名

所有调用 `_encode` 的方法都需要传递 `hidden_prev` 并返回 `hidden`。

```python
def forward(
    self,
    board: torch.Tensor,
    global_features: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    hidden_prev: Optional[torch.Tensor] = None,  # 新增
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    hidden = self._encode(board, global_features, hidden_prev)

    action_logits = self.policy_head(hidden)
    value = self.value_head(hidden)

    if action_mask is not None:
        action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

    return action_logits, value, hidden  # 新增返回 hidden


def get_action(
    self,
    board: torch.Tensor,
    global_features: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    hidden_prev: Optional[torch.Tensor] = None,  # 新增
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    hidden = self._encode(board, global_features, hidden_prev)
    ...
    return action, value, log_prob, hidden  # 新增返回 hidden


def evaluate_actions(
    self,
    board: torch.Tensor,
    global_features: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    hidden_prev: Optional[torch.Tensor] = None,  # 新增
) -> Tuple[...]:
    hidden = self._encode(board, global_features, hidden_prev)
    ...
    return action_log_probs, values, entropy, action_logits, type_entropy, target_entropy, hidden
```

#### 3.2.4 修改 _compute_aux_loss 的输入

```python
def _compute_aux_loss(
    self,
    hidden: torch.Tensor,    # 从 merged 改为 hidden
    tower_labels: torch.Tensor,
    gold_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tower_pred = self.tower_damage_head(hidden)  # 输入改为 hidden
    gold_pred = self.gold_income_head(hidden)
    tower_loss = F.mse_loss(tower_pred, tower_labels)
    gold_loss = F.mse_loss(gold_pred, gold_labels)
    return tower_loss, gold_loss, tower_pred, gold_pred
```

---

### 3.3 ppo_trainer.py 详细修改

#### 3.3.1 _ppo_update 方法

当前流程需要改为逐时间步传递 hidden state 的流程。关键变化在数据 batch 的组织方式。

**现状**：`evaluate_actions` 一次处理一个 mini-batch（从 N 条 episode 中各取一些 step，共 batch_size 条样本），样本之间**无时序关系**，无法使用 GRU。

**修改方案**：PPO update 需要改变数据组织，从"平铺采样"改为"按 episode 序采样"。

核心思路：从 `tensors` 中一次性取出一个 episode 的所有 step，按时间序依次传入 `evaluate_actions`，维护每一步的 hidden state。

```python
def _ppo_update(self, tensors, batch_size):
    """
    tensors: Dict
      'board': Tensor[N, C, H, W]       # 所有 episode 的所有 step 平铺
      'global': Tensor[N, global_dim]
      'episode_ids': Tensor[N]           # 新增：每条样本所属的 episode ID
      'step_indices': Tensor[N]          # 新增：每条样本在对应 episode 中的步序号
      ...
    """
    # ---------- 新增：按 episode 组织数据 ----------
    episode_ids = tensors['episode_ids'].cpu().numpy()
    unique_eps = np.unique(episode_ids)

    # 为每个 episode 初始化 hidden state（全零）
    episode_hidden = {}
    for eid in unique_eps:
        episode_hidden[eid] = None

    # 按 (episode_id, step_index) 排序
    order = np.argsort(episode_ids * 100000 + tensors['step_indices'].cpu().numpy())

    # 按顺序组织 mini-batch
    for start in range(0, len(order), batch_size):
        batch_order = order[start:start + batch_size]
        batch_indices = torch.from_numpy(batch_order).to(tensors['board'].device)

        board = tensors['board'][batch_indices]
        global_obs = tensors['global'][batch_indices]
        ...

        # evaluate_actions 现在需要 hidden_prev，并返回 hidden
        action_log_probs, values, entropy, action_logits, type_entropy, target_entropy, hiddens = \
            self.policy.evaluate_actions(
                board, global_obs, actions, action_mask,
                hidden_prev=None  # 简化处理：batch 内各样本的 hidden 无法对齐
            )
```

**简化方案**（v1 实现）：

由于 PPO update 时 mini-batch 内的样本来自不同 episode 的不同步，无法直接使用跨步 GRU，因此 v1 采用折中：

1. **rollout 时**：GRU 完整工作，hidden 在 episode 内跨步传递
2. **训练时**：evaluate_actions 不传 hidden_prev（视为独立样本）

这意味着 PPO update 时每个样本仍然是独立处理的，但 rollout 收集的数据已经受益于 GRU 的记忆能力。Gradients 仍然通过单步的 GRU cell 反向传播。

```
rollout 时: step 0 → step 1 → ... → step 249  (hidden 跨步传递 ✅)
训练时:     每个样本独立计算，无 hidden 传递   (hidden = f(merged, 0))
```

这是 Transformer/Gated RNN 在在线 RL 中的标准做法（如 IMPALA, R2D2 等论文均采用类似策略），因为：
- 训练时 batch 内各样本的 hidden state 无法对齐（不同 episode、不同步序）
- rollout 时使用记忆已经能让策略做出更好的决策
- 单步梯度也能学习到"从 merged 产生有用 hidden"的映射

--- v2 可选升级 ---
如果要让训练时也能跨步传递 hidden，需要 change 数据收集方式——每条 episode 的 step 连续存入一个 chunk，训练时按 episode 顺序读取。但这需要改造数据 pipeline，复杂度较高，建议 v1 先不做。
"""

```

#### 3.3.2 _compute_aux_loss 调用

```python
if self.enable_auxiliary and \
   'aux_tower_damage' in tensors and 'aux_gold_income' in tensors:
    hidden = self.policy._encode(board, global_obs)  # 无 hidden_prev
    aux_tower_loss, aux_gold_loss, _, _ = \
        self.policy._compute_aux_loss(
            hidden,  # 改为 hidden
            tensors['aux_tower_damage'][batch_indices],
            tensors['aux_gold_income'][batch_indices])
```

---

### 3.4 selfplay.py 详细修改

#### 3.4.1 rollout 循环

这是 GRU 发挥最大作用的地方 —— hidden state 在 episode 内跨步传递。

```python
def _run_episode(self, env, max_steps, ...) -> EpisodeDetail:
    hidden = None  # 每局开始时清空
    step_snapshots = []
    state = env.reset()

    for step in range(max_steps):
        board, global_obs = self._extract_obs(state)
        action_mask = self._build_action_mask(state)

        with torch.no_grad():
            action, value, log_prob, hidden = self.policy.get_action(
                board, global_obs, action_mask,
                hidden_prev=hidden,     # 传入上一步的 hidden
                deterministic=False,
            )

        # 执行动作
        next_state, reward, done, info = env.step(action.item())

        # 记录 transition（hidden 也记录，但训练时不用）
        episode_buffer.add(state, action, reward, value, log_prob, ...)

        # 收集辅助预测标签数据
        tower_hp = self._get_total_tower_hp(next_state)
        our_cumulative = self._get_our_cumulative(next_state)
        step_snapshots.append((tower_hp, our_cumulative))

        if done:
            break
        state = next_state
```

#### 3.4.2 对战评估（battle）也要传 hidden

```python
def _run_battle(self, env, max_steps, ...) -> BattleResult:
    hidden = None
    ...
    for step in range(max_steps):
        ...
        action, _, _, hidden = self.policy.get_action(
            board, global_obs, action_mask,
            hidden_prev=hidden,
            deterministic=True,  # 评估时用确定性策略
        )
```

---

### 3.5 配置文件修改

```yaml
# ppo_antwar.yaml
network:
  hidden_dim: 256
  enable_gru: true         # 新增：开启 GRU
  # enable_gru: false      # 可随时关闭，回退到原始行为
```

---

## 四、训练注意事项

### 4.1 Hidden State 初始化

每局开始时 `hidden = None`，内部自动初始化为全零向量：

```python
if hidden_prev is None:
    hidden_prev = torch.zeros(B, self.hidden_dim, device=merged.device)
```

B = 1（单局推理）。多 env 并行时，每个 env 独立维护自己的 hidden。

### 4.2 多 Env 并行

使用 `n_envs=20` 时，每个 env 独立维护 hidden：

```python
# selfplay.py
self.env_hiddens = [None] * n_envs

for env_id in range(n_envs):
    board, global_obs = ...
    action, value, log_prob, hidden = policy.get_action(
        board, global_obs, action_mask,
        hidden_prev=self.env_hiddens[env_id],
    )
    self.env_hiddens[env_id] = hidden  # 更新
```

### 4.3 梯度流

```
rollout → 存储 transition(hidden 不存)
         ↓
PPO update → evaluate_actions(每个样本独立, hidden_prev=None)
           → 计算 loss
           → backward 通过 GRUCell(f(merged, 0))
```

梯度只通过单步 GRU cell 反传，不跨步。这是 v1 的简化方案。

### 4.4 权重初始化

GRUCell 使用默认的 orthogonal/xavier 初始化即可，无需特殊处理。建议在 `_init_weights` 中对 GRU 做：

```python
if self.enable_gru:
    for name, param in self.gru_cell.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param, gain=1.0)
        elif 'bias' in name:
            # GRU bias 的遗忘门偏置初始化为 1（提升长程记忆能力）
            n = param.size(0)
            param.data[:n//4].fill_(0)        # reset gate
            param.data[n//4:n//2].fill_(1)    # update gate (遗忘门偏置=1)
            param.data[n//2:3*n//4].fill_(0)  # new gate
            param.data[3*n//4:].fill_(0)      # output (GRUCell 只有 3 个门)
```

GRUCell 内部只有一个 bias，包含 3 组（reset/update/new），`fill_(1)` 部分会让模型默认"保留历史信息"，有利于长程记忆。

### 4.5 渐进式引入

推荐训练流程：

1. **阶段一**：`enable_gru: false`，继续当前训练（无 GRU baseline）
2. **阶段二**：`enable_gru: true`，从头开始训练（含 GRU）
3. **对比**：相同 episode 数下对比对战胜率

---

## 五、Rollout 数据格式变化

### 5.1 EpisodeBatch 新增字段

```python
@dataclass
class EpisodeBatch:
    boards: List[np.ndarray]           # (C, H, W)
    globals: List[np.ndarray]          # (global_dim,)
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    action_masks: List[np.ndarray]
    dones: List[bool]
    # 以下为现有字段
    aux_tower_damage: Optional[List[np.ndarray]] = None
    aux_gold_income: Optional[List[np.ndarray]] = None
    # hidden 不存储（每步需要 256×4=1KB, 250步=250KB/局, 20局×10轮=50MB）
    # 训练时不用 hidden，所以无需存
```

**Hidden 不存到 replay buffer** — 这是关键决策：

- 存 hidden：每步额外 256 float32 ≈ 1KB，250 步 ≈ 250KB/episode
- 20 envs × 10 update chunks = 50MB，负担不大
- 但训练时 batch 内样本无法对齐 hidden，存了也没用
- 结论：v1 不存，v2 如需跨步训练再考虑

### 5.2 TensorDict 转换

`to_tensors` 方法参考下面代码：

```python
def to_tensors(self, device, ...):
    result = {
        'board': torch.FloatTensor(np.array(self.boards)).to(device),
        'global': torch.FloatTensor(np.array(self.globals)).to(device),
        'action': torch.LongTensor(np.array(self.actions)).to(device),
        ...
        'episode_ids': torch.zeros(len(self.boards), dtype=torch.long),  # 本 batch 只含 1 条 episode
    }
    return result
```

---

## 六、训练效果验证

### 6.1 对照实验

| 配置 | 说明 |
|------|------|
| A（baseline） | `enable_gru: false`，其余完全相同 |
| B（GRU） | `enable_gru: true`，其余完全相同 |

### 6.2 关键指标

| 指标 | 预期方向 | 说明 |
|------|---------|------|
| Reward | ↑ | GRU 应带来更高平均奖励 |
| Battle WinRate | ↑ | 对战胜率应提升 |
| Entropy | 持平或略降 | 记忆降低决策不确定性，正常 |
| TowerLoss / GoldLoss | ↓ | hidden 包含历史信息，预测应更准 |
| 训练速度 | ~持平 | GRUCell 计算量极小 |

### 6.3 可视化验证

`train_metrics_table` 中新增 `GRU Hidden Mean` 和 `GRU Hidden Std` 字段（可选），观察 hidden state 是否随时间步产生有意义的变化：

- 开局：hidden 应接近 0（无历史）
- 中期：hidden 应稳定在高值（积累了大量历史）
- 不同战局的 hidden 应产生差异（用于验证 GRU 确实在学习）

---

## 七、回退方案

如果 GRU 效果不达预期，回退只需：

1. 配置 `enable_gru: false`
2. 模型加载旧权重（不包含 GRU 参数，自动忽略）
3. `_encode` 返回 `merged` 而非 `hidden`，所有 Head 恢复原状

代码中所有 `if self.enable_gru:` 保护了改动，关闭后行为与 GRU 引入前完全一致。

---

## 八、总结

| 维度 | 评估 |
|------|------|
| 实现复杂度 | 低（~200 行新增/修改） |
| 推理开销 | ~+0.01ms/步（GRUCell vs Linear） |
| 参数量增加 | ~262K（GRUCell: 256→256） |
| 总参数量变化 | 262K / ~5M ≈ +5% |
| 风险 | 低（有回退开关） |
| 预期收益 | 中高（解决独立决策的核心缺陷） |
