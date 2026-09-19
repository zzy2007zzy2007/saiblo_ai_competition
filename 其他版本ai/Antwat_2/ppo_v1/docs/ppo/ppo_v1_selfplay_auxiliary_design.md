# PPO v1 辅助预测任务技术方案（真实轨迹版）

## 一、方案概述

在现有 PPO 模型的基础上，增加两个辅助预测头（Auxiliary Prediction Heads），以多任务学习（Multi-Task Learning）的方式，让模型预测未来若干回合的"业务指标"：

1. **塔攻击强度预测**：1/2/4/8 回合后，我方防御塔会承受的总 HP 损失
2. **金币收入预测**：1/2/4/8 回合后，我方可以赚取的金币总额

标签来源：**同一 episode 的真实游戏轨迹**——即当前步之后实际发生的事。辅助预测头**不参与推理时决策**（不做 feedback loop），仅通过共享表征的梯度反传来改善策略/价值网络的训练质量。

### 1.1 核心假设

- AntWar 的决策效果滞后明显（建塔 → 30 轮后才产生击杀回报），GAE 的有效回溯窗口有限
- 让模型显式预测"当前状态在真实游戏中的后续发展"，可以改善长距离信用分配
- 多任务学习通过辅助梯度丰富共享表征，间接提升策略质量
- 本方案的首要目标是**快速验证思路可行性**，因此选择实现成本最低的真实轨迹方案

### 1.2 与现有架构的关系

```
原架构:
  merged → [PolicyHead, ValueHead]

新架构:
  merged → [PolicyHead, ValueHead, TowerDamageHead, GoldIncomeHead]
                                        ↑              ↑
                                     新增辅助头      新增辅助头
```

辅助头只接收 `merged` 表征，不参与 `get_action` 推理流程。

---

## 二、辅助预测标签设计

### 2.1 预测目标

| 预测头 | 预测内容 | 预测步长 | 输出维度 |
|--------|---------|----------|----------|
| TowerDamageHead | 我方所有塔在未来 N 回合内的总 HP 损失 | [1, 2, 4, 8] | 4 |
| GoldIncomeHead | 我方在未来 N 回合内获得的金币总数 | [1, 2, 4, 8] | 4 |

### 2.2 标签来源：真实轨迹回看

**核心思路**：不模拟任何东西。在 episode 数据收集过程中，用列表记录每个时间步的（塔总 HP, 金币数），episode 结束后通过回看来计算标签。

```
┌──────────────────────────────────────────────────────────────────┐
│ Episode 时间线 (250 步)                                          │
│                                                                  │
│ Step 0  Step 1  Step 2  ...  Step 50  ...  Step 58  ...  Step 249│
│   │       │       │             │             │              │    │
│   └── 快照列表: [(hp₀,c₀), (hp₁,c₁), (hp₂,c₂), ..., (hp₂₄₉,c₂₄₉)]│
│                                                                  │
│ 为 Step 50 计算标签:                                              │
│   horizon=1: hp₅₀ - hp₅₁  (或 hp₅₁ - hp₅₀)                      │
│   horizon=2: hp₅₀ - hp₅₂                                         │
│   horizon=4: hp₅₀ - hp₅₄                                         │
│   horizon=8: hp₅₀ - hp₅₈  ← 直接取 8 步后的真实游戏值             │
│                                                                  │
│ 为 Step 245 计算标签 (episode 末尾不足 8 步):                      │
│   horizon=8: hp₂₄₅ - hp₂₄₉  ← 用 episode 最后一帧的值            │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 实现伪代码

```python
def compute_aux_labels_from_trajectory(snapshots):
    """
    snapshots: List[Tuple[int, int]]  — 每步的 (塔总HP, 累计收入)

    注意：第二个字段是“累计收入”，不是“余额”。
    累计收入 = 自游戏开始以来通过被动收入、击杀奖励、蚂蚁破家获得的金币总和。
    它只增不减，因此不受建塔/升级消费的干扰。

    返回: 两个 List[np.ndarray]，每个元素是 shape (4,)
    """
    horizons = [1, 2, 4, 8]
    n = len(snapshots)
    tower_labels = []
    gold_labels = []

    for t in range(n):
        tower_dmg = np.zeros(4, dtype=np.float32)
        gold_inc = np.zeros(4, dtype=np.float32)
        cur_hp, cur_earned = snapshots[t]

        for i, h in enumerate(horizons):
            future_idx = min(t + h, n - 1)  # 末尾不足时用最后一帧
            future_hp, future_earned = snapshots[future_idx]

            # 塔损失 = HP 减少量（不会为负）
            tower_dmg[i] = max(0.0, float(cur_hp - future_hp))
            # 金币收入 = 累计收入差值（天然 ≥ 0，无需 max）
            gold_inc[i] = float(future_earned - cur_earned)

        tower_labels.append(tower_dmg)
        gold_labels.append(gold_inc)

    return tower_labels, gold_labels
```

### 2.4 数据收集流程

在 `_collect_episode_with_swap()` 中，每个时间步额外记录两个数字。关键：金币标签使用**累计收入**而非余额，以避免建塔/升级消费干扰收入计算。

```python
# === 数据收集循环中的改动 ===

step_snapshots = []  # [(tower_hp, cumulative_earned), ...]

# episode 开始时初始化累计收入（用初始余额作为 baseline）
cumulative_earned = state.coins[player]

for step in range(max_steps):
    # --- 新增：记录当前快照 ---
    if hasattr(env, '_runtime') and env._runtime and hasattr(env._runtime, 'state'):
        state = env._runtime.state
        tower_hp = sum(t.hp for t in state.towers_of(player))
        # 注意：记录的是累计收入，不是余额
    else:
        tower_hp, cumulative_earned = 0, 0
    step_snapshots.append((tower_hp, cumulative_earned))

    # --- 原有代码：观测编码、模型推理、环境步进 ---
    obs = env.encode(state)
    action, log_prob, value = model(obs)
    next_state, reward, done = env.step(action)
    batch.add(obs, action, reward, value, log_prob, done)

    # --- 新增：累积本轮的纯收入 ---
    # own_income = 本轮 advance_round 带来的被动收入 + 击杀奖励 + 破家奖励
    # 不包含建塔/升级的消费（消费在 ops 阶段已扣除）
    reward_detail = env._last_reward_detail.get(agent_key, {})
    cumulative_earned += reward_detail.get('own_income', 0)

# === Episode 结束后，计算辅助标签 ===
if enable_auxiliary:
    tower_labels, gold_labels = compute_aux_labels_from_trajectory(step_snapshots)
    batch.aux_tower_damage = tower_labels
    batch.aux_gold_income = gold_labels
```

### 2.5 EpisodeBatch 改动

新增两个字段：

```python
@dataclass
class EpisodeBatch:
    # ... 原有字段 ...
    aux_tower_damage: List[np.ndarray] = None   # 每个是 shape (4,) 的 float32
    aux_gold_income: List[np.ndarray] = None     # 每个是 shape (4,) 的 float32
```

默认 `None` 保证向后兼容（关掉辅助任务时无需填充）。

`merge()` 和 `to_tensors()` 中增加对应处理。

### 2.6 性能影响

| 操作 | 额外开销 |
|------|---------|
| 每步记录 2 个 int | **零**（纳秒级） |
| episode 结束后回算标签 | ~1ms（纯 Python 循环，n=250，O(n×4) 复杂度） |
| 总计 | **可忽略** |

与 NOOP 模拟方案（每步 clone + 8 次 advance_round，约 2.5ms/步、12.5 秒/batch）相比，真实轨迹方案的额外计算开销为零。

### 2.7 标签值域

| 指标 | 典型范围 | 说明 |
|------|---------|------|
| 塔总 HP（所有塔之和） | 0 ~ 60+（取决于塔数量） | 模型预测该值在 [1,2,4,8] 回合后的减少量 |
| 累计收入 | 0 ~ 500+（初始 ~50，每轮约 +1.5） | 只增不减，不受建塔/升级消费干扰。模型预测未来 h 轮的增量 |

**设计决定：不在标签上做显式归一化。** 原因：
- 塔 HP 和累计收入都是绝对值，规模可预测（HP ≤ 100，收入 ≤ 500）
- 不加归一化让辅助头学到原始量纲的意义，与 value_head 学 200 量纲的值是类似的
- 简化实现，减少一个需要调的超参数（归一化系数）
- 如果训练中发现 aux_loss 量纲过大（>100），再考虑加 `/50.0` 或 `/20.0` 归一化

---

## 三、模型架构改动

### 3.1 辅助预测头设计

在 `merged`（256 维）上新增两个轻量级预测头：

```python
class TowerDamageHead(nn.Module):
    """预测塔在 [1, 2, 4, 8] 回合后的总 HP 损失"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, merged):
        return self.net(merged)  # (batch, 4)


class GoldIncomeHead(nn.Module):
    """预测 [1, 2, 4, 8] 回合内的金币收入"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, merged):
        return self.net(merged)  # (batch, 4)
```

### 3.2 插入位置

```python
class AntWarPolicyValueNetwork(nn.Module):
    def __init__(self, ..., enable_auxiliary=False):
        # ... 原有初始化 (cnn_encoder, mlp_encoder, projection, policy_head, value_head) ...

        self.enable_auxiliary = enable_auxiliary
        if enable_auxiliary:
            self.tower_damage_head = TowerDamageHead(hidden_dim)
            self.gold_income_head = GoldIncomeHead(hidden_dim)

    def forward(self, board, global_features, action_mask=None):
        merged = self._encode(board, global_features)
        action_logits = self.policy_head(merged)
        value = self.value_head(merged) * self.value_bound

        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

        if self.enable_auxiliary:
            tower_damage_pred = self.tower_damage_head(merged)
            gold_income_pred = self.gold_income_head(merged)
            return action_logits, value, tower_damage_pred, gold_income_pred

        return action_logits, value
```

### 3.3 `get_action` 和 `evaluate_actions` 不改动

辅助头不参与推理，`get_action` 和 `evaluate_actions` 保持原样。辅助预测只在 `_ppo_update` 的训练阶段触发。

### 3.4 参数量增量

| 组件 | 参数量 |
|------|--------|
| TowerDamageHead | 256×64 + 64 + 64×4 + 4 = 16,708 |
| GoldIncomeHead | 256×64 + 64 + 64×4 + 4 = 16,708 |
| **总计** | **~33K**（< 2% 基础模型） |

---

## 四、训练流程改动

### 4.1 辅助损失函数

MSE，不做额外归一化：

```python
def _compute_aux_loss(self, merged, aux_tower_labels, aux_gold_labels):
    """merged: (batch, 256) — 融合层输出"""
    tower_pred = self.tower_damage_head(merged)    # (batch, 4)
    gold_pred = self.gold_income_head(merged)       # (batch, 4)

    tower_loss = F.mse_loss(tower_pred, aux_tower_labels)
    gold_loss = F.mse_loss(gold_pred, aux_gold_labels)

    return tower_loss, gold_loss, tower_pred, gold_pred
```

### 4.2 总损失公式

```python
if self.enable_auxiliary:
    aux_tower_loss, aux_gold_loss, tower_pred, gold_pred = \
        self.policy._compute_aux_loss(merged, aux_tower_labels, aux_gold_labels)
else:
    aux_tower_loss = torch.tensor(0.0)
    aux_gold_loss = torch.tensor(0.0)

loss = (
    policy_loss
    + vf_coef * value_loss
    + ent_coef * entropy_loss
    + hard_entropy_penalty
    + aux_tower_coef * aux_tower_loss    # 新增
    + aux_gold_coef * aux_gold_loss       # 新增
)
```

### 4.3 超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `enable_auxiliary` | False | 开关，默认关闭 |
| `aux_tower_coef` | 0.1 | 塔攻击强度损失的权重 |
| `aux_gold_coef` | 0.1 | 金币收入损失的权重 |

权重原则（与 NOOP 版本一致）：
- 不能太大，否则辅助任务主导梯度
- 不能太小，否则无实际影响
- 0.1 是保守起点

### 4.4 训练指标新增

```python
return {
    # ... 原有指标 ...
    'aux_tower_loss': aux_tower_loss.item(),
    'aux_gold_loss': aux_gold_loss.item(),
}
```

### 4.5 Mini-batch 中 auxiliary loss 的计算位置

在 `_ppo_update` 中，`evaluate_actions` 已经计算了 `merged`（通过 `self.policy._encode(board, global_obs)`），辅助损失可以直接复用这个 `merged`：

```python
# _ppo_update 中（伪代码，标注改动位置）

for batch_indices in mini_batches:
    board, global_obs, action_mask, actions = ...
    
    # evaluate_actions 内部调用了 _encode，产生 merged
    action_log_probs, values, entropy, action_logits, type_entropy, target_entropy = \
        self.policy.evaluate_actions(board, global_obs, actions, action_mask)
    
    # ... PPO loss 计算 ...

    # === 新增：辅助损失 ===
    if self.enable_auxiliary:
        merged = self.policy._encode(board, global_obs)
        aux_tower_loss, aux_gold_loss, _, _ = \
            self.policy._compute_aux_loss(merged,
                tensors['aux_tower_damage'][batch_indices],
                tensors['aux_gold_income'][batch_indices])
    
    # ... 总 loss ...
```

**注意**：虽然 `merged` 被计算了两次（`evaluate_actions` 内部一次 + 辅助损失这里一次），但这保证了代码改动的最小侵入性——不需要修改 `evaluate_actions` 的签名和返回值。额外的 `_encode` 调用开销极小（ReLU + Linear，~0.1ms）。

---

## 五、不涉及标签的时间步

episode 末尾的步（最后 7 步）的标签使用 episode 最后一帧的值（见 2.3 节 `future_idx = min(t + h, n - 1)`）。这意味着：

- **Step 0~241**：每个 horizon 都有真实对应帧的标签
- **Step 242~249**：部分或全部 horizon 使用最后一帧的值

这近似于"隐含假设 episode 结束后状态不再变化"。对于塔 HP 和金币来说，在游戏结束后这确实是正确的（terminal 状态不再推进）。对于辅助任务的可学习性来说，这些"不够完整"的标签占比约 7/250 ≈ 3%，影响可忽略。

**可选优化**：如果不希望最后几步参与辅助训练，可以在 loss 计算时 mask 掉 `t + horizon >= n` 的项。但这会增加代码复杂度，对 3% 的数据量来说不值得。

---

## 六、验证方案

### 6.1 A/B 对比实验

相同配置（seed 已去掉），跑两轮训练：

| 组别 | 配置 | 耗时 |
|------|------|------|
| **Baseline** | `enable_auxiliary=False` | 与当前相同 |
| **Auxiliary** | `enable_auxiliary=True`, `aux_tower_coef=0.1`, `aux_gold_coef=0.1` | 与当前几乎相同（标签计算零开销） |

### 6.2 评估指标

| 维度 | 指标 | 期望变化 |
|------|------|----------|
| 训练效率 | Baseline 对战 WinRate 收敛速度 | 期望 Auxiliary 更快或持平 |
| 训练稳定性 | ratio_spike / grad_explosion 次数 | 期望不劣化 |
| 策略质量 | MediumRuleAI 胜率 | 期望提升 |
| 辅助任务质量 | aux_tower_loss / aux_gold_loss 下降曲线 | 期望持续下降 |
| 计算开销 | 单 batch 耗时 | 期望无变化 |

### 6.3 实验周期

单次实验约 600 episodes（当前一轮训练的量），对比两轮结果即可初步判断。

---

## 七、改动文件清单

| 文件 | 改动类型 | 改动行数 | 改动内容 |
|------|---------|---------|----------|
| `network/antwar_net.py` | 修改 | ~50 行 | 新增 `TowerDamageHead`、`GoldIncomeHead` 类；`AntWarPolicyValueNetwork` 增加 `enable_auxiliary` 开关、辅助头初始化、`_compute_aux_loss` 方法 |
| `trainer/ppo_trainer.py` | 修改 | ~20 行 | `EpisodeBatch` 增加 `aux_tower_damage`、`aux_gold_income` 字段（含 `merge`/`to_tensors` 适配）；`_ppo_update` 中增加辅助损失计算和梯度反传 |
| `trainer/selfplay.py` | 修改 | ~15 行 | `_collect_episode_with_swap` 中记录 `step_snapshots`，episode 结束后调用标签计算函数，结果存入 `EpisodeBatch` |
| `utils/aux_labels.py` | **新增** | ~25 行 | `compute_aux_labels_from_trajectory(snapshots)` 函数 |
| `configs/ppo_antwar.yaml` | 修改 | ~3 行 | 增加 `enable_auxiliary`、`aux_tower_coef`、`aux_gold_coef` |
| **总计** | | **~115 行** | 5 个文件 |

### 7.1 各文件详细改动

#### 7.1.1 `network/antwar_net.py`

```python
# === 新增类 ===
class TowerDamageHead(nn.Module):
    """..."""
    # 如上文 3.1 节

class GoldIncomeHead(nn.Module):
    """..."""
    # 如上文 3.1 节

# === AntWarPolicyValueNetwork 改动 ===
class AntWarPolicyValueNetwork(nn.Module):
    def __init__(self, ..., enable_auxiliary=False):
        # ... 原有不变 ...
        self.enable_auxiliary = enable_auxiliary
        if enable_auxiliary:
            self.tower_damage_head = TowerDamageHead(hidden_dim)
            self.gold_income_head = GoldIncomeHead(hidden_dim)

    def _compute_aux_loss(self, merged, tower_labels, gold_labels):
        tower_pred = self.tower_damage_head(merged)
        gold_pred = self.gold_income_head(merged)
        tower_loss = F.mse_loss(tower_pred, tower_labels)
        gold_loss = F.mse_loss(gold_pred, gold_labels)
        return tower_loss, gold_loss, tower_pred, gold_pred

    # forward(), get_action(), evaluate_actions() 保持原样不变
```

#### 7.1.2 `trainer/ppo_trainer.py`

```python
# === EpisodeBatch ===
@dataclass
class EpisodeBatch:
    # ... 原有 8 个字段不变 ...
    aux_tower_damage: List[np.ndarray] = None  # 新增
    aux_gold_income: List[np.ndarray] = None    # 新增

    def merge(self, others):
        # 原有字段的处理不变
        # 增加 aux_tower_damage 和 aux_gold_income 的 extend 逻辑
        for attr in ['aux_tower_damage', 'aux_gold_income']:
            lst = getattr(merged, attr)
            if lst is None:
                lst = []
            my_attr = getattr(self, attr)
            if my_attr is not None:
                lst.extend(my_attr)
            for b in others:
                if getattr(b, attr) is not None:
                    lst.extend(getattr(b, attr))
            setattr(merged, attr, lst)

    def to_tensors(self, device):
        result = { ... }  # 原有不变
        # 新增
        for key in ['aux_tower_damage', 'aux_gold_income']:
            data = getattr(self, key)
            if data is not None and len(data) > 0:
                result[key] = torch.FloatTensor(np.array(data)).to(device)
        return result

# === PPOTrainer._ppo_update ===
# 在 mini-batch 循环中，计算完 policy_loss/value_loss 后：

if self.enable_auxiliary and \
   'aux_tower_damage' in tensors and 'aux_gold_income' in tensors:
    merged = self.policy._encode(board, global_obs)
    aux_tower_loss, aux_gold_loss, _, _ = \
        self.policy._compute_aux_loss(
            merged,
            tensors['aux_tower_damage'][batch_indices],
            tensors['aux_gold_income'][batch_indices])
else:
    aux_tower_loss = torch.tensor(0.0, device=self.device)
    aux_gold_loss = torch.tensor(0.0, device=self.device)

loss = (
    policy_loss
    + self.ppo_config.vf_coef * value_loss
    + self.current_ent_coef * entropy_loss
    + hard_entropy_penalty
    + self.aux_tower_coef * aux_tower_loss
    + self.aux_gold_coef * aux_gold_loss
)
```

#### 7.1.3 `trainer/selfplay.py`

```python
# _collect_episode_with_swap 中

# 在循环开始前
step_snapshots = []  # [(tower_hp, cumulative_earned), ...]
# 使用累计收入而非余额，避免消费干扰
cumulative_earned = state.coins[player] if has_state else 0

# 在数据收集循环中（每次 step 开始时，记录快照）
tower_hp = sum(t.hp for t in state.towers_of(player)) if has_state else 0
step_snapshots.append((tower_hp, cumulative_earned))

# ... 原有观测编码、动作选择、环境步进 ...

# step 结束后，累积本轮的纯收入
reward_detail = env._last_reward_detail.get(agent_key, {})
cumulative_earned += reward_detail.get('own_income', 0)

# Episode 结束后
if getattr(self.trainer, 'enable_auxiliary', False):
    from ppo_antwar.utils.aux_labels import compute_aux_labels_from_trajectory
    tower_labels, gold_labels = compute_aux_labels_from_trajectory(step_snapshots)
    
    if batch.aux_tower_damage is None:
        batch.aux_tower_damage = []
    if batch.aux_gold_income is None:
        batch.aux_gold_income = []
    batch.aux_tower_damage.extend(tower_labels)
    batch.aux_gold_income.extend(gold_labels)
```

#### 7.1.4 `utils/aux_labels.py`（新文件）

```python
"""辅助任务标签计算"""
import numpy as np

HORIZONS = [1, 2, 4, 8]

def compute_aux_labels_from_trajectory(snapshots):
    """
    从真实轨迹回看计算辅助标签。

    第二个字段使用“累计收入”而非“余额”。
    累计收入 = 自游戏开始以来通过被动收入、击杀奖励、蚂蚁破家获得的金币总和，
    只增不减，不受建塔/升级消费的干扰。

    Args:
        snapshots: List[Tuple[int, int]]  — 每步的 (塔总HP, 累计收入)

    Returns:
        tower_labels: List[np.ndarray]  — 每个是 shape (4,) 的 float32
        gold_labels: List[np.ndarray]   — 每个是 shape (4,) 的 float32
    """
    n = len(snapshots)
    tower_labels = []
    gold_labels = []

    for t in range(n):
        tower_dmg = np.zeros(4, dtype=np.float32)
        gold_inc = np.zeros(4, dtype=np.float32)
        cur_hp, cur_earned = snapshots[t]

        for i, h in enumerate(HORIZONS):
            future_idx = min(t + h, n - 1)
            future_hp, future_earned = snapshots[future_idx]
            tower_dmg[i] = max(0.0, float(cur_hp - future_hp))
            gold_inc[i] = float(future_earned - cur_earned)

        tower_labels.append(tower_dmg)
        gold_labels.append(gold_inc)

    return tower_labels, gold_labels
```

#### 7.1.5 `configs/ppo_antwar.yaml`

```yaml
# 在 ppo 或 training 段中新增
enable_auxiliary: false
aux_tower_coef: 0.1
aux_gold_coef: 0.1
```

---

## 八、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 辅助损失不下降 | 中 | 梯度无贡献，退化到 Baseline | 监控 aux_loss 曲线；调整 coef 或预测步长 |
| 标签含自引用噪声（模型动作影响标签） | 中 | 辅助任务学到的是"我一般会做什么"而非"状态蕴含什么" | 可接受——探针实验中 85%+ 是 NOOP，自引用影响有限 |
| 辅助梯度干扰 PPO 主任务 | 低 | 训练质量下降 | coef=0.1 保守起点；可通过 enable_auxiliary=False 关闭 |
| 最后几步的标签不够完整 | 极低 | 3% 的数据质量略差 | 影响可忽略；必要时可 mask 掉 |

---

## 九、与 NOOP 模拟方案的对比

| 维度 | 真实轨迹（本方案） | NOOP 模拟 |
|------|-------------------|-----------|
| 代码量 | **~115 行** | ~200 行 |
| 额外计算开销 | **零** | ~2.5ms/步，~12.5s/batch |
| 需要访问引擎内部 | **否** | 是（clone + advance_round） |
| 标签噪声 | 有（对手行为 + 自身动作） | 无 |
| 标签完整性 | 最后 7 步不完整 | 每步都有 |
| 探针实验适用性 | **极佳** | 适合后续优化 |

---

## 十、实施步骤

### Phase 1：准备（不改训练逻辑）

1. 在 `EpisodeBatch` 中增加字段，确保向后兼容（`None` 默认值）
2. 新增 `aux_labels.py` 模块
3. 在 `AntWarPolicyValueNetwork` 中增加辅助头和 `_compute_aux_loss` 方法
4. 配置文件增加参数

### Phase 2：标签收集验证

1. 在 `_collect_episode_with_swap` 中增加标签计算
2. **不启用辅助损失参与训练**（`aux_coef=0.0`）
3. 跑一个 batch，检查标签值是否合理（范围、是否非负、是否有 NaN）

### Phase 3：训练实验

1. 设置 `enable_auxiliary=True`, `aux_tower_coef=0.1`, `aux_gold_coef=0.1`
2. 跑一轮 600 episodes 的训练，记录所有指标
3. 与 Baseline 对比

### Phase 4：评估与决策

1. 如果 WinRate 提升 → 保留，调优 coef
2. 如果持平但 aux_loss 下降 → 辅助头在学习，尝试调大 coef
3. 如果下降或 aux_loss 不变 → 放弃或转向 NOOP 模拟方案
