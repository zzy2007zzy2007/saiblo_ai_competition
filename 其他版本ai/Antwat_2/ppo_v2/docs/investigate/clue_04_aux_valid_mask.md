# 线索 04：辅助损失标签截断未过滤无效步

**调查日期**：2026-06-11
**结论**：**确认问题存在**

---

## 1. 问题描述

`compute_aux_labels_from_trajectory()` 中最后 `max(AUX_HORIZON_STEPS)=16` 步没有有效标签，`_attach_aux_labels()` 对这些步用零填充并设置 `aux_valid_mask=False`。但 `compute_auxiliary_loss()` 中直接使用 `F.mse_loss(pred, label)` 计算，没有使用 `aux_valid_mask` 过滤无效步，导致末尾步的零标签被当作真实标签训练，产生错误的梯度信号。

---

## 2. 逐文件代码审查

### 2.1 辅助标签生成：`utils/aux_labels.py`

**文件路径**：`ppo_v2/src/ppo_antwar/utils/aux_labels.py`

**关键代码**（第 28-36 行）：

```python
max_horizon = max(AUX_HORIZON_STEPS)  # = 16
n = len(step_aux_data)
T = n - max_horizon
if T <= 0:
    return (
        np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
        np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
        np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
    )
```

**分析**：函数只返回前 `T = n - 16` 步的标签，最后 16 步的标签无法计算（因为需要未来 16 步的数据），所以不在输出数组中。输出数组形状为 `(T, 10)`，而非 `(n, 10)`。

**确认**：标签生成逻辑正确，最后 16 步确实没有有效标签。

---

### 2.2 标签填充与掩码设置：`trainer/episode_collector.py`

**文件路径**：`ppo_v2/src/ppo_antwar/trainer/episode_collector.py`

**关键代码**（第 324-348 行）：

```python
@staticmethod
def _attach_aux_labels(batch: EpisodeBatch, step_aux_data: List[Dict]):
    """从 per-round 辅助数据计算累积标签并附加到 batch。"""
    tower_labels, gold_labels, base_labels = compute_aux_labels_from_trajectory(step_aux_data)
    if tower_labels.shape[0] > 0:
        num_steps = len(batch)
        valid_count = tower_labels.shape[0]
        if valid_count < num_steps:
            padded_tower = np.zeros((num_steps, 10), dtype=np.float32)
            padded_gold = np.zeros((num_steps, 10), dtype=np.float32)
            padded_base = np.zeros((num_steps, 10), dtype=np.float32)
            padded_tower[:valid_count] = tower_labels
            padded_gold[:valid_count] = gold_labels
            padded_base[:valid_count] = base_labels
            batch.aux_tower_damage = padded_tower
            batch.aux_gold_income = padded_gold
            batch.aux_base_damage = padded_base
        else:
            valid_count = num_steps
            batch.aux_tower_damage = tower_labels
            batch.aux_gold_income = gold_labels
            batch.aux_base_damage = base_labels
        mask = np.zeros(num_steps, dtype=bool)
        mask[:valid_count] = True
        batch.aux_valid_mask = mask
```

**分析**：
- 当 `valid_count < num_steps` 时（即最后 16 步），用零填充标签数组至完整轨迹长度
- 正确设置了 `aux_valid_mask`：前 `valid_count` 步为 `True`，后 `num_steps - valid_count` 步为 `False`
- 掩码逻辑正确，但关键问题在于下游是否使用了这个掩码

**确认**：`aux_valid_mask` 被正确生成，标记了哪些步有有效标签。

---

### 2.3 张量传递：`trainer/batch.py`

**文件路径**：`ppo_v2/src/ppo_antwar/trainer/batch.py`

**关键代码**（第 87-89 行）：

```python
tensors["aux_valid_mask"] = _safe_tensor(
    self.aux_valid_mask, torch.bool, device
)
```

**分析**：`to_tensors()` 方法确实将 `aux_valid_mask` 转换为 tensor 并放入返回字典中。

**确认**：`aux_valid_mask` 被传递到了 tensors 字典中。

---

### 2.4 Minibatch 准备：`trainer/ppo_trainer.py`

**文件路径**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py`

**关键代码**（第 308-332 行）：

```python
@staticmethod
def _prepare_minibatch(tensors, advantages_norm, returns, mb_indices):
    mb_board = tensors["board"][mb_indices]
    mb_global = tensors["global"][mb_indices]
    mb_mask = tensors["action_mask"][mb_indices]
    mb_actions = tensors["actions"][mb_indices]
    mb_old_log_probs = tensors["log_probs"][mb_indices]
    mb_advantages_norm = advantages_norm[mb_indices]
    mb_returns = returns[mb_indices]

    return {
        "board": mb_board,
        "global": mb_global,
        "mask": mb_mask,
        "actions": mb_actions,
        "old_log_probs": mb_old_log_probs,
        "advantages_norm": mb_advantages_norm,
        "returns": mb_returns,
        "returns_raw": mb_returns,
        "advantages_raw": mb_advantages_norm,
        "old_values": tensors["values"][mb_indices],
        "aux_tower_damage": tensors["aux_tower_damage"][mb_indices],
        "aux_gold_income": tensors["aux_gold_income"][mb_indices],
        "aux_base_damage": tensors["aux_base_damage"][mb_indices],
    }
```

**分析**：`_prepare_minibatch()` 提取了 `aux_tower_damage`、`aux_gold_income`、`aux_base_damage`，但**完全没有提取 `aux_valid_mask`**。这是问题的关键断裂点——掩码数据在 `to_tensors()` 中被正确传递，但在构建 minibatch 时被丢弃了。

**确认**：`aux_valid_mask` 在 minibatch 准备阶段被丢弃，**问题确认存在**。

---

### 2.5 辅助损失计算：`network/ant_war_policy_value_network.py`

**文件路径**：`ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py`

**关键代码**（第 335-415 行）：

```python
def compute_auxiliary_loss(
    self,
    features: torch.Tensor,
    aux_tower_damage: torch.Tensor,
    aux_gold_income: torch.Tensor,
    aux_base_damage: Optional[torch.Tensor] = None,
    value_features: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # ...
    tower_pred = self.tower_damage_head(features)  # (batch, 10)
    gold_pred = self.gold_income_head(features)  # (batch, 10)

    mid = tower_pred.size(1) // 2  # 5

    aux_tower_loss = F.mse_loss(tower_pred[:, :mid], aux_tower_damage[:, :mid])
    aux_enemy_tower_loss = F.mse_loss(
        tower_pred[:, mid:], aux_tower_damage[:, mid:]
    )
    aux_gold_loss = F.mse_loss(gold_pred[:, :mid], aux_gold_income[:, :mid])
    aux_enemy_gold_loss = F.mse_loss(gold_pred[:, mid:], aux_gold_income[:, mid:])
    # ... 同样模式用于 base_damage 和 value encoder 的辅助头
```

**分析**：
- `compute_auxiliary_loss()` 的参数列表中**没有 `aux_valid_mask`**
- 所有 `F.mse_loss()` 调用都是直接对整个 batch 计算，**没有过滤无效步**
- 这意味着末尾 16 步的零标签被当作真实标签参与 MSE 计算

**确认**：损失计算完全忽略了 `aux_valid_mask`，**问题确认存在**。

---

### 2.6 辅助损失添加：`trainer/ppo_trainer.py`

**文件路径**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py`

**关键代码**（第 631-696 行）：

```python
def _add_auxiliary_loss(
    self,
    total_loss: torch.Tensor,
    mb_data: Dict[str, torch.Tensor],
    aux_preds: Optional[Dict],
    ppo_cfg: Dict[str, Any],
    features: torch.Tensor,
    value_features: Optional[torch.Tensor],
    enable_auxiliary: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not enable_auxiliary or aux_preds is None:
        return total_loss, {}
    aux_base_damage = mb_data.get("aux_base_damage")
    aux_losses = self.policy.compute_auxiliary_loss(
        features,
        mb_data["aux_tower_damage"],
        mb_data["aux_gold_income"],
        aux_base_damage=aux_base_damage,
        value_features=value_features,
    )
    # ... 系数加权后加入 total_loss
```

**分析**：`_add_auxiliary_loss()` 调用 `compute_auxiliary_loss()` 时也没有传递 `aux_valid_mask`（因为 `mb_data` 中根本就不包含它）。

**确认**：辅助损失添加流程完全缺失掩码过滤，**问题确认存在**。

---

## 3. 影响评估

### 3.1 末尾无效步占比

- `AUX_HORIZON_STEPS = [1, 2, 4, 8, 16]`，`max_horizon = 16`
- 假设平均 episode 长度 250 步
- 无效步数 = 16
- **占比 = 16 / 250 = 6.4%**

### 3.2 零标签对 MSE Loss 的贡献估算

以塔伤害为例：
- 有效步的标签值：取决于游戏状态，通常为正数（伤害累积量）
- 无效步的标签值：**全零**（零填充）
- 网络预测值：对于末尾步，网络会输出非零预测（因为输入特征与有效步类似）

假设：
- 网络对末尾步的塔伤害预测值约为 `p`（与有效步同量级）
- 有效步的平均 MSE ≈ `σ²`（正常方差）
- 无效步的 MSE ≈ `p²`（因为标签为 0，MSE = pred²）

如果 `p` 与有效步标签的均值量级相当，则无效步的 MSE 贡献可能与有效步相当甚至更大（因为有效步的 pred 和 label 接近，MSE 较小；而无效步的 pred 远离 0，MSE 较大）。

**粗略估算**：无效步占总 loss 的比例可能远超 6.4%，因为零标签与网络预测的偏差可能大于有效步的偏差。极端情况下，如果网络对末尾步的预测与有效步类似，则无效步的 MSE 贡献可达 **10-20%**。

### 3.3 对策略学习的影响

1. **梯度方向错误**：零标签会推动辅助头的预测向零收敛，但末尾步的真实标签不应为零。这导致辅助头在末尾步产生错误的梯度信号。

2. **特征表示污染**：辅助任务的核心目的是通过多任务学习改善编码器的特征表示。错误的辅助梯度会污染 policy encoder 和 value encoder 的特征学习，使编码器在末尾步的特征提取能力下降。

3. **策略偏向**：如果辅助损失系数较大（如 `aux_tower_coef`、`aux_gold_coef` 等），错误的辅助梯度可能通过共享的编码器影响策略梯度的方向，导致策略在游戏末尾阶段做出次优决策。

### 3.4 对价值学习的影响

- Value encoder 也有独立的辅助头（`value_tower_damage_head` 等），同样受零标签影响
- 价值网络的辅助任务本应帮助编码器学习更好的状态表示，但零标签会引入噪声
- 由于 value encoder 的梯度独立于 policy encoder，影响可能相对可控，但仍然降低了辅助任务对价值估计的正向贡献

---

## 4. 问题链路总结

```
compute_aux_labels_from_trajectory()  →  输出 (T, 10)，T = n-16，最后 16 步无标签
         ↓
_attach_aux_labels()  →  零填充至 (n, 10)，设置 aux_valid_mask[:valid]=True, [valid:]=False
         ↓
EpisodeBatch.aux_valid_mask  →  正确存储
         ↓
to_tensors()  →  tensors["aux_valid_mask"] 正确传递
         ↓
_prepare_minibatch()  →  ❌ 未提取 aux_valid_mask，掩码丢失
         ↓
_add_auxiliary_loss()  →  ❌ 未传递 aux_valid_mask
         ↓
compute_auxiliary_loss()  →  ❌ 不接受 aux_valid_mask 参数，F.mse_loss 全量计算
         ↓
末尾 16 步零标签被当作真实标签训练  →  错误梯度信号
```

---

## 5. 结论

**确认问题存在**。`aux_valid_mask` 在数据生成和存储阶段被正确处理，但在训练流程的两个关键环节被丢弃：

1. **`_prepare_minibatch()`**（ppo_trainer.py 第 308-332 行）：未将 `aux_valid_mask` 加入 minibatch 数据字典
2. **`compute_auxiliary_loss()`**（ant_war_policy_value_network.py 第 335-415 行）：不接受也不使用 `aux_valid_mask`，直接对全量数据计算 MSE

---

## 6. 修复建议

### 6.1 修改 `_prepare_minibatch()` 传递掩码

在 `ppo_trainer.py` 的 `_prepare_minibatch()` 返回字典中添加：

```python
"aux_valid_mask": tensors["aux_valid_mask"][mb_indices],
```

### 6.2 修改 `compute_auxiliary_loss()` 接受并使用掩码

在 `ant_war_policy_value_network.py` 的 `compute_auxiliary_loss()` 中：

1. 添加 `aux_valid_mask: Optional[torch.Tensor] = None` 参数
2. 将所有 `F.mse_loss(pred, label)` 替换为掩码版本：

```python
if aux_valid_mask is not None:
    # 仅对有效步计算 MSE
    valid_pred = pred[aux_valid_mask]
    valid_label = label[aux_valid_mask]
    if valid_pred.numel() > 0:
        loss = F.mse_loss(valid_pred, valid_label)
    else:
        loss = torch.tensor(0.0, device=pred.device)
else:
    loss = F.mse_loss(pred, label)
```

### 6.3 修改 `_add_auxiliary_loss()` 传递掩码

在 `ppo_trainer.py` 的 `_add_auxiliary_loss()` 中，将 `aux_valid_mask` 传递给 `compute_auxiliary_loss()`：

```python
aux_valid_mask = mb_data.get("aux_valid_mask")
aux_losses = self.policy.compute_auxiliary_loss(
    features,
    mb_data["aux_tower_damage"],
    mb_data["aux_gold_income"],
    aux_base_damage=aux_base_damage,
    value_features=value_features,
    aux_valid_mask=aux_valid_mask,
)
```

### 6.4 验证方案

修复后，可通过以下方式验证：
1. 对比修复前后 20 episode 的 `aux_tower_loss`、`aux_gold_loss` 等指标，修复后应有所下降（因为排除了无效步的噪声贡献）
2. 检查末尾步的辅助头预测值是否不再被强制向零收敛
3. 观察训练曲线是否更稳定
