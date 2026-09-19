# PPO v1 — 结构化动作头（Structured Action Head）设计 v3

> 基于 `ppo/src/ppo_antwar` 重构后的代码编写。本文档反映 [v2 设计文档](ppo_v1_selfplay_ann_design_v2.md) 实施后的实际代码状态，与 v2 的关键差异在对应章节以「⚠️ v3 变更」标注。
>
> 核心文件：
> - [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py) — 网络结构（StructuredActionHead 实现）
> - [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py) — 动作空间常量 + TYPE_CONFIG
> - [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py) — 训练循环（未修改）

---

## 目录

1. [当前问题：为什么 `Linear(256, 96)` 是扁平动作头](#一当前问题为什么-linear256-96-是扁平动作头)
2. [核心思想](#二核心思想)
3. [详细设计](#三详细设计)
4. [entropy 坍缩根治的数学原理](#四-entropy-坍缩根治的数学原理)
5. [Action Mask 的映射处理](#五-action-mask-的映射处理)
6. [对 ε-greedy 探索的影响](#六对-ε-greedy-探索的影响)
7. [改动范围](#七改动范围)
8. [与当前方案的关系](#八与当前方案的关系)
9. [潜在风险与限制](#九潜在风险与限制)
10. [验证方案](#十验证方案)
11. [与 v2 设计文档的关键差异](#十一与-v2-设计文档的关键差异)

---

## 一、当前问题：为什么 `Linear(256, 96)` 是扁平动作头

当前代码 [antwar_net.py:L306](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L306)：

```python
self.policy_head = StructuredActionHead(hidden_dim)  # 原始：Linear(256, 96)
```

改造前的 `Linear(256, 96)` 是一个单一的全连接层，输出 96 维 logits。**"扁平"的含义是：模型把 96 个 action_id 视为完全独立的类别，它们之间没有任何结构性关系。**

### 问题 1：无法共享「类型级」知识

建塔（action_id 1–10）和升级塔（action_id 11–50）在游戏规则上高度相关——两者都涉及防御塔。但扁平头里这是两组不相关的参数：

```
W_policy shape = (256, 96)
    ┌──────────────────────────────────────────────────────┐
    │ w_{1,1}  w_{1,2}  ...  w_{1,10}  w_{1,11}  ...  │   建塔的权重
    │ w_{2,1}  w_{2,2}  ...  w_{2,10}  w_{2,11}  ...  │   和升级塔的权重
    │ ...                                               │   之间没有约束关系
    └──────────────────────────────────────────────────────┘
```

学到"在 (4,2) 建塔好"不能帮助模型理解"在 (4,2) 升级塔也好"。每个 target 都需要独立记忆。

### 问题 2：语义相关的动作在学习中互不关联

`action_id=11`（升级塔位#1）和 `action_id=16`（降级塔位#1）语义相反——一个增强塔，一个削弱塔。但扁平头里这是两个完全独立的输出神经元，模型需要从数据中独立学习"action 11 在什么情况下好"和"action 16 在什么情况下好"，无法利用它们之间的"语义相反"关系。

### 问题 3：越多的无效维度意味着越多的无用参数

53 个保留位（43–95）永远被 mask 为 0，对应的 `256 × 53 = 13,568` 个权重参数在 forward 中每次都参与计算但永不产生有效梯度。

---

## 二、核心思想

将 `Linear(256, 96)` **拆解为两层结构**：

```
当前：
    隐藏层 (256) ——Linear——> 96 个 flat logits

改造后：
    隐藏层 (256)
        │
        ├── 类型头 (type_head)     → 9 个类型 logits
        │
        └── 目标头 (target_heads)  → 每个类型一组独立的目标 logits

    最终动作 logit = type_logit + target_logit
```

### 数学模型

扁平头：

$$
\text{logit}(a_j) = W_j \cdot h + b_j \quad (\text{每个 action 独立参数})
$$

结构化头：

$$
\text{logit}(a_{t,k}) = \underbrace{W^{\text{type}}_t \cdot h + b^{\text{type}}_t}_{\text{类型 logit}} + \underbrace{W^{\text{target}}_t[k] \cdot h + b^{\text{target}}_t[k]}_{\text{目标 logit}}
$$

其中：
- $t$ = 动作类型（0–8，共 9 种）
- $k$ = 在该类型内的目标索引
- $W^{\text{type}}$ = 类型头权重 shape `(256, 9)`
- $W^{\text{target}}_t$ = 第 t 个目标头权重，各类型独立

**关键差异**：类型 logit **跨所有目标共享**。如果模型学到"建塔在当前是好选择"，这个知识通过类型头的 logit 传递到所有建塔目标上，即使某些塔位从未被建过也能获得合理的先验概率。

---

## 三、详细设计

### 3.1 类型配置定义

基于 [action_constants.py:L119-L129](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L119-L129)：

```python
TYPE_CONFIG: List[Dict] = [
    {'name': 'noop',              'flat_start': 0,   'flat_end': 1,    'max_targets': 1},
    {'name': 'build_tower',       'flat_start': 1,   'flat_end': 11,   'max_targets': 16},
    {'name': 'upgrade_tower',     'flat_start': 11,  'flat_end': 51,   'max_targets': 40},
    {'name': 'downgrade_tower',   'flat_start': 51,  'flat_end': 61,   'max_targets': 10},
    {'name': 'lightning_storm',   'flat_start': 61,  'flat_end': 66,   'max_targets': 5},
    {'name': 'emp_blaster',       'flat_start': 66,  'flat_end': 71,   'max_targets': 5},
    {'name': 'deflector',         'flat_start': 71,  'flat_end': 76,   'max_targets': 5},
    {'name': 'evasion',           'flat_start': 76,  'flat_end': 81,   'max_targets': 5},
    {'name': 'tech_upgrade',      'flat_start': 81,  'flat_end': 83,   'max_targets': 2},
]
```

| type_id | 类型名称 | flat range | 最大目标数 | 说明 |
|:-------:|:---------|:----------:|:---------:|:-----|
| 0 | NO_OP | 0–1 | 1 | 始终合法 |
| 1 | BUILD_TOWER | 1–11 | 16 | 最多 16 个塔位 |
| 2 | UPGRADE_TOWER | 11–51 | 40 | 10 塔 × 4 升级方向 |
| 3 | DOWNGRADE_TOWER | 51–61 | 10 | 最多 10 个己方塔 |
| 4 | LIGHTNING_STORM | 61–66 | 5 | 5 个投放位置 |
| 5 | EMP_BLASTER | 66–71 | 5 | 5 个投放位置 |
| 6 | DEFLECTOR | 71–76 | 5 | 5 个投放位置 |
| 7 | EMERGENCY_EVASION | 76–81 | 5 | 5 个投放位置 |
| 8 | TECH_UPGRADE | 81–83 | 2 | 两种科技 |

**注意**：目标头的输出维度 = `max_targets`（该类型的理论最大目标数），但实际有效的 target 数由 action_mask 动态决定，只有 `flat_end - flat_start` 个是有效的。这种"过配置"设计确保类型头可以处理游戏过程中目标数量变化的情况。

---

### 3.2 网络结构定义

实际代码 [antwar_net.py:L199-L258](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L199-L258)：

```python
class StructuredActionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.num_types = len(TYPE_CONFIG)                     # 9
        self.type_head = nn.Linear(hidden_dim, self.num_types) # (256, 9)

        target_dims = [config['max_targets'] for config in TYPE_CONFIG]
        self.target_heads = nn.ModuleList([                    # 每个类型独立
            nn.Linear(hidden_dim, dim) for dim in target_dims  # [1,16,40,10,5,5,5,5,2]
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.type_head.weight, gain=0.5)
        nn.init.zeros_(self.type_head.bias)
        for head in self.target_heads:
            if head.out_features <= 1:          # NO_OP
                nn.init.orthogonal_(head.weight, gain=0.01)
            else:
                nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.zeros_(head.bias)

    def forward(self, x):  # x: (B, hidden_dim)
        type_logits = self.type_head(x)           # (B, 9)

        B = x.size(0)
        flat_logits = torch.full((B, MAX_ACTIONS), fill_value=-1e9, device=x.device)

        for type_id, config in enumerate(TYPE_CONFIG):
            start = config['flat_start']
            end = config['flat_end']
            n_valid = end - start                 # 有效目标数
            target_logits = self.target_heads[type_id](x)
            target_logits_valid = target_logits[:, :n_valid]   # 截取有效部分
            flat_logits[:, start:end] = type_logits[:, type_id:type_id+1] + target_logits_valid

        return flat_logits                        # (B, 96)
```

#### ⚠️ v3 变更

与 v2 设计相比，实际实现有以下差异：
1. **`MAX_ACTIONS` 替代硬编码 `96`**：`torch.full((B, MAX_ACTIONS), ...)`
2. **使用 `flat_start`/`flat_end`**：与 `TYPE_CONFIG` 的 key 命名一致（v2 §3.2 的伪代码使用了 `flat_range` tuple）
3. **`self._init_weights()` 方法**：显式初始化方法，在构造器中自动调用，并在 `AntWarPolicyValueNetwork.__init__` 中再次调用以确保 `apply(orthogonal_init)` 后覆盖

---

### 3.3 `forward()` 的变化

#### ⚠️ v3 变更

v2 预期 `forward()` 接口完全不变。实际实现中，`forward()` 现在**内部调用了 `_encode()`**：

```python
def forward(self, board, global_features, action_mask=None):
    merged = self._encode(board, global_features)  # 编码器逻辑移入 forward 内部

    action_logits = self.policy_head(merged)        # StructuredActionHead → 96 logits
    value = self.value_head(merged)

    if action_mask is not None:
        action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

    return action_logits, value
```

**接口不变**。ppo_trainer.py 中的调用点 `self.policy(board, global_obs, action_mask)` 仍返回 `(logits, value)`。

但 `get_action()` 不再调用 `self.forward()`，而是**内联展开**以避免重复编码。

---

### 3.4 `get_action()` 的实际实现 — 单一编码 + 类型优先采样 + 分层 log_prob

实际代码 [antwar_net.py:L339-L423](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L339-L423)：

```
1. 编码：merged = self._encode(board, global_features)    ← 仅调用 1 次
2. 从 merged 同时计算：
   - action_logits = self.policy_head(merged)         → (B, 96) flat logits
   - value = self.value_head(merged)                  → (B, 1)
   - type_logits = self.policy_head.type_head(merged) → (B, 9)
3. 对 type_logits 和 action_logits 分别应用 mask
4. noise_std > 0 时对 action_logits 加高斯噪声（用于外部配置的探索噪声）

5. 分支选择：
   ├── ε-greedy（以 ε 概率）：从合法 flat 动作中均匀采样
   └── 类型优先采样（1-ε 概率）：
        ├── 在类型空间采样 → type_id（含 clamp + 重归一化保数值安全）
        └── 在被选类型的 target 空间采样 → target_k（含 clamp + 重归一化）

6. 分层 log_prob（ε-greedy 和类型优先采样后均执行）：
   ├── action_id → type + target 分解（action_id_to_type_and_target）
   ├── type_log_prob_selected = log_softmax(type_logits)[:, type_id]
   ├── target_logits → n_valid 截断 + mask 后 → log_softmax → target_log_prob_selected
   └── action_log_prob = type_log_prob_selected + target_log_prob_selected
```

核心代码：

```python
merged = self._encode(board, global_features)                # 仅 1 次编码

action_logits = self.policy_head(merged)                      # policy head
value = self.value_head(merged)                               # value head
type_logits = self.policy_head.type_head(merged)              # type head

# mask 处理
if action_mask is not None:
    type_mask = self.policy_head.flat_mask_to_type_mask(action_mask)
    type_logits = type_logits.masked_fill(type_mask == 0, float('-inf'))

if action_mask is not None:
    action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

# 可选探索噪声（外部通过 setattr 配置 logit_noise_std）
noise_std = getattr(self, 'logit_noise_std', 0.0)
if not deterministic and noise_std > 0:
    action_logits = action_logits + torch.randn_like(action_logits) * noise_std

eps = getattr(self, 'exploration_epsilon', 0.0)
if not deterministic and eps > 0 and random.random() < eps:
    action = valid_indices[torch.randint(...)]                 # ε-greedy 分支
else:
    # 类型优先采样
    if deterministic:
        type_id = torch.argmax(type_logits, dim=-1)
    else:
        type_probs = F.softmax(type_logits, dim=-1)
        type_probs = type_probs.clamp(min=1e-10)              # 数值安全
        type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)
        type_id = torch.multinomial(type_probs, 1)

    type_id_scalar = type_id.item()
    config = TYPE_CONFIG[type_id_scalar]
    n_valid = config['flat_end'] - config['flat_start']

    target_logits = self.policy_head.target_heads[type_id_scalar](merged)
    target_logits_valid = target_logits[:, :n_valid]           # 截断有效维度

    if action_mask is not None:
        target_mask = action_mask[:, config['flat_start']:config['flat_end']]
        target_logits_valid = target_logits_valid.masked_fill(target_mask == 0, float('-inf'))

    if deterministic:
        target_k = torch.argmax(target_logits_valid, dim=-1)
    else:
        target_probs = F.softmax(target_logits_valid, dim=-1)
        target_probs = target_probs.clamp(min=1e-10)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
        target_k = torch.multinomial(target_probs, 1)

    action = flat_start + target_k

# 分层 log_prob
type_ids, target_ks = action_id_to_type_and_target(action)
type_id_0 = type_ids.item()
target_k_0 = target_ks.item()

type_log_prob = F.log_softmax(type_logits, dim=-1)
type_log_prob_selected = type_log_prob[:, type_id_0:type_id_0+1]

config = TYPE_CONFIG[type_id_0]
n_valid = config['flat_end'] - config['flat_start']
target_logits_0 = self.policy_head.target_heads[type_id_0](merged)
target_logits_0 = target_logits_0[:, :n_valid]                # 截断
if action_mask is not None:
    target_mask_0 = action_mask[:, config['flat_start']:config['flat_end']]
    target_logits_0 = target_logits_0.masked_fill(target_mask_0 == 0, float('-inf'))  # mask
target_log_prob = F.log_softmax(target_logits_0, dim=-1)
target_log_prob_selected = target_log_prob[:, target_k_0:target_k_0+1]

action_log_prob = (type_log_prob_selected + target_log_prob_selected).squeeze(-1)
```

#### ⚠️ v3 变更

1. **消除双编码**：v2 伪代码中先调 `self.forward()` 又调 `self._encode()` 导致两次编码；实际代码仅调用一次 `_encode()`，然后从 merged 计算所有 logits
2. **分层 log_prob**：v2 伪代码中 `get_action` 仍然使用扁平 log_prob（`log_prob.gather(1, action)`）；实际代码使用分层 log_prob（`type_log_prob + target_log_prob`），与 `evaluate_actions` 一致
3. **`action_id_to_type_and_target()`**：v2 设计为拆分函数但伪代码未体现；实际实现在 [antwar_net.py:L260-L269](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L260-L269)

---

### 3.5 `evaluate_actions()` 的实际实现 — 分层 log_prob + 分层 entropy

实际代码 [antwar_net.py:L425-L497](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L425-L497)：

**分层 log_prob**：

```python
# 拆解 action → type_id + target_k
type_ids, target_ks = action_id_to_type_and_target(actions.squeeze(-1))

# 类型级 log_prob（应用 type_mask）
type_logits = self.policy_head.type_head(merged)
if action_mask is not None:
    type_mask = self.policy_head.flat_mask_to_type_mask(action_mask)
    type_logits = type_logits.masked_fill(type_mask == 0, float('-inf'))
type_log_prob = F.log_softmax(type_logits, dim=-1)
type_log_prob_selected = type_log_prob.gather(1, type_ids.unsqueeze(-1)).squeeze(-1)

# 目标级 log_prob（逐类型计算，仅计算实际出现的类型）
target_log_prob_selected = torch.zeros_like(type_log_prob_selected)
for type_id_val in range(len(TYPE_CONFIG)):
    mask = (type_ids == type_id_val)
    if not mask.any():
        continue
    config = TYPE_CONFIG[type_id_val]
    n_valid = config['flat_end'] - config['flat_start']
    target_logits_t = self.policy_head.target_heads[type_id_val](merged)
    target_logits_t = target_logits_t[:, :n_valid]            # 截断有效维度
    if action_mask is not None:
        target_mask = action_mask[:, config['flat_start']:config['flat_end']]
        target_logits_t = target_logits_t.masked_fill(target_mask == 0, float('-inf'))
    t_log_prob = F.log_softmax(target_logits_t, dim=-1)
    target_log_prob_selected[mask] = t_log_prob.gather(1, target_ks[mask].unsqueeze(-1)).squeeze(-1)

# 总 log_prob
action_log_prob = type_log_prob_selected + target_log_prob_selected
```

**分层 entropy**：

```python
type_probs = torch.exp(type_log_prob)
type_probs = type_probs.clamp(min=1e-10)                     # 数值安全，防 0 * -inf = NaN
type_log_prob = torch.log(type_probs)
type_entropy = -(type_probs * type_log_prob).sum(dim=-1)     # 类型级熵

target_entropy_sum = torch.zeros_like(type_entropy)
for type_id, config in enumerate(TYPE_CONFIG):
    start = config['flat_start']
    n_valid = config['flat_end'] - start

    # 重算 target_logits（不可复用 log_prob 循环中的变量，因 batch 中多类型交叉时需独立计算）
    target_logits_t = self.policy_head.target_heads[type_id](merged)
    target_logits_t = target_logits_t[:, :n_valid]            # 截断有效维度

    if action_mask is not None:
        target_mask = action_mask[:, start:config['flat_end']]
        target_logits_t = target_logits_t.masked_fill(target_mask == 0, float('-inf'))

    target_log_prob_t = F.log_softmax(target_logits_t, dim=-1)
    target_probs_t = torch.exp(target_log_prob_t)

    target_probs_t = torch.nan_to_num(target_probs_t, nan=0.0)     # 处理全部 mask 时 log_softmax 产生的 NaN
    target_probs_t = target_probs_t.clamp(min=1e-10)               # 数值安全
    target_log_prob_t = torch.log(target_probs_t)

    target_entropy_t = -(target_probs_t * target_log_prob_t).sum(dim=-1)
    target_entropy_sum += type_probs[:, type_id] * target_entropy_t  # 按类型概率加权

dist_entropy = type_entropy + target_entropy_sum              # 总熵
```

#### ⚠️ v3 变更

1. **mask 应用位置**：v2 伪代码（附录 A.4）在 softmax 后用 `type_probs = type_probs * type_mask` 做 mask；实际代码在 **softmax 之前**对 type_logits 直接 `masked_fill(type_mask == 0, -inf)`。后者是正确的做法——logits 在 softmax 前 mask 确保正规化时不计入非法类型
2. **目标头循环**：`evaluate_actions` 中 log_prob 和 entropy 各循环 9 次（实际只有 `max(type_count_with_data)` 次有效计算）
3. **`type_entropy + target_entropy_sum` 加总**：符合 v2 公式 $H_{\text{total}} = H_{\text{type}} + \sum \pi_{\text{type}}(t) \cdot H_{\text{target}}^t$

---

## 四、Entropy 坍缩根治的数学原理

当前 PPO loss 中的 entropy bonus：

$$
L = L^{\text{CLIP}} + c \cdot H(\pi(\cdot|s))
$$

其中 $c$ 从 0.2 退火到 0.05（[yaml L17-L19](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L17-L19)）。

当策略分布退化到 one-hot 时：

- **扁平头**：$H(\pi) = 0$，$\frac{\partial H}{\partial \theta} = 0$，entropy bonus 完全失效
- **结构化头**：

  $$
  H(\pi) = H(\pi_{\text{type}}) + \sum_t \pi_{\text{type}}(t) \cdot H(\pi_{\text{target}}^t)
  $$

  即使类型完全确定（$H(\pi_{\text{type}}) = 0$），只要目标级还有不确定性，整体熵就 > 0。

更关键地，entropy 的梯度不会消失：

$$
\frac{\partial H}{\partial \theta} = \frac{\partial H_{\text{type}}}{\partial \theta} + \sum_t \left[ \frac{\partial \pi_{\text{type}}(t)}{\partial \theta} \cdot H_{\text{target}}^t + \pi_{\text{type}}(t) \cdot \frac{\partial H_{\text{target}}^t}{\partial \theta} \right]
$$

即使 $\frac{\partial H_{\text{type}}}{\partial \theta} \to 0$，只要存在任何 $\pi_{\text{type}}(t) > 0$ 且 $H_{\text{target}}^t > 0$，梯度就持续存在。

```
熵坍缩的对比：

扁平头：                       结构化头：
H_total = 0.0                  H_type = 0.0 (确定只要建塔)
                                H_target = 1.2 (还在探索建在哪)
                                ─────────────────
                                H_total = 1.2  ← 仍有 gradient

扁平头一旦坍缩为 0，             类型确定后目标级仍在探索，
entropy gradient 消失。        总 entropy > 0，梯度持续。
```

---

## 五、Action Mask 的映射处理

当前 action_mask 是 96 维 flat 的，结构化头需要将其分解为**类型级 mask** 和**目标级 mask**。

### 5.1 类型级 mask

实际代码 [antwar_net.py:L240-L250](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L240-L250)：

```python
@staticmethod
def flat_mask_to_type_mask(flat_mask: torch.Tensor) -> torch.Tensor:
    B = flat_mask.size(0)
    type_mask = torch.zeros(B, len(TYPE_CONFIG), device=flat_mask.device)
    for type_id, config in enumerate(TYPE_CONFIG):
        start = config['flat_start']
        end = config['flat_end']
        has_legal = flat_mask[:, start:end].any(dim=1)      # 类型 t 合法 ⟺ 存在任意合法目标
        type_mask[:, type_id] = has_legal.float()
    return type_mask
```

### 5.2 目标级 mask

实际代码 [antwar_net.py:L252-L257](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L252-L257)：

```python
@staticmethod
def flat_mask_to_target_mask(flat_mask: torch.Tensor, type_id: int) -> torch.Tensor:
    config = TYPE_CONFIG[type_id]
    start = config['flat_start']
    end = config['flat_end']
    return flat_mask[:, start:end]                          # 直接截取对应区间
```

### 5.3 动作拆解

实际代码 [antwar_net.py:L260-L269](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L260-L269)：

```python
def action_id_to_type_and_target(action_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    type_ids = torch.zeros_like(action_ids)
    target_ks = torch.zeros_like(action_ids)
    for type_id, config in enumerate(TYPE_CONFIG):
        start = config['flat_start']
        end = config['flat_end']
        in_range = (action_ids >= start) & (action_ids < end)
        type_ids[in_range] = type_id
        target_ks[in_range] = action_ids[in_range] - start    # 减去 flat_start 得到 target_k
    return type_ids, target_ks
```

---

## 六、对 ε-greedy 探索的影响

实际实现 [antwar_net.py:L363-L402](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L363-L402) 采用 **v2 推荐策略 1（flat ε-greedy）**：

```python
eps = getattr(self, 'exploration_epsilon', 0.0)
if not deterministic and eps > 0 and random.random() < eps:
    if action_mask is not None:
        valid_indices = torch.where(action_mask[0] > 0)[0]
        action = valid_indices[torch.randint(len(valid_indices), (1,))].unsqueeze(0)
        action = action.squeeze(0)
    else:
        action = torch.randint(0, action_logits.size(-1), (1,), device=action_logits.device)
```

**语义不变**：5% 的概率在全部合法动作中均匀随机选。ε-greedy 路径后的 log_prob 计算同样使用分层方式。

---

## 七、改动范围

| 文件 | 改动内容 | 行数 |
|:-----|:---------|:----:|
| **`antwar_net.py`** | 新增 `StructuredActionHead` 类（含 type_head + 9 个 target_heads + forward + mask 映射 + `_init_weights`） | ~60 行 |
| **`antwar_net.py`** | `AntWarPolicyValueNetwork.__init__` 替换 `self.policy_head` | ~5 行 |
| **`antwar_net.py`** | 提取 `_encode()` 方法（forward 内部逻辑分离） | ~6 行 |
| **`antwar_net.py`** | `forward()` 接口不变，内部逻辑重构 | ~10 行 |
| **`antwar_net.py`** | `get_action()` 重写为单一编码 + 类型优先采样 + 分层 log_prob | ~60 行 |
| **`antwar_net.py`** | `evaluate_actions()` 重写为分层 log_prob + 分层 entropy | ~70 行 |
| **`antwar_net.py`** | 新增模块级函数 `action_id_to_type_and_target()` | ~10 行 |
| **`action_constants.py`** | 新增 `TYPE_CONFIG` 结构化配置 | ~11 行 |
| **`ppo_antwar.yaml`** | 无变化 | 0 行 |
| **`ppo_trainer.py`** | 无变化 | 0 行 |
| **`selfplay.py`** | 无变化 | 0 行 |
| **Checkpoint** | ❌ 不兼容（policy_head 权重名字和 shape 完全改变） | — |

**总计：约 232 行新增/修改代码，涉及 2 个文件，其他文件无需修改。**

---

## 八、与当前方案的关系

结构化动作头与之前讨论的"增加采局数"方案**正交**：

| 方向 | 解决的问题 | 优化性质 |
|:-----|:----------|:--------:|
| 增加采局数（games_per_update=4） | 数据量不足 → 更新方差大 | **程度优化** |
| 结构化动作头 | 扁平动作空间 → 熵坍缩 | **结构性优化** |

两种可以独立实施，互不冲突。

---

## 九、潜在风险与限制

### 风险 1：target_head 过参数化

UPGRADE_TOWER 的 target_head 输出 40 维（10 塔 × 4 方向），但实际每局游戏中通常只有 1-3 个塔可升级。这导致 37+ 个输出维度被 mask 掉。但跟 flat head 的 53 个保留位不同——这些维度对应的是**可能在未来游戏中出现**的目标，不是"永远无用"的。

### 风险 2：mask 映射正确性依赖 TYPE_CONFIG 与 SDK ActionBundle 顺序一致性

如果 SDK 升级改变了 ActionBundle 的排列顺序，结构化头的类型分解会出错。

### 风险 3：极端 mask 场景下的采样表现

类型优先采样在某些极端 mask 场景下可能表现不如 flat 采样。例如当某个类型的部分目标被 mask、部分未被 mask 时，类型级 mask 只能做 binary 判断（合法/不合法）。

### 风险 4：分层 entropy 加总的偏差

`evaluate_actions` 中的分层 entropy 计算采用：

$$
H_{\text{total}} = H_{\text{type}} + \sum_t \pi_{\text{type}}(t) \cdot H_{\text{target}}^t
$$

当某些类型的目标数差异大时（UPGRADE 有 40 个目标，NO_OP 只有 1 个），直接的 entropy 加总可能引入偏差。备选方案：乘以类型权重因子 `weight_t = log(n_targets_t)` 进行归一化。

---

## 十、验证方案

改造后，与 flat head 对比训练，验证指标：

### 主要指标

| 指标 | 目标值 | 说明 |
|:----|:-----:|:-----|
| 策略熵保持 | ≥ 0.5 | 核心指标，熵不坍缩即成功 |
| 训练胜率收敛性 | ≥ flat head 水平 | 不应因结构化而降低 |
| Baseline 对抗胜率 | ≥ flat head 水平 | 最终对战效果 |

### 辅助指标

| 指标 | 说明 |
|:----|:-----|
| type_entropy 日志 | 训练中记录类型级 entropy，便于诊断 |
| target_entropy 日志 | 训练中记录目标级 entropy，便于诊断 |
| log_prob(type) 分布 | 验证分层 log_prob 的计算正确性 |
| type 选择频率 | 各类动作的被选频率分布 |

### 诊断策略

如果结构化头改造后熵仍然下降过快，按以下顺序排查：

1. 检查 `TYPE_CONFIG` 的 `flat_range` 是否与 SDK `ActionCatalog.build()` 生成的 ActionBundle 顺序一致
2. 检查 type_mask 生成逻辑是否正确（flat_mask 到 type_mask 的映射）
3. 检查类型优先采样是否真的按 `type_logits → target_logits` 的顺序采样
4. 确认 `evaluate_actions` 中的分层 entropy 加总方式无误

---

## 十一、与 v2 设计文档的关键差异

以下列出现有代码与 [v2 设计文档](ppo_v1_selfplay_ann_design_v2.md) 之间的主要差异：

| # | 方面 | v2 设计 | v3 代码实现 | 理由 |
|:-:|:-----|:--------|:-----------|:------|
| 1 | `get_action` 编码调用 | 先 `self.forward()`（含编码），再 `self._encode()`（双编码） | 单一 `self._encode()`，内联展开所有 head 计算 | 消除 50% 策略推理计算量浪费 |
| 2 | `get_action` log_prob | 扁平 softmax（`log_prob.gather(1, action)`） | 分层 log_prob（`type_log_prob + target_log_prob`） | 与采样分布一致；有 mask 时扁平 ≠ 分层 |
| 3 | `evaluate_actions` log_prob | 分层 log_prob（附录 A.4） | 分层 log_prob | 与 `get_action` 统一，PPO ratio 正确 |
| 4 | `evaluate_actions` mask 应用 | softmax 后 `type_probs * type_mask` | softmax 前 `logits.masked_fill(-inf)` | 标准做法，确保非法类型不计入正规化 |
| 5 | `flat_mask_to_*` 命名 | `_flat_mask_to_*`（private 命名） | `flat_mask_to_*`（public `@staticmethod`） | 命名简化 |
| 6 | `StructuredActionHead.forward` | 硬编码 `(B, 96)` | `(B, MAX_ACTIONS)` | 常量引用，避免魔数 |
| 7 | `TYPE_CONFIG` key | 伪代码使用 `flat_range`（tuple） | 使用 `flat_start`/`flat_end`（独立 key） | 与附录 A.1 一致 |
| 8 | `forward()` 内部结构 | 不调 `_encode()`，直接调用 `self.policy_head(merged)` | 调 `self._encode(board, global_features)` | 接口不变，内部重构 |
| 9 | `action_id_to_type_and_target` | 设计为两个独立函数 | 一个联合函数返回 tuple | 更紧凑 |
| 10 | 参数初始化 | 未详细指定 | `orthogonal_init(gain=0.5)` + NO_OP 特殊处理（gain=0.01） | 更精细的初始化策略 |

### 差异原因总结

- **#1, #2, #3**：代码审查中发现的设计缺陷修复（双编码性能问题 + 扁平/分层 log_prob 不一致）
- **#4**：代码审查中发现的 mask 应用位置修正
- **#5, #6, #7**：代码实现中的命名和细节优化
- **#8, #9, #10**：实现过程中的自然演化和完善

---

## 附录 A：关键代码索引

| 组件 | 文件 | 行号 |
|:-----|:-----|:----:|
| `StructuredActionHead` 类定义 | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L199-L258) | L199–258 |
| `StructuredActionHead.forward()` | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L223-L238) | L223–238 |
| `flat_mask_to_type_mask` | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L240-L250) | L240–250 |
| `flat_mask_to_target_mask` | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L252-L257) | L252–257 |
| `action_id_to_type_and_target` | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L260-L269) | L260–269 |
| `StructuredActionHead` 初始化配置 | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L213-L221) | L213–221 |
| `_encode()` 编码器 | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L316-L321) | L316–321 |
| `forward()` | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L323-L337) | L323–337 |
| `get_action()` 类型优先采样 + 分层 log_prob | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L339-L423) | L339–423 |
| `evaluate_actions()` 分层 log_prob + 分层 entropy | [antwar_net.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/network/antwar_net.py#L425-L497) | L425–497 |
| `TYPE_CONFIG` 定义 | [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L119-L129) | L119–129 |
| `ACTION_SPACE_CONFIG` | [action_constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/utils/action_constants.py#L107-L117) | L107–117 |
| ent_coef 配置 | [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml#L17-L19) | L17–19 |
