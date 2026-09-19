# PPO 训练参数调整报告

> 数据来源：server9 `20260609_000042` run, EpisodeBatch 窗口 E20、E40
> 基准: 20 envs, batch_size=1024, vs BasicTowerAI/BasicRandomAI/MediumRuleAI

---

## 一、当前指标分析

### 1.1 训练指标

```
Epid   PLoss    VLoss       TwrL     GldL    BaseL   EBaseL   Entropy  ClipFrac  GradNorm
E20    0.115    4796.0      87.2     925.3   3.22    3.07     0.461    0.874     1.41
E40    0.066    4529.3      86.4     956.5   2.15    2.14     0.268    0.098     0.54
```

### 1.2 业务指标

```
Epid   Rounds  Reward  WinRate
E20    249.2   368.6   55%
E40    251.9   377.5   55%
```

### 1.3 问题诊断

| 指标 | 观测值 | 问题 | 严重度 |
|---|---|---|---|
| VLoss | 4500-4800 | 值过大，即使 × vf_coef(0.001) 也贡献 ~4.5，远超 PLoss(0.1) | 🔴 严重 |
| ClipFrac | E20=0.874 | 87% 的 policy 更新被 clip，policy 变化过于剧烈 | 🟡 中等 |
| Entropy | 0.46→0.27 | 20 个 batch 内下降 42%，收敛过快 | 🟡 中等 |
| TwrL | 87 | 原始 MSE 大（标签值域 0-500），coef 压到 ~0.0003 | 🟢 预期内 |
| GldL | 925 | 同上，值域 0-250，coef 压到 ~0.001 | 🟢 预期内 |
| BaseL | 2-3 | 值域 0-5，MSE 合理 | 🟢 正常 |

---

## 二、参数调整原则框架

### 2.1 调整目标

1. **稳定性**：ClipFrac 保持在 0.05-0.20，Entropy 稳定在 0.15-0.40
2. **收敛效率**：Reward/WinRate 单调上升，PLoss 在 0.01-0.20 波动
3. **损失比例协调**：各 loss 分量的加权贡献保持在合理数量级

### 2.2 各损失分量的理想加权贡献比例

以 PLoss 为基准（约 0.1），其他分量目标：

| 分量 | 原始值域 | 目标加权贡献 | 推导 coef 范围 |
|---|---|---|---|
| PLoss | 0.01-0.20 | 1×（基准） | - |
| VLoss | 1000-5000 | 0.5-2× | `vf_coef × VLoss / PLoss ≈ 0.00001-0.0004` |
| TwrL | 20-200 | 0.01-0.05× | `coef × TwrL / PLoss ≈ 5e-6-2.5e-4` |
| GldL | 200-2000 | 0.01-0.05× | `coef × GldL / PLoss ≈ 5e-7-2.5e-5` |
| BaseL | 0.5-10 | 0.01-0.05× | `coef × BaseL / PLoss ≈ 1e-4-5e-3` |

### 2.3 调整方法

1. **测量阶段**：收集 3-5 个窗口的正常训练数据，计算各 loss 分量的均值和方差
2. **校准阶段**：调整 coef，使各分量的加权贡献落在目标比例区间
3. **验证阶段**：跑 5-10 个窗口，检查 ClipFrac、Entropy 是否稳定

### 2.4 验证流程

```
窗口 1-3  → 观察基线
窗口 4-6  → 调整后数据，对比 ClipFrac / Entropy / WinRate 趋势
窗口 7+   → 稳定后确认
```

---

## 三、具体参数建议

### 3.1 VLoss 问题（🔴 最高优先级）

VLoss 4500+ 的根因是 **returns 未归一化**（returns 值域数百，value head 输出数百，MSE 自然为数千）。

**方案 A（纯调参）**：降低 `vf_coef`

```
vf_coef: 0.001 → 0.00003
```

理由：目标加权贡献 = 0.5 × PLoss = 0.05，VLoss ≈ 4500，coef = 0.05/4500 ≈ 1e-5。但经验上 PPO 需要适当的 value 梯度，建议保守取 3e-5。

**方案 B（代码改造，推荐）**：在 `_compute_gae` 中对 returns 做 z-score 归一化（运行时均值/标准差），然后在 value head 输出端反归一化。这样 VLoss 降至 0.1-10 量级，`vf_coef` 可用标准值 0.5。

### 3.2 Policy 稳定性

| 参数 | 当前值 | 建议值 | 理由 |
|---|---|---|---|
| `lr` | 0.001 | **0.0005** | ClipFrac 曾达 87%，表明步长过大。减半后应降至 20-40% 区间 |
| `clip_eps` | 0.15 | **0.10** | 配合 lr 降低，双重约束 policy 变化幅度 |
| `ent_coef` | 0.1 | **0.15** | 熵从 0.46→0.27 仅 20 个 batch，减慢崩塌需提高探索奖励 |

### 3.3 Aux 系数校准

基于 E40 的观测值，按目标加权贡献 0.02 × PLoss ≈ 0.0013 校准：

| 参数 | 当前值 | 加权贡献 | 建议值 | 新加权贡献 |
|---|---|---|---|---|
| `aux_tower_coef` | 3e-6 | 0.00026 | **5e-6** | 0.00043 |
| `aux_gold_coef` | 1e-6 | 0.00096 | **2e-6** | 0.00191 |
| `aux_base_dmg_coef` | 3e-4 | 0.00065 | **5e-4** | 0.00107 |

> 注：aux 标签刚改为 `mid→after` 纯值，只采集了 E20/E40 两个窗口。建议等 E60 后再确认。

### 3.4 保持不变

| 参数 | 当前值 | 理由 |
|---|---|---|
| `gamma` | 0.99 | 回合长度 ~250 步，γ²⁵⁰≈0.08，合理 |
| `gae_lambda` | 0.95 | 标准 PPO 值 |
| `ppo_epochs` | 2 | 配合 clip_eps=0.10 够用 |
| `max_grad_norm` | 0.5 | GradNorm 目前 0.5-1.4，无需调整 |
| `exploration_epsilon` | 0.05 | 标准值 |

---

## 四、建议配置

```yaml
ppo:
  lr: 0.0005
  lr_vf: 0.00025
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.10
  clip_eps_vf: 0.2
  ent_coef: 0.15
  exploration_epsilon: 0.05
  logit_noise_std: 0.2
  vf_coef: 0.00003
  max_grad_norm: 0.5
  max_grad_norm_vf: 50.0
  ppo_epochs: 2
  batch_size: 1024
  target_kl: 0.05
  lr_cosine_decay: true
  enable_auxiliary: true
  aux_tower_coef: 0.000005
  aux_gold_coef: 0.000002
  aux_enemy_tower_coef: 0.000005
  aux_enemy_gold_coef: 0.000002
  aux_base_dmg_coef: 0.0005
  aux_enemy_base_dmg_coef: 0.0005
```

---

## 五、预期效果

| 指标 | 调整前 (E40) | 预期 (E60+) |
|---|---|---|
| VLoss 加权贡献 | ~4.5 | ~0.13 |
| ClipFrac | 0.098 | 0.05-0.15 |
| Entropy | 0.27 | 0.20-0.35 |
| PLoss | 0.066 | 0.03-0.10 |
| Reward | 377 | ↑ 缓慢上升 |
| WinRate | 55% | ↑ 55-65% |

---

## 六、风险与后续

1. **2 个数据点不足以定论** — 建议 E60 后再次评估，若 ClipFrac < 0.05 则切回 `lr=0.0005`；若 Entropy < 0.1 则提高 `ent_coef` 至 0.2
2. **VLoss 临时方案** — `vf_coef=3e-5` 只是权宜之计。长期建议实施方案 B（returns 归一化）以彻底解决值域不匹配问题
3. **BaseDamageHead 系数** — 当前 BaseL 仅 2-3，`5e-4` 的 coef 使贡献约 0.001。需 E60+ 确认此值是否足以驱动学习，不足则再提高
