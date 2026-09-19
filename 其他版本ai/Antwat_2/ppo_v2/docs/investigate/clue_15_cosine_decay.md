# Clue 15: Cosine Decay 学习率调度在训练初期就开始衰减

## 调查结论：**确认问题存在**

---

## 1. 问题现象

Cosine decay 学习率调度在训练初期就开始衰减，导致策略网络和价值网络在尚未充分学习时就面临学习率下降，可能影响训练效果。

---

## 2. 代码分析

### 2.1 核心代码

**文件**: `ppo_v2/src/ppo_antwar/trainer/lr_scheduler.py`

```python
def compute(self, episode: int, total_episodes: int) -> Tuple[float, float, str]:
    cfg = self._config

    if cfg.warmup_episodes > 0 and episode <= cfg.warmup_episodes:
        ratio = episode / max(cfg.warmup_episodes, 1)
        schedule_type = "warmup"
        lr = cfg.lr_base * ratio
        lr_vf = cfg.lr_vf_base * ratio
        return lr, lr_vf, schedule_type

    if cfg.cosine_decay:
        progress = episode / max(total_episodes, 1)
        factor = (1.0 + math.cos(math.pi * progress)) / 2.0
        lr = cfg.lr_base * factor
        lr_vf = cfg.lr_vf_base * factor
        schedule_type = "cosine"
        return lr, lr_vf, schedule_type

    schedule_type = "fixed"
    return cfg.lr_base, cfg.lr_vf_base, schedule_type
```

### 2.2 配置

**文件**: `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml`

```yaml
ppo:
  lr: 0.0005
  lr_vf: 0.00025
  lr_cosine_decay: true
  # 注意：没有 lr_warmup_episodes 配置项
```

### 2.3 调用方式

**文件**: `ppo_v2/src/ppo_antwar/trainer/selfplay.py`

```python
self._episode_count = 0  # 起始值

while self._episode_count < num_episodes:
    self._episode_count += 1  # 从 1 开始
    # ... 训练逻辑 ...

# 在 PPO 更新后调用
self._trainer.update_lr_schedule(self._episode_count, self._total_episodes)
```

---

## 3. 问题根因分析

### 3.1 Cosine Decay 公式

```
progress = episode / total_episodes
factor   = (1 + cos(π × progress)) / 2
lr       = lr_base × factor
```

这是一个**从 1.0 单调递减到 0.0** 的标准余弦退火公式：
- `progress = 0` → `factor = (1 + cos(0)) / 2 = 1.0` → lr = lr_base
- `progress = 0.5` → `factor = (1 + cos(π/2)) / 2 = 0.5` → lr = lr_base × 0.5
- `progress = 1.0` → `factor = (1 + cos(π)) / 2 = 0.0` → lr = 0

### 3.2 关键问题：episode 从 1 开始，progress 永远不为 0

由于 `self._episode_count` 从 1 开始递增，`progress` 的最小值为 `1/2000 = 0.0005`，而非 0。这意味着**训练第一步的学习率就已经不是 lr_base**，而是略低于 lr_base。

### 3.3 无 Warmup 保护

配置中**没有设置 `lr_warmup_episodes`**，`from_ppo_config` 中的默认值为 0：

```python
warmup_episodes=ppo_cfg.get("lr_warmup_episodes") or 0,  # 未配置 → 0
```

因此 warmup 分支永远不会进入，cosine decay 从 episode=1 就开始生效。

### 3.4 各阶段学习率数值（total_episodes=2000）

| Episode | progress | factor | lr_policy | lr_value | 说明 |
|---------|----------|--------|-----------|----------|------|
| 1       | 0.0005   | 0.9999996 | 4.99998e-4 | 2.49999e-4 | ≈满值，问题不大 |
| 100     | 0.05     | 0.99381   | 4.969e-4   | 2.485e-4  | 下降 0.6% |
| 200     | 0.10     | 0.97553   | 4.878e-4   | 2.439e-4  | 下降 2.4% |
| 400     | 0.20     | 0.90451   | 4.523e-4   | 2.261e-4  | 下降 9.5% |
| 600     | 0.30     | 0.79389   | 3.969e-4   | 1.985e-4  | 下降 20.6% |
| 800     | 0.40     | 0.65451   | 3.273e-4   | 1.636e-4  | 下降 34.5% |
| 1000    | 0.50     | 0.50000   | 2.500e-4   | 1.250e-4  | 下降 50% |
| 1500    | 0.75     | 0.14645   | 7.322e-5   | 3.661e-5  | 下降 85.4% |
| 2000    | 1.00     | 0.00000   | 0.000e+0   | 0.000e+0  | lr=0 |

### 3.5 核心问题总结

1. **从 episode=1 开始就衰减**：虽然 episode=1 时衰减极小（0.04%），但 cosine decay 是一条连续曲线，不存在"平坦期"。在训练前 20% 的阶段（episode 1-400），学习率已经下降了约 10%。

2. **无 warmup**：PPO 训练通常在初期需要稳定的学习率来建立基础策略。缺少 warmup 意味着从第一个 episode 就开始衰减，与常见实践不符。

3. **最终 lr=0**：cosine decay 的终点是 lr=0，这意味着训练末期学习率趋近于零，模型几乎停止学习。通常应设置一个 `lr_min` 下限（如 lr_base 的 1/10）。

4. **衰减速度过快**：在 episode=600（仅完成 30% 训练）时，学习率已经下降 20.6%，这对于 PPO 这种需要大量样本的策略梯度方法来说，可能导致中后期学习速度不足。

---

## 4. 修复建议

### 方案 A：添加 Warmup 阶段（推荐）

在 `ppo_antwar.yaml` 中添加 warmup 配置：

```yaml
ppo:
  lr: 0.0005
  lr_vf: 0.00025
  lr_cosine_decay: true
  lr_warmup_episodes: 100  # 新增：前 100 个 episode 线性 warmup
```

同时在 `lr_scheduler.py` 中修改 cosine decay 的 progress 计算方式，使 warmup 结束后 progress 从 0 开始：

```python
def compute(self, episode: int, total_episodes: int) -> Tuple[float, float, str]:
    cfg = self._config

    if cfg.warmup_episodes > 0 and episode <= cfg.warmup_episodes:
        ratio = episode / max(cfg.warmup_episodes, 1)
        schedule_type = "warmup"
        lr = cfg.lr_base * ratio
        lr_vf = cfg.lr_vf_base * ratio
        return lr, lr_vf, schedule_type

    if cfg.cosine_decay:
        # 修复：warmup 结束后 progress 从 0 开始
        effective_start = cfg.warmup_episodes
        progress = (episode - effective_start) / max(total_episodes - effective_start, 1)
        progress = max(0.0, min(1.0, progress))  # clamp 到 [0, 1]
        factor = (1.0 + math.cos(math.pi * progress)) / 2.0
        lr = cfg.lr_base * factor
        lr_vf = cfg.lr_vf_base * factor
        schedule_type = "cosine"
        return lr, lr_vf, schedule_type

    schedule_type = "fixed"
    return cfg.lr_base, cfg.lr_vf_base, schedule_type
```

### 方案 B：设置学习率下限

避免学习率衰减至 0：

```python
if cfg.cosine_decay:
    progress = (episode - cfg.warmup_episodes) / max(total_episodes - cfg.warmup_episodes, 1)
    progress = max(0.0, min(1.0, progress))
    factor = (1.0 + math.cos(math.pi * progress)) / 2.0
    # 设置最低学习率为 base_lr 的 10%
    lr_min_ratio = 0.1
    factor = lr_min_ratio + (1.0 - lr_min_ratio) * factor
    lr = cfg.lr_base * factor
    lr_vf = cfg.lr_vf_base * factor
```

### 方案 C：方案 A + B 组合（最佳实践）

同时添加 warmup 和 lr 下限，这是 PPO 训练中最常见的 cosine decay 实践：

- Warmup 阶段：前 5% 的 episode（约 100 个），学习率从 0 线性增长到 lr_base
- Cosine decay 阶段：从 lr_base 平滑衰减到 lr_base × 0.1
- 保证训练末期仍有最小学习率

---

## 5. 影响评估

| 维度 | 影响 |
|------|------|
| 严重程度 | **中等** |
| 训练初期（ep 1-200） | 影响较小，lr 仅下降 2.4%，几乎可忽略 |
| 训练中期（ep 200-800） | **影响显著**，lr 下降 2.4%-34.5%，可能导致学习速度不足 |
| 训练后期（ep 800-2000） | **影响严重**，lr 下降 34.5%-100%，模型几乎停止学习 |
| 总体 | 缺少 warmup + lr 衰减至 0 是两个叠加问题，可能导致训练不充分 |

---

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `ppo_v2/src/ppo_antwar/trainer/lr_scheduler.py` | 学习率调度器核心实现 |
| `ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py` | PPO 训练器，调用 `update_lr_schedule` |
| `ppo_v2/src/ppo_antwar/trainer/selfplay.py` | 自对弈训练循环，传递 episode/total_episodes |
| `ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml` | 训练配置，缺少 warmup 配置 |
