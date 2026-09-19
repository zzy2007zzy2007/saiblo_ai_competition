# 调查报告：step_reward_clip=2000 过大，基本不生效

**调查日期**：2026-06-11
**问题编号**：clue_16
**状态**：✅ 确认

---

## 1. 问题描述

PPO V2 训练中 `step_reward_clip = 2000.0`，远大于实际单步 reward 的典型值域，导致 clip 形同虚设，无法起到约束 reward 极端值的作用。

---

## 2. 代码审查

### 2.1 Clip 配置

**文件**：`ppo_v2/src/ppo_antwar/utils/action_constants.py:258`

```python
REWARD_CONFIG: Dict = {
    ...
    "step_reward_clip": 2000.0,
}
```

### 2.2 Clip 执行逻辑

**文件**：`ppo_v2/src/ppo_antwar/env/antwar_env.py:301-303`

```python
# clip
clip_val = REWARD_CONFIG["step_reward_clip"]
r = max(-clip_val, min(clip_val, r))
```

clip 在 `_compute_battle_rewards` 方法末尾执行，对汇总后的 `r`（含终局奖励）做对称裁剪。

### 2.3 Clip 位置的关键问题：动作奖励未被 clip

**文件**：`ppo_v2/src/ppo_antwar/env/antwar_env.py:112-121`

```python
# 动作奖励（只算 self）
reward_self_action = self._compute_action_reward(action_self, self_player)

try:
    reward_self, reward_opponent, terminated, truncated, info = (
        self._resolve_turn_from_ops(self_ops, opponent_ops, self_first)
    )
    reward_self += reward_self_action   # ← 动作奖励在 clip 之后叠加
```

`_compute_battle_rewards` 内部对 `r` 做了 clip，但返回后 `step()` 方法又叠加了 `reward_self_action`（动作奖励），这部分**未被 clip 覆盖**。不过动作奖励值域很小（最大 15.0），对结论影响不大。

---

## 3. 单步 Reward 值域分析

### 3.1 非终局步（terminated=False）

各奖励分量的典型值域：

| 奖励分量 | 计算公式 | 典型值域 | 极端值域 |
|---------|---------|---------|---------|
| 基地伤害 | `hp_dmg × 2.0` | 0 ~ +6 | 0 ~ +40 |
| 塔伤害 | `tower_dmg × 0.2` | 0 ~ +4 | 0 ~ +20 |
| 金币收入 | `coin_gain × 0.05` | +0.5 ~ +2 | 0 ~ +5 |
| 金币惩罚 | `-coin_gain_opp × 0.01` | -0.3 ~ 0 | -0.5 ~ 0 |
| 塔存活 | `own_survival - enemy_survival` | -0.1 ~ +0.4 | -0.2 ~ +0.8 |
| 科技加成 | `gen_level × 1.5 + ant_level × 1.0` | 0 ~ +7.5 | 0 ~ +7.5 |
| 余额奖励 | `coins × 0.02` | +0.2 ~ +10 | 0 ~ +20 |
| 死亡惩罚 | `die_delta × (-0.05)` | -0.5 ~ +0.5 | -5 ~ +5 |
| 动作奖励 | 各类动作 | -9 ~ +15 | -9 ~ +15 |
| **非终局步总计** | | **-10 ~ +20** | **-15 ~ +80** |

### 3.2 终局步（terminated=True, 含 win/loss）

| 场景 | 终局奖励 | 加上中间奖励后 | clip=2000 是否触发 |
|------|---------|--------------|-------------------|
| 赢局 | +500.0 | +490 ~ +580 | ❌ 不触发 |
| 输局 | -500.0 | -515 ~ -490 | ❌ 不触发 |

### 3.3 极端场景推演

即使考虑最极端情况——单步内同时出现：
- 基地伤害 20 HP × 2.0 = +40
- 塔伤害 100 HP × 0.2 = +20
- 金币收入 100 × 0.05 = +5
- 余额 1000 × 0.02 = +20
- 科技满级 = +7.5
- 动作奖励 upgrade_gen_speed_l1 = +15
- 赢局终局奖励 = +500

**极端最大值 ≈ 40 + 20 + 5 + 20 + 7.5 + 15 + 500 = 607.5**

即使所有分量同时取极端正值，也远低于 2000。clip=2000 在任何实际场景下都不会被触发。

---

## 4. 结论

**✅ 确认问题存在**：`step_reward_clip = 2000.0` 完全不生效。

### 4.1 核心事实

1. **非终局步 reward 典型值域**：-10 ~ +20，极端不超过 ±80
2. **终局步 reward 最大值**：约 ±580（含 win/loss ±500）
3. **clip 阈值 2000 是实际最大值的 3.4 倍以上**，形同虚设
4. **clip 的实际作用**：仅作为极端安全网，防止代码 bug 产生 NaN/Inf 传入训练，对正常 reward 分布无约束力

### 4.2 clip 不生效的影响

clip 本身不生效并不是一个严重问题——如果 reward 值域合理，不需要 clip 起作用。但当前情况是：

- **终局奖励 ±500 占据单步 reward 的绝对主导**（参见 clue_02），导致 returns 双峰化
- clip=2000 本应约束终局步的 reward spike，但阈值过高完全不起作用
- 如果 clip 设为合理值（如 50-100），可以直接抑制终局步的 reward spike，缓解 returns 双峰化

### 4.3 与 clue_02 的关系

- **clue_02** 指出终局奖励 ±500 过大导致 returns 双峰化
- **clue_16**（本报告）指出 step_reward_clip=2000 无法约束终局步的 reward spike
- 两者是同一问题的不同侧面：终局奖励过大 + clip 阈值过高 = 终局 reward spike 完全不受控

---

## 5. 修复建议

### 方案 A：降低 clip 值使其生效（推荐，与 clue_02 方案联动）

将 `step_reward_clip` 从 2000 降至合理值，使 clip 能约束终局步的 reward spike：

```python
# ppo_v2/src/ppo_antwar/utils/action_constants.py
REWARD_CONFIG: Dict = {
    ...
    "step_reward_clip": 50.0,   # 原 2000.0，降至 50
    ...
}
```

**效果分析**：
- 非终局步（典型 -10 ~ +20）：不受影响
- 终局步赢局（约 +500）：被 clip 到 +50，终局 spike 被抑制
- 终局步输局（约 -500）：被 clip 到 -50，终局 spike 被抑制

**优点**：
- 最简单的修改，只改一个数字
- 直接抑制终局 reward spike
- 保留胜负信号（+50 vs -50 仍能区分赢输）

**缺点**：
- clip 是硬截断，会丢失终局奖励的精确值信息
- 赢局和输局的 reward 差异从 1000 压缩到 100，信号强度降低
- clip 后 reward 不再反映真实收益，可能影响 value function 学习

### 方案 B：降低终局奖励 + 同步降低 clip（推荐，与 clue_02 方案 A 联动）

同时降低终局奖励和 clip 值：

```python
# ppo_v2/src/ppo_antwar/utils/action_constants.py
REWARD_CONFIG: Dict = {
    ...
    "win_reward": 20.0,          # 原 500.0
    "loss_reward": -20.0,        # 原 -500.0
    "step_reward_clip": 100.0,   # 原 2000.0，作为安全网
    ...
}
```

**效果分析**：
- 终局步赢局：约 +20 ~ +35（中间奖励 + win_reward），远低于 clip=100
- 终局步输局：约 -30 ~ -15（中间奖励 + loss_reward），远高于 clip=-100
- clip=100 作为安全网，仅在异常情况下触发

**优点**：
- 从根源解决终局 reward spike 问题（降低 win/loss reward）
- clip 作为安全网存在，阈值合理
- 不丢失终局信号的精确值

**缺点**：
- 需要同时修改两个参数
- 终局奖励降低后，需要确认策略仍能充分重视胜负

### 方案 C：仅降低 clip，不改终局奖励（不推荐）

将 clip 降至 50-100，但保持 win/loss reward = ±500。

**不推荐原因**：
- 终局步 reward 被硬截断到 ±50~±100，但 value function 仍需预测 ±500 的 return
- clip 后的 reward 与 GAE 计算中的 return 不一致，导致 TD error 偏大
- 本质上是"掩耳盗铃"——reward 被 clip 了但 return 仍受终局奖励影响

### 推荐方案

**方案 B**（降低终局奖励 + 同步降低 clip），与 clue_02 的修复方案联动。

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `win_reward` | 500.0 | 20.0 | 与中间步奖励量级匹配 |
| `loss_reward` | -500.0 | -20.0 | 与中间步奖励量级匹配 |
| `step_reward_clip` | 2000.0 | 100.0 | 作为安全网，约为最大正常 reward 的 2 倍 |

---

## 6. 代码引用索引

| 内容 | 文件 | 行号 |
|------|------|------|
| step_reward_clip 配置 | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 258 |
| win_reward / loss_reward 配置 | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 256-257 |
| clip 执行逻辑 | `ppo_v2/src/ppo_antwar/env/antwar_env.py` | 301-303 |
| 终局奖励叠加逻辑 | `ppo_v2/src/ppo_antwar/env/antwar_env.py` | 292-299 |
| 动作奖励在 clip 后叠加 | `ppo_v2/src/ppo_antwar/env/antwar_env.py` | 119 |
| 奖励分量权重 | `ppo_v2/src/ppo_antwar/utils/action_constants.py` | 230-254 |
