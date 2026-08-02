# PPO 位置通道纳入策略优化 — 方案

## 问题背景

当前 PPO 只优化 class 头（head_logits），位置通道（action_map）完全不在损失里：

```
策略 π(a|s) = π(class|s)  ×  π(position|class, s)
              ↑ PPO 优化      ↑ 完全不动（argmax 解码，预训练锁死）
```

**两个问题**：
1. **位置漂移**：PPO 更新 backbone → spatial_feat 变化 → 位置输出随输入漂移，但 action_map_conv 权重不变 → 位置选择越来越不准
2. **上限锁死**：位置偏好来自 score_distill 蒸馏 ExampleAI，位置能力被锁在 example 水平，无法随训练突破

**目标**：让位置通道也纳入 PPO 的 `logπ(a|s)`，使 action_map_conv 有梯度，位置随训练演化。

## 核心设计

### 策略分布分解

```
logπ(a|s) = logπ_class(class|s) + logπ_pos(position|class, s)
```

- `logπ_class`：已有（head_logits 的 raw softmax）
- `logπ_pos`：**新增**，= log softmax(action_map[class_id])[x, y]，需在合法位置上做

### 位置采样的温度问题

当前 class 用 z-score+T 温度采样，位置用 argmax。为了让位置有梯度，位置也应该**温度采样**（在合法位置内 softmax 采样，而非 argmax）。

但采样后的 logπ_pos 计算有个关键难点：

### 位置通道的 z-score 归一化（与 class 头对称）

位置采样的分布定义为**对该 class 通道的 19×19 值做 per-channel z-score**，再除以**独立的位置温度** `T_pos`：

```
channel_map = action_map[class_id]        # (19, 19)
z = (channel_map - mean) / (std + 1e-8)   # per-channel z-score（numpy ddof=0）
probs = softmax(z / T_pos)                # 位置温度采样
x, y ~ probs（mask 掉非法位置）
```

- **z-score 理由**：不同 class 通道的数值范围不同（建塔通道 vs 升级通道），z-score 消除尺度差异，保证 `T_pos` 采样一致性——和 class 头的 z-score 逻辑完全对称
- **独立温度理由**：class 错误代价低（24 类都可接受），位置错误代价高（建错位置浪费资源），所以 `T_pos` 应该低于 class 温度，让位置采样更接近 argmax

### 温度参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--temperature` | 0.5 | class 通道采样温度（已有） |
| `--pos-temperature` | 0.3 | 位置通道采样温度（新增，默认低于 class） |

## 技术难点：训练时无法重建采样分布

rollout 时位置采样用 `softmax(action_map[masked] / T)`，训练时算 `logπ_pos` 需要**完全相同**的分布。但：

1. **position_mask 依赖游戏状态**（塔的位置、金币、冷却等），训练时只有 board/stats，**没有 state 对象**，无法重建 mask
2. 因此必须**在 rollout 时把采样相关信息存进 npz**

### 解决方案：rollout 时存"采样位置 + 该位置的 mask 概率"

rollout 存位置采样时，把**采样到的 (x, y)** 和**该 class 的合法位置概率分布**一起存。具体：

| 新增 npz 字段 | 形状 | 说明 |
|--------------|------|------|
| `pos_class` | (T, N_heads) int | 每个 head 采样时的 class_id（可能不同于 head argmax） |
| `pos_sampled` | (T, N_heads, 2) int | 每个 head 采样的 (x, y) |
| `pos_logprob` | (T, N_heads) float32 | rollout 时算好的 logπ_pos = log softmax(z_masked/T_pos)[x,y] |

**关键决策**：`pos_logprob` 在 rollout 时就存好（用 rollote 时的分布算），训练时直接读取，不重算。这和 class 通道的 `normalized_log_probs` 思路一致（存归一化 logits）。

### 训练时 new logπ_pos 的计算 — 必须用同一份 mask（方案 B）

**经验教训（2026-07-31 实测）**：全图 softmax 算合法位置的 logp 会**系统性偏负**——361 格中合法格少，非法格数量多，softmax 后合法格概率被严重稀释（实测合法位置全图概率和仅 ~0.2，logp 均值 -5~-27，量级远超 class 项），导致 ratio 爆炸（policy_loss 到 1e33）。

**最终方案（2026-07-31 实测确定）：位置项作为辅助 loss，不参与 on-policy ratio。**

**为什么不做进 ratio**：位置通道的 logp 天然离散跳变——模型微变 → 位置 argmax 换一个格 → logp 从 ~0 跳到 -17。实测 |diff| 大部分样本仅 0.28，但 1.6% 样本 diff > 5，这些极端样本主导 KL（policy_loss 到 1e33、KL 771）。class 通道能稳定是因为 24 类 softmax 平滑，位置 361 格做不到。

**2026-07-31 二次验证**：用户提出归一化可能解决（固定缩放 ÷100 而非 z-score）。单样本实验确认固定缩放确实让模型间位置 logp 差异降到 0.002（raw 0.28 / z-score 0.54），但**完整训练中位置进 ratio 依然爆炸**（KL 到 184063，entropy 崩到 0.23，rollouts=40 也一样）。原因：PPO 优化位置这个离散目标时，梯度会主动把 361 格中某个位置概率推高/拉低，logp 剧变 → ratio 爆炸。**问题不是数值尺度，而是位置进 ratio 本身本质不稳，归一化解决不了。**

**结论：位置进 ratio 不可行（raw / z-score / 固定缩放 / 各种 T 都试过）。辅助 loss 是唯一稳定方案。**

**方案**：
- `L_total = L_ppo(class) + λ_pos · L_pos_aux`
- `L_pos_aux = -log softmax(masked(raw action_map[class])/T_pos)[x, y]`（BC 式，让采样位置概率变高）
- action_map 有梯度、位置可优化，但**不进入 ratio** → 不扰动 PPO 策略更新稳定性

**存储方案**：rollout 时存每个采样 head 的**合法位置 mask**（不是全 24 通道，只存采样时的 class 对应 mask）：

| npz 字段 | 形状 | 大小/局 |
|---------|------|---------|
| `pos_mask` | (T, N_heads, 19, 19) float16 (0/1) | ~0.7 MB |
| `pos_record` | (T, N_heads, 3) = (x, y, 保留) | 小 |

位置 logp（old）在辅助 loss 方案下**不需要**（ratio 不含位置项），`pos_record` 只保留 x, y。

## 改动点

### 1. `decoder.py` — 位置温度采样（z-score + 独立温度）

`decode_head` 中位置选择从 argmax 改为温度采样：

```python
# 位置采样（对选中的 class 通道做 per-channel z-score + T_pos）
channel_map = action_map[class_id]                  # (19, 19)
z = (channel_map - channel_map.mean()) / (channel_map.std() + 1e-8)
masked_z = np.where(pos_mask, z, -np.inf)           # mask 非法位置
probs = softmax(masked_z / T_pos)
x, y = sample(probs)
```

- 新增参数 `pos_temperature`、`sampled_pos_out`
- 位置 z-score 用 numpy `std()`（ddof=0），与 class 头对称

### 2. `_eval_worker.py` — 存位置采样信息

rollout 时对每个 head：
- 记录采样 class_id、采样 (x, y)
- 记录该 head 位置采样概率（rollout 时用 masked z + T_pos 算好 logπ_pos）

新增 npz 字段：`pos_class`、`pos_sampled`、`pos_logprob`。

### 3. `ppo_train.py` — logπ 加位置项

- 新增 `--pos-temperature` 参数
- `PPODataset` 加载新字段
- `compute_action_log_probs`（new）：加位置项 `log softmax(zscore(action_map_new[class])/T_pos)[x,y]`（全图，不 mask）
- old_logp：已有 class 项 + 新增位置项（直接读 pos_logprob）
- ratio = exp(new - old) 现在包含位置项

## 工作量估算

| 文件 | 改动 | 代码量 |
|------|------|--------|
| `decoder.py` | 位置温度采样 + sampled_pos_out + pos_temperature | ~35 行 |
| `_eval_worker.py` | 存 pos_class/pos_sampled/pos_logprob | ~25 行 |
| `ppo_train.py` | --pos-temperature + PPODataset + logπ 位置项 | ~45 行 |
| **合计** | | **~105 行** |

## 风险与验证

1. **位置采样导致行为不稳**：位置温度 T 可调低（如 0.3），大多数时候接近 argmax。验证方式：diagnose 看相邻代位置分布是否渐进变化（不突变）
2. **new_logπ 的 mask 偏差**（方案 A）：验证方式：对比 rollout 存的实际分布 vs 训练时全图 softmax 在采样位置的 log prob，差异应该 < 0.1
3. **训练稳定性**：KL 是否保持正常范围（修复后 1-2 区间）
4. **位置是否真的在优化**：对比训练前后，同一局面下 argmax 位置是否变化（说明 action_map_conv 在学）

## 备选：不动位置（维持现状）

如果改动风险大，可以先用**更低成本的中间方案**：
- rollout 位置仍 argmax，但训练时给 action_map 加一个辅助 loss（比如让 action_map 预测"哪个位置最近被选"）—— 但这等于引入额外监督，不如直接纳入策略干净。

推荐直接按方案 A 纳入策略。
