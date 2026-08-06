# Bundle MCTS 的 AlphaZero 训练方案

> 目标：用已验证的 bundle 采样 MCTS（自对局 67-69% 增强）做 AlphaZero 自对弈训练。
> 核心挑战：MCTS 访问分布在 bundle 空间，网络输出在意图空间——需要边际化桥接。
> 日期：2026-08-06。状态：设计阶段。

## 1. 背景

搜索已验证有效（`docs/az_bundle_mcts_results.md`）：bundle 采样 MCTS（128 迭代/
深度 4，t_class 0.5 / t_pos 0.3）对原始 gen0120_warm 自对局 66.7-68.8%，增强成立。
下一步是把搜索接回 AlphaZero 自对弈训练。

**标准 AlphaZero 训练**：每决策跑 MCTS → 根节点访问分布（温度缩放）作为策略目标 →
`policy_loss = CE(网络输出, 访问分布)` + 价值回归。**要求网络输出空间 = MCTS 动作空间**。

**我们的差异**：网络输出是**每头意图**（类×位置），MCTS 分支是**采样出的 bundle**
（完整 3 动作组）——两者空间不同，不能直接 CE。

## 2. 核心设计：边际化

**核心挑战**：MCTS 的 visit 分布在 **bundle 空间**，网络输出在**意图空间**。
桥接 = 把每个 bundle 的 visit 质量**按"意图被采样到的次数"归属**到每头意图。
（2026-08-06 采用用户方案：蒙特卡洛估计，比"存被保留组合"更正确，比"枚举反查表"更简单）

**关键洞察**：同一解码 bundle 可来自多种意图组合（如 3 个 HOLD 可来自
`闪电×3`、`EMP+hold+闪电` 等）。采样的意图分布 = 网络的先验，所以
**某意图在"解码出 bundle b 的采样"里出现的频率 ≈ P(意图 | 解码出 b)**——
这是条件概率的蒙特卡洛估计。按采样计数分配 visit 质量 = 期望意义上的正确归属。

```
对每个解码出的 bundle b、每个头 h：
  target_h(意图 i) += visit(b) × count_h(b, i) / Σ_j count_h(b, j)
```

**具体例子**（采样中 30 个采样解码出 WAIT，其中 head1 采到闪电 20 次、emp 3 次、
hold 5 次、护盾 2 次）：
```
head1 的 WAIT 归属: 闪电 20/30, emp 3/30, hold 5/30, 护盾 2/30
WAIT visit=0.5 → 闪电意图 += 0.5×2/3 ≈ 0.333, emp += 0.05, ...
```
闪电拿大头——符合"网络最可能想闪电"（闪电占无操作类概率的 91%）。

**归一化**：对每个 (bundle, head)，计数归一化成比例（和 = 1），再乘 visit。
每头的目标总和 = 1（合法分布），CE 干净。

## 3. 训练损失

```
对每个头 h:
  L_h = CE(网络 P_h(意图), target_h)      # 网络 P_h = P(c)·P(p|c)（解码器掩码）
L = Σ_头 L_h + λ_value · MSE(价值头, HP-diff目标) + L_anchor(§3.2)
```

### 3.1 分解式 CE 的精确推导（网络输出是因子化的，损失必须跟着拆）

网络输出 `P_h(意图) = P_h(c) × P_h(p|c)`（类 softmax × 类内位置 softmax），
网络给不出单个"意图概率"，只能给两个因子。所以 CE 必须拆开：

```
L_h = -Σ_i target_h(i) · log P_h(意图 i)
    = -Σ_i target_h(i) · [log P_h(c_i) + log P_h(p_i|c_i)]
    = -Σ_i target_h(i) · log P_h(c_i)        ← 类部分
      -Σ_i target_h(i) · log P_h(p_i|c_i)    ← 位置部分
```

**类部分**（target 对位置求和 = 类边际）：
```
target_class(c) = Σ_p target_h(c, p)
L_class = -Σ_c target_class(c) · log P_h(c)
```

**位置部分**（完整 (c,p) target，按类加权的位置 CE）：
```
L_pos = -Σ_c target_class(c) · Σ_p [target_h(c,p)/target_class(c)] · log P_h(p|c)
      = -Σ_c target_class(c) · CE(归一化位置目标_c, P_h(·|c))
```

**等价视角**：把联合 target 分解成"类分布 + 每类条件位置分布"，
损失 = 类 CE + 按类质量加权的位置 CE。**这是精确分解（非近似）**——
因为 `log P(意图) = log P(c) + log P(p|c)`，log 把乘积变加法，联合 CE 自然分裂。

### 3.2 策略锚定损失（trust-region，2026-08-06 补充）

**动机**：target 很稀疏（只覆盖采到的意图），CE 梯度集中在少数格子上，
一次激进更新可能把策略拉偏。且 value_warmup 已出现过策略漂移（锚定 0.147，
强度减半）。参考 PPO 的位置图 MSE trust-region（`compute_pos_mse_loss`）。

```
L_anchor = λ_anchor × [ MSE(action_map_current, action_map_recorded)
                        + Σ_头 MSE(head_logits_current, head_logits_recorded) ]
```

- **记录的输出**：自对弈时保存每个决策的网络策略输出（action_map + 3 头 logits），
  训练时锚向它（= PPO 的"旧"输出）
- **位置图头锚定**：防稀疏梯度拉偏（对应 PPO 的位置担忧——但 PPO 是 ratio 爆炸，
  这里是稀疏 target；机制不同，都需要稳定）
- **类头锚定**：保策略核心——类偏好（闪电 vs 建塔）是策略的命脉，被拉偏等于
  重演"闪电被剪掉"
- **λ_anchor 双刃剑**：太大 → 训练只复刻自对弈输出、学不到搜索改进；太小 →
  策略漂移。初版给 1.0（与价值损失同量级），可调

### 3.3 训练操作步骤（每决策、每头）

1. 网络前向 → head_logits、action_map
2. 从记录的意图计数算出 `target_h(c,p)`
3. `target_class(c) = Σ_p target_h(c,p)`
4. `L_class = -Σ_c target_class(c)·log softmax(head_logits)[c]`（需 class_mask 算分母）
5. 对每个有质量的类 c：
   `L_pos += target_class(c) × CE(类内归一化目标, softmax(action_map[c]))`
   （需 position_mask 算分母）
6. `L_h = L_class + L_pos`；Σ 所有头 + λ_value·价值 MSE

**数据存掩码的原因**：算 `P_h(c)` 和 `P_h(p|c)` 的 softmax 需要合法类/位置集合
（分母）——class_mask / position_mask。

## 4. 采样策略优化（2026-08-06 补充）

**实测**：单次 `sample_bundle` ~0.46ms，其中**掩码计算 0.28ms（61%）**，且每次采样
都在重算掩码（掩码只依赖状态，节点内不变）。

**优化 1：掩码节点级缓存**——`_expand` 里掩码只算一次，传给 `sample_bundle`。
采样本体只剩 ~0.18ms/次。

**优化 2：采样 k×n 次、按计数取 top-k**（用户方案）：
- 当前：采样直到 k 个唯一（max_attempts=k×6），保留最高先验
- 改为：采样 k×n 次（如 k=24, n=15 → 360 次），去重后**按采样次数取前 k 个**
- 好处：候选集 = 策略下最可能的 bundle；每个保留 bundle 有可靠计数（供边际化）
- 耗时：掩码缓存后 360×0.18 ≈ 65ms/节点 ≈ 当前（144×0.46），**几乎免费**

## 5. 训练数据（每决策记录）

```
board       (28,19,19) f16          ← 网络输入
stats       (42,)      f16
player      (0/1)                    ← 价值视角
class_mask  (24,)      uint8         ← 合法类（训练时算 P(c) 分母）
position_mask (24,19,19) bitmask     ← 合法位置（训练时算 P(p|c) 分母）
bundles     k × 3 头意图(class, x, y) ← 保留的 k 个 bundle 的每头意图
counts      k × 3 × 意图计数          ← 每个 bundle 每头的意图采样计数
visit       k × f32                   ← 温度缩放的访问分布
action_map  (24,19,19) f16          ← 自对弈时的网络输出（§3.2 锚定用）
head_logits (3,24)     f16          ← 自对弈时的网络输出（§3.2 锚定用）
value_target f32                      ← 终局 HP-diff，player 视角
```

**存储量**：~44KB/决策（board + action_map 占大头），一局 ~500 决策 ≈ 22MB，
npz 压缩后更小。
**训练时**：按 §2 公式（计数归一化 × visit）算出每头意图目标，再做 CE；
§3.2 的锚定用记录的 action_map / head_logits。

## 6. 自对弈采集

- 双方都用 bundle MCTS（自对弈，搜索增强的两个玩家）
- 搜索配置：与验证一致（128 迭代/深度 4，t_class 0.5 / t_pos 0.3，加上 §4 的采样优化）
- 每决策：search → 温度缩放的访问分布 + 每个 bundle 的意图计数 → 记录
- 终局 → 每决策赋 HP-diff 价值目标
- 温度调度：AlphaZero 式（前期 1.0 采样、后期 ~0）

**⚠️ 搜索配置必须用 128 迭代/深度 4（2026-08-06 修正）**：自对局增强的可靠证据
只有 128/深度4（66.7%，48 局）；64/深度2 是否增强未验证（曾跑 5 局 1W/4L，样本
不足）。**降低配置可能导致搜索增强消失**，训练目标退化为无信号。所以训练用
128/深度4（已验证配置）。

**吞吐量实测**：128/深度4 自对局 48 局 × 8 worker ≈ 6 小时（~60 分钟/局墙钟，
8 worker 并行）。**10 局一个训练 batch ≈ 1.3 小时**；3-5 batch ≈ 4-7 小时
（一个通宵的量）。

## 7. 训练循环

```
for batch in range(batches):
    games = [selfplay(当前网络, 搜索配置)]      # 采集 N 局
    边际化目标 → 每头意图 target + 价值 target
    for epoch in range(epochs):
        for batch 样本:
            L = Σ_头 CE + λ_value·MSE(价值)
            optimizer.step()
    保存 checkpoint（每 batch）
```

- 优化器：AdamW，lr 1e-3，weight_decay 1e-4
- 初始网络：gen0120_warm（已有策略 + 价值头）
- 每个决策是独立的训练样本（board/stats + 目标）

## 8. 超参数（初版）

| 参数 | 建议 |
|------|------|
| 搜索配置（训练） | **128 迭代/深度 4**（已验证增强配置，§6） |
| 采样温度 | t_class 0.5 / t_pos 0.3 |
| 每 batch 局数 | 10（验证起点；见 §9 讨论） |
| epochs | 3-5 |
| batch_size | 32 |
| λ_value | 1.0 |
| λ_anchor | 1.0（策略锚定，§3.2；太大阻塞学习，太小策略漂移） |
| 目标温度 | 前 30 回合 1.0，之后 ~0 |

## 9. 验证计划

1. **训练后模型 vs 训练前（gen0120_warm）自对局——胜率 >50% 说明训练提升**
   （**真正的判据**：损失下降只说明网络拟合了搜索目标，那必然发生，不算数）
2. 训练后 vs ExampleAI / rule_v4——绝对强度对比
3. 搜索健全性：训练后搜索 ≥ 训练后原始——训练没有破坏搜索基础

### 9.1 10 局/batch 够不够（2026-08-06 讨论）

- **够做首轮验证**：10 局 ≈ 5000 样本/batch，价值头已预训练（微调而非从零学）
- **弱点**：价值目标局内共享 → 每 batch 只有 10 个独立价值目标（噪声大）；
  3-5 batch 累计到 30-50 局后价值趋势才可靠
- **流程**：10 局/batch 采 3-5 batch → 训练 → 训练后 vs 训练前自对局。
  若 >50% → 机制成立，再扩规模；若 ~50% → 再考虑加大 batch（15-20 局）

## 10. 里程碑

| # | 内容 | 验证 |
|---|------|------|
| T1 | 搜索改造：掩码缓存 + k×n 采样取 top-k + 意图计数记录 + 自对弈采集存储 | 一局跑通，数据格式正确 |
| T2 | 训练循环（计数→意图目标 + 每头 CE + 价值 MSE） | 损失下降 |
| T3 | 小规模验证（几 batch） | 训练后 vs 训练前胜率 >50% |
| T4 | 扩大规模 + 调参 | 绝对强度提升 |

## 11. 风险

| 风险 | 缓解 |
|------|------|
| 训练自对弈太慢（搜索重） | 先降搜索预算跑通管道，再升 |
| 边际化目标稀疏（未采样意图无目标） | 采样 k 够大 + 温度多样；与标准 AlphaZero 未搜动作 0 目标同理 |
| 价值头精度仍是天花板 | T3 后评估；必要时加强价值训练 |
| 策略漂移（价值训练拉偏策略） | 复用 value_warmup 的锚定机制 |

## 12. 一句话总结

**bundle MCTS 的 AlphaZero 训练 = 自对弈时采样 k×n 个动作组（掩码缓存，几乎免费）、
按计数取 top-k 并记录每头的意图采样计数；训练时把根节点访问分布按"意图被采样
次数"（条件概率的蒙特卡洛估计）归属成每头意图目标，用 CE（网络意图输出 vs
目标）+ 价值 MSE 监督更新。关键难点是动作空间不同（bundle vs 意图）需计数归属
桥接，以及训练自对弈的搜索吞吐量（先降预算跑通）。**
