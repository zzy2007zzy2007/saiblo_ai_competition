# 蚁洋陷役2 — 神经网络模型设计文档

> 基于 AlphaZero 风格架构：CNN 提取空间特征 + 策略头（空间图+分类头）+ 价值头

---

## 1. 总体架构

```
Input (board + stats)
        │
   ┌────▼────┐
   │ Encoder │  ← Conv7×7 + ResBlock×6 + Stats MLP
   └────┬────┘
        │
   state_embedding (128)
        │
   ┌────┴──────────────┐
   ▼                    ▼
┌──────┐          ┌──────────┐
│Value │          │ Policy   │
│Head  │          │ Head     │
│→ 1   │          │→ action_map(23,19,19) + class heads×3
└──────┘          └──────────┘
```

### 完整前向流程

```
board (28,19,19)
  → Conv7×7, pad=3, 28→64  (快速融合多通道输入)
  → BatchNorm → ReLU
  → ResBlock ×6~10 (Conv3×3, 64ch, 无池化)
  → spatial_feat (64,19,19)
      ├── Global Average Pooling → board_emb (64)
      └── Conv1×1, 64→23 → action_map (23,19,19)

stats (~22维)
  → MLP [22→64] → ReLU → stats_emb (64)

state_emb = concat(board_emb, stats_emb) = (128)

# 分类头（×3，每份轻量23维）
policy_base = MLP(state_emb: 128→64) → ReLU
head1 = Linear(64→23) → head1_logits
head2 = Linear(64→23) → head2_logits
head3 = Linear(64→23) → head3_logits

# 价值头
value = Linear(128→64) → ReLU → Linear(64→1) → tanh → [-1, 1]
```

### 参数量

| 组件 | 参数量 |
|------|--------|
| Conv7×7 (28→64) + BN | ~88K |
| ResBlock ×6 (Conv3×3, 64ch) | ~442K |
| Stats MLP (22→64→64) | ~5.6K |
| Conv1×1 (64→23) | ~1.5K |
| Policy heads (GAP→3×23) | ~8.6K |
| Value Head (128→64→1) | ~8.3K |
| **总计** | **~0.55M** |

### CPU 推理耗时

- 单次 forward（含解码）：**~15-50ms**（视 CPU 型号）
- 远低于 10 秒限制

---

## 2. 输入

直接复用官方 `SDK/utils/features.py` — `FeatureExtractor`：

```python
observation = extractor.encode_observation(state, player, action_mask)
# → {"board": np.ndarray(28,19,19), "stats": np.ndarray(N,), "action_mask": np.ndarray(96,)}
```

### 空间特征 `board` (28 通道 × 19 × 19)

| 通道 | 内容 |
|------|------|
| 0-3 | 地形（路径、己方高台、对方高台、障碍） |
| 4-5 | 己方/敌方防御塔位置 |
| 6 | 塔等级 |
| 7 | 塔攻击范围 |
| 8 | 塔冷却 CD |
| 9 | 塔伤害 |
| 10-11 | 己方/敌方蚂蚁（归一化血量累加） |
| 12 | 蚂蚁等级 |
| 13 | 蚂蚁年龄 |
| 14-15 | 己方/敌方信息素 |
| 16-19 | 蚂蚁状态（冻结、随机、蛊惑、免控） |
| 20-21 | 闪电风暴区域 |
| 22-23 | EMP 轰炸区域 |
| 24-25 | 引力护盾区域 |
| 26-27 | 紧急回避区域 |

### 全局统计 `stats` (~22 维)

金币、基地血量、超级武器冷却、前线距离、塔数量/等级、击杀数、基地等级等。

### 设计要点

- **无池化层**：保留精确位置信息，棋盘游戏不需要
- **第一层 Conv7×7**：快速融合 28 个异构通道（参考 AlphaGo Zero）
- **后续 Conv3×3** ResBlock：逐步扩大感受野至全图
  - +3 个 ResBlock → 覆盖 19×19 全图
  - +6 个 ResBlock → 27×27，留有余量
- **Stats 不进入 CNN**：全局特征通过 MLP 单独编码后与 board_emb concat

---

## 3. 输出：策略头

### 设计原则：意图驱动（Intent-level）

网络只输出"想要什么"和"在哪做"，不关心塔 ID 等运行时细节。解码器根据位置查表，将意图映射为具体操作。

### 动作分类（23 类）

| 通道 | 动作类 | 解码逻辑 |
|------|-------|---------|
| 0 | Basic 塔 | 空地 → 建 Basic；已有 Basic → pass |
| 1 | Heavy 塔 | 空地 → 建 Basic；有 Basic → 升级 Heavy |
| 2 | Heavy+ 塔 | 空地 → 建 Basic；有 Basic → 升级 Heavy；有 Heavy → 升级 Heavy+ |
| 3 | Ice 塔 | 空地 → 建 Basic；有 Basic → 升级 Heavy；有 Heavy → 升级 Ice |
| 4 | Bewitch 塔 | 空地 → 建 Basic；有 Basic → 升级 Heavy；有 Heavy → 升级 Bewitch |
| 5 | Quick 塔 | 空地 → 建 Basic；有 Basic → 升级 Quick |
| 6 | Quick+ 塔 | 空地 → 建 Basic；有 Basic → 升级 Quick；有 Quick → 升级 Quick+ |
| 7 | Double 塔 | 空地 → 建 Basic；有 Basic → 升级 Quick；有 Quick → 升级 Double |
| 8 | Sniper 塔 | 空地 → 建 Basic；有 Basic → 升级 Quick；有 Quick → 升级 Sniper |
| 9 | Mortar 塔 | 空地 → 建 Basic；有 Basic → 升级 Mortar |
| 10 | Mortar+ 塔 | 空地 → 建 Basic；有 Basic → 升级 Mortar；有 Mortar → 升级 Mortar+ |
| 11 | Pulse 塔 | 空地 → 建 Basic；有 Basic → 升级 Mortar；有 Mortar → 升级 Pulse |
| 12 | Missile 塔 | 空地 → 建 Basic；有 Basic → 升级 Mortar；有 Mortar → 升级 Missile |
| 13 | Producer+ 塔 | 空地 → 建 Basic；有 Basic → 升级 Producer；有 Producer → 升级 Producer+ |
| 14 | Siege 塔 | 空地 → 建 Basic；有 Basic → 升级 Producer；有 Producer → 升级 Siege |
| 15 | Medic 塔 | 空地 → 建 Basic；有 Basic → 升级 Producer；有 Producer → 升级 Medic |
| 16 | 降级 | 查 (x,y) 塔 ID → 降级一级；已是 Basic → 拆除 |
| 17 | 闪电风暴 | 验证冷却和金币 → 部署 |
| 18 | EMP 轰炸 | 验证冷却和金币 → 部署 |
| 19 | 引力护盾 | 验证冷却和金币 → 部署 |
| 20 | 紧急回避 | 验证冷却和金币 → 部署 |
| 21 | 升基地出兵速度 | 无位置，仅由分类头 logit 决定 |
| 22 | 升基地兵种血量 | 无位置，仅由分类头 logit 决定 |

### 输出结构

```
spatial_feat (64, 19, 19)
      │
      ├── Conv1×1, 64→23 → action_map (23, 19, 19)
      │    每个通道表示"该类动作在对应位置的好坏"
      │    共享 1 份，不复制
      │
      └── GAP → MLP(64→64) → policy_base
                ┌── Linear(64→23) → head1_logits (23)
                ├── Linear(64→23) → head2_logits (23)
                └── Linear(64→23) → head3_logits (23)
```

- `action_map (23,19,19)`：8,303 logits，空间决策共享
- `head_logits ×3`：各 23 维，共 69 个额外 logits
- 每个头独立对应一个操作，支持最多 3 个操作的组合

### 解码流程

```
对于每个头 i:
  1. head_i_logits → mask 非法动作类 → argmax → class_id
  2. action_map[class_id] → mask 非法位置 → argmax → (x, y)
  3. 解码器根据 (class_id, x, y) 查场上状态，生成 Operation:
     - 塔类 (0-15): tower_at(x,y) 查塔 → 建/升级/降级
     - 武器类 (17-20): 验证冷却和金币 → Operation(USE_*, x, y)
     - 基地升级 (21-22): 无位置 → 直接生成
     - 若操作非法或无可做之事 → pass（合并到前一个有效操作的 bundle 中）

合并所有有效操作为 bundle 发送。
```

---

## 4. 输出：价值头

```
state_emb (128)
  → Linear(128→64)
  → ReLU
  → Linear(64→1)
  → tanh → [-1, 1]
```

输出标量，表示当前局面下当前玩家的胜率估计。

---

## 5. 与 AlphaZero 的对比

| 组件 | AlphaZero（围棋） | 本设计 |
|------|------------------|--------|
| Board size | 19×19 | 19×19 |
| Encoder | Conv+ResBlock | Conv7×7 + ResBlock×6 |
| Policy head | softmax over (19×19+1) | softmax over (23×19×19) |
| Value head | scalar | scalar |
| 池化 | 无 | 无 |
| 可做 MCTS 先验 | ✅ | ✅（8303 维分布） |

当前输出可自然转化为 AlphaZero 风格的动作先验分布：

```python
logits = head_logits[:, None, None] + action_map   # (23,19,19)
logits = logits.reshape(-1)                          # (8303,)
logits = where(mask, logits, -inf)
prior = softmax(logits)                              # MCTS 先验
```

---

## 6. 训练方案

采用**进化策略（ES）**：

- 用高斯噪声扰动网络权重，生成 N 个个体的种群
- 每个个体与基准对手（如 greedy AI 或上一代精英）对战 N 局
- 以**胜率**作为 fitness
- 加权平均更新均值参数
- **不设计奖励函数**，消除 RL 奖励工程的调参问题

> 后续可升级为 AlphaZero 自对弈 + MCTS（当前架构天然支持）

---

## 7. 升级树参考

```python
TOWER_UPGRADE_TREE = {
    BASIC:       (HEAVY, QUICK, MORTAR, PRODUCER),
    HEAVY:       (HEAVY_PLUS, ICE, BEWITCH),
    QUICK:       (QUICK_PLUS, DOUBLE, SNIPER),
    MORTAR:      (MORTAR_PLUS, PULSE, MISSILE),
    PRODUCER:    (PRODUCER_FAST, PRODUCER_SIEGE, PRODUCER_MEDIC),
}
```

解码器按此树走一步：若当前塔 `T_cur` 到目标 `T_target` 有一条边，则升级；若无，则向目标方向沿树走一步。
