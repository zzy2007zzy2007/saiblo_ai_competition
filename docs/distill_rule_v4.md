# 蒸馏 RuleV4 方案设计

## 目标

将规则 AI `rule_v4` 的策略知识迁移到神经网络模型，得到一个具有 rule_v4 同等水平、且可作为 ss_train 初始权重的 checkpoint。蒸馏后的模型用作 ss_train 的初始均值，减少从随机权重探索的成本。

## 方法：行为克隆（Behavior Cloning）

纯规则 AI 没有梯度可传播，不能用 logit-level KD。采用 **行为克隆**——让 rule_v4 打比赛，记录其决策轨迹作为监督数据，训练神经网络拟合规则策略。

```
rule_v4 (老师)  →  打比赛  →  记录 (board, stats, action, position)  →  BC训练  →  神经网络 (学生)
```

## 数据采集

### 比赛配置

- rule_v4 vs rule_v4（自我对称），避免策略偏差
- 也可用 rule_v4 vs ExampleAI，或在两个 player 槽位各放一个 rule_v4
- 建议采集 200-500 局，每局约 200-300 回合 → 总计 40K-150K 样本
- 使用多进程加速（纯 CPU，不需要 GPU）

### 每回合记录字段

| 字段 | 形状 | 说明 |
|:----|:-----|:-----|
| `board` | (28, 19, 19) float16 | 局面编码（与 ss_train BC 数据相同） |
| `stats` | (42,) float16 | 统计数据编码（与 ss_train BC 数据相同） |
| `class_` | (N_heads,) int64 | rule_v4 的操作类别，3 个 head 用相同标签 |
| `action_map` | (24, 19, 19) float16 | 位置峰值图 |
| `head_logits` | (N_heads, 24) float16 | 仅用于格式兼容，实际设为 one-hot |

### 标签映射

rule_v4 的 `Operation` → 神经网络 class label + position：

| rule_v4 操作 | Class | 位置来源 | 说明 |
|:------------|:------|:---------|:-----|
| `BUILD_TOWER(x, y)` | 0 (build_Basic) | `(x, y)` | 建塔位置 |
| `UPGRADE_TOWER(id, target_type)` | 见 TowerType→Channel 映射表 | `tower.pos` | 升级的塔所在位置 |
| `DOWNGRADE_TOWER(id)` | 16 | `tower.pos` | 拆塔位置 |
| `USE_LIGHTNING_STORM(x, y)` | 17 | `(x, y)` | 闪电中心 |
| HOLD（什么都没做） | 23 | 无 | 全零 |

TowerType→Channel 映射（来自 `decoder.py` 的 `CHANNEL_TO_TOWER_TYPE`）：

| TowerType | Channel |
|:----------|:--------|
| HEAVY | 1 |
| HEAVY_PLUS | 2 |
| ICE | 3 |
| BEWITCH | 4 |
| QUICK | 5 |
| QUICK_PLUS | 6 |
| DOUBLE | 7 |
| SNIPER | 8 |
| MORTAR | 9 |
| MORTAR_PLUS | 10 |
| PULSE | 11 |
| MISSILE | 12 |
| PRODUCER_FAST | 13 |
| PRODUCER_SIEGE | 14 |
| PRODUCER_MEDIC | 15 |

注意 rule_v4 只升到 Producer 系列的 4 种，不会升级到 Heavy/Ice 等战斗塔。

### action_map 构建

```python
action_map = np.zeros((24, 19, 19), dtype=np.float32)
if class_label != 23:
    action_map[class_label, pos_x, pos_y] = 10.0  # 单峰标签
```

3 个 head 都用相同的 class 和 action_map（rule_v4 是确定性策略，没有多头差异）。

### head_logits 构建

```python
head_logits = np.full((num_heads, 24), -10.0, dtype=np.float32)
head_logits[:, class_label] = 10.0  # one-hot 风格 logits
```

ss_train 的 SSDataset 要求这个字段存在，但蒸馏时不会用它做温度采样变异，所以 one-hot 即可。

### 数据存储格式

每局一个 `.npz` 文件，命名格式（与 ss_train 的 `write_npz` 兼容）：

```
distill_gen_0000_ind000_seed42.npz
```

文件内容：

```python
np.savez_compressed(
    path,
    board=board.astype(np.float16),
    stats=stats.astype(np.float16),
    class_=class_labels,               # (T, N_heads) int64
    action_map=action_map.astype(np.float16),  # (T, 24, 19, 19)
    head_logits=head_logits.astype(np.float16),  # (T, N_heads, 24)
)
```

## 训练

直接复用 `ss_train.py` 的 `SSDataset` 和 `ss_supervised_update`：

```python
npz_paths = sorted(Path("distill_data/").glob("distill_*.npz"))
ds = SSDataset(npz_paths, p_hold=1.0)  # rule_v4 没有 HOLD downsampling 必要
ss_ret = ss_supervised_update(
    model, ds, device="cuda",
    epochs=10,
    lr=1e-3,
    batch_size=64,
    lambda_class=1.0,
    lambda_map=1.0,
)
```

### 超参数建议

| 参数 | 值 | 理由 |
|:----|:---|:-----|
| `epochs` | 10-20 | 纯 BC 不需要太多 epoch，5-10 轮后 loss 趋于平稳 |
| `lr` | 1e-3 | 标准学习率 |
| `batch_size` | 64 | 标准 batch size |
| `lambda_class` | 1.0 | 分类准确率优先 |
| `lambda_map` | 1.0 | 位置精度同等重要 |
| `p_hold` | 1.0 | 不降采样 HOLD（rule_v4 的 hold 行为也是策略的一部分） |

训练完成后保存 checkpoint：

```
distill_rulev4_gen_0020.pt
```

## 集成到现有训练流程

```
蒸馏 checkpoint
       │
       ▼
ss_train.py --checkpoint distill_rulev4_gen_0020.pt --sigma 0.0002 ...
       │
       ▼
    自我对弈进化
```

关键点：
- `--sigma` 可以用小值（0.0002），因为初始策略已经可用，不需要大量参数噪声探索
- 蒸馏后的权重作为 mean 的初始值，ss_train 在其基础上做 BC 变异 + 标签变异探索
- 如果蒸馏效果好，ss_train 的收敛速度会比从随机初始化快 5-10 倍

## 评估方法

### 训练过程中评估

每 N 个 epoch 后对战 rule_v4，监控蒸馏损失：

```python
蒸馏 epoch 1:  vs rule_v4  胜率 12%
蒸馏 epoch 5:  vs rule_v4  胜率 35%
蒸馏 epoch 10: vs rule_v4  胜率 58%  ← 接近规则上限
蒸馏 epoch 20: vs rule_v4  胜率 62%  ← 略有超出（泛化优势）
```

### 蒸馏完成后评估

用 `eval_checkpoint.py` 评估蒸馏模型：

```bash
# vs rule_v4
python code/test_match/eval_checkpoint.py distill_rulev4.pt --opponent rule_v4 --games 50

# vs ExampleAI
python code/test_match/eval_checkpoint.py distill_rulev4.pt --games 50
```

## 预期效果

| 指标 | 预期值 |
|:----|:-------|
| 蒸馏后 vs rule_v4 胜率 | ~50-60%（超越规则水平说明学到了泛化） |
| 蒸馏后 vs ExampleAI 胜率 | ~80%（rule_v4 本身对该对手的胜率） |
| ss_train 收敛代数（从蒸馏开始） | 20-50 代（从随机开始需要 100+ 代） |
| 数据采集时间（500 局） | ~5-10 分钟（CPU 多进程） |
| BC 训练时间 | ~1-2 分钟（GPU） |

## 风险和注意事项

### 1. 分布偏移

rule_v4 的策略非常简单——只在最边上（y=1 或 y=17）建塔和升级。蒸馏后的网络在 BC 阶段只见过这些"边上"局面。进入 ss_train 自我对弈后，网络可能遇到 rule_v4 不会访问的局面（例如中线或后方建塔），输出可能会退化。

缓解：ss_train 的自我对弈 + 标签变异会自然修正这个问题。

### 2. 动作空间覆盖不全

rule_v4 只用以下操作：
- BUILD_TOWER（建 Basic 塔）
- UPGRADE_TOWER（升级到 Producer 系列）
- DOWNGRADE_TOWER（拆塔）
- LIGHTNING（闪电风暴）
- HOLD（什么都不做）

不用的操作：EMP、Deflector、Evasion、UPGRADE_GENERATION_SPEED、UPGRADE_GENERATED_ANT，以及 Heavy/Ice/Sniper 等战斗塔升级路线。蒸馏后网络对这些操作的 logits 会很低，ss_train 需要额外探索才能学会。

### 3. rule_v4 的多头兼容

rule_v4 是确定性策略，每个回合只有 1 个最优操作。3 个 policy head 的训练标签完全一样，不会出现 ss_train 中不同 head 学到不同策略的情况。这可能导致蒸馏后 3 个 head 的权重趋同，ss_train 变异时 head swap 的效果会减弱。

### 4. HOLD 行为不一致

rule_v4 的 HOLD 行为是有策略意图的（"攒钱等闪电"、"等钱升级"），但 BC 训练时神经网络看到的是 `board + stats → HOLD` 的映射。神经网络可能无法从 board/stats 中推断出背后的"等钱"逻辑，导致学到的是"盲目 HOLD"。ss_train 的探索阶段会修正这个问题。

## 实现计划

### 阶段 1：数据采集脚本（distill_collect.py）

- 加载 rule_v4 AI
- 用 GameState 运行多局比赛
- 每回合记录 board/stats/class/action_map/head_logits
- 写入 .npz 文件

### 阶段 2：训练脚本（distill_train.py）

- 读取 .npz 文件，检查数据完整性
- 调用 SSDataset + ss_supervised_update
- 定期保存 checkpoint
- 评估 vs rule_v4 胜率

### 阶段 3：ss_train 集成验证

- 用蒸馏 checkpoint 初始化 ss_train
- 对比从随机初始化的收敛曲线
- 确认不会 diverge
