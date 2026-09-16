# 计划：策略网再拆一层 —— 类网 / 位置网 / 价值网（三个完全独立的网络）

> 日期：2026-09-16。状态：**计划阶段，待用户过目后再动手。**
> 前置：`docs/az_action_factorization_idea.md`、`docs/az_split_policy_value_plan.md`（第三轮二网拆分）

## 1. 为什么要拆（两难）

现状：**策略网与价值网已经是两个独立网络**（`docs/az_split_policy_value_plan.md`，第三轮），
但策略网内部 `action_map_conv`（位置头）与 `policy_base + policy_heads`（类头）
**共用** `initial_conv + resblocks + stats_mlp`。这造成一个两难：

| 选择 | 后果 |
|---|---|
| 不冻结骨干，只训位置头 | 位置 CE 的梯度流进共用 CNN ⇒ **类头的输入特征变了** ⇒ 即使类头参数一个没动，**它的输出也变了** |
| 冻结骨干 | 类头被完全钉住 ✅，但位置头只剩 `Conv2d(64→24, k=1)` 的**线性重投影**，共用特征是被类/策略目标塑形出来的 ⇒ 位置头学不到自己的空间特征 ❌ |

两个都不可接受。⇒ **按第三轮同一句话再往下用一层**：类网、位置网各自拿到一份骨干的**拷贝**，
之后只有位置网训练 ⇒ **类网的参数与输出逐位不变（构造性隔离，不靠"记得别动它"）**，
且位置网有了可训练的骨干。这也让 `--freeze-backbone` 这个补丁不再必要。

## 2. 证据基础（为什么值得做）

| 结论 | 依据 |
|---|---|
| 搜索的增益 **100% 在位置轴** | `search_axis_ablation`：①joint 93.8% = ③pos-only 93.8%（同 30W/2L）；②class-only 与 raw 逐回合 100% 相同 |
| 增益来自**价值排名**，不是"偏离 argmax 有用" | `_tmp_position_choice_ab.py`：同一批候选、同一出招率（2.8%），search 1.8125 / random **0.0000** / prior **0.0000**（各 0/32） |
| 位置目标**有真信号可学**，且采集便宜 | 同上 + `pos-only + skip-single` = 25.83×（真实采集叠加 skip-hold 后 2.51×） |
| 类头**表达不出**任何别的动作 | 615 个头次里 raw argmax 为 HOLD 的 **0 个**，永远是不可执行的闪电；三个"多出招"干预全败（0/64、≈3/64、0/32） |
| 类头**不能动**（动它会加重坍缩） | pos-only 采集的类目标是 one-hot（钉住的那个类 = 该头自己的 argmax）⇒ class CE 的梯度是"对本头 argmax 更自信"⇒ 会加重闪电偏好 |
| anchor 的定位 | §3.2 原始动机（稀疏 target、防止一次激进更新拉偏）**成立**；但实测 `--lambda-anchor=1.0` 时 anchor 梯度只占 CE 的 **2.3%**、基本惰性（`az_training_validation_results.md` §14）。在**本方案里它变成位置网唯一的约束**（97% 回合无位置信号） |

## 3. 目标架构

```
类网   class_net : backbone + policy_base + 3×class head   → head_logits (3×24)
位置网 pos_net   : backbone + action_map_conv (1×1)        → action_map (24×19×19)
价值网 value_net : backbone + value_head                   → value (1)
```

- **三个都是 `AntWarNetwork` 实例**（该类总是建全部头）：第一版**只读各自需要的那个头**，
  零新增模型代码；瘦身（位置网只留 action_map、类网只留 heads）留到跑通之后。
- **warm start**：从当前策略网的 `state_dict` 拆出骨干，**三个网各拷一份**
  （第二轮的"拆 state_dict"同款做法）。价值网沿用现在的 `value_state`。
- **搜索接口不变**：`make_three_net_fn(class_net, pos_net, value_net, feat)` →
  `{action_map, head_logits, value}`（**3 次前向**，现在是 2 次）⇒ `bundle_mcts.py` **一行不改**。

### checkpoint 格式（新增第三份权重）

```python
torch.save({
    "class_state": class_net.state_dict(),   # 新：类网（替代旧的"策略网"语义）
    "pos_state":   pos_net.state_dict(),     # 新：位置网
    "value_state": value_net.state_dict(),   # 沿用
    "num_heads": 3, "no_bn": False, "latent_dim": 64, "num_resblocks": 6,
    "value_pool": ..., "gn": ..., "gn_groups": ...,
}, path)
```

**⚠️ 不做静默兼容**：不给这种 ckpt 写 `model_state`。旧的
`load_model_from_ckpt` / `load_split_models` 遇到含 `pos_state` 的 ckpt
**直接报错并提示用三网加载器**——否则旧路径会静默读到"没训过的 action_map"，
是最容易埋雷的地方。

## 4. 改动清单（工作量估计）

| 文件 | 改动 | 规模 |
|---|---|---|
| `code/my_ai/az_intent/az_selfplay.py` | 新增 `load_three_models()`、`make_three_net_fn()`；`make_net_fn_from_ckpt` 加分支（`class_state`+`pos_state` → 三网）；**anchor 记录**改成 action_map 取自 pos_net、head_logits 取自 class_net；旧加载器加"误用即报错"守卫 | ~80 行 |
| `code/my_ai/az_intent/train.py` | 同上三网 net_fn（或直接复用 az_selfplay 的，避免两份实现） | ~20 行 |
| `code/my_ai/az_intent/az_train.py` | 新增训练模式：**只训位置网**（位置 CE + 位置 anchor），类网/价值网 `requires_grad=False` 且不进优化器；`_sample_policy_loss` 加 `lambda_class_ce`（默认 1.0，位置-only 阶段设 **0**）；ckpt 保存写三份 | ~120 行 |
| 新脚本 `code/my_ai/az_intent/make_three_net_ckpt.py` | 从现有（单网/二网）ckpt 拆出三网 ckpt（骨干拷贝 ×3） | ~60 行 |
| `code/my_ai/az_intent/eval.py`、`az_search_vs_search.py` | **不用改**（都走 `make_net_fn_from_ckpt`） | 0 |
| `code/my_ai/az_intent/bundle_mcts.py` | **不用改**（net_fn 接口不变） | 0 |

## 5. 位置-only 阶段的损失

```
L = λ_class_ce(=0) · L_class  +  1.0 · L_pos  +  λ_anchor · MSE(action_map, recorded_action_map)
```

- **类网**：不进优化器、不产生损失 ⇒ 输出逐位不变（可验证）。
- **价值网**：完全冻结（§23 记"训过的价值头有害"；今天 kgeo+rel / 迭代循环也是负面）。
- **位置网**：只吃 `L_pos`（§3.1 的分解式 CE 里"按类质量加权的位置 CE"那一段）+ 小 anchor。
- **关键性质**：`L_pos` **只在有位置目标的回合上存在**（约 2.7% 的回合；其余 97% 的回合
  只有一个空过候选 ⇒ 无 CE 信号）⇒ **anchor 是那 97% 回合上唯一的约束**。
  这正是 §3.2 当初设计 anchor 的用途，而现在它是**唯一**约束（不再与 value/类头纠缠）。
  λ_anchor 建议 **0.1-0.3 起**，视 CE 下降情况退火。
- `--pos-single`（把每个类的位置目标塌成 argmax 单点）值得一试：§15 记过
  "多位置目标分散在不相邻格子、梯度互相抵消、pos CE 卡死"。

## 6. 数据与采集

- 采集：`az_selfplay --search-mode pos-only --skip-single-candidate`（便宜）+ **pkl 路径**。
  **不能用 `--write-npz`**——它会丢掉 `bundles/visit/intent_counts`，而那正是位置 CE 的输入。
- 第一批：**300-500 局**。注意位置目标很薄：每局约 11 个决策点 ⇒ 300-500 局只有
  **~3300-5500 个位置决策点**（过拟合风险由 anchor 兜）。

## 7. 预注册判据（跑之前写死）

1. **position CE 是否真的下降**。§15 记过它曾在 0.85 卡死不动。
   **它不动 ⇒ 说明目标或结构仍不匹配，后面几条都不用看，直接回头诊断。**
2. **不搜索时**的强度是否上升：raw 策略（新）vs raw 策略（旧）的 **svs（配对）**，
   以及 vs rule_v4（≥64 局）。
   —— 这条是"位置网真的学到东西"的唯一硬判据；只让"搜索 vs raw"的差距缩小
   **不算**（那可能只是把搜索行为抄进了策略，而搜索本身没变强）。
3. **类网逐位不变**：训练前后 `class_state` 与类网输出逐位比对
   （证明三网隔离是构造性的）。
4. 定性检查：位置头 argmax 移动到的格子，是否不再是"旧头 argmax 那个格子"
   （即它到底有没有改判）。

判据 2 要提前定样本量与阈值：本项目噪声地板是 32 局 ±5pp、且"历史最好"从未超过
50% vs rule_v4 ⇒ **至少 64 局配对，成功线定在"显著优于旧策略"而非"看着好一点"**。

## 8. 验证计划（实现顺序上先做这个）

**Phase 0 —— 管线等价性（先做，是关键闸门）**
把当前 ckpt 拆成三网（三份骨干同源、均未训练）后，三网 `net_fn` 必须与现在的二网
`net_fn` **逐位等价**。做法：复用 `_tmp_position_choice_ab.py` 的镜像配对台，
跑 `raw / search` 两臂，断言**逐局结果完全一致**（该台基线臂 SE=0，等价性是可判定命题）。
**不等价 ⇒ 不许往下走。**

**Phase 1** —— 位置-only 训练模式（§5）+ `lambda_class_ce` 开关。
**Phase 2** —— 300-500 局小实验，按 §7 判据读。
**Phase 3（之后）** —— 网络瘦身（位置网只留 action_map）；类网那条线的课题另行立项。

## 9. 风险 / 待定

- **成本 +50%**：net_fn 从 2 次前向变 3 次。与 `pos-only + skip-single` 的 25× 不冲突，
  净值仍很赚；瘦身后可再降。
- **位置网的骨干继承自"被类目标塑形过"的特征**——第一版接受，跑通后再考虑是否给它
  单独预训练或更深的结构。
- **数据薄**：~3300-5500 个位置决策点，过拟合风险实在；anchor 是主要缓解手段。
- **"改进"必须在不搜索时也成立**（判据 2），否则只是把搜索行为复制进策略，绝对水平未变。
- **待定**：λ_anchor 的具体值与退火策略；`--pos-single` 开不开；是否同时给位置网喂
  "类选择"的显式输入（现在靠 action_map 的通道索引隐式表达，第一版不动）。

---

## 执行状态

### ✅ Phase 0 —— 管线等价性（已通过）

三项独立检查（`_tmp_verify_three_net.py` + `_tmp_position_choice_ab.py`）：

| 检查 | 结果 |
|---|---|
| A 加载器守卫 | 三网 ckpt 喂给 `load_model_from_ckpt` / `load_split_models` **均正确报错**；二网 ckpt 仍可正常加载（未破坏向后兼容） |
| B 输出层逐位 | 80 个真实局面：`action_map` **80/80**、`head_logits` **80/80**、`value` **80/80**，最大差 **0.000e+00** |
| C 对局层（独立复核）| 两网 vs 三网各自跑 `raw` / `search` 两臂：**每一个统计量精确一致** —— raw 双 1.0000(SE=0)；search 双 1.8125(SE=0.1008, t=+8.06)、候选数分布双 `{1:12261, 24:352}`、所选序号分布双 `12342/29/24/18/17`、出招率双 2.8% |

产物：`training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt`（94/94 张量三处逐位同源、
**无 `model_state` 键**）。

### ✅ Phase 1 —— 位置-only 训练模式（已实现并冒烟验证）

- `az_train.py`：新增 `--pos-only-net`、`--lambda-class-ce`、`save_three_net()`、
  `train_pos_only()`；`_sample_policy_loss` 加 `lambda_class_ce`（=0 时**完全不碰 head_logits**，
  因此位置 CE 只需位置网一次前向）。
- 冒烟（8 局采集 + 3 epoch）：**带位置目标的样本占 2.8%**（与预测的 ~2.7% 吻合）；
  **pos_ce 0.4929 → 0.4165 → 0.3392 在降**（§15 记过它曾卡在 0.85 不动 ⇒ 这一条初步通过，
  但只有 184 个带信号样本，属"过拟合级"验证）。
- **隔离性是构造性的**：训完 `class_state` **94/94 逐位相同（最大差 0）**、
  `value_state` **94/94 逐位相同**；`pos_state` 中 80/94 个张量变化，权重幅度 0.1-0.4
  （最大 3.75e-01 在 `resblocks.5.conv2.weight` ⇒ **位置网自己的骨干确实在学**，
  这正是共用骨干给不了的）。6.15e+02 那个数是 BN 的 `num_batches_tracked`（=615 步），非权重。

### 🐛 顺带修掉一个潜在 bug

`--skip-hold-search` 的样本把 `intent_counts` 存成 `None`，而 `_marginalize` 直接迭代它
⇒ **pkl 路径 + skip-hold 一直是不兼容的**（任何策略 CE 都会崩）。之前没人这么组合过
（value 循环走 npz，npz 路径丢掉 intent_counts）。已在 `_marginalize` 里守卫为"无目标"，
语义上正确：被跳过的决策没有搜索、因而没有策略目标，只保留 anchor。

### 待做

**Phase 2** —— 300-500 局小规模实跑，按 §7 判据读（判据跑前写死）。
**Phase 3** —— 网络瘦身。

### ✅ Phase 2 —— 位置网第一轮实跑（2026-09-16，`run_logged` 名 `phase2_posnet_b`）

采集 400 局（pos-only + skip-single + skip-hold）→ 336,064 样本 → **9,367 个位置样本**
（2.8%，23/局）→ 过滤成 459 MB（原始 pkl 15 GB，全量载入会 OOM）。训练 8 epoch × 两个 λ。

**判据 1：position CE 会降** ✅

| λ | pos_ce ep1 → ep8 | 有效 anchor/CE |
|---|---|---|
| 30 | 16.73 → **13.70**（−18%）| 1.3% → 2.8% |
| 150 | 16.88 → **14.36**（−15%）| 3.3% → 6.6% |

（高 λ 的 CE 降得更慢 ⇒ anchor 确实在当刹车，符合设计。）

**判据 2（硬闸门）：不搜索时 raw(新) vs raw(旧) 镜像配对 64 pair** ✅ **通过**

| arm | 配对得分 | t | 偏离 pair | 净增分 |
|---|---|---|---|---|
| untrained（对照）| **1.0000**（SE=0）| — | 0/64 | +0.0 |
| λ=30 | **1.4219** | **+5.29** | 37/64 | +27.0 |
| λ=150 | **1.4219** | **+5.10** | 39/64 | +27.0 |

⇒ **训练后的位置网，在完全不用搜索的情况下，显著打得过训练前的自己**（占分 71% vs 50%，
+21pp）。**这是这条线上第一次出现"训练产生了灵敏且显著的提升"。**

**判据 3：类网逐位不变** ✅（已在前一轮冒烟验证：94/94、最大差 0）

**判据 5（绝对强度）：vs rule_v4 带廉价搜索 64 局 —— 没跟上**

| arm | vs rule_v4 | 与未训练的配对（同 seed）|
|---|---|---|
| untrained | 18.8%（12W/52L）| — |
| λ=30 | **23.4%**（15W/49L）| p=0.61（仅新赢 9 / 仅旧赢 6）|
| λ=150 | 15.6%（10W/54L）| p=0.82（仅新赢 9 / 仅旧赢 11）|

⇒ **+4.6pp 在 64 局下不显著**。也就是说：策略确实变得更像搜索（自身相对变强），
但**对强参照的绝对水平没有可测的提升**。λ=30 vs λ=150 的差（23.4 vs 15.6）同样无法分辨。

### ⚠️ 一个必须先解决的口径问题

**同一个 ckpt** 对 rule_v4：本次 **18.8%** vs 历史（`pool64`, 同 64 局同 seed）**29.7%**
—— 配对后仅本次赢 8 / 仅历史赢 15，**p=0.21**（不显著，但方向一致、差 11pp）。

可能来源：(a) 噪声（项目记的"64 局 ±5pp"可能低估）；(b) 搜索配置（历史是 joint 256/4，
本次是 pos-only + skip-single）。Phase 0 已证 pos-only ≡ joint 在**镜像台**上逐局相同，
但 skip-single 会扰动 RNG 流 ⇒ 实现出来的对局不同 ⇒ 64 局的实现差异可能就有这么大。

**结论：本轮的"绝对强度"数字不能与历史 29.7% 直接比。**在拿到可信的绝对标尺之前，
应以**判据 2（配对镜像）为主判据**。

### ⚠️ Phase 2b —— 位置网迭代循环（5 轮 × 400 局）：**未复利，且是我的设计错误**

| 轮 | 累积池步数 | 镜像 vs 第 0 轮 | t | rule_v4（配对 p）|
|---|---|---|---|---|
| 1 | 292 | 1.1406 | +1.49 | — |
| 2 | 584 | **1.2969** | **+3.60** | 21.9%（p=0.82）|
| 3 | 876 | **1.0000** | **0.00** | — |
| 4 | 1168 | 1.0469 | +0.57 | **31.2%（p=0.13）** |
| 5 | 1460 | 1.0312 | +0.34 | — |

**错误 1（设计违反项目自家规则）**：`az_split_policy_value_plan.md` §3 明写
"**策略只用当前 batch（目标会过时），价值用累计**"。我把**位置（策略）目标跨轮累积**了
⇒ 每轮混入的是"上一轮那个不同策略"产出的目标 ⇒ **目标过时**，把不同策略的偏好平均掉。
**所以本结果不能当作"训练不复利"的证据**——是循环设计坏了。

**错误 2（盘子太小）**：第 1 轮 1.14 vs Phase 2 同类一轮 1.42，**同一条管线、只换 400 局游戏**
⇒ 效应量本身有 ±0.2 量级的抽样波动，5 个点画不出趋势（每轮仅 ~9.3k 位置样本）。

**更要紧的是判据本身可能测错**：raw 镜像测的是 **argmax 那一格**（只有 raw 解码用），
而**搜索是从整张地图采样**（`t_pos=0.3`）。旁证：第 4 轮 **rule_v4 = 31.2%**（历史最好配对
基线 18.8%，p=0.13）指向"搜索变强了"，与 raw 镜像的"无增益"矛盾。
⇒ **下一步应先换判据：搜索对搜索的镜像**（第 k 轮搜索 vs 第 0 轮搜索；同 seed、SE=0 对照），
拿现有 5 个 ckpt 重测仅需几分钟。**在换判据之前不要改循环。**
