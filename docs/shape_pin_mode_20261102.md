# shape_pin 模式：介于 pin 与 unified 之间的候选生成（2026-11-02）

> 状态：**实现完成，16 局 A/B 进行中/已出结果**（结果见 §4）。
> 设计演进记录：本模式是 `docs/unified_cand_design_20261029.md` 的后续扩展，那一篇里
> 「§2 末尾 shape_pin 模式」小节有本模式的精简版描述，本文是完整版（模式定义 + 实现 + 结果）。

## 0. 一句话

`AZAI_CAND_MODE=shape_pin`：阶段 1 照跑 unified 的 bandit + 位置游标，但 s1 只用来挑**动作类的
有序序列**（"先降级→再建塔"这类有顺序的多重集）；锁定该序列后，**位置由位置网先验（游标顺序，
argmax→rank k+1）直接取前 k 个**，不再按 s1 给所有候选排序。阶段 2（flat rollout 评估器）**一字不动**。

## 1. 模式定义（用户原话转述 + 澄清）

### 1.1 两阶段

- **第一阶段**：照常跑统一搜索（`unified_cand` 的 bandit + 位置游标），得到各候选的 s1（idle rollout 分）。
- **第二阶段（改）**：不直接按 s1 给所有候选排序，而是：
  1. 找到 s1 排第一的那个候选，提取它的**动作类序列**（有顺序的多重集，比如 `[降级, 建塔]`，注意顺序）；
  2. 锁定这个"有序序列类型"，只从它里面、按**网络先验分（= 游标顺序，argmax→rank k+1）**挑 top-k 个
     位置（k 默认 48，凑不满就少输出）；
  3. 这些 top-k 位置就是进阶段 2（flat rollout）的候选。

### 1.2 顺序合法性

候选动作的合法性依赖顺序（如"先降级回收金币→再建塔"合法，"先建塔→再降级"可能不合法）。所以锁定的是
**有序序列**，不是无序多重集；生成时顺序要和 s1 第一名一致。

> ⚠️ 实现口径说明：现役 `unified_cand` 的 decode 顺序 = **排序后的类序列**（`_decode_group` 按
> `shape`（排序多重集）逐类解码，见 `unified_cand.py`）。所以从 s1 第一名候选里提取的"有序类序列"
> 就是它的**实际 decode 顺序**（= 排序顺序），本模式原样锁定、不重新排序。顺序语义（先降级还是先建塔）
> 由现役 decode 决定，本模式不改变 decode —— 只改"阶段 1 结束后选哪些候选进阶段 2"这一处。

### 1.3 关键澄清（不确定时按这个理解）

- "s1 排第一的动作组" = 阶段 1 里 s1 分数最高的那个候选，它的 class 序列（按 decode 顺序，如
  `[16(降级), 0(建塔)]`）。
- "按游标顺序挑 k 个位置" = 对那个 class 序列，用 `unified_cand` 里已有的位置游标顺序
  （网络先验 argmax 顺序）生成位置，取前 k 个，而不是用 s1 排序。
- 若该序列只有一个 class（如纯建塔），就是"pin 到那个 class + 网络先验取前 k 位置"。
- 若 s1 第一名是 HOLD（空束），锁定"空序列" = pin 到 HOLD：候选 = 仅 HOLD 一条（见 §2 实现）。

## 2. 实现

### 2.1 与 pin / unified 的区别（各一句）

| 模式 | 候选来源 |
|---|---|
| **pin** | 一个类（MC 仲裁钉住 `pinned=[_ccls]*3`）+ 该类内位置采样 ⇒ 只搜一个类 |
| **unified** | 跨类 × 位置 bandit，最后按 **s1（阶段 1 idle rollout 分）** 对全部候选排序取 top-k ⇒ s1 同时决定"选组"和"选位置" |
| **shape_pin** | s1 只决定"选哪个动作类序列"，**位置由位置网先验（游标）决定** ⇒ 把"s1 选组"与"s1 选位置"两个角色拆开 |

### 2.2 改动点（只改一处）

- `code/my_ai/az_intent/unified_cand.py`：
  - `UnifiedCandidateGen.__init__` 新增 `mode="unified"` 参数（默认 unified ⇒ 现役行为逐位不变）；
  - `_decode_group` 现在返回 `(ops, classes)`（`classes[i]` = 产出 `ops[i]` 的通道号，按 decode 顺序）；
  - `_emit` / `picked_out` 记录每个候选的 `classes`（有序类序列）；
  - 新增 `_shape_pin_topk(seq, k)`：锁定有序类序列，按位置游标顺序（混合进制，槽 0 变化最快，步 0 =
    全体 argmax）枚举位置，取前 k 个合法候选（canon_ops 去重、空束跳过、cache 复用/填 s1）；
  - `generate()` 在阶段 1 结束后分支：`mode=="shape_pin"` 时按上述逻辑选候选，否则按原 s1 top-k。
- `code/test_match/az_bridge_ai.py`：
  - `AZAI_CAND_MODE` 合法值从 `pin|unified` 扩为 `pin|unified|shape_pin`（默认仍 pin）；
  - `shape_pin` 与 `unified` 共用 `make_unified_searcher_class()`，并把 `cand_mode` 传给搜索器
    （修掉了"`AZAI_CAND_MODE=shape_pin` 但搜索器仍按 unified 走"的传递漏洞）；
  - shape_pin 侧 stderr **无条件**打印 `[shape_pin] engine=flat rollouts=64 horizon=16 ...`，
    证明阶段 2 参数与部署 BASE 逐字一致（与 unified/pin 的 stderr 同款）。

### 2.3 阶段 2 参数与 BASE 逐字一致（父任务硬约束）

shape_pin **不碰** `FlatRolloutGreedy` 的任何参数。阶段 2 仍是：

```
rollouts=64 horizon=16 batch=32 rollout_mode=random side=opponent
prob=decay(1.0,3.0) rounds=49152 seed=53 share_prefix=1 eval=net k=48
```

与 `_us_vs_external.sh:68` 的 BASE 逐字相同；两臂（shape_pin vs pin）唯一差别 = `AZAI_CAND_MODE`。
stderr 里的 `[shape_pin] engine=flat ...` 行即证明（把 self 上的阶段 2 参数原样打出）。

## 3. 实验设计

| 项 | 值 |
|---|---|
| A 臂 | `AZAI_CAND_MODE=shape_pin` |
| B 臂 | `AZAI_CAND_MODE=pin`（现役 pin+flat） |
| 两臂共同 | 部署 BASE（`_us_vs_external.sh:68` 逐字）：ckpt=posnet_A_k5_m32.pt、AZAI_SEARCH=flat、K=48、rollouts=64、horizon=16、batch=32、random+opponent+decay(1,3)、rounds=49152、seed=53、share_prefix=1 |
| 对局 | 16 局 = seeds 100..107 × 2 座位（镜像配对，抵消先手偏） |
| 并行 | `--jobs 6`（留 CPU 给别的实验） |
| 驱动 | `code/test_match/_shape_pin_ab.sh` → `ab_search_modes.py`（ai0/ai1 都指 az_bridge_ai.py，各自 env） |
| 有效性 | 只算 `verdict=engine` |

## 4. 结果

> 16 局 A/B 跑完后填：shape_pin vs pin+flat 的胜率（Wilson CI）+ 血差 Δhp（t CI）+ 候选构成。

_待补_

## 5. 关键观察（shape_pin 候选生成出来是什么样）

> 跑完后填：shape_pin 每决策锁定的有序类序列直方图、n_cand 分布（vs pin）、含 HOLD/升级占比、
> 长度分布，与 pin/unified 的差异。

_待补_
