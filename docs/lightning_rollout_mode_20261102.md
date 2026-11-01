# flat rollout 新模式 lightning：我方随机放闪电（2026-11-02）

> 状态：**实现完成、默认关、冒烟单测通过；16 局 A/B 进行中**（结果见 §5）。
> 一句话：`AZAI_FLAT_ROLLOUT_MODE=lightning` —— 把 flat rollout 的随机动作从
> "对手随机建塔/降级"（现役 `random`）换成 "**我方**以 decay(1,3) 概率随机放闪电"（class 17）。

## 0. 语义（最终澄清口径，避免混淆）

现役 `AZAI_FLAT_ROLLOUT_MODE=random` 的随机机制（`flat_rollout_greedy.py:_decide_actions`）是：

- 只随机 **对手** 那一侧（`rollout_random_side="opponent"` ⇒ `_sides=(1-root_player,)`）；
- 随机动作 = `if rng < rollout_build_ratio: 建塔(随机位置) else: 降级(随机拆塔)`；
- 没有闪电、没有升级、没有其它类型。

新模式 `lightning` 相对上面**改两处**：

| 维度 | `random`（现役） | `lightning`（新） |
|---|---|---|
| **侧** | 对手 `1-root_player` | **我方 `root_player`**（忽略 `rollout_random_side`） |
| **类** | 建塔(11) / 降级(13) | **闪电(21，class 17 = `USE_LIGHTNING_STORM`)** |
| 落点 | 随机合法高地格（build 同款 `ctx["cells"]` 随机位置） | 随机合法高地格（同上） |
| 合法性 | 金币够才建/有塔才降 | `cd==0` 且 `金币≥90` 才放；否则**空过**（这一模式只有闪电一种动作，不建塔也不降级） |
| 概率调度 / seed / 其余 | `decay(1,3)` 逐回合 `p(r)=C·exp(-r/k)` | **逐字共用** |

即：rollout 的 16 回合里，**我方**以 decay 概率随机放闪电，对手不动（保持现有 idle 行为）。

## 1. 为什么做这个实验

`docs/lightning_diagnostic_summary_20261102.md` 的结论是：值网是静态叶子评估器（horizon=16
看不到闪电一次性伤害的长期后果），导致 unified 侧"攒不到 90 放闪电"。本实验把 flat rollout 的
随机动作换成"我方放闪电"，让值网在叶子前 16 回合里**反复看到我方放闪电**，观察 unified 侧的
放闪电/攒钱行为是否随之改变。

## 2. 实现

### 2.1 改动点（默认关，random 逐字不变）

- `code/my_ai/az_intent/flat_rollout_greedy.py`：
  - `__init__` 的 `assert rollout_mode in ("idle","random","lightning")`（默认仍 `idle`，BASE 显式给 `random`）；
  - `_decide_actions` 加 `lightning` 分支：`_sides=(root_player,)`（我方）+ 命中时发
    `USE_LIGHTNING_STORM`（落点 = `ctx["cells"][_pl]` 随机，`cd==0 && coins≥90` 才发，否则空过）；
  - `search()` 里 `_rctx` 的构建条件从 `== "random"` 扩为 `in ("random","lightning")`（随机落点要读 `cells`）。
  - **random 分支一字未动**（`_sides` 的 `elif rollout_random_side=="opponent"` = 原逻辑；建塔/降级循环原样）。
- `code/my_ai/az_intent/unified_cand.py`：
  - `[shape_pin] engine=flat ...` 那行推广为 `[{cand_mode}] engine=flat ...`，让 unified 侧 stderr 也
    无条件打一行阶段 2 参数（含 `rollout_mode=lightning`）；lightning 时 `side` 显示成 `ours(root)`
    （因为忽略 `rollout_random_side`，直接显示 `opponent` 会误导）。
- `code/test_match/ab_search_modes.py`：
  - 新增 `AB_SEARCH_SUFFIX_TAG`（默认空 ⇒ `DSH_PROGRESS_SUFFIX` 与改动前**逐字相同**）；
    `_lightning_rollout_ab.sh` 设 `=_lightning`，让本批 stderr 文件名与并发批次（如 shape_pin）区分。

### 2.2 单测

`code/test_match/_smoke_lightning_rollout.py`（最小 fake sim 直测 `_decide_actions`）：

- 构造 `rollout_mode="lightning"` 通过；我方 cd==0+coins≥90 ⇒ `(我方, 21, x, y)` 且落点 ∈ 我方高地格；
- cd>0 或 coins<90 ⇒ 空；root=1 时侧 = 我方 1（不是对手 0）。全部通过。

## 3. 阶段 2 参数与 BASE 逐字一致（父任务硬约束）

lightning 臂**只**改 `AZAI_FLAT_ROLLOUT_MODE`（以及 `AZAI_CAND_MODE=unified`）；其余 flat rollout 参数
与 `_us_vs_external.sh:68` 的 BASE 逐字相同：

```
ckpt=checkpoints/posnet_A_k5_m32.pt  AZAI_SEARCH=flat  K=48
rollouts=64 horizon=16 batch=32 eval=net share_prefix=1
prob=decay(1.0,3.0) rounds=49152 seed=53
```

unified 侧 stderr 的 `[unified] engine=flat ... rollout_mode=lightning ...` 行即证明（把 self 上的
阶段 2 参数原样打出）；`_lightning_rollout_ab.sh` 里还有 BASE 逐字断言。

## 4. 实验设计

| 项 | 值 |
|---|---|
| A 臂 | `AZAI_CAND_MODE=unified` + `AZAI_FLAT_ROLLOUT_MODE=lightning` |
| B 臂 | `AZAI_CAND_MODE=pin` + BASE 自带 `AZAI_FLAT_ROLLOUT_MODE=random` |
| 两臂唯一差别 | 候选生成（unified vs pin）+ 随机动作的侧/类（我方闪电 vs 对手建塔/降级） |
| 对局 | 16 局 = seeds 100..107 × 2 座位（镜像配对，抵消先手偏） |
| 并行 | `--jobs 8`（串行：shape_pin 跑完让出 6 进程后开 8，总 ≤ 16 核纪律） |
| 驱动 | `code/test_match/_lightning_rollout_ab.sh` → `ab_search_modes.py`（ai0/ai1 都指 az_bridge_ai.py） |
| 记录 | `bash code/run_logged.sh lightning_rollout_ab bash code/test_match/_lightning_rollout_ab.sh`（WMI detached） |
| 有效性 | 只算 `verdict=engine`；stderr 唯一后缀 `_lightning` |

## 5. 结果

> 16 局 A/B 跑完后填：unified(闪电rollout) vs pin 的胜率（Wilson CI）+ 血差 Δhp（t CI）。

_待补_

## 6. 关键观察（unified 侧放闪电/攒钱行为有没有变）

> 跑完后填：对比 shape_pin 那批（unified 侧攒不到 90 放闪电），本批 unified 侧
> 放闪电次数 / 攒到 90 的决策 / `[act]` 里的 op21 频率。

_待补_

## 7. commit

- `1e0d461` 加 flat rollout 新模式 lightning（默认关）+ unified(闪电rollout) vs pin A/B 驱动/冒烟
- `6917d6b` ab_search_modes 加 `AB_SEARCH_SUFFIX_TAG`（默认空）避免并发批次 stderr 覆盖
