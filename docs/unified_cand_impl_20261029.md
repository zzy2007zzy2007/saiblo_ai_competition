# 统一候选生成（方案 D）实现报告 —— 阶段 1（实现 + 零影响 + 1 局冒烟）

> 状态：**已实现并验证**（2026-10-29/30）。
> 冻结设计：`docs/unified_cand_design_20261029.md`（下称"设计"）。
> 开关：`AZAI_CAND_MODE=pin|unified`（默认 `pin` = 现役行为逐位不变）。

## 0. 一句话

新增候选生成器"类 UCB bandit + 位置游标"（`code/my_ai/az_intent/unified_cand.py`）：
**只换候选来源**，阶段 2 的 flat rollout 评估器一行不动；`pin` 时逐位不变（两层证据）。

## 1. 改了什么（3 个文件）

| 文件 | 改动 | 零影响理由 |
|---|---|---|
| `code/my_ai/az_intent/unified_cand.py`（**新增**） | `UnifiedCandidateGen`（方案 D）+ `make_unified_searcher_class()`（子类覆盖 `_expand`） | 新文件，没人 import 就一个字节都不跑 |
| `code/my_ai/az_intent/flat_rollout_greedy.py` | +62/-3 行：① `idle_rollout_values()`（**新方法**）② `search()` 里 `_reuse` 钩子 ③ 4 个属性 | `cand_mode` 默认 `"pin"` ⇒ `_reuse` 恒 `None` ⇒ 只多一次 `is not None` 判空；`idle_rollout_values` 只在统一生成器里当 `rollout_fn` |
| `code/test_match/az_bridge_ai.py` | +42/-1 行：读 `AZAI_CAND_MODE` + `if unified` 换搜索器类 + 冲突开关 warning | `AZAI_CAND_MODE` 不设/`pin` ⇒ `_searcher_cls = FlatRolloutGreedy`（**同一个类**），参数逐字相同 |

**`bundle_mcts.py` / `decoder.py` / `network.py` 零改动**（`git status` 干净）：现役候选束构造
`pinned=[_ccls]*3`（`bundle_mcts.py:884`）那段**原样保留**，统一候选用的是**子类覆盖 `_expand`**
（`flat_rollout_greedy.py` 的 `search()` 只认 `root.bundles`）⇒ pin 路径连"新增语句"都没有。

## 2. 实现说明（逐条对应设计 §2 方案 D）

### 2.1 候选来源：类 UCB + 位置游标（`UnifiedCandidateGen.generate`）

1. **类选择 = UCB**（`_ucb_pick`）：
   * 利用项 `_lse(c) = (1/β)·log((1/n)·Σ exp(β·x))`，`β = AZAI_CAND_BETA`（默认 3.0）；
   * 探索项 `+ c·√(log N / n)`，`c = AZAI_CAND_UCBC`（默认 1.0）；未试过的臂 = `+inf`（UCB1 标准初始化）；
   * `Q[c]` 初始**全相等（均匀）**；**类网先验接口预留但暂不用**（`net_out["head_logits"]` 一行都没读）。
2. **位置游标**（`_next_cell`）：该类**上一次**提议的位置在该通道位置图里的名次 `k` ⇒ 取第 `k+1` 名；
   首次 ⇒ 该通道位置图 argmax。取第 k 名用 **`np.argpartition` 定"第 k 大的值"**（O(n)），
   并列只在"等于第 k 大"的那一小撮里按 `(x,y)` 字典序补 ⇒ **不做全排序**。
3. **游标键 = 动作类的规范多重集**：`key = tuple(sorted({类} - {HOLD}))`；
   `(hold,build,lightning)` 与 `(build,hold,lightning)` 都归一到 `(build,lightning)`。
   位置分（日志用）= 各类位置图分数之**积**（`_pv`）。**λ = 0**：位置分**不参与**候选选择权重。
4. **无位置的类**（HOLD=23 / 基地升级=21,22）：`cells.get(c) is None` ⇒ 返回 `NO_POSITION`，
   游标恒指向那一个动作（单位置臂）。
5. **一个 bundle ≤ 3 op**（与网络 3 个头一致）：每轮按 UCB 选 3 个**互不相同**的类（同一束内不重复；
   HOLD 例外——它可以当"填充位"，因为 HOLD 不产生 op ⇒ 束长自然变成 1/2 op）。
6. **通道 → op 精确映射**（`decode_channel_op`，照 `decoder.py`，不自己发明）：

   | 通道 | 产出 | 实现 |
   |---|---|---|
   | 0–15 | BUILD(11) / UPGRADE(12) | 直接调 `decoder._decode_tower_action`（`decoder.py:433-456`）：空格 ⇒ `BUILD_TOWER(x,y)`（**建 Basic，忽略 target_type**）；我方塔 ⇒ `UPGRADE_TOWER(tower_id, step)`（step 由 `upgrade_step`，`decoder.py:82-94`）；敌方塔 ⇒ `None` |
   | 16 | DOWNGRADE(13) | 我方塔格 ⇒ `DOWNGRADE_TOWER(tower_id)`（`decoder.py:426-428`） |
   | 17–20 | 21/22/23/24 | `CHANNEL_TO_SUPER_WEAPON` → `SUPER_WEAPON_TO_OP_TYPE`，位置即 `(x,y)` |
   | 21 | 31 | `UPGRADE_GENERATION_SPEED`（`decoder.py:369-371`） |
   | 22 | 32 | `UPGRADE_GENERATED_ANT`（`decoder.py:372-373`） |
   | 23 | 空 bundle `()` | HOLD，不产生 op |

7. **逐条 `can_apply`**（`_decode_group`）：每个 op 与**已 pending 的 op** 一起检查
   （`state.can_apply_operation(player, op, pending)`）；重复 op 三元组先去掉（重复施加会崩）；
   `can_apply` **抛异常也当非法丢弃**（严格模式一条非法 = 判负，绝不能漏）。
   位置掩码本身用 `intent_decoding=False`（**连金币一起查**）⇒ 枚举阶段就已"买得起"，`can_apply` 是第二道防线。
8. **去重**：键 = `canon_ops(bundle)` = `(op_type,arg0,arg1)` **排序后**的元组（HOLD = `()`）。
   排序只当键用，**真实下发顺序不变**（顺序有语义：降级退钱 → 建塔）。

### 2.2 两阶段 rollout

* **阶段 1**（`generate` 内，逐候选 1 次）：`rollout_fn = FlatRolloutGreedy.idle_rollout_values`
  ——**确定性 idle rollout**（双方不出招）：`clone → apply(ops) → (P1 才 advance_round) → h 次
  advance_round → 叶值网`，与 `search()` 里 `rollout_mode="idle"` 那条路径逐位同语义。
* **候选选择**：按阶段 1 分**降序取 top-k**（生成池 = `k × AZAI_CAND_POOL`，默认 1.5 ⇒ 72 条，
  再截到 k=48）。λ=0。
* **阶段 2**：`FlatRolloutGreedy.search()` **原样执行**（`random` + opponent-only + decay(1.0,3.0)
  + horizon16 + share_prefix=1 + N=64；预算公式 `max_total_rounds/(k×h)` 不变）。
  ⇒ "阶段 2 不动"是**结构上**成立的（子类只覆盖 `_expand`），不是靠对齐参数。
* **缓存 / 复用**：`self.cache[canon_ops] = [value]`（同一 bundle 只算一次）。
  阶段 2 复用由 `FlatRolloutGreedy.search()` 里的 `_reuse` 钩子实现：
  **只有 P1 与 P2 的估计器签名完全一致时才复用**（`_p1_signature() == _p2_signature()`）。

### 2.3 早停（避免空转）

连续 `max(32, 2k)` 次尝试都只解出**已见过**的 bundle ⇒ 该局面的可达候选空间已被枚举完
（例：开局金币不够建塔/升级，只剩降级 + HOLD）⇒ 立刻停。实测把"开局 432 次空转到 96 次、
37 ms → 22 ms"。

### 2.4 接入点

`az_bridge_ai.build_engine()`：
* `AZAI_CAND_MODE=unified` **且** `AZAI_SEARCH=flat` ⇒ 构造 `UnifiedCandFlatSearch`；
* 其它组合（`AZAI_SEARCH=mcts|unified`）⇒ 一行 warning，**按 pin 跑**（阶段 2 必须是现役 flat 评估器）；
* `AZAI_K` 未设时 unified 用 **48**（pin 的默认仍是 1）；
* 设计 §4.6：`AZAI_PIN_K`/`AZAI_ALWAYS_HOLD`/`AZAI_MC_UPG`/`AZAI_MC_CELLS`/`AZAI_UPG_BUNDLES`
  在 unified 下**不再生效**（打 warning）；`AZAI_ONLY_EXTRA` 会丢掉统一候选（也 warning）；
  注入类（`AZAI_ORACLE_INJECT*`）**不忽略**——设计 §3 要求与统一候选合并去重（走 `set_extra_bundles`）。

### 2.5 新开关一览

| 开关 | 默认 | 含义 |
|---|---|---|
| `AZAI_CAND_MODE` | `pin` | `pin` = 现役；`unified` = 方案 D 候选生成 |
| `AZAI_K` | pin: 1 / unified: 48 | 候选数 k（两侧共用同一个变量） |
| `AZAI_CAND_BETA` | 3.0 | LogSumExp 的 β |
| `AZAI_CAND_UCBC` | 1.0 | UCB 探索常数 c |
| `AZAI_CAND_POOL` | 1.5 | 生成池 = k×pool（再按阶段 1 分取 top-k） |
| `AZAI_CAND_ATTEMPTS` | 6 | 去重后凑不满时最多再试 k×该值 次 |
| `AZAI_CAND_FORCE_HOLD` | 1 | 候选里**恒含 1 条空束**（设计 §6 第 3 问未拍板 ⇒ 取保守选项） |
| `AZAI_CAND_VERBOSE` | 0 | 每决策一行 `[cand] mode=unified n_cand=… 含HOLD=… 含升级=…`（stderr） |

## 3. 零影响证据（两层）

### 3.1 搜索器级：同局面 pre vs live，逐字段哈希（8 个真实局面，全部相同）

脚本 `code/test_match/probe_cand_mode_zero_impact.py`；日志
`match_results/unified_cand/zero_impact_probe.log`。
做法：`code/test_match/_flat_rollout_greedy_preunifiedcand.py`（**动手那一刻**的副本，
md5 `5B92A4865B849BFCE16BCE65938E1424`）用 `importlib` 单独加载 + `sys.modules` 临时替换
（**不动工作区**、不换文件 ⇒ 不污染正在跑的实验）；两侧用**同一份** `az_bridge_ai.build_engine`
+ 同一 ckpt + 同一 BASE 配置；局面从 `INJECT_VS_RU_B/8.log` 的 `--trace-ops` 重放得到。

读数（`match_results/unified_cand/zero_impact_probe.log`，一行 = 一个真实局面）：

```
[zi] ✓ 旧模块里没有 idle_rollout_values / cand_mode（确认它是改动前那一份） md5=5B92A4865B849BFCE16BCE65938E1424
[zi] pre=FlatRolloutGreedy(cand_mode=<无此属性>)  live=FlatRolloutGreedy(cand_mode=pin)
[zi] BASE 配置 30 项（从 _inject_vs_runnerup.sh 读）；K=48 rounds=49152 rollouts=64 mode=random AZAI_CAND_MODE=<未设>
[zi] r20   p0 | 候选=1   rollouts=64 回合=1024  | chosen=()            scores_sha=18d02e3ac2610752 policy_sha=6c3c396ed6b5c36d bundles_sha=b18a48f02566e615 | ✓ 逐字段相同
[zi] r20   p1 | 候选=1   rollouts=64 回合=1024  | chosen=()            scores_sha=9fe7ceb74e0e141f policy_sha=6c3c396ed6b5c36d bundles_sha=b18a48f02566e615 | ✓ 逐字段相同
[zi] r60   p0 | 候选=1   rollouts=64 回合=1024  | chosen=()            scores_sha=86fa156f959d9b02 policy_sha=6c3c396ed6b5c36d bundles_sha=b18a48f02566e615 | ✓ 逐字段相同
[zi] r60   p1 | 候选=1   rollouts=64 回合=1024  | chosen=()            scores_sha=732fad331655feb9 policy_sha=6c3c396ed6b5c36d bundles_sha=b18a48f02566e615 | ✓ 逐字段相同
[zi] r120  p0 | 候选=1   rollouts=64 回合=1024  | chosen=()            scores_sha=ee5dba23617913ce policy_sha=6c3c396ed6b5c36d bundles_sha=b18a48f02566e615 | ✓ 逐字段相同
[zi] r120  p1 | 候选=48  rollouts=64 回合=49152 | chosen=((21,2,6),)   scores_sha=0e7bcc9dc92e09db policy_sha=7ae78e7dd356af5e bundles_sha=f74bc3e0cd7d7d01 | ✓ 逐字段相同
[zi] r200  p0 | 候选=1   rollouts=64 回合=1024  | chosen=()            scores_sha=3e8fdbf6038120f8 policy_sha=6c3c396ed6b5c36d bundles_sha=b18a48f02566e615 | ✓ 逐字段相同
[zi] r200  p1 | 候选=48  rollouts=64 回合=49152 | chosen=((21,2,6),)   scores_sha=eb4a56f88590a482 policy_sha=f3280353575f0599 bundles_sha=68453406472c92ab | ✓ 逐字段相同
[zi] 小结：8 个真实局面（pre vs live）不一致 = 0（必须 0）
```

判据：`last_scores`（float64 **原始字节** sha256）/ `visit_policy` / `bundles` / `chosen_bundle` /
`root_value` / `intent_counts` / `last_rollouts` / `last_rounds` **全等**。
其中 2 个局面是 **n_cand=48、走满 49152 回合预算**（= 最贵的那条路径），另 6 个是 n_cand=1。

> 最终代码（把 `cand_mode=` 的日志追加也改成"只 unified 才追加"）**又跑了一遍**：
> `match_results/unified_cand/zero_impact_probe2.log`（4 个局面）⇒ 不一致 = 0，退出码 0。
> （同一个局面在两次探针里 `root_value` 不同是**正常的**：`_decision_idx` 随 search 调用次数递增、
> 阶段 2 的随机种子由它派生；pre/live 两侧**调用序列相同**，所以比较依然严格对齐。）

### 3.2 对局级：同 seed self-play，`RESULT` + 双方 op 序列逐字节

脚本 `code/test_match/_cand_mode_zero_impact_game.sh`；日志
`match_results/unified_cand/zero_impact_{before,after}_seed8.log`。
`before` = `code/test_match/_az_bridge_ai_preunifiedcand.py`（加代码那一刻的 `az_bridge_ai.py`
原样副本，**只把** `build_engine` 里 `from ...flat_rollout_greedy import FlatRolloutGreedy`
换成"加载上面那份副本"），`after` = 现行 `az_bridge_ai.py`（**不设** `AZAI_CAND_MODE`）；
两边各自 self-play（ai0=ai1=同一份代码）⇒ 代码行为相同则两局必然逐字节相同。
配置 = `INJECT_VS_RU_B` 的 BASE **逐字**（`AZAI_SEARCH=flat`、`AZAI_K=48`、`AZAI_FLAT_ROUNDS=49152`、
`FLAT_SEED=53`、`posnet_A_k5_m32.pt`），seed=8，`--rounds=120`，`--trace-ops`。

```
[zig]   before pid=963 rc=0  10:16:55
[zig]   after  pid=964 rc=0  10:16:55
[cmp] before: RESULT seed=8 p0=<AI> p1=<AI> winner=<AI> winner_side=p1 verdict=hp_tiebreak engine_winner=None rounds=120 terminal=False base_hp=36,43 coins=46,62 illegal=0
[cmp] after : RESULT seed=8 p0=<AI> p1=<AI> winner=<AI> winner_side=p1 verdict=hp_tiebreak engine_winner=None rounds=120 terminal=False base_hp=36,43 coins=46,62 illegal=0
[cmp] before 自检: illegal-summary seed=8 cold=False illegal_total=0 checked_ops=159 holds=p0:75,p1:67 by_side={0:0,1:0} by_kind={} mismatch=0 shadow_ms=<t>
[cmp] after  自检: illegal-summary seed=8 cold=False illegal_total=0 checked_ops=159 holds=p0:75,p1:67 by_side={0:0,1:0} by_kind={} mismatch=0 shadow_ms=<t>
[cmp] ✅ 所有行为字段逐字段相同（winner_side/verdict/rounds/terminal/base_hp/coins/illegal + 自检行）
[zig] before ops 行数=240  after ops 行数=240
[zig] ✅ op 序列**逐字节相同**（sha256 e336bd304bbbde2a）
[zig] ✅ RESULT 行（AI 标签归一化后）**逐字节相同**
[zig] 总退出码=0（0 = 对局级零影响成立）
```

> `--rounds=120` 是**为了不压满 CPU**（另一个批次在用 16 个进程）而截的；两臂都被同一上限截断，
> 逐字节比较的有效性不受影响。`verdict=hp_tiebreak` 也是截断的后果（引擎没判胜负），不是行为差异。


## 4. 冒烟读数

### 4.1 1 局冒烟（对手 = 亚军 cpp_lure_v4，seed=8，**跑满到引擎终局**）

命令：`ROUNDS=512 TAG=UNIFIED_CAND_FULL bash code/test_match/_unified_cand_smoke.sh`
（配置 = `INJECT_VS_RU_B` 的 BASE **逐字** + `AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1`；
`AZAI_K=48`、`FLAT_ROLLOUTS=64`、`FLAT_ROUNDS=49152`、`FLAT_SEED=53`、`--jobs 1`（本就是一局））
日志：`match_results/unified_cand/UNIFIED_CAND_FULL_seed8.log`、
AI stderr `match_results/ai0_az_bridge_ai_seed8_UC.stderr.log`、
统计 `match_results/unified_cand/stat_full.txt`（脚本 `code/test_match/analyze_unified_cand_smoke.py`）。

**对局结果（要求的四项判定）**

```
RESULT seed=8 p0=az_bridge_ai p1=runnerup winner=runnerup winner_side=p1
       verdict=engine engine_winner=1 rounds=346 terminal=True
       base_hp=0,32 coins=15,380 illegal=0
illegal-summary seed=8 cold=False illegal_total=0 checked_ops=264 holds=p0:273,p1:240
       by_side={0:0,1:0} by_kind={} mismatch=0
```

* **`verdict=engine` ✓**（打到 346 回合我方基地血量归零，引擎自己判的终局）
* **`illegal=0` ✓**（264 次自检 op，0 非法，`mismatch=0`）
* 用时 34 分钟（09:56:15 → 10:30:32），346 个决策点
* 我方实际发出的 op：`BUILD_TOWER 38 / DOWNGRADE_TOWER 36 / UPGRADE_TOWER 6 /
  USE_DEFLECTOR 1 / USE_LIGHTNING_STORM 1`（对照 pin 口径：升级**从来没有**进过候选，
  所以历史上一直是 0 次）

**每决策 `n_cand`（去重后）分布**（346 个决策）

| 统计量 | 值 |
|---|---|
| min / p25 / 中位 / p75 / p95 / max | **2 / 3 / 4 / 4 / 48 / 48** |
| 均值 | 9.0 |
| `n_cand==1` 占比 | **0.0%**（现役 pin 口径：**82.3%**） |
| `n_cand>=24` 占比 | 12.1% |
| `n_cand==48`（跑满 k）占比 | 11.6%（40 次） |

直方图：`n_cand=2 → 24 次(6.9%)`、`3 → 90 次(26.0%)`、`4 → 188 次(54.3%)`、
`17/19/43/44 各 1 次`、`48 → 40 次(11.6%)`。

> ⚠️ **中位数只有 4，不是 48**：不是生成器不够，而是**这个局面的可达候选空间就这么大** ——
> 我方金币长期在 13–56（日志 `round=N coins=[..]`），而建塔/升级/超武/闪电的价钱都高于它
> ⇒ 买得起的只剩"降级（退钱）+ HOLD"，再加少数金币够格的回合。
> 生成器的早停在 **335/346** 个决策上触发（= 可达空间已枚举完）。
> 48 条只出现在金币充裕的 11.6% 回合。

**候选构成**

| 指标 | 值 |
|---|---|
| 候选**含 HOLD** 的回合占比 | **99.1%**（343/346；3 个决策里 HOLD 进了池但被 top-k 按阶段 1 分挤掉） |
| 候选**含升级(op 12)** 的回合占比 | **3.8%**（13/346；每决策含升级条数 min 0 / 中位 0 / max 12，合计 69 条） |
| 长度分布（全部候选合计） | 0 op 11.0% / 1 op 60.8% / 2 op 27.3% / 3 op 0.9% |
| **升级被选中** | **6 次（1.7% 的决策）**，且**真的发出去了 6 次 `UPGRADE_TOWER`** |
| HOLD 被选中 | 273 次（78.9%） |
| 其它 | 67 次（19.4%） |

> 升级候选只有 13 个回合出现、且只有 6 次被选中 —— 原因同上：**升级要 60–120 金币**，
> 而这些局面金币常年 13–56。⇒ "候选里有没有升级"已经不是瓶颈了，**金币/经济的瓶颈**才是。

**决策耗时（ms）**

| 指标 | 值 |
|---|---|
| 决策总耗时 p50 | **1 519** |
| p95 | **23 761** |
| max | **41 694** |
| mean | 3 990 |
| 阶段 1（候选生成 + 确定性 idle rollout）p50 / p95 / max | 29.0 / 637.8 / 1 402.3 |
| 阶段 1 占整决策 | **2.4%** |

> 耗时读数是在**机器被另一个 16 进程批次占满**时测的（冠军批次全程在跑）⇒ 是**上界**；
> 48 候选 × 64 采样 × 16 回合 = 49 152 回合/决策 与设计 §3 的"48 ≈ 26–31 s/决策"同量级。
> p50 只有 1.5 s 是因为一半以上的决策 n_cand ≤ 4（预算公式给每候选更多采样，但总回合仍受
> `AZAI_FLAT_ROUNDS` 封顶）。

样例日志（`AZAI_CAND_VERBOSE=1` 时每决策一条，stderr）：

```
[cand] mode=unified n_cand=48 含HOLD=1 含升级=4 池=69 重复=232 长度分布={1: 25, 2: 22, 0: 1} p1_ms=785.7 决策_ms=23684.9 选中=升级
[cand] gen 池=72 取=48 重复=98 HOLD=1(兜底1) 升级=0 长度分布={2: 24, 1: 23, 0: 1} 513 ms
[cand] mode=unified n_cand=3 含HOLD=1 含升级=0 池=3 重复=96 长度分布={0: 1, 1: 2} p1_ms=22.1 决策_ms=1033.0 选中=HOLD
[cand] 早停：连续 96 次无新候选（可达空间已枚举完）
```

### 4.2 离线探针：6 个真实局面（**独立于对局进程**，便于复现）

`code/test_match/smoke_unified_cand.py`（逐条候选 **重新 `can_apply`** 自检）：
日志 `match_results/unified_cand/smoke_probe_final.log` / `.json`。

```
[uc] r20   p0 n_cand=48 池=63 重复=250 长度={1:25, 2:22, 0:1} 含HOLD=1 含升级=4 建塔=43 降级=22 | 决策=13296ms 选中=[(12, 1, 4)]
[uc] r20   p1 n_cand=2  池=2  重复=96  长度={1:1, 0:1}        含HOLD=1 含升级=0 建塔=0  降级=1  | 决策=569ms   选中=[(13, 2, -1)]
[uc] r200  p0 n_cand=4  池=4  重复=96  长度={1:3, 0:1}        含HOLD=1 含升级=0 建塔=0  降级=3  | 决策=1579ms  选中=[]
[uc] r200  p1 n_cand=48 池=72 重复=121 长度={1:24,2:9,3:15}   含HOLD=0 含升级=6 建塔=39 降级=23 闪电=14 超武=5 | 决策=16793ms 选中=[(11,13,7),(13,32,-1),(21,6,8)]
[uc] r300  p0 n_cand=48 池=57 重复=219 长度={1:29, 2:18, 0:1} 含HOLD=1 含升级=0 建塔=47 降级=18 | 决策=17707ms 选中=[(11, 7, 10)]
[uc] r300  p1 n_cand=48 池=72 重复=17  长度={1:5,2:21,3:22}   含HOLD=0 含升级=5 建塔=60 降级=7 闪电=12 超武=13 基地=16 | 决策=17038ms 选中=[(11,12,9),(21,3,7)]
[uc] 合计：6 个决策，非法候选/选中 = 0（必须 0）
```

要点：
* **升级（12）** 在金币够的回合稳定出现（4–6 条/决策），并且**两次被选中**（`(12,1,4)`）；
* **闪电（21）/超武（23,24）/基地升级（31,32）** 都在金币够的回合出现（r300 p1 甚至有 16 条基地升级），
  r200 p1 选中的就是 `(11,13,7)+(13,32,-1)+(21,6,8)` = 建塔 + 降级 + 闪电 的 3-op 束；
* **合法性**：6 个决策、每条候选都重新 dry-run 校验，**0 非法**（含 3-op 束）。


## 5. 已知未覆盖 / 与设计的偏差

### 5.1 偏离 / 取舍（逐条）

1. **位置游标"按类"而不是"按多重集"**（设计内部矛盾，必须选一个）。
   设计单动作版说"每类一个 `cursor[c]`"，多动作组版说"游标按**规范多重集**索引"。两者对多 op 束**冲突**：
   若按多重集重启，每个新键都从 argmax 开始，而**塔通道在空格上解码结果与类无关**（都解成同一个
   `BUILD_TOWER(x,y)`）⇒ 大量键解出同一个 bundle ⇒ 去重后候选数塌到个位数（实测 7 条 / 426 次重复）。
   ⇒ 取**按类**跟踪"该类最后那个位置"（= 单动作版的规则），多重集 `cursor[key]` 仍保留作提议记录。
2. **两阶段复用带"估计器一致"门控**。设计说"阶段 1 的结果直接喂给阶段 2，只补足采样数"，
   但默认阶段 1 = `idle`（确定性）、阶段 2 = `random`（随机化）⇒ **把 idle 的分混进 random 的样本均值是错的**。
   ⇒ 只有签名一致（`AZAI_FLAT_ROLLOUT_MODE=idle` 且没开 `AZAI_FLAT_RNG_PERTURB`）才复用；
   默认配置下不复用（打日志说明）。**已验证复用路径真的生效**：idle 配置下 `复用样本数=8/8`、
   `last_rounds` 从 8192 → 8064（正好少 1 个样本 × 16 回合）。
3. **HOLD 恒进"候选池"、但不保证进 top-k**（`AZAI_CAND_FORCE_HOLD=1`）。设计 §6 第 3 问
   "恒含 1 条还是按分数决定"未拍板，取保守选项（与"补 HOLD ⇒ +29.2pp"的证据口径一致）：
   空束**一定**进池并拿到自己的阶段 1 分，但**最终 top-k 由分数决定** ⇒ 实测 343/346 = **99.1%**
   的回合候选里含 HOLD（3 个回合金币充裕、48 条高分候选把它挤掉了）。
   注意：HOLD **仍然**是 bandit 的一个臂（`Q[23]/N[23]` 在跑），所以"含 HOLD 占比"这个指标
   在 unified 下判别力很低（恒 ~100%）；要看"空束被**选中**的比例"（本局 78.9%）才有信息。
4. **一个 bundle 内类不重复**（`{类}` 是集合）。设计说"规范多重集"，但同一类出现两次（例：两个
   不同通道都建塔）在位置槽位上歧义 ⇒ 取集合语义（这样"规范多重集"退化成"规范集合"，
   与设计给的归一化例子 `(hold,build,lightning) → (build,lightning)` 一致）。
5. **k=48 时成本与设计一致地高**：实测决策 p50 ≈ **1.52 s** / p95 ≈ **23.8 s** / max ≈ **41.7 s**
   （48 候选 × 64 采样 × 16 回合 = 49 152 回合/决策，与设计 §3 的 26–31 s 同量级；本机当时还有
   另一个 16 进程批次在跑 ⇒ 读数是**上界**）。⇒ 阶段 2 的 24 局 A/B 必须重新定 k 或预算。
6. **"48 条候选"只覆盖 11.6% 的决策**（中位 4 条）：不是生成器的锅，是**金币买不起**（见 §4.1）。
   ⇒ 若目标是"让评估器有得选"，卡点在**经济/金币**，不在候选枚举宽度；这一点对阶段 2 的读数
   解读很关键（不然会误判成"统一候选没用"）。
7. **不变的另一半**：本局评测中 `选中=HOLD` 仍占 **78.9%**（与现役 pin 口径的 78% 几乎一样）。
   ⇒ 候选生成修好**不等于**决策会变 —— 阶段 2 若 A/B 无提升，第一嫌疑应指向**评估器**
   （`random` rollout + 值网对"花钱"的系统性惩罚），而不是候选生成。

### 5.2 已知未覆盖

* **没跑 A/B**（按任务要求：24 局 A/B 等冠军批次跑完再说）。
* **位置网 / 类网先验没接**（设计明说类网现在不可用、需重训；接口留了但没实现回调）。
* **`AZAI_VISIT_RANK` / `AZAI_SEED_STATE` 等采集侧开关没在 unified 下验证**（不在本阶段范围）。
* **unified 与 `AZAI_SEARCH=mcts|unified` 不兼容**（会 warning 后回退 pin）；`BundleMCTS` 路径没有统一候选。
* **跨决策不保留 Q/cursor**（每次 `search` 重置）。设计没规定，但"跨决策累积"可能更省预算 ⇒ 留作后续。
* **候选空间受金币限制**：早期/低金币局面可达候选只有 2–4 条（真实限制，不是 bug；见 §4.1）。
* **本次只跑了 1 局**（`--jobs 1`，按任务要求不跑 A/B）：1 局的胜负/op 计数**不能**当强度读数。

## 6. 交给阶段 2 的三条判断

1. **候选项已经不再是瓶颈**：`n_cand==1` 从 82.3% → **0%**，升级/闪电/超武/基地升级都能进候选
   （金币够的回合），合法性 0 非法。⇒ 设计 §5 阶段 2 的 A/B 可以放心跑，"候选塌缩"这个假设已被修掉。
2. **但"选中 HOLD 78.9%"没变** ⇒ 若 24 局 A/B 无提升，**下一步该查评估器**（`random` rollout 的
   `decay(1.0,3.0)` + 值网对"花钱"的惩罚），不是继续加候选。这也是设计 §2「关键设计判断」警告的
   "弱评估器主导"的镜像：**强评估器也可能系统性偏好 HOLD**。
3. **成本要提前定**：k=48 + N=64 时 p95 ≈ 24 s/决策、单局 34 分钟；24 局 A/B 在这个 k 下要
   约 14 核·小时（本机还被别的批次占着）⇒ 建议 A/B 先把 `AZAI_FLAT_ROUNDS` 或 k 降一档，
   或者按设计 §3 固定 k=48 但只跑 12 局。


