# 方案 B 诊断臂：把「升级」作为额外候选送进 flat rollout 评估器（2026-10-28）

> 工作目录 `D:\2026智能体大赛_新2_distill`。上游只读诊断：`docs/why_no_upgrade_20261028.md`（commit `f9afd27`）。
> 本文 = **预注册**（§0-§4，跑之前写）+ **实测读数**（§5 起，跑完补）。
> 一句话变更：新增 `AZAI_UPG_BUNDLES=N`（**默认 0**），把最多 N 条"合法且买得起"的升级候选走
> **已有的**额外候选通道塞进 48 束的 flat rollout 评估器，让它们**同台比较**。

---

## 0. 结论（三句话）

1. **评估器认升级**：C 臂（`AZAI_UPG_BUNDLES=4`）塞进去 **3961** 条升级候选，被 flat rollout **选中 526 条
   = 13.3%/条**；按回合看，**有升级候选的 995 个决策点里 526 个（52.9%）真的走了升级** ⇒
   上游"瓶颈在候选生成、不在评估器"的结论在**真实对局**里得到确认（离线探针 17% 同量级）。
   副作用：我方 `UPGRADE_TOWER` 从 **0（0.0%）→ 526（22.8% 的 op）**，塔等级从 **100% 一级** →
   **58.8 / 39.8 / 1.5%**（一/二/三级）。
2. **但棋力没变**：C 臂 vs 亚军本体 **2W-14L = 12.5%**，血量差 **−20.25**；对照 B 臂
   （不注入、不加升级）**2W-14L = 12.5%**、**−20.12** ⇒ 配对 ΔHP = **−0.12**（95%CI −12.22…+11.97）、
   McNemar **p=1.0**、胜率差 **+0.0pp**。和 A 臂（注入亚军那一手）50.0% / +0.56 比，
   **C 只拿到 A−B 那 +20.69 血量差里的 −0.6%（≈0）**。
3. **所以"升级进候选"值不值得转正：不值得**（至少在这套对手/配置/16 局分辨率下）。它把
   "升级到底有没有用"从**没试过**变成**试过了、没有可测收益**——而且是在**正确的机制上**
   （进候选、同一个评估器裁决，不是 `AZAI_MC_UPG` 那种绕过评估器直接接管）。默认 0 继续保持。

---

## 1. 实现（`code/test_match/az_bridge_ai.py`，开关式，默认 0 = 逐位不变）

### 1.1 插入点（不改搜索算法）

| 位置 | 内容 |
|---|---|
| `main()` 内 `_oracle_init()` 之后 | `_UPG` 状态字典（**模块级**，为了 `_report` 也能读）+ `_upg_enable()` / `_upg_bundles()` / `_upg_step()` / `_upg_after_search()` / `_upg_count_sent()` |
| `decide()` 里 `mcts.search(...)` **之前**（`:1597-1604`） | `_upg_step()`：枚举 → 逐条 `can_apply` → 逐条 `_block_legal_gate` → `mcts.set_extra_bundles(注入臂的 + 我的)` |
| `decide()` 里 `_oracle_after_search()` 之后 | `_upg_after_search(res)`：判定"被评估器选中" + 记 `(n_cand, n_rollouts)` |
| `_decide_timed()` 里 `_oracle_log_round(_ops)` 之后 | `_upg_count_sent(_ops)`：最终发出去的 op 里数 `op=12` + 每 50 决策打一次汇总 |
| `_report()` | 收尾一行 `[upg] 统计: …` |

`AZAI_UPG_BUNDLES` 也写进了模块 docstring 的环境变量表。**没有**动 `code/my_ai/az_intent/**`
（通道 `set_extra_bundles` / `_collect_extras` 是现成的，只调不改）。

### 1.2 候选怎么挑（**确定性、无 RNG、可复现**）

1. `facade.towers_of(player)` 按 `tower_id` 升序（不依赖列表顺序）；
2. 每座塔按 `TOWER_UPGRADE_TREE[tower_type]` 的目标顺序（**塔外层、目标内层**，与
   `bundle_mcts.py:784-820` 的 `for _tw: for _tgt:` 同构）；
3. 逐条 `facade.can_apply_operation(player, Operation(12, tower_id, target))`（**只读** dry-run）——
   与 `_oracle_bundle:1022` **同一个检查器**（`game_state_facade.py:176-183` → 引擎 `can_apply_dry`，
   复刻冷路径的金币/塔位/营地上限）。上游 §4.3 的"方法 B"就是它，且与"买得起"口径**同值**
   （22368 = 22368）⇒ **合法即买得起**，不需要另写金币闸门；
4. 便宜的预筛：金币 < 60（最便宜升级价）⇒ 直接返回空（省掉 towers×targets 次 dry-run）；
5. 取前 N 条；异常一律 **fail-closed**（当非法丢掉）。

### 1.3 合法性：双保险 + 一个**真 bug**（冒烟就抓到）

* 单 op 候选 ⇒ `pending` 恒空，**不存在"半块非法"**（那是 `AZAI_NO_DOWNGRADE` 多 op 块的问题）；
* 仍然把每条**单独**过一遍 `_block_legal_gate`（`:327-356`，`MVS.check_ops` 独立校验器、fail-closed）。
  被拦下 ⇒ 只丢那一条（`gate_drop` 计数），绝不发半块。
* ⚠️ **踩过的坑（值得记）**：第一版把 4 条候选**当成一整块**交给 `_block_legal_gate`，结果 4 条全被
  `tower_already_used_in_round(id=1)` 拦掉——因为闸门的 `pending=` 语义是"同一回合**先后执行**"，
  而这 4 条对评估器是**互斥的备选**（同回合只会发一条）。冒烟日志里 9 个回合全是这个假阳性。
  改成**逐条调用**后 `gate_drop=0`（16 局累计 0，见 §7）。

### 1.4 与注入通道共存（设计文档 §8.3 的"合并而不是覆盖"）

`set_extra_bundles` 是**跨回合持久**的字段（`search()` 读 `self.extra_bundles`），所以每回合：
**先摘掉自己上一回合写进去的那份** → `set_extra_bundles(注入臂那份 + 我的)`；本回合没有候选
⇒ 只撤自己的，注入臂的候选原样保留。⚠️ 角落（未触发）：若某回合我没有候选、且注入臂那一手
**恰好等于**我上一回合塞过的升级 bundle，它会被一并摘掉（白丢一个候选，不非法、不崩）。

### 1.5 日志（**一律 stderr**，绝不写 stdout）

```
[upg] 升级候选已启用: 每回合最多 4 条（seed=… player=… K=48 search=FlatRolloutGreedy；通道=set_extra_bundles…）
[upg] round=12 加入 4 条升级候选 targets=[(12,3,1), (12,3,2), (12,3,7), (12,3,4)]
[upg] round=13 加入 0 条升级候选 targets=[]                       # 没有合法且买得起的升级
[upg] sel round=12 hit=1 mine=4 extra=4 n_cand=5 n_rollouts=64 chosen=(12,3,4)
[upg] 统计: n_max=4 decisions=500 rounds_with_cand=115(23.0% of decisions) offered=460 sel=56(12.2% …) …
```

* `加入 k 条` 每决策**一条** ⇒ 分母（决策点数）与"有候选的决策点占比"直接可算；
* `sel … hit=1` = 最终 `chosen_bundle` **就是**我塞的某一条；
* `n_cand`/`n_rollouts` 取 `mcts.last_scores`/`last_rollouts`（flat 自己算的，不猜）；
* 解析器：`code/test_match/analyze_upg_bundles.py`。

### 1.6 默认关（N=0）的零影响

三个钩子函数**第一行**都是 `if _UPG["n"] <= 0: return` ⇒ 不枚举、不 dry-run、不调
`set_extra_bundles`、不打日志、不动 RNG。对局级证据见 §2。

---

## 2. 零影响证据（N=0 vs 改动前，同 seed 逐字段相同）

* 改动前 = `code/test_match/_az_bridge_ai_preupg.py`（**加代码那一刻**的工作区原样副本；
  md5 **`8841F843C058D68564077ABC1978EC8E`**，99860 字节。⚠️ 不用 `git show HEAD:` —— HEAD 里
  已经有本次改动，等于自欺欺人）；
* 改动后 = `code/test_match/az_bridge_ai.py`（115114 字节），**不设** `AZAI_UPG_BUNDLES`；
* 配置 = `_inject_vs_runnerup.sh` 里的 BASE **逐字**（含 `K=48 / FLAT_ROUNDS=49152 / FLAT_SEED=53`），
  `--rounds=200` 截断，`seeds=7 7r`，两版**并发**跑（4 局 × 2 个 AI = 8 进程）；
* 比较器 `code/test_match/compare_result_lines.py`（标签归一化 + 剔掉墙钟 `shadow_ms`）。

```
seed=7  before: RESULT seed=7 p0=<AI> p1=<AI> winner=<AI> winner_side=p1 verdict=hp_tiebreak
                rounds=200 terminal=False base_hp=33,36 coins=24,71 illegal=0
        after : RESULT seed=7 … 逐字段相同
        自检行: illegal-summary … illegal_total=0 checked_ops=210 holds=p0:123,p1:125 by_side={0:0,1:0}
                by_kind={} mismatch=0   ← 两边**逐字相同**
seed=7r 同上，逐字段相同；总退出码 0
```

* 脚本：`bash code/test_match/_upg_bundles_zero_impact.sh`（默认行为逐字段不变时返回 0）；
* 旁证：这 4 局的 AI stderr 里 `[upg]` 行数 = **0**（开关关 ⇒ 一条日志都不打）；
* 另一侧：冒烟里 **ai1（开关关）的 `[upg]` 行数 = 0**，ai0（开关开）正常（§5）。

---

## 3. 诊断臂设计与预算

### 3.1 两臂

| 臂 | 配置 | 来源 |
|---|---|---|
| **C（诊断臂）** | 现役 BASE + `AZAI_UPG_BUNDLES=4` | 本次新跑 16 局（`INJECT_VS_RU_C`） |
| **D（今次对照）** | 现役 BASE（不加升级、不注入） | **复用** `INJECT_VS_RU_B`（同 BASE、同 seeds 7..14×2 座位、同 harness），有效性由 §2 的零影响证明兜住 |

* BASE = `_inject_vs_runnerup.sh:45` 那串（`K=48 / FLAT_ROLLOUTS=64 / FLAT_ROUNDS=49152 /
  random+decay(1.0,3.0) / horizon16 / share_prefix=1 / FLAT_SEED=53`），**逐字未改**
  （脚本自带断言：BASE 与 `_oracle_inject_ab.sh` 逐字相同，600 字符）；
* C 臂 env **没有** `AZAI_MC_UPG` / `AZAI_ORACLE_INJECT` / `AZAI_ORACLE_PLACEBO`
  （脚本里加了三条前置断言，任一不满足就 `exit 4`）；
* 驱动：`bash code/run_logged.sh upg_bundles_ab bash code/test_match/_inject_vs_runnerup.sh`，
  `ARMS=C UPG_N=4 JOBS=8 RETRY=2`。任务书里的 `--resume --retry-invalid 2 --game-timeout-s 7200`
  在这个 harness 里是**内建**的：`--resume` = 脚本的 `remaining_tokens()`（跳过已有
  `verdict=engine` 的 token）、`--retry-invalid 2` = `RETRY=2`、`--game-timeout-s 7200` =
  `_tmp_ladder.py:33 TIMEOUT=7200`；
* 实测：16 局、8 并行、**59 分钟**跑完，第 1 轮就全部 `verdict=engine`（**没有触发重试**）。

### 3.2 预算：候选 48 → 52，`N_eff` **59（−7.8%）**

`flat_rollout_greedy.py:520-523`：
`per_candidate_rounds = max_total_rounds // (n_cand × horizon)`，`n_rollouts = min(64, ↑)`

| n_cand | `49152/(n_cand×16)` | `n_rollouts` | 相对 48 |
|---|---|---|---|
| **48**（B/D 臂） | 64.0 | **64** | — |
| 49 | 62.7 | 62 | −3.1% |
| 50 | 61.4 | 61 | −4.7% |
| 51 | 60.2 | 60 | −6.3% |
| **52**（+4，C 臂上界） | 59.1 | **59** | **−7.8%** |

⇒ **C 臂在采样上略吃亏**：最坏情况（每回合真的塞满 4 条）每条候选的 rollout 数 64 → 59。
**但 §8 的实测说明这几乎不发生**（只有 **1.74%** 的决策 `n_rollouts` 被压到 64 以下）。

---

## 4. 预注册判据（跑之前写死）

| 读数 | 结论 |
|---|---|
| 选中率 ≈ 0 | 评估器**不认**升级 ⇒ 瓶颈在评估器 ⇒ 不转正 |
| 选中率 ≫ 0 而胜率不涨 | 评估器认、但对赢没帮助 ⇒ 与三次 `MC_UPG` A/B（23.4/18.8/31.2%）同结论 ⇒ **不转正** |
| 选中率 ≫ 0 且胜率/血量显著更好 | 值得转正并单独复跑 |

实际落在**第 2 行**（选中率 13.3%/条，胜率与 B 完全同分）。

---

## 5. 指标 1（核心）：升级候选的选中率 = **13.3%/条**

源：16 局的 AI stderr `[upg]` 行（`analyze_upg_bundles.py --stderr <C 臂 16 局 stderr>`；
9/16 局的逐决策累计与收尾 `统计` 行**完全对上**，其余只到周期快照 ⇒ 表里标 `--`）。

| file（我方 16 局） | 决策 | 有候选 | 占比 | 塞入 | 选中 | 选中/条 | 选中/回合 | ⚠️ |
|---|---|---|---|---|---|---|---|---|
| ai0_seed7 | 387 | 96 | 24.8% | 384 | 29 | 7.6% | 30.2% | 0 |
| ai0_seed8 | 505 | 115 | 22.8% | 460 | 56 | 12.2% | 48.7% | 0 |
| ai0_seed9 | 392 | 86 | 21.9% | 344 | 32 | 9.3% | 37.2% | 0 |
| ai0_seed10 | 414 | 44 | 10.6% | 175 | 29 | 16.6% | 65.9% | 0 |
| ai0_seed11 | 465 | 38 | 8.2% | 147 | 29 | 19.7% | 76.3% | 0 |
| ai0_seed12 | 454 | 74 | 16.3% | 296 | 34 | 11.5% | 45.9% | 0 |
| ai0_seed13 | 410 | 95 | 23.2% | 380 | 31 | 8.2% | 32.6% | 0 |
| ai0_seed14 | 459 | 63 | 13.7% | 252 | 38 | 15.1% | 60.3% | 0 |
| ai1_seed7 | 442 | 53 | 12.0% | 208 | 31 | 14.9% | 58.5% | 0 |
| ai1_seed8 | 456 | 55 | 12.1% | 220 | 38 | 17.3% | 69.1% | 0 |
| ai1_seed9 | 512 | 42 | 8.2% | 159 | 35 | 22.0% | 83.3% | 0 |
| ai1_seed10 | 498 | 77 | 15.5% | 308 | 42 | 13.6% | 54.5% | 0 |
| ai1_seed11 | 387 | 51 | 13.2% | 204 | 30 | 14.7% | 58.8% | 0 |
| ai1_seed12 | 419 | 49 | 11.7% | 196 | 29 | 14.8% | 59.2% | 0 |
| ai1_seed13 | 315 | 11 | 3.5% | 44 | 8 | 18.2% | 72.7% | 0 |
| ai1_seed14 | 423 | 46 | 10.9% | 184 | 35 | 19.0% | 76.1% | 0 |
| **合计** | **6938** | **995** | **14.3%** | **3961** | **526** | **13.3%** | **52.9%** | **0** |

* **指标 1（任务书口径）= 13.3%/条**（526/3961）；等价读法 = **有候选的回合里 52.9% 走了升级**；
* **"有升级候选的决策点占比" = 995/6938 = 14.3%**（= 上游 §4.3 那个"合法且买得起"的机会率
  在本臂的版本；B 臂是 42.3%，差距来自**我们变穷了**，见 §6 金币）；
* 每回合塞入条数：`{4 条: 976 回合, 3 条: 19}` ⇒ 基本总是塞满；
* **评估器偏好哪个升级**（每目标：被选中 / 被塞入）：
  **PRODUCER(4) 261/976 = 26.7%** > **MORTAR(3) 107/976 = 11.0%** > HEAVY(1) 73/976 = 7.5%
  ≈ QUICK(2) 71/976 = 7.3%（三级目标各 15 次被塞入：31 → 1、32 → 5、33 → 5）
  ⇒ **它偏好经济塔**，而不是 `bundle_mcts` 默认会挑的那个 HEAVY（"统一升成最差的那个"的历史坑）；
* `[upg] sel` 与**实际发出去**的 `op=12` 数**完全相等**（526 = §6 的 526）⇒ 被选中的升级一条都没被
  执行层过滤掉（`illegal=0` 的另一面）。
* （口径说明：`analyze_inject_vs_runnerup.py` 的耗时口径数到 **6737** 个决策，比这里的 6938 少 201 条，
  = 每局末尾没来得及 flush 的那几条日志，不影响任何比率的分母口径。）

---

## 6. 指标 2：op 类型分布 / 塔等级 / 金币（C vs 已知基线 B）

`probe_why_no_upgrade.py --tag INJECT_VS_RU_C`（与 B 臂**完全同口径**，纯日志重算）：

| op | **C 我方** | B 我方（基线） | C 对手（亚军） | B 对手 |
|---|---|---|---|---|
| `BUILD_TOWER(11)` | 604（**26.2%**） | 1544（48.6%） | 995（42.1%） | 1119（41.5%） |
| **`UPGRADE_TOWER(12)`** | **526（22.8%）** | **0（0.0%）** | 199（8.4%） | 252（9.3%） |
| `DOWNGRADE_TOWER(13)` | 1062（46.0%） | 1461（46.0%） | 1091（46.1%） | 1250（46.3%） |
| `LIGHTNING(21)` | 117（5.1%） | 170（5.4%） | 76（3.2%） | 61（2.3%） |
| 合计 op | **2309** | 3175 | 2366 | 2697 |
| op / 有出手的决策 | **1.20** | 1.64 | — | — |

| 我方塔·等级（塔·回合计数） | **C** | B（基线） |
|---|---|---|
| 一级（BASIC） | **4871（58.8%）** | 12468（**100.0%**） |
| 二级 | **3298（39.8%）** | 0（0.0%） |
| 三级 | **122（1.5%）** | 0（0.0%） |
| 塔·回合总数 | 8291 | 12468 |
| 有 ≥1 座二级塔的回合 | **45.9%** | 0.0% |
| 一座塔都没有的回合 | 8.5% | 14.6% |

| 金币（决策时刻，p50） | **C 50**（p10 13 / p90 126 / mean 59.4） | B 64（p10 21 / p90 143 / mean 75.6） |
|---|---|---|
| `P(金币≥60)` | 39.7% | 54.2% |
| `P(金币≥200)` | 0.6% | 3.8% |
| 机会转化率（发出升级/买得起的升级组合） | **11.2%**（526/4709） | 0.0%（0/22368） |

三点读数：

1. **就是目标该做的事**：`UPGRADE_TOWER` 从 **0.0% → 22.8%**、塔等级从 **100% 一级 → 58.8/39.8/1.5%**，
   而且**真的升到了三级**（122 个塔·回合、1.5%）——上游"我们 100% 一级"的结构性问题被打开。
2. **代价是建塔**：`BUILD_TOWER` 48.6% → 26.2%，总 op 3175 → 2309，金币 p50 64 → 50。
   即**升级挤掉了建塔**（60/200 的钱换成塔等级），塔·回合总数 12468 → 8291（−33.5%）。
   "把同样的钱换成另一种花法"是这个臂最本质的变化——这解释了 §9 为什么收益为 0。
3. `DOWNGRADE` 占比纹丝不动（46.0% vs 46.0%）：`AZAI_UPG_BUNDLES` 只加升级候选，
   **没有**动"建塔→降级→再建"的 churn 结构（那是另一条独立的问题）。

---

## 7. 指标 3/4/5：胜负、健康度、耗时（三臂并排）

`analyze_inject_vs_runnerup.py`（同一批 seeds 7..14×2 座位，可逐 token 配对）：

| 臂 | n | W-D-L | **胜率**（Wilson 95%CI） | **血量差**（t 95%CI） | 平均回合 | op/决策 |
|---|---|---|---|---|---|---|
| **A（注入亚军那一手）** | 16 | 8-0-8 | **50.0%** [28.0, 72.0] | **+0.56** [−9.16, +10.29] | 507 | 1.50 |
| **B = D（不注入，基线）** | 16 | 2-0-14 | **12.5%** [3.5, 36.0] | **−20.12** [−28.33, −11.92] | 492 | 1.64 |
| **C（`AZAI_UPG_BUNDLES=4`）** | 16 | 2-0-14 | **12.5%** [3.5, 36.0] | **−20.25** [−27.98, −12.52] | 436 | **1.20** |

配对（同 token = 同图同座位）：

| 配对 | ΔHP (A−B) | t 95%CI | 胜负表 | McNemar |
|---|---|---|---|---|
| **C − B** | **−0.12** | **[−12.22, +11.97]**（跨 0） | 都赢 0 / 只 C 赢 2 / 只 B 赢 2 / 都输 12 | p=**1.0000**，**+0.0pp** |
| C − A | −20.81 | [−33.76, −7.86]（不跨 0） | 都赢 2 / 只 C 赢 0 / 只 A 赢 6 / 都输 8 | p=**0.0312**，**−37.5pp** |

* **C 与 B 完全同分（12.5%）**，血量差差 **0.13**；赢的局**换了两局**（C 赢 `8r/9r`，B 赢 `12/14r`），
  没赢的那 12 局两边都输 ⇒ 逐 seed 得分差 `{7:0, 8:+1, 9:+1, 10:0, 11:0, 12:−1, 13:0, 14:−1}`；
* **C 只拿到 A−B 那 +20.69 血量差里的 −0.12（≈−0.6%，即 0）**；上游报告 §5.2 留的那个替代解释
  （"注入的收益可能只是候选数/类型偏好"）**被这一臂否掉**：一个"升级占菜单 80%"的候选集
  （§8）**复现不出**注入收益的任何一部分；
* 座位内：C 在 P0 **0/8**、P1 **2/8**；B 是 P0 1/8、P1 1/8 ⇒ 座位内数字太小，不足以说"座位效应"。

**指标 4（健康度）**：

| 项 | C 臂 |
|---|---|
| `verdict` | **16/16 = `engine`**（官方 5 级级联，无 `hp_tiebreak`/`timeout`/`aborted`） |
| 裁判侧 `illegal` | **0**（16 局全部 `illegal=0`） |
| AI 侧自检 `illegal_total` | **0**（最大跨局 0） |
| 升级通道异常 | `[upg] ⚠️` 行 **0 条**；`gate_drop=0`（16 局累计） |
| 去重损耗 | `added == offered`（3829 = 3829，收尾快照口径）⇒ 升级候选**一条都没被当成重复丢掉** |
| 注入通道 | 未启用（`asks=0`，这是预期的：C 臂不注入） |

**指标 5（决策耗时；每局各算再取中位）**：

| 口径 | A（注入） | B（基线） | **C（升级候选）** |
|---|---|---|---|
| 决策数 | 8081 | 7675 | 6737 |
| **p50** | 1.73 s | 0.56 s | **0.54 s** |
| **p95**（每局中位 / 最坏单局） | 14.58 / 21.73 s | 19.76 / 21.17 s | **16.80 / 19.02 s** |
| **max**（每局中位 / 最坏单局） | 33.48 / 63.12 s | 28.84 / 42.22 s | **25.90 / 45.44 s** |

⇒ **C 不比 B 慢**（p50 0.54 vs 0.56、p95 16.80 vs 19.76、max 25.90 vs 28.84），也远快于 A 的 p50 1.73 s。
原因见 §8：绝大多数决策的候选集只有 1-5 条，离 `FLAT_ROUNDS` 的硬预算还很远。

---

## 8. 预算稀释的**实测**：−7.8% 是上界，真实代价 ≈ 0（但菜单被"升级"占满）

`[upg] 统计` 的 `cand/rollouts` 直方图（**每决策**一条，16 局共 7574 个样本）：

| `n_cand/n_rollouts` | 决策数 | 占比 | 备注 |
|---|---|---|---|
| 1/64 | 5034 | 66.5% | 网络去重后只剩 **1** 条（本回合没有升级候选） |
| **5/64** | 1692 | 22.3% | 其中 **878 = 网络 1 + 升级 4**；另 814 = 网络自己就 5 条 |
| 48/64 | 277 | 3.7% | 网络真的满 48 条（本回合没有升级候选） |
| 4/64 | 153 | 2.0% | 16 = 网络 1 + 升级 3；其余网络 4 条 |
| **52/59**（唯一的满额+4 档） | 126 | 1.7% | 64 = 网络 48 + 升级 4；62 = 网络自己 52 |
| 3/64 | 86 | 1.1% | 网络 3 条 |
| 其余（8…33 条） | 206 | 2.7% | — |

* **`n_rollouts` 被压到 64 以下的决策 = 132/7574 = 1.74%** ⇒ §3.2 那个 **−7.8%** 只作用于
  **1.7%** 的决策，加权后 **≈ −0.14%**。C 臂"采样上略吃亏"这句话**成立但可忽略**；
* 两种"吃亏"要分清：真正的代价不是**每候选**采样下降，而是**总 rollout 量上升**
  （1 候选 → 5 候选 = 5 倍 × 64 次 × 16 回合）；但耗时读数（§7）说明这在 10s/决策的规则内**没有体现**；
* ⚠️ **一个必须说清的发现（本次新测到）**：在这个配置下，**48 束实际去重后常常只剩 1 条候选**
  （`n_cand=1` 占 **66.5%** 的决策；`n_cand≥48` 只占 **5.4%**）。原因与上游 §3.2 是同一个：
  整束被钉在经济类上，类 16（降级）的合法位置 = 我方塔格（1-2 个）⇒ 48 次采样全撞在同一个 `(类,格)` 上。
  **所以 C 臂的干预不是"48→52 的边缘扰动"，而是"1→5"**——在 88.2% 的有候选回合里，
  升级候选占了菜单的 **80%**（4/5）。这条既解释了 §7 的耗时为什么没变，也说明
  §5 的 13.3% 选中率是在"升级占 80% 菜单"的条件下测出来的：
  **评估器有 47% 的回合仍然选了网络那唯一一条**（不选升级），所以它不是"无脑偏好升级"。

---

## 9. 结论：值不值得把 `AZAI_UPG_BUNDLES` 转正？——**不值得**（但值得留着当诊断开关）

**支持"不转正"的三条**：

1. **棋力**：C 与 B **完全同分**（12.5% vs 12.5%，McNemar p=1.0），血量差差 **0.13**
   （配对 CI [−12.2, +12.0] 完全跨 0）；与 A 臂 50.0% 差 37.5pp（p=0.031）。
   ⇒ 在这套对手上，**"升级进候选"没有可测收益**，同时也不是有害（自对弈历史说有害，这里没复现）。
2. **机制上它做对了、但不够**：升级真的被塞进候选、真的被同一个评估器选中（13.3%/条）、
   真的执行（526 = op 数），塔等级真的变成 58.8/39.8/1.5% —— 但**钱被换了个花法**：
   建塔 48.6%→26.2%、塔·回合 −33.5%、金币 p50 64→50。在"建塔→降级→再建"的 churn 结构里，
   升级只是把同一笔钱从"塔的数量"挪到"塔的等级"，而**亚军赢我们的方式不是塔等级**
   （对手 8.4% 升级、46.1% 降级，和我们一样的 churn）。所以 0 收益是可解释的，
   不是"没测到"那么简单。
3. **上游 §5.2 的替代解释被否掉**：A 臂 +20.69 的血量差里，**C 拿到 −0.12（≈0%）**。
   所以注入的收益**不是**"候选数/类型偏好"——那 4 条升级候选在 88% 的回合里占满 80% 菜单，
   却什么都没带来。这把注入的归因**收窄到"亚军那一手的知识含量"**（下一步该测的就是安慰剂臂）。

**为什么仍然值得留着这个开关**：

* 它是**第一个"进候选、同台评估"口径**的升级实验（区别于 `AZAI_MC_UPG` 的"MC 仲裁直接接管"），
  默认 0 且零影响（§2），**不占任何默认路径的成本**；
* 它把"升级是否有用"这个问题从"没试过"变成"试过、0 收益"，并且**给出读数工具**
  （`analyze_upg_bundles.py` 的选中率/预算/菜单份额），下一步任何"改钱流"的实验都能复用；
* 如果还要再试一次升级，**该改的不是这个开关，而是钱的分配**：例如只允许升到二级（60 金）、
  每 K 回合最多升 1 次、或与 `AZAI_NO_DOWNGRADE` 组合（先止住 46% 的拆塔 churn）。
  这一条**不在本报告范围内**，属于新实验（需要重新预注册）。

---

## 10. 没覆盖 / 风险 / 诚实清单

1. **16 局的胜率分辨率极低**：12.5% vs 12.5%，Wilson CI 都是 [3.5, 36.0]；要检出 +20pp 的差异
   大概需要 50+ 局。血量差口径更稳，ΔHP = −0.12（CI [−12.2, +12.0]）⇒ 只能说"没测到收益"，
   不能说"绝对没收益"（CI 允许 ±12 HP 的变化）。
2. **只有一个 N=4**：没扫 N=1/2/8。N 越大越挤掉建塔（§6）；N=1 可能"便宜地"测出
   与 N=4 不同的收益曲线，但那是新的预注册。
3. **"有升级候选的决策点占比"是 14.3%，不是 42.3%**：这是**结果不是原因**——C 臂把钱花在升级上
   ⇒ 金币 p50 从 64 掉到 50 ⇒ 买得起升级的回合从 42.3% 掉到 14.3%。两者不可直接比。
4. **只扫了 `tower_id` 升序 × 目标顺序的前 4 条**：没有做"哪个目标更值"的搜索；
   评估器自己偏好 PRODUCER（26.6%），说明"目标选择"本身是有信息量的维度（留给后续）。
5. **旧的 stderr 归档撞车（操作层面，不影响读数）**：`match_results/ai?_az_bridge_ai_seed<N>.stderr.log`
   是**全局共享**文件名（`MVS_STDERR_SUFFIX` 默认关）。本批跑完时另一个无关批次正好以同 seed 启动，
   把 `inject_vs_ru/C/stderr/ai0_az_bridge_ai_seed7.stderr.log` 覆盖成了别人的内容（3701 字节）。
   **已恢复**：`_tmp_ladder.py` 的**每局即时归档** `match_results/ladder_logs/INJECT_VS_RU_C/stderr/`
   是干净的 16 份；§5 的读数全部来自那里（并复制成 `match_results/upgrade_bundles/C_stderr_clean/`）。
   **教训**：以后并发跑不同批次时设 `MVS_STDERR_SUFFIX=1`，或只信 ladder 级归档。
6. **`AZAI_UPG_BUNDLES` 与注入同时开**没有实跑（§1.4 的合并语义只在代码层面验证过）：
   C 臂断言里明确禁掉了注入/安慰剂，避免归因混淆。
7. **两个工作流共用 `code/test_match/az_bridge_ai.py`**：我改的时候工作区里还有另一个工作流
   （安慰剂臂 / E4 `AZAI_ORACLE_INJECT_B` + `AZAI_ONLY_EXTRA`）的**未提交**改动。为了不替别人提交，
   `c0d23d0` 有意**只含本功能**：用 `_build_upg_only_variant.py` 把本功能的 6 个块插到 `HEAD` 版上
   （`git hash-object -w` + `update-index --cacheinfo`，**不碰工作区**）。所以：
   * 提交里的 `az_bridge_ai.py` = `f9afd27` + 本功能；工作区 = 提交版 + 同事的改动（原样保留）；
   * 两边的开关**默认全关** ⇒ 行为等价；C 臂实验跑的是工作区版，重现时用默认 env 即可；
   * `_upg_bundles_zero_impact.sh` 依赖的 `compare_result_lines.py` 也来自那个工作流
     （当时未跟踪）；若它不在，先确认那个工作流是否已提交。
8. **"改动前"副本的存放**：`code/test_match/_az_bridge_ai_preupg.py`（未跟踪，md5 `8841F843…`）
   是零影响对照的 before；同时在 `match_results/upgrade_bundles/_az_bridge_ai_preupg.py` 留了一份
   （`match_results/` 被 gitignore，属本地工件）。两份都能用：脚本按顺序找。

---

## 11. 复现命令

```bash
PY=D:/anaconda3/envs/pytorch-gpu/python.exe

# ① 冒烟（40-120 回合，看 [upg] 日志/通道/合法性；用 seed 3 避免污染 7..14）
bash code/test_match/_upg_bundles_smoke.sh 4 3 120

# ② 零影响（N=0 vs 改动前；返回 0 = 逐字段不变）
bash code/run_logged.sh upg_bundles_zero_impact bash code/test_match/_upg_bundles_zero_impact.sh

# ③ 诊断臂 C（16 局 = seeds 7..14 × 2 座位，8 并行，~59 min）
ARMS=C UPG_N=4 JOBS=8 RETRY=2 \
  bash code/run_logged.sh upg_bundles_ab bash code/test_match/_inject_vs_runnerup.sh

# ④ 读数
$PY -u code/test_match/analyze_upg_bundles.py \
    --stderr match_results/ladder_logs/INJECT_VS_RU_C/stderr \
    --json match_results/upgrade_bundles/upg_rates_C.json
$PY -u code/test_match/probe_why_no_upgrade.py --tag INJECT_VS_RU_C \
    --json match_results/upgrade_bundles/probe_why_no_upgrade_C.json
$PY -u code/test_match/analyze_inject_vs_runnerup.py \
    --arm inject_A=match_results/ladder_logs/INJECT_VS_RU_A --stderr inject_A=match_results/inject_vs_ru/A/stderr \
    --arm inject_B=match_results/ladder_logs/INJECT_VS_RU_B --stderr inject_B=match_results/inject_vs_ru/B/stderr \
    --arm inject_C=match_results/ladder_logs/INJECT_VS_RU_C --stderr inject_C=match_results/upgrade_bundles/C_stderr_clean \
    --pair inject_C inject_B --pair inject_C inject_A
```

---

## 12. 工件清单

| 路径 | 内容 |
|---|---|
| `code/test_match/az_bridge_ai.py` | `AZAI_UPG_BUNDLES`（新增，默认 0） |
| `code/test_match/_inject_vs_runnerup.sh` | 加 **C 臂**（`ARMS=C UPG_N=4`），A/B 分支逐字未动 |
| `code/test_match/_upg_bundles_smoke.sh` | 冒烟（40-120 回合） |
| `code/test_match/_upg_bundles_zero_impact.sh` | 零影响 A/B（同 seed，逐字段） |
| `code/test_match/_az_bridge_ai_preupg.py` | **改动前**原样副本（md5 `8841F843…`，零影响对照的 before） |
| `code/test_match/analyze_upg_bundles.py` | 读数脚本（选中率 / 目标偏好 / 预算 / 菜单份额） |
| `code/test_match/_build_upg_only_variant.py` | 只把**本功能**的 6 个块插到 `HEAD` 版上的构建器（见下） |
| `match_results/upgrade_bundles/upg_rates_C.json` | 指标 1 的原始读数（含逐局） |
| `match_results/upgrade_bundles/probe_why_no_upgrade_C.json` | 指标 2（op 构成 / 塔等级 / 金币 / 机会） |
| `match_results/upgrade_bundles/arms_ABC.{json,txt}` | 指标 3/4/5（三臂 + 配对） |
| `match_results/upgrade_bundles/zero_impact/*.log` | 零影响的 4 局原始日志 |
| `match_results/upgrade_bundles/C_stderr_clean/` | C 臂 16 局的干净 stderr 副本（每局即时归档） |
| `match_results/ladder_logs/INJECT_VS_RU_C/` | C 臂 16 局对局日志 + 即时 stderr 归档 |
| `training_history/progress/INJECT_VS_RU_C/` | 逐回合 trace（op/金币/血量） |
| `training_history/runs/20261028_215428_upg_bundles_ab/output.log` | 驱动脚本完整输出 |
| `training_history/runs/20261028_213201_upg_bundles_zero_impact/output.log` | 零影响完整输出 |
