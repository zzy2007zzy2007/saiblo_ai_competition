# 亚军候选注入 A/B（P2）+ 事后过滤「整块合法性闸门」（P1）

> 2026-10-28。工作目录 `D:\2026智能体大赛_新2_distill`。
> 两件前后依赖的事：**先修真非法操作 bug（P1）**，再跑**注入 A/B（P2，主目的）**。
>
> 相关文档：`docs/illegal_op_strict_mode_20261028.md`（§5.3 = P1 的复现与根因）、
> `docs/oracle_inject_design.md`（注入通道设计）、`docs/runnerup_oracle_protocol.md`（协议）、
> `docs/match_retry_resume_20261028.md`（`--resume/--retry-invalid/--game-timeout-s`）。

---

## 1. 计划（写在动手之前）

### 1.1 P1：为什么必须修

`docs/illegal_op_strict_mode_20261028.md` §5.3 已经用真实对局抓到了 2/2 局非法操作，机制干净：

```
AZAI_ORACLE_INJECT 把亚军整块候选 [(13,3,-1),(11,8,2)] 注入搜索（整块合法：先降级退钱再建塔）
AZAI_NO_DOWNGRADE=1 在执行层只删掉 (13,3,-1)  ⇒ 剩下 (11,8,2) 买不起（coins=3 < cost=45）
严格模式（=官方口径）⇒ 立刻判负
```

这是**一类** bug，不是一条：所有"事后从已合法的块里删 op"的开关都同构
（`AZAI_NO_DOWNGRADE`、`AZAI_BAN_OPS`，以及它们与注入/`mc_last_explicit`/`FORCE_UPG` 的组合）。

### 1.2 P1：修法（最小改动、默认零影响）

在 `code/test_match/az_bridge_ai.py` 的 `decide()` 里，四个事后过滤块**之后**加一道
**整块合法性闸门**：

1. 记下过滤前的 op 数 `_n_raw`；
2. 四个过滤块（`NO_DOWNGRADE` / `LIGHT_MIN_COINS` / `BUILD_MIN_COINS` / `BAN_OPS`）**原样不动**；
3. **只有真的删过东西**（`len(ops) != _n_raw`）才复核：逐条 `can_apply`（dry + clone 冷路径双查，
   复用 `match_via_sdk.check_ops`），`pending=` 累计已接受前缀（同一回合内的金币/塔位/基地升级语义）；
4. 只要有一条非法 ⇒ **整块丢弃**，本回合退化为 **HOLD**（返回 `[]`），并打一行带
   `过滤前/过滤后/非法 op 与 reason` 的日志 + 计数 `filter_gate_dropped`。

设计选择与理由：

| 选择 | 理由 |
|---|---|
| 退化为 **HOLD**（而不是"回到搜索的下一名"） | ① 必然合法；② 不需要重排 `chosen_index/visit_policy/AZ_DUMP`，改动面最小；③ 任务允许两者取一 |
| **不**回退到"未过滤前" | 那等于把开关关掉，会让实验读数变成"开关没生效" |
| 闸门用**独立**的 `IllegalStats` | `ILLEGAL_STATS` 的口径是"**真的发出去了**多少非法 op"；被拦下的不能算进去，否则 `RESULT ... illegal=` 虚高 |
| 闸门日志把 `ILLEGAL ` 改写成 `GATE-BLOCK ` | `_tmp_ladder.py:192` 数的是"含 `ILLEGAL` 的行数"（= 发出去的非法 op），闸门拦下的没发出去 |
| **没删 op 就不复核** | 保持默认行为逐位不变、零额外成本（开关默认全关 ⇒ 闸门一次都不进） |
| 闸门自身异常 ⇒ **丢弃**（fail-closed） | 与 `decide_final` 自检的 fail-open 相反：这里 fail-open = 可能发出非法 op = 当场判负，而 HOLD 一定安全 |
| **不**给 `FORCE_UPG` / `mc_last_explicit` 加闸门 | 它们不是"事后过滤"（它们在过滤之前就 `return` 了）；`FORCE_UPG=1` 的 `(x,y)` 形式**本来就是**故意发非法 op 的诊断模式（F3），拦住等于废掉它 |

### 1.3 P2：注入 A/B 设计

| 项 | 值 |
|---|---|
| A = 被测臂 | `AZAI_ORACLE_INJECT=<亚军 exe 绝对路径>`（注入开启） |
| B = 基准臂 | 同配置，注入关闭 |
| 其余配置 | 两臂**完全相同**（BASE 串见 §4.1），`AZAI_FLAT_SEED=53` 固定 |
| 禁用开关 | **不开** `AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS`（这是 P1 的雷区，且会污染主读数） |
| 规模 | 地图 seeds **3600-3607**（8）× 2 座位 = **16 局**，`--jobs 8`、`--paired-sides` |
| 驱动 | `ab_search_modes.py --resume --retry-invalid 2 --game-timeout-s 5400` |
| 主读数 | `flat 胜率` = **A（注入臂）得分**；再按 `@P0`/`@P1` 拆分，另给血量差 |
| 副读数 | AI stderr 里的注入次数 / `hit_rate` / `illegal` / oracle 平均·最大耗时 |

已知坑与对策：`ab_search_modes --paired-sides` 同一个 seed 的两局**并发**，而
`match_via_sdk` 的 stderr 文件名只含 `(seat, label, seed)` ⇒ 两局会写**同一个文件**
（`docs/oracle_inject_design.md` §4.3 记录过）。为避免"注入统计读不出来"，
给 `match_via_sdk` 加一个**默认关闭**的 opt-in：`MVS_STDERR_SUFFIX=1` 时把驱动已经按局设置的
`DSH_PROGRESS_SUFFIX`（`_p0`/`_p1`）拼进 stderr 文件名。不设该环境变量 ⇒ 文件名与改动前逐字相同。

---

## 2. P1 修复（代码）

### 2.1 `code/test_match/az_bridge_ai.py`（共 +94/−6 行，全部围着"过滤之后"）

| 位置 | 改动 |
|---|---|
| 模块级（`ILLEGAL_STATS` 之后） | 新增 `_gate_log()` + `_block_legal_gate(facade, pl, ops)`：**整块**只读复核，复用 `MVS.check_ops`（`can_apply_dry` + clone 版 `can_apply_cold` 双查、`pending=` 前缀语义、reason 分类），用**独立**的 `IllegalStats`，日志词 `ILLEGAL`→`GATE-BLOCK` |
| `decide()` 的四个过滤块之前 | 记 `_n_raw = len(ops)`、`_raw_ops = ops`（只存引用，**零拷贝**） |
| `decide()` 的四个过滤块之后 | **闸门**：`if len(ops) != _n_raw:` ⇒ 复核剩余整块；有非法 ⇒ `ops = []`（本回合 HOLD）+ `filter_gate_dropped` + 一行 `🛡️ [gate] 过滤后整块非法 ⇒ 整块丢弃` 日志（含过滤前/过滤后/非法 op 与 reason） |
| `AZAI_NO_DOWNGRADE` 块里的旧自检 | **删掉** `_legal_check(..., "after_no_downgrade")`：它只记录不拦，且会把"**没发出去**的 op"计进 `ILLEGAL_STATS` ⇒ `illegal_total` 失真。同一件事现在由闸门做（而且会真的丢） |
| `_report()` / 每 25 决策的周期日志 | 多打 `gate_checks` / `gate_dropped`（收尾报告常被 bridge 硬杀，所以必须对局中也打） |
| `main()` init | `STATS["seed"] = seed`（闸门日志要带 seed） |

关键设计点（全部可核对）：

* **开关全关 ⇒ 闸门一次都不进**（`len(ops) == _n_raw`）⇒ 默认行为逐位不变、零额外成本；
* **fail-closed**：闸门自己异常、或 `check_ops` 报 `CHECK-ERROR`（校验器没给出判定）⇒ 按非法处理（丢块）。
  与 `decide_final` 自检的 fail-open 故意相反：那里只是记录，这里决定"发不发"；
* `ILLEGAL_STATS`（口径 = 真的发出去的非法 op）**不被闸门污染** —— 拦下的量只在 `gate_dropped` 里；
* 没有碰 `code/my_ai/az_intent/**` 一行。

### 2.2 `code/test_match/match_via_sdk.py`（+9/−2 行，**只影响 stderr 文件名**）

`MVS_STDERR_SUFFIX=1`（**默认关**）时，把驱动已按局设置的 `DSH_PROGRESS_SUFFIX`（`_p0`/`_p1`）拼进
stderr 文件名：`ai0_<label>_seed<N>{suffix}.stderr.log`。动机见 §1.3 ——
`--paired-sides` 同一 seed 两局并发会写同一个 stderr 文件，注入统计读不出来。
**不设该变量 ⇒ 文件名与改动前逐字相同**（`_tmp_ladder.py` 的 `ai?_*_seed{N}.stderr.log` 等老消费者不受影响）。

### 2.3 新增工具

| 文件 | 作用 |
|---|---|
| `code/test_match/probe_filter_gate.py` | 闸门的**确定性**探针（不起子进程、不走搜索）：用引擎把局面推到"降级退钱 + 建塔合法、但只剩建塔非法"，验 4 件事（抓得住 / 不误杀 / 不污染计数 / fail-closed）。失败返回 1，可当回归测试 |
| `code/test_match/_filter_gate_matrix.sh` | 「注入 × 各过滤开关」的组合矩阵（4 局并行），见 §3.3 |
| `code/test_match/_oracle_inject_ab.sh` | P2 的 A/B 驱动（16 局），见 §4 |
| `code/test_match/analyze_oracle_inject_ab.py` | A/B 读数汇总（每局明细 / 胜率 / 座位拆分 / 血量差 / 注入统计 / 闸门计数） |

## 3. P1 验证

### 3.1 确定性探针：`probe_filter_gate.py`（8/8 PASS，2 秒）

它用引擎把局面推到"**整块合法、半块非法**"，这个局面本身就是 bug 的存在性证明：

```
局面: seed=7 round=2 coins=8 build_cost=45 塔 id=0 type=0 level=0 降级退钱=54 cell=(4, 9)
[PASS] 构造出「整块（降级+建塔）合法」的局面  raw=[(13, 0, -1), (11, 4, 9)]
[PASS] 构造出「半块（只剩建塔）非法」的局面  half=[(11, 4, 9)]
[PASS] 闸门：过滤后的半块被判非法（⇒ 调用方会整块丢弃）  bad=[((11, 4, 9), 'build_no_gold(cost=45,coins=8)')]
[PASS] 闸门给出的 reason 是 build_no_gold（钱不够）  build_no_gold(cost=45,coins=8)
[PASS] 闸门**没有**污染 ILLEGAL_STATS（口径 = 真发出去的非法 op）  ILLEGAL_STATS.total=0（调用前 0）
[PASS] 闸门：过滤前的整块判为合法（不误杀，不会把好块丢成 HOLD）  bad=[]
[PASS] 闸门：空块（HOLD）判为合法
[PASS] 闸门：检查器失效 ⇒ fail-closed（按非法处理）  bad=[(None, 'checker_unavailable')]
=== 全部 PASS ===
```

读法：`coins=8 < cost=45` 而 `降级退钱=54` ⇒ 引擎的建塔检查（内含金币）**单独看**判非法、
带上降级 op 作为 `pending` 就判合法。这与 §5.3 实测的两个局面（`coins=3` / `coins=30` vs `cost=45`）**同构**。

### 3.2 默认行为不变（逐字段相同，跨过这次改动）

`_illegal_smoke.sh` 的 canonical 批（`AZAI_ITERS=256` 部署口径、无任何过滤开关）seed 8101 这一局，
与 **2026-10-27 旧代码**（`runs/20261027_185410_illegal_cold_control`）跑出的结果**逐字段相同**：

```
旧代码: RESULT seed=8101 ... rounds=383 terminal=True base_hp=0,31 coins=55,507 illegal=0
新代码: RESULT seed=8101 ... rounds=383 terminal=True base_hp=0,31 coins=55,507 illegal=0     ← 一字不差
旧代码: illegal-summary seed=8101 cold=True  illegal_total=0 checked_ops=123 holds=p0:355,p1:326 by_side={0:0,1:0} by_kind={} mismatch=0
新代码: illegal-summary seed=8101 cold=False illegal_total=0 checked_ops=123 holds=p0:355,p1:326 by_side={0:0,1:0} by_kind={} mismatch=0
```

`checked_ops` / `holds` / `by_kind` 全同 ⇒ **两边发出去的 op 序列一模一样**（不是"看起来差不多"）。
结构上也成立：闸门被 `if len(ops) != _n_raw:` 挡住，四个开关默认全关时一个字节都不会多读。

### 3.3 复现路径：`_illegal_smoke.sh` 批次 2（§5.3 的那个组合）

批次 2 = `AZAI_ORACLE_INJECT` + `AZAI_NO_DOWNGRADE=1` + `AZAI_MC_UPG=1` + flat 搜索，
配置与 §5.3 **逐字相同**，seed 也还是 **8103 / 8103r**。日志：
`training_history/runs/20261028_120847_illegal_strict_highrisk/output.log`
（stderr 归档：`training_history/p1_logs/smoke_batch2_stderr/`）。

| 局 | 修复前（§5.3，2026-10-27） | 修复后（本次） |
|---|---|---|
| seed 8103 我方 p0 | `ILLEGAL r61 (11,8,2) build_no_gold(45,3)` ⇒ `verdict=illegal_strict`，**第 61 回合判负** | **`verdict=engine rounds=338 base_hp=0,39 illegal=0`**，`illegal_total=0`、`mismatch=0`、闸门 `82 checks / 14 dropped` |
| seed 8103r 我方 p1 | `ILLEGAL r244 (11,10,8) build_no_gold(45,30)` ⇒ `verdict=illegal_strict`，第 244 回合判负 | **`verdict=engine rounds=401 base_hp=30,0 illegal=0`**，`illegal_total=0`、闸门 `74 checks / 29 dropped` |

**同一手、同一回合，现在被闸门整块丢掉**（stderr 原文；两个 `round` 与两个 op 三元组都与 §5.3 逐字一致）：

```
[azai] 🛡️ [gate] 过滤后整块非法 ⇒ **整块丢弃（本回合 HOLD）**: round=61        ← 8103(p0)
       过滤前=[(13, 3, -1), (11, 8, 2)] 过滤后=[(11, 8, 2)]
       非法=[(0, (11, 8, 2), 'build_no_gold(cost=45,coins=3)')] （累计丢弃 1/2 次）
[azai] 🛡️ [gate] 过滤后整块非法 ⇒ **整块丢弃（本回合 HOLD）**: round=141
       过滤前=[(13, 10, -1), (13, 9, -1), (11, 7, 1)] 过滤后=[(11, 7, 1)]
       非法=[(0, (11, 7, 1), 'build_no_gold(cost=90,coins=16)')] （累计丢弃 2/15 次）
[azai] 🛡️ [gate] 过滤后整块非法 ⇒ **整块丢弃（本回合 HOLD）**: round=244       ← 8103r(p1)
       过滤前=[(13, 16, -1), (13, 33, -1), (11, 10, 8)] 过滤后=[(11, 10, 8)]
       非法=[(0, (11, 10, 8), 'build_no_gold(cost=45,coins=30)')] （累计丢弃 1/23 次）
```

两局的注入统计：`8103` = `asks=338 inject=338 hit=142 hit_rate=69.3% timeout=0 avg_ms=667 max_ms=1438`；
`8103r` = `asks=401 inject=401 hit=137 hit_rate=71.4% timeout=0 avg_ms=674 max_ms=1688`
—— 注入**全程活着**（`timeout=0 dead=0`），所以"0 非法"不是"注入早退"换来的。

### 3.4 组合矩阵：注入 × 各过滤开关（1 局/组合，4 局并行）

脚本：`code/test_match/_filter_gate_matrix.sh`；日志：
`training_history/runs/20261028_114004_filter_gate_matrix_final/output.log`。
每局都是 **`AZAI_ORACLE_INJECT` 开启** + 一个事后过滤开关（HIGHRISK 底配置，
`_illegal_smoke.sh` 批次 2 同款；**op 的合法性与搜索强度无关**）。

| 组合 | seed / 我方座位 | verdict | 裁判侧 `illegal=` | AI 侧 `illegal_total` | 闸门 checks / **dropped** |
|---|---|---|---|---|---|
| `AZAI_NO_DOWNGRADE=1` | 8103 / p0 | **engine** | **0**（0 行 ILLEGAL） | **0** | 82 / **14** |
| `AZAI_BAN_OPS=13` | 8103r / p1 | **engine** | **0**（0 行 ILLEGAL） | **0** | 74 / **29** |
| `AZAI_LIGHT_MIN_COINS=180` | 8201 / p0 | **engine** | **0**（0 行 ILLEGAL） | **0** | 81 / 0 |
| `AZAI_BUILD_MIN_COINS=90` | 8202 / p0 | **engine** | **0**（0 行 ILLEGAL） | **0** | 34 / 0 |

⇒ **4/4 组合 0 非法**（`[matrix] ✅ 全部组合 0 非法操作`），且两个"删降级"的组合里闸门**真的拦下了半块**
（不是"没触发所以没非法"）：

```
─ no_downgrade ─  round=61 过滤前=[(13, 3, -1), (11, 8, 2)] 过滤后=[(11, 8, 2)]
                 非法=[(0, (11, 8, 2), 'build_no_gold(cost=45,coins=3)')]      ← §5.3 的**同一手**
─ no_downgrade ─  round=141 过滤前=[(13,10,-1),(13,9,-1),(11,7,1)] 过滤后=[(11,7,1)]
                 非法=[(0, (11,7,1), 'build_no_gold(cost=90,coins=16)')]
─ ban_ops13 ─     round=396 过滤前=[(13, 66, -1), (12, 61, 1)] 过滤后=[(12, 61, 1)]
                 非法=[(0, (12, 61, 1), 'upgrade_no_gold(cost=60,coins=55)')]   ← **升级**版（新变种）
```

三个读数点：

1. **不是只有建塔会中招**：`BAN_OPS=13` 那局被拦的是**升级**（`upgrade_no_gold(cost=60,coins=55)`）
   —— 同一机制的第二个形态，docs/illegal_op_strict_mode §7 第 1 条当时只"推测同构"，现在有实例了；
2. **"没删降级"的两个组合里闸门 dropped=0**（`LIGHT_MIN_COINS` / `BUILD_MIN_COINS` 只可能让**少花钱**，
   删掉的东西本身是消费项）⇒ 闸门没有乱杀；这两组仍是 `illegal=0`、`verdict=engine`；
3. 4 局全部跑到引擎终局（426/401/426/512 回合），**没有一局因为非法操作提前结束**。

### 3.5 复现路径：`_illegal_smoke.sh`（整脚本 4 批，全部走 `run_logged`）

```bash
bash code/test_match/_illegal_smoke.sh          # 故障注入 2 局 + canonical 严格 4 局 + 高风险 2 局 + 旧行为对照 1 局
```

**批次 4（故障注入，`runs/20261028_111714_illegal_fault_inject_strict` / `…111721…cold`）**：
判负逻辑没被我改坏 —— 故意发非法 op 的 `faulty_ai.py` 仍然被抓：

```
严格模式: RESULT seed=7501 ... winner=runnerup winner_side=p0 verdict=illegal_strict rounds=0 base_hp=50,50 illegal=1
旧行为  : RESULT seed=7501 ... winner=draw winner_side=draw verdict=hp_tiebreak rounds=15 base_hp=50,50 illegal=15
```

**批次 1（canonical 部署配置 × 4 局 vs 亚军，`runs/20261028_111727_illegal_strict_smoke`）**：
`illegal_total=0`、`mismatch=0`、`verdict=engine` 4/4：

| seed | 我方座位 | verdict | illegal | rounds | base_hp | checked_ops | holds |
|---|---|---|---|---|---|---|---|
| 8101 | p0 | engine | 0 | 383 | 0,31 | 123 | p0:355,p1:326 |
| 8101r | p1 | engine | 0 | 475 | 29,0 | 241 | p0:364,p1:416 |
| 8102 | p0 | engine | 0 | 476 | 0,25 | 215 | p0:425,p1:377 |
| 8102r | p1 | engine | 0 | 449 | 19,0 | 258 | p0:328,p1:405 |

（我方 0/4 胜 —— canonical 配置本来就打不过亚军，与本次改动无关；判据是"0 非法 + 引擎终局"。）

**批次 2（§5.3 的高风险组合）× 2 局** 与 **批次 3（旧行为对照）**：见 §3.3。

## 4. P2 A/B 结果

_待填_

## 5. 结论

_待填_
