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

### 4.1 设计（照任务书，逐字）

```bash
bash code/run_logged.sh oracle_inject_ab bash code/test_match/_oracle_inject_ab.sh
# 日志：training_history/runs/20261028_132806_oracle_inject_ab/output.log
#       match_results/oracle_inject_ab/ab16/{seed<N>_p<S>.log, _progress.json, stderr/*.stderr.log}
```

| 项 | 值 |
|---|---|
| A（被测臂） | `BASE + AZAI_ORACLE_INJECT=<亚军 exe 绝对路径> + AZAI_ORACLE_TIMEOUT=15` |
| B（基准臂） | `BASE`（逐字等于任务书那串，600 字符，用脚本核对过 `identical=True`） |
| 共同项 | `AZAI_FLAT_SEED=53`，两臂环境串除上面两处外**完全相同** |
| 未开 | `AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS`（一个都没开） |
| 规模 | seeds 3600-3607 × 2 座位 = 16 局，`--paired-sides --jobs 8` |
| 驱动 | `ab_search_modes.py --resume --retry-invalid 2 --game-timeout-s 5400` |
| 代码 | commit **`475894a`**（跑之前提交；跑的过程中没有再改代码） |

**唯一一处字面偏离 + 理由**（重要，先说清楚）：`AZAI_ORACLE_TIMEOUT` 从默认 2.5s 放宽到 **15s**。
第一版严格按默认 2.5s 跑，8 个 A 臂里有 **5 个**在 round 55/123/150/187/207 触发 `OracleTimeout`
——每次都停在 **2515-2516ms**（刚刚超线），而同一局的 p95 只有 ~1.4s ⇒ 是调度/内存抖动造成的**尖峰**，
不是"亚军算得慢"。超时后注入**永久停用**（读端错位，协议上不能重连，见 `oracle_inject_design.md` §3.4），
后半局 A 臂就退化成 B 臂 ⇒ 主读数被稀释。证据（第一版 16 个 stderr 全留档）：
`training_history/p1_logs/ab16_attempt1_stderr/`、第一版进度 `…/ab16_attempt1_progress.json`。
这个阈值只决定"等亚军多久算失败"，不动搜索/网络/B 臂；放宽到 15s（= p95 的 10 倍）后仍能抓真死。

### 4.1.1 两个"静默失效"（都留了证据，值得记下来）

1. **`AZAI_ORACLE_TIMEOUT=2.5` 在 `--jobs 8` 下太紧** ⇒ 5/8 局注入中途永久停用（见上）。
2. **Git-bash 的 `$REPO` 是 `/d/...` 形态** ⇒ `az_bridge_ai._oracle_init` 的
   `Path(exe).is_absolute()` 在 Windows 上对 `/d/...` 判 **False**，于是拼成 `<repo>/d/...`
   ⇒ `Popen` 抛 `FileNotFoundError` ⇒ **注入整局静默不启用，而 A/B 看起来照跑**。
   第二版就是这样白跑了 ~21 分钟（1247 s）：8 个 A 臂全是 `⚠️ 启用失败`，且因为两臂行为完全相同，
   配对的两局连 `HP/金币` 都**逐字节相同**（`…/ab16_attempt2_noinject/`）。
   修法：`RU="$(pwd -W)/$RU_REL"`（`D:/...` 形态）⇒ 第三版每个 seed 里**恰好一个座位**
   `注入已启用=1`、另一个 `=0`，且注入统计真的有 `asks/hit`（见 §4.3）。

### 4.2 每局明细 + 汇总

第 3 版（有效的那版）：`13W / 0D / 3L`，**16/16 有效局**（全部 `verdict=engine`、`illegal=0`）。
A = 注入臂；`hp_A / hp_B` = 终局基地血量（`base_hp` 按 A 的座位换算）；血量差 = `hp_A - hp_B`。

| seed | A 座位 | 结果 | hp_A | hp_B | 血量差 | 回合 |
|---|---|---|---|---|---|---|
| 3600 | P0 | 胜 | 19 | 3 | **+16** | 512 |
| 3600 | P1 | 胜 | 33 | 1 | **+32** | 512 |
| 3601 | P0 | 胜 | 26 | 0 | **+26** | 444 |
| 3601 | P1 | 胜 | 30 | 0 | **+30** | 466 |
| 3602 | P0 | 胜 | 20 | 0 | **+20** | 469 |
| 3602 | P1 | 胜 | 19 | 11 | **+8** | 512 |
| 3603 | P0 | 胜 | 23 | 10 | **+13** | 512 |
| 3603 | P1 | **负** | 15 | 16 | −1 | 512 |
| 3604 | P0 | 胜 | 27 | 0 | **+27** | 504 |
| 3604 | P1 | 胜 | 12 | 8 | **+4** | 512 |
| 3605 | P0 | 胜 | 14 | 11 | **+3** | 512 |
| 3605 | P1 | **负** | 15 | 18 | −3 | 512 |
| 3606 | P0 | 胜 | 13 | 7 | **+6** | 512 |
| 3606 | P1 | 胜 | 19 | 12 | **+7** | 512 |
| 3607 | P0 | **负** | 15 | 18 | −3 | 512 |
| 3607 | P1 | 胜 | 22 | 9 | **+13** | 512 |

| 口径 | n | W/D/L | 胜率 | SE | 平均血量差 | 平均回合 |
|---|---|---|---|---|---|---|
| **全部** | 16 | **13W/0D/3L** | **81.2%** | ±9.8% | **+12.38**（95%CI **+6.6 ~ +18.1**） | 502 |
| `@P0`（A 执先手） | 8 | 7W/0D/1L | 87.5% | ±11.7% | +13.50 | 497 |
| `@P1`（A 执后手） | 8 | 6W/0D/2L | 75.0% | ±15.3% | +11.25 | 506 |

* **统计复核**：上表所有数字都由独立脚本**从 16 个 `RESULT` 行重新解析**过一遍（不看我这边的汇总输出），
  得 `13W-0D-3L / 81.2%`、`血量差 mean=+12.375 sd=11.74 SE=2.93 95%CI [+6.62, +18.13]`、
  `@P0 7W-1L 87.5% / @P1 6W-2L 75.0%`、`符号检验 p=0.0213`、配对分
  `{3600:1.0, 3601:1.0, 3602:1.0, 3603:0.5, 3604:1.0, 3605:0.5, 3606:1.0, 3607:0.5}`
  （5 个 seed 2-0 / 3 个 1-1 / 0 个 0-2）。日志：`training_history/p1_logs/ab16_stats_recheck*.log`。
* 驱动自己的汇总：`flat 胜率 81.2% (13W/0D/3L) ±25.0%(2SE)，wall=104.1 min，无效局=0`，
  `A 在 P0: 7W-1L = 87.5% (n=8)`、`A 在 P1: 6W-2L = 75.0% (n=8)`、`=== 最终有效率 16/16 (100.0%) ===`。
* **配对（按 seed）**：5 个 seed 2-0 拿下（3600/3601/3602/3604/3606）、3 个 seed 1-1（3603/3605/3607）、
  **0 个 seed 0-2**。配对分 = 6.5/8 = 0.8125，与逐局口径一致。
* **符号检验**（13W vs 3L，忽略平局）：双侧 p ≈ 0.021（`2·Σ_{k=13..16} C(16,k)/2^16 = 0.0213`）。
* **3 场败局都是 1-3 血**（−1 / −3 / −3），而 13 场胜局平均 +16.4 血 —— 不是"互有胜负的硬币"，
  更像是"赢了就赢很多、输的只是差一口气"。
* **座位不是解释**：两个座位都 >70%，且同一 seed 的两局是同一张图、只换座位（配对设计）。
* **成本**（同一局里 A、B 各占一边，所以这是"带注入"的单局墙钟）：44-58 min/局（`--jobs 8`，
  wave 2 的 `min` 读数含排队时间，不可比）。单次 oracle 查询均值 1.14 s × 512 决策 ≈ **+9.7 min/局**
  （约占 20%）。

**口径复核（怎么确认"`_p0` 的日志 = A 臂执先手"）**：`ab_search_modes.py:399-401` 是
`flat_is_p0 = (flat_side_in == 0)` ⇒ `env0 = flat_env if flat_is_p0 else mcts_env`，
而日志名 `seed<N>_p<S>.log` 的 `S` 就是 **flat（= A）臂的座位**（`_log_name`，同文件 80-82 行）；
`score_of()` 再用同一个 `S` 把 `winner_side` 折成 A 的得分。所以 `_p0` 那 8 局 = A 执先手，
`_p1` 那 8 局 = A 执后手；本报告的 `hp_A/hp_B` 也是按这个座位从 `base_hp=(hp0,hp1)` 里取的。
核对结果：`_p0` 7W-1L、`_p1` 6W-2L，与驱动自己打的 `A 在 P0 / A 在 P1` 两行**逐字一致**。

> ⚠️ **局限（别把 81.2% 读成"能赢亚军"）**：
> ① **n=16**（8 张图 × 2 座位），单一 `AZAI_FLAT_SEED=53`；
> ② **自我对弈**：A、B 两臂是**同一个网络、同一份配置**，唯一差别是注不注入
>   ⇒ 这 81.2% 只能说明"**在同一套自己人里，注入这一侧更强**"，**不等于**"对亚军本体更强"；
> ③ **12/16 局打满 `max_rounds=512`**（靠引擎自己的 hp 级联判胜），只有 4 局在 444-504 回合提前结束
>   ⇒ 主读数是"血量差/回合上限"口径，不是"推掉基地"口径（§4.2 的 hp 差已把这一点带出来）。
### 4.3 注入侧读数（A 臂 stderr 汇总，16 局）

| 字段 | 值 | 说明 |
|---|---|---|
| `asks` / `ok` | **8027 / 8027** | 每回合都问、每回合都拿到回包（**0 次失败**） |
| `inject` | 8027 | 都交给了搜索（含 HOLD） |
| `new` | **3051（38.0%）** | 去重后**真的新增**了候选 |
| `dup` | **4976（62.0%）** | 与已有候选重复（主要是双方都 HOLD，见 §6） |
| `hit`（**老口径** = 新代码的 `hit_sel`） | **1269** | 注入的额外候选被搜索选中 |
| `hit/new` | **41.6%** | 老口径；`hit/asks` = 15.8% |
| `illegal` | **0** | 亚军给的候选没有一条非法（逐条 `can_apply` 复核了 2149 个 op） |
| `timeout` / `dead` | **0 / 0** | 放宽到 15 s 后**一局都没丢注入** |
| 单次查询耗时 | 每局 avg **810-1438 ms**（16 局均值 1141 ms）、p50 719-1531 ms、**全体最大 6859 ms** | ⚠️ 最大 6.9 s **远超默认 2.5 s 阈值** |
| 过滤闸门 | `checks=0 dropped=0` | P2 没开任何过滤开关（符合设计） |
| 实际发出的非法 op | **0**（16 局 `illegal-summary illegal_total=0`） | 严格模式下没人判负 |

**逐局交叉（A 臂注入读数 × 战果；`hit_rate` 是老口径 hit/new）**

| seed | A 座位 | 结果 | 血量差 | asks | new | hit(new) | hit/new | avg_ms | max_ms |
|---|---|---|---|---|---|---|---|---|---|
| 3600 | P0 | 胜 | +16 | 512 | 197 | 75 | 38.1% | 1438 | 4562 |
| 3600 | P1 | 胜 | +32 | 512 | 173 | 70 | 40.5% | 1349 | 5985 |
| 3601 | P0 | 胜 | +26 | 444 | 188 | 86 | 45.7% | 1187 | 5281 |
| 3601 | P1 | 胜 | +30 | 466 | 162 | 76 | 46.9% | 1308 | 5953 |
| 3602 | P0 | 胜 | +20 | 469 | 166 | 81 | 48.8% | 1188 | 4969 |
| 3602 | P1 | 胜 | +8 | 512 | 181 | 73 | 40.3% | 1070 | 4984 |
| 3603 | P0 | 胜 | +13 | 512 | 218 | 69 | 31.7% | 1313 | 4687 |
| 3603 | P1 | **负** | −1 | 512 | 175 | 57 | 32.6% | 871 | 3968 |
| 3604 | P0 | 胜 | +27 | 504 | 213 | 86 | 40.4% | 1438 | 6859 |
| 3604 | P1 | 胜 | +4 | 512 | 233 | 91 | 39.1% | 1140 | 2984 |
| 3605 | P0 | 胜 | +3 | 512 | 224 | 102 | 45.5% | 1143 | 3125 |
| 3605 | P1 | **负** | −3 | 512 | 169 | 74 | 43.8% | 810 | 2468 |
| 3606 | P0 | 胜 | +6 | 512 | 219 | 104 | 47.5% | 1239 | 3578 |
| 3606 | P1 | 胜 | +7 | 512 | 171 | 61 | 35.7% | 969 | 3531 |
| 3607 | P0 | **负** | −3 | 512 | 185 | 95 | 51.4% | 974 | 3203 |
| 3607 | P1 | 胜 | +13 | 512 | 177 | 69 | 39.0% | 813 | 2437 |

按 `hit_rate` 中位数（40.5%）切两半：高组（n=8）平均得分 **0.75**、平均血量差 +13.88；
低组（n=8）平均得分 **0.88**、平均血量差 +10.88 —— **没有"命中越多越好"的正相关**（n 太小，
只能说没看到信号；也说明"命中率"不是这轮胜负的驱动量，§5 会再谈）。

> ⚠️ 这批日志产在 2026-10-28 16:00 之前的**老口径**（`hit` = 额外候选被选中）。
> 新口径（`hit` = 动作一致率、`hit_sel` = 老口径）从 §5 的那次改动开始，**下一批 A/B 才有**。

### 4.4 决策耗时：长尾**不是注入引入的**（两臂分开量）

问题（来自 16 局退出报告里的 `dec_max=25.34-26.59s`、`dec_slow=11-14`）：
这条长尾是注入带来的，还是本来就有？——**本来就有，而且基准臂更长一点。**
（`dec_slow` 的阈值 = **`AZAI_SLOW_DECISION_SEC`，默认 20 s**，不是 10 s；`dec_max` = 该局最慢的**单次决策**墙钟。）

每局 stderr 的 `决策耗时: n=… p50=… p95=… max=…`（收尾）与 `⏱️ 耗时累计: … dec_max=… dec_slow=…`
（每 25 决策）逐局抽出来（16 局 × 2 臂 = 32 个日志，共 7825 vs 7825 个决策）：

| 口径（每局统计后再取中位） | **A（注入）** | **B（基准）** |
|---|---|---|
| 决策 p50 | **1.80 s**（1.47-2.28） | **0.55 s**（0.50-0.58） |
| 决策 p95 | 16.87 s（7.98-19.61） | 19.20 s（18.73-20.00） |
| 决策 max | 26.89 s（22.72-55.42） | 26.72 s（23.19-**74.72**） |
| ≥20 s 的决策占比 | **3.03%**（237/7825，每局 7-24 次） | **4.05%**（317/7825，每局 4-30 次） |

三个读数点：

1. **注入只抬中位数，不造长尾**：p50 从 0.55 s → 1.80 s（+1.25 s ≈ 一次 oracle 查询 + 排队），
   但 ≥20 s 的比例 **A 3.03% 反而低于 B 4.05%**，B 的最慢一次 74.72 s 也比 A 的 55.42 s 更慢。
   ⇒ "没有注入是否也会超 10 s"的答案是**会**（B 的 p95 ≈ 19 s、max 24-75 s）。
2. **长尾是这套 BASE 配置的性质**，与注入无关：同一台机器上**部署口径的 canonical 配置**
   （`K=24 / MC_EVERY=5 / FLAT_ROUNDS=39000`，**完全不开注入**）实测
   `dec_max=8.2 s dec_slow=0`（`match_results/ai1_az_bridge_ai_seed8101.stderr.log` 的 `耗时累计`）。
   BASE 那套（`K=48 / FLAT_ROLLOUTS=64 / FLAT_ROUNDS=49152`）本身就重 2-3 倍，
   而 `FLAT_ROUNDS` 是**按"回合数"计的预算、不是按墙钟** ⇒ 局面越大（后期单位越多）同样预算越慢，
   这就是长尾的来源。
3. **对判据的影响**：本轮 16/16 都跑到 `verdict=engine`（bridge 读超时 `MVS_TIMEOUT_SECONDS`
   默认 300 s），所以**这批读数没被长尾污染**。但若正式平台对"每步思考时间"有硬限制，
   **需要先看的是 BASE 配置的 p95≈17-20 s，而不是注入**（注入只是把中位数从 0.55 s 推到 1.8 s）。

---

## 5. 每决策一条注入日志（2026-10-28 追加）

### 5.1 为什么加

原来的注入日志**只在命中时**打一行（`🎯 注入命中`），事后无法回答"亚军这一手我们到底改没改"，
也算不出一致率；而 `hit_sel` 只回答"它有没有被搜索选中"，对**重复候选（dup）**这一大类完全失明。

### 5.2 加了什么（`code/test_match/az_bridge_ai.py`，只加日志、不改决策）

每次**真的注入了候选**的决策都打一条（stderr；`AZAI_ORACLE_INJECT` 为空时一条都没有）：

```
[oracle] inject round=<r> action=<...> dup=<0/1|NA> new=<0/1|NA> chosen=<...> hit=<0/1>
```

* `action` = 亚军这一手：`()` = HOLD、`(t,a0,a1)` = 单 op、`((t,a0,a1),(t,a0,a1))` = 多 op；
* `chosen` = **我们最终真的发出去**的那串 op（同一格式；已过执行层过滤/闸门）⇒ `hit` 直接就是
  "我们改没改它"；`hit = int(chosen == action)`；
* `dup=1` = 亚军这一手与**已有候选重复**（没新增候选），`new=1` = **新增候选**，两者互斥；
  `NA` 只出现在 `AZAI_FORCE_UPG` 早退（搜索没跑）这种诊断模式里。

真实样例（本地 55 回合冒烟，`training_history/p1_logs/oracle_inject_log_smoke.log`）：

```
[oracle] inject round=0  action=()                 dup=1 new=0 chosen=()                 hit=1
[oracle] inject round=18 action=(11,4,9)           dup=0 new=1 chosen=(11,4,9)           hit=1
[oracle] inject round=31 action=(13,0,-1)          dup=1 new=0 chosen=(13,0,-1)          hit=1
[oracle] inject round=32 action=((13,0,-1),(11,4,16)) dup=0 new=1 chosen=(21,5,14)       hit=0
[oracle] inject round=33 action=((13,0,-1),(11,5,15)) dup=0 new=1 chosen=((13,0,-1),(11,5,15)) hit=1
```

汇总行同时报（**分母都写在括号里**）：

```
注入累计: asks=50 ok=50 checked=7 inject=50 new=15 dup=35 dup_hold=34 dup_nonhold=1
          hit=49 hit_new=14 hit_dup=35 hit_sel=14 illegal=0 timeout=0 dead=0
          avg_ms=582 p50_ms=359 p95_ms=1203 max_ms=1234
          hit_rate=326.7%(hit/new) hit_rate_asks=98.0%(hit/asks)
          hit_new_rate=93.3%(hit_new/new) hit_dup_rate=100.0%(hit_dup/dup) hit_sel_rate=93.3%(hit_sel/new)
```

⚠️ **`hit_rate`（hit/new）在新口径下不是比率、可以 >100%**：`hit` 含 `hit_dup`（亚军说"空过"、
我们也空过——这类"一致"不带来任何新动作），而分母 `new` 只数**新增候选**（这局 15）。
要当比率读的是 `hit/asks`、`hit_new/new`、`hit_dup/dup`。这条也写进了代码注释，
免得后人把它当胜率一样的百分比看。

**"默认行为不变"的实测**（老版 = commit `475894a`、未含新日志；新版 = 含新日志；同一 seed 7、canonical 配置、**注入关**）：

```
老版: RESULT seed=7 ... rounds=200 base_hp=27,33 coins=39,167 illegal=0
新版: RESULT seed=7 ... rounds=200 base_hp=27,33 coins=39,167 illegal=0      ← 逐字段相同
老版: illegal-summary ... checked_ops=78 holds=p0:186,p1:167 mismatch=0
新版: illegal-summary ... checked_ops=78 holds=p0:186,p1:167 mismatch=0      ← op 序列一模一样
```

并且新版的 stderr 里 `[oracle] inject round=` 行数 = **0**（注入没开 ⇒ 新日志零输出）。
日志：`training_history/p1_logs/logging_equivalence2.log`、`runs/20261028_160834_p1_logging_equiv_old2`、
`runs/20261028_160014_p1_logging_equiv_new`。
### 5.3 "dup 里有多少是 HOLD" —— 核实结果

两个来源，结论一致：**dup 几乎全是 HOLD**。

1. **旧 pilot 的逐回合 verbose 日志**（`match_results/oracle_inject/inject/seed3000_p0_ai0.stderr.log`
   等 3 局，每回合都打了亚军的 ops）：**1461 个决策里 1138 次是 HOLD = 77.9%**
   （3 局分别 88.7% / 73.0% / 71.2%；多 op 候选只占 5.8%）；
2. **新日志的 55 回合冒烟**：`dup=35` 里 **`dup_hold=34`、`dup_nonhold=1`**（97%）；
   40 回合那版的累计行（前 25 决策）是 `dup=18 dup_hold=18 dup_nonhold=0`。

⇒ "注入的增量信息"要按 `new` 那一档算：A/B 里 `new = 3051/8027 = 38.0%`，其余 62% 的 dup 里
绝大多数只是"亚军也空过"。**这也解释了 §4.3 那个"命中率与胜负不相关"**：老口径 `hit` 里
有一部分是"它在 HOLD、我们也 HOLD"这种天然一致。

### 5.4 以后怎么用

* **任意 A/B 都能直接统计一致率**：A 臂 stderr 里 `grep '\[oracle\] inject round='` ⇒
  `hit/总数` 就是动作一致率，`new=1` 里的一致率（`hit_new/new`）才是"亚军真的改了我们一手"的比例；
* 每局的收尾 `注入统计:` 行已经带 `dup_hold / dup_nonhold / hit_new / hit_dup / hit_sel` 与 5 个比率；
* `code/test_match/analyze_oracle_inject_ab.py` 已适配：见到 `hit_sel` 就按新口径解释，
  见不到就提示"这批日志是老口径"。

## 6. 结论

### 6.1 P1：修好了（复现路径上 0 非法，且"确实拦住了"）

1. **同一组合、同一 seed、同一回合，修复前判负 → 修复后跑满**：
   `8103`（我方 p0）从 `verdict=illegal_strict @ r61` 变成 `verdict=engine 338 回合 illegal=0`；
   `8103r`（我方 p1）从 `@ r244` 判负变成 `verdict=engine 401 回合 illegal=0`。
   而且 stderr 里能看到闸门**就在那两个回合、那两个 op 上**把它们拦下来的原文 ⇒ 不是"没触发"，
   是"拦住了"。
2. **组合矩阵 4/4 干净**：注入 × {`NO_DOWNGRADE`, `BAN_OPS=13`, `LIGHT_MIN_COINS=180`,
   `BUILD_MIN_COINS=90`} 各 1 局，全部 `verdict=engine`、裁判侧 `illegal=0`、AI 侧 `illegal_total=0`；
   前两个组合闸门真的丢了 14 / 29 次半块。
3. **默认行为不变**：seed 8101 canonical 局与旧代码的 `RESULT` 与 `illegal-summary` **逐字段相同**。
4. **顺带发现**：同一机制的第二个形态是**升级**（`BAN_OPS=13` 那局拦下 `upgrade_no_gold(60,55)`），
   不只是建塔；`docs/illegal_op_strict_mode_20261028.md` §7 第 1 条的"推测同构"现在有实例。
5. **代价**：闸门只在"真的删过 op"时跑（约 24 µs/op），默认配置下**一次都不进**。

**残留风险（诚实写出来）**：闸门只覆盖"事后删 op"这一类。`AZAI_FORCE_UPG=1`（诊断模式）和
`mc_last_explicit`（MC 显式升级）两条路径在过滤**之前**就 `return`，仍只有 `decide_final` 记录、
不拦截；`FORCE_UPG=1` 的 `(x,y)` 形式本来就故意发非法 op（F3）。要用它们做实验时得自己看着。

### 6.2 P2：注入值得继续（81.2% vs 50%，成本已量化）

**判断：值得继续，但"成本"和"增量信息"要一起抓。**

依据（按重要性排）：

1. **效应大且不在座位上**：16 局 `13W-0D-3L = 81.2% ± 9.8%`，两个座位分别 87.5% / 75.0%，
   5 个 seed 2-0、3 个 seed 1-1、0 个 seed 0-2；符号检验 p ≈ 0.021。
   3 场败局只输 1-3 血，13 场胜局平均 +16.4 血 ⇒ 像"真实力差"，不像硬币。
2. **干净**：16/16 `verdict=engine`、`illegal=0`、`mismatch=0`、`timeout=0`、`dead=0`
   （注入全程活着 ⇒ 读数不是被"早退"稀释的）。
3. **成本可量化、可接受**：单次查询均值 1.14 s × 512 决策 ≈ **+9.7 min/局**，单局 44-58 min
   ⇒ 约 **+20% 墙钟**；16 局 8 并行 104 min 跑完（预算按 `~1.1 h / 8 局` 估）。
4. **但"增量信息"只有 38%**：`new = 3051 / 8027`，剩下 62% 的 dup 里 97% 只是"亚军也 HOLD"
   （§5.3）。所以 ① `hit_sel/new = 41.6%` 要读成"新增候选里有 41.6% 被搜索选中"，
   不是"41.6% 的回合被它改主意"；② 老口径 `hit` 与胜负**没有**正相关（高组 0.75 vs 低组 0.88），
   别把命中率当强度指标。

**下一步建议（按性价比）**：

| # | 建议 | 理由 |
|---|---|---|
| 1 | 把 `AZAI_ORACLE_TIMEOUT` 默认从 2.5 s 提到 **≥10 s** | 本批实测单次 max **6859 ms**；2.5 s 下有 5/8 局丢了注入（§4.1.1） |
| 2 | 用新的**每决策日志**在正式配置上量 `dup_hold / new / hit_new` | 这批 A/B 只有老口径；有它才算得出"一致率/真增量" |
| 3 | 省掉明显无用的查询：`dup_hold` 占 62% ⇒ 若能**本地**判定"我们本来就 HOLD"，可少一次 1.1 s 查询 | 直接把 +20% 成本砍掉一大半（但要小心：HOLD 的一致本身也算一种确认） |
| 4 | 换外部标尺再验一次：A（注入）vs **亚军本体** | 现在测的是"相对自己变强多少"，没回答"能不能赢亚军" |
| 5 | 多固定几个 `AZAI_FLAT_SEED` 复现（本轮只 53） | 单一种子下的效应，别当成与种子无关的常数 |

**一句话**：P1 的洞已经堵住（且有"拦住"的证据），P2 的注入在自对弈里是 **+31 pp** 级别的真增益、
成本 +20% 墙钟；继续做，但下一步要补上"每决策一致率/增量"和"对亚军本体的胜率"这两把尺子。

---

## 7. 复现清单（全部有落盘日志）

```bash
# P1：确定性探针（8/8 PASS，秒级）
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/probe_filter_gate.py

# P1：复现路径（4 批：故障注入 / canonical 4 局 / 高风险 2 局 / 旧行为对照）
bash code/test_match/_illegal_smoke.sh

# P1：组合矩阵（4 局并行；1 局/组合）
bash code/run_logged.sh filter_gate_matrix_final bash code/test_match/_filter_gate_matrix.sh

# P2：16 局 A/B（8 seeds × 2 座位，jobs 8）
bash code/run_logged.sh oracle_inject_ab bash code/test_match/_oracle_inject_ab.sh
#   续跑：同一条命令原样再敲（--resume 会跳过已 verdict=engine 的局）

# P2：读数汇总
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_oracle_inject_ab.py \
    --out-dir match_results/oracle_inject_ab/ab16
```

| 工件 | 路径 |
|---|---|
| P2 主日志 | `training_history/runs/20261028_132806_oracle_inject_ab/output.log` |
| P2 逐局日志 / 进度 / stderr | `match_results/oracle_inject_ab/ab16/{seed*_p*.log,_progress.json,stderr/}` |
| P2 读数 | `training_history/p1_logs/ab16_analysis.log` |
| P2 作废的两版（证据） | `training_history/p1_logs/ab16_attempt1_stderr/`（2.5 s 超时丢注入）、`…/ab16_attempt2_noinject/`（POSIX 路径 ⇒ 静默不注入） |
| P1 canonical / 高风险批 | `training_history/runs/20261028_1117*_illegal_*`、`…_120847_illegal_strict_highrisk`、`…_123515_illegal_cold_control` |
| P1 组合矩阵 | `training_history/runs/20261028_114004_filter_gate_matrix_final/output.log`；旧版留档 `match_results/filter_gate_matrix_v0/` |
| P1 高风险批 stderr | `training_history/p1_logs/smoke_batch2_stderr/` |
| 新日志冒烟 | `training_history/p1_logs/oracle_inject_log_smoke.log`、`match_results/ai0_az_bridge_ai_seed3600_smoke.stderr.log` |
| 代码 commit | `475894a`（P1 + 工具）、本报告与每决策日志见 `git log` |

## 8. 没做 / 没覆盖的（诚实清单）

1. **只跑了一个 `AZAI_FLAT_SEED=53`**（按任务书约定：本轮先看单一种子下的效应）。
2. **A/B 是自对弈消融**（A、B 都是我们的 AI），没有对亚军本体、也没有对其它对手。
3. **新口径的 `hit`（动作一致率）这批没有**：16 局日志产在新日志之前，只有老口径 `hit_sel`；
   `dup_hold` 的核实靠旧 pilot 的 verbose 日志（3 局）+ 一次 55 回合冒烟（1 局），不是 16 局。
4. **闸门只覆盖事后删 op**：`FORCE_UPG` / `mc_last_explicit` 仍是"记录不拦"（见 §6.1 残留风险）。
5. **`AZAI_ORACLE_TIMEOUT` 动了**（2.5→15 s）：唯一一处偏离任务书配置，理由与证据见 §4.1.1。
6. **没测"注入 + 过滤开关"在强度上的影响**：那是 P1 的雷区，本轮刻意不开（矩阵只验合法性）。
7. **没有**做跨局的 oracle 复用/缓存（协议上会话有状态，不能复用）。
8. **P2 期间没跑其它实验**（除了最后 1 局 55 回合的日志冒烟）：`--jobs 8` 已经把机器吃满。
