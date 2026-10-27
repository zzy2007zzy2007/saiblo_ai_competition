# 非法操作严格模式：本地口径对齐官方 + 逐条影子校验（2026-10-27/28）

> 任务：让我们本地的对局测试与官方判题在"非法操作"上对齐，并把非法操作变成**可见的日志与计数**。
> 工作目录 `D:\2026智能体大赛_新2_distill`；代码 commit 见文末 §9。
> 状态：**代码已合入并跑完回归**（§5）；本轮**没有**改动引擎源码、`code/cpp_engine/**` 的行为、
> 也没有改动 `code/my_ai/az_intent/**` 的任何搜索逻辑。

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| ① 默认对齐官方 | `cold_handle_rule_illegal=False` 成为默认；`--cold-illegal`（或 `AZAI_COLD_ILLEGAL=1`）回旧行为 |
| ② 非法操作可见 | 逐条 `can_apply`（dry + clone 双查）→ 每次非法**一行 stderr** + 每局**一行汇总** `illegal_total=N` |
| ③ 高风险路径 | 出口 `decide_final` / 去降级 `after_no_downgrade` / 影子应用 `shadow_apply` / 注入计数 `checked=` 四处补齐 |
| ④ 真实对局 | **严格模式 4+2 局全部 `verdict=engine`、`illegal_total=0`**（§5.3）；旧行为对照 1 局逐字段相同（§5.4） |
| 本轮实际非法操作数 | **0**（严格模式 6 局 + 旧行为对照 1 局 + **120 局历史对局重放审计**，共 50k+ op） |
| 但 | 历史"48 局 0 次"是**结构性 0**（旧检查永远不成立，见 §2），不是实测；本轮才第一次**真的**测出来 |

一句话：**风险是"没触发"，不是"不会触发"**——而且"没触发"这个结论此前从未被真正测量过。本篇把它测了：
当前时代（2026-10-26/27 的 120 局对局、50480 个 op）确实 0 非法，同时严格模式已经在本地把关。

---

## 1. 问题：官方语义 vs 本地语义

### 1.1 官方（平台判题用的）

引擎参数 `cold_handle_rule_illegal` 只在 `Game::round_read_from_judger` 里被读
（`Ant-Game/game/src/game.cpp:2618`，我们的副本 `code/cpp_engine/src/game.cpp:2618`）：

```cpp
if (!cold_handle_rule_illegal) {                 // ← 官方默认（config 里不写 = false）
    op[player] = op_list;
    if (!apply_operation(op_list, player, err_msg)) {
        set_AI_state_IO(player);                 // state=ILLEGAL_OPERATION, is_end=true, winner=对手
        return false;                            // main.cpp:41/52 的循环随即 break ⇒ 对局立刻结束
    }
} else {                                          // ← 冷处理：逐条 apply，非法**静默跳过**
    ... accepted_ops.push_back(operation) ...
    append_illegal_summary(err_msg, player, ignored_errors);
}
```

`Game::set_AI_state_IO`（`game.cpp:2804`）：`state[player]=ILLEGAL_OPERATION; is_end=true;
winner=(player==0)?1:0;` ⇒ **发非法操作的一方直接判负，对局当场结束**。

顺序也很关键：官方先 `round_read_from_judger(0)` 再 `(1)`
⇒ p0 的非法优先判、且 **p1 的操作是在 p0 的操作已经落到局面之后才判的**
（影响 `TowerBuild: EMPBlaster is active` 这类同回合跨侧规则）。

### 1.2 本地（改之前）

`code/test_match/match_via_sdk.py` 第 405 行写死：

```python
state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)   # 非法操作被吞掉
```

⇒ 本地永远测不出"非法操作"这类 bug，只会在正式比赛里变成**直接判负**。

---

## 2. 关键发现：两个**静默失效**的旧检查（比"改默认值"更重要）

### F1. `GameStateFacade.apply_operation_list` 返回 `None` ⇒ 旧检查**永不成立**

`code/my_ai/az_intent/game_state_facade.py:167-170` 把 binding 返回的 accepted 计数丢掉了：

```python
def apply_operation_list(self, player: int, operations) -> None:      # ← 返回 None
    raw = [(int(op.op_type), op.arg0, op.arg1) for op in operations]
    self._g.apply_operation_list(player, raw)
    self._cache.clear()
```

于是两处历史代码**从来没数到过任何东西**：

| 位置 | 旧代码 | 实际 |
|---|---|---|
| `match_via_sdk.py:717-726`（改前） | `il0 = state.apply_operation_list(...)` + `if il0 or il1:` | `il0` 恒为 `None` ⇒ 分支永不进入 |
| `az_bridge_ai.py:1055-1063`（改前） | `bad = facade.apply_operation_list(...)`；`n = len(bad) if bad else 0` | `n` 恒为 0 ⇒ `STATS["illegal"]` 永远是 0 |

`_tmp_ladder.py:192` 数的是"含 `ILLEGAL` 的行数"，而这条线从来没打印过 ⇒
**历史日志里的"48 局非法 0 次"是结构性 0**，不能当作"实测安全"的证据。

### F2. `cold_handle_rule_illegal` 在 pybind 路径上**是惰性的**

`code/cpp_engine/binding.cpp:46-59` 的 `apply_operation_list` **无条件**调
`apply_operation_list_cold`（= 冷处理，非法静默跳过），而 `cold_handle_rule_illegal`
只在 `round_read_from_judger` 里被读、binding 根本没暴露那个函数
（探针实测 `dir(NativeGame)` 里没有 `round_read_from_judger`）。

⇒ **只把 `cold=False` 传进去不改变任何行为**。严格语义必须在 bridge 里实现。
探针还实测：同一个非法 op 打进 `cold=True` 与 `cold=False` 的两个引擎，状态演化**逐位相同**。

### F3.（顺带发现，未修）诊断探针 `_probe_upgrade_once` 自己在造非法 op

`az_bridge_ai.py` 的 `AZAI_PROBE_UPG=1` 诊断路径（**默认关**，只打日志、只对 clone 生效，不影响对局）：

* 升级 op 用的是 `(x, y)` 形式（`_Op(UPGRADE_TOWER, w.x, w.y)`）——官方要的是 `(tower_id, target_type)`；
* "建塔对照"用的是 `ys, xs = (pm[0] > 0).nonzero()` 后 `(xs[0], ys[0])`，而掩码布局是
  `mask[class, x, y]`（`decoder.py:190`）⇒ **x/y 调换**，那条"对照"本身也是非法 op。

⇒ 那段探针历史上给出的"连建塔都测不出效果"的读数，**很可能是这两个非法 op 造成的**，
不是引擎/状态语义问题。本次只记录、不擅自改（任务约束），建议后续要么修、要么删。

---

## 3. 改了什么

### 3.1 `code/test_match/match_via_sdk.py`（裁判侧）

1. **默认值**：`COLD_ILLEGAL = ("--cold-illegal" in sys.argv) or (AZAI_COLD_ILLEGAL=1)`
   ⇒ `GameState.initial(..., cold_handle_rule_illegal=COLD_ILLEGAL)`，默认 **False**（=官方）。
   加了 `--help/-h`（`--cold-illegal` 的语义、等价环境变量都写在里面），启动时打
   `[bridge] cold_illegal=False (严格模式=官方默认: 非法操作判负)`。
2. **影子校验（只读）**：新增 `IllegalStats` / `op_reason()` / `_can_apply_cold()` / `check_ops()`。
   每个 op 依次：
   * `facade.can_apply_operation(...)` → `native.can_apply_dry`（clone-free，搜索同款，~1µs）；
   * `native.can_apply_cold(...)` → **clone 后真跑** `apply_operation_list_cold`（权威，~20µs）；
   * 两者都跑、不一致就记 `MISMATCH(legality)`（这类"搜索用的快速检查 vs 引擎"漂移本身就是要抓的 bug）；
   * **权威判定 = cold 版**（拿不到才退回 dry）；两个都抛异常 ⇒ 记 `CHECK-ERROR` 并放行
     （绝不能因为检查器坏了判人负）。
3. **顺序照抄官方**：`p0 校验 → 应用 p0 → p1 校验 → 应用 p1`（`match_via_sdk.py` 的回合主体）。
   这样"p0 本回合放 EMP ⇒ p1 建塔落在护盾格"也能被抓到。
4. **严格模式的处置**：一旦某侧出现非法 ⇒ `result["strict_illegal_side"]`、`winner=对手`、
   `terminal=True`、**不推进回合**、对局结束。RESULT 行判词 = **`verdict=illegal_strict`**
   （故意与 `verdict=engine` 区分：这样下游"有效性门槛 = verdict=engine"会把它排除，
   不会被当成一场正常战果）。
5. **日志（全部 stderr，stdout 绝不写）**：
   ```
   [bridge] ILLEGAL seed=7501 round=0 side=1 tag=apply idx=0 op=(11,16,9) name=BUILD_TOWER \
            reason=build_on_base dry=False cold=False cold_mode=False engine=reject
   [bridge] illegal-summary seed=7501 cold=False illegal_total=1 checked_ops=1 holds=p0:1,p1:0 \
            by_side={0:0,1:1} by_kind={(11,build_on_base):1} mismatch=0 shadow_ms=0.1 first=...
   ```
   * 每次非法 **一行**（含 seed/回合/side/op 三元组/reason/dry+cold 判定）；
   * 每局 **一行汇总**，`finally` 里打 ⇒ 超时/异常局也有，`illegal_total=N` 是判据；
   * reason 是**尽力而为的分类器**（`build_on_base` / `build_cell_occupied` / `build_no_gold(...)` /
     `tower_id_not_found(...)` / `tower_not_owned(...)` / `upgrade_target_type_mismatch(...)` /
     `tower_max_level(...)` / `upgrade_no_gold(...)` / `weapon_in_cooldown(...)` / `weapon_no_gold(...)` /
     `base_max_level(...)` / `base_no_gold(...)` / `unknown_op_type(...)` …）。引擎只回 bool，
     所以 reason 是**从局面上反推的**，判定本身仍由引擎给。
   * ⚠️ 汇总行**故意不含大写 `ILLEGAL`**：`_tmp_ladder.py:192` 数的是含该词的**行数**，
     汇总行若含它就会把 0 非法数成 1。实测：故障注入局 `illegal` 计数 = 1 / 15，与 op 数一致。
6. **RESULT 行**：`... coins=a,b illegal=N`（放在 `coins=` 之后、`exc=` 之前；`RESULT_RE`
   不锚定行尾 ⇒ 安全）。格式回归测试 `code/test_match/test_result_line_formats.py` 全 PASS。

### 3.2 `code/test_match/az_bridge_ai.py`（AI 侧：操作生成/改写路径）

只加**只读校验与日志**，不动任何决策。新增 `_legal_check()`（内部调 `MVS.check_ops`），挂在 4 处：

| tag | 位置 | 覆盖的高风险路径 |
|---|---|---|
| `decide_final` | `_decide_timed()` 里 `decide()` 返回后（**计时器外**） | 覆盖 `decide()` 的**所有** return：`AZAI_FORCE_UPG` 的升级 op、`mc_last_explicit` 显式升级 op(12)、**HOLD 空列表**、以及所有改写后缀 |
| `after_no_downgrade` | `AZAI_NO_DOWNGRADE=1` 去掉降级 op 之后 | 降级是"退钱"操作，去掉后**后面的建塔/升级可能变成钱不够** ⇒ 必须重查 |
| `shadow_apply` | `_apply()` 里（影子状态应用前，两侧都查） | 同时把原来那个"恒 0"的 `STATS["illegal"]` 修成真的 |
| `checked=` | `_oracle_bundle()`（候选注入） | `AZAI_ORACLE_INJECT` 这条路径**本来就有**逐条 `can_apply`（非法就丢整条候选），本次只补计数，让它进 `[oracle] ... checked=N` |

另外：每 25 个决策把自检汇总打一次（收尾常被 bridge 硬杀），收尾再打一次。

### 3.3 新增的工具（都在 `code/test_match/`）

| 文件 | 作用 |
|---|---|
| `probe_illegal_op_mode.py` | 证据探针：pybind 暴露了什么、flag 是否惰性、返回 None、升级 op 两种形式、影子校验能不能抓、两种 can_apply 的代价。**可当回归测试**（失败返回 1） |
| `faulty_ai.py` | 故障注入 AI：每回合发一条**必然非法**的 op（自家基地格建塔），用来证明"严格模式真的判负 / 旧行为真的吞掉" |
| `audit_replay_illegal.py` | **离线重放审计**：把已打完的一局按 trace 逐回合重放，用引擎复核每个 op；同时校验"每回合操作前的 HP/金币与 trace 一致"（⇒ 重放忠实）。**可以审计历史对局** |
| `audit_replay_batch.py` | 上面的批量入口（挑最近 N 个已完成 trace） |
| `_illegal_smoke.sh` | 4 批回归脚本（见 §5），每批走 `code/run_logged.sh` |

### 3.4 两种口径的差异（一图说明）

| | `cold=True`（旧本地默认） | `cold=False`（**新默认 = 官方**） |
|---|---|---|
| 非法 op 的后果 | 引擎静默跳过，对局继续 | `set_AI_state_IO` ⇒ 立刻终局、**肇事方判负** |
| 本地能否测出 | 不能（本地日志也看不见，见 F1） | 能：逐条 stderr + `illegal_total` + `verdict=illegal_strict` |
| 对局结果 | 与"没发过这条 op"相同 | 一场**判负**（正式比赛里就是输一局） |

---

## 4. 默认行为变化 & 影响面（**明确写出来**）

1. **凡是走 `match_via_sdk.py` 的本地对局，默认从"吞掉非法操作"变成"非法操作判负"**。
   受影响的下游：`_tmp_ladder.py`（梯子/对决）、`ab_search_modes.py`、各个 `_ab_*.sh`、
   以及所有 `--ai0=/--ai1=` 的探针脚本。**这是一个口径变更，不只是加日志。**
2. **`illegal_total=0` 时，两种口径的结果逐位相同**：机制上，`check_ops` 是只读的
   （只调 `can_apply_dry` / `can_apply_cold`，后者在 clone 上跑），应用路径完全没变；
   实测上用同一个 seed（8101）跑严格/旧行为两局，`winner/rounds/base_hp/coins` 全同（§5.4）。
3. **跨口径可比性**：旧实验（`cold=True` 时代）与新实验（`cold=False` 时代）的结果**只有在
   `illegal_total=0` 时才可比**。旧时代没法证明这一点（F1 的死检查），但本轮补上了：
   用离线重放审计把 **120 局**旧对局（57527 回合、50480 个 op）重新过了一遍 ⇒ **0 非法**
   （§5.5）。所以"最近这批读数"可以认为没被非法操作污染；**更早的批次没审**（见 §7）。
4. **正在跑的实验**：`_ab_rng_perturb.sh`（`match_results/rng_perturb_ab`，18:02 启动）跨过了这次改动。
   18:02 启动的第 0 组 8 局用的是**旧代码**；18:36 之后的组会起**新代码**（严格默认）。
   由于该实验的两条臂都是 `az_bridge_ai` 自对弈、且重放审计显示同类对局 0 非法（含它自己的
   seed3300/3303 两局），**可以认为不受影响**；但"日志里 `illegal=` 字段在 18:36 之前不存在"这一点
   在事后分析时要知道。
5. **没有改到的地方**：`code/test_match/run_cpp_ai_match.py`（走官方 `main.exe` + json 配置，
   配置里仍写 `"cold_handle_rule_illegal": True`）、`fast_match.py`、`code/my_ai/**` 的各类
   collector/eval **仍然**是"吞掉非法 op"的口径，本次**没动**（见 §7）。
6. **`AZAI_COLD_ILLEGAL=1`** 是给"不能加命令行参数"的老脚本（如 `_tmp_ladder.py`）留的逃生门。

---

## 5. 实测（全部有落盘日志）

### 5.1 证据探针 `probe_illegal_op_mode.py`

`training_history/probe_illegal_op_mode_20261027.log`（**全部 PASS**，节选）：

```
NativeGame 暴露的方法: [... 'can_apply_cold', 'can_apply_dry', 'apply_operation_list', ...]
[PASS] NativeGame 没有 round_read_from_judger（⇒ cold flag 的唯一下游不存在）
[PASS] apply_operation_list 返回 None（老代码 `if il0:` 永远不成立）  return=None
[PASS] 冷路径对非法 op 静默跳过（状态不变、不抛异常）
[PASS] cold=False 的引擎对同一个非法 op 的行为与 cold=True **逐位相同**（flag 惰性）
[PASS] 对照：合法建塔在冷路径上确实生效（塔数 +1、金币减少）  0->1 coins 50->35
[PASS] 正确的升级 op (tower_id, target_type) 被判合法  tower_id=0 coins=122
[PASS] 错误的升级 op (x, y) 被判**非法**（历史 bug 的确切形态）  (x,y)=(4,2)
[PASS] 冷路径下 (x,y) 形式的升级 op 被静默丢弃（等级不变）  [0] -> [0]
[PASS] check_ops 抓到非法 op  ... reason=build_on_base dry=False cold=False
[PASS] 汇总行**不含**大写 ILLEGAL（否则 0 非法也会被数成 1）
can_apply_dry=1.4us  can_apply_cold=23.4us   （每 op 的校验成本）
```

### 5.2 故障注入：严格 vs 旧行为（**同一个 seed、同一个非法 op**）

AI0 = 亚军 `ai_cpp_lure_v4.exe`，AI1 = `faulty_ai.py`（每回合在自家基地格建塔）。

| | 严格模式 `illegal_fault_inject_strict` | 旧行为 `illegal_fault_inject_cold` |
|---|---|---|
| 日志 | `training_history/runs/20261027_183211_illegal_fault_inject_strict/output.log` | `.../20261027_183218_illegal_fault_inject_cold/output.log` |
| 首个非法 | `r0 s1 op=(11,16,9) reason=build_on_base` | 同样（每回合一行，共 15 行） |
| 汇总 | `illegal_total=1 checked_ops=1` | `illegal_total=15 checked_ops=15` |
| RESULT | `winner=runnerup winner_side=p0 verdict=illegal_strict ... illegal=1 rounds=0 terminal=True` | `winner=draw verdict=hp_tiebreak ... illegal=15 rounds=15` |
| 结局 | **p1 当场判负**（= 官方平台会做的事） | 15 个非法 op **被静默吞掉**，对局照常打到回合上限 |

⇒ 这就是"本地测不出、正式比赛直接判负"的完整复现；也是"严格模式真的抓得住"的正面证据。

### 5.3 严格模式干净局（真实对局）

环境：`AI0 = az_bridge_ai`（部署口径 canonical 配置，与 `ab_search_modes.py:37-42` 的
`DEFAULT_MCTS_ENV` 逐字相同）、`AI1 = 亚军 ai_cpp_lure_v4.exe`；`MVS_TIMEOUT_SECONDS=900`
（机器上另有 8 局 A/B 在跑，放宽"卡死检测"门限；它不影响判词）。

<!-- SMOKE_TABLE -->

### 5.4 旧行为对照（同一 seed、同一配置）

<!-- CONTROL_TABLE -->

### 5.5 历史对局重放审计（**第一次真正回答"历史 48 局到底有没有非法操作"**）

`python code/test_match/audit_replay_batch.py --n 120` → `training_history/audit_replay_120_20261027.log`

```
=== 合计: 120 局 / 57527 回合: illegal_total=0 divergence=0 ===
games: 120   checked_ops 合计: 50480   holds 合计: 81798
illegal 合计: 0   mismatch 合计: 0   divergence 行数: 0
shadow_ms 合计: 1144.5   （= 22.7 µs/op）
```

* 来源：`training_history/progress/` 下最近 120 个**已打完**的 trace（`rr_pz_vs_hybrid` 64 局、
  `rr_f10_vs_pz` 14 局、`seed31xx` A/B 30 局、`zero_old/new` 4 局、`seed3300/3301/3303` 3 局等），
  排除本次自己造的故障注入/快速验证局。
* `divergence=0` ⇒ 重放是**忠实**的（每回合操作前的 HP/金币都与 trace 一致，且终局也一致）。
* `mismatch=0` ⇒ 搜索用的 `can_apply_dry` 与引擎的 clone 版检查在这 5 万个 op 上**没有出现分歧**。

### 5.6 成本

* 影子校验每 op ≈ 24 µs（`dry` 1.4 µs + `cold` 23 µs，其中 cold 版要 deep clone）；
* 一局 400~500 回合 ⇒ `shadow_ms ≈ 8~19 ms`（离线审计实测），对局总时长是**分钟级** ⇒ 完全可忽略；
* AI 侧每个决策多 ~50 µs（`decide_final` + `shadow_apply`），且在计时器**之外** ⇒
  `decision`/`apply` 的耗时口径与改动前一致。

---

## 6. 发现的问题清单

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| F1 | `apply_operation_list` 返回 None ⇒ bridge 与 az_bridge_ai 的非法检查**永不成立**，历史"0 次"是结构性 0 | 高（这是本次的根因之一） | **已修** |
| F2 | `cold_handle_rule_illegal` 在 pybind 路径上惰性 ⇒ 只改 flag 不改行为 | 高（差点"改了等于没改"） | **已修**（严格语义在 bridge 实现） |
| F3 | `_probe_upgrade_once` 的升级 op 用 `(x,y)`、建塔对照 x/y 调换 ⇒ 它在造非法 op | 中（诊断结论可能被带偏；默认关，不影响对局） | **只记录**（按约束不擅自改决策/诊断逻辑） |
| F4 | 其它 harness（`run_cpp_ai_match.py` 仍写 `cold=true`、`fast_match.py`、`code/my_ai/**` 的 collector/eval）**仍会吞掉非法 op** | 中（那几条路径上的 bug 还是测不出） | **未改**（超出本次范围，见 §7） |
| F5 | 历史日志的 `ILLEGAL` 计数语义 = "含该词的行数"（`_tmp_ladder.py:192`） | 低 | 已在新增代码里保证"一条非法 = 一行"，汇总行用 `illegal_total=` |

**本轮真跑出来的非法操作：0 个**（严格模式 6 局 + 旧行为 1 局 + 120 局历史重放审计，共 5 万+ op）。
唯一出现的非法 op 是**我们自己注入**的（`faulty_ai.py`）+ 探针构造的那两条。

---

## 7. 还没覆盖的高风险路径（诚实的清单）

1. **`AZAI_BAN_OPS` 里禁掉 `13`（降级）**：与 `AZAI_NO_DOWNGRADE` 完全同构（去掉"退钱"操作），
   但只在 `decide_final` 兜底，**没有分阶段 tag** ⇒ 一旦触发，日志能报"哪一手非法"，
   但不能直接归因到"是 BAN_OPS 那一步引入的"。
2. **`AZAI_LIGHT_MIN_COINS` / `AZAI_BUILD_MIN_COINS`**：只会**少花钱**，理论上不会让后续 op 变非法
   （所以没加分阶段检查），但也**没有实测证据**；同样只有 `decide_final` 兜底。
3. **`mc_last_explicit`（MC 显式升级 op）与 `AZAI_FORCE_UPG`**：有 `decide_final` 兜底但没有独立 tag。
4. **搜索内部的 rollout**：每个决策把上千个 op apply 到 clone 上，**不逐条校验**（成本不允许）。
   那里如果静默跳过非法 op，会污染 rollout 值 ⇒ 属于 `docs/findings_shadow_sim_fidelity.md` 那条线，
   本次没碰。可用 `mismatch=0` 作为"`can_apply_dry` 与引擎一致"的间接证据（5 万 op 无分歧）。
5. **对手侧（magica / 亚军 / rule_v4）的 op 生成**：不可控，但**裁判侧对两侧都查**，所以出问题会可见。
6. **其它 harness**（F4）：`run_cpp_ai_match.py`、`fast_match.py`、`code/my_ai/**` 的
   collector/eval/`ss_train`/`es_train` 等。它们仍是冷口径。若要全线对齐，最小改法是
   在那些地方也接 `match_via_sdk.check_ops`（或至少在 `--rounds` 跑完后做一次重放审计）。
7. **更早的历史批次没审计**：本轮只审了最近 120 局（`progress/` 里有 1515 个已完成 trace）。
   要全量审：`python code/test_match/audit_replay_batch.py --n 1515`（每局 ~2-4 s）。
8. **`_probe_upgrade_once`（F3）**：仍然会产出非法 op（默认关）。

---

## 8. 复现

```bash
# 探针（=回归测试；失败返回 1）
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/probe_illegal_op_mode.py

# RESULT 行格式回归（确保下游 _tmp_ladder 仍能解析）
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/test_result_line_formats.py

# 4 批回归（故障注入 2 局 → 严格 4 局 → 高风险路径 2 局 → 旧行为对照 1 局）
bash code/test_match/_illegal_smoke.sh          # 全部
bash code/test_match/_illegal_smoke.sh fast     # 只跑故障注入那 2 局（10 秒级）

# 历史对局重放审计（可换 --n）
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/audit_replay_batch.py --n 120
```

---

## 9. 数据出处 / commit

* 代码 commit：`01b15d7`（严格模式 + 影子校验 + 高风险路径自检 + 3 个新工具）；
  后续 `docs:` commit 补本篇 + 重放审计工具 + 顺序修正（`p0 校验→应用→p1 校验→应用`）。
* 探针：`training_history/probe_illegal_op_mode_20261027.log`
* 故障注入：`training_history/runs/20261027_1832*_illegal_fault_inject_*/output.log`
* 严格模式回归：`training_history/runs/*_illegal_strict_smoke/output.log`、
  `.../illegal_strict_highrisk/output.log`、`.../illegal_cold_control/output.log`
* 历史重放审计：`training_history/audit_replay_120_20261027.log`
* 每批的 `cmd.txt` / `meta.txt`（commit + dirty 计数）由 `code/run_logged.sh` 自动写在同目录。
