# 冠军 oracle 的崩溃根因 + 会话重建/回放 + E4/E4-rand 四臂（2026-10-29）

> **一句话**：冠军当 oracle **不是"会用着用着崩"，而是从第 1 回合起就变成一台 HOLD 机器**
> ——`OnRoundState` 每回合都会和它的影子局面比对，一旦不一致就 `force_pass_=true`（**整局不再复位**），
> 而它的影子局面里执行的是**它自己挑的操作**、真对局执行的是**我们的操作** ⇒ 两者从第一次分叉起
> 就再也对不上。实测 E2 的 16 局：**4177/4180 = 99.93% 的回合它给的是 `()`**，
> 而 `force_pass` 在 seed10 那个会话里 310/311 个回合都是 1（只有 round 0 是 0）。
> 崩（`apply_operations.called_after_terminal`，0xC0000409）只是这条链的**尾巴**：
> 影子里的自己从不防守 ⇒ 影子基地血归零 ⇒ 下一次喂对手操作就抛异常。
>
> ⇒ **"会话重建 + 回放追赶"能救"会话活着"（覆盖度），救不了"它给的动作是不是真动作"**。
> 本会话把这个区分做实了（§1.6 / §1.7），并据此把 E4 的 A 臂降级为"HOLD 候选"（§2.2）。
>
> ⛔ **E4 / E4-rand 这批 64 局在 2026-10-29 13:32 被父任务叫停**（只跑完 13 局、只有一个臂）。
> 原因：既然 A 臂恒为 HOLD，E4 测的就不是"冠军 vs 亚军"而是"{HOLD, 亚军}"；而且父任务解析
> 亚军注入臂后发现**被采纳的"新候选"里 70.5% 是 HOLD** ⇒ "注入有效"很可能只是"补了一个 HOLD 选项"。
> 于是整批**作废**（§7.2），改做一个不需要 oracle 的干净消融：`AZAI_ALWAYS_HOLD`
> （`docs/always_hold_ab_20261029.md`）。**E2 的 25.0% 与 E4 的任何胜负读数一律标注不可用。**

---

## 0. 本会话动了什么（清单）

| 文件 | 改动 | 默认行为 |
|---|---|---|
| `code/test_match/runnerup_oracle.py` | 记录每回合**真正发出去的帧**（`history`）+ `replay()` + `rebuild_session()` | `keep_history=True`（每局多存 1-2 MB 字符串）；不改协议 |
| `code/test_match/az_bridge_ai.py` | oracle 失败 ⇒ **重建 + 回放 + 重试本回合**（上限 `AZAI_ORACLE_REBUILD_MAX=6`）；新增**主动重启**（`AZAI_ORACLE_RESTART_EVERY`，默认 0 = 关）；`注入统计` 增加 `rebuild/restart/replay_*/hold` | 未注入时**一个字节都不变**（所有新代码都在 `_ORACLE` 分支内，且 `off=True` 提前返回） |
| `code/test_match/analyze_xoracle.py` | 覆盖度表增加 **`它 HOLD`** 列 + `rebuild/restart` 列；汇总行报 **HOLD 比例**（分母 = `[xor]` 回合数） | 老日志兼容（没有 `hold=` 就用 `[xor]` 行兜底） |
| `code/test_match/_us_vs_external.sh` | 抓取并回填 4 个 oracle 开关（它原本会 `unset` 掉所有 `AZAI_*`）；`${LADDER_EXTRA_ARGS}` 透传 | 不设 ⇒ 逐字不变（E1/E2 历史批次不受影响） |
| `code/test_match/_e4_only_extra.sh` / `_e4r_only_extra_rand.sh` | 写入 E4 的 oracle 加固 + 过夜三件套的默认值 | 只在 E4/E4R 里生效 |
| `code/test_match/_e4_overnight_all.sh` | **新增**：4 个 tag 顺序跑 + 每个 tag 跑完立刻出读数 + 最后出配对分析 | — |
| `code/test_match/_oracle_rebuild_accept.sh` | **新增**：任务 A 的两臂验收（`rebuild` / `restart`） | — |
| `code/test_match/_champ_oracle_rebuild_probe.py` | **新增**：回放能不能救回来（§1.4） | — |
| `code/test_match/_champ_oracle_shadow_probe.py` | **新增**：喂"它自己的世界"能不能让它不再 HOLD（§1.5） | — |
| `code/test_match/_e4_stratify.py` | **新增**：四臂汇总 + `[dscore]` 分层（§4） | — |

---

## 1. 任务 A：冠军 oracle 的崩溃——根因、修法、验收

### 1.1 先说结论（三条，按重要性排）

1. **"崩溃"不是主要问题，"HOLD"才是。** E2 的 16 局 4180 个决策里，冠军 oracle 给了 **4177 次 `()`**
   （99.93%）。它不是"后半局崩了所以只有半程剂量"，而是**全程几乎没有给过真动作**。
2. **根因在 `main.cpp` 的两行代码 + 一个架构事实**（§1.2）：影子局面失配 ⇒ `force_pass_` 永久置位；
   而"失配"是**必然**的，因为它的影子只执行**它自己挑的操作**，真对局执行**我们的操作**。
3. **崩溃是这条链的尾巴**：影子里的自己永远 HOLD ⇒ 只挨打不防守 ⇒ 影子基地血归零 ⇒
   `is_ended()` ⇒ 下一次 `ReplacePendingOperations` 抛异常 ⇒ 未捕获 ⇒ abort。
   ⇒ **"回放追赶"到崩点之前可以（实测 16-17 s / 239 回合），但一到崩点又会崩**（2/2 复现），
   所以对冠军来说"重建"只能救会话存活，救不了内容。

### 1.2 根因（源码级，`其他版本ai/ant-war2-magica-v3/main.cpp`）

| 位置 | 代码 | 后果 |
|---|---|---|
| `main.cpp:26706-26709` | `OnRoundState`: 影子 vs 真局面 `CompareProjectedState` 不一致 ⇒ `MarkDesynced("round_sync.desync")` **且** `force_pass_ = true` | 一次失配 ⇒ 整局不再搜索 |
| `main.cpp:12611-12614` | `IsFatalDiagnostic()` 只认 `env.desynced / env.init_error / step_round.desynced` | `round_sync.desync` **不在**里面 ⇒ 会话"健康"，只是永远 HOLD（所以日志里每回合都还在打 `runtime mismatch`） |
| `main.cpp:26638` | `force_pass_ = false` **只在 `OnMatchStart`** | 没有任何"恢复"路径 |
| `main.cpp:26663-26691` | `OnSelfOperations` / `OnOpponentOperations` 把**双方的操作**打进影子 | 影子里的"自己"= 它自己挑的操作；而真对局里那个座位执行的是**我们**的操作 ⇒ 从第 1 个不同的决策起 `coins[self]` 就对不上 |
| `main.cpp:4204-4222` | `judge_base_camp()`：任一方基地 HP ≤ 0 ⇒ `is_end = 1` | 影子里的自己 HOLD、我们的伤害照样进它的影子 ⇒ 它的基地血在影子模型里单调下降 |
| `main.cpp:12656-12657` | `ReplacePendingOperations`: `if (game_.is_ended() \|\| Phase::Terminal) throw ... "apply_operations.called_after_terminal"` | 影子自认终局后，**我们下一次喂对手操作块**就抛 ⇒ 未捕获 ⇒ `terminate` ⇒ 0xC0000409 |

**直接证据（E2 的真实日志）**：

```
[MagicaV0Policy] runtime mismatch seed=10 player=0 round=1 field=coins[0] artifact=
[Policy] Collect: round=0 shadow_healthy=1 force_pass=0 mcts_search max_sim=20
[Policy] Collect: round=1 shadow_healthy=1 force_pass=1 mcts_search max_sim=20
...（round 1..310 全是 force_pass=1；该会话 311 个回合里 310 个是 1，只有 round 0 是 0）
```

### 1.3 "回放追赶"为什么注定只救一半（`_champ_oracle_rebuild_probe.py`）

用 E1 的真实对局（`match_results/ladder_logs/US_VS_CHAMPION_BASE/11.log`，冠军执 P1）逐回合重放：

| 步骤 | 读数 |
|---|---|
| 基线会话 | 喂成功 **239** 回合，崩于 r239（`OracleDead`，进程退出码 `0xC0000409`） |
| 重建 #1：新进程 + 回放 0..238 | 回放**成功**（239 回合，**16.3 s**），但 `ask(r239)` **又崩在同一个回合** |
| 重建 #2：同上 | 回放 17.1 s，`ask(r239)` **再次崩**（2/2） |

⇒ 崩点由"它影子账本自认终局"决定，**与进程活了多久无关**。同一串历史重放 = 同一个终点。
（顺带：`--max-rounds 300` 的那次运行里它的答案与真实冠军动作一致率 61.8%，与历史读数 61.1% 同源
⇒ 冠军本身是随机的，这条限制继续成立。）

### 1.4 那"喂它自己的世界"呢？（`_champ_oracle_shadow_probe.py`）

思路：既然失配来自"它自己挑的操作没被执行"，那就**维护一份它视角的局面**（自己执行它的操作、
对手执行真实操作），把这份局面喂给它 ⇒ 影子与输入同源 ⇒ 应该不失配。

| 量 | 读数 |
|---|---|
| 前 18 回合 | ✅ 影子**完全对得上**（无 mismatch）——证明这个思路本身是对的 |
| r19 起 | ❌ `towers[1].x` 失配（我们的 pybind 引擎与它内部模型对某个塔位的理解不同）⇒ `force_pass` ⇒ 又变 HOLD |
| 259 回合合计 | HOLD **249/259 = 96.1%**；它自己那条线在第 259 回合**基地被打爆**（世界终局） |

⇒ **"冠军 oracle"这个用法有一条硬边界**：它的影子模型要求**闭环执行**（它挑的招必须真的被执行）。
只要不是闭环（oracle 天生不是），它就一定会在 ~20 回合内失配、然后整局 HOLD。
这不是"修一个 bug"能解决的，是它的设计前提。**任何"重建/重启/回放"都改变不了这一点。**

### 1.5 修法：会话自动重建 + 回放追赶（按任务书实现）+ 主动重启（备选 A）

**默认行为（任务书要求 1-3）**：

1. 检测到会话死亡/异常（`dead=1` / 进程退出 / `OracleDead` / `OracleDesync` / `OracleTimeout`）
   ⇒ **立刻重建会话**，把本局 round 0..当前回合的历史重新喂一遍（只喂状态与对手操作、**丢弃它的输出**），
   追上后**重试本回合**；
2. 重建上限 `AZAI_ORACLE_REBUILD_MAX`（默认 **6**，`0` = 退回旧行为"死了就永久停用"）；
3. stderr 逐次记录：`[oracle] 重建会话 #k at round=R（回放 R 回合，耗时 X s）`
   （**每次重建写一个新的 stderr 文件**，旧会话的崩溃日志不会被覆盖）；
4. 同一回合最多重试 2 次（防"崩-重建-崩"死循环）。

**主动重启（`AZAI_ORACLE_RESTART_EVERY=N`，默认 0 = 关）**：不等它崩，每 N 回合换一个新进程 +
**回合号重标**（把当前真局面当成它的第 0/1/2… 回合喂，读端不错位，不重放历史）。
理由：对冠军来说"重建+回放"在崩点必崩（§1.3），而**它的内容本来就是 HOLD**（§1.1），
所以"每 N 回合换一张白纸"是唯一能让**会话**活完整局的办法。日志：`[oracle] 主动重启会话 #k at round=R`。

### 1.6 验收读数（对手 = 冠军，注入 = 冠军；全部 `verdict=engine`、`illegal=0`）

| 臂 | 局 | 回合 | asks | **覆盖度** | dead | 成功重建/失败 | 主动重启 | 单次重建墙钟 | 它 HOLD |
|---|---|---|---|---|---|---|---|---|---|
| **E2（旧代码）** | 16 | — | 4276 | **54.4%** | 14/16 局 | —（死一次就永久停用） | 0 | — | **99.93%**（4177/4180） |
| **A1** `rebuild`：只开"崩了才重建" | 7（P0） | 512 | 283 | **55.3%** | **7** | **6 / 0** | 0 | **20.9 s**（回放 283 回合） | **100%**（273/273） |
| **A2** `restart`：每 150 回合主动重启 | 7/8/9（P0） | 512 / 324 / 453 | 512 / 324 / 453 | **100%** | **0** | 0/0 | **8** | 重启 ≈0.5-1 s（不回放） | **99.8%**（1257/1259） |
| **A3** `kill test`：第 41 回合**按 PID 精确杀** oracle | 10（P0） | 90（截断冒烟） | 90 | **100%** | 1 | **1 / 0** | 0 | **3.2 s**（回放 41 回合） | — |

三条结论（**第 1 条是"修好了"，第 2 条是"修不好"**）：

1. **机制本身是对的**（A3，`_oracle_kill_rebuild_test.sh`）：精确杀掉 oracle 子进程后，
   桥接第 41 回合立刻报 `OracleDead` ⇒ 重建 ⇒ **回放 41 回合 3.2 s** ⇒ `↻ 重试本回合` 成功
   ⇒ 该局覆盖度仍是 **100%**（只丢"那半个回合"的候选）。真实日志：

   ```
   [oracle] ⚠️ 第 41 回合查询失败（0ms）⇒ 尝试重建会话: OracleDead: 亚军进程已退出 (exit code 4294967295)
   [oracle] 重建会话 #1 at round=41（回放 41 回合，耗时 3.2 s）（崩溃后追赶）
   [oracle] ↻ 第 41 回合：会话已重建，重试本回合（第 1 次）
   ```

2. **但它救不了冠军的"自然崩"**（A1）：崩点由它影子账本里的"终局"决定，回放同一串历史 =
   同一个终点 ⇒ 6 次重建**回放全部成功**、**每次重试都在同一回合再崩**，最后按上限停用。
   覆盖度 54.4% → **55.3%（等于没变）**。这与 §1.3 的探针（2/2 在同一回合复现）完全一致。
   ⇒ **"回放追赶"对冠军这条路是死路**，不是实现 bug。

3. **真正能让会话活完整局的是"主动重启"**（A2）：每 150 回合换一张白纸（回合号重标、不回放）
   ⇒ 覆盖度 **100%**、`dead=0`、8 次重启、3 局分别跑到 512/324/453 回合。
   ⚠️ **但它的内容仍然是 HOLD（99.8%）** ——"覆盖度"只证明**它答了**，不证明**它给了真动作**。

**验收判据逐条对照**：

| 判据（任务书 / 父任务） | A1 rebuild | A2 restart | 结论 |
|---|---|---|---|
| 注入覆盖 ≥95%（asks / rounds） | 55.3% ❌ | **100%** ✅ | 只有"主动重启"这一条路能到 95% |
| `dead` 重建后仍能继续 | 重建成功但**同一回合再崩** ⇒ 6 次后停用 ❌ | 不触发（无 dead） | 机制对**可恢复**故障有效（A3 ✅） |
| 单次重建 <60 s | **20.9 s**（283 回合）✅ | ≈0.5-1 s ✅ | 回放速率 ≈ 68-78 ms/回合 ⇒ 512 回合 ≈ 35-40 s，**仍在 60 s 内** |
| ≥3 局、跑到 ≥450 回合 | 1 局 512 ✅ | 2/3 局（512、453）✅；第 3 局 **324 回合是引擎判终局**（我们基地被打爆，`verdict=engine`），不是截断 | 见右 |

> **一句话**：**任务 A 的"会话重建 + 回放追赶"按任务书实现了、也验收了（A3），
> 但对冠军这条特定的崩溃（影子自认终局）它是死路（A1）；能让覆盖率到 100% 的只是"主动重启"，
> 而主动重启换来的是"会话活着 + 内容仍是 HOLD"**。这个区分是本次最重要的工程结论。

### 1.7 对 E2 结论的修正（必须落档，`docs/us_vs_champion_20261029.md` §4 需要按这里重读）

| E2 报告里的说法 | 本次实测 | 应该怎么读 |
|---|---|---|
| "冠军 68% 的回合是 HOLD" | **99.93%**（4177/4180） | 不是"常常 HOLD"，是"**恒 HOLD**"（97.2% 的会话从第 1 回合起 `force_pass=1`；余下 2.8% 是它失配**之前**的几回合） |
| "`new/asks = 31.7%` = 冠军那一手**不在我们候选集里**的比例（真·增量）" | `new` = "**HOLD 不在候选集里**"的回合数 | 它说明的是"我们的 48 条网络候选有 31.7% 的回合**不含 HOLD**"——与冠军无关 |
| "`hit_new/new = 43.1%` 真·增量候选里我们采纳的比例" | 同上 | = "我们把 HOLD 加进候选后**真的选了 HOLD**的比例" |
| "E2 = 半程注入（覆盖 54.4%）" | 覆盖度确实 54.4%，但**内容**是 HOLD | E2 的干预 = **给评估器补一个 HOLD 选项**，不是"借冠军的招" |

⇒ **E2 的 25%（4W-12L）vs E1 的 6.2%（1W-15L）应当重读为**：
"在 31.7% 的回合里把 HOLD 补进候选集、并让评估器自己决定要不要空过" 的效果
（配对 Δhp +5.19，p=0.375 不显著，方向为正）。
这与"注入冠军动作"是**两件不同的事**，之前的报告把两者混为一谈了。
**本会话没有重跑 E2**（64 局 E4 优先），所以 E2 的胜负读数本身是有效的、只是**解释要改**。



### 2.1 预注册的设计（`docs/us_vs_champion_20261029.md` §5，未改）

| 臂 | 候选集 | 挑法 | 对手 | 局数 |
|---|---|---|---|---|
| **E4** | `AZAI_ONLY_EXTRA=1` ⇒ **只有**「冠军那一手 + 亚军那一手」 | 我们的 flat 评估器 argmax | 冠军 / 亚军 | 16 + 16 |
| **E4-rand** | 同上 | **随机**挑一个（`sha256("onlyextra\|seed\|dec_n")`，可复现） | 冠军 / 亚军 | 16 + 16 |

判据（跑之前登记）：E4 vs E4-rand（同 token 配对）**McNemar p<0.05 且胜率差 ≥+12.5pp** 才算
"评估器在这类选择上有贡献"；否则"不可分辨"。

### 2.2 实际跑出来的 A 臂 = **HOLD**（这是 §1 的直接后果，必须在读数前说清）

| 事实 | 读数 | 来源 |
|---|---|---|
| 冠军 oracle 的答案里 `()` 的比例 | **99.8% ~ 100%**（E2 是 99.93%） | §1.1 / §1.6 |
| `force_pass` 从第几回合起为 1 | **第 1 回合**（seed10 那个会话 310/311） | §1.2 |
| 自然崩的局里它**崩前**给过几个非 HOLD | 7/250（2.8%），且**全部**被我们影子校验判非法 | A1 的 `illegal=7` |
| 主动重启后的新会话 | 第 1 回合就 `force_pass` ⇒ 一律 HOLD | §1.5 |

⇒ **E4 的候选集实际是 `{HOLD, 亚军那一手}`**，不是 `{冠军动作, 亚军动作}`。
本次**没有**偷偷换掉 A 臂（也没有把"冠军动作"这个词继续用在读数上），而是：

* 照预注册把 64 局跑完（数据本身是干净的：`verdict=engine`、`illegal=0`、两个会话都真的在跑）；
* 把主读数**重新命名**为"**评估器在 {HOLD, 亚军那一手} 两个候选里挑 vs 随机挑**"；
* 在 §4 里把"哪些决策**两臂才可能不同**"（`N_pair_differs`）与"高 Δscore 子集"都报出来，
  因为这才是这批数据能支撑的结论强度。

### 2.3 为什么**没有**用父任务给的备选 B / C 换掉 A 臂

| 备选 | 为什么不用 |
|---|---|
| **B：只统计前 200 回合** | 救不了：冠军的退化从**第 1 回合**就开始了（§1.2 的 `force_pass` 从 round 1 就是 1），前 200 回合里它已经 99% 是 HOLD ⇒ 截前 200 回合只是**样本更小**，不是**更干净**。 |
| **C：换"我们自己的旧强配置"当 oracle** | 结构性同样不 faithful：我们的 AI 也维护**影子 facade**（`az_bridge_ai` 主循环里自己 `apply` 双方操作，还用收到的局面文本做 `verify`），当 oracle 时"它挑的招"不会被执行 ⇒ 影子照样分叉（只是它不会 `force_pass`，所以**看起来**还在出招，动作却是在一个漂移的世界里算的）。用它跑 64 局 = 拿一个**没法验证 faithfulness** 的 oracle 去当主读数，比"承认 A 臂是 HOLD"更糟。 |
| 换 `ai_cpp_lure_v4_custom.exe`（同族另一变体）当第二 oracle | 有可能，但**没验证**它的强度/合法性/行为，且会把预注册的"冠军 + 亚军"改成"亚军 + 亚军变体"——那是另一个实验，不是本次能在一夜内验证的东西。**留给下一轮**（§5 第 2 条）。 |

### 2.4 一个必须记住的口径差别

* **E2** = `{我们的 48 条网络候选} ∪ {HOLD}`（冠军那一手是 HOLD，作为**第 49 条**加进去）；
  `new=1324` 的真正含义是"**HOLD 本来不在候选集里**的回合数"（31.7%），**不是**"冠军给了新招"。
* **E4** = `{HOLD, 亚军那一手}`（**网络候选一条都不进**）⇒ 这是"两个候选里挑"的**纯净**版本，
  但其中一个候选**恒为 HOLD**。
* 所以 **E2 与 E4 的读数不能互相解释**：E2 是"多一个 HOLD 选项"，E4 是"只有 HOLD 与亚军"。

---

## 3. E4 / E4-rand 结果

> ⛔ **本节的正式读数不存在**：4 个 tag 只跑完 `ONLYEXTRA_VS_CHAMPION` 的 **13/16 局**，
> 就被父任务叫停（理由见 §7.2）。E4-rand 一局都没跑 ⇒ **没有配对比较、没有四臂读数**。
> 那 13 局只能当**试点**，读数与口径在 §7.3；**不得**与 E1/E2 做差、**不得**当结论引用。

---

## 4. 预注册判据落档 + `[dscore]` 分层诊断

预注册判据（§2.1）：E4 vs E4-rand 配对 McNemar `p<0.05` 且胜率差 ≥ +12.5pp。

| 判据 | 能否评估 | 原因 |
|---|---|---|
| McNemar p | **无法评估** | E4-rand 0 局 ⇒ 0 对 |
| 胜率差 | **无法评估** | 同上 |
| `N_rounds / N_pair_differs / N_one_cand / N_both_hold` | 只有单臂试点值：6499 / 2120 / 4379 / 4304（§7.3） | 单臂也能报"可分辨决策数"，但**没有对照臂**就只是描述 |
| `N_high_dscore` | 试点：`|Δscore|` ≥ p75(0.0569) 的决策 **531** 个 | 同上 |
| `|Δscore|` 分位 | 试点：p25 0.0118 / 中位 0.0290 / p75 0.0569 / p90 0.1153 / max 0.5567 | 这条**单臂也成立**，是有用的检验力信息 |
| `pick_hi` 正向对照 | 试点：**2120/2120 = 100%** ✅ | 说明"E4 臂按评估器挑"这件事确实在跑 |

> **口径不变**（仍然有效，留给下一轮做真 E4 的人）：`E4≈E4-rand` **不构成**"评估器无用"的证据，
> 除非在"高 Δscore 子集"上也无差别；而"高 Δscore 子集"必须先用 `N_pair_differs` 与
> `|Δscore|` 分位证明**真的有可分辨决策**。本次试点给出的是这两个分母（32.6% 可分辨、
> 中位分差 0.029），**不是**判据的答案。


---

## 7. ⛔ 实验作废说明（2026-10-29 13:32，父任务叫停）

### 7.1 实际跑到哪一步

| tag | 局数 | 状态 |
|---|---|---|
| `ONLYEXTRA_VS_CHAMPION`（E4 vs 冠军） | **13/16 完成**（+3 局在跑时被杀） | 日志留存，**不作结论** |
| `ONLYEXTRA_RAND_VS_CHAMPION` | 0/16 | 没开始（顺序执行，被停在第 1 个 tag） |
| `ONLYEXTRA_VS_RUNNERUP` / `ONLYEXTRA_RAND_VS_RUNNERUP` | 0/16 / 0/16 | 没开始 |

停止方式：按 **PID 精确**杀 ladder → 它启动的 3 局（bridge + 两个 oracle + 对手）→ 包装 shell
（没有用镜像名/通配符；完成局的日志一字节没删）。停完复查：机器上只剩
`send_videos.py`（别人的，10-21 起）与 `exp28_coevolution`（别人的），两者都没动。

### 7.2 为什么整批作废（三条独立事实，任一成立就足以作废）

| # | 事实 | 读数 | 来源 |
|---|---|---|---|
| 1 | **冠军 oracle 从第 1 回合起就永久空过**：`OnRoundState` 发现影子与真局面不一致 ⇒ `MarkDesynced` + `force_pass_=true`，而 `force_pass_` 只在 `OnMatchStart` 复位 | E2 16 局 **4177/4180 = 99.93%** 的回合它给 `()`；`force_pass` 在 seed10 会话里 **310/311** 个回合为 1（只有 round 0 是 0）；本批 13 局实测 **1.4%** 非 HOLD | §1.1 / §1.2 + 本批 `[xor]` 逐回合行 |
| 2 | 因此 **E4 的 A 臂 ≠ "冠军动作"**，而是常量 HOLD ⇒ E4 实际是 `{HOLD, 亚军那一手}`，**不是**预注册的 `{冠军动作, 亚军动作}` | 见 §2.2 | — |
| 3 | **亚军注入臂的"增益"里 HOLD 占绝大多数**（父任务直接解析 `match_results/inject_vs_ru/A/stderr/`）：注入 8117 次里 **6857 次(84.5%) 是 HOLD**、`new` 候选 2721 里 **1522(55.9%) 是 HOLD**、`hit_new` 1244 里 **877(70.5%) 是 HOLD** ⇒ 非 HOLD 新增动作只被采纳 **367 次(4.5%)** | 父任务口径 | 见 `docs/always_hold_ab_20261029.md` |

⇒ **E2 的 25.0%（4W-12L）与 E4 的任何胜负读数一律标注"不可用"**。
理由不是"结果不好看"，而是**干预内容与预注册不是同一件事**：
E2/E4 里那个"外部意见"实际是一个常量 `()`，所以它们只能测"**给评估器补一个 HOLD 选项**"的效应，
而那件事已经被单独做成一个干净的消融实验（`docs/always_hold_ab_20261029.md`，`AZAI_ALWAYS_HOLD`）。

### 7.3 那 13 局不是白跑：作为**试点**留下的三个读数（**不作结论**，只用于定下一步）

| 量 | 试点读数（13 局 vs 冠军） | 并排（同一对手、同 seeds，历史批次） |
|---|---|---|
| 胜负 | **3W-10L = 23.1%**（Wilson [8.2%, 50.3%]） | E1（我们 48 条候选）= 1W-15L = **6.2%**；E2（48 条 + HOLD）= 4W-16 = **25.0%** |
| 血量差 | **−19.08**（t 95%CI [−29.23, −8.93]） | E1 = −22.62 [−28.68, −16.57]；E2 = −17.44 [−24.91, −9.97] |
| 我们 HOLD 的回合占比 | **5645/6499 = 86.9%** | 现役（E1 口径）约 78% |
| 亚军出招时我们采纳它 | **778/2086 = 37.3%**（<50% ⇒ 评估器比随机**更不愿意**采纳外部动作） | E4-rand 按构造 = 50% |
| `N_pair_differs` / `N_rounds` | 2120 / 6499 = **32.6%** | — |
| `|Δscore|`（两候选） | p25 **0.0118** / 中位 **0.0290** / p75 **0.0569** / p90 **0.1153** | — |
| `pick_hi`（正向对照） | **2120/2120 = 100%** ✅ 臂按设计在跑 | — |

**读法（三句，别超读）**：
1. 只用 `{HOLD, 亚军那一手}` 两个候选、且 **86.9% 的回合空过**，就拿到了 23.1%
   —— 与"48 条网络候选 + HOLD"的 25.0% 几乎一样，**远好于** 48 条候选单干的 6.2%。
   这与父任务的假设（"缺的是 HOLD 这个选项"）**方向一致**，但 n=13，**不能当证据**。
2. 我们的评估器在"亚军出招"的 2086 个决策里只采纳了 37.3%（比随机的 50% 更低）
   ⇒ 它**有偏好**（偏向 HOLD），不是随便挑；`pick_hi=100%` 说明它确实是按分数挑的。
3. `|Δscore|` 中位只有 **0.029**（p25 0.012）⇒ 两个候选的价值差本来就很小
   —— 这也是为什么"E4≈E4-rand"必须先看 `N_pair_differs` 与高 Δscore 子集（§4 的口径），
   否则会把"分差本来就小"误读成"评估器没用"。

### 7.4 下一步（按父任务的决定）

* **替换实验**：`AZAI_ALWAYS_HOLD`（HOLD 恒为候选）A/B 消融，对手 = 亚军，每臂 16 局
  —— 已实现、已冒烟、已开跑，报告在 `docs/always_hold_ab_20261029.md`。
* **冠军 oracle 这条线到此为止**：除非换成一个**闭环执行**的 oracle（它挑的招必须真的被执行），
  否则"冠军动作"永远拿不到（§1.4 的硬边界）。可行的替代是**无状态**的 oracle
  （亚军就是：每回合重读局面，不维护影子），或换一个同族的第二个无状态策略（未验证）。


---

## 5. 已知限制（诚实清单）

1. **A 臂 = HOLD（最重的一条）**：E4 的"冠军那一手"实际恒为 `()`（99.8-100%）。所以本批**不能**
   回答"评估器在两个**外部动作**里挑得对不对"，只能回答"评估器在 **{HOLD, 亚军那一手}** 里挑
   vs 随机挑"。想要前者，先要一个**能闭环执行**的第二 oracle（§5 第 2 条 / §2.3）。
2. **亚军的候选也常常是 HOLD**：`cpp_lure_v4` 历史上 77.9% 的回合 HOLD。两个候选都 HOLD 时
   `n_extra=1`（按构造两臂完全相同）⇒ 真实可分辨的决策数远小于回合数。**先看 §4 的
   `N_pair_differs`，再看胜率**。
3. **冠军不是确定性对手**（`docs/us_vs_champion_20261029.md` §0.3 / §4.2.2）：
   同一局面它自己的两个答案只有 **61.1%** 相同 ⇒ 16 局的方差里有一部分是**对手的随机性**，
   不是我们的。何况 E4 的 A 臂本来就退化了，这条只影响"对手侧"。
4. **64 局的分辨率**：16 对配对、二分类，McNemar 只有在"两臂不一致的配对"够多时才有检验力
   （比如 8 对全一边倒也只到 p≈0.008；4-4 就 p=1）。胜率差的 Wilson 区间在 16 局下大约
   ±20pp ⇒ **只够分辨大效应**。
5. **两臂的对局不是同一个游戏**：`E4` 与 `E4-rand` 在同一 token 下也是**两条不同的棋**
   （第 1 个不同的决策之后局面就分叉了）⇒ 决策级反事实做不到；§4 的"高 Δscore 分层"是
   **按局的协变量分层**，属**探索性**，不是随机化证据。
6. **`pick_hi` 是正向对照，不是效果量**：E4 ≈100% / E4-rand ≈50% 只说明"臂按设计在跑"。
   若 E4 的 `pick_hi` 明显 <100%，那是**执行层过滤**（候选被删 op / 整块闸门丢掉）造成的，
   不是"评估器没挑它"。
7. **对手 = 亚军那一组有一个同源问题**：B 会话问的就是亚军本体，而对手也是它
   ⇒ 候选集里那一手是"**同一个策略在另一个座位上**会怎么走"（不是抄对手当前那一手，
   但同源）。读这一组时不要把"评估器偏好亚军的手"当成"评估器强"。
8. **主动重启的代价没有量化**：A2 里每次重启都让它的影子从头攒（= 它更 HOLD）。
   由于它本来就是 HOLD，这个代价在本批数据里**看不出来**；但如果将来换成一个 faithful 的 oracle，
   `AZAI_ORACLE_RESTART_EVERY` 必须关掉（默认就是 0）。
9. **没有**做"冠军为什么在第 19 回合开始与我们的引擎分歧"（`towers[1].x`）的代码级追查：
   那需要把 magica 的规则实现与我们 pybind 引擎逐条对拍，超出本会话范围（§1.4 只做到"有分歧"）。


---

## 6. 复现清单

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"

# ── 任务 A：根因探针（各 ~1-3 min，只读）──────────────────────────────────────
#  ① 同一个会话能不能"回放救回来"：基线崩于 r239，重建后回放到 238 成功、ask(239) 又崩（2/2）
$P -u code/test_match/_champ_oracle_rebuild_probe.py \
    --log match_results/ladder_logs/US_VS_CHAMPION_BASE/11.log --max-rounds 512 --attempts 2
#  ② 喂"它自己的世界"能不能让它不失配：前 18 回合对得上，r19 起 towers[1].x 失配 ⇒ 96% HOLD
$P -u code/test_match/_champ_oracle_shadow_probe.py \
    --log match_results/ladder_logs/US_VS_CHAMPION_BASE/11.log --max-rounds 300 --mode shadow

# ── 任务 A：验收（3 个臂）────────────────────────────────────────────────────
#  A1 只开"崩了才重建"：覆盖度 55.3%（回放成功但同一回合再崩，6 次后停用）
ARM=rebuild SEEDS=7 JOBS=1 bash code/run_logged.sh oracle_fix_rebuild \
    bash code/test_match/_oracle_rebuild_accept.sh
#  A2 每 150 回合主动重启：覆盖度 100%、dead=0、重启 8 次
ARM=restart SEEDS="7 8 9" JOBS=3 bash code/run_logged.sh oracle_fix_restart \
    bash code/test_match/_oracle_rebuild_accept.sh
#  A3 第 41 回合**按 PID 精确杀** oracle：重建 + 回放 41 回合 3.2s + 重试成功
SEED=10 KILL_AT_ROUND=40 ROUNDS=90 bash code/run_logged.sh oracle_kill_test \
    bash code/test_match/_oracle_kill_rebuild_test.sh
#  读数（覆盖度 + 它 HOLD 的比例 + 重建/重启次数 + 单次重建墙钟）
$P -u code/test_match/analyze_xoracle.py --tag ORACLE_FIX_REBUILD \
    --stderr match_results/us_vs_external/ORACLE_FIX_REBUILD/stderr
$P -u code/test_match/analyze_xoracle.py --tag ORACLE_FIX_RESTART \
    --stderr match_results/us_vs_external/ORACLE_FIX_RESTART/stderr
#  旧口径重算（证明 E2 那 16 局其实是 99.93% HOLD）
$P -u code/test_match/analyze_xoracle.py --tag US_VS_CHAMPION_INJ \
    --stderr match_results/us_vs_external/US_VS_CHAMPION_INJ/stderr

# ── 任务 B：E4 / E4-rand（64 局，一条命令跑完 4 个 tag）──────────────────────
#  过夜三件套写在 `_e4_only_extra.sh` / `_e4r_only_extra_rand.sh` 的默认值里：
#    LADDER_TIMEOUT=21600  LADDER_EXTRA_ARGS="--retry-invalid 2 --final-sweep --watchdog-stall-min 90"
#    MVS_TIMEOUT_SECONDS=300  MVS_STDERR_SUFFIX=1（由 `_us_vs_external.sh` 统一 export）
#  ⚠️ **不**起 `_keepawake.ps1`：宿舍夜里断电，靠电池保活会耗光电池；主动接受休眠 + retry/sweep 补跑。
JOBS=10 bash code/run_logged.sh e4_overnight_all bash code/test_match/_e4_overnight_all.sh

# ── 读数（4 个 tag）──────────────────────────────────────────────────────────
for T in ONLYEXTRA_VS_CHAMPION ONLYEXTRA_RAND_VS_CHAMPION ONLYEXTRA_VS_RUNNERUP ONLYEXTRA_RAND_VS_RUNNERUP; do
  $P -u code/test_match/_batch_table.py --tag $T \
      --stderr match_results/us_vs_external/$T/stderr --opp-label ${T%%_VS_*} \
      --markdown-out match_results/us_vs_external/$T/table.md
  $P -u code/test_match/analyze_xoracle.py --tag $T \
      --stderr match_results/us_vs_external/$T/stderr
done
# 四臂汇总 + |Δscore| 分层 + 配对 McNemar
$P -u code/test_match/_e4_stratify.py
# 单个对手的配对检验（同 seeds/座位）
$P -u code/test_match/analyze_inject_vs_runnerup.py \
    --arm E4=match_results/ladder_logs/ONLYEXTRA_VS_CHAMPION \
    --stderr E4=match_results/us_vs_external/ONLYEXTRA_VS_CHAMPION/stderr \
    --arm E4R=match_results/ladder_logs/ONLYEXTRA_RAND_VS_CHAMPION \
    --stderr E4R=match_results/us_vs_external/ONLYEXTRA_RAND_VS_CHAMPION/stderr \
    --pair E4 E4R
```

> 实验日志：`docs/experiment_log.md`（`run_logged` 自动追加，含 commit/dirty/命令/输出路径）；
> 原始输出：`training_history/runs/2026102*_oracle_fix_*/output.log`、`..._e4_overnight_all/output.log`。
