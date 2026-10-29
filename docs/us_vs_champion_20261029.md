# 我们 vs 冠军：E1 基线（16 局补完）+ E2 / E4 预注册（2026-10-29）

> 一句话：**E1（不注入）= 1/16 = 6.2%（Wilson [1.1%, 28.3%]），血量差 −22.62（t 95%CI [−28.68, −16.57]）**；
> **E2（注入冠军动作）= 4/16 = 25.0%（[10.2%, 49.5%]），血量差 −17.44（[−24.91, −9.97]）**，
> 与 E1 配对 McNemar **p=0.375**、配对 Δhp **+5.19 [−14.50, +4.13]** ⇒ 方向为正但**不显著**；
> 16 局全部 `verdict=engine`、`illegal=0`（裁判侧 + 我方自检），冠军侧 stderr 全留存。
> 三个必须知道的坑：① 上一批"3 局 timeout + 8 局空壳"的真根因是**整机进 S0 待机被冻 2h06m**（§1）；
> ② **冠军当 oracle 会在第 ~250 回合自己 abort**（`apply_operations.called_after_terminal`），
> E2 的注入覆盖只有 **54.4%**（§4.2.1）；③ E4/E4-rand 必须在修掉②之后再跑，否则候选集后半局会塌成一个（§5）。

---

## 0. 口径与这一篇管什么

### 0.1 三批实验（同一个 BASE、同一批 seeds、同一套 harness）

| 实验 | 被试 | 对手 | 局数 | 状态 |
|---|---|---|---|---|
| **E1** | 我们（**不注入**） | 冠军 `magica_v3_O0.exe` | 16 = seeds 7..14 × 2 座位 | **本次补完** |
| **E2** | 我们 + **注入冠军动作**（`AZAI_ORACLE_INJECT`=冠军，timeout 20s） | 冠军 | 16 | 见 §4 |
| **E4** | 候选集**只留**「冠军动作 + 亚军动作」，**由我们的评估器挑**（`AZAI_ONLY_EXTRA=1` + 两个 oracle 会话） | 冠军 / 亚军 | 16 + 16 | 见 §5 |
| **E4-rand** | 同一候选集，**随机挑一个**（`AZAI_ONLY_EXTRA_RANDOM=1`，固定种子可复现） | 冠军 / 亚军 | 16 + 16 | 见 §5 |

- 配置 `BASE` 与历史所有"我们 vs 亚军"批次**逐字相同**（`_us_vs_external.sh` / `_us_vs_champion_e1_resume.sh`
  里都有 `BASE == _inject_vs_runnerup.sh` 的逐字节断言，600 字符）。
- token 语义：`7` = 我们执 **P0**；`7r` = 镜像，我们执 **P1**。
- 与 `docs/us_vs_champion_20261028.md` 的分工：那篇是**设计与工程坑**（DLL/0xC0000139、冠军 stderr 爆量、
  协议换行、超时），本篇是**读数与统计**。

### 0.2 与"vs 亚军"的读数**不能直接相减**

对手不同（冠军 vs 亚军）⇒ 只做**并排展示**（§6），不做差、不配对检验。
配对只在同一批内成立（同 seeds 的 N/Nr 是"先手互换的两盘棋"）。

### 0.3 冠军**不是确定性对手**

`experiment_log` 2026-09-22 那批（冠军 vs 亚军）记录过**同 seed 同配置胜负翻转**（seed 9 / 11），
最可能的原因是冠军的决策受机器状态影响。⇒ ① 不要拿"同 seed 重跑应逐位相同"校验冠军侧 harness；
② 16 局的分辨率只够分辨**大效应**（Wilson 95%CI 在 0/16 时上界 ≈ 19.4%，在 0/12 时 ≈ 24.3%）。

---

## 1. 运行环境风险：整机进 Modern Standby（S0）—— 上一批 timeout 的**真正根因**

### 1.1 证据（系统事件日志，`Microsoft-Windows-Kernel-Power`）

| 时间 | 事件 | 含义 |
|---|---|---|
| 2026-10-29 **00:10:48** | 506 | 系统**进入**新型待机（S0） |
| 2026-10-29 **02:16:47** | 507 | 系统**退出**新型待机 |
| 2026-10-29 07:52:18 | 507 | 退出新型待机（父任务记录的那次） |
| 2026-10-29 02:16:55 / 02:16:59 | 42 / 107 | 进入睡眠 / 从睡眠恢复 |

**冻结窗口 = 00:10:48 → 02:16:47（2 小时 6 分钟）**。这一窗口**正好**覆盖上一批 E1 的第二、三波：

- `9r / 10 / 10r` 三局在 00:10 冻在 **回合 285 / 309 / 285**；恢复瞬间 bridge 的
  `MVS_TIMEOUT_SECONDS`（当时是默认 300s）到期 ⇒ `TimeoutError: timed out reading ...` ⇒
  `verdict=timeout`、`winner=INVALID`（日志里三条都是这个栈）。
- `11 11r 12 12r 13 13r 14 14r` 八局在 00:10 冻在 **回合 ~196–223**，恢复瞬间被父 ladder 的
  7200s 墙钟一起判死 ⇒ 日志被截断成"空壳"（`_attempts` 里留档，`rounds≈200`）。
- 冻结期间冠军自己的日志里 `elapsed_us` 会跳到小时级（`docs/pin_k2_ab_20261028.md` §2.1 有同源证据）。

> ⚠️ 我第一版归因写的是"并发抬到 11–12 局 ⇒ 单局慢 7 倍"，**那是错的**：慢 7 倍是因为
> 2 小时里 CPU 根本没在跑，而墙钟（和 300s 读超时）照走。并发只是次要因素。

### 1.2 三件套（本批用的就是这三样）

| 件 | 路径 | 作用 | 本批使用 |
|---|---|---|---|
| **保活** | `code/test_match/_keepawake.ps1` | `SetThreadExecutionState(ES_CONTINUOUS\|ES_SYSTEM_REQUIRED\|ES_DISPLAY_REQUIRED)`，每 30s 刷一次 ⇒ 进程活着期间系统不进待机；**不改任何系统设置** | ✅ 08:23:57 起（PID 58908），日志 `match_results/champion_e1/keepawake.log` |
| **看门狗** | `code/test_match/_ladder_watchdog.ps1` | 按 `training_history/progress/<tag>/az_progress_seed*.txt` 的 **mtime** 判卡死（>StallMinutes）；默认**只看不杀**，`-Kill` 才按 PID 精确清那一局 | ✅ 08:31:57 起（PID 43896，`-StallMinutes 12`，**未开 -Kill**），日志 `match_results/champion_e1/watchdog.log` |
| **超时** | 环境变量 | `LADDER_TIMEOUT=10800`（单局 3h 墙钟）+ `MVS_TIMEOUT_SECONDS`（单次读包卡死检测） | ✅ 本批 `LADDER_TIMEOUT=10800`；`MVS_TIMEOUT_SECONDS=1800`（见下方口径） |

⚠️ `powercfg /change standby-timeout-ac 0` **挡不住 S0**：本机 `STANDBYIDLE`(AC) 已经是 0，仍然进过待机。
`powercfg /requests` 需要管理员权限才能核验，所以保活是否生效只能靠"事件日志里没有新的 506/507"来反证。

**本批的保活覆盖情况（诚实说明）**：E1 续跑批 07:56 开跑，**保活从 08:23:57 才起**
（07:56–08:24 这 28 分钟只有 `STANDBYIDLE=0` 兜着）。事实核查：**07:52 之后没有任何新的 506/507 事件**，
且每局 progress 文件一直在前进（看门狗 `idle` 最大 0.7 分钟）⇒ 这 28 分钟**没有**被冻。

### 1.3 超时参数的口径差异（要写清楚，别当成"两批不可比"）

| 参数 | 上一批（5 局已完成） | 本次续跑（11 局） | 依据 |
|---|---|---|---|
| `LADDER_TIMEOUT` | **未设** ⇒ 默认 7200s | **10800s** | 冠军是 O0 编译 + 全项目最重的 MCTS；上一批正是 2h 被掐 |
| `MVS_TIMEOUT_SECONDS` | 未设 ⇒ 默认 300s | **1800s** | 父任务后续改为"用 300，避免被冻 30 分钟才判死"；本批已开跑，**保持 1800 跑完**，E2/E4 改回 300 |

这两个**只**决定"多慢算卡死/掐断"，**不是每步预算**（我们的搜索是预算制，不吃墙钟）⇒
在没有卡死的局上它们对**胜负与血量差没有任何影响**；本次 16 局都没有卡死（§3.3 逐局墙钟 31–44 min）。

---

## 2. 续跑是怎么做的（只补没跑的局，一个字节都不重跑已完成局）

脚本：`code/test_match/_us_vs_champion_e1_resume.sh`（本批新增，可复用）。

1. **resume 判据** = 日志里有没有 `verdict=engine`：
   `remaining_tokens()` 扫 `match_results/ladder_logs/US_VS_CHAMPION_BASE/<token>.log`，
   只有**没有** `verdict=engine` 的 token 才进 `_tmp_ladder.py` 的命令行。
   本批实测：已完成 `7 7r 8 8r 9`（**跳过**），待补 `9r 10 10r 11 11r 12 12r 13 13r 14 14r`（11 局）。
2. **旧 stderr 先搬走**（只搬不删，只搬本批 tag 后缀）：
   按 `(座位, seed)` 精确算 —— 同一个 seed 的两个座位是**两个文件**，其中已完成局那一份是证据
   （例：`ai1_champion_seed9_US_VS_CHAMPION_BASE.stderr.log` 属于已完成的 `9`，绝不能动；
   属于 timeout 的 `9r` 那份是 `ai0_champion_seed9_...`）。搬到
   `match_results/champion_e1/_attempts/<时间戳>/`。
3. **stderr 唯一化**：`MVS_STDERR_SUFFIX=1` + `DSH_PROGRESS_SUFFIX=_<TAG>`（与 2026-10-28 那批**逐字相同**，
   否则新老 stderr 落不到同一套文件名上）。
4. **逐局归档**：`code/test_match/archive_champion_e1_stderr.py`（本批新增，可复用）
   把 `ai?_<label>_seed<N>_<TAG>.stderr.log` 规范化成历史名 → `match_results/champion_e1/stderr/`，
   冠军那份 **gzip -1**（实测 128–143 MB → 14–16 MB），我们那份原样。
5. **清场纪律**：本批没有 kill 任何进程（包括别的实验）；若必须清，**先杀 ladder 再杀 bridge**
   （反过来 `_tmp_ladder.py` 会立刻抢信号量把下一波 token 拉起来 ⇒ 一堆孤儿）。

**为什么这次不会再 timeout**：① 3h 墙钟；② 5 并发（不是 11–12）；③ 保活跑着；④ 看门狗盯着。

---

## 3. E1 结果：16 局（补完）

### 3.1 逐局读数（`match_results/champion_e1/table.md` 原样）

| token | 我方座位 | verdict | 回合 | hp(我,敌) | Δhp | 裁判ill | 我ill | 墙钟(min) | 冠军stderr(gz) | 决策 p50/p95/max (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | P0 | engine | 512 | 2,21 | −19 | 0 | 0 | 43.6 | 16.4MB | 0.55/19.02/35.38 |
| 7r | P1 | engine | 512 | 6,30 | −24 | 0 | 0 | 39.8 | 14.7MB | 0.58/18.25/41.36 |
| 8 | P0 | engine | 512 | 2,31 | −29 | 0 | 0 | 40.3 | 14.6MB | 0.52/17.94/50.72 |
| 8r | P1 | engine | 497 | 0,34 | −34 | 0 | 0 | 35.0 | 14.6MB | 0.52/16.53/29.19 |
| 9 | P0 | engine | 439 | 0,34 | −34 | 0 | 0 | 36.8 ~ | 10.7MB | 0.69/19.20/63.42 |
| 9r | P1 | engine | 338 | 0,29 | −29 | 0 | 0 | 31.6 | 8.9MB | 0.70/25.67/33.22 |
| 10 | P0 | engine | 512 | 10,21 | −11 | 0 | 0 | 45.0 | 13.4MB | 0.62/21.67/31.12 |
| **10r** | P1 | engine | 512 | **24,15** | **+9** | 0 | 0 | 46.9 | 9.7MB | 0.59/21.02/28.84 |
| 11 | P0 | engine | 512 | 8,25 | −17 | 0 | 0 | 44.8 | 12.7MB | 0.61/20.33/32.31 |
| 11r | P1 | engine | 512 | 11,26 | −15 | 0 | 0 | 38.6 | 11.8MB | 0.61/20.67/28.31 |
| 12 | P0 | engine | 506 | 0,25 | −25 | 0 | 0 | 44.3 | 15.2MB | 0.53/19.66/35.30 |
| 12r | P1 | engine | 511 | 0,34 | −34 | 0 | 0 | 36.1 | 14.2MB | 0.55/17.86/26.14 |
| 13 | P0 | engine | 482 | 0,30 | −30 | 0 | 0 | 41.6 | 14.2MB | 0.61/21.53/29.80 |
| 13r | P1 | engine | 471 | 0,33 | −33 | 0 | 0 | 40.9 | 11.8MB | 0.64/20.61/30.83 |
| 14 | P0 | engine | 512 | 12,27 | −15 | 0 | 0 | 38.2 | 12.8MB | 0.56/19.45/30.31 |
| 14r | P1 | engine | 453 | 0,22 | −22 | 0 | 0 | 35.8 | 13.9MB | 0.61/21.77/26.77 |

（`~` = 该局墙钟由日志 mtime 反推：它的 ladder 进程在 2026-10-29 02:16:47 被杀，没来得及打 `rc=.. N min` 那行）

### 3.2 统计（16 局）

| 口径 | 局数 | 胜-负 | 胜率 | Wilson 95%CI | 血量差（我−敌） | t 95%CI |
|---|---|---|---|---|---|---|
| **全部** | 16 | **1W-15L** | **6.2%** | [1.1%, 28.3%] | **−22.62**（sd 11.36） | [−28.68, −16.57] |
| `@P0`（我们执先手） | 8 | 0W-8L | 0.0% | [0.0%, 32.4%] | −22.50 | [−29.34, −15.66] |
| `@P1`（我们执后手） | 8 | 1W-7L | 12.5% | [2.2%, 47.1%] | −22.75 | [−34.86, −10.64] |

- **按座位完全对称**：先手 −22.50 / 后手 −22.75（差 0.25，远小于 SE 2.8）⇒ 没有"座位效应"这类可修的系统偏差。
- 唯一一胜 = **seed 10 的镜像局（`10r`）**，我们执后手、血差 +9。它的同 seed 另一局（`10`，我们执先手）输了 −11
  ⇒ 这一对是"先手互换后结果翻转"（`_tmp_ladder` 的镜像配对读数：seed 10 = 1.0 分，其余 7 对 = 0.0，均值 0.125）。
- 8 对镜像里 AI0 得分 **0.125 / 方差 0.1094**（等强度预期 1.0/0.0）⇒ 我们**远弱于**冠军，不是"座位或运气"。

### 3.3 健康度核查（三条硬判据全过）

| 判据 | 结果 |
|---|---|
| `verdict=engine`（官方 5 级级联，不是 timeout/aborted/hp_tiebreak） | **16/16** ✅ |
| 裁判侧 `illegal=0` | **16/16** ✅ |
| 我方 `illegal_total=0`（我们自己的 stderr，影子校验） | **16/16** ✅ |
| 冠军侧 stderr 留存（证它没秒退/没报 desync） | **16/16**，gzip 后 8.9–16.4 MB/局（原始 90–143 MB） ✅ |
| 僵局嫌疑（全场双方操作数 ≤2） | 无 ✅ |
| 每局墙钟 | min/med/max = **31.6 / 40.0 / 46.9 min**（合计 639.3 min；jobs=5 ⇒ 整批墙钟 110.9 min） |
| 我方决策耗时（每局最后一行取中位） | p50 **0.60s**、p95 **19.99s**、max 63.4s（预算制，不吃墙钟） |

### 3.4 结论

1. **我们（BASE，不注入）对冠军 = 1/16 = 6.2%（Wilson [1.1%, 28.3%]），血量差 −22.62 [−28.68, −16.57]。**
   16 局全部是官方判词、零非法操作 ⇒ 这个基线是可信的。
2. **输法有两半**：8 局我们基地被打爆（HP=0：`8r 9 9r 12 12r 13 13r 14r`），8 局打满 512 回合上限后
   按血量判负（`7 7r 8 10 10r 11 11r 14`，其中只有 `10r` 反过来是我们血多）。
   ⇒ 冠军**不是**靠"终局一波"赢我们，前中期就已经把我们压进"血量永远追不上"的轨道。
3. **行为画像（`match_results/champion_e1/reanalyze.log`，白拿的读数）**：
   - 冠军**每局稳定升级 23–42 次**（`UPGRADE_TOWER` 是它 top-3 之一），而我们**16 局 0 次升级**
     （我们 top-3 恒为 `BUILD_TOWER / DOWNGRADE_TOWER / USE_LIGHTNING_STORM`）。
     这与 `docs/why_no_upgrade_20261028.md` 的结论一致（我们的候选生成器几乎不产生升级），
     也把"升级/经济曲线"推到 E2/E4 之外的第一嫌疑位。
   - 双方出招密度：冠军 193–320 op/局（act 131–215 回合），我们 130–300 op/局（act 84–186 回合）
     ⇒ 冠军**每局都比我们多出 30–70 个 op**，且多出来的那部分集中在上表第 3 列的升级/降级。
4. 与 2026-10-28 那次"5 局 0W-5L、Δhp −28.0"相比，补完 16 局后 Δhp 收敛到 **−22.62**
   （`10r` 一胜 + `10/11/14` 三局血差在 −11..−17）⇒ **只看 5 局的 −28.0 是高估了差距**。

---


## 4. E2：我们 + 注入冠军动作 vs 冠军（16 局）

### 4.1 设计与预注册判据（跑之前登记）

| 项 | 值 |
|---|---|
| 唯一变量 | `AZAI_ORACLE_INJECT=<冠军 exe 的 D:/ 绝对路径>` + `AZAI_ORACLE_TIMEOUT=20` |
| 其余 | 与 E1 **逐字相同**（BASE、seeds 7..14 × 2 座位、对手 = 冠军、`LADDER_TIMEOUT=10800`、`MVS_STDERR_SUFFIX=1`） |
| 并发 | `jobs=6`；`MVS_TIMEOUT_SECONDS=300`（S0 已保活 ⇒ 300s 只当"真死锁"门限） |
| 驱动 | `code/test_match/_e2_inject_vs_champion.sh` → `_us_vs_external.sh`（`EXP=E2`） |
| 逐局硬断言 | ① `verdict=engine`；② 裁判侧 `illegal=0`；③ 我方 `illegal_total=0`；④ stderr 里有 **`注入已启用`**；⑤ `asks>0`；⑥ 冠军侧 stderr 非空 |
| 主读数 | 胜率（Wilson）、血量差（t CI，**与 E1 配对**：同 seeds/座位）、注入采纳率、`dvn`（注入候选 vs 我们自己最佳候选的分差）分布 |
| 预注册判据 | 与 E1 配对 McNemar 精确 `p<0.05` **且** 胜率差 ≥ +12.5pp（2/16）才算"注入有效" |

### 4.2 注入覆盖率（**本节的关键健康指标**）

`[oracle]` 会话有个硬性质：**一次读包超时或进程退出就永久停用**（读端错位），之后整局静默退化成 E1。
所以除了 `asks>0`，还必须报 **覆盖率 = asks / 本局回合数**（`analyze_xoracle.py --tag US_VS_CHAMPION_INJ`）：

| token | 回合数 | asks | 覆盖度 | inject | new | dup(全是 HOLD) | hit | hit_new | timeout | **dead** | oracle p50/p95/max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 512 | 261 | 51.0% | 253 | 78 | 175 | 209 | 34 | 0 | 1 | 79/234/422 |
| 7r | 512 | 295 | 57.6% | 288 | 106 | 182 | 222 | 40 | 0 | 1 | 78/125/188 |
| 8 | 426 | 269 | 63.1% | 261 | 88 | 173 | 204 | 31 | 0 | 1 | 93/172/546 |
| 8r | 512 | 264 | 51.6% | 256 | 99 | 157 | 188 | 31 | 0 | 1 | 78/219/360 |
| 9 | 411 | 259 | 63.0% | 252 | 65 | 187 | 211 | 24 | 0 | 1 | 78/187/250 |
| 9r | 433 | 261 | 60.3% | 253 | 77 | 176 | 214 | 38 | 0 | 1 | 78/156/250 |
| 10 | 457 | 313 | 68.5% | 306 | 73 | 233 | 264 | 32 | 0 | 1 | 78/188/360 |
| 10r | 512 | 261 | 51.0% | 254 | 85 | 169 | 213 | 44 | 0 | 1 | 78/110/188 |
| 11 | 512 | - | - | - | - | - | - | - | - | 1 | -/-/- |
| 11r | 512 | 267 | 52.1% | 260 | 84 | 176 | 210 | 35 | 0 | 1 | 78/218/297 |
| 12 | 512 | 274 | 53.5% | 267 | 87 | 180 | 214 | 34 | 0 | 1 | 78/172/313 |
| 12r | 497 | 256 | 51.5% | 250 | 77 | 173 | 208 | 35 | 0 | 1 | 78/140/219 |
| 13 | 512 | - | - | - | - | - | - | - | - | 1 | -/-/- |
| 13r | 512 | 267 | 52.1% | 259 | 78 | 181 | 212 | 31 | 0 | 1 | 63/140/344 |
| 14 | 512 | 258 | 50.4% | 250 | 69 | 181 | 214 | 33 | 0 | 1 | 78/187/219 |
| 14r | 512 | 271 | 52.9% | 264 | 91 | 173 | 223 | 50 | 0 | 1 | 63/141/297 |

**覆盖度 median 52.5%（min 50.4% / max 68.5%）；`dead=1` 的局 14/16（另 2 局是被硬杀的收尾行缺失）。**

#### 4.2.1 为什么只有一半：冠军当 oracle **会自己崩**（本次新发现的坑）

`match_results/oracle_logs/oracle_seed*.stderr.log` 里 18 个会话都有同一条序列：

```
[MagicaV0Policy] runtime mismatch seed=<N> player=<P> round=<1..11> field=coins[P]        ← 影子模型早早对不上
[PolicyEnv:ERROR] where=ReplacePendingOperations.called_after_terminal round=<250..310> self_player=<P>
                  error=apply_operations.called_after_terminal
terminate called after throwing an instance of 'std::runtime_error'
  what():  apply_operations.called_after_terminal                                          ← 未捕获 ⇒ abort
```

- Windows 侧看到的退出码是 **`3221226505 = 0xC0000409`**（mingw 的 `terminate()` abort），
  `RunnerUpOracle` 把它记成 `OracleDead`，**该局注入就此永久停止**。
- 机制（推测，但证据一致）：magica 内部维护一份**自己的影子局面**，每回合拿真实状态去对；
  它对我们这一侧（对手）的操作只做"预测"而不是照单全收 ⇒ 从第 1 回合起 `coins` 就对不上，
  越走越偏，到第 250–310 回合它的影子局面里"有一方基地已经死了" ⇒ 我们下一次喂操作时它抛
  `apply_operations.called_after_terminal` ⇒ abort。
- **影响**：E2 的"剂量"只有 ~半局。所以本节的读数应当理解为 **"半程注入"的效果**，
  而不是"整局注入"的效果（§4.4 会据此写结论）。
- **对 E4 的直接影响（必读）**：E4 的候选集 = {冠军那一手, 亚军那一手}，冠军会话在第 ~250 回合崩掉后
  候选集就只剩亚军那一手 ⇒ **后半局退化成 E4-only-runnerup**。⇒ 跑 E4 之前必须先解决：
  ① 每 ~100 回合**重启** oracle 会话（kill + 重新 `ReadInit` + 用当前局面继续），或
  ② 只在前 200 回合记 E4 的读数（把"局"截成 200 回合），或
  ③ 换一个**不会崩**的冠军实现（`main.cpp` 重编译、修掉影子模型对账）。
  ⚠️ 不建议"就这么跑"：那样 E4 与 E4-rand 的差别会被稀释到无法判读。

### 4.3 结果

| 口径 | 局数 | 胜-负 | 胜率 | Wilson 95%CI | 血量差 | t 95%CI |
|---|---|---|---|---|---|---|
| **E2 全部** | 16 | **4W-12L** | **25.0%** | [10.2%, 49.5%] | **−17.44**（sd 14.02） | [−24.91, −9.97] |
| `@P0`（我们执先手） | 8 | 1W-7L | 12.5% | [2.2%, 47.1%] | −25.25 | [−35.17, −15.33] |
| `@P1`（我们执后手） | 8 | 3W-5L | 37.5% | [13.7%, 69.4%] | −9.62 | [−19.56, +0.31] |

**与 E1 的配对比较（同 token = 同图同座位）**：

| 量 | 读数 |
|---|---|
| 胜负表 | 两臂都赢 **0** / 只 E1 赢 **1**（`10r`）/ 只 E2 赢 **4**（`7r 8r 11r 13`）/ 都输 **11** |
| McNemar 精确双侧 | **p = 0.375**（不显著） |
| 胜率差 | E2 − E1 = **+18.8pp**（6.2% → 25.0%） |
| 配对血量差（E1 − E2） | **−5.19**（sd 17.49，SE 4.37），t 95%CI **[−14.50, +4.13]** ⇒ 区间跨 0 |

**预注册判据**（§4.1：McNemar p<0.05 **且** ≥+12.5pp）：**满足幅度（+18.8pp）、不满足显著性（p=0.375）**
⇒ 按预注册口径，结论只能是 **"方向为正、不可分辨"**，不能宣称"注入有效"。

**注入本身的读数（16 局合计）**：

| 量 | 读数 | 说明 |
|---|---|---|
| `注入已启用` | 16/16 | 路径是 `D:/` 形态（16/16），没有静默失效 |
| `asks` 总数 | 4276 | 平均 267/局 = 覆盖 54.4% |
| `new` / `asks` | 1324 / 4180 = **31.7%** | 冠军那一手**不在**我们候选集里的比例（其余 68% 是"HOLD 我们本来就有"） |
| `hit_new / new` | **43.1%** | 真·增量候选里，我们最终采纳的比例（与亚军那批的 45.7% 几乎相同） |
| `hit / asks` | 81.9% | 含"HOLD 也算一致"的虚高值，**别当采纳率读** |
| oracle 单次延迟 | avg **92.5ms**、p50 78ms、p95 164.5ms、max 546ms | 20s 阈值绰绰有余；`timeout=0` 全批 |
| `dvn`（冠军那一手 减 我们最佳网络候选） | 中位 **−0.0047**（p25 −0.0312 / p75 +0.0118） | 我们的评估器**略微更偏爱自己的候选**（差得很小） |
| 冠军提出的非法 op | 102 / 4276 = **2.4%** | 我们的影子校验判它非法（它自己的协议下合法） |

### 4.4 判读

1. **E2 的正向信号是真的、但没有统计力**：4W-12L vs E1 的 1W-15L，配对血量差 +5.19（E2 更好），
   幅度 +18.8pp 正好等于预注册门槛（2/16），但 McNemar p=0.375 ⇒ **不能宣称"注入冠军动作有效"**。
2. **而且剂量只有一半**（§4.2.1）：冠军会话在第 ~250 回合崩掉，之后 250 回合退化成 E1。
   所以 25.0% 应读成 **"半程注入 = 25.0%"**；若修掉崩溃（重启会话），还有上升空间——**这是下一步最省力的改进**。
3. **座位不对称，且方向与 E1 相反**：E1 两边都是 −22.7（对称）；E2 是 `@P1 37.5% / −9.62` vs
   `@P0 12.5% / −25.25`。也就是说"注入冠军那一手"在我们**执后手**时明显更有用
   （后手本来就是"看对手出招再反应"，借它的招最直接）；执先手时反而把血差从 −22.50 拉到 −25.25。
   ⚠️ n=8/座位，这只能当**假设**（下一步预注册：`@P1` 单独跑 32 局）。
4. **增量信息本来就少**：冠军 68% 的回合是 HOLD（且 HOLD 已在我们的候选里）⇒ 真正"新信息"的回合只有 31.7%，
   其中我们采纳 43.1% ⇒ 全局限 ≈ 13.7% 的决策被真正改变。这个上限不解决，E2/E4 都不可能有大效应。
5. 与亚军那批对照：注入亚军动作时胜率 12.5% → 50.0%（配对 Δhp +20.7，p≈0.002）；
   **注入冠军动作只有 6.2% → 25.0%（配对 Δhp +5.19，p 不显著）** ⇒ 至少在这 16 局里，
   "借冠军的招"不如"借亚军的招"——与"亚军那 77.9% HOLD、我们与它 45.7% 一致"的既往读数一致
   （亚军的动作更贴近我们自己的动作分布，借过来更容易被我们的评估器接住）。

## 5. E4 / E4-rand：候选只留「冠军动作 + 亚军动作」（**预注册 + 仪表就绪，本会话没跑完**）

### 5.1 这条线要回答什么

E2 只是"多一个候选"；E4 是**把我们的网络候选整个拿掉**，候选集 = {冠军那一手, 亚军那一手}，
由**我们自己的评估器**（flat rollout 叶值）挑一个。于是：

| 臂 | 候选集 | 挑法 | 含义 |
|---|---|---|---|
| **E4** | 只有额外候选（`AZAI_ONLY_EXTRA=1`） | 我们的评估器 argmax | "评估器从 2 个外部候选里挑对"的**上限** |
| **E4-rand** | 同上（`AZAI_ONLY_EXTRA_RANDOM=1`） | **随机**挑一个（固定种子可复现） | 同一候选集**不看分数**的对照 |
| **E1/基线** | 我们自己的 48 束（去重后常只剩 1 条） | 评估器 | 现役水平（§3） |

两臂各 16 局，对手分别跑**冠军**和**亚军** ⇒ 4 个 tag：
`ONLYEXTRA_VS_CHAMPION` / `ONLYEXTRA_RAND_VS_CHAMPION` / `ONLYEXTRA_VS_RUNNERUP` / `ONLYEXTRA_RAND_VS_RUNNERUP`。

判据（**跑之前登记**）：E4 vs E4-rand（配对，同 seeds/座位）**McNemar 精确 p<0.05 且胜率差 ≥ +12.5pp** 才算
"评估器在这类选择上有贡献"；否则"不可分辨"。

### 5.2 ⚠️ 为什么"E4 ≈ E4-rand"**不能**直接读成"评估器没用"

用户的判断是对的：**动作之间的价值差本来就小**。要让两臂产生分歧，必须同时满足两件事：
① 两个候选**真的不同**（`agree=0`，即冠亚军不是同一手）；② 我们的评估器给它们的分**有明显差别**。
只满足①的两个候选，若是"两个都差不多好"，随机挑和挑高的**期望收益一样** ⇒ 两臂本来就该打平。

所以本次新增了仪表 `[dscore]`（`az_bridge_ai.py`，**只读、默认零日志**），逐决策记：

```
[dscore] round=R n_extra=K n_net=N d=|Δscore| dvn=额外−我们最佳网络候选 pick=hi|lo|net|only1|na s_extra=[...] s_netmax=...
[dscore-sum] rounds=N two_cand=N one_cand=N pick_hi=N pick_lo=N pick_net=N d_p25=.. d_med=.. d_p75=.. d_p90=.. dvn_p25=.. dvn_med=.. dvn_p75=..
```

分析口径（`analyze_xoracle.py` 已实现，直接出表 + JSON）：

| 量 | 从哪来 | 用途 |
|---|---|---|
| `N_rounds` | `[xor]`/`[dscore]` 行数 | 有记录的决策数 |
| `N_pair_differs` | `[dscore] … n_extra≥2` 或 `[xor] agree=0` | **两臂才可能不同**的决策数 |
| `N_one_cand` | `[dscore] … n_extra=1` | 两臂**按构造完全相同**（必须剔除） |
| `N_both_hold` | `[xor] both_hold=1` | 同上（都 HOLD，两臂一样） |
| `|Δscore|` 分位 | `[dscore] d=` | 中位 / p25 / p75 / p90（**检验力**：分差有多小） |
| `N_high_dscore` | `d ≥ p75` | **高 Δscore 子集**（评估器选对/选错最可能体现在结果上） |
| `pick_hi` 比例 | `[dscore] pick=` | **正向对照**：E4 应≈100%、E4-rand 应≈50%（偏离 ⇒ 臂没按预期跑） |

**分层比较**（报告 §5.4 会照这个写）：
- 全样本：E4 vs E4-rand 的胜率 / 血量差（配对 McNemar + 配对 Δhp 的 t CI）；
- **高 Δscore 子集**：按每局"高 Δscore 决策数"给局分档（或按该局 `d` 的中位数分档），
  在**高档组**里再比一次两臂；
- 若高差子集上也无差别 ⇒ 才能说"评估器在这类选择上没贡献"；
- ⚠️ 诚实的限制：两臂的对局**不是同一个游戏**，所以"决策级反事实"做不到，
  高差子集的分层只是**探索性**（会明确标注），不是随机化证据。

### 5.3 冒烟实测（已做，证明仪表真的在跑）

`EXP=E4 OPP=runnerup ROUNDS=40 JOBS=1 SEEDS="7"`（`_us_vs_external.sh`）：

| 量 | 读数 |
|---|---|
| `rounds`（有记录决策） | 40 |
| **`both_hold`** | **38/40 = 95%** ⇒ 两臂按构造相同 |
| `N_pair_differs`（n_extra=2） | **2/40 = 5%** |
| 那 2 个决策的 `|Δscore|` | 0.0421 / 0.0466 |

⇒ **这条线的检验力瓶颈不在"局数"，而在"可分辨决策数"**：按 5% 算，16 局 ≈ 8 个可分辨决策/局 ≈ 128 个，
而其中"我们自己也觉得差得多"的（≥p75）只有 ~32 个 ⇒ 想在 E4 vs E4-rand 上检出胜率差，
**要么提高可分辨比例（换更强的对手组合/更长的局），要么接受"只能报区间与是否同向"**。

### 5.4 一键复现（本会话留给下一轮，脚本与判据都已就位）

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
# 4 个 tag，各 16 局；jobs=6，MVS_TIMEOUT_SECONDS=300，LADDER_TIMEOUT=10800（驱动内默认）
JOBS=6 RETRY=0 MVS_TIMEOUT_SECONDS=300 bash code/run_logged.sh only_extra_vs_champion \
    bash code/test_match/_e4_only_extra.sh
JOBS=6 RETRY=0 MVS_TIMEOUT_SECONDS=300 OPP=runnerup bash code/run_logged.sh only_extra_vs_runnerup \
    bash code/test_match/_e4_only_extra.sh
JOBS=6 RETRY=0 MVS_TIMEOUT_SECONDS=300 bash code/run_logged.sh only_extra_rand_vs_champion \
    bash code/test_match/_e4r_only_extra_rand.sh
JOBS=6 RETRY=0 MVS_TIMEOUT_SECONDS=300 OPP=runnerup bash code/run_logged.sh only_extra_rand_vs_runnerup \
    bash code/test_match/_e4r_only_extra_rand.sh

# 读数（每臂：逐局表 + Wilson/t CI + 逐局注入/分差诊断）
for T in ONLYEXTRA_VS_CHAMPION ONLYEXTRA_RAND_VS_CHAMPION ONLYEXTRA_VS_RUNNERUP ONLYEXTRA_RAND_VS_RUNNERUP; do
  $P -u code/test_match/_batch_table.py --tag $T \
      --stderr match_results/us_vs_external/$T/stderr --opp-label champion \
      --markdown-out match_results/us_vs_external/$T/table.md
  $P -u code/test_match/analyze_xoracle.py --tag $T \
      --stderr match_results/us_vs_external/$T/stderr      # ← 含 §5.2 的检验力诊断表
done
```

**验收（每个 tag 都必须过）**：`verdict=engine`、裁判侧 `illegal=0`、我方 `illegal_total=0`、
stderr 里**有** `注入已启用` 且 `asks>0`（驱动里已改成硬失败）、`[dscore]` 行数与 `[xor]` 行数同量级、
`pick_hi` 比例符合臂的构造（E4≈100% / E4-rand≈50%）。

---

## 6. 与"我们 vs 亚军"的并排（**不做差**，对手不同）

对照组固定为 `docs/inject_vs_runnerup_20261028.md` §3.3 的 **B 臂**（同 BASE、同 seeds 7..14 × 2 座位、
不注入、16 局）：`2W-14L = 12.5%`（Wilson [3.5, 36.0]），血量差 `−20.12`（t 95%CI [−28.33, −11.92]），
平均 492 回合。两次读数**只并排**，不做配对检验、不做差（§0.2）。

| 对手 | 局数 | 胜-负 | 胜率 [Wilson 95%CI] | 血量差 [t 95%CI] | 平均回合 | 我们升级数 |
|---|---|---|---|---|---|---|
| **亚军**（`ai_cpp_lure_v4`，2026-10-28） | 16 | 2W-14L | **12.5%** [3.5%, 36.0%] | **−20.12** [−28.33, −11.92] | 492 | 0 |
| **冠军**（`magica_v3_O0`，本篇 E1） | 16 | 1W-15L | **6.2%** [1.1%, 28.3%] | **−22.62** [−28.68, −16.57] | 502 | 0 |

读法（三条，别超读）：

1. **方向一致、量级很小**：换成冠军后胜率 12.5% → 6.2%（−6.3pp）、血量差 −20.12 → −22.62（−2.5）。
   两个 CI **大面积重叠**（胜率区间共享 [3.5%, 28.3%]；血量差区间几乎整段重叠）⇒
   **这 32 局不足以证明"冠军比亚军强多少"**，只能说"冠军至少不比亚军弱，且我们两边都远低于 50%"。
2. **同样的 16 局分辨率**：Wilson 区间在 1/16 时上界 28.3%、在 2/16 时上界 36.0% ⇒
   要分辨"冠军 vs 亚军"这 2.5 点血差，需要的局数远大于 16（见 `docs/inject_vs_runnerup_20261028.md` §7 的检验力表）。
3. **共同点比差异更有信息**：对**两个**对手，我们都是 0 次升级、平均 490–500 回合、血差 −20 上下
   ⇒ 我们对"同一条经济曲线"的弱点，与对手是谁基本无关。这条一致性比"冠军比亚军强"更值得当结论用。

---

## 7. 复现清单（全部落盘）

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
# 0) 保活 + 看门狗（跨夜批次**先起保活**）
powershell -NoProfile -ExecutionPolicy Bypass -File code/test_match/_keepawake.ps1 &
powershell -NoProfile -ExecutionPolicy Bypass -File code/test_match/_ladder_watchdog.ps1 \
    -Tag US_VS_CHAMPION_BASE -StallMinutes 12 -IntervalSeconds 120 &

# 1) E1 续跑（只补没有 verdict=engine 的局；已完成的局自动跳过）
bash code/run_logged.sh us_vs_champion_e1_resume bash code/test_match/_us_vs_champion_e1_resume.sh

# 2) 逐局归档（冠军 gzip；幂等，可边跑边用 --max-age-min 跳过在飞的局）
$P -u code/test_match/archive_champion_e1_stderr.py \
    --dst match_results/champion_e1/stderr --src match_results \
    --src "match_results/us_vs_external/_stale/US_VS_CHAMPION_BASE_20261028_234244" \
    --tag US_VS_CHAMPION_BASE

# 3) 读数（逐局表 + 胜率 Wilson + 血量差 t CI + 按座位 + 墙钟 + verdict·illegal 核查）
$P -u code/test_match/_batch_table.py --tag US_VS_CHAMPION_BASE \
    --stderr match_results/champion_e1/stderr --opp-label champion \
    --markdown-out match_results/champion_e1/table.md --json match_results/champion_e1/summary.json

# 4) 行为画像（不重跑，只重新解析已有日志：双方每局 act/ops 直方图）
$P -u _tmp_ladder.py --tag US_VS_CHAMPION_BASE --reanalyze \
    --ai0=code/test_match/az_bridge_ai.py --ai1="其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe" \
    7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r

# 5) E2 / E4（脚本已就绪；见 §4 / §5）
JOBS=6 MVS_TIMEOUT_SECONDS=300 bash code/run_logged.sh us_vs_champion_e2 bash code/test_match/_e2_inject_vs_champion.sh
JOBS=6 MVS_TIMEOUT_SECONDS=300 bash code/run_logged.sh only_extra_vs_champion bash code/test_match/_e4_only_extra.sh
JOBS=6 MVS_TIMEOUT_SECONDS=300 bash code/run_logged.sh only_extra_rand_vs_champion bash code/test_match/_e4r_only_extra_rand.sh
```

---

## 8. 没做 / 没覆盖（诚实清单）

1. **E4 / E4-rand 没跑**（4 个 tag × 16 局 = 64 局，按 33–45 min/局 × 6 并发 ≈ 11–13 小时）：
   本会话把 E1 补完 + E2 跑完（合计 ~3 小时），E4 的**脚本、仪表、预注册判据、检验力诊断**都已就位（§5），
   但**必须先修掉"冠军 oracle 第 250 回合自杀"**（§4.2.1），否则后半局候选集塌成一个、跑出来没法判读。
2. **E2 只跑了一半的剂量**（覆盖 54.4%）：这不是设计，是 §4.2.1 那个崩溃导致的既有事实。
   修法（重启会话 / 截断到 200 回合 / 重编译冠军）都**还没做**。
3. **`@P1` 的座位不对称只是假设**（E2 `@P1` 37.5% vs `@P0` 12.5%，每边 n=8）：要坐实需要
   预注册一个"只跑 `@P1`、32 局"的批次，本会话没跑。
4. **E1 的第 5 局（`9`）墙钟是反推的**（`36.8 min ~`）：它的 ladder 进程在 02:16:47 那一波被杀，
   没来得及打 `rc=.. N min`。其余 15 局都是驱动亲口报的。
5. **我们自己的 stderr 归档少 2 处**：`11`/`13` 两局的 oracle 会话在崩溃前被硬杀，
   `注入统计` 收尾行没写出来（所以覆盖度表里那两行是 `-`）。逐回合 `[xor]`/`[dscore]` 行是完整的。
6. 冠军**不是确定性对手**（§0.3）：同 seed 重跑不保证同样结果，本批只能做"批内"比较。
7. `ANTWAR_MAGICA_LOG_LEVEL` 在 O0 版里无效（`docs/us_vs_champion_20261028.md` §1.2），
   所以它一局仍然写 100+ MB stderr；归档一律 gzip（本批 32 个文件共 ~400 MB）。
8. 保活/看门狗是**本次才补起来的**（08:23 / 08:31 起）；07:56–08:24 那 28 分钟只有
   `STANDBYIDLE(AC)=0` 兜着（事后核对：无 506/507 事件、progress 一直前进 ⇒ 实际没被冻）。
9. **没有**解释"冠军第 250 回合 abort"的**代码级**根因：只从日志定位到
   `runtime mismatch … field=coins[*]` → `apply_operations.called_after_terminal` 这条链（§4.2.1），
   没去读 `main.cpp` 的影子模型对账逻辑。
