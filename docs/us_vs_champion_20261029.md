# 我们 vs 冠军：E1 基线（16 局补完）+ E2 / E4 预注册（2026-10-29）

> 一句话：**E1 = 1/16 = 6.2%（Wilson [1.1%, 28.3%]），血量差 −22.62（t 95%CI [−28.68, −16.57]）**，
> 16 局全部 `verdict=engine`、`illegal=0`（裁判侧 + 我方自检），冠军侧 stderr 16/16 留存。
> 上一批"3 局 timeout + 8 局空壳"的**真正根因是整机进了 Modern Standby（S0）被冻了 2 小时 6 分钟**，
> 不是我们的搜索变慢、也不是对手变慢（§1，附系统事件日志）。
> E2（注入冠军动作）与 E4 / E4-rand（候选只留冠亚军动作）的读数见 §4 / §5。

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


## 4. E2：我们 + 注入冠军动作 vs 冠军（待跑）

## 5. E4 / E4-rand：候选只留「冠军动作 + 亚军动作」（待跑）

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

（跑完回填）

1. **E2 / E4 尚未跑完**（见 §4 / §5）：本会话先把 E1 补完，E2 之后接。
2. E4 的"高 Δscore 子集"分层比较**不是随机化实验**：两臂的对局不同，无法在决策级做反事实，
   所以那一层只能作为**探索性**证据（§5 会写清"什么才算评估器无用的证据"）。
3. 冠军侧不可复现（§0.3）⇒ 同一 seed 重跑不保证同样结果，本批只能做"批内"比较。
4. 冠军 stderr 的 `ANTWAR_MAGICA_LOG_LEVEL` 在 O0 版里无效（`docs/us_vs_champion_20261028.md` §1.2），
   所以它一局仍然写 100+ MB；归档一律 gzip。
