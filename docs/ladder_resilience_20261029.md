# 过夜批次：把「无效局重试 / 早上补跑 / 卡死看门狗 / 收尾汇总表」做进对局驱动本体

**日期**：2026-10-29
**改动**：`_tmp_ladder.py`（**只动驱动层**；不碰 `code/my_ai/**`、不碰 `match_via_sdk.py` 的协议/判负逻辑、
**没动** `code/test_match/ab_search_modes.py`）
**计划**：`docs/ladder_resilience_plan_20261029.md`
**验证工件**：`_tmp_lr_env.sh`（公共 env）· `_tmp_lr_v1.sh` / `_tmp_lr_v2.sh` / `_tmp_lr_v3.sh` /
`_tmp_lr_v4.sh`（四个场景）· `_tmp_lr_cmp.py`（RESULT 逐字段比对）· `_tmp_lr_pids.py`（进程普查，只读）·
`_tmp_lr_faultinject/sitecustomize.py`（故障注入）· `_tmp_noopai.py`（空操作假 AI）·
`_tmp_hangai.py`（`sleep 600` 假 AI）· `_tmp_lr_final.sh`（终版整套复跑）·
`_tmp_ladder_head.py`（`git show HEAD:_tmp_ladder.py` 的副本）

---

## 0. 动机

2026-10-29 那批 E1（`US_VS_CHAMPION_BASE`）的真实形态：**整机进 Modern Standby 冻了 2h06m**
（系统事件 00:10:48 → 02:16:47，见 `docs/us_vs_champion_20261029.md` §1.1）
⇒ 3 局 `verdict=timeout` + 8 局 0 字节日志，没有一局 `engine`。
补跑只能靠外层 shell 脚本再 `grep -q 'verdict=engine'` 拼一遍命令行
（`code/test_match/_us_vs_champion_e1_resume.sh` 里那套）——既容易写错，也看不出"哪几局没成、各试过几次、
到底还剩谁"。这三项能力应该长在驱动里，而不是再加一层包装。

有效性门槛沿用 `code/test_match/analyze_paired.py` §4 的口径：**只有 `verdict=engine` 算有效**，
`timeout / aborted / illegal_strict / hp_tiebreak / 无 RESULT / 无日志` 一律无效 ⇒ 重跑。

## 1. 三个开关（**默认全关 ⇒ 行为与改动前逐位相同**）

| 开关 | 默认 | 作用 |
|---|---|---|
| `--retry-invalid N` | `0`（关） | 一批跑完后扫 `<tag>` 日志目录，凡**没有 `verdict=engine`** 的 token 重跑，最多 N 轮 |
| `--final-sweep` | 关 | 主循环（含重试）结束后**再扫一遍**，仍有非 engine ⇒ 最后再跑一次（**不占** N 轮） |
| `--watchdog-stall-min M` | `0`（关） | 运行期间盯着**自己启动的**对局：progress/日志 mtime 连续 M 分钟不前进 ⇒ 精确 kill 该局 PID 树 |

两个带值的开关同时支持 `--retry-invalid 2` 与 `--retry-invalid=2`（老参数仍然只有 `=` 形式，没动）。

### 1.1 `--retry-invalid N`

- 每轮开头打印：**第几轮** / **待跑 token 列表** / **重跑前的 verdict 计数分布**；
- **只重跑失败的**：清单从"当前无效局"生成，已 `engine` 的 token 永远不会进入；
  重跑前还会再判一次（日志已 `engine` 就跳过并告警）；
- 重跑前把上一轮的失败日志**复制**留证据：`<dir>/_attempts/<token>.attempt<k>.log`
  （`k` = 产下这份日志的那次尝试序号；只复制，不删/不改原日志）；
- 判据覆盖：`timeout` / `aborted` / `illegal_strict` / `hp_tiebreak` / **空壳日志** / **没有 RESULT 行** / **没有日志**；
- 计分口径 = 最后一次尝试（成功局不会再跑，所以重试过的局用的就是它最后一次有效结果）。

### 1.2 `--final-sweep`

- 主循环 + `--retry-invalid` 的 N 轮都用完之后，**再扫一遍**；仍有非 engine 的 token ⇒ 再跑一次；
- **不占** `--retry-invalid` 的轮数（N 轮用完也照样能跑这一轮）；
- 单独一个区块打印（`=== final-sweep（主循环 + 重试 N 轮之后）：K 局没有 verdict=engine ===`），
  不会和重试轮混算；
- 用途：机器夜里休眠 ⇒ 重试轮数在凌晨就被"超时 + 重试"烧完了，这一轮通常在**早上**执行，把漏的补完。

### 1.3 `--watchdog-stall-min M`

- **只盯本进程启动的对局**（`Popen` 句柄登记在 `_WATCH` 表里，一局一登记、结束就注销）；
- 判定依据：该局的 progress 文件（`training_history/progress/<tag>/az_progress_*<token>*.txt`）
  **或**该局日志的 mtime 连续 M 分钟不前进（取"更活跃"的那个 ⇒ 日志被缓冲时不会误判）；
- 命中 ⇒ 标记 `verdict=timeout_watchdog`，并按 ① 的规则参与重试；
- 动作顺序**严格**是：**① 先停调度（`_SCHED_RUN.clear()`）→ ② `taskkill /PID <pid> /T /F` →
  ③ 用已死的根 PID 复查后代闭包 → ④ 恢复调度**。
  踩过的坑：先杀对局会让驱动立刻抢信号量、把下一波 token 拉起来——所以 `run_one` 在**信号量内部**
  等这道闸门，停调度期间不会起任何新局；
- **绝不按镜像名/命令行通配符批量杀**：只用自己 `Popen` 的那一个 PID（`/T` 带它的子进程）。
  `taskkill /IM python.exe` 那种写法会把本机**别人正在跑的对局**一起杀掉（2026-10 刚踩过）；
- 已终局的局**不再更新** progress 文件 ⇒ mtime 会一直变老，必须靠最后一行的 `terminal` 排除
  （否则会对已完成的局狂杀）——判据沿用 `code/test_match/_ladder_watchdog.ps1`；
- 默认关：它是"停滞"判据，**不是死锁检测**，一局合法但很慢的对局也会被杀（阈值必须明显大于单回合耗时）。

### 1.4 收尾汇总表（④）

所有开关用完、结算前打印（**任一开关打开才打印**，全关时不打印 ⇒ 默认 stdout 与改动前一致）：

```
=== 收尾汇总（每个 token 一行；判有效 = 日志里 verdict=engine）===
token   | verdict          | rounds  | 墙钟(min)  | 尝试次数  | 是否有效
--------------------------------------------------------------------------
992     | engine           |     178 |        0.8 |         2 | ✓ 有效
992r    | engine           |     207 |        0.6 |         1 | ✓ 有效

engine 2/2
仍无效的 token: 无（全部 engine）
```

- `墙钟(min)` = **各次尝试耗时之和**（重跑过的局显示"总共花了多久"，不是最后一次）；
- `rounds` 取自该局日志的 RESULT 行（日志没有 RESULT 行时是 `-`）；
- 最后单独列出**仍无效的 token**（带判词）——早上人工补跑就照这个列表给命令行。

## 2. 实现要点 / 踩到的坑

1. **`subprocess.run` → `Popen`**（看门狗必须拿到**自己的** PID）。超时分支仍按旧语义：
   只杀直接子进程、`rc=-2`、只在 `TimeoutExpired` 时触发 ⇒ 默认路径读数与改动前一致。
2. **调度闸门 `_SCHED_RUN`**（`threading.Event`，默认 open）。`run_one` 在
   `with sem:` **内部** `_SCHED_RUN.wait()` ⇒ 停调度期间即使有游戏刚结束释放了信号量，也不会拉起下一个 token。
3. **进程表用 `CreateToolhelp32Snapshot`**（ctypes，只读、毫秒级），不用 `wmic`/PowerShell
   （`wmic` 在新 Windows 上已被移除，PowerShell 一次 100–500ms）。
   `Windows 不会重新挂父`：根进程死后孤儿的 PPID 仍指向那个**已死的 PID**
   ⇒ "用已死根 PID 复查后人闭包"是可行的，这也是杀后还能继续用 `_descendants(pid)` 的依据。
4. **终局误杀**：bridge 终局后只写一行 `terminal …`，progress 文件不再更新 ⇒ 必须先看最后一行，
   见到 `terminal` 就跳过（`_ladder_watchdog.ps1` 踩过这个坑）。
5. **token 前缀的 glob 陷阱**：`az_progress_*10*.txt` 会把镜像局 `10r` 的文件一起捞进来
   ⇒ 拿**别人那局**的 mtime 判卡死。`_progress_candidates()` 的判据是
   "seed 段以 token 开头，且 token 之后紧跟的东西**不是字母/数字**"（`10` 后面只能是 `_`/结尾）。
   本批实际目录里 `9/9r/10/10r/...` 并存，用朴素 glob 一定会错。
6. **日志优先**：被 kill 时如果日志里已经是 `verdict=engine`（进程写完 RESULT 才退出），以日志为准算有效，
   不把好局误判成超时；反过来，日志没有 RESULT 行时才用进程内记下的杀因（`timeout_watchdog` / `timeout_hard`）。
7. **进程内记忆的判词只做兜底**：判"这局有没有效"永远以日志为准（`_MEM_VERDICT` 只在日志没有判词时用）。
8. 汇总表的中文表头用 `% -Ns` 对不齐（CJK 占 2 列）⇒ 加了个按**显示宽度**补空格的 `_pad()`。
9. **重跑前再判一次**（`run_batch(..., is_retry=True)`）：重试轮 / final-sweep 真正启动前，
   对清单里的每个 token 再读一次日志 —— 已经是 `engine` 就跳过并告警
   （`[retry] <token> 已有 verdict=engine（有效）⇒ 跳过，不覆盖`）。这是"成功局绝不被覆盖"的第二道防线
   （第一道是重跑清单只由"当前无效局"生成）。正常批次里这个分支不会触发，所以四个验证里都没走到它。

## 3. 验证（全部 `bash code/run_logged.sh <名字> <命令>` 落盘）

> ⚠️ 验证期间机器上**同时有一批真实的 16 局**（`US_VS_CHAMPION_BASE`, jobs=5）+ 一个 40 worker 的
> coevo 在跑 ⇒ 验证只用 1–2 局、`--jobs=1`，**没有 kill 任何别人的进程**（下面 §3.5 有进程普查）。
>
> **对局选择**（被现场实测逼出来的）：驱动层验证只需要"跑真对局、拿到 `verdict=engine`、别把机器占满"。
> 试过两种更慢的：① `_illegal_smoke_fast.sh` 的 FAST bridge 档 ⇒ **双方每回合都空过**，
> 512 回合僵局、`verdict=hp_tiebreak`（不是 engine）、单局 ~34 min；② 亚军 C++ ×2 ⇒ 每回合 ~3s、单局 ~25 min。
> 最终用 **`code/test_match/rv4_pkg/main.py`（rule_v4）vs `_tmp_noopai.py`（空操作假 AI）**：
> noop 不防守 ⇒ rule_v4 在 ~180–230 回合推平基地 ⇒ **引擎真的判终局**、单局 ~40s、纯 CPU 不占 GPU。
> 驱动层代码与"用哪两个 AI"无关，结论不受影响。

### 3.1 验证 1：默认行为不变（`_tmp_lr_v1.sh`）

HEAD 版驱动（`_tmp_ladder_head.py`，358 行）与新驱动（960 行）各跑同一 2 局（token `920` / `920r`）；
HEAD 版**跑两遍**先证明这个对局本身是可复现的（否则"新旧一样"可能只是"两次运行本来就会飘"）。

| 运行 | 命令 | 日志 |
|---|---|---|
| V1-old | `python -u _tmp_ladder_head.py --tag=LR_V1_OLD --jobs=1 --ai0=…/rv4_pkg/main.py --ai1=_tmp_noopai.py 920 920r` | `training_history/runs/20261029_091544_ladder_resilience_v1_old/output.log` |
| V1-old2 | 同上，`LR_V1_OLD2` | `…/20261029_091701_ladder_resilience_v1_old2/output.log` |
| V1-new | `python -u _tmp_ladder.py --tag=LR_V1_NEW …`（**不加任何新开关**） | `…/20261029_092758_ladder_resilience_v1_new/output.log（终版）` |

`_tmp_lr_cmp.py` 的逐字段输出（(a) 可复现性，(b) 新旧对比）：两份都 `==> RESULT 逐字段相同: True`

```
=== token 920 ===
  RESULT 行整行相同: True
    ✓ seed 920 | p0 main | p1 _tmp_noopai | winner main | winner_side p0 | verdict engine
    ✓ engine_winner 0 | rounds 180 | terminal True | base_hp 32,0 | coins 41,497
  OLD: RESULT seed=920 p0=main p1=_tmp_noopai winner=main winner_side=p0 verdict=engine engine_winner=0 rounds=180 terminal=True base_hp=32,0 coins=41,497 illegal=0
  NEW: RESULT seed=920 p0=main p1=_tmp_noopai winner=main winner_side=p0 verdict=engine engine_winner=0 rounds=180 terminal=True base_hp=32,0 coins=41,497 illegal=0

=== token 920r ===
  RESULT 行整行相同: True
    ✓ seed 920 | p0 _tmp_noopai | p1 main | winner main | winner_side p1 | verdict engine
    ✓ engine_winner 1 | rounds 221 | terminal True | base_hp 0,31 | coins 570,18
  OLD: RESULT seed=920 p0=_tmp_noopai p1=main winner=main winner_side=p1 verdict=engine engine_winner=1 rounds=221 terminal=True base_hp=0,31 coins=570,18 illegal=0
  NEW: RESULT seed=920 p0=_tmp_noopai p1=main winner=main winner_side=p1 verdict=engine engine_winner=1 rounds=221 terminal=True base_hp=0,31 coins=570,18 illegal=0

新版 stdout（20261029_092758_ladder_resilience_v1_new）里的新开关区块: 无
==> RESULT 逐字段相同: True
```

两个驱动 stdout 的"逐局表"也逐字相同：

```
token  p0         p1         winner     verdict      rounds hp          ill  p0 act/ops …
920    main       _tmp_noopai main       engine       180    (32, 0)     0    25/25 {'BUILD_TOWER': 8, 'UPGRADE_TOWER': 7, 'USE_LIGHTNING_STORM': 5} 0/0 {}
920r   _tmp_noopai main       main       engine       221    (0, 31)     0    0/0 {}                         36/36 {'BUILD_TOWER': 12, 'UPGRADE_TOWER': 9, 'DOWNGRADE_TOWER': 9} 0
```

**结论：默认路径（三个开关全关）逐字段不变，stdout 也没有多出任何区块。**

### 3.2 验证 2：`--retry-invalid 1` 只重跑失败的那一局（`_tmp_lr_v2.sh`）

故障注入：`_tmp_lr_faultinject/sitecustomize.py` 让**第 1 个**对局进程（`match_via_sdk.py`）在启动瞬间
`os._exit(3)`——等价"一个会立刻退出的假 AI"：没有 RESULT 行、日志是空壳。
（用 `sitecustomize` 而不是换掉 `az_bridge_ai.py`/换 `--ai0`，是因为前者会污染别人正在跑的批次，
后者没法让"第 K+1 次"变回一局真对局。）

`training_history/runs/20261029_092913_ladder_resilience_v2_retry/output.log`：

```
[ladder] 过夜批次开关: retry_invalid=1  final_sweep=False  watchdog_stall_min=0
[992] rc=3  0.0 min
[992]   第 1 次尝试：verdict=no_result  ⇒ 无效，等重试  墙钟累计 0.0 min
[992r] rc=0  0.5 min
[992r]   第 1 次尝试：verdict=engine  墙钟累计 0.5 min

=== 重试第 1/1 轮：1 局没有 verdict=engine ===
    重跑前 verdict 计数分布: {'no_result': 1}
    待跑 token: 992
[retry] 992 的失败日志已留档: match_results/ladder_logs/LR_V2_RETRY/_attempts/992.attempt1.log
[992] rc=0  0.6 min
[992]   第 2 次尝试：verdict=engine  墙钟累计 0.6 min

[ladder] 全部结束 wall=68.1s  单局 min/med/max = 0.5/0.6/0.6 min

=== 收尾汇总（每个 token 一行；判有效 = 日志里 verdict=engine）===
token   | verdict          | rounds  | 墙钟(min)  | 尝试次数  | 是否有效
--------------------------------------------------------------------------
992     | engine           |     178 |        0.6 |         2 | ✓ 有效
992r    | engine           |     207 |        0.5 |         1 | ✓ 有效

engine 2/2
仍无效的 token: 无（全部 engine）
```

落盘证据（`ls --time-style=full-iso`）：

```
match_results/ladder_logs/LR_V2_RETRY/992.log    2026-10-29 09:30:23.458  (83798 B，重跑后)
match_results/ladder_logs/LR_V2_RETRY/992r.log   2026-10-29 09:29:48.085  (93733 B，**没被动过**)
match_results/ladder_logs/LR_V2_RETRY/_attempts/992.attempt1.log   64 B
  内容: [faultinject] 第 1/1 次对局进程被注入 ⇒ os._exit(3)
```

读数要点：
1. **只重跑了 1 局**（`待跑 token: 992`），`992r` 的 mtime 停在它自己跑完的时刻（09:29:48）、
   比 `992.log` 重跑完（09:30:23）早 35s ⇒ **已 engine 的日志没被覆盖**；
2. `_attempts/` 里**只有** `992.attempt1.log`（失败那局的空壳日志，64 字节，无 RESULT 行）；
3. 收尾表里两局 `尝试次数` 分别是 2 / 1 ⇒ 好的那局确实只跑了一次。

### 3.3 验证 3：`--final-sweep` 在重试轮数用完后补跑（`_tmp_lr_v3.sh`）

注入改成"**前 4 个**对局进程立刻退出"：第一批 2 局 + 重试第 1 轮 2 局全废 ⇒ 重试轮数用完、仍 2 局无效
⇒ `--final-sweep` 再跑一次（这时注入已用完 ⇒ 真对局 ⇒ engine）。

`training_history/runs/20261029_093024_ladder_resilience_v3_final_sweep/output.log`：

```
[ladder] 过夜批次开关: retry_invalid=1  final_sweep=True  watchdog_stall_min=0
[994] rc=3  0.0 min     …  第 1 次尝试：verdict=no_result  ⇒ 无效，等重试
[994r] rc=3  0.0 min    …  第 1 次尝试：verdict=no_result  ⇒ 无效，等重试

=== 重试第 1/1 轮：2 局没有 verdict=engine ===
    重跑前 verdict 计数分布: {'no_result': 2}
    待跑 token: 994, 994r
[retry] 994 的失败日志已留档: …/_attempts/994.attempt1.log
[retry] 994r 的失败日志已留档: …/_attempts/994r.attempt1.log
[994] rc=3  0.0 min     …  第 2 次尝试：verdict=no_result  ⇒ 无效，等重试
[994r] rc=3  0.0 min    …  第 2 次尝试：verdict=no_result  ⇒ 无效，等重试

[retry] ⚠️ 已用满 1 轮，仍有 2 局无效: 994(no_result), 994r(no_result)

=== final-sweep（主循环 + 重试 1 轮之后）：2 局没有 verdict=engine ===
    当前 verdict 计数分布: {'no_result': 2}
    待跑 token: 994, 994r
    （这一轮通常在早上执行：机器夜里休眠导致超时、醒来时重试轮数已用完 ⇒ 把漏的补完；不占 --retry-invalid 的轮数）
[retry] 994 的失败日志已留档: …/_attempts/994.attempt2.log
[retry] 994r 的失败日志已留档: …/_attempts/994r.attempt2.log
[994] rc=0  0.8 min     …  第 3 次尝试：verdict=engine  墙钟累计 0.8 min
[994r] rc=0  0.6 min    …  第 3 次尝试：verdict=engine  墙钟累计 0.6 min

[ladder] 全部结束 wall=84.9s  单局 min/med/max = 0.6/0.8/0.8 min

=== 收尾汇总 …
994     | engine           |     207 |        0.8 |         3 | ✓ 有效
994r    | engine           |     228 |        0.6 |         3 | ✓ 有效
engine 2/2
```

`_attempts/` 里 4 份留档：`994.attempt1.log`、`994.attempt2.log`、`994r.attempt1.log`、`994r.attempt2.log`
（attempt1 = 重试前、attempt2 = final-sweep 前）⇒ **重试轮数（1 轮）没有被 final-sweep 占用**，两轮都留了证据。

### 3.4 验证 4：`--watchdog-stall-min 1` 精确杀掉卡住的那一局（`_tmp_lr_v4.sh`）

卡住的构造：`--ai0=_tmp_hangai.py`（假 AI，`sleep 600`）+ `--ai1=_tmp_noopai.py` ⇒ 该局卡在 round 0，
progress/日志 mtime 都不再前进。**同时并发**跑一个**对照批次**（真对局、没有任何新开关）⇒ 证明"别的局不受影响"。

`training_history/runs/20261029_093158_ladder_resilience_v4_watchdog/output.log`（节选，逐字）：

```
[ladder] 过夜批次开关: retry_invalid=1  final_sweep=False  watchdog_stall_min=1
[watchdog] 启动：tag=LR_V4_WD  阈值 1 min（progress/日志 mtime 连续这么久不前进 ⇒ 判卡死）  每 10s 检查一次
[watchdog] 只按本进程 Popen 的**那一个 PID** 精确杀（taskkill /PID <pid> /T /F）；先停调度再杀，杀完用已死根 PID 复查后代闭包
[watchdog] 命中 1 局 ⇒ 先停调度（_SCHED_RUN=closed），再逐局精杀
[watchdog] 995: progress/日志已 1.0 min 没有更新（阈值 1 min） ⇒ 精确 kill PID 67792 及其子进程（taskkill /PID 67792 /T /F，不碰其它进程）
[watchdog] 995: 杀前闭包 3 个 PID: [12224, 37072, 67792]
[watchdog] taskkill rc=0: SUCCESS: The process with PID 37072 (child process of PID 67792) has been terminated.
[watchdog] taskkill rc=0: SUCCESS: The process with PID 12224 (child process of PID 67792) has been terminated.
[watchdog] taskkill rc=0: SUCCESS: The process with PID 67792 (child process of PID 46396) has been terminated.
[watchdog] 995: 用已死根 PID 67792 复查后代闭包 ⇒ 0 个存活（干净）
[watchdog] 恢复调度（_SCHED_RUN=open）
[995] rc=1  1.2 min
[995]   第 1 次尝试：verdict=timeout_watchdog  ⇒ 无效，等重试  墙钟累计 1.2 min

=== 重试第 1/1 轮：1 局没有 verdict=engine ===
    重跑前 verdict 计数分布: {'timeout_watchdog': 1}
    待跑 token: 995
[retry] 995 的失败日志已留档: …/_attempts/995.attempt1.log
[watchdog] 命中 1 局 ⇒ 先停调度（_SCHED_RUN=closed），再逐局精杀
[watchdog] 995: progress/日志已 1.0 min 没有更新（阈值 1 min） ⇒ 精确 kill PID 5768 及其子进程（taskkill /PID 5768 /T /F，不碰其它进程）
[watchdog] 995: 杀前闭包 3 个 PID: [5768, 20988, 59288]
… taskkill 三条 SUCCESS …
[995] rc=1  1.2 min
[995]   第 2 次尝试：verdict=timeout_watchdog  ⇒ 无效，等重试  墙钟累计 2.4 min

[retry] ⚠️ 已用满 1 轮，仍有 1 局无效: 995(timeout_watchdog)
[watchdog] 995: 用已死根 PID 5768 复查后代闭包 ⇒ 0 个存活（干净）
[watchdog] 恢复调度（_SCHED_RUN=open）
[watchdog] 停止

[ladder] 全部结束 wall=141.9s  单局 min/med/max = 1.2/1.2/1.2 min

=== 收尾汇总（每个 token 一行；判有效 = 日志里 verdict=engine）===
token   | verdict          | rounds  | 墙钟(min)  | 尝试次数  | 是否有效
--------------------------------------------------------------------------
995     | timeout_watchdog |       - |        2.4 |         2 | ✗ 无效

engine 0/1
仍无效的 token（早上人工补跑就照这个列表给命令行）: 995(timeout_watchdog)
```

对照批次（同一时刻在跑的真对局，`training_history/runs/20261029_093153_ladder_resilience_v4_ctl/`）：

```
[run_logged] exit=0  (37s)
RESULT seed=996 p0=main p1=_tmp_noopai winner=main winner_side=p0 verdict=engine engine_winner=0
       rounds=176 terminal=True base_hp=38,0 coins=21,431 illegal=0
```

读数要点：

1. **~1 分钟命中并精确杀了 3 个 PID**（`match_via_sdk` + 假 AI + noop AI），都是自己 `Popen` 那棵树的，
   杀前打印了完整闭包、杀后用**已死的根 PID** 复查 ⇒ **0 个存活**；
2. **先停调度再杀**：日志顺序是 `命中 ⇒ 先停调度` → kill → `复查后代闭包` → `恢复调度`；
3. 被杀那局标 `verdict=timeout_watchdog`，并按 ① 的规则进入重试（第 2 次尝试同样被杀）；
4. **同批/同机别的局不受影响**：并发的对照批次在 37s 跑完、`verdict=engine`；
5. 收尾表把这一局报成 `✗ 无效` + `engine 0/1` + 仍无效清单（不会假装成功）。

### 3.5 进程安全：没有 kill 任何别人的进程

跑前/跑后各查一次别人那批真实对局的主 PID（`_tmp_lr_pids.py`，只读 `CreateToolhelp32Snapshot`）：

```
跑前: [census] PID 58044: alive=True  存活后代 3 个: [31800, 43216, 60964]
跑后: [census] PID 58044: alive=True  存活后代 3 个: [31800, 43216, 60964]
```

`58044`（`US_VS_CHAMPION_BASE` 的 ladder 进程）**跑前跑后都活着，连后代 PID 都是同一组**
（更早那次 09:12 的运行时它是 6 → 3：少掉的是**它自己的**两局在这 147s 里跑完了，不是我杀的）。
本次全部 kill 只出现在上面的 watchdog 日志里，每一条都带精确 PID，且都是我的 ladder 的子进程。
（另外：`_tmp_lr_v4.sh` 结尾只对**假 AI 自己写进 pidfile 的 PID** 做兜底清理，两个 PID 都是 `alive=False`
⇒ 看门狗没有漏网。）

## 4. 怎么用（过夜一条命令）

```bash
# 夜里跑：单局 3h 墙钟、卡死 90min 就看门狗杀掉、跑完把无效局重试 2 轮、早上再补最后一次
# （AZAI_* 部署配置仍由外层脚本 export；驱动只管调度/判有效/补跑）
LADDER_TIMEOUT=10800 MVS_TIMEOUT_SECONDS=1800 \
D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_ladder.py \
    --tag=US_VS_CHAMPION_BASE --jobs=5 \
    --ai0=code/test_match/az_bridge_ai.py \
    --ai1="其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe" \
    --retry-invalid 2 --final-sweep --watchdog-stall-min 90 \
    7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r
```

- **阈值怎么取**：`--watchdog-stall-min` 必须明显大于"单回合最坏耗时"（冠军 O0 那批实测单回合 ~4s、
  单局 30–45 min ⇒ 60–120 是安全区；取 5 这种小值会把正常局也杀掉，见 §5.2）。
- **早上怎么补**：驱动没有 `--resume`（见 §5.1）。看昨晚最后那张 `收尾汇总` 的
  "仍无效的 token" 一行，把它喂回去（同一 tag ⇒ 日志目录不变）：

  ```bash
  LADDER_TIMEOUT=10800 MVS_TIMEOUT_SECONDS=1800 \
  D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_ladder.py \
      --tag=US_VS_CHAMPION_BASE --jobs=2 --ai0=… --ai1=… \
      --retry-invalid 1 --final-sweep 9r 12
  ```
- 也可以只用 `--final-sweep`（不重试）：主循环跑完直接补最后一轮。

## 5. 已知限制（哪些情况救不了）

1. **进程内重试救不了断电/关机**：`--retry-invalid` / `--final-sweep` 只在驱动进程活着时有效；
   断电、蓝屏、任务被杀之后只能**人工补跑**（把汇总表里的"仍无效 token"列表喂回命令行）。
   本驱动**没有 `--resume`**（这次刻意不加，避免超范围改"第一批跑什么"的语义）——
   要全自动续跑，得给 ladder 加 `--resume`（语义可照抄 `ab_search_modes.py`：
   启动时跳过已有 `verdict=engine` 的 token，并把旧结果采纳进本轮汇总）。
2. **看门狗是"停滞"判据，不是死锁检测**：一局合法但很慢的对局也会被杀（所以默认关）。
   实测反例：`ab_search_modes` 那篇里 `--game-timeout-s 40` 比正常单局墙钟还小 ⇒ 两局都被杀。
3. **整机休眠期间看门狗也冻着**：Modern Standby 时线程不跑，所以"休眠中"不会触发；
   **醒来那一刻** `time.time()` 已经跳过阈值 ⇒ 会立刻判卡死并杀掉那一局、然后进重试——
   对"夜里冻住、早上恢复"这个场景正好合用，但**夜里它什么也做不了**（真正的根治手段是
   `code/test_match/_keepawake.ps1` 保活）。
4. **`terminal` 之后的卡死不管**：bridge 写完 `terminal` 行但进程不退出（比如等 AI 子进程退出）时，
   看门狗会跳过它（避免把已经写好 RESULT 的好局杀掉）⇒ 这种只能等 `LADDER_TIMEOUT`。
5. **重试不换配置**：第一轮为什么失败，重跑还是同一套 env/参数。偶发崩溃/资源竞争能救；
   配置性错误（ckpt 路径错、显存不够、AI 本身非法操作）会**稳定失败 N 轮**，
   最后老实打 `仍无效` 清单（`illegal_strict` 这种判负也会被重试，见 §3 的判据）。
6. **`taskkill /T` 依赖父子关系**：当前 `match_via_sdk.start_ai` 是普通 `Popen` ⇒ 关系成立
   （实测每次都是 3 个 PID）。若将来某个 AI 被显式 detach，`/T` 可能漏掉它；
   杀后的"已死根 PID 闭包复查"能发现（PPID 链仍在），并**只对杀前闭包里出现过的 PID** 补刀，
   不在闭包里的进程一律不碰（宁可漏杀）。
7. **同一 tag 并发跑两个驱动**仍会互相覆盖日志（本改动不解决这个）；同一 tag 的 `_attempts/`
   会一直累积（每次失败一份留档）。
8. **收尾表的"墙钟"是各次尝试之和**：重跑 3 次的局显示的是总耗时，不是最后一次。
9. 三个开关全关时**不打印收尾汇总表**（为了"默认 stdout 与改动前一致"）。想要那张表又不想重试，
   任给一个开关即可（例如 `--final-sweep`，没有无效局时它一局都不跑）。

## 6. 复现

```bash
bash code/run_logged.sh ladder_resilience_v1_old  D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_ladder_head.py --tag=LR_V1_OLD  --jobs=1 --ai0=code/test_match/rv4_pkg/main.py --ai1=_tmp_noopai.py 920 920r
bash _tmp_lr_v1.sh     # 验证 1（HEAD 副本 ×2 + 新版，逐字段比对）
bash _tmp_lr_v2.sh     # 验证 2（注入第 1 个对局进程失败 + --retry-invalid 1）
bash _tmp_lr_v3.sh     # 验证 3（注入前 4 个失败 + --retry-invalid 1 --final-sweep）
bash _tmp_lr_v4.sh     # 验证 4（假 AI sleep 600 + --watchdog-stall-min 1；并发对照批次 + 进程普查）
bash _tmp_lr_final.sh  # 终版整套复跑（V1-new + V2 + V3 + V4 一次跑完，上面报告的输出来自这次）
```

完整命令与输出在 `training_history/runs/<时间戳>_ladder_resilience_*/{cmd.txt,meta.txt,output.log}`，
每次运行的记录也追加进了 `docs/experiment_log.md`。
