# 计划：把「过夜批次」的三项能力做进对局驱动本体（`_tmp_ladder.py`）

**日期**：2026-10-29
**目标文件**：`_tmp_ladder.py`（项目标准对局驱动；所有"我们 vs 外部 AI"批次都走它）
**参考实现**：`code/test_match/ab_search_modes.py` 的 `--retry-invalid` / `--resume` / `--game-timeout-s`
（语义照抄，但**不动那个文件**）
**不做**：不再加一层 shell 包装（`_us_vs_champion_e1_resume.sh` 里那套 `grep verdict=engine` 的
外部补跑逻辑，这次直接搬进驱动）。

## 0. 为什么是现在

2026-10-29 那批 E1（`US_VS_CHAMPION_BASE`）的形态：**机器进 Modern Standby 冻了 2h06m**
⇒ 3 局 `verdict=timeout` + 8 局 0 字节日志。补跑只能靠外层 shell 脚本 `grep` 日志再拼命令行，
既容易写错，也漏掉"到底哪几局没成、各试过几次"。这三项能力应该长在驱动里。

## 1. 三项能力（默认全关 ⇒ 行为与改动前逐位相同）

| 开关 | 默认 | 作用 |
|---|---|---|
| `--retry-invalid N` | `0`（关） | 一批跑完后扫 `<tag>` 日志目录，凡**没有 `verdict=engine`** 的 token 重跑，最多 N 轮 |
| `--final-sweep` | 关 | 主循环（含重试）结束后**再扫一遍**，仍有非 engine ⇒ 最后再跑一次（不计入 N 轮） |
| `--watchdog-stall-min M` | `0`（关） | 运行期间盯本进程启的对局，progress/日志 mtime 连续 M 分钟不前进 ⇒ 精确 kill 该局 PID 树 |

### 1.1 `--retry-invalid N`

- 每轮打印：第几轮、**待跑 token 列表**、**重跑前 verdict 计数分布**；
- **只重跑失败的**：清单由"当前无效局"生成 ⇒ 已 engine 的局永不进入（重跑前再判一次，已 engine 就跳过）；
- 重跑前把失败局日志**复制**到 `<dir>/_attempts/<token>.attempt<k>.log`（只复制，不删原日志）；
- 无效判据 = `verdict != engine`，覆盖 `timeout / aborted / illegal_strict / hp_tiebreak /
  空壳日志 / 没有 RESULT 行 / 没有日志`。

### 1.2 `--final-sweep`

- 主循环 + 重试都用完之后再扫一遍；仍有非 engine 的 token ⇒ 再跑一次（**不计入** `--retry-invalid` 的轮数）；
- 用途：机器夜里休眠 ⇒ 重试轮数在凌晨就被"超时+重试"烧完了，这一轮在早上（醒来）执行，把漏的补完；
- 单独打印一张 `final-sweep` 区块（当前 verdict 分布 + 待跑列表），与重试轮分开，不会混算。

### 1.3 `--watchdog-stall-min M`

- **只盯自己启动的对局**（`Popen` 句柄在 `_WATCH` 表里）；判定依据：
  `training_history/progress/<tag>/az_progress_*<token>*.txt` 的 mtime **或** 该局日志 mtime
  连续 M 分钟不前进（两者取"更活跃"的那个，避免日志被缓冲时误判）；
- 命中 ⇒ 标记 `timeout_watchdog`，并且**顺序严格是：先停调度（`_SCHED_RUN.clear()`）→ 再
  `taskkill /PID <pid> /T /F` → 用**已死的根 PID** 复查后代闭包 → 最后恢复调度**；
  踩过的坑：先杀对局会让驱动立刻抢信号量、把下一波 token 拉起来；
- **绝不按镜像名/命令行通配符批量杀**（`taskkill /IM python.exe` 会连别人正在跑的对局一起杀）；
- 已终局的局不更新 progress 文件 ⇒ mtime 会一直变老，**必须**靠 progress 最后一行的 `terminal`
  排除（否则会对已完成的局狂杀）——沿用 `code/test_match/_ladder_watchdog.ps1` 的判据；
- 默认关：它是"墙钟/停滞"判据，不是死锁检测，一局合法但很慢的局也会被杀。

### 1.4 收尾汇总表

任一开关打开时，**结算前**打印（全关时不打印 ⇒ 默认 stdout 与改动前一致）：

```
token | verdict | rounds | 墙钟(min) | 尝试次数 | 是否有效
...
engine K/N
仍无效的 token（早上人工补跑就照这个列表给命令行）: 8(no_result), 9r(timeout_watchdog)
```

## 2. 实现要点

1. `run_one` 从 `subprocess.run(timeout=)` 改成 `Popen` + `wait(timeout=)`：看门狗必须拿到
   **自己的 PID**；超时分支仍按旧语义（只杀直接子进程、`rc=-2`）⇒ 默认路径读数不变。
2. 全局调度闸门 `_SCHED_RUN`（`threading.Event`，默认 open）；`run_one` 在**信号量内**等它
   ⇒ 停调度期间不会有新 token 起来，且不会因为"被杀那局释放了 sem"而抢跑。
3. 进程表用 `CreateToolhelp32Snapshot`（ctypes，只读、ms 级），**不用** wmic/PowerShell
   （wmic 在新 Windows 上已被移除）。Windows 不重新挂父 ⇒ 已死根 PID 的 PPID 链仍可复查闭包。
4. 判有效 / 解析一律走既有 `read_game`（历史判据唯一实现，不重写）。
5. `_attempts` 归档命名 `<token>.attempt<k>.log`，k = 产下这份日志的那次尝试序号。

## 3. 验证计划（全部 `bash code/run_logged.sh ladder_resilience_* <命令>`，1–2 局、`--jobs 1`）

| # | 场景 | 判据 |
|---|---|---|
| 1 | `git show HEAD:_tmp_ladder.py` 副本 vs 新版，各跑同样 2 局 | `RESULT` 行**逐字段相同** |
| 2 | 故障注入（`sitecustomize` 让第 1 次对局进程立刻 `os._exit`）+ `--retry-invalid 1` | 只重跑那 1 局、转 engine；另一局日志未被动过 |
| 2b | 手工把某 token 的 `verdict=engine` 改成 `verdict=timeout` + `--retry-invalid 1` | 同上（用户点名的第二种注入方式） |
| 3 | 注入前 4 次都失败 + `--retry-invalid 1 --final-sweep` | 重试 1 轮后仍无效 ⇒ final-sweep 再跑一次并转 engine |
| 4 | 假 AI `sleep 600` + `--watchdog-stall-min 1` | ~1 分钟内**只**杀掉那一局（打印精确 PID/闭包复查）；同批另一局照常 engine；别人的批次 PID 不变 |

## 4. 交付

1. `_tmp_ladder.py`（三个开关 + 收尾汇总表，默认关）
2. `docs/ladder_resilience_20261029.md`（开关说明 + 真实输出 + 已知限制）
3. `git commit`
