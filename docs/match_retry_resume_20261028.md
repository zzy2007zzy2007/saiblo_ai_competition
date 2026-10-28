# 对局驱动健壮性：异常局自动重试 + 卡死看门狗 + 断点续跑

**日期**：2026-10-28
**改动文件**：`code/test_match/ab_search_modes.py`
**状态**：计划（改代码前先落档，实施/验证结果见文末各节）

---

## 0. 动机（为什么现在做）

`ab_search_modes.py` 是 `--seeds ... --jobs N --paired-sides` 的批量对局驱动。宿舍熄灯 /
休眠 / 断电会让一批局直接变成 `verdict=timeout` 或"日志 0 字节"，而**整批重跑代价极大**
（实测 32 局 / jobs=4 一轮 41 min；64 局 / jobs=8 一轮 47 min，见 `docs/experiment_log.md`）。
其中大部分局其实是好的，只有少数几局掉队。

2026-10-28 07:53 的 `match_results/rng_ab_ctl/efs4303` 就是活例：16 局里 8 局
`verdict=timeout exc=TimeoutError rounds=0`、8 局日志 0 字节（还在跑），**没有一局是 engine**。

已有有效性门槛：`code/test_match/analyze_paired.py` §4（`verdict=engine` 才算有效；
`timeout/aborted/illegal_strict` 全部记无效）。本任务**复用这个口径**，不另立标准。

## 1. 目标与约束

| # | 能力 | 参数 | 默认 |
|---|------|------|------|
| ① | 一批跑完后自动重跑无效局 | `--retry-invalid N` | 0（关） |
| ② | 单局卡死看门狗（按 PID 精确 kill） | `--game-timeout-s S` | 0（关） |
| ③ | 断电/重启后断点续跑 | `--resume` | 关 |
| ③' | 每局结束更新的轻量状态文件 | `<out-dir>/_progress.json` | 总是写（纯元数据） |

硬约束：

1. **默认行为不变**：三个新参数全关时，per-game 日志的 `RESULT` 行与改动前逐字一致，
   汇总行（胜率/无效局）也一致。为了便于机器比对，新参数全关时**不打印**新增的有效率区块。
2. **只 kill 自己的那一个 PID 树**：`subprocess.Popen` 句柄 → `taskkill /PID <pid> /T /F`。
   **绝不允许** `taskkill /IM python.exe` 或任何通配符/镜像名批量杀（2026-10 刚误杀过别人的进程）。
3. 不碰 `code/my_ai/az_intent/**`，不碰 `match_via_sdk.py` 的协议/判负逻辑。
4. 验证只用 1–2 局、`--jobs 1`，因为**同时有一批 8 局的 `rng_ab_ctl` 在跑**（不 kill 任何进程）。

## 2. 方案

### 2.1 单局执行层：`subprocess.run` → `subprocess.Popen` + 看门狗

现在：`subprocess.run(cmd, timeout=args.timeout)`。两个问题：
（a）`--timeout` 命中时 `TimeoutExpired` 直接把线程炸掉 ⇒ 该局结果**丢失**（既不进胜率也不计无效）；
（b）`subprocess.run` 的 timeout 只会 kill 直接子进程，AI 子进程会变孤儿。

改成：

```
proc = Popen(cmd, stdout=log_handle, stderr=STDOUT, env=..., cwd=REPO)
timer = threading.Timer(S, _kill_tree)      # 仅在 --game-timeout-s > 0 时启动
rc = proc.wait(timeout=args.timeout or None)
```

- `_kill_tree(proc)`：`taskkill /PID {proc.pid} /T /F`（Windows）/ `os.killpg`（POSIX），
  打完再 `proc.wait(60)`，仍未退则 `proc.kill()`。打印被杀的 PID。
- 看门狗触发 ⇒ 该局 verdict 记 `timeout_watchdog`；`--timeout` 命中 ⇒ `timeout_hard`。
- 句柄持有 PID ⇒ 进程活着时 PID 不会被回收（Windows 上 Popen 持有进程句柄），
  所以"按 PID 杀"是安全的；通配符杀才是危险的。
- **日志优先**：若被 kill 时日志里已经是 `verdict=engine`（进程写完 RESULT 才退出），
  以日志为准（算有效），避免把好局误判成 watchdog 超时。

### 2.2 有效性判定（复用 analyze_paired 口径）

单一函数 `_judge_log(task)`：

| 日志状态 | verdict | 有效 |
|---|---|---|
| 无日志 | `no_log` | 否 |
| 有日志但无 RESULT 行（0 字节 / 被杀） | `no_result`（或本进程记录的杀因） | 否 |
| `verdict=engine` | `engine` | **是** |
| `verdict=timeout` / `aborted` / `illegal_strict` | 原样 | 否 |

> 注意（顺带发现的坑）：老代码里 `winner_side is None` 时**不**计无效局，
> 所以 0 字节日志会被当成 0.5 平局**混进胜率**。新代码不动那行汇总（保默认行为），
> 但新增的有效率区块按真 verdict 统计，把这个洞显式化。

### 2.3 重试（`--retry-invalid N`）

```
跑第一批 → for rnd in 1..N:
    扫描全部 task → 找 verdict != engine 的
    若空 ⇒ break
    打印：轮次 / 重跑清单 / 重跑前 verdict 计数（Counter）
    只跑这些 task（attempt += 1）
```

- 成功局**永不出现在重跑清单**里 ⇒ 不会被覆盖。重跑前把上一轮失败日志复制到
  `<out-dir>/_attempts/<label>.attempt<k>.log` 留证据（不删原日志）。
- 计数：`results[label]` 每轮被**最新一次尝试**覆盖 ⇒ 最终统计天然是"最后一次有效结果"。
- 防御性兜底：重跑前若发现该局日志其实已经 `engine`（例如人工改过），跳过并告警。
- 收尾打 `=== 最终有效率 X/Y (Z%) ===` + 有效局专用胜率 + 仍无效清单。

### 2.4 断点续跑（`--resume`）+ `_progress.json`

- 启动时扫 out-dir：`verdict=engine` 的 task 直接跳过，并**采纳**它的结果到本轮汇总
  （否则 resume 那轮的胜率只覆盖重跑的几局，读数会骗人）。
- 打印 `续跑计划：已完成 K 个、待跑 M 个`，并列出待跑原因（`no_log` / `verdict=timeout` …）。
- `_progress.json`：`{label: {seed, side, status, verdict, ts, attempts, score, minutes}}`，
  每局结束（含失败）就更新；`写 .tmp + os.replace` 原子替换（和 `progress.py` 同一套写法）。
- **日志是唯一权威**，状态文件只是缓存/审计线索：如果 `_progress.json` 说 ok 但日志被删了
  ⇒ 仍然重跑（对应用户的验证 ③：删日志后必须只补那一局）。
- 状态文件里额外记 `ok`/`invalid`/`running`，跑批中途打开就能看出进行到哪。

### 2.5 已知限制（预设）

1. **进程内重试跨不了断电**：`--retry-invalid` 只在驱动进程活着时有效；断电/被杀就只剩
   `--resume` 一条路。两者正交，建议一起用（`--resume --retry-invalid 2 --game-timeout-s ...`）。
2. 看门狗是"墙钟超时"，不是"卡死检测"：一局合法但很慢的对局同样会被杀（所以默认关）。
3. `taskkill /T` 靠父子关系；若某个 AI 子进程被父进程显式 detach 而父进程先死，
   `/T` 可能漏掉它（当前 `match_via_sdk.start_ai` 是普通 Popen ⇒ 父子关系成立）。
4. 重试不改变配置：第一轮为什么失败，重跑还是那套 env/参数（偶发资源竞争能救，配置性错误
   救不了，会稳定失败 N 轮后报"最终有效率 < 100%"）。
5. 同一 out-dir 并发跑两个驱动会互相覆盖日志（状态文件有锁但日志没有），本改动不解决这个。

## 3. 验证计划

| # | 场景 | 做法 | 判据 |
|---|------|------|------|
| 1 | 默认行为不变 | 同一 2 局小批次，HEAD 版驱动（`_tmp_ab_old.py`）vs 新版驱动各跑一次 | 两局 `RESULT` 行逐字相同；汇总行一致（耗时除外） |
| 2 | 看门狗 + 重试 | 用一次性 `sitecustomize.py` 注入"首个 AI 进程 sleep"，`--game-timeout-s` 小 | 被 kill 的只有该局 PID 树；记 `timeout_watchdog`；重跑清单只有那 1 局；最终有效率 2/2 |
| 3 | 续跑 | 跑 2 局 → 把 1 局日志改成 `verdict=timeout` → `--resume` 再跑 | 打印"已完成 1 个、待跑 1 个"，只重跑那 1 局 |

全部用 `bash code/run_logged.sh <name> <cmd>` 落盘。

---

## 4. 实施（代码）

（待填）

## 5. 验证输出

（待填）

## 6. 已知限制

（待填）
