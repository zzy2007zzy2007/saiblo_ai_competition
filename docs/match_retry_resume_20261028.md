# 对局驱动健壮性：异常局自动重试 + 卡死看门狗 + 断电断点续跑

**日期**：2026-10-28
**改动**：`code/test_match/ab_search_modes.py`（只动驱动层；不碰 `code/my_ai/az_intent/**`、
不碰 `match_via_sdk.py` 的协议/判负逻辑）
**验证工件**：`_tmp_v1_new.sh` / `_tmp_v2_retry.sh` / `_tmp_v3_resume.sh` / `_tmp_v4_noop.sh` /
`_tmp_faultinject/sitecustomize.py` / `code/test_match/_tmp_ab_old.py`（HEAD 版驱动的副本，做对照用）

---

## 0. 动机

`ab_search_modes.py` 是 `--seeds ... --jobs N --paired-sides` 的批量对局驱动。宿舍熄灯 / 休眠 /
断电会让一批局变成 `verdict=timeout` 或者干脆"日志 0 字节"，而整批重跑代价极大
（`docs/experiment_log.md` 里 32 局/jobs=4 一轮 41 min、64 局/jobs=8 一轮 47 min）。

2026-10-28 07:53 的 `match_results/rng_ab_ctl/efs4303` 是活例：16 局里 8 局
`verdict=timeout exc=TimeoutError rounds=0`、8 局 0 字节，**没有一局 engine**。

有效性门槛复用 `code/test_match/analyze_paired.py` §4 的口径：**只有 `verdict=engine` 算有效**，
`timeout / aborted / illegal_strict / hp_tiebreak / 无 RESULT / 无日志` 一律无效。

## 1. 参数说明

| 参数 | 默认 | 作用 |
|---|---|---|
| `--retry-invalid N` | `0`（关） | 一批跑完后扫 out-dir，凡**没有 `verdict=engine`** 的局重跑，最多 N 轮 |
| `--game-timeout-s S` | `0`（关） | 单局墙钟看门狗：超 S 秒未结束 ⇒ 精确 kill 该局 PID 树，记 `verdict=timeout_watchdog` |
| `--resume` | 关 | 启动时跳过已有 `verdict=engine` 的 seed/side，只跑缺失/无效的；并把旧结果采纳进本轮汇总 |
| `<out-dir>/_progress.json` | 总是写 | 每局结束更新一次（`.tmp`+`os.replace` 原子写）：`{label: {status, verdict, ts, attempts, score, minutes, rounds, seed, side}}` |

**三个开关全关时行为与改动前一致**（见 §3 验证 1）：stdout 只在"某局无效"时多一行 `⚠️` 告警，
`RESULT` 行逐字相同。

### 1.1 `--retry-invalid N`（运行中自动重试）

- 每轮开头打印：**第几轮**、**重跑清单**（哪些 seed/side）、**重跑前的 verdict 计数**；
- **只重跑失败的**：重跑清单从"当前无效局"生成，成功局永不进入 ⇒ 成功局的日志不会被覆盖
  （重跑前还有一道防御：再读一次日志，已经 `engine` 就跳过并告警）；
- 重跑前把上一轮失败的日志**复制**到 `<out-dir>/_attempts/<label>.attempt<k>.log` 留证据（不删原日志）；
- 收尾打印 `=== 最终有效率 X/Y (Z%) ===` + 本轮重试覆盖了几局、几局转有效、仍然无效的是哪些，
  以及**只看有效局**的胜率（分母 = `verdict=engine` 的局）；
- 计分口径 = **最后一次尝试**；因为成功局不会再跑，所以重试过的局用的就是它最后一次有效结果。

### 1.2 `--game-timeout-s S`（卡死看门狗）

- 单局执行从 `subprocess.run(timeout=...)` 改成 `Popen` + `threading.Timer(S)`；
- 超时 ⇒ `taskkill /PID <pid> /T /F`（**只按 Popen 句柄那一个 PID**，`/T` 带上它的子进程）；
  **没有任何按镜像名/通配符的杀法**（`taskkill /IM python.exe` 那种会连别人正在跑的对局一起杀）；
- 杀完记 `verdict=timeout_watchdog`，然后按 §1.1 的规则重试；
- **日志优先**：如果被 kill 时日志里已经是 `verdict=engine`（进程写完 RESULT 才退出），
  以日志为准算有效，不把好局误判成超时。

### 1.3 `--resume`（断电断点续跑）

- 启动时扫 out-dir，`verdict=engine` 的直接跳过，并**采纳**它的比分进本轮汇总
  （否则续跑那轮的胜率只覆盖重跑的几局，读数会骗人）；
- 打印 `续跑计划：已完成 K 个、待跑 M 个` + 跳过清单 + 待跑清单（附原因：`timeout` / `no_log` / `no_result` …）；
- **日志是唯一权威**：`_progress.json` 里记着 `ok` 但日志被删/被改坏 ⇒ 照样重跑
  （打印里会写"但 `_progress.json` 记为 ok ⇒ 以日志为准重跑"）。这条就是用户验证③要的行为。

## 2. 实现要点 / 踩到的坑

1. **精杀进程树**：`Popen` 句柄持有该 PID；进程活着时 Windows 不会回收这个 PID ⇒ 按 PID 杀是安全的。
   `taskkill /T` 靠父子关系找子进程（`match_via_sdk.start_ai` 是普通 Popen，父子关系成立）。
2. **`--timeout` 的老坑**：旧代码 `subprocess.run(timeout=...)` 命中时 `TimeoutExpired` 直接把线程炸掉
   ⇒ 该局结果**彻底丢失**（既不进胜率也不计无效）。现在走同一条 kill 路径，记 `timeout_hard` 并参与重试。
3. **0 字节日志的老坑**：旧代码在 `winner_side is None`（根本没有 RESULT 行）时**不**计无效局
   ⇒ 断电那批局被当成 **0.5 平局混进胜率**。现在这类局计入"无效局"（`no_result / no_log /
   timeout_watchdog / timeout_hard / driver_error`）。**只在这一种情况下汇总行的数字才会变大**，
   有效局（engine）的读数与旧版逐字相同。
4. **死锁（验证时真踩到，已修）**：第一版把"数无效局"写成了 `invalid_count_for_summary()` 并让它
   `with lock:`，而 `write_status()` 是**持锁**调用它的 ⇒ `threading.Lock` 非重入 ⇒ 整批静默卡死
   （实测：第一局跑完后不再打印，第二局跑完堵在 lock 上，599s 后人工 kill；见 `docs/experiment_log.md`
   08:17:39 那条）。修法：`lock` 改 `RLock` + 该函数不再抢锁（只做 dict 只读访问）。
5. **原子写状态文件**：`_progress.json.tmp` → `os.replace`（和 `code/test_match/progress.py` 同一套写法），
   断电最多丢最后一次更新，不会留半个 JSON。
6. **驱动层异常不丢局**：线程入口包了 `try/except`，异常记 `verdict=driver_error` 参与重试
   （旧实现里该局结果会直接消失）。

## 3. 验证（全部 `bash code/run_logged.sh <名字> <命令>` 落盘）

> ⚠️ 验证期间机器上**同时有一批 8 局在跑**（`match_results/rng_ab_ctl/efs4303`、`rng_p...`），
> 所以验证只用 1–2 局、`--jobs 1`，并且**没有 kill 任何别人的进程**。
> 另外为了让验证便宜，"小批次"用的是把 `AZAI_ITERS/FLAT_ROLLOUTS` 调小的 tiny 配置
> （单局 55–90s 而不是 5 min）；驱动层代码与配置无关，结论不受影响。

### 验证 1：默认行为不受影响（V1）

同一 2 局小批次（`--paired-sides --seeds 900 --jobs 1`），HEAD 版驱动（`code/test_match/_tmp_ab_old.py`）
vs 新版驱动各跑一次，`--out-dir` 分别是 `match_results/retry_resume_test/v1_old` / `v1_new`。

| | 命令 |
|---|---|
| 旧 | `bash code/run_logged.sh v1_old_2games <py> -u code/test_match/_tmp_ab_old.py --paired-sides --seeds 900 --jobs 1 --out-dir match_results/retry_resume_test/v1_old ...` |
| 新 | 同上，只把 `_tmp_ab_old.py` → `ab_search_modes.py`、`v1_old` → `v1_new` |

日志目录：`training_history/runs/20261028_080910_v1_old_2games/`、`training_history/runs/20261028_082746_v1_new_2games/`

`RESULT` 行**逐字相同**：

```
OLD seed900_p0 : RESULT seed=900 p0=az_bridge_ai p1=az_bridge_ai winner=az_bridge_ai winner_side=p0 verdict=engine engine_winner=0 rounds=426 terminal=True base_hp=9,0 coins=227,97 illegal=0
NEW seed900_p0 : RESULT seed=900 ...（同上一字不差）
OLD seed900_p1 : RESULT seed=900 p0=az_bridge_ai p1=az_bridge_ai winner=az_bridge_ai winner_side=p1 verdict=engine engine_winner=1 rounds=378 terminal=True base_hp=0,17 coins=83,251 illegal=0
NEW seed900_p1 : RESULT seed=900 ...（同上一字不差）
SAME: True / True
```

整份日志（1953 行）也一致，唯一差异是 bridge 的**计时字段**：

```
[bridge] illegal-summary seed=900 ... mismatch=0 shadow_ms=7.6 ...   ← 旧
[bridge] illegal-summary seed=900 ... mismatch=0 shadow_ms=6.8 ...   ← 新
```

驱动 stdout 也一致（只有耗时不同）：

```
旧: [seed 900] flat=P0 score=1.0 rc=0 rounds=426 hp=(9, 0) (0.9 min)
    [seed 900] flat=P1 score=1.0 rc=0 rounds=378 hp=(0, 17) (1.6 min)
    === flat vs mcts：2 局  flat 胜率 100.0%  (2W/0D/0L)  ±70.7%(2SE)  wall=1.6 min  无效局=0 ===
        A 在 P0: 1W-0L = 100.0% (n=1)
        A 在 P1: 1W-0L = 100.0% (n=1)

新: [seed 900] flat=P0 score=1.0 rc=0 rounds=426 hp=(9, 0) (0.8 min)
    [seed 900] flat=P1 score=1.0 rc=0 rounds=378 hp=(0, 17) (1.5 min)
    === flat vs mcts：2 局  flat 胜率 100.0%  (2W/0D/0L)  ±70.7%(2SE)  wall=1.5 min  无效局=0 ===
        A 在 P0: 1W-0L = 100.0% (n=1)
        A 在 P1: 1W-0L = 100.0% (n=1)
```

文件系统唯一差别：新目录多一个 `_progress.json`（纯元数据）。**结论：默认行为不变。**

### 验证 2：看门狗 + 自动重试（V2）

故障注入：`_tmp_faultinject/sitecustomize.py` 让**第一个** AI 进程睡 600s（`mkdir` 原子抢占，
只影响第一局；换掉 `az_bridge_ai.py` 会污染正在跑的另一批，所以没换文件）。看门狗 150s。

`training_history/runs/20261028_082943_v2_retry_watchdog/output.log`：

```
[watchdog] seed910: 墙钟超过 150s 仍未结束 ⇒ 精确 kill PID 59284 及其子进程（taskkill /PID 59284 /T /F，不碰其它进程）
[watchdog] taskkill rc=0: SUCCESS: The process with PID 61100 (child process of PID 59284) has been terminated.
SUCCESS: The process with PID 59208 (child process of PID 59284) has been terminated.
SUCCESS: The process with PID 59284 (child process of PID 45632) has been terminated.
[seed 910] flat=P0 score=0.5 rc=1 rounds=None hp=None (2.5 min)
  ⚠️ seed910 判无效：verdict=timeout_watchdog
[seed 911] flat=P1 score=1.0 rc=0 rounds=383 hp=(0, 17) (3.4 min)

=== 重试第 1/2 轮：1 局没有 verdict=engine ===
    重跑前失败原因分布（verdict 计数）: {'timeout_watchdog': 1}
    重跑清单: seed910
[retry] 上一轮失败日志已留档: match_results/retry_resume_test/v2_retry/_attempts/seed910.attempt1.log
[seed 910] flat=P0 score=1.0 rc=0 rounds=497 hp=(6, 0) (1.0 min)

[retry] 第 2/2 轮：没有无效局了，停止重试

=== flat vs mcts：2 局  flat 胜率 100.0%  (2W/0D/0L)  ±70.7%(2SE)  wall=4.4 min  无效局=0 ===
    A 在 P0: 1W-0L = 100.0% (n=1)
    A 在 P1: 1W-0L = 100.0% (n=1)

=== 最终有效率 2/2 (100.0%) ===
    重跑过的局 1 个：seed910（其中转有效 1 个）
    有效局胜率（口径=verdict=engine，分母 2）: 100.0%  (2W/0D/0L)
    说明：上面的汇总按**最后一次尝试**计分；成功局不会被重跑，所以重试过的局用的就是最后一次有效结果。
```

读数要点：

1. **只杀了 3 个 PID**（`match_via_sdk` + 两个 AI），同批的另一局 `seed911` 照常跑完（`rc=0`、
   383 回合、`verdict=engine`），**没有被波及**；
2. 被杀的那局标记成 `timeout_watchdog`，进重跑清单；重跑后 `seed910.log` 变成一局完整的 225 KB 日志，
   而**失败那次的日志**被留档成 `_attempts/seed910.attempt1.log`（1701 字节，只到对局表头、没有 RESULT）；
3. `_progress.json`：`seed910.attempts=2 / verdict=engine`，`seed911.attempts=1`；
4. 进程普查：跑之前 26 个 python 进程，跑完还是这 26 个（**一个都没少**），
   包括别人那批 `rng_ab_ctl` 的主进程 39084 —— 没有任何误杀。

> 反例记录（第一次跑 V2 时）：`--game-timeout-s 40` 比正常单局墙钟（55–90s）还小 ⇒ **两局都被杀**，
> 重试 2 轮仍然 0/2（`docs/experiment_log.md` 08:12:59 那条）。这正好说明看门狗是"墙钟超时"不是
> "死锁检测"，阈值必须大于正常单局耗时（见 §5 限制 2）。
>
> 录制时间说明：这段输出录于 08:29，之后只改了一处**纯显示**口径（"重跑过的局"从"累计尝试>1"
> 改成"本轮尝试>1"）。本场景两种算法结果相同（seed910 本轮尝试 2 次）；修正后的口径由验证 5 覆盖。

### 验证 3：断点续跑（V3）

`training_history/runs/20261028_083808_v3_resume_s1_normal/` → 先正常跑 2 局（seed 901/902）；
然后**手动把 seed902 的 RESULT 改成 `verdict=timeout`**，再 `--resume`：

```
[resume] 续跑计划：已完成 1 个、待跑 1 个
[resume] 跳过（日志 verdict=engine）: seed901
[resume] 待跑明细（原因）:
    seed902: timeout，但 _progress.json 记为 ok ⇒ 以日志为准重跑
[seed 902] flat=P0 score=1.0 rc=0 rounds=470 hp=(6, 0) (0.9 min)

=== flat vs mcts：2 局  flat 胜率 100.0%  (2W/0D/0L)  ±70.7%(2SE)  wall=0.9 min  无效局=0 ===
...
=== 最终有效率 2/2 (100.0%) ===
    有效局胜率（口径=verdict=engine，分母 2）: 100.0%  (2W/0D/0L)
```

**只跑了那 1 局**（`wall=0.9 min`、只有一行 `[seed 902]`），seed901 直接复用旧日志进汇总
（所以汇总仍是"2 局"，不是骗人的 1 局）。

再**手动删掉 seed901.log**（模拟"日志没落盘"），继续 `--resume`：

```
[resume] 续跑计划：已完成 1 个、待跑 1 个
[resume] 跳过（日志 verdict=engine）: seed902
[resume] 待跑明细（原因）:
    seed901: no_log，但 _progress.json 记为 ok ⇒ 以日志为准重跑
[seed 901] flat=P1 score=1.0 rc=0 rounds=304 hp=(0, 23) (0.7 min)
```

即：**`_progress.json` 说 ok 也不算数，日志没了就重跑**。

### 验证 4（补充）：`--resume --retry-invalid` 同时开、但都已完成

`training_history/runs/20261028_083759_v4_resume_noop/output.log`（**0 秒、一局都没跑**）：

```
[resume] 续跑计划：已完成 2 个、待跑 0 个
[resume] 跳过（日志 verdict=engine）: seed901, seed902

[retry] 第 1/1 轮：没有无效局了，停止重试

=== flat vs mcts：2 局  flat 胜率 100.0%  (2W/0D/0L)  ±70.7%(2SE)  wall=0.0 min  无效局=0 ===
...
=== 最终有效率 2/2 (100.0%) ===
```

### 验证 5（补充）：重试**救不回来**时的读数（V5，25s）

看门狗故意设成 5s（远小于正常单局墙钟）⇒ 每局都被杀、重试 1 轮仍然被杀。
`training_history/runs/20261028_084245_v5_retry_cannot_fix/output.log`（已提交的最终代码）：

```
[watchdog] seed920: 墙钟超过 5s 仍未结束 ⇒ 精确 kill PID 32412 及其子进程（taskkill /PID 32412 /T /F，不碰其它进程）
...
[seed 920] flat=P0 score=0.5 rc=1 rounds=None hp=None (0.1 min)
  ⚠️ seed920 判无效：verdict=timeout_watchdog
[seed 921] flat=P1 score=0.5 rc=1 rounds=None hp=None (0.2 min)
  ⚠️ seed921 判无效：verdict=timeout_watchdog

=== 重试第 1/1 轮：2 局没有 verdict=engine ===
    重跑前失败原因分布（verdict 计数）: {'timeout_watchdog': 2}
    重跑清单: seed920, seed921
[retry] 上一轮失败日志已留档: .../v5_retry_cannot_fix/_attempts/seed920.attempt1.log
[retry] 上一轮失败日志已留档: .../v5_retry_cannot_fix/_attempts/seed921.attempt1.log
...
[retry] ⚠️ 已用满 1 轮，仍有 2 局无效: seed920(timeout_watchdog), seed921(timeout_watchdog)

=== flat vs mcts：2 局  flat 胜率 50.0%  (0W/2D/0L)  ±70.7%(2SE)  wall=0.4 min  无效局=2 ===

=== 最终有效率 0/2 (0.0%) ===
    重跑过的局 2 个：seed920, seed921（其中转有效 0 个）
    仍然无效 2 个: seed920(timeout_watchdog), seed921(timeout_watchdog)
    说明：上面的汇总按**最后一次尝试**计分；成功局不会被重跑，所以重试过的局用的就是最后一次有效结果。
```

两个读数点：

1. 重试是"最多 N 轮"而不是"一定成功"——救不回来时它会**明确报 0/2 + 仍然无效清单**，
   不会假装成功；
2. 汇总行的 `无效局=2` 是 §2 坑 3 的修正效果：这两局**根本没有 RESULT 行**，旧代码会把它们
   当 0.5 平局（`无效局=0`），现在是 2。这正好是"断电那批局"的形态。

## 4. 怎么用（一条命令）

```bash
# 夜里跑：卡死 90min 就杀掉、跑完把无效局重试 2 轮、白天断电后同一条命令加 --resume 只补漏的
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/ab_search_modes.py \
    --seeds 3300-3331 --jobs 8 --paired-sides \
    --out-dir match_results/rng_ab_ctl/efs4303 \
    --resume --retry-invalid 2 --game-timeout-s 5400
```

第二天的正确姿势：**同一条命令原样再敲一遍**（`--resume` 会跳过所有已 engine 的局）；
只想补跑、不想动其它参数时也可以先不带 `--retry-invalid`，跑完再补一轮。

## 5. 已知限制（哪些情况救不了）

1. **进程内重试跨不了断电**：`--retry-invalid` 只在驱动进程活着时有效。断电/关机/被 kill 之后，
   只能靠 `--resume` 重来；两者是正交的，建议一起开。
2. **看门狗是墙钟超时，不是死锁检测**：一局合法但很慢的对局也会被杀（所以默认关）。
   阈值必须明显大于正常单局墙钟：tiny 配置 55–90s、默认配置约 5 min（本次实测 `--game-timeout-s 40`
   把两局都误杀了）。保守取值 60–120 min。
3. **重试不换配置**：第一轮为什么失败，重跑还是同一套 env/参数。偶发资源竞争能救，配置性错误
   （ckpt 路径错、显存不够）会稳定失败 N 轮，最后打 `最终有效率 < 100%` 和"仍然无效"清单。
4. **`taskkill /T` 依赖父子关系**：若某个 AI 子进程被父进程显式 detach 而父进程先死，`/T` 可能漏掉它
   （当前 `match_via_sdk.start_ai` 是普通 Popen ⇒ 父子关系成立；本次验证里被杀的就是 3 个 PID）。
5. **"无效局"仍按旧口径计 0.5 分混进 `flat 胜率`**（为了不改变既有读数习惯）。要看干净读数请用
   `有效局胜率` 那行，或直接 `analyze_paired.py`（它把非 engine 一律排除）。
6. **`_progress.json` 的 `attempts` 是"该 out-dir 的累计次数"**：同一目录换参数重跑会让它继续累加。
   它只是审计线索，判定永远以日志为准。
7. **同一 out-dir 并发跑两个驱动**仍会互相覆盖日志（状态文件有锁、日志没有）——本改动不解决这个。
8. **断电瞬间的残局**：最后 1–2 秒内写完 RESULT 但没落盘的局会被判 `no_result` 并重跑
   （行为正确，只是浪费一局）。
9. 重试/续跑都不会修复**引擎或 AI 本身的 bug**（例如 `illegal_strict` 判负、崩溃）——它们只是让你
   知道有多少局坏了、并把偶发坏局救回来。

## 6. 复现

```bash
bash code/run_logged.sh v1_old_2games <py> -u code/test_match/_tmp_ab_old.py --paired-sides --seeds 900 --jobs 1 \
    --out-dir match_results/retry_resume_test/v1_old --flat-env="..." --mcts-env="..."   # 见 _tmp_v1_new.sh
bash _tmp_v1_new.sh     # V1（新驱动，默认参数）
bash _tmp_v2_retry.sh   # V2（故障注入 + 150s 看门狗 + 重试 2 轮）
bash _tmp_v3_resume.sh  # V3（正常跑 2 局 → 改日志 → --resume → 删日志 → --resume）
bash _tmp_v4_noop.sh    # V4（--resume + --retry-invalid 但全部已完成）
bash _tmp_v5_cannotfix.sh  # V5（看门狗 5s：重试救不回来时的读数）
```

完整命令（含所有 env 串）也记在 `training_history/runs/<时间戳>_<名字>/cmd.txt` 里。
