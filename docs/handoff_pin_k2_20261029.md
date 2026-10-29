# 交接：`AZAI_PIN_K`（放宽类轴）实验为什么停了、那次"疑似卡死"的真因、怎么接着做

> 写于 **2026-10-29 08:25**。上游：`docs/pin_k2_ab_20261028.md`（预注册 + 中止记录）。
> 本文是给**下一个人**看的：不需要重读全部上下文，读这一页就能接手。

---

## 0. 30 秒版

1. **实验已暂停**（用户优先级调整到"评估够不够用"E2/E4 那条线）。
   **0/16 局有官方判词** ⇒ **没有任何棋力结论**。`配对 ΔHP` 算不出来。
2. **"疑似卡死"不是 `AZAI_PIN_K=2` 引起的**，是**整台机器进了 Modern Standby 待机**。
   同一份代码加了保活之后，8 局跑 1325 个决策 / 160-214 回合 / **0 次 timeout** / 8.19 回合每分。
3. ⚠️ **本机上任何跨夜/长时批次，开跑前必须先起 `code/test_match/_keepawake.ps1`**，
   否则 `verdict=timeout` 是必然的（不是概率事件：这台机器**每晚都睡**）。
4. ⚠️ 那个保活进程**现在还在跑**（我起的，PID 见 §5）。它在保护当前所有批次。
   如果你不需要它了，按 §5 的方式**按 PID** 停掉，别用镜像名批量杀。

---

## 1. `AZAI_PIN_K` 是什么

### 1.1 一句话

`AZAI_PIN_K=K`（`K>1`）让**整束的 720 次采样轮流落在 K 个不同的类上**，
而不是现役那样**全部钉在一个类**上。默认 **0 = 完全关闭、现役行为逐位不变**。

### 1.2 为什么需要它（上游已确立的诊断）

`docs/why_effective_candidates_collapse_20261028.md`（commit `ae5fd84`）：
`bundle_mcts.py:884` 的 `pinned=[_ccls]*3` 把 720 次采样（`k=48 × sample_mult=15`）**全钉在一个类**上，
而 `mc_quiet` 在 **78.0%** 的决策里选 **HOLD（类 23）**——HOLD 的动作集基数 = 1（空束）
⇒ 720 次采样解出**同一个空束** ⇒ `agg` 只有一个键 ⇒ 送进 flat rollout 的候选数 = **1**。
**不是去重 bug，是采样本身只产出 1 条。**

放开后（同一批 1518 个真实决策点实测）：

| 配置 | `n_cand==1` | 中位 `n_cand` | `=48` |
|---|---|---|---|
| `PIN_K=0`（现役） | **82.3%** | 1 | 8.7% |
| **`PIN_K=2`** | **0.7%** | **14** | 19.2% |
| `PIN_K=3` | 0.2% | 42 | 44.7% |
| `PIN_K=5` | 0.2% | 48 | 56.9% |

每候选的采样数**不稀释**（`n_rollouts = min(64, 49152//(n_cand×16))`，`n_cand ≤ 48` 时仍是 64）。

### 1.3 代码在哪（+21 行代码，已在 `ae5fd84` 里提交）

| 文件 | 位置 | 改动 |
|---|---|---|
| `code/my_ai/az_intent/bundle_mcts.py` | `__init__` 加参数 `pin_k: int = 0`；`:515` `self.pin_k = max(0, int(pin_k))` | +2 行 |
| 同上 | `:1010-1027` `if self.pin_k > 1:` 分支：类序列 = `[MC 的赢家] + [_econ_rank 剩下的] + [HOLD(23), 闪电(17)]`，去重取前 K，然后 `class_ids` 把 `n_samples` 次采样**轮流**分给这 K 个类 | +19 行 |
| `code/test_match/az_bridge_ai.py` | `:979` `pin_k=int(_env("AZAI_PIN_K", "0"))` + docstring | +1 行 |

**只改"哪些类参与采样"。** 去重（`:1084`）、`top-k` 截断（`:1097`）、flat rollout 的预算公式、
评估器、`mc_last_explicit` 的升级直发路径**一行没动**；`pinned` 本身也不动
（它还喂 `pos_prior_fn` 与训练目标）。

### 1.4 零影响证据（`PIN_K=0`）

1. **代码路径**：`if self.pin_k > 1:` ⇒ `0` 时只有 `else` 那一行（`class_ids = [[c]*n_samples ...]`）执行，
   与改动前逐字相同。
2. **实测**：把**打过补丁的**代码用 `--pin-k 0` 重跑同一批 1518 个决策点，与改动前那一跑的
   JSON **逐字段比对**（`n_cand` / `n_unique_raw` / `pinned` / MC 菜单 / 每类不同 bundle / 全部候选束）
   ⇒ **0 处不同**（`match_results/collapse/fullscan_B_pinK0_after.json` vs `fullscan_B.json`）。
3. `AZAI_PIN_K` 未设 ⇒ `_env("AZAI_PIN_K","0")` = 0。

⇒ 所以"正在跑的别的批次"不会被这个开关影响，**可以放心让它留在代码里**。

### 1.5 生效旁证（本次冒烟，`AZAI_FLAT_VERBOSE=1`）

60 回合截断、2 局，`[flat] 候选 N 个` 的实测序列：

```
ai0_az_bridge_ai_seed7: n=20 vals=48,5,5,5,5,4,4,4,4,4,4,4,4,4,2,2,2,2,1,48   ⇒ n_cand==1 只占 5%
ai1_az_bridge_ai_seed7: n=23 vals=48,5,2,2,2,2,2,2,2,2,2,2,2,1,48,5,5,4,5,4,5,2,1 ⇒ 1/23 = 4.3%
```

对照现役 **82.3%** ⇒ 开关真的生效。

---

## 2. "疑似卡死"：证据与结论（**这是本文最重要的一节**）

### 2.1 现象（第一次起跑 2026-10-29 00:01:41，`--jobs=4`）

4 局（`7 7r 8 8r`）在 **135.1–135.2 min** 后落成 `verdict=timeout / winner=INVALID / terminal=False`，
只跑到 round 40-45，报文是：

```
EXC timed out reading ai_cpp_lure_v4 (need 4B, have 0B) buf_head_hex= buf_head_repr=b''
```

另一会话据此提出："候选数 1→14 可能触发了隐藏的死锁/阻塞"，旁证是
"bridge 累计 CPU 只有 52 s / 5.6 h"、"8 个 `ai_cpp_lure_v4.exe` 累计 CPU = 0"。

### 2.2 结论：**不是死锁，是整台机器进了 Modern Standby（S0 待机）**

三条独立证据，全部同向：

| # | 证据 | 内容 |
|---|---|---|
| 1 | **系统事件** | `Microsoft-Windows-Power-Troubleshooter`：`2026/10/29 7:52:19 系统已从低功耗状态中恢复`。前几次恢复：10-28 `7:53`、10-27 `9:39`、10-26 `7:32` ⇒ **这台机器每晚都睡**。`powercfg /a` 第一行就是 `待机 (S0 低电量待机)`。`STANDBYIDLE` 的 AC 值 = **1800 s（30 min）**。起跑 10 min 后（00:11，无人操作）进待机，与 30 min 空闲超时**完全吻合**。 |
| 2 | **对手自己的日志** | `ai1_runnerup_seed7_*.stderr.log` 里同一个进程相邻回合：round 42 `elapsed_us=3031714`（3.0 s）→ round 43 **`235139862`（235.1 s）** → round 45 **`7551388975`（= 2.10 h）**。亚军自己的"思考时间"涨到 2 小时 ⇒ **时间是被墙钟冻住的**，不是它算的，也不是管道写满。 |
| 3 | **我们这侧正常** | 同一局我们的 `⏱️ 决策耗时 p50=2.86s p95=30.27s max=37.19s`，最慢单决策 37 s，离 bridge 的读超时差 48 倍。 |

**最硬的反证（决定性）**：同一份代码、同一个 `PIN_K=2`、同一台机器，**加了保活之后**
（第二次起跑 07:56:36，`--jobs=8`）8 局跑满 **1325 个决策 / 160-214 回合、0 次 timeout、0 次卡死**，
速率稳定在 **8.19 回合/min ≈ 491 回合/小时**。
**死锁会复现；它没有复现。**

⇒ "候选数 1→14 触发死锁"这个假设**被否掉**。
`PIN_K=2` 确实把长尾决策的**密度**抬高了一点（`dec_ge10` 10.49% → 12.30%），
而 bridge 的**单次读包超时**（`MVS_TIMEOUT_SECONDS`，默认 300 s）在**机器被冻住**时一定先到期
⇒ 表现为 `verdict=timeout`。**它是"机器睡"的放大器，不是它自己造成的死锁。**

### 2.3 为什么会被误判成"死锁"

因为**被冻住的时候，症状和死锁一模一样**：CPU 归零、progress 不前进、bridge 在读管道。
区别只有两个地方能看到：**系统电源事件**、和**对端进程自己记的 `elapsed_us`**。
以后再遇到"所有进程 CPU 都接近 0 但没退出"，**先查 `Get-WinEvent` 的 Power-Troubleshooter**，
再谈代码。

---

## 3. 本机跑长批次的正确姿势（**必读**）

```bash
# ① 先起保活（**必须在起跑之前**；它一退出，约束自动解除，不改系统设置）
powershell -NoProfile -ExecutionPolicy Bypass -File code/test_match/_keepawake.ps1 &
#   自检：应该打出 `[keepawake] start ... flags=0x80000003` 且 `ret=0x80000000`

# ② 起看门狗（第二道防线：某局 progress 超过 N 分钟不前进就报警；-Kill 才动手）
powershell -NoProfile -ExecutionPolicy Bypass \
    -File code/test_match/_ladder_watchdog.ps1 -Tag <TAG> -StallMinutes 10
#   加 -Kill 就会**按 PID 精确**清掉那一局（该局由 --resume 语义在下一轮补跑）
```

`powercfg /change standby-timeout-ac 0` **挡不住 S0 待机**（本次实测：设了照样睡），
**只有 `SetThreadExecutionState` 靠得住**。

---

## 4. 恢复这个实验：命令照抄

```bash
# 0) 先保活（见 §3）
# 1) 冒烟（截断 60 回合，验证 PIN_K 真的生效；RETRY=0 ⇒ 只跑一轮，不重试）
ROUNDS=60 JOBS=2 RETRY=0 SEEDS="7 7r" FLAT_VERBOSE=1 TAG=PIN_K2_SMOKE \
    bash code/test_match/_pin_k2_vs_runnerup.sh
# 2) 正式 16 局（跑完再敲一遍 = 只补没 verdict=engine 的局）
JOBS=8 LADDER_TIMEOUT=10800 bash code/run_logged.sh pin_k2_ab \
    bash code/test_match/_pin_k2_vs_runnerup.sh
# 3) 读数
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_pin_k2_ab.py \
    --arm-e match_results/ladder_logs/PIN_K2_VS_RU \
    --arm-b match_results/ladder_logs/INJECT_VS_RU_B \
    --stderr-e match_results/pin_k2_vs_ru/PIN_K2_VS_RU/stderr \
    --stderr-b match_results/inject_vs_ru/B/stderr \
    --json match_results/pin_k2_vs_ru/summary.json
```

**参数取值与理由**：

| 参数 | 值 | 为什么 |
|---|---|---|
| `JOBS` | **8**（有别的批次在跑时 **4**） | 16 核，每局 = 我们 1 进程 + 对手 1 进程。2026-10-29 实测 `jobs=8` + 别人 `jobs=5` = 13 局并发仍能跑 8.19 回合/min |
| `LADDER_TIMEOUT` | **10800**（3 h） | 只放宽"单局最长等多久"，防"慢但没死"被掐断。实测一局 512 回合约 62.5 min |
| `MVS_TIMEOUT_SECONDS` | **300**（默认，与基线臂一致） | 卡死检测。本臂最慢单决策 37 s，8 倍余量。**别设成 1800**（第一次起跑那样只会让"机器睡"多浪费 30 min） |
| `RETRY` | 2（正式）/ 0（冒烟） | `--resume` 语义：只补没有 `verdict=engine` 的局 |
| `MVS_STDERR_SUFFIX` | **脚本里已强制 =1** | 不加的话同 seed 的 stderr 会互相覆盖（踩过两次） |

**预计耗时**：8 局并发 × 2 波 × ~65 min ≈ **2.2 h**（不含重试）。
**基线臂可以直接复用** `INJECT_VS_RU_B`（16 局 2W-14L、血量差 −20.12、`verdict` 全 `engine`、
`illegal` 全 0）——可比性已核对：16/16 token、座位相同、对手同一份 exe、
`BASE` 与 `_inject_vs_runnerup.sh:48` 逐字相同（脚本里有断言）。

---

## 5. 保活进程现状（**别用镜像名批量杀**）

我起的保活是 `powershell -File code/test_match/_keepawake.ps1`，进程树根在
`powershell.exe`。要停就**按 PID**：

```powershell
# 先看有哪些（会匹配到 2-3 个：外层 powershell + 内层子进程）
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -match '_keepawake' } |
    Select-Object ProcessId, ParentProcessId, CreationDate
# 再逐个 Stop-Process -Id <PID>
```

**注意**：它停了以后，机器会**在 30 min 空闲后重新进待机** ⇒ 正在跑的批次会重演 §2.1。
如果你自己起了别的保活，就把我这个停掉，两者不冲突（`SetThreadExecutionState` 是可叠加的）。

---

## 6. 如果接着做，建议的顺序（我的意见）

1. **先跑完 16 局**拿到 `配对 ΔHP`，按 `docs/pin_k2_ab_20261028.md` §0.3 的四档判读。
   预注册判据**已经写好、别改**（改了就失去"预注册"的意义）。
2. 若落在 **A 档**（`ΔHP` 显著 > 0）：再扫 `K=3/5`，但先看 §7 的耗时——`K=3` 的中位是 42 条候选，
   按 §7.1 的修正因子，p50 大概会到 5-8 s、`dec_ge10` 可能破 25% ⇒ 很可能要先降
   `AZAI_K`（候选上限 16）再谈转正。
3. 若落在 **B 档**（`ΔHP ≈ 0`）：**别再往"加候选"方向投入**。
   按预注册的下一步：测**"把 HOLD 也放进 flat 长 rollout 再决定"**
   （现在 HOLD 是 `mc_quiet` 用 4×16 回合 rollout 的**廉价仲裁**定的，在 78% 的决策里赢；
   48 束 × 64 次 × 16 回合的预算**完全没参与**这个决定）。
4. 若落在 **C 档**：`PIN_K` 只当诊断留档。

---

## 7. 顺手纠正上游的一处成本模型（**下次估预算请用修正因子**）

`docs/why_effective_candidates_collapse_20261028.md` §8.4 用
`每决策 rollout 回合数 × 62.7 µs + 固定开销 0.48 s` 估墙钟，实测偏**乐观**：

| 场景 | §8.4 的估算 | 实测 | 偏差 |
|---|---|---|---|
| `PIN_K=2` 的 p50（`n_cand` 中位 14） | ~1.4 s | **2.53 s** | ×1.8 |
| 单条 `n_cand=48` 的决策 | 49152×62.7 µs + 0.48 ≈ **3.6 s** | **26.8-31.4 s** | **×7-9** |

`62.7 µs/回合` 只在小 `n_cand` 下成立；`n_cand` 大了以后主导成本是
**叶子批量估值网的 forward**（`n_cand × 64` 个叶子）与状态拷贝，不是 rollout 推演。
⇒ **估 `PIN_K` 的成本乘 2（中位）到 8（大 `n_cand`）的修正因子。**

---

## 8. 本次留下的工件（都在，没删）

| 路径 | 内容 |
|---|---|
| `docs/pin_k2_ab_20261028.md` | 预注册（§0，**跑之前写定，未改**）+ 中止记录 + interim 耗时 + 卡死排查 |
| `docs/handoff_pin_k2_20261029.md` | 本文 |
| `code/test_match/_pin_k2_vs_runnerup.sh` | 实验臂脚本（BASE 逐字断言 / 臂开关断言 / resume / stderr 后缀 / 归档 / 四条硬断言） |
| `code/test_match/analyze_pin_k2_ab.py` | 读数汇总（配对 ΔHP + 95%CI + 符号检验 + Wilson + 耗时红线判定） |
| `code/test_match/_keepawake.ps1` | **保活**（`SetThreadExecutionState`）——本机长批次必备 |
| `code/test_match/_ladder_watchdog.ps1` | **看门狗**（默认只看不杀；`-Kill` 按 PID 精确清那一局） |
| `match_results/ladder_logs/PIN_K2_VS_RU/` | 8 局 `aborted`（我杀的）+ 8 局只有 ladder 头 |
| `match_results/pin_k2_vs_ru/PIN_K2_VS_RU/stderr/` | 30 份双方 stderr（interim 耗时来源） |
| `match_results/pin_k2_vs_ru/PIN_K2_SMOKE/stderr/` | 冒烟（证明 `n_cand` 被抬起来） |
| `match_results/pin_k2_vs_ru/_stale/` | 第一次作废批的 stderr（`stash_stderr` 挪走的） |

### 8.1 一个必须记住的坑：清场时会把**下一波**拉起来

我第一次按 PID 清场时，先杀了 8 个对局进程，`_tmp_ladder.py` 的线程**立刻拿到信号量、
把下一波 token（`11 11r …`）拉起来了**，于是在我杀完 ladder 之后又留下一批**孤儿**
（7 个 `match_via_sdk` + 7 个 `az_bridge_ai` + 7 个 `ai_cpp_lure_v4.exe`）。
**正确顺序**：先杀 ladder（不再派新局），再杀 bridge/AI。
**清完必须再查一遍后代闭包**（用已死的根 PID 当种子也能查到，`ParentProcessId` 还在）。
本文这次已经补杀干净并复核（`none left ✓`，且复核过没有 `magica` 进程被卷进来）。
