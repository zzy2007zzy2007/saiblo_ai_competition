# 诊断报告：`pos_pin=mc_light` 的"原生崩溃"其实是 **bridge 的 300 秒读超时**（2026-10-06）

> 结论先行：**不是引擎 bug，也没有原生崩溃**。是**我方 AI 某一手算得太久**（MC 被写在搜索树的每次节点展开里），
> 超过 `match_via_sdk.py` 的 `TIMEOUT_SECONDS = 300`，bridge 判 `aborted` ⇒ 记成 `winner=INVALID`。
> 这也解释了当初 M6 验收批 **41/128 局 INVALID（32%）**。

## 1. 现象（当初的观察）

* `M6_mc_128` / `M6_retry8`：`seed 11` 我方先手**两次**都在 **round 55** 中止：
  `RESULT ... winner=INVALID verdict=aborted exc=str rounds=55 terminal=False base_hp=46,47 coins=0,0`；
* 我方 stderr **没有任何 Python traceback**，看起来像"C++ 原生崩溃（无栈）"；
* 换边（`11r`）正常打完；单局跑同一局面也正常。

## 2. 决定性证据（本轮）

| # | 证据 | 读数 |
|---|---|---|
| 1 | **真实异常**（bridge 会打印 `异常: ...` 一行）| `TimeoutError: timed out reading az_bridge_ai (need 4B, have 0B) buf_head_hex= buf_head_repr=b''` |
| 2 | 含义 | bridge 连**4 字节长度前缀**都没收到 ⇒ **AI 那一手一个字都没写**（不是"包写坏了/分帧错位"）|
| 3 | **用时**（`ENG_mem8` 8 局并发）| 挂掉的局：`5.2 / 5.5 / 5.7 / 6.6 min` ≈ 对局正常时长 + **300 s 超时** |
| 4 | **faulthandler**（`MVS_FAULTHANDLER=1` 给 AI 加 `-X faulthandler`）| 全部 stderr **零** `Fatal Python error / access violation / Current thread` 标记 ⇒ **没有原生故障** |
| 5 | **挂掉的时刻** | `seed 11: round=55 coins=[95,63]`；`seed 18: round=26 coins=[99,89]` ⇒ **都是金币首次突破闪电价（~90）** |
| 6 | 与门控对照 | `mc_light` 的门控要求"闪电**真的放得出来**"（含金币）⇒ **金币到 90 之前 MC 完全不动**；一到就触发 |
| 7 | 单局 vs 并发 | 同一局面**单局**（`ENG_mc_light_s11`）**正常通过**（跑到 160+ 回合）；**8 局并发**才超时 ⇒ CPU 争抢 |
| 8 | 为什么 `exc=str` | bridge 第 397 行 `result["exception"] = f"{type(exc).__name__}: {exc}"` ⇒ **异常被转成字符串**，RESULT 行打印的 `exc=` 只是"字符串的类型"，**完全不含信息** |

## 3. 机理（为什么会慢到这个程度）

`mc_light` 的蒙特卡洛比较**写在 `_expand` 内部**，而 `_expand` 在 **256 次搜索迭代里对每个被访问的叶节点**都会被调用
（`bundle_mcts.py` → `search()` 的 `value = self._expand(node, self.k)`）。
⇒ **每一次节点展开都要跑 2–3 个候选 × 最多 100 回合 rollout × 每回合「网络前向 + 落子」**。
⇒ 一手决策的开销可达**分钟级**；8 局并发时更糟，轻松越过 300 s。

（同一现象后来被我以"MC 每局跑 ~780 次、连对手节点都仲裁"的形式观测到——那正是"写在 `_expand` 里"的后果。
`mc_quiet` 之所以没事：它的 rollout **只 `advance_round()`、不调网络**；`mc_root_only` 又把它压到**每次搜索 1 次**。）

## 4. 影响与修正

* **对已记录的结论**：无影响。`M6_mc_128` 那批本来就因 32% INVALID 被**整批判为无效**（未采信）；
  `M6b`（`mc_quiet`）与 `M6c-ROOT`（+ 根节点）两批**无效 0%**，与本次诊断一致（它们不慢）。
* **对你（用户）此前的猜测**：**"重新封装的 C++ 引擎有 bug"这条不成立**——本次症状里**没有任何原生故障证据**；
  换来的好消息是**裁判（judge）完整性没有被这件事动摇**。
  （⚠️ 但这只排除**这一个症状**；真要有别的原生故障，需要另找证据。）
* **已做的小修**：`match_via_sdk.py` 加 `MVS_FAULTHANDLER=1` 开关（默认关）⇒ 以后真遇到原生崩溃能直接拿到栈。
* **建议（**2026-10-06 用户批准，三条已做**）**：
  1. ✅ **RESULT 行打印真实异常类型**：`result["exception_type"]` 单独存；INVALID 行现在打 `exc=TimeoutError`
     （以前恒为 `exc=str` ⇒ 会把人引向"原生崩溃"）；
  2. ✅ **我方 AI 每手决策耗时埋点**：`AZAI_SLOW_DECISION_SEC`（默认 20s）以上打 `⚠️ 慢决策` 告警，
     退出报告带 `dec_n / dec_max / dec_slow`；
  3. ✅ **`MVS_TIMEOUT_SECONDS`（默认 300）** + 超时单独判词：超时局记 **`verdict=timeout`**，
     分析器把它**从"坏"里单列**（`⏱️ 超时局 ...`），但**仍然算无效、绝不计分**（"慢"与"坏"分开看，但不放过）。
  4. ⏳ 未做（建议保留）：任何"在搜索树内做昂贵评估"的设计，**先量每手耗时**再上批量。

### 4.2 落地时又发现并修掉的 3 个静默缺陷（2026-10-06 19:0x）

| # | 缺陷 | 后果 | 修法 |
|---|---|---|---|
| C | `match_results/ai?_*_seed*.stderr.log` **不按 tag 分目录** | 跨批跑同一 seed **互相覆盖** ⇒ 事后无法复核"那局慢在哪" | ladder 把本局两个 AI 的 stderr **归档进 `<tag>/stderr/`**（只复制；用 `--ai0/--ai1` stem 过滤）|
| D | `dec_max/dec_slow` 只在**退出报告**里 | bridge 结束时 `terminate()` ⇒ **正常对局里永远打不出来**（只在异常局可见）| AI **对局中周期性**打印 `⏱️ 耗时累计`（`AZAI_TIMING_EVERY`，默认 25 手）；分析器从归档汇总 |
| E | 计时只写在 `player == 0` 分支 | **镜像局（我方执后手）没有耗时数据** | 抽出 `_decide_timed()`，覆盖两处决策点 |
| F | 我自己引用不存在的常量 `MATCH_RESULTS`，且 `except: pass` | 归档**静默失效**整整一轮 | 改用 `LOG_ROOT.parent`，失败**打可见警告** |

**验收**：分析器现在打印 `⏱️ 单决策耗时（本 tag 归档 N 个 AI 日志）: dec_max 最大 = …s, dec_slow 合计 = …`，
且 `⏱️ 超时局 …` 与本行**同在分析器输出里** ⇒ 章程「跑批必看两类诊断」可一处执行。
⚠️ 归档只对 **2026-10-06 之后**跑的批次有效。

### 4.1 顺带挖出的两个**真缺陷**（都已修，且都曾把信息埋掉）

* **缺陷 A（严重）**：`_tmp_ladder.RESULT_RE` 要求行里有 `engine_winner=` 字段，而 bridge 的 **INVALID 行从来不打印它**
  ⇒ **无效/超时局从来没有被解析过**，分析器只能显示「winner 缺失/rounds=0」
  ⇒ 「第 55 回合超时」这条关键信息**整整被埋了一轮**。
  修：INVALID 行补齐 `winner_side=None engine_winner=None`（改法可用真 `RESULT_RE` 逐例复验：缺该字段 / `exc=` 居中
  ⇒ **不匹配**；补齐且 `exc=` 在行尾 ⇒ **匹配**）；
  另外分析器现在会把"**该行未被解析**"单独标出来，防止同类问题再潜伏。
* **缺陷 B**：我第一版把 `exc=` 插在 `engine_winner=` 与 `rounds=` 之间 ⇒ **又把正则挡住**（正则要求这两者相邻）
  ⇒ 已把 `exc=` 挪到**行尾**。教训：**改对外输出格式后，必须用消费它的那个正则做单元测试**（本次连踩两次；✅ **该测试现已入库**：`code/test_match/test_result_line_formats.py`，`python -u code/test_match/test_result_line_formats.py`，失败返回码 1）。

## 5. 复现方法（都在库里）

```bash
# ① 单局（不崩、通过）：验证"同一局面单跑没问题"
MVS_FAULTHANDLER=1 AZAI_POSPIN=mc_light AZAI_MC_EVERY=4 AZAI_MC_HORIZON=100 \
  python -u _tmp_ladder.py --tag=ENG_mc_light_s11 --jobs=1 \
  --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py 11

# ② 8 局并发（会超时 ⇒ INVALID）：验证"并发才触发"
MVS_FAULTHANDLER=1 ... --tag=ENG_mem8 --jobs=8 ... 11 11r 13 13r 14 14r 18 18r
#   然后看：match_results/ladder_logs/ENG_mem8/11.log 里的 `异常: TimeoutError: timed out reading ...`
#          以及 match_results/ai0_az_bridge_ai_seed11.stderr.log 的最后一行（停在 coins≈90 那一手）

# ③ 离线重放（对状态/链路做定点实验）
python -X faulthandler -u code/test_match/repro_rollout_crash.py --seed 11 --stop-round 55   # 手写贪心 rollout：不崩
python -X faulthandler -u code/test_match/repro_search_crash.py --seed 11 --stop-round 55 \
   --pos-pin mc_light --mc-every 4 --mc-horizon 100 --out training_history/vprior/crash_trace2.log
#   ⚠️ ③ 里那次 search 的 mc_taken=0（round 55 与门控不对齐）⇒ 没走到 MC 分支，这是脚本的局限，已记档
```
