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
* **建议（未做，等你定）**：
  1. RESULT 行改成打印**真实异常类型**（从 `exception` 字符串里切出 `TypeName:`），因为现在的 `exc=str` 会把人引向"原生崩溃"；
  2. 我方 AI **每手决策记录耗时**（超过阈值就打印警告）⇒ 这类"慢"故障一眼可见，而不是伪装成 INVALID；
  3. 把 `TIMEOUT_SECONDS` 变成环境变量并在跑批时显式登记（协议层面：**"超时"应记为"太慢"而不是"无效"**，
     否则慢的方法会被误判）；
  4. 任何"在搜索树内做昂贵评估"的设计，都必须**先量每手耗时**再上批量。

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
