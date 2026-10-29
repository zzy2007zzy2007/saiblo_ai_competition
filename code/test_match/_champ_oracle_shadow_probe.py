#!/usr/bin/env python
"""探针：**让冠军 oracle 的影子局面保持同步**（2026-10-29，E4 的真正前置条件）。

## 为什么原来的 oracle 是个"永远 HOLD"的机器（已从源码 + 16 局数据确认）

`其他版本ai/ant-war2-magica-v3/main.cpp`：
```
OnRoundState  → CompareProjectedState(shadow, state)；不一致 ⇒ MarkDesynced + force_pass_ = true
get_actions   → request.force_pass = force_pass_ || !healthy  ⇒ 直接返回 pass（HOLD）
```
`force_pass_` 只在 `OnMatchStart` 里被复位 ⇒ **一次 mismtach = 整局 HOLD**。
而我们喂它的局面是**真对局**（我方真实操作被执行），它影子里的自己却执行**它自己挑的操作** ⇒
两边从第 1 个非 HOLD 决策起就分叉（日志 `field=coins[self]`）⇒ 之后它每回合都 HOLD。
实测 E2 16 局：**4177/4180 = 99.93% 的回合它给的是 `()`**（`[xor]` 行逐条数出来的）。

最后影子里的自己从不升级/不建塔 ⇒ 只挨打 ⇒ 血在影子模型里归零 ⇒ `is_ended()` ⇒
下一次 `ReplacePendingOperations` 抛 `apply_operations.called_after_terminal` ⇒ abort（0xC0000409）。
⇒ **崩溃只是表面**："会崩"和"永远 HOLD"是**同一个根因**（影子不同步）。

## 本探针要验证的修法：喂"它自己的世界"

不再把**真对局**的局面喂它，而是维护一份**它视角下的局面**：
  * 我方（它自己）执行**它挑的操作**；对手执行**对手的真实操作**；
  * 每回合把这个世界推进一步，把它的局面文本喂给它。
这样它的影子局面与喂进去的局面**同源** ⇒ 不 mismatch ⇒ 不 force_pass ⇒
每回合都真的搜索、真的出招，且影子里的自己也会防守/升级 ⇒ 血不会平白归零 ⇒ 不崩。

代价（必须写进报告）：它的答案回答的是"**在它自己那条线上**它怎么走"，
不是"在我们这条线上它怎么走"。对 E4（把它的手当候选）来说这是**可接受的近似**，
但它比"真实对局里的冠军"多了一层反事实，必须在报告里点名。

## 用法

```bash
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/_champ_oracle_shadow_probe.py \
    --log match_results/ladder_logs/US_VS_CHAMPION_BASE/11.log --max-rounds 512
```
输出：`[shadow-sum]` 一行（HOLD 率 / 存活回合 / 延迟）+ 与真实冠军动作的一致率（参考）。
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

OP_RE = re.compile(r"Operation\(op_type=<OperationType\.(\w+): (-?\d+)>, arg0=(-?\d+), arg1=(-?\d+)\)")
ROUND_RE = re.compile(r"^\s*\[round (\d+)\] (p0|p1)\(([^)]*)\) .*?-> \[(.*)\]\s*$")
RESULT_RE = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+)")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_rounds(log: Path):
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = RESULT_RE.search(txt)
    if not m:
        raise SystemExit(f"!! {log} 里没有 RESULT 行")
    seed = int(m.group(1))
    labels = {0: m.group(2), 1: m.group(3)}
    champ_seat = 0 if labels[0] == "champion" else (1 if labels[1] == "champion" else None)
    if champ_seat is None:
        raise SystemExit(f"!! 两侧都不是 champion: {labels}")
    rounds: dict[int, dict[int, list]] = {}
    for line in txt.splitlines():
        mm = ROUND_RE.match(line)
        if not mm:
            continue
        r, who, ops_txt = int(mm.group(1)), int(mm.group(2)[1:]), mm.group(4)
        rounds.setdefault(r, {})[who] = [(n, int(a0), int(a1)) for (n, _v, a0, a1) in OP_RE.findall(ops_txt)]
    return seed, champ_seat, rounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--mode", choices=("shadow", "real"), default="shadow",
                    help="shadow = 喂它自己的世界（本探针的修法）；real = 喂真对局（复现 100% HOLD）")
    args = ap.parse_args()

    log = (REPO / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    seed, champ_seat, rounds = parse_rounds(log)
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    ro = _load("runnerup_oracle", REPO / "code" / "test_match" / "runnerup_oracle.py")
    champ = REPO / "其他版本ai" / "ant-war2-magica-v3" / "magica_v3_O0.exe"
    stderr = REPO / "match_results" / "oracle_logs" / f"probe_shadow_{args.mode}_seed{seed}.stderr.log"
    stderr.parent.mkdir(parents=True, exist_ok=True)

    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    other = 1 - champ_seat
    n_ask = n_hold = n_agree = 0
    ms: list[float] = []
    crash = None
    t0 = time.time()
    print(f"[probe] mode={args.mode} log={log.name} seed={seed} 冠军座位=P{champ_seat}", flush=True)
    try:
        with ro.RunnerUpOracle(seed=seed, player=champ_seat, exe=str(champ),
                               timeout=float(args.timeout), stderr_path=stderr) as orc:
            for r in range(min(args.max_rounds, max(rounds) + 1)):
                if st.terminal:
                    print(f"[probe] r{r} 世界已终局（shadow 模式下 = 它自己那条线打完了）", flush=True)
                    break
                ops_r = rounds.get(r, {})
                # ★ 必须登记对手操作块（第一版漏了 —— 那样它的影子里对手从不花钱，
                #   `coins[对手]` 每回合都 mismatch，测出来全是 HOLD，等于白测）：
                #   协议 P1：state(r) → opp_ops(r)；P0：opp_ops(r-1) → state(r)。
                oth_r = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(other, [])]
                if champ_seat == 1:
                    orc.set_opponent_ops(r, [Operation(t, a0, a1) for (t, a0, a1) in oth_r])
                elif r >= 1:
                    prev = [(OperationType[n], a0, a1)
                            for (n, a0, a1) in rounds.get(r - 1, {}).get(other, [])]
                    orc.set_opponent_ops(r - 1, [Operation(t, a0, a1) for (t, a0, a1) in prev])
                ta = time.monotonic()
                got = orc.ask(st)
                ms.append((time.monotonic() - ta) * 1000.0)
                n_ask += 1
                got_t = [(OperationType(int(o.op_type)).name, int(o.arg0), int(o.arg1)) for o in got]
                if not got_t:
                    n_hold += 1
                # 参考读数：与"真对局里冠军那一手"的一致率（shadow 模式下**不该**当准确性读）
                want = [(n, a0, a1) for (n, a0, a1) in ops_r.get(champ_seat, [])]
                n_agree += int(got_t == want)
                if args.mode == "shadow":
                    # ★ 关键：它挑的操作**真的**落进它自己的世界（它影子里的自己也是这么走的）
                    if got_t:
                        st.apply_operation_list(champ_seat,
                                                [Operation(OperationType[n], a0, a1) for (n, a0, a1) in got_t])
                else:
                    # 复现旧行为：世界里执行的是**真对局**里冠军的操作（它影子里的自己 ≠ 世界）
                    if want:
                        st.apply_operation_list(champ_seat,
                                                [Operation(OperationType[n], a0, a1) for (n, a0, a1) in want])
                if oth_r:
                    st.apply_operation_list(other, [Operation(t, a0, a1) for (t, a0, a1) in oth_r])
                st.advance_round()
                if r % 50 == 0:
                    print(f"[probe] r{r:<4} ask#{n_ask} HOLD={n_hold} 与真实冠军一致={n_agree} "
                          f"最近 {ms[-1]:.0f}ms hp={[b.hp for b in st.bases]} "
                          f"op类型={[t for (t, _a, _b) in got_t]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        crash = f"{type(exc).__name__}: {exc}"
        print(f"[probe] !! 第 {n_ask} 次 ask 崩: {crash}", flush=True)

    dt = time.time() - t0
    print(f"\n[shadow-sum] mode={args.mode} asks={n_ask} HOLD={n_hold}({100.0 * n_hold / max(1, n_ask):.1f}%) "
          f"agree_real={n_agree}({100.0 * n_agree / max(1, n_ask):.1f}%) "
          f"wall={dt:.1f}s avg_ms={sum(ms) / max(1, len(ms)):.0f} max_ms={max(ms) if ms else -1:.0f} "
          f"crash={crash or 'None'} "
          f"rounds={st.round_index}", flush=True)
    return 0 if crash is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
