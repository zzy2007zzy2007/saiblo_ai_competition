#!/usr/bin/env python3
"""离线"合法性审计"：把**已打完**的一局逐回合重放，用引擎复核每个 op 是否合法（2026-10-27）。

为什么需要它：
  * 线上的影子校验（`match_via_sdk.check_ops`）只能管**从今以后**的对局；
  * 历史日志里的"非法操作 0 次"其实是**结构性 0**（`apply_operation_list` 返回 None，
    旧代码的判据永远不成立）⇒ 那些对局到底有没有非法操作，**只能事后重放才能回答**；
  * 重放还能覆盖线上检查的边角（例如"p0 本回合放 EMP ⇒ p1 建塔落在护盾格"这种**同回合跨侧**
    影响），并**校验重放本身是否忠实**（每回合操作前的 HP/金币必须与 trace 一致）。

数据来源：`code/test_match/progress.py` 写的逐回合 trace
（`training_history/progress/<tag>/az_progress_seed<seed>.txt`，格式见该文件头注释）。

用法：
    python -u code/test_match/audit_replay_illegal.py --trace <trace.txt> [--trace ...] [--seed 8101]

判据：`illegal_total=0` 且 `divergence=0`（HP/金币逐回合对得上 ⇒ 重放忠实）。
退出码：有非法操作或重放不一致 ⇒ 1。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "test_match"))
sys.path.insert(0, str(REPO / "Ant-Game"))

import os
os.add_dll_directory(str(REPO / "code" / "cpp_engine"))
os.add_dll_directory(r"C:/mingw64/bin")

import match_via_sdk as MVS  # noqa: E402
from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import OperationType  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

# progress.py 的 fmt_op 反解（**只用它自己的几种形式**，不认识就报错而不是猜）
_RE = {
    11: re.compile(r"^BUILD\((-?\d+),(-?\d+)\)$"),
    12: re.compile(r"^UPGRADE\(id=(-?\d+),tgt=(-?\d+)\)$"),
    13: re.compile(r"^DOWNGRADE\(id=(-?\d+)\)$"),
    21: re.compile(r"^LIGHTNING\((-?\d+),(-?\d+)\)$"),
    22: re.compile(r"^EMP\((-?\d+),(-?\d+)\)$"),
    23: re.compile(r"^DEFLECTOR\((-?\d+),(-?\d+)\)$"),
    24: re.compile(r"^EVASION\((-?\d+),(-?\d+)\)$"),
    31: re.compile(r"^UP_SPEED$"),
    32: re.compile(r"^UP_ANT$"),
}
_HEAD = re.compile(r"^round=(\d+) HP0=(-?\d+) HP1=(-?\d+) G0=(-?\d+) G1=(-?\d+) (.*)$")


def parse_op(tok: str) -> tuple[int, int, int]:
    """`BUILD(4,16)` / `UPGRADE(id=3,tgt=2)` → (op_type, arg0, arg1)。"""
    tok = tok.strip()
    for t, rx in _RE.items():
        m = rx.match(tok)
        if m:
            g = [int(x) for x in m.groups()]
            if t in (11, 21, 22, 23, 24):        # (x, y)
                return (t, g[0], g[1])
            if t == 12:                          # (id, target)
                return (t, g[0], g[1])
            if t == 13:                          # (id)
                return (t, g[0], -1)
            return (t, -1, -1)                   # 31/32 无参数
    m = re.match(r"^OP(\d+)\((-?\d+),(-?\d+)\)$", tok)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise ValueError(f"不认识的 op token: {tok!r}")


def parse_ops_field(s: str) -> list:
    """`[BUILD(4,16), LIGHTNING(7,3)]` → list[Operation]；`[]` → 空。"""
    s = s.strip()
    if s in ("[]", ""):
        return []
    assert s.startswith("[") and s.endswith("]"), s
    body = s[1:-1].strip()
    out = []
    for tok in body.split(", "):
        t, a0, a1 = parse_op(tok)
        out.append(Operation(OperationType(t), a0, a1))
    return out


def split_sides(rest: str) -> tuple[list, list]:
    """一行里 `P0=[...]   P1=[...]`（P0 段用空格补齐 36 列）→ (ops0, ops1)。"""
    i = rest.find("P1=")
    if i < 0:
        raise ValueError(f"没有 P1= 段: {rest!r}")
    p0s = rest[:i].strip()
    p1s = rest[i:]
    assert p0s.startswith("P0="), p0s
    return parse_ops_field(p0s[3:]), parse_ops_field(p1s[3:])


def audit(trace: Path, seed: int | None) -> tuple[int, int, int]:
    """重放一局。返回 (rounds, illegal_total, divergence)。"""
    if seed is None:
        m = re.search(r"seed(\d+)", trace.name)
        seed = int(m.group(1)) if m else 0
    stats = MVS.IllegalStats()
    st = GameStateFacade.initial(seed=int(seed), cold_handle_rule_illegal=False)
    rounds = 0
    diverge = 0
    bad_all: list[dict] = []
    terminal_line = ""
    for ln in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.rstrip()
        if not ln:
            continue
        if ln.startswith("terminal"):
            terminal_line = ln
            break
        m = _HEAD.match(ln)
        if not m:
            print(f"  ⚠️ 跳过不认识的行: {ln[:80]!r}", flush=True)
            diverge += 1
            continue
        rnd = int(m.group(1))
        hp = (int(m.group(2)), int(m.group(3)))
        coins = (int(m.group(4)), int(m.group(5)))
        ops0, ops1 = split_sides(m.group(6))
        # 重放忠实性：操作前的 HP / 金币必须与 trace 一致
        cur_hp = tuple(int(b.hp) for b in st.bases)
        cur_coins = tuple(int(c) for c in st.coins)
        if rnd != int(st.round_index) or cur_hp != hp or cur_coins != coins:
            diverge += 1
            if diverge <= 5:
                print(f"  ⚠️ 重放分歧 round={rnd} (引擎 round={st.round_index}) "
                      f"hp {cur_hp} vs {hp}  coins {cur_coins} vs {coins}", flush=True)
        bad0 = MVS.check_ops(st, 0, ops0, stats, seed=int(seed), round_index=rnd,
                             tag="replay", cold=False, log=lambda s: print(f"  {s}", flush=True))
        bad_all += bad0
        if bad0:
            break
        st.apply_operation_list(0, ops0)
        bad1 = MVS.check_ops(st, 1, ops1, stats, seed=int(seed), round_index=rnd,
                             tag="replay", cold=False, log=lambda s: print(f"  {s}", flush=True))
        bad_all += bad1
        if bad1:
            break
        st.apply_operation_list(1, ops1)
        st.advance_round()
        rounds += 1
    print(f"  {stats.summary(int(seed), False)} rounds={rounds} "
          f"trace_terminal={terminal_line!r} replay_terminal={bool(st.terminal)} "
          f"divergence={diverge}", flush=True)
    return rounds, stats.total, diverge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", default=[], help="progress trace 文件（可多次）")
    ap.add_argument("--glob", default="", help="批量：相对仓库根的 glob（只审计**已打完**的，"
                                              "即末行是 terminal 的）")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    traces = [Path(t) if Path(t).is_absolute() else (REPO / t) for t in args.trace]
    if args.glob:
        cand = sorted(REPO.glob(args.glob))
        done: list[Path] = []
        skipped = 0
        for p in cand:
            try:
                tail = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            if tail.startswith("terminal"):
                done.append(p)
            else:
                skipped += 1
        print(f"[glob] 命中 {len(cand)} 个 trace：已打完 {len(done)}，跳过（未结束/空）{skipped}",
              flush=True)
        traces += done
    if not traces:
        print("必须给 --trace 或 --glob", flush=True)
        return 2
    tot_illegal = 0
    tot_diverge = 0
    tot_rounds = 0
    for tr in traces:
        print(f"\n== 重放 {tr.relative_to(REPO) if tr.is_relative_to(REPO) else tr} ==", flush=True)
        if not tr.exists():
            print("  !! 文件不存在", flush=True)
            tot_diverge += 1
            continue
        rounds, illegal, diverge = audit(tr, args.seed)
        tot_illegal += illegal
        tot_diverge += diverge
        tot_rounds += rounds
    print(f"\n=== 合计: {len(traces)} 局 / {tot_rounds} 回合: illegal_total={tot_illegal} "
          f"divergence={tot_diverge} ===", flush=True)
    return 1 if (tot_illegal or tot_diverge) else 0


if __name__ == "__main__":
    raise SystemExit(main())
