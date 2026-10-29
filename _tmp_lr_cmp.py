#!/usr/bin/env python
"""V1 对照：HEAD 版 `_tmp_ladder.py` 与新版驱动跑同一批 token，**逐字段比对 RESULT 行**。

用法：
    python _tmp_lr_cmp.py LR_V1_OLD LR_V1_NEW 990 990r

判据（用户要求"RESULT 行与改动前逐字段相同"）：
  · 每个 token 的 `RESULT` 行整行字符串相同（逐字段必然相同）；
  · 再把 13 个字段单独对一遍并打印（便于人眼复核）；
  · 顺带确认新版 stdout 里**没有**新开关的区块（默认全关 ⇒ 默认行为不变）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOGS = REPO / "match_results" / "ladder_logs"

FIELDS = ["seed", "p0", "p1", "winner", "winner_side", "verdict", "engine_winner",
          "rounds", "terminal", "base_hp", "coins"]
RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) (?:winner_side=(\S+) )?verdict=(\S+) "
    r"engine_winner=(\S+) rounds=(\S+) terminal=(\S+) base_hp=(\d+),(\d+) coins=(\d+),(\d+)")


def result_lines(tag: str, token: str) -> list[str]:
    p = LOGS / tag / f"{token}.log"
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if "RESULT seed=" in ln]


def main() -> int:
    if len(sys.argv) < 4:
        print("用法: python _tmp_lr_cmp.py <old_tag> <new_tag> <token> [token ...]")
        return 2
    old_tag, new_tag, tokens = sys.argv[1], sys.argv[2], sys.argv[3:]
    all_same = True
    for token in tokens:
        lo, ln = result_lines(old_tag, token), result_lines(new_tag, token)
        print(f"\n=== token {token} ===")
        print(f"  OLD({old_tag}) {len(lo)} 条 RESULT 行")
        print(f"  NEW({new_tag}) {len(ln)} 条 RESULT 行")
        if not lo or not ln:
            print("  ✗ 有一边没有 RESULT 行（该局没跑完/日志缺失）")
            all_same = False
            continue
        for i, (a, b) in enumerate(zip(lo, ln)):
            mo, mn = RESULT_RE.search(a), RESULT_RE.search(b)
            same = (a == b)
            all_same = all_same and same
            print(f"  RESULT 行整行相同: {same}")
            for idx, name in enumerate(FIELDS):
                va = mo.group(idx + 1) if mo else None
                vb = mn.group(idx + 1) if mn else None
                flag = "✓" if va == vb else "✗"
                print(f"    {flag} {name:<14} OLD={va!r:<24} NEW={vb!r}")
            print(f"  OLD: {a}")
            print(f"  NEW: {b}")
    # 默认全关 ⇒ 新版 stdout 里不该出现新开关的区块
    runs = sorted((REPO / "training_history/runs").glob("*_ladder_resilience_v1_new"))
    if runs:
        out = (runs[-1] / "output.log").read_text(encoding="utf-8", errors="replace")
        marks = [m for m in ("[retry]", "[watchdog]", "收尾汇总", "final-sweep") if m in out]
        print(f"\n新版 stdout（{runs[-1].name}）里的新开关区块: {marks or '无'}")
        if marks:
            all_same = False
    print(f"\n==> RESULT 逐字段相同: {all_same}")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
