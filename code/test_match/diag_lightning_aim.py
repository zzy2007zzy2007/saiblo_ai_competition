"""诊断：**闪电打在哪里？** —— 从对局日志里量"我方闪电落点到敌方基地的六边形距离"。

动机：A1 配置下我方的胜负是**基地竞赛**（32 局里 16 局我方基地被打爆、16 局对方被打爆），
而我方**唯一武器就是闪电**（11.2 次/局、从不建塔）。所以"闪电打哪儿"很可能就是 16.9pp 的来源。
本脚本**只读日志**（不跑对局、不要引擎）：

  * 从每局日志里抓 `USE_LIGHTNING_STORM` 的操作坐标（`arg0=x, arg1=y`）；
  * 用**官方几何**（`SDK.utils.geometry.hex_distance`）算它到**敌方基地**的距离
    （`PLAYER_BASES = ((2,9),(16,9))`，`constants.py:12`）；
  * 分别统计**我方**与**对手**的落点距离分布，并按**胜负**切开看我方。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_lightning_aim.py \
      --tag=C1_a1_hist32 --seeds=7-22
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Ant-Game"))
sys.path.insert(0, str(REPO / "code"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from SDK.utils.constants import PLAYER_BASES            # noqa: E402
from SDK.utils.geometry import hex_distance             # noqa: E402

LT_RE = re.compile(r"Operation\(op_type=<OperationType\.USE_LIGHTNING_STORM: 21>, arg0=(\d+), arg1=(-?\d+)\)")
OPS_LINE_RE = re.compile(r"\[round (\d+)\] p(\d)\(([^)]+)\)")
RESULT_RE = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+)")


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="7-22")
    ap.add_argument("--us-label", default="az_bridge_ai")
    a = ap.parse_args()

    d = REPO / "match_results" / "ladder_logs" / a.tag
    seeds = parse_seeds(a.seeds)
    ours, theirs = [], []
    ours_by_outcome = collections.defaultdict(list)
    per_game = []

    for s in seeds:
        for suf in ("", "r"):
            f = d / f"{s}{suf}.log"
            if not f.is_file():
                continue
            us_p0 = (suf == "")
            enemy_base = PLAYER_BASES[1] if us_p0 else PLAYER_BASES[0]
            txt = f.read_text(encoding="utf-8", errors="replace")
            m = RESULT_RE.search(txt)
            winner = m.group(4) if m else "?"
            us_label = a.us_label
            us_won = (winner == us_label)
            n_us = n_them = 0
            for ln in txt.splitlines():
                mo = OPS_LINE_RE.search(ln)
                if not mo:
                    continue
                side = int(mo.group(2))
                is_us = (side == 0) == us_p0
                for x, y in LT_RE.findall(ln):
                    dist = hex_distance(int(x), int(y), *enemy_base)
                    (ours if is_us else theirs).append(dist)
                    if is_us:
                        ours_by_outcome["胜" if us_won else "负"].append(dist)
                        n_us += 1
                    else:
                        n_them += 1
            per_game.append((f"{s}{suf}", "胜" if us_won else "负", n_us, n_them))

    def stats(xs: list[int]) -> str:
        if not xs:
            return "n=0"
        c = collections.Counter(xs)
        return (f"n={len(xs)} mean={st.mean(xs):.2f} median={st.median(xs):.1f} "
                f"min={min(xs)} max={max(xs)}  分布={dict(sorted(c.items()))}")

    print(f"=== {a.tag}  seeds={seeds[0]}..{seeds[-1]} ===")
    print("【我方闪电 → 敌方基地 的六边形距离】")
    print("  " + stats(ours))
    print("【对手闪电 → 我方基地 的距离】")
    print("  " + stats(theirs))
    print()
    for k in ("胜", "负"):
        if ours_by_outcome[k]:
            print(f"我方[{k}]局：" + stats(ours_by_outcome[k]))
    print()
    print("逐局（我方闪电次数 / 对手闪电次数）：")
    for tok, out, nu, nt in per_game:
        print(f"  {tok:>5}  {out}  我方={nu:3d}  对手={nt:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
