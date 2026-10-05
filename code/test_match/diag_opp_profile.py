"""诊断：**对手（rule_v4）在我方胜局与负局里，打法有什么不同？**

动机：A1 下我方 32 局恰好 16 胜 16 负（全是基地竞赛）。若"我方输的局里对手建塔/升级更多"，
说明**经济**是它赢的原因（那 D 的失败就只是"建太多"而不是"建塔没用"）；若只在"闪电更多"上
有差别，说明是**打击密度**的事。本脚本只读日志。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_opp_profile.py --tag=C1_a1_hist32 --seeds=7-22
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) (?:winner_side=(\S+) )?verdict=(\S+) "
    r"engine_winner=(\S+) rounds=(\d+) terminal=(\S+) base_hp=(\d+),(\d+) coins=(\d+),(\d+)")
OPS_LINE_RE = re.compile(r"\[round (\d+)\] p(\d)\(([^)]+)\)")
OPNAME_RE = re.compile(r"OperationType\.(\w+):")


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
    rows = []
    for s in parse_seeds(a.seeds):
        for suf in ("", "r"):
            f = d / f"{s}{suf}.log"
            if not f.is_file():
                continue
            us_p0 = (suf == "")
            txt = f.read_text(encoding="utf-8", errors="replace")
            m = RESULT_RE.search(txt)
            if not m:
                continue
            win = m.group(4)
            us_won = (win == a.us_label)
            ours, theirs = collections.Counter(), collections.Counter()
            act_us = act_them = 0
            first_act = {0: None, 1: None}
            for ln in txt.splitlines():
                mo = OPS_LINE_RE.search(ln)
                if not mo:
                    continue
                side = int(mo.group(2))
                names = OPNAME_RE.findall(ln)
                if not names:
                    continue
                tgt = ours if ((side == 0) == us_p0) else theirs
                for nm in names:
                    tgt[nm] += 1
                if (side == 0) == us_p0:
                    act_us += 1
                else:
                    act_them += 1
                if first_act[side] is None:
                    first_act[side] = int(mo.group(1))
            hp = (int(m.group(10)), int(m.group(11)))
            rows.append({"token": f"{s}{suf}", "won": us_won, "rounds": int(m.group(8)),
                         "our_hp": hp[0] if us_p0 else hp[1], "opp_hp": hp[1] if us_p0 else hp[0],
                         "ours": ours, "theirs": theirs, "act_us": act_us, "act_them": act_them,
                         "their_first": first_act[1 if us_p0 else 0]})

    def summarise(sel, key_side, op):
        xs = [r[key_side][op] for r in sel]
        return f"{st.mean(xs):7.1f}" if xs else "   n/a"

    print(f"=== {a.tag}  共 {len(rows)} 局（我方 {sum(r['won'] for r in rows)} 胜）===\n")
    for label, sel in (("我方胜局", [r for r in rows if r["won"]]),
                       ("我方负局", [r for r in rows if not r["won"]])):
        print(f"--- {label}（n={len(sel)}）---")
        print(f"  回合数        平均 {st.mean([r['rounds'] for r in sel]):7.1f}")
        print(f"  我方 建塔/升级/闪电/出招   "
              f"{summarise(sel,'ours','BUILD_TOWER')} / {summarise(sel,'ours','UPGRADE_TOWER')} / "
              f"{summarise(sel,'ours','USE_LIGHTNING_STORM')} / "
              f"{st.mean([r['act_us'] for r in sel]):7.1f}")
        print(f"  对手 建塔/升级/闪电/出招   "
              f"{summarise(sel,'theirs','BUILD_TOWER')} / {summarise(sel,'theirs','UPGRADE_TOWER')} / "
              f"{summarise(sel,'theirs','USE_LIGHTNING_STORM')} / "
              f"{st.mean([r['act_them'] for r in sel]):7.1f}")
        fa = [r["their_first"] for r in sel if r["their_first"] is not None]
        print(f"  对手首次出招回合 中位 {st.median(fa) if fa else 'n/a'}")
        print(f"  我方终局基地血量 平均 {st.mean([r['our_hp'] for r in sel]):7.1f} | "
              f"对手 {st.mean([r['opp_hp'] for r in sel]):7.1f}")
    print("\n逐局明细：")
    for r in rows:
        print(f"  {r['token']:>5} {'胜' if r['won'] else '负'}  r={r['rounds']:3d}  "
              f"我方hp={r['our_hp']:2d} 对手hp={r['opp_hp']:2d}  "
              f"我方(建{r['ours']['BUILD_TOWER']:3d} 升{r['ours']['UPGRADE_TOWER']:2d} "
              f"闪{r['ours']['USE_LIGHTNING_STORM']:2d} 招{r['act_us']:3d})  "
              f"对手(建{r['theirs']['BUILD_TOWER']:3d} 升{r['theirs']['UPGRADE_TOWER']:2d} "
              f"闪{r['theirs']['USE_LIGHTNING_STORM']:2d} 招{r['act_them']:3d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
