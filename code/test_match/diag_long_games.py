"""针对"长局失分"的取证：解析 ladder 日志的**终局状态**（回合数、金币、塔数）与胜负的关系。

动机：`seg_compare.py` 显示固定段的失分集中在 ~250 回合的消耗战（短局 0.790 / 长局 0.576），
但"输在长局"有很多种可能：钱没花出去（金币堆积）、塔被拆光、被拖到超时……本脚本用**现成日志**区分它们。

读法：`RESULT ... base_hp=17,0 coins=237,74` 给终局金币；正文最后一条 `[round N] towers=[...]` 给终局塔数。
用法：python code/test_match/diag_long_games.py --tag AB_MC1_dec_vs_baseline
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RES_RE = re.compile(r"RESULT\s+seed=(\d+).*?winner_side=(\S+)\s+verdict=(\S+).*?rounds=(\d+).*?base_hp=([\d,\-]+).*?coins=([\d,\-]+)")
TOW_RE = re.compile(r"\[round (\d+)\] towers=\[(.*?)\]\s")


def parse(tag: str):
    d = REPO / "match_results" / "ladder_logs" / tag
    rows = []
    for f in sorted(d.glob("*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RES_RE.search(txt)
        if not m:
            continue
        seed, side, verdict, rounds, hp, coins = m.groups()
        if verdict != "engine":
            continue
        swapped = f.stem.endswith("r")
        ai0_win = (side == "p0") if not swapped else (side == "p1")
        # 终局塔数：取最后一条 towers= 行里属于 p0/p1 的计数
        t0 = t1 = None
        for tm in TOW_RE.finditer(txt):
            body = tm.group(2)
            items = re.findall(r"\((\d+), ([01]), ", body)
            t0 = sum(1 for _id, p in items if p == "0")
            t1 = sum(1 for _id, p in items if p == "1")
        try:
            hp0, hp1 = (int(x) for x in hp.split(","))
            c0, c1 = (int(x) for x in coins.split(","))
        except ValueError:
            continue
        # 统一到"我方(p0 视角)"：换边时把 0/1 对调
        ours = dict(rounds=int(rounds), win=bool(ai0_win), coins=c0, opp_coins=c1,
                    towers=t0, opp_towers=t1)
        if swapped:
            ours["coins"], ours["opp_coins"] = c1, c0
            ours["towers"], ours["opp_towers"] = t1, t0
        rows.append(ours)
    return rows


def summarize(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"   {name}: 无有效局"); return
    med = statistics.median(r["rounds"] for r in rows)
    print(f"   {name}: n={len(rows)} 全段中位回合={med:.0f}")
    for label, sel in (("短局(<中位)", [r for r in rows if r["rounds"] < med]),
                       ("长局(>=中位)", [r for r in rows if r["rounds"] >= med])):
        w = [r for r in sel if r["win"]]; l = [r for r in sel if not r["win"]]
        if not sel:
            continue
        f = lambda rs, k: (statistics.mean(r[k] for r in rs) if rs else float("nan"))
        print(f"      {label}: n={len(sel)} 胜率={len(w)}/{len(sel)}")
        print(f"         我方终局金币 胜={f(w,'coins'):.0f} 负={f(l,'coins'):.0f} | "
              f"对手金币 胜={f(w,'opp_coins'):.0f} 负={f(l,'opp_coins'):.0f}")
        print(f"         我方终局塔数 胜={f(w,'towers'):.1f} 负={f(l,'towers'):.1f} | "
              f"对手塔数 胜={f(w,'opp_towers'):.1f} 负={f(l,'opp_towers'):.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    print("=== 长局失分取证（终局金币/塔数）")
    summarize(a.tag, parse(a.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
