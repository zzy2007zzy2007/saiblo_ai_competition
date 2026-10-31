#!/usr/bin/env python
"""汇总值网 A/B（`_value_ab_vs_ru.sh`）的对局日志：胜率 + 血量差，**按 (seed, 座位) 配对**。

读什么：`match_results/value_ab/<TAG>/*.log` 里 match_via_sdk 的 RESULT 行：

    RESULT seed=7 p0=az_bridge_ai p1=ai_cpp_lure_v4 winner=az_bridge_ai winner_side=p0 \
           verdict=engine engine_winner=0 rounds=433 terminal=True base_hp=12,0 coins=..

座位：token 无 `r` ⇒ 我们（AI0）坐 P0；token 带 `r` ⇒ 我们坐 P1（match_via_sdk.py:936-938）。

输出：每臂 W/L/胜率/均值血差；两臂之间按 (seed, 座位) 配对的差值（同 seed 左右手各一局）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RE_RESULT = re.compile(
    r"RESULT\s+seed=(\d+)\s+p0=(\S+)\s+p1=(\S+)\s+winner=(\S+)\s+winner_side=(p[01]|None)"
    r".*?verdict=(\S+).*?rounds=(\d+).*?base_hp=(\d+),(\d+)")


def parse_dir(d: Path) -> list[dict]:
    rows = []
    for f in sorted(d.glob("*.log")):
        tok = f.stem
        seat = 1 if tok.endswith("r") else 0
        txt = f.read_text(encoding="utf-8", errors="replace")
        for m in RE_RESULT.finditer(txt):
            seed, p0, p1, winner, wside, verdict, rounds, hp0, hp1 = m.groups()
            hp = (int(hp0), int(hp1))
            rows.append({"token": tok, "seed": int(seed), "seat": seat,
                         "verdict": verdict, "rounds": int(rounds),
                         "winner_side": (None if wside == "None" else int(wside[1])),
                         "hp": hp, "our_hp": hp[seat], "opp_hp": hp[1 - seat],
                         "win": (wside != "None" and int(wside[1]) == seat),
                         "draw": (wside == "None")})
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    w = sum(1 for r in rows if r["win"])
    d = sum(1 for r in rows if r["draw"])
    l = n - w - d
    margin = [r["our_hp"] - r["opp_hp"] for r in rows]
    return {"n": n, "W": w, "D": d, "L": l,
            "winrate": (w + 0.5 * d) / n,
            "margin_mean": sum(margin) / n,
            "margin_list": margin,
            "illegal": sum(1 for r in rows if r["verdict"] != "engine")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True, help="一个或多个臂的日志目录")
    ap.add_argument("--name", nargs="*", default=None, help="对应臂名（缺省用目录名）")
    a = ap.parse_args()
    names = a.name or [Path(d).name for d in a.dir]
    all_rows = {}
    print(f"{'臂':<18}{'n':>4}{'W':>4}{'D':>4}{'L':>4}{'胜率':>8}{'均值血差':>10}{'非法':>6}")
    for nm, d in zip(names, a.dir):
        rows = parse_dir(Path(d))
        all_rows[nm] = rows
        s = summarize(rows)
        if not s:
            print(f"{nm:<18}{'(空)':>4}")
            continue
        print(f"{nm:<18}{s['n']:>4}{s['W']:>4}{s['D']:>4}{s['L']:>4}"
              f"{s['winrate']*100:>7.1f}%{s['margin_mean']:>+10.2f}{s['illegal']:>6}")
    # 配对比较（前两个臂）
    if len(all_rows) >= 2:
        (na, ra), (nb, rb) = list(all_rows.items())[:2]
        ka = {(r["seed"], r["seat"]): r for r in ra}
        kb = {(r["seed"], r["seat"]): r for r in rb}
        common = sorted(set(ka) & set(kb))
        if common:
            dw = sum(1 for k in common if kb[k]["win"] and not ka[k]["win"])
            dl = sum(1 for k in common if ka[k]["win"] and not kb[k]["win"])
            dm = [kb[k]["our_hp"] - kb[k]["opp_hp"] - (ka[k]["our_hp"] - ka[k]["opp_hp"])
                  for k in common]
            print(f"\n[配对] {len(common)} 对（同 seed 同座位）")
            print(f"  {nb} 赢 而 {na} 没赢：{dw} 对；反之 {dl} 对")
            print(f"  血差差（{nb} − {na}）均值 = {sum(dm)/len(dm):+.2f} HP"
                  f"（中位 {sorted(dm)[len(dm)//2]:+.1f}）")
    out = REPO / "match_results/value_ab" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: summarize(v) for k, v in all_rows.items()},
                             ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
