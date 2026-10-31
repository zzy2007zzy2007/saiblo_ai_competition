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
    out = {"n": n, "W": w, "D": d, "L": l,
           "winrate": (w + 0.5 * d) / n,
           "margin_mean": sum(margin) / n,
           "margin_list": margin,
           "illegal": sum(1 for r in rows if r["verdict"] != "engine")}
    out["wilson"] = _wilson(w + 0.5 * d, n)                 # 胜率（和棋算半胜）的 Wilson 95% CI
    out["wilson_decided"] = _wilson(w, w + l) if (w + l) else (float("nan"), float("nan"))
    out["margin_t"] = _mean_t_ci(margin)                     # 血量差的 t 95% CI
    return out


def _wilson(k: float, n: int, z: float = 1.959963985) -> tuple:
    """Wilson 二项比例 95% CI（`k` 允许是 W+0.5D 这种"有效胜场"）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _tq(n: int) -> float:
    """t_{0.975, n−1}；没有 scipy 就退化成正态 1.96。"""
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, n - 1))
    except Exception:                                          # noqa: BLE001
        return 1.959963985


def _mean_t_ci(xs: list[float]) -> tuple:
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    half = _tq(n) * sd / (n ** 0.5)
    return (m - half, m + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True, help="一个或多个臂的日志目录")
    ap.add_argument("--name", nargs="*", default=None, help="对应臂名（缺省用目录名）")
    a = ap.parse_args()
    names = a.name or [Path(d).name for d in a.dir]
    all_rows = {}
    print(f"{'臂':<18}{'n':>4}{'W':>4}{'D':>4}{'L':>4}{'胜率':>8}{'Wilson95':>16}"
          f"{'均值血差':>10}{'t95':>17}{'非法':>6}")
    for nm, d in zip(names, a.dir):
        rows = parse_dir(Path(d))
        all_rows[nm] = rows
        s = summarize(rows)
        if not s:
            print(f"{nm:<18}{'(空)':>4}")
            continue
        wi = s["wilson"]; mt = s["margin_t"]
        print(f"{nm:<18}{s['n']:>4}{s['W']:>4}{s['D']:>4}{s['L']:>4}"
              f"{s['winrate']*100:>7.1f}%"
              f"{f'[{wi[0]*100:.0f}%,{wi[1]*100:.0f}%]':>16}"
              f"{s['margin_mean']:>+10.2f}"
              f"{f'[{mt[0]:+.1f},{mt[1]:+.1f}]':>17}{s['illegal']:>6}")
    # 配对比较（每对臂 vs 第一个臂）
    if len(all_rows) >= 2:
        base_name, base_rows = list(all_rows.items())[0]
        for nb, rb in list(all_rows.items())[1:]:
            ka = {(r["seed"], r["seat"]): r for r in base_rows}
            kb = {(r["seed"], r["seat"]): r for r in rb}
            common = sorted(set(ka) & set(kb))
            if not common:
                continue
            dw = sum(1 for k in common if kb[k]["win"] and not ka[k]["win"])
            dl = sum(1 for k in common if ka[k]["win"] and not kb[k]["win"])
            dm = [kb[k]["our_hp"] - kb[k]["opp_hp"] - (ka[k]["our_hp"] - ka[k]["opp_hp"])
                  for k in common]
            ci = _mean_t_ci(dm)
            print(f"\n[配对] {nb} − {base_name}：{len(common)} 对（同 seed 同座位）"
                  f"｜{nb} 独赢 {dw} 对 / {base_name} 独赢 {dl} 对"
                  f"｜血差差 {sum(dm)/len(dm):+.2f} HP（t95 [{ci[0]:+.1f},{ci[1]:+.1f}]）")
    out = (Path(a.dir[0]).resolve().parent / "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: summarize(v) for k, v in all_rows.items()},
                             ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
