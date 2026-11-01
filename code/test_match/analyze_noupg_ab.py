"""unified_noupg（候选层删升级） vs pin+flat 的 16 局 A/B 分析（2026-11-02）。

主读数：
  1. A 臂（unified_noupg）胜率（Wilson 95% CI）；
  2. 血差 Δhp = A_hp − B_hp（按 seed 配对，t 95% CI）；
  3. 行为：A/B 两臂各自**真的发出**的 op 构成（闪电 op21 / 建塔 11 / 降级 13 / 升级 12 / HOLD）、
     "攒到 90"（决策前金币 ≥90）的决策占比 —— 回答"升级被删后，搜索选了啥替代、攒钱/闪电有没有变多"。

输入：
  · 对局日志 = `match_results/noupg_ab/seed<seed>_p<side>.log`（ab_search_modes.py 产出）；
  · stderr    = `match_results/ai?_az_bridge_ai_seed<seed>_noupg_p<side>.stderr.log`
    （`AB_SEARCH_SUFFIX_TAG=_noupg` + `MVS_STDERR_SUFFIX=1`）。

座位约定：side=0 ⇒ A(unified_noupg) 坐 P0；side=1 ⇒ 坐 P1。
用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_noupg_ab.py \
      --seeds 100-107 [--json match_results/noupg_ab/summary.json]
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "match_results" / "noupg_ab"
VALID_VERDICT = "engine"
STDERR_TAG = "_noupg"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _load_ladder():
    spec = importlib.util.spec_from_file_location("ladder_for_noupg", REPO / "_tmp_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def t_ci(xs: list[float], z: float = None) -> tuple[float, float, float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"),) * 4
    mean = st.mean(xs)
    if n == 1:
        return (mean, float("nan"), mean, mean)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n)
    if z is None:
        _T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
              30: 2.042, 60: 2.000, 120: 1.980}
        df = n - 1
        if df <= 0:
            tv = float("nan")
        elif df in _T:
            tv = _T[df]
        else:
            keys = sorted(_T)
            for a, b in zip(keys, keys[1:]):
                if a <= df <= b:
                    tv = _T[a] + (_T[b] - _T[a]) * (df - a) / (b - a)
                    break
            else:
                tv = _T[keys[-1]] if df > keys[-1] else _T[keys[0]]
        z = tv
    return (mean, se, mean - z * se, mean + z * se)


RE_ACT = re.compile(r"\[act\] round=(?P<r>\d+) n=(?P<n>\d+) ops=\[(?P<ops>.*?)\]")
RE_OP = re.compile(r"\((\d+), (-?\d+), (-?\d+)\)")
RE_ROUND = re.compile(r"\[azai\] round=(?P<r>\d+) hp=\[(?P<h0>-?\d+), (?P<h1>-?\d+)\] "
                      r"coins=\[(?P<c0>-?\d+), (?P<c1>-?\d+)\]")
OP_NAMES = {11: "建塔", 12: "升级", 13: "降级", 21: "闪电", 22: "EMP",
            23: "偏转", 24: "闪避", 31: "升速", 32: "升蚁"}


def _behavior(stderr_path: Path, seat: int) -> dict:
    out = dict(decisions=0, holds=0, ops=collections.Counter(), n90=0, n_coins=0)
    try:
        txt = stderr_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for ln in txt.splitlines():
        m = RE_ACT.search(ln)
        if m:
            out["decisions"] += 1
            ops = [(int(a), int(b), int(c)) for a, b, c in RE_OP.findall(m.group("ops"))]
            if not ops:
                out["holds"] += 1
            for (t, _a, _b) in ops:
                out["ops"][t] += 1
            continue
        m = RE_ROUND.search(ln)
        if m:
            coins = int(m.group("c0" if seat == 0 else "c1"))
            out["n_coins"] += 1
            if coins >= 90:
                out["n90"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-107")
    ap.add_argument("--out-dir", default=str(OUTDIR))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    ladder = _load_ladder()

    rows: list[dict] = []
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    for seed in seeds:
        for side in (0, 1):
            label = f"seed{seed}_p{side}"
            log_path = out_dir / f"{label}.log"
            if not log_path.is_file():
                missing.append(label)
                continue
            g = ladder.read_game(log_path, str(seed))
            verdict = g.get("verdict")
            if verdict != VALID_VERDICT:
                invalid.append((label, str(verdict)))
                continue
            hp = g.get("hp") or (0, 0)
            ws = g.get("winner_side")
            a_seat = side
            a_won = (ws == ("p0" if a_seat == 0 else "p1"))
            a_hp = hp[a_seat]
            b_hp = hp[1 - a_seat]
            rows.append(dict(seed=seed, side=side, label=label,
                             winner_side=ws, rounds=g.get("rounds"),
                             a_won=a_won, dhp=a_hp - b_hp,
                             a_hp=a_hp, b_hp=b_hp))

    n_valid = len(rows)
    wins = sum(1 for r in rows if r["a_won"])
    losses = sum(1 for r in rows if not r["a_won"])
    rate = wins / n_valid if n_valid else float("nan")
    w_lo, w_hi = wilson(wins, n_valid)

    by_seed: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    paired_dhp: list[float] = []
    for seed in seeds:
        gs = by_seed.get(seed, [])
        if len(gs) == 2:
            paired_dhp.append(st.mean([g["dhp"] for g in gs]))
    mean_dhp, se_dhp, dhp_lo, dhp_hi = t_ci(paired_dhp)

    by_seat = collections.defaultdict(list)
    for r in rows:
        by_seat[r["side"]].append(r["dhp"])

    print("=== unified_noupg(候选层删升级) vs pin+random：16 局 A/B（seeds "
          f"{seeds[0]}..{seeds[-1]} × 2 座位）===")
    print(f"有日志 {n_valid + len(invalid) + len(missing)} 有效 {n_valid} 无效 {len(invalid)} 缺 {len(missing)}")
    if missing:
        print(f"  缺日志: {missing}")
    if invalid:
        print(f"  无效（不计分）: {invalid}")
    print(f"\n--- 主读数 ①：A(unified_noupg) 胜率 ---")
    print(f"  {wins}W-{losses}L / {n_valid} 局 = {rate:.4f}  "
          f"Wilson 95% = [{w_lo:.4f}, {w_hi:.4f}]")
    print(f"\n--- 主读数 ②：血差 Δhp = A_hp − B_hp ---")
    print(f"  按 seed 配对（n={len(paired_dhp)} 对）: mean={mean_dhp:+.1f}  "
          f"t95=[{dhp_lo:+.1f}, {dhp_hi:+.1f}]  (se={se_dhp:.1f})")
    if rows:
        all_dhp = [r["dhp"] for r in rows]
        print(f"  参考（{len(all_dhp)} 局不配对）: mean={st.mean(all_dhp):+.1f}  "
              f"median={st.median(all_dhp):+.1f}")
    for seat in (0, 1):
        xs = by_seat.get(seat, [])
        if xs:
            print(f"  A 坐 P{seat}（n={len(xs)}）: Δhp mean={st.mean(xs):+.1f}")

    print("\n--- 逐局 ---")
    print("  %-12s %-4s %-14s %-7s %-6s %-6s %-6s" %
          ("token", "座位", "winner", "rounds", "A_hp", "B_hp", "Δhp"))
    for r in rows:
        print("  %-12s P%d  %-14s %-7s %-6d %-6d %+6d" %
              (r["label"], r["side"], ("A(uni_noupg)" if r["a_won"] else "B(pin)"),
               str(r["rounds"]), r["a_hp"], r["b_hp"], r["dhp"]))

    print("\n--- 主读数 ③：行为（实际发出的 op 构成 + 攒钱）---")
    agg = {"A": dict(decisions=0, holds=0, ops=collections.Counter(), n90=0, n_coins=0),
           "B": dict(decisions=0, holds=0, ops=collections.Counter(), n90=0, n_coins=0)}
    per_game: list[dict] = []
    for seed in seeds:
        for side in (0, 1):
            a_seat = side
            a_file = out_dir.parent / (f"ai{a_seat}_az_bridge_ai_seed{seed}_{STDERR_TAG}_p{side}.stderr.log")
            b_file = out_dir.parent / (f"ai{1 - a_seat}_az_bridge_ai_seed{seed}_{STDERR_TAG}_p{side}.stderr.log")
            ba = _behavior(a_file, a_seat)
            bb = _behavior(b_file, 1 - a_seat)
            for k, v in ba.items():
                if k == "ops":
                    agg["A"]["ops"].update(v)
                else:
                    agg["A"][k] += v
            for k, v in bb.items():
                if k == "ops":
                    agg["B"]["ops"].update(v)
                else:
                    agg["B"][k] += v
            per_game.append(dict(seed=seed, side=side,
                                 A_light=ba["ops"].get(21, 0), B_light=bb["ops"].get(21, 0),
                                 A_upg=ba["ops"].get(12, 0), B_upg=bb["ops"].get(12, 0),
                                 A_n90=ba["n90"], B_n90=bb["n90"],
                                 A_dec=ba["decisions"], B_dec=bb["decisions"]))

    def _prt(tag: str, a: dict, b: dict) -> None:
        a_dec, b_dec = a["decisions"], b["decisions"]
        print(f"  {tag}: 决策 A={a_dec} B={b_dec}")
        for t in (11, 12, 13, 21, 22, 23, 24, 31, 32):
            if a["ops"].get(t, 0) or b["ops"].get(t, 0):
                print(f"      {OP_NAMES.get(t, t)}({t}): A={a['ops'].get(t,0)}  B={b['ops'].get(t,0)}")
        print(f"    HOLD（空束）回合: A={a['holds']}  B={b['holds']}")
        if a["n_coins"] and b["n_coins"]:
            print(f"    金币≥90 的决策（攒到 90）: A={a['n90']}/{a['n_coins']} "
                  f"= {100.0*a['n90']/a['n_coins']:.1f}%   "
                  f"B={b['n90']}/{b['n_coins']} = {100.0*b['n90']/b['n_coins']:.1f}%")

    _prt("两臂合计", agg["A"], agg["B"])

    print("\n  逐局（A=unified_noupg，B=pin）：升级次数 A/B + 闪电次数 A/B + 攒90决策 A/B")
    print("  %-12s %-4s %-8s %-8s %-8s %-8s %-8s %-8s" %
          ("token", "座位", "A升级", "B升级", "A闪电", "B闪电", "A攒90", "B攒90"))
    for g in per_game:
        print("  %-12s P%d %-8d %-8d %-8d %-8d %-8d %-8d" %
              (f"seed{g['seed']}_p{g['side']}", g["side"], g["A_upg"], g["B_upg"],
               g["A_light"], g["B_light"], g["A_n90"], g["B_n90"]))

    out = dict(
        seeds=seeds, n_valid=n_valid, missing=missing, invalid=invalid,
        wins=wins, losses=losses, rate=rate, wilson=[w_lo, w_hi],
        paired_dhp=paired_dhp, dhp_mean=mean_dhp, dhp_t95=[dhp_lo, dhp_hi],
        dhp_by_seat={str(k): v for k, v in by_seat.items()},
        behavior_A=dict(decisions=agg["A"]["decisions"], holds=agg["A"]["holds"],
                        ops=dict(agg["A"]["ops"]), n90=agg["A"]["n90"], n_coins=agg["A"]["n_coins"]),
        behavior_B=dict(decisions=agg["B"]["decisions"], holds=agg["B"]["holds"],
                        ops=dict(agg["B"]["ops"]), n90=agg["B"]["n90"], n_coins=agg["B"]["n_coins"]),
        per_game=per_game,
    )
    if args.json:
        p = Path(args.json)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
