"""叶子金币项 α 扫描分析：unified(+AZAI_LEAF_COIN_ALPHA=α) vs pin+flat。

主读数（每个 α 各一份）：
  1. A 臂（unified+α）胜率（Wilson 95% CI）；
  2. 血差 Δhp = A_hp − B_hp（按 seed 配对，t 95% CI）；
  3. 行为：A/B 两臂各自**真的发出**的 op 构成（闪电 op21 / 建塔 11 / 降级 13 / 升级 12 / HOLD）、
     "攒到 90"（决策前金币 ≥90）的决策占比 —— 回答"加大 α 是否让搜索更爱攒钱/放闪电"。

输入：
  · 对局日志 = `<out-root>/a<α>/seed<seed>_p<side>.log`（ab_search_modes.py 产出）；
  · stderr    = `match_results/ai?_az_bridge_ai_seed<seed>_lca<α>_p<side>.stderr.log`
    （`AB_SEARCH_SUFFIX_TAG=_lca<α>` + `MVS_STDERR_SUFFIX=1`）。

座位约定：side=0 ⇒ A(unified+α) 坐 P0；side=1 ⇒ 坐 P1。
用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_leaf_coin_alpha.py \
      --alphas 0,0.02,0.037,0.07,0.15,0.30 --seeds 100-101 \
      [--out-root match_results/leaf_coin_alpha] [--json match_results/leaf_coin_alpha/summary.json]
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
VALID_VERDICT = "engine"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _load_ladder():
    spec = importlib.util.spec_from_file_location("ladder_for_lca", REPO / "_tmp_ladder.py")
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


def _parse_alphas(spec: str) -> list[str]:
    return [a.strip() for a in spec.split(",") if a.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def t_ci(xs: list[float]) -> tuple[float, float, float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"),) * 4
    mean = st.mean(xs)
    if n == 1:
        return (mean, float("nan"), mean, mean)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n)
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
    return (mean, se, mean - tv * se, mean + tv * se)


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


def analyze_alpha(alpha: str, seeds: list[int], out_root: Path, ladder) -> dict:
    a_dir = out_root / f"a{alpha}"
    tag = f"lca{alpha}"
    rows: list[dict] = []
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    for seed in seeds:
        for side in (0, 1):
            label = f"seed{seed}_p{side}"
            log_path = a_dir / f"{label}.log"
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

    # 行为
    agg = {"A": dict(decisions=0, holds=0, ops=collections.Counter(), n90=0, n_coins=0),
           "B": dict(decisions=0, holds=0, ops=collections.Counter(), n90=0, n_coins=0)}
    for seed in seeds:
        for side in (0, 1):
            a_seat = side
            a_file = out_root.parent / (f"ai{a_seat}_az_bridge_ai_seed{seed}_{tag}_p{side}.stderr.log")
            b_file = out_root.parent / (f"ai{1 - a_seat}_az_bridge_ai_seed{seed}_{tag}_p{side}.stderr.log")
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

    def frac(a: dict, key: str) -> float:
        den = a["decisions"] if key == "holds" else a["n_coins"]
        return (a[key] / den) if den else float("nan")

    return dict(
        alpha=alpha, n=n_valid, wins=wins, losses=losses, rate=rate,
        wilson=[w_lo, w_hi], paired_dhp=paired_dhp, dhp_mean=mean_dhp,
        dhp_t95=[dhp_lo, dhp_hi], dhp_se=se_dhp, missing=missing, invalid=invalid,
        rows=rows,
        A_light=agg["A"]["ops"].get(21, 0), B_light=agg["B"]["ops"].get(21, 0),
        A_build=agg["A"]["ops"].get(11, 0), B_build=agg["B"]["ops"].get(11, 0),
        A_dn=agg["A"]["ops"].get(13, 0), B_dn=agg["B"]["ops"].get(13, 0),
        A_upg=agg["A"]["ops"].get(12, 0), B_upg=agg["B"]["ops"].get(12, 0),
        A_dec=agg["A"]["decisions"], B_dec=agg["B"]["decisions"],
        A_holds=agg["A"]["holds"], B_holds=agg["B"]["holds"],
        A_hold_frac=frac(agg["A"], "holds"), B_hold_frac=frac(agg["B"], "holds"),
        A_n90=agg["A"]["n90"], A_n_coins=agg["A"]["n_coins"],
        B_n90=agg["B"]["n90"], B_n_coins=agg["B"]["n_coins"],
        A_n90_frac=frac(agg["A"], "n90"), B_n90_frac=frac(agg["B"], "n90"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0,0.02,0.037,0.07,0.15,0.30")
    ap.add_argument("--seeds", default="100-103")
    ap.add_argument("--out-root", default=str(REPO / "match_results" / "leaf_coin_alpha"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    alphas = _parse_alphas(args.alphas)
    seeds = _parse_seeds(args.seeds)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    ladder = _load_ladder()

    results = [analyze_alpha(a, seeds, out_root, ladder) for a in alphas]

    print("=== 叶子金币项 α 扫描：unified(+AZAI_LEAF_COIN_ALPHA=α) vs pin+flat "
          f"（seeds {seeds[0]}..{seeds[-1]} × 2 座位）===")
    print(f"α ∈ {alphas}   有效局/α 以日志 verdict=engine 为准\n")

    # ── 主表 ──
    hdr = (f"  {'α':>7} {'n':>3} {'胜率':>8} {'W-L':>7} {'Wilson95':>18} "
           f"{'Δhp配对':>9} {'t95':>15} | "
           f"{'攒90% A':>8} {'闪电 A':>7} {'HOLD% A':>8} {'建塔 A':>7} {'降级 A':>7} | "
           f"{'攒90% B':>8} {'闪电 B':>7} {'HOLD% B':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        a = r["alpha"]
        if r["n"] == 0:
            print(f"  {a:>7} {r['n']:>3}  (无有效局) missing={r['missing']} invalid={r['invalid']}")
            continue
        wl = f"{r['wins']}W-{r['losses']}L"
        dhp = f"{r['dhp_mean']:+.1f}" if r["paired_dhp"] else "  n/a"
        t95 = f"[{r['dhp_t95'][2]:+.1f},{r['dhp_t95'][3]:+.1f}]" if len(r["paired_dhp"]) > 1 else "  n/a"
        print(f"  {a:>7} {r['n']:>3} {r['rate']:>7.1%} {wl:>7} "
              f"[{r['wilson'][0]:.3f},{r['wilson'][1]:.3f}] "
              f"{dhp:>9} {t95:>15} | "
              f"{r['A_n90_frac']:>7.1%} {r['A_light']:>7} {r['A_hold_frac']:>7.1%} "
              f"{r['A_build']:>7} {r['A_dn']:>7} | "
              f"{r['B_n90_frac']:>7.1%} {r['B_light']:>7} {r['B_hold_frac']:>7.1%}")

    # ── 逐局 Δhp（每个 α）──
    for r in results:
        if not r["rows"]:
            continue
        print(f"\n--- α={r['alpha']} 逐局 Δhp（A−B）---")
        dhps = [x["dhp"] for x in r["rows"]]
        print("  " + "  ".join(f"{x['label']}:{x['dhp']:+d}" for x in r["rows"]))
        print(f"  不配对 mean={st.mean(dhps):+.1f} median={st.median(dhps):+.1f} "
              f"min={min(dhps):+d} max={max(dhps):+d}")

    out = dict(seeds=seeds, alphas=alphas, results=results)
    if args.json:
        p = Path(args.json)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        # 去掉 rows（逐局详情太重），换成紧凑字段
        slim = [dict(kv for kv in r.items() if kv[0] != "rows") for r in results]
        out["results"] = slim
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
