"""统一候选 A/B（`AZAI_CAND_MODE=unified` vs `pin`）的**配对分析**（2026-10-31）。

输入（都由 `_unified_cand_ab.sh` 产出）：
* `match_results/ladder_logs/<tag>/<token>.log`：`RESULT` 行（胜负/回合/血量/coins/illegal）
  + `match_results/ladder_logs/<tag>/_run.log`（每局墙钟）
* `match_results/ai{0,1}_az_bridge_ai_seed<N><SUF>.stderr.log`：
  * `[cand] mode=unified|pin ...`（每决策：n_cand / 含HOLD / 含升级 / 选中）
  * `[azai] [act] round=.. n=.. ops=[(t,a0,a1), ...]`（每回合**真的发出去的** op）

输出：
1. 配对胜负/血量：胜率 Wilson CI、配对 Δhp 的 t-CI、McNemar 精确 p、Δhp 符号检验
2. 按座位拆分（我方 P0 / P1）
3. 行为差异：HOLD 率、降级 churn 占比、升级次数、`n_cand` 分布
4. 健康度：verdict=engine / illegal=0 的局数
5. 每局墙钟

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/analyze_unified_cand_ab.py \
    --tag-a UNIFIED_AB_A --tag-b UNIFIED_AB_B --suf-a _UNA --suf-b _UNB \
    --out match_results/unified_cand/ab_report.txt --json match_results/unified_cand/ab.json
```
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LADDER = REPO / "match_results" / "ladder_logs"
MR = REPO / "match_results"

RE_RESULT = re.compile(
    r"RESULT seed=(?P<seed>\d+) .*?winner=(?P<winner>\S+) winner_side=(?P<ws>\S+) "
    r"verdict=(?P<verdict>\S+).*?rounds=(?P<rounds>\d+) terminal=(?P<term>\S+) "
    r"base_hp=(?P<hp0>-?\d+),(?P<hp1>-?\d+) coins=(?P<c0>-?\d+),(?P<c1>-?\d+)"
    r"(?: illegal=(?P<illegal>\d+))?")
RE_ACT = re.compile(r"\[act\] round=(?P<r>\d+) n=(?P<n>\d+) ops=\[(?P<ops>.*?)\]")
RE_OP = re.compile(r"\((\d+), (-?\d+), (-?\d+)\)")
RE_WALL = re.compile(r"^\[(?P<tok>[0-9]+r?)\]\s+rc=(?P<rc>-?\d+)\s+(?P<min>[\d.]+) min")
RE_CAND = re.compile(r"\[cand\] mode=(?P<m>unified|pin) n_cand=(?P<n>-?\d+) "
                     r"含HOLD=(?P<h>\d+) 含升级=(?P<u>\d+)")
RE_HIT = re.compile(r"选中=(?P<p>\S+)")


# ─── 统计工具（无 scipy 也跑得动）────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz 连分数（不完全 Beta 的数值实现，免依赖 scipy）。"""
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: int) -> float:
    """P(T > |t|)，双侧。"""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)


def t_ppf(p: float, df: int) -> float:
    """t 分布分位数（二分法，够用）。"""
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(T <= mid) = 1 - t_sf(mid)/2
        cdf = 1.0 - t_sf(mid, df) / 2.0
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def binom_two_sided(k: int, n: int) -> float:
    """McNemar 精确检验（discordant 里 k 个偏向一边的两侧二项 p）。"""
    if n <= 0:
        return float("nan")
    k = min(k, n - k)
    s = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * s)


def sign_test(deltas: list) -> tuple[int, int, int, float]:
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    tie = len(deltas) - pos - neg
    return pos, neg, tie, binom_two_sided(pos, pos + neg)


def mean_ci(xs: list) -> tuple[float, float, float, int]:
    n = len(xs)
    if n == 0:
        return (float("nan"),) * 3 + (0,)
    m = sum(xs) / n
    if n == 1:
        return m, float("nan"), float("nan"), 1
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return m, m, m, n
    tq = t_ppf(0.975, n - 1)
    return m, m - tq * se, m + tq * se, n


# ─── 读数据 ────────────────────────────────────────────────────────────────
def _side_of(token: str) -> int:
    """token 不带 r ⇒ 我们执 P0；带 r ⇒ 我们执 P1（AI1 先手）。"""
    return 1 if token.endswith("r") else 0


def _stderr_path(seed: int, token: str, suf: str) -> Path:
    seat = "ai1" if token.endswith("r") else "ai0"
    return MR / f"{seat}_az_bridge_ai_seed{seed}{suf}.stderr.log"


def read_arm(tag: str, suf: str, tokens: list) -> dict:
    logdir = LADDER / tag
    walls: dict = {}
    runlog = logdir / "_run.log"
    if runlog.is_file():
        for line in runlog.read_text(encoding="utf-8", errors="replace").splitlines():
            m = RE_WALL.match(line.strip())
            if m:
                walls[m.group("tok")] = float(m.group("min"))
    out = {}
    for tok in tokens:
        seed = int(tok.rstrip("r"))
        side = _side_of(tok)
        rec = {"token": tok, "seed": seed, "side": side, "verdict": None, "win": None,
               "hp_ours": None, "hp_opp": None, "rounds": None, "illegal": None,
               "wall_min": walls.get(tok), "has_log": False,
               "decisions": 0, "holds": 0, "ops": {}, "upg_rounds": 0, "same_type_rounds": 0,
               "n_cand": [], "cand_has_hold": 0, "cand_has_upg": 0, "pick_hold": 0,
               "pick_upg": 0, "pick_other": 0, "mode": None}
        f = logdir / f"{tok}.log"
        if f.is_file():
            txt = f.read_text(encoding="utf-8", errors="replace")
            m = RE_RESULT.search(txt)
            if m:
                rec["has_log"] = True
                rec["verdict"] = m.group("verdict")
                rec["rounds"] = int(m.group("rounds"))
                rec["illegal"] = int(m.group("illegal") or 0)
                hp0, hp1 = int(m.group("hp0")), int(m.group("hp1"))
                rec["hp_ours"] = hp0 if side == 0 else hp1
                rec["hp_opp"] = hp1 if side == 0 else hp0
                ws = m.group("ws")
                rec["win"] = int(ws == f"p{side}")
                rec["draw"] = int(ws == "draw")
        sf = _stderr_path(seed, tok, suf)
        if sf.is_file():
            st = sf.read_text(encoding="utf-8", errors="replace")
            for line in st.splitlines():
                ma = RE_ACT.search(line)
                if ma:
                    n = int(ma.group("n"))
                    ops = [(int(a), int(b), int(c)) for a, b, c in RE_OP.findall(ma.group("ops"))]
                    rec["decisions"] += 1
                    if n == 0:
                        rec["holds"] += 1
                    types = [t for (t, _a, _b) in ops]
                    for t in types:
                        rec["ops"][t] = rec["ops"].get(t, 0) + 1
                    if 12 in types:
                        rec["upg_rounds"] += 1
                    if len(types) >= 2 and len(set(types)) < len(types):
                        rec["same_type_rounds"] += 1
                    continue
                mc = RE_CAND.search(line)
                if mc:
                    rec["mode"] = mc.group("m")
                    rec["n_cand"].append(int(mc.group("n")))
                    rec["cand_has_hold"] += int(mc.group("h") == "1")
                    rec["cand_has_upg"] += int(int(mc.group("u")) > 0)
                    mh = RE_HIT.search(line)
                    if mh:
                        p = mh.group("p")
                        if "HOLD" in p:
                            rec["pick_hold"] += 1
                        elif "升级" in p:
                            rec["pick_upg"] += 1
                        else:
                            rec["pick_other"] += 1
        out[tok] = rec
    return out


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{100.0*x:5.1f}%"


def _f(x: float, nd: int = 2) -> str:
    return "n/a" if x != x else f"{x:+.{nd}f}"


def _arm_row(recs: list) -> dict:
    valid = [r for r in recs if r["verdict"] == "engine"]
    wins = sum(1 for r in valid if r["win"])
    p, lo, hi = wilson(wins, len(valid))
    decisions = sum(r["decisions"] for r in recs)
    holds = sum(r["holds"] for r in recs)
    ops: dict = {}
    for r in recs:
        for t, c in r["ops"].items():
            ops[t] = ops.get(t, 0) + c
    build = ops.get(11, 0)
    down = ops.get(13, 0)
    upg = ops.get(12, 0)
    n_cand = [n for r in recs for n in r["n_cand"]]
    walls = [r["wall_min"] for r in recs if r["wall_min"]]
    return dict(n=len(recs), valid=len(valid), wins=wins, wr=p, wr_lo=lo, wr_hi=hi,
                decisions=decisions, holds=holds,
                hold_rate=(holds / decisions if decisions else float("nan")),
                ops=ops, build=build, down=down, upg=upg,
                churn=(down / (build + down) if (build + down) else float("nan")),
                upg_rounds=sum(r["upg_rounds"] for r in recs),
                same_type_rounds=sum(r["same_type_rounds"] for r in recs),
                n_cand=n_cand,
                n_cand_median=(sorted(n_cand)[len(n_cand) // 2] if n_cand else float("nan")),
                n_cand_max=(max(n_cand) if n_cand else float("nan")),
                n_cand_single=(sum(1 for n in n_cand if n == 1) / len(n_cand) if n_cand else float("nan")),
                n_cand_ge24=(sum(1 for n in n_cand if n >= 24) / len(n_cand) if n_cand else float("nan")),
                cand_has_hold=sum(r["cand_has_hold"] for r in recs),
                cand_has_upg=sum(r["cand_has_upg"] for r in recs),
                pick_hold=sum(r["pick_hold"] for r in recs),
                pick_upg=sum(r["pick_upg"] for r in recs),
                pick_other=sum(r["pick_other"] for r in recs),
                wall=walls, illegal=sum(r["illegal"] or 0 for r in recs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-a", default="UNIFIED_AB_A")
    ap.add_argument("--tag-b", default="UNIFIED_AB_B")
    ap.add_argument("--suf-a", default="_UNA")
    ap.add_argument("--suf-b", default="_UNB")
    ap.add_argument("--seeds", default="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r "
                                    "14 14r 15 15r 16 16r 17 17r 18 18r")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    tokens = args.seeds.split()

    A = read_arm(args.tag_a, args.suf_a, tokens)
    B = read_arm(args.tag_b, args.suf_b, tokens)
    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    ra, rb = _arm_row(list(A.values())), _arm_row(list(B.values()))
    emit("=" * 78)
    emit("统一候选 A/B —— `AZAI_CAND_MODE=unified`(A) vs `pin`(B)")
    emit("=" * 78)
    emit("")
    emit("## 1. 逐臂健康度 / 总体")
    emit(f"  {'':22} {'A(unified)':>16} {'B(pin)':>16}")
    emit(f"  {'局数（有 RESULT）':22} {ra['valid']:>16} {rb['valid']:>16}")
    emit(f"  {'verdict=engine':22} {ra['valid']:>16} {rb['valid']:>16}")
    emit(f"  {'illegal 合计':22} {ra['illegal']:>16} {rb['illegal']:>16}")
    emit(f"  {'胜场':22} {ra['wins']:>16} {rb['wins']:>16}")
    emit(f"  {'胜率 [95% Wilson]':22} {_pct(ra['wr'])+' ['+_pct(ra['wr_lo']).strip()+','+_pct(ra['wr_hi']).strip()+']':>16} "
         f"{_pct(rb['wr'])+' ['+_pct(rb['wr_lo']).strip()+','+_pct(rb['wr_hi']).strip()+']':>16}")
    if ra["wall"] and rb["wall"]:
        emit(f"  {'每局墙钟 min 中位':22} {sorted(ra['wall'])[len(ra['wall'])//2]:>16.1f} "
             f"{sorted(rb['wall'])[len(rb['wall'])//2]:>16.1f}")
        emit(f"  {'每局墙钟 min 合计':22} {sum(ra['wall']):>16.1f} {sum(rb['wall']):>16.1f}")

    # ── 配对 ────────────────────────────────────────────────────────────────
    common = [t for t in tokens if A[t]["verdict"] == "engine" and B[t]["verdict"] == "engine"]
    onlyA = [t for t in tokens if A[t]["verdict"] == "engine" and B[t]["verdict"] != "engine"]
    onlyB = [t for t in tokens if B[t]["verdict"] == "engine" and A[t]["verdict"] != "engine"]
    emit("")
    emit(f"## 2. 配对（两臂都 engine 的 token = {len(common)}；只 A 有效 {len(onlyA)}；只 B 有效 {len(onlyB)}）")
    wa = sum(A[t]["win"] for t in common)
    wb = sum(B[t]["win"] for t in common)
    both = sum(1 for t in common if A[t]["win"] and B[t]["win"])
    nei = sum(1 for t in common if not A[t]["win"] and not B[t]["win"])
    a_only = sum(1 for t in common if A[t]["win"] and not B[t]["win"])
    b_only = sum(1 for t in common if B[t]["win"] and not A[t]["win"])
    mc_p = binom_two_sided(a_only, a_only + b_only)
    emit(f"  A 胜 {wa} / B 胜 {wb}（配对子集 n={len(common)}）")
    emit(f"  2x2: 都赢 {both} / 只 A 赢 {a_only} / 只 B 赢 {b_only} / 都输 {nei}")
    emit(f"  McNemar 精确双侧 p = {mc_p:.4f}（discordant={a_only + b_only}）")
    dh = [A[t]["hp_ours"] - A[t]["hp_opp"] - (B[t]["hp_ours"] - B[t]["hp_opp"]) for t in common]
    if dh:
        m, lo, hi, n = mean_ci(dh)
        emit(f"  Δhp（A−B，逐 token）均值 {_f(m)}  95% CI [{_f(lo)}, {_f(hi)}]  n={n}")
        pos, neg, tie, sp = sign_test(dh)
        emit(f"  Δhp 符号检验：A>B {pos} / A<B {neg} / 平 {tie}   p = {sp:.4f}")
    # 座位拆分
    for side, name in ((0, "我方 P0"), (1, "我方 P1")):
        toks = [t for t in common if A[t]["side"] == side]
        if not toks:
            continue
        a_w = sum(A[t]["win"] for t in toks)
        b_w = sum(B[t]["win"] for t in toks)
        d = [A[t]["hp_ours"] - A[t]["hp_opp"] - (B[t]["hp_ours"] - B[t]["hp_opp"]) for t in toks]
        m, lo, hi, n = mean_ci(d)
        emit(f"  [{name}] n={len(toks)}  A 胜 {a_w} / B 胜 {b_w}；"
             f"Δhp 均值 {_f(m)} CI [{_f(lo)}, {_f(hi)}]")

    # ── 行为差异 ────────────────────────────────────────────────────────────
    emit("")
    emit("## 3. 行为差异（全部决策/回合汇总）")
    emit(f"  {'':30} {'A(unified)':>14} {'B(pin)':>14}")
    emit(f"  {'决策数':30} {ra['decisions']:>14} {rb['decisions']:>14}")
    emit(f"  {'HOLD 率（选中空束/决策）':30} {_pct(ra['hold_rate']):>14} {_pct(rb['hold_rate']):>14}")
    emit(f"  {'BUILD 次数':30} {ra['build']:>14} {rb['build']:>14}")
    emit(f"  {'DOWNGRADE 次数':30} {ra['down']:>14} {rb['down']:>14}")
    emit(f"  {'UPGRADE 次数':30} {ra['upg']:>14} {rb['upg']:>14}")
    emit(f"  {'降级 churn = D/(B+D)':30} {_pct(ra['churn']):>14} {_pct(rb['churn']):>14}")
    emit(f"  {'升级回合数（真的发出）':30} {ra['upg_rounds']:>14} {rb['upg_rounds']:>14}")
    emit(f"  {'同类型多 op 回合数':30} {ra['same_type_rounds']:>14} {rb['same_type_rounds']:>14}")
    _oa = str({k: v for k, v in sorted(ra["ops"].items()) if k not in (11, 12, 13)})
    _ob = str({k: v for k, v in sorted(rb["ops"].items()) if k not in (11, 12, 13)})
    emit(f"  {'其它 op（闪电/超武/基地）':30} {_oa:>14} {_ob:>14}")
    emit("")
    emit("  n_cand（每决策，去重后候选数）：")
    emit(f"  {'':30} {'A(unified)':>14} {'B(pin)':>14}")
    emit(f"  {'决策条数':30} {len(ra['n_cand']):>14} {len(rb['n_cand']):>14}")
    emit(f"  {'中位':30} {ra['n_cand_median']:>14.0f} {rb['n_cand_median']:>14.0f}")
    emit(f"  {'max':30} {ra['n_cand_max']:>14.0f} {rb['n_cand_max']:>14.0f}")
    emit(f"  {'n_cand==1 占比':30} {_pct(ra['n_cand_single']):>14} {_pct(rb['n_cand_single']):>14}")
    emit(f"  {'n_cand>=24 占比':30} {_pct(ra['n_cand_ge24']):>14} {_pct(rb['n_cand_ge24']):>14}")
    emit(f"  {'候选含 HOLD 的决策占比':30} {_pct(ra['cand_has_hold']/max(1,len(ra['n_cand']))):>14} "
         f"{_pct(rb['cand_has_hold']/max(1,len(rb['n_cand']))):>14}")
    emit(f"  {'候选含升级 的决策占比':30} {_pct(ra['cand_has_upg']/max(1,len(ra['n_cand']))):>14} "
         f"{_pct(rb['cand_has_upg']/max(1,len(rb['n_cand']))):>14}")
    _pa = f"{ra['pick_hold']}/{ra['pick_upg']}/{ra['pick_other']}"
    _pb = f"{rb['pick_hold']}/{rb['pick_upg']}/{rb['pick_other']}"
    emit(f"  {'选中 HOLD / 升级 / 其它':30} {_pa:>14} {_pb:>14}")

    # ── 逐 token 明细 ──────────────────────────────────────────────────────
    emit("")
    emit("## 4. 逐 token 明细（hp = 我方基地血量；Δhp = A−B）")
    emit(f"  {'token':6} {'seat':4} {'A verdict':10} {'A hp':>5} {'A 回合':>6} | "
         f"{'B verdict':10} {'B hp':>5} {'B 回合':>6} | {'Δhp':>5} {'A胜':>4} {'B胜':>4} {'墙钟A/B min':>12}")
    for t in tokens:
        a, b = A[t], B[t]
        d = (a["hp_ours"] - b["hp_ours"]) if (a["hp_ours"] is not None and b["hp_ours"] is not None) else None
        emit(f"  {t:6} P{a['side']:<3} {str(a['verdict']):10} {str(a['hp_ours']):>5} {str(a['rounds']):>6} | "
             f"{str(b['verdict']):10} {str(b['hp_ours']):>5} {str(b['rounds']):>6} | "
             f"{('n/a' if d is None else f'{d:+d}'):>5} "
             f"{('Y' if a['win'] else ('n' if a['win'] is not None else '-')):>4} "
             f"{('Y' if b['win'] else ('n' if b['win'] is not None else '-')):>4} "
             f"{str(a['wall_min'])}/{str(b['wall_min'])}")

    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[ab] 已写入 {p}")
    if args.json:
        p = Path(args.json)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"A": ra, "B": rb, "common": common,
                                 "mcnemar_p": mc_p, "dhp": dh,
                                 "per_token": {"A": A, "B": B}},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[ab] 已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
