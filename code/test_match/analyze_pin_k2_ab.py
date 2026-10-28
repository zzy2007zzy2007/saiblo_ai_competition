#!/usr/bin/env python
"""`AZAI_PIN_K=2` A/B 的读数汇总（报告 `docs/pin_k2_ab_20261028.md`）。

口径**全部来自报告 §0 的预注册**（跑之前写定），本脚本只负责把它们算出来：

* **主判据** = **逐 token 配对**的 `ΔHP = Δhp_E − Δhp_B`（不是两批均值相减），
  报点估计、95%CI（t 分布，df=n−1）、双侧符号检验 p、sd、`+`/`−`/`0` 计数；
* **副判据** = 胜率（**Wilson** 95%CI，不用正态近似）+ 按座位拆分（`@P0` n=8 / `@P1` n=8）；
* **可行性** = 每局 stderr 里 `⏱️ 决策耗时 p50/p95/max` 与 `dec_ge10/20/30` 的
  **每局读数 → 对局取中位**，分母 = 全部局 `dec_n` 之和；两臂**同一口径**重算；
  另按预注册的两条红线判"可接受 / 边缘 / 不可接受"；
* **健康度** = `verdict=engine` / 裁判侧 `illegal=0` / 我方 `illegal_total=0` / `terminal=True`。

`Δhp` 一律从 `RESULT` 行的 `base_hp=(p0, p1)` **按我方座位**取（不靠标签比较）。

⚠️ 只读：不跑对局、不改任何被读的文件（`--json` 除外，那是我自己写的产物）。

用法：
  $PY -u code/test_match/analyze_pin_k2_ab.py \\
      --arm-e match_results/ladder_logs/PIN_K2_VS_RU \\
      --arm-b match_results/ladder_logs/INJECT_VS_RU_B \\
      --stderr-e match_results/pin_k2_vs_ru/PIN_K2_VS_RU/stderr \\
      --stderr-b match_results/inject_vs_ru/B/stderr \\
      --json match_results/pin_k2_vs_ru/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUR_LABEL = "az_bridge_ai"
SEEDS = list(range(7, 15))
TOKENS = [f"{s}{'r' if mirror else ''}" for s in SEEDS for mirror in (False, True)]

RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) (?:winner_side=(\S+) )?verdict=(\S+) "
    r"engine_winner=(\S+) rounds=(\S+) terminal=(\S+) base_hp=(\d+),(\d+) "
    r"coins=(\d+),(\d+)(?: illegal=(\d+))?")
# 每 25 决策一行：`⏱️ 决策耗时 p50=0.56s p95=19.62s max=26.75s n=500`
RE_PCT = re.compile(r"决策耗时 p50=([\d.]+)s p95=([\d.]+)s max=([\d.]+)s n=(\d+)")
# 每 25 决策一行：`⏱️ 耗时累计: dec_n=500 dec_max=26.75s dec_slow=21 dec_ge10=51(10.2%) …`
RE_CUM = re.compile(r"耗时累计: dec_n=(\d+) dec_max=([\d.]+)s dec_slow=(\d+)")
RE_GE = {th: re.compile(rf"dec_ge{th}=(\d+)\(([\d.]+)%\)") for th in (10, 20, 30)}
RE_ILLSUM = re.compile(r"illegal_total=(\d+)")
# 冒烟用：`[flat] 候选 48 个 × 64 次 × 16 回合`
RE_NCAND = re.compile(r"\[flat\] 候选 (\d+) 个")


# ── 统计小工具（不硬依赖 scipy，本项目只用 df≤30）───────────────────────────────
_T_TAB = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
          9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
          16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def tcrit(df: int, alpha: float = 0.05) -> float:
    """双侧 95% 的 t 临界值（优先 scipy；缺了查表，df>30 用 1.96）。"""
    try:
        from scipy import stats as _st          # type: ignore
        return float(_st.t.ppf(1 - alpha / 2, df))
    except Exception:
        if df in _T_TAB:
            return _T_TAB[df]
        ks = [k for k in _T_TAB if k <= df]
        return _T_TAB[max(ks)] if ks else 12.706


def sign_test(w: int, l: int) -> float:
    """双侧符号检验（**忽略 0 差**）：2·P(X ≥ max(w,l))，X ~ Bin(w+l, 0.5)。"""
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """胜率的 Wilson 95%CI（n 很小，正态近似不可用）。返回**比例** (lo, hi)。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mean_ci(xs: list[float]) -> tuple[float, float, float, float]:
    """返回 (mean, lo, hi, sd)。n<2 时 CI 退化成 (mean, mean)。"""
    n = len(xs)
    if n == 0:
        return (float("nan"),) * 4
    m = statistics.fmean(xs)
    if n < 2:
        return (m, m, m, 0.0)
    sd = statistics.stdev(xs)
    half = tcrit(n - 1) * sd / math.sqrt(n)
    return (m, m - half, m + half, sd)


# ── 解析 ───────────────────────────────────────────────────────────────────────
def read_arm(log_dir: Path) -> dict[str, dict]:
    """读一个 tag 目录的 16 个 token 日志 → {token: 读数}。缺文件/RESULT 行的记 `missing`。"""
    out: dict[str, dict] = {}
    for tok in TOKENS:
        f = log_dir / f"{tok}.log"
        rec: dict = {"token": tok, "exists": f.is_file(), "missing": None}
        if not f.is_file():
            rec["missing"] = "没有日志"
            out[tok] = rec
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = None
        for m in RESULT_RE.finditer(txt):     # 取**最后一条**（重试时前面的不算）
            pass
        if m is None:
            rec["missing"] = "没有 RESULT 行"
            out[tok] = rec
            continue
        seat = 0 if m.group(2) == OUR_LABEL else (1 if m.group(3) == OUR_LABEL else None)
        rec.update({"p0": m.group(2), "p1": m.group(3), "winner": m.group(4),
                    "winner_side": m.group(5), "verdict": m.group(6),
                    "rounds": int(m.group(8)), "terminal": m.group(9),
                    "illegal_judge": int(m.group(14) or 0), "seat": seat})
        if seat is None:
            rec["missing"] = f"两侧都不是 {OUR_LABEL}"
            out[tok] = rec
            continue
        hp0, hp1 = int(m.group(10)), int(m.group(11))
        rec["hp_ours"], rec["hp_opp"] = (hp0, hp1) if seat == 0 else (hp1, hp0)
        rec["d_hp"] = rec["hp_ours"] - rec["hp_opp"]
        # 胜负：**以 winner_side 为准**（官方级联的输出），没有就按 HP 兜底（旧日志）
        ws = m.group(5)
        if ws is None and rec["verdict"] != "aborted":
            ws = "p0" if hp0 > hp1 else ("p1" if hp1 > hp0 else None)
            rec["winner_side_src"] = "hp~"
        rec["score"] = (1.0 if ws == f"p{seat}" else 0.0) if ws else None
        out[tok] = rec
    return out


def read_latency(stderr_dir: Path) -> dict:
    """读一臂全部局的**最终**决策耗时读数；两臂必须用同一个函数（§0.5 的口径要求）。"""
    games: list[dict] = []
    if not stderr_dir.is_dir():
        return {"games": games, "dir": str(stderr_dir), "found": 0}
    for seat in (0, 1):
        for s in SEEDS:
            f = stderr_dir / f"ai{seat}_{OUR_LABEL}_seed{s}.stderr.log"
            if not f.is_file():
                continue
            txt = f.read_text(encoding="utf-8", errors="replace")
            g: dict = {"file": f.name, "seed": s, "seat": seat}
            pct = RE_PCT.findall(txt)
            if pct:
                p50, p95, mx, n = pct[-1]
                g.update({"p50": float(p50), "p95": float(p95), "max": float(mx),
                          "n_pct": int(n)})
            cum = RE_CUM.findall(txt)
            if cum:
                g.update({"dec_n": int(cum[-1][0]), "dec_max": float(cum[-1][1]),
                          "dec_slow": int(cum[-1][2])})
            for th in (10, 20, 30):
                mm = RE_GE[th].findall(txt)
                if mm:
                    g[f"ge{th}"] = int(mm[-1][0])
                    g[f"ge{th}_pct"] = float(mm[-1][1])
            ill = RE_ILLSUM.findall(txt)
            g["our_illegal_total"] = max((int(x) for x in ill), default=None)
            nc = [int(x) for x in RE_NCAND.findall(txt)]
            if nc:
                g["n_cand_median"] = statistics.median(nc)
                g["n_cand_mean"] = round(statistics.fmean(nc), 2)
                g["n_cand_eq1_pct"] = round(100.0 * sum(1 for x in nc if x == 1) / len(nc), 1)
                g["n_cand_n"] = len(nc)
            if "p50" in g or "dec_n" in g:
                games.append(g)
    return {"games": games, "dir": str(stderr_dir), "found": len(games)}


def lat_summary(lat: dict) -> dict:
    """一臂的汇总：p50/p95 取**局的**中位，max 取全局最大，dec_ge* 求和 / 决策总数。"""
    gs = lat["games"]
    if not gs:
        return {}
    n_dec = sum(int(g.get("dec_n") or 0) for g in gs)
    out = {"n_games": len(gs), "n_dec": n_dec,
           "p50": statistics.median([g["p50"] for g in gs if "p50" in g]),
           "p95": statistics.median([g["p95"] for g in gs if "p95" in g]),
           "max": max(g["max"] for g in gs if "max" in g),
           "p50_worst": max(g["p50"] for g in gs if "p50" in g),
           "p95_worst": max(g["p95"] for g in gs if "p95" in g)}
    for th in (10, 20, 30):
        tot = sum(int(g.get(f"ge{th}") or 0) for g in gs)
        out[f"ge{th}"] = tot
        out[f"ge{th}_pct"] = 100.0 * tot / n_dec if n_dec else float("nan")
    return out


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-e", required=True, help="实验臂（PIN_K=2）的 ladder tag 目录")
    ap.add_argument("--arm-b", required=True, help="基线臂（复用 INJECT_VS_RU_B）的 ladder tag 目录")
    ap.add_argument("--stderr-e", default="", help="实验臂归档后的 stderr 目录")
    ap.add_argument("--stderr-b", default="", help="基线臂归档后的 stderr 目录")
    ap.add_argument("--json", default="")
    ap.add_argument("--label-e", default="PIN_K=2")
    ap.add_argument("--label-b", default="BASE(B 臂)")
    ap.add_argument("--no-rule", action="store_true", help="只打表，不判档（想自己判时用）")
    args = ap.parse_args()

    de, db = resolve(args.arm_e), resolve(args.arm_b)
    E, B = read_arm(de), read_arm(db)

    # ── 0. 可比性自检（配对的前提）─────────────────────────────────────────────
    print("=" * 108)
    print("可比性自检（配对的前提）")
    print("=" * 108)
    same = [t for t in TOKENS
            if E[t].get("seat") is not None and B[t].get("seat") is not None
            and E[t]["seat"] == B[t]["seat"]]
    print(f"  * token 清单：两臂都按 {len(TOKENS)} 个 token（8 seed × 2 座位）取")
    print(f"  * 两臂都在、且**我方座位相同**的 token：**{len(same)}/16** "
          f"{'✓ 可逐 token 配对' if len(same) == len(TOKENS) else '✗ 配对会缺 token'}")
    if len(same) != len(TOKENS):
        print(f"    缺：{[t for t in TOKENS if t not in same]}")

    def opp_of(arm: dict, toks: list[str]) -> list[str]:
        return sorted({(arm[t]["p1"] if arm[t]["seat"] == 0 else arm[t]["p0"])
                       for t in toks if arm[t].get("seat") is not None})

    print(f"  * 对手标签：E = {opp_of(E, same)}   B = {opp_of(B, same)}"
          f"   {'✓ 同一份对手' if opp_of(E, same) == opp_of(B, same) else '✗ 对手不同，配对不成立'}")
    print("  * 配置：脚本前置断言已保证两臂 BASE 与 `_inject_vs_runnerup.sh:48` **逐字相同**、"
          "实验臂只多一个 `AZAI_PIN_K=2`")
    print("    （见 `_pin_k2_vs_runnerup.sh` 输出头：`✓ BASE 与 _inject_vs_runnerup.sh 逐字相同（600 字符）`）")

    # ── 1. 逐 token 表 ────────────────────────────────────────────────────────
    print()
    print("=" * 108)
    print("逐 token 配对表（Δhp = 我方基地血 − 对方基地血；ΔHP = 实验臂 − 基线臂）")
    print("=" * 108)
    print(f"{'token':<6}{'座位':>4}{'E':>3}{'B':>3}{'hpE(我,敌)':>13}{'ΔhpE':>7}"
          f"{'hpB(我,敌)':>13}{'ΔhpB':>7}{'ΔHP':>7}")
    rows: list[dict] = []
    for tok in TOKENS:
        e, b = E[tok], B[tok]
        if not (e.get("seat") is not None and b.get("seat") is not None
                and e["seat"] == b["seat"]):
            print(f"{tok:<6}   —— 不可配对（E: {e.get('missing') or e.get('verdict')} / "
                  f"B: {b.get('missing') or b.get('verdict')}）")
            continue
        d = e["d_hp"] - b["d_hp"]
        we = "胜" if e.get("score") == 1.0 else ("负" if e.get("score") == 0.0 else "?")
        wb = "胜" if b.get("score") == 1.0 else ("负" if b.get("score") == 0.0 else "?")
        hp_e = f"{e['hp_ours']},{e['hp_opp']}"
        hp_b = f"{b['hp_ours']},{b['hp_opp']}"
        seat_lab = "P" + str(e["seat"])
        print(f"{tok:<6}{seat_lab:>4}{we:>3}{wb:>3}{hp_e:>13}{e['d_hp']:>+7d}"
              f"{hp_b:>13}{b['d_hp']:>+7d}{d:>+7d}")
        rows.append({"token": tok, "seat": e["seat"], "d_hp_e": e["d_hp"], "d_hp_b": b["d_hp"],
                     "d_hp": d, "score_e": e.get("score"), "score_b": b.get("score"),
                     "verdict_e": e.get("verdict"), "verdict_b": b.get("verdict"),
                     "rounds_e": e.get("rounds"), "rounds_b": b.get("rounds"),
                     "illegal_e": e.get("illegal_judge"), "illegal_b": b.get("illegal_judge"),
                     "terminal_e": e.get("terminal")})

    n = len(rows)
    if n == 0:
        print("\n!! 没有任何可配对的 token —— 实验臂还没跑完？")
        return 1

    # ── 2. 主判据 ─────────────────────────────────────────────────────────────
    diffs = [r["d_hp"] for r in rows]
    m, lo, hi, sd = mean_ci(diffs)
    w = sum(1 for x in diffs if x > 0)
    l = sum(1 for x in diffs if x < 0)
    z = sum(1 for x in diffs if x == 0)
    p_sign = sign_test(w, l)
    me = statistics.fmean([r["d_hp_e"] for r in rows])
    mb = statistics.fmean([r["d_hp_b"] for r in rows])
    print()
    print("=" * 108)
    print("【主判据】配对 ΔHP（实验臂 − 基线臂）")
    print("=" * 108)
    print(f"  n = {n} 个配对 token（覆盖 {len({r['seat'] for r in rows})} 个座位）")
    print(f"  **mean(ΔHP) = {m:+.2f}   95%CI [{lo:+.2f}, {hi:+.2f}]   sd = {sd:.2f}**")
    print(f"  符号：+ {w} / − {l} / 0 {z}   ⇒ **双侧符号检验 p = {p_sign:.4f}**")
    print(f"  对照：E 的血量差均值 = {me:+.2f}、B = {mb:+.2f}（差 {me - mb:+.2f}，"
          f"应等于 mean(ΔHP) = {m:+.2f}）")

    # ── 3. 副判据：胜率 + 座位拆分 ─────────────────────────────────────────────
    print()
    print("=" * 108)
    print("【副判据】胜率（Wilson 95%CI）与按座位拆分")
    print("=" * 108)

    def winrep(sub: list[dict], label: str) -> None:
        nn = len(sub)
        if not nn:
            print(f"  {label}: n=0")
            return
        pe = sum(1 for r in sub if r["score_e"] == 1.0)
        pb = sum(1 for r in sub if r["score_b"] == 1.0)
        le_w, he_w = wilson(pe, nn)
        lb_w, hb_w = wilson(pb, nn)
        dd = [r["d_hp"] for r in sub]
        mm_, ll_, hh_, _ = mean_ci(dd)
        print(f"  {label:<6} E: {pe}W-{nn - pe}L = {100.0 * pe / nn:5.1f}% "
              f"(Wilson 95%CI {100.0 * le_w:.1f}–{100.0 * he_w:.1f}%)"
              f"   |   B: {pb}W-{nn - pb}L = {100.0 * pb / nn:5.1f}% "
              f"(Wilson 95%CI {100.0 * lb_w:.1f}–{100.0 * hb_w:.1f}%)")
        print(f"  {'':<6} 配对 ΔHP = {mm_:+.2f} (95%CI {ll_:+.2f}~{hh_:+.2f}；"
              f"+{sum(1 for x in dd if x > 0)}/−{sum(1 for x in dd if x < 0)}/"
              f"0{sum(1 for x in dd if x == 0)})")

    winrep(rows, "全部")
    for seat in (0, 1):
        winrep([r for r in rows if r["seat"] == seat], f"@P{seat}")
    se0 = [r["d_hp"] for r in rows if r["seat"] == 0]
    se1 = [r["d_hp"] for r in rows if r["seat"] == 1]
    if se0 and se1:
        m0, m1 = statistics.fmean(se0), statistics.fmean(se1)
        hetero = (m0 > 0) != (m1 > 0)
        tag_h = "符号相反 ⇒ 必须同时报这条（§0.4 第 2 条）" if hetero else "符号一致"
        print(f"\n  ⚠️ 座位异质性：@P0 mean(ΔHP) = {m0:+.2f}、@P1 = {m1:+.2f} ⇒ **{tag_h}**")

    # ── 4. 可行性：决策耗时 ───────────────────────────────────────────────────
    print()
    print("=" * 108)
    print("【可行性】决策耗时（口径 = 每局 stderr 最后一条 `⏱️ 决策耗时` / `耗时累计`；对局取中位）")
    print("=" * 108)
    le = lat_summary(read_latency(resolve(args.stderr_e))) if args.stderr_e else {}
    lb = lat_summary(read_latency(resolve(args.stderr_b))) if args.stderr_b else {}
    print(f"{'臂':<14}{'局数':>5}{'决策数':>8}{'p50(s)':>9}{'p95(s)':>9}{'max(s)':>9}"
          f"{'≥10s':>7}{'占比':>9}{'≥20s':>7}{'≥30s':>7}")
    for lab, s in ((args.label_e, le), (args.label_b, lb)):
        if not s:
            print(f"{lab:<14}   —— 没读到（没给 --stderr 或归档里没有我方 stderr）")
            continue
        print(f"{lab:<14}{s['n_games']:>5}{s['n_dec']:>8}{s['p50']:>9.2f}{s['p95']:>9.2f}"
              f"{s['max']:>9.2f}{s['ge10']:>7}{s['ge10_pct']:>8.2f}%{s['ge20']:>7}{s['ge30']:>7}")
    if le and lb:
        print(f"\n  放大倍数：p50 ×{le['p50'] / lb['p50']:.2f}、p95 ×{le['p95'] / lb['p95']:.2f}、"
              f"max ×{le['max'] / lb['max']:.2f}；dec_ge10 占比 "
              f"{lb['ge10_pct']:.2f}% → {le['ge10_pct']:.2f}%"
              f"（×{(le['ge10_pct'] / lb['ge10_pct']) if lb['ge10_pct'] else float('nan'):.2f}）")
        red_a = le["p95"] > 3.0 * lb["p95"]
        red_b = le["ge10_pct"] > 25.0
        amb_a = le["p95"] > 2.0 * lb["p95"]
        amb_b = le["ge10_pct"] > 2.0 * lb["ge10_pct"]
        if red_a or red_b:
            hits = "+".join([x for x, f in (("p95 > 3× 基线", red_a),
                                            ("dec_ge10 > 25%", red_b)) if f])
            print(f"  预注册红线判定：❌ **不可接受**（{hits}）")
        elif amb_a or amb_b:
            hits = "+".join([x for x, f in (("p95 > 2× 基线", amb_a),
                                            ("dec_ge10 > 2× 基线", amb_b)) if f])
            print(f"  预注册红线判定：⚠️ **边缘**（{hits}）")
        else:
            print("  预注册红线判定：✅ **可接受**（两条红线都没触发）")
        print(f"    · 红线 (a) p95 > 3× 基线（> {3.0 * lb['p95']:.2f}s）：实测 {le['p95']:.2f}s "
              f"⇒ {'触发' if red_a else '未触发'}")
        print(f"    · 红线 (b) dec_ge10 占比 > 25%：实测 {le['ge10_pct']:.2f}% "
              f"⇒ {'触发' if red_b else '未触发'}")
        print(f"    · 边缘线：p95 > 2×（> {2.0 * lb['p95']:.2f}s）= {le['p95'] > 2.0 * lb['p95']}"
              f"、dec_ge10 > 2×（> {2.0 * lb['ge10_pct']:.2f}%）= "
              f"{le['ge10_pct'] > 2.0 * lb['ge10_pct']}")
        print("    · 分母口径：`dec_n` 取每 25 决策一行的**最后一条**（每局末尾最多漏 <25 个决策），"
              "两臂同一口径 ⇒ 只影响绝对占比的第 3 位")

    # ── 5. 健康度 ─────────────────────────────────────────────────────────────
    print()
    print("=" * 108)
    print("【健康度】verdict / illegal / terminal（§0.6 四条硬断言）")
    print("=" * 108)
    for lab, arm, sdir in (("E", E, args.stderr_e), ("B", B, args.stderr_b)):
        valid = [t for t in TOKENS if arm[t].get("seat") is not None]
        vs = [arm[t].get("verdict") for t in valid]
        bad = [(t, arm[t].get("verdict")) for t in valid if arm[t].get("verdict") != "engine"]
        ill = sum(int(arm[t].get("illegal_judge") or 0) for t in valid)
        term = [(t, arm[t].get("terminal")) for t in valid if arm[t].get("terminal") != "True"]
        print(f"  {lab} 臂：有效局 {len(valid)}/16；判词分布 "
              f"{dict((v, vs.count(v)) for v in sorted(set(vs)))}；裁判侧 illegal 合计 = {ill}")
        if bad:
            print(f"    !! 不是 verdict=engine 的：{bad}")
        if term:
            print(f"    !! terminal != True 的：{term}")
        if sdir:
            st = resolve(sdir)
            ours = []
            for seat in (0, 1):
                for s in SEEDS:
                    f = st / f"ai{seat}_{OUR_LABEL}_seed{s}.stderr.log"
                    if f.is_file():
                        mm = RE_ILLSUM.findall(f.read_text(encoding="utf-8", errors="replace"))
                        ours.append((f.name, max((int(x) for x in mm), default=None)))
            nz = [(f, v) for f, v in ours if v not in (0, None)]
            print(f"    我方 stderr 归档 {len(ours)} 份；`illegal_total` 非 0 的：{nz if nz else '无 ✓'}")

    # ── 6. 预注册判读档位 ─────────────────────────────────────────────────────
    print()
    print("=" * 108)
    print("【判读】按 §0.3 的预注册四档")
    print("=" * 108)
    tier = None
    if args.no_rule:
        print("  （--no-rule：不判档）")
    elif lo > 0 and p_sign < 0.05:
        tier = "A"
        print("  ➜ **A 类轴塌缩是真短板**（95%CI 下界 > 0 且 符号检验 p < 0.05）")
        print("     下一步：转正 `PIN_K=2` 并扫 K（3/5），同时按上面的耗时判定处理成本。")
    elif hi < 0 and p_sign < 0.05:
        tier = "C"
        print("  ➜ **C 现役的 HOLD 偏好是对的**（95%CI 上界 < 0 且 符号检验 p < 0.05）")
        print("     下一步：`PIN_K` 只当诊断留档，不再往「加候选」方向走。")
    elif abs(m) <= 5.0 and lo <= 0 <= hi:
        tier = "B"
        print(f"  ➜ **B 问题不在「没候选」**（|mean| = {abs(m):.2f} ≤ 5 且 CI "
              f"{lo:+.2f}~{hi:+.2f} 含 0）")
        print("     下一步：测**「把 HOLD 也放进长 rollout 再决定」**"
              "（`MC_SCORE=net` 的 4×16 回合证据 → flat 长 rollout）。")
    else:
        tier = "D"
        print(f"  ➜ **D 不确定（灰区）**（mean = {m:+.2f}、CI {lo:+.2f}~{hi:+.2f}、"
              f"p = {p_sign:.4f}）")
        print("     下一步：扩样本（加 seed / 换对手）再判 —— **不许**把 D 读成 A 或 B。")
    print(f"\n  ⚠️ n = {n} 个配对 token 的统计功效很低：CI 半宽约 ±{(hi - lo) / 2:.1f} 血量。"
          "报结论时必须带上这个区间。")

    if args.json:
        outp = resolve(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "paired": rows, "mean_d_hp": m, "ci": [lo, hi], "sd": sd,
            "sign": {"plus": w, "minus": l, "zero": z, "p": p_sign},
            "latency": {"E": le, "B": lb},
            "tier": tier,
            "arm_e": str(de), "arm_b": str(db),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
