"""三臂读数汇总：A（亚军注入）/ P（安慰剂）/ P2（安慰剂·78% HOLD）对同一个 B（基准）。

2026-10-28；报告 `docs/placebo_inject_ab_20261028.md`。

读三份 `ab_search_modes.py --paired-sides` 的产出目录（**只读**，不覆盖任何日志）：

| 臂 | 目录（默认） | 来源 |
|---|---|---|
| A  | `match_results/oracle_inject_ab/ab16`  | 已完成的那次 A/B（13W-0D-3L = 81.2%） |
| P  | `match_results/placebo_inject_ab/ab16_h0` | 本次新跑（hold_p=0，每次都给一个随机合法 op） |
| P2 | `match_results/placebo_inject_ab/ab16_h078` | 本次新跑（hold_p=0.78） |

输出五块：

1. **每局明细**（每个臂：arm 座位 / 输赢 / `hp_arm - hp_B` / 回合 / 判词 / `illegal=`）；
2. **三臂汇总**：W/D/L、胜率 + Wilson 95%CI、平均血量差 + t 95%CI、平均回合、符号检验 p、座位拆分、
   逐 seed 配对分；
3. **判据档**（预注册，见报告 §1.1）：P 的胜率落在 A/B/C 哪一档；
4. **注入侧读数**：A vs P vs P2 的 `asks/inject/new/dup/hit/hit_sel` 与比率，外加 P/P2 的每决策日志统计
   （action 类型直方图、`hit` 一致率、`hit_nonhold`）；A 的日志产在"新口径"之前 ⇒ `hit` 就是老口径；
5. **决策耗时**：每局 p50/p95/max → 取中位；`≥10/20/30 s` 的占比（固定阈值口径）。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_placebo_ab.py
  # 可选：--arm P2=match_results/placebo_inject_ab/ab16_h078
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RE_RESULT = re.compile(
    r"RESULT seed=(?P<seed>\d+) .*?winner=(?P<winner>\S+) winner_side=(?P<ws>\S+) "
    r"verdict=(?P<verdict>\S+).*?rounds=(?P<rounds>\d+) terminal=(?P<term>\S+) "
    r"base_hp=(?P<hp0>-?\d+),(?P<hp1>-?\d+) coins=(?P<c0>-?\d+),(?P<c1>-?\d+)"
    r"(?: illegal=(?P<illegal>\d+))?")
RE_STATS = re.compile(r"注入(?:统计|累计): (?P<body>.+)$", re.MULTILINE)
RE_ILLSUM = re.compile(r"illegal-summary .*?illegal_total=(?P<tot>\d+)")
RE_DEC = re.compile(r"决策耗时: n=(?P<n>\d+) p50=(?P<p50>[\d.]+)s p95=(?P<p95>[\d.]+)s "
                    r"max=(?P<max>[\d.]+)s")
RE_DEC_P = re.compile(r"决策耗时 p50=(?P<p50>[\d.]+)s p95=(?P<p95>[\d.]+)s max=(?P<max>[\d.]+)s "
                      r"n=(?P<n>\d+)")
RE_CUM = re.compile(r"耗时累计: dec_n=(?P<n>\d+) dec_max=(?P<mx>[\d.]+)s dec_slow=(?P<slow>\d+)"
                    r"\s*(?P<rest>.*)$", re.MULTILINE)
RE_INJLINE = re.compile(r"\[(?:oracle|placebo)\] inject round=(?P<round>\d+) action=(?P<act>\S+) "
                        r"dup=(?P<dup>\w+) new=(?P<new>\w+) chosen=(?P<chosen>\S+) hit=(?P<hit>[01])")
RE_ARM_NAME = re.compile(r"ai(\d)_.*_seed(\d+)_p(\d)\.stderr\.log$")

# t 分布双侧 97.5% 分位（df=1..30；n=16 ⇒ df=15）
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
         30: 2.042}


def _num(txt: str, key: str, cast=float, default=None):
    m = re.search(rf"\b{key}=([-\d.]+)", txt)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except ValueError:
        return default


def _wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间（比例小样本比 Wald 稳）。k 可以是"得分和"（含平局 0.5）。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _binom_two_sided(w: int, l: int) -> float:
    """符号检验（忽略平局）：p = 2·P(X ≥ max(w,l))，X~Bin(w+l, .5)。"""
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _pct(vals: list, q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(float(v) for v in vals)
    k = int(round(q * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, k))]


def parse_games(out_dir: Path) -> list[dict]:
    games = []
    for log in sorted(out_dir.glob("seed*_p*.log")):
        m = re.search(r"seed(\d+)_p(\d)\.log$", log.name)
        if not m:
            continue
        seed, side = int(m.group(1)), int(m.group(2))
        txt = log.read_text(encoding="utf-8", errors="replace")
        r = RE_RESULT.search(txt)
        rec = {"seed": seed, "side": side, "log": log.name, "result": bool(r),
               "verdict": None, "score": None, "hp_arm": None, "hp_b": None,
               "rounds": None, "illegal": None}
        if r:
            ws = r.group("ws")
            ws_i = int(ws[1]) if ws in ("p0", "p1") else None
            hp = [int(r.group("hp0")), int(r.group("hp1"))]
            rec.update(verdict=r.group("verdict"), rounds=int(r.group("rounds")),
                       illegal=(int(r.group("illegal")) if r.group("illegal") is not None else None),
                       hp_arm=hp[side], hp_b=hp[1 - side],
                       score=(0.5 if ws_i is None else (1.0 if ws_i == side else 0.0)))
        games.append(rec)
    return games


def parse_stderr(stderr_dir: Path) -> dict:
    """被测臂（= flat 侧 = 注入/安慰剂臂）的 stderr 读数。

    ⚠️ 文件名 `ai{s}_..._seed{N}_p{s}` 里两个数字相同 = 该局被测臂那一份。
    """
    tot = {"games": 0, "asks": 0, "ok": 0, "inject": 0, "new": 0, "dup": 0, "dup_hold": 0,
           "dup_nonhold": 0, "hit": 0, "hit_new": 0, "hit_dup": 0, "hit_sel": 0,
           "illegal": 0, "timeout": 0, "dead": 0, "checked": 0,
           "illegal_sent": 0, "gate_checks": 0, "gate_dropped": 0,
           "gen_ms": [], "oracle_ms": [], "hold_p": None, "kinds": {},
           "ms_avg": [], "ms_p50": [], "ms_p95": [], "ms_max": [],
           "inj_lines": 0, "inj_hit": 0, "inj_hit_nonhold": 0, "inj_new": 0, "inj_dup": 0,
           "inj_act_kind": {}, "per_game": []}
    for f in sorted(stderr_dir.glob("ai?_*_p?.stderr.log")):
        m = RE_ARM_NAME.search(f.name)
        if not m or m.group(1) != m.group(3):
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        rec = {"file": f.name, "seed": int(m.group(2)), "side": int(m.group(3))}
        st = RE_STATS.findall(txt)
        if st:
            body = st[-1]
            for k in ("asks", "ok", "inject", "new", "dup", "dup_hold", "dup_nonhold",
                      "hit", "hit_new", "hit_dup", "hit_sel", "illegal", "timeout", "dead",
                      "checked", "avg_ms", "p50_ms", "p95_ms", "max_ms",
                      "gen_ms_avg", "gen_ms_max"):
                rec[k] = _num(body, k, float, None)
            hr = re.search(r"hit_rate=([\d.]+)%", body)
            rec["hit_rate"] = float(hr.group(1)) if hr else None
            hra = re.search(r"hit_rate_asks=([\d.]+)%", body)
            rec["hit_rate_asks"] = float(hra.group(1)) if hra else None
            rec["hit_sel_rate"] = _num(body, "hit_sel_rate", float, None)
            rec["hold_p"] = _num(body, "hold_p", float, None)
            km = re.search(r"kinds=\{([^}]*)\}", body)
            if km:
                for part in km.group(1).split(","):
                    kk, _, vv = part.partition(":")
                    if kk.strip() and vv.strip().isdigit():
                        rec.setdefault("kinds", {})[kk.strip()] = int(vv)
            # 老口径（A/B 那次）没有 hit_sel 字段 ⇒ 它的 `hit` 就是"额外候选被选中"
            rec["hit_is_old_semantics"] = rec.get("hit_sel") is None
            for k in ("asks", "ok", "inject", "new", "dup", "dup_hold", "dup_nonhold",
                      "hit", "hit_new", "hit_dup", "hit_sel", "illegal", "timeout",
                      "dead", "checked"):
                if rec.get(k):
                    tot[k] += int(rec[k])
            for k in ("gen_ms_avg", "max_ms", "p50_ms", "p95_ms", "avg_ms"):
                if rec.get(k):
                    tot["gen_ms" if k == "gen_ms_avg" else "oracle_ms"].append(float(rec[k]))
            for _k, _dst in (("avg_ms", "ms_avg"), ("p50_ms", "ms_p50"),
                             ("p95_ms", "ms_p95"), ("max_ms", "ms_max")):
                if rec.get(_k) is not None:
                    tot[_dst].append(float(rec[_k]))
            if rec.get("hold_p") is not None:
                tot["hold_p"] = rec["hold_p"]
            for kk, vv in (rec.get("kinds") or {}).items():
                tot["kinds"][kk] = tot["kinds"].get(kk, 0) + int(vv)
            tot["games"] += 1
        # 每决策一条注入日志（只有 2026-10-28 之后产的日志才有）
        for mm in RE_INJLINE.finditer(txt):
            tot["inj_lines"] += 1
            act = mm.group("act")
            hitv = int(mm.group("hit"))
            if mm.group("new") == "1":
                tot["inj_new"] += 1
            if mm.group("dup") == "1":
                tot["inj_dup"] += 1
            if hitv:
                tot["inj_hit"] += 1
            if act == "()":
                kind = "hold"
            elif mm.group("act").startswith("(("):
                kind = "multi"
            else:
                t = re.match(r"\((\d+),", act)
                kind = f"op{t.group(1)}" if t else "?"
            tot["inj_act_kind"][kind] = tot["inj_act_kind"].get(kind, 0) + 1
            if hitv and act != "()":
                tot["inj_hit_nonhold"] += 1
        ill = RE_ILLSUM.findall(txt)
        rec["illegal_sent"] = max((int(x) for x in ill), default=0)
        tot["illegal_sent"] += rec["illegal_sent"]
        g = re.findall(r"过滤闸门累计: checks=(\d+) dropped=(\d+)", txt)
        if g:
            rec["gate_checks"], rec["gate_dropped"] = int(g[-1][0]), int(g[-1][1])
            tot["gate_checks"] += rec["gate_checks"]
            tot["gate_dropped"] += rec["gate_dropped"]
        # 决策耗时：收尾那行优先，其次最后一条周期行
        dm = RE_DEC.findall(txt)
        if dm:
            rec["dec"] = {"n": int(dm[-1][0]), "p50": float(dm[-1][1]),
                          "p95": float(dm[-1][2]), "max": float(dm[-1][3]), "src": "final"}
        else:
            dp = RE_DEC_P.findall(txt)
            if dp:
                rec["dec"] = {"n": int(dp[-1][3]), "p50": float(dp[-1][0]),
                              "p95": float(dp[-1][1]), "max": float(dp[-1][2]), "src": "periodic"}
        cm = RE_CUM.findall(txt)
        if cm:
            rest = cm[-1][3]
            rec["cum"] = {"n": int(cm[-1][0]), "dec_max": float(cm[-1][1]),
                          "dec_slow": int(cm[-1][2]),
                          "ge10": _num(rest, "dec_ge10", int, 0) or 0,
                          "ge20": _num(rest, "dec_ge20", int, 0) or 0,
                          "ge30": _num(rest, "dec_ge30", int, 0) or 0}
            for k in ("ge10", "ge20", "ge30"):
                tot[k] = tot.get(k, 0) + rec["cum"][k]
            tot["dec_slow"] = tot.get("dec_slow", 0) + rec["cum"]["dec_slow"]
            tot["dec_n"] = tot.get("dec_n", 0) + rec["cum"]["n"]
        rec["stopped"] = "停用注入" in txt
        tot["per_game"].append(rec)
    return tot


def _load_arm(label: str, out_dir: Path) -> dict:
    games = parse_games(out_dir)
    st = Path(out_dir) / "stderr"
    if not st.is_absolute():
        st = REPO / st
    inj = parse_stderr(st)
    inj["hit_sel_equiv"] = (inj["hit"] if (inj["games"] and
                                          all(r.get("hit_is_old_semantics") for r in inj["per_game"]))
                            else inj["hit_sel"])
    inj["hit_sel_semantics"] = ("老口径 hit（日志里没有 hit_sel 字段）"
                               if (inj["games"] and inj["hit_sel"] == 0
                                   and all(r.get("hit_is_old_semantics") for r in inj["per_game"]))
                               else "新口径 hit_sel")
    return {"label": label, "dir": out_dir, "games": games, "inj": inj}


def _stat_block(arm: dict) -> dict:
    valid = [g for g in arm["games"] if g["result"] and g["verdict"] == "engine"]
    n = len(valid)
    w = sum(1 for g in valid if g["score"] == 1.0)
    d = sum(1 for g in valid if g["score"] == 0.5)
    l = sum(1 for g in valid if g["score"] == 0.0)
    rate = 100.0 * sum(g["score"] for g in valid) / n if n else float("nan")
    lo, hi = _wilson(sum(g["score"] for g in valid), n)
    diffs = [g["hp_arm"] - g["hp_b"] for g in valid if g["hp_arm"] is not None]
    if diffs:
        mean = sum(diffs) / len(diffs)
        sd = (sum((x - mean) ** 2 for x in diffs) / max(1, len(diffs) - 1)) ** 0.5
        se = sd / math.sqrt(len(diffs))
        t = _T975.get(len(diffs) - 1, 1.96)
        ci = (mean - t * se, mean + t * se)
    else:
        mean = se = float("nan")
        ci = (float("nan"), float("nan"))
    rds = [g["rounds"] for g in valid if g["rounds"] is not None]
    per_seed: dict = {}
    for g in valid:
        per_seed.setdefault(g["seed"], []).append(g["score"])
    pair = {s: (sum(v) / len(v)) for s, v in per_seed.items()}
    return {"n": n, "w": w, "d": d, "l": l, "rate": rate, "ci95": (100 * lo, 100 * hi),
            "hp_mean": mean, "hp_se": se, "hp_ci": ci,
            "rounds": (sum(rds) / len(rds)) if rds else float("nan"),
            "p_sign": _binom_two_sided(w, l), "pair": pair,
            "p0": [g for g in valid if g["side"] == 0], "p1": [g for g in valid if g["side"] == 1],
            "bad": [g for g in arm["games"] if g not in valid],
            "illegal": sum(g["illegal"] or 0 for g in arm["games"])}


def _seat_line(groups: list, label: str) -> str:
    valid = [g for g in groups if g["result"] and g["verdict"] == "engine"]
    n = len(valid)
    if not n:
        return f"{label:<6} n=0"
    w = sum(1 for g in valid if g["score"] == 1.0)
    l = sum(1 for g in valid if g["score"] == 0.0)
    d = n - w - l
    r = 100.0 * sum(g["score"] for g in valid) / n
    diffs = [g["hp_arm"] - g["hp_b"] for g in valid if g["hp_arm"] is not None]
    hp_txt = f"平均血量差={sum(diffs) / len(diffs):+6.2f}" if diffs else "平均血量差=--"
    return f"{label:<6} n={n:<3} {w}W/{d}D/{l}L  胜率={r:5.1f}%  {hp_txt}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="label=dir；可重复。默认 A/P/P2 三个")
    args = ap.parse_args()
    spec = {
        "A": "match_results/oracle_inject_ab/ab16",
        "P": "match_results/placebo_inject_ab/ab16_h0",
        "P2": "match_results/placebo_inject_ab/ab16_h078",
    }
    order = ["A", "P", "P2"]
    for item in args.arm:
        k, _, v = item.partition("=")
        spec[k.strip()] = v.strip()
        if k.strip() not in order:
            order.append(k.strip())

    arms = []
    for label in order:
        d = Path(spec[label])
        if not d.is_absolute():
            d = REPO / d
        if not d.is_dir():
            print(f"[skip] {label}: {d} 不存在")
            continue
        arms.append(_load_arm(label, d))
    if not arms:
        print("!! 没有任何臂的目录")
        return 1

    print("=" * 118)
    print("三臂读数（每个臂都是「被测臂 vs 同一个 B = BASE 不注入」；A 复用已有 A/B，P/P2 本次新跑）")
    print(f"共同项：AZAI_FLAT_SEED=53，seeds 3600-3607 × 2 座位 = 16 局，--paired-sides --jobs 8；"
          f"`verdict=engine` 才算有效局")
    for a in arms:
        print(f"  {a['label']:<3} = {a['dir'].relative_to(REPO).as_posix() if a['dir'].is_relative_to(REPO) else a['dir']}")

    # ── ① 每局明细 ────────────────────────────────────────────────────────
    for a in arms:
        print()
        print("=" * 118)
        print(f"① 每局明细 —— {a['label']} 臂（座位 = 该臂执哪边）")
        print(f"{'seed':>6} {'座位':>4} {'结果':>4} {'得分':>4} {'hp_arm':>6} {'hp_B':>5} "
              f"{'血量差':>6} {'回合':>5} {'判词':>14} {'illegal':>7}")
        for g in sorted(a["games"], key=lambda x: (x["seed"], x["side"])):
            tag = ("胜" if g["score"] == 1.0 else ("负" if g["score"] == 0.0 else
                                                   ("平" if g["score"] == 0.5 else "?")))
            diff = (g["hp_arm"] - g["hp_b"]) if g["hp_arm"] is not None else None
            score_txt = "--" if g["score"] is None else f"{g['score']:.1f}"
            ha = "--" if g["hp_arm"] is None else str(g["hp_arm"])
            hb = "--" if g["hp_b"] is None else str(g["hp_b"])
            dt = "--" if diff is None else f"{diff:+d}"
            rd = "--" if g["rounds"] is None else str(g["rounds"])
            il = "--" if g["illegal"] is None else str(g["illegal"])
            print(f"{g['seed']:>6} {'P' + str(g['side']):>4} {tag:>4} {score_txt:>4} "
                  f"{ha:>6} {hb:>5} {dt:>6} {rd:>5} {str(g['verdict']):>14} {il:>7}")

    # ── ② 三臂汇总 ────────────────────────────────────────────────────────
    print()
    print("=" * 118)
    print("② 三臂汇总（B 的得分 = 1 − 该臂得分；「B = 50%」是配对自对弈的期望值，不是独立测量）")
    print(f"{'臂':<3} {'n':>3} {'W/D/L':>10} {'胜率':>7} {'95%CI(Wilson)':>18} "
          f"{'平均血量差':>10} {'95%CI':>18} {'平均回合':>7} {'符号检验p':>9}")
    stats = {}
    for a in arms:
        s = _stat_block(a)
        stats[a["label"]] = s
        wdl = f"{s['w']}W/{s['d']}D/{s['l']}L"
        print(f"{a['label']:<3} {s['n']:>3} {wdl:>10} "
              f"{s['rate']:>6.1f}% [{s['ci95'][0]:>5.1f},{s['ci95'][1]:>5.1f}] "
              f"{s['hp_mean']:>+9.2f} [{s['hp_ci'][0]:>+6.2f},{s['hp_ci'][1]:>+6.2f}] "
              f"{s['rounds']:>7.0f} {s['p_sign']:>9.4f}")
        print(f"    {_seat_line(s['p0'], '@P0')}")
        print(f"    {_seat_line(s['p1'], '@P1')}")
        print(f"    逐 seed 配对分: " + "  ".join(f"{k}:{v:.1f}" for k, v in sorted(s["pair"].items())))
        print(f"    无效局 {len(s['bad'])}/{len(a['games'])}"
              + (f" -> {[(g['seed'], 'P' + str(g['side']), g['verdict']) for g in s['bad']]}"
                 if s["bad"] else "")
              + f"    RESULT 行 illegal= 合计 {s['illegal']}")

    # ── ③ 判据档 ──────────────────────────────────────────────────────────
    if "P" in stats and "A" in stats:
        sP, sA = stats["P"], stats["A"]
        print()
        print("=" * 118)
        print("③ 判据档（预注册，见报告 §1.1）")
        print(f"  A = {sA['rate']:.1f}%（n={sA['n']}）   P = {sP['rate']:.1f}%  "
              f"95%CI [{sP['ci95'][0]:.1f}, {sP['ci95'][1]:.1f}]   B = 50%")
        if sP["rate"] >= 65.0:
            print("  ⇒ **判据档 A**：P 也显著 > B ⇒ +31pp 主要来自「多一个候选/预算效应」，"
                  "**不是知识** ⇒ 「注入证明先验是瓶颈」不成立")
        elif sP["ci95"][1] < 65.0:
            print("  ⇒ **判据档 B**：P 的 95%CI 上界 < 65% ⇒ 与 B 的差别不足以解释 A 的 81.2% "
                  "⇒ 「知识确实有用」，旧结论立得住")
        else:
            print("  ⇒ **判据档 C**：介于两者之间 ⇒ 部分来自候选数、部分来自知识，需要进一步分离")

    # ── ④ 注入侧读数 ──────────────────────────────────────────────────────
    print()
    print("=" * 118)
    print("④ 注入侧读数（每个臂的 stderr：`注入统计` 收尾行 + 每决策日志统计）")
    for a in arms:
        o = a["inj"]
        if not o["games"]:
            print(f"  {a['label']}: 没找到被测臂 stderr")
            continue
        print(f"  ── {a['label']}（{o['games']} 局）"
              + (f"  hold_p={o['hold_p']:g}" if o.get("hold_p") is not None else ""))
        _old = all(r.get("hit_is_old_semantics") for r in o["per_game"])
        if _old:
            print(f"     ⚠️ 这批日志是**老口径**（`hit` = 额外候选被搜索选中，没有 hit_sel/dup_hold 字段）"
                  f"⇒ dup_hold/dup_nonhold/hit_new/hit_dup/hit_sel 全部缺失，不能与 P 的新口径字段混算")
        else:
            print(f"     dup_hold={o['dup_hold']} dup_nonhold={o['dup_nonhold']} "
                  f"hit_new={o['hit_new']} hit_dup={o['hit_dup']} hit_sel={o['hit_sel']}"
                  f"（hit=动作一致率；hit_sel=老口径）")
        print(f"     **跨版本可比**的读数：额外候选被搜索选中 = {o['hit_sel_equiv']}"
              f"（{o['hit_sel_semantics']}）⇒ 被选中率(hit_sel/new)="
              f"{100.0 * o['hit_sel_equiv'] / max(1, o['new']):.1f}%")
        print(f"     asks={o['asks']} ok={o['ok']} inject={o['inject']} new={o['new']} "
              f"dup={o['dup']}  new/asks={100.0 * o['new'] / max(1, o['asks']):.1f}%  "
              f"dup/asks={100.0 * o['dup'] / max(1, o['asks']):.1f}%")
        print(f"     illegal(候选整条丢)={o['illegal']} timeout={o['timeout']} dead={o['dead']}  "
              f"实际发出的非法 op（illegal-summary 合计）={o['illegal_sent']}  "
              f"过滤闸门 checks={o['gate_checks']} dropped={o['gate_dropped']}")
        if o["inj_lines"]:
            print(f"     每决策日志: {o['inj_lines']} 条  new={o['inj_new']} dup={o['inj_dup']}  "
                  f"hit={o['inj_hit']}（{100.0 * o['inj_hit'] / o['inj_lines']:.1f}%）"
                  f"  hit_nonhold={o['inj_hit_nonhold']}"
                  f"（{100.0 * o['inj_hit_nonhold'] / max(1, o['inj_lines'] - o['inj_act_kind'].get('hold', 0)):.1f}% of 非 HOLD）")
            print(f"     action 类型直方图: "
                  + "  ".join(f"{k}:{v}" for k, v in sorted(o["inj_act_kind"].items())))
        else:
            print("     每决策日志: **这批日志没有**（产在 2026-10-28 每决策日志之前）⇒ 只有上面的汇总；"
                  "A 的 `hit` 是**老口径**（额外候选被搜索选中）")
        if o.get("kinds"):
            print(f"     生成结果直方图（kinds）: " + "  ".join(f"{k}:{v}" for k, v in sorted(o["kinds"].items())))
        if o["ms_avg"]:
            print(f"     oracle 单次查询（每局的值，再取中位/最大）: "
                  f"avg 中位={_pct(o['ms_avg'], .5):.0f} max={max(o['ms_avg']):.0f} ms  "
                  f"p95 中位={_pct(o['ms_p95'], .5):.0f} ms  "
                  f"单次最大={max(o['ms_max']):.0f} ms")
        if o["gen_ms"]:
            print(f"     本地生成 gen_ms（每局 avg）: 中位={_pct(o['gen_ms'], .5):.3f} "
                  f"max={max(o['gen_ms']):.3f} ms")

    # ── ⑤ 决策耗时 ────────────────────────────────────────────────────────
    print()
    print("=" * 118)
    print("⑤ 决策耗时（被测臂；每局先算 p50/p95/max，再取中位）")
    print(f"{'臂':<3} {'局数':>4} {'p50 中位':>9} {'p50 范围':>16} {'p95 中位':>9} "
          f"{'p95 范围':>18} {'max 中位':>9} {'max 范围':>18} {'≥10s 占比':>10} {'≥20s 占比':>10}")
    for a in arms:
        o = a["inj"]
        decs = [r["dec"] for r in o["per_game"] if r.get("dec")]
        if not decs:
            print(f"{a['label']:<3}  （没有耗时读数）")
            continue
        p50 = [d["p50"] for d in decs]
        p95 = [d["p95"] for d in decs]
        mx = [d["max"] for d in decs]
        n_tot = o.get("dec_n", 0) or 0
        has_ge = bool(o.get("ge10", 0) or o.get("ge20", 0))
        g10 = 100.0 * o.get("ge10", 0) / n_tot if (n_tot and has_ge) else float("nan")
        # A 那批日志产在 `dec_ge*` 之前 ⇒ 用 `dec_slow`（阈值 AZAI_SLOW_DECISION_SEC，默认 20s）当 ≥20s 口径
        g20 = (100.0 * o.get("ge20", 0) / n_tot if (n_tot and has_ge) else
               (100.0 * o.get("dec_slow", 0) / n_tot if n_tot else float("nan")))
        _g10 = f"{g10:>9.2f}%" if has_ge else f"{'NA':>10}"
        print(f"{a['label']:<3} {len(decs):>4} {_pct(p50, .5):>8.2f}s "
              f"[{min(p50):>5.2f},{max(p50):>5.2f}]s {_pct(p95, .5):>8.2f}s "
              f"[{min(p95):>5.2f},{max(p95):>6.2f}]s {_pct(mx, .5):>8.2f}s "
              f"[{min(mx):>5.2f},{max(mx):>6.2f}]s {_g10} {g20:>9.2f}%")
    print("  注：`≥20s 占比` = dec_ge20 之和 ÷ dec_n 之和；这批 A 的日志产在 `dec_ge*` 之前 ⇒ 用 "
          "`dec_slow/dec_n`（阈值 AZAI_SLOW_DECISION_SEC，默认 20s）代替，`≥10s` 记为 NA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
