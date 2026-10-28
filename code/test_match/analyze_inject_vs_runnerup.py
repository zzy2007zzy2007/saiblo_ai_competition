#!/usr/bin/env python
"""外部标尺 A/B 的读数汇总：**把"注入臂 / 非注入臂 vs 亚军本体"的落盘日志算成表**。

用途（`docs/inject_vs_runnerup_20261028.md`）：
  * 每臂：胜率 + Wilson 95%CI + 血量差（均值 / t 95%CI）+ 按座位拆分；
  * A−B：同一 token（同图同座位）的配对分析（McNemar 精确检验 + 配对血量差）；
  * A 臂注入有效性：`注入已启用` / `asks` / `timeout` / `dead` / 路径形态 —— **静默失效**的核查；
  * 决策耗时：p50 / p95 / max（每局统计，再取中位与最坏）；
  * 有效性门槛：`verdict=engine`、`illegal=0`。

⚠️ 解析一律复用 `_tmp_ladder.read_game`（历史判据的唯一实现）—— 不在这里重写一遍
   （重写会把"败"记成"平"，见 `_tmp_ladder.py` 文件头的第 1 条）。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_inject_vs_runnerup.py \
      --arm inject_A=match_results/ladder_logs/INJECT_VS_RU_A \
      --stderr inject_A=match_results/inject_vs_ru/A/stderr \
      --arm inject_B=match_results/ladder_logs/INJECT_VS_RU_B \
      --stderr inject_B=match_results/inject_vs_ru/B/stderr \
      --pair inject_A inject_B \
      --ref VS_RU_newcfg=match_results/ladder_logs/VS_RU_newcfg \
      --ref VS_RUNNERUP_cells4=match_results/ladder_logs/VS_RUNNERUP_cells4 \
      --json match_results/inject_vs_ru/summary.json
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("_tmp_ladder", REPO / "_tmp_ladder.py")
_ladder = importlib.util.module_from_spec(_spec)
sys.modules["_tmp_ladder"] = _ladder
_spec.loader.exec_module(_ladder)

OUR_LABEL = "az_bridge_ai"
OPP_LABEL = "runnerup"

# t_{0.975, df}（双侧 95%）—— 小样本必须用 t，不能用 1.96
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
         30: 2.042, 31: 2.040, 32: 2.037, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    return _T975.get(df) or (1.96 if df > 120 else _T975[min(k for k in _T975 if k >= df)])


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间（k 可以是 0.5 步长的得分和；n = 局数）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mean_ci(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0}
    m = statistics.fmean(xs)
    if n == 1:
        return {"n": 1, "mean": m, "sd": None, "se": None, "lo": None, "hi": None}
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(n)
    h = t975(n - 1) * se
    return {"n": n, "mean": m, "sd": sd, "se": se, "lo": m - h, "hi": m + h}


def binom_two_sided(k: int, n: int) -> float:
    """精确二项双侧 p（p0=0.5）。"""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-12))


# ── 每局 ─────────────────────────────────────────────────────────────────────
NUM_RE = re.compile(r"([a-z_0-9]+)=(-?[\d.]+)")
INJ_LINE_RE = re.compile(r"\[(?:oracle|placebo)\] (?:注入统计|注入累计):\s*(.+)")
INJ_STEP_RE = re.compile(r"\[(?:oracle|placebo)\] inject round=(\d+).*?dup=(\S+) new=(\S+) "
                         r"chosen=(\S+) hit=(\S+)")
DEC_NEW_RE = re.compile(r"决策耗时 p50=([\d.]+)s p95=([\d.]+)s max=([\d.]+)s n=(\d+)")
DEC_END_RE = re.compile(r"决策耗时: n=(\d+) p50=([\d.]+)s p95=([\d.]+)s max=([\d.]+)s")
DEC_MAX_RE = re.compile(r"dec_max=([\d.]+)s")


def parse_stderr(path: Path) -> dict:
    """我们的 bridge 的 stderr → 注入/耗时/非法读数。文件不存在 ⇒ `{}`（B 臂也照样读）。"""
    out: dict = {"path": str(path), "exists": path.is_file()}
    if not out["exists"]:
        return out
    txt = path.read_text(encoding="utf-8", errors="replace")
    out["inj_enabled"] = 0
    out["inj_enabled_path"] = None
    out["inj_disabled_marks"] = 0
    for ln in txt.splitlines():
        if "注入已启用" in ln:
            out["inj_enabled"] = 1
            m = re.search(r"exe=(\S+)", ln)
            out["inj_enabled_path"] = m.group(1) if m else None
        if ("注入已停用" in ln) or ("永久停用" in ln) or ("启用失败" in ln):
            out["inj_disabled_marks"] += 1
    out["inj_fail_lines"] = sum(1 for ln in txt.splitlines()
                                if "查询失败" in ln or "OracleTimeout" in ln
                                or "OracleDesync" in ln or "OracleDead" in ln)
    # 注入统计：取**最后一条**（收尾那条最全）
    stats: dict = {}
    for ln in txt.splitlines():
        m = INJ_LINE_RE.search(ln)
        if m:
            stats = {k: float(v) for k, v in NUM_RE.findall(m.group(1))}
    out["inj_stats"] = stats
    # 每决策日志（新口径）：直接数行，与统计行交叉核对
    steps = INJ_STEP_RE.findall(txt)
    out["steps"] = len(steps)
    out["steps_new"] = sum(1 for s in steps if s[2] == "1")
    out["steps_dup"] = sum(1 for s in steps if s[1] == "1")
    out["steps_hit"] = sum(1 for s in steps if s[4] == "1")
    out["steps_hit_new"] = sum(1 for s in steps if s[2] == "1" and s[4] == "1")
    out["steps_hit_dup"] = sum(1 for s in steps if s[1] == "1" and s[4] == "1")
    out["steps_dup_hold"] = sum(1 for s in steps if s[1] == "1" and s[3] == "()")
    # 决策耗时（p50/p95/max）：取对局中周期打印的最后一条 + 收尾报告
    dec = None
    for ln in txt.splitlines():
        m = DEC_NEW_RE.search(ln)
        if m:
            dec = {"n": int(m.group(4)), "p50": float(m.group(1)),
                   "p95": float(m.group(2)), "max": float(m.group(3))}
        m2 = DEC_END_RE.search(ln)
        if m2:
            dec = {"n": int(m2.group(1)), "p50": float(m2.group(2)),
                   "p95": float(m2.group(3)), "max": float(m2.group(4))}
    out["dec"] = dec
    dm = DEC_MAX_RE.findall(txt)
    out["dec_max_hist"] = max((float(x) for x in dm), default=None)
    out["illegal_summary"] = 0
    for ln in txt.splitlines():
        if "illegal-summary" in ln or "非法操作自检汇总" in ln:
            m = re.search(r"illegal_total=(\d+)", ln)
            if m:
                out["illegal_summary"] = max(out["illegal_summary"], int(m.group(1)))
    return out


def read_arm(log_dir: Path, stderr_dir: Path | None) -> dict:
    games = []
    for lp in sorted(log_dir.glob("*.log")):
        token = lp.stem
        seed_m = re.match(r"(\d+)", token)
        if not seed_m:
            continue
        try:
            g = _ladder.read_game(lp, token)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[warn] 解析 {lp} 失败: {exc}", file=sys.stderr)
            continue
        p0, p1 = g.get("p0_label"), g.get("p1_label")
        if p0 == OUR_LABEL:
            seat = 0
        elif p1 == OUR_LABEL:
            seat = 1
        else:
            seat = None
        rec = {
            "token": token, "seed": int(seed_m.group(1)),
            "mirror": token.endswith("r"),
            "seat": seat, "p0": p0, "p1": p1,
            "winner": g.get("winner"), "winner_side": g.get("winner_side"),
            "verdict": g.get("verdict"), "rounds": g.get("rounds"),
            "hp": g.get("hp"), "coins": g.get("coins"),
            "illegal_judge": g.get("illegal"),
        }
        if seat is not None and g.get("hp"):
            rec["hp_ours"] = g["hp"][seat]
            rec["hp_opp"] = g["hp"][1 - seat]
            rec["hp_diff"] = rec["hp_ours"] - rec["hp_opp"]
        ws = (g.get("winner_side") or "").strip().lower()
        if seat is None:
            rec["score"] = None
        elif ws == f"p{seat}":
            rec["score"] = 1.0
        elif ws in ("p0", "p1"):
            rec["score"] = 0.0
        else:
            rec["score"] = 0.5 if ws in ("draw", "none", "") else None
        rec["valid"] = (rec["verdict"] == "engine")
        # 行为画像（"治疗有没有真的咬住"）：我方/对手的出招构成。数据来自 `--trace-ops` 的
        # `[round N] pX(...) -> [Operation(...)]` 行（`_ladder.read_game` 已经统计好）。
        if seat is not None:
            who, opp = f"p{seat}", f"p{1 - seat}"
            rec["our_ops"] = g["ops"][who]
            rec["our_act"] = g["act"][who]
            rec["our_hist"] = dict(g["hist"][who])
            rec["opp_hist"] = dict(g["hist"][opp])
            rec["opp_ops"] = g["ops"][opp]
            rec["opp_act"] = g["act"][opp]
        if stderr_dir is not None:
            sp = stderr_dir / f"ai{seat}_az_bridge_ai_seed{rec['seed']}.stderr.log" \
                if seat is not None else None
            rec["stderr"] = parse_stderr(sp) if sp is not None else {}
        games.append(rec)
    return {"log_dir": str(log_dir), "games": games}


def summarize(arm: dict) -> dict:
    games = arm["games"]
    valid = [g for g in games if g["valid"]]
    scored = [g for g in valid if g["score"] is not None]
    n = len(scored)
    win = sum(1 for g in scored if g["score"] == 1.0)
    loss = sum(1 for g in scored if g["score"] == 0.0)
    draw = sum(1 for g in scored if g["score"] == 0.5)
    pts = sum(g["score"] for g in scored)
    lo, hi = wilson(pts, n)
    hps = [g["hp_diff"] for g in scored if "hp_diff" in g]
    out = {
        "n_valid": len(valid), "n_games": len(games), "n_scored": n,
        "wins": win, "draws": draw, "losses": loss,
        "winrate": (pts / n) if n else None, "winrate_ci": (lo, hi),
        "hp_diff": mean_ci(hps),
        "byd_side": {},
        "verdicts": dict(collections.Counter(g["verdict"] for g in games)),
        "illegal_judge": sum(int(g.get("illegal_judge") or 0) for g in games),
        "rounds": [g["rounds"] for g in scored],
    }
    for seat in (0, 1):
        sub = [g for g in scored if g["seat"] == seat]
        if sub:
            p = sum(g["score"] for g in sub)
            l2, h2 = wilson(p, len(sub))
            out["byd_side"][f"P{seat}"] = {
                "n": len(sub), "pts": p, "winrate": p / len(sub), "ci": (l2, h2),
                "wins": sum(1 for g in sub if g["score"] == 1.0),
                "losses": sum(1 for g in sub if g["score"] == 0.0),
                "hp_diff": mean_ci([g["hp_diff"] for g in sub if "hp_diff" in g]),
            }
    # 注入 / 耗时（只有带 stderr 的臂才有）
    subs = [g["stderr"] for g in games if g.get("stderr", {}).get("exists")]
    if subs:
        asks = sum(int(s["inj_stats"].get("asks", 0)) for s in subs)
        # 注入覆盖率：asks 应该 ≈ 该局的回合数（每回合问一次）。明显偏小 ⇒ 注入中途死了/没启用。
        # ⚠️ 阈值 0.90 而不是 0.98：周期日志每 **25 个决策**才打一次（`_report()` 的收尾那行
        #    在对局结束时可能被 bridge 硬杀吃掉）⇒ `asks` 最多滞后 24 个决策（512 回合下 = 4.7%）。
        short = []
        for g in games:
            st = g.get("stderr") or {}
            if not st.get("exists"):
                continue
            a = int((st.get("inj_stats") or {}).get("asks", 0))
            r = int(g.get("rounds") or 0)
            if r and a < 0.90 * r:
                short.append({"token": g["token"], "asks": a, "rounds": r})
        out["inj_short"] = short
        out["inj_cover"] = (
            sum(int((g.get("stderr") or {}).get("inj_stats", {}).get("asks", 0)) for g in games
                if (g.get("stderr") or {}).get("exists"))
            / max(1, sum(int(g.get("rounds") or 0) for g in games
                         if (g.get("stderr") or {}).get("exists"))))
        # 覆盖率只在"真的开了注入"的臂里才有意义（B 臂 asks 恒 0 ⇒ 不要报成"覆盖不足"）
        if not any(s.get("inj_enabled") for s in subs):
            out["inj_short"] = []
            out["inj_cover"] = None
        out["inj"] = {
            "games_with_stderr": len(subs),
            "games_inj_enabled": sum(1 for s in subs if s.get("inj_enabled")),
            "paths_ok": sum(1 for s in subs
                            if (s.get("inj_enabled_path") or "").replace("\\", "/")
                            .upper().startswith("D:/")),
            "games_with_asks": sum(1 for s in subs if int(s["inj_stats"].get("asks", 0)) > 0),
            "disabled_marks": sum(int(s.get("inj_disabled_marks") or 0) for s in subs),
            "fail_lines": sum(int(s.get("inj_fail_lines") or 0) for s in subs),
            "asks_total": asks,
            "totals": {k: sum(int(s["inj_stats"].get(k, 0)) for s in subs)
                       for k in ("ok", "checked", "inject", "new", "dup", "dup_hold",
                                 "dup_nonhold", "hit", "hit_new", "hit_dup", "hit_sel",
                                 "illegal", "timeout", "dead")},
            "lat_ms": {
                k: (statistics.median([s["inj_stats"][k] for s in subs if k in s["inj_stats"]])
                    if any(k in s["inj_stats"] for s in subs) else None)
                for k in ("avg_ms", "p50_ms", "p95_ms", "max_ms")},
            "lat_max": max([s["inj_stats"].get("max_ms", 0) for s in subs], default=None),
            "steps": sum(int(s.get("steps") or 0) for s in subs),
            "steps_new": sum(int(s.get("steps_new") or 0) for s in subs),
            "steps_dup": sum(int(s.get("steps_dup") or 0) for s in subs),
            "steps_hit": sum(int(s.get("steps_hit") or 0) for s in subs),
            "steps_hit_new": sum(int(s.get("steps_hit_new") or 0) for s in subs),
            "steps_hit_dup": sum(int(s.get("steps_hit_dup") or 0) for s in subs),
            "steps_dup_hold": sum(int(s.get("steps_dup_hold") or 0) for s in subs),
            "illegal_ai": max([int(s.get("illegal_summary") or 0) for s in subs], default=0),
        }
        decs = [s["dec"] for s in subs if s.get("dec")]
        if decs:
            out["dec"] = {
                "games": len(decs),
                "decisions": sum(d["n"] for d in decs),
                "p50_median": statistics.median([d["p50"] for d in decs]),
                "p95_median": statistics.median([d["p95"] for d in decs]),
                "p95_max": max(d["p95"] for d in decs),
                "max_median": statistics.median([d["max"] for d in decs]),
                "max_worst": max(d["max"] for d in decs),
            }
    # ── 行为画像：出招构成（每 100 个我方决策的 op 数 / 各类 op 占比）──
    act = [g for g in games if g.get("our_act") is not None]
    if act:
        n_dec = sum(int(g["our_act"]) for g in act)
        n_ops = sum(int(g["our_ops"]) for g in act)
        types: collections.Counter = collections.Counter()
        for g in act:
            for k, v in (g.get("our_hist") or {}).items():
                types[k] += int(v)
        opp_types: collections.Counter = collections.Counter()
        opp_dec = sum(int(g.get("opp_act") or 0) for g in act)
        for g in act:
            for k, v in (g.get("opp_hist") or {}).items():
                opp_types[k] += int(v)
        tot = sum(types.values()) or 1
        out["behavior"] = {
            "decisions": n_dec, "ops": n_ops,
            "ops_per_decision": (n_ops / n_dec) if n_dec else None,
            "act_rate": (sum(1 for g in act if int(g["our_act"]) > 0) / len(act)) if act else None,
            "mix_pct": {k: 100.0 * v / tot for k, v in types.most_common(6)},
            "mix_counts": {k: int(v) for k, v in types.most_common(8)},
            "opp_decisions": opp_dec,
            "opp_mix_pct": ({k: 100.0 * v / (sum(opp_types.values()) or 1)
                             for k, v in opp_types.most_common(6)}),
        }
    return out


def paired(a: dict, b: dict) -> dict:
    """同一 token 的配对：A−B 的血量差、胜负列联表、McNemar 精确检验。"""
    ga = {g["token"]: g for g in a["games"] if g["valid"] and g["score"] is not None}
    gb = {g["token"]: g for g in b["games"] if g["valid"] and g["score"] is not None}
    common = sorted(set(ga) & set(gb), key=lambda t: (int(re.match(r"\d+", t).group()),
                                                      t.endswith("r")))
    diffs, rows = [], []
    both_w = a_only = b_only = both_l = 0
    for t in common:
        x, y = ga[t], gb[t]
        d = x.get("hp_diff", 0) - y.get("hp_diff", 0)
        diffs.append(d)
        sw = x["score"] >= 1.0
        sb = y["score"] >= 1.0
        if sw and sb:
            both_w += 1
        elif sw:
            a_only += 1
        elif sb:
            b_only += 1
        else:
            both_l += 1
        rows.append({"token": t, "seat": x["seat"], "score_A": x["score"], "score_B": y["score"],
                     "hp_A": x.get("hp_diff"), "hp_B": y.get("hp_diff"), "d_hp": d,
                     "accept_A": x.get("accept")})
    per_seed: dict[int, list[float]] = collections.defaultdict(list)
    for t in common:
        per_seed[ga[t]["seed"]].append(ga[t]["score"] - gb[t]["score"])
    return {
        "n_pairs": len(common),
        "rows": rows,
        "d_hp": mean_ci(diffs),
        "table": {"both_win": both_w, "A_only": a_only, "B_only": b_only, "both_lose": both_l},
        "mcnemar_p": binom_two_sided(a_only, a_only + b_only) if (a_only + b_only) else 1.0,
        "sign_test_p": binom_two_sided(a_only, a_only + b_only) if (a_only + b_only) else 1.0,
        "per_seed_delta": {str(s): sum(v) for s, v in sorted(per_seed.items())},
        "delta_winrate": ((a_only - b_only) / len(common)) if common else None,
    }


def shares(sa: dict, sb: dict) -> dict:
    """§2.4 预注册的份额公式：G = 1−B；Δ = A−B；s = Δ/G（含 Wilson 界的 min/max）。

    ⚠️ 这只是**操作化定义**（"候选通道 = 亚军那一手 + 它自己的搜索/评估"），不是因果份额。
    """
    if not sa.get("n_scored") or not sb.get("n_scored"):
        return {}
    a, b = sa["winrate"], sb["winrate"]
    a_lo, a_hi = sa["winrate_ci"]
    b_lo, b_hi = sb["winrate_ci"]
    g = 1.0 - b
    return {
        "A": a, "B": b, "G": g, "delta": a - b,
        "s_point": (a - b) / g if g > 0 else None,
        "s_min": (a_lo - b_hi) / (1.0 - b_lo) if b_lo < 1 else None,
        "s_max": (a_hi - b_lo) / (1.0 - b_hi) if b_hi < 1 else None,
    }


def print_arm(label: str, arm: dict, s: dict) -> None:
    print(f"\n{'='*100}\n### {label}   (log_dir={arm['log_dir']})\n{'='*100}")
    print(f"{'token':<6}{'seat':<6}{'score':<7}{'hp我':<6}{'hp敌':<6}{'Δhp':<7}"
          f"{'回合':<6}{'verdict':<10}{'ill':<5}{'asks':<6}{'new':<6}{'hit':<6}{'时out':<7}{'时dead':<7}")
    for g in arm["games"]:
        st = g.get("stderr") or {}
        ij = st.get("inj_stats") or {}
        print(f"{g['token']:<6}{('P%d' % g['seat']) if g['seat'] is not None else '?':<6}"
              f"{('%.1f' % g['score']) if g['score'] is not None else '?':<7}"
              f"{g.get('hp_ours', '-'):<6}{g.get('hp_opp', '-'):<6}{g.get('hp_diff', '-'):<7}"
              f"{str(g.get('rounds')):<6}{str(g.get('verdict')):<10}"
              f"{int(g.get('illegal_judge') or 0):<5}"
              f"{int(ij.get('asks', 0)) if ij else '-':<6}"
              f"{int(ij.get('new', 0)) if ij else '-':<6}"
              f"{int(ij.get('hit', 0)) if ij else '-':<6}"
              f"{int(ij.get('timeout', 0)) if ij else '-':<7}"
              f"{int(ij.get('dead', 0)) if ij else '-':<7}")
    n = s["n_scored"]
    print(f"\n有效局 {s['n_valid']}/{s['n_games']}；计分局 n={n}；"
          f"{s['wins']}W-{s['draws']}D-{s['losses']}L = "
          f"{(100*s['winrate'] if s['winrate'] is not None else float('nan')):.1f}%  "
          f"Wilson95%CI [{100*s['winrate_ci'][0]:.1f}, {100*s['winrate_ci'][1]:.1f}]")
    hp = s["hp_diff"]
    if hp.get("n"):
        print(f"血量差 mean={hp['mean']:+.2f}  sd={hp['sd']:.2f}  SE={hp['se']:.2f}  "
              f"t95%CI [{hp['lo']:+.2f}, {hp['hi']:+.2f}]")
    for k, v in s["byd_side"].items():
        print(f"  @{k}: n={v['n']} {v['wins']}W-{v['losses']}L = {100*v['winrate']:.1f}% "
              f"CI[{100*v['ci'][0]:.1f},{100*v['ci'][1]:.1f}] "
              f"血量差 {v['hp_diff'].get('mean', float('nan')):+.2f}")
    print(f"判词: {s['verdicts']}  裁判侧 illegal={s['illegal_judge']}")
    if s.get("inj"):
        j = s["inj"]
        t = j["totals"]
        t["asks_total"] = j["asks_total"]
        print(f"注入有效性: stderr={j['games_with_stderr']}局  注入已启用={j['games_inj_enabled']}局  "
              f"路径 D:/ 形态={j['paths_ok']}局  有 asks={j['games_with_asks']}局  "
              f"停用标记={j['disabled_marks']}  失败行={j['fail_lines']}")
        if s.get("inj_cover") is not None:
            print(f"注入覆盖: asks/回合 = {100*s['inj_cover']:.1f}%"
                  + (f"  ⚠️ 覆盖不足的局: {s['inj_short']}" if s["inj_short"] else "  ✓ 每局都覆盖整局"))
        else:
            print("注入覆盖: —（本臂没开注入，asks 恒 0 是预期的）")
        print(f"注入统计: asks={j['asks_total']} new={t['new']} dup={t['dup']} "
              f"dup_hold={t['dup_hold']} hit={t['hit']} hit_new={t['hit_new']} "
              f"hit_sel={t['hit_sel']} illegal={t['illegal']} timeout={t['timeout']} "
              f"dead={t['dead']}")
        print(f"每决策日志: steps={j['steps']}（dup={j['steps_dup']} new={j['steps_new']} "
              f"hit={j['steps_hit']} hit_new={j['steps_hit_new']} hit_dup={j['steps_hit_dup']} "
              f"dup_hold={j['steps_dup_hold']}）⇒ "
              f"new/asks={100*j['steps_new']/max(1,j['steps']):.1f}%  "
              f"hit_new/new={100*j['steps_hit_new']/max(1,j['steps_new']):.1f}%  "
              f"hit/asks={100*j['steps_hit']/max(1,j['steps']):.1f}%")
        print(f"oracle 耗时(每局中位): avg={j['lat_ms']['avg_ms']}ms p50={j['lat_ms']['p50_ms']}ms "
              f"p95={j['lat_ms']['p95_ms']}ms  全体最大={j['lat_max']}ms")
        print(f"AI 侧 illegal_total(max over games)={j['illegal_ai']}")
    if s.get("behavior"):
        b = s["behavior"]
        print(f"行为画像: 我方 {b['decisions']} 决策 / {b['ops']} op "
              f"= {b['ops_per_decision']:.2f} op/决策  （出手决策占比 {100*b['act_rate']:.0f}%）")
        print(f"  我方 op 构成: {[(k, f'{v:.1f}%') for k, v in b['mix_pct'].items()]}")
        print(f"  对手 op 构成: {[(k, f'{v:.1f}%') for k, v in b['opp_mix_pct'].items()]}")
    if s.get("dec"):
        d = s["dec"]
        print(f"决策耗时: {d['games']}局/{d['decisions']}决策  "
              f"p50={d['p50_median']:.2f}s(中位) p95={d['p95_median']:.2f}s(中位, 最坏{d['p95_max']:.2f}) "
              f"max={d['max_median']:.2f}s(中位, 最坏{d['max_worst']:.2f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=LOGDIR")
    ap.add_argument("--stderr", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--ref", action="append", default=[], metavar="LABEL=LOGDIR",
                    help="参考读数（历史批次；不给 stderr）")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sd = {}
    for spec in args.stderr:
        k, v = spec.split("=", 1)
        sd[k] = REPO / v
    arms: dict[str, dict] = {}
    for spec in args.arm + args.ref:
        k, v = spec.split("=", 1)
        arms[k] = read_arm(REPO / v, sd.get(k))
    if not arms:
        print("没有给 --arm/--ref", file=sys.stderr)
        return 2

    summ = {k: summarize(v) for k, v in arms.items()}
    for k in arms:
        print_arm(k, arms[k], summ[k])

    pairs = {}
    for a_lab, b_lab in args.pair:
        if a_lab in arms and b_lab in arms:
            p = paired(arms[a_lab], arms[b_lab])
            pairs[f"{a_lab}-{b_lab}"] = p
            print(f"\n{'='*100}\n### 配对 {a_lab} − {b_lab}（同 token＝同图同座位）\n{'='*100}")
            print(f"{'token':<6}{'seat':<6}{'scoreA':<8}{'scoreB':<8}{'ΔhpA':<8}{'ΔhpB':<8}{'ΔΔhp':<8}")
            for r in p["rows"]:
                print(f"{r['token']:<6}{'P%d' % r['seat']:<6}{r['score_A']:<8.1f}"
                      f"{r['score_B']:<8.1f}{r['hp_A']:>+7}{r['hp_B']:>+8}{r['d_hp']:>+8.0f}")
            d = p["d_hp"]
            if d.get("n"):
                print(f"\n配对血量差 (A−B) mean={d['mean']:+.2f}  sd={d['sd']:.2f}  "
                      f"SE={d['se']:.2f}  t95%CI [{d['lo']:+.2f}, {d['hi']:+.2f}]  n={d['n']}")
            t = p["table"]
            print(f"胜负表: 两臂都赢 {t['both_win']} / 只 A 赢 {t['A_only']} / "
                  f"只 B 赢 {t['B_only']} / 都输 {t['both_lose']}")
            print(f"McNemar 精确双侧 p={p['mcnemar_p']:.4f}；"
                  f"胜率差 = {(p['delta_winrate'] or 0)*100:+.1f}pp（n={p['n_pairs']} 对）")
            print(f"逐 seed 得分差 (A−B，每 seed 两局合计，范围 −2..+2): {p['per_seed_delta']}")
            sh = shares(summ[a_lab], summ[b_lab])
            if sh:
                pairs[f"{a_lab}-{b_lab}"]["shares"] = sh
                print(f"\n§2.4 份额（操作化定义：候选通道 = {a_lab} 的注入）: "
                      f"G=1−B={100*sh['G']:.1f}pp  Δ=A−B={100*sh['delta']:+.1f}pp")
                print(f"  候选/先验份额 s = Δ/G = {100*(sh['s_point'] or 0):+.1f}%  "
                      f"（95% 区间 [{100*(sh['s_min'] or 0):+.1f}%, {100*(sh['s_max'] or 0):+.1f}%]）"
                      f" ⇒ 评估/搜索份额 = {100*(1-(sh['s_point'] or 0)):.1f}%")

    if args.json:
        outp = REPO / args.json
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"arms": summ, "pairs": pairs}, ensure_ascii=False,
                                   indent=2, default=str), encoding="utf-8")
        print(f"\n[json] 已写 {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
