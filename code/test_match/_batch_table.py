#!/usr/bin/env python
"""把一批 `_tmp_ladder` 实验的日志算成**报告用的表**（胜率/血量差/按座位/每局墙钟/verdict·illegal 核查）。

2026-10-28（`docs/us_vs_champion_20261028.md`）：`analyze_inject_vs_runnerup.py` 已经能算胜率与
血量差，但它**不读每局墙钟**（`_run.log` 里的 `[tok] rc=.. N min`），也不核对"对手侧 stderr 是否留存"。
本脚本补这两块，并把逐局读数打成 markdown 表（直接贴进报告，避免手抄）。

2026-10-29（`docs/us_vs_champion_20261029.md`）：补上**报告要的两个区间**——
Wilson 95%CI（胜率，小样本不能拿 p±1.96·se 糊）与 **t 95%CI**（血量差，df=n−1），
总表/按座位拆分都给区间；区间函数与 `analyze_inject_vs_runnerup.py` **同一套实现**（同一读数口径）。

判据（与驱动里的三条断言同一口径）：
  * `verdict=engine`（不是 aborted/timeout/illegal_strict）；
  * 裁判侧 `illegal=0` 且我方 stderr 的 `illegal_total=0`；
  * 每局都能找到**对手侧** stderr（`.log` 或 `.log.gz`；冠军那份是 gzip）。
    ⚠️ 2026-10-29 起这条默认降级为**软判据**（`--strict-opp-stderr` 可升回硬判据）：
    历史局（如 2026-10-28 那批的 7/7r/8/8r）的对手 stderr 已被清掉，但它们的
    `RESULT`/`illegal` 是完整的 —— 拿"文件没了"去否定"局跑完了"是错的；失去的只是
    "冠军有没有自己报 desync"这一条旁证。表里仍然逐局列出 `-`，不漏报。

墙钟来源（按优先级）：
  1. `_run.log` 里 `[<tok>] rc=<rc>  <min> min` —— 驱动亲口报的（最准）；
  2. 缺失时（例如 ladder 被中途掐掉）用 `<tok>.log` 的**创建时间→最后修改时间**近似（表里标 `~`）。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/_batch_table.py \
      --tag US_VS_CHAMPION_BASE --stderr match_results/champion_e1/stderr \
      --opp-label champion --seeds 7 8 9 10 11 12 13 14 \
      --markdown-out match_results/champion_e1/table.md --json match_results/champion_e1/summary.json
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

RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) winner_side=(\S+) verdict=(\S+) "
    r"engine_winner=(\S+) rounds=(\d+) terminal=(\S+) base_hp=(\d+),(\d+) "
    r"coins=(\d+),(\d+)(?: illegal=(\d+))?")
WALL_RE = re.compile(r"^\[(\d+r?)\] rc=(-?\d+)\s+([\d.]+) min", re.M)

# t_{0.975, df}（双侧 95%）—— 小样本必须用 t，不能用 1.96；与 analyze_inject_vs_runnerup.py 同一张表
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
    """均值 + t 95%CI（df=n−1）。n<2 时给 None（不硬编区间）。"""
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


def _per_game_wall(tag_dir: Path) -> dict:
    """从 `_run.log` 读每局墙钟（分钟）；后出现的覆盖先出现的（重试取最后一次）。"""
    out: dict[str, float] = {}
    p = tag_dir / "_run.log"
    if p.is_file():
        txt = p.read_text(encoding="utf-8", errors="replace")
        for m in WALL_RE.finditer(txt):
            out[m.group(1)] = float(m.group(3))
    return out


def _wall_from_mtime(log: Path) -> float:
    st = log.stat()
    return max(0.0, (st.st_mtime - st.st_ctime) / 60.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stderr", default="", help="归档后的 stderr 目录（含 .gz）")
    ap.add_argument("--opp-label", default="", help="对手侧标签（champion/runnerup；留空 = 不检查）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12, 13, 14])
    ap.add_argument("--json", default="")
    ap.add_argument("--markdown-out", default="", help="把逐局表 + 统计块写成 markdown")
    ap.add_argument("--strict-opp-stderr", action="store_true",
                    help="对手侧 stderr 缺失算硬失败（默认只警告；见文件头 2026-10-29 那条）")
    args = ap.parse_args()

    tag_dir = REPO / "match_results" / "ladder_logs" / args.tag
    if not tag_dir.is_dir():
        print(f"!! 没有日志目录 {tag_dir}", file=sys.stderr)
        return 2
    sd = (REPO / args.stderr) if args.stderr else None
    walls = _per_game_wall(tag_dir)

    tokens: list[str] = []
    for sd_ in args.seeds:
        tokens.extend([str(sd_), f"{sd_}r"])

    rows: list[dict] = []
    fails: list[str] = []
    warns: list[str] = []
    for tok in tokens:
        log = tag_dir / f"{tok}.log"
        rec: dict = {"token": tok, "seed": int(re.sub(r"r$", "", tok)),
                     "mirror": tok.endswith("r"), "exists": log.is_file()}
        if not log.is_file():
            rec["verdict"] = None
            rows.append(rec)
            fails.append(f"{tok}: 没有日志")
            continue
        txt = log.read_text(encoding="utf-8", errors="replace")
        m = RESULT_RE.search(txt)          # 取**第一条**（重试时日志会被覆盖，正常只有一条）
        if not m:
            rec["verdict"] = None
            rows.append(rec)
            fails.append(f"{tok}: 没有 RESULT 行")
            continue
        (rec["seed"], rec["p0"], rec["p1"], rec["winner"], rec["winner_side"],
         rec["verdict"], rec["engine_winner"], rec["rounds"], rec["terminal"]) = (
            int(m.group(1)), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6),
            m.group(7), int(m.group(8)), m.group(9))
        hp0, hp1 = int(m.group(10)), int(m.group(11))
        rec["illegal_judge"] = int(m.group(14) or 0)
        rec["seat"] = 0 if rec["p0"] == OUR_LABEL else (1 if rec["p1"] == OUR_LABEL else None)
        if rec["seat"] is None:
            fails.append(f"{tok}: 两侧标签都不是 {OUR_LABEL}")
            rows.append(rec)
            continue
        rec["hp_ours"], rec["hp_opp"] = (hp0, hp1) if rec["seat"] == 0 else (hp1, hp0)
        rec["d_hp"] = rec["hp_ours"] - rec["hp_opp"]
        rec["score"] = 1.0 if rec["winner_side"] == f"p{rec['seat']}" else 0.0
        if tok in walls:
            rec["wall_min"], rec["wall_src"] = walls[tok], "runlog"
        else:
            rec["wall_min"], rec["wall_src"] = round(_wall_from_mtime(log), 1), "mtime~"
        # 我方 stderr（非法自检 + 决策耗时）
        rec["our_illegal_total"] = None
        rec["dec_p50"] = rec["dec_p95"] = rec["dec_max"] = None
        if sd is not None:
            f = sd / f"ai{rec['seat']}_{OUR_LABEL}_seed{rec['seed']}.stderr.log"
            if f.is_file():
                st = f.read_text(encoding="utf-8", errors="replace")
                mm = re.findall(r"illegal_total=(\d+)", st)
                rec["our_illegal_total"] = max((int(x) for x in mm), default=0)
                mm = re.findall(r"p50=([\d.]+)s p95=([\d.]+)s max=([\d.]+)s", st)
                if mm:
                    p50, p95, mx = mm[-1]
                    rec["dec_p50"], rec["dec_p95"], rec["dec_max"] = (
                        float(p50), float(p95), float(mx))
        # 对手侧 stderr 留存（冠军那份有 .gz）
        rec["opp_stderr_mb"] = None
        if sd is not None and args.opp_label:
            cands = list(sd.glob(f"ai{1 - rec['seat']}_*{args.opp_label}*_seed{rec['seed']}*.log.gz")) + \
                    list(sd.glob(f"ai{1 - rec['seat']}_*{args.opp_label}*_seed{rec['seed']}*.log"))
            if cands:
                rec["opp_stderr_mb"] = round(sum(c.stat().st_size for c in cands) / (1024 * 1024), 1)
        rows.append(rec)
        # ── 硬判据 ──
        if rec["verdict"] != "engine":
            fails.append(f"{tok}: verdict={rec['verdict']}")
        if rec["illegal_judge"] != 0:
            fails.append(f"{tok}: 裁判侧 illegal={rec['illegal_judge']}")
        if rec["our_illegal_total"] not in (0, None):
            fails.append(f"{tok}: 我方 illegal_total={rec['our_illegal_total']}")
        # ── 软判据 ──
        if sd is not None and args.opp_label and rec["opp_stderr_mb"] is None:
            msg = f"{tok}: 找不到对手侧 stderr（局本身有效，只少了'对手没报 desync'的旁证）"
            (fails if args.strict_opp_stderr else warns).append(msg)

    ok = [r for r in rows if r.get("score") is not None and r.get("verdict") == "engine"]
    n = len(ok)
    pts = sum(r["score"] for r in ok)
    d = [r["d_hp"] for r in ok]
    rounds = [r["rounds"] for r in ok]
    wl = [r["wall_min"] for r in ok]

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"\n### {args.tag}：逐局读数（有效局 {n}/{len(rows)}）\n")
    out("| token | 我方座位 | verdict | 回合 | hp(我,敌) | Δhp | 裁判ill | 我ill | 墙钟(min) | 对手stderr | "
        "决策 p50/p95/max (s) |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r.get("verdict"):
            out(f"| {r['token']} | ? | **缺失** | - | - | - | - | - | - | - | - |")
            continue
        seat = "P0" if r["seat"] == 0 else "P1"
        wsrc = "" if r.get("wall_src") == "runlog" else " ~"
        dec = (f"{r['dec_p50']:.2f}/{r['dec_p95']:.2f}/{r['dec_max']:.2f}"
               if r.get("dec_p50") is not None else "-")
        out(f"| {r['token']} | {seat} | {r['verdict']} | {r['rounds']} | "
            f"{r['hp_ours']},{r['hp_opp']} | {r['d_hp']:+d} | {r['illegal_judge']} | "
            f"{r['our_illegal_total'] if r['our_illegal_total'] is not None else '-'} | "
            f"{r['wall_min']:.1f}{wsrc} | "
            f"{r['opp_stderr_mb'] if r['opp_stderr_mb'] is not None else '-'}MB | {dec} |")

    stats: dict = {}
    if ok:
        lo, hi = wilson(pts, n)
        hci = mean_ci(d)
        stats.update({"n": n, "wins": int(pts), "losses": n - int(pts),
                      "winrate": pts / n, "winrate_ci": [lo, hi], "d_hp": hci,
                      "rounds": {"min": min(rounds), "med": statistics.median(rounds),
                                 "max": max(rounds)},
                      "wall": {"min": min(wl), "med": statistics.median(wl), "max": max(wl),
                               "sum": sum(wl)}})
        out("")
        out(f"- 胜率：**{int(pts)}W-{n - int(pts)}L = {100.0 * pts / n:.1f}%**"
            f"（n={n}，Wilson 95%CI [{100 * lo:.1f}%, {100 * hi:.1f}%]）")
        if hci["n"] > 1:
            out(f"- 血量差（我方 − 对手，基准 50/50）：均值 **{hci['mean']:+.2f}**"
                f"（sd={hci['sd']:.2f}，SE={hci['se']:.2f}，t 95%CI [{hci['lo']:+.2f}, {hci['hi']:+.2f}]）")
        else:
            out(f"- 血量差：均值 {hci['mean']:+.2f}（n=1，不给区间）")
        seat_stats: dict = {}
        for seat in (0, 1):
            sub = [r for r in ok if r["seat"] == seat]
            if not sub:
                continue
            p = sum(r["score"] for r in sub)
            l2, h2 = wilson(p, len(sub))
            sc = mean_ci([r["d_hp"] for r in sub])
            seat_stats[f"p{seat}"] = {"n": len(sub), "wins": int(p),
                                      "winrate": p / len(sub), "winrate_ci": [l2, h2],
                                      "d_hp": sc}
            extra = (f"，血量差 {sc['mean']:+.2f}（t 95%CI [{sc['lo']:+.2f}, {sc['hi']:+.2f}]）"
                     if sc.get("lo") is not None else f"，血量差 {sc['mean']:+.2f}")
            out(f"  - @P{seat}（{'我们执先手' if seat == 0 else '我们执后手'}）："
                f"{int(p)}W-{len(sub) - int(p)}L = {100.0 * p / len(sub):.1f}%"
                f"（n={len(sub)}，Wilson 95%CI [{100 * l2:.1f}%, {100 * h2:.1f}%]{extra}）")
        stats["by_seat"] = seat_stats
        out(f"- 回合数：min/med/max = {min(rounds)}/{int(statistics.median(rounds))}/{max(rounds)}"
            f"（512 = 打满回合上限，胜负由引擎 5 级级联判）")
        out(f"- 每局墙钟：min/med/max = {min(wl):.1f}/{statistics.median(wl):.1f}/{max(wl):.1f} min"
            f"（合计 {sum(wl):.1f} min，jobs 并发 ⇒ 墙钟 ≠ 之和）")
        _p50 = [r["dec_p50"] for r in ok if r.get("dec_p50") is not None]
        _p95 = [r["dec_p95"] for r in ok if r.get("dec_p95") is not None]
        _mx = [r["dec_max"] for r in ok if r.get("dec_max") is not None]
        if _p50:
            out(f"- 决策耗时（每局最后一行的 p50/p95/max，取中位）："
                f"p50={statistics.median(_p50):.2f}s p95={statistics.median(_p95):.2f}s "
                f"max={max(_mx):.2f}s")
        else:
            out("- 决策耗时：没读到（没给 --stderr 或归档里没有我方 stderr）")
        # 逐局胜负点（方便报告正文引用）
        seq = " ".join("%s%s" % (r["token"], "胜" if r["score"] == 1.0 else "负") for r in ok)
        out("- 逐局：%s" % seq)
    out("")
    if not fails and not warns:
        out("**判据核查**：✅ 全部通过（verdict=engine / illegal=0 / 对手 stderr 留存）")
    else:
        out(f"**判据核查**：硬判据 {'✅ 全过' if not fails else f'❌ {len(fails)} 条不通过'}"
            + (f"；软判据 {len(warns)} 条警告" if warns else ""))
    for f in fails:
        out(f"  - ❌ {f}")
    for w in warns:
        out(f"  - ⚠️ {w}")

    if args.markdown_out:
        outp = REPO / args.markdown_out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        print(f"\n[md] {outp}")

    if args.json:
        outp = REPO / args.json
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"tag": args.tag, "stats": stats, "rows": rows,
                                    "fails": fails, "warns": warns},
                                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[json] {outp}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
