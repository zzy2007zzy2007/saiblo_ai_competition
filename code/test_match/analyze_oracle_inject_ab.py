"""亚军候选注入 A/B 的读数汇总（2026-10-28；报告 `docs/oracle_inject_ab_20261028.md`）。

读 `ab_search_modes.py --paired-sides` 的产出目录，输出三块读数：

1. **每局明细**：A（注入臂）坐哪边、输赢、`base_hp` 差、回合数、判词、`illegal=`；
2. **汇总**：A 的胜率（整体 / `@P0` / `@P1`）、平均血量差、平均回合数、有效性（`verdict=engine` 占比）；
3. **注入侧**：从 A 臂 AI 的 stderr 汇总 oracle 的 `asks/ok/inject/new/dup/hit/hit_rate/illegal/timeout`
   与耗时（avg/p50/p95/max），外加非法自检（`illegal-summary`）与过滤闸门计数。

⚠️ 有效性口径照抄 `analyze_paired.py` §4：**只有 `verdict=engine` 算有效局**，
其余（`timeout/aborted/illegal_strict/hp_tiebreak/无 RESULT`）一律排除在主读数之外、单独列出。

用法：
  python code/test_match/analyze_oracle_inject_ab.py \
      --out-dir match_results/oracle_inject_ab/ab16 --stderr-dir match_results/oracle_inject_ab/ab16/stderr
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RE_RESULT = re.compile(
    r"RESULT seed=(?P<seed>\d+) .*?winner=(?P<winner>\S+) winner_side=(?P<ws>\S+) "
    r"verdict=(?P<verdict>\S+).*?rounds=(?P<rounds>\d+) terminal=(?P<term>\S+) "
    r"base_hp=(?P<hp0>-?\d+),(?P<hp1>-?\d+) coins=(?P<c0>-?\d+),(?P<c1>-?\d+)"
    r"(?: illegal=(?P<illegal>\d+))?")
RE_ORACLE = re.compile(r"注入(?:统计|累计): (?P<body>.+)$")
RE_ILLSUM = re.compile(r"illegal-summary .*?illegal_total=(?P<tot>\d+)")
RE_GATE = re.compile(r"过滤闸门累计: checks=(?P<chk>\d+) dropped=(?P<drop>\d+)")


def _num(txt: str, key: str, cast=float, default=None):
    m = re.search(rf"\b{key}=([-\d.]+)", txt)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except ValueError:
        return default


def parse_games(out_dir: Path) -> list[dict]:
    games = []
    for log in sorted(out_dir.glob("seed*_p*.log")):
        m = re.search(r"seed(\d+)_p(\d)\.log$", log.name)
        if not m:
            continue
        seed, side = int(m.group(1)), int(m.group(2))
        txt = log.read_text(encoding="utf-8", errors="replace")
        r = RE_RESULT.search(txt)
        rec = {"seed": seed, "a_side": side, "log": log.name,
               "result": bool(r), "verdict": None, "a_score": None, "hp_a": None, "hp_b": None,
               "rounds": None, "illegal": None, "winner_side": None}
        if r:
            ws = r.group("ws")
            ws_i = int(ws[1]) if ws in ("p0", "p1") else None
            hp = [int(r.group("hp0")), int(r.group("hp1"))]
            rec.update(verdict=r.group("verdict"), rounds=int(r.group("rounds")),
                       winner_side=ws, illegal=(int(r.group("illegal"))
                                                if r.group("illegal") is not None else None),
                       hp_a=hp[side], hp_b=hp[1 - side],
                       a_score=(0.5 if ws_i is None else (1.0 if ws_i == side else 0.0)))
        games.append(rec)
    return games


def parse_ai_stderr(stderr_dir: Path) -> dict:
    """A 臂（注入臂）的注入统计。**A 的文件名满足 `ai{s}_..._seed{N}_p{s}`**（两个数字相同）。"""
    tot = {"games": 0, "asks": 0, "ok": 0, "inject": 0, "new": 0, "dup": 0, "hit": 0,
           "illegal": 0, "timeout": 0, "dead": 0, "checked": 0,
           "illegal_sent": 0, "gate_checks": 0, "gate_dropped": 0,
           "files": [], "per_game": []}
    for f in sorted(stderr_dir.glob("ai?_*_p?.stderr.log")):
        m = re.search(r"ai(\d)_.*_seed(\d+)_p(\d)\.stderr\.log$", f.name)
        if not m or m.group(1) != m.group(3):     # 只要 A 臂（座位 == flat_side）
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        orc = RE_ORACLE.findall(txt)
        rec = {"file": f.name, "seed": int(m.group(2)), "side": int(m.group(3))}
        if orc:
            body = orc[-1]
            for k in ("asks", "ok", "inject", "new", "dup", "hit", "illegal", "timeout", "dead",
                      "checked", "avg_ms", "p50_ms", "p95_ms", "max_ms"):
                rec[k] = _num(body, k, int, None)
            hr = re.search(r"hit_rate=([\d.]+)%", body)
            rec["hit_rate"] = float(hr.group(1)) if hr else None
            for k in ("asks", "ok", "inject", "new", "dup", "hit", "illegal", "timeout",
                      "dead", "checked"):
                if rec.get(k):
                    tot[k] += int(rec[k])
            tot["games"] += 1
            tot["files"].append(f.name)
        ill = RE_ILLSUM.findall(txt)
        rec["illegal_sent"] = max((int(x) for x in ill), default=0)
        tot["illegal_sent"] += rec["illegal_sent"]
        g = RE_GATE.findall(txt)
        if g:
            rec["gate_checks"] = int(g[-1][0])
            rec["gate_dropped"] = int(g[-1][1])
            tot["gate_checks"] += rec["gate_checks"]
            tot["gate_dropped"] += rec["gate_dropped"]
        rec["stopped"] = "停用注入" in txt
        tot["per_game"].append(rec)
    tot["hit_rate"] = (100.0 * tot["hit"] / tot["new"]) if tot["new"] else 0.0
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="match_results/oracle_inject_ab/ab16")
    ap.add_argument("--stderr-dir", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO / out
    st = Path(args.stderr_dir) if args.stderr_dir else out / "stderr"
    if not st.is_absolute():
        st = REPO / st

    games = parse_games(out)
    if not games:
        print(f"!! {out} 里没有 seed*_p*.log")
        return 1

    print("=" * 108)
    print(f"每局明细（A = 注入臂）：{out}")
    print(f"{'seed':>6} {'A座位':>5} {'结果':>4} {'A分':>4} {'hp_A':>5} {'hp_B':>5} {'血量差':>6} "
          f"{'回合':>5} {'判词':>14} {'illegal':>7}")
    valid = []
    for g in sorted(games, key=lambda x: (x["seed"], x["a_side"])):
        if g["result"] and g["verdict"] == "engine":
            valid.append(g)
        tag = ("胜" if g["a_score"] == 1.0 else
               ("负" if g["a_score"] == 0.0 else ("平" if g["a_score"] == 0.5 else "?")))
        diff = (g["hp_a"] - g["hp_b"]) if (g["hp_a"] is not None) else None
        score_txt = "--" if g["a_score"] is None else f"{g['a_score']:.1f}"
        ha = "--" if g["hp_a"] is None else g["hp_a"]
        hb = "--" if g["hp_b"] is None else g["hp_b"]
        dt = "--" if diff is None else f"{diff:+d}"
        rd = "--" if g["rounds"] is None else g["rounds"]
        il = "--" if g["illegal"] is None else g["illegal"]
        print(f"{g['seed']:>6} {'P' + str(g['a_side']):>5} {tag:>4} {score_txt:>4} "
              f"{ha:>5} {hb:>5} {dt:>6} {rd:>5} {str(g['verdict']):>14} {il:>7}")

    def stat(groups: list[dict], label: str) -> None:
        n = len(groups)
        if not n:
            print(f"  {label}: n=0")
            return
        w = sum(1 for g in groups if g["a_score"] == 1.0)
        d = sum(1 for g in groups if g["a_score"] == 0.5)
        l = sum(1 for g in groups if g["a_score"] == 0.0)
        rate = 100.0 * sum(g["a_score"] for g in groups) / n
        se = (rate * (100 - rate) / n) ** 0.5 if n else 0.0
        diffs = [g["hp_a"] - g["hp_b"] for g in groups if g["hp_a"] is not None]
        rds = [g["rounds"] for g in groups if g["rounds"] is not None]
        print(f"  {label:<10} n={n:<3} {w}W/{d}D/{l}L  胜率={rate:5.1f}%  ±{se:.1f}%  "
              f"平均血量差={sum(diffs) / len(diffs) if diffs else float('nan'):+.2f}  "
              f"平均回合={sum(rds) / len(rds) if rds else float('nan'):.0f}")

    print()
    print("=" * 108)
    print("汇总（主口径 = 只有 verdict=engine 算有效局，与 analyze_paired.py §4 一致）")
    stat(valid, "全部")
    stat([g for g in valid if g["a_side"] == 0], "@P0")
    stat([g for g in valid if g["a_side"] == 1], "@P1")
    bad = [g for g in games if g not in valid]
    print(f"  无效局: {len(bad)}/{len(games)}"
          + (f" -> {[(g['seed'], 'P' + str(g['a_side']), g['verdict']) for g in bad]}" if bad else ""))

    tot_ill = [g["illegal"] for g in games if g["illegal"] is not None]
    print(f"  RESULT 行 illegal= 合计: {sum(tot_ill)}（有该字段的 {len(tot_ill)} 局）"
          f"  非零: {[(g['seed'], g['a_side'], g['illegal']) for g in games if g['illegal']]}")

    print()
    print("=" * 108)
    print(f"注入侧读数（A 臂 AI stderr：{st}）")
    o = parse_ai_stderr(st)
    if not o["games"]:
        print("  !! 没找到 A 臂 stderr（检查 MVS_STDERR_SUFFIX=1 与 --stderr-dir）")
    else:
        print(f"  覆盖 {o['games']} 局  文件: {', '.join(o['files'])}")
        print(f"  asks={o['asks']} ok={o['ok']} inject={o['inject']} new={o['new']} dup={o['dup']} "
              f"hit={o['hit']} hit_rate={o['hit_rate']:.1f}%（分母 new）  "
              f"注入候选被判非法={o['illegal']}  checked(逐条复核 op)={o['checked']}")
        print(f"  失败: timeout={o['timeout']} dead={o['dead']}"
              f"   实际发出的非法 op（illegal-summary 合计）={o['illegal_sent']}")
        print(f"  过滤闸门: checks={o['gate_checks']} dropped={o['gate_dropped']}（dropped>0 = 拦下半块）")
        mss = [r for r in o["per_game"] if r.get("max_ms") is not None]
        if mss:
            print(f"  oracle 单次耗时: avg_ms(每局) = "
                  f"{[r.get('avg_ms') for r in mss]}")
            print(f"                  p50_ms = {[r.get('p50_ms') for r in mss]}  "
                  f"max_ms = {[r.get('max_ms') for r in mss]}")
            allmax = max(r["max_ms"] for r in mss)
            print(f"                  全体最大 = {allmax} ms（超时阈值 AZAI_ORACLE_TIMEOUT 默认 2500 ms）")
        stp = [r["file"] for r in o["per_game"] if r.get("stopped")]
        print(f"  中途停用注入的局: {len(stp)}" + (f" -> {stp}" if stp else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
