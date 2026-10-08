"""Phase 0 报表：把若干 tag 的子博弈读数汇到一张表（预注册 `docs/prereg_20261017_phase0_nolightning.md` §2）。

每个 tag 输出：
  * 配对胜率（读该 tag 的 `_analysis.txt`，若没有则现算裸胜率）；
  * **我方 / 对手** 的 `建塔 / 降级 / 升级 / 闪电` 每局均值；
  * 我方"有操作的回合数"；
  * 对局回合数、单局墙钟（从 `training_history/runs/*_<tag>/output.log` 的 `rc=0  X min` 行取均值）；
  * 无效局数。

用法: python code/test_match/phase0_report.py <tag> [<tag> ...]
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RESULT = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+)")
OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OPNAME = re.compile(r"OperationType\.(\w+): (\d+)>")
PAIRED = re.compile(r"配对胜率 p̂ = ([\d.]+)")
WALL = re.compile(r"rc=0\s+([\d.]+) min")


def one(tag: str) -> dict:
    d: dict = {"tag": tag}
    logdir = REPO / "match_results" / "ladder_logs" / tag
    files = sorted(logdir.glob("*.log")) if logdir.exists() else []
    ours = {"BUILD_TOWER": 0, "DOWNGRADE_TOWER": 0, "UPGRADE_TOWER": 0, "USE_LIGHTNING_STORM": 0}
    theirs = dict(ours)
    n = rounds = act_rounds = 0
    wins = inval = 0
    tot_rounds = []
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RESULT.search(txt)
        if not m:
            continue
        n += 1
        if m.group(4) == "INVALID":
            inval += 1
        elif m.group(4) == "az_bridge_ai":
            wins += 1
        # 2026-10-18: decide OUR side from the token convention (no r -> AI0 is p0,
        # r -> AI0 is p1), matching analyze_paired. Taking "the first side labelled
        # az_bridge_ai" silently resolves to p0 for self-mirror tags.
        us = (0 if m.group(2) == "az_bridge_ai" else 1) if m.group(2) != m.group(3) \
            else (1 if f.stem.endswith("r") else 0)
        # 2026-10-18: take rounds ONLY from the RESULT line. Searching the whole file matches
        # the header's max_rounds=512 first, which silently wrote 512 for every game.
        for _line in txt.splitlines():
            _s = _line.strip()
            if _s.startswith("RESULT "):
                _rr = re.search(r"rounds=(\d+)", _s)
                if _rr:
                    tot_rounds.append(int(_rr.group(1)))
                break
        for mm in OPLINE.finditer(txt):
            side = int(mm.group(2)[1])
            names = [nm for nm, _ in OPNAME.findall(mm.group(5))]
            if side == us:
                # ⚠️ 只统计**真的有操作**的回合（空列表 = HOLD，不算"有操作"）
                if names:
                    act_rounds += 1
                for nm in names:
                    if nm in ours:
                        ours[nm] += 1
            else:
                for nm in names:
                    if nm in theirs:
                        theirs[nm] += 1
    d.update(n=n, inval=inval, wins=wins, act_rounds=act_rounds,
             ours=ours, theirs=theirs,
             rounds=statistics.mean(tot_rounds) if tot_rounds else float("nan"))
    ana = logdir / "_analysis.txt"
    if ana.exists():
        pm = PAIRED.search(ana.read_text(encoding="utf-8", errors="replace"))
        d["phat"] = float(pm.group(1)) if pm else float("nan")
    else:
        d["phat"] = float("nan")
    # 单局墙钟
    walls = []
    for rd in sorted((REPO / "training_history" / "runs").glob(f"*_{tag}")):
        p = rd / "output.log"
        if p.exists():
            walls += [float(x) for x in WALL.findall(p.read_text(encoding="utf-8", errors="replace"))]
    d["wall"] = statistics.mean(walls) if walls else float("nan")
    return d


def main() -> int:
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        return 2
    print(f"{'arm':22s} {'局':>4s} {'p̂':>7s} {'我方/局: 建/降/升/闪':>22s} {'有操作回合':>10s} "
          f"{'对手/局: 建/降/升/闪':>22s} {'回合':>6s} {'min/局':>7s} {'无效':>4s}")
    for t in tags:
        d = one(t)
        n = max(d["n"], 1)
        o, h = d["ours"], d["theirs"]
        print(f"{t:22s} {d['n']:4d} {d['phat']:7.4f} "
              f"{o['BUILD_TOWER']/n:10.1f}/{o['DOWNGRADE_TOWER']/n:5.1f}/{o['UPGRADE_TOWER']/n:4.1f}/{o['USE_LIGHTNING_STORM']/n:5.1f} "
              f"{d['act_rounds']/n:10.1f} "
              f"{h['BUILD_TOWER']/n:10.1f}/{h['DOWNGRADE_TOWER']/n:5.1f}/{h['UPGRADE_TOWER']/n:4.1f}/{h['USE_LIGHTNING_STORM']/n:5.1f} "
              f"{d['rounds']:6.0f} {d['wall']:7.2f} {d['inval']:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
