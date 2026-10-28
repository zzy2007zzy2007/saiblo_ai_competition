#!/usr/bin/env python
"""把 A/B 两臂的逐局读数打成 **markdown 表行**（报告 `docs/inject_vs_runnerup_20261028.md` §3 用）。

为什么不直接抄 `analyze_inject_vs_runnerup.py` 的屏幕输出：报告里要的是**逐局**的
`asks/new/hit_new/hit_sel`，而屏幕汇总只有总数。这里按"报告表格的列顺序"逐局打印，
避免手工誊抄出错（本文件第一版就因为手抄 hit_sel 写错过一个数字）。

用法：
  python code/test_match/_inject_vs_ru_tables.py A B
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("_tmp_ladder", REPO / "_tmp_ladder.py")
_ladder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ladder)

NUM = re.compile(r"([a-z_0-9]+)=(-?[\d.]+)")


def stats_of(path: Path) -> dict:
    if not path.is_file():
        return {}
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "注入统计:" in ln or "注入累计:" in ln:
            out = {k: float(v) for k, v in NUM.findall(ln.split(":", 1)[1])}
    return out


def main() -> int:
    arms = sys.argv[1:] or ["A", "B"]
    for arm in arms:
        log_dir = REPO / f"match_results/ladder_logs/INJECT_VS_RU_{arm}"
        sd = REPO / f"match_results/inject_vs_ru/{arm}/stderr"
        print(f"\n### {arm} 臂逐局（`--stderr` 目录 = {sd.relative_to(REPO).as_posix()}）\n")
        print("| token | 我方座位 | 结果 | 我方 HP | 对手 HP | 血量差 | 回合 | asks | new | hit_new | hit_sel |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for lp in sorted(log_dir.glob("*.log")):
            tok = lp.stem
            m = re.match(r"(\d+)", tok)
            if not m:
                continue
            g = _ladder.read_game(lp, tok)
            seat = 0 if g.get("p0_label") == "az_bridge_ai" else 1
            st = stats_of(sd / f"ai{seat}_az_bridge_ai_seed{m.group(1)}.stderr.log")
            hp = g.get("hp") or (0, 0)
            oh = hp[seat]
            th = hp[1 - seat]
            ws = (g.get("winner_side") or "").lower()
            res = "**胜**" if ws == f"p{seat}" else ("**负**" if ws in ("p0", "p1") else "?")
            rows.append((int(m.group(1)), tok.endswith("r"), tok, seat, res, oh, th, oh - th,
                         g.get("rounds"), st))
        rows.sort(key=lambda r: (r[0], r[1]))
        for _s, _mir, tok, seat, res, oh, th, d, rounds, st in rows:
            print(f"| {tok} | P{seat} | {res} | {oh} | {th} | {d:+d} | {rounds} | "
                  f"{int(st.get('asks', 0))} | {int(st.get('new', 0))} | "
                  f"{int(st.get('hit_new', 0))} | {int(st.get('hit_sel', 0))} |")
        n = len(rows)
        wins = sum(1 for r in rows if r[4] == "**胜**")
        print(f"\n合计: {wins}W-{n - wins}L / {n}；"
              f"asks={sum(int(r[9].get('asks', 0)) for r in rows)} "
              f"new={sum(int(r[9].get('new', 0)) for r in rows)} "
              f"hit_new={sum(int(r[9].get('hit_new', 0)) for r in rows)} "
              f"hit_sel={sum(int(r[9].get('hit_sel', 0)) for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
