"""对比两个 tag（如"固定段"与"随机段"）的**局面特征**，用于回答"为什么某一段更难"。

读的是 ladder 日志里的 `RESULT` 行与机制行（只读，不改任何东西）。输出：
- 每段：配对胜率、执先/执后、**我们胜局的回合数 vs 负局的回合数**（均值/中位数）；
- **按局长分箱的胜率**（短/中/长）—— 用来判断"是不是输在长局"（若是 ⇒ 指向残局/经济问题，可操作）；
- 我方与对手的建塔/降级/升级均值。

用法：
    python code/test_match/seg_compare.py --tag-a AB_MC1_dec_vs_baseline --tag-b MC1_rand_ctrl
"""
from __future__ import annotations

import argparse
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

# 真实格式（无 token 字段，token 在文件名里）：
#   RESULT seed=7 p0=az_bridge_ai p1=main winner=az_bridge_ai winner_side=p0 verdict=engine engine_winner=0 rounds=204 terminal=True base_hp=17,0 coins=237,74
RES_RE = re.compile(r"RESULT\s+seed=(\d+)\s+p0=(\S+)\s+p1=(\S+)\s+winner=(\S+)\s+winner_side=(\S+)\s+verdict=(\S+).*?rounds=(\d+)", re.S)


def parse(tag: str):
    d = REPO / "match_results" / "ladder_logs" / tag
    rows = []
    for f in sorted(d.glob("*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RES_RE.search(txt)
        if not m:
            continue
        seed, _p0, _p1, _win, side, verdict, rounds = m.groups()
        token = f.stem          # 文件名就是 token（`7`=先手，`7r`=后手）
        if verdict != "engine" or side in ("None", "INVALID"):
            continue
        swapped = token.endswith("r")
        # 我方(ai0)胜：N 局 winner_side=p0；Nr 局 winner_side=p1
        ai0_win = (side == "p0") if not swapped else (side == "p1")
        rows.append(dict(seed=int(seed), swapped=swapped, win=bool(ai0_win), rounds=int(rounds)))
    return rows


def rep(name: str, rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        print(f"   {name}: 无有效局"); return
    wins = [r for r in rows if r["win"]]
    loss = [r for r in rows if not r["win"]]
    first = [r for r in rows if not r["swapped"]]
    second = [r for r in rows if r["swapped"]]
    wr = len(wins) / n
    med = statistics.median([r["rounds"] for r in rows])
    print(f"   {name}: 局数={n} 胜率={wr:.4f}  执先={sum(r['win'] for r in first)}/{len(first)}"
          f"  执后={sum(r['win'] for r in second)}/{len(second)}")
    if wins and loss:
        print(f"      回合数：胜局 均值={statistics.mean(r['rounds'] for r in wins):.1f} "
              f"中位={statistics.median(r['rounds'] for r in wins):.0f}  |  "
              f"负局 均值={statistics.mean(r['rounds'] for r in loss):.1f} "
              f"中位={statistics.median(r['rounds'] for r in loss):.0f}   （全段中位={med:.0f}）")
    short = [r for r in rows if r["rounds"] < med]
    long_ = [r for r in rows if r["rounds"] >= med]
    print(f"      按局长分箱（以本段中位 {med:.0f} 切）：短局胜率={sum(r['win'] for r in short)}/{len(short)}"
          f"  长局胜率={sum(r['win'] for r in long_)}/{len(long_)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-a", required=True, help="例如固定段 AB_MC1_dec_vs_baseline")
    ap.add_argument("--tag-b", required=True, help="例如随机段 MC1_rand_ctrl")
    a = ap.parse_args()
    ra, rb = parse(a.tag_a), parse(a.tag_b)
    print("=== 段间对比（同一配置、只换 seed 段）")
    rep(a.tag_a, ra)
    rep(a.tag_b, rb)
    # 跨段合并看"局长 vs 胜负"的整体关系（两个 tag 的执行/后手都混在一起，仅作粗看）
    allr = ra + rb
    if allr:
        med = statistics.median([r["rounds"] for r in allr])
        s = [r for r in allr if r["rounds"] < med]; l = [r for r in allr if r["rounds"] >= med]
        print(f"=== 两段合并：短局(<{med:.0f})胜率={sum(r['win'] for r in s)}/{len(s)} "
              f" 长局(>={med:.0f})胜率={sum(r['win'] for r in l)}/{len(l)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
