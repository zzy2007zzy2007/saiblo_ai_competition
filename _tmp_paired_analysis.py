"""Paired analysis of the two kgeo_gap svs runs.

Run 1 (unflipped): seeds 0-63, A is P0 when seed is even.
Run 2 (flipped):   seeds 0-63, A is P1 when seed is even.
So each seed contributes one pair = (A-as-P0 game, A-as-P1 game).  Pair score in
{0,1,2}.  Under the null (A==B) the #(2-0) and #(0-2) pairs are exchangeable, so
a McNemar exact test on those discordant pairs is the clean statistic — it
cancels the P0/P1 asymmetry entirely.
"""
import re
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent
R1 = REPO / "training_history/runs/20260912_214250_svs_kgeo_gap_vs_term/output.log"
R2 = REPO / "training_history/runs/20260913_070523_svs_kgeo_flip_paired/output.log"

PAT = re.compile(r"seed=\s*(\d+)\s+(WIN|DRAW|LOSS)\s+\(A=P(\d)\)")


def load(p):
    out = {}
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = PAT.search(ln)
        if m:
            out[int(m.group(1))] = ({"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}[m.group(2)],
                                    int(m.group(3)))
    return out


a = load(R1)
b = load(R2)
seeds = sorted(set(a) & set(b))
print(f"unflipped games: {len(a)}   flipped games: {len(b)}   paired seeds: {len(seeds)}")


def side_stats(d, name):
    p0 = [v for v, s in d.values() if s == 0]
    p1 = [v for v, s in d.values() if s == 1]
    print(f"  {name:10s} A=P0: {sum(p0):.0f}W/{len(p0)-sum(p0):.0f}L ({sum(p0)/max(len(p0),1):.1%})   "
          f"A=P1: {sum(p1):.0f}W/{len(p1)-sum(p1):.0f}L ({sum(p1)/max(len(p1),1):.1%})")


print("\n[unpaired]")
side_stats(a, "run1")
side_stats(b, "run2")
allv = [v for v, _ in a.values()] + [v for v, _ in b.values()]
n = len(allv)
w = sum(allv)
print(f"  combined: {w:.0f}W/{n-w:.0f}L = {w/n:.1%}  (n={n})")

print("\n[paired] 每对 2 局（A 执先/后各一）")
two = zero = one = 0
for s in seeds:
    t = a[s][0] + b[s][0]
    if t == 2.0:
        two += 1
    elif t == 0.0:
        zero += 1
    else:
        one += 1
print(f"  2-0 (A 双胜): {two}   1-1 (平): {one}   0-2 (B 双胜): {zero}")
disc = two + zero
if disc:
    # exact two-sided McNemar = binomial(disc, 0.5) at min(two,zero)
    k = min(two, zero)
    p = sum(comb(disc, i) for i in range(k + 1)) / 2 ** disc * 2
    print(f"  不一致对 {disc}，McNemar 精确双侧 p = {min(p,1.0):.4f}")
    print(f"  A 的不一致对胜率 = {two}/{disc} = {two/disc:.1%}")
else:
    print("  无不一致对（全部 1-1）")
print(f"  配对平均得分 = {sum(a[s][0]+b[s][0] for s in seeds)/len(seeds):.3f} "
      f"(0.5=打平, 1.0=全胜)")
print(f"  先后手抵消后的隐含 A 胜率 = {sum(a[s][0]+b[s][0] for s in seeds)/(2*len(seeds)):.1%}")
