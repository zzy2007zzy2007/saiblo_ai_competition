"""算清楚"穷举式枚举"到底要多少条候选（不需要网络裁剪）。"""
import glob
import json
import sys

import numpy as np

for p in ["D:/2026智能体大赛_新2_distill/Ant-Game", "D:/2026智能体大赛_新2_distill/Ant-Game/SDK"]:
    sys.path.insert(0, p)
from SDK.utils.constants import HIGHLAND_CELLS, MAP_PROPERTY, Terrain  # noqa: E402


def cube(x, y):
    z = y - (x - (x & 1)) // 2
    return x, -x - z, z


def hexdist(x1, y1, x2, y2):
    a, b = cube(x1, y1), cube(x2, y2)
    return max(abs(a[i] - b[i]) for i in range(3))


nonvoid = [(x, y) for x in range(19) for y in range(19) if MAP_PROPERTY[x][y] != Terrain.VOID]
light = [c for c in nonvoid if hexdist(c[0], c[1], 9, 9) <= 5]
print(f"非 VOID 格 = {len(nonvoid)}")
print(f"闪电合法中心（半径 5 内）= {len(light)}")
print(f"我方高地点（可建塔）= {len(HIGHLAND_CELLS[0])}")

counts = []
occupied = []
for path in sorted(glob.glob("training_history/distill_top2/abl_linear2_raw/ents_seed*.jsonl"))[:6]:
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        for player in (0, 1):
            own = [t for t in (record.get("towers") or []) if int(t[1]) == player]
            counts.append(len(own))
            occupied.append(sum(1 for t in own if (int(t[2]), int(t[3])) in HIGHLAND_CELLS[player]))
counts = np.array(counts)
occupied = np.array(occupied)
print(f"我方塔数: 均值 {counts.mean():.2f}  最大 {counts.max()}  "
      f"分布 {dict(zip(*[a.tolist() for a in np.unique(counts, return_counts=True)]))}")
legal_build = len(HIGHLAND_CELLS[0]) - occupied
print(f"可建塔格（33 − 已占）: 均值 {legal_build.mean():.1f}  最小 {legal_build.min()}")

print("\n=== 每阶段的穷举候选数（不做任何网络裁剪）===")
print(f"  hold                     1")
print(f"  建塔 @ 全部合法高地点     ≤ {len(HIGHLAND_CELLS[0])}")
print(f"  降级 @ 我方每座塔         ≤ {counts.max()}")
print(f"  升级 @ 每塔 × 3 个目标型  ≤ {3 * counts.max()}")
print(f"  闪电 @ 全部合法中心       ≤ {len(light)}")
total = 1 + len(HIGHLAND_CELLS[0]) + counts.max() + 3 * counts.max() + len(light)
print(f"  --- 合计上限约 {total} 条/阶段")
print(f"  （若闪电不可用则 −{len(light)} ⇒ 约 {total - len(light)} 条/阶段）")
