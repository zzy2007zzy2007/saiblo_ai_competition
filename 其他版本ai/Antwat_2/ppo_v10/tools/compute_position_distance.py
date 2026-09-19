"""
计算四种 (x,y) 极坐标方案的编码位置到最近合法位置的平均 hex_distance.

建塔合法 = 己方 HIGHLAND（含已建塔占用）
超武合法 = 非 VOID
"""
import itertools
import math
from typing import List, Tuple

# ─── 地图数据 ───
MAP_SIZE = 19
Terrain_VOID = -1
Terrain_PATH = 0
Terrain_BARRIER = 1
Terrain_P0_HIGHLAND = 2
Terrain_P1_HIGHLAND = 3

MAP_PROPERTY: Tuple[Tuple[int, ...], ...] = (
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, 0, 0, 1, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, -1, -1, -1, -1),
    (-1, -1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1),
    (0, 0, 2, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0),
    (0, 0, 0, 2, 0, 0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 2, 0, 0, 0),
    (0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0),
    (0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0),
    (0, 1, 3, 0, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 0, 3, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0),
    (0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0),
    (0, 3, 0, 0, 0, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 3, 0),
    (0, 0, 3, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 3, 0, 0),
    (-1, 0, 0, 3, 0, 1, 1, 0, 0, 3, 0, 0, 1, 1, 0, 3, 0, 0, -1),
    (-1, -1, -1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, -1, -1, -1),
    (-1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 1, 1, 0, 0, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
)


# ─── hex_distance (from SDK) ───
def hex_distance(x0: int, y0: int, x1: int, y1: int) -> int:
    dy = abs(y0 - y1)
    if dy % 2:
        if x0 > x1:
            dx = max(0, abs(x0 - x1) - dy // 2 - (y0 % 2))
        else:
            dx = max(0, abs(x0 - x1) - dy // 2 - (1 - (y0 % 2)))
    else:
        dx = max(0, abs(x0 - x1) - dy // 2)
    return dx + dy


# ─── 预计算合法位置集 ───
# Player 0 的 HIGHLAND 集
p0_highlands: List[Tuple[int, int]] = []
# 所有非 VOID 格
non_void_cells: List[Tuple[int, int]] = []

for x in range(MAP_SIZE):
    for y in range(MAP_SIZE):
        t = MAP_PROPERTY[x][y]
        if t == Terrain_P0_HIGHLAND:
            p0_highlands.append((x, y))
        if t != Terrain_VOID:
            non_void_cells.append((x, y))

print(f"Player 0 HIGHLAND cells: {len(p0_highlands)}")
print(f"Non-VOID cells: {len(non_void_cells)}")


# ─── 极坐标解码 ───
def decode_polar(x: int, y: int, player: int = 0) -> Tuple[int, int]:
    """将极坐标 (x,y) 解码为地图坐标 player=0"""
    return (2 + x, 9 - y)


def nearest_distance(coord: Tuple[int, int], targets: List[Tuple[int, int]]) -> int:
    """计算到 target 列表的最近 hex_distance"""
    if not targets:
        return 999
    return min(hex_distance(coord[0], coord[1], t[0], t[1]) for t in targets)


# ─── 方案定义 ───
# 命名规则 [x_count]×[y_count]
schemes = {
    # 新增方案
    "3×3":  (range(0, 3),  range(-1, 2)),     # x=0..2, y=-1..+1 → 9组合，6维
    "3×4":  (range(0, 3),  range(-2, 2)),     # x=0..2, y=-2..+1 → 12组合，7维
    "4×4":  (range(0, 4),  range(-2, 2)),     # x=0..3, y=-2..+1 → 16组合，8维
    "4×5":  (range(0, 4),  range(-2, 3)),     # x=0..3, y=-2..+2 → 20组合，9维
    # 原有方案
    "5×5":  (range(0, 5),  range(-2, 3)),     # x=0..4, y=-2..+2 → 25组合，10维
    "6×7":  (range(0, 6),  range(-3, 4)),     # x=0..5, y=-3..+3 → 42组合，13维
    "7×7":  (range(0, 7),  range(-3, 4)),     # x=0..6, y=-3..+3 → 49组合，14维
    "7×9":  (range(0, 7),  range(-4, 5)),     # x=0..6, y=-4..+4 → 63组合，16维
}

print(f"\n{'方案':<12} {'总组合':>6} {'建塔距离(均值±σ)':>20} {'中位':>5} {'max':>4} {'超武距离':>12} {'p1':>7} {'p2':>9} {'d≤1&c≤1':>10}")
print("-" * 95)

for name, (x_range, y_range) in schemes.items():
    tower_dists = []
    weapon_dists = []
    
    # 新增：p1_p2 统计
    tower_p1_counts = []  # 每个编码点，最近距离处的合法位置数量 c
    tower_p2_products = []  # 每个编码点，c × d
    near_and_unique = []  # d≤1 且 c≤1
    
    for px, py in itertools.product(x_range, y_range):
        actual = decode_polar(px, py, player=0)
        tower_d = nearest_distance(actual, p0_highlands)
        weapon_d = nearest_distance(actual, non_void_cells)
        tower_dists.append(tower_d)
        weapon_dists.append(weapon_d)
        
        # p1/p2: 统计距离最近的合法位置"多少个"
        c = sum(1 for hl in p0_highlands
                if hex_distance(actual[0], actual[1], hl[0], hl[1]) == tower_d)
        tower_p1_counts.append(c)
        tower_p2_products.append(c * tower_d)
        
        # p3: 距离 ≤ 1 且 最近合法位置数 ≤ 1
        near_and_unique.append(1 if tower_d <= 1 and c <= 1 else 0)
    
    n_pos = len(tower_dists)
    tower_avg = sum(tower_dists) / n_pos
    tower_std = math.sqrt(sum((d - tower_avg) ** 2 for d in tower_dists) / n_pos)
    tower_med = sorted(tower_dists)[n_pos // 2]
    tower_max = max(tower_dists)
    weapon_avg = sum(weapon_dists) / n_pos
    weapon_std = math.sqrt(sum((d - weapon_avg) ** 2 for d in weapon_dists) / n_pos)
    weapon_med = sorted(weapon_dists)[n_pos // 2]
    weapon_max = max(weapon_dists)
    
    tower_p1 = sum(tower_p1_counts) / n_pos
    tower_p2 = sum(tower_p2_products) / n_pos
    p3_pct = sum(near_and_unique) / n_pos * 100
    
    print(f"{name:<12} {n_pos:>6} {tower_avg:>9.3f}±{tower_std:.3f} {tower_med:>5} {tower_max:>4} {weapon_avg:>8.3f}±{weapon_std:.3f}  {tower_p1:>10.2f}  {tower_p2:>10.3f}  {p3_pct:>6.1f}%")

# 补充：看看各方案的"距最近高地格的距离"分布
print("\n\n=== 各方案距离分布直方图 ===")
for name, (x_range, y_range) in schemes.items():
    tower_dists = []
    for px, py in itertools.product(x_range, y_range):
        actual = decode_polar(px, py, player=0)
        tower_dists.append(nearest_distance(actual, p0_highlands))
    
    max_d = max(tower_dists)
    hist = {}
    for d in tower_dists:
        hist[d] = hist.get(d, 0) + 1
    
    print(f"\n{name} (总 {len(tower_dists)}) 到最近高地的距离分布:")
    dist_list = sorted(hist.items())
    for d, count in dist_list:
        pct = count / len(tower_dists) * 100
        bar = "█" * int(pct / 2)
        print(f"  dist={d}: {count:>3}  ({pct:>5.1f}%) {bar}")

print("\n\n=== 当前 8 固定位方案（基准） ===")
# 当前 8 固定位的平均距离（只有这 8 个位置本身）
_PLAYER_TOWER_POS_0 = [(5, 9), (4, 9), (7, 8), (6, 7), (8, 7), (5, 6), (6, 14), (4, 2)]
for pos in _PLAYER_TOWER_POS_0:
    d = nearest_distance(pos, p0_highlands)
    print(f"  固定位 {pos}: dist_to_highland = {d}")
print(f"  固定位方案建塔平均距离: 0.000 (8 位置全是 HIGHLAND)")
