"""
评估 5×5 极坐标 + 指数缩放 Y 轴的覆盖能力。

原理：Y 轴用指数函数代替线性均匀分布，
让编码点在中线附近密集、在侧翼稀疏，扩大覆盖范围。

解码函数（Player 0）:
    actual_x = 2 + x_enc
    actual_y = 9 - sign(y_enc) * round(k * (exp(|y_enc|) - 1))
"""
import itertools
import math
from typing import List, Tuple

# ─── 地图数据 ───
MAP_SIZE = 19
Terrain_VOID = -1
Terrain_P0_HIGHLAND = 2

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

# Player 0 的 HIGHLAND 格
p0_highlands: List[Tuple[int, int]] = []
for x in range(MAP_SIZE):
    for y in range(MAP_SIZE):
        if MAP_PROPERTY[x][y] == Terrain_P0_HIGHLAND:
            p0_highlands.append((x, y))

# 8 个指定塔位
TARGETS = [
    (5, 9), (4, 9), (7, 8), (6, 7),
    (8, 7), (5, 6), (6, 14), (4, 2),
]

# X 轴始终使用 5 个均匀值
x_range = list(range(0, 5))  # x=0..4 → map x=2..6

# ─── 测试不同的指数缩放参数 k ───
# y 指数映射: 对于 y_enc ∈ {-2, -1, 0, 1, 2}:
#   map_y = 9 - sign(y_enc) * round(k * (exp(|y_enc|) - 1))
k_values = [0.5, 0.8, 1.0, 1.1, 1.2, 1.5]

for k in k_values:
    # 生成 5×5 解码位置
    decoded_positions = []
    y_map_coords = {}  # 记录 y_enc 到 map_y 的映射
    for x_enc in x_range:
        for y_enc in range(-2, 3):
            if y_enc == 0:
                y_offset = 0
            else:
                y_offset = round(k * (math.exp(abs(y_enc)) - 1)) * (1 if y_enc < 0 else -1)
                # y_enc < 0 → map 右侧 (y_enc=-2 → y_map = 9 - (-6) = 15)
                # y_enc > 0 → map 左侧 (y_enc=+2 → y_map = 9 - (+6) = 3)
                # map_y = 9 - y_offset where y_offset = sign(y_enc) * round(k*(exp-1))
                # Actually: y_enc=-2 → y_offset = -round(k*(e^2-1)) → map_y = 9 - (-round(k*(e^2-1))) = 9 + round(k*(e^2-1))
                # So map_y = 9 - sign(y_enc) * round(k*(exp-1))... let me redo this.
                pass
            actual_x = 2 + x_enc
            # 统一公式: map_y = 9 + (y_enc_sign) * round(k * (exp(|y_enc|) - 1))
            # 当 y_enc < 0 (偏向右侧): offset 为正 → map_y = 9 + positive = larger y
            if y_enc < 0:
                offset = round(k * (math.exp(abs(y_enc)) - 1))
            elif y_enc > 0:
                offset = -round(k * (math.exp(abs(y_enc)) - 1))
            else:
                offset = 0
            actual_y = 9 + offset
            decoded_positions.append((actual_x, actual_y, x_enc, y_enc))  # (map_x, map_y, x_enc, y_enc)
            y_map_coords[y_enc] = actual_y

    # 计算覆盖统计
    y_map_vals = [y_map_coords[ye] for ye in sorted(y_map_coords.keys())]
    
    # 对每个目标位置计算最近距离
    results = {}
    for target in TARGETS:
        min_d = 999
        min_pt = None
        for dp in decoded_positions:
            d = hex_distance(target[0], target[1], dp[0], dp[1])
            if d < min_d:
                min_d = d
                min_pt = dp  # (map_x, map_y, x_enc, y_enc)
        
        if min_pt is not None:
            # 同距离下其他 HIGHLAND 的数量
            other_count = sum(1 for hl in p0_highlands
                             if hex_distance(min_pt[0], min_pt[1], hl[0], hl[1]) == min_d)
        else:
            other_count = 0
        
        results[target] = {"min_d": min_d, "n_other_at_d": other_count, "from": min_pt}
    
    all_covered_d3 = all(r["min_d"] <= 3 for r in results.values())
    all_covered_d2 = all(r["min_d"] <= 2 for r in results.values())
    
    # HIGHLAND 覆盖率
    hl_covered_d1 = sum(1 for hl in p0_highlands
                        if min(hex_distance(hl[0], hl[1], d[0], d[1]) for d in decoded_positions) <= 1)
    
    print(f"\n{'='*70}")
    print(f"k={k} | y_map: {y_map_vals}")
    print(f"{'='*70}")
    print(f"{'目标':<8} {'极坐标(x,y)':>12} {'距离':>6} {'同距H数':>8}")
    print("-" * 40)
    for target in TARGETS:
        r = results[target]
        status = "✅" if r["min_d"] <= 1 else "❌" if r["min_d"] > 2 else "⚠️"
        mp = r["from"]
        polar = f"({mp[2]},{mp[3]})" if mp else "N/A"
        print(f"{status} {target}   {polar:>10}      {r['min_d']:>2d}       {r['n_other_at_d']:>2d}")
    print(f"\n全部距离≤2? {'是 ✅' if all_covered_d2 else '否 ❌'}")
    print(f"全部距离≤3? {'是 ✅' if all_covered_d3 else '否 ❌'}")
    print(f"HIGHLAND覆盖(d≤1): {hl_covered_d1}/{len(p0_highlands)} = {hl_covered_d1/len(p0_highlands)*100:.1f}%")

# ─── 另外测试手动非均匀方案 ───
manual_y_map = {-2: 16, -1: 12, 0: 9, 1: 7, 2: 2}
print(f"\n{'='*70}")
print(f"手动: y_enc→map_y: {manual_y_map}")
print(f"{'='*70}")
decoded_manual = []
for x_enc in range(0, 5):
    for y_enc, y_map in manual_y_map.items():
        decoded_manual.append((2 + x_enc, y_map, x_enc, y_enc))  # (map_x, map_y, x_enc, y_enc)

print(f"{'目标':<8} {'极坐标(x,y)':>12} {'距离':>6} {'同距H数':>8}")
print("-" * 40)
all_d2 = True
for target in TARGETS:
    min_d = 999
    min_pt = None
    for dp in decoded_manual:
        d = hex_distance(target[0], target[1], dp[0], dp[1])
        if d < min_d:
            min_d = d
            min_pt = dp
    other = sum(1 for hl in p0_highlands
                if hex_distance(min_pt[0], min_pt[1], hl[0], hl[1]) == min_d)
    if min_d > 2: all_d2 = False
    status = "✅" if min_d <= 1 else "❌" if min_d > 2 else "⚠️"
    polar = f"({min_pt[2]},{min_pt[3]})" if min_pt else "N/A"
    print(f"{status} {target}   {polar:>10}      {min_d:>2d}       {other:>2d}")
print(f"全部距离≤2? {'是 ✅' if all_d2 else '否 ❌'}")
hl_cov = sum(1 for hl in p0_highlands
             if min(hex_distance(hl[0], hl[1], d[0], d[1]) for d in decoded_manual) <= 1)
print(f"HIGHLAND覆盖(d≤1): {hl_cov}/{len(p0_highlands)} = {hl_cov/len(p0_highlands)*100:.1f}%")
