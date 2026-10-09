# 复现 antwar/antwar2 地图构造 (map.cpp Map::Map())
L = 10
M = 2 * L - 1  # 19
valid = [[False] * M for _ in range(M)]
k = 2 * L - 1
for i in range(L - 1, -1, -1):
    for j in range(k):
        valid[(L - 1 - i) // 2 + j][i] = True
    k -= 1
k = 2 * L - 1
for i in range(L - 1, 2 * L - 1):
    for j in range(k):
        valid[(i - L + 1) // 2 + j][i] = True
    k -= 1
shape = sum(sum(r) for r in valid)
rows = [sum(valid[x][y] for x in range(M)) for y in range(M)]

blocks = [(6,1),(7,1),(9,1),(11,1),(12,1),(4,2),(6,2),(8,2),(9,2),(11,2),(13,2),(4,3),(5,3),(13,3),(14,3),(6,4),(8,4),(9,4),(11,4),(3,5),(4,5),(7,5),(9,5),(11,5),(14,5),(15,5),(3,6),(5,6),(12,6),(14,6),(2,7),(5,7),(6,7),(8,7),(9,7),(10,7),(12,7),(13,7),(16,7),(1,8),(2,8),(7,8),(10,8),(15,8),(16,8),(0,9),(4,9),(5,9),(6,9),(9,9),(12,9),(13,9),(14,9),(18,9),(1,10),(2,10),(7,10),(10,10),(15,10),(16,10),(2,11),(5,11),(6,11),(8,11),(9,11),(10,11),(12,11),(13,11),(16,11),(3,12),(5,12),(12,12),(14,12),(3,13),(4,13),(7,13),(9,13),(11,13),(14,13),(15,13),(6,14),(8,14),(9,14),(11,14),(4,15),(5,15),(13,15),(14,15),(4,16),(6,16),(8,16),(9,16),(11,16),(13,16),(6,17),(7,17),(9,17),(11,17),(12,17)]
for x, y in blocks:
    assert valid[x][y], f"block {(x,y)} was not valid in shape!"
    valid[x][y] = False
walk = sum(sum(r) for r in valid)

t0 = [(6,1),(7,1),(4,2),(6,2),(8,2),(4,3),(5,3),(6,4),(8,4),(7,5),(5,6),(5,7),(6,7),(8,7),(7,8),(4,9),(5,9),(6,9),(7,10),(5,11),(6,11),(8,11),(5,12),(7,13),(6,14),(8,14),(4,15),(5,15),(4,16),(6,16),(8,16),(6,17),(7,17)]
t1 = [(11,1),(12,1),(9,2),(11,2),(13,2),(13,3),(14,3),(9,4),(11,4),(11,5),(12,6),(10,7),(12,7),(13,7),(10,8),(12,9),(13,9),(14,9),(10,10),(10,11),(12,11),(13,11),(12,12),(11,13),(9,14),(11,14),(13,15),(14,15),(9,16),(11,16),(13,16),(11,17),(12,17)]
# tower cell on blocked (non-walkable) cells?
t0_blocked = sum(1 for c in t0 if c in blocks)
t1_blocked = sum(1 for c in t1 if c in blocks)
# symmetry check: t1 should be mirror of t0: x -> 18-x? check
mirror = {(18 - x, y) for x, y in t0}
print("shape cells:", shape)
print("row counts:", rows)
print("blocked:", len(blocks), "walkable:", walk)
print("tower cells P0:", len(t0), " P1:", len(t1))
print("P0 tower cells that are blocked(unwalkable):", t0_blocked, " P1:", t1_blocked)
print("mirror symmetric:", mirror == set(t1))
# camps walkable?
print("camp0 valid:", valid[2][9], "camp1 valid:", valid[16][9])
# distance between camps (hex distance per map.cpp)
def dist(a, b):
    dy = abs(a[1] - b[1])
    if dy % 2:
        if a[0] > b[0]:
            dx = max(0, abs(a[0]-b[0]) - dy//2 - (a[1] % 2))
        else:
            dx = max(0, abs(a[0]-b[0]) - dy//2 - (1 - a[1] % 2))
    else:
        dx = max(0, abs(a[0]-b[0]) - dy//2)
    return dx + dy
print("base-to-base distance:", dist((2,9),(16,9)))
