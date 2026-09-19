import json, sys

filepath = sys.argv[1]
with open(filepath) as f:
    data = json.load(f)

details = data.get("round_details", [])
actions = data.get("actions", [])

# Coin analysis
print("=== 金币分析 ===")
our_coins_traj = [r["our_coins"] for r in details]
enemy_coins_traj = [r["enemy_coins"] for r in details]
print(f"我方金币范围: {min(our_coins_traj)} - {max(our_coins_traj)}")
print(f"敌方金币范围: {min(enemy_coins_traj)} - {max(enemy_coins_traj)}")

# Periodic coin snapshots
print("\n=== 金币阶段性快照 (每30步) ===")
for i in range(0, len(details), 30):
    r = details[i]
    print(f"  step={r['step']}: our_coins={r['our_coins']}, enemy_coins={r['enemy_coins']}")

# Tower count analysis
our_tower_counts = [len(r["our_towers"]) for r in details]
enemy_tower_counts = [len(r["enemy_towers"]) for r in details]
print("\n=== 塔数量分析 ===")
print(f"我方塔数量范围: {min(our_tower_counts)} - {max(our_tower_counts)}")
print(f"敌方塔数量范围: {min(enemy_tower_counts)} - {max(enemy_tower_counts)}")

# Tower types at end
last = details[-1]
print("\n=== 最终状态 ===")
print(f"我方HP: {last['our_hp']}, 敌方HP: {last['enemy_hp']}")
print(f"我方金币: {last['our_coins']}, 敌方金币: {last['enemy_coins']}")
print(f"我方蚂蚁: {last['our_ants']}, 敌方蚂蚁: {last['enemy_ants']}")
print(f"我方塔数: {len(last['our_towers'])}, 敌方塔数: {len(last['enemy_towers'])}")

# Action type mapping
print("\n=== 动作类型详细分布 ===")
type_map = {}
for a in actions:
    t = a.get("action_type", "N/A")
    if t in type_map:
        type_map[t] += 1
    else:
        type_map[t] = 1
for k, v in sorted(type_map.items(), key=lambda x: -x[1]):
    print(f"  type={k}: count={v}")

# HP Delta analysis
print("\n=== HP重大变化事件 (>5 HP的变化) ===")
for i in range(1, len(details)):
    our_hp_delta = details[i]["our_hp"] - details[i-1]["our_hp"]
    enemy_hp_delta = details[i]["enemy_hp"] - details[i-1]["enemy_hp"]
    if abs(our_hp_delta) > 5 or abs(enemy_hp_delta) > 5:
        print(f"  step={details[i]['step']}: our_hp_delta={our_hp_delta:+d}, enemy_hp_delta={enemy_hp_delta:+d}")

# Tower event frequency analysis
print("\n=== 塔事件频率统计 ===")
our_builds = 0
our_losses = 0
enemy_builds = 0
enemy_losses = 0
prev_our_towers = 0
prev_enemy_towers = 0
for r in details:
    our_t = len(r["our_towers"])
    enemy_t = len(r["enemy_towers"])
    if our_t > prev_our_towers:
        our_builds += our_t - prev_our_towers
    elif our_t < prev_our_towers:
        our_losses += prev_our_towers - our_t
    if enemy_t > prev_enemy_towers:
        enemy_builds += enemy_t - prev_enemy_towers
    elif enemy_t < prev_enemy_towers:
        enemy_losses += prev_enemy_towers - enemy_t
    prev_our_towers = our_t
    prev_enemy_towers = enemy_t
print(f"我方建塔: {our_builds}, 失塔: {our_losses}")
print(f"敌方建塔: {enemy_builds}, 失塔: {enemy_losses}")
