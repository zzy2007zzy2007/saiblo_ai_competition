import json, sys, math, os

filepath = sys.argv[1]
with open(filepath) as f:
    data = json.load(f)

ep = data["episode"]
opp = data["opponent_id"]
pos = data.get("player_position", "N/A")
result = data["result"]
total_reward = data["total_reward"]
rounds = data["rounds"]
duration = data.get("duration", 0)
start_time = data.get("start_time", "")
end_time = data.get("end_time", "")

details = data.get("round_details", [])
final_hp = data.get("final_hp", {})
action_counts = data.get("action_counts", {})
actions = data.get("actions", [])

# --- 基本信息 ---
print(f"episode={ep}")
print(f"opponent_id={opp}")
print(f"player_position={pos}")
print(f"result={result}")
print(f"total_reward={total_reward:.2f}")
print(f"rounds={rounds}")
print(f"duration={duration}")
print(f"start_time={start_time}")
print(f"end_time={end_time}")

# --- first_move / second_move ---
for k in ["first_move_wins","first_move_losses","second_move_wins","second_move_losses"]:
    if k in data:
        print(f"{k}={data[k]}")

# --- final_hp ---
fh = final_hp
if isinstance(fh, dict):
    print(f"final_our_hp={fh.get('our', 'N/A')}")
    print(f"final_enemy_hp={fh.get('enemy', 'N/A')}")

# --- 动作分布 ---
print("action_counts:")
ac_total = sum(action_counts.values()) if action_counts else 0
for k, v in sorted(action_counts.items(), key=lambda x: -x[1]):
    pct = v / ac_total * 100 if ac_total > 0 else 0
    print(f"  {k}={v}({pct:.1f}%)")
print(f"  total={ac_total}")

# --- 奖励分析 ---
reward_list = [a.get("reward", 0) for a in actions if isinstance(a.get("reward"), (int, float))]
if reward_list:
    r_mean = sum(reward_list) / len(reward_list)
    r_var = sum((r - r_mean) ** 2 for r in reward_list) / len(reward_list)
    r_std = math.sqrt(r_var)
    pos_r = sum(1 for r in reward_list if r > 0)
    neg_r = sum(1 for r in reward_list if r < 0)
    zero_r = sum(1 for r in reward_list if r == 0)
    print(f"reward_mean={r_mean:.4f}")
    print(f"reward_std={r_std:.4f}")
    print(f"reward_pos={pos_r}")
    print(f"reward_neg={neg_r}")
    print(f"reward_zero={zero_r}")
    print(f"reward_count={len(reward_list)}")

# --- 回合轨迹采样（每~30步 + 首尾）---
if details:
    interval = max(1, len(details) // 30)
    print("trajectory:")
    print("step,our_hp,enemy_hp,our_coins,enemy_coins,our_ants,enemy_ants,our_towers,enemy_towers")
    for i in range(0, len(details), interval):
        r = details[i]
        print(f"{r['step']},{r['our_hp']},{r['enemy_hp']},{r['our_coins']},{r['enemy_coins']},{r['our_ants']},{r['enemy_ants']},{len(r['our_towers'])},{len(r['enemy_towers'])}")
    r = details[-1]
    print(f"{r['step']},{r['our_hp']},{r['enemy_hp']},{r['our_coins']},{r['enemy_coins']},{r['our_ants']},{r['enemy_ants']},{len(r['our_towers'])},{len(r['enemy_towers'])}")

# --- 动作序列（action_type 随时间的变化）---
print("actions_seq:")
for a in actions:
    at = a.get("action_type", "N/A")
    rw = a.get("reward", 0)
    print(f"  step={a.get('step','?')} type={at} reward={rw:.4f}")

# --- 塔建造/拆除/升级事件 ---
print("tower_events:")
prev_our_towers = 0
prev_enemy_towers = 0
for i, r in enumerate(details):
    our_t = len(r.get("our_towers", []))
    enemy_t = len(r.get("enemy_towers", []))
    if our_t != prev_our_towers or enemy_t != prev_enemy_towers:
        print(f"  round={r['step']} our_towers={our_t}({'+' if our_t>prev_our_towers else ''}{our_t-prev_our_towers}) enemy_towers={enemy_t}({'+' if enemy_t>prev_enemy_towers else ''}{enemy_t-prev_enemy_towers})")
    prev_our_towers = our_t
    prev_enemy_towers = enemy_t
