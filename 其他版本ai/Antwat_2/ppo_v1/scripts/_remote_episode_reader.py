import json, sys, math, os
from collections import Counter

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

# 动作索引 → 类型映射
ACTION_NAMES = {}
for i in range(0, 1):
    ACTION_NAMES[i] = "NO_OP"
for i in range(1, 11):
    ACTION_NAMES[i] = f"BUILD({i-1})"
for i in range(11, 51):
    ACTION_NAMES[i] = f"UPGRADE({i-11})"
for i in range(51, 61):
    ACTION_NAMES[i] = f"DOWNGRADE({i-51})"
for i in range(61, 66):
    ACTION_NAMES[i] = f"LS({i-61})"
for i in range(66, 71):
    ACTION_NAMES[i] = f"EMP({i-66})"
for i in range(71, 76):
    ACTION_NAMES[i] = f"DEFLECTOR({i-71})"
for i in range(76, 81):
    ACTION_NAMES[i] = f"EVASION({i-76})"
for i in range(81, 82):
    ACTION_NAMES[i] = "GEN_SPEED"
for i in range(82, 83):
    ACTION_NAMES[i] = "GEN_ANT"
for i in range(83, 96):
    ACTION_NAMES[i] = f"UNUSED({i})"

# --- 基本信息 ---
print(f"episode={ep}")
print(f"opponent_id={opp}")
print(f"player_position={pos}")
print(f"result={result}")
print(f"total_reward={total_reward:.2f}")
print(f"rounds={rounds}")
print(f"duration={duration:.2f}s")
print(f"start_time={start_time}")
print(f"end_time={end_time}")

for k in ["first_move_wins","first_move_losses","second_move_wins","second_move_losses"]:
    if k in data:
        print(f"{k}={data[k]}")

fh = final_hp
if isinstance(fh, dict):
    print(f"final_our_hp={fh.get('our', 'N/A')}")
    print(f"final_enemy_hp={fh.get('enemy', 'N/A')}")

# --- 动作分布 ---
print("action_counts:")
ac_total = sum(action_counts.values()) if action_counts else 0
type_groups = Counter()
for k, v in sorted(action_counts.items(), key=lambda x: -x[1]):
    kidx = int(k)
    name = ACTION_NAMES.get(kidx, f"UNKNOWN({kidx})")
    pct = v / ac_total * 100 if ac_total > 0 else 0
    print(f"  {k}={name}: {v}({pct:.1f}%)")
    # 按大类汇总
    if kidx == 0:
        type_groups["NO_OP"] += v
    elif 1 <= kidx <= 10:
        type_groups["BUILD"] += v
    elif 11 <= kidx <= 50:
        type_groups["UPGRADE"] += v
    elif 51 <= kidx <= 60:
        type_groups["DOWNGRADE"] += v
    elif 61 <= kidx <= 65:
        type_groups["LIGHTNING_STORM"] += v
    elif 66 <= kidx <= 70:
        type_groups["EMP"] += v
    elif 71 <= kidx <= 75:
        type_groups["DEFLECTOR"] += v
    elif 76 <= kidx <= 80:
        type_groups["EVASION"] += v
    elif 81 <= kidx <= 82:
        type_groups["TECH_UPGRADE"] += v
    else:
        type_groups[f"OTHER({kidx})"] += v

print("action_type_summary:")
for t, c in type_groups.most_common():
    pct = c / ac_total * 100 if ac_total > 0 else 0
    print(f"  {t}: {c}({pct:.1f}%)")
print(f"  total={ac_total}")
print(f"  unique_action_indices={len(action_counts)}")

# --- 奖励分析 ---
reward_list = [a.get("reward", 0) for a in actions if isinstance(a.get("reward"), (int, float))]
if reward_list:
    r_mean = sum(reward_list) / len(reward_list)
    r_var = sum((r - r_mean) ** 2 for r in reward_list) / len(reward_list)
    r_std = math.sqrt(r_var)
    pos_r = sum(1 for r in reward_list if r > 0)
    neg_r = sum(1 for r in reward_list if r < 0)
    zero_r = sum(1 for r in reward_list if r == 0)
    reward_max = max(reward_list)
    reward_min = min(reward_list)
    print(f"reward_mean={r_mean:.4f}")
    print(f"reward_std={r_std:.4f}")
    print(f"reward_max={reward_max:.4f}")
    print(f"reward_min={reward_min:.4f}")
    print(f"reward_pos={pos_r}")
    print(f"reward_neg={neg_r}")
    print(f"reward_zero={zero_r}")
    print(f"reward_count={len(reward_list)}")

# --- 回合轨迹（全量，不采样）---
if details:
    print("full_round_trajectory:")
    print("step,our_hp,enemy_hp,our_coins,enemy_coins,our_ants,enemy_ants,our_tower_count,enemy_tower_count")
    for r in details:
        print(f"{r['step']},{r['our_hp']},{r['enemy_hp']},{r['our_coins']},{r['enemy_coins']},{r['our_ants']},{r['enemy_ants']},{len(r.get('our_towers',[]))},{len(r.get('enemy_towers',[]))}")

# --- 动作序列（含回合经济上下文）---
print("actions_seq:")
print("step,action_idx,action_name,reward,round_our_hp,round_enemy_hp,round_our_coins,round_enemy_coins,round_our_towers,round_enemy_towers")
# 构建 step → round_detail 索引
step_to_detail = {r['step']: r for r in details}
for a in actions:
    at = a.get("action_type", "N/A")
    try:
        aidx = int(at)
        aname = ACTION_NAMES.get(aidx, f"UNKNOWN({at})")
    except:
        aname = str(at)
    rw = a.get("reward", 0)
    s = a.get("step", -1)
    rd = step_to_detail.get(s, {})
    print(f"{s},{at},{aname},{rw:.4f},{rd.get('our_hp','N/A')},{rd.get('enemy_hp','N/A')},{rd.get('our_coins','N/A')},{rd.get('enemy_coins','N/A')},{len(rd.get('our_towers',[]))},{len(rd.get('enemy_towers',[]))}")

# --- 塔事件序列 ---
print("tower_events:")
print("step,our_tower_count,enemy_tower_count,delta_our,delta_enemy")
prev_our = 0
prev_enemy = 0
for r in details:
    our_t = len(r.get("our_towers", []))
    enemy_t = len(r.get("enemy_towers", []))
    if our_t != prev_our or enemy_t != prev_enemy:
        print(f"{r['step']},{our_t},{enemy_t},{our_t-prev_our},{enemy_t-prev_enemy}")
    prev_our = our_t
    prev_enemy = enemy_t
