#!/usr/bin/env python3
"""
从 SelfPlay 对战 JSON 日志中分解 total_reward 的各分量贡献。

使用方法:
  analyze_reward_breakdown.py <json_file1> [json_file2 ...]

输出每局对战的 reward 分解表 + 跨对局汇总表。
"""
import json
import sys
import os
from collections import defaultdict

REWARD_CONFIG = {
    "hp_attack_weight": 0.3,
    "own_coin_gain_weight": 0.02,
    "enemy_coin_gain_weight": 0.01,
    "tower_survival_per_tower": 0.20,
    "enemy_tower_survival_per_tower": 0.10,
    "tower_survival_level_multipliers": [1.0, 1.5, 2.5],
    "build_tower_tiers": [0.6, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1],
    "upgrade_tower_l2": 0.6,
    "upgrade_tower_l3": 1.0,
    "downgrade_tower_penalty": -3.15,
    "noop_tolerance": 3,
    "noop_base_penalty": -0.04,
    "noop_max_penalty": -0.80,
    "win_reward": 500.0,
    "loss_reward": -100.0,
    "step_reward_clip": 20.0,
}

def _calc_tower_cost(tower_count: int) -> int:
    n = tower_count
    return 15 * (3 ** (n // 2)) * (2 if n % 2 else 1)

def analyze_battle(log_path: str) -> dict:
    with open(log_path) as f:
        raw = json.load(f)

    rounds = raw.get("round_details", [])
    if not rounds:
        return None

    episode = raw.get("episode", 0)
    total_reward = raw.get("total_reward", 0)
    result = raw.get("result", "?")
    player_position = raw.get("player_position", 0)
    final_our_hp = rounds[-1].get("our_hp", 0)
    final_enemy_hp = rounds[-1].get("enemy_hp", 0)

    cfg = REWARD_CONFIG

    r_build = 0.0
    r_upgrade = 0.0
    r_noop = 0.0
    r_hp = 0.0
    r_coin = 0.0
    r_tower = 0.0
    r_terminal = 0.0
    r_tech = 0.0
    r_other = 0.0

    build_count = 0
    upgrade_count = 0
    upgrade_l2_count = 0
    upgrade_l3_count = 0
    noop_count = 0
    tower_build_slot_counts = defaultdict(int)

    noop_streak = 0

    prev_our_hp = rounds[0].get("our_hp", 50)
    prev_enemy_hp = rounds[0].get("enemy_hp", 50)
    prev_our_coins = rounds[0].get("our_coins", 50)
    prev_enemy_coins = rounds[0].get("enemy_coins", 50)

    for i in range(1, len(rounds)):
        r = rounds[i]
        p = rounds[i - 1]

        step = r["step"]
        our_hp = r.get("our_hp", 0)
        enemy_hp = r.get("enemy_hp", 0)
        our_coins = r.get("our_coins", 0)
        enemy_coins = r.get("enemy_coins", 0)

        curr_our_towers = {t["id"]: t for t in r.get("our_towers", [])}
        prev_our_towers = {t["id"]: t for t in p.get("our_towers", [])}
        curr_our_tech = r.get("our_tech", {})

        # --- 动作奖励 ---
        # BUILD 检测
        for tid, t in curr_our_towers.items():
            if tid not in prev_our_towers:
                pos_key = tuple(t["position"])
                slot_count = tower_build_slot_counts[pos_key]
                tiers = cfg["build_tower_tiers"]
                reward_val = tiers[min(slot_count, len(tiers) - 1)]
                r_build += reward_val
                build_count += 1
                tower_build_slot_counts[pos_key] += 1

        # UPGRADE 检测
        for tid, t in curr_our_towers.items():
            if tid in prev_our_towers and t["level"] > prev_our_towers[tid]["level"]:
                if t["level"] == 1:
                    reward_val = cfg["upgrade_tower_l2"]
                    upgrade_l2_count += 1
                else:
                    reward_val = cfg["upgrade_tower_l3"]
                    upgrade_l3_count += 1
                r_upgrade += reward_val
                upgrade_count += 1

        # TECH_UPGRADE 检测
        prev_tech = p.get("our_tech", {})
        for key in ("gen_speed", "ant_strength"):
            if curr_our_tech.get(key, 0) > prev_tech.get(key, 0):
                r_tech += 0.5

        # NO-OP 检测 - 同时考虑 NO_OP 动作和动作列表为空
        step_has_action = False
        for tid, t in curr_our_towers.items():
            if tid not in prev_our_towers:
                step_has_action = True
                break
        for tid, t in prev_our_towers.items():
            if tid not in curr_our_towers:
                step_has_action = True
                break
        for tid, t in curr_our_towers.items():
            if tid in prev_our_towers and t["level"] > prev_our_towers[tid]["level"]:
                step_has_action = True
                break

        if not step_has_action:
            noop_streak += 1
            noop_count += 1
            if noop_streak > cfg["noop_tolerance"]:
                penalty = cfg["noop_base_penalty"] + cfg["noop_base_penalty"] * (
                    (noop_streak - cfg["noop_tolerance"]) // 10
                )
                penalty = max(penalty, cfg["noop_max_penalty"])
                r_noop += penalty
        else:
            noop_streak = 0

        # --- 回合奖励 ---
        enemy_hp_damage = prev_enemy_hp - enemy_hp
        own_hp_damage = prev_our_hp - our_hp
        r_hp += (enemy_hp_damage - own_hp_damage) * cfg["hp_attack_weight"]

        own_income = our_coins - prev_our_coins
        enemy_income = enemy_coins - prev_enemy_coins
        r_coin += own_income * cfg["own_coin_gain_weight"]
        r_coin -= enemy_income * cfg["enemy_coin_gain_weight"]

        own_tower_count = len(curr_our_towers)
        enemy_tower_count = len({t["id"] for t in r.get("enemy_towers", [])})
        own_tower_val = own_tower_count * cfg["tower_survival_per_tower"]
        enemy_tower_val = enemy_tower_count * cfg["enemy_tower_survival_per_tower"]
        r_tower += own_tower_val - enemy_tower_val

        prev_our_hp = our_hp
        prev_enemy_hp = enemy_hp
        prev_our_coins = our_coins
        prev_enemy_coins = enemy_coins

    # --- 终局奖励 ---
    if result == "win":
        r_terminal = cfg["win_reward"]
    elif result == "loss":
        r_terminal = cfg["loss_reward"]

    estimated_known = r_build + r_upgrade + r_noop + r_hp + r_coin + r_tower + r_terminal + r_tech
    r_unknown = total_reward - estimated_known

    return {
        "episode": episode,
        "result": result,
        "position": "先手" if player_position == 0 else "后手",
        "total_reward": total_reward,
        "estimated_known": estimated_known,
        "final_hp": f"{final_our_hp}:{final_enemy_hp}",
        "r_build": r_build,
        "r_upgrade": r_upgrade,
        "r_noop": r_noop,
        "r_hp": r_hp,
        "r_coin": r_coin,
        "r_tower": r_tower,
        "r_terminal": r_terminal,
        "r_tech": r_tech,
        "r_unknown": r_unknown,
        "build_count": build_count,
        "upgrade_count": upgrade_count,
        "upgrade_l2": upgrade_l2_count,
        "upgrade_l3": upgrade_l3_count,
        "noop_count": noop_count,
        "total_steps": len(rounds) - 1,
    }


def format_analysis(data: dict) -> str:
    ep = data["episode"]
    pos = data["position"]
    result = data["result"]
    hp = data["final_hp"]
    total = data["total_reward"]

    lines = []
    sep = "═" * 67
    lines.append(sep)
    lines.append(f" Episode {ep}  {pos}  {'胜' if result == 'win' else '败'}  "
                 f"HP {hp}  Total Reward: {total:+.1f}")
    lines.append(sep)
    lines.append("")

    components = [
        ("R_build   建塔奖励", data["r_build"]),
        ("R_upgrade 升级奖励", data["r_upgrade"]),
        ("R_hp      HP伤害",   data["r_hp"]),
        ("R_coin    金币收入", data["r_coin"]),
        ("R_tower   塔存活",   data["r_tower"]),
        ("R_noop    NO-OP惩罚", data["r_noop"]),
        ("R_term   终局奖励",  data["r_terminal"]),
        ("R_tech    科技升级", data["r_tech"]),
    ]

    abs_total = sum(abs(v) for _, v in components) or 1

    lines.append(f"  {'分量':<16} {'数值':>10} {'占比':>8}")
    lines.append(f"  {'─'*16} {'─'*10} {'─'*8}")
    for name, val in components:
        pct = val / abs_total * 100 if abs_total else 0
        sign = "+" if val >= 0 else ""
        lines.append(f"  {name:<16} {sign}{val:>+8.2f}  {pct:>6.1f}%")
    lines.append(f"  {'─'*16} {'─'*10} {'─'*8}")
    lines.append(f"  {'合计(已知)':<16} {data['estimated_known']:>+10.1f}")
    lines.append(f"  {'未建模奖励':<16} {data['r_unknown']:>+10.1f}  (超级武器/蚂蚁击杀等)")
    lines.append(f"  {'实际 Reward':<16} {total:>+10.1f}")
    lines.append("")

    lines.append("  动作效率:")
    lines.append(f"    BUILD:  {data['build_count']}次 → +{data['r_build']:.2f}  ({data['r_build']/max(data['build_count'],1):.3f}/次)")
    lines.append(f"    UPG:    {data['upgrade_count']}次(L2:{data['upgrade_l2']} L3:{data['upgrade_l3']}) → +{data['r_upgrade']:.2f}  ({data['r_upgrade']/max(data['upgrade_count'],1):.3f}/次)")
    lines.append(f"    NO-OP:  {data['noop_count']}步/{data['total_steps']}步({data['noop_count']/max(data['total_steps'],1)*100:.1f}%) → {data['r_noop']:.2f}")
    lines.append("")

    return "\n".join(lines)


def format_summary_table(all_results: list) -> str:
    lines = []
    lines.append("=" * 120)
    lines.append("跨对局 Reward 分解汇总")
    lines.append("=" * 120)
    lines.append("")

    header = (f"  {'Ep':<6} {'结果':<4} {'HP终局':<8} {'Reward':>8} "
              f"{'R_build%':>8} {'R_upg%':>7} {'R_hp%':>6} {'R_coin%':>7} "
              f"{'R_tower%':>8} {'R_noop%':>7} {'R_term%':>7} "
              f"{'BUILD':>6} {'UPG':>4}")
    sep_line = "  " + "─" * (len(header) - 2)

    lines.append(header)
    lines.append(sep_line)

    for d in all_results:
        total = d["total_reward"]
        abs_total = sum(abs(d[k]) for k in ("r_build","r_upgrade","r_noop","r_hp","r_coin","r_tower","r_terminal","r_tech")) or 1

        def pct(val):
            return val / abs_total * 100

        result_cn = "胜" if d["result"] == "win" else "败"
        lines.append(
            f"  Ep{d['episode']:<4} {result_cn:<4} {d['final_hp']:<8} "
            f"{d['total_reward']:>+8.1f} "
            f"{pct(d['r_build']):>7.1f}% {pct(d['r_upgrade']):>6.1f}% "
            f"{pct(d['r_hp']):>5.1f}% {pct(d['r_coin']):>6.1f}% "
            f"{pct(d['r_tower']):>7.1f}% {pct(d['r_noop']):>6.1f}% "
            f"{pct(d['r_terminal']):>6.1f}% "
            f"{d['build_count']:>4} {d['upgrade_count']:>4}"
        )

    lines.append("")
    lines.append("─" * 120)

    wins = [d for d in all_results if d["result"] == "win"]
    losses = [d for d in all_results if d["result"] == "loss"]

    if wins:
        avg_win_r = sum(d["total_reward"] for d in wins) / len(wins)
        lines.append(f"  胜局平均 Reward: {avg_win_r:.1f}  ({len(wins)}局)")
    if losses:
        avg_loss_r = sum(d["total_reward"] for d in losses) / len(losses)
        lines.append(f"  败局平均 Reward: {avg_loss_r:.1f}  ({len(losses)}局)")

    avg_build = sum(d["build_count"] for d in all_results) / len(all_results)
    avg_upg = sum(d["upgrade_count"] for d in all_results) / len(all_results)
    avg_noop_pct = sum(d["noop_count"]/max(d["total_steps"],1)*100 for d in all_results) / len(all_results)
    lines.append(f"  平均 BUILD: {avg_build:.1f}次  平均 UPG: {avg_upg:.1f}次  平均 NO-OP%: {avg_noop_pct:.1f}%")
    lines.append("")

    return "\n".join(lines)


def main():
    import glob
    paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("tmp/selfplay_6[5-9][0-9]_*.json")

    all_results = []
    for path in paths:
        if not os.path.exists(path):
            print(f"跳过: 文件不存在 {path}", file=sys.stderr)
            continue
        data = analyze_battle(path)
        if data is None:
            print(f"跳过: 数据无效 {path}", file=sys.stderr)
            continue
        all_results.append(data)
        print(format_analysis(data))

    if len(all_results) > 1:
        print(format_summary_table(all_results))


if __name__ == "__main__":
    main()
