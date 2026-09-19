#!/usr/bin/env python3
"""
扫描所有 SelfPlay 对战 JSON，统计超级武器和科技升级。
检测方式：
- 科技升级：round_details 中 our_tech / enemy_tech 的 gen_speed / ant_strength 等级变化
- 超级武器：reward_detail 各分量（action_reward 包含超武奖励）
"""
import json
import os
import sys
import glob
from collections import defaultdict, Counter

BATTLE_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/selfplay/selfplay_battles/detailed_battles"

SUPER_WEAPON_LABELS = {
    "lightning_storm": "闪电风暴",
    "emp": "电磁爆",
    "deflector": "偏转护盾",
    "evasion": "紧急规避",
}

TECH_LABELS = {
    "gen_speed": "产量速度",
    "ant_strength": "兵种强度",
}


def scan_file(filepath):
    """扫描单个文件，返回超武和科技升级信息"""
    with open(filepath) as f:
        data = json.load(f)

    ep = int(os.path.basename(filepath).split("_")[1])
    ppo_won = 1 if data.get("result") == "win" else 0

    rd = data.get("round_details", [])
    actions = data.get("actions", [])

    # --- 科技升级检测：tech 等级变化 ---
    ppo_tech_upgrades = 0
    opp_tech_upgrades = 0
    ppo_upgrade_types = Counter()
    opp_upgrade_types = Counter()

    prev_ppo_tech = {"gen_speed": 0, "ant_strength": 0}
    prev_opp_tech = {"gen_speed": 0, "ant_strength": 0}

    for step_info in rd:
        our_tech = step_info.get("our_tech", {})
        enemy_tech = step_info.get("enemy_tech", {})

        for key in ["gen_speed", "ant_strength"]:
            cur_val = our_tech.get(key, 0)
            if cur_val > prev_ppo_tech.get(key, 0):
                ppo_upgrade_types[key] += cur_val - prev_ppo_tech.get(key, 0)
                ppo_tech_upgrades += cur_val - prev_ppo_tech.get(key, 0)
            prev_ppo_tech[key] = cur_val

            cur_val = enemy_tech.get(key, 0)
            if cur_val > prev_opp_tech.get(key, 0):
                opp_upgrade_types[key] += cur_val - prev_opp_tech.get(key, 0)
                opp_tech_upgrades += cur_val - prev_opp_tech.get(key, 0)
            prev_opp_tech[key] = cur_val

    # --- 超级武器检测：reward_detail ---
    ppo_sw_count = 0
    opp_sw_count = 0
    ppo_sw_types = Counter()
    opp_sw_types = Counter()

    for act in actions:
        rd_detail = act.get("reward_detail", {})
        action_reward = rd_detail.get("action_reward", {})

        if isinstance(action_reward, dict):
            for sw_key, sw_label in SUPER_WEAPON_LABELS.items():
                val = action_reward.get(sw_key, 0)
                if val > 0:
                    ppo_sw_count += 1
                    ppo_sw_types[sw_key] += 1

        # opponent_action 是 int 索引，无法直接解析
        # 改为从 PPO action 中检测 opponent 的 tech_bonus

    # Opponent 也使用同样的 reward_detail 结构
    # 在 selfplay 日志中，ppo 视角记录自己动作的 reward_detail
    # opponent 的 action 是整数索引，没有对应的 reward_detail
    # 所以 opponent 侧的超武无法直接从 actions 检测

    # 但我们可以从 round_details 中检测 opponent 的 tower 变化来推断
    # 或者检查 action_counts 分布

    action_counts = data.get("action_counts", {})

    result = {
        "ep": ep,
        "ppo_won": ppo_won,
        "ppo_tech_upgrades": ppo_upgrade_types,
        "opp_tech_upgrades": opp_upgrade_types,
        "ppo_sw": ppo_sw_types,
        "ppo_total_sw": ppo_sw_count,
        "ppo_total_tech": ppo_tech_upgrades,
        "opp_total_tech": opp_tech_upgrades,
        "action_counts": action_counts,
    }
    return result


def main():
    files = sorted(glob.glob(os.path.join(BATTLE_DIR, "*.json")))
    print(f"对战总数: {len(files)}\n")

    total_ppo_sw = 0
    total_ppo_tech = 0
    total_opp_tech = 0
    ppo_sw_types = Counter()
    ppo_tech_types = Counter()
    opp_tech_types = Counter()
    battles_with_ppo_sw = 0
    battles_with_ppo_tech = 0
    battles_with_opp_tech = 0

    ppo_sw_by_ep_bin = Counter()
    ppo_tech_by_ep_bin = Counter()
    opp_tech_by_ep_bin = Counter()

    sw_ep_battle_count = Counter()
    tech_ep_battle_count = Counter()

    ppo_wins_with_sw = 0
    ppo_wins_with_tech = 0
    total_ppo_wins = 0

    ppo_sw_first_ep = {}
    ppo_tech_first_ep = {}

    all_results = []

    for filepath in files:
        res = scan_file(filepath)
        all_results.append(res)
        ep = res["ep"]
        ep_bin = (ep // 200) * 200

        if res["ppo_won"]:
            total_ppo_wins += 1

        # 超级武器
        if res["ppo_total_sw"] > 0:
            battles_with_ppo_sw += 1
            ppo_sw_by_ep_bin[ep_bin] += res["ppo_total_sw"]
            sw_ep_battle_count[ep_bin] += 1
            if res["ppo_won"]:
                ppo_wins_with_sw += 1
            for k, v in res["ppo_sw"].items():
                ppo_sw_types[k] += v
            if res["ppo_total_sw"] > 0 and not ppo_sw_first_ep:
                ppo_sw_first_ep["first_use"] = ep

        total_ppo_sw += res["ppo_total_sw"]

        # PPO 科技升级
        if res["ppo_total_tech"] > 0:
            battles_with_ppo_tech += 1
            ppo_tech_by_ep_bin[ep_bin] += res["ppo_total_tech"]
            if res["ppo_won"]:
                ppo_wins_with_tech += 1
            for k, v in res["ppo_tech_upgrades"].items():
                ppo_tech_types[k] += v
                if k not in ppo_tech_first_ep:
                    ppo_tech_first_ep[k] = ep

        total_ppo_tech += res["ppo_total_tech"]

        # Opponent 科技升级
        if res["opp_total_tech"] > 0:
            battles_with_opp_tech += 1
            opp_tech_by_ep_bin[ep_bin] += res["opp_total_tech"]
            for k, v in res["opp_tech_upgrades"].items():
                opp_tech_types[k] += v

        total_opp_tech += res["opp_total_tech"]

    # ===== 输出 =====

    print("=" * 60)
    print("一、科技升级分析")
    print("=" * 60)

    print(f"\nPPO 科技升级:")
    print(f"  出现局数: {battles_with_ppo_tech}/{len(files)} ({battles_with_ppo_tech/len(files)*100:.1f}%)")
    print(f"  总升级次数: {total_ppo_tech}")
    print(f"  局均: {total_ppo_tech/len(files):.2f}")
    for k, v in sorted(ppo_tech_types.items(), key=lambda x: -x[1]):
        label = TECH_LABELS.get(k, k)
        first = ppo_tech_first_ep.get(k, "?")
        print(f"    {label}: {v} 次 (首次 Ep {first})")

    print(f"\n对手科技升级:")
    print(f"  出现局数: {battles_with_opp_tech}/{len(files)} ({battles_with_opp_tech/len(files)*100:.1f}%)")
    print(f"  总升级次数: {total_opp_tech}")
    print(f"  局均: {total_opp_tech/len(files):.2f}")
    for k, v in sorted(opp_tech_types.items(), key=lambda x: -x[1]):
        label = TECH_LABELS.get(k, k)
        print(f"    {label}: {v} 次")

    print(f"\nPPO 胜率对比:")
    print(f"  有科技升级局: {ppo_wins_with_tech}/{battles_with_ppo_tech} ({ppo_wins_with_tech/battles_with_ppo_tech*100:.1f}% 胜率)" if battles_with_ppo_tech > 0 else "  有科技升级局: 0")
    print(f"  全部局:       {total_ppo_wins}/{len(files)} ({total_ppo_wins/len(files)*100:.1f}% 胜率)")

    print(f"\n科技升级随轮次分布 (每 200 轮):")
    for ep_bin in range(0, 2000, 200):
        ppo = ppo_tech_by_ep_bin.get(ep_bin, 0)
        opp = opp_tech_by_ep_bin.get(ep_bin, 0)
        battles = sum(1 for r in all_results if (r["ep"] // 200) * 200 == ep_bin)
        if battles > 0:
            print(f"  Ep {ep_bin:>4d}-{ep_bin+200:<4d}: PPO {ppo:>3d} 次 (局均 {ppo/battles:.2f}), 对手 {opp:>3d} 次 (局均 {opp/battles:.2f})")

    print()
    print("=" * 60)
    print("二、超级武器分析")
    print("=" * 60)

    print(f"\nPPO 超级武器使用:")
    print(f"  出现局数: {battles_with_ppo_sw}/{len(files)} ({battles_with_ppo_sw/len(files)*100:.1f}%)")
    print(f"  总使用次数: {total_ppo_sw}")
    print(f"  局均: {total_ppo_sw/len(files):.2f}")
    if ppo_sw_types:
        for k, v in sorted(ppo_sw_types.items(), key=lambda x: -x[1]):
            label = SUPER_WEAPON_LABELS.get(k, k)
            first = ppo_sw_first_ep.get("first_use", "?")
            print(f"    {label}: {v} 次 (首次 Ep {first})")
    else:
        print("    (PPO 未使用任何超级武器)")

    # 备注
    print(f"\n备注:")
    print(f"  - PPO 动作存为整数 action 索引 (Bundle 编号)，无法直接解析具体操作")
    print(f"  - 超级武器检测依赖 reward_detail.action_reward 中的分量字段")
    print(f"  - 对手 (baseline agent) 的动作同样为整数索引，非 PPO 方无法解析")
    print(f"  - 科技升级检测通过 round_details.our_tech/enemy_tech 等级变化，准确可靠")

    print()
    print("=" * 60)
    print("三、综合趋势 (每 200 轮)")
    print("=" * 60)

    print(f"{'轮次范围':>12s} | {'PPO科技':>8s} | {'PPO科技局%':>9s} | {'对手科技':>8s} | {'PPO超武':>8s} | {'PPO胜率':>8s}")
    print("-" * 66)
    for ep_bin in range(0, 2000, 200):
        battles = sum(1 for r in all_results if (r["ep"] // 200) * 200 == ep_bin)
        if battles == 0:
            continue
        ppo_t = ppo_tech_by_ep_bin.get(ep_bin, 0)
        opp_t = opp_tech_by_ep_bin.get(ep_bin, 0)
        ppo_s = ppo_sw_by_ep_bin.get(ep_bin, 0)
        tech_battle_pct = tech_ep_battle_count.get(ep_bin, 0) / battles * 100
        wins = sum(1 for r in all_results if (r["ep"] // 200) * 200 == ep_bin and r["ppo_won"] > 0)
        print(f"  Ep {ep_bin:>3d}-{ep_bin+200:<4d} | {ppo_t:>8d} | {tech_battle_pct:>8.1f}% | {opp_t:>8d} | {ppo_s:>8d} | {wins/battles*100:>7.1f}%")

    print()
    print("=" * 60)
    print("四、结论")
    print("=" * 60)

    print(f"\n科技升级 (PPO):  {total_ppo_tech} 次 / {battles_with_ppo_tech} 局 ({battles_with_ppo_tech/len(files)*100:.1f}%)")
    print(f"科技升级 (对手):  {total_opp_tech} 次 / {battles_with_opp_tech} 局 ({battles_with_opp_tech/len(files)*100:.1f}%)")
    print(f"超级武器 (PPO):   {total_ppo_sw} 次 / {battles_with_ppo_sw} 局 ({battles_with_ppo_sw/len(files)*100:.1f}%)")

    if total_ppo_tech > 0 or total_opp_tech > 0:
        print(f"\n科技升级比率 (PPO/对手): {total_ppo_tech}/{total_opp_tech} = {total_ppo_tech/total_opp_tech:.2f}x" if total_opp_tech > 0 else f"\n科技升级仅 PPO 方有")
    if total_ppo_sw > 0:
        print(f"\nPPO 超武使用频次: 每 {len(files)/total_ppo_sw:.1f} 局 1 次")

    if ppo_tech_types:
        print(f"\nPPO 科技偏好: {', '.join(f'{TECH_LABELS.get(k,k)}({v})' for k,v in sorted(ppo_tech_types.items(), key=lambda x:-x[1]))}")
    if opp_tech_types:
        print(f"对手科技偏好: {', '.join(f'{TECH_LABELS.get(k,k)}({v})' for k,v in sorted(opp_tech_types.items(), key=lambda x:-x[1]))}")


if __name__ == "__main__":
    main()
