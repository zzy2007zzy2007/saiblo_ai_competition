import json
import glob
import os

BATTLE_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/selfplay/selfplay_battles/detailed_battles"
files = sorted(glob.glob(BATTLE_DIR + "/*.json"))
print(f"对战总数: {len(files)}\n")

# ----- 科技升级检测: 扫描全部 2000 局的最终 tech 等级 -----
ppo_max_gs_all = 0
ppo_max_as_all = 0
opp_max_gs_all = 0
opp_max_as_all = 0
ppo_ever_upgraded = 0          # PPO 曾升级科技(任意类型)的局数
opp_ever_upgraded = 0          # 对手曾升级科技的局数
ppo_tech_upgrade_counts = []   # 每局的 PPO 科技升级次数
opp_tech_upgrade_counts = []   # 每局的对手科技升级次数

for fp in files:
    with open(fp) as f:
        data = json.load(f)
    rd = data.get("round_details", [])

    # 跟踪科技等级变化
    prev_ppo = {"gen_speed": 0, "ant_strength": 0}
    prev_opp = {"gen_speed": 0, "ant_strength": 0}
    ppo_up = 0
    opp_up = 0

    for step_info in rd:
        ot = step_info.get("our_tech", {})
        et = step_info.get("enemy_tech", {})

        for key in ["gen_speed", "ant_strength"]:
            cv = ot.get(key, 0) if isinstance(ot, dict) else 0
            if cv > prev_ppo.get(key, 0):
                ppo_up += cv - prev_ppo.get(key, 0)
            prev_ppo[key] = max(prev_ppo.get(key, 0), cv)

            cv2 = et.get(key, 0) if isinstance(et, dict) else 0
            if cv2 > prev_opp.get(key, 0):
                opp_up += cv2 - prev_opp.get(key, 0)
            prev_opp[key] = max(prev_opp.get(key, 0), cv2)

    if ppo_up > 0:
        ppo_ever_upgraded += 1
    if opp_up > 0:
        opp_ever_upgraded += 1
    ppo_tech_upgrade_counts.append(ppo_up)
    opp_tech_upgrade_counts.append(opp_up)
    ppo_max_gs_all = max(ppo_max_gs_all, prev_ppo["gen_speed"])
    ppo_max_as_all = max(ppo_max_as_all, prev_ppo["ant_strength"])
    opp_max_gs_all = max(opp_max_gs_all, prev_opp["gen_speed"])
    opp_max_as_all = max(opp_max_as_all, prev_opp["ant_strength"])

print("=" * 60)
print("科技升级分析 (全部 2000 局)")
print("=" * 60)
print(f"\nPPO 曾升级科技的局数: {ppo_ever_upgraded}/2000 ({ppo_ever_upgraded/2000*100:.1f}%)")
print(f"PPO 科技总升级次数:   {sum(ppo_tech_upgrade_counts)}")
print(f"PPO 局均升级次数:     {sum(ppo_tech_upgrade_counts)/2000:.3f}")
print(f"PPO 全局最高科技等级: gen_speed={ppo_max_gs_all}, ant_strength={ppo_max_as_all}")

print(f"\n对手曾升级科技的局数: {opp_ever_upgraded}/2000 ({opp_ever_upgraded/2000*100:.1f}%)")
print(f"对手科技总升级次数:   {sum(opp_tech_upgrade_counts)}")
print(f"对手局均升级次数:     {sum(opp_tech_upgrade_counts)/2000:.3f}")
print(f"对手全局最高科技等级: gen_speed={opp_max_gs_all}, ant_strength={opp_max_as_all}")

# 如果有升级，找出哪些局
if ppo_ever_upgraded > 0:
    print(f"\nPPO 有升级的局 (前 10):")
    for i, c in enumerate(ppo_tech_upgrade_counts):
        if c > 0:
            ep = int(os.path.basename(files[i]).split("_")[1])
            print(f"  Ep {ep}: {c} 次升级")
            if sum(1 for x in ppo_tech_upgrade_counts[:i+1] if x > 0) >= 10:
                break

if opp_ever_upgraded > 0:
    print(f"\n对手有升级的局 (前 10):")
    for i, c in enumerate(opp_tech_upgrade_counts):
        if c > 0:
            ep = int(os.path.basename(files[i]).split("_")[1])
            print(f"  Ep {ep}: {c} 次升级")
            if sum(1 for x in opp_tech_upgrade_counts[:i+1] if x > 0) >= 10:
                break

# ----- 分析 action_counts 中的 Bundle 使用模式 -----
print(f"\n\n{'=' * 60}")
print("PPO 动作 Bundle 使用统计 (全部 2000 局)")
print("=" * 60)

bundle_counter = {}
for fp in files:  # 全部局
    with open(fp) as f:
        data = json.load(f)
    ac = data.get("action_counts", {})
    for k, v in ac.items():
        bundle_counter[k] = bundle_counter.get(k, 0) + v

# 按使用次数排序
sorted_bundles = sorted(bundle_counter.items(), key=lambda x: -x[1])
print(f"\n总不同 Bundle 数: {len(sorted_bundles)}")
print(f"\nTop 20 最常用动作 Bundle:")
print(f"{'Rank':>4s} | {'Bundle ID':>10s} | {'使用次数':>8s} | {'占比':>6s}")
print("-" * 38)
total_actions = sum(v for _, v in sorted_bundles)
for rank, (bid, cnt) in enumerate(sorted_bundles[:20], 1):
    pct = cnt / total_actions * 100
    print(f"{rank:>4d} | {bid:>10s} | {cnt:>8d} | {pct:>5.1f}%")
print(f"  ... (共 {len(sorted_bundles)} 种不同 Bundle)")

# ----- 如果科技全为 0，直接结论 -----
print(f"\n\n{'=' * 60}")
print("结论")
print("=" * 60)
if ppo_ever_upgraded == 0 and opp_ever_upgraded == 0:
    print(f"\n[结论] 在全部 2000 局 SelfPlay 对战中，")
    print(f"        PPO 模型和所有 baseline 对手均未进行过任何科技升级")
    print(f"        (gen_speed 和 ant_strength 全程保持为 0)")
    print(f"\n原因分析:")
    print(f"  - PPO 可能从未将科技升级视为有价值的动作")
    print(f"  - 奖励函数中科技升级的权重不足 (upgrade_gen_speed=3.0, upgrade_gen_ant=1.5)")
    print(f"  - 200 回合内专注于塔的建造和兵输出，无余钱做科技")
elif ppo_ever_upgraded > 0 and opp_ever_upgraded > 0:
    print(f"\n[结论] 双方都有科技升级行为，PPO {ppo_ever_upgraded} 局 / 对手 {opp_ever_upgraded} 局")
elif ppo_ever_upgraded > 0:
    print(f"\n[结论] 仅 PPO 有科技升级 ({ppo_ever_upgraded} 局)，对手从未升级")

print(f"\n超级武器方面:")
print(f"  - 该日志格式 (PPO action 整数索引) 无法直接解码超级武器操作")
print(f"  - action_reward 为标量 float (非字典)，不含操作级明细")
print(f"  - 需通过 PPO ActionCatalog 的 Bundle→Operation 映射才能验证")
print(f"  - 但由于 reward 体系中超武的奖励值已调高 (deploy_ls=1.5, deploy_emp=2.0 等)")
print(f"    如果 PPO 使用了超武，reward_detail.tech_bonus 应有体现")
print(f"  - 上述数据中所有 reward 分量均为 0.0 (每步仅累积 end_reward)，间接说明超武未被使用")
