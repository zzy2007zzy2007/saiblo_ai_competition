import json, sys, math

d = json.load(sys.stdin)

print(f"{'个体ID':20s} | vs RandomAI p | vs TowerAI p | vs MediumAI p")
print("-" * 70)

pairs = []
for ind_id, refs in sorted(d.items()):
    p_random = refs["BasicRandomAI"]["agent1_wins"] / 50
    p_tower = refs["BasicTowerAI"]["agent1_wins"] / 50
    p_medium = refs["MediumRuleAI"]["agent1_wins"] / 50
    print(f"{ind_id:20s} | {p_random:.2f}          | {p_tower:.2f}         | {p_medium:.2f}")

    for ref_name, base_elo in [("BasicRandomAI", 400), ("BasicTowerAI", 1000), ("MediumRuleAI", 1600)]:
        wins = refs[ref_name]["agent1_wins"]
        p_obs = wins / 50
        if 0 < p_obs < 1:
            diff = -400 * math.log10(1/p_obs - 1)
        elif p_obs == 1.0:
            diff = 400
        else:
            diff = -400
        pairs.append((diff, p_obs, base_elo, ref_name, ind_id))

# Sort by ELO diff
pairs.sort(key=lambda x: x[0])

print()
print(f"{'ELO差':>8s} | {'实测p':>6s} | {'理论p':>6s} | {'偏差':>6s} | 参照物          | 个体")
print("-" * 72)
for elo_diff, p_obs, base_elo, ref_name, ind_id in pairs:
    p_theory = 1.0 / (1.0 + 10**(-elo_diff/400))
    deviation = p_obs - p_theory
    print(f"{elo_diff:8.0f} | {p_obs:6.2f} | {p_theory:6.2f} | {deviation:+6.2f} | {ref_name:16s} | {ind_id}")

# Summary stats
deviations = [p_obs - (1.0/(1.0+10**(-diff/400))) for diff, p_obs, _, _, _ in pairs]
abs_devs = [abs(d) for d in deviations]
print()
print("=" * 50)
print("偏差统计 (实测p - 理论p):")
print(f"  平均偏差: {sum(deviations)/len(deviations):.4f}")
print(f"  平均绝对偏差: {sum(abs_devs)/len(abs_devs):.4f}")
print(f"  最大绝对偏差: {max(abs_devs):.4f}")
print(f"  RMS偏差: {math.sqrt(sum(d*d for d in deviations)/len(deviations)):.4f}")

# Group by |deviation|
print()
print("偏差分布:")
bins = [0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
for i, threshold in enumerate(bins):
    count = sum(1 for d in abs_devs if d <= threshold and (i == 0 or d > bins[i-1]))
    lo = 0 if i == 0 else bins[i-1]
    print(f"  |偏差| in [{lo:.2f}, {threshold:.2f}]: {count} 个")

# Key insight: how does p distribute across the population?
print()
print("=" * 50)
print("p 值的种群分布:")
for ref_name in ["BasicRandomAI", "BasicTowerAI", "MediumRuleAI"]:
    p_vals = [refs[ref_name]["agent1_wins"]/50 for _, refs in sorted(d.items())]
    print(f"  {ref_name}: min={min(p_vals):.2f}, max={max(p_vals):.2f}, mean={sum(p_vals)/len(p_vals):.2f}, std={math.sqrt(sum((p-sum(p_vals)/len(p_vals))**2 for p in p_vals)/len(p_vals)):.3f}")
