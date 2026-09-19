#!/usr/bin/env python3
"""根因调查数据采集脚本 - 一次性收集 Tasks B~E 所需的所有远程数据"""
import json
import statistics

BASE = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260527_082648"
METRICS = f"{BASE}/training/training_metrics_history.jsonl"
PER_EP = f"{BASE}/training/per_episode_stats.jsonl"
OPP_POOL = f"{BASE}/checkpoint/league_state/opponent_pool.json"
PAYOFF = f"{BASE}/checkpoint/league_state/payoff.json"

import os

# ================== 读取数据 ==================
def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

metrics = load_jsonl(METRICS)
per_ep = load_jsonl(PER_EP)
pool = json.load(open(OPP_POOL)) if os.path.exists(OPP_POOL) else {}
payoff = json.load(open(PAYOFF)) if os.path.exists(PAYOFF) else {}

# ================== Task B: 奖励分量分析 ==================
print("=" * 70)
print("【Task B】奖励分量分析")
print("=" * 70)

rw_keys = sorted([k for k in metrics[0] if k.startswith("rw_")])
print(f"奖励分量字段: {rw_keys}")

# 按阶段分组
n = len(metrics)
phases = {"early(0~33%)": metrics[:n//3], "mid(33~66%)": metrics[n//3:2*n//3], "late(66~100%)": metrics[2*n//3:]}

for phase_name, phase_data in phases.items():
    print(f"\n  [{phase_name}]")
    total_rw = sum(e.get("avg_reward", 0) for e in phase_data)
    for rk in rw_keys:
        vals = [e.get(rk, 0) for e in phase_data]
        avg = sum(vals) / len(vals)
        pct = avg * 100 / (total_rw / len(phase_data)) if total_rw > 0 else 0
        print(f"    {rk:20s}  avg={avg:>8.4f}  pct_of_total={pct:>6.2f}%")

# 各分量趋势
print(f"\n  各分量趋势 (late - early):")
early_vals = {rk: sum(e.get(rk, 0) for e in phases["early(0~33%)"]) / len(phases["early(0~33%)"]) for rk in rw_keys}
late_vals = {rk: sum(e.get(rk, 0) for e in phases["late(66~100%)"]) / len(phases["late(66~100%)"]) for rk in rw_keys}
for rk in rw_keys:
    diff = late_vals[rk] - early_vals[rk]
    trend = "↑上升" if diff > 0.01 else ("↓下降" if diff < -0.01 else "→持平")
    print(f"    {rk:20s}  early={early_vals[rk]:>8.4f}  late={late_vals[rk]:>8.4f}  diff={diff:>+8.4f}  {trend}")

# ================== Task E: 梯度/熵/Value Loss 分析 ==================
print("\n" + "=" * 70)
print("【Task E】PPO 训练稳定性分析")
print("=" * 70)

grads = [e.get("gradient_norm", 0) for e in metrics]
grad_v = [e.get("grad_norm_value", 0) for e in metrics]
vloss = [e.get("value_loss", 0) for e in metrics]
entropy = [e.get("entropy", 0) for e in metrics]
ent_coefs = [e.get("entropy_coef", 0) for e in metrics]
clip_frac = [e.get("clip_fraction", 0) for e in metrics]
vi_mean = [e.get("value_input_mean", 0) for e in metrics]
vi_std = [e.get("value_input_std", 0) for e in metrics]

print(f"\n  梯度分析:")
n_total = len(grads)
print(f"    总更新次数: {n_total}")
for threshold, label in [(50, ">50"), (100, ">100"), (200, ">200"), (500, ">500"), (1000, ">1000")]:
    count = sum(1 for g in grads if g > threshold)
    print(f"    gradient_norm {label}: {count}/{n_total} ({100*count/n_total:.1f}%)")
print(f"    梯度均值: {sum(grads)/n_total:.1f}")
print(f"    梯度中位数: {sorted(grads)[n_total//2]:.1f}")
print(f"    梯度峰值: {max(grads):.1f}")

# 梯度分段趋势
seg_size = n_total // 5
print(f"\n  梯度分段趋势:")
for si in range(5):
    seg = grads[si*seg_size:(si+1)*seg_size]
    print(f"    段{si+1}(ep{metrics[si*seg_size].get('episode',0)}~{metrics[min((si+1)*seg_size-1, n_total-1)].get('episode',0)}): mean={sum(seg)/len(seg):.1f}  max={max(seg):.1f}  >200={sum(1 for g in seg if g>200)}/{len(seg)} ({100*sum(1 for g in seg if g>200)/len(seg):.1f}%)")

# Value Loss
print(f"\n  Value Loss 分析:")
print(f"    初始 value_loss: {vloss[0]:.4f}")
print(f"    最终 value_loss: {vloss[-1]:.4f}")
print(f"    value_loss 均值: {sum(vloss)/len(vloss):.4f}")
vl_first = vloss[:n_total//2]
vl_last = vloss[n_total//2:]
print(f"    前半段均值: {sum(vl_first)/len(vl_first):.4f} → 后半段均值: {sum(vl_last)/len(vl_last):.4f}")

# value_input 分布
print(f"\n  Value Input 分布:")
print(f"    initial: mean={vi_mean[0]:.4f}  std={vi_std[0]:.4f}")
print(f"    final:   mean={vi_mean[-1]:.4f}  std={vi_std[-1]:.4f}")
vi_first = [vi_std[i] for i in range(n_total//2)]
vi_last = [vi_std[i] for i in range(n_total//2, n_total)]
print(f"    std 前半段均值: {sum(vi_first)/len(vi_first):.4f} → 后半段均值: {sum(vi_last)/len(vi_last):.4f}")

# Entropy
print(f"\n  Entropy 分析:")
print(f"    初始 entropy: {entropy[0]:.4f}")
print(f"    最终 entropy: {entropy[-1]:.4f}")
print(f"    entropy 均值: {sum(entropy)/len(entropy):.4f}")
print(f"    entropy 最低值: {min(entropy):.4f}")
# entropy 骤降检测
for i in range(1, len(entropy)):
    if entropy[i] < entropy[i-1] * 0.6:
        print(f"    ⚠ 骤降: Ep {metrics[i].get('episode',0)} 从 {entropy[i-1]:.4f} 跌到 {entropy[i]:.4f}")

# EntCoef adaptive 触发
default_ec = ent_coefs[0]
adaptive_triggers = sum(1 for ec in ent_coefs if ec > default_ec * 1.5)
print(f"    Adaptive ent_coef 触发次数: {adaptive_triggers}/{n_total} ({100*adaptive_triggers/n_total:.1f}%)")
print(f"    EntCoef 初始: {ent_coefs[0]:.4f}  峰值: {max(ent_coefs):.4f}  最终: {ent_coefs[-1]:.4f}")

# Clip fraction
print(f"\n  Clip Fraction 分析:")
print(f"    clip_fraction 均值: {sum(clip_frac)/len(clip_frac):.4f}")
high_clip = sum(1 for cf in clip_frac if cf > 0.5)
print(f"    clip_fraction > 0.5: {high_clip}/{n_total} ({100*high_clip/n_total:.1f}%)")
print(f"    clip_fraction 峰值: {max(clip_frac):.4f}")

# ================== Task B/C: 动作分布 ==================
print("\n" + "=" * 70)
print("【Task B/C】动作分布")
print("=" * 70)
type_keys = sorted([k for k in metrics[0] if k.startswith("type_") and "valid" not in k])
print(f"{'动作':>20s} {'整体':>8s} {'前1/3':>8s} {'中1/3':>8s} {'后1/3':>8s} {'趋势':>6s}")
print("-" * 60)
n3 = n // 3
for tk in type_keys:
    vals = [e.get(tk, 0) for e in metrics]
    avg = sum(vals)/n
    f3 = sum(vals[:n3])/n3
    m3 = sum(vals[n3:2*n3])/n3
    l3 = sum(vals[-n3:])/n3
    if l3 > f3 * 1.1:
        trend = "↑"
    elif l3 < f3 * 0.9:
        trend = "↓"
    else:
        trend = "→"
    print(f"{tk.replace('type_',''):>20s} {avg:>8.2f} {f3:>8.2f} {m3:>8.2f} {l3:>8.2f} {trend:>6s}")

# NO-OP 与其他动作的相关性
print(f"\n  NO-OP 与其他动作的相关性分析:")
noop_vals = [e.get("type_noop", 0) for e in metrics]
for tk in type_keys:
    if tk == "type_noop":
        continue
    tk_vals = [e.get(tk, 0) for e in metrics]
    if sum(tk_vals) == 0:
        print(f"    {tk.replace('type_',''):>20s}:  始终为 0, 无法计算相关系数")
        continue
    # Pearson correlation
    n_corr = len(noop_vals)
    mean_noop = sum(noop_vals)/n_corr
    mean_tk = sum(tk_vals)/n_corr
    num = sum((noop_vals[i]-mean_noop)*(tk_vals[i]-mean_tk) for i in range(n_corr))
    den = (sum((v-mean_noop)**2 for v in noop_vals) * sum((v-mean_tk)**2 for v in tk_vals))**0.5
    corr = num/den if den > 0 else 0
    print(f"    {tk.replace('type_',''):>20s}:  r={corr:>+.4f}")

# ================== Task C: 金币峰值分析 ==================
print("\n" + "=" * 70)
print("【Task C】金币峰值分析")
print("=" * 70)
# 从 detailed_battles 中采样分析金币峰值
import glob
detail_dir = f"{BASE}/selfplay/selfplay_battles/detailed_battles/"
detail_files = sorted(os.listdir(detail_dir)) if os.path.exists(detail_dir) else []
print(f"详细对战文件数: {len(detail_files)}")
if detail_files:
    # 采样分析早期、中期、后期的对战
    samples = []
    for idx in [0, len(detail_files)//4, len(detail_files)//2, 3*len(detail_files)//4, len(detail_files)-1]:
        try:
            with open(os.path.join(detail_dir, detail_files[idx])) as f:
                data = json.load(f)
            rds = data.get("round_details", [])
            our_coins = [r.get("our_coins", 0) for r in rds]
            max_coin = max(our_coins) if our_coins else 0
            ge_60 = sum(1 for c in our_coins if c >= 60)
            ge_200 = sum(1 for c in our_coins if c >= 200)
            total = len(our_coins)
            samples.append({
                "file": detail_files[idx],
                "ep": data.get("episode", "?"),
                "opponent": data.get("opponent_id", "?"),
                "result": data.get("result", "?"),
                "total_steps": total,
                "max_our_coins": max_coin,
                "pct_ge_60": 100*ge_60/total if total else 0,
                "pct_ge_200": 100*ge_200/total if total else 0,
                "our_cumulative": data.get("our_cumulative_coins", 0),
                "enemy_cumulative": data.get("enemy_cumulative_coins", 0),
            })
        except Exception as ex:
            print(f"  读取失败 {detail_files[idx]}: {ex}")
    if samples:
        print(f"\n  采样分析 (5 场对战, 从早到晚):")
        print(f"  {'Ep':>6s} {'对手':>20s} {'结果':>6s} {'步数':>5s} {'最大余额':>8s} {'>=60%':>7s} {'>=200%':>8s} {'我累计':>7s} {'敌累计':>7s}")
        print(f"  {'-'*84}")
        for s in samples:
            opp_short = s["opponent"][:18] if len(s["opponent"]) > 18 else s["opponent"]
            print(f"  {s['ep']:>6d} {opp_short:>20s} {s['result']:>6s} {s['total_steps']:>5d} {s['max_our_coins']:>8d} {s['pct_ge_60']:>6.1f}% {s['pct_ge_200']:>7.1f}% {s['our_cumulative']:>7.0f} {s['enemy_cumulative']:>7.0f}")
        
        # 汇总统计
        all_max = [s["max_our_coins"] for s in samples]
        all_ge60 = [s["pct_ge_60"] for s in samples]
        all_cum = [s["our_cumulative"] for s in samples]
        print(f"\n  采样汇总:")
        print(f"    最大余额: max={max(all_max)}, min={min(all_max)}, avg={sum(all_max)/len(all_max):.0f}")
        print(f"    步数中余额>=60: avg={sum(all_ge60)/len(all_ge60):.1f}%")
        print(f"    累计金币: avg={sum(all_cum)/len(all_cum):.0f}")

# ================== Task D: 经济流向分析 ==================
print("\n" + "=" * 70)
print("【Task D】经济流向分析")
print("=" * 70)
cum_our = [e.get("avg_our_cumulative_coins", 0) for e in metrics]
cum_enemy = [e.get("avg_enemy_cumulative_coins", 0) for e in metrics]
bal_our = [e.get("avg_our_coins", 0) for e in metrics]
bal_enemy = [e.get("avg_enemy_coins", 0) for e in metrics]
rw_cg = [e.get("rw_coin_gain", 0) for e in metrics]

print(f"  累计金币 (ours): mean={sum(cum_our)/len(cum_our):.1f}  min={min(cum_our):.1f}  max={max(cum_our):.1f}")
print(f"  累计金币 (enemy): mean={sum(cum_enemy)/len(cum_enemy):.1f}  min={min(cum_enemy):.1f}  max={max(cum_enemy):.1f}")

# 累计金币差值趋势
diff = [cum_our[i] - cum_enemy[i] for i in range(n)]
print(f"  累计金币差 (our-enemy): mean={sum(diff)/len(diff):.1f}")
diff_first = diff[:n//2]
diff_last = diff[n//2:]
print(f"    前半段均值: {sum(diff_first)/len(diff_first):.2f} → 后半段均值: {sum(diff_last)/len(diff_last):.2f}")
print(f"    差值为正的比例: {sum(1 for d in diff if d>0)}/{n} ({100*sum(1 for d in diff if d>0)/n:.1f}%)")

# 余额差值趋势
bal_diff = [bal_our[i] - bal_enemy[i] for i in range(n)]
print(f"\n  余额差 (our-enemy): mean={sum(bal_diff)/len(bal_diff):.2f}")
print(f"    差值为正的比例: {sum(1 for d in bal_diff if d>0)}/{n} ({100*sum(1 for d in bal_diff if d>0)/n:.1f}%)")

# rw_coin_gain 趋势
print(f"\n  rw_coin_gain (金币增益奖励):")
rw_cg_first = rw_cg[:n//2]
rw_cg_last = rw_cg[n//2:]
print(f"    前半段: {sum(rw_cg_first)/len(rw_cg_first):.2f} → 后半段: {sum(rw_cg_last)/len(rw_cg_last):.2f}")

# ================== Task A: 对手池分析 ==================
print("\n" + "=" * 70)
print("【Task A】对手池分析")
print("=" * 70)
if pool:
    print(f"  对手池大小: {pool.get('max_size', '?')}")
    print(f"  对手数量: {len(pool.get('opponent_ids', []))}")
    print(f"  Min games threshold: {pool.get('min_games_threshold', '?')}")
    print(f"\n  对手列表:")
    oids = pool.get('opponent_ids', [])
    games = pool.get('games_played', {})
    eps = pool.get('added_episodes', {})
    print(f"  {'ID':>30s} {'Games':>6s} {'Added Ep':>10s}")
    print(f"  {'-'*48}")
    for oid in oids:
        g = games.get(oid, '?')
        ep = eps.get(oid, '?')
        short = oid[:28] if len(oid) > 28 else oid
        print(f"  {short:>30s} {str(g):>6s} {str(ep):>10s}")
else:
    print("  opponent_pool.json 不存在")

if payoff:
    print(f"\n  TrueSkill 评分:")
    ratings = payoff.get("trueskill_ratings", {})
    print(f"  {'Player':>30s} {'mu':>8s} {'sigma':>8s} {'Games':>6s}")
    print(f"  {'-'*56}")
    for pid, rating in sorted(ratings.items(), key=lambda x: x[1].get("mu", 0), reverse=True):
        short = pid[:28] if len(pid) > 28 else pid
        mu = rating.get("mu", 0)
        sigma = rating.get("sigma", 0)
        gp = rating.get("games_played", 0)
        print(f"  {short:>30s} {mu:>8.2f} {sigma:>8.2f} {gp:>6d}")

# ================== 学习率进度 ==================
print("\n" + "=" * 70)
print("【辅助】学习率进度")
print("=" * 70)
lrs = [e.get("learning_rate", 0) for e in metrics]
print(f"  初始 LR: {lrs[0]:.10f}")
print(f"  最终 LR: {lrs[-1]:.10f}")
lr_gt_zero = sum(1 for lr in lrs if lr > 0)
print(f"  LR > 0 的更新次数: {lr_gt_zero}/{n_total}")
lr_gt_pct1 = sum(1 for lr in lrs if lr > lrs[0] * 0.01)
print(f"  LR > 1%初始的更新次数: {lr_gt_pct1}/{n_total}")

# 找到 LR 归零的节点
for i, lr in enumerate(lrs):
    if lr < 1e-10:
        print(f"  LR 归零于更新 #{i+1} (Ep ~{metrics[i].get('episode',0)})")
        break
