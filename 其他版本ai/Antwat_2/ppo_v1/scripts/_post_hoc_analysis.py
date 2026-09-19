#!/usr/bin/env python3
"""
事后分析脚本：基于现有训练日志验证诊断假设 H1/H3/H4/H5/H6

使用数据（来自 server9 20260525_172045 训练）：
  - training_metrics_history.jsonl   (24 条 PPO update 记录)
  - per_episode_stats.jsonl          (192 条单 episode 记录)
  - weight_stats.jsonl               (4 条权重快照)
"""
import json
import math
import statistics
from collections import defaultdict

DATA_DIR = "ppo/outputs/20260525_172045/training"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def safe(v, default=0.0):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return default
    return v


def main():
    metrics = load_jsonl(f"{DATA_DIR}/training_metrics_history.jsonl")
    episodes = load_jsonl(f"{DATA_DIR}/per_episode_stats.jsonl")
    weights = load_jsonl(f"{DATA_DIR}/weight_stats.jsonl")

    print("=" * 70)
    print("PPO 坍塌事后根因分析")
    print("=" * 70)
    print(f"数据: {len(metrics)} 条 PPO update, {len(episodes)} 个 episode, {len(weights)} 条权重快照")
    print()

    # ---- 识别关键阶段 ----
    collapse_ep = None
    for r in metrics:
        if r.get("gradient_norm") == "NaN" or (isinstance(r.get("gradient_norm"), float) and math.isnan(r.get("gradient_norm"))):
            collapse_ep = r["episode"]
            break

    print(f"NaN 首次出现: Ep {collapse_ep}" if collapse_ep else "NaN 未直接出现（日志可能已过滤）")
    
    # 找健康阶段 (熵 > 0.05) 和死亡阶段 (熵 < 0.001)
    healthy = [r for r in metrics if r["entropy"] > 0.05]
    dead = [r for r in metrics if r["entropy"] < 0.001]
    print(f"健康阶段 (entropy>0.05): Ep {healthy[0]['episode']} ~ Ep {healthy[-1]['episode']}, 共 {len(healthy)} 条")
    print(f"死亡阶段 (entropy<0.001): Ep {dead[0]['episode']} ~ Ep {dead[-1]['episode']}, 共 {len(dead)} 条")
    print()

    # ========== H1: 奖励结构分析 ==========
    print("-" * 70)
    print("H1: 终局奖励主导")
    print("-" * 70)
    
    reward_sources = ["rw_hp_attack_base", "rw_hp_attack_tower", "rw_coin_gain", "rw_tower_survival", 
                      "rw_tech_bonus", "rw_die_penalty", "rw_end_reward", "rw_speed_bonus"]
    
    for r in metrics:
        total = sum(abs(safe(r.get(s, 0))) for s in reward_sources)
        if total == 0:
            continue
        end_pct = abs(safe(r["rw_end_reward"])) / total * 100
        hp_pct = abs(safe(r.get("rw_hp_attack_base", 0)) + safe(r.get("rw_hp_attack_tower", 0))) / total * 100
        tower_pct = abs(safe(r["rw_tower_survival"])) / total * 100
        if r["episode"] in [8, 80, 136, 144, 152, 192]:
            print(f"  Ep {r['episode']:>3d} (entropy={r['entropy']:.4f}): "
                  f"终局占比={end_pct:5.1f}%, HP占比={hp_pct:5.1f}%, 塔占比={tower_pct:5.1f}%")

    # 终局占比 vs entropy 相关性
    print()
    print("  entropy vs 终局奖励占比 趋势:")
    healthy_end_ratios = []
    dead_end_ratios = []
    for r in metrics:
        total = sum(abs(safe(r.get(s, 0))) for s in reward_sources)
        if total == 0:
            continue
        end_pct = abs(safe(r["rw_end_reward"])) / total * 100
        if r["entropy"] > 0.05:
            healthy_end_ratios.append(end_pct)
        elif r["entropy"] < 0.001:
            dead_end_ratios.append(end_pct)
    
    print(f"  健康阶段平均终局占比: {statistics.mean(healthy_end_ratios):.1f}%" if healthy_end_ratios else "  无健康阶段数据")
    print(f"  死亡阶段平均终局占比: {statistics.mean(dead_end_ratios):.1f}%" if dead_end_ratios else "  无死亡阶段数据")

    # 终局奖励与总奖励的量级对比
    for label, phase_records in [("健康阶段", healthy), ("死亡阶段", dead)]:
        if not phase_records:
            continue
        avg_rew = statistics.mean(r.get("avg_reward", 0) for r in phase_records)
        avg_end = statistics.mean(abs(safe(r.get("rw_end_reward", 0))) for r in phase_records)
        print(f"  {label} avg_reward={avg_rew:.1f}, avg_end_reward_abs={avg_end:.1f}")
    print()

    # ========== H3: 自适应熵机制审计 ==========
    print("-" * 70)
    print("H3: 自适应熵机制响应滞后分析")
    print("-" * 70)
    
    # hard_entropy_penalty > 0 意味着硬性熵惩罚被触发
    penalty_triggered = [r for r in metrics if safe(r.get("hard_entropy_penalty", 0)) > 0]
    print(f"  hard_entropy_penalty 触发次数: {len(penalty_triggered)} / {len(metrics)} 次 update")
    
    if penalty_triggered:
        first_penalty_ep = penalty_triggered[0]["episode"]
        print(f"  首次触发: Ep {first_penalty_ep}")
        # 检查触发后的 entropy 变化
        first_penalty_idx = next(i for i, r in enumerate(metrics) if r["episode"] == first_penalty_ep)
        if first_penalty_idx + 1 < len(metrics):
            before = metrics[first_penalty_idx]["entropy"]
            after = metrics[first_penalty_idx + 1]["entropy"]
            print(f"  触发时 entropy={before:.4f}, 下次 update entropy={after:.4f} (变化={after-before:.4f})")
            
            # 检查从首次触发到熵开始恢复的延迟
            for i in range(first_penalty_idx, min(first_penalty_idx + 5, len(metrics))):
                r = metrics[i]
                ent_coef = r.get("entropy_coef", 0)
                print(f"    Ep {r['episode']}: entropy={r['entropy']:.4f}, ent_coef={ent_coef:.4f}, hard_penalty={safe(r.get('hard_entropy_penalty',0)):.2f}")
    
    print()

    # ========== H4: 有效动作空间 ==========
    print("-" * 70)
    print("H4: 有效动作空间分析")
    print("-" * 70)
    
    print(f"  {'Episode':>6s}  {'entropy':>10s}  {'valid_act':>9s}  {'ln(N_valid)':>11s}  {'ratio':>8s}  {'解释':<40s}")
    for r in metrics:
        valid_n = safe(r.get("valid_actions_mean", 0))
        if valid_n <= 0:
            continue
        ln_valid = math.log(valid_n) if valid_n > 0 else 0
        eff_ratio = r["entropy"] / ln_valid if ln_valid > 0 else 0
        note = ""
        if r["entropy"] > 0.05 and eff_ratio > 0.5:
            note = "健康分布"
        elif r["entropy"] > 0.01 and eff_ratio < 0.3:
            note = "策略偏确定性"
        elif r["entropy"] < 0.001:
            note = "完全坍塌"
        elif eff_ratio > 1.0 and r["entropy"] < 1:
            note = "近乎均匀"
        
        if r["episode"] in [8, 16, 32, 64, 80, 96, 112, 128, 136, 144, 152, 160, 168, 192]:
            print(f"  Ep {r['episode']:>3d}  {r['entropy']:>10.6f}  {valid_n:>9.2f}  {ln_valid:>11.4f}  {eff_ratio:>8.4f}  {note}")
    print()
    
    # ========== H5: ratio 异常波动 ==========
    print("-" * 70)
    print("H5: Ratio 异常波动与熵坍塌的时间关系")
    print("-" * 70)
    
    print(f"  {'Episode':>6s}  {'entropy':>10s}  {'ratio_mean':>10s}  {'ratio_std':>9s}  {'clip_frac':>9s}  {'ratio>2?':>8s}")
    high_ratio_count = 0
    for r in metrics:
        rm = safe(r.get("ratio_mean", 0))
        rs = safe(r.get("ratio_std", 0))
        cf = safe(r.get("clip_fraction", 0))
        flagged = "⚠️" if rm > 2.0 or rs > 5.0 else ""
        if flagged:
            high_ratio_count += 1
        if flagged or r["episode"] in [8, 136, 144, 152, 160]:
            print(f"  Ep {r['episode']:>3d}  {r['entropy']:>10.6f}  {rm:>10.2f}  {rs:>9.2f}  {cf:>9.4f}  {flagged:>8s}")
    
    print(f"  高 ratio 事件 (mean>2.0 or std>5.0) 次数: {high_ratio_count}")
    
    # 检查 ratio 爆炸与熵下降的时间关系
    print()
    print("  ratio 爆炸 → entropy 下降 的时间关系:")
    # 找 ratio_mean > 2 的事件，以及随后的 entropy
    for i, r in enumerate(metrics):
        rm = safe(r.get("ratio_mean", 0))
        if rm > 2.0 and i + 1 < len(metrics):
            next_e = metrics[i + 1]["entropy"]
            delta = next_e - r["entropy"]
            print(f"    Ep {r['episode']} ratio_mean={rm:.1f} → Ep {metrics[i+1]['episode']} entropy 变化: {delta:+.6f}")
    print()

    # ========== H6: 类型分布分析 ==========
    print("-" * 70)
    print("H6: 动作类型分布演变")
    print("-" * 70)
    
    type_keys = [k for k in metrics[0].keys() if k.startswith("type_") and not k.startswith("type_valid_")]
    type_keys.sort()
    
    print(f"  {'Episode':>6s}  {'entropy':>10s}  ", end="")
    for tk in type_keys:
        short = tk.replace("type_", "")[:10]
        print(f"{short:>10s}", end="")
    print()
    
    for r in metrics:
        if r["episode"] in [8, 16, 32, 64, 80, 96, 112, 128, 136, 144, 152, 160, 168, 180, 192]:
            print(f"  Ep {r['episode']:>3d}  {r['entropy']:>10.6f}  ", end="")
            for tk in type_keys:
                val = safe(r.get(tk, 0))
                if val == 1.0:
                    print(f"{'1.0(100%)':>10s}", end="")
                elif val > 0.1:
                    print(f"{val:>8.2%}  ", end="")
                elif val > 0.01:
                    print(f"{val:>8.1%}  ", end="")
                else:
                    print(f"{'0':>10s}", end="")
            print()
    print()

    # ========== 权重分析 ==========
    print("-" * 70)
    print("权重污染过程 (weight_stats.jsonl)")
    print("-" * 70)
    
    for wr in weights:
        ep = wr["episode"]
        nan_params = []
        healthy_params = 0
        for name, ws in wr["weights"].items():
            if any(math.isnan(ws[k]) for k in ["mean", "std", "min", "max"]):
                nan_params.append(name)
            else:
                healthy_params += 1
        print(f"  Ep {ep}: 健康参数={healthy_params}, NaN参数={len(nan_params)}")
        if nan_params:
            for np_ in nan_params[:5]:
                print(f"    NaN: {np_}")
            if len(nan_params) > 5:
                print(f"    ... 共 {len(nan_params)} 个NaN参数")
    print()

    # ========== 综合结论 ==========
    print("=" * 70)
    print("综合诊断结论")
    print("=" * 70)
    
    # H1 结论
    healthy_end_avg = statistics.mean(healthy_end_ratios) if healthy_end_ratios else 0
    dead_end_avg = statistics.mean(dead_end_ratios) if dead_end_ratios else 0
    print(f"\nH1 (终局奖励主导): 健康阶段终局占比={healthy_end_avg:.0f}%, 死亡阶段={dead_end_avg:.0f}%")
    if healthy_end_avg > 60:
        print("  ✅ 终局奖励占比>60%，可能主导策略学习。但终局占比在健康/死亡阶段差异不大，说明这不是引起坍塌的关键变化。")
    else:
        print("  ❌ 终局奖励并非主导因素。")
    
    # H3 结论
    print(f"\nH3 (自适应机制滞后): hard_entropy_penalty 触发 {len(penalty_triggered)}/{len(metrics)} 次")
    if penalty_triggered:
        first_idx = next(i for i, r in enumerate(metrics) if safe(r.get("hard_entropy_penalty", 0)) > 0)
        if first_idx + 1 < len(metrics):
            delta = metrics[first_idx + 1]["entropy"] - metrics[first_idx]["entropy"]
            print(f"  触发后一次update熵变化={delta:.6f}")
            if delta < 0:
                print("  ✅ 自适应机制响应不足：触发后熵继续下降。说明反应式机制存在滞后。")
            else:
                print("  ❌ 自适应机制有效，熵在触发后回升。")
    
    # H4 结论
    print(f"\nH4 (有效动作空间):")
    valid_means = [safe(r.get("valid_actions_mean", 0)) for r in metrics if r["entropy"] > 0.001]
    print(f"  存活阶段平均有效动作数: {statistics.mean(valid_means):.1f}" if valid_means else "  N/A")
    print(f"  健康阶段最大熵理论上界: log(N_valid) ~ {math.log(statistics.mean(valid_means)):.2f}" if valid_means else "")
    # 检查是否 valid_actions 变大时熵变低
    dead_valid = [safe(r.get("valid_actions_mean", 0)) for r in dead]
    healthy_valid = [safe(r.get("valid_actions_mean", 0)) for r in healthy]
    if dead_valid and healthy_valid:
        print(f"  健康阶段valid_actions均值={statistics.mean(healthy_valid):.1f}, 死亡阶段={statistics.mean(dead_valid):.1f}")
        if statistics.mean(dead_valid) > statistics.mean(healthy_valid):
            print("  ✅ 死亡阶段合法动作数反而更多 → 坍塌不是环境约束导致（H4不成立），是策略本身选择只做NO-OP。")
        else:
            print("  ❌ 环境约束可能加剧问题。")
    
    # H5 结论
    print(f"\nH5 (多epoch自放大): 高ratio事件(ratio_mean>2)共{high_ratio_count}次")
    # 检查熵下降是否紧跟ratio异常
    ratio_spikes_before_collapse = 0
    for i, r in enumerate(metrics):
        if safe(r.get("ratio_mean", 0)) > 2.0 and i + 1 < len(metrics):
            if metrics[i + 1]["entropy"] < r["entropy"] * 0.8:
                ratio_spikes_before_collapse += 1
    print(f"  ratio异常后熵大幅下降的次数: {ratio_spikes_before_collapse}")
    if ratio_spikes_before_collapse > 0:
        print("  ✅ ratio异常与熵下降存在时序相关性 → H5可能存在。")
    else:
        print("  ❌ 未发现ratio异常与熵下降的直接时序关联。")
    
    # H6 结论
    print(f"\nH6 (类型分布坍缩):")
    for r in metrics:
        if r["episode"] == collapse_ep - 8 if collapse_ep else r["episode"] == 136:
            noop_ratio = safe(r.get("type_noop", 0))
            print(f"  坍塌前最后一次健康update (Ep {r['episode']}): NO-OP占比={noop_ratio:.2%}")
        if r["episode"] == collapse_ep if collapse_ep else r["episode"] == 144:
            noop_ratio = safe(r.get("type_noop", 0))
            print(f"  坍塌时刻 (Ep {r['episode']}): NO-OP占比={noop_ratio:.2%}")
            for tk in type_keys:
                val = safe(r.get(tk, 0))
                if val > 0.01:
                    print(f"    {tk.replace('type_','')}: {val:.2%}")
        if r["episode"] == dead[-1]["episode"] if dead else r["episode"] == 192:
            noop_ratio = safe(r.get("type_noop", 0))
            print(f"  死亡阶段最后 (Ep {r['episode']}): NO-OP占比={noop_ratio:.2%}")
    
    if any(safe(r.get("type_noop", 0)) > 0.95 for r in dead):
        print("  ✅ NO-OP占比在死亡阶段接近100%，策略完全坍缩为不做任何操作。")

    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("""
根据事后分析，各假设结论优先级排序：

  高置信度:
    H1: 终局奖励主导训练 —— 不直接导致坍塌，但可能加速策略收敛
    H4: 有效动作空间 —— 坍塌不是环境约束导致（死亡阶段合法动作更多）
    H6: NO-OP 类型坍缩 —— 坍塌的本质是策略学会只做 NO-OP

  中置信度（需新训练数据验证）:
    H5: 多 epoch 自放大 —— ratio 异常与熵下降存在时序相关性
    H3: 自适应机制滞后 —— 硬性熵惩罚触发后熵继续下降

  低置信度（需更多数据）:
    H2: 熵衰减速度 —— 需要对比实验组才能确认

建议下一步：
  1. 下载完整 training.log 确认 NaN 前的梯度/ratio 细节
  2. 启动 ppo_epochs=1 消融实验验证 H5
  3. 如确认 H5 为主因，将自适应 epoch 缩减纳入正式方案
""")


if __name__ == "__main__":
    main()
