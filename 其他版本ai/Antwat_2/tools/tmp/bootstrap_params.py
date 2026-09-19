"""
P0 参数测算脚本：粗排 n_battles / pass_ratio / 精排 n_battles
基于 baseline 50 局高精度数据，用 bootstrap 模拟少量对局的表现
"""
import json, sys, math, random
import numpy as np
from collections import defaultdict

# ── ELO 工具 ──
K = 64
INIT = 1200

def compute_elo(individuals, refs_data, n_battles, n_bootstrap=200):
    """
    对一批个体，用 n_battles 局模拟对战，计算 ELO 排序。
    返回 n_bootstrap 次模拟的排名结果。
    
    refs_data: [(ref_name, base_elo, p_dict)]  其中 p_dict 是 {ind_id: true_p}
    """
    all_rankings = []
    
    for _ in range(n_bootstrap):
        ratings = {}
        for ref_name, base_elo, p_dict in refs_data:
            ratings[ref_name] = base_elo  # 固定参照物 ELO
        
        for ind_id in individuals:
            ratings[ind_id] = INIT
        
        # 模拟对战
        for ref_name, base_elo, p_dict in refs_data:
            for ind_id in individuals:
                true_p = p_dict.get(ind_id, 0.5)
                # 模拟 n_battles*2 局（先后手各半）
                n_games = n_battles * 2
                wins = np.random.binomial(n_games, true_p)
                observed_p = wins / n_games
                
                # ELO 更新：基于模拟的对战结果
                ra = ratings[ind_id]
                rb = ratings[ref_name]
                ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
                
                # 每局独立更新（更真实）或者批量更新
                # 简化：用观测胜率一次性更新
                for _ in range(n_games):
                    # 每局独立随机+独立ELO更新
                    game_win = 1 if random.random() < true_p else 0
                    ra = ratings[ind_id]
                    rb = ratings[ref_name]
                    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
                    ratings[ind_id] = ra + K * (game_win - ea)
                    # 参照物评分不更新
        
        # 排序
        ranked = sorted(
            [(iid, ratings[iid]) for iid in individuals],
            key=lambda x: x[1], reverse=True
        )
        all_rankings.append([iid for iid, _ in ranked])
    
    return all_rankings


def compute_elo_fine(individuals, refs_data, n_battles, n_bootstrap=200):
    """精排用：更多参照物，更多局数"""
    return compute_elo(individuals, refs_data, n_battles, n_bootstrap)


def course_filter(individuals, refs_data, n_battles, pass_ratio, n_bootstrap=200):
    """粗排：只用 2 个参照物，少量局数，筛选 pass_ratio 比例"""
    n_pass = max(1, int(len(individuals) * pass_ratio))
    all_passed = []
    
    for _ in range(n_bootstrap):
        ratings = {}
        for ref_name, base_elo, p_dict in refs_data:
            ratings[ref_name] = base_elo
        
        for ind_id in individuals:
            ratings[ind_id] = INIT
        
        for ref_name, base_elo, p_dict in refs_data:
            for ind_id in individuals:
                true_p = p_dict.get(ind_id, 0.5)
                n_games = n_battles * 2
                for _ in range(n_games):
                    game_win = 1 if random.random() < true_p else 0
                    ra = ratings[ind_id]
                    rb = ratings[ref_name]
                    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
                    ratings[ind_id] = ra + K * (game_win - ea)
        
        ranked = sorted(
            [(iid, ratings[iid]) for iid in individuals],
            key=lambda x: x[1], reverse=True
        )
        passed = [iid for iid, _ in ranked[:n_pass]]
        all_passed.append(passed)
    
    return all_passed


def kendall_tau(ranking_a, ranking_b):
    """计算两个排名的 Kendall tau 相关系数"""
    n = len(ranking_a)
    if n != len(ranking_b):
        raise ValueError("Rankings must have same length")
    
    idx_a = {x: i for i, x in enumerate(ranking_a)}
    idx_b = {x: i for i, x in enumerate(ranking_b)}
    
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = int(idx_a[ranking_a[i]] < idx_a[ranking_a[j]])
            b_order = int(idx_b[ranking_a[i]] < idx_b[ranking_a[j]])
            if a_order == b_order:
                concordant += 1
            else:
                discordant += 1
    
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def top_k_recall(pred_ranking, true_top_k, k):
    """召回率：预测 top-K 中有多少在真实 top-K 中"""
    pred_set = set(pred_ranking[:k])
    return len(pred_set & true_top_k) / k


def load_data(eval_json_path):
    """加载 evaluation.json，提取每个个体的真实 p 值"""
    with open(eval_json_path) as f:
        d = json.load(f)
    
    individuals = []
    true_p = {}  # {(ind_id, ref_name): p}
    
    for ind_id, refs in d.items():
        individuals.append(ind_id)
        for ref_name in refs:
            wins = refs[ref_name]["agent1_wins"]
            total = refs[ref_name]["total_battles"]
            true_p[(ind_id, ref_name)] = wins / total if total > 0 else 0.5
    
    return individuals, true_p


def compute_theoretical_elo(individuals, true_p, refs):
    """基于 50 局的真实 p 值，解析计算理论 ELO（避免迭代误差）"""
    # 对每个个体，从每个参照物反推 ELO，求平均
    ind_elos = {}
    for ind_id in individuals:
        elos = []
        for ref_name, base_elo in refs:
            p = true_p.get((ind_id, ref_name), 0.5)
            if 0 < p < 1:
                diff = -400 * math.log10(1/p - 1)
            elif p == 1.0:
                diff = 400  # cap
            else:
                diff = -400
            elos.append(base_elo + diff)
        ind_elos[ind_id] = sum(elos) / len(elos) if elos else INIT
    
    return sorted(individuals, key=lambda x: ind_elos[x], reverse=True), ind_elos


def analyze_course(individuals, true_p, course_refs):
    """分析粗排参数"""
    refs_data = []
    for ref_name, base_elo in course_refs:
        p_dict = {}
        for ind in individuals:
            p_dict[ind] = true_p.get((ind, ref_name), 0.5)
        refs_data.append((ref_name, base_elo, p_dict))
    
    # Ground truth: 基于 50 局真实 p 的解析 ELO
    gt_ranking, gt_elos = compute_theoretical_elo(individuals, true_p, course_refs)
    
    print(f"\n  Ground truth ranking (based on 50-game true p):")
    for i, ind in enumerate(gt_ranking[:8]):
        print(f"    {i+1}. {ind:20s}  ELO={gt_elos[ind]:.1f}")
    
    true_top_k = {8: set(gt_ranking[:8]), 4: set(gt_ranking[:4])}
    
    # 测试不同 n_battles 和 pass_ratio
    print(f"\n  {'n_battles':>10s} | {'pass_ratio':>10s} | {'recall@8':>10s} | {'recall@4':>10s} | {'mean_pass':>10s}")
    print("  " + "-" * 65)
    
    for n_b in [1, 2, 3]:
        for pr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            passed_sets = course_filter(individuals, refs_data, n_b, pr, n_bootstrap=200)
            recalls_8 = [top_k_recall(ps, true_top_k[8], 8) for ps in passed_sets]
            recalls_4 = [top_k_recall(ps, true_top_k[4], 4) for ps in passed_sets]
            mean_recall_8 = np.mean(recalls_8)
            mean_recall_4 = np.mean(recalls_4)
            mean_passed = np.mean([len(ps) for ps in passed_sets])
            print(f"  {n_b:10d} | {pr:10.1%}  | {mean_recall_8:10.1%} | {mean_recall_4:10.1%} | {mean_passed:10.1f}")
    
    return gt_ranking


def analyze_fine(individuals, true_p, all_refs):
    """分析精排 n_battles"""
    refs_data = []
    for ref_name, base_elo in all_refs:
        p_dict = {}
        for ind in individuals:
            p_dict[ind] = true_p.get((ind, ref_name), 0.5)
        refs_data.append((ref_name, base_elo, p_dict))
    
    # Ground truth: 基于所有参照物 50 局的解析 ELO
    gt_ranking, gt_elos = compute_theoretical_elo(individuals, true_p, all_refs)
    
    print(f"\n  Ground truth (all refs, 50 games):")
    for i, ind in enumerate(gt_ranking[:8]):
        print(f"    {i+1}. {ind:20s}  ELO={gt_elos[ind]:.1f}")
    
    # 测试不同 n_battles
    print(f"\n  {'n_battles':>10s} | {'Kendall tau':>12s} | {'tau std':>10s} | {'recall@8':>10s} | {'recall@4':>10s} | {'#8 stable':>10s}")
    print("  " + "-" * 75)
    
    for n_b in [1, 2, 3, 5, 8, 10]:
        rankings = compute_elo(individuals, refs_data, n_b, n_bootstrap=200)
        taus = [kendall_tau(r, gt_ranking) for r in rankings]
        recalls_8 = [top_k_recall(r, set(gt_ranking[:8]), 8) for r in rankings]
        recalls_4 = [top_k_recall(r, set(gt_ranking[:4]), 4) for r in rankings]
        
        # #8 位置稳定性：前 8 中排在第 8 的个体在多少次模拟中进了前 8
        rank8_stable = sum(1 for r in rankings if gt_ranking[7] in r[:8]) / len(rankings)
        
        print(f"  {n_b:10d} | {np.mean(taus):12.3f} | {np.std(taus):10.3f} | {np.mean(recalls_8):10.1%} | {np.mean(recalls_4):10.1%} | {rank8_stable:10.1%}")


def main():
    import sys
    data_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    
    print(f"=" * 70)
    print(f"  P0 参数测算 - {label}")
    print(f"  数据: {data_path}")
    print(f"=" * 70)
    
    individuals, true_p = load_data(data_path)
    print(f"\n  个体数: {len(individuals)}")
    
    # ── 粗排分析 ──
    print(f"\n{'='*70}")
    print(f"  【P0-1】粗排参数: 用 BasicTowerAI + MediumRuleAI 做低门槛参照物")
    print(f"{'='*70}")
    
    course_refs = [
        ("BasicTowerAI", 1000),
        ("MediumRuleAI", 1600),
    ]
    analyze_course(individuals, true_p, course_refs)
    
    # ── 精排分析 ──
    print(f"\n{'='*70}")
    print(f"  【P0-2】精排参数: n_battles 对排序精度的影响")
    print(f"{'='*70}")
    
    all_refs = [
        ("BasicRandomAI", 400),
        ("BasicTowerAI", 1000),
        ("MediumRuleAI", 1600),
    ]
    analyze_fine(individuals, true_p, all_refs)


if __name__ == "__main__":
    main()
