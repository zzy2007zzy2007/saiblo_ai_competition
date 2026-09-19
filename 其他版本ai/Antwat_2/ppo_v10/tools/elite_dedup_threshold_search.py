#!/usr/bin/env python3
"""阈值优化实验：基于模拟 gen4 家族结构，网格搜索最优去重阈值。

模拟数据包含 4 个家族 + 散兵，结构如下：
  - 家族 A (强): 28 人, ELO 1500~1800, 内部 cos 0.65~0.95
  - 家族 B (中): 18 人, ELO 1300~1550, 内部 cos 0.65~0.92
  - 家族 C (弱但独立): 12 人, ELO 1100~1350, 内部 cos 0.60~0.90
  - 家族 D (弱但独立): 8 人,  ELO 1000~1200, 内部 cos 0.55~0.88
  - 散兵:           14 人, ELO 900~1400,  随机方向
"""

import numpy as np
import itertools
import sys
from typing import List, Tuple, Dict

# ============================================================
# 1. 余弦相似度
# ============================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ============================================================
# 2. 生成合成数据
# ============================================================
def _random_unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    """生成随机单位向量（高维空间中方向均匀分布）。"""
    v = rng.normal(0, 1, dim)
    return v / np.linalg.norm(v)


def generate_synthetic_gen4(
    dim: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成模拟 gen4 的 80 个权重向量和 ELO 评分。

    家族通过"锚点插值"生成，精确控制内部余弦相似度：
    - 家族成员 w = α * anchor + √(1-α²) * random_unit
    - 两个同家族成员的期望 cos ≈ α²（高维空间中 random_unit 近似正交）
    - 跨家族 cos ≈ α_i * α_j * (anchor_i · anchor_j)，而随机锚点在高维几乎正交
    """
    rng = np.random.default_rng(seed)
    n_total = 80

    # 家族定义:
    #   target_cos: 期望的家族内部余弦相似度
    #   alpha: 插值系数 α = √target_cos
    #   elo_center: ELO 中心值
    #   elo_spread: ELO 散布范围（均匀 ± elo_spread）
    family_specs = [
        # 家族 A: 强，紧密 (对标文档 seed_1↔seed_2 cos≈0.903)
        {"n": 28, "target_cos": 0.88, "elo_center": 1650, "elo_spread": 150, "label": 0},
        # 家族 B: 中，紧密 (对标文档 seed_3↔seed_4 cos≈0.917)
        {"n": 18, "target_cos": 0.90, "elo_center": 1400, "elo_spread": 130, "label": 1},
        # 家族 C: 弱但独立
        {"n": 12, "target_cos": 0.85, "elo_center": 1200, "elo_spread": 100, "label": 2},
        # 家族 D: 弱但独立
        {"n": 8,  "target_cos": 0.82, "elo_center": 1050, "elo_spread": 80,  "label": 3},
    ]

    # 为每个家族生成随机锚点（高维空间中几乎正交）
    anchors = {}
    for fam in family_specs:
        anchors[fam["label"]] = _random_unit(rng, dim)

    weights_list = []
    elos_list = []
    labels = []

    for fam in family_specs:
        anchor = anchors[fam["label"]]
        alpha = np.sqrt(fam["target_cos"])  # α² = target_cos
        beta = np.sqrt(max(0, 1 - alpha**2))

        for k in range(fam["n"]):
            # w = α * anchor + β * random_unit → E[cos(w_i, w_j)] = α²
            w = alpha * anchor + beta * _random_unit(rng, dim)
            w /= np.linalg.norm(w)  # 确保单位向量（修正数值误差）
            weights_list.append(w)

            elo = fam["elo_center"] + rng.uniform(-fam["elo_spread"], fam["elo_spread"])
            elos_list.append(elo)
            labels.append(fam["label"])

    # 散兵: 纯随机方向（彼此几乎正交）
    for k in range(14):
        w = _random_unit(rng, dim)
        weights_list.append(w)
        elos_list.append(rng.uniform(900, 1420))
        labels.append(-1)

    weights = np.array(weights_list, dtype=np.float32)
    elos = np.array(elos_list, dtype=np.float32)
    labels = np.array(labels, dtype=int)

    assert weights.shape == (n_total, dim)
    assert elos.shape == (n_total,)

    return weights, elos, labels, np.array(list(anchors.values()))


# ============================================================
# 3. 模拟种子选择
# ============================================================
def select_seeds_sim(
    weights: np.ndarray,
    elos: np.ndarray,
    hard_th: float,
    soft_th: float,
    soft_max: int,
    n_seeds: int = 8,
    verbose: bool = False,
) -> List[int]:
    """离线模拟 select_seeds_diverse，返回选中的个体索引列表（按 ELO 降序）。"""
    order = np.argsort(elos)[::-1]
    selected = []

    for idx in order:
        w = weights[idx]
        hard_dup = False
        family_count = 0
        for s_idx in selected:
            cos = cosine_similarity(w, weights[s_idx])
            if cos > hard_th:
                hard_dup = True
                if verbose:
                    print(f"  [{idx:3d}] hard-dup by [{s_idx:3d}] cos={cos:.4f}")
                break
            if cos > soft_th:
                family_count += 1
        if hard_dup or family_count >= soft_max:
            if verbose and not hard_dup:
                print(f"  [{idx:3d}] soft-throttled (family_count={family_count})")
            continue
        selected.append(idx)
        if verbose:
            print(f"  [{idx:3d}] SELECTED (ELO={elos[idx]:.0f})")
        if len(selected) >= n_seeds:
            break
    return selected


# ============================================================
# 4. 并查集计算家族数（连通分量）
# ============================================================
def compute_family_count(weights: np.ndarray, indices: List[int], soft_th: float) -> int:
    """用并查集计算 seeds 间 cos > soft_th 形成的连通分量数。"""
    n = len(indices)
    if n <= 1:
        return n
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(weights[indices[i]], weights[indices[j]]) > soft_th:
                union(i, j)
    return len({find(i) for i in range(n)})


# ============================================================
# 5. 实验主逻辑
# ============================================================
def run_experiment(weights: np.ndarray, elos: np.ndarray, labels: np.ndarray):
    """网格搜索最优阈值组合。"""
    n_total = len(elos)
    n_seeds = 8
    orig_top8_elo_mean = np.mean(np.sort(elos)[::-1][:n_seeds])

    print(f"数据规模: {n_total} 个个体, 权重维度={weights.shape[1]}")
    print(f"原始 top-{n_seeds} ELO 均值: {orig_top8_elo_mean:.1f}")
    print()

    # 搜索空间
    hard_candidates = [0.90, 0.92, 0.95, 0.97, 0.99]
    soft_candidates = [0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
    soft_max_candidates = [1, 2, 3]

    results = []
    for hard_th in hard_candidates:
        for soft_th in soft_candidates:
            for soft_max in soft_max_candidates:
                sel = select_seeds_sim(weights, elos, hard_th, soft_th, soft_max, n_seeds)

                if len(sel) >= 2:
                    cos_vals = [
                        cosine_similarity(weights[sel[i]], weights[sel[j]])
                        for i in range(len(sel))
                        for j in range(i + 1, len(sel))
                    ]
                    cos_max = max(cos_vals)
                    cos_mean = np.mean(cos_vals)
                else:
                    cos_max = 0.0
                    cos_mean = 0.0

                family_count = compute_family_count(weights, sel, soft_th)
                elo_drop = orig_top8_elo_mean - np.mean(elos[sel])

                # 统计选中种子的家族来源分布（ground truth）
                sel_labels = labels[sel]
                unique_labels = sorted(set(labels))
                label_counts = {
                    f"fam_{lbl}": int(np.sum(sel_labels == lbl))
                    for lbl in unique_labels
                }
                # 真实家族数：ground truth 中出现的不同家族标签数
                true_families = len([lbl for lbl in unique_labels
                                     if np.sum(sel_labels == lbl) > 0])

                results.append({
                    "hard_th": hard_th,
                    "soft_th": soft_th,
                    "soft_max": soft_max,
                    "n_selected": len(sel),
                    "family_count": family_count,
                    "true_families": true_families,
                    "cos_max": cos_max,
                    "cos_mean": cos_mean,
                    "elo_drop": elo_drop,
                    **label_counts,
                })

    # ── 找出家族内部 cos 的最大值，用于判断 soft_threshold 是否设得太高 ──
    max_intra_cos = 0.0
    for lbl in sorted(set(labels)):
        idx_a = np.where(labels == lbl)[0]
        if len(idx_a) >= 2:
            for i in idx_a[:10]:
                for j in idx_a[:10]:
                    if i < j:
                        max_intra_cos = max(max_intra_cos,
                                            cosine_similarity(weights[i], weights[j]))

    # ── 分类统计 ──
    # A. 满足 true_families >= 4 且 cos_max < 0.90 的最优组合
    strict = [r for r in results
              if r["true_families"] >= 4 and r["cos_max"] < 0.90]
    strict.sort(key=lambda r: r["elo_drop"])

    # B. true_families >= 3 的次优组合
    medium = [r for r in results
              if r["true_families"] >= 3 and r["cos_max"] < 0.90]
    medium.sort(key=lambda r: r["elo_drop"])

    # ── 诊断：检查 soft_threshold 是否超过了家族内部 cos ──
    soft_diag = {}
    for r in results:
        key = r["soft_th"]
        if key not in soft_diag:
            soft_diag[key] = r["family_count"]
    ineffective_soft = [st for st, fc in soft_diag.items()
                        if st > max_intra_cos and fc >= 4]

    # ── 输出 ──
    print(f"家族内部最大 cos: {max_intra_cos:.3f}")
    if ineffective_soft:
        print(f"⚠️  以下 soft_threshold 高于家族内部 cos ({max_intra_cos:.3f})，"
              f"无法检测家族聚类，family_count 指标不可信: "
              f"{ineffective_soft}")
    print()

    print("=" * 90)
    print("【A 档】true_families ≥ 4 且 cos_max < 0.90 （最优，基于 ground truth）")
    print("=" * 90)
    if strict:
        print_top_results(strict, max_n=10)
    else:
        print("  (无满足条件的组合)")

    print()
    print("=" * 90)
    print("【B 档】true_families ≥ 3 且 cos_max < 0.90 （次优，基于 ground truth）")
    print("=" * 90)
    if medium:
        # 去重：同一参数组合只显示最低 elo_drop 的
        print_top_results(medium, max_n=10)
    else:
        print("  (无满足条件的组合)")

    print()
    print("=" * 90)
    print("【参考】原版阈值 (hard=0.95, soft=0.85, max=2) 的表现")
    print("=" * 90)
    baseline = [r for r in results if r["hard_th"] == 0.95 and r["soft_th"] == 0.85 and r["soft_max"] == 2]
    if baseline:
        print_top_results(baseline, max_n=1)

    # ── 推荐 ──
    print()
    print("=" * 90)
    print("【推荐】")
    print("=" * 90)

    # 从 strict 和 medium 中排除 soft_threshold > max_intra_cos 的组合（不可信）
    def is_trustworthy(r):
        return r["soft_th"] <= max_intra_cos + 0.02  # 允许 0.02 的余量

    candidates = [r for r in (strict if strict else medium) if is_trustworthy(r)]

    if candidates:
        best = candidates[0]
        print(f"  推荐阈值: hard={best['hard_th']:.2f}, soft={best['soft_th']:.2f}, "
              f"soft_max={best['soft_max']}")
        print(f"  true_families={best['true_families']}, cos_max={best['cos_max']:.3f}, "
              f"cos_mean={best['cos_mean']:.3f}, elo_drop={best['elo_drop']:.1f}")

        close = [r for r in candidates if r["elo_drop"] - best["elo_drop"] < 10.0]
        if len(close) > 1:
            print(f"\n  以下 {len(close)} 组 elo_drop 差距 < 10，均可考虑：")
            print_top_results(close, max_n=len(close))
            close_strict = max(close, key=lambda r: r["soft_th"])
            print(f"\n  最终推荐（选最严格阈值）: hard={close_strict['hard_th']:.2f}, "
                  f"soft={close_strict['soft_th']:.2f}, soft_max={close_strict['soft_max']}")
    else:
        # 如果 strict/medium 都不可信，直接输出所有 true_families >= 3 的结果
        fallback = [r for r in results if r["true_families"] >= 3]
        fallback.sort(key=lambda r: (r["elo_drop"], -r["soft_th"]))
        if fallback:
            print("  strict/medium 档无可信组合，输出所有 true_families >= 3 的结果：")
            print_top_results(fallback, max_n=10)
            best = fallback[0]
            print(f"\n  建议阈值: hard={best['hard_th']:.2f}, soft={best['soft_th']:.2f}, "
                  f"soft_max={best['soft_max']}")
        else:
            print("  无法给出推荐，所有组合均不满足最低多样性要求。")

    return results


def print_top_results(results: list, max_n: int = 10):
    """格式化输出结果表格。"""
    header = (
        f"{'hard':>6}  {'soft':>6}  {'max':>4}  "
        f"{'families':>9}  {'true_f':>6}  {'cos_max':>8}  {'cos_mean':>8}  {'elo_drop':>9}  "
        f"{'fam_-1':>7}  {'fam_0':>6}  {'fam_1':>6}  {'fam_2':>6}  {'fam_3':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results[:max_n]:
        print(
            f"{r['hard_th']:6.2f}  {r['soft_th']:6.2f}  {r['soft_max']:4d}  "
            f"{r['family_count']:9d}  {r['true_families']:6d}  {r['cos_max']:8.3f}  {r['cos_mean']:8.3f}  {r['elo_drop']:9.1f}  "
            f"{r.get('fam_-1', 0):7d}  {r.get('fam_0', 0):6d}  {r.get('fam_1', 0):6d}  "
            f"{r.get('fam_2', 0):6d}  {r.get('fam_3', 0):6d}"
        )


# ============================================================
# 6. 数据诊断
# ============================================================
def diagnose_data(weights: np.ndarray, elos: np.ndarray, labels: np.ndarray):
    """输出合成数据的基本统计，验证是否符合预期模式。"""
    print("=" * 60)
    print("【数据诊断】合成 gen4 数据统计")
    print("=" * 60)

    # 家族间锚点 cos
    unique_labels = sorted(set(labels))
    for la in unique_labels:
        mask_a = labels == la
        count_a = np.sum(mask_a)
        elo_a = np.mean(elos[mask_a])
        print(f"\n家族 fam_{la}: {count_a} 人, 平均 ELO={elo_a:.1f}")

        # 家族内部 cos 分布
        idx_a = np.where(mask_a)[0]
        if len(idx_a) >= 2:
            intra_cos = []
            for i in range(min(len(idx_a), 50)):  # 采样避免 O(n²)
                for j in range(i + 1, min(len(idx_a), 50)):
                    intra_cos.append(cosine_similarity(weights[idx_a[i]], weights[idx_a[j]]))
            print(f"  内部 cos: min={min(intra_cos):.3f}, max={max(intra_cos):.3f}, "
                  f"mean={np.mean(intra_cos):.3f}")

        # 与其他家族的代表性 cos
        for lb in unique_labels:
            if lb >= la:
                continue
            mask_b = labels == lb
            idx_b = np.where(mask_b)[0]
            cross_cos = []
            for i in idx_a[:5]:  # 采样
                for j in idx_b[:5]:
                    cross_cos.append(cosine_similarity(weights[i], weights[j]))
            print(f"  vs fam_{lb}: cos min={min(cross_cos):.3f}, max={max(cross_cos):.3f}, "
                  f"mean={np.mean(cross_cos):.3f}")

    # top-8 的家族分布（无去重）
    top8_idx = np.argsort(elos)[::-1][:8]
    top8_labels = labels[top8_idx]
    label_counts = {lbl: int(np.sum(top8_labels == lbl)) for lbl in unique_labels}
    print(f"\n原始 top-8 家族分布 (无去重): {label_counts}")
    print(f"  家族 A (fam_0) 占 {label_counts.get(0, 0)}/8 = {label_counts.get(0,0)/8*100:.0f}%")
    print()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # 生成数据
    weights, elos, labels, anchors = generate_synthetic_gen4(dim=2000, seed=42)

    # 诊断
    diagnose_data(weights, elos, labels)

    # 实验
    run_experiment(weights, elos, labels)
