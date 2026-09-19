"""指标计算 — Spearman ρ, TopK 重叠率, 计算量统计"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def spearman_rho(ranking_a: List[str], ranking_b: List[str]) -> float:
    """计算两个排名的 Spearman 相关系数

    Args:
        ranking_a: ID 列表，从前到后 = 从强到弱
        ranking_b: ID 列表，从前到后 = 从强到弱

    Returns:
        ρ ∈ [-1, 1]
    """
    if len(ranking_a) != len(ranking_b):
        raise ValueError("两个排名长度必须相同")

    n = len(ranking_a)
    if n < 2:
        return 0.0  # 不足以计算相关系数

    # 构建 rank map: ID → rank (0 = 最强)
    rank_map_a = {id_: i for i, id_ in enumerate(ranking_a)}
    rank_map_b = {id_: i for i, id_ in enumerate(ranking_b)}

    ranks_a = [rank_map_a[id_] for id_ in ranking_a]
    ranks_b = [rank_map_b[id_] for id_ in ranking_a]

    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return 0.0

    rho, _ = stats.spearmanr(ranks_a, ranks_b)
    return float(rho)


def topk_overlap_rate(
    ranking_a: List[str],
    ranking_b: List[str],
    k: int,
) -> float:
    """计算 TopK 交集率

    Args:
        ranking_a, ranking_b: 排名列表
        k: 取前 k 个

    Returns:
        交集率 ∈ [0, 1]
    """
    top_a = set(ranking_a[:k])
    top_b = set(ranking_b[:k])
    intersection = top_a & top_b
    return len(intersection) / k


def recall_at_k_within_topm(
    algo_ranking: List[str],
    gt_ranking: List[str],
    k: int,
    m: int,
) -> float:
    """算法 Top-m 中召回了 GT Top-k 的比例

    定义: |algo[:m] ∩ gt[:k]| / k
    含义: 算法选出 m 名候选时，其中包含了多少个真正的 GT 前 k 强。

    Args:
        algo_ranking: 算法输出的排名
        gt_ranking: Ground Truth 排名
        k: GT 中要召回的前 k 强
        m: 算法选出的前 m 名候选池

    Returns:
        召回率 ∈ [0, 1]
    """
    algo_top = set(algo_ranking[:m])
    gt_top = set(gt_ranking[:k])
    return len(algo_top & gt_top) / k


def project_to_subset(ranking: List[str], subset_ids) -> List[str]:
    """保留 ranking 中属于 subset 的元素，相对顺序不变

    Args:
        ranking: 完整排名列表
        subset_ids: 目标子集 ID（可为 list/set）

    Returns:
        投影后的子序列
    """
    s = set(subset_ids)
    return [iid for iid in ranking if iid in s]


def compute_all_metrics(
    ground_truth: List[str],
    bisection_ranking: List[str],
    round_robin_ranking: Optional[List[str]] = None,
    top_k_values: List[int] = [3, 5, 8, 16],
    battled_ids: Optional[List[str]] = None,
) -> dict:
    """计算所有评估指标

    Args:
        ground_truth: p 决定的真实排名
        bisection_ranking: 二分种子方案输出的排名（已仅含已对战个体）
        round_robin_ranking: 全量 RR 排名（可选）
        top_k_values: 需要计算重叠率的 TopK 值
        battled_ids: 若提供，则将 ground_truth 与 round_robin_ranking 投影到该子集，
                     再与 bisection_ranking 等长比较

    Returns:
        指标字典
    """
    metrics = {}

    # 投影到已对战子集（如有）
    if battled_ids is not None:
        gt_for_compare = project_to_subset(ground_truth, battled_ids)
        rr_for_compare = (
            project_to_subset(round_robin_ranking, battled_ids)
            if round_robin_ranking is not None else None
        )
    else:
        gt_for_compare = ground_truth
        rr_for_compare = round_robin_ranking

    # Spearman ρ vs Ground Truth
    metrics["spearman_rho_vs_ground_truth"] = spearman_rho(
        bisection_ranking, gt_for_compare
    )

    # Spearman ρ vs RoundRobin
    if rr_for_compare is not None:
        metrics["spearman_rho_vs_round_robin"] = spearman_rho(
            bisection_ranking, rr_for_compare
        )
        # RR vs GT 仍使用全集对比，作为 RR 自身基准
        metrics["spearman_rho_rr_vs_ground_truth"] = spearman_rho(
            round_robin_ranking, ground_truth
        )

    # TopK 重叠率
    for k in top_k_values:
        k_actual = min(k, len(gt_for_compare))
        metrics[f"top{k}_overlap_vs_ground_truth"] = topk_overlap_rate(
            bisection_ranking, gt_for_compare, k_actual
        )
        if rr_for_compare is not None:
            metrics[f"top{k}_overlap_vs_rr"] = topk_overlap_rate(
                bisection_ranking, rr_for_compare, k_actual
            )

    # Recall@K within Top-16: 算法 Top16 召回了 GT Top-K 的比例
    # 含义: 算法把"前 16 进精排"作为候选池，能覆盖多少个真正的 GT Top-K
    M = 16
    for k in [3, 5, 8, 16]:
        k_actual = min(k, len(gt_for_compare))
        m_actual = min(M, len(gt_for_compare))
        metrics[f"recall{k}_within_top{M}_vs_ground_truth"] = recall_at_k_within_topm(
            bisection_ranking, gt_for_compare, k_actual, m_actual
        )

    # 额外信息
    if battled_ids is not None:
        metrics["n_battled"] = len(battled_ids)
        metrics["n_ground_truth"] = len(ground_truth)

    return metrics


def format_metrics(metrics: dict) -> str:
    """格式化输出指标"""
    lines = []
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)
