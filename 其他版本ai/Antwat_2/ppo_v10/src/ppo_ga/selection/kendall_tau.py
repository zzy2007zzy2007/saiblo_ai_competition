import numpy as np


def kendall_tau(ranking_a: list, ranking_b: list) -> float:
    """
    计算两个排名列表的 Kendall τ 相关系数。

    Args:
        ranking_a: 排名 A，元素为可比较的个体 ID 列表，按排名从高到低
        ranking_b: 排名 B，格式同上

    Returns:
        τ ∈ [-1, 1]
    """
    # 仅考虑两个排名共同包含的元素
    common = [item for item in ranking_a if item in set(ranking_b)]
    n = len(common)
    if n < 2:
        return 1.0

    # 建立 B 排名中每个元素的位次映射
    rank_b = {item: i for i, item in enumerate(ranking_b)}

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = 1  # common[i] 在 ranking_a 中排在 common[j] 前面
            # 在 B 中的相对顺序
            b_order = 1 if rank_b[common[i]] < rank_b[common[j]] else -1
            if a_order * b_order > 0:
                concordant += 1
            else:
                discordant += 1

    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def ranking_from_trueskill(
    individual_ids: list,
    trueskill_mus: dict,       # {id: mu}
) -> list:
    """按 TrueSkill mu 降序生成排名列表"""
    return sorted(individual_ids, key=lambda iid: trueskill_mus.get(iid, 0.0), reverse=True)


def ranking_from_head_to_head(
    individual_ids: list,
    results: dict,             # {individual_id: {"wins": int, "losses": int, "draws": int}}
) -> list:
    """根据对战结果生成排名列表（按胜率降序）"""
    def win_rate(iid):
        r = results.get(iid, {})
        total = r.get("wins", 0) + r.get("losses", 0) + r.get("draws", 0)
        if total == 0:
            return 0.0
        return (r.get("wins", 0) + 0.5 * r.get("draws", 0)) / total

    return sorted(individual_ids, key=win_rate, reverse=True)
