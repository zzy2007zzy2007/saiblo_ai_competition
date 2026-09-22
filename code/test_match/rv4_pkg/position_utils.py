#!/usr/bin/env python3
"""位置判断工具"""
from __future__ import annotations

from SDK.utils.constants import PLAYER_BASES


# 中线位置（两个基地的中间）
# 玩家0基地: (2, 9)
# 玩家1基地: (16, 9)
# 中线 x 坐标: 9
MIDLINE_X = 9


def get_distance_to_midline(x: int, y: int) -> int:
    """计算位置到中线的距离（曼哈顿距离）"""
    return abs(x - MIDLINE_X)


def is_near_midline(x: int, y: int, threshold: int = 3) -> bool:
    """
    判断位置是否靠近中线
    
    Args:
        x, y: 位置坐标
        threshold: 距离阈值，默认3
    
    Returns:
        True 表示靠近中线，False 表示远离中线
    """
    return get_distance_to_midline(x, y) <= threshold


def position_classify(x: int, y: int, player: int) -> str:
    """
    分类位置：前线、中线、后方
    
    Returns:
        'front': 靠近敌方（前线）
        'mid': 中线附近
        'back': 靠近己方基地（后方）
    """
    distance = get_distance_to_midline(x, y)
    
    if distance <= 2:
        return 'mid'
    elif player == 0:
        # 玩家0在左边，前线是右边
        return 'front' if x > MIDLINE_X else 'back'
    else:
        # 玩家1在右边，前线是左边
        return 'front' if x < MIDLINE_X else 'back'


def get_build_position_info(x: int, y: int, player: int) -> dict:
    """
    获取建塔位置的详细信息
    
    Returns:
        dict 包含位置信息
    """
    base_pos = PLAYER_BASES[player]
    distance_to_midline = get_distance_to_midline(x, y)
    distance_to_base = abs(x - base_pos[0]) + abs(y - base_pos[1])
    
    return {
        'position': (x, y),
        'distance_to_midline': distance_to_midline,
        'distance_to_base': distance_to_base,
        'is_near_midline': is_near_midline(x, y),
        'classify': position_classify(x, y, player),
    }


# 测试代码
if __name__ == '__main__':
    print("中线位置 x =", MIDLINE_X)
    print()
    
    test_positions = [
        (5, 9, 0),
        (8, 9, 0),
        (9, 9, 0),
        (10, 9, 0),
        (12, 9, 0),
        (5, 9, 1),
        (8, 9, 1),
        (9, 9, 1),
        (10, 9, 1),
        (12, 9, 1),
    ]
    
    for x, y, player in test_positions:
        info = get_build_position_info(x, y, player)
        print(f"玩家{player} 位置({x},{y}):")
        print(f"  到中线距离: {info['distance_to_midline']}")
        print(f"  到基地距离: {info['distance_to_base']}")
        print(f"  是否靠近中线: {info['is_near_midline']}")
        print(f"  位置分类: {info['classify']}")
        print()
