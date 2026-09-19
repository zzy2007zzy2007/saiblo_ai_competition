from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PlayerState:
    """单个玩家的状态快照。"""

    hp: float
    coins: float
    tower_hp: Dict[int, int]
    tower_count: int
    tower_levels: List[int]
    gen_level: int
    ant_level: int
    die_count: int
    build_cost: float


__all__ = ["PlayerState"]
