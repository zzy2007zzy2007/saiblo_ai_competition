"""状态快照数据容器 - 替代 _snapshot_state() 返回的扁平 dict。

使用 PlayerState/StateSnapshot 数据类代替 own_/enemy_ 前缀的扁平字典，
使 self/opponent 语义更清晰。
"""

from __future__ import annotations

from dataclasses import dataclass
from .player_state import PlayerState


@dataclass
class StateSnapshot:
    """完整状态快照，包含双方状态。"""

    self_state: PlayerState
    opponent_state: PlayerState
    round_index: int = 0


__all__ = ["StateSnapshot"]
