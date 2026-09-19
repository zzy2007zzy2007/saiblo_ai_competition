from typing import Any, List


class AntWarAgent:
    """所有对战 Agent 的基类"""

    def __init__(self, player_id: int, name: str = ""):
        self.player_id = player_id
        self.name = name or self.__class__.__name__

    def choose_operations(self, state) -> List[Any]:
        raise NotImplementedError

    @property
    def illegal_action_count(self) -> int:
        """子类可覆写此属性以返回非法动作计数（默认 0）。"""
        return 0


__all__ = ["AntWarAgent"]
