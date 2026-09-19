from typing import Any, List


class AntWarAgent:
    """所有对战 Agent 的基类"""

    def __init__(self, player_id: int, name: str = ""):
        self.player_id = player_id
        self.name = name or self.__class__.__name__

    def choose_operations(self, state) -> List[Any]:
        raise NotImplementedError


__all__ = ["AntWarAgent"]
