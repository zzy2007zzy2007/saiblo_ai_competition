from typing import Dict, Optional


class ELORating:
    """ELO 评分系统 —— 用于代内个体排名"""

    def __init__(
        self,
        initial_rating: float = 1200.0,
        k_factor: float = 64.0,
    ):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}
        self._dynamic_ref_ratings: Dict[str, float] = {}

    def ensure_player(self, player_id: str) -> None:
        """注册玩家（若未注册则使用初始评分）"""
        if player_id not in self._ratings:
            self._ratings[player_id] = self.initial_rating

    def update(self, player_a: str, player_b: str, result: float) -> None:
        """更新双方 ELO 评分。

        Args:
            player_a: 个体 ID
            player_b: 参照物 ID
            result: 1.0 = A 胜, 0.5 = 平, 0.0 = A 负
        """
        self.ensure_player(player_a)
        self.ensure_player(player_b)

        ra = self._ratings[player_a]
        rb = self._ratings[player_b]

        # 期望得分
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

        # 更新评分
        self._ratings[player_a] = ra + self.k_factor * (result - ea)
        # 参照物评分不更新（固定参照物保持不变）

    def get_rating(self, player_id: str) -> float:
        """获取玩家评分"""
        return self._ratings.get(player_id, self.initial_rating)

    def set_rating(self, player_id: str, rating: float) -> None:
        """直接设置玩家评分（用于批量 ELO 计算后同步）"""
        self._ratings[player_id] = rating

    def get_rankings(self) -> list:
        """获取按评分降序排列的 (player_id, rating) 列表"""
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def reset_individuals(self) -> None:
        """重置个体评分，保留动态参照物的跨代评分"""
        dynamic_names = set(self._dynamic_ref_ratings.keys())
        self._ratings = {k: v for k, v in self._ratings.items() if k in dynamic_names}

    def save_dynamic_ref_rating(self, ref_name: str, rating: float) -> None:
        """保存动态参照物评分（代结束时调用）"""
        self._dynamic_ref_ratings[ref_name] = rating

    def get_dynamic_ref_rating(self, ref_name: str) -> float:
        """获取动态参照物的上一代评分，不存在则返回 initial_rating"""
        return self._dynamic_ref_ratings.get(ref_name, self.initial_rating)
