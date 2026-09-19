import json
from typing import Any, Dict, List

import trueskill

from loguru import logger

from .trueskill_rating import TrueSkillRating


# ─── 评分矩阵 ──────────────────────────────────────────────────────────────


class BattleSharedPayoff:
    """管理所有玩家间的对战记录和 TrueSkill 评分矩阵"""

    _PSEUDO_COUNT = 1
    _DRAW_WEIGHT = 0.5
    _EXPLOITABILITY_SCALE = 0.5

    def __init__(self, decay: float, min_win_rate_games: int):
        self._players: List[str] = []
        self._players_ids: Dict[str, int] = {}
        self._data: Dict[str, Dict[str, int]] = {}
        self._trueskill_ratings: Dict[str, TrueSkillRating] = {}
        self._decay = decay
        self._min_win_rate_games = min_win_rate_games

    def ensure_player(self, player_id: str) -> None:
        """注册玩家但不创建对战记录"""
        if player_id not in self._players_ids:
            self._players.append(player_id)
            self._players_ids[player_id] = len(self._players) - 1
            self._trueskill_ratings[player_id] = TrueSkillRating()

    def update(self, home: str, away: str, result: int) -> None:
        """更新对战记录和 TrueSkill 评分

        Args:
            result: 1 = home 胜, -1 = away 胜, 0 = 平局
        """
        if home not in self._players_ids:
            self._players.append(home)
            self._players_ids[home] = len(self._players) - 1
            self._trueskill_ratings[home] = TrueSkillRating()
        if away not in self._players_ids:
            self._players.append(away)
            self._players_ids[away] = len(self._players) - 1
            self._trueskill_ratings[away] = TrueSkillRating()

        key = f"{home}-{away}"
        if key not in self._data:
            self._data[key] = {"wins": 0, "draws": 0, "losses": 0, "games": 0}

        if result == 1:
            self._data[key]["wins"] += 1
        elif result == -1:
            self._data[key]["losses"] += 1
        else:
            self._data[key]["draws"] += 1

        self._data[key]["games"] += 1

        home_rating = self._trueskill_ratings[home].to_trueskill()
        away_rating = self._trueskill_ratings[away].to_trueskill()

        if result == 1:
            new_home, new_away = trueskill.rate_1vs1(home_rating, away_rating)
        elif result == -1:
            new_away, new_home = trueskill.rate_1vs1(away_rating, home_rating)
        else:
            new_home, new_away = trueskill.rate_1vs1(
                home_rating, away_rating, drawn=True
            )

        old_home_games = self._trueskill_ratings[home].games_played
        old_away_games = self._trueskill_ratings[away].games_played
        self._trueskill_ratings[home] = TrueSkillRating.from_trueskill(
            new_home, games_played=old_home_games + 1
        )
        self._trueskill_ratings[away] = TrueSkillRating.from_trueskill(
            new_away, games_played=old_away_games + 1
        )

        rev_key = f"{away}-{home}"
        if rev_key not in self._data:
            self._data[rev_key] = {"wins": 0, "draws": 0, "losses": 0, "games": 0}
        if result == 1:
            self._data[rev_key]["losses"] += 1
        elif result == -1:
            self._data[rev_key]["wins"] += 1
        else:
            self._data[rev_key]["draws"] += 1
        self._data[rev_key]["games"] += 1

    def get_win_rate(self, home: str, away: str) -> float:
        """计算贝叶斯胜率（含信度加权收缩）

        当对战场次不足时，胜率向 0.5 收缩。
        """
        key = f"{home}-{away}"
        if key not in self._data:
            return 0.5

        data = self._data[key]
        games = data["games"]
        wins = data["wins"]
        losses = data["losses"]
        draws = data["draws"]

        alpha = self._PSEUDO_COUNT + wins + draws * self._DRAW_WEIGHT
        beta = self._PSEUDO_COUNT + losses + draws * self._DRAW_WEIGHT
        raw_win_rate = alpha / (alpha + beta)
        confidence_weight = min(1.0, games / max(self._min_win_rate_games, 1))
        return 0.5 + confidence_weight * (raw_win_rate - 0.5)

    def decay_all(self) -> None:
        """对所有对战记录应用衰减因子，使旧记录影响递减"""
        for key in self._data:
            d = self._data[key]
            d["wins"] = int(d["wins"] * self._decay)
            d["draws"] = int(d["draws"] * self._decay)
            d["losses"] = int(d["losses"] * self._decay)
            d["games"] = int(d["games"] * self._decay)

    def get_exploitability(self, current_player_id: str) -> float:
        """计算当前智能体的可利用性

        Returns:
            0.0: 远强于对手池
            0.5: 与对手池平均相当
            1.0: 远弱于对手池
        """
        if current_player_id not in self._trueskill_ratings:
            return 0.5

        current_rating = self._trueskill_ratings[current_player_id]
        if not self._players:
            return 0.5

        opponent_mus = []
        for pid in self._players:
            if pid != current_player_id:
                opponent_mus.append(
                    self._trueskill_ratings.get(pid, TrueSkillRating()).mu
                )

        if not opponent_mus:
            return 0.5

        avg_opponent_mu = sum(opponent_mus) / len(opponent_mus)
        diff = avg_opponent_mu - current_rating.mu
        scaled = (
            0.5 + diff / max(current_rating.sigma * 2, 1.0) * self._EXPLOITABILITY_SCALE
        )
        return max(0.0, min(1.0, scaled))

    def get_all_win_rates(self, player_id: str) -> Dict[str, float]:
        """获取该玩家对所有对手的胜率"""
        rates = {}
        for opponent_id in self._players:
            if opponent_id != player_id:
                rates[opponent_id] = self.get_win_rate(player_id, opponent_id)
        return rates

    def get_payoff_matrix(self) -> List[List[float]]:
        """获取完整 payoff 矩阵（n×n 胜率矩阵）"""
        n = len(self._players)
        matrix = [[0.5] * n for _ in range(n)]
        for i, p1 in enumerate(self._players):
            for j, p2 in enumerate(self._players):
                if i != j:
                    matrix[i][j] = self.get_win_rate(p1, p2)
        return matrix

    def get_battle_stats(self) -> Dict[str, Any]:
        """获取所有玩家的对战统计"""
        stats = {}
        for pid in self._players:
            rating = self._trueskill_ratings.get(pid, TrueSkillRating())
            stats[pid] = {
                "mu": rating.mu,
                "sigma": rating.sigma,
                "games_played": rating.games_played,
            }
        return stats

    def save_state(self, filepath: str) -> None:
        """保存评分状态到 JSON"""
        state = {
            "players": self._players,
            "players_ids": self._players_ids,
            "data": self._data,
            "trueskill_ratings": {
                pid: {"mu": r.mu, "sigma": r.sigma, "games_played": r.games_played}
                for pid, r in self._trueskill_ratings.items()
            },
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Payoff state saved to {filepath}")

    def get_trueskill_mus(self, player_ids: List[str]) -> Dict[str, float]:
        """批量获取多个玩家的 TrueSkill mu"""
        result = {}
        for pid in player_ids:
            rating = self._trueskill_ratings.get(pid, TrueSkillRating())
            result[pid] = rating.mu
        return result

    def get_trueskill_sigmas(self, player_ids: List[str]) -> Dict[str, float]:
        """批量获取多个玩家的 TrueSkill sigma"""
        result = {}
        for pid in player_ids:
            rating = self._trueskill_ratings.get(pid, TrueSkillRating())
            result[pid] = rating.sigma
        return result

    def load_state(self, filepath: str) -> None:
        """从 JSON 加载评分状态"""
        with open(filepath, "r") as f:
            state = json.load(f)
        required_fields = ["players", "players_ids", "data", "trueskill_ratings"]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Payoff state missing required field: {field}")
        self._players = state["players"]
        self._players_ids = state["players_ids"]
        self._data = state["data"]
        self._trueskill_ratings = {
            pid: TrueSkillRating(**r) for pid, r in state["trueskill_ratings"].items()
        }
        logger.info(f"Payoff state loaded from {filepath}")
