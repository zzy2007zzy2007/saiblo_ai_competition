from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import trueskill


@dataclass
class TrueSkillRating:
    mu: float = 25.0
    sigma: float = 8.33
    games_played: int = 0

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return (self.mu - 3 * self.sigma, self.mu + 3 * self.sigma)

    @property
    def is_confident(self) -> bool:
        return self.sigma < 3.0

    def to_trueskill(self) -> trueskill.Rating:
        return trueskill.Rating(mu=self.mu, sigma=self.sigma)

    @classmethod
    def from_trueskill(cls, rating: trueskill.Rating, games_played: int = 0) -> "TrueSkillRating":
        return cls(mu=rating.mu, sigma=rating.sigma, games_played=games_played)


class BattleRecordDict(dict):
    data_keys = ['wins', 'draws', 'losses', 'games']

    def __init__(self) -> None:
        super(BattleRecordDict, self).__init__()
        for k in self.data_keys:
            self[k] = 0

    def __mul__(self, decay: float) -> dict:
        obj = copy.deepcopy(self)
        for k in obj.keys():
            obj[k] *= decay
        return obj


class BattleSharedPayoff:
    def __init__(
        self,
        decay: float = 0.99,
        min_win_rate_games: int = 8,
    ) -> None:
        self._players = []
        self._players_ids = []
        self._data: Dict[str, BattleRecordDict] = defaultdict(BattleRecordDict)
        self._decay = decay
        self._min_win_rate_games = min_win_rate_games
        self._trueskill_ratings: Dict[str, TrueSkillRating] = {}

    @property
    def players(self) -> list:
        return self._players

    def _get_or_create_rating(self, player_id: str) -> TrueSkillRating:
        if player_id not in self._trueskill_ratings:
            self._trueskill_ratings[player_id] = TrueSkillRating()
        return self._trueskill_ratings[player_id]

    def add_player(self, player_id: str) -> None:
        if player_id not in self._players_ids:
            self._players.append(player_id)
            self._players_ids.append(player_id)

    def update(self, home: str, away: str, result: int) -> None:
        self.add_player(home)
        self.add_player(away)

        key = self.get_key(home, away)
        record = self._data[key]

        record['games'] += 1
        if result == 0:
            record['draws'] += 1
        elif result == 1:
            record['wins'] += 1
        else:
            record['losses'] += 1

        reverse_key = self.get_key(away, home)
        reverse_record = self._data[reverse_key]
        reverse_record['games'] += 1
        if result == 0:
            reverse_record['draws'] += 1
        elif result == 1:
            reverse_record['losses'] += 1
        else:
            reverse_record['wins'] += 1

        home_rating = self._get_or_create_rating(home)
        away_rating = self._get_or_create_rating(away)

        ts_home = home_rating.to_trueskill()
        ts_away = away_rating.to_trueskill()

        if result == 1:
            new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away)
        elif result == -1:
            new_away, new_home = trueskill.rate_1vs1(ts_away, ts_home)
        else:
            new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away, drawn=True)

        self._trueskill_ratings[home] = TrueSkillRating.from_trueskill(
            new_home, games_played=home_rating.games_played + 1
        )
        self._trueskill_ratings[away] = TrueSkillRating.from_trueskill(
            new_away, games_played=away_rating.games_played + 1
        )

    def get_key(self, home: str, away: str) -> str:
        return f'{home}-{away}'

    def get_win_rate(self, home: str, away: str) -> float:
        key = self.get_key(home, away)
        if key not in self._data:
            return 0.5

        record = self._data[key]
        games = record['games']

        if games == 0:
            return 0.5

        alpha = 1 + record['wins'] + record['draws'] * 0.5
        beta = 1 + record['losses'] + record['draws'] * 0.5

        win_rate = alpha / (alpha + beta)

        confidence_weight = min(1.0, games / self._min_win_rate_games)
        win_rate = 0.5 + confidence_weight * (win_rate - 0.5)

        return win_rate

    def decay_all(self) -> None:
        for key in self._data:
            self._data[key] = self._data[key] * self._decay

    def get_all_win_rates(self, player_id: str) -> Dict[str, float]:
        win_rates = {}
        for opp_id in self._players_ids:
            if opp_id != player_id:
                win_rates[opp_id] = self.get_win_rate(player_id, opp_id)
        return win_rates

    def get_payoff_matrix(self) -> np.ndarray:
        n = len(self._players)
        matrix = np.zeros((n, n))
        for i, home in enumerate(self._players):
            for j, away in enumerate(self._players):
                if i != j:
                    matrix[i, j] = self.get_win_rate(home, away)
        return matrix

    def get_battle_stats(self) -> Dict[str, Dict]:
        stats = {}
        for player_id in self._players_ids:
            total_wins = 0
            total_losses = 0
            total_draws = 0
            total_games = 0
            total_reward = 0.0

            for opp_id in self._players_ids:
                if opp_id == player_id:
                    continue
                key = self.get_key(player_id, opp_id)
                if key in self._data:
                    record = self._data[key]
                    total_wins += record['wins']
                    total_losses += record['losses']
                    total_draws += record['draws']
                    total_games += record['games']

            if total_games > 0:
                stats[player_id] = {
                    'wins': total_wins,
                    'losses': total_losses,
                    'draws': total_draws,
                    'total_battles': total_games,
                    'win_rate': (total_wins + total_draws * 0.5) / total_games if total_games > 0 else 0.0,
                    'avg_reward': 0.0
                }

        return stats

    def get_trueskill_rating(self, player_id: str) -> Optional[TrueSkillRating]:
        return self._trueskill_ratings.get(player_id)

    def get_all_trueskill_ratings(self) -> Dict[str, TrueSkillRating]:
        return dict(self._trueskill_ratings)

    def get_exploitability(self) -> float:
        if not self._trueskill_ratings:
            return 1.0

        current = self._trueskill_ratings.get("current_agent")
        if current is None:
            return 1.0

        opponent_ratings = [
            r for pid, r in self._trueskill_ratings.items()
            if pid != "current_agent" and r.games_played > 0
        ]

        if not opponent_ratings:
            return 1.0

        avg_mu = sum(r.mu for r in opponent_ratings) / len(opponent_ratings)
        avg_sigma = sum(r.sigma for r in opponent_ratings) / len(opponent_ratings)

        if current.mu <= avg_mu:
            advantage = (avg_mu - current.mu) / (current.sigma + avg_sigma + 1e-6)
            return min(1.0, 0.5 + advantage * 0.5)
        else:
            advantage = (current.mu - avg_mu) / (current.sigma + avg_sigma + 1e-6)
            return max(0.0, 0.5 - advantage * 0.5)

    def save_state(self, filepath: str) -> None:
        """保存payoff状态"""
        import json
        state = {
            'players': self._players,
            'players_ids': self._players_ids,
            'trueskill_ratings': {
                pid: {
                    'mu': rating.mu,
                    'sigma': rating.sigma,
                    'games_played': rating.games_played
                }
                for pid, rating in self._trueskill_ratings.items()
            },
            'battle_records': dict(self._data),
            'decay': self._decay,
            'min_win_rate_games': self._min_win_rate_games,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> "BattleSharedPayoff":
        """加载payoff状态"""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        payoff = cls(
            decay=state.get('decay', 0.99),
            min_win_rate_games=state.get('min_win_rate_games', 8)
        )
        payoff._players = state.get('players', [])
        payoff._players_ids = state.get('players_ids', [])
        payoff._trueskill_ratings = {
            pid: TrueSkillRating(**rating_dict)
            for pid, rating_dict in state.get('trueskill_ratings', {}).items()
        }
        payoff._data = defaultdict(BattleRecordDict, state.get('battle_records', {}))
        return payoff
