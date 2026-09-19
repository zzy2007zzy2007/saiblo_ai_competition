from __future__ import annotations

import os
import random
from typing import List, Optional

from loguru import logger

from .payoff import BattleSharedPayoff


class OpponentSelector:
    def __init__(
        self,
        payoff: BattleSharedPayoff,
        exploit_prob: float = 0.7,
        adaptive: bool = True,
        schedule: str = "linear",
    ) -> None:
        self.payoff = payoff
        self.base_exploit_prob = exploit_prob
        self.adaptive = adaptive
        self.schedule = schedule
        self._min_confidence_sigma = 5.0

        self._total_episodes = 0
        self._total_selections = 0

    def update_progress(self, episodes: int) -> None:
        """更新训练进度"""
        self._total_episodes = episodes

    def _get_adaptive_exploit_prob(self) -> float:
        """根据训练进度计算动态exploit_prob"""
        if not self.adaptive:
            return self.base_exploit_prob

        progress = min(1.0, self._total_episodes / 10000.0)

        if self.schedule == "linear":
            return 0.3 + 0.6 * progress
        elif self.schedule == "sigmoid":
            import math
            return 0.3 + 0.6 * (1 / (1 + math.exp(-10 * (progress - 0.5))))
        elif self.schedule == "exp":
            import math
            return 0.3 + 0.6 * (1 - math.exp(-5 * progress))
        else:
            return self.base_exploit_prob

    def _is_confident_opponent(self, opponent_id: str) -> bool:
        rating = self.payoff.get_trueskill_rating(opponent_id)
        if rating is None:
            return False
        return rating.sigma < self._min_confidence_sigma

    def select(self, player_id: str, opponent_candidates: List[str]) -> str:
        if not opponent_candidates:
            raise ValueError("No opponent candidates available")

        exploit_prob = self._get_adaptive_exploit_prob()

        if random.random() < exploit_prob:
            selected = self._exploit_select(player_id, opponent_candidates)
            if selected is not None:
                self._total_selections += 1
                return selected

        self._total_selections += 1
        return self._explore_select(opponent_candidates)

    def _exploit_select(self, player_id: str, opponent_candidates: List[str]) -> Optional[str]:
        if not opponent_candidates:
            return None

        confident_opponents = [
            opp_id for opp_id in opponent_candidates
            if self._is_confident_opponent(opp_id)
        ]

        if confident_opponents:
            rated_opponents = [
                (opp_id, self.payoff.get_trueskill_rating(opp_id).mu)
                for opp_id in confident_opponents
            ]
            rated_opponents.sort(key=lambda x: x[1], reverse=True)
            top_opponents = [opp_id for opp_id, _ in rated_opponents[:min(3, len(rated_opponents))]]
            return random.choice(top_opponents) if top_opponents else None

        all_ratings = [
            (opp_id, self.payoff.get_trueskill_rating(opp_id).sigma)
            for opp_id in opponent_candidates
            if self.payoff.get_trueskill_rating(opp_id) is not None
        ]
        if not all_ratings:
            return None
        all_ratings.sort(key=lambda x: x[1])
        return all_ratings[0][0]

    def _explore_select(self, opponent_candidates: List[str]) -> str:
        if not opponent_candidates:
            raise ValueError("No opponent candidates available")

        filtered = [
            opp_id for opp_id in opponent_candidates
            if not self.payoff.get_trueskill_rating(opp_id) or
               self.payoff.get_trueskill_rating(opp_id).sigma < self._min_confidence_sigma * 1.5
        ]

        if not filtered:
            filtered = opponent_candidates

        return random.choice(filtered)

    def select_batch(
        self,
        player_id: str,
        opponent_candidates: List[str],
        num_selections: int = 1,
    ) -> List[str]:
        if num_selections == 1:
            return [self.select(player_id, opponent_candidates)]

        results = []
        available = list(opponent_candidates)
        for _ in range(num_selections):
            if not available:
                available = list(opponent_candidates)
            selected = self.select(player_id, available)
            results.append(selected)
            available.remove(selected)
        return results

    def get_battle_stats(self) -> dict:
        if hasattr(self.payoff, 'get_battle_stats'):
            return self.payoff.get_battle_stats()
        return {}

    def get_recent_battle_stats(self) -> dict:
        return self.get_battle_stats()


class OpponentPool:
    def __init__(self, max_size: int = 10, min_games_threshold: int = 10,
                 eviction_log_dir: Optional[str] = None) -> None:
        self.max_size = max_size
        self.min_games_threshold = min_games_threshold
        self._opponents: List[str] = []
        self._checkpoint_paths: Dict[str, str] = {}
        self._added_timestamps: Dict[str, float] = {}
        self._added_episodes: Dict[str, int] = {}
        self._games_played: Dict[str, int] = {}
        self._payoff = None
        self._current_episode: int = 0
        self._eviction_log_dir: Optional[str] = eviction_log_dir

    def set_payoff_reference(self, payoff) -> None:
        """设置payoff引用，用于淘汰决策"""
        self._payoff = payoff

    def set_current_episode(self, episode: int) -> None:
        """设置当前 episode 编号，用于淘汰日志"""
        self._current_episode = episode

    def add(self, opponent_id: str, checkpoint_path: str = None, episode: int = None) -> None:
        if opponent_id not in self._opponents:
            self._opponents.append(opponent_id)
            self._added_timestamps[opponent_id] = 0
            self._added_episodes[opponent_id] = episode if episode is not None else 0
            self._games_played[opponent_id] = 0
            if checkpoint_path:
                self._checkpoint_paths[opponent_id] = checkpoint_path
            if len(self._opponents) > self.max_size:
                self._evict_worst()

    def _evict_worst(self) -> None:
        """淘汰策略：低场次对手获得保护加成，评分最低的淘汰"""
        def eviction_score(opp_id: str) -> float:
            games = self._games_played.get(opp_id, 0)
            if self._payoff is not None:
                rating = self._payoff.get_trueskill_rating(opp_id)
                mu = rating.mu if rating else 25.0
            else:
                mu = 25.0

            if games < self.min_games_threshold:
                protection_bonus = (self.min_games_threshold - games) * 0.5
            else:
                protection_bonus = 0

            return mu + protection_bonus

        victim = min(self._opponents, key=eviction_score)

        victim_games = self._games_played.get(victim, 0)
        victim_mu = 25.0
        if self._payoff is not None:
            victim_rating = self._payoff.get_trueskill_rating(victim)
            if victim_rating is not None:
                victim_mu = victim_rating.mu
        victim_protection = 0.0
        if victim_games < self.min_games_threshold:
            victim_protection = (self.min_games_threshold - victim_games) * 0.5
        victim_final_score = victim_mu + victim_protection
        pool_size_before = len(self._opponents)

        logger.info(
            f"[对手池淘汰] victim={victim[:20]} "
            f"mu={victim_mu:.2f} protection={victim_protection:.2f} "
            f"final_score={victim_final_score:.2f} "
            f"games={victim_games} "
            f"ep={self._current_episode} "
            f"pool={pool_size_before}→{pool_size_before - 1}/{self.max_size}"
        )

        if self._eviction_log_dir is not None:
            import json, time as time_module
            eviction_record = {
                'timestamp': time_module.strftime('%Y-%m-%dT%H:%M:%S'),
                'episode': self._current_episode,
                'victim_id': victim,
                'victim_mu': round(victim_mu, 2),
                'victim_protection': round(victim_protection, 2),
                'victim_final_score': round(victim_final_score, 2),
                'victim_games': victim_games,
                'pool_size_before': pool_size_before,
                'pool_size_after': pool_size_before - 1,
                'max_size': self.max_size,
            }
            eviction_file = os.path.join(self._eviction_log_dir, "eviction_history.jsonl")
            try:
                with open(eviction_file, 'a') as f:
                    f.write(json.dumps(eviction_record, ensure_ascii=False) + '\n')
            except IOError:
                pass

        self._remove_opponent(victim)

    def _remove_opponent(self, opponent_id: str) -> None:
        if opponent_id in self._opponents:
            self._opponents.remove(opponent_id)
        self._on_remove(opponent_id)

    def _on_remove(self, opponent_id: str) -> None:
        """淘汰钩子，可用于清理资源"""
        if opponent_id in self._checkpoint_paths:
            self._checkpoint_paths.pop(opponent_id)
        if opponent_id in self._added_timestamps:
            self._added_timestamps.pop(opponent_id)
        if opponent_id in self._added_episodes:
            self._added_episodes.pop(opponent_id)
        if opponent_id in self._games_played:
            self._games_played.pop(opponent_id)

    def increment_games(self, opponent_id: str) -> None:
        """每对战一场后调用，增加对战场次"""
        if opponent_id in self._games_played:
            self._games_played[opponent_id] += 1

    def get_checkpoint_path(self, opponent_id: str) -> Optional[str]:
        return self._checkpoint_paths.get(opponent_id)

    def get_all(self) -> List[str]:
        return list(self._opponents)

    def get_random(self) -> str:
        if not self._opponents:
            raise ValueError("Opponent pool is empty")
        return random.choice(self._opponents)

    def remove(self, opponent_id: str) -> bool:
        if opponent_id in self._opponents:
            self._opponents.remove(opponent_id)
            self._on_remove(opponent_id)
            return True
        return False

    def __len__(self) -> int:
        return len(self._opponents)

    def __contains__(self, opponent_id: str) -> bool:
        return opponent_id in self._opponents

    def save_state(self, filepath: str) -> None:
        """保存对手池状态"""
        import json
        state = {
            'max_size': self.max_size,
            'min_games_threshold': self.min_games_threshold,
            'opponent_ids': self._opponents,
            'checkpoint_paths': self._checkpoint_paths,
            'added_timestamps': self._added_timestamps,
            'added_episodes': self._added_episodes,
            'games_played': self._games_played,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> "OpponentPool":
        """加载对手池状态"""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        pool = cls(
            max_size=state.get('max_size', 10),
            min_games_threshold=state.get('min_games_threshold', 10)
        )
        pool._opponents = state.get('opponent_ids', [])
        pool._checkpoint_paths = state.get('checkpoint_paths', {})
        pool._added_timestamps = state.get('added_timestamps', {})
        pool._added_episodes = state.get('added_episodes', {})
        pool._games_played = state.get('games_played', {})
        return pool
