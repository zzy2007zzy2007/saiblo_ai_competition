"""Leaderboard - a lightweight sorted opponent pool (strongest → weakest).

Replaces WinGraph for ss_train when the strategy space is relatively
monotonic (no circular counter relationships).
"""
from __future__ import annotations

from dataclasses import dataclass
import bisect
from typing import Any

import numpy as np


@dataclass
class LeaderboardEntry:
    """A single entry on the leaderboard."""
    gen: int
    params: np.ndarray
    score: float
    lbd: float = 1.0  # adaptive sampling weight


class Leaderboard:
    """Sorted opponent pool from strongest to weakest.

    Maintains a linear ordering of entries sorted by score descending.
    Assumption: entry[i] beats entry[i+1] (win rate > 50 %).
    """

    def __init__(self, max_size: int = 20, param_count: int = 0,
                 threshold: float = 0.6, max_challenges: int = 1):
        self.max_size = max_size
        self.threshold = threshold
        self.max_challenges = max_challenges
        self.entries: list[LeaderboardEntry] = []  # sorted strongest → weakest

        # Zero-vector placeholder used for cold-start opponent sampling
        # before any real entries exist in the pool.
        self._placeholder = (np.zeros(param_count, dtype=np.float32), 0.0)

    # ── public API ───────────────────────────────────────────────────

    def add_candidate(self, gen: int, params: np.ndarray,
                      score: float | None = None,
                      match_fn=None) -> bool:
        """Try to insert a new candidate, keeping the pool sorted.

        If ``match_fn`` is provided, the candidate challenges entries
        from strongest to weakest and is inserted at the highest rank
        it can beat (score > threshold). If it beats no one, it is
        appended to the tail (if room) or rejected.

        If ``score`` is provided directly (no match_fn), standard
        score-based insertion is used.

        Args:
            gen: generation number.
            params: candidate parameter vector.
            score: pre-computed win rate (optional if match_fn given).
            match_fn: callable ``(params_a, params_b) -> win_rate``.

        Returns:
            True if inserted, False if rejected.
        """
        default_lbd = self.entries[0].lbd if self.entries else 1.0
        if match_fn is not None and self.entries:
            # ── Challenge ladder: strongest → up to max_challenges ──
            for rank, entry in enumerate(self.entries[:self.max_challenges]):
                s = match_fn(params, entry.params)
                if s > self.threshold:
                    # Beats this opponent → insert above it
                    self.entries.insert(rank, LeaderboardEntry(
                        gen=gen, params=params.copy(), score=s, lbd=default_lbd))
                    if len(self.entries) > self.max_size:
                        self.entries.pop()
                    return True
            # Couldn't beat anyone in the top-k → append to tail if room
            if len(self.entries) < self.max_size:
                self.entries.append(LeaderboardEntry(
                    gen=gen, params=params.copy(), score=0.0))
                return True
            return False

        # ── Direct score insertion ──
        if score is None:
            score = 1.0

        if self.entries and score <= self.threshold:
            return False
        if len(self.entries) >= self.max_size and score <= self.entries[-1].score:
            return False

        entry = LeaderboardEntry(gen=gen, params=params.copy(), score=score, lbd=default_lbd)
        neg_scores = [-e.score for e in self.entries]
        pos = bisect.bisect_left(neg_scores, -score)
        self.entries.insert(pos, entry)
        if len(self.entries) > self.max_size:
            self.entries.pop()
        return True

    def get_ranked_entries(self) -> list[dict[str, Any]]:
        """Return all entries sorted strongest → weakest with rank info.

        Returns:
            [{"gen": int, "params": np.ndarray, "rank": int, "score": float}, ...]
            rank=0 is the strongest.
        """
        return [
            {
                "gen": e.gen,
                "params": e.params,
                "rank": rank,
                "score": e.score,
            }
            for rank, e in enumerate(self.entries)
        ]

    def get_strongest(self) -> dict[str, Any] | None:
        """Return the top entry, or None if pool is empty.

        Dict keys: gen, params, score, rank=0.
        """
        if not self.entries:
            return None
        e = self.entries[0]
        return {"gen": e.gen, "params": e.params, "score": e.score, "rank": 0}

    def get_opponents(self, k: int = 5) -> list[dict[str, Any]]:
        """Rank-weighted sampling of k opponents from the pool.

        Sampling is always with replacement when k > pool size, so that
        all k opponents are real strategies (no placeholders). Stronger
        entries are sampled more frequently (exponential decay weights).

        Returns:
            [{"gen": int, "params": np.ndarray}, ...]
        """
        ranked = self.get_ranked_entries()
        n = len(ranked)

        if n == 0:
            # Cold start — empty pool, caller handles fallback
            return []

        weights = self._compute_weights(n)
        rng = np.random.default_rng()
        indices = rng.choice(n, size=k, replace=True, p=weights)

        sampled = [ranked[i] for i in indices]
        return [
            {"gen": d["gen"], "params": d["params"]}
            for d in sampled
        ]

    def get_opponents_adaptive(
        self, k: int = 5,
        target_wr: float = 0.5,
        sigma: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Adaptive weighted sampling — rank × λ.

        λ is updated per-entry based on observed win rates so that
        individuals achieve ~target_wr win rate against the pool.
        """
        ranked = self.get_ranked_entries()
        n = len(ranked)
        if n == 0:
            return []

        # Weights = 0.5^{rank} × λ
        raw_weights = [(0.5 ** i) * self.entries[i].lbd for i in range(n)]
        rng = np.random.default_rng()
        indices = rng.choice(n, size=k, replace=True, p=np.array(raw_weights) / sum(raw_weights))

        sampled = [ranked[i] for i in indices]
        return [{"gen": d["gen"], "params": d["params"]} for d in sampled]

    def update_lambdas(
        self,
        wr_by_gen: dict[int, float],
        target_wr: float = 0.5,
        sigma: float = 0.25,
        momentum: float = 0.3,
    ):
        """Update adaptive λ weights based on observed win rates.

        Args:
            wr_by_gen: {gen: win_rate} — individual win rate against that entry.
            target_wr: desired individual win rate (default 0.5 → balanced).
            sigma: Gaussian kernel width for λ smoothing.
            momentum: λ update momentum (0.9 = smooth, 0 = instant).
        """
        sigma2 = sigma * sigma
        for entry in self.entries:
            wr = wr_by_gen.get(entry.gen, None)
            if wr is not None:
                raw = np.exp(-abs(wr - target_wr) ** 2 / sigma2)
                entry.lbd = momentum * entry.lbd + (1 - momentum) * raw

    def prune(self, max_size: int) -> list[int]:
        """Remove excess weakest entries so that size ≤ max_size.

        Returns the list of removed generation numbers.
        """
        if max_size < 0:
            max_size = 0
        removed = []
        while len(self.entries) > max_size:
            victim = self.entries.pop()
            removed.append(victim.gen)
        return removed

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the leaderboard."""
        return {
            "max_size": self.max_size,
            "param_count": len(self._placeholder[0]),
            "entries": [
                {
                    "gen": e.gen,
                    "params": e.params.tolist(),
                    "score": e.score,
                    "lbd": e.lbd,
                }
                for e in self.entries
            ],
        }

    def load_state_dict(self, d: dict[str, Any]):
        """Restore the leaderboard from a state_dict."""
        self.max_size = d["max_size"]
        param_count = d.get("param_count", 0)
        self._placeholder = (np.zeros(param_count, dtype=np.float32), 0.0)

        self.entries = []
        for ed in d["entries"]:
            self.entries.append(LeaderboardEntry(
                gen=ed["gen"],
                params=np.array(ed["params"], dtype=np.float32),
                score=ed["score"],
                lbd=ed.get("lbd", 1.0),
            ))
        # entries are assumed to already be sorted strongest → weakest

    def get_info(self) -> dict[str, Any]:
        """Return summary statistics for logging / monitoring."""
        strongest = self.get_strongest()
        return {
            "size": len(self.entries),
            "max_size": self.max_size,
            "strongest_score": strongest["score"] if strongest else None,
        }

    # ── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _compute_weights(n: int) -> list[float]:
        """Compute rank-based sampling weights using exponential decay.

        Weight = 0.5^{rank}, then normalized.
        Rank 0 gets ~50%, rank 1 ~25%, rank 2 ~12.5%, etc.
        """
        if n <= 0:
            return []
        raw = [0.5 ** rank for rank in range(n)]
        total = sum(raw)
        return [w / total for w in raw]
