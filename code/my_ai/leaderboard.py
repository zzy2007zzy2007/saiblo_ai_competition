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


class Leaderboard:
    """Sorted opponent pool from strongest to weakest.

    Maintains a linear ordering of entries sorted by score descending.
    Assumption: entry[i] beats entry[i+1] (win rate > 50 %).
    """

    def __init__(self, max_size: int = 20, param_count: int = 0,
                 threshold: float = 0.5):
        self.max_size = max_size
        self.threshold = threshold
        self.entries: list[LeaderboardEntry] = []  # sorted strongest → weakest

        # ExampleAI placeholder used for cold-start opponent sampling
        self.example_ai = (np.zeros(param_count, dtype=np.float32), 0.0)

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
        if match_fn is not None and self.entries:
            # ── Challenge ladder: strongest → weakest ──
            for rank, entry in enumerate(self.entries):
                s = match_fn(params, entry.params)
                if s > self.threshold:
                    # Beats this opponent → insert above it
                    self.entries.insert(rank, LeaderboardEntry(
                        gen=gen, params=params.copy(), score=s))
                    if len(self.entries) > self.max_size:
                        self.entries.pop()
                    return True
            # Couldn't beat anyone → append to tail (if room)
            if len(self.entries) < self.max_size:
                s = match_fn(params, self.entries[-1].params)
                self.entries.append(LeaderboardEntry(
                    gen=gen, params=params.copy(), score=s))
                return True
            return False

        # ── Direct score insertion ──
        if score is None:
            score = 1.0

        if self.entries and score <= self.threshold:
            return False
        if len(self.entries) >= self.max_size and score <= self.entries[-1].score:
            return False

        entry = LeaderboardEntry(gen=gen, params=params.copy(), score=score)
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

        The returned list has no 'score' field, matching the WinGraph
        interface for ``select_opponents``.

        If the pool is empty, all k opponents are ExampleAI placeholders.
        If the pool has fewer than k entries, available entries are
        sampled and the remainder are filled with ExampleAI.

        Returns:
            [{"gen": int, "params": np.ndarray}, ...]
        """
        ranked = self.get_ranked_entries()
        n = len(ranked)

        if n == 0:
            # Cold start — all ExampleAI
            return [
                {"gen": -1, "params": self.example_ai[0].copy()}
                for _ in range(k)
            ]

        # Compute normalised weights
        weights = self._compute_weights(n)

        # Sample without replacement, as many as possible
        sample_size = min(k, n)
        rng = np.random.default_rng()
        indices = rng.choice(n, size=sample_size, replace=False, p=weights)
        sampled = [ranked[i] for i in indices]

        # Sort sampled entries by rank (strongest first)
        sampled.sort(key=lambda d: d["rank"])

        result = [
            {"gen": d["gen"], "params": d["params"]}
            for d in sampled
        ]

        # Fill remaining slots with ExampleAI
        while len(result) < k:
            result.append({
                "gen": -1,
                "params": self.example_ai[0].copy(),
            })

        return result

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
            "param_count": len(self.example_ai[0]),
            "entries": [
                {
                    "gen": e.gen,
                    "params": e.params.tolist(),
                    "score": e.score,
                }
                for e in self.entries
            ],
        }

    def load_state_dict(self, d: dict[str, Any]):
        """Restore the leaderboard from a state_dict."""
        self.max_size = d["max_size"]
        param_count = d.get("param_count", 0)
        self.example_ai = (np.zeros(param_count, dtype=np.float32), 0.0)

        self.entries = []
        for ed in d["entries"]:
            self.entries.append(LeaderboardEntry(
                gen=ed["gen"],
                params=np.array(ed["params"], dtype=np.float32),
                score=ed["score"],
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
        """Compute rank-based sampling weights for *n* entries.

        Rank 0 → 0.30, rank 1 → 0.20, rank 2 → 0.15,
        remaining ranks share the leftover weight uniformly.
        """
        if n <= 0:
            return []

        fixed = {0: 0.30, 1: 0.20, 2: 0.15}
        weights = []

        if n <= 3:
            # Only use the first n entries from fixed weights
            for rank in range(n):
                weights.append(fixed[rank])
        else:
            remaining = 1.0 - 0.30 - 0.20 - 0.15
            per_rest = remaining / (n - 3)
            for rank in range(n):
                if rank in fixed:
                    weights.append(fixed[rank])
                else:
                    weights.append(per_rest)

        # Normalise (handles floating point drift and n < 3 cases)
        total = sum(weights)
        return [w / total for w in weights]
