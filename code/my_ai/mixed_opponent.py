"""Mixed Strategy Opponent — per-turn sampling from random / ExampleAI / rule_v4.

Usage in _eval_worker::

    from my_ai.mixed_opponent import MixedStrategyOpponent
    opponent = MixedStrategyOpponent(probs=(0.2, 0.5, 0.3))
    ops_opp = opponent.choose_operations(state, opp_player)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure Ant-Game and rule_v4 paths
_ANT_GAME = Path(__file__).resolve().parents[2] / "Ant-Game"
_RULE_V4 = Path(__file__).resolve().parents[2] / "其他版本ai" / "rule_v4"
for p in (_ANT_GAME, _RULE_V4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from SDK.backend.state import BackendState
from AI.ai_random import RandomAgent
from AI.ai_example import ExampleAgent

# rule_v4 — lazy import (heavy, only when needed for strategy 2)
_RULE_V4_AGENT: "RuleBasedAI | None" = None


def _get_rule_v4():
    global _RULE_V4_AGENT
    if _RULE_V4_AGENT is None:
        # pylint: disable=import-outside-toplevel
        from ai import RuleBasedAI  # type: ignore
        _RULE_V4_AGENT = RuleBasedAI()
    return _RULE_V4_AGENT


class MixedStrategyOpponent:
    """Per-turn opponent that samples from three strategies.

    Each turn, an action source is selected with probability (a, b, c):
      a — RandomAgent (fully random, produces diverse board states)
      b — ExampleAgent (reference AI, moderate strength)
      c — rule_v4 (strong rule-based AI)

    The ``probs`` tuple is continuously adjusted during training to keep
    the win rate near the target (see ``adjust_probs``).
    """

    STRATEGY_RANDOM = 0
    STRATEGY_EXAMPLE = 1
    STRATEGY_RULE_V4 = 2

    def __init__(
        self,
        probs: tuple[float, float, float] = (0.33, 0.33, 0.34),
        seed: int | None = None,
    ):
        assert len(probs) == 3 and abs(sum(probs) - 1.0) < 1e-6, \
            f"probs must sum to 1, got {probs}"
        self.probs = probs
        self.rng = np.random.RandomState(seed)

        # Lazy init: create agents on first use (so this works in mp workers)
        self._random_agent: RandomAgent | None = None
        self._example_agent: ExampleAgent | None = None
        # rule_v4 is lazily imported via _get_rule_v4

    def _get_random(self) -> RandomAgent:
        if self._random_agent is None:
            self._random_agent = RandomAgent()
        return self._random_agent

    def _get_example(self) -> ExampleAgent:
        if self._example_agent is None:
            self._example_agent = ExampleAgent()
        return self._example_agent

    def choose_operations(self, state: "BackendState", player: int) -> list:
        """Choose operations for the opponent this turn.

        Returns a list of Operation objects (can be empty).
        """
        source = self.rng.choice(3, p=self.probs)

        if source == self.STRATEGY_RANDOM:
            agent = self._get_random()
            bundles = agent.list_bundles(state, player)
            bundle = agent.choose_bundle(state, player, bundles)
            return list(bundle.operations)

        elif source == self.STRATEGY_EXAMPLE:
            agent = self._get_example()
            bundles = agent.list_bundles(state, player)
            bundle = agent.choose_bundle(state, player, bundles)
            return list(bundle.operations)

        else:  # STRATEGY_RULE_V4
            rule = _get_rule_v4()
            bundles = rule.list_bundles(state, player)
            bundle = rule.choose_bundle(state, player, bundles)
            return list(bundle.operations)

    def _choose_operations(self, state: "BackendState", player: int) -> list:
        """Alias for choose_operations (used by _eval_worker's uniform interface)."""
        return self.choose_operations(state, player)


def adjust_probs(
    current_probs: tuple[float, float, float],
    win_rate: float,
    target_rate: float = 0.55,
    step: float = 0.1,
) -> tuple[float, float, float]:
    """Dynamically adjust (a, b, c) to keep win_rate near target_rate.

    When win_rate is too high → increase difficulty: shift weight from
    random (a) toward rule_v4 (c).

    When win_rate is too low → decrease difficulty: shift weight from
    rule_v4 (c) toward random (a).

    Returns:
        New (a, b, c) tuple that sums to 1.
    """
    a, b, c = current_probs

    if win_rate > target_rate + 0.05:  # too easy → harder
        shift = min(a, step)
        a -= shift
        c = min(1.0, c + shift)
    elif win_rate < target_rate - 0.05:  # too hard → easier
        shift = min(c, step)
        c -= shift
        a = min(1.0, a + shift)
    else:
        return current_probs  # in sweet spot, no change

    # Renormalize to sum to 1
    total = a + b + c
    return (a / total, b / total, c / total)
