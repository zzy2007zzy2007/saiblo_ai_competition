"""P2: Self-play loop — run games, collect AlphaZero training samples.

Each round, both players play up to 3 operations (one per policy head), each
decided by its own intent-space MCTS search.  After P1's third head the round
is resolved (advance_round).  Every search records a training sample:
observation + the full legal intent space + the temperature-scaled visit
distribution over it (the policy target for that head).

The value target is the terminal saturated HP-difference, assigned to every
sample of the game from the deciding player's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND

from my_ai.az_intent.mcts import HP_SCALE, IntentMCTS, intent_to_operation


@dataclass(slots=True)
class GameSamples:
    samples: list = field(default_factory=list)
    winner: int | None = None  # 0/1/None
    rounds: int = 0
    hp: tuple = (0, 0)

    @property
    def num_samples(self) -> int:
        return len(self.samples)


def terminal_value_target(state: GameState, player: int) -> float:
    """Saturated HP-difference value for the terminal state, from ``player``'s view."""
    if not state.terminal:
        return 0.0
    diff = state.bases[player].hp - state.bases[1 - player].hp
    v = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
    if diff == 0 and state.winner is not None:
        v = 0.1 if state.winner == player else -0.1
    return v


def run_game(
    net_fn,
    feature_extractor,
    mcts: IntentMCTS,
    seed: int,
    *,
    max_rounds: int = MAX_ROUND,
    temp_rounds: int = 30,
    max_actions: int = 96,
) -> GameSamples:
    """Play one full self-play game and collect training samples."""
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    samples: list[dict] = []

    for round_idx in range(max_rounds):
        if state.terminal:
            break
        temperature = 1.0 if round_idx < temp_rounds else 1e-6
        explore = round_idx < temp_rounds
        for player in (0, 1):
            if state.terminal:
                break
            for head_idx in (0, 1, 2):
                if state.terminal:
                    break
                res = mcts.search(
                    state, player, head_idx=head_idx,
                    temperature=temperature, add_root_noise=explore,
                )
                obs = feature_extractor.encode_observation(state, player, np.zeros(max_actions))
                samples.append(
                    {
                        "board": obs["board"].astype(np.float16),
                        "stats": obs["stats"].astype(np.float16),
                        "head_idx": head_idx,
                        "player": player,
                        "intent_class": np.asarray(
                            [it.class_id for it in res.full_intents], dtype=np.int8
                        ),
                        "intent_x": np.asarray([it.x for it in res.full_intents], dtype=np.int16),
                        "intent_y": np.asarray([it.y for it in res.full_intents], dtype=np.int16),
                        "target": res.full_target.astype(np.float32),
                    }
                )
                op = intent_to_operation(state, player, res.chosen_intent)
                if op is not None:
                    state.apply_operation_list(player, [op])
            if player == 1 and not state.terminal:
                state.advance_round()

    # Terminal value from each player's perspective, assigned to all samples.
    v_p0 = terminal_value_target(state, 0)
    for s in samples:
        s["value_target"] = v_p0 if s["player"] == 0 else -v_p0

    return GameSamples(
        samples=samples,
        winner=state.winner,
        rounds=state.round_index,
        hp=(state.bases[0].hp, state.bases[1].hp),
    )
