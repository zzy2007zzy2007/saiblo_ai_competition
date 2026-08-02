"""MCTS agent that selects between two expert policy models using a value network.

Architecture:
  Each turn, two expert models (e.g. gen_0030, gen_0101) each propose one action
  bundle.  MCTS simulates both candidates forward to depth D, using a third model
  to generate opponent actions during simulation, and evaluates resulting states
  with a learned value network.  The candidate with the highest average value
  estimate is executed.

  Since branching factor is exactly 2 (one per expert), no UCB/UCT is needed.
  The search degenerates to: expand both candidate paths → value_net evaluates →
  pick the better one.  Depth > 1 means we alternate expert proposals and
  opponent responses for multiple turns before evaluating.

Usage:
    from my_ai.mcts_agent import MCTSAgent
    agent = MCTSAgent(experts=[agent_0030, agent_0101],
                      value_net=value_net,
                      sim_opponent=agent_opp,
                      depth=3)
    ops = agent.choose_operations(state, player)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _CODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from AI.common import BaseAgent
from SDK.backend.state import BackendState
from SDK.utils.features import FeatureExtractor

from my_ai.value_network import ValueNetwork
from my_ai.network import AntWarNetwork


class MCTSAgent(BaseAgent):
    """Agent that uses two expert policy models + value network for MCTS.

    Each turn:
        1. Each expert proposes one action bundle for the current player.
        2. For each candidate: simulate the turn (using sim_opponent for the
           other side), then evaluate the resulting state with the value network.
        3. Optionally recurse deeper (expert proposal + opponent response).
        4. Execute the candidate with the highest average value estimate.
    """

    def __init__(
        self,
        experts: list,
        value_net: ValueNetwork,
        sim_opponent=None,
        depth: int = 3,
        seed: int | None = None,
        max_actions: int = 96,
    ):
        super().__init__(seed=seed, max_actions=max_actions)
        assert len(experts) == 2, "MCTSAgent requires exactly 2 expert models"
        self.experts = experts
        self.value_net = value_net
        self.value_net.eval()
        self.sim_opponent = sim_opponent  # NeuralAgent for opponent simulation
        self.depth = depth
        self.max_actions = max_actions
        self.feature_extractor = FeatureExtractor(max_actions=max_actions)
        self.last_expert_idx: int = -1  # set by choose_operations

        # Decide device from value_net parameters
        self._device = next(self.value_net.parameters()).device

    # ── Public interface ────────────────────────────────────────────────────

    def choose_operations(self, state: BackendState, player: int) -> list:
        """Main entry: run MCTS and return the best operations found."""
        opp_player = 1 - player

        # Step 1: get candidate actions from each expert
        candidates: list[list] = []
        for expert in self.experts:
            ops = expert._choose_operations(state, player)
            candidates.append(ops)

        # Step 2: MCTS to pick the best candidate
        best_idx = self._mcts_search(state, player, opp_player, candidates, self.depth)
        self.last_expert_idx = best_idx  # track which expert was chosen
        return candidates[best_idx]

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles=None,
    ):
        """Wrap choose_operations as an ActionBundle (compatible with game SDK)."""
        operations = self.choose_operations(state, player)
        from SDK.utils.actions import ActionBundle
        score = 0.0
        if operations:
            score = len(operations) * 10.0 + sum(
                5.0 if op.op_type != 11 else 0.0 for op in operations
            )
        return ActionBundle(
            name="mcts",
            operations=tuple(operations),
            score=score,
            tags=("mcts",),
        )

    # ── MCTS internals ──────────────────────────────────────────────────────

    def _mcts_search(
        self,
        state: BackendState,
        player: int,
        opp_player: int,
        candidates: list[list],
        depth: int,
    ) -> int:
        """Run MCTS search and return the index of the best candidate.

        For each candidate action, simulate forward and evaluate the resulting
        state.  When depth > 1, recursively expand further (experts propose
        next actions, opponent responds, evaluate deeper).  The value returned
        for each candidate is the best reachable value from that branch (not
        an average), since we deterministically select the expert's best action
        at each level.

        Args:
            state: Current game state (root node).
            player: Our player index.
            opp_player: Opponent player index.
            candidates: List of operation lists from each expert.
            depth: Remaining search depth.

        Returns:
            Index into ``candidates`` with the highest value.
        """
        values = [self._simulate_and_value(state, player, opp_player, ops, depth)
                  for ops in candidates]

        if not values:
            return 0
        return int(np.argmax(values))

    def _simulate_and_value(
        self,
        state: BackendState,
        player: int,
        opp_player: int,
        ops: list,
        depth: int,
    ) -> float:
        """Simulate one turn with ``ops`` and return a value estimate.

        Steps:
          1. Get opponent's response via sim_opponent.
          2. Clone state and resolve the turn.
          3. If depth > 1 and not terminal: recurse with the best next action.
             Otherwise: evaluate with value_net.

        Returns:
            Value estimate (positive = good for us).
        """
        # ── Get opponent action for simulation ──────────────────────────
        if self.sim_opponent is not None:
            opp_ops = self.sim_opponent._choose_operations(state, opp_player)
        else:
            opp_ops = []

        # ── Clone state and simulate one turn ───────────────────────────
        sim_state = state.clone()
        if player == 0:
            sim_state.resolve_turn(ops, opp_ops)
        else:
            sim_state.resolve_turn(opp_ops, ops)

        # ── Evaluate ────────────────────────────────────────────────────
        if depth > 1 and not sim_state.terminal:
            # Recurse: get next candidates from both experts
            next_candidates: list[list] = []
            for expert in self.experts:
                next_ops = expert._choose_operations(sim_state, player)
                next_candidates.append(next_ops)
            best_idx = self._mcts_search(
                sim_state, player, opp_player, next_candidates, depth - 1
            )
            best_ops = next_candidates[best_idx]
            return self._simulate_and_value(
                sim_state, player, opp_player, best_ops, depth - 1
            )
        else:
            return self._evaluate(sim_state, player)

    @torch.no_grad()
    def _evaluate(self, state: BackendState, player: int) -> float:
        """Use the value network to estimate state value from player's view.

        Returns:
            Scalar value.  Positive → player leads; negative → player trails.
        """
        obs = self.feature_extractor.encode_observation(
            state, player, np.zeros(self.max_actions)
        )
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float().to(self._device)
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float().to(self._device)

        value = self.value_net(board, stats)  # (1,) tensor
        return value.item()


def create_mcts_agent(
    model_a_path: str,
    model_b_path: str,
    value_net_path: str,
    opponent_path: str | None = None,
    depth: int = 3,
    small: bool = False,
    device: str = "cuda",
) -> MCTSAgent:
    """Factory function: load checkpoints and build an MCTSAgent.

    Args:
        model_a_path: Path to first expert checkpoint (e.g. gen_0030.pt).
        model_b_path: Path to second expert checkpoint (e.g. gen_0101.pt).
        value_net_path: Path to value network checkpoint (.pt).
        opponent_path: Optional path to opponent model.  If None, uses
            model_a as the simulation opponent.
        depth: MCTS search depth.
        small: Whether the value network is the small variant.
        device: Torch device string.

    Returns:
        Configured MCTSAgent ready for use.
    """
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent
    from my_ai.value_network import create_value_network, create_small_value_network

    # ── Load value network ──────────────────────────────────────────────
    print(f"[MCTS] Loading value network from {value_net_path}")
    if small:
        vnet = create_small_value_network()
    else:
        vnet = create_value_network()
    ckpt = torch.load(value_net_path, map_location=device, weights_only=True)
    if "state_dict" in ckpt:
        vnet.load_state_dict(ckpt["state_dict"])
    else:
        vnet.load_state_dict(ckpt)
    vnet.to(device)
    vnet.eval()
    print(f"[MCTS] Value network loaded ({vnet.count_parameters():,} params)")

    # ── Load expert models ──────────────────────────────────────────────
    def _load_expert(path: str) -> NeuralAgent:
        print(f"[MCTS] Loading expert from {path}")
        try:
            ckpt_model = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt_model = torch.load(path, map_location="cpu", weights_only=False)
        num_heads = ckpt_model.get("num_heads", 3)
        is_small = ckpt_model["mean"].shape[0] < 200000
        is_no_bn = ckpt_model.get("no_bn", False)
        model = create_model(num_heads=num_heads, small=is_small, no_bn=is_no_bn)
        if "mean" in ckpt_model:
            model.set_parameters_from_vector(ckpt_model["mean"].numpy())
        elif "top2_params" in ckpt_model and len(ckpt_model["top2_params"]) > 0:
            model.set_parameters_from_vector(ckpt_model["top2_params"][0].numpy())
        elif "model_state" in ckpt_model:
            model.load_state_dict(ckpt_model["model_state"])
        else:
            # Last resort: try any vector-like key
            for k in ckpt_model:
                v = ckpt_model[k]
                if hasattr(v, "shape") and len(v.shape) == 1 and v.shape[0] > 1000:
                    model.set_parameters_from_vector(v.numpy())
                    break
        model.to(device)
        model.eval()
        return NeuralAgent(model=model)

    experts = [_load_expert(model_a_path), _load_expert(model_b_path)]

    # ── Load simulation opponent ────────────────────────────────────────
    sim_opponent = None
    if opponent_path:
        sim_opponent = _load_expert(opponent_path)
    else:
        sim_opponent = experts[0]  # default: use first expert
    print(f"[MCTS] Simulation opponent: {opponent_path or 'model_a (default)'}")

    agent = MCTSAgent(
        experts=experts,
        value_net=vnet,
        sim_opponent=sim_opponent,
        depth=depth,
    )
    print(f"[MCTS] Agent ready (depth={depth}, device={device})")
    return agent
