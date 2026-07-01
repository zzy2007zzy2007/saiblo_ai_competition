"""Neural AI for 蚁洋陷役2 - Submission entry point."""
from __future__ import annotations
from pathlib import Path

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.backend import BackendState
from SDK.utils.actions import ActionBundle

import torch
import numpy as np

# Import our model
import sys
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from my_ai.network import create_model
from my_ai.decoder import decode_network_output


class NeuralAgent(BaseAgent):
    """Neural network based AI agent."""

    def __init__(self, seed=None, max_actions=96):
        super().__init__(seed=seed, max_actions=max_actions)
        self.max_actions = max_actions
        ckpt_path = _THIS_DIR / "gen_0006.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.num_heads = ckpt.get("num_heads", 3)
        self.params = ckpt["top2_params"][0].numpy()
        self.model = create_model(num_heads=self.num_heads)
        self.model.set_parameters_from_vector(self.params)
        self.model.eval()

    def _choose_operations(self, state: BackendState, player: int):
        obs = self.feature_extractor.encode_observation(state, player, np.zeros(self.max_actions))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats_t = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            output = self.model(board, stats_t)
        ops = decode_network_output({k: v for k, v in output.items()}, state, player)
        return ops

    def choose_bundle(self, state: BackendState, player: int, bundles=None):
        ops = self._choose_operations(state, player)
        if not ops:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
        # Pack operations into a bundle (use first bundle that matches)
        if bundles:
            for b in bundles:
                if list(b.operations) == ops:
                    return b
        # Fallback: create a pseudo-bundle
        score = sum(1 for _ in ops)
        return ActionBundle(name="neural", operations=tuple(ops), score=score)


class AI(NeuralAgent):
    pass
