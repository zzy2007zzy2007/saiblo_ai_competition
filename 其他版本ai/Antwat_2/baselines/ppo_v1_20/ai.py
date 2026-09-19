from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.serialization
from easydict import EasyDict

from SDK.backend.model import Operation
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle

from common import BaseAgent

from ppo_antwar.network.antwar_net import AntWarPolicyValueNetwork
from ppo_antwar.env.observation import ObservationEncoder
from ppo_antwar.env.action_mask import ActionMaskHandler


class AI(BaseAgent):
    def __init__(self, seed: int | None = None, max_actions: int = 96) -> None:
        super().__init__(seed=seed, max_actions=max_actions)

        model_dir = Path(__file__).resolve().parent / "models"
        model_paths = sorted(model_dir.glob("*.pt"))
        if not model_paths:
            raise FileNotFoundError(
                f"No model files found in {model_dir}. "
                f"Place a PPO checkpoint (.pt) in baselines/ppo_v1_20/models/"
            )
        model_path = str(model_paths[-1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            import warnings
            warnings.warn(f"PPO agent running on CPU (no CUDA available); model: {model_path}")

        keep_fn = None
        if hasattr(torch.serialization, 'safe_globals'):
            keep_fn = torch.serialization.safe_globals([EasyDict])
            keep_fn.__enter__()
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        finally:
            if keep_fn is not None:
                keep_fn.__exit__(None, None, None)

        config = checkpoint.get('config', None)
        if config is None:
            raise ValueError(f"Model checkpoint does not contain 'config' key: {model_path}")

        self.policy = AntWarPolicyValueNetwork(
            board_shape=tuple(config.network.board_shape),
            global_dim=config.network.global_dim,
            action_dim=config.network.action_dim,
            hidden_dim=config.network.hidden_dim,
        ).to(device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        self.device = device

        self.observation_encoder = ObservationEncoder()
        self.action_mask_handler = ActionMaskHandler()

    def choose_operations(
        self,
        state: BackendState,
        player: int,
        bundles: Any | None = None,
    ) -> list[Operation]:
        obs = self.observation_encoder.encode(state, player)
        obs['action_mask'] = self.action_mask_handler.get_action_mask(state, player)

        board = torch.from_numpy(obs['board']).float().unsqueeze(0).to(self.device)
        global_obs = torch.from_numpy(obs['global']).float().unsqueeze(0).to(self.device)
        mask = torch.from_numpy(obs['action_mask']).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_id, _, _ = self.policy.get_action(
                board, global_obs, mask, deterministic=True
            )

        op = self.action_mask_handler.action_id_to_operation(
            int(action_id.item()), state, player
        )
        return [op] if op is not None else []

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        ops = self.choose_operations(state, player, bundles=bundles)
        return ActionBundle(operations=ops)
