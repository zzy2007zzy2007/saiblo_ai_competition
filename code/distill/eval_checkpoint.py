"""Evaluate a distilled checkpoint vs ExampleAI.

Usage:
    python code/distill/eval_checkpoint.py distill_data_v2/model_large_bal.pt --games 10
    python code/distill/eval_checkpoint.py distill_data_v2/model_all_bal300.pt --games 10 --small
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import numpy as np
from SDK.backend.engine import GameState
from AI.ai_example import AI as ExampleAI
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent


def evaluate(ckpt_path: str, num_games: int = 10, small: bool = False, seed_offset: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "mean" in ckpt:
        params = ckpt["mean"].numpy()
    elif "top2_params" in ckpt and len(ckpt["top2_params"]) > 0:
        params = ckpt["top2_params"][0]
        params = params.numpy() if isinstance(params, torch.Tensor) else params
    else:
        raise ValueError("No params found in checkpoint")

    # Determine num_heads from checkpoint
    if "num_heads" in ckpt:
        num_heads = ckpt["num_heads"]
    elif "config" in ckpt:
        num_heads = ckpt["config"].get("num_heads", 3)
    else:
        # Count heads from model_state keys
        import re
        hk = [k for k in ckpt.get("model_state", ckpt)
              if re.match(r"policy_heads\.\d+\.weight", k)]
        num_heads = max(len(hk), 1) if hk else (1 if small else 3)

    model = create_model(small=small, num_heads=num_heads)
    model.set_parameters_from_vector(params)
    model.to(device)
    model.eval()
    print(f"Model: {model.count_parameters():,} params, "
          f"{model.num_heads} heads, small={small}")

    agent = NeuralAgent(model=model)
    opponent = ExampleAI()
    wins = 0
    total_turns = 0

    for g in range(num_games):
        seed = g + seed_offset
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

        for turn in range(MAX_ROUND):
            if state.terminal:
                break
            u = agent._choose_operations(state, 0)
            opp = opponent.choose_operations(state, 1)
            state.resolve_turn(u, opp)

        result = "WIN" if state.bases[0].hp > state.bases[1].hp else "LOSS"
        if state.bases[0].hp > state.bases[1].hp:
            wins += 1
        total_turns += turn + 1
        print(f"  Game {g+1}: {result} ({state.bases[0].hp} vs {state.bases[1].hp}) "
              f"in {turn+1} turns")

    avg_turns = total_turns / num_games
    wr = wins / num_games * 100
    print(f"\nResults: {wins}/{num_games} wins ({wr:.0f}%) over {avg_turns:.0f} avg turns")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--small", action="store_true",
                        help="use small model architecture")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.ckpt, args.games, args.small, args.seed_offset)
