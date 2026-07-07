"""Quick diagnose what classes a distilled model predicts during gameplay."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent


def diagnose(ckpt_path: str, small: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "mean" in ckpt:
        params = ckpt["mean"].numpy()
    elif "top2_params" in ckpt and len(ckpt["top2_params"]) > 0:
        params = ckpt["top2_params"][0]
        params = params.numpy() if isinstance(params, torch.Tensor) else params
    else:
        raise ValueError("No params found")

    if "num_heads" in ckpt:
        num_heads = ckpt["num_heads"]
    elif "config" in ckpt:
        num_heads = ckpt["config"].get("num_heads", 3)
    else:
        num_heads = 1 if small else 3

    model = create_model(small=small, num_heads=num_heads)
    model.set_parameters_from_vector(params)
    model.to(device)
    model.eval()
    print(f"Model: {model.count_parameters():,} params, {num_heads} heads")

    agent = NeuralAgent(model=model)

    all_classes = [[] for _ in range(num_heads)]
    all_logit_range = [[] for _ in range(num_heads)]

    for seed in range(3):  # 3 games
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        for turn in range(MAX_ROUND):
            if state.terminal:
                break
            # Get model output
            obs = agent.feature_extractor.encode_observation(state, 0, np.zeros(agent.max_actions))
            board = torch.from_numpy(obs["board"]).unsqueeze(0).float().to(device)
            st = torch.from_numpy(obs["stats"]).unsqueeze(0).float().to(device)
            with torch.no_grad():
                output = model(board, st)

            u = agent._choose_operations(state, 0)
            opp = agent._choose_operations(state, 1)
            state.resolve_turn(u, opp)

            for h in range(num_heads):
                key = f"head{h+1}_logits"
                logits = output[key].detach().cpu().numpy().squeeze()
                cls = int(np.argmax(logits))
                all_classes[h].append(cls)
                all_logit_range[h].append(logits.max() - logits.min())

    print(f"\n{'Head':>6}  Top-5 classes (count)                  Logit range")
    print("-" * 65)
    for h in range(num_heads):
        cls = all_classes[h]
        top5 = sorted(set(cls), key=lambda c: cls.count(c), reverse=True)[:5]
        top5_str = ", ".join(f"{c}({cls.count(c)})" for c in top5)
        logit_rng = np.mean(all_logit_range[h])
        print(f"  H{h+1}:  {top5_str:<40s}  {logit_rng:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    diagnose(args.ckpt, args.small)
