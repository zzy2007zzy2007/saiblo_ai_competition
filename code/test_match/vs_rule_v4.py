"""Compare neural model vs rule_v4 AI."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
_RULE = Path(__file__).resolve().parents[2] / "其他版本ai" / "rule_v4"
for p in (_REPO, _CODE, _RULE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import numpy as np
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from ai import AI as RuleV4


def main():
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    params = ckpt["top2_params"][0].numpy()
    num_heads = ckpt.get("num_heads", 3)
    model = create_model(num_heads=num_heads)
    model.set_parameters_from_vector(params)
    neural = NeuralAgent(model=model)
    rule = RuleV4()

    wins = 0
    for seed in range(10):
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        for _ in range(MAX_ROUND):
            if state.terminal:
                break
            u = neural._choose_operations(state, 0)
            v = rule.choose_operations(state, 1)
            state.resolve_turn(u or [], v or [])
        h0, h1 = state.bases[0].hp, state.bases[1].hp
        if h0 > h1:
            wins += 1
        print(f"  seed={seed:2d}  {'WIN' if h0 > h1 else 'LOSS':4s}  HP {h0:2d} vs {h1:2d}")
    print(f"\nResult: {wins}/10 ({wins*10}%)")


if __name__ == "__main__":
    main()
