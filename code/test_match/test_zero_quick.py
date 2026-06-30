"""Quick single-process test of zero model vs ExampleAI."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Ant-Game"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_num_threads(1)
import numpy as np

from my_ai.network import create_zero_model, create_model
from my_ai.agent import NeuralAgent
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from AI.ai_example import AI as ExampleAI

def test_one(label, model_fn, seed):
    m = model_fn()
    a = NeuralAgent(model=m)
    o = ExampleAI(seed=seed)
    our_p = seed % 2
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        u = a._choose_operations(state, our_p)
        v = o.choose_operations(state, 1 - our_p)
        if our_p == 0:
            state.resolve_turn(u, v)
        else:
            state.resolve_turn(v, u)
    hp = state.bases[our_p].hp
    hp_o = state.bases[1 - our_p].hp
    s = 1.0 if hp > hp_o else (0.0 if hp_o > hp else 0.5)
    print(f"  {label}: score={s:.1f}  hp={hp}/{hp_o}")
    return s

print("\nZero model vs ExampleAI (5 games):")
scores = [test_one(f"seed={s}", create_zero_model, s) for s in range(5)]
print(f"  Win rate: {sum(1 for s in scores if s==1.0)}/{len(scores)} = {np.mean(scores):.3f}")

print("\nDefault init model vs ExampleAI (5 games):")
scores2 = [test_one(f"seed={s}", create_model, s) for s in range(5)]
print(f"  Win rate: {sum(1 for s in scores2 if s==1.0)}/{len(scores2)} = {np.mean(scores2):.3f}")
