"""Debug: print operations from NeuralAgent vs ExampleAI for first N rounds."""
from __future__ import annotations
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Ant-Game"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch; torch.set_num_threads(1)
import numpy as np
from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from AI.ai_example import AI as ExampleAI


def _op_name(op) -> str:
    """Return human-readable name for an operation."""
    from SDK.utils.constants import OperationType, TowerType, SuperWeaponType
    t = OperationType(op.op_type).name if hasattr(op, 'op_type') else str(op.op_type)
    if op.arg0 is not None and op.arg1 is not None:
        return f"{t}(x={op.arg0},y={op.arg1})"
    elif op.arg0 is not None:
        return f"{t}(id={op.arg0})"
    return t


def debug_game(ckpt_path: str, seed: int = 42, max_rounds: int = 30):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = create_model()
    model.set_parameters_from_vector(ckpt["mean"].numpy())
    agent = NeuralAgent(model=model)
    opp = ExampleAI(seed=seed)

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    print(f"\nCheckpoint: {ckpt_path}  seed={seed}")
    print(f"{'rnd':>4}  {'Player0 (Ours)':<40}  {'Player1 (Example)':<40}  hp0  hp1  coins0  coins1")
    print("-" * 110)

    for rnd in range(min(max_rounds, MAX_ROUND)):
        if state.terminal:
            print(f"  Game ended at round {rnd}")
            break

        ops0 = agent._choose_operations(state, 0)
        ops1 = opp.choose_operations(state, 1)
        result = state.resolve_turn(ops0, ops1)

        op0_str = ", ".join(_op_name(op) for op in ops0) if ops0 else "noop"
        op1_str = ", ".join(_op_name(op) for op in ops1) if ops1 else "noop"
        il0 = len(result.illegal[0])
        il1 = len(result.illegal[1])
        if il0:
            op0_str += f" [+{il0} illegal]"
        if il1:
            op1_str += f" [+{il1} illegal]"

        print(f"{rnd:4d}  {op0_str:<40}  {op1_str:<40}  "
              f"{state.bases[0].hp:3d}  {state.bases[1].hp:3d}  "
              f"{state.coins[0]:5d}  {state.coins[1]:5d}")

    hp0, hp1 = state.bases[0].hp, state.bases[1].hp
    print(f"\nResult: {'Win' if hp0>hp1 else ('Lose' if hp1>hp0 else 'Draw')}  ({hp0} vs {hp1})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    debug_game(args.ckpt, args.seed, args.rounds)
