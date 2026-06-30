"""Diagnose a model's behavior: what actions does it take vs ExampleAI?"""
from __future__ import annotations
import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND, TowerType, OperationType, SUPER_WEAPON_STATS
from AI.ai_example import AI as ExampleAI
from my_ai.network import create_model
from my_ai.agent import NeuralAgent

NUM_CLASSES = 24

OP_NAME_MAP = {
    OperationType.BUILD_TOWER: "BUILD", OperationType.UPGRADE_TOWER: "UPGRADE",
    OperationType.DOWNGRADE_TOWER: "DOWNGRADE",
    OperationType.USE_LIGHTNING_STORM: "LIGHTNING", OperationType.USE_EMP_BLASTER: "EMP",
    OperationType.USE_DEFLECTOR: "DEFLECTOR", OperationType.USE_EMERGENCY_EVASION: "EVASION",
    OperationType.UPGRADE_GENERATION_SPEED: "UP_SPEED", OperationType.UPGRADE_GENERATED_ANT: "UP_ANT_HP",
}


def op_desc(op) -> str:
    name = OP_NAME_MAP.get(op.op_type, f"OP_{op.op_type}")
    if op.op_type in (OperationType.BUILD_TOWER,):
        return f"{name}({op.arg0},{op.arg1})"
    elif op.op_type == OperationType.UPGRADE_TOWER:
        return f"{name}(id={op.arg0}->type={op.arg1})"
    elif op.op_type == OperationType.DOWNGRADE_TOWER:
        return f"{name}(id={op.arg0})"
    elif OperationType.USE_LIGHTNING_STORM <= op.op_type <= OperationType.USE_EMERGENCY_EVASION:
        return f"{name}({op.arg0},{op.arg1})"
    else:
        return name
CLASS_NAMES = {
    0: "build_Basic", 1: "toward_Heavy", 2: "toward_Heavy+", 3: "toward_Ice",
    4: "toward_Bewitch", 5: "toward_Quick", 6: "toward_Quick+", 7: "toward_Double",
    8: "toward_Sniper", 9: "toward_Mortar", 10: "toward_Mortar+", 11: "toward_Pulse",
    12: "toward_Missile", 13: "toward_Producer+", 14: "toward_Siege", 15: "toward_Medic",
    16: "downgrade", 17: "lightning", 18: "emp", 19: "deflector", 20: "evasion",
    21: "upgrade_speed", 22: "upgrade_ant_hp", 23: "HOLD",
}


def diagnose(ckpt_path: str, n_games: int = 10, seed_offset: int = 0, verbose: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    param_vec = ckpt["top2_params"][0].numpy() if "top2_params" in ckpt else ckpt["mean"].numpy()

    model = create_model(single_head=True)
    if model.count_parameters() != len(param_vec):
        model = create_model(single_head=False)
    model.set_parameters_from_vector(param_vec)
    agent = NeuralAgent(model=model)

    # Aggregated stats
    total_turns = 0
    total_ops = 0  # total operations executed
    total_hold = 0  # turns where model took no actions
    op_type_counts: dict[int, int] = {}
    game_results = []

    for g in range(n_games):
        seed = seed_offset + g
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        opp = ExampleAI(seed=seed)
        our_player = seed % 2
        opp_player = 1 - our_player
        side = "1st" if our_player == 0 else "2nd"

        turns_played = 0
        ops_this_game = 0
        hold_this_game = 0

        print(f"\n--- Game {g+1}: seed={seed}, we are {side} ---")

        for _ in range(MAX_ROUND):
            if state.terminal:
                break

            ops = agent._choose_operations(state, our_player)

            if len(ops) == 0:
                hold_this_game += 1
                total_hold += 1
                if verbose:
                     print(f"  turn={turns_played:3d}  HOLD")
            else:
                ops_this_game += len(ops)
                total_ops += len(ops)
                for op in ops:
                    op_type_counts[op.op_type] = op_type_counts.get(op.op_type, 0) + 1
                if verbose:
                     descs = [op_desc(op) for op in ops]
                     print(f"  turn={turns_played:3d}  {' | '.join(descs)}")

            opp_ops = opp.choose_operations(state, opp_player)

            if our_player == 0:
                state.resolve_turn(ops, opp_ops)
            else:
                state.resolve_turn(opp_ops, ops)

            turns_played += 1

        hp_us = state.bases[our_player].hp
        hp_opp = state.bases[opp_player].hp
        if hp_us > hp_opp:
            result = "WIN"
        elif hp_opp > hp_us:
            result = "LOSS"
        else:
            result = "DRAW"

        game_results.append({"seed": seed, "result": result, "hp_us": int(hp_us), "hp_opp": int(hp_opp),
                             "turns": turns_played, "ops": ops_this_game, "holds": hold_this_game})
        total_turns += turns_played

        print(f"  Game {g+1}/{n_games}: seed={seed}  {result}  us={int(hp_us):2d}  opp={int(hp_opp):2d}  "
              f"turns={turns_played:3d}  ops={ops_this_game}  holds={hold_this_game}")

    # ─── Summary ─────────────────────────────────────────────────────
    wins = sum(1 for r in game_results if r["result"] == "WIN")
    losses = sum(1 for r in game_results if r["result"] == "LOSS")

    print(f"\n{'='*60}")
    print(f"DIAGNOSE: {ckpt_path}")
    print(f"Games: {n_games}")
    print(f"Win rate: {wins}/{n_games} ({100*wins/n_games:.1f}%)")
    print(f"Avg turns per game: {total_turns/n_games:.1f}")
    print(f"Avg ops per turn:   {total_ops/total_turns:.3f}")
    print(f"Hold rate:          {total_hold}/{total_turns} ({100*total_hold/total_turns:.1f}% of turns)")
    print()

    # Op type breakdown
    print(f"Operation type distribution (total {total_ops} operations):")
    op_name_map = {
        OperationType.BUILD_TOWER: "BUILD", OperationType.UPGRADE_TOWER: "UPGRADE",
        OperationType.DOWNGRADE_TOWER: "DOWNGRADE",
        OperationType.USE_LIGHTNING_STORM: "LIGHTNING", OperationType.USE_EMP_BLASTER: "EMP",
        OperationType.USE_DEFLECTOR: "DEFLECTOR", OperationType.USE_EMERGENCY_EVASION: "EVASION",
        OperationType.UPGRADE_GENERATION_SPEED: "UP_SPEED", OperationType.UPGRADE_GENERATED_ANT: "UP_ANT_HP",
    }
    for opt, cnt in sorted(op_type_counts.items(), key=lambda x: -x[1]):
        name = op_name_map.get(opt, f"OP_{opt}")
        print(f"  {name:15s}: {cnt:4d} ({100*cnt/total_ops:5.1f}%)")
    print()

    # Game-by-game
    print("Per-game detail:")
    for r in game_results:
        print(f"  seed={r['seed']:3d}  {r['result']:4s}  HP {r['hp_us']:2d}/{r['hp_opp']:2d}  "
              f"turns={r['turns']:3d}  ops={r['ops']:2d}  holds={r['holds']:2d}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true", help="print per-turn actions")
    args = parser.parse_args()
    diagnose(args.ckpt, args.games, verbose=args.verbose)
