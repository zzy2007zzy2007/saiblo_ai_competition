"""Diagnose a model's behavior: what actions does it take vs ExampleAI?"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys, time
from pathlib import Path

import torch
torch.set_num_threads(1)

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
from my_ai.decoder import decode_network_output
from utils.logger import get_logger

NUM_CLASSES = 24

OP_NAME_MAP = {
    OperationType.BUILD_TOWER: "BUILD", OperationType.UPGRADE_TOWER: "UPGRADE",
    OperationType.DOWNGRADE_TOWER: "DOWNGRADE",
    OperationType.USE_LIGHTNING_STORM: "LIGHTNING", OperationType.USE_EMP_BLASTER: "EMP",
    OperationType.USE_DEFLECTOR: "DEFLECTOR", OperationType.USE_EMERGENCY_EVASION: "EVASION",
    OperationType.UPGRADE_GENERATION_SPEED: "UP_SPEED", OperationType.UPGRADE_GENERATED_ANT: "UP_ANT_HP",
}


def op_desc(op) -> str:
    """动作表示与自对弈进度文件一致：元组 (op_type, arg0, arg1)，如 (11,5,9)。"""
    return f"({int(op.op_type)},{op.arg0},{op.arg1})"
CLASS_NAMES = {
    0: "build_Basic", 1: "toward_Heavy", 2: "toward_Heavy+", 3: "toward_Ice",
    4: "toward_Bewitch", 5: "toward_Quick", 6: "toward_Quick+", 7: "toward_Double",
    8: "toward_Sniper", 9: "toward_Mortar", 10: "toward_Mortar+", 11: "toward_Pulse",
    12: "toward_Missile", 13: "toward_Producer+", 14: "toward_Siege", 15: "toward_Medic",
    16: "downgrade", 17: "lightning", 18: "emp", 19: "deflector", 20: "evasion",
    21: "upgrade_speed", 22: "upgrade_ant_hp", 23: "HOLD",
}


def diagnose(ckpt_path: str, n_games: int = 10, seed_offset: int = 0, verbose: bool = False, num_heads: int = 3, log=None, action_dropout: float = 0.0, ind: int | None = None):
    _print = print
    _empty = lambda: _print()
    if log is not None:
        _print = lambda *a, **kw: log.print(*a, timestamp=False, **kw)
        _empty = lambda: log.print(timestamp=False)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ind is not None and "ga_pop" in ckpt and ind < len(ckpt["ga_pop"]):
        param_vec = ckpt["ga_pop"][ind]
        label = f"ga_pop[{ind}]"
    elif "top2_params" in ckpt:
        param_vec = ckpt["top2_params"][0].numpy()
        label = "TOP1"
    elif "mean" in ckpt:
        raw = ckpt["mean"]
        param_vec = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
        label = "TOP1"
    else:
        param_vec = None  # az_intent/AZ checkpoint format: model_state only
        label = "model_state"

    model = create_model(num_heads=num_heads, no_bn=ckpt.get("no_bn", False))
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])  # BN stats + params
    if param_vec is not None:
        model.set_parameters_from_vector(param_vec)  # overwrite trainable params
    agent = NeuralAgent(model=model, action_dropout=action_dropout)
    _print(f"num_heads={num_heads}, params={len(param_vec):,}  [{label}]" if param_vec is not None
           else f"num_heads={num_heads}  [{label}]")

    # Aggregated stats
    total_turns = 0
    total_ops = 0  # total operations executed
    total_hold = 0  # turns where model took no actions
    op_type_counts: dict[int, int] = {}          # executed (after dropout override)
    model_op_counts: dict[int, int] = {}          # model's true intent (before dropout)
    dropout_op_counts: dict[int, int] = {}        # forced by dropout only
    model_hold_turns = 0                          # turns where model wanted to do nothing
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

        _print(f"\n--- Game {g+1}: seed={seed}, we are {side} ---")

        for _ in range(MAX_ROUND):
            if state.terminal:
                break

            ops = agent._choose_operations(state, our_player)

            # Decode model's true intent (before dropout override)
            is_dropout = action_dropout > 0 and agent.dropout_this_turn
            if agent.last_output is not None and is_dropout:
                model_ops = decode_network_output(agent.last_output, state, our_player)
            else:
                model_ops = ops  # no dropout, executed = intended

            # Count model's intended ops
            if len(model_ops) == 0:
                model_hold_turns += 1
            else:
                for op in model_ops:
                    model_op_counts[op.op_type] = model_op_counts.get(op.op_type, 0) + 1

            # Count dropout-forced ops separately
            if is_dropout:
                for op in ops:
                    dropout_op_counts[op.op_type] = dropout_op_counts.get(op.op_type, 0) + 1

            if len(ops) == 0:
                hold_this_game += 1
                total_hold += 1
                if verbose:
                     tag = " [dropout→HOLD]" if is_dropout else ""
                     _print(f"  turn={turns_played:3d}  HOLD{tag}")
            else:
                ops_this_game += len(ops)
                total_ops += len(ops)
                for op in ops:
                    op_type_counts[op.op_type] = op_type_counts.get(op.op_type, 0) + 1
                if verbose:
                     tag = " [dropout]" if is_dropout else ""
                     descs = [op_desc(op) for op in ops]
                     _print(f"  turn={turns_played:3d}  {' | '.join(descs)}{tag}")

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

        _print(f"  Game {g+1}/{n_games}: seed={seed}  {result}  us={int(hp_us):2d}  opp={int(hp_opp):2d}  "
              f"turns={turns_played:3d}  ops={ops_this_game}  holds={hold_this_game}")

    # ─── Summary ─────────────────────────────────────────────────────
    wins = sum(1 for r in game_results if r["result"] == "WIN")
    losses = sum(1 for r in game_results if r["result"] == "LOSS")

    if action_dropout > 0:
        _print(f"Action dropout: {action_dropout:.2f}")

    _print(f"\n{'='*60}")
    _print(f"DIAGNOSE: {ckpt_path}")
    _print(f"Games: {n_games}")
    _print(f"Win rate: {wins}/{n_games} ({100*wins/n_games:.1f}%)")
    _print(f"Avg turns per game: {total_turns/n_games:.1f}")
    _print(f"Avg ops per turn:   {total_ops/total_turns:.3f}")
    _print(f"Hold rate:          {total_hold}/{total_turns} ({100*total_hold/total_turns:.1f}% of turns)")
    _empty()

    # Op type breakdown
    _print(f"Operation type distribution (total {total_ops} operations):")
    op_name_map = {
        OperationType.BUILD_TOWER: "BUILD", OperationType.UPGRADE_TOWER: "UPGRADE",
        OperationType.DOWNGRADE_TOWER: "DOWNGRADE",
        OperationType.USE_LIGHTNING_STORM: "LIGHTNING", OperationType.USE_EMP_BLASTER: "EMP",
        OperationType.USE_DEFLECTOR: "DEFLECTOR", OperationType.USE_EMERGENCY_EVASION: "EVASION",
        OperationType.UPGRADE_GENERATION_SPEED: "UP_SPEED", OperationType.UPGRADE_GENERATED_ANT: "UP_ANT_HP",
    }
    for opt, cnt in sorted(op_type_counts.items(), key=lambda x: -x[1]):
        name = op_name_map.get(opt, f"OP_{opt}")
        _print(f"  {name:15s}: {cnt:4d} ({100*cnt/total_ops:5.1f}%)")
    _empty()

    if action_dropout > 0 and total_turns > 0:
        # Model's true intent breakdown
        total_model_ops = sum(model_op_counts.values())
        _print("Model-intended actions (before dropout):")
        for opt, cnt in sorted(model_op_counts.items(), key=lambda x: -x[1]):
            name = op_name_map.get(opt, f"OP_{opt}")
            _print(f"  {name:15s}: {cnt:4d}")
        if model_hold_turns > 0:
            _print(f"  {'HOLD':15s}: {model_hold_turns:4d}")
        _empty()

        # Dropout-forced breakdown
        total_dropout_ops = sum(dropout_op_counts.values())
        _print(f"Dropout-forced actions (total {total_dropout_ops}):")
        for opt, cnt in sorted(dropout_op_counts.items(), key=lambda x: -x[1]):
            name = op_name_map.get(opt, f"OP_{opt}")
            _print(f"  {name:15s}: {cnt:4d}")
        _empty()

    # Game-by-game
    _print("Per-game detail:")
    for r in game_results:
        _print(f"  seed={r['seed']:3d}  {r['result']:4s}  HP {r['hp_us']:2d}/{r['hp_opp']:2d}  "
              f"turns={r['turns']:3d}  ops={r['ops']:2d}  holds={r['holds']:2d}")


if __name__ == "__main__":
    import argparse, torch, re
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=3, help="number of policy heads")
    parser.add_argument("--verbose", "-v", action="store_true", help="print per-turn actions")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="log directory (dual output to terminal + file)")
    args = parser.parse_args()

    if args.num_heads == 3:
        # Auto-detect num_heads from checkpoint
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if "num_heads" in ckpt:
            args.num_heads = ckpt["num_heads"]
        elif "config" in ckpt and "num_heads" in ckpt["config"]:
            args.num_heads = ckpt["config"]["num_heads"]
        elif "model_state" in ckpt:
            heads = [k for k in ckpt["model_state"] if re.match(r"policy_heads\.\d+\.weight", k)]
            args.num_heads = max(len(heads), 1)
        elif "policy_head2.weight" in ckpt.get("model_state", {}):
            args.num_heads = 3

    log = None
    if args.log_dir:
        log = get_logger(Path(args.log_dir) / "diagnose_model.log", mode="a")
    diagnose(args.ckpt, args.games, verbose=args.verbose, num_heads=args.num_heads, log=log)
