"""Sequential evaluation script — runs games one by one, NO multiprocessing.

This script avoids the Windows page-file crash caused by spawning many
Python subprocesses that each need to import torch CUDA DLLs.

Usage:
    # Single model baseline
    python code/test_match/run_eval_seq.py ^
        training_history/ss_20260709_191050/gen_0030.pt ^
        --games 30

    # MCTS mode
    python code/test_match/run_eval_seq.py ^
        training_history/ss_20260709_191050/gen_0030.pt ^
        --mcts --model-b training_history/ss_20260710_155327/gen_0101.pt ^
        --value-net value_net.pt --small-value --games 30
"""

from __future__ import annotations
import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

# ── Operation name table ────────────────────────────────────────────────

_OP_NAMES = {
    11: "BUILD_TOWER", 12: "UPGRADE_TOWER", 13: "DOWNGRADE_TOWER",
    21: "LIGHTNING", 22: "EMP_BLASTER", 23: "DEFLECTOR", 24: "EVASION",
    31: "UP_SPEED", 32: "UP_ANT",
}


def _summarize_ops(ops: list) -> str:
    """Turn a list of Operation objects into a compact readable string."""
    if not ops:
        return "(none)"
    parts = []
    for op in ops:
        name = _OP_NAMES.get(op.op_type, f"OP{op.op_type}")
        if op.arg0 != -1 and op.arg1 != -1:
            parts.append(f"{name}({op.arg0},{op.arg1})")
        elif op.arg0 != -1:
            parts.append(f"{name}({op.arg0})")
        else:
            parts.append(name)
    return ", ".join(parts)

# ── Helpers ────────────────────────────────────────────────────────────

def _result_str(score: float) -> str:
    return {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[score]


def _finish_game(state, our_player, opp_player, seed) -> dict:
    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    side = "1st" if our_player == 0 else "2nd"
    if hp_us <= 0 and hp_opp <= 0:
        score = 0.5
    elif hp_us > hp_opp:
        score = 1.0
    else:
        score = 0.0 if hp_opp > hp_us else 0.5
    tag = _result_str(score)
    print(f"  [{side}] seed={seed:3d}  {tag}  us={int(hp_us):2d}  opp={int(hp_opp):2d}", flush=True)
    return {"score": score, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}


# ── Single-model ───────────────────────────────────────────────────────

def run_single_eval(ckpt_path, games, opponent="example", gpu=False):
    print(f"\nSingle-model evaluation: {Path(ckpt_path).name}")
    print(f"  Opponent: {opponent}, {games} games  {'[GPU]' if gpu else '[CPU]'}")
    print("-" * 60)

    import os
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "top2_params" in ckpt:
        param_vec = ckpt["top2_params"][0].numpy()
    elif "mean" in ckpt:
        param_vec = ckpt["mean"].numpy()
    elif "model_state" in ckpt:
        num_heads = ckpt.get("num_heads", 3)
        small = (ckpt["mean"].shape[0] < 200000) if "mean" in ckpt else False
        model = create_model(num_heads=num_heads, small=small)
        model.load_state_dict(ckpt["model_state"])
        param_vec = model.get_parameters_as_vector()
    else:
        raise KeyError(f"Unknown checkpoint format: {list(ckpt.keys())}")

    small = (param_vec.shape[0] < 200000)
    num_heads = ckpt.get("num_heads", 3 if not small else 1)
    model = create_model(num_heads=num_heads, small=small)
    model.set_parameters_from_vector(param_vec)
    agent = NeuralAgent(model=model)

    # Opponent
    from AI.ai_example import AI as ExampleAI
    opp = ExampleAI()

    from SDK.backend.state import PythonBackendState
    from SDK.utils.constants import MAX_ROUND

    results = []
    t0 = time.perf_counter()
    for i in range(games):
        seed = i
        our_player = seed % 2
        opp_player = 1 - our_player

        state = PythonBackendState.initial(seed=seed, cold_handle_rule_illegal=True)
        for _ in range(MAX_ROUND):
            if state.terminal:
                break
            ops_us = agent._choose_operations(state, our_player)
            ops_opp = opp.choose_operations(state, opp_player)
            if our_player == 0:
                state.resolve_turn(ops_us, ops_opp)
            else:
                state.resolve_turn(ops_opp, ops_us)

        result = _finish_game(state, our_player, opp_player, seed)
        results.append(result)

    dt = time.perf_counter() - t0
    _print_summary(results, games, dt, Path(ckpt_path).stem)


# ── MCTS ───────────────────────────────────────────────────────────────

def run_mcts_eval(model_a_path, model_b_path, value_net_path, games,
                  depth=3, small_value=False, opponent="example", gpu=False):
    print(f"\nMCTS evaluation")
    print(f"  Models:  {Path(model_a_path).stem} + {Path(model_b_path).stem}")
    print(f"  Value:   {Path(value_net_path).name}")
    print(f"  Depth:   {depth}")
    print(f"  Games:   {games}")
    print(f"  Device:  {'GPU' if gpu else 'CPU'}")
    print("-" * 60)

    import os
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from my_ai.mcts_agent import create_mcts_agent

    device = "cuda" if gpu else "cpu"
    agent = create_mcts_agent(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        value_net_path=value_net_path,
        depth=depth,
        small=small_value,
        device=device,
    )

    # ── Load opponent ────────────────────────────────────────────────────
    if opponent == "rule_v4":
        _rv4_root = Path(__file__).resolve().parents[2] / "其他版本ai" / "rule_v4"
        import sys as _sys
        if str(_rv4_root) not in _sys.path:
            _sys.path.insert(0, str(_rv4_root))
        from ai import AI as RuleV4AI
        opp = RuleV4AI(seed=0)
    else:
        from AI.ai_example import AI as ExampleAI
        opp = ExampleAI()

    opp_label = opponent

    from SDK.backend.state import PythonBackendState
    from SDK.utils.constants import MAX_ROUND

    results = []
    t0 = time.perf_counter()
    for i in range(games):
        seed = i
        our_player = seed % 2
        opp_player = 1 - our_player
        side = "1st" if our_player == 0 else "2nd"
        print(f"\n--- Game {i+1}/{games} [seed={seed}, {side}) ---", flush=True)

        state = PythonBackendState.initial(seed=seed, cold_handle_rule_illegal=True)
        for turn in range(1, MAX_ROUND + 1):
            if state.terminal:
                break

            ops_us = agent.choose_operations(state, our_player)
            expert_idx = agent.last_expert_idx
            expert_label = "A" if expert_idx == 0 else "B"

            ops_opp = opp.choose_operations(state, opp_player)

            hp_us_before = state.bases[our_player].hp
            hp_opp_before = state.bases[opp_player].hp
            if our_player == 0:
                state.resolve_turn(ops_us, ops_opp)
            else:
                state.resolve_turn(ops_opp, ops_us)
            hp_us = state.bases[our_player].hp
            hp_opp = state.bases[opp_player].hp

            print(f"  [T{turn:3d}] HP:{hp_us_before:2d},{hp_opp_before:2d}→{hp_us:2d},{hp_opp:2d}"
                  f"  Expert:{expert_label}"
                  f"  us={_summarize_ops(ops_us)}"
                  f"  opp={_summarize_ops(ops_opp)}", flush=True)

        result = _finish_game(state, our_player, opp_player, seed)
        results.append(result)
        elapsed = time.perf_counter() - t0
        wins_so_far = sum(1 for r in results if r["score"] == 1.0)
        print(f"  => Game {i+1} done  [{wins_so_far}/{i+1} wins]  ({elapsed:.0f}s elapsed)", flush=True)

    dt = time.perf_counter() - t0
    label = f"MCTS({Path(model_a_path).stem}+{Path(model_b_path).stem})"
    _print_summary(results, games, dt, label)


# ── Summary ────────────────────────────────────────────────────────────

def _print_summary(results, total_games, dt, label):
    scores = [r["score"] for r in results]
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = sum(1 for s in scores if s == 0.0)
    hp_us_all = [r["hp_us"] for r in results]
    hp_opp_all = [r["hp_opp"] for r in results]

    print(f"\n{'='*60}")
    print(f"Results: {label}")
    print(f"  Time:    {dt:.0f}s ({dt/total_games:.0f}s/game)")
    print(f"  Win:     {wins}/{total_games} ({wins/total_games:.3f})")
    print(f"  Draw:    {draws}/{total_games} ({draws/total_games:.3f})")
    print(f"  Loss:    {losses}/{total_games} ({losses/total_games:.3f})")
    print(f"  Avg HP:  {np.mean(hp_us_all):.1f} / {np.mean(hp_opp_all):.1f}")
    print(f"  Median:  {np.median(hp_us_all):.0f} / {np.median(hp_opp_all):.0f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="checkpoint path (model_a for MCTS)")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--mcts", action="store_true")
    parser.add_argument("--model-b", type=str, default=None)
    parser.add_argument("--value-net", type=str, default=None)
    parser.add_argument("--small-value", action="store_true")
    parser.add_argument("--mcts-depth", type=int, default=3)
    parser.add_argument("--opponent", type=str, default="example",
                        choices=["example", "rule_v4"],
                        help="opponent AI (default: example)")
    parser.add_argument("--gpu", action="store_true", help="use GPU for inference (MCTS mode)")
    args = parser.parse_args()

    if args.mcts:
        if not args.model_b or not args.value_net:
            parser.error("--mcts requires --model-b and --value-net")
        run_mcts_eval(
            model_a_path=str(Path(args.ckpt).resolve()),
            model_b_path=str(Path(args.model_b).resolve()),
            value_net_path=str(Path(args.value_net).resolve()),
            games=args.games,
            depth=args.mcts_depth,
            small_value=args.small_value,
            opponent=args.opponent,
            gpu=args.gpu,
        )
    else:
        run_single_eval(
            ckpt_path=str(Path(args.ckpt).resolve()),
            games=args.games,
            gpu=args.gpu,
        )
