"""Evaluate a trained checkpoint (single model or MCTS) against various opponents.

Modes:
  Single-model: evaluate one checkpoint vs opponent (original behavior).
  MCTS:         evaluate dual-expert + value network MCTS agent vs opponent.

Examples:
  # Single model
  python code/test_match/eval_checkpoint.py training_history/.../gen_0030.pt --games 50

  # MCTS mode
  python code/test_match/eval_checkpoint.py training_history/.../gen_0030.pt ^
      --mcts --model-b training_history/.../gen_0101.pt ^
      --value-net value_net.pt --games 50
"""
from __future__ import annotations
import sys, time, multiprocessing as mp
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

# NOTE: torch is imported INSIDE worker functions, NOT at module level.
# This prevents multiprocessing workers from loading CUDA DLLs simultaneously
# and crashing the Windows page file.

# ── Single-model worker (original) ──────────────────────────────────────

def _worker(ckpt_path: str, seed: int, top1: bool = True, num_heads: int = 3, opponent: str = "example", small: bool = False, action_dropout: float = 0.0) -> dict:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    import random
    random.seed(seed)  # deterministic dropout behavior per game seed

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    from AI.ai_example import AI as ExampleAI
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "top2_params" in ckpt:
        param_vec = ckpt["top2_params"][0].numpy() if top1 else ckpt["mean"].numpy()
    elif "mean" in ckpt:
        param_vec = ckpt["mean"].numpy()
    elif "model_state" in ckpt:
        # ss_train.py checkpoint: has model_state but might not have mean
        model_local = create_model(num_heads=num_heads, small=small)
        model_local.load_state_dict(ckpt["model_state"])
        param_vec = model_local.get_parameters_as_vector()
    else:
        raise KeyError(f"Checkpoint keys: {list(ckpt.keys())}")
    model = create_model(num_heads=num_heads, small=small)
    model.set_parameters_from_vector(param_vec)
    agent = NeuralAgent(model=model, action_dropout=action_dropout)

    # Opponent
    if opponent == "rule_v4":
        _rv4_root = Path(__file__).resolve().parents[2] / "其他版本ai" / "rule_v4"
        import sys as _sys
        if str(_rv4_root) not in _sys.path:
            _sys.path.insert(0, str(_rv4_root))
        from ai import AI as RuleV4AI
        opp = RuleV4AI(seed=seed)
    else:
        opp = ExampleAI(seed=seed)

    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops_us = agent._choose_operations(state, our_player)
        ops_opp = opp.choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    side = "1st" if our_player == 0 else "2nd"
    if hp_us <= 0 and hp_opp <= 0:
        result = {"score": 0.5, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}
    elif hp_us > hp_opp:
        result = {"score": 1.0, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}
    elif hp_opp > hp_us:
        result = {"score": 0.0, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}
    else:
        result = {"score": 0.5, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}
    tag = {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[result["score"]]
    print(f"  [{side}] seed={seed:3d}  {tag}  us={int(hp_us):2d}  opp={int(hp_opp):2d}", flush=True)
    return result


# ── MCTS worker ─────────────────────────────────────────────────────────

def _mcts_worker(
    model_a_path: str,
    model_b_path: str,
    value_net_path: str,
    seed: int,
    opponent: str = "example",
    depth: int = 3,
    small_value: bool = False,
) -> dict:
    """Play one game using MCTSAgent vs opponent.  Returns win/draw/loss."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU to avoid OOM with many workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    from SDK.utils.constants import MAX_ROUND
    from SDK.backend.state import PythonBackendState
    from AI.ai_example import AI as ExampleAI
    from my_ai.mcts_agent import create_mcts_agent

    # Determine device (workers use CPU if CUDA is unavailable)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = create_mcts_agent(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        value_net_path=value_net_path,
        opponent_path=None,   # uses model_a as sim opponent
        depth=depth,
        small=small_value,
        device=device,
    )

    # Load opponent
    if opponent == "rule_v4":
        _rv4_root = Path(__file__).resolve().parents[2] / "其他版本ai" / "rule_v4"
        import sys as _sys
        if str(_rv4_root) not in _sys.path:
            _sys.path.insert(0, str(_rv4_root))
        from ai import AI as RuleV4AI
        opp = RuleV4AI(seed=seed)
    else:
        opp = ExampleAI(seed=seed)

    our_player = seed % 2
    opp_player = 1 - our_player

    state = PythonBackendState.initial(seed=seed, cold_handle_rule_illegal=True)
    expert_labels = ["A(Lightning)", "B(Towers)"]
    for turn in range(MAX_ROUND):
        if state.terminal:
            break

        ops_us = agent.choose_operations(state, our_player)
        expert = agent.last_expert_idx
        ops_opp = opp.choose_operations(state, opp_player)

        hp_us = state.bases[our_player].hp
        hp_opp = state.bases[opp_player].hp
        us_action_str = ";".join(str(o) for o in ops_us[:3]) if ops_us else "(none)"
        print(f"  [T{turn+1:3d}] HP:{hp_us:2d},{hp_opp:2d}  Expert:{expert_labels[expert]}  us={us_action_str}  opp={ops_opp[0] if ops_opp else '(none)'}", flush=True)

        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    side = "1st" if our_player == 0 else "2nd"
    if hp_us <= 0 and hp_opp <= 0:
        result = {"score": 0.5, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}
    elif hp_us > hp_opp:
        result = {"score": 1.0, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}
    else:
        result = {"score": 0.0, "first": our_player == 0, "hp_us": int(hp_us), "hp_opp": int(hp_opp)}
    tag = {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[result["score"]]
    print(f"  [MCTS]  [{side}] seed={seed:3d}  {tag}  us={int(hp_us):2d}  opp={int(hp_opp):2d}", flush=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path to checkpoint .pt file (model_a in MCTS mode)")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mean", action="store_true", help="use mean instead of top2_params[0]")
    parser.add_argument("--num-heads", type=int, default=3, help="number of policy heads")
    parser.add_argument("--small", action="store_true", help="use small model (87K params, 1 head)")
    parser.add_argument("--opponent", type=str, default="example", choices=["example", "rule_v4"],
                        help="opponent AI to evaluate against (default: example)")
    parser.add_argument("--action-dropout", type=float, default=0.0,
                        help="probability of replacing agent's action with random legal action per turn")

    # MCTS mode
    parser.add_argument("--mcts", action="store_true", help="enable MCTS dual-expert mode")
    parser.add_argument("--model-b", type=str, default=None,
                        help="path to second expert checkpoint (required for --mcts)")
    parser.add_argument("--value-net", type=str, default=None,
                        help="path to value network checkpoint (required for --mcts)")
    parser.add_argument("--mcts-depth", type=int, default=3,
                        help="MCTS search depth (default: 3)")
    parser.add_argument("--small-value", action="store_true",
                        help="use small value network (84K params)")
    args = parser.parse_args()

    if args.mcts:
        # ── Validate MCTS args ─────────────────────────────────────────
        if not args.model_b:
            parser.error("--mcts requires --model-b")
        if not args.value_net:
            parser.error("--mcts requires --value-net")
        _run_mcts_eval(args)
    else:
        _run_single_eval(args)


def _run_single_eval(args):
    """Original single-model evaluation."""
    top1 = not args.mean

    num_heads = args.num_heads
    if args.small:
        num_heads = 1
    # Note: auto-detection of num_heads from metadata is skipped to avoid
    # loading torch (with CUDA DLLs) in the main process, which exhausts
    # the Windows page file.  Pass --num-heads explicitly if != 3.

    ckpt_path = str(Path(args.ckpt).resolve())

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.starmap(_worker, [(ckpt_path, s, top1, num_heads, args.opponent, args.small, args.action_dropout) for s in range(args.games)])
    dt = time.perf_counter() - t0

    scores = [r["score"] for r in results]
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = sum(1 for s in scores if s == 0.0)
    as_first = [r for r in results if r["first"]]
    as_second = [r for r in results if not r["first"]]
    hp_us_all = [r["hp_us"] for r in results]
    hp_opp_all = [r["hp_opp"] for r in results]

    label = " (TOP1)" if top1 else " (mean)"
    opp_label = "RuleV4" if args.opponent == "rule_v4" else "ExampleAI"
    print(f"\nCheckpoint: {ckpt_path}{label}")
    print(f"Opponent:   {opp_label}")
    if args.action_dropout > 0:
        print(f"Dropout:    {args.action_dropout:.2f}")
    print(f"Games: {args.games} ({args.workers} workers, {dt:.1f}s)")
    print(f"  Win rate:  {wins / args.games:.3f} ({wins}/{args.games})")
    print(f"  Draw rate: {draws / args.games:.3f} ({draws}/{args.games})")
    print(f"  Loss rate: {losses / args.games:.3f} ({losses}/{args.games})")
    if as_first:
        w1 = sum(1 for r in as_first if r["score"] == 1.0)
        print(f"  As first:  {w1}/{len(as_first)} ({w1/len(as_first):.3f})")
    if as_second:
        w2 = sum(1 for r in as_second if r["score"] == 1.0)
        print(f"  As second: {w2}/{len(as_second)} ({w2/len(as_second):.3f})")
    print(f"  Avg HP (us/opp): {np.mean(hp_us_all):.1f} / {np.mean(hp_opp_all):.1f}")
    print(f"  Median HP:       {np.median(hp_us_all):.0f} / {np.median(hp_opp_all):.0f}")
    print()


def _run_mcts_eval(args):
    """MCTS dual-expert evaluation with value network."""
    model_a_path = str(Path(args.ckpt).resolve())
    model_b_path = str(Path(args.model_b).resolve())
    value_net_path = str(Path(args.value_net).resolve())

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.starmap(_mcts_worker, [
            (model_a_path, model_b_path, value_net_path, s,
             args.opponent, args.mcts_depth, args.small_value)
            for s in range(args.games)
        ])
    dt = time.perf_counter() - t0

    scores = [r["score"] for r in results]
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = sum(1 for s in scores if s == 0.0)
    as_first = [r for r in results if r["first"]]
    as_second = [r for r in results if not r["first"]]
    hp_us_all = [r["hp_us"] for r in results]
    hp_opp_all = [r["hp_opp"] for r in results]

    opp_label = "RuleV4" if args.opponent == "rule_v4" else "ExampleAI"
    model_a_name = Path(model_a_path).stem
    model_b_name = Path(model_b_path).stem
    print(f"\n{'='*60}")
    print(f"MCTS Evaluation")
    print(f"  Models:       {model_a_name} + {model_b_name}")
    print(f"  Value net:    {Path(value_net_path).name}")
    print(f"  MCTS depth:   {args.mcts_depth}")
    print(f"  Opponent:     {opp_label}")
    print(f"  Games:        {args.games} ({args.workers} workers, {dt:.1f}s)")
    print(f"{'='*60}")
    print(f"  Win rate:     {wins / args.games:.3f} ({wins}/{args.games})")
    print(f"  Draw rate:    {draws / args.games:.3f} ({draws}/{args.games})")
    print(f"  Loss rate:    {losses / args.games:.3f} ({losses}/{args.games})")
    if as_first:
        w1 = sum(1 for r in as_first if r["score"] == 1.0)
        print(f"  As first:     {w1}/{len(as_first)} ({w1/len(as_first):.3f})")
    if as_second:
        w2 = sum(1 for r in as_second if r["score"] == 1.0)
        print(f"  As second:    {w2}/{len(as_second)} ({w2/len(as_second):.3f})")
    print(f"  Avg HP:       {np.mean(hp_us_all):.1f} / {np.mean(hp_opp_all):.1f}")
    print(f"  Median HP:    {np.median(hp_us_all):.0f} / {np.median(hp_opp_all):.0f}")
    print()


if __name__ == "__main__":
    main()
