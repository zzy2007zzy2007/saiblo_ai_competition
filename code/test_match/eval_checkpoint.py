"""Evaluate a trained checkpoint against ExampleAI."""
from __future__ import annotations
import sys, time, multiprocessing as mp
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

def _worker(ckpt_path: str, seed: int, top1: bool = True, num_heads: int = 3, opponent: str = "example", small: bool = False) -> dict:
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

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
    agent = NeuralAgent(model=model)

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path to checkpoint .pt file")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mean", action="store_true", help="use mean instead of top2_params[0]")
    parser.add_argument("--num-heads", type=int, default=3, help="number of policy heads")
    parser.add_argument("--small", action="store_true", help="use small model (87K params, 1 head)")
    parser.add_argument("--opponent", type=str, default="example", choices=["example", "rule_v4"],
                        help="opponent AI to evaluate against (default: example)")
    args = parser.parse_args()

    top1 = not args.mean

    if args.small:
        num_heads = 1
    elif args.num_heads != 3:
        num_heads = args.num_heads
    else:
        # Auto-detect num_heads from checkpoint metadata
        ckpt_meta = torch.load(str(Path(args.ckpt).resolve()), map_location="cpu", weights_only=True)
        if "num_heads" in ckpt_meta:
            num_heads = ckpt_meta["num_heads"]
        elif "config" in ckpt_meta and "num_heads" in ckpt_meta["config"]:
            num_heads = ckpt_meta["config"]["num_heads"]
        elif "model_state" in ckpt_meta:
            import re
            keys = list(ckpt_meta["model_state"].keys())
            heads = [k for k in keys if re.match(r"policy_heads\.\d+\.weight", k)]
            num_heads = max(len(heads), 1)
        elif "policy_head2.weight" in ckpt_meta.get("model_state", {}):
            num_heads = 3
        else:
            num_heads = 3  # old single_head=True or unknown

    ckpt_path = str(Path(args.ckpt).resolve())

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.starmap(_worker, [(ckpt_path, s, top1, num_heads, args.opponent, args.small) for s in range(args.games)])
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


if __name__ == "__main__":
    main()
