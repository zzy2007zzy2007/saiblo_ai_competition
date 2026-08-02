"""Collect game data from model checkpoints for value network training.

Each game randomly picks two players from the pool {checkpoints, "example"}
and records board/stats from Player A's perspective with value = (hpA - hpB).

Usage:
    python code/my_ai/collect_value_data.py \\
        path/to/gen_0030.pt path/to/gen_0101.pt \\
        --games 1000 --workers 16 --output-dir value_data

Output .npz files (compatible with train_value_net.py):
  - board:  (T, 28, 19, 19) float16
  - stats:  (T, 42)         float16
  - value:  (T, 2)          float32 — (hp_us, hp_opp) per frame
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _CODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from utils.logger import get_logger, redirect_stderr_to_log

# NOTE: torch and my_ai imports are INSIDE worker functions to avoid
# Windows page-file crash from spawning CUDA-loaded processes.


# ---------------------------------------------------------------------------
# Helper: load a player agent (checkpoint path or "example")
# ---------------------------------------------------------------------------
_PLAYER_CACHE: dict[str, NeuralAgent] = {}


def _load_player(spec: str, num_heads: int, small: bool):
    """Load a player agent.  ``spec`` is a checkpoint path or ``"example"``.

    Results are cached so each worker loads each checkpoint only once.
    """
    # Return cached agent if available
    if spec in _PLAYER_CACHE:
        return _PLAYER_CACHE[spec]

    import torch
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    if spec == "example":
        from AI.ai_example import AI as ExampleAI
        agent = ExampleAI()
        _PLAYER_CACHE[spec] = agent
        return agent

    ckpt = torch.load(spec, map_location="cpu", weights_only=True)
    model = create_model(num_heads=num_heads, small=small)
    if "mean" in ckpt:
        model.set_parameters_from_vector(ckpt["mean"].numpy())
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    agent = NeuralAgent(model=model)
    _PLAYER_CACHE[spec] = agent
    return agent


def _choose_ops(agent, state, player):
    """Uniform interface: call the right method for NeuralAgent vs BaseAgent."""
    if hasattr(agent, "_choose_operations"):
        return agent._choose_operations(state, player)
    return agent.choose_operations(state, player)


def _worker_wrapper(args):
    """Unpack tuple for imap_unordered."""
    return _worker(*args)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(us_spec: str, them_spec: str, seed: int, output_dir: str,
            small: bool, num_heads: int) -> dict:
    """Play one game: ``us_spec`` vs ``them_spec``, save .npz from our side."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    from SDK.backend.state import PythonBackendState
    from SDK.utils.constants import MAX_ROUND

    from my_ai.ss_train import write_npz

    us = _load_player(us_spec, num_heads, small)
    them = _load_player(them_spec, num_heads, small)

    our_player = seed % 2
    opp_player = 1 - our_player

    state = PythonBackendState.initial(seed=seed, cold_handle_rule_illegal=True)

    boards, statss, hp_traj = [], [], []

    for _ in range(MAX_ROUND):
        if state.terminal:
            break

        ops_us = _choose_ops(us, state, our_player)

        # Record observation + per-frame HP (only when "us" is a checkpoint)
        if us_spec != "example":
            obs = us.feature_extractor.encode_observation(
                state, our_player, np.zeros(us.max_actions))
            boards.append(obs["board"].copy())
            statss.append(obs["stats"].copy())
            hp_traj.append((state.bases[our_player].hp,
                            state.bases[opp_player].hp))

        ops_opp = _choose_ops(them, state, opp_player)

        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp

    # ── Save .npz (only when us is a neural model) ────────────────────────
    fname_prefix = "us" if us_spec == "example" else Path(us_spec).stem
    if boards:
        boards_arr = np.stack(boards, axis=0).astype(np.float16)
        stats_arr = np.stack(statss, axis=0).astype(np.float16)
        value_labels = np.array(hp_traj, dtype=np.float32)  # (T, 2) per-frame HP

        fname = f"{fname_prefix}_vs_{Path(them_spec).stem}_seed{seed}.npz"
        write_npz(
            Path(output_dir) / fname,
            boards_arr, stats_arr,
            class_=np.zeros((len(boards), 1), dtype=np.int64),
            action_map=np.zeros((len(boards), 24, 19, 19), dtype=np.float16),
            head_logits=np.zeros((len(boards), 1, 24), dtype=np.float16),
            value=value_labels,
        )

    return {
        "us": Path(us_spec).name if us_spec != "example" else "example",
        "them": Path(them_spec).name if them_spec != "example" else "example",
        "seed": seed,
        "hp_us": hp_us,
        "hp_opp": hp_opp,
        "turns": len(boards),
        "saved": len(boards),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect value training data")
    parser.add_argument("checkpoints", nargs="+",
                        help="Paths to .pt checkpoint files")
    parser.add_argument("--games", type=int, default=1000,
                        help="Total games to play (distributed across matchups)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="value_data")
    parser.add_argument("--opponents", nargs="+",
                        default=["example"],
                        help="Opponent pool entries (checkpoint paths or 'example')")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──────────────────────────────────────────────────────────────
    log_path = output_dir / "collect.log"
    log = get_logger(log_path)
    redirect_stderr_to_log(log)
    # Suppress logger's own timestamp for cleaner output
    _ts = False

    # Build the full player pool
    pool = list(args.checkpoints)
    for o in args.opponents:
        if o not in pool:
            pool.append(o)
    pool = [str(Path(p).resolve()) if p != "example" else "example" for p in pool]

    log.print(f"Player pool: {pool}", timestamp=_ts)

    # Determine model config from first checkpoint
    log.print("Loading checkpoint to determine model config...", timestamp=_ts)
    import torch
    first_ckpt = [p for p in pool if p != "example"][0]
    ckpt_data = torch.load(first_ckpt, map_location="cpu", weights_only=True)
    num_heads = ckpt_data.get("num_heads", 3)
    small = ckpt_data["mean"].shape[0] < 200000

    log.separator("=")
    log.print("Value Data Collection", timestamp=_ts)
    log.print(f"  Player pool:  {pool}", timestamp=_ts)
    log.print(f"  Games:        {args.games}", timestamp=_ts)
    log.print(f"  Workers:      {args.workers}", timestamp=_ts)
    log.print(f"  Output dir:   {output_dir}", timestamp=_ts)
    log.print(f"  Model:        {'small' if small else 'large'} "
              f"({ckpt_data['mean'].shape[0]:,} params, {num_heads} heads)", timestamp=_ts)
    log.separator("=")

    # Build tasks: for each game, randomly pick us and them from pool
    rng = np.random.RandomState(args.seed)
    tasks = []
    for i in range(args.games):
        # Pick two (possibly same; self-play is informative for value)
        us = pool[rng.randint(len(pool))]
        them = pool[rng.randint(len(pool))]
        tasks.append((us, them, args.seed + i, str(output_dir.resolve()),
                      small, num_heads))

    log.print(f"Total games to play: {len(tasks)}", timestamp=_ts)
    log.print(f"  Matchups: {len(set((u, t) for u, t, *_ in tasks))} unique (us, them) pairs", timestamp=_ts)

    start = time.time()
    total_frames = 0
    total_saved = 0

    try:
        with mp.Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker_wrapper, tasks), 1):
                total_frames += result["turns"]
                total_saved += result["saved"]
                if i % 200 == 0 or i == len(tasks):
                    elapsed = time.time() - start
                    rate = i / elapsed
                    log.print(f"  [{i}/{len(tasks)}]  {total_saved} files  {total_frames} frames  "
                              f"{rate:.1f} games/s  ({elapsed:.0f}s)", timestamp=_ts)
    except Exception as e:
        log.print(f"ERROR: pool crashed: {e}", timestamp=_ts)
        import traceback
        log.print(traceback.format_exc(), timestamp=_ts)
        raise

    elapsed = time.time() - start

    n_files = total_saved
    log.print(f"Done in {elapsed:.0f}s ({elapsed / max(len(tasks), 1):.1f}s/game)", timestamp=_ts)
    log.print(f"  Saved .npz:   {n_files} files → {output_dir}/", timestamp=_ts)
    log.print(f"  Total frames: {total_frames}", timestamp=_ts)
    log.print(f"  Frames/sec:   {total_frames / max(elapsed, 1):.0f}", timestamp=_ts)


if __name__ == "__main__":
    main()
