"""Sequential value data collection — no multiprocessing, runs on GPU.

Models are loaded once onto GPU.  Games run one at a time, each saving
per-frame HP trajectory for tau label computation.

Usage:
    D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/collect_value_seq.py ^
        training_history/ss_20260709_191050/gen_0030.pt ^
        training_history/ss_20260710_155327/gen_0101.pt ^
        --games 5000 --output-dir value_data_v2
"""

from __future__ import annotations
import sys, time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _CODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from utils.logger import get_logger, redirect_stderr_to_log

from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from my_ai.ss_train import write_npz
from AI.ai_example import AI as ExampleAI
from SDK.backend.state import PythonBackendState
from SDK.utils.constants import MAX_ROUND


def _load_expert(path: str, device: str):
    """Load a checkpoint as NeuralAgent on device."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    num_heads = ckpt.get("num_heads", 3)
    small = ckpt["mean"].shape[0] < 200000
    model = create_model(num_heads=num_heads, small=small)
    if "mean" in ckpt:
        model.set_parameters_from_vector(ckpt["mean"].numpy())
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return NeuralAgent(model=model)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sequential value data collection on GPU")
    parser.add_argument("checkpoints", nargs="+", help="checkpoint .pt paths")
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="value_data_v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Print progress every N games")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────
    log = get_logger(output_dir / "collect.log")
    redirect_stderr_to_log(log)
    _ts = False

    # ── Load models once ──────────────────────────────────────────────────
    log.print(f"Loading {len(args.checkpoints)} models...", timestamp=_ts)
    agents = [_load_expert(p, device) for p in args.checkpoints]
    log.print(f"  → {len(agents)} agents on {device}", timestamp=_ts)

    # Build player pool: checkpoints + example
    pool = list(agents) + [None]  # None = ExampleAI
    pool_names = [Path(p).stem for p in args.checkpoints] + ["example"]

    # Pre-create example agent
    example_ai = ExampleAI()

    def _choose(agent, state, player):
        if agent is None:
            return example_ai.choose_operations(state, player)
        return agent._choose_operations(state, player)

    # ── Game loop ─────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    total_frames = 0
    t0 = time.perf_counter()

    for game_idx in range(args.games):
        # Pick random matchup
        us_idx = rng.randint(len(pool))
        them_idx = rng.randint(len(pool))

        us = pool[us_idx]
        them = pool[them_idx]
        us_name = pool_names[us_idx]
        them_name = pool_names[them_idx]

        seed = args.seed + game_idx
        our_player = seed % 2
        opp_player = 1 - our_player

        state = PythonBackendState.initial(seed=seed, cold_handle_rule_illegal=True)

        boards, statss, hp_traj = [], [], []

        for _ in range(MAX_ROUND):
            if state.terminal:
                break

            ops_us = _choose(us, state, our_player)

            # Record observation + per-frame HP (only when us is a neural model)
            if us is not None:
                obs = us.feature_extractor.encode_observation(
                    state, our_player, np.zeros(us.max_actions))
                boards.append(obs["board"].copy())
                statss.append(obs["stats"].copy())
                hp_traj.append((state.bases[our_player].hp,
                                state.bases[opp_player].hp))

            ops_opp = _choose(them, state, opp_player)

            if our_player == 0:
                state.resolve_turn(ops_us, ops_opp)
            else:
                state.resolve_turn(ops_opp, ops_us)

        # ── Save ──────────────────────────────────────────────────────────
        if boards:
            boards_arr = np.stack(boards, axis=0).astype(np.float16)
            stats_arr = np.stack(statss, axis=0).astype(np.float16)
            value_labels = np.array(hp_traj, dtype=np.float32)

            fname = f"{us_name}_vs_{them_name}_seed{seed}.npz"
            write_npz(
                output_dir / fname,
                boards_arr, stats_arr,
                class_=np.zeros((len(boards), 1), dtype=np.int64),
                action_map=np.zeros((len(boards), 24, 19, 19), dtype=np.float16),
                head_logits=np.zeros((len(boards), 1, 24), dtype=np.float16),
                value=value_labels,
            )
            total_frames += len(boards)

        # Progress
        if (game_idx + 1) % args.save_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (game_idx + 1) / elapsed
            fps = total_frames / elapsed
            log.print(f"  [{game_idx+1}/{args.games}]  {total_frames} frames  "
                      f"{rate:.1f} games/s  {fps:.0f} frames/s  "
                      f"({elapsed:.0f}s elapsed)", timestamp=_ts)

    dt = time.perf_counter() - t0
    n_files = len(list(output_dir.glob("*.npz")))
    log.print(f"Done in {dt:.0f}s", timestamp=_ts)
    log.print(f"  Total frames: {total_frames:,}", timestamp=_ts)
    log.print(f"  Saved .npz:   {n_files} files → {output_dir}/", timestamp=_ts)
    log.print(f"  Avg:          {total_frames/dt:.0f} frames/s", timestamp=_ts)


if __name__ == "__main__":
    main()
