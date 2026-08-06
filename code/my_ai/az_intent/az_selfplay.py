"""Self-play collection for bundle-MCTS AlphaZero training (T1).

Both players use the bundle MCTS.  At each decision we record a training sample:
board/stats (inputs), player, legal masks, the k sampled bundles with their
per-head intent sample counts, the temperature-scaled visit distribution, the
network's recorded policy outputs (anchor for §3.2), and the terminal
HP-difference value target.

Samples are saved per game (pickle) so training can stream / load them.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_model_from_ckpt(ckpt_path: str):
    from my_ai.network import create_model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_model(
        num_resblocks=ckpt.get("num_resblocks", 6),
        num_heads=ckpt.get("num_heads", 3),
        latent_dim=ckpt.get("latent_dim", 64),
        no_bn=True,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def collect_game(net_fn, model, feature_extractor, mcts, seed, *,
                 max_rounds: int = 512, temp_rounds: int = 30) -> list[dict]:
    """Play one self-play game (both sides = bundle MCTS), record training samples."""
    from SDK.backend.engine import GameState
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.decoder import make_class_mask, make_position_masks
    from my_ai.az_intent.mcts import HP_SCALE

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    samples: list[dict] = []

    for round_idx in range(max_rounds):
        if state.terminal:
            break
        temperature = 1.0 if round_idx < temp_rounds else 1e-6
        for player in (0, 1):
            if state.terminal:
                break
            # recorded policy outputs (anchor) + masks (training denominators)
            obs = feature_extractor.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            recorded_am = out["action_map"].squeeze(0).numpy().astype(np.float16)
            recorded_hl = np.stack(
                [out[f"head{i + 1}_logits"].squeeze(0).numpy()
                 for i in range(model.num_heads)]
            ).astype(np.float16)
            pm = make_position_masks(state, player, intent_decoding=True)
            cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)

            res = mcts.search(state, player, temperature=temperature)
            samples.append(
                {
                    "board": obs["board"].astype(np.float16),
                    "stats": obs["stats"].astype(np.float16),
                    "player": player,
                    "class_mask": cm,
                    "position_mask": pm,
                    "bundles": list(res.bundles),
                    "intent_counts": res.intent_counts,
                    "visit": res.visit_policy.astype(np.float32),
                    "recorded_action_map": recorded_am,
                    "recorded_head_logits": recorded_hl,
                }
            )
            ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in res.chosen_bundle
            ]
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()

    diff = state.bases[0].hp - state.bases[1].hp
    v_p0 = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
    if diff == 0 and state.winner is not None:
        v_p0 = 0.1 if state.winner == 0 else -0.1
    for s in samples:
        s["value_target"] = v_p0 if s["player"] == 0 else -v_p0
    return samples


def _collect_and_save(seed: int, out_dir: str, ckpt_path: str, iterations: int,
                      max_depth_rounds: int, t_class: float, t_pos: float,
                      k: int, sample_mult: int, max_rounds: int, temp_rounds: int) -> dict:
    torch.set_num_threads(1)  # avoid thread thrash across parallel workers
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.az_intent.train import make_net_fn

    model = load_model_from_ckpt(ckpt_path)
    model.eval()
    feat = FeatureExtractor(max_actions=96)
    net_fn = make_net_fn(model, feat)
    mcts = BundleMCTS(net_fn, iterations=iterations, max_depth_rounds=max_depth_rounds,
                      k=k, sample_mult=sample_mult, t_class=t_class, t_pos=t_pos, seed=seed)
    samples = collect_game(net_fn, model, feat, mcts, seed,
                           max_rounds=max_rounds, temp_rounds=temp_rounds)
    path = Path(out_dir) / f"az_selfplay_seed{seed:05d}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"seed": seed, "samples": samples}, f)
    print(f"  [collect] seed={seed} samples={len(samples)} -> {path.name}", flush=True)
    return {"seed": seed, "samples": len(samples)}


def collect_games_parallel(ckpt_path: str, seeds: list[int], out_dir: str, workers: int,
                           iterations: int, max_depth_rounds: int, t_class: float,
                           t_pos: float, k: int, sample_mult: int, max_rounds: int,
                           temp_rounds: int) -> list[Path]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(s, out_dir, ckpt_path, iterations, max_depth_rounds, t_class, t_pos,
             k, sample_mult, max_rounds, temp_rounds) for s in seeds]
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.starmap(_collect_and_save, jobs)
    else:
        results = [_collect_and_save(*j) for j in jobs]
    total = sum(r["samples"] for r in results)
    print(f"[selfplay] collected {total} samples from {len(results)} games", flush=True)
    return [Path(out_dir) / f"az_selfplay_seed{s:05d}.pkl" for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle-MCTS AlphaZero self-play collection (T1)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="training_history/az_intent/az_selfplay_data")
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--max-depth-rounds", type=int, default=4)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--sample-mult", type=int, default=15)
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--temp-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seeds = [args.seed * 10000 + g for g in range(args.games)]
    print(f"[selfplay] collecting {args.games} games ({args.workers} workers, "
          f"{args.iterations} iters / depth {args.max_depth_rounds})...", flush=True)
    paths = collect_games_parallel(
        args.checkpoint, seeds, args.out_dir, args.workers,
        args.iterations, args.max_depth_rounds, args.t_class, args.t_pos,
        args.k, args.sample_mult, args.max_rounds, args.temp_rounds,
    )
    print(f"[selfplay] done -> {len(paths)} files", flush=True)


if __name__ == "__main__":
    main()
