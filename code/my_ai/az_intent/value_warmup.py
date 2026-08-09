"""Value warm-up for intent-space AlphaZero (价值头预热 + 策略锚定).

Collects gen_0120 (expanded 3-head, TOP1) raw-policy SELF-PLAY data:
  - full network outputs (action_map + 3 head logits) per decision = policy anchor targets
  - terminal HP-difference = value target
then trains so that:
  - the POLICY is anchored to its own recorded outputs (no drift while the value learns),
  - the VALUE HEAD learns the HP-difference (becomes informative for the MCTS search).

Usage:
    python code/my_ai/az_intent/value_warmup.py --hotstart .../gen_0120.pt \
        --games 40 --checkpoint training_history/az_intent/gen0120_warm.pt
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def collect_game(model, feature_extractor, seed: int,
                 native_engine: bool = False) -> tuple[list[dict], dict]:
    """Play one self-play game with the raw policy (argmax decode), record samples.

    Each sample = one player's turn: board/stats inputs, the full policy outputs
    (action_map + 3 head logits) as anchor targets, the player, and the terminal
    HP-difference value target (same for all samples of the game, player-signed).

    ``native_engine`` uses the C++ engine (official rules) instead of the Python
    SDK engine.  Returns (samples, game_info).
    """
    from SDK.utils.constants import MAX_ROUND
    from my_ai.decoder import decode_network_output
    from my_ai.az_intent.mcts import HP_SCALE
    from my_ai.az_intent.az_selfplay import make_initial_state

    state = make_initial_state(seed, native_engine)
    samples: list[dict] = []

    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            obs = feature_extractor.encode_observation(state, player, np.zeros(96))
            board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
            stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
            with torch.no_grad():
                out = model(board, stats)
            ops = decode_network_output(out, state, player, temperature=0.0, intent_decoding=True)
            state.apply_operation_list(player, ops)
            samples.append(
                {
                    "board": obs["board"].astype(np.float16),
                    "stats": obs["stats"].astype(np.float16),
                    "action_map": out["action_map"].squeeze(0).numpy().astype(np.float16),
                    "head_logits": np.stack(
                        [out[f"head{i+1}_logits"].squeeze(0).numpy()
                         for i in range(model.num_heads)],
                        axis=0,
                    ).astype(np.float16),
                    "player": player,
                }
            )
        if not state.terminal:
            state.advance_round()

    diff = state.bases[0].hp - state.bases[1].hp
    v_p0 = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
    if diff == 0 and state.winner is not None:
        v_p0 = 0.1 if state.winner == 0 else -0.1
    for s in samples:
        s["value_target"] = v_p0 if s["player"] == 0 else -v_p0
    winner = state.winner
    info = {
        "rounds": state.round_index,
        "winner": "p0" if winner == 0 else ("p1" if winner == 1 else "draw"),
        "hp": (state.bases[0].hp, state.bases[1].hp),
        "samples": len(samples),
    }
    return samples, info


def _collect_and_save(seed: int, out_dir: str, hotstart: str | None,
                      native_engine: bool = False) -> dict:
    """Worker: build the model (same hot-start, deterministic), play one self-play game,
    save the samples to an npz, then free them.  Mirrors code/distill/collect.py's _worker."""
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.train import build_model

    model = build_model(hotstart)
    model.eval()
    feat = FeatureExtractor(max_actions=96)
    samples, info = collect_game(model, feat, seed, native_engine=native_engine)
    path = Path(out_dir) / f"warm_seed{seed:05d}.npz"
    if samples:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            board=np.stack([s["board"] for s in samples], axis=0),
            stats=np.stack([s["stats"] for s in samples], axis=0),
            action_map=np.stack([s["action_map"] for s in samples], axis=0),
            head_logits=np.stack([s["head_logits"] for s in samples], axis=0),
            player=np.asarray([s["player"] for s in samples], dtype=np.int8),
            value_target=np.asarray([s["value_target"] for s in samples], dtype=np.float32),
        )
    print(f"  [collect] seed={seed:5d} rounds={info['rounds']:3d} "
          f"winner={info['winner']} hp={info['hp']} samples={info['samples']}", flush=True)
    return {"seed": seed, "samples": len(samples), "path": str(path)}


def collect_games_parallel(hotstart: str | None, seeds: list[int], out_dir: str, workers: int,
                           native_engine: bool = False) -> list[Path]:
    """Collect self-play data in parallel, one npz per game (workers save to disk,
    parent only receives paths — bounded memory)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(seed, out_dir, hotstart, native_engine) for seed in seeds]
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.starmap(_collect_and_save, jobs)
    else:
        results = [_collect_and_save(*j) for j in jobs]
    total = sum(r["samples"] for r in results)
    print(f"[warmup] collected {total} samples from {len(results)} games", flush=True)
    return [Path(r["path"]) for r in results]


def load_all_data(npz_paths: list[Path]) -> dict:
    """Load all collected games into memory as float16 stacked arrays.

    ~37.5 KB/sample float16 (board + action_map + head_logits + stats + value).
    200 games x ~480 samples = ~96K samples ≈ 3.6 GB — fits comfortably in RAM
    (machine has 15 GB), so training runs in memory instead of re-reading disk
    every epoch (10 epochs of streaming would be ~36 GB of disk I/O).
    """
    boards, stats, a_maps, h_logits, v_targets = [], [], [], [], []
    for i, path in enumerate(npz_paths):
        if not path.exists():
            continue
        d = np.load(path)
        boards.append(d["board"])
        stats.append(d["stats"])
        a_maps.append(d["action_map"])
        h_logits.append(d["head_logits"])
        v_targets.append(d["value_target"])
        d.close()
        if (i + 1) % 25 == 0 or i == len(npz_paths) - 1:
            print(f"  [load] {i+1}/{len(npz_paths)} files", flush=True)
    return {
        "board": np.concatenate(boards),
        "stats": np.concatenate(stats),
        "action_map": np.concatenate(a_maps),
        "head_logits": np.concatenate(h_logits),
        "value_target": np.concatenate(v_targets),
    }


def _train_batch(model, optimizer, board, stats, a_map, h_logits, v_tgt, lambda_value):
    """One optimizer step: anchor loss (keep policy) + value loss (learn HP-diff)."""
    out = model(board, stats)
    anchor_loss = F.mse_loss(out["action_map"], a_map) + F.mse_loss(
        torch.stack([out[f"head{i+1}_logits"] for i in range(model.num_heads)], dim=1),
        h_logits,
    )
    value_loss = F.mse_loss(out["value"].squeeze(-1), v_tgt)
    loss = anchor_loss + lambda_value * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item(), anchor_loss.detach().item(), value_loss.detach().item()


def train_in_memory(
    model,
    data: dict,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_value: float = 1.0,
    seed: int = 0,
    log_interval: int = 200,
) -> dict:
    """Policy-anchor + value training on in-memory float16 arrays (per-batch f32)."""
    import time

    v_t = data["value_target"]
    print(f"  [train] value targets: min={v_t.min():.3f} max={v_t.max():.3f} "
          f"mean={v_t.mean():.3f} std={v_t.std():.3f} "
          f"positive={100*(v_t>0).mean():.1f}% negative={100*(v_t<0).mean():.1f}%", flush=True)

    # value head output BEFORE training on a fixed probe batch
    probe_idx = np.arange(min(64, data["board"].shape[0]))
    probe_board = torch.from_numpy(data["board"][probe_idx]).float()
    probe_stats = torch.from_numpy(data["stats"][probe_idx]).float()
    with torch.no_grad():
        v_before = model(probe_board, probe_stats)["value"].squeeze(-1).numpy()
    print(f"  [train] value head BEFORE: pred min={v_before.min():.3f} max={v_before.max():.3f} "
          f"std={v_before.std():.3f}  (target std={v_t.std():.3f})", flush=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    n = data["board"].shape[0]
    steps_per_epoch = max((n + batch_size - 1) // batch_size, 1)
    epoch_start = time.perf_counter()
    for epoch in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        total_l = total_a = total_v = 0.0
        n_steps = 0
        batch_t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            board = torch.from_numpy(data["board"][idx]).float()
            stats = torch.from_numpy(data["stats"][idx]).float()
            a_map = torch.from_numpy(data["action_map"][idx]).float()
            h_logits = torch.from_numpy(data["head_logits"][idx]).float()
            v_tgt = torch.from_numpy(data["value_target"][idx]).float()
            loss, a, v = _train_batch(
                model, optimizer, board, stats, a_map, h_logits, v_tgt, lambda_value
            )
            total_l += loss
            total_a += a
            total_v += v
            n_steps += 1
            if n_steps % log_interval == 0:
                avg_l = total_l / n_steps
                avg_a = total_a / n_steps
                avg_v = total_v / n_steps
                elapsed = time.perf_counter() - batch_t0
                eta = (steps_per_epoch - n_steps) * (elapsed / max(n_steps, 1))
                print(f"    epoch {epoch} batch {n_steps}/{steps_per_epoch}: "
                      f"loss={avg_l:.4f} anchor={avg_a:.4f} value={avg_v:.4f} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
        print(f"  epoch {epoch} DONE: loss={total_l/max(n_steps,1):.4f} "
              f"anchor={total_a/max(n_steps,1):.4f} value={total_v/max(n_steps,1):.4f} "
              f"({time.perf_counter()-epoch_start:.0f}s total)", flush=True)
    model.eval()

    with torch.no_grad():
        v_after = model(probe_board, probe_stats)["value"].squeeze(-1).numpy()
    print(f"  [train] value head AFTER:  pred min={v_after.min():.3f} max={v_after.max():.3f} "
          f"std={v_after.std():.3f}", flush=True)
    print(f"  [train] probe value targets: {v_t[probe_idx].round(2)}", flush=True)
    print(f"  [train] probe preds AFTER:   {v_after.round(2)}", flush=True)
    return {"samples": n, "epochs": epochs, "steps": n_steps}


def main() -> None:
    from my_ai.az_intent.train import build_model

    parser = argparse.ArgumentParser(description="Value warm-up (策略锚定 + 价值头预热)")
    parser.add_argument("--hotstart", type=str, default=None, help="ES checkpoint (gen_0120)")
    parser.add_argument("--checkpoint", type=str, default="training_history/az_intent/gen0120_warm.pt")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="training_history/az_intent/warm_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--native-engine", action="store_true",
                        help="collect on the C++ engine (official rules) instead of the Python SDK engine")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = build_model(args.hotstart)
    model.eval()

    seeds = [args.seed * 1000 + g for g in range(args.games)]
    print(f"[warmup] collecting {args.games} self-play games ({args.workers} workers) "
          f"[engine={'C++' if args.native_engine else 'python'}]...", flush=True)
    npz_paths = collect_games_parallel(args.hotstart, seeds, args.data_dir, args.workers,
                                       native_engine=args.native_engine)

    print("[warmup] loading all data into memory...", flush=True)
    data = load_all_data(npz_paths)
    print(f"[warmup] {data['board'].shape[0]} samples in memory "
          f"(~{data['board'].nbytes // (1 << 20)} MB f16)", flush=True)

    metrics = train_in_memory(
        model, data,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        lambda_value=args.lambda_value, seed=args.seed,
    )
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_heads": model.num_heads,
            "no_bn": model.no_bn,
            "latent_dim": model.LATENT_DIM,
            "num_resblocks": model.num_resblocks,
            "completed_batches": 0,
            "warmup": True,
        },
        args.checkpoint,
    )
    print(f"[warmup] saved -> {args.checkpoint} {metrics}", flush=True)


if __name__ == "__main__":
    main()
