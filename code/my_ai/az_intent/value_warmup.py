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
    import torch
    torch.set_num_threads(1)  # avoid thread thrash across parallel workers
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


def _exp_weighted_labels(stats: np.ndarray, player: np.ndarray, *,
                         tau: float, label_scale: float,
                         label_mode: str = "abs",
                         mix_alpha: float = 0.5,
                         label_weight: str = "geo") -> np.ndarray:
    """Per-frame exp-weighted future HP-diff label for ONE game's frames.

    With ``label_weight="geo"`` this is an exact re-implementation of
    ``az_train.add_weighted_labels`` — same gamma, same unified-P0 view, same
    clip/scale, same per-player sign flip — so the two code paths produce
    identical labels.  Reads the per-frame HP difference from ``stats[:, 1]``
    (``hp_delta`` = player.hp - enemy.hp; features.py's `named` dict puts
    round_ratio at 0 and hp_delta at 1).  ``player`` (n,) is used both to unify
    the view and to re-sign the label.

    ``label_weight`` selects the weight on the future offset k (d is the unified
    P0-view HP diff, the label is then re-signed per player):

      geo  : w_k = gamma^k      — peak at k=0, i.e. the current frame has the
                                  largest weight.  Highly correlated with the
                                  CURRENT HP diff, which the net reads straight
                                  off ``stats[1]`` (measured corr 0.937 at tau=50)
                                  -> the value head can "copy the input".
      kgeo : w_k = k * gamma^k  — w_0 = 0, so the current frame is EXCLUDED and
                                  the peak moves to k~tau.  Same shortcut at a
                                  lower level (corr 0.863 / R^2 0.72 at tau=50)
                                  while keeping more per-frame variation than
                                  simply raising tau (see
                                  docs/az_pool_tau_label_plan.md).
    """
    from my_ai.az_intent.mcts import HP_SCALE

    n = int(stats.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    hp = stats[:, 1].astype(np.float64)
    d = np.where(player == 0, hp, -hp)  # unified P0 view
    gamma = float(np.exp(-1.0 / tau))

    def finish(raw: float, d_t: float, p: int) -> float:
        if label_mode == "abs":
            lab = raw
        elif label_mode == "mix":
            lab = mix_alpha * d_t + (1.0 - mix_alpha) * raw
        else:  # "rel"
            lab = raw - d_t
        label = float(np.clip(lab / HP_SCALE, -1.0, 1.0)) * label_scale
        return label if p == 0 else -label

    out = np.empty(n, dtype=np.float64)
    if label_weight == "kgeo":
        # B_t = sum_{k>=0} k g^k d_{t+k};  splitting off k=0 (which is 0) gives
        # B_t = g (B_{t+1} + A_{t+1}) where A is the plain geometric suffix sum.
        A = B = WA = WB = 0.0
        for t in range(n - 1, -1, -1):
            B = gamma * (B + A)
            WB = gamma * (WB + WA)
            A = d[t] + gamma * A
            WA = 1.0 + gamma * WA
            # WB -> 0 on the final frame (only k=0 contributes, weight 0): fall
            # back to the current frame so the label is still defined.
            raw = (B / WB) if WB > 1e-9 else d[t]
            out[t] = finish(raw, d[t], int(player[t]))
    else:
        suffix = wsum = 0.0
        for t in range(n - 1, -1, -1):
            suffix = d[t] + gamma * suffix
            wsum = 1.0 + gamma * wsum
            out[t] = finish(suffix / wsum, d[t], int(player[t]))
    return out.astype(np.float32)


def _load_chunk(path: Path, label_cfg: dict | None = None) -> dict:
    """Load ONE npz (a per-game file or a merged chunk) into float16 arrays.

    Shared by the all-in-memory and the streaming trainers so both paths produce
    identical labels.  ``label_cfg`` as in ``load_all_data``.
    """
    cfg = dict(label_cfg or {})
    mode = str(cfg.get("mode", "terminal"))
    d = np.load(path)
    out = {
        "board": d["board"],
        "stats": d["stats"],
        "action_map": d["action_map"],
        "head_logits": d["head_logits"],
    }
    if mode == "terminal":
        out["value_target"] = d["value_target"].astype(np.float32)
    else:
        out["value_target"] = _exp_weighted_labels(
            d["stats"], d["player"],
            tau=float(cfg.get("tau", 20.0)),
            label_scale=float(cfg.get("label_scale", 1.0)),
            label_mode=mode,
            mix_alpha=float(cfg.get("mix_alpha", 0.5)),
            label_weight=str(cfg.get("weight", "geo")),
        )
    d.close()
    return out


def load_all_data(npz_paths: list[Path], label_cfg: dict | None = None) -> dict:
    """Load ALL games into memory as float16 stacked arrays (non-streaming path).

    ~32 MB per game decompressed (board 17 MB + action_map 15 MB), so this caps
    the pool at a few hundred games on a 15 GB machine.  For larger pools use
    ``--stream`` (merged_*.npz chunks loaded one at a time).
    """
    cfg = dict(label_cfg or {})
    if str(cfg.get("mode", "terminal")) != "terminal":
        print(f"  [label] recomputing tau-weighted labels from stats: mode={cfg.get('mode')} "
              f"weight={cfg.get('weight', 'geo')} tau={cfg.get('tau')} "
              f"scale={cfg.get('label_scale')} mix_alpha={cfg.get('mix_alpha')}", flush=True)
    parts = []
    for i, path in enumerate(npz_paths):
        if not path.exists():
            continue
        parts.append(_load_chunk(path, cfg))
        if (i + 1) % 25 == 0 or i == len(npz_paths) - 1:
            print(f"  [load] {i+1}/{len(npz_paths)} files", flush=True)
    if not parts:
        raise SystemExit("[warmup] no data loaded")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


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
    device: str = "cpu",
) -> dict:
    """Policy-anchor + value training on in-memory float16 arrays (per-batch f32)."""
    import time

    v_t = data["value_target"]
    print(f"  [train] device={device}", flush=True)
    print(f"  [train] value targets: min={v_t.min():.3f} max={v_t.max():.3f} "
          f"mean={v_t.mean():.3f} std={v_t.std():.3f} "
          f"positive={100*(v_t>0).mean():.1f}% negative={100*(v_t<0).mean():.1f}%", flush=True)
    absv = np.abs(v_t)
    print("  [label] |label| deciles: "
          + " ".join(f"p{p}={np.percentile(absv, p):.3f}" for p in (10, 25, 50, 75, 90))
          + f"  frac<0.05={float((absv < 0.05).mean()):.3f}", flush=True)

    model.to(device)
    print(f"  [train] model on {next(model.parameters()).device}", flush=True)
    # value head output BEFORE training on a fixed probe batch
    probe_idx = np.arange(min(64, data["board"].shape[0]))
    probe_board = torch.from_numpy(data["board"][probe_idx]).float().to(device)
    probe_stats = torch.from_numpy(data["stats"][probe_idx]).float().to(device)
    with torch.no_grad():
        v_before = model(probe_board, probe_stats)["value"].squeeze(-1).cpu().numpy()
    print(f"  [train] value head BEFORE: pred min={v_before.min():.3f} max={v_before.max():.3f} "
          f"std={v_before.std():.3f}  (target std={v_t.std():.3f})", flush=True)

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  [train] trainable tensors: {len(trainable)} "
          f"({sum(p.numel() for p in trainable):,} params)", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
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
            board = torch.from_numpy(data["board"][idx]).float().to(device)
            stats = torch.from_numpy(data["stats"][idx]).float().to(device)
            a_map = torch.from_numpy(data["action_map"][idx]).float().to(device)
            h_logits = torch.from_numpy(data["head_logits"][idx]).float().to(device)
            v_tgt = torch.from_numpy(data["value_target"][idx]).float().to(device)
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
        v_after = model(probe_board, probe_stats)["value"].squeeze(-1).cpu().numpy()
    print(f"  [train] value head AFTER:  pred min={v_after.min():.3f} max={v_after.max():.3f} "
          f"std={v_after.std():.3f}", flush=True)
    print(f"  [train] probe value targets: {v_t[probe_idx].round(2)}", flush=True)
    print(f"  [train] probe preds AFTER:   {v_after.round(2)}", flush=True)
    return {"samples": n, "epochs": epochs, "steps": n_steps}


def train_streaming(model, chunk_paths: list[Path], label_cfg: dict | None = None, *,
                    epochs: int = 10, batch_size: int = 32, lr: float = 1e-3,
                    lambda_value: float = 1.0, seed: int = 0,
                    log_interval: int = 200, device: str = "cpu") -> dict:
    """Value training that never holds more than ONE merged chunk in memory.

    Each epoch shuffles the chunk order, loads a chunk (one batch group, produced
    by ``merge_az_batches.py``), shuffles its rows, trains batches, then frees it.
    Peak memory = one chunk (~1.6 GB for 50 games), independent of pool size —
    this is what lets the pool grow to thousands of games.

    Chunk order is reshuffled every epoch, so batches still mix during training.
    """
    import time

    rng = random.Random(seed)
    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  [stream] device={device} chunks={len(chunk_paths)} "
          f"trainable={len(trainable)} ({sum(p.numel() for p in trainable):,} params)",
          flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    epoch_start = time.perf_counter()
    rows_total = 0
    n_steps = 0
    for epoch in range(epochs):
        order = list(chunk_paths)
        rng.shuffle(order)
        el = ea = ev = 0.0
        ns = 0
        for path in order:
            d = _load_chunk(path, label_cfg)
            n = int(d["board"].shape[0])
            rows_total += n
            if epoch == 0 and ns == 0:
                vt = d["value_target"]
                print(f"  [stream] first chunk {path.name}: {n} rows  "
                      f"label mean={vt.mean():+.3f} std={vt.std():.3f} "
                      f"range=[{vt.min():.3f},{vt.max():.3f}]", flush=True)
            idx = list(range(n))
            rng.shuffle(idx)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                board = torch.from_numpy(d["board"][b]).float().to(device)
                stats = torch.from_numpy(d["stats"][b]).float().to(device)
                a_map = torch.from_numpy(d["action_map"][b]).float().to(device)
                h_logits = torch.from_numpy(d["head_logits"][b]).float().to(device)
                v_tgt = torch.from_numpy(d["value_target"][b]).float().to(device)
                loss, a, v = _train_batch(model, optimizer, board, stats, a_map,
                                          h_logits, v_tgt, lambda_value)
                el += loss
                ea += a
                ev += v
                ns += 1
                n_steps += 1
                if log_interval and n_steps % log_interval == 0:
                    print(f"    [stream] epoch {epoch} step {n_steps}: "
                          f"loss={el/ns:.4f} anchor={ea/ns:.4f} value={ev/ns:.4f}", flush=True)
            del d
        print(f"  epoch {epoch} DONE: loss={el/max(ns,1):.4f} anchor={ea/max(ns,1):.4f} "
              f"value={ev/max(ns,1):.4f} ({time.perf_counter()-epoch_start:.0f}s total)", flush=True)
    model.eval()
    return {"samples": rows_total, "epochs": epochs, "steps": n_steps,
            "chunks": len(chunk_paths)}


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
    parser.add_argument("--skip-collect", action="store_true",
                        help="reuse existing npz files in --data-dir instead of collecting new games")
    parser.add_argument("--stream", action="store_true",
                        help="stream training over merged_*.npz chunks (one chunk resident at a "
                             "time) instead of loading the whole pool into memory.  Required for "
                             "pools beyond a few hundred games; build the chunks with "
                             "merge_az_batches.py")
    parser.add_argument("--keep-bn", action="store_true",
                        help="keep BatchNorm from the hotstart checkpoint instead of folding it "
                             "into no_bn (hotstart must be a BN checkpoint)")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' picks cuda when available; use 'cpu' to force CPU")
    parser.add_argument("--threads", type=int, default=-1,
                        help="intra-op threads; -1 = auto (1 on GPU to avoid thread thrash, "
                             "PyTorch default on CPU)")
    parser.add_argument("--no-tf32", action="store_true",
                        help="disable TF32 for matmul+cuDNN (GPU trains in full FP32; "
                             "TF32's 10-bit mantissa can send training to a different basin)")
    parser.add_argument("--value-pool", type=str, default="gap",
                        choices=["gap", "gapmask", "gapmax", "region", "grid", "attn", "stats"],
                        help="value-head spatial pooling (default 'gap' = original "
                             "global average over all 361 grid cells). 'stats' = no "
                             "spatial path at all (value sees only the 42-dim stats) — "
                             "control for whether board features help the value head")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="freeze initial_conv + resblocks and train only the heads "
                             "(removes the under-constrained backbone drift; the value head "
                             "then fits a fixed feature space)")
    parser.add_argument("--label-mode", type=str, default="terminal",
                        choices=["terminal", "abs", "rel", "mix"],
                        help="value label form. 'terminal' (default) = the per-game HP-diff "
                             "stored in the npz (original behaviour). 'abs'/'rel'/'mix' "
                             "RECOMPUTE an exp-weighted future HP-diff label from the stored "
                             "per-frame stats (mirrors az_train.add_weighted_labels)")
    parser.add_argument("--tau", type=float, default=50.0,
                        help="time constant for the exp-weighted future HP-diff labels "
                             "(view distance; only used when --label-mode != terminal)")
    parser.add_argument("--label-scale", type=float, default=2.0,
                        help="amplify recomputed labels so the value head's output magnitude "
                             "matches what the search expects (only for non-terminal modes)")
    parser.add_argument("--label-mix-alpha", type=float, default=0.5,
                        help="for --label-mode mix: weight of the current HP-diff (rest goes "
                             "to the weighted future average)")
    parser.add_argument("--label-weight", type=str, default="geo",
                        choices=["geo", "kgeo"],
                        help="weight on the future offset k: 'geo' = gamma^k (peak at the "
                             "current frame, the original behaviour) or 'kgeo' = k*gamma^k "
                             "(current frame excluded, peak at k~tau — cuts the 'copy the "
                             "current HP diff' shortcut, see docs/az_pool_tau_label_plan.md)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    elif device == "cuda":
        # GPU does the compute; pin CPU threads to 1 so tiny batches don't thrash
        # 30+ intra-op threads (PyTorch default) and steal cores from other jobs.
        torch.set_num_threads(1)
    print(f"[warmup] device={device} tf32(matmul={torch.backends.cuda.matmul.allow_tf32}, "
          f"cudnn={torch.backends.cudnn.allow_tf32}) threads={torch.get_num_threads()}", flush=True)

    torch.manual_seed(args.seed)
    model = build_model(args.hotstart, keep_bn=args.keep_bn,
                        value_pool=args.value_pool)
    model.eval()
    if args.freeze_backbone:
        n_frozen = 0
        for name, p in model.named_parameters():
            if name.startswith("initial_conv") or name.startswith("resblocks"):
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"[warmup] backbone frozen: {n_frozen:,} params", flush=True)

    if args.skip_collect:
        npz_paths = sorted(Path(args.data_dir).glob("*.npz"))
        print(f"[warmup] skip-collect: {len(npz_paths)} npz files in {args.data_dir}", flush=True)
        if not npz_paths:
            raise SystemExit(f"[warmup] no npz found in {args.data_dir}")
    else:
        seeds = [args.seed * 1000 + g for g in range(args.games)]
        print(f"[warmup] collecting {args.games} self-play games ({args.workers} workers) "
              f"[engine={'C++' if args.native_engine else 'python'}]...", flush=True)
        npz_paths = collect_games_parallel(args.hotstart, seeds, args.data_dir, args.workers,
                                           native_engine=args.native_engine)

    label_cfg = {
        "mode": args.label_mode,
        "weight": args.label_weight,
        "tau": args.tau,
        "label_scale": args.label_scale,
        "mix_alpha": args.label_mix_alpha,
    }
    if args.stream:
        chunks = sorted(Path(args.data_dir).glob("merged_*.npz"))
        print(f"[warmup] stream mode: {len(chunks)} merged chunks in {args.data_dir}", flush=True)
        if not chunks:
            raise SystemExit(f"[warmup] no merged_*.npz in {args.data_dir} — "
                             f"run merge_az_batches.py first")
        metrics = train_streaming(
            model, chunks, label_cfg,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            lambda_value=args.lambda_value, seed=args.seed, device=device,
        )
    else:
        print("[warmup] loading all data into memory...", flush=True)
        data = load_all_data(npz_paths, label_cfg=label_cfg)
        print(f"[warmup] {data['board'].shape[0]} samples in memory "
              f"(~{data['board'].nbytes // (1 << 20)} MB f16)", flush=True)
        metrics = train_in_memory(
            model, data,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            lambda_value=args.lambda_value, seed=args.seed, device=device,
        )
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    model.cpu()  # save CPU tensors regardless of training device
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_heads": model.num_heads,
            "no_bn": model.no_bn,
            "gn": getattr(model, "gn", False),
            "gn_groups": getattr(model, "gn_groups", 8),
            "value_pool": getattr(model, "value_pool", "gap"),
            "latent_dim": model.LATENT_DIM,
            "num_resblocks": model.num_resblocks,
            "completed_batches": 0,
            "warmup": True,
            "label_mode": args.label_mode,
            "label_weight": args.label_weight,
            "tau": args.tau,
            "label_scale": args.label_scale,
            "label_mix_alpha": args.label_mix_alpha,
        },
        args.checkpoint,
    )
    print(f"[warmup] saved -> {args.checkpoint} {metrics}", flush=True)


if __name__ == "__main__":
    main()
