"""Strategy Space Evolution — supervised self-play training from top-K individuals.

Completely independent from es_train.py and elite_bc.py (no imports to either).
Uses only network.py, agent.py, and decoder.py as dependencies.

Core idea:
  1. ES-style mirrored sampling generates a population of parameter vectors
  2. Each individual plays games, collecting board/stats/class_/action_map/head_logits
  3. Top-K individuals' data is assembled into SSDataset with stochastic mutations
  4. Supervised update trains the mean model on mutated elite data
  5. The model parameters become the new mean for the next generation
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
import csv
import signal
import time
import multiprocessing as mp
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from my_ai.network import create_model, AntWarNetwork
from my_ai.agent import NeuralAgent
from utils.logger import get_logger


def _eval_worker(
    params_flat: np.ndarray,
    opp_params_flat: np.ndarray,
    seed: int,
    num_heads: int = 3,
    bc_dir=None,
    gen=0,
    ind=0,
) -> dict:
    """Run one match: params vs opponent params, collect game data to .npz."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import sys as _sys
    from pathlib import Path as _Path
    _RP = _Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = _Path(__file__).resolve().parents[1]
    for _p in (_RP, _CODE):
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))

    import torch as _torch
    _torch.set_num_threads(1)

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND

    model = create_model(num_heads=num_heads)
    model.set_parameters_from_vector(params_flat)
    agent = NeuralAgent(model=model)

    opp_model = create_model(num_heads=num_heads)
    opp_model.set_parameters_from_vector(opp_params_flat)
    opponent = NeuralAgent(model=opp_model)

    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    bc_boards, bc_stats = [], []
    bc_class_labels, bc_action_maps, bc_head_logits = [], [], []

    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops_us = agent._choose_operations(state, our_player)
        if bc_dir:
            feat = agent.feature_extractor.encode_observation(
                state, our_player, np.zeros(agent.max_actions))
            bc_boards.append(feat["board"].copy())
            bc_stats.append(feat["stats"].copy())

            # class labels: argmax of each head's logits
            cls_labels = []
            head_logits_list = []
            for hi in range(num_heads):
                logits = agent.last_output[f"head{hi+1}_logits"].squeeze(0)  # (24,)
                cls_labels.append(logits.argmax().item())
                head_logits_list.append(logits.cpu().numpy())
            bc_class_labels.append(np.array(cls_labels))  # (num_heads,)
            bc_head_logits.append(np.stack(head_logits_list, axis=0))  # (num_heads, 24)

            # full action_map
            bc_action_maps.append(
                agent.last_output["action_map"].squeeze(0).cpu().numpy()
            )  # (NUM_CLASSES, 19, 19)

        ops_opp = opponent._choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    if bc_dir and bc_boards:
        boards_arr = np.stack(bc_boards, axis=0)       # (T, 28, 19, 19)
        stats_arr = np.stack(bc_stats, axis=0)          # (T, 42)
        class_arr = np.stack(bc_class_labels, axis=0)   # (T, num_heads)
        map_arr = np.stack(bc_action_maps, axis=0)      # (T, NUM_CLASSES, 19, 19)
        logits_arr = np.stack(bc_head_logits, axis=0)   # (T, num_heads, 24)

        from my_ai.ss_train import write_npz
        write_npz(
            _Path(bc_dir) / f"gen_{gen:04d}_ind{ind:03d}_seed{seed}.npz",
            boards_arr, stats_arr, class_arr, map_arr, logits_arr,
        )

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    if hp_us <= 0 and hp_opp <= 0:
        result = {"score": 0.5, "our_player": our_player}
    elif hp_us > hp_opp:
        result = {"score": 1.0, "our_player": our_player}
    elif hp_opp > hp_us:
        result = {"score": 0.0, "our_player": our_player}
    else:
        result = {"score": 0.5, "our_player": our_player}
    score = result["score"]
    if score == 1.0:
        color = "\033[92m"
    elif score == 0.0:
        color = "\033[91m"
    else:
        color = "\033[93m"
    reset = "\033[0m"
    print(f"{color}.{reset}", end="", flush=True)
    return result


class SSDataset(Dataset):
    """Dataset from top-K game .npz files with HOLD filtering + stochastic mutations.

    Fields:
      - board:        (T, 28, 19, 19) float16
      - stats:        (T, 42)         float16
      - class_label:  (T, N_heads)    int64   (argmax class per head)
      - action_map:   (T, NUM_CLASSES, 19, 19) float16
      - head_logits:  (T, N_heads, 24) float16

    During __getitem__, random mutations are applied:
      - class mutation: replace argmax with softmax-sampled class (p=p_mutate)
      - position mutation: add Gaussian noise to action_map (p=p_mutate)
    """

    HOLD_CLASS = 23

    def __init__(self, npz_paths, p_hold=0.1, p_mutate=0.1, temperature=3.0, pos_noise_std=0.1):
        boards, statss, classes, maps, logits = [], [], [], [], []
        for p in npz_paths:
            data = np.load(p)
            boards.append(data["board"])
            statss.append(data["stats"])
            classes.append(data["class_"])
            maps.append(data["action_map"])
            logits.append(data["head_logits"])

        self.board = np.concatenate(boards, axis=0)        # (T_total, 28, 19, 19)
        self.stats = np.concatenate(statss, axis=0)        # (T_total, 42)
        self.class_label = np.concatenate(classes, axis=0)  # (T_total, N_heads)
        self.action_map = np.concatenate(maps, axis=0)     # (T_total, NUM_CLASSES, 19, 19)
        self.head_logits = np.concatenate(logits, axis=0)  # (T_total, N_heads, 24)

        # HOLD downsampling: rounds where all heads == HOLD_CLASS, kept at p_hold
        all_hold = (self.class_label == self.HOLD_CLASS).all(axis=1)
        rng_hold = np.random.default_rng()
        keep_hold = rng_hold.random(len(self.class_label)) < p_hold
        valid = (~all_hold) | (all_hold & keep_hold)

        n_before = len(self.board)
        self.board = self.board[valid]
        self.stats = self.stats[valid]
        self.class_label = self.class_label[valid]
        self.action_map = self.action_map[valid]
        self.head_logits = self.head_logits[valid]
        n_after = len(self.board)

        if n_before > 0 and n_after < n_before:
            print(f"  [SS] HOLD downsampled: {n_before} -> {n_after} "
                  f"({100 * (n_before - n_after) // n_before}% removed)")

        self.p_mutate = p_mutate
        self.temperature = temperature
        self.pos_noise_std = pos_noise_std

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        board = torch.from_numpy(self.board[idx]).float()         # (28, 19, 19)
        stats = torch.from_numpy(self.stats[idx]).float()         # (42,)
        cls_label = torch.from_numpy(self.class_label[idx].copy())  # (N_heads,)
        action_map = torch.from_numpy(self.action_map[idx].copy()).float()  # (NUM_CLASSES, 19, 19)

        rng_item = torch.Generator()
        rng_item.manual_seed(int(torch.randint(0, 2**31, (1,)).item()))

        # Class mutation: temperature-softmax sampling, EXCLUDING HOLD (class 23)
        if torch.rand(1).item() < self.p_mutate:
            logits_row = self.head_logits[idx]  # (N_heads, 24) — numpy
            for hi in range(logits_row.shape[0]):
                logits_t = torch.from_numpy(logits_row[hi])  # (24,)
                logits_t = torch.clamp(logits_t, -50.0, 50.0)  # 安全兜底防 inf/nan
                # Mask out HOLD class so mutation always explores non-HOLD actions
                logits_t[self.HOLD_CLASS] = float("-inf")
                probs = F.softmax(logits_t / self.temperature, dim=0)
                cls_label[hi] = torch.multinomial(probs, 1).item()

        # Position mutation: add Gaussian noise to action_map
        if torch.rand(1).item() < self.p_mutate:
            noise = torch.randn_like(action_map) * self.pos_noise_std
            action_map = action_map + noise

        return {
            "board": board,
            "stats": stats,
            "class_label": cls_label,
            "action_map": action_map,
        }


def ss_supervised_update(model, dataset, device, epochs=3, lr=1e-3, batch_size=64,
                         lambda_class=1.0, lambda_map=1.0):
    """Train model on SSDataset with classification + action-map distillation loss.

    Args:
        model: AntWarNetwork to train (modified in-place).
        dataset: SSDataset instance.
        device: torch device.
        epochs: number of full passes over dataset.
        lr: learning rate.
        batch_size: DataLoader batch size.
        lambda_class: weight for class cross-entropy loss.
        lambda_map: weight for action-map KL divergence loss.

    Returns:
        dict with keys: class_loss, map_loss, total_loss, samples.
    """
    if len(dataset) == 0:
        return {"class_loss": 0.0, "map_loss": 0.0, "total_loss": 0.0, "samples": 0}

    model.train()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_cls = 0.0
    total_map = 0.0
    total = 0.0
    n_batches = 0

    for _ in range(epochs):
        for batch in loader:
            board = batch["board"].to(device)                     # (B, 28, 19, 19)
            stats = batch["stats"].to(device)                     # (B, 42)
            cls_label = batch["class_label"].to(device)           # (B, N_heads)
            target_map = batch["action_map"].to(device)           # (B, NUM_CLASSES, 19, 19)

            optimizer.zero_grad()
            output = model(board, stats)

            # Class loss: average CE over all policy heads
            cls_loss = 0.0
            for i in range(model.num_heads):
                cls_loss += F.cross_entropy(
                    output[f"head{i+1}_logits"], cls_label[:, i])
            cls_loss /= model.num_heads

            # Map loss: KL(softmax(pred) || softmax(target).detach())
            # Flatten spatial dims: (B, NUM_CLASSES, 19, 19) -> (B, NUM_CLASSES, 361)
            B, C, H, W = target_map.shape
            pred_flat = output["action_map"].view(B, C, -1)      # (B, C, 361)
            target_flat = target_map.view(B, C, -1)              # (B, C, 361)
            # Permute to (B, 361, C) for softmax over class dim
            pred_flat = pred_flat.permute(0, 2, 1).contiguous()  # (B, 361, C)
            target_flat = target_flat.permute(0, 2, 1).contiguous()  # same
            # KL: pred_log_softmax vs target_softmax
            map_loss = F.kl_div(
                F.log_softmax(pred_flat, dim=-1),
                F.softmax(target_flat.detach(), dim=-1),
                reduction="batchmean",
            )

            loss = lambda_class * cls_loss + lambda_map * map_loss
            loss.backward()
            optimizer.step()

            total_cls += cls_loss.item()
            total_map += map_loss.item()
            total += loss.item()
            n_batches += 1

    model.cpu()
    model.eval()
    return {
        "class_loss": total_cls / n_batches,
        "map_loss": total_map / n_batches,
        "total_loss": total / n_batches,
        "samples": len(dataset),
    }


# ════════════════════════════════════════════════
# Data I/O
# ════════════════════════════════════════════════


def write_npz(path, board, stats, class_, action_map, head_logits):
    """Save game timestep data as compressed .npz.

    board/action_map/head_logits/stats saved as float16 (with clip to safe range),
    class_ saved as int64.
    """
    F16_MAX = 65504.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        board=board.astype(np.float16),
        stats=stats.astype(np.float16),
        class_=class_,
        action_map=np.clip(action_map, -F16_MAX, F16_MAX).astype(np.float16),
        head_logits=np.clip(head_logits, -F16_MAX, F16_MAX).astype(np.float16),
    )


def collect_npz(bc_dir, selected_indices, gen):
    """Glob-match gen_NNNN_ind{idx}_*.npz for each selected index."""
    paths = []
    for idx in selected_indices:
        paths.extend(sorted(bc_dir.glob(f"gen_{gen:04d}_ind{idx:03d}_*.npz")))
    return paths


def cleanup_gen_npz(bc_dir, gen):
    """Delete all .npz files for a given generation."""
    for f in bc_dir.glob(f"gen_{gen:04d}_*.npz"):
        f.unlink()


# ════════════════════════════════════════════════
# Checkpoint
# ════════════════════════════════════════════════


def save_checkpoint(path, mean, model, generation, config=None):
    """Save training state to checkpoint file."""
    data = {
        "mean": torch.from_numpy(mean),
        "model_state": model.state_dict(),
        "generation": generation,
        "num_heads": model.num_heads,
    }
    if config is not None:
        data["config"] = config
    torch.save(data, path)


# ════════════════════════════════════════════════
# Opponent selection
# ════════════════════════════════════════════════


def select_opponents(population_size, games_per_individual, rng):
    """Select opponent indices from the current population.

    Returns ``games_per_individual // 2`` opponent indices.
    Each opponent is played twice (first player / second player parity).
    """
    n_pairs = games_per_individual // 2
    return [rng.randint(0, population_size - 1) for _ in range(n_pairs)]


# ════════════════════════════════════════════════
# Main training loop
# ════════════════════════════════════════════════


def reload_config(config_path: str, args) -> bool:
    """Hot-reload training parameters from config.txt at generation boundary.
    Returns True if any parameter was changed.
    """
    try:
        kv: dict[str, str] = {}
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip()
        float_keys = {"sigma", "lr"}
        int_keys = {"generations", "pop_size", "games", "workers", "k", "p_mutate", "temperature"}
        changed = False
        for key in float_keys:
            if key in kv:
                new = float(kv[key])
                if getattr(args, key.replace("-", "_"), None) != new:
                    setattr(args, key.replace("-", "_"), new)
                    changed = True
        for key in int_keys:
            if key in kv:
                new = int(kv[key])
                if getattr(args, key.replace("-", "_"), None) != new:
                    setattr(args, key.replace("-", "_"), new)
                    changed = True
        return changed
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Strategy Space Evolution training")
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--games", type=int, default=18,
                        help="games per individual per generation")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--k", type=int, default=5,
                        help="top-K individuals selected for supervised update")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--p-mutate", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--p-hold", type=float, default=0.1)
    parser.add_argument("--pos-noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: auto timestamp)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="resume from checkpoint")

    args = parser.parse_args()

    # ── Output directory ──────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("training_history") / f"ss_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_dir = out_dir / "bc_data"
    bc_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger & CSV ──────────────────────────────────────────────
    log = get_logger(out_dir / "train.log")
    csv_path = out_dir / "history.csv"

    # Save config
    with open(out_dir / "config.txt", "w") as f:
        for key, val in vars(args).items():
            f.write(f"{key}={val}\n")
        f.write(f"timestamp={ts}\n")

    # ── Model ─────────────────────────────────────────────────────
    model = create_model(num_heads=args.num_heads)
    param_count = model.count_parameters()

    # Elite retention
    elite_params: list[np.ndarray] = []

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        start_gen = ckpt.get("generation", 0)
        log.print(key="resume", value=f"from generation {start_gen} @ {args.checkpoint}")
    else:
        start_gen = 0

    mean = model.get_parameters_as_vector()
    rng = np.random.RandomState(args.seed)

    # ── Ctrl+C handler ────────────────────────────────────────────
    interrupted = False

    def _signal_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            log.print("Forced exit.")
            sys.exit(1)
        interrupted = True
        log.print("")
        log.print(key="interrupt", value="Ctrl+C received, stopping after current generation...")
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except AttributeError:
        pass

    # ── Print header ──────────────────────────────────────────────
    log.header("Strategy Space Evolution Training")
    log.print(key="out_dir", value=out_dir)
    log.print(key="params", value=f"{param_count:,}")
    log.print(key="seed", value=args.seed)
    log.print(key="pop_size", value=args.pop_size)
    log.print(key="sigma", value=args.sigma)
    log.print(key="games_per_ind", value=args.games)
    log.print(key="workers", value=args.workers)
    log.print(key="generations", value=args.generations)
    log.print(key="num_heads", value=args.num_heads)
    log.print(key="k", value=args.k)
    log.print(key="epochs", value=args.epochs)
    log.print(key="lr", value=args.lr)
    log.print(key="batch_size", value=args.batch_size)
    log.print(key="p_mutate", value=args.p_mutate)
    log.print(key="temperature", value=args.temperature)
    log.print(key="p_hold", value=args.p_hold)
    log.print(key="pos_noise_std", value=args.pos_noise_std)
    log.separator("-")

    log.print_table(gen="gen", best="best_fit", avg="avg_fit",
                    eval_s="eval(s)", total_s="total(s)",
                    ss="ss_loss")
    log.separator("-", width=60, timestamp=False)
    log.print("  [Ctrl+C: stop after current gen | Second Ctrl+C: force quit]")
    log.separator("-", width=60, timestamp=False)

    # ── CSV header ────────────────────────────────────────────────
    csv_header = ["generation", "best_fitness", "avg_fitness",
                  "eval_time_s", "total_time_s",
                  "ss_samples", "ss_class_loss", "ss_map_loss", "ss_total_loss"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)

    # ── Training loop ─────────────────────────────────────────────
    pool = mp.Pool(args.workers)
    pause_file = out_dir / "PAUSE"
    pause_file.write_text("resume", encoding="utf-8")
    latest_result = None

    try:
        for gen in range(start_gen, args.generations):
            if interrupted:
                break

            # PAUSE check + config reload
            while pause_file.read_text(encoding="utf-8").strip().lower() == "pause" and not interrupted:
                time.sleep(2)
            if interrupted:
                break
            if reload_config(str(out_dir / "config.txt"), args):
                log.print(key="config_reload",
                          value=f"sigma={args.sigma} lr={args.lr} "
                                f"pop={args.pop_size} games={args.games} workers={args.workers}")

            t0 = time.time()

            # 1. Mirrored sampling (with elite retention injection)
            assert args.pop_size % 2 == 0, "pop_size must be even (mirrored sampling)"
            n_noise = args.pop_size // 2
            noise = rng.randn(n_noise, param_count).astype(np.float32)
            params_list = []
            for n in noise:
                params_list.append(mean + args.sigma * n)
                params_list.append(mean - args.sigma * n)

            # Elite retention: replace bottom individuals with stored elites
            if elite_params:
                n_elite = max(1, int(args.pop_size ** 0.25))
                for i in range(min(n_elite, len(elite_params))):
                    params_list[-(i + 1)] = elite_params[i].copy()

            # 2. Select opponents
            opp_indices = select_opponents(args.pop_size, args.games, rng)
            k_per_ind = args.games // 2
            opp_params_list = [params_list[i] for i in opp_indices]

            # 3. Build all evaluation tasks
            eval_start = time.time()
            all_args = []
            for idx in range(args.pop_size):
                for k_idx in range(k_per_ind):
                    opp_params = opp_params_list[k_idx]
                    base_seed = (args.seed + gen * args.pop_size * args.games
                                 + idx * args.games + k_idx * 2)
                    all_args.append((
                        params_list[idx], opp_params, base_seed,
                        args.num_heads,
                        str(bc_dir), gen, idx,
                    ))
                    all_args.append((
                        params_list[idx], opp_params, base_seed + 1,
                        args.num_heads,
                        str(bc_dir), gen, idx,
                    ))

            # 4. Submit and collect results
            async_result = pool.starmap_async(_eval_worker, all_args)
            while True:
                try:
                    all_results = async_result.get(timeout=2)
                    break
                except mp.TimeoutError:
                    if interrupted:
                        break
                    continue
                except (mp.context.BrokenProcessPool, OSError, ValueError):
                    if interrupted:
                        break
                    raise

            if interrupted:
                break

            # 5. Parse fitness
            scores = np.array([r["score"] if isinstance(r, dict) else r
                               for r in all_results], dtype=np.float32)
            fitness = np.mean(scores.reshape(args.pop_size, args.games), axis=1)
            eval_time = time.time() - eval_start

            # 6. Select top-K and train
            top_k_idx = sorted(range(len(fitness)), key=lambda i: -fitness[i])[:args.k]
            npz_paths = collect_npz(bc_dir, top_k_idx, gen)

            ss_ret = {"samples": 0, "class_loss": 0.0, "map_loss": 0.0, "total_loss": 0.0}
            if npz_paths:
                ds = SSDataset(
                    npz_paths,
                    p_hold=args.p_hold,
                    p_mutate=args.p_mutate,
                    temperature=args.temperature,
                    pos_noise_std=args.pos_noise_std,
                )
                if len(ds) > 0:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ss_ret = ss_supervised_update(
                        model, ds, device=device,
                        epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                    )
                    mean = model.get_parameters_as_vector()
                else:
                    log.print(key="ss", value="all-HOLD filtered, no valid samples")
            else:
                log.print(key="ss", value="no data files for top-K")

            total_time = time.time() - t0

            # 7. Print log
            log.print_table(
                gen=gen,
                best=f"{fitness.max():.4f}",
                avg=f"{fitness.mean():.4f}",
                eval_s=f"{eval_time:.1f}",
                total_s=f"{total_time:.1f}",
                ss=f"{ss_ret['total_loss']:.4f}",
            )

            # Fitness distribution histogram
            n_candidates = [8, 6, 5, 4]
            n_bins = next((n for n in n_candidates if args.games % n == 0), 6)
            n_buckets = n_bins + 1
            step = 1.0 / n_bins
            bin_labels = [f"{(i * step):.3f}" for i in range(n_buckets)]
            hist = {lbl: 0 for lbl in bin_labels}
            for fv in fitness:
                bkt = f"{int(fv / step) * step:.3f}"
                hist[bkt] = hist.get(bkt, 0) + 1
            non_empty = {k: v for k, v in sorted(hist.items()) if v > 0}
            hist_str = " ".join(f"{k}:{v}" for k, v in non_empty.items())
            best_i = int(fitness.argmax())
            log.print_table(**{"ind_dist": hist_str, "best": f"#{best_i} {fitness[best_i]:.3f}"})

            # Update elite retention pool with top performers
            n_elite = max(1, int(args.pop_size ** 0.25))
            top_k_idx_full = np.argsort(fitness)[-n_elite:]
            new_elites = [params_list[i].copy() for i in reversed(top_k_idx_full)]
            combined = []
            for i in range(max(len(new_elites), len(elite_params))):
                if i < len(new_elites):
                    combined.append(new_elites[i])
                if i < len(elite_params):
                    combined.append(elite_params[i])
            elite_params = combined[:n_elite]

            # 8. CSV
            row = [
                gen,
                f"{fitness.max():.6f}",
                f"{fitness.mean():.6f}",
                f"{eval_time:.3f}",
                f"{total_time:.3f}",
                ss_ret["samples"],
                f"{ss_ret['class_loss']:.6f}",
                f"{ss_ret['map_loss']:.6f}",
                f"{ss_ret['total_loss']:.6f}",
            ]
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(row)

            # 9. Save checkpoint
            if args.save_every > 0 and (gen + 1) % args.save_every == 0:
                ckpt_path = out_dir / f"gen_{gen+1:04d}.pt"
                config_dict = {
                    "pop_size": args.pop_size,
                    "sigma": args.sigma,
                    "games": args.games,
                    "num_heads": args.num_heads,
                    "k": args.k,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "p_mutate": args.p_mutate,
                    "temperature": args.temperature,
                    "p_hold": args.p_hold,
                    "pos_noise_std": args.pos_noise_std,
                }
                save_checkpoint(ckpt_path, mean, model, gen + 1, config=config_dict)
                log.print(key="checkpoint", value=ckpt_path)

            # 10. Cleanup .npz files
            cleanup_gen_npz(bc_dir, gen)

            latest_result = {
                "generation": gen,
                "best_fitness": float(fitness.max()),
                "avg_fitness": float(fitness.mean()),
            }

    except KeyboardInterrupt:
        interrupted = True
        log.print("")
        log.print(key="interrupt", value="KeyboardInterrupt, saving checkpoint...")
    finally:
        pool.terminate()
        pool.join()

    # ── Final ─────────────────────────────────────────────────────
    if interrupted:
        ckpt_path = out_dir / f"interrupt_gen_{gen + 1:04d}.pt" if 'gen' in dir() else out_dir / "interrupt.pt"
        save_checkpoint(ckpt_path, mean, model,
                        gen + 1 if 'gen' in dir() else start_gen)
        log.print(key="interrupt_checkpoint", value=ckpt_path)
    else:
        save_checkpoint(out_dir / "final.pt", mean, model, args.generations)
        log.print(key="final_checkpoint", value=out_dir / "final.pt")

    log.separator("=")
    if latest_result is not None and not interrupted:
        log.print(key="best_fitness", value=f"{latest_result['best_fitness']:.4f}")
        log.print(key="avg_fitness", value=f"{latest_result['avg_fitness']:.4f}")
    log.print(key="history", value=csv_path)
    log.print("Done.")


if __name__ == "__main__":
    main()
