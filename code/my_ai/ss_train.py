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
from my_ai.leaderboard import Leaderboard
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

    def __init__(self, npz_paths, p_hold=1.0):
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

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        board = torch.from_numpy(self.board[idx]).float()         # (28, 19, 19)
        stats = torch.from_numpy(self.stats[idx]).float()         # (42,)
        cls_label = torch.from_numpy(self.class_label[idx].copy())  # (N_heads,)
        action_map = torch.from_numpy(self.action_map[idx].copy()).float()  # (NUM_CLASSES, 19, 19)

        return {
            "board": board,
            "stats": stats,
            "class_label": cls_label,
            "action_map": action_map,
            "head_logits": torch.from_numpy(self.head_logits[idx]).float(),  # (N_heads, 24)
        }


def ss_supervised_update(model, dataset, device, epochs=3, lr=1e-3, batch_size=64,
                         lambda_class=1.0, lambda_map=1.0, log=None):
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

    for epoch in range(epochs):
        epoch_cls = 0.0
        epoch_map = 0.0
        epoch_tot = 0.0
        epoch_n = 0
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

            epoch_cls += cls_loss.item()
            epoch_map += map_loss.item()
            epoch_tot += loss.item()
            epoch_n += 1

        if log and epoch_n > 0:
            log.print(key="epoch",
                      value=f"{epoch+1}/{epochs} cls={epoch_cls/epoch_n:.4f} "
                            f"map={epoch_map/epoch_n:.4f} tot={epoch_tot/epoch_n:.4f}")

    model.cpu()
    model.eval()
    return {
        "class_loss": total_cls / n_batches,
        "map_loss": total_map / n_batches,
        "total_loss": total / n_batches,
        "samples": len(dataset),
    }


def bc_mutate_population(mean_model, dataset, device, n_individuals,
                          p_mutate=0.1, temperature=3.0,
                          mutation_step=0.01, sigma=0.0002,
                          n_dirs=20, log=None):
    """Generate N individuals via linear combination of gradient directions.

    Phase 1: Pre-compute ``n_dirs`` gradient basis directions (each from a
    different batch × mutation seed). Phase 2: For each individual, sample
    random weights and linearly combine basis directions as the perturbation.

    Returns:
        list of np.ndarray parameter vectors (len=n_individuals).
    """
    if len(dataset) == 0:
        return []

    N = mean_model.num_heads
    mean_params = mean_model.get_parameters_as_vector()
    model_template = create_model(num_heads=N)

    # Phase 1: Pre-compute gradient basis
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    all_batches = list(loader)
    grad_basis = []

    for j in range(min(n_dirs, len(all_batches))):
        batch = all_batches[j % len(all_batches)]
        board = batch["board"].to(device)
        stats = batch["stats"].to(device)
        cls_label_ref = batch["class_label"].to(device)
        head_logits = batch["head_logits"].to(device)

        model_template.load_state_dict(mean_model.state_dict())
        model_template.train()
        model_template.to(device)

        # Mutate labels
        cls_label = cls_label_ref.clone()
        torch.manual_seed(j)
        B, NH = cls_label.shape
        for hi in range(NH):
            mask = torch.rand(B) < p_mutate
            if not mask.any():
                continue
            logits_i = head_logits[mask, hi, :]
            logits_i = torch.clamp(logits_i, -50.0, 50.0)
            # Category rebalancing: each super-category gets ~equal total weight
            # Tower-intent classes (0-15, 16 classes) → weight 1.0
            # Demolish (16, 1 class) → weight 2.5
            # Super weapons (17-20, 4 classes) → weight 2.5
            # Base upgrades (21-22, 2 classes) → weight 2.5
            # HOLD (23, 1 class) → weight 2.5
            w = torch.ones(24, device=logits_i.device, dtype=logits_i.dtype)
            w[16] = 2.5
            w[17:21] = 2.5
            w[21:23] = 2.5
            w[23] = 2.5
            logits_i = logits_i + torch.log(w) * temperature
            probs = F.softmax(logits_i / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            cls_label[mask, hi] = sampled

        # Freeze encoder: gradient only for policy_heads + action_map_conv
        for name, param in model_template.named_parameters():
            param.requires_grad = name.startswith("policy_heads") or "action_map_conv" in name

        # Forward + backward
        model_template.zero_grad()
        output = model_template(board, stats)
        cls_loss = 0.0
        for hi in range(NH):
            cls_loss += F.cross_entropy(output[f"head{hi+1}_logits"], cls_label[:, hi])
        cls_loss /= NH
        cls_loss.backward()

        # Extract gradient with per-head normalization
        grad = _extract_grad_per_head(model_template, NH)
        grad_basis.append(grad)

        # Restore requires_grad for next iteration
        for param in model_template.parameters():
            param.requires_grad = True
        model_template.zero_grad()

    n_dirs_actual = len(grad_basis)
    if n_dirs_actual == 0:
        return []

    # Phase 2: Generate individuals as linear combinations
    # Each direction is per-head normalized → each head changes equally
    # mutation_step ~ 100.0 gives per-weight change ≈ 0.9 (enough to flip argmax)
    params_list = []
    rng = np.random.RandomState()
    rng.seed(42)

    for i in range(n_individuals):
        w = rng.randn(n_dirs_actual).astype(np.float32) / np.sqrt(n_dirs_actual)

        delta = np.zeros_like(mean_params, dtype=np.float32)
        for j in range(n_dirs_actual):
            delta += w[j] * grad_basis[j]

        noise = rng.randn(*mean_params.shape).astype(np.float32) * sigma
        params_list.append(mean_params + mutation_step * delta + noise)

        if log and (i + 1) % max(1, n_individuals // 5) == 0:
            log.print(key="gmut", value=f"{i+1}/{n_individuals}")

    model_template.cpu()
    return params_list


def _extract_grad_vector(model):
    """Flatten all parameter gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1).cpu().numpy().astype(np.float32))
        else:
            grads.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(grads)


def get_head_offsets(model, num_heads):
    """For each policy head, return (weight_start, weight_size, bias_start, bias_size)
    in the flattened parameter vector.
    """
    offsets = [(0, 0, 0, 0)] * num_heads
    idx = 0
    for name, p in model.named_parameters():
        size = p.numel()
        for hi in range(num_heads):
            if name == f"policy_heads.{hi}.weight":
                offsets[hi] = (idx, size, None, None)
            if name == f"policy_heads.{hi}.bias":
                w_start, w_size, _, _ = offsets[hi]
                offsets[hi] = (w_start, w_size, idx, size)
        idx += size
    return offsets  # list of (w_start, w_size, b_start, b_size)


def _extract_grad_per_head(model, num_heads):
    """Extract gradient with per-head normalization.

    Uses parameter NAMES to identify head-specific parameters.
    Each head's gradient (weight+bias) is normalized independently.
    Returns a full-size gradient vector (same shape as get_parameters_as_vector())
    with encoder and action_map_conv gradients zeroed out.
    """
    full = _extract_grad_vector(model)
    result = np.zeros_like(full)

    # Collect head parameters by name
    named_params = dict(model.named_parameters())
    idx = 0
    for name, p in model.named_parameters():
        size = p.numel()
        # Check if this is a head-specific parameter using NAME
        is_head_param = False
        for hi in range(num_heads):
            if name.startswith(f"policy_heads.{hi}."):
                is_head_param = True
                break
        if is_head_param and p.grad is not None:
            g = p.grad.detach().view(-1).cpu().numpy().astype(np.float32)
            gn = np.linalg.norm(g)
            if gn > 1e-8:
                result[idx:idx + size] = g / gn
        # Everything else stays zero (encoder, action_map_conv)
        idx += size
    return result


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


def save_checkpoint(path, mean, model, generation, config=None, leaderboard=None):
    """Save training state to checkpoint file."""
    data = {
        "mean": torch.from_numpy(mean),
        "model_state": model.state_dict(),
        "generation": generation,
        "num_heads": model.num_heads,
        # Compat shim: es_train tools expect top2_params
        "top2_params": [torch.from_numpy(mean.copy())],
        "top2_scores": [1.0],
    }
    if config is not None:
        data["config"] = config
    if leaderboard is not None:
        data["leaderboard"] = leaderboard.state_dict()
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


def build_eval_args(params_list, opp_params_list, pop_size, games, num_heads, bc_dir, gen, seed, only_idx=None, seed_offset=0):
    """Build argument tuples for _eval_worker.
    
    k_per_ind is derived from len(opp_params_list) so that Phase 2
    always uses valid indices regardless of games vs data_games.
    """
    k_per_ind = len(opp_params_list)
    all_args = []
    indices = only_idx if only_idx is not None else range(pop_size)
    for idx in indices:
        for k_idx in range(k_per_ind):
            base_seed = (seed + gen * pop_size * games
                         + idx * games + k_idx * 2 + seed_offset)
            all_args.append((
                params_list[idx], opp_params_list[k_idx], base_seed,
                num_heads,
                bc_dir, gen, idx,
            ))
            all_args.append((
                params_list[idx], opp_params_list[k_idx], base_seed + 1,
                num_heads,
                bc_dir, gen, idx,
            ))
    return all_args


def run_eval(pool, all_args):
    """Submit and collect evaluation results, handling interrupts."""
    if not all_args:
        return []
    async_result = pool.starmap_async(_eval_worker, all_args)
    while True:
        try:
            return async_result.get(timeout=2)
        except mp.TimeoutError:
            # Check for interrupt (handled via global)
            continue
        except (mp.context.BrokenProcessPool, OSError, ValueError):
            raise


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
        float_keys = {"sigma", "lr", "p_mutate", "temperature", "pos_noise_std", "p_hold", "swap_p"}
        int_keys = {"generations", "pop_size", "games", "workers", "k", "data_games", "lb_inject"}
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
    parser.add_argument("--sigma", type=float, default=0.0002)
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
    parser.add_argument("--p-hold", type=float, default=1.0)
    parser.add_argument("--pos-noise-std", type=float, default=0.1)
    parser.add_argument("--mutation-step", type=float, default=50.0,
                        help="gradient-guided mutation step size (per-head normalized)")
    parser.add_argument("--n-grad-dirs", type=int, default=100,
                        help="number of gradient basis directions for mutation")
    parser.add_argument("--swap-ratio", type=float, default=0.3,
                        help="fraction of population generated by head swapping")
    parser.add_argument("--swap-p", type=float, default=0.5,
                        help="probability per individual to undergo head swap")
    parser.add_argument("--lb-inject", type=int, default=0,
                        help="number of strongest leaderboard entries injected into population")
    parser.add_argument("--no-lb", action="store_true",
                        help="disable Leaderboard opponent selection (use random from population)")
    parser.add_argument("--save-data", action="store_true", dest="save_data", default=False,
                        help="single-phase eval: save all individuals' data to disk")
    parser.add_argument("--data-games", type=int, default=6,
                        help="games per top-K individual for data collection (2-phase eval)")
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
    from utils.logger import redirect_stderr_to_log
    redirect_stderr_to_log(log)
    csv_path = out_dir / "history.csv"

    # Save config
    with open(out_dir / "config.txt", "w") as f:
        for key, val in vars(args).items():
            f.write(f"{key}={val}\n")
        f.write(f"timestamp={ts}\n")

    # ── Model ─────────────────────────────────────────────────────
    model = create_model(num_heads=args.num_heads)
    param_count = model.count_parameters()

    # ── Leaderboard ──────────────────────────────────────────────
    if not args.no_lb:
        leaderboard = Leaderboard(max_size=20, param_count=param_count)
        log.print(key="leaderboard", value=f"enabled (max_size=20)")
    else:
        leaderboard = None
        log.print(key="leaderboard", value="disabled (random opponents)")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        start_gen = ckpt.get("generation", 0)
        if leaderboard is not None and "leaderboard" in ckpt:
            leaderboard.load_state_dict(ckpt["leaderboard"])
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
    log.print(key="leaderboard", value="enabled (max_size=20)" if not args.no_lb else "disabled")
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
        bc_dataset = None
        elite_saved: list[np.ndarray] = []  # cross-generation elite params
        for gen in range(start_gen, args.generations):
            if interrupted:
                break

            # PAUSE check + config reload
            while pause_file.read_text(encoding="utf-8").strip().lower() == "pause" and not interrupted:
                time.sleep(2)
            if interrupted:
                break
            if reload_config(str(out_dir / "config.txt"), args) and log:
                float_keys = {"sigma", "lr", "p_mutate", "temperature", "pos_noise_std", "p_hold", "swap_p"}
                int_keys = {"generations", "pop_size", "games", "workers", "k", "data_games", "lb_inject"}
                changed_keys = []
                for key in sorted(float_keys | int_keys):
                    val = getattr(args, key.replace("-", "_"), None)
                    if val is not None:
                        changed_keys.append(f"{key}={val}")
                log.print(key="config_reload",
                          value=" ".join(changed_keys))

            t0 = time.time()

            # 1. Generate population (BC mutation or fallback)
            assert args.pop_size > 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if bc_dataset is not None and len(bc_dataset) > 0:
                params_list = bc_mutate_population(
                    model, bc_dataset, device, args.pop_size,
                    p_mutate=args.p_mutate, temperature=args.temperature,
                    mutation_step=args.mutation_step, sigma=args.sigma,
                    n_dirs=args.n_grad_dirs, log=log,
                )
            else:
                # Fallback (gen=0 only): tiny parameter noise
                noise = rng.randn(args.pop_size, param_count).astype(np.float32)
                params_list = [mean + args.sigma * n for n in noise]

            # Head swap mutation: swap two heads within selected individuals
            if bc_dataset is not None and len(bc_dataset) > 0 and args.swap_ratio > 0 and args.pop_size >= 2 and len(params_list) >= 2:
                n_swap = max(1, int(args.pop_size * args.swap_ratio))
                n_swap = min(n_swap, len(params_list) // 2)  # safety
                
                # Pre-compute head offsets once
                head_offsets = get_head_offsets(model, args.num_heads)
                
                rng_swap = np.random.RandomState()
                for i in range(n_swap):
                    idx = rng_swap.randint(len(params_list))
                    ha, hb = rng_swap.choice(args.num_heads, size=2, replace=False)
                    ha, hb = int(ha), int(hb)
                    
                    child = params_list[idx].copy().astype(np.float32)
                    
                    # Swap weights
                    wa_start, wa_size, ba_start, ba_size = head_offsets[ha]
                    wb_start, wb_size, bb_start, bb_size = head_offsets[hb]
                    child[wa_start:wa_start + wa_size], child[wb_start:wb_start + wb_size] = \
                        child[wb_start:wb_start + wb_size].copy(), child[wa_start:wa_start + wa_size].copy()
                    child[ba_start:ba_start + ba_size], child[bb_start:bb_start + bb_size] = \
                        child[bb_start:bb_start + bb_size].copy(), child[ba_start:ba_start + ba_size].copy()
                    
                    # Replace end of list
                    params_list[-(i + 1)] = child
                    
                if log:
                    log.print(key="head_swap",
                              value=f"swapped {n_swap} individuals, "
                                    f"pop={len(params_list)} p={args.swap_ratio:.2f}")

            # Elite retention: inject top past elites, replace weakest positions
            n_elite = max(1, int(args.pop_size ** 0.25))
            for i in range(min(n_elite, len(elite_saved))):
                params_list[-(i + 1)] = elite_saved[i].copy()

            # LB inject: pull strongest entries from leaderboard into population
            if leaderboard is not None and args.lb_inject > 0:
                lb_entries = leaderboard.get_ranked_entries()
                n_lb = min(args.lb_inject, len(lb_entries))
                offset = n_elite  # inject after elite retention
                for i in range(n_lb):
                    pos = -(i + 1 + offset)
                    if abs(pos) <= len(params_list):
                        params_list[pos] = lb_entries[i]["params"].copy()
                if log and n_lb > 0:
                    log.print(key="lb_inject",
                              value=f"{n_lb} entries from leaderboard injected")

            # Mean injection: ensure the mean policy is always in the population
            params_list[-1] = mean.copy().astype(np.float32)

            # 2. Select opponents
            k_per_ind = args.games // 2
            if leaderboard is not None:
                opp_list = leaderboard.get_opponents(k=k_per_ind)
                if opp_list:
                    opp_params_list = [entry["params"] for entry in opp_list]
                else:
                    # Cold start: pool empty → random from population
                    opp_indices = select_opponents(args.pop_size, args.games, rng)
                    opp_params_list = [params_list[i] for i in opp_indices]
            else:
                opp_indices = select_opponents(args.pop_size, args.games, rng)
                opp_params_list = [params_list[i] for i in opp_indices]

            # 3. Build all evaluation tasks
            eval_start = time.time()

            if args.save_data:
                # Single phase: save all data
                all_args = build_eval_args(
                    params_list, opp_params_list, args.pop_size, args.games,
                    args.num_heads, str(bc_dir), gen, args.seed,
                    only_idx=None, seed_offset=0)
                all_results = run_eval(pool, all_args)
                if interrupted:
                    break
                print()  # newline after eval dots
                scores = np.array([r["score"] if isinstance(r, dict) else r
                                   for r in all_results], dtype=np.float32)
                fitness = np.mean(scores.reshape(args.pop_size, args.games), axis=1)
                eval_time = time.time() - eval_start
            else:
                # Phase 1: evaluate all individuals, no data saving
                all_args_1 = build_eval_args(
                    params_list, opp_params_list, args.pop_size, args.games,
                    args.num_heads, None, gen, args.seed,
                    only_idx=None, seed_offset=0)
                all_results_1 = run_eval(pool, all_args_1)
                if interrupted:
                    break
                print()  # newline after eval dots
                scores_1 = np.array([r["score"] if isinstance(r, dict) else r
                                     for r in all_results_1], dtype=np.float32)
                fitness = np.mean(scores_1.reshape(args.pop_size, args.games), axis=1)
                eval_time = time.time() - eval_start

                # Select top-K
                top_k_idx = sorted(range(len(fitness)), key=lambda i: -fitness[i])[:args.k]

                # Phase 2: select opponents (same weighted sampling as Phase 1)
                n_opp_phase2 = args.data_games // 2
                if leaderboard is not None:
                    opp_list_2 = leaderboard.get_opponents(k=n_opp_phase2)
                    if opp_list_2:
                        opp_params_phase2 = [entry["params"] for entry in opp_list_2]
                    else:
                        opp_indices = select_opponents(args.pop_size, args.data_games, rng)
                        opp_params_phase2 = [params_list[i] for i in opp_indices]
                else:
                    opp_indices = select_opponents(args.pop_size, args.data_games, rng)
                    opp_params_phase2 = [params_list[i] for i in opp_indices]
                n_games_phase2 = len(top_k_idx) * len(opp_params_phase2) * 2
                log.print(key="collect_data",
                          value=f"k={len(top_k_idx)} × {len(opp_params_phase2)} opponents × 2 = {n_games_phase2} games")
                all_args_2 = build_eval_args(
                    params_list, opp_params_phase2, args.pop_size, args.data_games,
                    args.num_heads, str(bc_dir), gen, args.seed,
                    only_idx=top_k_idx, seed_offset=10000)
                run_eval(pool, all_args_2)
                if interrupted:
                    break
                print()

            # 6. Select top-K and train
            top_k_idx = sorted(range(len(fitness)), key=lambda i: -fitness[i])[:args.k]
            npz_paths = collect_npz(bc_dir, top_k_idx, gen)

            # Update elite_saved: merge top-N with saved, preserve old if still strong
            n_elite = max(1, int(args.pop_size ** 0.25))
            top_k_idx = sorted(range(len(fitness)), key=lambda i: -fitness[i])[:n_elite]
            new_elites = [params_list[i].copy() for i in reversed(top_k_idx)]
            combined = []
            for i in range(max(len(new_elites), len(elite_saved))):
                if i < len(new_elites):
                    combined.append(new_elites[i])
                if i < len(elite_saved):
                    combined.append(elite_saved[i])
            elite_saved = combined[:n_elite]

            ss_ret = {"samples": 0, "class_loss": 0.0, "map_loss": 0.0, "total_loss": 0.0}
            if npz_paths:
                ds = SSDataset(
                    npz_paths,
                    p_hold=args.p_hold,
                )
                if len(ds) > 0:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ss_ret = ss_supervised_update(
                        model, ds, device=device,
                        epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        log=log,
                    )
                    mean = model.get_parameters_as_vector()
                    bc_dataset = ds
                else:
                    log.print(key="ss", value="all-HOLD filtered, no valid samples")
            else:
                log.print(key="ss", value="no data files for top-K")

            # Update Leaderboard
            if leaderboard is not None:
                print("")
                log.print(key="lb", value=f"challenging (pool={len(leaderboard.entries)})")

                def _vs_strongest(me, opponent):
                    match_tasks = [(me, opponent, args.seed + 999999 + gen * 100 + s,
                                    args.num_heads, None, gen, -1)
                                   for s in range(args.games)]
                    scores = [r["score"] if isinstance(r, dict) else r
                              for r in pool.starmap(_eval_worker, match_tasks)]
                    wr = float(np.mean(scores))
                    log.print(key="lb_wr", value=f"{wr:.3f} (threshold={leaderboard.threshold})")
                    return wr
                added = leaderboard.add_candidate(gen, mean.copy(),
                                                   match_fn=_vs_strongest)
                log.print(key="lb", value=f"{'added' if added else 'rejected'}")

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
                save_checkpoint(ckpt_path, mean, model, gen + 1, config=config_dict, leaderboard=leaderboard)
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
                        gen + 1 if 'gen' in dir() else start_gen,
                        leaderboard=leaderboard)
        log.print(key="interrupt_checkpoint", value=ckpt_path)
    else:
        save_checkpoint(out_dir / "final.pt", mean, model, args.generations,
                        leaderboard=leaderboard)
        log.print(key="final_checkpoint", value=out_dir / "final.pt")

    log.separator("=")
    if latest_result is not None and not interrupted:
        log.print(key="best_fitness", value=f"{latest_result['best_fitness']:.4f}")
        log.print(key="avg_fitness", value=f"{latest_result['avg_fitness']:.4f}")
    log.print(key="history", value=csv_path)
    log.print("Done.")


if __name__ == "__main__":
    main()
