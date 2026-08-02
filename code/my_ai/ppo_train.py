"""
PPO training — policy gradient with clipped surrogate objective.

Usage:
    python -m my_ai.ppo_train \\
        --checkpoint path/to/checkpoint.pt \\
        --opp-checkpoint path/to/opp_checkpoint.pt \\
        --rollouts 200 --workers 12 --temperature 1.0 \\
        --generations 200 --save-every 20

On-policy: each generation collects fresh rollouts, then discards them.
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
import time
import multiprocessing as mp
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from my_ai.network import create_model, AntWarNetwork
from my_ai._eval_worker import _ppo_rollout_and_save, _eval_worker
from my_ai.leaderboard import Leaderboard, LeaderboardEntry
from my_ai.ga_ss_boilerplate import (
    init_csv,
    write_csv_row,
    get_device,
    setup_signal_handler,
)
from utils.logger import get_logger, redirect_stderr_to_log


# ═══════════════════════════════════════════════════════════════════
# 1. GAE
# ═══════════════════════════════════════════════════════════════════

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(λ) — compute advantages and return targets.

    Args:
        rewards: (T,) float32, r_0 .. r_{T-1}
        values:  (T,) float32, V(s_0) .. V(s_{T-1})
                 (terminal state V(s_T) = 0 implicitly)
        gamma:   discount factor
        lam:     GAE trace decay

    Returns:
        advantages: (T,) float32
        returns:    (T,) float32 = advantages + values
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ═══════════════════════════════════════════════════════════════════
# 2. Dataset
# ═══════════════════════════════════════════════════════════════════

class PPODataset(Dataset):
    """On-policy trajectory dataset for PPO update."""

    def __init__(self, trajectories: list[dict]):
        """trajectories: list of dicts from npz load."""
        boards, stats = [], []
        action_classes, head_logits_old = [], []
        advantages, returns = [], []
        for traj in trajectories:
            boards.append(traj["boards"])
            stats.append(traj["stats"])
            action_classes.append(traj["action_classes"])
            head_logits_old.append(traj["head_logits"])
            advantages.append(traj["advantages"])
            returns.append(traj["returns"])
        self.boards = torch.from_numpy(np.concatenate(boards, axis=0)).float()
        self.stats = torch.from_numpy(np.concatenate(stats, axis=0)).float()
        self.action_classes = torch.from_numpy(
            np.concatenate(action_classes, axis=0))
        self.head_logits_old = torch.from_numpy(
            np.concatenate(head_logits_old, axis=0)).float()
        self.advantages = torch.from_numpy(
            np.concatenate(advantages, axis=0)).float()
        self.returns = torch.from_numpy(
            np.concatenate(returns, axis=0)).float()

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return {
            "board": self.boards[idx],
            "stats": self.stats[idx],
            "action_class": self.action_classes[idx],    # (N_heads,)
            "head_logits_old": self.head_logits_old[idx], # (N_heads, 24)
            "advantage": self.advantages[idx],
            "return": self.returns[idx],
        }


# ═══════════════════════════════════════════════════════════════════
# 3. PPO helpers
# ═══════════════════════════════════════════════════════════════════

def zscore_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """z-score normalize per-row then scale by temperature (if > 0).

    Must match the decoder's sampling distribution (see ``decode_head``),
    so PPO's logπ and entropy use the SAME distribution as rollout.

    Uses ``correction=0`` so the std matches numpy's default ``std()``
    (ddof=0), which is what the decoder uses for sampling.
    """
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True, correction=0) + 1e-8
    out = (logits - mean) / std
    if temperature > 0:
        out = out / temperature
    return out


def normalized_log_probs(
    normalized_logits: torch.Tensor,  # (B, N_heads, 24) already z-scored
    action_classes: torch.Tensor,     # (B, N_heads)
    temperature: float = 1.0,
) -> torch.Tensor:                    # (B,) sum of logπ per head
    """logπ for ALREADY normalized logits (rollout stored z-scored form).

    The decoder sampled from ``softmax(z / T)``, so old logπ is simply
    ``log_softmax(z / T)`` — no re-normalization (re-normalizing float16
    quantized ±960 logits produced a different distribution than rollout).
    """
    log_probs_sum = 0.0
    for hi in range(normalized_logits.shape[1]):
        logits = normalized_logits[:, hi]  # (B, 24)
        if temperature > 0:
            logits = logits / temperature
        log_probs = F.log_softmax(logits, dim=-1)  # (B, 24)
        cls = action_classes[:, hi]  # (B,)
        log_probs_sum += log_probs.gather(1, cls.unsqueeze(1)).squeeze(1)
    return log_probs_sum


def compute_action_log_probs(
    head_logits: torch.Tensor,      # (B, N_heads, 24) raw model logits
    action_classes: torch.Tensor,   # (B, N_heads)
    temperature: float = 1.0,
) -> torch.Tensor:                   # (B,) sum of logπ per head
    """Compute log π(a|s) for CURRENT policy's raw logits.

    The behavioral policy samples from ``softmax(zscore(logits)/T)``, so
    the new policy's logπ uses the same z-scored distribution.
    """
    log_probs_sum = 0.0
    for hi in range(head_logits.shape[1]):
        logits = head_logits[:, hi]  # (B, 24)
        norm = zscore_logits(logits, temperature)  # (B, 24)
        log_probs = F.log_softmax(norm, dim=-1)  # (B, 24)
        cls = action_classes[:, hi]  # (B,)
        log_probs_sum += log_probs.gather(1, cls.unsqueeze(1)).squeeze(1)
    return log_probs_sum


def ppo_policy_loss(
    log_probs_new: torch.Tensor,   # (B,)
    log_probs_old: torch.Tensor,   # (B,)
    advantages: torch.Tensor,      # (B,)
    clip_epsilon: float = 0.2,
) -> torch.Tensor:                 # scalar
    """PPO clipped surrogate loss."""
    ratio = torch.exp(log_probs_new - log_probs_old)  # (B,)
    pg_loss1 = -ratio * advantages
    pg_loss2 = -torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return torch.mean(torch.max(pg_loss1, pg_loss2))


def entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute entropy of the policy distribution (z-scored, temperature-scaled).

    Args:
        logits: (B, 24) or (B, N_heads, 24)
    Returns:
        scalar entropy (averaged over batch)
    """
    norm = zscore_logits(logits, temperature)
    probs = F.softmax(norm, dim=-1)
    log_probs = F.log_softmax(norm, dim=-1)
    ent = -(probs * log_probs).sum(dim=-1)  # (B,) or (B, N_heads)
    return ent.mean()


# ═══════════════════════════════════════════════════════════════════
# 4. Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════

def load_model_params(path: str, device: torch.device,
                      num_heads: int = 3, no_bn: bool = False) -> AntWarNetwork:
    """Load a checkpoint and return the model with loaded weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(num_heads=num_heads, no_bn=no_bn)
    sd = ckpt.get("model_state")
    if sd is None:
        # Try loading from mean vector
        mean = ckpt.get("mean") or ckpt.get("params")
        if mean is None:
            raise ValueError(f"Checkpoint has neither model_state nor mean: {list(ckpt.keys())[:8]}")
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        model.set_parameters_from_vector(mean.cpu().numpy())
    else:
        # Handle BN folding
        has_bn = any("running_mean" in k for k in sd)
        if no_bn and has_bn:
            sd = AntWarNetwork.fold_bn_into_state_dict(sd)
        model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_params_vec(model: AntWarNetwork) -> np.ndarray:
    """Extract parameter vector from a model."""
    return model.get_parameters_as_vector()


def extract_bn_stats(model: AntWarNetwork) -> dict | None:
    """Extract BatchNorm running stats as numpy dict (or None if no BN).

    Needed because the parameter vector (``get_parameters_as_vector``)
    does NOT include BN running_mean/running_var buffers.
    """
    bn = {k: v.cpu().numpy() for k, v in model.state_dict().items()
          if "running_mean" in k or "running_var" in k}
    return bn if bn else None


# ═══════════════════════════════════════════════════════════════════
# 5. Parser
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PPO training for Ant-Game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoints
    p.add_argument("--checkpoint", required=True,
                   help="path to initial model checkpoint")
    p.add_argument("--opp-checkpoint", default=None,
                   help="path to opponent checkpoint (default: same as --checkpoint)")

    # Rollout
    p.add_argument("--rollouts", type=int, default=200,
                   help="games per PPO iteration")
    p.add_argument("--workers", type=int, default=8,
                   help="multiprocessing workers")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sampling temperature for exploration")
    p.add_argument("--num-heads", type=int, default=3,
                   help="number of policy heads")
    p.add_argument("--no-bn", action="store_true",
                   help="remove BatchNorm layers")
    p.add_argument("--no-intent-decoding", action="store_true",
                   help="disable auto-downgrade for super weapon gold shortage")

    # PPO
    p.add_argument("--generations", type=int, default=200,
                   help="number of PPO iterations")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="discount factor")
    p.add_argument("--lam", type=float, default=0.95,
                   help="GAE lambda")
    p.add_argument("--clip-epsilon", type=float, default=0.2,
                   help="PPO clip ratio")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="learning rate")
    p.add_argument("--ppo-epochs", type=int, default=2,
                   help="number of PPO epochs per iteration")
    p.add_argument("--batch-size", type=int, default=64,
                   help="mini-batch size")
    p.add_argument("--c-v", type=float, default=0.5,
                   help="value loss coefficient")
    p.add_argument("--c-e", type=float, default=0.01,
                   help="entropy bonus coefficient")
    p.add_argument("--max-grad-norm", type=float, default=0.5,
                   help="gradient clipping norm")

    # Leaderboard (adaptive opponent pool)
    p.add_argument("--no-lb", action="store_true",
                   help="disable Leaderboard (use fixed opponent)")
    p.add_argument("--lb-max-size", type=int, default=20,
                   help="leaderboard max entries")
    p.add_argument("--lb-threshold", type=float, default=0.6,
                   help="win rate threshold to insert into leaderboard")
    p.add_argument("--lb-games", type=int, default=6,
                   help="games per challenge match in add_candidate")

    # Logging / save
    p.add_argument("--save-every", type=int, default=20,
                   help="save checkpoint every N iterations")
    p.add_argument("--eval-every", type=int, default=10,
                   help="run argmax evaluation vs opponent every N iterations")
    p.add_argument("--eval-games", type=int, default=20,
                   help="number of evaluation games")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output directory (default: auto timestamp)")

    # Model variant
    p.add_argument("--small", action="store_true",
                   help="use small model variant")
    return p


# ═══════════════════════════════════════════════════════════════════
# 6. Evaluation (argmax, no temperature)
# ═══════════════════════════════════════════════════════════════════

def run_tasks(pool: mp.Pool, func, tasks: list, interrupted_ref=None) -> list:
    """Submit tasks via starmap_async and poll with 2s timeout.

    Windows: a blocking ``pool.starmap()`` cannot be interrupted by
    Ctrl+C, forcing a terminal kill.  Polling with timeout lets the
    main process notice the interrupt flag and return early.
    """
    if not tasks:
        return []
    async_result = pool.starmap_async(func, tasks)
    while True:
        if interrupted_ref is not None and interrupted_ref[0]:
            return []  # interrupted during pool wait → return empty
        try:
            return async_result.get(timeout=2)
        except mp.TimeoutError:
            continue


def evaluate(params: np.ndarray, opp_params: np.ndarray,
             workers: int, num_heads: int, no_bn: bool,
             games: int = 20, interrupted_ref=None,
             bn_stats: dict | None = None) -> float:
    """Evaluate params vs ONE opponent with argmax (eval_temperature=0).
    Returns win rate.
    """
    pool = mp.Pool(workers)
    tasks = []
    seeds = [42 + i for i in range(games)]
    for seed in seeds:
        tasks.append((params, opp_params, seed, num_heads,
                       None, 0, 0, 0.0, False, no_bn, bn_stats, 0.0, False))
    results = run_tasks(pool, _eval_worker, tasks, interrupted_ref)
    pool.close()
    pool.join()
    if not results:
        return 0.0
    scores = [r["score"] for r in results]
    return float(np.mean(scores))


def evaluate_vs_pool(params: np.ndarray, opponents: list[np.ndarray],
                     workers: int, num_heads: int, no_bn: bool,
                     games: int = 20, interrupted_ref=None,
                     bn_stats: dict | None = None) -> tuple[float, list[float]]:
    """Evaluate params vs a list of opponents (from leaderboard).

    Games are spread evenly across opponents.

    Returns:
        (overall win rate, per-opponent win rates)
    """
    pool = mp.Pool(workers)
    tasks = []
    n_opp = len(opponents)
    per_opp = max(1, games // max(n_opp, 1))
    for oi, opp_params in enumerate(opponents):
        for j in range(per_opp):
            seed = 42 + oi * 1000 + j
            tasks.append((params, opp_params, seed, num_heads,
                           None, 0, 0, 0.0, False, no_bn, bn_stats, 0.0, False))
    results = run_tasks(pool, _eval_worker, tasks, interrupted_ref)
    pool.close()
    pool.join()
    if not results:
        return 0.0, []
    scores = [r["score"] for r in results]
    overall = float(np.mean(scores))
    per_opp_wr = []
    for oi in range(n_opp):
        chunk = scores[oi * per_opp:(oi + 1) * per_opp]
        if chunk:
            per_opp_wr.append(float(np.mean(chunk)))
    return overall, per_opp_wr


# ═══════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()
    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device()

    # ── Setup output directory ──
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"training_history/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(exist_ok=True)

    log = get_logger(out_dir / "train.log")
    redirect_stderr_to_log(log)  # stderr → also to log file
    log.print(f"Output: {out_dir}")
    log.print(f"Device: {device}")
    log.print(f"Args: {vars(args)}")

    # ── Load model ──
    log.print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    start_iter = ckpt.get("iteration", 0)  # resume support: continue from saved iter
    model = load_model_params(args.checkpoint, device,
                              num_heads=args.num_heads, no_bn=args.no_bn)
    log.print(f"Model parameters: {model.count_parameters():,}")

    # ── Leaderboard (adaptive opponent pool) ──
    if not args.no_lb:
        leaderboard = Leaderboard(
            max_size=args.lb_max_size,
            param_count=model.count_parameters(),
            threshold=args.lb_threshold,
        )
        # Cold start: seed the pool with the initial checkpoint, so early
        # iterations have a real opponent to play against.
        ckpt_mean = ckpt.get("mean")
        if ckpt_mean is not None:
            raw = ckpt_mean
            seed_params = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
            leaderboard.entries.append(
                LeaderboardEntry(gen=-1, params=seed_params.astype(np.float32), score=0.5))
            log.print("Leaderboard cold-started with initial checkpoint.")
        # Resume: restore saved LB
        if "leaderboard" in ckpt:
            leaderboard.load_state_dict(ckpt["leaderboard"])
            log.print(f"Leaderboard restored: {len(leaderboard.entries)} entries")
        log.print(f"Leaderboard: max_size={args.lb_max_size} threshold={args.lb_threshold}")
    else:
        leaderboard = None
        # Fixed opponent fallback
        opp_path = args.opp_checkpoint or args.checkpoint
        log.print(f"Loading fixed opponent: {opp_path}")
        opp_model = load_model_params(opp_path, "cpu",
                                      num_heads=args.num_heads, no_bn=args.no_bn)
        opp_params = extract_params_vec(opp_model)
        log.print(f"Opponent parameters: {len(opp_params):,}")

    # ── Training state ──
    params = extract_params_vec(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if start_iter > 0:
        log.print(f"Resuming from iteration {start_iter} "
                  f"(previous run's optimizer state not restored).")

    # ── CSV ──
    csv_path = out_dir / "history.csv"
    if csv_path.exists() and start_iter > 0:
        log.print(f"CSV exists — appending (resume).")
    else:
        init_csv(csv_path, ["iter", "avg_reward", "win_rate",
                 "value_loss", "policy_loss", "entropy",
                 "approx_kl", "explained_var"])

    # ── Signal handler (for graceful Ctrl+C) ──
    interrupted = [False]
    setup_signal_handler(interrupted, log)

    # ── Process pool (reused across iter) ──
    pool = mp.Pool(args.workers)

    for it in range(start_iter, args.generations):
        if interrupted[0]:
            log.print("Interrupted — stopping.")
            break
        iter_start = time.time()

        # ── 7a. Rollout ──
        # Clear previous rollouts
        for f in rollout_dir.glob("ppo_*.npz"):
            f.unlink(missing_ok=True)

        # Extract BN stats (only relevant when model uses BatchNorm)
        bn_stats = None if args.no_bn else extract_bn_stats(model)

        # Select opponents for this iteration's rollout
        if leaderboard is not None and leaderboard.entries:
            k = min(args.lb_max_size, max(1, args.rollouts // 2))
            opps = leaderboard.get_opponents_adaptive(k=k)
            opp_list = [o["params"] for o in opps]
            if not opp_list:
                opp_list = [params.copy()]  # self-play fallback
        else:
            opp_list = [opp_params.copy()]  # fixed opponent / cold start

        tasks = []
        for i in range(args.rollouts):
            seed = 42 + i + it * 10000
            opp = opp_list[i % len(opp_list)]
            tasks.append((
                params.copy(), opp.copy(), seed,
                str(rollout_dir), args.temperature,
                args.num_heads, args.small, args.no_bn,
                bn_stats,
                not args.no_intent_decoding,
            ))

        results = run_tasks(pool, _ppo_rollout_and_save, tasks, interrupted)
        rollout_time = time.time() - iter_start
        if interrupted[0]:
            log.print("Interrupted during rollout — stopping.")
            break

        # ── 7b. Load trajectories ──
        npz_paths = sorted(rollout_dir.glob("ppo_*.npz"))
        trajectories = []
        for p in npz_paths:
            data = dict(np.load(p))
            trajectories.append(data)

        total_steps = sum(len(t["rewards"]) for t in trajectories)
        mean_reward = float(np.mean([t["rewards"].mean() for t in trajectories]))

        # ── 7c. Compute GAE ──
        for traj in trajectories:
            adv, ret = compute_gae(
                traj["rewards"], traj["values"],
                gamma=args.gamma, lam=args.lam)
            traj["advantages"] = adv
            traj["returns"] = ret

        # Normalize advantages
        all_adv = np.concatenate([t["advantages"] for t in trajectories])
        adv_mean = all_adv.mean()
        adv_std = all_adv.std() + 1e-8
        for traj in trajectories:
            traj["advantages"] = (traj["advantages"] - adv_mean) / adv_std

        # ── 7d. PPO update ──
        dataset = PPODataset(trajectories)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        approx_kl = 0.0
        n_batches = 0

        # Pre-compute old log probs (used across all epochs)
        old_log_probs_list = []
        with torch.no_grad():
            for batch in loader:
                log_probs_old = normalized_log_probs(
                    batch["head_logits_old"].to(device),
                    batch["action_class"].to(device),
                    temperature=args.temperature)
                old_log_probs_list.append(log_probs_old.cpu())
        old_log_probs_all = torch.cat(old_log_probs_list, dim=0)

        for epoch in range(args.ppo_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_kl = 0.0
            batch_count = 0

            # Rebuild loader to reshuffle
            epoch_loader = DataLoader(dataset, batch_size=args.batch_size,
                                      shuffle=True)
            batch_offset = 0
            for batch in epoch_loader:
                board = batch["board"].to(device)          # (B, 28, 19, 19)
                stats = batch["stats"].to(device)          # (B, 42)
                sampled_class = batch["action_class"].to(device)   # (B, N_heads)
                adv = batch["advantage"].to(device)        # (B,)
                ret = batch["return"].to(device)           # (B,)

                B = board.shape[0]
                old_logp = old_log_probs_all[batch_offset:batch_offset + B].to(device)
                batch_offset += B

                # Forward
                output = model(board, stats)
                new_logits = torch.stack([
                    output[f"head{hi+1}_logits"]
                    for hi in range(args.num_heads)
                ], dim=1)  # (B, N_heads, 24)
                new_value = output["value"].squeeze(-1)  # (B,)

                # Log probs (z-score normalized + temperature, matching rollout)
                log_probs_new = compute_action_log_probs(
                    new_logits, sampled_class, temperature=args.temperature)

                # Policy loss (clipped)
                pg_loss = ppo_policy_loss(log_probs_new, old_logp, adv,
                                          clip_epsilon=args.clip_epsilon)

                # Value loss
                vf_loss = F.mse_loss(new_value, ret)

                # Entropy bonus
                ent = entropy_from_logits(new_logits, temperature=args.temperature)

                # Total
                loss = pg_loss + args.c_v * vf_loss - args.c_e * ent

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                # Stats
                with torch.no_grad():
                    ratio = torch.exp(log_probs_new - old_logp)
                    kl = (ratio - 1 - log_probs_new + old_logp).mean().item()

                epoch_policy_loss += pg_loss.item()
                epoch_value_loss += vf_loss.item()
                epoch_entropy += ent.item()
                epoch_kl += kl
                batch_count += 1

            n_batches += batch_count
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy += epoch_entropy
            approx_kl += epoch_kl

        avg_policy_loss = total_policy_loss / (args.ppo_epochs * max(n_batches // args.ppo_epochs, 1))
        avg_value_loss = total_value_loss / (args.ppo_epochs * max(n_batches // args.ppo_epochs, 1))
        avg_entropy = total_entropy / (args.ppo_epochs * max(n_batches // args.ppo_epochs, 1))
        avg_kl = approx_kl / n_batches

        # Extract new params
        params = extract_params_vec(model.cpu())
        model.to(device)
        model.train()

        update_time = time.time() - iter_start - rollout_time

        # ── 7e. Evaluate + Leaderboard challenge (on eval steps) ──
        win_rate = 0.0
        lb_added = False
        if it % args.eval_every == 0 or it == args.generations - 1:
            model.eval()
            if leaderboard is not None and leaderboard.entries:
                # Evaluate vs sampled LB opponents (track gen for lambda updates)
                k = min(args.lb_max_size, max(1, args.eval_games // 2))
                opps = leaderboard.get_opponents_adaptive(k=k)
                eval_opps = [o["params"] for o in opps]
                opp_gens = [o.get("gen") for o in opps]
                win_rate, per_opp_wr = evaluate_vs_pool(
                    params, eval_opps,
                    workers=min(args.workers, 8),
                    num_heads=args.num_heads,
                    no_bn=args.no_bn,
                    games=args.eval_games,
                    interrupted_ref=interrupted,
                    bn_stats=bn_stats,
                )
                # Adaptive lambda update: keep each entry sampled ~50/50
                wr_by_gen = {}
                for g, wr in zip(opp_gens, per_opp_wr):
                    if g is not None:
                        wr_by_gen[g] = wr
                leaderboard.update_lambdas(wr_by_gen)

                # Challenge ladder: current model tries to enter the pool
                def _vs_lb(me, opponent):
                    tasks = [
                        (me, opponent, 42 + 999999 + it * 100 + s,
                         args.num_heads, None, 0, -1,
                         0.0, args.small, args.no_bn, bn_stats, 0.0, False)
                        for s in range(args.lb_games)
                    ]
                    res = run_tasks(pool, _eval_worker, tasks, interrupted)
                    scores = [r["score"] if isinstance(r, dict) else r for r in res]
                    return float(np.mean(scores)) if scores else 0.0

                lb_added = leaderboard.add_candidate(it, params.copy(), match_fn=_vs_lb)
            else:
                # Fixed opponent evaluation
                win_rate = evaluate(
                    params, opp_params,
                    workers=min(args.workers, 8),
                    num_heads=args.num_heads,
                    no_bn=args.no_bn,
                    games=args.eval_games,
                    interrupted_ref=interrupted,
                    bn_stats=bn_stats,
                )
            model.train()
            log.print_table(iter=it, win_rate=f"{win_rate:.3f}",
                            eval_games=args.eval_games, lb_added=lb_added)

        # ── 7f. Log + CSV (single row, always includes win_rate) ──
        log.print_table(
            iter=it,
            reward=f"{mean_reward:.3f}",
            win_rate=f"{win_rate:.3f}",
            steps=total_steps,
            policy_loss=f"{avg_policy_loss:.4f}",
            value_loss=f"{avg_value_loss:.4f}",
            entropy=f"{avg_entropy:.4f}",
            kl=f"{avg_kl:.4f}",
            rollout_s=f"{rollout_time:.0f}",
            update_s=f"{update_time:.0f}",
        )
        write_csv_row(csv_path, [
            it, f"{mean_reward:.4f}", f"{win_rate:.4f}",
            f"{avg_value_loss:.4f}", f"{avg_policy_loss:.4f}",
            f"{avg_entropy:.4f}", f"{avg_kl:.4f}", "0.0",
        ])

        # ── 7g. Save checkpoint ──
        if it % args.save_every == 0 or it == args.generations - 1:
            ckpt_path = out_dir / f"ppo_iter_{it:04d}.pt"
            torch.save({
                "model_state": model.cpu().state_dict(),
                "mean": params,
                "iteration": it,
                "num_heads": args.num_heads,
                "no_bn": args.no_bn,
                "config": vars(args),
                "leaderboard": leaderboard.state_dict() if leaderboard else None,
            }, ckpt_path)
            model.to(device)
            log.print(f"Saved: {ckpt_path}")

        # ── 7h. Interrupt checkpoint (always save, even if not save-every) ──
        if interrupted[0]:
            ckpt_path = out_dir / f"ppo_interrupt_{it:04d}.pt"
            torch.save({
                "model_state": model.cpu().state_dict(),
                "mean": params,
                "iteration": it,
                "num_heads": args.num_heads,
                "no_bn": args.no_bn,
                "config": vars(args),
                "leaderboard": leaderboard.state_dict() if leaderboard else None,
            }, ckpt_path)
            log.print(f"Interrupt checkpoint: {ckpt_path}")
            break

    pool.terminate()
    pool.join()
    log.print("PPO training complete.")


if __name__ == "__main__":
    main()
