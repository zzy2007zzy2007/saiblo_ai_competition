"""Minimal Evolution Strategies trainer for Ant-Game AI (self-play).

Core algorithm (OpenAI-ES style):
  1. Sample noise -> create perturbed models: theta +/- sigma*eps
  2. Self-play: each individual plays against random population member
     (opponent selected by select_opponents() — extensible to elite pools)
  3. Fitness = win rate, shaped via rank-based normalization
  4. Gradient estimate: g = 1/(N*sigma) * sum(f_i * eps_i)
  5. Update: theta <- theta + lr * g
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
import sys
import time
import multiprocessing as mp
from datetime import datetime

import numpy as np
import torch

from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from utils.logger import get_logger


def select_opponents(
    population_size: int,
    games_per_individual: int,
    rng: np.random.RandomState,
) -> list[list[int]]:
    """Select opponent indices for each individual.

    Returns `games_per_individual // 2` opponents per individual.
    Each opponent is played twice: once as first player, once as second player
    (handled in step() by creating two tasks per opponent with alternating parity).

    Returns:
        opp_indices[i][k]: opponent index for individual i, pair k.
            Always != i (never plays against self).  k < games_per_individual // 2.
    """
    assert games_per_individual % 2 == 0, "games_per_individual must be even (for fair first/second player swap)"
    n_pairs = games_per_individual // 2
    opp_indices: list[list[int]] = []
    for i in range(population_size):
        candidates = [j for j in range(population_size) if j != i]
        opps = rng.choice(candidates, size=n_pairs, replace=False)
        opp_indices.append([int(o) for o in opps])
    return opp_indices


def _eval_worker(
    params_flat: np.ndarray,
    opp_params_flat: np.ndarray,
    seed: int,
    single_head: bool = False,
    synthetic_target: np.ndarray | None = None,
) -> dict:
    """Run one match: params vs opponent params.

    If synthetic_target is provided, fitness = distance to target (no game).
    Otherwise, runs a full Ant-Game match.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import sys
    from pathlib import Path
    _RP = Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = Path(__file__).resolve().parents[1]
    for p in (_RP, _CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import torch
    torch.set_num_threads(1)

    if synthetic_target is not None:
        # ── Synthetic mode ──────────────────────────────────────────
        target_exists = synthetic_target != 0
        sub_flat = params_flat[target_exists]
        sub_target = synthetic_target[target_exists]
        dist = float(np.linalg.norm(sub_flat - sub_target))
        return {"score": 1.0 / (1.0 + dist), "our_player": 0}

    # ── Real game mode ────────────────────────────────────────────
    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND

    model = create_model(single_head=single_head)
    model.set_parameters_from_vector(params_flat)
    agent = NeuralAgent(model=model)

    opp_model = create_model(single_head=single_head)
    opp_model.set_parameters_from_vector(opp_params_flat)
    opponent = NeuralAgent(model=opp_model)

    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops_us = agent._choose_operations(state, our_player)
        ops_opp = opponent._choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    if hp_us <= 0 and hp_opp <= 0:
        return {"score": 0.5, "our_player": our_player}
    if hp_us > hp_opp:
        return {"score": 1.0, "our_player": our_player}
    if hp_opp > hp_us:
        return {"score": 0.0, "our_player": our_player}
    return {"score": 0.5, "our_player": our_player}


class ESTrainer:
    """Evolution Strategies trainer."""

    def __init__(
        self,
        population_size: int = 16,
        sigma: float = 0.2,
        lr: float = 0.01,
        momentum: float = 0.9,
        num_workers: int = 4,
        games_per_individual: int = 2,
        seed: int = 0,
        single_head: bool = False,
    ):
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.momentum = momentum
        self.num_workers = num_workers
        self.games_per_individual = games_per_individual
        self.seed = seed
        self.single_head = single_head
        self.rng = np.random.RandomState(seed)

        self.model = create_model(single_head=single_head)
        self.param_count = self.model.count_parameters()
        self.mean = self.model.get_parameters_as_vector()
        self.synthetic_target: np.ndarray | None = None
        self.velocity: np.ndarray | None = None

    def step(self, generation: int, pool: mp.Pool) -> dict:
        """Run one ES generation with mirrored sampling (reduces variance by 2x)."""
        t0 = time.time()

        assert self.population_size % 2 == 0, "population_size must be even (mirrored sampling)"
        n_noise = self.population_size // 2

        noise = self.rng.randn(n_noise, self.param_count).astype(np.float32)

        # Mirrored sampling: +sigma and -sigma for each noise vector
        params_list: list[np.ndarray] = []
        for n in noise:
            params_list.append(self.mean + self.sigma * n)
            params_list.append(self.mean - self.sigma * n)
        eval_start = time.time()

        # Select opponent pairs (each pair = one first-player + one second-player game)
        k_per_ind = self.games_per_individual // 2
        opp_indices = select_opponents(self.population_size, self.games_per_individual, self.rng)

        # Build task arguments: each opponent pair creates 2 games (alternating first/second)
        all_args = []
        for idx in range(self.population_size):
            for k in range(k_per_ind):
                opp_idx = opp_indices[idx][k]
                base_seed = self.seed + generation * self.population_size * self.games_per_individual + (idx * self.games_per_individual + k * 2)
                # base_seed is always even → our_player=0; base_seed+1 → our_player=1
                all_args.append((params_list[idx], params_list[opp_idx], base_seed, self.single_head, self.synthetic_target))
                all_args.append((params_list[idx], params_list[opp_idx], base_seed + 1, self.single_head, self.synthetic_target))

        # Submit all tasks and wait with timeout polling (so Ctrl+C works on Windows)
        async_result = pool.starmap_async(_eval_worker, all_args)
        while True:
            try:
                all_results = async_result.get(timeout=2)
                break  # Got all results
            except mp.TimeoutError:
                if getattr(pool, '_interrupted', False):
                    return {"generation": generation, "best_fitness": 0.0, "avg_fitness": 0.0,
                            "eval_time": 0.0, "total_time": 0.0}
                continue
            except (mp.context.BrokenProcessPool, OSError, ValueError):
                if getattr(pool, '_interrupted', False):
                    return {"generation": generation, "best_fitness": 0.0, "avg_fitness": 0.0,
                            "eval_time": 0.0, "total_time": 0.0}
                raise

        # Parse results
        # Supports both dict return (score, our_player) and float return (backward compat)
        game_info = []
        for r in all_results:
            if isinstance(r, dict):
                game_info.append(r)
            else:
                game_info.append({"score": r, "our_player": 0})

        scores = np.array([g["score"] for g in game_info], dtype=np.float32)
        fitness = np.mean(scores.reshape(self.population_size, self.games_per_individual), axis=1)
        eval_time = time.time() - eval_start

        # Per-individual stats (first/second player breakdown)
        ind_details = []
        for idx in range(self.population_size):
            games = game_info[idx * self.games_per_individual : (idx + 1) * self.games_per_individual]
            p0 = [g for g in games if g["our_player"] == 0]
            p1 = [g for g in games if g["our_player"] == 1]
            ind_details.append({
                "p0_w": int(sum(1 for g in p0 if g["score"] == 1.0)),
                "p0_n": int(len(p0)),
                "p1_w": int(sum(1 for g in p1 if g["score"] == 1.0)),
                "p1_n": int(len(p1)),
                "score": float(fitness[idx]),
            })

        ranks = np.argsort(np.argsort(fitness))
        shaped = (ranks + 1) / (self.population_size + 1) - 0.5

        # Mirrored sampling gradient: sum of (f_pos - f_neg) * noise / (N * sigma)
        shaped_pairs = shaped.reshape(n_noise, 2)
        pair_diffs = shaped_pairs[:, 0] - shaped_pairs[:, 1]
        gradient = (noise.T @ pair_diffs) / (self.population_size * self.sigma)

        # Momentum update: v = μ·v + lr·g ; θ += v
        if self.velocity is None:
            self.velocity = np.zeros_like(self.mean)
        self.velocity = self.momentum * self.velocity + self.lr * gradient.astype(self.mean.dtype)
        self.mean += self.velocity
        self.model.set_parameters_from_vector(self.mean)

        total_time = time.time() - t0

        # Top-2 individuals' parameters
        top2_idx = np.argsort(fitness)[-2:]
        top2_params = [params_list[i].copy() for i in top2_idx]
        top2_scores = [float(fitness[i]) for i in top2_idx]

        result = {
            "generation": generation,
            "best_fitness": float(fitness.max()),
            "avg_fitness": float(fitness.mean()),
            "eval_time": eval_time,
            "total_time": total_time,
            "ind_details": ind_details,
            "top2_params": top2_params,
            "top2_scores": top2_scores,
        }
        if self.synthetic_target is not None:
            target_exists = self.synthetic_target != 0
            sub_mean = self.mean[target_exists]
            sub_target = self.synthetic_target[target_exists]
            result["dist_to_target"] = float(np.linalg.norm(sub_mean - sub_target))
        return result

    def save_checkpoint(self, path: str | Path, top2_params: list[np.ndarray] | None = None,
                        top2_scores: list[float] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "mean": torch.from_numpy(self.mean),
            "model_state": self.model.state_dict(),
            "generation": self.step_count if hasattr(self, "step_count") else 0,
        }
        if top2_params is not None:
            data["top2_params"] = [torch.from_numpy(p) for p in top2_params]
            data["top2_scores"] = top2_scores
        if self.velocity is not None:
            data["velocity"] = torch.from_numpy(self.velocity)
        torch.save(data, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.mean = ckpt["mean"].numpy()
        self.model.set_parameters_from_vector(self.mean)
        if "velocity" in ckpt:
            self.velocity = ckpt["velocity"].numpy()


def main():
    parser = argparse.ArgumentParser(description="ES training for Ant-Game AI (self-play)")
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=2, help="games per individual per generation")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-head", action="store_true",
                        help="train with only 1 policy head (reduce conflicting actions)")
    parser.add_argument("--checkpoint", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--synthetic-test", action="store_true",
                        help="synthetic fitness: converge toward random target (test ES correctness)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: training_history_<timestamp>)")
    args = parser.parse_args()

    # ── Prepare output directory ──────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("training_history")
    base_dir.mkdir(exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / f"{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────
    log = get_logger(out_dir / "train.log")
    csv_path = out_dir / "history.csv"

    # Save config
    with open(out_dir / "config.txt", "w") as f:
        for key, val in vars(args).items():
            f.write(f"{key}={val}\n")
        f.write(f"timestamp={ts}\n")

    # ── Trainer ───────────────────────────────────────────────────
    trainer = ESTrainer(
        population_size=args.pop_size,
        sigma=args.sigma,
        lr=args.lr,
        momentum=args.momentum,
        num_workers=args.workers,
        games_per_individual=args.games,
        seed=args.seed,
        single_head=args.single_head,
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # ── Synthetic test: generate random target ──────────────────
    if args.synthetic_test:
        # Use first N_DIMS dimensions for the distance computation
        # (full param space is too large for ES to show convergence in reasonable time)
        target_sub_dim = 1000
        target = trainer.rng.uniform(-0.5, 0.5, size=target_sub_dim).astype(np.float32)
        # Pad to full param size with zeros (unused in distance computation)
        full_target = np.zeros(trainer.mean.shape, dtype=np.float32)
        full_target[:target_sub_dim] = target
        trainer.synthetic_target = full_target
        init_dist = float(np.linalg.norm(trainer.mean[:target_sub_dim] - target))

    # ── Print config ──────────────────────────────────────────────
    log.header("ES Training")
    log.print(key="out_dir", value=out_dir)
    log.print(key="params", value=f"{trainer.param_count:,}")
    log.print(key="seed", value=args.seed)
    log.print(key="pop_size", value=args.pop_size)
    log.print(key="sigma", value=args.sigma)
    log.print(key="lr", value=args.lr)
    log.print(key="momentum", value=args.momentum)
    log.print(key="workers", value=args.workers)
    log.print(key="games_per_ind", value=args.games)
    log.print(key="generations", value=args.generations)
    log.print(key="single_head", value=args.single_head)
    log.print(key="opponents", value="random from population (self-play)")
    log.separator("-")

    # ── Print mode info ──────────────────────────────────────────
    log.print(key="mode", value="synthetic test" if args.synthetic_test else "real game (self-play)")
    if args.synthetic_test:
        log.print(key="init_dist_to_target", value=f"{init_dist:.4f}")

    # ── CSV header ────────────────────────────────────────────────
    csv_header = ["generation", "best_fitness", "avg_fitness", "eval_time_s", "total_time_s"]
    if args.synthetic_test:
        csv_header.append("dist_to_target")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)

    # ── Training loop ─────────────────────────────────────────────
    log.print_table(gen="gen", best="best_fit", avg="avg_fit", eval_s="eval(s)", total_s="total(s)",
                    **({"dist": "dist_to_target"} if args.synthetic_test else {}))
    log.separator("-", width=50 if not args.synthetic_test else 65, timestamp=False)
    log.print("  [Ctrl+C: stop after current gen | Second Ctrl+C: force quit]")
    log.print("  [PAUSE file: create PAUSE in out_dir to pause, delete to resume]")
    log.separator("-", width=50 if not args.synthetic_test else 65, timestamp=False)

    pool = mp.Pool(args.workers)
    interrupted = False
    pause_file = out_dir / "PAUSE"
    result = None
    trainer.step_count = 0

    def _signal_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            log.print("Forced exit.")
            pool.terminate()
            sys.exit(1)
        interrupted = True
        pool._interrupted = True  # Signal step() to abort
        log.print("")
        log.print(key="interrupt", value="Ctrl+C received, stopping after current generation...")
        pool.terminate()
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except AttributeError:
        pass

    try:
        for gen in range(args.generations):
            if interrupted:
                break
            # Check for PAUSE file before starting a generation
            while pause_file.exists() and not interrupted:
                time.sleep(2)
            if interrupted:
                break
            result = trainer.step(gen, pool)
            if interrupted:
                break
            trainer.step_count = gen + 1

            log.print_table(
                gen=result["generation"],
                best=f"{result['best_fitness']:.4f}",
                avg=f"{result['avg_fitness']:.4f}",
                eval_s=f"{result['eval_time']:.1f}",
                total_s=f"{result['total_time']:.1f}",
                **({"dist": f"{result.get('dist_to_target', 0):.2f}"} if args.synthetic_test else {}),
            )

            # Print per-individual breakdown
            details = result.get("ind_details", [])
            if details and not args.synthetic_test:
                best_i = max(range(len(details)), key=lambda i: details[i]["score"])
                b = details[best_i]
                # Fitness distribution histogram (auto-select bucket count)
                # Try [8, 6, 5, 4] that divide games evenly, pick the largest match
                n_candidates = [8, 6, 5, 4]
                n_bins = next((n for n in n_candidates if args.games % n == 0), 6)
                n_buckets = n_bins + 1
                step = 1.0 / n_bins
                bin_labels = [f"{(i * step):.3f}" for i in range(n_buckets)]
                hist = {lbl: 0 for lbl in bin_labels}
                for d in details:
                    bkt = f"{round(d['score'] / step) * step:.3f}"
                    hist[bkt] = hist.get(bkt, 0) + 1
                # Only show non-empty buckets
                non_empty = {k: v for k, v in sorted(hist.items()) if v > 0}
                hist_str = " ".join(f"{k}:{v}" for k, v in non_empty.items())
                log.print_table(
                    **{"ind_dist": hist_str,
                       "best": f"#{best_i} {b['score']:.3f} "
                               f"[1st:{b['p0_w']}/{b['p0_n']} 2nd:{b['p1_w']}/{b['p1_n']}]"},
                )

            # Append to CSV
            row = [
                result["generation"],
                f"{result['best_fitness']:.6f}",
                f"{result['avg_fitness']:.6f}",
                f"{result['eval_time']:.3f}",
                f"{result['total_time']:.3f}",
            ]
            if args.synthetic_test:
                row.append(f"{result['dist_to_target']:.6f}")
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(row)

            if args.save_every > 0 and (gen + 1) % args.save_every == 0:
                ckpt_path = out_dir / f"gen_{gen+1:04d}.pt"
                trainer.save_checkpoint(ckpt_path,
                                        top2_params=result.get("top2_params"),
                                        top2_scores=result.get("top2_scores"))
                log.print(key="checkpoint", value=ckpt_path)

    except KeyboardInterrupt:
        # Fallback in case signal handler didn't catch it
        interrupted = True
        log.print("")
        log.print(key="interrupt", value="KeyboardInterrupt, saving checkpoint...")
    finally:
        pool.terminate()
        pool.join()

    # ── Final ─────────────────────────────────────────────────────
    if interrupted:
        ckpt_path = out_dir / f"interrupt_gen_{trainer.step_count:04d}.pt"
        trainer.save_checkpoint(ckpt_path,
                                top2_params=result.get("top2_params") if result else None,
                                top2_scores=result.get("top2_scores") if result else None)
        log.print(key="interrupt_checkpoint", value=ckpt_path)
    else:
        trainer.save_checkpoint(out_dir / "final.pt",
                                top2_params=result.get("top2_params") if result else None,
                                top2_scores=result.get("top2_scores") if result else None)

    log.separator("=")
    if result is not None and not interrupted:
        log.print(key="best_fitness", value=f"{result['best_fitness']:.4f}")
        log.print(key="avg_fitness", value=f"{result['avg_fitness']:.4f}")
    if args.synthetic_test and result is not None and not interrupted:
        final_dist = result.get("dist_to_target", 0)
        log.print(key="final_dist_to_target", value=f"{final_dist:.4f}")
        log.print(key="dist_reduction", value=f"{init_dist / max(final_dist, 1e-10):.1f}x")
        if final_dist < init_dist * 0.1:
            log.print("  ✅ ES converged toward the target!")
        else:
            log.print("  ⚠️  ES did NOT converge; check hyperparameters.")
    log.print(key="history", value=csv_path)
    log.print("Done.")


if __name__ == "__main__":
    main()

