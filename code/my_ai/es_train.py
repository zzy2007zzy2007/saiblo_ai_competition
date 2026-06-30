"""Minimal Evolution Strategies trainer for Ant-Game AI.

Core algorithm (OpenAI-ES style):
  1. Sample noise -> create perturbed models: theta +/- sigma*eps
  2. Evaluate each against baseline opponent -> fitness = win rate
  3. Fitness shaping (rank-based normalization)
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


def _eval_worker(params_flat: np.ndarray, seed: int, opponent: str) -> float:
    """Run one match: params vs opponent. Return 1/0.5/0 for win/draw/loss."""
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

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    from AI.ai_random import AI as RandomAI

    model = create_model()
    model.set_parameters_from_vector(params_flat)
    agent = NeuralAgent(model=model)

    if opponent == "random":
        opp = RandomAI(seed=seed)
    elif opponent == "example":
        from AI.ai_example import AI as ExampleAI
        opp = ExampleAI(seed=seed)
    else:
        raise ValueError(f"unknown opponent: {opponent}")

    # Alternate first/second player by seed parity for fairness
    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops_us = agent._choose_operations(state, our_player)
        ops_opp = opp.choose_operations(state, opp_player)
        # resolve_turn expects (player0_ops, player1_ops) in that order
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    if hp_us <= 0 and hp_opp <= 0:
        return 0.5
    if hp_us > hp_opp:
        return 1.0
    if hp_opp > hp_us:
        return 0.0
    return 0.5


class ESTrainer:
    """Evolution Strategies trainer."""

    def __init__(
        self,
        population_size: int = 16,
        sigma: float = 0.05,
        lr: float = 0.01,
        num_workers: int = 4,
        games_per_individual: int = 2,
        opponent: str = "random",
        seed: int = 0,
    ):
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.num_workers = num_workers
        self.games_per_individual = games_per_individual
        self.opponent = opponent
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.model = create_model()
        self.param_count = self.model.count_parameters()
        self.mean = self.model.get_parameters_as_vector()

    def step(self, generation: int, pool: mp.Pool) -> dict:
        """Run one ES generation using an external pool (for interrupt safety)."""
        t0 = time.time()

        noise = self.rng.randn(self.population_size, self.param_count).astype(np.float32)

        seeds = [
            self.seed + generation * self.population_size * self.games_per_individual + i
            for i in range(self.population_size * self.games_per_individual)
        ]

        params_list = [self.mean + self.sigma * n for n in noise]
        eval_start = time.time()

        tasks = []
        for idx in range(self.population_size):
            for g in range(self.games_per_individual):
                s = seeds[idx * self.games_per_individual + g]
                tasks.append(pool.apply_async(_eval_worker, (params_list[idx], s, self.opponent)))
        all_scores = [t.get() for t in tasks]

        fitness = np.mean(
            np.array(all_scores).reshape(self.population_size, self.games_per_individual),
            axis=1,
        )
        eval_time = time.time() - eval_start

        ranks = np.argsort(np.argsort(fitness))
        shaped = (ranks + 1) / (self.population_size + 1) - 0.5

        gradient = (noise.T @ shaped) / (self.population_size * self.sigma)
        self.mean += self.lr * gradient.astype(self.mean.dtype)
        self.model.set_parameters_from_vector(self.mean)

        total_time = time.time() - t0
        return {
            "generation": generation,
            "best_fitness": float(fitness.max()),
            "avg_fitness": float(fitness.mean()),
            "eval_time": eval_time,
            "total_time": total_time,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean": torch.from_numpy(self.mean),
                "model_state": self.model.state_dict(),
                "generation": self.step_count if hasattr(self, "step_count") else 0,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.mean = ckpt["mean"].numpy()
        self.model.set_parameters_from_vector(self.mean)


def main():
    parser = argparse.ArgumentParser(description="ES training for Ant-Game AI")
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=2, help="games per individual per generation")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--opponent", default="example", choices=["random", "example"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: training_history_<timestamp>)")
    args = parser.parse_args()

    # ── Prepare output directory ──────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"training_history_{ts}")
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
        num_workers=args.workers,
        games_per_individual=args.games,
        opponent=args.opponent,
        seed=args.seed,
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # ── Print config ──────────────────────────────────────────────
    log.header("ES Training")
    log.print(key="out_dir", value=out_dir)
    log.print(key="params", value=f"{trainer.param_count:,}")
    log.print(key="seed", value=args.seed)
    log.print(key="pop_size", value=args.pop_size)
    log.print(key="sigma", value=args.sigma)
    log.print(key="lr", value=args.lr)
    log.print(key="workers", value=args.workers)
    log.print(key="games_per_ind", value=args.games)
    log.print(key="opponent", value=args.opponent)
    log.print(key="generations", value=args.generations)
    log.separator("-")

    # ── CSV header ────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_fitness", "avg_fitness", "eval_time_s", "total_time_s"])

    # ── Training loop ─────────────────────────────────────────────
    log.print_table(gen="gen", best="best_fit", avg="avg_fit", eval_s="eval(s)", total_s="total(s)")
    log.separator("-", width=50, timestamp=False)

    pool = mp.Pool(args.workers)
    interrupted = False

    def _signal_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C: force quit
            log.print("Forced exit.")
            pool.terminate()
            sys.exit(1)
        interrupted = True
        log.print("")
        log.print(key="interrupt", value="Ctrl+C received, stopping after current generation...")
        log.print("  (press Ctrl+C again to force exit)")
    signal.signal(signal.SIGINT, _signal_handler)
    # Windows doesn't have SIGALRM, only SIGINT and SIGTERM are common.
    # But Python on Windows does support SIGTERM.
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except AttributeError:
        pass

    try:
        for gen in range(args.generations):
            if interrupted:
                break
            result = trainer.step(gen, pool)
            trainer.step_count = gen + 1

            log.print_table(
                gen=result["generation"],
                best=f"{result['best_fitness']:.4f}",
                avg=f"{result['avg_fitness']:.4f}",
                eval_s=f"{result['eval_time']:.1f}",
                total_s=f"{result['total_time']:.1f}",
            )

            # Append to CSV
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    result["generation"],
                    f"{result['best_fitness']:.6f}",
                    f"{result['avg_fitness']:.6f}",
                    f"{result['eval_time']:.3f}",
                    f"{result['total_time']:.3f}",
                ])

            if args.save_every > 0 and (gen + 1) % args.save_every == 0:
                ckpt_path = out_dir / f"gen_{gen+1:04d}.pt"
                trainer.save_checkpoint(ckpt_path)
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
        trainer.save_checkpoint(ckpt_path)
        log.print(key="interrupt_checkpoint", value=ckpt_path)
    else:
        trainer.save_checkpoint(out_dir / "final.pt")

    log.separator("=")
    log.print(key="best_fitness", value=f"{result['best_fitness']:.4f}" if not interrupted else "N/A (interrupted)")
    log.print(key="avg_fitness", value=f"{result['avg_fitness']:.4f}" if not interrupted else "N/A (interrupted)")
    log.print(key="history", value=csv_path)
    log.print("Done.")


if __name__ == "__main__":
    main()

