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
import time
import multiprocessing as mp

import numpy as np
import torch

from my_ai.network import create_model
from my_ai.agent import NeuralAgent


def _eval_worker(params_flat: np.ndarray, seed: int, opponent: str) -> float:
    """Run one match: params vs opponent. Return 1/0.5/0 for win/draw/loss."""
    # Worker must set up its own import paths
    import sys
    from pathlib import Path
    _RP = Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = Path(__file__).resolve().parents[1]
    for p in (_RP, _CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

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

    # Use cold_handle_rule_illegal=True so illegal ops are filtered instead of
    # instantly losing -- critical during early training when the network is random.
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops0 = agent._choose_operations(state, 0)
        ops1 = opp.choose_operations(state, 1)
        # resolve_turn handles: apply ops (with validation), check terminal, advance round
        state.resolve_turn(ops0, ops1)

    hp0, hp1 = state.bases[0].hp, state.bases[1].hp
    if hp0 <= 0 and hp1 <= 0:
        return 0.5
    if hp0 > hp1:
        return 1.0
    if hp1 > hp0:
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

        # Create initial model and get parameter count
        self.model = create_model()
        self.param_count = self.model.count_parameters()

        # Mean parameters (current best estimate)
        self.mean = self.model.get_parameters_as_vector()

        print(f"Model parameters: {self.param_count:,}")
        print(f"ES config: pop={population_size}, sigma={sigma}, lr={lr}, workers={num_workers}")
        print(f"Games per eval: {games_per_individual}")

    def step(self, generation: int) -> dict:
        """Run one ES generation."""
        t0 = time.time()

        # 1. Sample noise
        noise = self.rng.randn(self.population_size, self.param_count).astype(np.float32)

        # 2. Generate seeds for each individual's games
        seeds = [
            self.seed + generation * self.population_size * self.games_per_individual + i
            for i in range(self.population_size * self.games_per_individual)
        ]

        # 3. Evaluate all individuals in parallel
        params_list = [self.mean + self.sigma * n for n in noise]
        eval_start = time.time()

        with mp.Pool(self.num_workers) as pool:
            tasks = []
            for idx in range(self.population_size):
                for g in range(self.games_per_individual):
                    s = seeds[idx * self.games_per_individual + g]
                    tasks.append(pool.apply_async(_eval_worker, (params_list[idx], s, self.opponent)))
            all_scores = [t.get() for t in tasks]

        # Reshape: (pop_size * games_per_individual) -> (pop_size, games_per_individual)
        fitness = np.mean(
            np.array(all_scores).reshape(self.population_size, self.games_per_individual),
            axis=1,
        )
        eval_time = time.time() - eval_start

        # 4. Fitness shaping (rank-based)
        ranks = np.argsort(np.argsort(fitness))  # 0..pop_size-1
        shaped = (ranks + 1) / (self.population_size + 1) - 0.5  # ~[-0.5, 0.5]

        # 5. Gradient estimate & update
        gradient = (noise.T @ shaped) / (self.population_size * self.sigma)
        self.mean += self.lr * gradient.astype(self.mean.dtype)

        # 6. Update model with new mean
        self.model.set_parameters_from_vector(self.mean)

        total_time = time.time() - t0
        best_fitness = float(fitness.max())
        avg_fitness = float(fitness.mean())

        return {
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "eval_time": eval_time,
            "total_time": total_time,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
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
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.mean = ckpt["mean"].numpy()
        self.model.set_parameters_from_vector(self.mean)
        print(f"  Checkpoint loaded: {path} (gen {ckpt.get('generation', '?')})")


def main():
    parser = argparse.ArgumentParser(description="ES training for Ant-Game AI")
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=2, help="games per individual per generation")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--opponent", default="random", choices=["random", "example"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

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

    print(f"\nStarting ES training for {args.generations} generations...")
    print(f"{'gen':>4}  {'best_fit':>8}  {'avg_fit':>8}  {'eval(s)':>7}  {'total(s)':>7}")
    print("-" * 45)

    for gen in range(args.generations):
        result = trainer.step(gen)
        trainer.step_count = gen + 1
        print(
            f"{result['generation']:>4}  {result['best_fitness']:>8.4f}  "
            f"{result['avg_fitness']:>8.4f}  {result['eval_time']:>7.1f}  "
            f"{result['total_time']:>7.1f}"
        )

        if args.save_every > 0 and (gen + 1) % args.save_every == 0:
            trainer.save_checkpoint(Path(args.save_dir) / f"gen_{gen+1:04d}.pt")

    trainer.save_checkpoint(Path(args.save_dir) / "final.pt")
    print(f"\nTraining complete. Final mean fitness: {result['avg_fitness']:.4f}")


if __name__ == "__main__":
    main()
