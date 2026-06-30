"""Quick synthetic test: ES convergence on a low-dimensional problem.

This tests whether the ES algorithm (noise → evaluate → rank → gradient → update)
can converge on a simple spherical function in 1000-D space.
"""
from __future__ import annotations
import numpy as np


def test_es(dim: int = 1000, pop_size: int = 32, sigma: float = 0.5, lr: float = 0.1,
            n_gen: int = 500, seed: int = 42):
    rng = np.random.RandomState(seed)

    # Random target in [-0.5, 0.5]
    target = rng.uniform(-0.5, 0.5, size=dim).astype(np.float32)
    mean = np.zeros(dim, dtype=np.float32)

    init_dist = np.linalg.norm(mean - target)

    for gen in range(n_gen):
        noise = rng.randn(pop_size, dim).astype(np.float32)
        params_list = [mean + sigma * n for n in noise]

        # Absolute fitness: negative distance to target
        fitness = np.array([-np.linalg.norm(p - target) for p in params_list], dtype=np.float32)

        # Rank-based shaping
        ranks = np.argsort(np.argsort(fitness))
        shaped = (ranks + 1) / (pop_size + 1) - 0.5

        # Gradient and update
        gradient = (noise.T @ shaped) / (pop_size * sigma)
        mean += lr * gradient.astype(mean.dtype)

        if gen % 100 == 0 or gen == n_gen - 1:
            dist = np.linalg.norm(mean - target)
            print(f"  gen={gen:4d}  best_fit={fitness.max():.4f}  dist={dist:.4f}  "
                  f"reduction={init_dist / max(dist, 1e-10):.2f}x")

    final_dist = np.linalg.norm(mean - target)
    print(f"\n  dim={dim}, pop_size={pop_size}, sigma={sigma}, lr={lr}")
    print(f"  Initial distance: {init_dist:.4f}")
    print(f"  Final distance:   {final_dist:.4f}")
    print(f"  Reduction:        {init_dist / max(final_dist, 1e-10):.1f}x")
    success = final_dist < init_dist * 0.1
    print(f"  {'✅ ES converged!' if success else '❌ ES did not converge'}")
    return success


if __name__ == "__main__":
    for dim in [100, 1000, 10000]:
        print(f"\n{'='*50}")
        test_es(dim=dim, pop_size=32, sigma=0.5, lr=0.1, n_gen=500)
