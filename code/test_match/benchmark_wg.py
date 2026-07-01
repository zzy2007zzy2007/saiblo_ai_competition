"""Benchmark WinGraph add_node to diagnose low CPU utilization."""
from __future__ import annotations
import sys, time, multiprocessing as mp
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
from collections import defaultdict

# ── track CPU usage ──────────────────────────────────────
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


def get_cpu_percent():
    if HAVE_PSUTIL:
        return psutil.cpu_percent(interval=1.0)
    return -1


def track_cpu(duration_sec=10, interval=0.5):
    """Sample CPU every `interval` seconds for `duration` seconds."""
    if not HAVE_PSUTIL:
        return []
    samples = []
    t_end = time.time() + duration_sec
    while time.time() < t_end:
        samples.append(psutil.cpu_percent(interval=interval))
    return samples


# ── quick test using add_node (the batched version) ─────
def benchmark(num_nodes=10, edge_games=10, workers=16):
    from my_ai.win_graph import WinGraph

    # Load some checkpoint parameters
    ckpt_dir = Path("training_history/20260701_004058")
    pts = sorted(ckpt_dir.glob("gen_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(pts)} checkpoints, using first {num_nodes}")

    params_list = []
    for pt in pts[:num_nodes]:
        data = torch.load(pt, map_location="cpu", weights_only=True)
        p = data["top2_params"][0].numpy()  # top1
        params_list.append(p)

    wg = WinGraph(n_max=num_nodes + 10, edge_games=edge_games,
                  workers=workers, num_heads=3)

    # Use a pre-created pool to avoid cold start overhead (same as training does)
    print(f"  Creating shared Pool with {workers} workers...")
    with mp.Pool(workers) as shared_pool:
        # Warm up: run one dummy task to force process spawn
        shared_pool.map(_dummy, range(workers))
        t_warm = time.time()
        print(f"  Pool warm-up: {time.time() - t_warm:.1f}s")

        for i, params in enumerate(params_list):
            t0 = time.time()
            result = wg.add_node(i, params, pool=shared_pool)
            dt = time.time() - t0
            print(f"  add_node gen={i:2d}:  {dt:5.1f}s  edges={wg.get_graph_info()['n_edges']:3d}  "
                  f"n_nodes={result['n_nodes']}")


def _dummy(_):
    return 1

    if cpu_samples:
        print(f"\n  CPU samples during add_node: {len(cpu_samples)} samples")
        print(f"  >90%: {sum(1 for c in cpu_samples if c > 90)}/{len(cpu_samples)}")
        print(f"  <50%: {sum(1 for c in cpu_samples if c < 50)}/{len(cpu_samples)}")

    info = wg.get_graph_info()
    print(f"\n  Final graph: {info['n_nodes']} nodes, {info['n_edges']} edges, {info['n_scc']} SCCs")
    print(f"  Topo ranks: {info['topo_ranks']}")

    # — manual check: simulate old-style loop to compare ———
    print(f"\n{'='*60}")
    print("Comparison: sequential Pool creation (old approach)")
    print(f"{'='*60}")
    from my_ai.win_graph import compute_win_rate
    t0 = time.time()
    for j in range(num_nodes - 1):  # first node vs previous
        if j < 1:
            continue
        _ = compute_win_rate(params_list[j], params_list[0],
                             edge_games, workers, num_heads=3)
    dt_seq = time.time() - t0
    print(f"  {num_nodes-1} sequential compute_win_rate calls: {dt_seq:.1f}s")
    print(f"  vs batched add_node with {num_nodes} nodes: ~{sum(1 for _ in range(1)):.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    benchmark(num_nodes=10, edge_games=16, workers=16)
