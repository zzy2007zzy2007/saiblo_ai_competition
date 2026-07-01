"""Quick WinGraph add_node benchmark with shared pool (for CPU monitoring)."""
from __future__ import annotations
import sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import multiprocessing as mp
import numpy as np
import torch

def _dummy(_):
    return 1

if __name__ == "__main__":
    mp.freeze_support()

    num_nodes = 10
    edge_games = 16
    workers = 22  # same as training

    from my_ai.win_graph import WinGraph

    ckpt_dir = Path("training_history/20260701_004058")
    pts = sorted(ckpt_dir.glob("gen_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    params_list = [torch.load(p, map_location="cpu", weights_only=True)["top2_params"][0].numpy() for p in pts[:num_nodes]]

    wg = WinGraph(n_max=num_nodes + 10, edge_games=edge_games,
                  workers=workers, num_heads=1)

    print(f"Creating shared Pool with {workers} workers... (watch CPU in Task Manager)")
    sys.stdout.flush()
    with mp.Pool(workers) as shared_pool:
        # Warm up the pool
        shared_pool.map(_dummy, range(workers))
        print("Pool ready. Now adding nodes (CPU should be high)...")
        sys.stdout.flush()
        time.sleep(5)

        for i, params in enumerate(params_list):
            t0 = time.time()
            result = wg.add_node(i, params, pool=shared_pool)
            dt = time.time() - t0
            n = result['n_nodes']
            print(f"  gen={i:2d}: {dt:5.1f}s  n_nodes={n:2d}  edges={wg.get_graph_info()['n_edges']:3d}  "
                  f"win_rate={result['win_rate']:.2f}")
            sys.stdout.flush()

    print(f"\nDone. Graph: {wg.get_graph_info()}")
