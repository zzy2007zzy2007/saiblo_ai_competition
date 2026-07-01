"""Build win-rate graph among a set of checkpoints."""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import numpy as np
import multiprocessing as mp
import time
import itertools
from my_ai.network import create_model
from my_ai.agent import NeuralAgent


def worker(args):
    s, pt_a, pt_b = args
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND

    da = torch.load(pt_a, map_location="cpu", weights_only=True)
    db = torch.load(pt_b, map_location="cpu", weights_only=True)
    va = da["top2_params"][0].numpy()
    vb = db["top2_params"][0].numpy()
    ma = create_model(single_head=True); ma.set_parameters_from_vector(va)
    mb = create_model(single_head=True); mb.set_parameters_from_vector(vb)
    aa = NeuralAgent(model=ma); ab = NeuralAgent(model=mb)

    state = GameState.initial(seed=s, cold_handle_rule_illegal=True)
    p = s % 2
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        if p == 0:
            u, v = aa._choose_operations(state, 0), ab._choose_operations(state, 1)
        else:
            u, v = ab._choose_operations(state, 0), aa._choose_operations(state, 1)
        state.resolve_turn(u, v) if p == 0 else state.resolve_turn(v, u)
    h0, h1 = state.bases[0].hp, state.bases[1].hp
    # 1 if first player wins, 0 if second wins, 0.5 draw
    if h0 > h1:
        return 1.0
    elif h1 > h0:
        return 0.0
    return 0.5


def compute_pair(pt_a, pt_b, games, workers):
    n2 = games // 2
    seeds = list(range(n2))
    with mp.Pool(workers) as pool:
        ra = pool.map(worker, [(s, pt_a, pt_b) for s in seeds])
        rb = pool.map(worker, [(s + 1000, pt_b, pt_a) for s in seeds])
    wins_a = sum(1 for r in ra if r == 1.0) + sum(1 for r in rb if r == 0.0)
    wins_b = sum(1 for r in rb if r == 1.0) + sum(1 for r in ra if r == 0.0)
    draws = games - wins_a - wins_b
    return wins_a, wins_b, draws


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=34)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pts = sorted(Path(args.outdir).glob("gen_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    sel = [p for p in pts if args.first <= int(p.stem.split("_")[1]) <= args.last]
    gens = [int(p.stem.split("_")[1]) for p in sel]

    print(f"Selected {len(sel)} checkpoints: gen_{gens[0]} → gen_{gens[-1]}")
    print(f"Pairs: {len(sel) * (len(sel) - 1) // 2}")
    print(f"Estimated time: ~{len(sel) * (len(sel) - 1) // 2 * args.games * 3 / args.workers}s\n")

    n = len(sel)
    matrix = np.zeros((n, n), dtype=float)  # matrix[i][j] = win rate of i against j

    for i, j in itertools.combinations(range(n), 2):
        t0 = time.time()
        wa, wb, d = compute_pair(sel[i], sel[j], args.games, args.workers)
        matrix[i][j] = wa / args.games
        matrix[j][i] = wb / args.games
        t = time.time() - t0
        print(f"  gen_{gens[i]:3d} vs gen_{gens[j]:3d}:  "
              f"{wa:2d}W/{wb:2d}L/{d:2d}D  {t:.0f}s")

    # Print matrix
    print(f"\n{'='*80}")
    print("Win rate matrix (row beats column)")
    print(f"{'':>8s}", " ".join(f"{g:>6d}" for g in gens))
    for i in range(n):
        row = " ".join(f"{matrix[i][j]*100:5.0f}%" for j in range(n))
        print(f"  gen_{gens[i]:3d}:  {row}")

    # SCC analysis
    print(f"\n{'='*80}")
    print("Building graph for SCC analysis...")
    # threshold: if win rate >= 60% -> directed edge
    # but if 40-60% -> no edge (too close)
    threshold = 0.55  # 55% threshold
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > threshold and matrix[j][i] < 1 - threshold:
                edges.append((gens[i], gens[j]))

    print(f"Edges (>={threshold*100:.0f}% win rate): {len(edges)}")
    for src, dst in edges:
        print(f"  gen_{src} → gen_{dst}")

    # Simple transitive closure -> SCC approximation
    # Group nodes that form cycles
    from collections import defaultdict

    adj = defaultdict(set)
    radj = defaultdict(set)
    for s, d in edges:
        adj[s].add(d)
        radj[d].add(s)

    # Kosaraju-like: find SCCs
    visited = set()
    order = []

    def dfs(u, graph, collect):
        stack = [u]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                if collect:
                    order.append(v)
                for w in graph[v]:
                    if w not in visited:
                        stack.append(w)

    for g in gens:
        if g not in visited:
            dfs(g, adj, True)

    visited.clear()
    comps = {}
    comp_of = {}

    def rdfs(u, cid):
        stack = [u]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                comp_of[v] = cid
                comps.setdefault(cid, []).append(v)
                for w in radj[v]:
                    if w not in visited:
                        stack.append(w)

    cid = 0
    for g in reversed(order):
        if g not in visited:
            rdfs(g, cid)
            cid += 1

    print(f"\nSCCs found: {cid}")
    for c in range(cid):
        nodes = comps[c]
        print(f"  SCC {c}: size={len(nodes)}  nodes={nodes}")

    # DAG edges between SCCs
    dag_edges = set()
    for s, d in edges:
        cs, cd = comp_of[s], comp_of[d]
        if cs != cd:
            dag_edges.add((cs, cd))
    print(f"\nDAG edges between SCCs: {len(dag_edges)}")
    print("Topological order of SCCs:")
    
    # Topological sort of SCC DAG
    dag_adj = defaultdict(set)
    for cs, cd in dag_edges:
        dag_adj[cs].add(cd)
    in_deg = {c: 0 for c in range(cid)}
    for cs, cd in dag_edges:
        in_deg[cd] += 1
    
    queue = [c for c in range(cid) if in_deg[c] == 0]
    topo = []
    while queue:
        c = queue.pop(0)
        topo.append(c)
        for cd in dag_adj[c]:
            in_deg[cd] -= 1
            if in_deg[cd] == 0:
                queue.append(cd)

    for rank, c in enumerate(topo):
        arrow = " ⊤" if rank == 0 else (" ⊥" if rank == len(topo) - 1 else "")
        print(f"  rank={rank}{arrow}: SCC {c}: {comps[c]}")
