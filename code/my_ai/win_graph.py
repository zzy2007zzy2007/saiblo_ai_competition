"""WinGraph - directed graph of head-to-head win relationships among elite individuals."""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import multiprocessing as mp
from collections import defaultdict


# ─── module-level worker (must be picklable for multiprocessing) ───
def _game_worker(args):
    """Play one game between two parameter vectors. Returns 1.0 if first player wins."""
    s, params_a, params_b, single_head = args
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    ma = create_model(single_head=single_head); ma.set_parameters_from_vector(params_a)
    mb = create_model(single_head=single_head); mb.set_parameters_from_vector(params_b)
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
        state.resolve_turn(u, v)
    h0, h1 = state.bases[0].hp, state.bases[1].hp
    if h0 > h1:
        return 1.0
    elif h1 > h0:
        return 0.0
    return 0.5


def compute_win_rate(params_a: np.ndarray, params_b: np.ndarray,
                     games: int, workers: int = 2,
                     single_head: bool = True) -> tuple[int, int, int]:
    """
    Play `games` games between two individuals (half as first, half as second).
    Returns: (wins_a, wins_b, draws)
    """
    n2 = games // 2
    seeds_fwd = list(range(n2))
    seeds_rev = list(range(1000, 1000 + n2))
    args_fwd = [(s, params_a, params_b, single_head) for s in seeds_fwd]
    args_rev = [(s, params_b, params_a, single_head) for s in seeds_rev]
    with mp.Pool(workers) as pool:
        ra = pool.map(_game_worker, args_fwd)
        rb = pool.map(_game_worker, args_rev)
    wins_a = sum(1 for r in ra if r == 1.0) + sum(1 for r in rb if r == 0.0)
    wins_b = sum(1 for r in rb if r == 1.0) + sum(1 for r in ra if r == 0.0)
    draws = games - wins_a - wins_b
    return wins_a, wins_b, draws


# ─── SCC utilities ───
def _kosaraju(adj: dict, nodes: list) -> tuple[dict, dict]:
    """
    Kosaraju SCC decomposition.
    Returns: (comp_of: dict[gen → comp_id], comps: dict[comp_id → list[gen]])
    """
    radj = defaultdict(set)
    for u in adj:
        for v in adj[u]:
            radj[v].add(u)

    visited = set()
    order = []

    def dfs(u):
        stack = [u]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                order.append(v)
                for w in adj.get(v, set()):
                    if w not in visited:
                        stack.append(w)

    for g in nodes:
        if g not in visited:
            dfs(g)

    visited.clear()
    comp_of = {}
    comps = {}
    cid = 0

    def rdfs(u, cid):
        stack = [u]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                comp_of[v] = cid
                comps.setdefault(cid, []).append(v)
                for w in radj.get(v, set()):
                    if w not in visited:
                        stack.append(w)

    for g in reversed(order):
        if g not in visited:
            rdfs(g, cid)
            cid += 1

    return comp_of, comps


def _topo_sort_sccs(dag_edges: set, all_cids: set) -> list:
    """Topological sort of SCC IDs."""
    dag_adj = defaultdict(set)
    in_deg = {c: 0 for c in all_cids}
    for cs, cd in dag_edges:
        if cs != cd:
            dag_adj[cs].add(cd)
            in_deg[cd] = in_deg.get(cd, 0) + 1
            in_deg.setdefault(cs, 0)

    queue = [c for c in all_cids if in_deg.get(c, 0) == 0]
    topo = []
    while queue:
        c = queue.pop(0)
        topo.append(c)
        for cd in dag_adj.get(c, set()):
            in_deg[cd] -= 1
            if in_deg[cd] == 0:
                queue.append(cd)
    return topo


# ─── WinGraph ────────────────────────────────────────────────────────

class WinGraph:
    """
    Directed graph of head-to-head win relationships among elite individuals.

    Nodes = elite individuals (identified by generation number).
    Edges = A → B if A beats B by a significant margin (>= win_threshold win rate).

    SCC condensation produces a DAG. The top of the DAG (source SCCs) contains
    the strongest individuals; the bottom (sink SCCs) contains the weakest.
    """

    def __init__(self, n_max: int = 30, edge_games: int = 10,
                 win_threshold: float = 0.55, workers: int = 2,
                 single_head: bool = True):
        self._n_max = n_max
        self._edge_games = edge_games
        self._win_threshold = win_threshold
        self._workers = workers
        self._single_head = single_head

        # node data: gen → {"params": np.ndarray, "example_win_rate": float | None}
        self.nodes: dict[int, dict] = {}

        # edge records: (gen_a, gen_b) → {"wins_a": int, "wins_b": int, "draws": int}
        self.edges: dict[tuple[int, int], dict] = {}

        # cached SCC state
        self._comp_of: dict[int, int] = {}   # gen → comp_id
        self._comps: dict[int, list[int]] = {}  # comp_id → [gen, ...]
        self._topo_order: list[int] = []  # comp_ids in topological order

    # ── properties ──────────────────────────────────────────────────

    @property
    def n_max(self) -> int:
        return self._n_max

    @n_max.setter
    def n_max(self, value: int):
        self._n_max = value
        if len(self.nodes) > self._n_max:
            self.prune(len(self.nodes) - self._n_max)

    @property
    def edge_games(self) -> int:
        return self._edge_games

    @edge_games.setter
    def edge_games(self, value: int):
        self._edge_games = value

    @property
    def win_threshold(self) -> float:
        return self._win_threshold

    @win_threshold.setter
    def win_threshold(self, value: float):
        self._win_threshold = value

    @property
    def workers(self) -> int:
        return self._workers

    @workers.setter
    def workers(self, value: int):
        self._workers = value

    @property
    def single_head(self) -> bool:
        return self._single_head

    @single_head.setter
    def single_head(self, value: bool):
        self._single_head = value

    # ── core operations ─────────────────────────────────────────────

    def add_node(self, gen: int, params: np.ndarray,
                 example_win_rate: float | None = None) -> dict:
        """
        Add a new elite individual and compute edges against all existing nodes.

        Args:
            gen: generation number (unique identifier)
            params: parameter vector (1D numpy array)
            example_win_rate: optional win rate vs ExampleAI (for logging)

        Returns:
            {"gen": gen, "wins": int, "losses": int, "draws": int,
             "win_rate": float, "n_nodes": int}
        """
        if gen in self.nodes:
            raise ValueError(f"gen {gen} already in graph")

        self.nodes[gen] = {
            "params": params.copy(),
            "example_win_rate": example_win_rate,
        }

        # Batch all games into a single Pool to avoid repeated Pool creation overhead
        existing_gens = [g for g in self.nodes if g != gen]
        if not existing_gens:
            self._rebuild_scc()
            return {
                "gen": gen, "wins": 0, "losses": 0, "draws": 0,
                "win_rate": 0.5, "n_nodes": 1,
            }

        n2 = self._edge_games // 2
        single_head = self._single_head

        # Build all args: for each opponent, (seed, params, opp_params, single_head)
        all_args = []
        pair_info = []  # (opp_gen, index_in_results)
        for opp_gen in existing_gens:
            opp_params = self.nodes[opp_gen]["params"]
            for s in range(n2):
                all_args.append((s, params, opp_params, single_head))
                pair_info.append((opp_gen, "fwd"))
                all_args.append((s + 10000, opp_params, params, single_head))
                pair_info.append((opp_gen, "rev"))

        with mp.Pool(self._workers) as pool:
            results = pool.map(_game_worker, all_args)

        # Distribute results into edge records
        total_wins = 0
        total_losses = 0
        total_draws = 0

        # Accumulate per-opponent
        opp_wins: dict[int, int] = defaultdict(int)
        opp_losses: dict[int, int] = defaultdict(int)
        opp_draws: dict[int, int] = defaultdict(int)

        for (opp_gen, direction), r in zip(pair_info, results):
            if direction == "fwd":
                # params is first player
                if r == 1.0:
                    opp_wins[opp_gen] += 1
                elif r == 0.0:
                    opp_losses[opp_gen] += 1
                else:
                    opp_draws[opp_gen] += 1
            else:
                # opp_params is first player
                if r == 1.0:
                    opp_losses[opp_gen] += 1
                elif r == 0.0:
                    opp_wins[opp_gen] += 1
                else:
                    opp_draws[opp_gen] += 1

        for opp_gen in existing_gens:
            wa = opp_wins[opp_gen]
            wb = opp_losses[opp_gen]
            d = opp_draws[opp_gen]
            self.edges[(gen, opp_gen)] = {"wins_a": wa, "wins_b": wb, "draws": d}
            total_wins += wa
            total_losses += wb
            total_draws += d

        total_games = total_wins + total_losses + total_draws
        win_rate = total_wins / total_games if total_games > 0 else 0.5

        self._rebuild_scc()

        # Prune if over capacity
        if len(self.nodes) > self._n_max:
            self.prune(len(self.nodes) - self._n_max)

        return {
            "gen": gen,
            "wins": total_wins,
            "losses": total_losses,
            "draws": total_draws,
            "win_rate": win_rate,
            "n_nodes": len(self.nodes),
        }

    def remove_node(self, gen: int) -> bool:
        """Remove a node and all its edges. Returns True if found."""
        if gen not in self.nodes:
            return False
        del self.nodes[gen]
        # Remove all edges involving this gen
        to_del = [k for k in self.edges if k[0] == gen or k[1] == gen]
        for k in to_del:
            del self.edges[k]
        self._rebuild_scc()
        return True

    def prune(self, k: int = 1) -> list[int]:
        """
        Remove k weakest nodes from sink SCCs (bottom of DAG).
        Returns list of removed gen numbers.
        """
        if k <= 0 or len(self.nodes) <= 1:
            return []
        if not self._topo_order:
            self._rebuild_scc()

        removed = []
        # Collect nodes ordered by DAG rank (sink first)
        candidates = []
        for cid in reversed(self._topo_order):
            for g in self._comps.get(cid, []):
                candidates.append(g)

        for _ in range(min(k, len(self.nodes) - 1)):
            if not candidates:
                break
            # Find the weakest from candidates: prefer sink, then lowest win rate
            # Get all edge records involving this candidate
            def _score(g):
                wins = 0
                games = 0
                for (a, b), rec in self.edges.items():
                    if a == g:
                        wins += rec["wins_a"]
                        games += rec["wins_a"] + rec["wins_b"] + rec["draws"]
                    elif b == g:
                        wins += rec["wins_b"]
                        games += rec["wins_a"] + rec["wins_b"] + rec["draws"]
                return wins / games if games > 0 else 0.0

            candidates.sort(key=_score)
            target = candidates.pop(0)
            removed.append(target)
            self.remove_node(target)

        return removed

    def get_top_opponents(self, k: int = 5) -> list[dict]:
        """
        Return top-k strongest individuals from source SCCs (top of DAG).

        Returns:
            [{"gen": int, "params": np.ndarray,
              "example_win_rate": float | None, "comp_rank": int}, ...]
        """
        if not self._topo_order:
            self._rebuild_scc()

        result = []
        for cid in self._topo_order:
            for g in self._comps.get(cid, []):
                result.append({
                    "gen": g,
                    "params": self.nodes[g]["params"],
                    "example_win_rate": self.nodes[g]["example_win_rate"],
                    "comp_rank": cid,
                })
                if len(result) >= k:
                    return result
        return result

    def get_graph_info(self) -> dict:
        """Get graph statistics for monitoring/logging."""
        if not self._topo_order:
            self._rebuild_scc()

        topo_ranks = []
        for cid in self._topo_order:
            topo_ranks.append(self._comps.get(cid, []))

        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_scc": len(self._comps),
            "topo_ranks": topo_ranks,
            "edge_games": self._edge_games,
            "n_max": self._n_max,
        }

    def get_node_depths(self) -> list[dict]:
        """
        Return all nodes sorted by DAG depth (longest path from source SCC).
        depth=0 → source SCC (strongest), larger values → farther from top.

        Returns:
            [{"gen": int, "params": np.ndarray, "depth": int,
              "example_win_rate": float | None, "win_rate": float}, ...]
        """
        if not self._topo_order:
            self._rebuild_scc()

        # Build SCC DAG adjacency
        dag_adj = defaultdict(set)
        for (a, b), rec in self.edges.items():
            ca, cb = self._comp_of.get(a), self._comp_of.get(b)
            if ca is not None and cb is not None and ca != cb:
                dag_adj[ca].add(cb)

        # Reverse adj for predecessor tracking
        radj = defaultdict(set)
        for ca, cbs in dag_adj.items():
            for cb in cbs:
                radj[cb].add(ca)

        # Compute SCC depths
        scc_depth = {}
        for cid in self._topo_order:
            if cid in radj:
                scc_depth[cid] = max(scc_depth.get(p, 0) for p in radj[cid]) + 1
            else:
                scc_depth[cid] = 0

        # Per-node win rate
        def _node_win_rate(g):
            wins, games = 0, 0
            for (a, b), rec in self.edges.items():
                if a == g:
                    wins += rec["wins_a"]
                    games += rec["wins_a"] + rec["wins_b"] + rec["draws"]
                elif b == g:
                    wins += rec["wins_b"]
                    games += rec["wins_a"] + rec["wins_b"] + rec["draws"]
            return wins / games if games > 0 else 0.0

        nodes_sorted = sorted(
            self.nodes.keys(),
            key=lambda g: (scc_depth.get(self._comp_of.get(g, -1), 999), -_node_win_rate(g)),
        )

        result = []
        for g in nodes_sorted:
            result.append({
                "gen": g,
                "params": self.nodes[g]["params"],
                "depth": scc_depth.get(self._comp_of.get(g, -1), -1),
                "example_win_rate": self.nodes[g]["example_win_rate"],
                "win_rate": _node_win_rate(g),
            })
        return result

    def recompute_scc(self) -> tuple[dict, dict]:
        """Force recompute SCC condensation. Returns (comp_of, comps)."""
        self._rebuild_scc()
        return self._comp_of, self._comps

    # ── serialization ───────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialize to dict for checkpoint saving."""
        return {
            "config": {
                "n_max": self._n_max,
                "edge_games": self._edge_games,
                "win_threshold": self._win_threshold,
                "workers": self._workers,
                "single_head": self._single_head,
            },
            "nodes": {
                str(gen): {
                    "example_win_rate": nd["example_win_rate"],
                }
                for gen, nd in self.nodes.items()
            },
            "node_params": {
                str(gen): nd["params"].tolist()
                for gen, nd in self.nodes.items()
            },
            # edges: store as list of tuples for JSON compat
            "edges": [
                (a, b, rec["wins_a"], rec["wins_b"], rec["draws"])
                for (a, b), rec in self.edges.items()
            ],
        }

    def load_state_dict(self, d: dict):
        """Restore from state_dict. Nodes need params injected separately."""
        cfg = d["config"]
        self._n_max = cfg["n_max"]
        self._edge_games = cfg["edge_games"]
        self._win_threshold = cfg["win_threshold"]
        self._workers = cfg.get("workers", 2)
        self._single_head = cfg.get("single_head", True)

        self.nodes = {}
        for gen_str, nd in d["nodes"].items():
            gen = int(gen_str)
            # Restore params from saved data, fall back to empty array for backward compat
            params_raw = d.get("node_params", {}).get(gen_str)
            params = np.array(params_raw, dtype=np.float32) if params_raw else np.array([], dtype=np.float32)
            self.nodes[gen] = {
                "params": params,
                "example_win_rate": nd["example_win_rate"],
            }

        self.edges = {}
        for a, b, wa, wb, dr in d["edges"]:
            self.edges[(a, b)] = {"wins_a": wa, "wins_b": wb, "draws": dr}

        self._rebuild_scc()

    def inject_params(self, gen: int, params: np.ndarray):
        """Inject parameter vector for a node (after load_state_dict)."""
        if gen in self.nodes:
            self.nodes[gen]["params"] = params.copy()

    # ── internal ─────────────────────────────────────────────────────

    def _build_directed_edges(self) -> dict:
        """
        Convert raw edge records into directed edges based on win_threshold.
        Returns: adj dict {gen: set[gen]} (A→B means A beats B significantly)
        """
        adj = defaultdict(set)
        for (a, b), rec in self.edges.items():
            total = rec["wins_a"] + rec["wins_b"] + rec["draws"]
            if total == 0:
                continue
            wr_a = rec["wins_a"] / total
            if wr_a > self._win_threshold:
                adj[a].add(b)
            elif wr_a < 1.0 - self._win_threshold:
                adj[b].add(a)
        return adj

    def _rebuild_scc(self):
        """Rebuild SCC condensation from current directed edges."""
        adj = self._build_directed_edges()
        all_gens = list(self.nodes.keys())
        if len(all_gens) <= 1:
            self._comp_of = {g: 0 for g in all_gens}
            self._comps = {0: list(all_gens)}
            self._topo_order = [0]
            return

        self._comp_of, self._comps = _kosaraju(adj, all_gens)

        # Build DAG edges between SCCs
        dag_edges = set()
        for (a, b), rec in self.edges.items():
            if a not in self._comp_of or b not in self._comp_of:
                continue
            ca, cb = self._comp_of[a], self._comp_of[b]
            if ca != cb:
                dag_edges.add((ca, cb))

        self._topo_order = _topo_sort_sccs(dag_edges, set(self._comps.keys()))
