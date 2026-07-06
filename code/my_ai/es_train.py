"""Evolution Strategies trainer for Ant-Game AI.

Two modes:
  Standard (self-play): individuals play against opponent models from
    leaderboard / elite pool / current population.

  Multi-Expert (mixed opponent): individuals play against a MixedStrategyOpponent
    that samples actions from (RandomAgent / ExampleAgent / rule_v4) per turn.
    The (a, b, c) probabilities are dynamically adjusted to keep win rate ~55%.

Core algorithm (OpenAI-ES style):
  1. Sample noise -> create perturbed models: theta +/- sigma*eps
  2. Evaluate: each individual plays games (self-play or mixed opponent)
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
from functools import partial

import numpy as np
import torch

from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from my_ai.elite_bc import TopKSelector, BCDataset, supervised_update, write_bc_npz, collect_bc_data, cleanup_gen_npz, BCConfig
from my_ai.leaderboard import Leaderboard
from my_ai.mixed_opponent import MixedStrategyOpponent, adjust_probs
from utils.logger import get_logger


def select_opponents(
    population_size: int,
    games_per_individual: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Select K opponent indices shared by ALL individuals this generation.

    Returns a single list of `games_per_individual // 2` opponent indices.
    Each opponent is played twice (first/second player parity handled in step()).
    All individuals face the same opponents for fair rank-based fitness comparison.
    """
    assert games_per_individual % 2 == 0, "games_per_individual must be even (for fair first/second player swap)"
    n_pairs = games_per_individual // 2
    return rng.choice(population_size, size=n_pairs, replace=False).tolist()


def _eval_worker(
    params_flat: np.ndarray,
    opp_spec,  # np.ndarray (model params) for standard, tuple (probs) for mixed, or None
    seed: int,
    num_heads: int = 3,
    small: bool = False,
    synthetic_target: np.ndarray | None = None,
    bc_dir=None, gen=0, ind=0,
    *,
    allowed_classes: list[int] | None = None,
) -> dict:
    """Run one match: params vs opponent.

    ``opp_spec`` determines opponent type:
      - np.ndarray: load NeuralAgent from model params (standard self-play)
      - tuple/list: create MixedStrategyOpponent with those (a, b, c) probs
      - None: synthetic test (no game needed)

    If synthetic_target is provided, fitness = distance to target (no game).
    Otherwise, runs a full Ant-Game match.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # workers use CPU only (small model, avoid CUDA DLL memory)
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

    model = create_model(small=small, num_heads=num_heads)
    model.set_parameters_from_vector(params_flat)
    agent = NeuralAgent(model=model, allowed_classes=allowed_classes)

    if isinstance(opp_spec, np.ndarray):
        opp_model = create_model(small=small, num_heads=num_heads)
        opp_model.set_parameters_from_vector(opp_spec)
        opponent = NeuralAgent(model=opp_model)
    elif isinstance(opp_spec, (tuple, list)):
        opponent = MixedStrategyOpponent(probs=tuple(opp_spec))
    else:
        opponent = None  # synthetic test only

    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    bc_boards, bc_stats, bc_class_labels, bc_map_labels = [], [], [], []

    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops_us = agent._choose_operations(state, our_player)
        if bc_dir:
            feat = agent.feature_extractor.encode_observation(
                state, our_player, np.zeros(agent.max_actions))
            bc_boards.append(feat["board"].copy())
            bc_stats.append(feat["stats"].copy())
            map_arg = agent.last_output["action_map"].reshape(-1).argmax().item()
            cls_labels = [agent.last_output[f"head{hi+1}_logits"].argmax().item() for hi in range(num_heads)]
            bc_class_labels.append(cls_labels)
            bc_map_labels.append([map_arg] * num_heads)
        ops_opp = opponent._choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    if bc_dir and bc_boards:
        from my_ai.elite_bc import write_bc_npz
        boards_arr = np.stack(bc_boards, axis=0)
        stats_arr = np.stack(bc_stats, axis=0)
        write_bc_npz(Path(bc_dir) / f"gen_{gen:04d}_ind{ind:03d}_seed{seed}.npz",
                     [boards_arr], [stats_arr],
                     [np.array(bc_class_labels)], [np.array(bc_map_labels)])

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
        color = "\033[92m"   # green = win
    elif score == 0.0:
        color = "\033[91m"   # red = loss
    else:
        color = "\033[93m"   # yellow = draw
    reset = "\033[0m"
    print(f"{color}.{reset}", end="", flush=True)
    return result


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
        num_heads: int = 3,
        log=None,
        bc_config: BCConfig | None = None,
        small: bool = False,
    ):
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.momentum = momentum
        self.num_workers = num_workers
        self.games_per_individual = games_per_individual
        self.seed = seed
        self.num_heads = num_heads
        self.log = log
        self.bc_config = bc_config or BCConfig()
        self.bc_dir: Path | None = None
        self.rng = np.random.RandomState(seed)
        self.small = small
        self.allowed_classes: list[int] | None = None  # multi-expert: restrict action classes
        self.mixed_probs: tuple[float, float, float] = (0.33, 0.33, 0.34)  # mixed opponent probs (a,b,c)

        self.model = create_model(small=small, num_heads=num_heads)
        self.param_count = self.model.count_parameters()
        self.mean = self.model.get_parameters_as_vector()
        self.synthetic_target: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        # Elite pool: [(mean_vector, top1_vector), ...] from past generations
        # Used as opponents so learners face diverse strategies.
        self.elite_pool: list[tuple[np.ndarray, np.ndarray]] = []
        # Elite retention: top-k individuals kept across generations (pop_size^0.25)
        self.elite_params: list[np.ndarray] = []
        self.step_count = 0
        self.out_dir: str | None = None  # set by main() for config hot-reload
        self.win_graph = None  # WinGraph instance (optional, kept for backward compat)
        self.leaderboard = None  # Leaderboard instance (optional, set by main())

    def reload_config(self, config_path: str) -> bool:
        """Hot-reload training parameters from config.txt at generation boundary.

        Reads the config file and updates trainer attributes that can be changed
        mid-training (sigma, lr, momentum, generations, pop_size, games, workers).
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
            float_keys = {"sigma", "lr", "momentum"}
            int_keys = {"generations", "pop_size", "games", "workers"}
            attr_map = {
                "sigma": "sigma", "lr": "lr", "momentum": "momentum",
                "generations": "generations", "pop_size": "population_size",
                "games": "games_per_individual", "workers": "num_workers",
            }
            changed = False
            for file_key, attr in attr_map.items():
                if file_key in kv:
                    old = getattr(self, attr, None)
                    new = float(kv[file_key]) if file_key in float_keys else int(kv[file_key])
                    if old is not None and new != old:
                        setattr(self, attr, new)
                        changed = True
            if changed and self.log is not None:
                self.log.print(key="config_reload",
                               value=f"sigma={self.sigma} lr={self.lr} momentum={self.momentum} "
                                     f"pop={self.population_size} games={self.games_per_individual} workers={self.num_workers}")
            return changed
        except Exception:
            return False

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

        # Elite retention: replace bottom individuals with stored elites
        if self.elite_params:
            n_elite = max(1, int(self.population_size ** 0.25))
            for i in range(min(n_elite, len(self.elite_params))):
                params_list[-(i + 1)] = self.elite_params[i].copy()

        eval_start = time.time()

        k_per_ind = self.games_per_individual // 2

        # Determine opponent specification per match
        if self.allowed_classes is not None:
            # Multi-Expert mode: all individuals face the same MixedStrategyOpponent
            opp_spec = self.mixed_probs  # tuple (a, b, c)
        elif self.leaderboard is not None:
            opp_list = self.leaderboard.get_opponents(k=k_per_ind)
            if opp_list:
                opp_params_list = [entry["params"] for entry in opp_list]
            else:
                opp_indices = select_opponents(self.population_size, self.games_per_individual, self.rng)
                opp_params_list = [params_list[i] for i in opp_indices]
        elif self.elite_pool:
            opp_pool: list[np.ndarray] = []
            for entry in reversed(self.elite_pool):
                opp_pool.append(entry[0])  # mean
                opp_pool.append(entry[1])  # top1
            n_pool = len(opp_pool)
            gen_weights = np.array([0.85 ** (i // 2) for i in range(n_pool)], dtype=np.float64)
            gen_weights /= gen_weights.sum()
            replace = k_per_ind > n_pool
            selected_opps = self.rng.choice(n_pool, size=k_per_ind, p=gen_weights, replace=replace)
            opp_params_list = [opp_pool[i] for i in selected_opps]
        else:
            opp_indices = select_opponents(self.population_size, self.games_per_individual, self.rng)
            opp_params_list = [params_list[i] for i in opp_indices]

        # Build unified opp_spec_list: one entry per opponent pair
        if self.allowed_classes is not None:
            opp_spec_list: list = [self.mixed_probs] * k_per_ind
        else:
            opp_spec_list = opp_params_list  # type: ignore

        all_args = []
        for idx in range(self.population_size):
            for k in range(k_per_ind):
                opp_spec = opp_spec_list[k]
                base_seed = self.seed + generation * self.population_size * self.games_per_individual + (idx * self.games_per_individual + k * 2)
                all_args.append((
                    params_list[idx], opp_spec, base_seed,
                    self.num_heads, self.small, self.synthetic_target,
                    str(self.bc_dir) if self.bc_config.enabled else None,
                    generation,  # gen
                    idx,         # ind
                ))
                all_args.append((
                    params_list[idx], opp_spec, base_seed + 1,
                    self.num_heads, self.small, self.synthetic_target,
                    str(self.bc_dir) if self.bc_config.enabled else None,
                    generation,  # gen
                    idx,         # ind
                ))

        # Submit all tasks and wait with timeout polling (so Ctrl+C works on Windows)
        worker_fn = partial(_eval_worker, allowed_classes=self.allowed_classes)
        async_result = pool.starmap_async(worker_fn, all_args)
        while True:
            try:
                all_results = async_result.get(timeout=2)
                break  # Got all results
            except mp.TimeoutError:
                if getattr(pool, '_interrupted', False):
                    return {"generation": generation, "best_fitness": 0.0, "avg_fitness": 0.0,
                            "eval_time": 0.0, "total_time": 0.0}
                continue
            except (OSError, ValueError):
                if getattr(pool, '_interrupted', False):
                    return {"generation": generation, "best_fitness": 0.0, "avg_fitness": 0.0,
                            "eval_time": 0.0, "total_time": 0.0}
                raise

        # Parse results
        print()  # newline after eval dots
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

        if not self.bc_config.enabled:
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

        # Update elite retention buffer with current top performers
        n_elite = max(1, int(self.population_size ** 0.25))
        top_k_idx = np.argsort(fitness)[-n_elite:]
        new_elites = [params_list[i].copy() for i in reversed(top_k_idx)]
        # Alternate new and old elites so old ones survive if they're strong
        combined = []
        for i in range(max(len(new_elites), len(self.elite_params))):
            if i < len(new_elites):
                combined.append(new_elites[i])
            if i < len(self.elite_params):
                combined.append(self.elite_params[i])
        self.elite_params = combined[:n_elite]

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
        # Hot-reload config if config.txt was modified during training
        if self.out_dir is not None:
            self.reload_config(str(Path(self.out_dir) / "config.txt"))

        return result

    def save_checkpoint(self, path: str | Path, top2_params: list[np.ndarray] | None = None,
                        top2_scores: list[float] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "mean": torch.from_numpy(self.mean),
            "model_state": self.model.state_dict(),
            "num_heads": self.num_heads,
            "generation": self.step_count,
        }
        if top2_params is not None:
            data["top2_params"] = [torch.from_numpy(p) for p in top2_params]
            data["top2_scores"] = top2_scores
        if self.velocity is not None:
            data["velocity"] = torch.from_numpy(self.velocity)
        if self.elite_pool:
            data["elite_pool"] = [
                (torch.from_numpy(m), torch.from_numpy(t))
                for m, t in self.elite_pool
            ]
        if self.elite_params:
            data["elite_params"] = [torch.from_numpy(p) for p in self.elite_params]
        if self.win_graph is not None:
            data["win_graph"] = self.win_graph.state_dict()
        if self.leaderboard is not None:
            data["leaderboard"] = self.leaderboard.state_dict()
        if self.bc_config.enabled:
            data["bc_config"] = {"k": self.bc_config.k, "epochs": self.bc_config.epochs,
                                 "lr": self.bc_config.lr, "batch_size": self.bc_config.batch_size,
                                 "lambda_map": self.bc_config.lambda_map,
                                 "lambda_class": self.bc_config.lambda_class}
        if self.allowed_classes is not None:
            data["allowed_classes"] = self.allowed_classes
            data["mixed_probs"] = self.mixed_probs
        torch.save(data, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.mean = ckpt["mean"].numpy()
        self.model.set_parameters_from_vector(self.mean)
        if "velocity" in ckpt:
            self.velocity = ckpt["velocity"].numpy()
        if "elite_pool" in ckpt:
            self.elite_pool = [(m.numpy(), t.numpy()) for m, t in ckpt["elite_pool"]]
        if self.win_graph is not None and "win_graph" in ckpt:
            self.win_graph.load_state_dict(ckpt["win_graph"])
        if self.leaderboard is not None and "leaderboard" in ckpt:
            self.leaderboard.load_state_dict(ckpt["leaderboard"])
        # Backward compat: old single_head checkpoint → remap state_dict keys
        sd = ckpt.get("model_state", {})
        if "policy_head1.weight" in sd:
            import collections
            new_sd = collections.OrderedDict()
            single = "policy_head2.weight" not in sd  # single_head=True vs False
            for key, val in sd.items():
                if key.startswith("policy_head"):
                    parts = key.split(".")
                    idx = parts[0].replace("policy_head", "")
                    new_key = f"policy_heads.{int(idx)-1}.{parts[1]}"
                    new_sd[new_key] = val
                else:
                    new_sd[key] = val
            self.model.load_state_dict(new_sd)
            # If old single_head=True (1 head), we need to set num_heads=1 for correct param count
            if single:
                self.num_heads = 1
        if "elite_params" in ckpt:
            self.elite_params = [p.numpy() for p in ckpt["elite_params"]]
        # If loading from top1_init checkpoint, seed elite_params with it
        # to protect the strategy from being immediately replaced
        if "top2_params" in ckpt and not self.elite_params:
            top1 = ckpt["top2_params"][0]
            if isinstance(top1, np.ndarray):
                top1 = torch.from_numpy(top1)
            self.elite_params = [top1.numpy()]
        if "generation" in ckpt:
            self.step_count = ckpt["generation"]
        if "bc_config" in ckpt and self.bc_config.enabled:
            for k, v in ckpt["bc_config"].items():
                setattr(self.bc_config, k, v)
        if "allowed_classes" in ckpt:
            self.allowed_classes = ckpt["allowed_classes"]
        if "mixed_probs" in ckpt:
            self.mixed_probs = tuple(ckpt["mixed_probs"])


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
    parser.add_argument("--num-heads", type=int, default=3,
                        help="number of policy heads (default: 3; ignored when --small)")
    parser.add_argument("--small", action="store_true",
                        help="use small model (87K params, 1 head)")
    parser.add_argument("--checkpoint", type=str, default=None, help="resume from ES checkpoint")
    parser.add_argument("--load-bc", type=str, default=None,
                        help="load BC checkpoint as initialization (for cold start)")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-lb", action="store_true",
                        help="disable Leaderboard opponent selection (fall back to elite pool)")
    parser.add_argument("--no-wg", action="store_true",
                        help=argparse.SUPPRESS)  # kept for backward compat, no-op
    parser.add_argument("--synthetic-test", action="store_true",
                        help="synthetic fitness: converge toward random target (test ES correctness)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: training_history_<timestamp>)")
    parser.add_argument("--bc", action="store_true", help="enable Elite Behavior Cloning")
    parser.add_argument("--bc-k", type=int, default=5)
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--bc-lambda-map", type=float, default=1.0)
    parser.add_argument("--bc-lambda-class", type=float, default=1.0)
    parser.add_argument("--bc-device", type=str, default="cuda")
    parser.add_argument("--allowed-classes", type=int, nargs="+", default=None,
                        help="restrict agent to specific action classes (multi-expert training)")
    parser.add_argument("--opp-probs", type=float, nargs=3, default=[0.33, 0.33, 0.34],
                        metavar=("A", "B", "C"),
                        help="mixed opponent probs: random, example, rule_v4")
    args = parser.parse_args()

    # ── Prepare output directory ──────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("training_history")
    base_dir.mkdir(exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / f"{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────
    log = get_logger(out_dir / "train.log")
    from utils.logger import redirect_stderr_to_log
    redirect_stderr_to_log(log)
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
        num_heads=1 if args.small else args.num_heads,
        log=log,
        bc_config=BCConfig.from_args(args) if args.bc else None,
        small=args.small,
    )
    trainer.out_dir = str(out_dir)
    trainer.generations = args.generations  # allows hot-reload from config
    trainer.allowed_classes = args.allowed_classes
    trainer.mixed_probs = tuple(args.opp_probs)

    # ── BC data directory ─────────────────────────────────────────
    if args.bc:
        bc_dir = out_dir / "bc_data"
        bc_dir.mkdir(parents=True, exist_ok=True)
        trainer.bc_dir = bc_dir

    # ── Leaderboard (rank-weighted opponent pool) ──
    if args.allowed_classes is not None:
        # Multi-expert mode: no model-vs-model leaderboard needed
        log.print(key="leaderboard", value="auto-disabled (multi-expert mode)")
    elif not args.no_lb:
        trainer.leaderboard = Leaderboard(max_size=20, param_count=trainer.param_count, threshold=0.4)
        log.print(key="leaderboard", value=f"enabled (max_size=20)")
    else:
        log.print(key="leaderboard", value="disabled (fallback to elite pool)")

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    if args.load_bc:
        ckpt = torch.load(args.load_bc, map_location="cpu", weights_only=True)
        trainer.mean = ckpt["mean"].numpy()
        trainer.model.set_parameters_from_vector(trainer.mean)
        trainer.velocity = None  # reset momentum
        log.print(key="load_bc", value=args.load_bc)

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
    log.print(key="num_heads", value=trainer.num_heads)
    log.print(key="small", value=args.small)
    if args.allowed_classes is not None:
        log.print(key="allowed_classes", value=args.allowed_classes)
        log.print(key="opp_probs", value=f"a={args.opp_probs[0]:.2f} b={args.opp_probs[1]:.2f} c={args.opp_probs[2]:.2f}")
        log.print(key="opponents", value="MixedStrategyOpponent (dynamic)")
    else:
        log.print(key="opponents", value="Leaderboard (rank-weighted)" if not args.no_lb else "elite pool (self-play)")
    log.separator("-")

    # ── Print mode info ──────────────────────────────────────────
    if args.synthetic_test:
        mode_str = "synthetic test"
    elif args.allowed_classes is not None:
        mode_str = f"multi-expert (classes={args.allowed_classes})"
    else:
        mode_str = "real game (self-play)"
    log.print(key="mode", value=mode_str)
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
    log.print("  [PAUSE: write 'pause' into PAUSE file to pause, 'resume' to continue]")
    log.separator("-", width=50 if not args.synthetic_test else 65, timestamp=False)

    pool = mp.Pool(args.workers)
    interrupted = False
    pause_file = out_dir / "PAUSE"
    # Start in running state by default
    pause_file.write_text("resume", encoding="utf-8")
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

    start_gen = trainer.step_count
    try:
        for gen in range(start_gen, trainer.generations):
            if interrupted:
                break
            # Check PAUSE file content before starting a generation
            while pause_file.read_text(encoding="utf-8").strip().lower() == "pause" and not interrupted:
                time.sleep(2)
            if interrupted:
                break
            result = trainer.step(gen, pool)
            if interrupted:
                break

            # ── Elite Behavior Cloning ──
            if trainer.bc_config.enabled and trainer.bc_dir is not None:
                fitness_arr = np.array([d["score"] for d in result.get("ind_details", [])])
                if len(fitness_arr) > 0:
                    selector = TopKSelector(k=trainer.bc_config.k)
                    selected = selector.select(fitness_arr.tolist())
                    npz_paths = collect_bc_data(trainer.bc_dir, selected, gen)
                    if npz_paths:
                        ds = BCDataset(npz_paths)
                        if len(ds) > 0:
                            bc_ret = supervised_update(trainer.model, ds,
                                device=trainer.bc_config.device,
                                epochs=trainer.bc_config.epochs,
                                lr=trainer.bc_config.lr,
                                batch_size=trainer.bc_config.batch_size,
                                lambda_map=trainer.bc_config.lambda_map,
                                lambda_class=trainer.bc_config.lambda_class)
                            trainer.mean = trainer.model.get_parameters_as_vector()
                            log.print(key="bc",
                                value=f"top{selected} samples={bc_ret['samples']} "
                                      f"loss={bc_ret['total_loss']:.4f} "
                                      f"cls={bc_ret['class_loss']:.4f} map={bc_ret['map_loss']:.4f}")
                        else:
                            log.print(key="bc", value="all-HOLD filtered, no valid samples")
                    else:
                        log.print(key="bc", value="no data files for top-K")
                cleanup_gen_npz(trainer.bc_dir, gen)

            # Leaderboard challenge: mean vs strongest entry (skipped in multi-expert mode)
            if trainer.leaderboard is not None:
                strongest = trainer.leaderboard.get_strongest()
                if strongest is not None:
                    opp = strongest["params"]
                    match_tasks = [
                        (trainer.mean, opp, trainer.seed + 999999 + gen * 100 + s,
                         trainer.num_heads, trainer.small, None, None, gen, -1)
                        for s in range(trainer.games_per_individual)
                    ]
                    lb_scores = pool.starmap(_eval_worker, match_tasks)
                    wr = float(np.mean([r["score"] if isinstance(r, dict) else r for r in lb_scores]))
                    trainer.leaderboard.add_candidate(gen, trainer.mean.copy(), score=wr)
                else:
                    # First generation: leaderboard empty, always add
                    trainer.leaderboard.add_candidate(gen, trainer.mean.copy(), score=1.0)

            # Adjust mixed opponent probs based on this generation's win rate
            if trainer.allowed_classes is not None and result is not None:
                avg_win_rate = result["avg_fitness"]
                trainer.mixed_probs = adjust_probs(trainer.mixed_probs, avg_win_rate, target_rate=0.55)
                log.print(key="mixed_probs",
                          value=f"a={trainer.mixed_probs[0]:.3f} "
                                f"b={trainer.mixed_probs[1]:.3f} "
                                f"c={trainer.mixed_probs[2]:.3f}")

            trainer.step_count = gen + 1

            # Update elite pool with this generation's mean and top1
            top2 = result.get("top2_params")
            if top2 and len(top2) >= 1:
                trainer.elite_pool.append((trainer.mean.copy(), top2[0].copy()))
                # Keep at most 20 generations (40 opponents) for diversity
                max_gens = 20
                if len(trainer.elite_pool) > max_gens:
                    trainer.elite_pool.pop(0)

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
                    bkt = f"{int(d['score'] / step) * step:.3f}"
                    hist[bkt] = hist.get(bkt, 0) + 1
                # Only show non-empty buckets
                non_empty = {k: v for k, v in sorted(hist.items()) if v > 0}
                hist_str = " ".join(f"{k}:{v}" for k, v in non_empty.items())
                log.print_table(
                    **{"ind_dist": hist_str,
                       "best": f"#{best_i} {b['score']:.3f} "
                               f"[1st:{b['p0_w']}/{b['p0_n']} 2nd:{b['p1_w']}/{b['p1_n']}]"},
                )
                if trainer.leaderboard is not None:
                    lb_info = trainer.leaderboard.get_info()
                    log.print(key="lb",
                              value=f"entries={lb_info.get('size', 0)} "
                                    f"strongest={lb_info.get('strongest_score', 0):.3f}")

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

