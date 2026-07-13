"""
GA + SS hybrid training.

GA core (generate_population) — TODO user writes this.
Peripheral code (argparse, I/O, logging, eval loop, BC training, checkpoint) — handled here.

Usage:
    python -m my_ai.ga_ss_train --pop-size 20 --games 2 --generations 100 --workers 8
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

from my_ai.leaderboard import Leaderboard
from my_ai._eval_worker import _eval_worker

# ── Import boilerplate (written by AI) ────────────────────────────
from my_ai.ga_ss_boilerplate import (
    bc_train,
    merge_datasets,
    setup_output_dir,
    init_csv,
    write_csv_row,
    get_device,
    setup_signal_handler,
    mutate_class_labels,
    mutate_action_map,
    subsample_dataset,
)


# ═══════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GA + SS hybrid training. You write generate_population().",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # GA
    p.add_argument("--pop-size", type=int, default=20, help="population size")
    p.add_argument("--generations", type=int, default=100, help="number of generations")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--p-cross", type=float, default=0.5, help="crossover probability")
    p.add_argument("--p-mutate", type=float, default=0.1, help="mutation probability")
    p.add_argument("--temperature", type=float, default=3.0,
                    help="sampling temperature for label mutation (0 = uniform random)")
    p.add_argument("--amp", type=float, default=1.0,
                    help="mutation amplitude: multiply BC parameter shift (1=normal)")
    p.add_argument("--opp-inject", type=int, default=3,
                    help="number of opponent params to inject into population")

    # Eval
    p.add_argument("--games", type=int, default=2,
                    help="games per individual per opponent (even number; 2 = 1 as P0 + 1 as P1)")
    p.add_argument("--workers", type=int, default=8, help="multiprocessing workers")
    p.add_argument("--action-dropout", type=float, default=0.0,
                    help="action dropout rate during eval (data diversity)")
    p.add_argument("--p-hold", type=float, default=1.0,
                    help="HOLD frame retention ratio in SSDataset")

    # BC training
    p.add_argument("--k", type=int, default=5, help="top-K individuals for BC training")
    p.add_argument("--epochs", type=int, default=3, help="BC training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="BC learning rate")
    p.add_argument("--batch-size", type=int, default=64, help="BC batch size")
    p.add_argument("--weight-decay", type=float, default=1e-5,
                    help="AdamW weight decay (0 = Adam, >0 = AdamW)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing for CE loss")
    p.add_argument("--lambda-class", type=float, default=1.0, help="class CE loss weight")
    p.add_argument("--lambda-map", type=float, default=1.0, help="action-map KL loss weight")
    p.add_argument("--lambda-div", type=float, default=0.0, help="head diversity loss weight")
    p.add_argument("--lambda-soft", type=float, default=0.1, help="soft-target KL loss weight")
    p.add_argument("--bias-decay", type=float, default=0.0,
                    help="extra L2 penalty on policy head biases")
    p.add_argument("--oversample-alpha", type=float, default=0.0,
                    help="rare-class oversampling alpha (0 = disabled)")

    # Architecture
    p.add_argument("--num-heads", type=int, default=3, help="number of policy heads")
    p.add_argument("--small", action="store_true", help="use small model variant")

    # Leaderboard
    p.add_argument("--no-lb", action="store_true", help="disable Leaderboard")

    # Checkpoint / resume
    p.add_argument("--save-every", type=int, default=10,
                    help="save checkpoint every N generations")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="resume from checkpoint path")
    p.add_argument("--out-dir", type=str, default=None,
                    help="output directory (default: auto timestamp)")

    return p


# ═══════════════════════════════════════════════════════════════════
# GA core — YOU write this
# ═══════════════════════════════════════════════════════════════════

def crossover(
    elite_pop: list[np.ndarray],
    parent_idx: list[int],
    rng: np.random.Generator,
    args,
    datasets: list[SSDataset],
    model,
    device,
    log=None,
) -> np.ndarray:
    """Crossover function for GA population."""
    if rng.random() > args.p_cross or len(set(parent_idx)) == 1:
        return elite_pop[parent_idx[0]].copy()
    dataset = datasets[parent_idx[0]]
    for i in range(1, len(parent_idx)):
        dataset = merge_datasets(dataset, datasets[parent_idx[i]])
    bc_res = bc_train(
        init_params=elite_pop[parent_idx[0]],
        model_template=model,      # 需要传进来        
        dataset=dataset,
        device=device,             # 需要传进来        
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lambda_class=args.lambda_class,
        lambda_map=args.lambda_map,
        lambda_div=args.lambda_div,
        lambda_soft=args.lambda_soft,
        bias_decay=args.bias_decay,
        log=log,
    )
    return bc_res

def mutate_dataset(dataset: SSDataset, rng: np.random.Generator, gen: int, args) -> SSDataset:
    """Mutate class labels and action maps in an SSDataset.

    Args:
        ds: source SSDataset
        rng: random number generator
        gen: current generation (for seed)
        args: parsed CLI args

    Returns:
        New SSDataset with mutated class labels and action maps.
    """
    import torch
    from my_ai.ss_train import SSDataset

    # 把 SSDataset 里的标签转成 tensor 传入变异
    cls_tensor = torch.from_numpy(dataset.class_label)
    logits_tensor = torch.from_numpy(dataset.head_logits) 

    mutated_labels = mutate_class_labels(
        cls_tensor,
        logits_tensor,
        p_mutate=args.p_mutate,
        temperature=args.temperature,
        seed=int(rng.integers(2**31))
    )                                           
    map_tensor = torch.from_numpy(dataset.action_map)  
    # (T, 24, 19, 19)
    mutated_map = mutate_action_map(
        map_tensor,
        p_mutate=args.p_mutate,
        noise_std=0.05,      # 噪声强度，可调
        seed=args.seed + gen,
    )
    result = SSDataset.__new__(SSDataset)
    result.board = dataset.board
    result.stats = dataset.stats
    result.class_label = mutated_labels.numpy()
    result.action_map = mutated_map.numpy()
    result.head_logits = dataset.head_logits
    result.class_scores = dataset.class_scores     
    result.value = dataset.value
    return result

def mutation(
    ind: np.ndarray,
    rng: np.random.Generator,
    args,
    datasets: list[SSDataset],
    model,
    device,
    gen: int,
    log=None,
) -> np.ndarray:
    """Mutation function for GA population."""
    dataset = datasets[0]
    for i in range(1, len(datasets)):
        dataset = merge_datasets(dataset, datasets[i])
    dataset1 = subsample_dataset(dataset, 64, rng)
    dataset2 = subsample_dataset(dataset, 192, rng)
    dataset2 = mutate_dataset(dataset2, rng, gen, args)
    dataset = merge_datasets(dataset1, dataset2)
    bc_res = bc_train(
        init_params=ind,
        model_template=model,      # 需要传进来        
        dataset=dataset,
        device=device,             # 需要传进来        
        epochs=4,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lambda_class=args.lambda_class,
        lambda_map=args.lambda_map,
        lambda_div=args.lambda_div,
        lambda_soft=args.lambda_soft,
        bias_decay=args.bias_decay,
        log=log,
    )
    if args.amp != 1.0:
        bc_res = ind + (bc_res - ind) * args.amp
    return bc_res


def _select_opponents(
    leaderboard: Leaderboard | None,
    pop_size: int,
    games: int,
    rng: np.random.Generator,
    mean: np.ndarray,
    log=None,
) -> list[np.ndarray]:
    """Select opponent parameter vectors for evaluation.

    If Leaderboard is available, use rank-weighted sampling from it.
    Otherwise, use the mean as the sole opponent (self-play).

    Returns:
        list of opponent parameter vectors (np.ndarray).
    """
    if leaderboard is not None and len(leaderboard.entries) > 0:
        k = max(1, games // 2)
        opps = leaderboard.get_opponents(k=k)
        if log and opps:
            names = [f"#{e['gen']}" for e in opps]
            log.print(key="lb_opponents", value=f"Leaderboard opponents: {names}")
        return [e["params"] for e in opps]
    else:
        # Self-play: use mean as opponent
        if log:
            log.print(key="self_play", value="no Leaderboard, using self-play")
        return [mean.copy()]



# ════════════════════════════════════════════════
# Checkpoint
# ════════════════════════════════════════════════


def save_checkpoint(path, mean, model, generation, ga_pop=None, config=None, leaderboard=None):
    """Save training state to checkpoint file."""
    import torch
    data = {
        "mean": torch.from_numpy(mean),
        "model_state": model.state_dict(),
        "generation": generation,
        "num_heads": model.num_heads,
        # Compat shim: es_train tools expect top2_params
        "top2_params": [torch.from_numpy(mean.copy())],
        "top2_scores": [1.0],
    }
    if ga_pop is not None:
        data["ga_pop"] = ga_pop
    if config is not None:
        data["config"] = config
    if leaderboard is not None:
        data["leaderboard"] = leaderboard.state_dict()
    torch.save(data, path)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import torch
    from my_ai.network import create_model
    from my_ai.ss_train import (
        build_eval_args,
        run_eval,
        SSDataset,
        collect_npz,
        reload_config,
    )

    # ── Parse args ─────────────────────────────────────────────────
    parser = build_parser()
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # ── Setup directories and logging ──────────────────────────────
    out_dir, bc_dir, log = setup_output_dir(args)
    log.print(key="args", value=str(args))
    log.print(key="out_dir", value=str(out_dir))
    log.print(key="bc_dir", value=str(bc_dir))

    # ── Device ─────────────────────────────────────────────────────
    device = get_device()
    log.print(key="device", value=str(device))

    # ── Model + initial parameters ─────────────────────────────────
    model = create_model(num_heads=args.num_heads, small=args.small)
    param_count = model.get_parameters_as_vector().shape[0]
    log.print(key="param_count", value=f"{param_count:,}")
    mean = model.get_parameters_as_vector().copy()

    # ── Leaderboard ────────────────────────────────────────────────
    leaderboard = Leaderboard(max_size=20, param_count=param_count) if not args.no_lb else None
    if leaderboard is None:
        log.print(key="lb", value="Leaderboard disabled")

    # ── Elite storage ──────────────────────────────────────────────
    elite_saved: list[tuple[np.ndarray, float]] = []

    # ── Resume from checkpoint ─────────────────────────────────────
    start_gen = 0
    ga_pop = []
    if args.checkpoint:
        log.print(key="resume", value=f"loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        mean = ckpt["mean"].cpu().numpy().copy()
        model.set_parameters_from_vector(mean)
        ga_pop = ckpt.get("ga_pop", [])
        start_gen = ckpt.get("generation", 0)
        if leaderboard is not None and "leaderboard" in ckpt:
            leaderboard.load_state_dict(ckpt["leaderboard"])
        log.print(key="resume_gen", value=f"resuming from generation {start_gen}")

    # ── CSV ────────────────────────────────────────────────────────
    csv_header = [
        "gen", "best_fitness", "avg_fitness", "min_fitness",
        "lb_size",
    ]
    csv_path = out_dir / "history.csv"
    init_csv(csv_path, csv_header)

    # ── Signal handler ─────────────────────────────────────────────
    interrupted = [False]
    setup_signal_handler(interrupted, log)

    # ── Process pool ───────────────────────────────────────────────
    pool = mp.Pool(args.workers)
    log.print(key="pool", value=f"started {args.workers} workers")

    # ── PAUSE file (write "pause" to pause, "resume" to continue) ──
    pause_file = out_dir / "PAUSE"
    pause_file.write_text("resume", encoding="utf-8")

    # ── Main loop ──────────────────────────────────────────────────
    latest_result = None

    for gen in range(start_gen, args.generations):
        if interrupted[0]:
            log.print(key="interrupt", value=f"stopping at generation {gen}")
            break

        # PAUSE check
        while pause_file.read_text(encoding="utf-8").strip().lower() == "pause" and not interrupted[0]:
            time.sleep(2)
        if interrupted[0]:
            break

        log.separator("-")
        log.print(key="gen", value=f"{gen + 1}/{args.generations}")
        t0 = time.time()

        if not ga_pop:
            for i in range(args.pop_size):
                noise = rng.normal(0, 0.01, size=mean.shape).astype(np.float32)
                ga_pop.append(noise)

        # ── (1) Select opponents ──────────────────────────────────
        opp_params_list = _select_opponents(
            leaderboard, args.pop_size, args.games, rng, mean, log=log,
        )
        n_opp = len(opp_params_list)
        log.print(key="opponents", value=f"{n_opp} opponent(s)")

        # ── (2) Evaluate all individuals (Phase 1) ────────────────

        params_list = ga_pop
        log.print(key="pop_size", value=f"{len(params_list)}")

        log.print(key="eval", value="evaluating all individuals...")
        all_args = build_eval_args(
            params_list, opp_params_list,
            pop_size=len(params_list), games=args.games,
            num_heads=args.num_heads, bc_dir=str(bc_dir),
            gen=gen, seed=args.seed + gen,
            only_idx=None, seed_offset=0,
            action_dropout=args.action_dropout, small=args.small,
        )
        results = run_eval(pool, all_args)

        # Aggregate scores (results come in order: all games for ind 0, then ind 1, ...)
        n_games_per_ind = n_opp * 2  # 2 games per opponent (P0 + P1)
        scores_ind = np.array([r["score"] for r in results], dtype=np.float64)
        scores_ind = scores_ind.reshape(len(params_list), n_games_per_ind)
        fitness = scores_ind.mean(axis=1)

        best_idx = int(np.argmax(fitness))
        best_fit = float(fitness[best_idx])
        best_params = ga_pop[best_idx].copy()  # save before ga_pop is reassigned
        mean = best_params.copy()              # update mean for checkpoint / eval tools
        avg_fit = float(fitness.mean())
        min_fit = float(fitness.min())
        print("")
        log.print(key="fitness",
                  value=f"best={best_fit:.4f}  avg={avg_fit:.4f}  min={min_fit:.4f}")
        # Fitness distribution
        unique, counts = np.unique(np.round(fitness, 4), return_counts=True)
        dist_str = ", ".join(f"{u:.4f}:{c}" for u, c in zip(unique, counts))
        log.print(key="fitness_dist", value=dist_str)

        # ── (3) Select top-K, load BC data ────────────────────────
        top_k_idx = np.argsort(fitness)[-args.k:][::-1].tolist()
        log.print(key="top_k", value=f"indices={top_k_idx}")

        npz_paths = [collect_npz(bc_dir, [top_k_idx[i]], gen) for i in range(args.k)]
        datasets = [
            SSDataset(
                npz_paths[i],
                p_hold=args.p_hold,
                seed=args.seed + gen,
                oversample_alpha=args.oversample_alpha
            ) for i in range(args.k)
        ]

        # ── (4) Generate new population: elite + LB + crossover + mutation ──
        top_k_pop = [ga_pop[top_k_idx[i]] for i in range(args.k)]
        lb_pop = []
        for i in range(min(args.opp_inject, len(opp_params_list))):
            lb_pop.append(opp_params_list[i].copy())

        new_ga_pop = []
        parents_idx = []
        new_pop_size = max(0, args.pop_size - len(top_k_pop) - len(lb_pop))
        for i in range(new_pop_size):
            parents_idx.append([rng.integers(len(top_k_pop)), rng.integers(len(top_k_pop))])
        log.print(key="crossover", value=f"new_pop_size={new_pop_size}")
        for i in range(new_pop_size):
            # log.print(key="crossover", value=f"parents={parents_idx[i]}")
            new_ga_pop.append(crossover(
                top_k_pop,
                parents_idx[i],
                rng,
                args,
                datasets,
                model,
                device,
                log,
            ))
        log.print(key="mutation", value=f"new_pop_size={new_pop_size}")
        for i in range(new_pop_size):
            # print(f"mutation:ind={i}")
            # log.print(key="mutation", value=f"ind={i}")
            new_ga_pop[i] = mutation(
                new_ga_pop[i],
                rng,
                args,
                datasets,
                model,
                device,
                gen,
                log,
            )

        ga_pop = top_k_pop + lb_pop + new_ga_pop

        # ── (5) Leaderboard challenge ─────────────────────────────
        if leaderboard is not None:
            log.print(key="lb", value=f"challenging (pool={len(leaderboard.entries)})")

            def _vs_lb(me, opponent):
                match_tasks = [(me, opponent, args.seed + 999999 + gen * 100 + s,
                                args.num_heads, None, gen, -1,
                                0.0, args.small)
                               for s in range(args.games)]
                scores = [r["score"] if isinstance(r, dict) else r
                          for r in pool.starmap(_eval_worker, match_tasks)]
                wr = float(np.mean(scores))
                log.print(key="lb_wr", value=f"{wr:.3f} (threshold={leaderboard.threshold})")
                return wr

            added = leaderboard.add_candidate(gen, best_params, match_fn=_vs_lb)
            log.print(key="lb", value=f"{'added' if added else 'rejected'}")
            lb_size = len(leaderboard.entries)
            log.print(key="lb_size", value=f"{lb_size} entries")
        else:
            lb_size = 0

        # ── (6) CSV logging ──────────────────────────────────────
        elapsed = time.time() - t0
        write_csv_row(csv_path, [
            gen + 1,
            f"{best_fit:.6f}",
            f"{avg_fit:.6f}",
            f"{min_fit:.6f}",
            lb_size,
        ])
        log.print(key="time", value=f"{elapsed:.1f}s")
        log.print(key="csv", value=str(csv_path))

        # ── (7) Save checkpoint ──────────────────────────────────
        if (gen + 1) % args.save_every == 0:
            ckpt_path = out_dir / f"gen_{gen + 1:04d}.pt"
            save_checkpoint(
                ckpt_path, mean, model,
                generation=gen + 1,
                ga_pop=ga_pop,
                config=vars(args),
                leaderboard=leaderboard,
            )
            log.print(key="checkpoint", value=str(ckpt_path))

        # ── (8) Hot-reload config ────────────────────────────────
        config_path = str(out_dir / "config.txt")
        changed = reload_config(config_path, args)
        if changed:
            log.print(key="reload", value="config hot-reloaded")

        latest_result = {
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "min_fitness": min_fit,
        }

    # ── Cleanup ──────────────────────────────────────────────────
    pool.terminate()
    pool.join()

    # ── Final checkpoint ───────────────────────────────────────────
    if interrupted[0]:
        ckpt_path = out_dir / f"interrupt_gen_{gen + 1:04d}.pt"
        save_checkpoint(ckpt_path, mean, model,
                        gen + 1, ga_pop=ga_pop,
                        leaderboard=leaderboard)
        log.print(key="interrupt_checkpoint", value=str(ckpt_path))
    else:
        save_checkpoint(out_dir / "final.pt", mean, model, args.generations,
                        ga_pop=ga_pop, leaderboard=leaderboard)
        log.print(key="final_checkpoint", value=str(out_dir / "final.pt"))

    # ── Summary ────────────────────────────────────────────────────
    log.separator("=")
    if latest_result is not None and not interrupted[0]:
        log.print(key="best_fitness", value=f"{latest_result['best_fitness']:.4f}")
        log.print(key="avg_fitness", value=f"{latest_result['avg_fitness']:.4f}")
    log.print(key="history", value=str(csv_path))
    log.print("Done.")


if __name__ == "__main__":
    main()
