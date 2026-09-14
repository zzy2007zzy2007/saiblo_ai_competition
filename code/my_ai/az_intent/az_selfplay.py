"""Self-play collection for bundle-MCTS AlphaZero training (T1).

Both players use the bundle MCTS.  At each decision we record a training sample:
board/stats (inputs), player, legal masks, the k sampled bundles with their
per-head intent sample counts, the temperature-scaled visit distribution, the
network's recorded policy outputs (anchor for §3.2), and the terminal
HP-difference value target.

Samples are saved per game (pickle) so training can stream / load them.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_model_from_ckpt(ckpt_path: str):
    from my_ai.network import create_model, model_kwargs_from_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_model(**model_kwargs_from_ckpt(ckpt))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_split_models(ckpt_path: str):
    """Load a split checkpoint into (policy_model, value_model).

    Both networks share the same architecture.  For a single-model checkpoint
    (model_state only, e.g. gen0120_warm) both nets warm-start from the same
    weights; for a split checkpoint (has ``value_state``) each loads its own.
    """
    from my_ai.network import create_model, model_kwargs_from_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _kw = model_kwargs_from_ckpt(ckpt)
    # A mixed checkpoint may record a separate normalization for the value net
    # (e.g. GroupNorm value + BatchNorm policy); older checkpoints omit these
    # keys and both nets share the policy's config.
    _vkw = dict(_kw)
    if "value_no_bn" in ckpt or "value_gn" in ckpt:
        _vkw["no_bn"] = bool(ckpt.get("value_no_bn", False))
        _vkw["gn"] = bool(ckpt.get("value_gn", False))
        _vkw["gn_groups"] = int(ckpt.get("value_gn_groups", 8))
    _vkw["value_pool"] = str(ckpt.get("value_value_pool", _kw.get("value_pool", "gap")))

    def _build():
        return create_model(**_kw)

    def _build_value():
        return create_model(**_vkw)

    policy_model = _build()
    policy_model.load_state_dict(ckpt["model_state"])
    value_model = _build_value() if "value_state" in ckpt else _build()
    value_model.load_state_dict(ckpt.get("value_state", ckpt["model_state"]))
    policy_model.eval()
    value_model.eval()
    return policy_model, value_model


def make_net_fn_from_ckpt(ckpt_path: str, feature_extractor, value_tanh: bool = True):
    """Build (anchor_model, net_fn) for a checkpoint — single or split.

    ``anchor_model`` is the policy network (used to record the anchor outputs
    during self-play); ``net_fn`` is the search interface (one or two forwards).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "value_state" in ckpt:
        from my_ai.az_intent.train import make_split_net_fn
        policy_model, value_model = load_split_models(ckpt_path)
        return policy_model, make_split_net_fn(policy_model, value_model, feature_extractor,
                                               value_tanh=value_tanh)
    from my_ai.az_intent.train import make_net_fn
    model = load_model_from_ckpt(ckpt_path)
    return model, make_net_fn(model, feature_extractor, value_tanh=value_tanh)


def make_initial_state(seed: int, native_engine: bool = False):
    """Create a fresh game state — pure-Python engine or the C++ engine.

    The C++ engine (``native_game`` + GameStateFacade) matches the official game
    binary; the Python SDK has known detail deviations.  Switching engines
    changes the simulated game, so results are only comparable within the same
    engine.  The C++ engine's advance_round is ~22x faster (MCTS bottleneck).
    """
    if native_engine:
        cpp_dir = Path(__file__).resolve().parents[2] / "cpp_engine"
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))
        from my_ai.az_intent.game_state_facade import GameStateFacade
        return GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    from SDK.backend.engine import GameState
    return GameState.initial(seed=seed, cold_handle_rule_illegal=True)


def _fmt_op(t: int, a0: int, a1: int) -> str:
    """动作命名表示（与 code/test_match/diagnose.py 的 op_desc 一致），如 BUILD(6,9)。"""
    from SDK.utils.constants import OperationType
    name = {
        OperationType.BUILD_TOWER: "BUILD",
        OperationType.UPGRADE_TOWER: "UPGRADE",
        OperationType.DOWNGRADE_TOWER: "DOWNGRADE",
        OperationType.USE_LIGHTNING_STORM: "LIGHTNING",
        OperationType.USE_EMP_BLASTER: "EMP",
        OperationType.USE_DEFLECTOR: "DEFLECTOR",
        OperationType.USE_EMERGENCY_EVASION: "EVASION",
        OperationType.UPGRADE_GENERATION_SPEED: "UP_SPEED",
        OperationType.UPGRADE_GENERATED_ANT: "UP_ANT_HP",
    }.get(OperationType(t))
    if name is None:
        return f"OP_{t}"
    if t == OperationType.BUILD_TOWER:
        return f"{name}({a0},{a1})"
    if t == OperationType.UPGRADE_TOWER:
        return f"{name}(id={a0}->type={a1})"
    if t == OperationType.DOWNGRADE_TOWER:
        return f"{name}(id={a0})"
    if OperationType.USE_LIGHTNING_STORM <= t <= OperationType.USE_EMERGENCY_EVASION:
        return f"{name}({a0},{a1})"
    return name


def _random_legal_bundle(model, out, state, player, rng, pm, cm, n_samples: int = 12):
    """A uniformly-random LEGAL bundle: uniform class dist per head (position
    sampling made ~uniform via a huge t_pos), then one bundle picked at random.
    """
    from my_ai.az_intent.bundle_mcts import sample_bundle

    head_logits = np.stack([out[f"head{i + 1}_logits"].squeeze(0).numpy()
                            for i in range(model.num_heads)])
    net_out = {"action_map": out["action_map"].squeeze(0).numpy(),
               "head_logits": head_logits}
    uni = []
    for h in range(model.num_heads):
        n_cls = head_logits[h].shape[-1]
        uni.append(np.ones(n_cls, dtype=np.float32) / n_cls)
    class_ids = [rng.choice(len(p), size=n_samples, p=p).tolist() for p in uni]
    bundles = []
    for j in range(n_samples):
        ops_j, _ = sample_bundle(
            net_out, state, player, t_class=1.0, t_pos=1e6, rng=rng,
            position_mask=pm, class_mask=cm,
            class_probs=uni, class_ids=[ci[j] for ci in class_ids])
        if ops_j:
            bundles.append(tuple((int(o.op_type), o.arg0, o.arg1) for o in ops_j))
    if not bundles:
        return None
    return bundles[int(rng.integers(len(bundles)))]


def collect_game(net_fn, model, feature_extractor, mcts, seed, *,
                 max_rounds: int = 512, temp_rounds: int = 30,
                 progress_path: str | None = None,
                 native_engine: bool = False,
                 random_action_prob: float = 0.0,
                 skip_hold_search: bool = False) -> list[dict]:
    """Play one self-play game (both sides = bundle MCTS), record training samples.

    If ``progress_path`` is given, a per-round text log (round index + each
    player's chosen operations) is written there and flushed every round, so the
    game's progress can be watched while it runs (games take ~2h at 128/depth4).

    ``random_action_prob`` (x): with probability x a decision ignores the
    search's choice and plays a UNIFORMLY random legal bundle (uniform over
    class + position within the legal masks).  Purpose: break the
    two-equal-searchers grind so games end with larger HP differences.  Keep x
    small (~0.05): random blunders are symmetric, so a small rate adds decisive
    outcomes without destroying the state->outcome signal.  The recorded policy
    target (visit distribution) is still the search's, so only the played move
    and the resulting states change.
    """
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.decoder import make_class_mask, make_position_masks
    from my_ai.az_intent.mcts import HP_SCALE

    state = make_initial_state(seed, native_engine)
    samples: list[dict] = []
    pfile = open(progress_path, "w", encoding="utf-8") if progress_path else None

    for round_idx in range(max_rounds):
        if state.terminal:
            break
        temperature = 1.0 if round_idx < temp_rounds else 1e-6
        round_ops: list[list] = [[], []]
        for player in (0, 1):
            if state.terminal:
                break
            # recorded policy outputs (anchor) + masks (training denominators)
            obs = feature_extractor.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            recorded_am = out["action_map"].squeeze(0).numpy().astype(np.float16)
            recorded_hl = np.stack(
                [out[f"head{i + 1}_logits"].squeeze(0).numpy()
                 for i in range(model.num_heads)]
            ).astype(np.float16)
            pm = make_position_masks(state, player, intent_decoding=True)
            cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)

            # HOLD-skip optimisation: the MCTS is the bottleneck (advance_round), and
            # ~91% of decisions end up HOLD — almost always because every action class
            # is masked out (cooldown / no valid position / unaffordable).  When the
            # *legal* class argmax is HOLD for every head the search can only pick HOLD
            # too (measured agreement: 2 wrong skips in 10,002 decisions), so skip it.
            skip = False
            if skip_hold_search:
                votes = []
                for i in range(model.num_heads):
                    lg = out[f"head{i + 1}_logits"].squeeze(0).numpy().astype(np.float64)
                    lg = np.where(cm, lg, -1e9)
                    votes.append(int(np.argmax(lg)))
                skip = all(v == 23 for v in votes)  # 23 = HOLD
            if skip:
                res = None
                chosen = ()
                bundles, intent_counts, visit = [()], None, np.ones(1, dtype=np.float32)
            else:
                res = mcts.search(state, player, temperature=temperature)
                chosen = res.chosen_bundle
                bundles, intent_counts, visit = (list(res.bundles), res.intent_counts,
                                                 res.visit_policy.astype(np.float32))
                if random_action_prob > 0.0 and mcts.rng.random() < random_action_prob:
                    rb = _random_legal_bundle(model, out, state, player, mcts.rng, pm, cm)
                    if rb:
                        chosen = rb
            samples.append(
                {
                    "board": obs["board"].astype(np.float16),
                    "stats": obs["stats"].astype(np.float16),
                    "player": player,
                    "class_mask": cm,
                    "position_mask": pm,
                    "bundles": bundles,
                    "intent_counts": intent_counts,
                    "visit": visit,
                    "recorded_action_map": recorded_am,
                    "recorded_head_logits": recorded_hl,
                    "search_skipped": skip,
                }
            )
            ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in chosen
            ]
            round_ops[player] = [(int(o.op_type), o.arg0, o.arg1) for o in ops]
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()
        if pfile is not None:
            hp0 = state.bases[0].hp
            hp1 = state.bases[1].hp
            g0 = state.coins[0]
            g1 = state.coins[1]
            p0 = "[" + ", ".join(_fmt_op(*op) for op in round_ops[0]) + "]" if round_ops[0] else "[]"
            while len(p0) < 32:
                p0 += " "
            p1 = "[" + ", ".join(_fmt_op(*op) for op in round_ops[1]) + "]" if round_ops[1] else "[]"
            pfile.write(f"round={round_idx} HP0={hp0} HP1={hp1} G0={g0} G1={g1} "
                        f"P0={p0} P1={p1}\n")
            pfile.flush()

    diff = state.bases[0].hp - state.bases[1].hp
    v_p0 = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
    if diff == 0 and state.winner is not None:
        v_p0 = 0.1 if state.winner == 0 else -0.1
    for s in samples:
        s["value_target"] = v_p0 if s["player"] == 0 else -v_p0
    if pfile is not None:
        pfile.write(f"terminal winner={state.winner} hp={[b.hp for b in state.bases]} "
                    f"rounds={state.round_index}\n")
        pfile.close()
    return samples


def _collect_and_save(seed: int, out_dir: str, ckpt_path: str, iterations: int,
                      max_depth_rounds: int, t_class: float, t_pos: float,
                      k: int, sample_mult: int, max_rounds: int, temp_rounds: int,
                      native_engine: bool = False, c_puct: float = 1.25,
                      random_action_prob: float = 0.0,
                      skip_hold_search: bool = False,
                      write_npz: bool = False) -> dict:
    torch.set_num_threads(1)  # avoid thread thrash across parallel workers
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    model, net_fn = make_net_fn_from_ckpt(ckpt_path, feat)
    mcts = BundleMCTS(net_fn, iterations=iterations, max_depth_rounds=max_depth_rounds,
                      k=k, sample_mult=sample_mult, t_class=t_class, t_pos=t_pos,
                      c_puct=c_puct, seed=seed)
    samples = collect_game(net_fn, model, feat, mcts, seed,
                           max_rounds=max_rounds, temp_rounds=temp_rounds,
                           progress_path=str(Path(out_dir) / f"az_progress_seed{seed:05d}.txt"),
                           native_engine=native_engine,
                           random_action_prob=random_action_prob,
                           skip_hold_search=skip_hold_search)
    path = Path(out_dir) / f"az_selfplay_seed{seed:05d}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_npz:
        # Light value-only format: drop bundles/visit/intent_counts/masks (heavy,
        # only needed for policy training).  The recorded anchors are kept AS-IS —
        # no re-forwarding — because they come from THIS collection model, which
        # for the iterative loop is the (fixed) policy we keep.
        import numpy as _np
        npz = path.with_suffix(".npz")
        if samples:
            _np.savez_compressed(
                npz,
                board=_np.stack([s["board"] for s in samples], axis=0),
                stats=_np.stack([s["stats"] for s in samples], axis=0),
                action_map=_np.stack([s["recorded_action_map"] for s in samples], axis=0),
                head_logits=_np.stack([s["recorded_head_logits"] for s in samples], axis=0),
                player=_np.asarray([s["player"] for s in samples], dtype=_np.int8),
                value_target=_np.asarray([s["value_target"] for s in samples], dtype=_np.float32),
            )
        path = npz
    else:
        with open(path, "wb") as f:
            pickle.dump({"seed": seed, "samples": samples}, f)
    print(f"  [collect] seed={seed} samples={len(samples)} -> {path.name}", flush=True)
    return {"seed": seed, "samples": len(samples)}


def collect_games_parallel(ckpt_path: str, seeds: list[int], out_dir: str, workers: int,
                           iterations: int, max_depth_rounds: int, t_class: float,
                           t_pos: float, k: int, sample_mult: int, max_rounds: int,
                           temp_rounds: int, native_engine: bool = False,
                           c_puct: float = 1.25,
                           random_action_prob: float = 0.0,
                           skip_hold_search: bool = False,
                           write_npz: bool = False) -> list[Path]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(s, out_dir, ckpt_path, iterations, max_depth_rounds, t_class, t_pos,
             k, sample_mult, max_rounds, temp_rounds, native_engine, c_puct,
             random_action_prob, skip_hold_search, write_npz) for s in seeds]
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.starmap(_collect_and_save, jobs)
    else:
        results = [_collect_and_save(*j) for j in jobs]
    total = sum(r["samples"] for r in results)
    print(f"[selfplay] collected {total} samples from {len(results)} games", flush=True)
    return [Path(out_dir) / f"az_selfplay_seed{s:05d}.pkl" for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle-MCTS AlphaZero self-play collection (T1)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="training_history/az_intent/az_selfplay_data")
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--max-depth-rounds", type=int, default=4)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--sample-mult", type=int, default=15)
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--temp-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-engine", action="store_true",
                        help="use the C++ engine (native_game) instead of the Python SDK engine")
    parser.add_argument("--random-action-prob", type=float, default=0.0,
                        help="x: with probability x a decision plays a uniformly random "
                             "LEGAL bundle instead of the search's choice (diversity / "
                             "decisive-game injection; keep small, e.g. 0.05)")
    parser.add_argument("--c-puct", type=float, default=1.25,
                        help="MCTS PUCT exploration constant (higher = more exploration)")
    parser.add_argument("--write-npz", action="store_true",
                        help="write a light value-only npz per game instead of the heavy pkl "
                             "(drops bundles/visit/masks; keeps the recorded anchors AS-IS). "
                             "Used by the iterative loop: az_selfplay npz -> merge -> train")
    parser.add_argument("--skip-hold-search", action="store_true",
                        help="skip the MCTS at decisions where the LEGAL class argmax is "
                             "HOLD for every head (the search can only pick HOLD there too; "
                             "measured 2 wrong skips in 10,002 decisions).  ~91%% of decisions "
                             "qualify, cutting collection cost by ~10x.  Value-only use.")
    args = parser.parse_args()

    seeds = [args.seed * 10000 + g for g in range(args.games)]
    print(f"[selfplay] collecting {args.games} games ({args.workers} workers, "
          f"{args.iterations} iters / depth {args.max_depth_rounds}) "
          f"[engine={'C++' if args.native_engine else 'python'}, c_puct={args.c_puct}, "
          f"random_action_prob={args.random_action_prob}]...", flush=True)
    paths = collect_games_parallel(
        args.checkpoint, seeds, args.out_dir, args.workers,
        args.iterations, args.max_depth_rounds, args.t_class, args.t_pos,
        args.k, args.sample_mult, args.max_rounds, args.temp_rounds,
        native_engine=args.native_engine, c_puct=args.c_puct,
        random_action_prob=args.random_action_prob,
        skip_hold_search=args.skip_hold_search,
        write_npz=args.write_npz,
    )
    print(f"[selfplay] done -> {len(paths)} files", flush=True)


if __name__ == "__main__":
    main()
