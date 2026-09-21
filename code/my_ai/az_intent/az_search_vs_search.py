"""Search-vs-search comparison of two checkpoints (full system test).

Both sides play with the bundle MCTS at the same config (default 128 iters /
depth 4, t_class 0.5 / t_pos 0.3 — matching self-play).  Alternates which
model is P0, reports A's win rate.  This is the definitive test of whether
AlphaZero training improved the full policy+value system.

Usage:
    python code/my_ai/az_intent/az_search_vs_search.py \
        --a .../az_az3.pt --b .../gen0120_warm.pt \
        --games 6 --workers 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from collections import Counter as _Counter  # 汇总打印用（_worker 内部另有一份局部 import）


def _ban_classes(net_fn, banned):
    """把给定类的 head logits 压到 -1e9 ⇒ 采样/argmax 永远不会选它们。

    用途（用户 2026-09-21）：**禁用闪电（类 17）**，让 joint 搜索只能从其余合法类里选，
    从而真正读到位置网的其他通道（`pos-only` 会把类钉死在类网 argmax=17 上，读不到）。
    head_logits 在 net_fn 里是"每个头一个 numpy 数组"的列表。"""
    if not banned:
        return net_fn
    ban = [int(c) for c in banned]

    def f(state, player):
        out = net_fn(state, player)
        hl = [h.copy() for h in out["head_logits"]]
        for arr in hl:
            for c in ban:
                arr[c] = -1e9
        return {**out, "head_logits": hl}

    return f


def _worker(args: tuple) -> dict:
    (a_path, b_path, seed, iterations, max_depth_rounds, t_class, t_pos,
     native_engine, flip, a_rel_abs, b_rel_abs, a_tanh, b_tanh,
     search_mode, skip_single, k, ban, a_prior, b_prior) = args
    import torch
    torch.set_num_threads(1)

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, make_initial_state
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    # Each side gets its OWN value-readout config: a head trained on RELATIVE
    # labels needs raw/scale + d_t/HP_SCALE (and tanh off), while an absolute-label
    # head is used as-is.  Without this the comparison would be unfair.
    _, net_fn_a = make_net_fn_from_ckpt(a_path, feat, value_tanh=a_tanh,
                                        value_rel_to_abs=a_rel_abs)
    _, net_fn_b = make_net_fn_from_ckpt(b_path, feat, value_tanh=b_tanh,
                                        value_rel_to_abs=b_rel_abs)
    # search_mode / skip_single_candidate（2026-09-19 新增，默认 = 原行为）：
    # 部署/采集口径是 `pos-only + skip-single-candidate`（实测与 joint 强度逐位相同、
    # 便宜 ~26x，因为 pos-only 下 ~97% 回合只有 1 个候选）。见 docs/az_forced_move_skip_plan.md
    _m = dict(iterations=iterations, max_depth_rounds=max_depth_rounds, k=k,
              t_class=t_class, t_pos=t_pos, search_mode=search_mode,
              skip_single_candidate=skip_single)
    def _make_prior(mode, ckpt_path, net_fn):
        """逐侧位置先验：'policy'=用该侧位置网自己的图（默认，无钩子）；'value'=根节点上换成
        1-ply 价值网的 z-score；'uniform'=同样的枚举但权重全等（对照）。

        'value' 用的价值网取自**该侧 ckpt 自己的 value_state** ⇒ 两侧必须是同一份价值网
        （本项目里 `posnet_*` 与 `posnet_r1` 都可挂同一个新价值网，或用 `--ckpt-our/--ckpt-opp`
        之外的拼法保证）。见 docs/az_posnet_v2_plan.md §5.1。"""
        if mode == "policy":
            return None
        from my_ai.az_intent.eval import _make_value_pos_prior
        vm = None
        if mode == "value":
            from my_ai.az_intent.az_selfplay import load_three_models
            _, _, vm = load_three_models(ckpt_path)
        return _make_value_pos_prior(feat, vm, mode=mode)

    net_fn_a = _ban_classes(net_fn_a, ban)
    net_fn_b = _ban_classes(net_fn_b, ban)
    mcts_a = BundleMCTS(net_fn_a, seed=seed, pos_prior_fn=_make_prior(a_prior, a_path, net_fn_a), **_m)
    mcts_b = BundleMCTS(net_fn_b, seed=seed + 1,
                        pos_prior_fn=_make_prior(b_prior, b_path, net_fn_b), **_m)

    # alternate which model is P0; --flip-sides inverts the assignment so the SAME
    # seed can be replayed with swapped sides -> pairs with the unflipped run
    from collections import Counter
    cls_cnt = {"A": Counter(), "B": Counter()}
    a_is_p0 = (seed % 2 == 0) != bool(flip)
    mcts = [mcts_a, mcts_b] if a_is_p0 else [mcts_b, mcts_a]
    state = make_initial_state(seed, native_engine)
    for _ in range(512):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            res = mcts[player].search(state, player, temperature=0.0)
            ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in res.chosen_bundle
            ]
            who = ("A" if a_is_p0 else "B") if player == 0 else ("B" if a_is_p0 else "A")
            for op in ops:
                cls_cnt[who][OperationType(int(op.op_type)).name] += 1
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()

    # score from model_a's perspective
    a_player = 0 if a_is_p0 else 1
    if state.terminal and state.winner is not None:
        if state.winner == a_player:
            score = 1.0
        elif state.winner == 1 - a_player:
            score = 0.0
        else:
            score = 0.5
    else:
        hp_a = state.bases[a_player].hp
        hp_b = state.bases[1 - a_player].hp
        score = 0.5 if hp_a == hp_b else (1.0 if hp_a > hp_b else 0.0)
    tag = {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[score]
    # 每局就把"两侧各自出了哪些类"打出来 ⇒ 跑几分钟就能判断 joint+禁类 有没有真的读到
    # 位置网的其他通道（只有覆盖率够，这个台子才测得到"多类记录"的效果）。
    _comp = {w: ",".join("%s%d" % (k[:3], v) for k, v in cls_cnt[w].most_common(5))
             for w in ("A", "B")}
    print(f"  seed={seed:3d} {tag} ({'A=P0' if a_is_p0 else 'A=P1'}) "
          f"hp={[b.hp for b in state.bases]} rounds={state.round_index} | "
          f"A[{_comp['A']}] B[{_comp['B']}]", flush=True)
    return {"score": score, "rounds": int(state.round_index), "terminal": bool(state.terminal),
            "cls": cls_cnt}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search-vs-search comparison (full system)")
    parser.add_argument("--a", required=True, help="model A (e.g. trained)")
    parser.add_argument("--b", required=True, help="model B (e.g. pre-training)")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--max-depth-rounds", type=int, default=4)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=1.0,
                        help="position sampling temperature (z-scored over legal cells); "
                             "see docs/az_t_pos_default_fix.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-engine", action="store_true",
                        help="use the C++ engine (native_game) for the game simulation")
    parser.add_argument("--a-rel-to-abs", type=float, default=0.0,
                        help="A: if A's value head was trained on RELATIVE labels, pass its "
                             "label_scale here (value = raw/scale + d_t/HP_SCALE)")
    parser.add_argument("--b-rel-to-abs", type=float, default=0.0, help="same, for B")
    parser.add_argument("--a-no-tanh", action="store_true", help="A: keep raw value (no tanh)")
    parser.add_argument("--b-no-tanh", action="store_true", help="B: keep raw value (no tanh)")
    parser.add_argument("--pos-prior-a", type=str, default="policy",
                        choices=["policy", "value", "uniform"],
                        help="A 侧的位置先验来源：policy=位置网自己的图（默认）；value=**1-ply 价值网**"
                             "在根节点上的 z-score（不蒸馏的形态）；uniform=同样枚举但等权（对照）")
    parser.add_argument("--pos-prior-b", type=str, default="policy",
                        choices=["policy", "value", "uniform"], help="B 侧同上")
    parser.add_argument("--ban-class", type=int, nargs="+", default=None,
                        help="禁用这些类（把 head logits 压到 -1e9）。用途：--ban-class 17 禁闪电，"
                             "让 joint 搜索去读位置网的其他通道；见 docs/az_posnet_v2_plan.md §5")
    parser.add_argument("--search-mode", type=str, default="joint",
                        choices=["joint", "class-only", "pos-only"],
                        help="传给 BundleMCTS。pos-only = 类钉在类头 argmax（= 部署/采集口径）")
    parser.add_argument("--skip-single-candidate", action="store_true",
                        help="单候选时跳过迭代（pos-only 下 ~97%% 回合命中，便宜 ~26x）")
    parser.add_argument("--k", type=int, default=24, help="每层候选数（BundleMCTS k）")
    parser.add_argument("--pairs", type=int, default=0,
                        help=">0：**真镜像配对**——每个 seed 各打一次 A=P0 与 A=P1，"
                             "报配对胜率 + SE + t（与 --games 二选一）")
    parser.add_argument("--flip-sides", action="store_true",
                        help="invert the seed%%2 side assignment. Running the SAME "
                             "seeds with this flag replays every game with swapped "
                             "sides, turning the two runs into 64 matched PAIRS "
                             "(cancels the P0/P1 asymmetry within each pair)")
    args = parser.parse_args()

    _common = (args.iterations, args.max_depth_rounds, args.t_class, args.t_pos,
               args.native_engine, args.a_rel_to_abs, args.b_rel_to_abs,
               not args.a_no_tanh, not args.b_no_tanh,
               args.search_mode, args.skip_single_candidate, args.k, args.ban_class,
               args.pos_prior_a, args.pos_prior_b)
    if args.pairs > 0:
        jobs = [(args.a, args.b, args.seed + s) + _common[:5] + (fl,) + _common[5:]
                for s in range(args.pairs) for fl in (False, True)]
    else:
        jobs = [(args.a, args.b, args.seed + s) + _common[:5] + (args.flip_sides,)
                + _common[5:] for s in range(args.games)]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    scores = [r["score"] for r in results]
    win = sum(1 for s in scores if s == 1.0)
    draw = sum(1 for s in scores if s == 0.5)
    loss = sum(1 for s in scores if s == 0.0)
    a_name = Path(args.a).name
    b_name = Path(args.b).name
    print(f"\n=== {a_name} (A) vs {b_name} (B) [search {args.iterations}/{args.max_depth_rounds}]: "
          f"A win rate = {win/len(scores):.1%} ({win}W/{draw}D/{loss}L) "
          f"[{args.search_mode}, skip_single={args.skip_single_candidate}, k={args.k}] ===")
    n_term = sum(1 for r in results if r["terminal"])
    rounds = np.asarray([r["rounds"] for r in results], dtype=np.float64)
    print(f"  决定性 {n_term}/{len(results)} = {100.0 * n_term / max(len(results), 1):.1f}%   "
          f"平均回合 {rounds.mean():.0f}（中位 {np.median(rounds):.0f}）")
    for who in ("A", "B"):
        c = _Counter()
        for r in results:
            c.update(r["cls"][who])
        tot = sum(c.values()) or 1
        print(f"  {who} 出招的类构成: " + ", ".join(
            "%s:%.1f%%" % (k, 100.0 * v / tot) for k, v in c.most_common(8)))
    if args.pairs > 0:
        sc = np.asarray(scores, dtype=np.float64).reshape(args.pairs, 2).mean(axis=1)
        se = sc.std(ddof=1) / np.sqrt(len(sc)) if len(sc) > 1 else 0.0
        t = (sc.mean() - 0.5) / se if se > 0 else 0.0
        print(f"  配对胜率(A) = {sc.mean():.4f}  (0.5 = 持平)  SE = {se:.4f}   t = {t:+.2f}"
              f"   [{args.pairs} 对 / {2 * args.pairs} 局]")


if __name__ == "__main__":
    main()
