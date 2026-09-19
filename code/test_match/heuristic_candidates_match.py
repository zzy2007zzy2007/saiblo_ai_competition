"""候选集来自启发式打分、由"随机 / 价值网"来选 —— 检验价值网能否给**非闪电动作**排序。

动机（用户 2026-09-19 提出）：`rv4_lightning_only` 已证"不放闪电以外的动作"值 ~80pp，
所以类轴是最大的一块。但当前价值网几乎没见过"有塔的局面"（自对弈里类头塌缩、只用闪电），
它可能**判不了其他动作的好坏**。而在我们自己的局面里"建塔"本来就该差 ⇒ 在那儿测不出东西。

本台子的设计（比我原先"和 rule_v4 比一致率"干净）：
  * 候选集 = `ActionCatalog.build()` 按 `score` 降序的前 K 个（= 官方 ExampleAI 用的那套启发式）
    ⇒ **两侧拿到的候选集完全相同** ⇒ 这是"同一堆选项，谁排得更好"的受控 A/B。
  * 启发式的 top-K 里**包含建塔/升级** ⇒ 这个智能体会走到**有塔的局面**，正好绕开那个坏不动点。
  * 随机 vs 价值网：
      - 价值网 ≫ 随机 ⇒ 它能排非闪电动作 ⇒ **不用重训价值网**
      - 价值网 ≈ 随机 ⇒ 它排不了 ⇒ 前提成立，需要重训

用法:
    python code/test_match/heuristic_candidates_match.py --our random --opp random --pairs 8   # 对照
    python code/test_match/heuristic_candidates_match.py --our value  --opp random --pairs 32 --k 8
    python code/test_match/heuristic_candidates_match.py --our value  --opp random --pairs 32 --k 24
    python code/test_match/heuristic_candidates_match.py --our value  --opp random --pairs 32 --k 8 --no-lightning
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORIG = REPO / "其他版本ai" / "rule_v4"
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _is_lightning(b) -> bool:
    from SDK.backend.model import OperationType
    return any(op.op_type == OperationType.USE_LIGHTNING_STORM for op in b.operations)


def _candidates(state, player, catalog, k: int, no_lightning: bool) -> list:
    """启发式 top-K。`ActionCatalog.build()` 的分 bundle.score 降序（空过分数 0 会排最后）。"""
    bundles = sorted(catalog.build(state, player), key=lambda b: -b.score)
    if no_lightning:
        bundles = [b for b in bundles if not _is_lightning(b)]
    return bundles[:k]


def _worker(job):
    seed, our_player, our_mode, opp_mode, k, no_lightning, ckpt, native, max_rounds = job
    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from SDK.utils.actions import ActionCatalog
    from SDK.backend.model import OperationType
    from my_ai.az_intent.az_selfplay import load_three_models, make_initial_state

    feat = FeatureExtractor(max_actions=96)
    catalog = ActionCatalog(max_actions=96, feature_extractor=feat)
    vmodel = None
    if "value" in (our_mode, opp_mode):
        _, _, vmodel = load_three_models(ckpt)
    OrigAI = None
    if "rule_v4" in (our_mode, opp_mode):
        m = _load_by_path("rv4_orig_ai_hm", ORIG / "ai.py")
        m.RuleBasedAI.log = staticmethod(lambda message: None)   # 关掉 GB 级日志
        OrigAI = m.AI

    stats = {"turns": 0, "cands": [], "cls": {"ours": Counter(), "opp": Counter()}}

    def _value_of(posts, pl):
        boards, statss = [], []
        for post in posts:
            ob = feat.encode_observation(post, 1 - pl, np.zeros(96))
            boards.append(ob["board"]); statss.append(ob["stats"])
        out = []
        for i in range(0, len(boards), 128):
            with torch.no_grad():
                v = vmodel(torch.from_numpy(np.stack(boards[i:i + 128])).float(),
                           torch.from_numpy(np.stack(statss[i:i + 128])).float())["value"]
            out.extend((-v.squeeze(-1).numpy()).tolist())
        return np.asarray(out, dtype=np.float64)

    def _pick(state, player, mode, rng, who):
        if mode == "rule_v4":
            ops = OrigAI(seed=seed).choose_operations(state, player)   # 只用于对手侧
            for op in ops:
                stats["cls"][who][OperationType(op.op_type).name] += 1
            return ops
        cands = _candidates(state, player, catalog, k, no_lightning)
        stats["cands"].append(len(cands))
        if not cands:
            return []
        if mode == "random":
            b = cands[int(rng.integers(len(cands)))]
        elif mode == "first":                     # = ExampleAI 的做法（启发式 top-K 里分数最高）
            b = cands[0]
        else:                                     # value：逐候选建 post-state 过价值网取最优
            posts = []
            for c in cands:
                post = state.clone()
                post.apply_operation_list(player, list(c.operations))
                posts.append(post)
            b = cands[int(np.argmax(_value_of(posts, player)))]
        for op in b.operations:
            stats["cls"][who][OperationType(op.op_type).name] += 1
        return list(b.operations)

    rng = np.random.default_rng(seed + 4242)
    st = make_initial_state(seed, native)
    _rnd = 0
    for _rnd in range(max_rounds):
        if st.terminal:
            break
        for pl in (0, 1):
            if st.terminal:
                break
            who = "ours" if pl == our_player else "opp"
            mode = our_mode if pl == our_player else opp_mode
            stats["turns"] += 1
            ops = _pick(st, pl, mode, rng, who)
            st.apply_operation_list(pl, ops)
        if not st.terminal:
            st.advance_round()

    if st.terminal and st.winner is not None:
        score = 1.0 if st.winner == our_player else (0.0 if st.winner == 1 - our_player else 0.5)
    else:
        a, b = st.bases[our_player].hp, st.bases[1 - our_player].hp
        score = 0.5 if a == b else (1.0 if a > b else 0.0)
    print("[game] seed=%d our_player=%d %s us=%d opp=%d rounds=%d"
          % (seed, our_player, "WIN" if score == 1.0 else ("LOSS" if score == 0.0 else "DRAW"),
             st.bases[our_player].hp, st.bases[1 - our_player].hp, _rnd), flush=True)
    return {"score": score, "turns": stats["turns"], "cands": stats["cands"],
            "cls": stats["cls"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", default="value", choices=["random", "value", "first", "rule_v4"])
    ap.add_argument("--opp", default="random", choices=["random", "value", "first", "rule_v4"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-lightning", action="store_true", help="候选集里剔除含闪电的 bundle")
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_r1.pt")
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native-engine", action="store_true", default=True)
    ap.add_argument("--max-rounds", type=int, default=512)
    a = ap.parse_args()

    jobs = []
    for i in range(a.pairs):
        s = a.seed + i
        for pl in (0, 1):
            jobs.append((s, pl, a.our, a.opp, a.k, a.no_lightning, a.ckpt,
                         a.native_engine, a.max_rounds))
    import multiprocessing as mp
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            res = pool.map(_worker, jobs)
    else:
        res = [_worker(j) for j in jobs]

    sc = np.asarray([r["score"] for r in res], dtype=np.float64)
    ps = sc.reshape(a.pairs, 2).sum(axis=1) / 2.0
    n = len(ps)
    se = ps.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    cand = np.asarray([c for r in res for c in r["cands"]], dtype=np.float64)
    t_str = "  (SE=0：镜像严格抵消)" if se == 0 else f"  t = {(ps.mean() - 0.5) / se:+.2f}"
    print(f"\n=== our={a.our} opp={a.opp} k={a.k} no_lightning={a.no_lightning}  "
          f"{n} pairs ({2 * n} games) seed={a.seed} ===")
    print(f"  我方配对胜率 = {ps.mean():.4f}  (0.5 = 持平, 1.0 = 全胜)  SE = {se:.4f}{t_str}")
    print(f"  偏离 0.5 的 pair 数 = {int((np.abs(ps - 0.5) > 1e-9).sum())}/{n}   "
          f"净胜局 = {int(sc.sum() - n)}")
    if len(cand):
        print(f"  候选集大小: 均值 {cand.mean():.1f}  中位 {np.median(cand):.0f}  "
              f"min {int(cand.min())}  max {int(cand.max())}")
    for who in ("ours", "opp"):
        c = Counter()
        for r in res:
            c.update(r["cls"][who])
        tot = sum(c.values())
        top = ", ".join(f"{kk}:{vv}({100.0 * vv / max(tot, 1):.0f}%)" for kk, vv in c.most_common(6))
        print(f"  {who} 的动作构成: {top}")


if __name__ == "__main__":
    main()
