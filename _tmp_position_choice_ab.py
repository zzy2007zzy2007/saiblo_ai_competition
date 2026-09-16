"""位置轴的控制实验：93.8% 的增益来自"价值头会挑位置"，还是"偏离 argmax 本身有用"？

做法：镜像配对（我方 vs 现行 raw 解码，同一策略网），每 seed 打两局（先后手互换）。
**候选集与触发条件在所有臂里完全相同**（都是 pos-only 搜索 _expand 出来的那批候选），
唯一差别是**从候选里挑哪一个**：

  raw      : 我方不用搜索，直接用现行 raw 解码        → 基线（应恰好 1.0/对）
  search   : 搜索的真实选择（visit 最大）             → 应复现 ≫1.0
  random   : 候选里**均匀随机**挑一个                  → 关键控制
  prior    : 按候选**先验**（采样频率）抽一个，忽略 value/visit 排名

判读：
  random ≈ search ≫ 1.0  ⇒ 增益来自"偏离 argmax 本身"，价值头排名无关
                            ⇒ 位置网络学不到真东西（会学到噪声）
  random ≈ 1.0 ≪ search  ⇒ 增益来自价值排名 ⇒ 位置网络有真信号可学
  prior 落在两者之间     ⇒ 可分离"先验"与"价值"各贡献多少
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
for p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
MAX_ROUNDS = 600


def _worker(job):
    seed, mode, ckpt = job
    import torch
    torch.set_num_threads(1)
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import (make_initial_state,
                                             make_net_fn_from_ckpt)
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.decoder import decode_network_output

    feat = FeatureExtractor(max_actions=96)
    # 自动分支：二网 / 三网 / 单网（anchor_model 提供 action_map + head_logits）
    policy_model, net_fn = make_net_fn_from_ckpt(ckpt, feat)
    rng = np.random.default_rng(seed + 999)
    stats = {"ours_turns": 0, "ours_acted": 0, "cands_hist": {}, "chosen_idx": {}}

    def policy_out(st, pl):
        obs = feat.encode_observation(st, pl, np.zeros(96))
        with torch.no_grad():
            return policy_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                torch.from_numpy(obs["stats"]).unsqueeze(0).float())

    def raw_ops(st, pl):
        return decode_network_output(policy_out(st, pl), st, pl, temperature=0.0,
                                     intent_decoding=True)

    def to_ops(bundle):
        return [Operation(OperationType(int(a)), int(b), int(c)) for a, b, c in bundle]

    def our_ops(st, pl, turn_idx):
        stats["ours_turns"] += 1
        if mode == "raw":
            ops = raw_ops(st, pl)
            stats["ours_acted"] += int(bool(ops))
            return ops
        m = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=24,
                       t_class=0.5, t_pos=0.3, seed=seed * 1000 + turn_idx,
                       search_mode="pos-only", skip_single_candidate=True)
        res = m.search(st, pl, temperature=0.0)
        cands = list(res.bundles)
        stats["cands_hist"][len(cands)] = stats["cands_hist"].get(len(cands), 0) + 1
        if not cands:
            return []
        if mode == "search":
            chosen, idx = res.chosen_bundle, res.chosen_index
        elif mode == "random":
            idx = int(rng.integers(len(cands)))
            chosen = cands[idx]
        else:  # prior
            priors = np.asarray([c.prior for c in m.last_root.children], dtype=np.float64)
            if priors.sum() <= 0:
                priors = np.ones(len(cands))
            idx = int(rng.choice(len(cands), p=priors / priors.sum()))
            chosen = cands[idx]
        stats["chosen_idx"][idx] = stats["chosen_idx"].get(idx, 0) + 1
        ops = to_ops(chosen)
        stats["ours_acted"] += int(bool(ops))
        return ops

    def play(our_player: int) -> float:
        st = make_initial_state(seed, True)
        turn = 0
        for _ in range(MAX_ROUNDS):
            if st.terminal:
                break
            for pl in (0, 1):
                if st.terminal:
                    break
                if pl == our_player:
                    turn += 1
                    ops = our_ops(st, pl, turn)
                else:
                    ops = raw_ops(st, pl)
                st.apply_operation_list(pl, ops)
            if pl == 1 and not st.terminal:
                st.advance_round()
        if st.terminal and st.winner is not None:
            return 1.0 if st.winner == our_player else (0.0 if st.winner == 1 - our_player else 0.5)
        a, b = st.bases[our_player].hp, st.bases[1 - our_player].hp
        return 0.5 if a == b else (1.0 if a > b else 0.0)

    s0 = play(0)
    s1 = play(1)
    return {"pair_score": s0 + s1, "score_p0": s0, "score_p1": s1, **stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CK)
    ap.add_argument("--mode", default="search",
                    choices=["raw", "search", "random", "prior"])
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jobs = [(args.seed + i, args.mode, args.checkpoint) for i in range(args.pairs)]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            res = pool.map(_worker, jobs)
    else:
        res = [_worker(j) for j in jobs]

    ps = np.asarray([r["pair_score"] for r in res])
    n = len(res)
    se = ps.std(ddof=1) / np.sqrt(n)
    turns = sum(r["ours_turns"] for r in res)
    acted = sum(r["ours_acted"] for r in res)
    cand_hist = {}
    for r in res:
        for k, v in r["cands_hist"].items():
            cand_hist[k] = cand_hist.get(k, 0) + v
    idx_hist = {}
    for r in res:
        for k, v in r["chosen_idx"].items():
            idx_hist[k] = idx_hist.get(k, 0) + v
    t_str = "  (SE=0：镜像严格抵消)" if se == 0 else f"  t = {(ps.mean() - 1.0) / se:+.2f}"
    print(f"\n=== mode={args.mode}  {n} pairs ({2 * n} games) seed={args.seed} ===")
    print(f"  配对得分均值 = {ps.mean():.4f}  (1.0 = 与对手持平)   SE = {se:.4f}{t_str}")
    print(f"  偏离 1.0 的 pair 数 = {int((ps != 1.0).sum())}/{n}   净增分 = {ps.sum() - n:+.1f}")
    print(f"  分布: 2分(双杀)={int((ps == 2).sum())}  1.5={int((ps == 1.5).sum())}  "
          f"1分={int((ps == 1).sum())}  0.5={int((ps == 0.5).sum())}  0分(双败)={int((ps == 0).sum())}")
    print(f"  我方出招率 = {acted / max(turns, 1):.1%}   候选数分布(前5) = "
          f"{sorted(cand_hist.items(), key=lambda kv: -kv[1])[:5]}")
    top_idx = sorted(idx_hist.items(), key=lambda kv: -kv[1])[:5]
    print(f"  所选候选的序号分布(前5) = {top_idx}   （序号按先验降序）")


if __name__ == "__main__":
    main()
