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
    seed, mode, ckpt, opp_ckpt, opp_mode, t_pos, k, pos_prior = job
    import torch
    torch.set_num_threads(1)
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import (load_three_models, make_initial_state,
                                             make_net_fn_from_ckpt)
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.decoder import (decode_head, decode_network_output, make_class_mask,
                               make_position_masks)

    feat = FeatureExtractor(max_actions=96)
    # 自动分支：二网 / 三网 / 单网（anchor_model 提供 action_map + head_logits）
    policy_model, net_fn = make_net_fn_from_ckpt(ckpt, feat)
    # 价值查询：三网 ckpt 走价值网本身（anchor 三网策略故意不含 value）；否则策略网自带 value
    _ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "pos_state" in _ck or "class_state" in _ck:
        _, _, vmodel = load_three_models(ckpt)
    else:
        vmodel = policy_model

    def _value_of(posts, pl):
        """对一串 post-state 批量查价值网，返回**我方优势** = -(对手视角价值)。"""
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

    # 对手用另一个 ckpt（None = 同一个）：判据"raw 新 vs raw 旧"需要两边不同
    opp_model = policy_model
    opp_net_fn = net_fn
    if opp_ckpt and opp_ckpt != ckpt:
        opp_model, opp_net_fn = make_net_fn_from_ckpt(opp_ckpt, feat)
    rng = np.random.default_rng(seed + 999)
    stats = {"ours_turns": 0, "ours_acted": 0, "cands_hist": {}, "chosen_idx": {}}

    def policy_out(st, pl):
        obs = feat.encode_observation(st, pl, np.zeros(96))
        with torch.no_grad():
            return policy_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                torch.from_numpy(obs["stats"]).unsqueeze(0).float())

    def _out_with(m, st, pl):
        obs = feat.encode_observation(st, pl, np.zeros(96))
        with torch.no_grad():
            return m(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                     torch.from_numpy(obs["stats"]).unsqueeze(0).float())

    def raw_ops(st, pl):
        return decode_network_output(policy_out(st, pl), st, pl, temperature=0.0,
                                     intent_decoding=True)

    def _value_prior(st, pl, net_out, pinned, is_root):
        """把每个钉类的 action_map 通道换成"该类合法格上 1-ply 我方优势"的 z-score。

        只在**根节点**生效（更深的节点维持网络自己的先验）——根节点的候选集才是决定最终
        选择的那个；若每层都算，每个扩展要 ~270 次价值前向 × 256 次迭代，成本不可接受。
        z-score 只取该类合法格（与采样器同一套），所以 t_pos 的语义与平时一致。
        """
        if not is_root:
            return None
        am = np.asarray(net_out["action_map"], dtype=np.float32)
        pm = make_position_masks(st, pl, intent_decoding=True)
        cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
        hl0 = np.asarray(net_out["head_logits"][0], dtype=np.float32)
        am2 = am.copy()
        for c in pinned:
            if c < 0 or c >= len(cm) or not cm[c]:
                continue
            cells = np.argwhere(pm[c])
            if len(cells) < 2:
                continue
            posts, keep = [], []
            for (x, y) in cells:
                am3 = am.copy(); am3[c] = -1e9; am3[c, x, y] = 1e9
                op = decode_head(hl0, am3, cm.copy(), pm.copy(), st, pl, class_id=int(c),
                                 temperature=0.0, pos_temperature=0.0, intent_decoding=True)
                if op is None or not st.can_apply_operation(pl, op, ()):
                    continue
                post = st.clone(); post.apply_operation_list(pl, [op])
                posts.append(post); keep.append((int(x), int(y)))
            if len(keep) < 2:
                continue
            a = _value_of(posts, pl)
            z = (a - a.mean()) / (a.std() + 1e-8)
            am2[c] = -1e9
            for (x, y), zz in zip(keep, z):
                am2[c, x, y] = float(zz)
        return am2

    ppf = _value_prior if pos_prior == "value" else None

    def search_bundle(nf, st, pl, gturn):
        """一次 pos-only 搜索，返回 chosen_bundle（两侧共用同一构造，保证对称）。"""
        # 注意：外部位置先验是**我方**的机制，对手侧（跑 raw）不用
        m = BundleMCTS(nf, iterations=256, max_depth_rounds=4, k=k,
                       t_class=0.5, t_pos=t_pos, seed=seed * 1000 + gturn,
                       search_mode="pos-only", skip_single_candidate=True)
        return m.search(st, pl, temperature=0.0).chosen_bundle

    def opp_search_ops(st, pl, gturn):
        return to_ops(search_bundle(opp_net_fn, st, pl, gturn))

    def opp_ops(st, pl, gturn=0):
        if opp_mode == "search":
            return opp_search_ops(st, pl, gturn)
        return decode_network_output(_out_with(opp_model, st, pl), st, pl,
                                     temperature=0.0, intent_decoding=True)

    def to_ops(bundle):
        return [Operation(OperationType(int(a)), int(b), int(c)) for a, b, c in bundle]

    def our_ops(st, pl, turn_idx):
        stats["ours_turns"] += 1
        if mode == "raw":
            ops = raw_ops(st, pl)
            stats["ours_acted"] += int(bool(ops))
            return ops

        if mode == "valueargmax":
            # 臂 C：完全不搜索 —— 钉类固定，在**全部**合法格上做 1-ply 价值 argmax。
            # 即 _tmp_pos_greedy_vs_raw.py 的规则，但现在跑在同一个台子里（消除口径差）。
            o = policy_out(st, pl)
            am = o["action_map"][0].numpy()
            hl = [o[f"head{h}_logits"].squeeze(0).numpy() for h in (1, 2, 3)]
            pm = make_position_masks(st, pl, intent_decoding=True)
            cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
            ops, pending = [], ()
            for h in range(3):
                c = int(np.argmax(hl[h]))
                if not cm[c] or c == 23:
                    continue
                cells = np.argwhere(pm[c])
                if len(cells) == 0:
                    continue
                posts, cand = [], []
                for (x, y) in cells:
                    am3 = am.copy(); am3[c] = -1e9; am3[c, x, y] = 1e9
                    op = decode_head(hl[h], am3, cm.copy(), pm.copy(), st, pl, class_id=c,
                                     temperature=0.0, pos_temperature=0.0, intent_decoding=True)
                    if op is None or not st.can_apply_operation(pl, op, pending):
                        continue
                    post = st.clone(); post.apply_operation_list(pl, [op])
                    posts.append(post); cand.append(op)
                if not posts:
                    continue
                op = cand[int(np.argmax(_value_of(posts, pl)))]
                ops.append(op); pending = pending + (op,)
            stats["ours_acted"] += int(bool(ops))
            stats["cands_hist"][len(ops)] = stats["cands_hist"].get(len(ops), 0) + 1
            return ops

        m = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=k,
                       t_class=0.5, t_pos=t_pos, seed=seed * 1000 + turn_idx,
                       search_mode="pos-only", skip_single_candidate=True,
                       pos_prior_fn=ppf)
        res = m.search(st, pl, temperature=0.0)
        cands = list(res.bundles)
        stats["cands_hist"][len(cands)] = stats["cands_hist"].get(len(cands), 0) + 1
        if not cands:
            return []
        if mode == "search":
            chosen, idx = res.chosen_bundle, res.chosen_index
        elif mode == "valuepick":
            # 臂 D：同一个候选集，但**不看访问次数**，直接用价值网选
            posts = []
            for bundle in cands:
                post = st.clone()
                post.apply_operation_list(pl, to_ops(bundle))
                posts.append(post)
            idx = int(np.argmax(_value_of(posts, pl)))
            chosen = cands[idx]
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
        gturn = 0
        for _ in range(MAX_ROUNDS):
            if st.terminal:
                break
            for pl in (0, 1):
                if st.terminal:
                    break
                gturn += 1                      # 全局回合号：两侧都用它定 seed（对称）
                if pl == our_player:
                    ops = our_ops(st, pl, gturn)
                else:
                    ops = opp_ops(st, pl, gturn)
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
    ap.add_argument("--opponent-checkpoint", type=str, default=None)
    ap.add_argument("--opp-mode", type=str, default="raw", choices=["raw", "search"])
    ap.add_argument("--mode", default="search",
                    choices=["raw", "search", "random", "prior", "valuepick", "valueargmax"],
                    help="search=采样候选+按访问次数选（现有）；valuepick=**同一候选集**但直接"
                         "按价值选（隔离'访问累积 vs 价值直接选'）；valueargmax=不搜索，"
                         "在全部合法格上做 1-ply 价值 argmax（= pos_greedy 规则，但同台）")
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=24, help="候选数（用于测缩 k）")
    ap.add_argument("--pos-prior", type=str, default="policy", choices=["policy", "value"],
                    help="位置采样分布的来源：policy=位置网（现状）；value=价值网在钉类合法格上"
                         "的 1-ply z-score（只在根节点生效）⇒ 测'让价值网决定候选菜单'")
    ap.add_argument("--t-pos", type=float, default=1.0,
                    help="位置采样温度（z-score 归一化后）；两侧共用，保证镜像对称。"
                         "默认 1.0（强度平台起点）；0.3 是极尖档，见 docs/az_t_pos_default_fix.md")
    args = ap.parse_args()

    jobs = [(args.seed + i, args.mode, args.checkpoint, args.opponent_checkpoint,
             args.opp_mode, args.t_pos, args.k, args.pos_prior) for i in range(args.pairs)]
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
    print(f"\n=== mode={args.mode}  t_pos={args.t_pos}  k={args.k}  "
          f"pos_prior={args.pos_prior}  {n} pairs ({2 * n} games) seed={args.seed} ===")
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
