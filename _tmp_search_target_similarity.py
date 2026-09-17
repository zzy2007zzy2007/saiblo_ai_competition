"""搜索目标的**软相似度**：不只比 argmax，还比"好区"是否一致。

对每个参照局面跑两遍搜索（只换 rng seed），逐头比较**训练目标分布**
（= trainer 用的那套边际化：每个候选的 visit 按该候选里各 intent 的采样次数摊到格子上）：

  (a) 目标分布余弦（361 维，逐头平均）  —— 好区是否重叠
  (b) top-5 格集合重叠率                —— 更直观的"好区是否一致"
  (c) 并列程度：选中格占该头 visit 的份额（≈1/m ⇒ m 个候选在并列）
  (d) 硬指标（对照）：两次是否选中同一个 bundle

判读：
  (a)(b) 高 而 (d) 低 ⇒ 区域偏好稳定、只是并列里 argmax 翻
      ⇒ 该走"多次抽样平均 visit 当目标"（确定、可学、忠实于搜索）
  (a)(b) 也低 ⇒ 连好区都不稳 ⇒ 只剩 1-ply 全局价值 argmax 那条路
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

CONFIGS = [("base k24 it256 tp0.3", 24, 0.3, 256, 15),
           ("k128 it1024 tp1.0", 128, 1.0, 1024, 15)]


def _run(job):
    (label, k, t_pos, iters, sm), ckpt, ref_games, max_ref, nh = job
    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.decoder import decode_network_output

    feat = FeatureExtractor(max_actions=96)
    policy_model, net_fn = make_net_fn_from_ckpt(ckpt, feat)

    def search(st, pl, seed, gturn, mode):
        m = BundleMCTS(net_fn, iterations=iters if mode == "cfg" else 256,
                       max_depth_rounds=4, k=k if mode == "cfg" else 24,
                       sample_mult=sm, t_class=0.5,
                       t_pos=t_pos if mode == "cfg" else 0.3,
                       seed=seed * 1000 + gturn, search_mode="pos-only",
                       skip_single_candidate=True)
        return m.search(st, pl, temperature=0.0), m

    # 参照局面：**按被测配置自己筛**（该配置下搜索候选数 >= 2 的回合 = 它真的有得选的地方）。
    # 对局推演只用 raw 策略（下面的 decode_network_output），与搜索配置无关 ⇒ 各配置走的是
    # **同一批轨迹**，只有"记哪些回合"不同。用基线配置去筛是不行的：z-score 修复后尖先验
    # 下 98% 的回合只剩 1 个候选，那样筛出来的参照集对别的配置几乎全退化。
    refs = []
    for g in range(ref_games):
        seed = 31 + g
        st = make_initial_state(seed, True)
        gturn = 0
        for _ in range(600):
            if st.terminal or len(refs) >= max_ref:
                break
            for pl in (0, 1):
                if st.terminal or len(refs) >= max_ref:
                    break
                gturn += 1
                r, _ = search(st, pl, seed + 100, gturn, "cfg")
                if len(set(r.bundles)) >= 2:
                    refs.append((st.clone(), pl, seed + 100, seed + 500, gturn))
                obs = feat.encode_observation(st, pl, np.zeros(96))
                with torch.no_grad():
                    o = policy_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                     torch.from_numpy(obs["stats"]).unsqueeze(0).float())
                st.apply_operation_list(pl, decode_network_output(o, st, pl, temperature=0.0,
                                                                  intent_decoding=True))
            if pl == 1 and not st.terminal:
                st.advance_round()

    def target_vecs(res, m):
        """逐头的 361 维目标分布 + 选中格的份额。"""
        vecs = [np.zeros(361, dtype=np.float64) for _ in range(nh)]
        share = [0.0] * nh
        ch = m.last_root.children
        vis = np.asarray([c.visits for c in ch], dtype=np.float64)
        totv = vis.sum()
        if totv <= 0:
            return vecs, share
        for j, ic in enumerate(res.intent_counts):
            for h in range(nh):
                d = ic[h] if j < len(ic) and h < len(ic[j]) else {}
                s = sum(d.values())
                if not s:
                    continue
                for (c, x, y), cnt in d.items():
                    if x >= 0:
                        vecs[h][x * 19 + y] += vis[j] * cnt / s
        for h in range(nh):
            t = vecs[h].sum()
            if t > 0:
                vecs[h] /= t
                # 选中候选 → 该头采样最多的 intent 的格子，占该头总 visit 的份额
                j = int(np.argmax(vis))
                d = res.intent_counts[j][h] if j < len(res.intent_counts) else {}
                if d:
                    (c0, x0, y0), _ = max(d.items(), key=lambda kv: kv[1])
                    if x0 >= 0:
                        share[h] = vecs[h][x0 * 19 + y0]
        return vecs, share

    cos_all, ov_all, sh_all = [], [], []
    nb = deg = 0
    for st, pl, s1, s2, gt in refs:
        r1, m1 = search(st, pl, s1, gt, "cfg")
        r2, m2 = search(st, pl, s2, gt, "cfg")
        if len(set(r1.bundles)) < 2 or len(set(r2.bundles)) < 2:
            deg += 1
            continue
        nb += int(tuple(r1.chosen_bundle) == tuple(r2.chosen_bundle))
        v1, sh1 = target_vecs(r1, m1)
        v2, _ = target_vecs(r2, m2)
        for h in range(nh):
            a, b = v1[h], v2[h]
            na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb_ > 0:
                cos_all.append(float(a @ b / (na * nb_)))
                t1 = set(np.argsort(-a)[:5].tolist())
                t2 = set(np.argsort(-b)[:5].tolist())
                ov_all.append(len(t1 & t2) / 5.0)
                sh_all.append(sh1[h])
    return {"label": label, "n": len(refs), "deg": deg, "n_eff": len(refs) - deg,
            "bundle": nb, "cos": float(np.mean(cos_all)) if cos_all else float("nan"),
            "ov": float(np.mean(ov_all)) if ov_all else float("nan"),
            "share": float(np.mean(sh_all)) if sh_all else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt")
    ap.add_argument("--ref-games", type=int, default=7)
    ap.add_argument("--max-ref", type=int, default=60)
    ap.add_argument("--num-heads", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--t-pos-list", type=str, default=None,
                    help="逗号分隔；给了就生成 k24/it256 的 t_pos 扫描配置")
    args = ap.parse_args()
    cfgs = CONFIGS
    if args.t_pos_list:
        cfgs = [("tp%-5g k24 it256" % tp, 24, tp, 256, 15)
                for tp in (float(x) for x in args.t_pos_list.split(",") if x.strip())]
    jobs = [(cfg, args.ckpt, args.ref_games, args.max_ref, args.num_heads) for cfg in cfgs]
    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        res = pool.map(_run, jobs)
    print(f"\n{'配置':22s} {'参照':>5s} {'可用':>5s} {'bundle同%':>9s} {'目标余弦':>9s} "
          f"{'top5重叠':>9s} {'选中格份额':>10s}")
    for r in res:
        print(f"{r['label']:22s} {r['n']:5d} {r['n_eff']:5d} "
              f"{r['bundle']/max(r['n_eff'],1):9.1%} {r['cos']:9.3f} "
              f"{r['ov']:9.1%} {r['share']:10.1%}", flush=True)
    print("\n[sim] 完成")


if __name__ == "__main__":
    main()
