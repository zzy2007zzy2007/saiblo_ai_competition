"""无噪声判据：**价值网挑中的候选，在官方启发式排序里处于什么分位？**

为什么不用对局：`cand_pick_k96` 三条臂互相矛盾（first>random +9、value>random +1、
value>first +7）⇒ 不可传递 ⇒ 对局当不了"谁排得更好"的判据（这个游戏有建塔/降级/升级/
闪电的相互克制，非传递性可能是真的）。

本脚本改用**局面级、无对局噪声**的判据：
  * 每回合取候选池（`ActionCatalog.build()`），按官方 `score`（含一步 rollout 重排）降序排序；
  * 比较三方"选中项"的分数排名：**价值网 1-ply argmax / 官方 top-1 / 随机**；
  * 随机是 50 分位基线；**价值网明显高于随机 ⇒ 它能排序**；
  * 另报 Spearman ρ（价值网自己的评估 vs 官方 score）。

用法: python _tmp_rank_quality.py --games 12 --k 96
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
for _p in (REPO/"Ant-Game", REPO/"code"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))


def _worker(job):
    seed, ngames, k, ckpt, native, max_rounds = job
    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from SDK.utils.actions import ActionCatalog
    from SDK.backend.model import OperationType
    from my_ai.az_intent.az_selfplay import load_three_models, make_initial_state
    from my_ai.decoder import decode_network_output

    feat = FeatureExtractor(max_actions=96)
    catalog = ActionCatalog(max_actions=96, feature_extractor=feat)
    pol, _ = None, None
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt
    pol, _ = make_net_fn_from_ckpt(ckpt, feat)
    _, _, vmodel = load_three_models(ckpt)

    def _val(posts, pl):
        bs, ss = [], []
        for post in posts:
            ob = feat.encode_observation(post, 1 - pl, np.zeros(96))
            bs.append(ob["board"]); ss.append(ob["stats"])
        out = []
        for i in range(0, len(bs), 128):
            with torch.no_grad():
                v = vmodel(torch.from_numpy(np.stack(bs[i:i+128])).float(),
                           torch.from_numpy(np.stack(ss[i:i+128])).float())["value"]
            out.extend((-v.squeeze(-1).numpy()).tolist())
        return np.asarray(out, dtype=np.float64)

    rng = np.random.default_rng(seed + 77)
    pct_v, pct_f, pct_r, rho = [], [], [], []
    cls = Counter()
    n_pool = []
    for g in range(ngames):
        st = make_initial_state(seed + g, native)
        for _ in range(max_rounds):
            if st.terminal: break
            for pl in (0, 1):
                if st.terminal: break
                cands = sorted(catalog.build(st, pl), key=lambda b: -b.score)
                if len(cands) >= 2:
                    ks = np.asarray([c.score for c in cands], dtype=np.float64)
                    order = np.argsort(-ks)           # 官方排序（0 = 最好）
                    rank_of = {int(j): int(r) for r, j in enumerate(order)}
                    N = len(cands)
                    # 价值网
                    posts = []
                    for c in cands:
                        post = st.clone()
                        post.apply_operation_list(pl, list(c.operations))
                        posts.append(post)
                    a = _val(posts, pl)
                    jv = int(np.argmax(a))
                    pct_v.append(1.0 - rank_of[jv] / (N - 1))
                    # 官方 top-1 / 随机
                    pct_f.append(1.0)
                    jr = int(rng.integers(N))
                    pct_r.append(1.0 - rank_of[jr] / (N - 1))
                    # Spearman（价值网评估 vs 官方 score）
                    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(ks))
                    rho.append(float(np.corrcoef(ra, rb)[0, 1]))
                    n_pool.append(N)
                    for op in cands[jv].operations:
                        cls[OperationType(op.op_type).name] += 1
                # 用价值网继续推演（保持状态分布与我们自己的智能体一致）
                obs = feat.encode_observation(st, pl, np.zeros(96))
                with torch.no_grad():
                    o = pol(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
                st.apply_operation_list(pl, decode_network_output(o, st, pl, temperature=0.0,
                                                                    intent_decoding=True))
            if not st.terminal:
                st.advance_round()
    return {"pct_v": pct_v, "pct_f": pct_f, "pct_r": pct_r, "rho": rho,
            "cls": cls, "n_pool": n_pool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--ckpt", default="training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--max-rounds", type=int, default=512)
    a = ap.parse_args()
    per = max(a.games // a.workers, 1)
    jobs = [(a.seed + i * 1000, per, a.k, a.ckpt, True, a.max_rounds) for i in range(a.workers)]
    import multiprocessing as mp
    with mp.Pool(a.workers) as pool:
        res = pool.map(_worker, jobs)
    pv = np.asarray([x for r in res for x in r["pct_v"]])
    pf = np.asarray([x for r in res for x in r["pct_f"]])
    pr = np.asarray([x for r in res for x in r["pct_r"]])
    rho = np.asarray([x for r in res for x in r["rho"]])
    npl = np.asarray([x for r in res for x in r["n_pool"]])
    c = Counter()
    for r in res: c.update(r["cls"])
    tot = sum(c.values())
    print("\n=== 有 ≥2 候选的决策点 n=%d（池子大小 均值 %.1f 中位 %.0f）===" % (len(pv), npl.mean(), np.median(npl)))
    for nm, arr in (("官方 top-1（=100%% 由构造）", pf), ("随机", pr), ("**价值网 1-ply argmax**", pv)):
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        print("  %-26s 分位 = %5.1f%%  (SE %.1f)  [0=最差, 100=最好]" % (nm, 100*arr.mean(), 100*se))
    se = rho.std(ddof=1) / np.sqrt(len(rho))
    print("  价值网评估 vs 官方 score 的 Spearman ρ = %.3f (SE %.3f)" % (rho.mean(), se))
    print("  价值网选中项的动作构成: " + ", ".join("%s:%.0f%%" % (k, 100*v/tot) for k, v in c.most_common(5)))


if __name__ == "__main__":
    main()
