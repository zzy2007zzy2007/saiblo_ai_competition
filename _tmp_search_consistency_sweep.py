"""搜索自一致率扫描（并行版）：调 k / t_pos / 迭代数，看"选中的那一步"能不能从
"抽样运气"变成"局面的函数"。

做法：
  * 参照局面集 = 用**基线配置**扫若干局、取"有得选"（root 候选数 >=2）的 (局面, 玩家)；
    每个 worker 内部**确定性重建**同一份参照集（避免跨进程传 C++ 状态）。
  * 每个配置在参照集上**跑两遍搜索（只换 rng seed）**，比较两次"选中"的每头意图
    （= 访问最高候选里采样次数最多的 intent，与 report_posnet_diag.py 同口径）。

防假信号：报告 ① 平均候选数、② 退化比例（某一遍只剩 <2 候选 ⇒ 选择被"逼"出来、
一致率虚高，故排除）、③ 选中 bundle 是否完全相同。

判读：基线一致率仅 1.5% ⇒ 目标基本是噪声（策略最多学到"最可能那格"）；
若加大 k/迭代后一致率显著上升 ⇒ 目标开始由局面决定 ⇒ 位置蒸馏才有可学的东西。
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

CONFIGS = [
    # 关键：k 与迭代必须同步放大，否则"每候选访问数"太少、visit 排名本身无意义
    # （上一版 k=128/it=256 => 每候选只有 2 次访问，结论不可信）
    ("base k24 it256 tp0.3", 24, 0.3, 256, 15),      # 访问/候选 ~11
    ("k64 it2048 tp0.3", 64, 0.3, 2048, 15),         # 32
    ("k64 it2048 tp1.0", 64, 1.0, 2048, 15),         # 32
    ("k128 it4096 tp1.0", 128, 1.0, 4096, 15),       # 32
    ("k200 it8192 tp1.0", 200, 1.0, 8192, 15),       # 41（最贵，单独跑）
]

# 2026-09-18 新增：k 与迭代数的**二维**扫描（全部 t_pos=1.0 = 强度最优档），
# 用来分离"一致率被哪一件事卡住"（用户 2026-09-18 提出的问题）：
#   k24 it256/it1024/it4096 → 固定 k，只加迭代数（若一致率上升 ⇒ 卡在访问噪声 ⇒ 多付算力可解）
#   k96 it1024              → 与 k24 it256 同为 ~11 访问/候选，只加大 k（若只有它上升
#                             ⇒ 卡在"两遍候选集重叠度" ⇒ 要修的是让候选集确定）
# 注意：修复前那组（k24→k200、it256→it8192，一致率 0%→18.8%）把两者一起放大了，分不清。
CONFIGS_GRID2D = [
    ("k24  it256  tp1", 24, 1.0, 256, 15),
    ("k24  it1024 tp1", 24, 1.0, 1024, 15),
    ("k24  it4096 tp1", 24, 1.0, 4096, 15),
    ("k96  it1024 tp1", 96, 1.0, 1024, 15),
]

# 2026-09-18：**价值先验下**的一致率，配 `--pos-prior value`。判据的判据 = 候选数：
# 低温时如果采样全落在先验的几个峰上、候选数远小于 k，那"一致率高"就是平凡的
# （不是"搜索找到了局面的答案"，而是"采样器在读价值网的 argmax"）。
# 各配置之间比一致率必须同时看候选数，否则会被峰度混淆（见日志 valprior_similarity）。
CONFIGS_VALPRIOR = [
    ("k24 tp0.5", 24, 0.5, 256, 15),
    ("k48 tp0.5", 48, 0.5, 256, 15),
    ("k24 tp1.0", 24, 1.0, 256, 15),
    ("k48 tp1.0", 48, 1.0, 256, 15),
]


def _refs(net_fn, policy_model, feat, ref_games: int, max_ref: int,
          ref_k: int, ref_t_pos: float, ref_iters: int, ref_sm: int, ppf=None):
    from my_ai.az_intent.az_selfplay import make_initial_state
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.decoder import decode_network_output

    def search(st, pl, seed, gturn, k, t_pos, iters, sm):
        m = BundleMCTS(net_fn, iterations=iters, max_depth_rounds=4, k=k, sample_mult=sm,
                       t_class=0.5, t_pos=t_pos, seed=seed * 1000 + gturn,
                       search_mode="pos-only", skip_single_candidate=True,
                       pos_prior_fn=ppf)
        return m.search(st, pl, temperature=0.0)

    out = []
    for g in range(ref_games):
        seed = 31 + g
        st = make_initial_state(seed, True)
        gturn = 0
        for _ in range(600):
            if st.terminal or len(out) >= max_ref:
                break
            for pl in (0, 1):
                if st.terminal or len(out) >= max_ref:
                    break
                gturn += 1
                r = search(st, pl, seed + 100, gturn, ref_k, ref_t_pos, ref_iters, ref_sm)
                if len(set(r.bundles)) >= 2:
                    out.append((st.clone(), pl, seed + 100, seed + 500, gturn))
                obs = feat.encode_observation(st, pl, np.zeros(96))
                with torch.no_grad():
                    o = policy_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                     torch.from_numpy(obs["stats"]).unsqueeze(0).float())
                st.apply_operation_list(pl, decode_network_output(o, st, pl, temperature=0.0,
                                                                  intent_decoding=True))
            if pl == 1 and not st.terminal:
                st.advance_round()
    return out


def _run(job):
    (label, k, t_pos, iters, sm), chunk, nchunk, ckpt, ref_games, max_ref, nh, pos_prior = job
    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.az_intent.eval import _make_value_pos_prior

    feat = FeatureExtractor(max_actions=96)
    policy_model, net_fn = make_net_fn_from_ckpt(ckpt, feat)
    ppf = None
    if pos_prior == "value":
        _, _, _vm = load_three_models(ckpt)
        ppf = _make_value_pos_prior(feat, _vm)
    # 参照集按**被测配置自己**筛（该配置下有 >=2 个候选的回合 = 它真的有得选的地方）。
    # 不能用固定基线去筛：那会系统性偏向某个配置。
    refs = _refs(net_fn, policy_model, feat, ref_games, max_ref, k, t_pos, iters, sm, ppf)
    lo, hi = chunk * len(refs) // nchunk, (chunk + 1) * len(refs) // nchunk

    def search(st, pl, seed, gturn):
        m = BundleMCTS(net_fn, iterations=iters, max_depth_rounds=4, k=k, sample_mult=sm,
                       t_class=0.5, t_pos=t_pos, seed=seed * 1000 + gturn,
                       search_mode="pos-only", skip_single_candidate=True,
                       pos_prior_fn=ppf)
        return m.search(st, pl, temperature=0.0)

    def intents(res):
        o = [None] * nh
        if res.intent_counts:
            d = res.intent_counts[res.chosen_index]
            for h in range(nh):
                if d[h]:
                    o[h] = max(d[h].items(), key=lambda kv: kv[1])[0]
        return o

    hit = [0] * nh
    tot = [0] * nh
    nb = deg = 0
    nc = []
    for st, pl, s1, s2, gt in refs[lo:hi]:
        r1, r2 = search(st, pl, s1, gt), search(st, pl, s2, gt)
        c1, c2 = len(set(r1.bundles)), len(set(r2.bundles))
        nc.append(max(c1, c2))
        if c1 < 2 or c2 < 2:
            deg += 1
            continue
        nb += int(tuple(r1.chosen_bundle) == tuple(r2.chosen_bundle))
        i1, i2 = intents(r1), intents(r2)
        for h in range(nh):
            if i1[h] is not None and i2[h] is not None:
                tot[h] += 1
                hit[h] += int(i1[h] == i2[h])
    return {"label": label, "n": hi - lo, "deg": deg, "nb": nb, "hit": hit, "tot": tot,
            "nc": float(np.mean(nc)) if nc else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt")
    ap.add_argument("--ref-games", type=int, default=7)
    ap.add_argument("--max-ref", type=int, default=80)
    ap.add_argument("--num-heads", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--labels", type=str, default=None,
                    help="只跑标签里含该子串的配置（逗号分隔）")
    ap.add_argument("--grid2d", action="store_true",
                    help="改用 k×迭代数 二维扫描配置（t_pos=1.0），见 CONFIGS_GRID2D")
    ap.add_argument("--valprior-grid", action="store_true",
                    help="改用 k∈{24,48}×t_pos∈{0.5,1.0} 配置（配 --pos-prior）")
    ap.add_argument("--pos-prior", type=str, default="policy", choices=["policy", "value"],
                    help="位置采样分布的来源：policy=位置网；value=价值网 1-ply z-score（仅根节点）")
    args = ap.parse_args()

    base = CONFIGS_VALPRIOR if args.valprior_grid else (
        CONFIGS_GRID2D if args.grid2d else CONFIGS)
    cfgs = base
    if args.labels:
        keys = [x.strip() for x in args.labels.split(",") if x.strip()]
        cfgs = [c for c in base if any(k in c[0] for k in keys)]
        print("[sweep] 只跑这些配置: " + str([c[0] for c in cfgs]), flush=True)
    print("[sweep] pos_prior=%s" % args.pos_prior, flush=True)
    jobs = [(cfg, c, args.chunks, args.ckpt, args.ref_games, args.max_ref, args.num_heads,
             args.pos_prior)
            for cfg in cfgs for c in range(args.chunks)]
    print(f"[sweep] {len(cfgs)} 配置 × {args.chunks} 块 = {len(jobs)} 任务，"
          f"{args.workers} 进程并行（脚本原本单核，机器 32 核）", flush=True)
    import multiprocessing as mp
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            res = pool.map(_run, jobs)
    else:
        res = [_run(j) for j in jobs]

    agg: dict = {}
    for r in res:
        a = agg.setdefault(r["label"], {"n": 0, "deg": 0, "nb": 0,
                                        "hit": [0] * args.num_heads,
                                        "tot": [0] * args.num_heads, "nc": []})
        a["n"] += r["n"]; a["deg"] += r["deg"]; a["nb"] += r["nb"]
        a["nc"].append(r["nc"])
        for h in range(args.num_heads):
            a["hit"][h] += r["hit"][h]; a["tot"][h] += r["tot"][h]

    print(f"\n{'配置':22s} {'候选数':>7s} {'访问/候选':>9s} {'退化%':>7s} {'bundle同%':>9s} "
          f"{'逐头一致%':>9s} {'头0/1/2':>14s}   n")
    for label, *_ in cfgs:
        a = agg[label]
        n_eff = a["n"] - a["deg"]
        agree = sum(a["hit"]) / max(sum(a["tot"]), 1)
        ph = "/".join(f"{(a['hit'][h]/a['tot'][h]):.0%}" if a["tot"][h] else "-"
                      for h in range(args.num_heads))
        vpc = float([c for c in cfgs if c[0] == label][0][3]) / float(
            [c for c in cfgs if c[0] == label][0][1])
        print(f"{label:22s} {np.mean(a['nc']):7.1f} {vpc:9.0f} {a['deg']/max(a['n'],1):7.1%} "
              f"{a['nb']/max(n_eff,1):9.1%} {agree:9.1%} {ph:>14s}   {a['n']}", flush=True)
    print("\n[sweep] 完成")


if __name__ == "__main__":
    main()
