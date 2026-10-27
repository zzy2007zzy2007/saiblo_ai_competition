"""三种随机调度对比：forward(decay) vs reverse vs uniform（2026-10-27）。

指标（同一批局面、同 C/k、同 N）：
  · 树节点数 / 去重叶子数（前缀树规模）
  · 单次搜索耗时
  · β（每名掉分）、σ_within（同一动作不同分支 SD）
  · 噪声跨名次 = (σ_within/√N_distinct)/|β|
  · **信息密度 = 1 / (噪声跨名次 × 耗时)**
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np

from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor

C, K, H, N = 1.0, 3.0, 16, 64


def run(mode: str, states, feat, net_fn, vm, tree: bool = True):
    s = FlatRolloutGreedy(
        net_fn, k=48, rollouts_per_candidate=N, horizon=H, max_total_rounds=10 ** 9,
        value_model=vm, feature_extractor=feat,
        rollout_mode="random", rollout_random_side="opponent",
        rollout_prob_mode=mode, rollout_decay_c=C, rollout_decay_k=K,
        rollout_act_prob=C, rollout_seed=53, rollout_share_prefix=tree)
    nodes, uniq, times, betas, sigmas, se_ranks = [], [], [], [], [], []
    for st in states:
        t0 = time.perf_counter()
        s.search(st, 0)
        dt = (time.perf_counter() - t0)
        times.append(dt)
        if not getattr(s, "last_tree_stats", None):
            continue
        nodes.append(s.last_tree_stats[0])
        uniq.append(s.last_tree_stats[2])
        smp = getattr(s, "last_samples", None) or []
        mats = [np.array(v, dtype=float) for v in smp if v and len(v) > 1]
        if len(mats) < 8:
            continue
        means = np.array([m.mean() for m in mats])
        ranks = np.arange(1, len(means) + 1)
        order = np.argsort(-means)
        b = np.polyfit(ranks, means[order], 1)[0]
        sw = float(np.mean([m.std(ddof=1) for m in mats]))
        nd = max(1, int(np.mean(uniq))) if uniq else N
        if abs(b) > 1e-9:
            betas.append(abs(b)); sigmas.append(sw)
            se_ranks.append((sw / math.sqrt(nd)) / abs(b))
    if not se_ranks:
        return None
    nr = statistics.median(se_ranks)
    t = statistics.median(times)
    if not nodes:
        nodes = [0]; uniq = [N]
    return {"nodes": statistics.median(nodes) if nodes else 0,
            "uniq": statistics.median(uniq) if uniq else 0,
            "ms": t * 1000, "beta": statistics.median(betas),
            "sigma": statistics.median(sigmas), "noise_ranks": nr,
            "density": 1.0 / (nr * t)}


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(str(REPO / "checkpoints/posnet_A_k5_m32.pt"), feat,
                                      value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(str(REPO / "checkpoints/posnet_A_k5_m32.pt"))
    vm.eval()
    states = []
    for sd in (10, 11, 12):
        st = GameStateFacade.initial(seed=sd, cold_handle_rule_illegal=True)
        for _ in range(60):
            st.advance_round()
        states.append(st.clone())

    print(f"  C={C} k={K} H={H} N={N}，3 个局面（round 60）")
    print(f"  {'模式':<9} {'节点':>7} {'去重叶':>7} {'耗时ms':>8} {'β':>8} {'σ':>7} {'噪名次':>7} {'信息密度':>9}")
    for mode in ("decay", "reverse", "const"):
        rt = run(mode, states, feat, net_fn, vm, tree=True)
        rn = run(mode, states, feat, net_fn, vm, tree=False)
        r = None
        if rt and rn:
            r = dict(rt); r.update({k: rn[k] for k in ("beta", "sigma", "noise_ranks", "density")})
            r["ms"] = rt["ms"]; r["density"] = 1.0 / (rn["noise_ranks"] * rt["ms"] / 1000.0)
        if not r:
            print(f"  {mode:<9} 无数据"); continue
        print(f"  {mode:<9} {r['nodes']:>7.0f} {r['uniq']:>7.0f} {r['ms']:>8.0f} "
              f"{r['beta']:>8.4f} {r['sigma']:>7.4f} {r['noise_ranks']:>7.2f} {r['density']:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
