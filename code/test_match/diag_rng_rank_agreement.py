"""四组 rollout 配置的排名一致性诊断（2026-10-28）。

目的：判断"引擎 RNG 扰动"是否真的损害决策质量，还是两臂差异出在 idle 上。

四组（同一批局面、k=48、N=64、H=16、prefix=1）：
  P  random  + 不扰动      ← 基线（对手会反抗）
  R  idle    + 不扰动      ← 纯确定性 rollout（老部署配置）
  Q  idle    + 扰动        ← 当前实验臂（⚠️ 双重差异）
  S  random  + 扰动        ← 只差"扰动"一个变量

指标（两两对比）：
  · Spearman(候选分数)
  · top-1 一致率
  · 每组自身的 σ_within / β（噪声水平）
"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
from scipy.stats import spearmanr

from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor

K, N, H = 48, 64, 16


def build(feat, net_fn, vm, mode: str, perturb: bool):
    return FlatRolloutGreedy(
        net_fn, k=K, rollouts_per_candidate=N, horizon=H, max_total_rounds=10 ** 9,
        value_model=vm, feature_extractor=feat,
        rollout_mode=mode, rollout_random_side="opponent",
        rollout_prob_mode="const", rollout_decay_c=0.5, rollout_decay_k=3.0,
        rollout_act_prob=0.5, rollout_seed=53, rollout_share_prefix=True,
        rollout_rng_perturb=perturb)


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

    cfgs = [("P random", "random", False), ("R idle", "idle", False),
            ("Q idle+perturb", "idle", True), ("S random+perturb", "random", True)]
    scores = {}
    for name, mode, perturb in cfgs:
        s = build(feat, net_fn, vm, mode, perturb)
        rows = []
        for st in states:
            try:
                s.search(st, 0)
                rows.append(np.asarray(s.last_scores, dtype=float))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] 搜索失败: {type(exc).__name__}: {exc}")
                rows.append(None)
        ok = [r for r in rows if r is not None and len(r) > 8]
        scores[name] = ok
        print(f"  [{name}] 有效局面 {len(ok)}/{len(states)}"
              + (f"  候选数 {len(ok[0])}" if ok else ""), flush=True)

    print("\n  两两对比（同局面）：")
    names = [c[0] for c in cfgs]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scores[names[i]], scores[names[j]]
            if not a or not b or len(a) != len(b):
                continue
            rhos, tops = [], []
            for x, y in zip(a, b):
                n = min(len(x), len(y))
                rho = spearmanr(x[:n], y[:n]).statistic
                if not np.isnan(rho):
                    rhos.append(rho)
                tops.append(int(np.argmax(x[:n])) == int(np.argmax(y[:n])))
            if rhos:
                print(f"    {names[i]:<16} vs {names[j]:<16} "
                      f"Spearman {statistics.mean(rhos):+.3f}   top-1 一致 {100*sum(tops)/len(tops):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
