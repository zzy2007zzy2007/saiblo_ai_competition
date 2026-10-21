"""扁平 rollout 搜索的冒烟测试 + 计时。

验证：
  1. 能不能跑通、返回合法 bundle；
  2. 每决策耗时是否落在预算内（max_total_rounds 自调节是否生效）；
  3. 和 M6c（BundleMCTS 256 iters / depth 4）在同一局面上的选择是否一致（仅供参考）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models, make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.bundle_mcts import BundleMCTS  # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

CKPT = str(_REPO / "checkpoints/grafted_sdk_A.pt")
K = 24


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(CKPT, feat, value_tanh=True, value_rel_to_abs=0.0)
    _, _, value_model = load_three_models(CKPT)
    value_model.eval()

    facade = GameStateFacade.initial(732)
    for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 200):
        if facade.terminal:
            break
        facade.advance_round()
    print(f"[局面] round={facade.round_index} 我方塔 {len(list(facade.towers))} "
          f"蚁 {len(list(facade.ants))}")

    # 参考：M6c
    baseline = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=K, sample_mult=15,
                          t_class=0.5, t_pos=1.0, c_puct=1.25, seed=0, pos_pin="mc_quiet",
                          skip_single_candidate=True, mc_horizon=256, mc_every=1)
    t0 = time.perf_counter()
    base_res = baseline.search(facade.clone(), 0, temperature=0.0)
    base_time = time.perf_counter() - t0
    print(f"\n[M6c] {base_time:.2f} s，候选 {len(base_res.bundles)} 个，"
          f"选中 {base_res.chosen_bundle}")

    for n_roll, horizon in ((8, 32), (16, 64), (32, 64)):
        searcher = FlatRolloutGreedy(net_fn, iterations=1, max_depth_rounds=4, k=K,
                                     sample_mult=15, t_class=0.5, t_pos=1.0, c_puct=1.25,
                                     seed=0, pos_pin="mc_quiet", skip_single_candidate=True,
                                     rollouts_per_candidate=n_roll, horizon=horizon,
                                     value_model=value_model, feature_extractor=feat,
                                     max_total_rounds=39_000)
        t0 = time.perf_counter()
        res = searcher.search(facade.clone(), 0, temperature=0.0)
        dt = time.perf_counter() - t0
        same = "相同" if res.chosen_bundle == base_res.chosen_bundle else "不同"
        print(f"[flat N={n_roll:<3} h={horizon:<3}] {dt:.2f} s，候选 {len(res.bundles)} 个，"
              f"实际每候选 {searcher.last_rollouts} 次，共 {searcher.last_rounds} 回合，"
              f"选中 {res.chosen_bundle}（与 M6c {same}）；"
              f"expand {searcher.last_timing.get('expand_s', 0):.2f}s / "
              f"rollout {searcher.last_timing.get('rollout_s', 0):.2f}s")
        print(f"          分数 top3 = {np.round(np.sort(searcher.last_scores)[::-1][:3], 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
