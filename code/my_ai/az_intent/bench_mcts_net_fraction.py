"""量我们自己的 MCTS：每个节点展开是否要一次网络前向？网络占搜索时间多少？

做法：用部署配置（iterations=256, k=24, max_depth_rounds=4）跑真实搜索，
把 net_fn 包一层计数器/计时器，得到：
  · 每次决策 net_fn 被调用多少次（= 展开的节点数 + 根）
  · 网络总耗时 vs 搜索总耗时（占比）
  · 每次 net_fn 内部：特征编码 vs 三个网络前向 各占多少
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.bundle_mcts import BundleMCTS  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/grafted_sdk_A.pt"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
K = 24
DEPTH = 4
ADVANCE = int(sys.argv[3]) if len(sys.argv) > 3 else 200


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(str(_REPO / CKPT), feat, value_tanh=True,
                                      value_rel_to_abs=0.0)
    stats = {"calls": 0, "net_ns": 0}

    def timed_net_fn(state, player):
        t0 = time.perf_counter_ns()
        out = net_fn(state, player)
        stats["net_ns"] += time.perf_counter_ns() - t0
        stats["calls"] += 1
        return out

    bmcts = BundleMCTS(timed_net_fn, iterations=ITERS, max_depth_rounds=DEPTH, k=K,
                       sample_mult=15, t_class=0.5, t_pos=1.0, c_puct=1.25, seed=0,
                       pos_pin="mc_quiet", skip_single_candidate=True,
                       mc_horizon=256, mc_every=1)
    facade = GameStateFacade.initial(732)
    for _ in range(ADVANCE):
        if facade.terminal:
            break
        facade.advance_round()

    # 预热
    _ = bmcts.search(facade.clone(), 0, temperature=0.0)
    stats["calls"] = 0
    stats["net_ns"] = 0

    print(f"[配置] iters={ITERS} k={K} depth={DEPTH} ckpt={Path(CKPT).name} 局面 round={facade.round_index}")
    dec_times, call_counts, net_fractions = [], [], []
    for rep in range(3):
        stats["calls"] = 0
        stats["net_ns"] = 0
        t0 = time.perf_counter()
        result = bmcts.search(facade.clone(), 0, temperature=0.0)
        total_s = time.perf_counter() - t0
        net_s = stats["net_ns"] / 1e9
        dec_times.append(total_s)
        call_counts.append(stats["calls"])
        net_fractions.append(net_s / total_s)
        print(f"  [{rep}] 搜索总耗时 {total_s:6.2f} s   net_fn 调用 {stats['calls']:>4} 次  "
              f"网络耗时 {net_s:6.2f} s  占比 {100 * net_s / total_s:5.1f}%  "
              f"每次 net_fn {net_s / max(stats['calls'],1)*1000:.2f} ms")

    # 拆解：一次 net_fn 里 编码 vs 三个前向
    obs_state = facade.clone()
    t_encode, t_forward = [], []
    import torch
    for _ in range(20):
        t0 = time.perf_counter()
        obs = feat.encode_observation(obs_state, 0, np.zeros(96))
        t_encode.append(time.perf_counter() - t0)
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats_t = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        t0 = time.perf_counter()
        net_fn(obs_state, 0)
        t_forward.append(time.perf_counter() - t0)
    print(f"\n[拆解] encode_observation ≈ {statistics.median(t_encode)*1000:.2f} ms；"
          f"一次 net_fn（含三网前向）≈ {statistics.median(t_forward)*1000:.2f} ms")
    print(f"\n[结论] 中位：搜索 {statistics.median(dec_times):.2f} s/决策，"
          f"net_fn {statistics.median(call_counts):.0f} 次/决策，"
          f"网络占比 {100 * statistics.median(net_fractions):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
