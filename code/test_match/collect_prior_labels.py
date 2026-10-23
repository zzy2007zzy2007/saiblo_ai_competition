"""采集位置先验的训练标签：用"枚举 + 值网排序"当教师（2026-10-23 夜）。

为什么：用户要求"要重训就把数据量搞大"，否则分不清是数据不够还是方法不对。
本脚本在**进程内自对弈**（不起 harness/AI 进程），每个决策：
  · 教师：枚举全部合法候选（k=300）+ 每个候选跑 h=16 rollout → 值网估值；
  · 记录：该局面的特征平面（28×19×19，float16）+ 每个候选的 (cls, x, y, value)；
  · 走子：用教师排序的 top-1（保证对局仍然"像真棋"）。
为控制成本：每 3 个回合只做 1 次教师评估，其余回合用 k=24 的廉价搜索走子。

输出：`training_history/prior_labels/g<seed>.npz`
  - `feat`: (S, 28, 19, 19) float16
  - `cand_cls/x/y/val`: 变长，按 `cand_off`（长度 S+1 的前缀和）索引
  - `rounds`: (S,) 该样本的回合号
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType
from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from my_ai.decoder import make_position_masks

CKPT = "checkpoints/posnet_A_k5_m32.pt"
H = 16
TEACHER_EVERY = 3          # 每 3 回合做 1 次教师评估
TEACHER_K = 300
CHEAP_K = 24
MAX_ROUNDS = int(__import__("os").environ.get("PRIOR_MAX_ROUNDS", "400"))
TEACHER = __import__("os").environ.get("PRIOR_TEACHER", "rollout")   # rollout | mcts
MCTS_ITERS = int(__import__("os").environ.get("PRIOR_MCTS_ITERS", "256"))
MCTS_K = int(__import__("os").environ.get("PRIOR_MCTS_K", "24"))


def ops_from_bundle(bundle):
    return [Operation(OperationType(int(t)), int(a0), int(a1)) for (t, a0, a1) in bundle]


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    out_dir = REPO / "training_history/prior_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    _prefix = "m" if TEACHER == "mcts" else "g"
    out_path = out_dir / f"{_prefix}{seed}.npz"
    if out_path.exists():
        print(f"  [skip] {out_path.name} 已存在", flush=True)
        return 0

    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(CKPT, feat, value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(CKPT)
    vm.eval()

    def make_search(k):
        if TEACHER == "mcts":
            from my_ai.az_intent.bundle_mcts import BundleMCTS
            return BundleMCTS(net_fn, iterations=MCTS_ITERS, k=MCTS_K, sample_mult=15,
                              search_mode="pos-only", pos_pin="mc_quiet", mc_every=1,
                              mc_horizon=16, mc_root_only=1, mc_cells=4, mc_score="net",
                              mc_net_samples=4)
        return FlatRolloutGreedy(net_fn, k=k, rollouts_per_candidate=1, horizon=H,
                                 max_total_rounds=10 ** 9, value_model=vm,
                                 feature_extractor=feat, rollout_mode="idle")

    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    feats, stats_list, ccls, cxs, cys, cvals, offs, rounds = [], [], [], [], [], [], [0], []
    t0 = time.time()
    n_teacher = 0
    for rnd in range(MAX_ROUNDS):
        if st.terminal:
            break
        use_teacher = (rnd % TEACHER_EVERY == 0)
        search = make_search(TEACHER_K if use_teacher else CHEAP_K)
        picked = []
        for player in (0, 1):
            if st.terminal:
                break
            try:
                _res = search.search(st.clone(), player)
                bundles = list(getattr(_res, 'bundles', None) or search.last_root.bundles)
            except Exception as exc:
                print(f"  ⚠️ round={rnd} player={player} 搜索失败 {type(exc).__name__}: {exc}", flush=True)
                bundles = []
            if not bundles:
                picked.append([])
                continue
            # 教师样本：只记 player 0 的（先验只用于我方）
            if use_teacher and player == 0:
                try:
                    board = np.asarray(feat.encode_board(st, player), dtype=np.float16)
                    stats_v = np.asarray(feat.encode_stats(st, player), dtype=np.float16)
                    feats.append(board)
                    stats_list.append(stats_v)
                    if TEACHER == "mcts":
                        vp = getattr(_res, "visit_policy", None)
                        if vp is None:
                            raise RuntimeError("mcts 结果缺少 visit_policy")
                        vals = np.asarray(vp, dtype=np.float32)
                    else:
                        vals = np.asarray(search.last_scores, dtype=np.float32)
                    for b, v in zip(bundles, vals):
                        if not b or len(b[0]) < 3:
                            continue          # HOLD（空 bundle）不参与位置先验训练
                        cls, a0, a1 = int(b[0][0]), int(b[0][1]), int(b[0][2])
                        ccls.append(cls); cxs.append(a0); cys.append(a1); cvals.append(float(v))
                    offs.append(len(ccls))
                    rounds.append(rnd)
                    n_teacher += 1
                except Exception as exc:
                    print(f"  ⚠️ 记录失败 {type(exc).__name__}: {exc}", flush=True)
            ops = ops_from_bundle(bundles[0])
            try:
                st.apply_operation_list(player, ops)
            except Exception:
                pass
            picked.append(ops)
        st.advance_round()
        if rnd % 50 == 0:
            print(f"  [g{seed}] round={rnd} 教师样本 {n_teacher} 累计候选 {len(ccls)} "
                  f"用时 {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(
        out_path,
        feat=np.asarray(feats, dtype=np.float16) if feats else np.zeros((0, 28, 19, 19), np.float16),
        stats=np.asarray(stats_list, dtype=np.float16) if stats_list else np.zeros((0, 1), np.float16),
        cand_cls=np.asarray(ccls, dtype=np.int8),
        cand_x=np.asarray(cxs, dtype=np.int16),
        cand_y=np.asarray(cys, dtype=np.int16),
        cand_val=np.asarray(cvals, dtype=np.float32),
        cand_off=np.asarray(offs, dtype=np.int32),
        rounds=np.asarray(rounds, dtype=np.int32),
    )
    dt = time.time() - t0
    print(f"  [done] teacher={TEACHER} g{seed}: 教师样本 {n_teacher} 候选 {len(ccls)} 用时 {dt / 60:.1f} min -> {out_path.name}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
