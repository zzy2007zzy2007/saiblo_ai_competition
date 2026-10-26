"""随机化 rollout 的噪声诊断（2026-10-26）。

在每个局面用 k=48、N=16 的随机化 rollout 跑一次搜索，记录"每候选 × 每分支"的分数，然后算：
  · β：把候选按平均分降序排名，回归 mean_score ~ rank 的斜率（每差一名少多少分）
  · R²：该回归的拟合度
  · σ_within：同一候选不同分支的 SD（噪声）
  · SE = σ_within/√N；噪声跨越名次数 = SE/|β|
  · 采样曲线：用前 n 个分支算 SE/|β|（n=1,2,4,8,16）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor
import re

REPLAY_LINE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文=.*? -> (\[.*?\])\s*$")


def replay_states(log: Path, take_every: int = 20, max_states: int = 20):
    """从日志重放，按间隔取局面（只取局面，不需要亚军动作）"""
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    OP_RE = re.compile(r"op_type=<OperationType\.\w+: (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
    txt = log.read_text(encoding="utf-8", errors="replace")
    seed = int(re.search(r"对战 seed=(\d+)", txt).group(1))
    per = {}
    for line in txt.splitlines():
        m = REPLAY_LINE.match(line.strip())
        if not m:
            continue
        per.setdefault(int(m.group(1)), {})[m.group(2)] = [
            (int(a), int(b), int(c)) for a, b, c in OP_RE.findall(m.group(4))
        ]
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    out = []
    for rnd in sorted(per):
        if st.terminal or len(out) >= max_states:
            break
        if rnd % take_every == 0:
            out.append(st.clone())
        for side in ("p0", "p1"):
            raw = per[rnd].get(side) or []
            if raw:
                st.apply_operation_list(int(side[1]), [Operation(OperationType(t), a, b) for t, a, b in raw])
        st.advance_round()
    return out


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(str(REPO / "checkpoints/posnet_A_k5_m32.pt"), feat,
                                      value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(str(REPO / "checkpoints/posnet_A_k5_m32.pt"))
    vm.eval()
    k, N = 48, 16
    search = FlatRolloutGreedy(net_fn, k=k, rollouts_per_candidate=N, horizon=16,
                               max_total_rounds=10 ** 9, value_model=vm, feature_extractor=feat,
                               rollout_mode="random", rollout_random_side="opponent",
                               rollout_prob_mode="decay", rollout_decay_c=0.3, rollout_decay_k=1.0,
                               rollout_act_prob=0.3, rollout_seed=71)
    logs = sorted((REPO / "match_results/ladder_logs/VS_RUNNERUP_cells4").glob("*.log"))[:6]
    rows = []
    for lp in logs:
        for st in replay_states(lp):
            for player in (0, 1):
                search.search(st, player)
                smp = search.last_samples
                if not smp:
                    continue
                mat = np.array([s for s in smp if s], dtype=float)  # (A, N)
                if mat.shape[0] < 8 or mat.shape[1] < 2:
                    continue
                means = mat.mean(1)
                order = np.argsort(-means)
                mean_sorted = means[order]
                ranks = np.arange(1, len(mean_sorted) + 1)
                b, a = np.polyfit(ranks, mean_sorted, 1)
                pred = a + b * ranks
                ss_res = float(((mean_sorted - pred) ** 2).sum())
                ss_tot = float(((mean_sorted - mean_sorted.mean()) ** 2).sum()) + 1e-12
                r2 = 1 - ss_res / ss_tot
                sw = float(mat.std(1, ddof=1).mean())
                curve = {}
                for n in (1, 2, 4, 8, 16):
                    if n > mat.shape[1]:
                        break
                    se = sw / np.sqrt(n)
                    curve[n] = se / max(abs(b), 1e-9)
                rows.append((abs(b), r2, sw, curve))
    if not rows:
        print("  没有采到数据"); return 1
    beta = np.array([r[0] for r in rows]); r2 = np.array([r[1] for r in rows]); sw = np.array([r[2] for r in rows])
    print(f"  局面×玩家数 = {len(rows)}   k={k} N={N}")
    print(f"  |β| 每名掉分: 中位 {np.median(beta):.3f}   均值 {beta.mean():.3f}")
    print(f"  R²: 中位 {np.median(r2):.3f}")
    print(f"  σ_within（同一动作不同分支）: 中位 {np.median(sw):.3f}")
    print("\n  噪声跨越的名次数 = (σ_within/√n)/|β|：")
    for n in (1, 2, 4, 8, 16):
        vals = [r[3][n] for r in rows if n in r[3]]
        if vals:
            print(f"    n={n:<3} 中位 {np.median(vals):6.2f} 名   均值 {np.mean(vals):6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
