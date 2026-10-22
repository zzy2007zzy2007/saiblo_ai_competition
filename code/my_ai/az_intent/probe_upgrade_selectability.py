"""统计：在真实中局里，**显式升级候选**在 MC 仲裁里到底能不能被选中？

对比三种仲裁评估器（其余完全相同：mc_cells=4, mc_upg=1）：
  · quiet   ：安静 rollout 到 256 回合，终局 HP 差（现状）
  · quiet+h32：安静 rollout 到 32 回合（新默认水平）
  · net     ：打一手 → 32 回合 → **值网**（亚军风格；连续分数、给花钱定价）

输出：菜单里有没有 op 候选、op 的最高分/排名、是否被选中、以及平局数。
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import OperationType  # noqa: E402
from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models, make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.az_intent.value_batch import make_batched_value_fn  # noqa: E402

SEED = 732
LOGS = _REPO / "match_results"


def load_ops(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if '"rank":1,' not in line or "base_heuristic" not in line:
            continue
        rec = json.loads(line)
        ops = []
        if rec["key"] != "hold":
            for seg in rec["key"].split("|"):
                if seg.startswith("F:"):
                    continue
                parts = seg.split(":")
                if len(parts) < 2:
                    continue
                v = [int(x) for x in parts[:3]] + [-1] * (3 - min(3, len(parts)))
                ops.append((v[0], v[1], v[2]))
        out[(rec["round"], rec["player"])] = ops
    return out


feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt(str(_REPO / "checkpoints/posnet_A_k5_m32.pt"), feat,
                                  value_tanh=True, value_rel_to_abs=0.0)
_, _, vm = load_three_models(str(_REPO / "checkpoints/posnet_A_k5_m32.pt"))
vm.eval()
vf = make_batched_value_fn(vm, feat)

ops0 = load_ops(LOGS / f"ai0_runnerup_seed{SEED}.stderr.log")
ops1 = load_ops(LOGS / f"ai1_ai3_linear_seed{SEED}.stderr.log")

MODES = {
    "quiet256": dict(mc_score="", mc_horizon=256),
    "net32": dict(mc_score="net", mc_horizon=32, mc_net_samples=4),
    "quiet+0.05": dict(mc_score="", mc_horizon=32, mc_upgrade_bonus=0.05),
    "quiet+0.10": dict(mc_score="", mc_horizon=32, mc_upgrade_bonus=0.10),
    "quiet+0.20": dict(mc_score="", mc_horizon=32, mc_upgrade_bonus=0.20),
    "net+0.10": dict(mc_score="net", mc_horizon=32, mc_net_samples=4, mc_upgrade_bonus=0.10),
}

facade = GameStateFacade.initial(SEED)
stats = {m: collections.Counter() for m in MODES}
op_rank = {m: [] for m in MODES}
states = 0
for r in range(340):
    if facade.terminal:
        break
    for player, table in ((0, ops0), (1, ops1)):
        ops = table.get((r, player), [])
        if ops:
            facade.apply_operation_list(player, [
                Operation(OperationType(int(t)), int(a0), int(a1)) for t, a0, a1 in ops])
    facade.advance_round()
    if r % 20 != 0 or r < 40:
        continue
    for player in (0, 1):
        if not list(facade.towers_of(player)):
            continue                      # 没塔 ⇒ 不可能有升级候选
        states += 1
        for name, cfg in MODES.items():
            m = BundleMCTS(net_fn, iterations=1, k=1, sample_mult=1, search_mode="pos-only",
                           pos_pin="mc_quiet", mc_every=1, mc_root_only=1,
                           mc_cells=4, mc_upg=1, mc_value_fn=vf, **cfg)
            root = BundleNode(state=facade.clone(), player=player)
            m.last_root = root
            m._mc_root = root
            try:
                m._expand(root, 24)
            except Exception as exc:  # noqa: BLE001
                stats[name]["异常"] += 1
                continue
            scores = getattr(m, "mc_last_scores", {}) or {}
            ops_keys = [k for k in scores
                        if isinstance(k, tuple) and len(k) >= 1 and k[0] == "op"]
            if ops_keys:
                stats[name]["菜单含升级"] += 1
                best_op = max(scores[k] for k in ops_keys)
                mx = max(scores.values())
                rank = 1 + sum(1 for v in scores.values() if v > best_op + 1e-9)
                ties = sum(1 for v in scores.values() if abs(v - mx) < 1e-9)
                stats[name]["平局(≥2项)" if ties >= 2 else "无平局"] += 1
                op_rank[name].append(rank)
            if getattr(m, "mc_last_explicit", None) is not None:
                stats[name]["升级被选中"] += 1

print(f"[样本] {states} 个 (局面, 玩家) 组合（双方都有塔）\n")
print(f"{'模式':<10}{'菜单含升级':>12}{'升级被选中':>12}{'平局占比':>10}{'op中位排名':>12}")
for name in MODES:
    s = stats[name]
    n = states
    tie = s["平局(≥2项)"]
    ranks = op_rank[name]
    med = float(np.median(ranks)) if ranks else float("nan")
    print(f"{name:<10}{s['菜单含升级']:>8}/{n:<4}{s['升级被选中']:>8}/{n:<4}"
          f"{100*tie/max(1,s['菜单含升级']):>9.0f}%{med:>12.1f}")
    if s["异常"]:
        print(f"    异常 {s['异常']} 次")
