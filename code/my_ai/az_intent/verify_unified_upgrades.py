"""验证两件事（都基于**真实对局中局**，而不是空转造出来的局面）：

1. `UnifiedActionSearch` 在**我方有塔**时，确实会枚举出「升级」候选并给它们打分；
2. `mc_score='net'`（用 rollout+值网当类仲裁器）能跑通，并打印各候选类分数。

局面来源：用日志里双方真实下发的 ops 在真实引擎里重放 seed 732 到第 N 回合。
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import OperationType  # noqa: E402
from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models, make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.az_intent.unified_action_search import UnifiedActionSearch  # noqa: E402

LOGS = _REPO / "match_results"
SEED = 732
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def load_ops(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if '"rank":1,' not in line or "base_heuristic" not in line:
            continue
        record = json.loads(line)
        ops = []
        if record["key"] != "hold":
            for segment in record["key"].split("|"):
                if segment.startswith("F:"):
                    continue
                parts = segment.split(":")
                if len(parts) < 2:
                    continue
                vals = [int(x) for x in parts[:3]] + [-1] * (3 - min(3, len(parts)))
                ops.append((vals[0], vals[1], vals[2]))
        out[(record["round"], record["player"])] = ops
    return out


ops0 = load_ops(LOGS / f"ai0_runnerup_seed{SEED}.stderr.log")
ops1 = load_ops(LOGS / f"ai1_ai3_linear_seed{SEED}.stderr.log")
facade = GameStateFacade.initial(SEED)
for r in range(TARGET):
    for player, table in ((0, ops0), (1, ops1)):
        ops = table.get((r, player), [])
        if ops:
            facade.apply_operation_list(player, [
                Operation(OperationType(int(t)), int(a0), int(a1)) for t, a0, a1 in ops])
    facade.advance_round()

towers0 = [(t.x, t.y, int(t.tower_type)) for t in facade.towers_of(0)]
towers1 = [(t.x, t.y, int(t.tower_type)) for t in facade.towers_of(1)]
print(f"[局面] round={facade.round_index} 金币={facade.coins}")
print(f"  我方塔 {len(towers0)}：{towers0}")
print(f"  对方塔 {len(towers1)}：{towers1}")

feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt(str(_REPO / "checkpoints/posnet_A_k5_m32.pt"), feat,
                                  value_tanh=True, value_rel_to_abs=0.0)
_, _, value_model = load_three_models(str(_REPO / "checkpoints/posnet_A_k5_m32.pt"))
value_model.eval()

# ── ① 统一搜索：是否枚举出升级候选 ──────────────────────────────────────────
searcher = UnifiedActionSearch(net_fn, horizon=32, candidate_cap=256, reeval_topk=8,
                               total_rounds=20000, min_samples_per_candidate=20,
                               value_model=value_model, feature_extractor=feat,
                               eval_mode="net", value_batch=32, verbose=True)
t0 = time.perf_counter()
res = searcher.search(facade, 0, temperature=0.0)
print(f"\n[① 统一搜索] {time.perf_counter()-t0:.2f} s，选中 {res.chosen_bundle}")
by_tag = collections.Counter(tag for _, tag, _, _ in searcher.last_actions)
print(f"  动作枚举：{dict(by_tag)}")
for tag in ("upgrade", "build", "downgrade", "lightning", "hold"):
    cands = sorted([(m, ops, k) for ops, t, m, k in searcher.last_actions
                    if t == tag and m == m], reverse=True)
    if cands:
        print(f"  {tag:<10} 最好 {cands[0][1]} 分数 {cands[0][0]:+.4f} 样本 {cands[0][2]}"
              f"（共 {len(cands)} 个）")
    else:
        print(f"  {tag:<10} 无")

# ── ② mc_score='net'：用 rollout+值网当类仲裁器 ─────────────────────────────
mcts = BundleMCTS(net_fn, iterations=1, max_depth_rounds=4, k=1, sample_mult=1,
                  search_mode="pos-only", pos_pin="mc_quiet",
                  mc_every=1, mc_horizon=32, mc_root_only=1, mc_score="net",
                  mc_net_samples=2)
root = BundleNode(state=facade.clone(), player=0)
mcts.last_root = root
mcts._mc_root = root
t0 = time.perf_counter()
mcts._expand(root, 24)
print(f"\n[② mc_score=net] {time.perf_counter()-t0:.2f} s")
print(f"  候选 {len(root.bundles)} 个，前 3：{root.bundles[:3]}")
scores = getattr(mcts, "mc_last_scores", None)
if scores:
    print("  各候选类分数：")
    for key, val in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"    {key}: {val:+.4f}")
