"""冒烟：UnifiedActionSearch 能否枚举出升级动作、跑通、耗时多少。

构造一个"有塔"的真实感局面（先建两座塔再推进几十回合），
然后看：① 各类动作数量 ② 类边际分数 ③ 选中什么 ④ 耗时。
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import OperationType  # noqa: E402
from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models, make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.az_intent.unified_action_search import UnifiedActionSearch  # noqa: E402

CKPT = str(_REPO / "checkpoints/posnet_A_k5_m32.pt")
feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt(CKPT, feat, value_tanh=True, value_rel_to_abs=0.0)
_, _, value_model = load_three_models(CKPT)
value_model.eval()

# 造一个有塔的中局：建两座塔 → 推进 → 再建一座 → 推进
facade = GameStateFacade.initial(732)
from SDK.utils.constants import HIGHLAND_CELLS
built = 0
for (x, y) in HIGHLAND_CELLS[0]:
    op = Operation(OperationType.BUILD_TOWER, int(x), int(y))
    if facade.can_apply_operation(0, op):
        facade.apply_operation_list(0, [op])
        built += 1
        if built >= 2:
            break
print(f"建塔 {built} 座")
for _ in range(40):
    facade.advance_round()
for (x, y) in HIGHLAND_CELLS[0]:
    op = Operation(OperationType.BUILD_TOWER, int(x), int(y))
    if facade.can_apply_operation(0, op):
        facade.apply_operation_list(0, [op])
        break
for _ in range(40):
    facade.advance_round()

print(f"[局面] round={facade.round_index} 我方塔={[(t.x, t.y, int(t.tower_type)) for t in facade.towers_of(0)]} "
      f"金币={facade.coins[0]}")

searcher = UnifiedActionSearch(net_fn, horizon=32, candidate_cap=256, reeval_topk=8,
                               total_rounds=39000, min_samples_per_candidate=30,
                               value_model=value_model, feature_extractor=feat,
                               eval_mode="net", value_batch=32, verbose=True)

t0 = time.perf_counter()
res = searcher.search(facade, 0, temperature=0.0)
dt = time.perf_counter() - t0
print(f"\n[搜索] {dt:.2f} s，总回合 {searcher.last_rounds}")
print(f"[选中] {res.chosen_bundle}")
print("[计时]", {k: round(v, 2) for k, v in searcher.last_timing.items()})

by_tag = collections.Counter(tag for _, tag, _, _ in searcher.last_actions)
print(f"\n[动作枚举] {sum(by_tag.values())} 个：{dict(by_tag)}")
print("\n[类边际分数]")
for tag, score in sorted(searcher.last_class_scores.items(), key=lambda kv: -kv[1]):
    n_tag = by_tag[tag]
    print(f"  {tag:<10} 边际均值 {score:+.4f}  （{n_tag} 个动作）")

print("\n[各类最好的动作]")
for tag in ("upgrade", "build", "downgrade", "lightning", "hold"):
    cands = [(m, ops, k) for ops, t, m, k in searcher.last_actions if t == tag and np.isfinite(m)]
    if not cands:
        print(f"  {tag}: 无")
        continue
    cands.sort(reverse=True)
    m, ops, k = cands[0]
    print(f"  {tag:<10} {ops}  分数 {m:+.4f}  样本 {k}")
