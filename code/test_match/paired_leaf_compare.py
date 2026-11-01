# -*- coding: utf-8 -*-
"""配对叶子对比：把两个候选放到**同一条对手随机轨迹**下跑 rollout，隔离"候选自身效果"。

（flat rollout 本身给每个候选不同的 rng 流 ⇒ s2 差里混了"对手随机轨迹不同"的噪声。
  这里用同一个 rng 流把两个候选各跑一遍，叶子值差 = 候选动作的纯净效果。）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
for p in [ROOT / "code", ROOT / "code" / "cpp_engine", ROOT / "Ant-Game", ROOT / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.backend.model import Operation                                   # noqa: E402
from SDK.utils.constants import OperationType                             # noqa: E402
from SDK.utils.features import FeatureExtractor                           # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models                 # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy         # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade            # noqa: E402

DATA = ROOT / "training_history" / "collect_v2_s1"
CKPT = str(ROOT / "checkpoints" / "posnet_A_k5_m32.pt")
CFG = dict(k=48, rollouts_per_candidate=64, horizon=16, value_batch=32,
           rollout_mode="random", rollout_random_side="opponent",
           rollout_prob_mode="decay", rollout_decay_c=1.0, rollout_decay_k=3.0,
           rollout_share_prefix=True, rollout_seed=53, max_total_rounds=49152,
           eval_mode="net")


def replay_state(seed, target_round, target_player):
    rp = json.loads((DATA / f"az_selfplay_seed{seed}.replay.json").read_text(encoding="utf-8"))
    by_round = {int(s["round"]): s for s in rp["steps"]}
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    max_r = max(by_round)
    for r in range(0, min(target_round, max_r) + 1):
        s = by_round.get(r)
        if s is None:
            continue
        if r < target_round:
            st.apply_operation_list(0, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in s["op0"]])
            st.apply_operation_list(1, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in s["op1"]])
            if not st.terminal:
                st.advance_round()
        elif r == target_round and target_player == 1:
            st.apply_operation_list(0, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in s["op0"]])
    return st


def child_for(state, player, bundle):
    child = state.clone()
    ops = [Operation(OperationType(int(t)), int(a0), int(a1)) for (t, a0, a1) in bundle]
    if ops:
        child.apply_operation_list(player, ops)
    if player == 1:
        child.advance_round()
    return child


def run_leaves(searcher, child, player, base):
    rngs = [np.random.default_rng(base + k) for k in range(64)]
    searcher._dump_tree = True
    searcher._tree_rollout_value(child, player, rngs)
    return np.asarray(searcher._tree_dump["leaf_values"])


def main():
    seed, rnd, player = 9000195, 160, 1
    cand_lines = (DATA / f"az_selfplay_seed{seed}.cand.jsonl").read_text(encoding="utf-8").splitlines()
    ent_lines = (DATA / f"az_selfplay_seed{seed}.ents.jsonl").read_text(encoding="utf-8").splitlines()
    for i, cl in enumerate(cand_lines):
        c = json.loads(cl)
        if int(c["round"]) == rnd and int(c["player"]) == player:
            cand, s1, s2 = c["cand"], c["s1"], c["s2"]
            line_idx = i
            break
    decision_idx = line_idx + 1
    light_idx = [i for i, b in enumerate(cand) if any(int(o[0]) == 21 for o in b)]
    li = max(light_idx, key=lambda i: s2[i])
    ci = c["chosen_idx"]

    st = replay_state(seed, rnd, player)
    _, _, vm = load_three_models(CKPT)
    vm.eval()
    feat = FeatureExtractor(max_actions=96)
    searcher = FlatRolloutGreedy(lambda s, p: None, **CFG, value_model=vm, feature_extractor=feat)

    light_bundle = cand[li]
    chosen_bundle = cand[ci]
    child_l = child_for(st, player, light_bundle)
    child_c = child_for(st, player, chosen_bundle)
    print(f"seed={seed} r={rnd} p{player}  闪电={light_bundle}  建塔(被选)={chosen_bundle}")
    print(f"实际 s2：闪电={s2[li]:+.4f}  建塔={s2[ci]:+.4f}")

    # 用同一条 rng 流（被选候选的流）把两个候选各跑一遍
    base_c = ((53 * 1000003 + decision_idx) * 100003 + ci * 101)
    base_l = ((53 * 1000003 + decision_idx) * 100003 + li * 101)
    lv_l_same = run_leaves(searcher, child_l, player, base_c)   # 闪电，用建塔的 rng
    lv_c_same = run_leaves(searcher, child_c, player, base_c)   # 建塔，用自己的 rng
    lv_l_own = run_leaves(searcher, child_l, player, base_l)    # 闪电，用自己的 rng

    def stat(v):
        return f"μ={np.mean(v):+.4f} 中位={np.median(v):+.4f} min={v.min():+.4f} max={v.max():+.4f} σ={np.std(v):+.4f}"

    print(f"\n同一条对手轨迹（建塔的 rng）下：")
    print(f"  闪电 叶子: {stat(lv_l_same)}")
    print(f"  建塔 叶子: {stat(lv_c_same)}")
    print(f"  纯净差 (闪电−建塔) = {np.mean(lv_l_same) - np.mean(lv_c_same):+.4f}")
    print(f"\n闪电用自己的 rng 流:")
    print(f"  闪电 叶子: {stat(lv_l_own)}")
    print(f"  (与建塔自己的 rng 差 = {np.mean(lv_l_own) - np.mean(lv_c_same):+.4f} ⇒ 这就是 s2 里看到的差)")


if __name__ == "__main__":
    main()
