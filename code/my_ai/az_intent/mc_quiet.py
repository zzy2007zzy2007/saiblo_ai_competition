"""方法侧 M6b：**安静 rollout 的蒙特卡洛类决策**（`pos_pin=mc_quiet`）。

为什么加这一版（M6 的实测故障）：
  * M6（`mc_light`，贪心 rollout：每一步都调 `net_fn` 解码并 `apply_operation_list`）
    在**确定性复现**的局面下崩掉（seed 11、我方先手、第 55 回合 = MC 第一次触发处；
    换边 `11r` 正常）⇒ **原生崩溃**（Python 侧没有 traceback）；128 局批因此 **32% INVALID**。
  * ⇒ 换成**更便宜、更安全**的模拟：**只把候选那一手打下去，然后让引擎"谁都不出招"地推进到终局**
    （纯 `advance_round()`，**不调网络、不 apply 别的 op**）。成本 ≈ 每候选 ~0.3–0.5 s，
    因此**每个回合都能比**（不像 M6 只能偶尔比）。
  * 语义变化（如实记）：安静 rollout 里双方都不再操作 ⇒ 终局 HP 差主要反映**这一手对基地竞赛的纯效应**
    （塔的伤害/闪电的伤害与清场），**不再是"双方正常发挥后的结果"**。这是**有偏但低方差**的信号。
"""
from __future__ import annotations

import numpy as np

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

from my_ai.az_intent.mcts import HP_SCALE


def _decode_class_op(head_logits, action_map, class_mask, position_mask, state, player, cid):
    from my_ai.decoder import decode_head
    hl = np.asarray(head_logits, dtype=np.float32)
    op = decode_head(hl, action_map, np.array(class_mask, copy=True),
                     np.array(position_mask, copy=True), state, player,
                     class_id=int(cid), temperature=0.0, pos_temperature=0.0,
                     intent_decoding=True)
    if op is None:
        return None
    return (int(op.op_type), int(op.arg0), int(op.arg1))


def quiet_value(state, player, max_rounds: int = 512) -> float:
    """把**当前这一步之后**的局面"谁都不出招"地推到终局，返回该玩家视角的终局 HP 差（clip ±1）。"""
    st = state.clone()
    r = 0
    while (not st.terminal) and r < max_rounds:
        st.advance_round()
        r += 1
    diff = float(st.bases[0].hp - st.bases[1].hp)
    v = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
    return float(v if player == 0 else -v)


def mc_quiet_choose(state, player, cand_classes, head_logits, action_map,
                    class_mask, position_mask, max_rounds: int = 512):
    """候选类各自"打下去 + 安静推进到终局"，取终局 HP 差最大者。"""
    scores = {}
    for cid in cand_classes:
        t = _decode_class_op(head_logits, action_map, class_mask, position_mask,
                             state, player, cid)
        st = state.clone()
        if t is not None:
            st.apply_operation_list(player,
                                    [Operation(OperationType(int(t[0])), int(t[1]), int(t[2]))])
        if player == 1:
            st.advance_round()
        scores[int(cid)] = quiet_value(st, player, max_rounds=max_rounds)
    best = max(scores, key=lambda c: scores[c])
    return int(best), scores
