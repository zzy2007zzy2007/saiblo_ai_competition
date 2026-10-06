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

# 埋点（AI 侧周期打印）：双向 rollout 的调用次数 / 打到终局次数 / 累计推进回合
BIDIR_STATS = {"n": 0, "terminal": 0, "steps": 0, "noop": 0}


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


def _policy_op(state, player, net_fn):
    """**rollout 内部**用我们自己的策略出一手（类 = 合法类里 logits 最大；位置 = 该类 map 最大）。

    返回 (op_type, arg0, arg1) 或 None（= 这一手不出招）。⚠️ 只用**我们自己**的网，
    绝不引入对手/人工规则（章程反捷径）。
    """
    from my_ai.decoder import decode_head, make_class_mask, make_position_masks
    out = net_fn(state, player)
    hl = np.asarray(out["head_logits"][0], dtype=np.float32)
    am = np.asarray(out["action_map"], dtype=np.float32)
    pm = make_position_masks(state, player, intent_decoding=True)
    cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)
    order = [int(c) for c in np.argsort(-hl)]
    for c in order[:4]:                      # 最多试 4 个类，省时间
        if not (0 <= c < len(cm)) or not cm[c] or not pm[c].any():
            continue
        op = decode_head(hl, am, np.array(cm, copy=True), np.array(pm, copy=True),
                         state, player, class_id=int(c), temperature=0.0,
                         pos_temperature=0.0, intent_decoding=True)
        if op is not None:
            return (int(op.op_type), int(op.arg0), int(op.arg1))
    return None


def bidir_value(state, player, net_fn, k_rounds: int, max_rounds: int = 512) -> float:
    """**K 步双向**：双方各按我们自己的策略出招 K 回合，之后安静推到终局，返回该玩家视角终局 HP 差。"""
    st = state.clone()
    steps = 0
    for _ in range(max(0, int(k_rounds))):
        if st.terminal:
            break
        for pl in (0, 1):
            op = _policy_op(st, pl, net_fn)
            if op is None:
                BIDIR_STATS["noop"] += 1
            else:
                st.apply_operation_list(pl, [Operation(OperationType(int(op[0])),
                                                       int(op[1]), int(op[2]))])
            if pl == 1:
                st.advance_round()
                steps += 1
        if st.terminal:
            break
    if st.terminal:
        BIDIR_STATS["terminal"] += 1
    BIDIR_STATS["n"] += 1
    BIDIR_STATS["steps"] += steps
    return quiet_value(st, player, max_rounds=max_rounds)


def mc_quiet_choose(state, player, cand_classes, head_logits, action_map,
                    class_mask, position_mask, max_rounds: int = 512,
                    bidir_k: int = 0, net_fn=None):
    """候选类各自"打下去 + 安静推进到终局"，取终局 HP 差最大者。

    `bidir_k > 0` ⇒ 用 **K 步双向 rollout**（预注册 `prereg_20261006_rollout_ladder.md` 的 R1/R2）：
    打候选那一手后，**双方各出招 K 回合**（用我们自己的策略），之后同样安静推到终局。
    """
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
        if int(bidir_k) > 0 and net_fn is not None:
            scores[int(cid)] = bidir_value(st, player, net_fn, int(bidir_k),
                                           max_rounds=max_rounds)
        else:
            scores[int(cid)] = quiet_value(st, player, max_rounds=max_rounds)
    best = max(scores, key=lambda c: scores[c])
    return int(best), scores
