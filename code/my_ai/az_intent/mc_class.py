"""方法侧 M6：**真跑到底的蒙特卡洛（greedy rollout）类决策**。

动机（本项目已证的瓶颈链）：
  * 搜索的**类决策**受制于**叶评价**——而价值网**看不到动作**（类级 adv ~80% 并列），
    所以"该不该花"在 1-ply / 4-plies 的价值里**分不出来**（M5：terminal 头 ⇒ 建塔 68/局、分数 0.5）；
  * 换视野（rel/kgeo）能改行为（建塔 68→38.8），但 128 局验收**反而更差**（−7.0pp）；
  * 手写储备规则（诊断）值 +13…20pp，但按章程第②条不能当交付物。

⇒ **本模块做的事**：把"这个类好不好"的判断**从学出来的价值换成真实模拟**——
  对每个候选类，**把局面克隆一份、把该候选打下去、然后双方都用贪心策略一路打到终局**，
  用**终局 HP 差**当分数。**没有任何手写规则**（比较全靠模拟结果），也不依赖价值网。

用法（由 `BundleMCTS(pos_pin="mc_light")` 调用）：
  * 只在**闪电可执行**的回合做模拟比较（≈ 11 次/局 ⇒ 成本可控）；
  * 其余回合退回原行为（argmax；闪电不可执行 ⇒ HOLD），与 A1 一致。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

OP_LIGHTNING = 17
OP_HOLD = 23


def _decode_class_op(head_logits, action_map, class_mask, position_mask, state, player, cid):
    """把一个类解成一个 op（贪心位置：该通道 action_map 最大的合法格）。"""
    from my_ai.decoder import decode_head
    hl = np.asarray(head_logits, dtype=np.float32)
    op = decode_head(hl, action_map, np.array(class_mask, copy=True),
                     np.array(position_mask, copy=True), state, player,
                     class_id=int(cid), temperature=0.0, pos_temperature=0.0,
                     intent_decoding=True)
    if op is None:
        return None
    return (int(op.op_type), int(op.arg0), int(op.arg1))


def greedy_step(net_fn, state, player):
    """贪心一步：类 = 类头 argmax（不套掩码）；位置 = 该通道 action_map 的 argmax 合法格。

    返回 op 元组或 None（解不出来 = 本回合不出手）。
    """
    from my_ai.decoder import make_class_mask, make_position_masks
    out = net_fn(state, player)
    hl = np.asarray(out["head_logits"][0], dtype=np.float32)
    am = np.asarray(out["action_map"], dtype=np.float32)
    pm = make_position_masks(state, player, intent_decoding=True)
    cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)
    cid = int(np.argmax(hl))
    return _decode_class_op(hl, am, cm, pm, state, player, cid)


def rollout_value(net_fn, state, player, max_rounds: int = 512) -> float:
    """从 `state` 起**克隆**并贪心打到终局，返回**该玩家视角**的终局 HP 差（/HP_SCALE，clip ±1）。"""
    from my_ai.az_intent.mcts import HP_SCALE
    st = state.clone()
    r = 0
    while (not st.terminal) and r < max_rounds:
        for pl in (0, 1):
            if st.terminal:
                break
            t = greedy_step(net_fn, st, pl)
            if t is not None:
                st.apply_operation_list(pl, [Operation(OperationType(int(t[0])), int(t[1]), int(t[2]))])
            if pl == 1:
                st.advance_round()
        r += 1
    diff = float(st.bases[0].hp - st.bases[1].hp)
    v = np.clip(diff / HP_SCALE, -1.0, 1.0)
    return float(v if player == 0 else -v)


def mc_choose_class(net_fn, state, player, cand_classes, base_head_logits, action_map,
                    class_mask, position_mask, max_rounds: int = 512):
    """在 `cand_classes` 里按"**先打这一手、再贪心打完**"的终局 HP 差选一个类。

    返回 (class_id, 每个候选的分数 dict)。全部候选都解不出来时返回 (None, {})。
    """
    scores = {}
    for cid in cand_classes:
        t = _decode_class_op(base_head_logits, action_map, class_mask, position_mask,
                             state, player, cid)
        st = state.clone()
        if t is not None:
            st.apply_operation_list(player, [Operation(OperationType(int(t[0])), int(t[1]), int(t[2]))])
        if player == 1:
            st.advance_round()
        scores[int(cid)] = rollout_value(net_fn, st, player, max_rounds=max_rounds)
    best = max(scores, key=lambda c: scores[c])
    return int(best), scores
