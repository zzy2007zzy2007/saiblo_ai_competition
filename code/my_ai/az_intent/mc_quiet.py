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
# 2026-10-18: rollout 内策略的"类预算"与埋点（R1 实测：4 个类的预算下约 55% 的调用解不出 op）
ROLLOUT_CLASSES = max(1, int(__import__("os").environ.get("AZAI_MC_ROLLOUT_CLASSES", "4")))
SPEND_STATS = {"calls": 0, "noop": 0, "terminal": 0, "steps": 0,
               "light": 0,          # rollout 内**真的**放出了闪电的次数（应为 0，若掩码禁类生效）
               "light_blocked": 0}  # 类 17 在候选顺序里出现、但被**掩码**挡下的次数（禁类生效的直接证据）


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


def quiet_value(state, player, max_rounds: int = 512, return_rounds: bool = False):
    """把**当前这一步之后**的局面"谁都不出招"地推到终局，返回该玩家视角的终局 HP 差（clip ±1）。"""
    st = state.clone()
    r = 0
    while (not st.terminal) and r < max_rounds:
        st.advance_round()
        r += 1
    diff = float(st.bases[0].hp - st.bases[1].hp)
    v = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
    _v = float(v if player == 0 else -v)
    return (_v, int(r)) if return_rounds else _v


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
    SPEND_STATS['calls'] += 1
    for c in order[:ROLLOUT_CLASSES]:   # 类预算 AZAI_MC_ROLLOUT_CLASSES（默认 4 = 历史值）

        if not (0 <= c < len(cm)) or not cm[c] or not pm[c].any():
            if int(c) == 17 and (not cm[c]):
                SPEND_STATS['light_blocked'] += 1   # 掩码把闪电挡下（禁类传播到 rollout 的直接证据）
            continue
        op = decode_head(hl, am, np.array(cm, copy=True), np.array(pm, copy=True),
                         state, player, class_id=int(c), temperature=0.0,
                         pos_temperature=0.0, intent_decoding=True)
        if op is not None:
            if int(op.op_type) == 21:
                SPEND_STATS['light'] += 1           # ⚠️ 若 >0 ⇒ 掩码禁类没传播进 rollout
            return (int(op.op_type), int(op.arg0), int(op.arg1))
    # 2026-10-18 补：原来**忘了在这条路上自增** ⇒ 打印出来的 `noop=0(0%)` 是恒零假读数
    #   （真实空过率看 BIDIR_STATS：D2 臂实测 530/836 = 63%）
    SPEND_STATS['noop'] += 1
    return None


def selfspend_value(state, player, net_fn, k_rounds: int, max_rounds: int = 512) -> float:
    """**单侧支出**：只有**我方**继续出招 K 回合（对手保持安静），之后照旧安静推到终局。

    这样"候选那一手花掉的钱"会以**已建成的塔**形式出现在终局里 ⇒ 评价开始给"花钱"定价（也就能看见代价）。
    """
    st = state.clone()
    for _ in range(max(0, int(k_rounds))):
        if st.terminal:
            break
        op = _policy_op(st, int(player), net_fn)
        if op is not None:
            st.apply_operation_list(int(player), [Operation(OperationType(int(op[0])),
                                                           int(op[1]), int(op[2]))])
        st.advance_round()
    return quiet_value(st, player, max_rounds=max_rounds)


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


def _decode_class_op_at_cell(head_logits, action_map, class_mask, position_mask,
                             state, player, cid, x, y):
    """把 `cid` 的 map 只留 (x,y) 一格再解码 ⇒ 强制落到那一格（用于「选类×选格」候选）。"""
    am = np.array(action_map, copy=True)
    am[cid] = -1e9
    am[cid, int(y), int(x)] = float(action_map[cid][int(y), int(x)])
    return _decode_class_op(head_logits, am, class_mask, position_mask, state, player, cid)


def mc_quiet_choose(state, player, cand_classes, head_logits, action_map,
                    class_mask, position_mask, max_rounds: int = 512,
                    bidir_k: int = 0, net_fn=None, margin_gate: float = 0.0,
                    score_mode: str = '', v_before: float = 0.0, selfspend_k: int = 0,
                    selfspend_mix: float = 1.0):
    """候选类各自"打下去 + 安静推进到终局"，取终局 HP 差最大者。

    `bidir_k > 0` ⇒ 用 **K 步双向 rollout**（预注册 `prereg_20261006_rollout_ladder.md` 的 R1/R2）：
    打候选那一手后，**双方各出招 K 回合**（用我们自己的策略），之后同样安静推到终局。
    """
    scores = {}
    for _cand in cand_classes:
        # 候选 = 类号(int) 或 (类号, x, y)（「选类×选格」）
        if isinstance(_cand, (tuple, list)) and len(_cand) == 4 and _cand[0] == 'op':
            # 显式 op 候选（不依赖 decoder）：('op', op_type, x, y) —— 用于「升级已有塔」
            cid, _x, _y = 12, int(_cand[2]), int(_cand[3])
            t = (int(_cand[1]), int(_cand[2]), int(_cand[3]))
        elif isinstance(_cand, (tuple, list)):
            cid, _x, _y = int(_cand[0]), int(_cand[1]), int(_cand[2])
            t = _decode_class_op_at_cell(head_logits, action_map, class_mask, position_mask,
                                         state, player, cid, _x, _y)
        else:
            cid, _x, _y = int(_cand), None, None
            t = _decode_class_op(head_logits, action_map, class_mask, position_mask,
                                 state, player, cid)
        st = state.clone()
        if t is not None:
            st.apply_operation_list(player,
                                    [Operation(OperationType(int(t[0])), int(t[1]), int(t[2]))])
        if player == 1:
            st.advance_round()
        # 2026-10-18 修（两个真 bug）：
        #   ① `_key` **先赋值**（原来 selfspend 分支在赋值前就 `scores[_key]` ⇒ UnboundLocalError
        #      ⇒ 被 bundle_mcts 的宽 except 吞掉 ⇒ MC 静默瘫痪、回退类头 argmax）；
        #   ② 评分走**单一 if/elif 链**（原来 bidir 的分数会被后面的安静分数无条件覆盖）。
        _key = (_cand if isinstance(_cand, (tuple, list)) else int(cid))
        if score_mode == 'value_only' and net_fn is not None:
            # 纯价值网差分：完全不做 rollout ⇒ 更快；价值网输入含 stats（金币在里）⇒ 会给花钱定价
            _v = float(net_fn(st, player)['value'])
            scores[_key] = _v - float(v_before)
        elif score_mode == 'delta' and net_fn is not None:
            # potential-based shaping（Φ=V）：只修掉"安静 rollout 看不见花钱代价"的短视
            _q = quiet_value(st, player, max_rounds=max_rounds)
            _v = float(net_fn(st, player)['value'])
            scores[_key] = _q + (_v - float(v_before))
        elif (int(selfspend_k) > 0 or int(bidir_k) > 0) and net_fn is not None:
            _s = (selfspend_value(st, player, net_fn, int(selfspend_k), max_rounds=max_rounds)
                  if int(selfspend_k) > 0 else
                  bidir_value(st, player, net_fn, int(bidir_k), max_rounds=max_rounds))
            _mix = min(1.0, max(0.0, float(selfspend_mix)))
            if _mix >= 1.0:
                scores[_key] = float(_s)
            elif _mix <= 0.0:
                scores[_key] = quiet_value(st, player, max_rounds=max_rounds)
            else:
                _q = quiet_value(st, player, max_rounds=max_rounds)
                scores[_key] = (1.0 - _mix) * float(_q) + _mix * float(_s)
        else:
            scores[_key] = quiet_value(st, player, max_rounds=max_rounds)
    # 分差门控：最高分与次高分相差太小 ⇒ **不接管**（返回 None，由调用方保留原 pin）
    # 依据：dump 里 23.6% 的决策是完全平局、另有 17.8% 在 1 HP 内 ⇒ 这些"接管"没有依据。
    if margin_gate and float(margin_gate) > 0 and len(scores) >= 2:
        _vals = sorted(scores.values(), reverse=True)
        if (_vals[0] - _vals[1]) < float(margin_gate):
            return None, scores
    best = max(scores, key=lambda c: scores[c])
    # 返回「最优候选」本身（可能是 (类,x,y)）；调用方取 [0] 当钉住的类
    return best, scores
