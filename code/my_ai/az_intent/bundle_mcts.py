"""Bundle-sampling MCTS (修复先验-解码不一致, 见 docs/az_bundle_sampling_mcts_plan.md).

Candidates = temperature-sampled COMPLETE action groups (up to 3 ops per player),
generated with the decoder's own sampling semantics (class sample -> position
sample -> decode).  The tree branches over bundles: 2 levels per round
(P0 bundle -> P1 bundle -> resolve).  The group prior = the joint
log-probability (sum of per-action log-probs).

net_fn(state, player) -> dict(action_map, head_logits, value): same interface
as the intent-space MCTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from SDK.backend.state import BackendState
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

from my_ai.az_intent.mcts import HP_SCALE


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    shifted = x - np.max(x)
    e = np.exp(shifted)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        out = np.zeros_like(e)
        out[0] = 1.0
        return out
    return e / s


def head_class_probs(head_logits: np.ndarray, t_class: float) -> np.ndarray:
    """Per-head class distribution (z-score softmax) — mirrors decode_head's
    sampling distribution.  Computed ONCE per expansion so the 360 samples of a
    head don't each re-do the mean/std/exp/sum."""
    head_logits = np.asarray(head_logits, dtype=np.float32)
    mean = head_logits.mean()
    std = head_logits.std() + 1e-8
    logits = (head_logits - mean) / std
    probs = np.exp(logits / t_class)
    probs /= probs.sum()
    return probs


def legal_nonlight_classes(class_mask: np.ndarray, position_mask: np.ndarray) -> list:
    """「随机钉类」的候选集：**可执行、且有合法格**、且**不是超武(17-20)/HOLD(23)** 的类。

    为什么排除超武：`intent_decoding=True` 下**买不起的超武仍在 class_mask 里**，但解码器会走
    通道 16 的自动降级 ⇒ 名义上钉 18/19/20、实际执行的是降级 ⇒ 那几回合的标注是错的
    （`eval.py:_legal_nonlight_classes` 的注释记录了 2026-09-21 的实测）。
    为什么排除 HOLD(23)：钉了也产不出位置目标。
    """
    out = []
    for c in range(len(class_mask)):
        if c == 23 or c in (17, 18, 19, 20):
            continue
        if class_mask[c] and position_mask[c].any():
            out.append(int(c))
    return out


def playable_class(
    head_logits: np.ndarray,
    action_map: np.ndarray,
    class_mask: np.ndarray,
    position_mask: np.ndarray,
    state: BackendState,
    player: int,
    *,
    intent_decoding: bool = True,
    max_probe: int = 8,
) -> int:
    """Highest-logit class that decodes to a real (non-None) operation.

    "The class the policy prefers, restricted to those that can actually be
    executed."  Used as the pinned class for the ``pos-only`` ablation when the
    raw argmax class is HOLD/unaffordable — otherwise pinning it would leave the
    position axis with no room at all and the arm would collapse to raw.
    """
    from my_ai.decoder import decode_head

    head_logits = np.asarray(head_logits, dtype=np.float32)
    for cid in np.argsort(-head_logits)[:max_probe]:
        op = decode_head(head_logits, action_map, class_mask.copy(),
                         position_mask.copy(), state, player,
                         class_id=int(cid), temperature=0.0,
                         pos_temperature=0.0, intent_decoding=intent_decoding)
        if op is not None:
            return int(cid)
    return 23  # HOLD — nothing executable


def reserve_class(
    head_logits: np.ndarray,
    action_map: np.ndarray,
    class_mask: np.ndarray,
    position_mask: np.ndarray,
    state: BackendState,
    player: int,
    reserve_coins: int,
    *,
    spend_classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
    intent_decoding: bool = True,
    max_probe: int = 24,
) -> int:
    """「留储备、花余钱」（候选 E，诊断臂；见 `docs/prereg_20261005_reserve_economy.md`）。

    规则：① `argmax` 类**可执行** ⇒ 用它（= 闪电优先，与 A1 一致）；
          ② 否则若 `state.coins[player] >= reserve_coins` ⇒ 在 `spend_classes`（建/升塔 0-16）
             里取 **logits 最高且真能解码**的那个；
          ③ 否则 ⇒ 返回 `argmax`（⇒ 该回合退化成 HOLD，与 A1 相同）。

    动机：闪电是**冷却 35 回合**限制的（实测 11.3 次/局 ≈ 上限），金币大量闲置（局末 ~500）；
    候选 D 之所以崩，是因为它没有储备概念、把金币花到买不起闪电（建塔成本 15→30→45→90→135…）。
    """
    from my_ai.decoder import decode_head

    order = np.argsort(-np.asarray(head_logits, dtype=np.float32))
    top = int(order[0])
    # ① 首选（与 A1 相同的 argmax）能执行吗？
    if decode_head(head_logits, action_map, class_mask.copy(), position_mask.copy(),
                   state, player, class_id=top, temperature=0.0,
                   pos_temperature=0.0, intent_decoding=intent_decoding) is not None:
        return top
    # ② 有储备富余 ⇒ 在"花钱类"里挑
    if int(state.coins[player]) >= int(reserve_coins):
        for cid in order[:max_probe]:
            cid = int(cid)
            if cid not in spend_classes or not class_mask[cid] or not position_mask[cid].any():
                continue
            op = decode_head(head_logits, action_map, class_mask.copy(), position_mask.copy(),
                             state, player, class_id=cid, temperature=0.0,
                             pos_temperature=0.0, intent_decoding=intent_decoding)
            if op is not None:
                return cid
    # ③ 没富余 ⇒ 维持原样（HOLD）
    return top


def _try_classes(head_logits, action_map, class_mask, position_mask, state, player,
                 order, allowed, intent_decoding, max_probe=24):
    """在 `allowed` 里按 logits 顺序找**第一个真能解码**的类；找不到返回 None。"""
    from my_ai.decoder import decode_head
    for cid in order[:max_probe]:
        cid = int(cid)
        if cid not in allowed or not class_mask[cid] or not position_mask[cid].any():
            continue
        op = decode_head(head_logits, action_map, class_mask.copy(), position_mask.copy(),
                         state, player, class_id=cid, temperature=0.0,
                         pos_temperature=0.0, intent_decoding=intent_decoding)
        if op is not None:
            return cid
    return None


def reserve_upgrade_class(
    head_logits: np.ndarray,
    action_map: np.ndarray,
    class_mask: np.ndarray,
    position_mask: np.ndarray,
    state: BackendState,
    player: int,
    reserve_coins: int,
    *,
    intent_decoding: bool = True,
) -> int:
    """候选 E3（诊断臂）：储备规则 + **优先"升级/出高级塔"（类 1-15）而不是"再建一座基础塔"（类 0）**，且**绝不降级**。

    动机（2026-10-05 实测的 ops 分布，E2 128 局）：我方 **建塔 10.70 / 升塔 0.37 / 降级 0.90 / 闪电 11.22**
    —— 也就是说"花余钱"时几乎总是**再建一座 BASIC 塔**、偶尔**升级**、还会**降级 0.9 次/局**（提示：
    "建塔成本 15→30→45→90…"涨上去以后，常数类头里排在前面且可执行的变成了降级类）。
    而对手 `rule_v4` 是 **建塔 16.3 / 升塔 9.8**。⇒ 问：**把余钱花在"升级/高级塔"上是否比"堆 BASIC 塔"更好？**

    规则：① `argmax`（闪电）可执行 ⇒ 用它；② 否则若 `coins ≥ reserve` ⇒ 先在 **1-15**（升级/高级塔）里挑，
    再退到 **0**（建基础塔）；**永不选 16（降级）**；③ 否则维持原样（HOLD）。
    """
    order = np.argsort(-np.asarray(head_logits, dtype=np.float32))
    top = int(order[0])
    from my_ai.decoder import decode_head
    if decode_head(head_logits, action_map, class_mask.copy(), position_mask.copy(),
                   state, player, class_id=top, temperature=0.0,
                   pos_temperature=0.0, intent_decoding=intent_decoding) is not None:
        return top
    if int(state.coins[player]) >= int(reserve_coins):
        cid = _try_classes(head_logits, action_map, class_mask, position_mask, state, player,
                           order, set(range(1, 16)), intent_decoding)
        if cid is not None:
            return cid
        cid = _try_classes(head_logits, action_map, class_mask, position_mask, state, player,
                           order, {0}, intent_decoding)
        if cid is not None:
            return cid
    return top


def sample_bundle(
    net_out: dict,
    state: BackendState,
    player: int,
    *,
    t_class: float = 1.0,
    t_pos: float = 1.0,
    rng: np.random.Generator,
    intent_decoding: bool = True,
    position_mask: np.ndarray | None = None,
    class_mask: np.ndarray | None = None,
    class_probs: list | None = None,
    class_ids: list | None = None,
) -> tuple[list[Operation], list]:
    """Sample one complete action group by calling the DECODER's own decode_head
    for each head (single source of truth — the search's candidates always match
    what the decoder would play, including the intent-decoding downgrade fallback).

    ``position_mask``/``class_mask`` may be passed in (cached at the node level);
    they are COPIED here since the sampling mutates the position mask.

    Returns (ops, intents) where ``intents`` = per-head (class, x, y) or None —
    the SAMPLED network intent for each head (the identity used for counting /
    marginalization in AlphaZero training).
    """
    from my_ai.decoder import decode_head, make_class_mask, make_position_masks

    if position_mask is None:
        position_mask = make_position_masks(state, player, intent_decoding=intent_decoding)
    else:
        position_mask = position_mask.copy()
    if class_mask is None:
        class_mask = make_class_mask(state, player, position_mask=position_mask,
                                     intent_decoding=intent_decoding)
    else:
        class_mask = class_mask.copy()
    action_map = np.asarray(net_out["action_map"], dtype=np.float32)

    ops: list[Operation] = []
    pending: tuple = ()
    intents: list = [None, None, None]

    for head in range(3):
        head_logits = np.asarray(net_out["head_logits"][head], dtype=np.float32)
        sampled_class: list[int] = []
        sampled_pos: list = []
        op = decode_head(
            head_logits, action_map, class_mask, position_mask, state, player,
            rng=rng, temperature=t_class, intent_decoding=intent_decoding,
            pos_temperature=t_pos,
            sampled_class_out=sampled_class, sampled_pos_out=sampled_pos,
            class_probs=(class_probs[head] if class_probs else None),
            class_id=(class_ids[head] if class_ids else None),
        )
        # record the sampled network intent (class, position) per head
        if sampled_class:
            c = int(sampled_class[0])
            if sampled_pos:
                _ch, x, y, _lp, _m = sampled_pos[0]
                intents[head] = (c, int(x), int(y))
            else:
                intents[head] = (c, -1, -1)
        if op is None:
            continue
        if not state.can_apply_operation(player, op, pending):
            continue
        ops.append(op)
        pending = pending + (op,)
        if op.op_type == OperationType.BUILD_TOWER:
            position_mask[:, op.arg0, op.arg1] = False
    return ops, intents


# ─── Tree ───────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class BundleNode:
    state: BackendState
    player: int
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: list = field(default_factory=list)
    bundles: list = field(default_factory=list)  # aligned with children (ops tuples)
    intent_counts: list = field(default_factory=list)  # aligned with children: per-child
        # list of 3 dicts {intent: sample_count} — for AlphaZero marginalization
    net_value: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class BundleSearchResult:
    bundles: list[tuple]
    visit_policy: np.ndarray
    chosen_index: int
    chosen_bundle: tuple
    root_value: float
    intent_counts: list = field(default_factory=list)


class BundleMCTS:
    def __init__(
        self,
        net_fn,
        *,
        iterations: int = 64,
        max_depth_rounds: int = 3,
        k: int = 24,
        c_puct: float = 1.25,
        t_class: float = 1.0,
        t_pos: float = 1.0,
        sample_mult: int = 15,
        seed: int = 0,
        search_mode: str = "joint",
        pos_pin: str = "argmax",
        class_pin_random_prob: float = 0.0,
        reserve_coins: int = 90,
        skip_single_candidate: bool = False,
        pos_prior_fn=None,
        candidate_fn=None,
    ) -> None:
        if search_mode not in ("joint", "class-only", "pos-only"):
            raise ValueError(f"unknown search_mode: {search_mode}")
        if pos_pin not in ("argmax", "playable", "reserve", "reserve_up", "masked", "q", "mc_light"):
            raise ValueError(f"unknown pos_pin: {pos_pin}")
        self.net_fn = net_fn
        self.iterations = iterations
        self.max_depth_rounds = max_depth_rounds
        self.k = k
        self.c_puct = c_puct
        self.t_class = t_class
        self.t_pos = t_pos
        self.sample_mult = sample_mult  # sample k*sample_mult times, keep top-k by count
        self.search_mode = search_mode
        self.pos_pin = pos_pin
        # OPTIONAL「随机钉类」(2026-10-05 实施 docs/az_posnet_random_class_plan.md 的计划)：
        # 在 pos-only 下**按头独立**以概率 p 把"钉住的类"从 argmax 换成一个**均匀随机的合法非闪电类**，
        # 位置仍由搜索做价值排序。用途：**采集**时给数据加"有塔局面"的多样性（部署侧不用它）。
        #   * 候选集 = `class_mask[c]` 且 `position_mask[c].any()` 且 c ∉ {17,18,19,20(超武), 23(HOLD)}
        #     （超武在 `intent_decoding` 下买不起也仍在 mask 里，会走通道 16 的自动降级 ⇒ 标注会错，
        #      与 `eval.py:_legal_nonlight_classes` 同一个坑，故一并排除）；
        #   * 用 `self.rng` 抽 ⇒ 同一 seed 可复现；
        #   * **默认 0.0 ⇒ 整段代码不执行，原有行为逐位不变**（A0/A1 的读数不受影响）。
        self.class_pin_random_prob = float(class_pin_random_prob)
        # 候选 E（诊断臂）：pos_pin="reserve" 时的"闪电储备"门槛（金币）。
        self.reserve_coins = int(reserve_coins)
        self.skip_single_candidate = skip_single_candidate
        # OPTIONAL external position prior (2026-09-18), default None = 完全维持原行为。
        # 用途：让调用方**替换**"钉类的合法格"上的采样分布，从而测"让价值网决定候选菜单"
        # 这类问题（见 _tmp_position_choice_ab.py 的 --pos-prior value）。签名：
        #   pos_prior_fn(state, player, net_out, pinned_classes, is_root) -> action_map | None
        # 返回 None 表示不覆盖。只碰被钉的类那几条通道 ⇒ 类轴与解码器的降级逻辑都不受影响。
        self.pos_prior_fn = pos_prior_fn
        # OPTIONAL external CANDIDATE MENU (2026-09-19), default None = 原有采样行为。
        # 用途：把候选来源换成"外部给定的菜单"（例如官方启发式 top-K），从而测
        # "在这份菜单里，价值网 + 搜索能不能选出比菜单自己的排序更好的招"
        # （见 docs/az_heuristic_menu_search_plan.md）。签名：
        #   candidate_fn(state, player, k) -> list[tuple[(op_type,arg0,arg1), ...]] | None
        #   返回 None 表示"这一层不接管"，回落原有采样。
        # 先验一律**均匀**（不掺官方 score，避免把"被检验的排序"又当先验偷回来）。
        self.candidate_fn = candidate_fn
        self.rng = np.random.default_rng(seed)
        self.last_root: BundleNode | None = None

    # ── internals ────────────────────────────────────────────────────────

    def _puct(self, parent: BundleNode, child: BundleNode) -> float:
        explore = self.c_puct * child.prior * math.sqrt(parent.visits + 1e-6) / (1 + child.visits)
        return -child.mean_value + explore

    def _terminal_value(self, state: BackendState, player: int) -> float:
        if state.winner is None:
            return 0.0
        diff = state.bases[player].hp - state.bases[1 - player].hp
        v = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
        if diff == 0:
            v = 0.1 if state.winner == player else -0.1
        return v

    def _expand(self, node: BundleNode, k: int) -> float:
        if node.state.terminal:
            node.expanded = True
            return self._terminal_value(node.state, node.player)
        net_out = self.net_fn(node.state, node.player)
        node.net_value = float(net_out["value"])

        # 外部候选菜单（见 __init__ 的 candidate_fn 注释 / docs/az_heuristic_menu_search_plan.md）：
        # 接管则**完全跳过**策略采样，直接用外部菜单建子节点（先验均匀）。
        if self.candidate_fn is not None:
            ext = self.candidate_fn(node.state, node.player, k)
            if ext:
                nxt = 1 - node.player
                pri = 1.0 / float(len(ext))
                for key in ext:
                    ops = [Operation(OperationType(int(x[0])), int(x[1]), int(x[2]))
                           for x in key]
                    child_state = node.state.clone()
                    if ops:
                        child_state.apply_operation_list(node.player, ops)
                    if node.player == 1:
                        child_state.advance_round()
                    node.children.append(
                        BundleNode(state=child_state, player=nxt, prior=pri))
                    node.bundles.append(tuple((int(x[0]), int(x[1]), int(x[2])) for x in key))
                    # 对局/评估用不到训练目标，留空即可（长度必须与 children 对齐）
                    node.intent_counts.append([dict(), dict(), dict()])
                node.expanded = True
                return node.net_value

        # cache masks once (they only depend on the state, not the sample)
        from my_ai.decoder import make_class_mask, make_position_masks
        base_pos_mask = make_position_masks(node.state, node.player, intent_decoding=True)
        base_cls_mask = make_class_mask(node.state, node.player,
                                        position_mask=base_pos_mask, intent_decoding=True)

        # sample k*sample_mult times; aggregate per decoded key the count and the
        # per-head intent sample counts (for AlphaZero marginalization).
        # Vectorized class sampling: compute each head's class distribution once
        # and batch-sample all k*sample_mult classes per head (one rng.choice
        # each instead of 360×3 sequential calls).
        n_samples = k * self.sample_mult
        head_logits_list = [np.asarray(net_out["head_logits"][h], dtype=np.float32)
                            for h in range(3)]
        # ── class/position factorization ablation (see docs/az_action_factorization_idea.md §8)
        #   joint      : classes sampled (t_class), positions sampled (t_pos) — the default
        #   pos-only   : class pinned to the decoder's own argmax (= raw's class choice),
        #                so candidates differ ONLY in position → isolates position search
        #   class-only : positions pinned to argmax (t_pos<=0), so candidates differ
        #                ONLY in class → isolates class search
        if self.search_mode == "pos-only":
            class_probs_arg = None
            if self.pos_pin == "playable":
                pinned = [playable_class(hl, net_out["action_map"], base_cls_mask,
                                         base_pos_mask, node.state, node.player)
                          for hl in head_logits_list]
            elif self.pos_pin == "reserve":
                # 候选 E（诊断）：闪电优先、有富余才花在建/升塔上（见 reserve_class 的注释）
                pinned = [reserve_class(hl, net_out["action_map"], base_cls_mask,
                                        base_pos_mask, node.state, node.player,
                                        self.reserve_coins)
                          for hl in head_logits_list]
            elif self.pos_pin == "reserve_up":
                # 候选 E3（诊断）：储备规则 + 优先升级/高级塔、不降级
                pinned = [reserve_upgrade_class(hl, net_out["action_map"], base_cls_mask,
                                                base_pos_mask, node.state, node.player,
                                                self.reserve_coins)
                          for hl in head_logits_list]
            elif self.pos_pin == "mc_light":
                # 方法侧 M6（见 code/my_ai/az_intent/mc_class.py + docs/prereg_20261006_mc_class.md）：
                # **只在闪电可执行的回合**做"真跑到底"的蒙特卡洛比较：候选 {闪电, HOLD, 经济类}，
                # 每个候选先打下去、再双方贪心打到终局，用**终局 HP 差**选类；其余回合 = A1（argmax ⇒ 闪电/HOLD）。
                _AM = np.asarray(net_out["action_map"], dtype=np.float32)
                _economy = None
                _best_v = -np.inf
                for _c in range(0, 17):
                    if base_cls_mask[_c] and base_pos_mask[_c].any():
                        _v = float(np.where(base_pos_mask[_c], _AM[_c], -np.inf).max())
                        if _v > _best_v:
                            _economy, _best_v = _c, _v
                pinned = []
                for _hl in head_logits_list:
                    _hl = np.asarray(_hl, dtype=np.float32)
                    _top = int(np.argmax(_hl))
                    _can_light = bool(base_cls_mask[17] and base_pos_mask[17].any())
                    if _top == 17 and _can_light:
                        from my_ai.az_intent.mc_class import mc_choose_class
                        _cands = [17, 23] + ([_economy] if _economy is not None else [])
                        try:
                            _c, _sc = mc_choose_class(
                                self.net_fn, node.state, node.player, _cands, _hl, _AM,
                                base_cls_mask, base_pos_mask)
                            pinned.append(int(_c))
                            self.mc_last_scores = _sc
                        except Exception:  # noqa: BLE001
                            pinned.append(_top)
                    else:
                        pinned.append(_top)
            elif self.pos_pin == "q":
                # 「类条件价值」决策（方法侧 M2，预注册 docs/prereg_20261005_qhead_class_decision.md）：
                # 在**合法类**里取 `argmax_c Q(s, c)`（Q 来自 `AZAI_Q_CKPT`，见 q_head.py）。
                # 与 masked 的区别：masked 用的是**类头的 logits**（= 模仿示范频率），
                # 这里用的是**对局结果的预测**（= 这个类在这儿会不会赢）。
                _qv = net_out.get("q_classes")
                if _qv is None:
                    pinned = [int(np.argmax(hl)) for hl in head_logits_list]
                else:
                    _qv = np.asarray(_qv, dtype=np.float64)
                    _c = int(np.argmax(np.where(base_cls_mask, _qv, -np.inf)))
                    pinned = [_c] * len(head_logits_list)
            elif self.pos_pin == "masked":
                # 「掩码感知的 argmax」（2026-10-05，方法侧修**决策机制**）：
                # 在**合法类**（`class_mask` 为真，**含 HOLD 23**）里取 argmax，而不是在全 24 类里取。
                # 动机（实测）：A1 的机制是"全 24 类 argmax ⇒ 恒为闪电"，闪电不可执行时整回合退化成 HOLD
                # ⇒ 策略**结构上无法表达**"闪电不能放时才花钱"这类条件行为
                # （候选 D / `pos_pin=playable` 都会崩成"见缝就花"：建塔 60+/降级 60+/闪电 0–1）。
                # 掩码感知后，"该 HOLD 就 HOLD、该花才花"变成**策略自己可以学的东西**（= 在合法动作集上决策）。
                pinned = [int(np.argmax(np.where(base_cls_mask, hl, -np.inf)))
                          for hl in head_logits_list]
            else:
                pinned = [int(np.argmax(hl)) for hl in head_logits_list]
            # 「随机钉类」：**按头独立**以 p 换成均匀随机的合法非闪电类（位置仍由搜索排序）。
            # 默认 p=0 ⇒ 这一整段不执行 ⇒ 与旧行为逐位相同。
            if self.class_pin_random_prob > 0.0:
                _cands = legal_nonlight_classes(base_cls_mask, base_pos_mask)
                if _cands:
                    _new = []
                    for _c in pinned:
                        if self.rng.random() < self.class_pin_random_prob:
                            _new.append(int(_cands[int(self.rng.integers(len(_cands)))]))
                        else:
                            _new.append(_c)
                    pinned = _new
            class_ids = [[c] * n_samples for c in pinned]
        else:
            pinned = None
            class_probs_arg = [head_class_probs(hl, self.t_class) for hl in head_logits_list]
            class_ids = [self.rng.choice(len(p), size=n_samples, p=p).tolist()
                         for p in class_probs_arg]

        # 可选的外部位置先验：改若干条**类通道**上的 action_map 值（= 改位置采样分布）。
        # 2026-09-21 **放宽门控**（用户 2026-09-21）：joint 模式也在**根节点**生效——此时没有
        # "被钉的类"，就把**全部合法类**交给回调（价值网只对"位置"有发言权，类仍由类网采样）。
        # 动机：要测"1-ply 价值偏好当位置先验"就必须允许 joint（pos-only 的"外部钉类"会
        # 把 agent 变成按指定类出招的怪东西，见 docs/az_posnet_v2_plan.md §5.1）。
        # 非根节点由回调自己返回 None（`_make_value_pos_prior` 就是这样）⇒ 成本只在根节点付一次。
        # ⚠️ 注意：调用方若在 **joint** 模式下传了这个回调，行为就与之前不同了（之前 joint 下被静默忽略）。
        _prior_classes = (pinned if pinned is not None
                          else [c for c in range(len(base_cls_mask)) if base_cls_mask[c]])
        if self.pos_prior_fn is not None and len(_prior_classes):
            am2 = self.pos_prior_fn(node.state, node.player, net_out,
                                    [int(c) for c in _prior_classes], node is self.last_root)
            if am2 is not None:
                net_out = {**net_out, "action_map": am2}

        eff_t_pos = 0.0 if self.search_mode == "class-only" else self.t_pos
        agg: dict[tuple, dict] = {}
        for idx in range(n_samples):
            ops, intents = sample_bundle(
                net_out, node.state, node.player,
                t_class=self.t_class, t_pos=eff_t_pos, rng=self.rng,
                position_mask=base_pos_mask, class_mask=base_cls_mask,
                class_probs=class_probs_arg,
                class_ids=[cid[idx] for cid in class_ids],
            )
            key = tuple((int(o.op_type), o.arg0, o.arg1) for o in ops)
            if key not in agg:
                agg[key] = {"count": 0, "heads": [dict(), dict(), dict()]}
            agg[key]["count"] += 1
            for h, intent in enumerate(intents):
                if intent is not None:
                    ic = agg[key]["heads"][h]
                    ic[intent] = ic.get(intent, 0) + 1
        if not agg:
            node.expanded = True
            return node.net_value  # nothing sampled -> value-only leaf

        # keep the top-k decoded bundles by sample count; prior = empirical frequency
        selected = sorted(agg, key=lambda kk: -agg[kk]["count"])[:k]
        counts = np.asarray([agg[kk]["count"] for kk in selected], dtype=np.float32)
        priors = counts / counts.sum()

        next_player = 1 - node.player
        for idx, key in enumerate(selected):
            info = agg[key]
            ops = [Operation(OperationType(int(x[0])), int(x[1]), int(x[2])) for x in key]
            child_state = node.state.clone()
            if ops:
                child_state.apply_operation_list(node.player, ops)
            if node.player == 1:
                child_state.advance_round()
            node.children.append(
                BundleNode(state=child_state, player=next_player, prior=float(priors[idx]))
            )
            node.bundles.append(key)
            node.intent_counts.append(info["heads"])
        node.expanded = True
        return node.net_value

    def _backup(self, path: list[BundleNode], value: float) -> None:
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.visits += 1
            node.value_sum += value
            if i > 0:
                value = -value  # players alternate every level (2 levels per round)

    # ── public ──────────────────────────────────────────────────────────

    def search(
        self,
        state: BackendState,
        player: int,
        *,
        temperature: float = 0.0,
    ) -> BundleSearchResult:
        max_levels = self.max_depth_rounds * 2
        root = BundleNode(state=state.clone(), player=player)
        self.last_root = root
        self._expand(root, self.k)

        # Forced-move shortcut (see docs/az_forced_move_skip_plan.md), OPT-IN: with a
        # single candidate the visit distribution is [1.0] no matter how many iterations
        # we run, so the 256 iterations only produce node visits nothing reads.  Measured
        # cost split: 256 iterations = 256 net_fn calls = ~85% of a search's wall time
        # (2.54ms each), while one expansion (1 forward + 360 sample_bundle calls) is
        # ~3.7ms.  The class head being collapsed means the argmax class is frequently
        # unexecutable, so in pos-only mode ~97% of turns have exactly one candidate ->
        # there the iterations are pure waste (measured 25.8x overall, 748ms->5ms/turn).
        # In joint mode only ~3% of turns qualify, so the win is negligible while the
        # rng-stream side effect below still bites -> keep the default OFF.
        #
        # chosen_bundle / bundles / visit_policy / intent_counts are bit-identical to the
        # un-skipped path; only root_value changes (visit mean -> root network value) and
        # last_root has no expanded subtree.
        #
        # CAVEAT: that identity is per-search, given the SAME rng state.  Across a whole
        # game the stream diverges, because the skipped iterations used to consume
        # self.rng (each expansion samples candidates) -- so realized games are NOT
        # reproducible across this flag and az_selfplay's random_action_prob (which
        # draws from mcts.rng) shifts too.  The decision rule/distribution is unchanged;
        # only the sampled realization is.  Verify strength with an N-game comparison,
        # never by comparing individual games.
        if self.skip_single_candidate and len(root.bundles) <= 1:
            if root.bundles:
                policy = np.ones(1, dtype=np.float32)
            else:
                policy = np.zeros(0, dtype=np.float32)
            return BundleSearchResult(
                bundles=list(root.bundles),
                visit_policy=policy,
                chosen_index=0,
                chosen_bundle=root.bundles[0] if root.bundles else (),
                root_value=float(root.net_value),
                intent_counts=list(root.intent_counts),
            )

        for _ in range(self.iterations):
            node = root
            path = [root]
            while (
                node.expanded and node.children and len(path) <= max_levels
                and not node.state.terminal
            ):
                node = max(node.children, key=lambda c: self._puct(node, c))
                path.append(node)
            if node.state.terminal:
                value = self._terminal_value(node.state, node.player)
            elif len(path) > max_levels:
                value = float(self.net_fn(node.state, node.player)["value"])
            else:
                value = self._expand(node, self.k)
            self._backup(path, value)

        visits = np.asarray([c.visits for c in root.children], dtype=np.float32)
        if visits.sum() <= 0:
            visits = np.asarray([c.prior for c in root.children], dtype=np.float32)
        policy = self._policy_from_visits(visits, temperature)
        chosen = int(np.argmax(visits)) if temperature <= 1e-6 else self._sample(policy)
        if not root.bundles:
            chosen = 0
        root_value = root.mean_value if root.visits else (root.net_value if root.expanded else 0.0)
        return BundleSearchResult(
            bundles=list(root.bundles),
            visit_policy=policy,
            chosen_index=chosen,
            chosen_bundle=root.bundles[chosen] if root.bundles else (),
            root_value=float(root_value),
            intent_counts=list(root.intent_counts),
        )

    def _policy_from_visits(self, visits: np.ndarray, temperature: float) -> np.ndarray:
        if visits.size == 0:
            return visits.astype(np.float32)
        if temperature <= 1e-6:
            policy = np.zeros_like(visits, dtype=np.float32)
            policy[int(np.argmax(visits))] = 1.0
            return policy
        scaled = np.power(np.maximum(visits, 1e-6), 1.0 / max(temperature, 1e-6)).astype(np.float32)
        s = scaled.sum()
        if s <= 0:
            out = np.zeros_like(visits, dtype=np.float32)
            out[0] = 1.0
            return out
        return scaled / s

    def _sample(self, policy: np.ndarray) -> int:
        threshold = float(self.rng.random())
        cumulative = 0.0
        for i, p in enumerate(policy.tolist()):
            cumulative += p
            if threshold <= cumulative:
                return i
        return int(np.argmax(policy))
