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


class _McSkip(Exception):
    """MC 分差不足 ⇒ 不接管的内部信号（预注册 prereg_20261007_mc_margin.md）。"""

import math
from dataclasses import dataclass, field

import os as _os

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


# ── C1.1（路 C 第一步，见 docs/plan_pathC_joint_class_search.md）：**类采样前的可行性掩蔽** ──
# 默认**关**（`AZAI_CLASS_FEAS_MASK=0`）⇒ 与旧行为逐位相同；=1 才启用。
import os as _os
_CLASS_FEAS_MASK = [_os.environ.get("AZAI_CLASS_FEAS_MASK", "0") == "1"]
_FEAS_STATS = {"masked_classes": 0, "expansions": 0}


def feasible_classes(class_mask, position_mask, state, player, *, include_hold: bool = True):
    """C1.1：**类采样前的可行性集合**（与 `mc_quiet` 的候选构造同源：都要求"可执行且能落子"）。

    可行 ⟺ `class_mask[c]` 且 `position_mask[c].any()`（该类有合法格）
            且（`c` 不是超武 **或** 买得起）——因为 `intent_decoding=True` 下**买不起的超武仍在 class_mask 里**，
               解码器会走**通道 16 自动降级** ⇒ 名义上出超武、实际执行降级（`decoder.py:114` 与
               `_legal_nonlight_classes` 的注释都记录了这一点）⇒ 这正是 joint 模式 churn 风暴的来源。
            且（`c != 23` 或 `include_hold=True`，HOLD 下限）。
    返回 `(allowed: list[int], n_masked: int)`；`n_masked` = 被剔除的、**本在 class_mask 里**的类数（便于诊断）。
    """
    try:
        from my_ai.decoder import CHANNEL_TO_SUPER_WEAPON, SUPER_WEAPON_STATS
    except Exception:  # noqa: BLE001  取不到就只做"有合法格"这一层，不冒险
        CHANNEL_TO_SUPER_WEAPON, SUPER_WEAPON_STATS = {}, {}
    allowed, masked = [], 0
    for c in range(len(class_mask)):
        if not class_mask[c]:
            continue
        if (c == 23 and not include_hold) or (not position_mask[c].any()):
            masked += 1
            continue
        sw = CHANNEL_TO_SUPER_WEAPON.get(int(c))
        if sw is not None and SUPER_WEAPON_STATS and state.coins[player] < SUPER_WEAPON_STATS[sw].cost:
            masked += 1
            continue
        allowed.append(int(c))
    return allowed, masked


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
    # 2026-10-11（用户追问逼出）：**原始访问计数**（`root.children[i].visits`）。
    # `visit_policy` 在 `TEMP<=1e-6` 下会被 `_policy_from_visits` **argmax 成 one-hot**（⇒ 不含分布），
    # 而 `visits_raw` 就是真正的计数向量 ⇒ 只有它才能回答"搜索的偏好是什么" ✓。
    visits_raw: list = field(default_factory=list)


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
        # ── 候选确定化（docs/pos_topk_candidates_plan.md，2026-10-11）─────────────────
        #   "sample" = 现状（采样 n_samples 次，按解码结果出现次数取前 k）
        #   "topk"   = **不采样**：每头把它钉住的类的合法格按 `action_map[类]` 降序（并列按 (x,y) 字典序）
        #              排序，第 i 个候选束 = 各头第 i 名拼成的 op 列表 ⇒ 候选集合 = 状态的确定性函数 ✓
        cands_mode: str = "sample",
        cands_k: int = 0,      # topk 模式取前几个格（0 ⇒ 用 k）
        cands_combo: int = 1,  # topk 模式：1 ⇒ 各头拼束（≤3 op，保留"一回合多动作"表达）；0 ⇒ 只用 head0（单 op）
        # ⚠️ 2026-10-12（用户追问逼出）：**topk 候选的先验**
        #   "uniform" = 每个候选 count=1 ⇒ PUCT 先验均匀 ✗（实测这会让 visits 摊平：top1 0.098、熵 4.49/4.58，
        #               部署取 argmax 近乎随机 ⇒ 固定段 −10.2pp；见日志 86p/86q）
        #   "map"     = 先验 = **采样器本来会给出的概率** Π_头 softmax(action_map[类]/t_pos)[该格] ⇒
        #               等价于"拿掉采样的噪声、保留图的全部先验信息" ⇒ 这才是与 sample 的**公平对照** ✓
        cands_prior: str = "uniform",
        # ── 「确定性播种」（AZAI_SEED_STATE=1，默认关；见 docs/experiment_log.md 86r）─────────────
        #   真实（禁闪电）制度下搜索的访问分布**有实质信息**（先验全平、却把 50% 压在一个"三格建造组合"上 ✓），
        #   但采样的候选**每次重采** ⇒ 目标不可复现 ✗。本开关**不改采样分布**，只在**根节点**用"状态指纹"
        #   重置 `self.rng` ⇒ 同一状态跑两次搜索得到同一串随机数 ⇒ **候选集合/访问分布 = 状态的确定性函数** ✓
        seed_from_state: bool = False,
        # ── 「按 visit 名次取动作」（AZAI_VISIT_RANK，默认 1 = 今天的行为逐位不变 ✓）──────────────
        #   动因（用户 2026-10-12 指出的方法论问题）：`TEMP=1.0`（按分布采样）与 argmax 的比较**没有判别力** ✗ ——
        #   任何分布（哪怕完美）都有非众数质量 ⇒ 采样必然不优于 argmax ⇒ 该对比只反映"随机化代价"，❌ 不能判"分布对不对"。
        #   正确测法：**取 visit 排序里的第 r 名**（r=1 即 argmax）⇒ 若 visit 排序有信息，强度应**随 r 单调下降** ✓；
        #   若各名次强度持平 ⇒ visit 名次不含质量信息 ⇒ 拿它当蒸馏目标从根上无意义 ✗。
        #   实现：`argsort(-visits, kind="stable")[r-1]`（并列按候选下标稳定排序 ⇒ 确定性 ✓）。
        visit_rank: int = 1,
        seed: int = 0,
        search_mode: str = "joint",
        pos_pin: str = "argmax",
        class_pin_random_prob: float = 0.0,
        reserve_coins: int = 90,
        mc_horizon: int = 100,
        mc_bidir: int = 0,
        mc_every: int = 1,
        mc_no_downgrade: int = 0,
        mc_m4: int = 0,
        mc_root_only: int = 0,
        mc_top2: int = 0,
        mc_topk: int = 0,
        mc_cells: int = 0,
        mc_upg: int = 0,
        mc_margin: float = 0.0,
        mc_score: str = '',
        mc_selfspend: int = 0,
        mc_no_light: bool = False,
    mc_selfspend_mix: float = 1.0,   # 2026-10-18: score=(1-a)*quiet + a*selfspend/bidir
        mc_net_samples: int = 4,       # 2026-10-21: mc_score='net' 时每个候选的 rollout 次数
        skip_single_candidate: bool = False,
        pos_prior_fn=None,
        candidate_fn=None,
    ) -> None:
        if search_mode not in ("joint", "class-only", "pos-only"):
            raise ValueError(f"unknown search_mode: {search_mode}")
        if pos_pin not in ("argmax", "playable", "reserve", "reserve_up", "masked", "q", "mc_light", "mc_quiet"):
            raise ValueError(f"unknown pos_pin: {pos_pin}")
        if cands_mode not in ("sample", "topk"):
            raise ValueError(f"unknown cands_mode: {cands_mode}")
        if cands_prior not in ("uniform", "map"):
            raise ValueError(f"unknown cands_prior: {cands_prior}")
        self.net_fn = net_fn
        self.iterations = iterations
        self.max_depth_rounds = max_depth_rounds
        self.k = k
        self.c_puct = c_puct
        self.t_class = t_class
        self.t_pos = t_pos
        self.sample_mult = sample_mult  # sample k*sample_mult times, keep top-k by count
        self.cands_mode = cands_mode
        self.cands_k = int(cands_k)
        self.cands_combo = int(cands_combo)
        self.cands_prior = cands_prior
        if cands_mode == "topk" and search_mode != "pos-only":
            print(f"[cands] ⚠️ CANDS=topk 只在 search_mode=pos-only 生效；当前 {search_mode} ⇒ 回退 sample（既有行为不变）",
                  flush=True)
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
        # M6（mc_light）：MC 的 rollout 截断长度 / 每 N 回合最多比一次（控成本）
        self.mc_horizon = int(mc_horizon)
        # R1/R2（预注册 prereg_20261006_rollout_ladder.md）：>0 ⇒ 安静段之前先来 K 回合双向真实出招
        self.mc_bidir = int(mc_bidir)
        self.mc_every = int(mc_every)
        # M6b-ND（预注册 §6.1）：候选里的"最优经济类"是否**排除类 16（降级）**
        self.mc_no_downgrade = int(mc_no_downgrade)
        # M6c-M4（预注册 docs/prereg_20261006_mc_menu.md）：菜单显式给出 {HOLD, 闪电(可执行时), 类16, 最优经济类}
        self.mc_m4 = int(mc_m4)
        # M6c-ROOT（预注册 docs/prereg_20261006_mc_menu.md §2 追加）：
        # 只在**根节点**做 MC 仲裁（部署时真正要做的那个决策），搜索树内部交回原采样。
        # 动机：`_expand` 在 256 次迭代里对**每个叶节点**都会调用 ⇒ 现状每局 ~780 次 MC（成本高、
        # 且连对手节点也被 MC 仲裁）；root-only 为 1 次/搜索。
        self.mc_root_only = int(mc_root_only)
        # M6c-2E（预注册 mc_menu.md §2）：候选里放**两个不同的**经济类（按 action_map 排序取 top-2）
        # —— 教训：上一版「显式 16 + 最优经济类」被 `!= 16` 守卫抵消成空操作（逐局 128/128 与 M6b 相同）。
        self.mc_top2 = int(mc_top2)
        self.mc_topk = int(mc_topk)
        # 选类×选格（>0 ⇒ 菜单放「类+格」候选，含我们已有的塔格 ⇒ 可表达升级）
        self.mc_cells = int(mc_cells)
        # 显式升级候选：>0 ⇒ 对我们前 n 座塔各放一个 Operation(12,x,y) 候选（见 mc_quiet.py 注释）
        self.mc_upg = int(mc_upg)
        # 分差门控（预注册 prereg_20261007_mc_margin.md）：分差 < mc_margin ⇒ MC 不接管
        self.mc_margin = float(mc_margin)
        self.mc_skipped = 0
        # 评分方式（预注册 prereg_20261008_mc_value_delta.md）：'' = 安静 rollout（现状）
        self.mc_score = str(mc_score or '')
        # 单侧支出（预注册 prereg_20261008_mc_selfspend.md）：>0 ⇒ rollout 里我方继续出招 K 回合
        self.mc_selfspend = int(mc_selfspend)
        # Phase 0：True ⇒ 闪电不进 MC 菜单（见 prereg_20261018_mc_nolight_real.md）
        self.mc_no_light = bool(mc_no_light)
        self.mc_selfspend_mix = float(mc_selfspend_mix)
        self.mc_net_samples = int(mc_net_samples)
        # 「显式 op」候选（如升级）：参数不是 (x,y) ⇒ class+position 通路表达不了，
        # 必须由 decide() 直接把它当本回合动作发出去（见 unify_upgrade 补丁注释）。
        self.mc_last_explicit = None
        self._mc_root = None
        # 埋点（独立验证者 2026-10-06 限制④）：MC 真正生效 / 静默回落的次数
        self.mc_taken = 0
        self.mc_fallback = 0
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
        self.seed_from_state = bool(seed_from_state)
        self.visit_rank = max(1, int(visit_rank))
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

    def _topk_candidates(self, net_out, node, pinned, k: int, head_logits_list,
                         base_cls_mask, base_pos_mask) -> dict:
        """**候选确定化**（`cands_mode="topk"`，见 `docs/pos_topk_candidates_plan.md`）。

        动因（本会话实测，见实验日志 86n）：现状是"采样 n_samples 次、按**出现次数**取前 k"，
        频率估计带噪 ⇒ 同一局面换 seed 两次搜索的候选集合只有 1/12 完全相同（Jaccard 0.45）✗。

        本函数**完全不调用 `self.rng`**：对每个头，把它**钉住的类**的合法格按 `action_map[类]` 降序
        （**并列按 (x,y) 字典序** ⇒ 严格确定性）排序；第 i 个候选束 = 各头第 i 名拼起来的 op 列表
        （`cands_combo=1` ⇒ ≤3 op，保留"一回合多动作"的表达 ✓；`=0` ⇒ 只用 head0 ⇒ 单 op）。
        ⇒ 候选集合 = **状态的确定性函数** ✓（两次搜索必然同清单）。

        返回结构与采样路径一致：`{key: {"count": 1, "heads": [3 个 dict]}}`（count 恒为 1 ⇒ prior 均匀；
        `heads` 用于 AZ 边缘化/dump，键与采样路径同形 `(类, x, y)`）。
        """
        from my_ai.az_intent.mc_quiet import _decode_class_op_at_cell

        _K = int(self.cands_k) if int(self.cands_k) > 0 else int(k)
        am = np.asarray(net_out["action_map"], dtype=np.float32)
        ranked: list = []
        for h, c in enumerate(pinned):
            c = int(c)
            if not (0 <= c < am.shape[0]) or not base_pos_mask[c].any():
                ranked.append([])
                continue
            xs, ys = np.nonzero(base_pos_mask[c])
            vals = am[c][xs, ys]
            order = sorted(range(len(xs)), key=lambda i: (-float(vals[i]), int(xs[i]), int(ys[i])))
            ranked.append([(int(xs[i]), int(ys[i])) for i in order])
        # "map" 先验：采样器本来会给出的概率（`t_pos` 温度的 softmax over 该类合法格）
        probs: list = []
        for _h, _c in enumerate(pinned):
            _c = int(_c)
            if not (0 <= _c < am.shape[0]) or not base_pos_mask[_c].any():
                probs.append({})
                continue
            _xs, _ys = np.nonzero(base_pos_mask[_c])
            _z = am[_c][_xs, _ys] / max(float(self.t_pos), 1e-6)
            _z = _z - float(np.max(_z))
            _e = np.exp(_z)
            _s = max(float(_e.sum()), 1e-12)
            probs.append({(int(_xs[k]), int(_ys[k])): float(_e[k]) / _s for k in range(len(_xs))})
        n_heads = 3 if int(self.cands_combo) else 1
        agg: dict = {}
        n_skip_invalid = 0
        for i in range(_K):
            ops, heads = [], [dict(), dict(), dict()]
            _cnt = 1.0
            for h in range(min(n_heads, len(ranked))):
                if i >= len(ranked[h]):
                    continue
                x, y = ranked[h][i]
                t = _decode_class_op_at_cell(head_logits_list[h], am, base_cls_mask, base_pos_mask,
                                             node.state, node.player, int(pinned[h]), x, y)
                if t is None:
                    continue
                op3 = (int(t[0]), int(t[1]), int(t[2]))
                if op3 in ops:            # 两头可能解出同一个 op ⇒ 去重（重复施加会崩）
                    continue
                ops.append(op3)
                heads[h][(int(pinned[h]), int(x), int(y))] = 1
                if self.cands_prior == "map":
                    _cnt *= max(probs[h].get((x, y), 0.0), 1e-12)
            if not ops:
                continue
            # 联合可执行性校验（只影响 topk 路径；采样路径一字不动）
            try:
                _cs = node.state.clone()
                _cs.apply_operation_list(node.player,
                                         [Operation(OperationType(int(a)), int(b), int(c)) for a, b, c in ops])
            except Exception:  # noqa: BLE001
                n_skip_invalid += 1
                continue
            key = tuple(ops)
            if key in agg:
                continue
            agg[key] = {"count": (_cnt if self.cands_prior == "map" else 1.0), "heads": heads}
        if n_skip_invalid:
            print(f"[cands] topk：{n_skip_invalid} 个候选束联合不可执行 ⇒ 已丢弃", flush=True)
        return agg

    def _expand(self, node: BundleNode, k: int) -> float:
        if node.state.terminal:
            node.expanded = True
            return self._terminal_value(node.state, node.player)
        net_out = self.net_fn(node.state, node.player)
        # ── 确定性播种（默认关）：只对**根节点**用状态指纹重置 rng ──────────────────────
        #   之后整棵树的所有随机抽样都来自这串确定性随机数 ⇒ 同一状态 ⇒ 同一候选集合/访问分布 ✓
        #   （只依赖 `net_out`（action_map + head_logits）与玩家号 ⇒ 零额外前向成本 ✓）
        if self.seed_from_state and node is self.last_root:
            import hashlib
            _fp = (np.asarray(net_out["action_map"], dtype=np.float32).tobytes()
                   + np.asarray(net_out["head_logits"], dtype=np.float32).tobytes()
                   + bytes([int(node.player) & 0xFF]))
            _h = hashlib.blake2b(_fp, digest_size=8).digest()
            self.rng = np.random.default_rng(int.from_bytes(_h, "little") % (2 ** 63))
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
            elif self.pos_pin == "mc_quiet":
                # 方法侧 M6b（见 code/my_ai/az_intent/mc_quiet.py）：**安静 rollout** 的蒙特卡洛类决策。
                # M6（贪心 rollout）在确定性局面下原生崩溃（128 局里 32% INVALID）⇒ 换成"只打候选那一手、
                # 之后谁都不出招地推进到终局"，成本 ~0.3–0.5 s/候选 ⇒ **每个回合都能比**。
                # 候选 = {闪电, HOLD} ∪ {最优经济类}（经济类只在"不用打闪电"时才有意义，但一并比较）。
                _AM = np.asarray(net_out["action_map"], dtype=np.float32)
                _economy = None
                _best_v = -np.inf
                _hi = 16 if self.mc_no_downgrade else 17      # M6b-ND：排除类 16（降级）
                _econ_rank = []
                for _c in range(0, _hi):
                    if base_cls_mask[_c] and base_pos_mask[_c].any():
                        _v = float(np.where(base_pos_mask[_c], _AM[_c], -np.inf).max())
                        if np.isfinite(_v):
                            _econ_rank.append((_v, int(_c)))
                _econ_rank.sort(key=lambda x: -x[0])
                if _econ_rank:
                    _best_v, _economy = _econ_rank[0]
                from my_ai.az_intent.mc_quiet import _decode_class_op as _dq, mc_quiet_choose
                _hl0 = np.asarray(head_logits_list[0], dtype=np.float32)
                _rnd = int(getattr(node.state, "round_index", 0))
                # ⚠️ 成本：只算**一次** MC（三个 head 钉同一个类——与 argmax 下的既有一致性相同），
                # 并用 `mc_every` 门控（实测：每回合 × 3 heads × 512 回合 rollout ⇒ ~20 min/局）。
                _is_root = (self._mc_root is not None and node is self._mc_root)
                if (self.mc_every > 0 and (_rnd % int(self.mc_every) == 0)
                        and (not self.mc_root_only or _is_root)):
                    _cands = [23]
                    # 🧪 Phase 0 发现（预注册 `docs/prereg_20261018_mc_nolight_real.md`）：
                    #   **闪电会在 MC 菜单里"挤掉"经济/升级候选**（真实对局里闪电候选几乎总赢 ⇒ 升级永远 0 次）。
                    #   `mc_no_light=1` ⇒ **闪电不进 MC 菜单**（仍由非仲裁回合的默认"买得起就放"负责），
                    #   让经济/升级候选在**没有闪电**的菜单里公平比较。
                    if (not self.mc_no_light) and _dq(_hl0, _AM, base_cls_mask, base_pos_mask,
                                                      node.state, node.player, 17) is not None:
                        _cands.insert(0, 17)
                    if self.mc_cells > 0:
                        # **选类×选格**：{闪电?, HOLD} ∪ {(经济类, 最热 n 个空格)} ∪ {(经济类, 我们已有塔的格子)}
                        # 后者会让解码器按格子状态给出升级/降级 ⇒ 首次让机制能「表达升级」。
                        _c0 = _economy if _economy is not None else 0
                        _pm0 = base_pos_mask[_c0]
                        _ys, _xs = np.where(_pm0)
                        if len(_ys):
                            _vals = _AM[_c0][_ys, _xs]
                            _order = np.argsort(-_vals)
                            for _i in _order[: max(1, int(self.mc_cells))]:
                                _cands.append((_c0, int(_xs[_i]), int(_ys[_i])))
                        try:
                            for _tw in node.state.towers_of(int(node.player)):
                                _cands.append((_c0, int(_tw.x), int(_tw.y)))
                        except Exception as _exc:
                            print(f'[mc_cells] towers_of 失败: {type(_exc).__name__}: {_exc}', flush=True)
                    elif self.mc_topk > 0:
                        # 菜单加宽（2026-10-07）：加入**前 k 个互不相同**的经济类（各自解码成功才加）
                        # ⚠️ 动机：现有菜单的经济候选实际上永远是"类 0 或类 16"（跨类 map 最大值不可比），
                        #    导致我们只会"铺一级塔 + 反复降级"、**从不升级**（对手 9.4 次升级/局）。
                        _added = 0
                        for _v, _c in _econ_rank:
                            if _added >= int(self.mc_topk):
                                break
                            if _dq(_hl0, _AM, base_cls_mask, base_pos_mask,
                                   node.state, node.player, _c) is not None:
                                if _c not in _cands:
                                    _cands.append(_c)
                                    _added += 1
                    elif self.mc_top2:
                        # M6c-2E：加入**前两个不同的**经济类（各自解码成功才加）
                        _added = 0
                        for _v, _c in _econ_rank:
                            if _added >= 2:
                                break
                            if _dq(_hl0, _AM, base_cls_mask, base_pos_mask,
                                   node.state, node.player, _c) is not None:
                                if _c not in _cands:
                                    _cands.append(_c)
                                    _added += 1
                    else:
                        if self.mc_m4 and _dq(_hl0, _AM, base_cls_mask, base_pos_mask,
                                              node.state, node.player, 16) is not None:
                            _cands.append(16)      # M6c-M4：显式给「类 16」一个候选位
                        if _economy is not None and _economy != 16 and _dq(
                                _hl0, _AM, base_cls_mask, base_pos_mask, node.state, node.player,
                                _economy) is not None:
                            _cands.append(_economy)
                    # **显式升级候选（可加性）**：直接构造 UPGRADE_TOWER(12) op ⇒ 机制第一次能"比较升级 vs 再建塔"。
                    # ⚠️ 为什么不用 decoder：实测（196 局）我们**从不升级**，而"格子已有塔"时 decoder 给的是**降级**。
                    # ⚠️ 必须**可加**（不能挤掉经济候选）：第一版写成 elif 链 ⇒ 建塔直接归零（冒烟实测 0 建塔）。
                    if self.mc_upg > 0:
                        # ✅ 正确参数（2026-10-07 从 SDK 官方样例查明）：UPGRADE_TOWER(tower_id, target_type)
                        _n_up = 0
                        try:
                            from SDK.utils.constants import TOWER_UPGRADE_TREE as _TREE
                            _tw_list = sorted(node.state.towers_of(int(node.player)),
                                              key=lambda w: (-int(getattr(w, 'tower_type', 0)), int(w.x), int(w.y)))
                            for _tw in _tw_list:
                                if _n_up >= int(self.mc_upg):
                                    break
                                _tgts = _TREE.get(_tw.tower_type, ())
                                if not _tgts:
                                    continue
                                _tgt = int(_tgts[0])
                                # 只加**买得起**的（买不起的 op 引擎会拒 ⇒ 等于空操作，白占候选位）
                                if int(node.state.coins[node.player]) < int(node.state.upgrade_tower_cost(_tgts[0])):
                                    continue
                                _cands.append(('op', 12, int(_tw.tower_id), _tgt))
                                _n_up += 1
                        except Exception as _exc:
                            print(f'[mc_upg] 构造升级候选失败: {type(_exc).__name__}: {_exc}', flush=True)
                    try:
                        # 评分用（delta / value_only 需要 V(before)）：只算一次，成本 ≈ 一次前向
                        _v_before = 0.0
                        if self.mc_score in ('delta', 'value_only'):
                            try:
                                _v_before = float(self.net_fn(node.state, node.player)['value'])
                            except Exception as _exc:
                                print(f'[mc_score] V(before) 计算失败: {type(_exc).__name__}: {_exc}', flush=True)
                        # C-效率（2026-10-09）：**截断补尾**开关（默认关 ⇒ 旧行为逐位不变）
                        _tail_fn = (None if _os.environ.get('AZAI_MC_TAIL', '') != 'value'
                                    else (lambda _s, _p: float(self.net_fn(_s, _p)['value'])))
                        _tail_mix = float(_os.environ.get('AZAI_MC_TAIL_MIX', '1.0'))
                        _c, _sc = mc_quiet_choose(node.state, node.player, _cands, _hl0, _AM,
                                                  base_cls_mask, base_pos_mask,
                                                  max_rounds=int(self.mc_horizon),
                                                  bidir_k=int(self.mc_bidir),
                                                  net_fn=self.net_fn,
                                                  margin_gate=float(self.mc_margin),
                                                  score_mode=str(self.mc_score),
                                                  tail_fn=_tail_fn, tail_mix=_tail_mix,

                                                  v_before=float(_v_before),
                                                  selfspend_k=int(self.mc_selfspend),
                                                  selfspend_mix=float(self.mc_selfspend_mix),
                     net_samples=int(self.mc_net_samples))
                        self.mc_last_scores = _sc
                        if _c is None:
                            # 分差不足 ⇒ **不接管**：由类头 argmax 决定（= 该配置下的默认 pin），只埋点
                            raise _McSkip()
                        self.mc_taken += 1
                        # 候选可能是 (类, x, y) ⇒ 钉住的类取它 [0]
                        _ccls = (int(_c[0]) if isinstance(_c, (tuple, list)) else int(_c))
                        if isinstance(_c, (tuple, list)) and _c[0] == 'op':
                            # 显式 op 候选（升级）：**class 通路表达不了它**（实测类 0-15 只会给建塔）
                            # ⇒ 记下来，由 az_bridge_ai.decide() 直接把这个 op 当作本回合动作发出去。
                            self.mc_last_explicit = (int(_c[1]), int(_c[2]), int(_c[3]))
                            _ccls = 12
                        else:
                            self.mc_last_explicit = None
                        pinned = [_ccls] * len(head_logits_list)
                        # 观测（可选，只读）：把这次 MC 的候选/分数/选择写成 JSONL
                        _dump = _os.environ.get("AZAI_MC_DUMP", "")
                        if _dump:
                            try:
                                import json as _json
                                with open(_dump, "a", encoding="utf-8") as _fh:
                                    _fh.write(_json.dumps({
                                        "round": _rnd, "player": int(node.player),
                                        "coins": [int(x) for x in node.state.coins],
                                        "hp": [int(b.hp) for b in node.state.bases],
                                        # ⚠️ 候选可能是 int（类）或元组（('op',12,id,tgt) / (类,x,y)）：
                                        #    原来写 `int(x[0])` ⇒ 遇 'op' 抛异常 ⇒ **整行被 except 吞掉**（静默丢样本，
                                        #    2026-10-07 我据此误判"升级候选从没进菜单"）。
                                        "cands": [list(x) if isinstance(x, (tuple, list)) else int(x)
                                                  for x in _cands],
                                        "scores": {str(k): round(float(v), 5) for k, v in _sc.items()},
                                        "chosen": int(_ccls),
                                    }, ensure_ascii=False) + "\n")
                            except Exception:  # noqa: BLE001
                                pass
                    except _McSkip:
                        # ⚠️ 必须**给 pinned 赋值**：mc_quiet 分支里 pinned 只在"接管/异常回退"两条路上被赋值，
                        #    少了这一条 ⇒ `UnboundLocalError: pinned`（2026-10-07 冒烟实测，该局 aborted）。
                        #    "不接管"的语义 = 让**类头 argmax** 决定（与该配置下的默认一致）。
                        self.mc_skipped += 1
                        pinned = [int(np.argmax(hl)) for hl in head_logits_list]
                    except Exception:  # noqa: BLE001
                        self.mc_fallback += 1
                        pinned = [int(np.argmax(hl)) for hl in head_logits_list]
                else:
                    pinned = [int(np.argmax(hl)) for hl in head_logits_list]
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
                    # ⚠️ 门控必须用「**真的放得出来**」：`intent_decoding` 只查冷却**不查金币**
                    # （实测：只看 class_mask 会让 MC 几乎每回合都跑 ⇒ ~75 s/回合）。
                    from my_ai.az_intent.mc_class import _decode_class_op, mc_choose_class
                    _can_light = _decode_class_op(_hl, _AM, base_cls_mask, base_pos_mask,
                                                  node.state, node.player, 17) is not None
                    _rnd = int(getattr(node.state, "round_index", 0))
                    if (_top == 17 and _can_light and self.mc_every > 0
                            and (_rnd % self.mc_every == 0)):
                        _cands = [17, 23]
                        try:
                            _c, _sc = mc_choose_class(
                                self.net_fn, node.state, node.player, _cands, _hl, _AM,
                                base_cls_mask, base_pos_mask,
                                max_rounds=int(self.mc_horizon))
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
            # ── C1.1（默认关）：把类分布里**不可行**的类权重清零并重新归一 ──
            if _CLASS_FEAS_MASK[0]:
                _allowed, _nmask = feasible_classes(base_cls_mask, base_pos_mask, node.state, node.player)
                if _allowed and _nmask > 0:
                    _keep = np.zeros(len(base_cls_mask), dtype=bool)
                    _keep[_allowed] = True
                    _newp = []
                    for _pp in class_probs_arg:
                        _q = np.asarray(_pp, dtype=np.float64).copy()
                        _q[~_keep] = 0.0
                        _s = float(_q.sum())
                        _newp.append((_q / _s) if _s > 0 else _pp)   # 全被掩蔽 ⇒ 安全阀：回退原分布
                    class_probs_arg = _newp
                    _FEAS_STATS["masked_classes"] += int(_nmask)
                    _FEAS_STATS["expansions"] += 1
                    if _FEAS_STATS["expansions"] % 200 == 0:
                        print(f"[feas] 已掩蔽 {_FEAS_STATS['masked_classes']} 个不可行类"
                              f" / {_FEAS_STATS['expansions']} 次扩展（超武买不起 / 无合法格 / HOLD 下限）", flush=True)
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
        # ── 候选确定化（AZAI_CANDS=topk，默认关）：不采样，直接取"每头按图分数前 K 名"拼束 ──
        #    见 docs/pos_topk_candidates_plan.md ⇒ 候选集合 = 状态的确定性函数 ✓（且省掉 n_samples 次解码）
        _use_topk = (self.cands_mode == "topk" and self.search_mode == "pos-only"
                     and pinned is not None)
        if _use_topk:
            agg = self._topk_candidates(net_out, node, pinned, k, head_logits_list,
                                        base_cls_mask, base_pos_mask)
        for idx in range(0 if _use_topk else n_samples):
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
        self._mc_root = root
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
                visits_raw=[float(c.visits) for c in root.children],
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
        # 名次实验（AZAI_VISIT_RANK>1）：改取 visit 排序里的第 r 名（r=1 即上面的 argmax ⇒ 旧行为不变 ✓）。
        # 判读：强度随 r 单调下降 ⇒ visit 排序有信息 ✓；持平 ⇒ 名次无信息 ✗（见 __init__ 的注释）。
        if int(self.visit_rank) > 1 and visits.size > 0:
            _ord = np.argsort(-visits, kind="stable")
            _ri = min(int(self.visit_rank), len(_ord)) - 1
            chosen = int(_ord[_ri])
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
            visits_raw=[float(v) for v in visits.tolist()],   # 原始访问计数（真分布，不是 argmax 后的 policy）
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
