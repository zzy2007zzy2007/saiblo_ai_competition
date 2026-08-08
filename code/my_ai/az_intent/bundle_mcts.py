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
    ) -> None:
        self.net_fn = net_fn
        self.iterations = iterations
        self.max_depth_rounds = max_depth_rounds
        self.k = k
        self.c_puct = c_puct
        self.t_class = t_class
        self.t_pos = t_pos
        self.sample_mult = sample_mult  # sample k*sample_mult times, keep top-k by count
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
        class_probs = [head_class_probs(hl, self.t_class) for hl in head_logits_list]
        class_ids = [self.rng.choice(len(p), size=n_samples, p=p).tolist()
                     for p in class_probs]
        agg: dict[tuple, dict] = {}
        for idx in range(n_samples):
            ops, intents = sample_bundle(
                net_out, node.state, node.player,
                t_class=self.t_class, t_pos=self.t_pos, rng=self.rng,
                position_mask=base_pos_mask, class_mask=base_cls_mask,
                class_probs=class_probs,
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
