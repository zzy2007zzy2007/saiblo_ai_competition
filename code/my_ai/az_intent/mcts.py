"""Intent-space MCTS core for Ant-Game AlphaZero (方案 B, P1).

Branches over ATOMIC INTENT actions:
  - classes 0-20: (class, x, y) intents (position-bearing)
  - classes 21/22 (base upgrades) and 23 (HOLD): class-only intents

Prior (factored, per design doc §3):
  P(c)   = softmax(head_logits[c] / T_class) over legal classes
  P(p|c) = softmax(action_map[c, p] / T_pos) over class c's legal positions
  prior(c, p) = P(c) * P(p|c)

Tree structure: 6 head-levels per round
  P0-h1 -> P0-h2 -> P0-h3 -> P1-h1 -> P1-h2 -> P1-h3 -> next round
Each edge clones the parent state and applies the decoded operation, so later
heads see earlier ops (P1 genuinely sees P0's ops). At the P1-h3 -> next-round
edge the state is cloned and advance_round() is called.  HOLD edges reuse the
parent state object (read-only, never mutated in place).

The network is injected as net_fn(state, player) -> dict:
  action_map:  (24, 19, 19) float ndarray
  head_logits: list of 3 (24,) float ndarrays (head0=head1_logits, ...)
  value:       float, from the node's player perspective
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np

from SDK.backend.state import BackendState
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

from my_ai.decoder import (
    CHANNEL_TO_SUPER_WEAPON,
    SUPER_WEAPON_TO_OP_TYPE,
    _decode_tower_action,
    make_class_mask,
    make_position_masks,
)

NUM_CLASSES = 24
HP_SCALE = 20.0  # value target = clip((hp_us - hp_opp) / HP_SCALE, -1, 1)


# ─── Atomic intent & decoding ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Intent:
    class_id: int
    x: int = -1
    y: int = -1


def intent_to_operation(state: BackendState, player: int, intent: Intent) -> Operation | None:
    """Deterministically decode an atomic intent into an Operation (or None for HOLD)."""
    c = intent.class_id
    if 0 <= c <= 15:
        return _decode_tower_action(state, player, c, intent.x, intent.y)
    if c == 16:
        tower = state.tower_at(intent.x, intent.y)
        if tower is not None and tower.player == player:
            return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
        return None
    if 17 <= c <= 20:
        op_type = SUPER_WEAPON_TO_OP_TYPE[CHANNEL_TO_SUPER_WEAPON[c]]
        return Operation(op_type, intent.x, intent.y)
    if c == 21:
        return Operation(OperationType.UPGRADE_GENERATION_SPEED)
    if c == 22:
        return Operation(OperationType.UPGRADE_GENERATED_ANT)
    return None  # c == 23 (HOLD) or unknown


# ─── Prior computation ──────────────────────────────────────────────────────


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


def compute_intent_prior(
    net_out: dict,
    state: BackendState,
    player: int,
    head_idx: int,
    t_class: float,
    t_pos: float,
) -> tuple[list[Intent], np.ndarray]:
    """Compute the factored prior over atomic intents for one head.

    Returns (intents, priors) aligned by index.  ``priors`` sums to 1.
    """
    position_mask = make_position_masks(state, player, intent_decoding=False)
    class_mask = make_class_mask(state, player, position_mask=position_mask, intent_decoding=False)
    head_logits = np.asarray(net_out["head_logits"][head_idx], dtype=np.float32)
    action_map = np.asarray(net_out["action_map"], dtype=np.float32)

    legal_classes = np.where(class_mask)[0]
    class_probs = _softmax(head_logits[legal_classes] / t_class)

    intents: list[Intent] = []
    priors: list[float] = []
    for idx, c in enumerate(legal_classes):
        p_c = class_probs[idx]
        if 0 <= c <= 20:
            # position_mask is indexed [c, x, y] (x is the first spatial dim).
            xs, ys = np.where(position_mask[c])
            if xs.size == 0:
                continue
            pos_logits = action_map[c, xs, ys] / t_pos
            pos_probs = _softmax(pos_logits)
            for (x, y), pp in zip(zip(xs.tolist(), ys.tolist()), pos_probs):
                intents.append(Intent(int(c), int(x), int(y)))
                priors.append(float(p_c * pp))
        else:  # class-only intents: 21/22/23
            intents.append(Intent(int(c)))
            priors.append(float(p_c))

    priors_arr = np.asarray(priors, dtype=np.float32)
    total = priors_arr.sum()
    if total <= 0 or not np.isfinite(total):
        priors_arr = np.zeros_like(priors_arr)
        if priors_arr.size:
            priors_arr[0] = 1.0
    else:
        priors_arr = priors_arr / total
    return intents, priors_arr


# ─── Tree nodes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class IntentNode:
    state: BackendState
    player: int
    head_idx: int
    depth: int = 0
    pending: tuple = ()  # ops selected this round by this player (for can_apply_operation)
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: list = field(default_factory=list)
    intents: list = field(default_factory=list)  # explored (kept) intents, aligned with children
    kept_indices: list = field(default_factory=list)  # index into full_intents for each child
    child_priors: np.ndarray | None = None
    full_intents: list = field(default_factory=list)  # complete legal intent space
    net_value: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


# ─── Search ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SearchResult:
    intents: list[Intent]        # explored (kept) intents at root
    visit_policy: np.ndarray     # over root.intents
    chosen_index: int
    chosen_intent: Intent
    root_value: float
    full_intents: list[Intent]   # complete legal intent space at root
    full_target: np.ndarray      # temperature-scaled visit distribution over full_intents


class IntentMCTS:
    def __init__(
        self,
        net_fn: Callable[[BackendState, int], dict],
        *,
        iterations: int = 32,
        max_depth_rounds: int = 2,
        c_puct: float = 1.25,
        t_class: float = 1.0,
        t_pos: float = 1.0,
        root_topk: int = 24,
        child_topk: int = 12,
        dirichlet_alpha: float = 0.35,
        dirichlet_epsilon: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.net_fn = net_fn
        self.iterations = iterations
        self.max_depth_rounds = max_depth_rounds
        self.c_puct = c_puct
        self.t_class = t_class
        self.t_pos = t_pos
        self.root_topk = root_topk
        self.child_topk = child_topk
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)
        self.last_root: IntentNode | None = None  # debug hook: root of the last search

    # ── internals ────────────────────────────────────────────────────────

    def _puct(self, parent: IntentNode, child: IntentNode) -> float:
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

    def _expand(self, node: IntentNode, topk: int) -> float:
        """Expand a leaf: compute priors, create top-k children. Returns the node value."""
        if node.state.terminal:
            node.expanded = True
            return self._terminal_value(node.state, node.player)

        net_out = self.net_fn(node.state, node.player)
        node.net_value = float(net_out["value"])
        intents, priors = compute_intent_prior(
            net_out, node.state, node.player, node.head_idx, self.t_class, self.t_pos
        )
        node.full_intents = intents
        if not intents:
            node.expanded = True
            return node.net_value  # nothing legal to do -> value-only leaf

        # Top-k by prior; HOLD (class 23) always kept.
        order = np.argsort(priors)[::-1]
        selected = set(int(i) for i in order[:topk])
        hold_idx = next((i for i, it in enumerate(intents) if it.class_id == 23), None)
        if hold_idx is not None:
            selected.add(hold_idx)

        kept_intents: list[Intent] = []
        kept_indices: list[int] = []
        kept_priors: list[float] = []
        kept_ops: list[Operation | None] = []
        for i in sorted(selected):
            intent = intents[i]
            if intent.class_id == 23:
                kept_intents.append(intent)
                kept_indices.append(i)
                kept_priors.append(float(priors[i]))
                kept_ops.append(None)  # HOLD
                continue
            op = intent_to_operation(node.state, node.player, intent)
            if op is None:
                continue
            if not node.state.can_apply_operation(node.player, op, node.pending):
                continue
            kept_intents.append(intent)
            kept_indices.append(i)
            kept_priors.append(float(priors[i]))
            kept_ops.append(op)

        node.intents = kept_intents
        node.kept_indices = kept_indices
        node.child_priors = np.asarray(kept_priors, dtype=np.float32)
        total = node.child_priors.sum()
        if total > 0:
            node.child_priors = node.child_priors / total
        else:
            node.child_priors = np.zeros_like(node.child_priors)
            if node.child_priors.size:
                node.child_priors[0] = 1.0

        # Phase transition for children.
        if node.player == 1 and node.head_idx == 2:
            next_player, next_head, round_end = 0, 0, True
        elif node.player == 0 and node.head_idx == 2:
            next_player, next_head, round_end = 1, 0, False
        else:
            next_player, next_head, round_end = node.player, node.head_idx + 1, False

        for intent, op in zip(node.intents, kept_ops):
            if op is None:  # HOLD: reuse read-only state, or advance round at round end
                if round_end:
                    child_state = node.state.clone()
                    child_state.advance_round()
                else:
                    child_state = node.state
                child_pending = () if next_player != node.player else node.pending
            else:
                child_state = node.state.clone()
                child_state.apply_operation_list(node.player, [op])
                if round_end:
                    child_state.advance_round()
                child_pending = () if next_player != node.player else node.pending + (op,)
            node.children.append(
                IntentNode(
                    state=child_state,
                    player=next_player,
                    head_idx=next_head,
                    depth=node.depth + 1,
                    pending=child_pending,
                    prior=float(node.child_priors[len(node.children)]),
                )
            )
        node.expanded = True
        return node.net_value

    def _backup(self, path: list[IntentNode], value: float) -> None:
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.visits += 1
            node.value_sum += value
            if i > 0 and path[i - 1].player != node.player:
                value = -value

    def _add_dirichlet_noise(self, node: IntentNode) -> None:
        n = len(node.child_priors)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * n)
        node.child_priors = (
            (1.0 - self.dirichlet_epsilon) * node.child_priors + self.dirichlet_epsilon * noise
        )
        node.child_priors /= node.child_priors.sum()

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
            policy = np.zeros_like(visits, dtype=np.float32)
            policy[0] = 1.0
            return policy
        return scaled / s

    # ── public ──────────────────────────────────────────────────────────

    def search(
        self,
        state: BackendState,
        player: int,
        *,
        head_idx: int = 0,
        temperature: float = 0.0,
        add_root_noise: bool = True,
        select_by_prior: bool = False,
    ) -> SearchResult:
        max_head = self.max_depth_rounds * 6
        root = IntentNode(state=state.clone(), player=player, head_idx=head_idx, depth=0, prior=1.0)
        self.last_root = root
        self._expand(root, self.root_topk)
        if add_root_noise and self.dirichlet_epsilon > 0 and len(root.children) > 1:
            self._add_dirichlet_noise(root)

        for _ in range(self.iterations):
            node = root
            path = [root]
            while (
                node.expanded and node.children and node.depth < max_head and not node.state.terminal
            ):
                node = max(node.children, key=lambda c: self._puct(node, c))
                path.append(node)
            if node.state.terminal:
                value = self._terminal_value(node.state, node.player)
            elif node.depth >= max_head:
                value = float(self.net_fn(node.state, node.player)["value"])
            else:
                value = self._expand(node, self.child_topk)
            self._backup(path, value)

        visits = np.asarray([c.visits for c in root.children], dtype=np.float32)
        if visits.sum() <= 0:
            visits = root.child_priors.copy() if root.child_priors is not None else visits
        policy = self._policy_from_visits(visits, temperature)
        if select_by_prior:
            chosen = int(np.argmax(root.child_priors)) if root.child_priors is not None else 0
        elif temperature <= 1e-6:
            chosen = int(np.argmax(visits))
        else:
            chosen = self._sample_from_policy(policy)
        if len(root.intents) == 0:
            chosen = 0
        root_value = root.mean_value if root.visits else (root.net_value if root.expanded else 0.0)

        # Full-space training target: temperature-scaled visits over all legal intents.
        full_target = np.zeros(len(root.full_intents), dtype=np.float32)
        for j, fi in enumerate(root.kept_indices):
            full_target[fi] = policy[j]
        if full_target.sum() > 0:
            full_target = full_target / full_target.sum()

        return SearchResult(
            intents=list(root.intents),
            visit_policy=policy,
            chosen_index=chosen,
            chosen_intent=root.intents[chosen] if root.intents else Intent(23),
            root_value=float(root_value),
            full_intents=list(root.full_intents),
            full_target=full_target,
        )

    def _sample_from_policy(self, policy: np.ndarray) -> int:
        threshold = float(self.rng.random())
        cumulative = 0.0
        for index, p in enumerate(policy.tolist()):
            cumulative += p
            if threshold <= cumulative:
                return index
        return int(np.argmax(policy))
