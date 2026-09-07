"""Depth-0 greedy search: try candidate actions, value the resulting position.

Depth-0 AlphaZero experiment (docs/az_depth0_greedy.md).  The idea: normal MCTS
rolls out several levels (each advance_round is the engine bottleneck ~99% of
time), and depth-4 search already shows strong enhancement.  Q: if we instead
just try candidate actions and directly value the post-apply position with the
value head (no deeper rollout), do we still beat raw?  This trades depth for
speed — advance_round is only called once per candidate (for P1 after a full
round), not per level.

Searcher:
  - expand root once exactly like BundleMCTS._expand (k*sample_mult samples,
    keep top-k distinct decoded bundles)
  - for each kept bundle: apply to a clone, net_fn-value the child state
    (P0: no advance; P1: advance_round once — mirrors bundle_mcts's
    "if node.player == 1: child_state.advance_round()")
  - pick the bundle whose child value is best for the root player; fall back
    to the highest-prior bundle if all values are degenerate.

It returns the same BundleSearchResult shape as BundleMCTS.search so the rest
of the pipeline (az_selfplay / eval) is unchanged.

Usage (eval a checkpoint against its own raw or rule_v4):
    python code/my_ai/az_intent/az_depth0_greedy.py \
        --checkpoint az_r85.pt --opponent rule_v4 --games 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode, BundleSearchResult  # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, make_initial_state  # noqa: E402


class Depth0Greedy(BundleMCTS):
    """Bundle sampling + direct child-value greedy (no tree rollout).

    Shares BundleMCTS's bundle sampling / decode machinery via _expand, but
    overrides search() to value each top-k child once with net_fn instead of
    running MCTS iterations.
    """

    def search(self, state, player, *, temperature: float = 0.0) -> BundleSearchResult:
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType

        root = BundleNode(state=state.clone(), player=player)
        # reuse the same sampling/aggregation that produces candidate bundles
        self._expand(root, self.k)
        if not root.bundles:
            return BundleSearchResult(bundles=[], visit_policy=np.array([], dtype=np.float32),
                                      chosen_index=0, chosen_bundle=(),
                                      root_value=root.net_value, intent_counts=[])

        # value each candidate child (post-apply state) from the ROOT player's view
        child_vals = []
        for key in root.bundles:
            ops = [Operation(OperationType(int(x[0])), int(x[1]), int(x[2])) for x in key]
            child_state = state.clone()
            if ops:
                child_state.apply_operation_list(player, ops)
            if player == 1:
                child_state.advance_round()
            v = float(self.net_fn(child_state, 1 - player)["value"])
            # net_fn(child, 1-player) returns the opponent's view of the child
            # state; the root player's view is always its negation.
            child_vals.append(-v)

        child_vals = np.asarray(child_vals)
        priors = np.asarray([c.prior for c in root.children])
        # pick argmax child value; ties/NaNs fall back to prior
        best = int(np.argmax(child_vals)) if child_vals.size else int(np.argmax(priors))
        chosen = root.bundles[best]
        # build a policy shaped like the MCTS visit policy (one-hot-ish over chosen)
        policy = np.zeros(len(root.bundles), dtype=np.float32)
        policy[best] = 1.0
        return BundleSearchResult(
            bundles=list(root.bundles),
            visit_policy=policy,
            chosen_index=best,
            chosen_bundle=chosen,
            root_value=float(child_vals[best]) if child_vals.size else 0.0,
            intent_counts=list(root.intent_counts),
        )


if __name__ == "__main__":
    print("[depth0] standalone module; use via az_selfplay/eval integration")
