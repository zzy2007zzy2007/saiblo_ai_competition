"""Smoke test for P1 intent-space MCTS core.

1. Random net_fn -> tree mechanics (search, expansion, backup, visit policy).
2. If a real checkpoint is given, run one search with the real network.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from SDK.backend.engine import GameState
from my_ai.az_intent.mcts import IntentMCTS, compute_intent_prior, intent_to_operation


def dummy_net(state, player):
    """Random network output (mechanism test only)."""
    rng = np.random.default_rng(state.round_index * 7 + player * 13)
    return {
        "action_map": rng.standard_normal((24, 19, 19)).astype(np.float32) * 0.5,
        "head_logits": [rng.standard_normal(24).astype(np.float32) for _ in range(3)],
        "value": float(rng.standard_normal()),
    }


def test_dummy() -> None:
    state = GameState.initial(seed=3, cold_handle_rule_illegal=True)
    mcts = IntentMCTS(dummy_net, iterations=16, max_depth_rounds=1, seed=1)
    for player in (0, 1):
        res = mcts.search(state, player, temperature=0.0, add_root_noise=False)
        n = len(res.intents)
        print(f"[dummy] player={player} intents={n} visits_sum={res.visit_policy.sum():.3f} "
              f"chosen=class{res.chosen_intent.class_id}@({res.chosen_intent.x},{res.chosen_intent.y}) "
              f"root_value={res.root_value:.3f}")
        assert n > 0, "root must have children"
        assert abs(res.visit_policy.sum() - 1.0) < 1e-3, "visit policy must sum to 1"
        op = intent_to_operation(state, player, res.chosen_intent)
        if res.chosen_intent.class_id != 23:
            assert op is not None, "chosen intent must decode to an op"
            ok = state.can_apply_operation(player, op, ())
            print(f"      chosen op={op.op_type} legal={ok}")
        else:
            print("      chosen = HOLD")
    # prior sanity
    net_out = dummy_net(state, 0)
    intents, priors = compute_intent_prior(net_out, state, 0, 0, 1.0, 1.0)
    assert abs(priors.sum() - 1.0) < 1e-3
    print(f"[dummy] head0 atomic intents={len(intents)} priors_sum={priors.sum():.4f}")
    print("P1 dummy OK")


def test_real(checkpoint: str) -> None:
    import torch
    torch.set_num_threads(4)
    from my_ai.network import create_model
    from SDK.utils.features import FeatureExtractor

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    num_heads = ckpt.get("num_heads", 3)
    no_bn = ckpt.get("no_bn", False)
    model = create_model(num_heads=num_heads, no_bn=no_bn)
    if "mean" in ckpt:
        model.set_parameters_from_vector(ckpt["mean"].numpy())
    elif "top2_params" in ckpt and len(ckpt["top2_params"]) > 0:
        model.set_parameters_from_vector(ckpt["top2_params"][0].numpy())
    else:
        raise KeyError(f"no usable params in {list(ckpt.keys())}")
    model.eval()

    feat = FeatureExtractor(max_actions=96)

    def net_fn(state, player):
        obs = feat.encode_observation(state, player, np.zeros(96))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            out = model(board, stats)
        heads = [out[f"head{i+1}_logits"].squeeze(0).numpy() for i in range(num_heads)]
        return {
            "action_map": out["action_map"].squeeze(0).numpy(),
            "head_logits": heads,
            "value": float(out["value"].squeeze(0).numpy()),
        }

    state = GameState.initial(seed=3, cold_handle_rule_illegal=True)
    mcts = IntentMCTS(net_fn, iterations=16, max_depth_rounds=1, seed=1)
    import time
    t0 = time.perf_counter()
    res = mcts.search(state, 0, temperature=0.0, add_root_noise=False)
    dt = time.perf_counter() - t0
    print(f"[real] {checkpoint} intents={len(res.intents)} chosen=class{res.chosen_intent.class_id} "
          f"@({res.chosen_intent.x},{res.chosen_intent.y}) value={res.root_value:.3f} search={dt:.2f}s")
    print("P1 real OK")


if __name__ == "__main__":
    test_dummy()
    if len(sys.argv) > 1:
        test_real(sys.argv[1])
