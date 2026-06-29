"""Test: verify the full neural network pipeline with zero and tuned weights."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch
import numpy as np

from my_ai.network import create_zero_model, create_tuned_model
from my_ai.decoder import decode_network_output
from SDK.backend import GameState
from SDK.utils.features import FeatureExtractor


def test_zero_weights():
    print("=" * 60)
    print("TEST 1: Zero weights")
    print("=" * 60)

    model = create_zero_model()
    state = GameState.initial(seed=7)
    player = 0

    extractor = FeatureExtractor()
    obs = extractor.encode_observation(state, player, np.zeros(96))

    board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
    stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()

    with torch.no_grad():
        output = model(board, stats)

    for key in ["action_map", "head1_logits", "head2_logits", "head3_logits", "value"]:
        val = output[key]
        print(f"  {key}: shape={val.shape}, range=[{val.min():.6f}, {val.max():.6f}]")

    detached = {k: v.detach() for k, v in output.items()}
    operations = decode_network_output(detached, state, player)
    print(f"\n  Decoded operations ({len(operations)}):")
    for op in operations:
        print(f"    {op}")

    for op in operations:
        valid = state.can_apply_operation(player, op)
        status = "[OK]" if valid else "[INVALID]"
        print(f"    {status} can_apply_operation({op}) = {valid}")

    state_clone = state.clone()
    invalid = state_clone.apply_operation_list(player, operations)
    ok = len(invalid) == 0
    print(f"  Batch apply: {len(invalid)} invalid {'[OK]' if ok else '[FAIL]'}")

    return operations


def test_tuned_weights():
    print("\n" + "=" * 60)
    print("TEST 2: Tuned weights")
    print("=" * 60)

    model = create_tuned_model()
    state = GameState.initial(seed=7)
    player = 0

    extractor = FeatureExtractor()
    obs = extractor.encode_observation(state, player, np.zeros(96))

    board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
    stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()

    with torch.no_grad():
        output = model(board, stats)

    for i in range(1, 4):
        logits = output[f"head{i}_logits"][0]
        argmax = logits.argmax().item()
        expected = {1: 0, 2: 16, 3: 17}[i]
        status = "[OK]" if argmax == expected else "[FAIL]"
        print(f"  head{i} argmax: {argmax} (expected {expected}) {status}")

    detached = {k: v.detach() for k, v in output.items()}
    operations = decode_network_output(detached, state, player)
    print(f"\n  Decoded operations ({len(operations)}):")
    for op in operations:
        print(f"    {op}")

    return operations


def test_full_match():
    print("\n" + "=" * 60)
    print("TEST 3: Full match with zero-weight model")
    print("=" * 60)

    import time
    from my_ai.agent import NeuralAgent
    from AI.ai_random import AI as RandomAI

    zero_agent = NeuralAgent(model=create_zero_model(), seed=7)
    random_agent = RandomAI(seed=7)

    state = GameState.initial(seed=7)
    t0 = time.time()

    for round_idx in range(512):
        if state.terminal:
            break

        ops0 = zero_agent._choose_operations(state, 0)
        state.apply_operation_list(0, ops0)

        ops1 = random_agent.choose_operations(state, 1)
        state.apply_operation_list(1, ops1)

        state.advance_round()

        if round_idx % 100 == 0:
            hp0, hp1 = state.bases[0].hp, state.bases[1].hp
            towers0 = len([t for t in state.towers if t.player == 0])
            print(f"  Round {round_idx}: HP {hp0} vs {hp1}, towers {towers0}")

    elapsed = time.time() - t0
    hp0, hp1 = state.bases[0].hp, state.bases[1].hp
    print(f"\n  Match finished in {round_idx + 1} rounds, {elapsed:.1f}s")
    print(f"  Final HP: {hp0} vs {hp1}")
    print(f"  No crashes: [OK]")
    return state


if __name__ == "__main__":
    test_zero_weights()
    test_tuned_weights()
    test_full_match()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE [OK]")
    print("=" * 60)
