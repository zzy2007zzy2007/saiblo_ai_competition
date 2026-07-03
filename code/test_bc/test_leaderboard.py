"""Unit tests for Leaderboard opponent pool.

Tests:
  1. Score-based insertion preserves descending order
  2. Leaderboard rejects low-score candidates
  3. Leaderboard prunes to max_size
  4. get_opponents returns correct number of opponents
  5. get_opponents weighted sampling prefers strong entries
  6. Challenge-ladder insertion via match_fn
  7. Serialization round-trip (state_dict / load_state_dict)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from my_ai.leaderboard import Leaderboard


# Helper
def _make_params(seed: int, dim: int = 10) -> np.ndarray:
    return np.full(dim, float(seed), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 1. Score-based insertion: preserved descending order
# ═══════════════════════════════════════════════════════════════

def test_score_insertion_order():
    lb = Leaderboard(max_size=5, param_count=10, threshold=0.0)
    for score in [0.3, 0.9, 0.7, 0.5, 0.8, 0.1]:
        lb.add_candidate(gen=0, params=_make_params(int(score * 10)), score=score)

    scores = [e.score for e in lb.entries]
    assert scores == sorted(scores, reverse=True), f"not sorted desc: {scores}"
    print(f"  [PASS] score_insertion_order: {scores}")


# ═══════════════════════════════════════════════════════════════
# 2. Low-score rejection (threshold)
# ═══════════════════════════════════════════════════════════════

def test_low_score_rejection():
    lb = Leaderboard(max_size=3, param_count=10, threshold=0.5)
    assert lb.add_candidate(0, _make_params(1), score=0.6), "should accept 0.6"
    assert lb.add_candidate(0, _make_params(2), score=0.7), "should accept 0.7"
    assert not lb.add_candidate(0, _make_params(3), score=0.3), "should reject 0.3"
    assert len(lb.entries) == 2, f"expected 2 entries, got {len(lb.entries)}"
    print("  [PASS] low_score_rejection")


# ═══════════════════════════════════════════════════════════════
# 3. Pruning respects max_size
# ═══════════════════════════════════════════════════════════════

def test_prune_to_max():
    lb = Leaderboard(max_size=10, param_count=10, threshold=0.0)
    for i in range(20):
        lb.add_candidate(i, _make_params(i), score=min(1.0, 0.3 + i * 0.05))
    assert len(lb.entries) == 10, f"expected 10 entries after auto-prune, got {len(lb.entries)}"
    print(f"  [PASS] prune_to_max: {len(lb.entries)} entries")


# ═══════════════════════════════════════════════════════════════
# 4. get_opponents returns correct count
# ═══════════════════════════════════════════════════════════════

def test_get_opponents_count():
    lb = Leaderboard(max_size=10, param_count=10, threshold=0.0)
    for i in range(5):
        lb.add_candidate(i, _make_params(i), score=0.5 + i * 0.1)

    # Request more than pool has → sample with replacement
    opps = lb.get_opponents(k=8)
    assert len(opps) == 8, f"expected 8 opponents, got {len(opps)}"
    gens = [o["gen"] for o in opps]
    assert -1 not in gens, f"no placeholder expected in results, got gen=-1"

    # Request fewer than pool has
    opps2 = lb.get_opponents(k=3)
    assert len(opps2) == 3, f"expected 3 opponents, got {len(opps2)}"
    print(f"  [PASS] get_opponents_count: pool=5, k=8→{len(opps)}, k=3→{len(opps2)}")


# ═══════════════════════════════════════════════════════════════
# 5. Weighted sampling: strong entries sampled more
# ═══════════════════════════════════════════════════════════════

def test_weighted_sampling():
    lb = Leaderboard(max_size=5, param_count=10, threshold=0.0)
    for i in range(5):
        lb.add_candidate(i, _make_params(i), score=0.1 + i * 0.2)  # 0.1, 0.3, 0.5, 0.7, 0.9

    # Sample many times and count
    counts = {i: 0 for i in range(5)}
    for _ in range(1000):
        opps = lb.get_opponents(k=1)
        # Find which entry we got by matching params
        for idx, e in enumerate(lb.entries):
            if np.allclose(e.params, opps[0]["params"]):
                counts[idx] += 1
                break

    # Rank 0 (strongest, score=0.9) should be sampled most
    assert counts[0] > counts[1], f"rank 0 ({counts[0]}) should > rank 1 ({counts[1]})"
    print(f"  [PASS] weighted_sampling: rank counts={counts}")


# ═══════════════════════════════════════════════════════════════
# 6. Challenge-ladder insertion
# ═══════════════════════════════════════════════════════════════

def test_challenge_ladder():
    lb = Leaderboard(max_size=5, param_count=10, threshold=0.5)

    def match(a, b):
        # Simple deterministic "stronger wins": higher first element = stronger
        return 1.0 if a[0] > b[0] else 0.0

    # Pre-populate with 3 entries
    for val in [1.0, 2.0, 3.0]:
        lb.add_candidate(0, np.full(10, val, dtype=np.float32), match_fn=match)

    # New candidate (strength 2.5) should slot between 2.0 and 3.0
    added = lb.add_candidate(0, np.full(10, 2.5, dtype=np.float32), match_fn=match)
    assert added, "candidate should be added"
    scores = [e.params[0] for e in lb.entries]
    assert scores == [3.0, 2.5, 2.0, 1.0], f"unexpected ladder order: {scores}"
    print(f"  [PASS] challenge_ladder: {scores}")

    # Weak candidate (strength 0.5) should be rejected
    added = lb.add_candidate(0, np.full(10, 0.5, dtype=np.float32), match_fn=match)
    assert not added, "weak candidate should be rejected"
    print("  [PASS] challenge_ladder rejection")


# ═══════════════════════════════════════════════════════════════
# 7. Serialization round-trip
# ═══════════════════════════════════════════════════════════════

def test_serialization():
    lb = Leaderboard(max_size=10, param_count=10, threshold=0.0)
    for i in range(5):
        lb.add_candidate(i, _make_params(i), score=0.5 + i * 0.1)

    sd = lb.state_dict()
    lb2 = Leaderboard(max_size=10, param_count=10, threshold=0.0)
    lb2.load_state_dict(sd)

    assert len(lb2.entries) == len(lb.entries)
    for e1, e2 in zip(lb.entries, lb2.entries):
        assert e1.gen == e2.gen
        assert np.allclose(e1.params, e2.params)
        assert abs(e1.score - e2.score) < 1e-6
    print(f"  [PASS] serialization: {len(lb2.entries)} entries match")


# ═══════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════

def main():
    tests = [
        test_score_insertion_order,
        test_low_score_rejection,
        test_prune_to_max,
        test_get_opponents_count,
        test_weighted_sampling,
        test_challenge_ladder,
        test_serialization,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{n_pass}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
