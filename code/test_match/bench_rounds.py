"""Count total rounds per full game across 3 scenarios."""
from __future__ import annotations
import sys, time, multiprocessing as mp
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND


def _run(agent_type: str, seed: int) -> dict:
    import sys
    from pathlib import Path
    _RP = Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = Path(__file__).resolve().parents[1]
    for p in (_RP, _CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    if agent_type == "example":
        from AI.ai_example import AI as AI0, AI as AI1
        a0, a1 = AI0(seed=seed), AI1(seed=seed + 10000)
    elif agent_type == "neural":
        from my_ai.agent import NeuralAgent
        from my_ai.network import create_zero_model
        a0 = NeuralAgent(model=create_zero_model())
        a1 = NeuralAgent(model=create_zero_model())
    elif agent_type == "neural_vs_example":
        from AI.ai_example import AI as AI1
        from my_ai.agent import NeuralAgent
        from my_ai.network import create_zero_model
        a0 = NeuralAgent(model=create_zero_model())
        a1 = AI1(seed=seed)

    our_p = seed % 2
    p0 = a0 if our_p == 0 else a1
    p1 = a1 if our_p == 0 else a0

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    t0 = time.perf_counter()
    for rnd in range(MAX_ROUND):
        if state.terminal:
            break
        state.resolve_turn(p0.choose_operations(state, 0), p1.choose_operations(state, 1))
    dt = time.perf_counter() - t0
    return {"rounds": rnd + 1, "time_s": dt, "winner": state.winner}


def main():
    import numpy as np
    for at in ["example", "neural", "neural_vs_example"]:
        print(f"\n  {at}  (12 workers, 24 games):")
        with mp.Pool(12) as pool:
            results = pool.starmap(_run, [(at, s) for s in range(24)])
        rounds = [r["rounds"] for r in results]
        times = [r["time_s"] for r in results]
        total_rounds = sum(rounds)
        total_time = sum(times)
        avg_r = np.mean(rounds)
        avg_t = np.mean(times)
        per_round_ms = avg_t / avg_r * 1000
        print(f"    avg rounds: {avg_r:.0f}  (min={min(rounds)}, max={max(rounds)})")
        print(f"    avg time:   {avg_t*1000:.0f}ms  (min={min(times)*1000:.0f}, max={max(times)*1000:.0f})")
        print(f"    ms/round:   {per_round_ms:.0f}")
        print(f"    total:      {total_rounds}rnd in {total_time:.0f}s")


if __name__ == "__main__":
    main()
