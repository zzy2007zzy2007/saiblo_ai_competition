"""Diagnose why Example vs Neural is slowest."""
import sys, time, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/code').resolve()))
sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/Ant-Game').resolve()))

def _worker(args):
    mode, seed = args
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/code').resolve()))
    sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/Ant-Game').resolve()))

    from SDK.backend.engine import GameState
    from AI.ai_example import AI as ExampleAI
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    MAX_R = 100

    # A: Ex vs Ex, cold_handle_rule_illegal=True
    if mode == "A":
        a0 = ExampleAI(seed=seed)
        a1 = ExampleAI(seed=seed + 1)
        a0.on_match_start(0, seed)
        a1.on_match_start(1, seed + 1)
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        for _ in range(MAX_R):
            if state.terminal: break
            ops0 = a0.choose_operations(state, 0)
            ops1 = a1.choose_operations(state, 1)
            state.resolve_turn(ops0, ops1)

    # B: Ex vs Ex, cold_handle_rule_illegal=False (default)
    elif mode == "B":
        a0 = ExampleAI(seed=seed)
        a1 = ExampleAI(seed=seed + 1)
        a0.on_match_start(0, seed)
        a1.on_match_start(1, seed + 1)
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=False)
        for _ in range(MAX_R):
            if state.terminal: break
            ops0 = a0.choose_operations(state, 0)
            ops1 = a1.choose_operations(state, 1)
            state.resolve_turn(ops0, ops1)

    # C: Ex vs Ex but each also creates a NeuralAgent (constructor overhead)
    elif mode == "C":
        model = create_model()
        _dummy = NeuralAgent(model=model, seed=seed)  # NeuralAgent init
        a0 = ExampleAI(seed=seed)
        a1 = ExampleAI(seed=seed + 1)
        a0.on_match_start(0, seed)
        a1.on_match_start(1, seed + 1)
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        for _ in range(MAX_R):
            if state.terminal: break
            ops0 = a0.choose_operations(state, 0)
            ops1 = a1.choose_operations(state, 1)
            state.resolve_turn(ops0, ops1)

    # D: Ex vs NN but with cold_handle_rule_illegal=False (should fail faster = shorter game)
    elif mode == "D":
        model = create_model()
        a0 = ExampleAI(seed=seed)
        a1 = NeuralAgent(model=model, seed=seed + 1)
        a0.on_match_start(0, seed)
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=False)
        for _ in range(MAX_R):
            if state.terminal: break
            ops0 = a0.choose_operations(state, 0)
            ops1 = a1._choose_operations(state, 1)
            state.resolve_turn(ops0, ops1)

    return mode, state.round_index, state.terminal

if __name__ == "__main__":
    WORKERS = 8
    GAMES_PER = 8

    tests = [
        ("A", "Ex+Ex, cold_handle=True"),
        ("B", "Ex+Ex, cold_handle=False"),
        ("C", "Ex+Ex, cold_handle=True + NN init"),
        ("D", "Ex+NN, cold_handle=False"),
    ]

    for mode, label in tests:
        t0 = time.perf_counter()
        with mp.Pool(WORKERS) as pool:
            results = pool.map(_worker, [(mode, i) for i in range(GAMES_PER)])
        dt = time.perf_counter() - t0
        avg_r = sum(r for _, r, _ in results) / len(results)
        all_terminal = all(t for _, _, t in results)
        print(f"{label:40s}  {GAMES_PER}场={dt:.1f}s  {dt/GAMES_PER:.2f}s/场  avg回合={avg_r:.0f}  {'全终局' if all_terminal else '有未终局'}")
