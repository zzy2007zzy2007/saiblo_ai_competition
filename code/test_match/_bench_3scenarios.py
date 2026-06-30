"""Benchmark 3 scenarios × 12 workers, 100 rounds each."""
import sys, time, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/code').resolve()))
sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/Ant-Game').resolve()))

def _worker(args):
    scenario, seed = args
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/code').resolve()))
    sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/Ant-Game').resolve()))

    from SDK.backend.engine import GameState
    from AI.ai_example import AI as ExampleAI
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    MAX_R = 100

    if scenario == "ex_ex":
        a0 = ExampleAI(seed=seed)
        a1 = ExampleAI(seed=seed + 1)
        a0.on_match_start(0, seed)
        a1.on_match_start(1, seed + 1)
        state = GameState.initial(seed=seed)
        for _ in range(MAX_R):
            if state.terminal: break
            ops0 = a0.choose_operations(state, 0)
            ops1 = a1.choose_operations(state, 1)
            state.resolve_turn(ops0, ops1)

    elif scenario == "ex_nn":
        model = create_model()
        a0 = ExampleAI(seed=seed)
        a1 = NeuralAgent(model=model, seed=seed + 1)
        a0.on_match_start(0, seed)
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        for _ in range(MAX_R):
            if state.terminal: break
            ops0 = a0.choose_operations(state, 0)
            ops1 = a1._choose_operations(state, 1)
            state.resolve_turn(ops0, ops1)

    else:  # nn_nn
        model0 = create_model()
        model1 = create_model()
        a0 = NeuralAgent(model=model0, seed=seed)
        a1 = NeuralAgent(model=model1, seed=seed + 1)
        state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
        for _ in range(MAX_R):
            if state.terminal: break
            ops0 = a0._choose_operations(state, 0)
            ops1 = a1._choose_operations(state, 1)
            state.resolve_turn(ops0, ops1)

    return scenario, seed, state.round_index

if __name__ == "__main__":
    WORKERS = 12
    GAMES_PER = 12

    for scenario, label in [("ex_ex", "Example vs Example"),
                            ("ex_nn", "Example vs Neural"),
                            ("nn_nn", "Neural vs Neural")]:
        t0 = time.perf_counter()
        with mp.Pool(WORKERS) as pool:
            results = pool.map(_worker, [(scenario, i) for i in range(GAMES_PER)])
        dt = time.perf_counter() - t0
        avg_rounds = sum(r for _, _, r in results) / len(results)
        print(f"{label:25s}  {GAMES_PER}场={dt:.1f}s  {dt/GAMES_PER:.2f}s/场  {avg_rounds:.0f}回合/场  {dt/GAMES_PER/avg_rounds*1000:.0f}ms/回合")
