"""Neural vs rule_v4 with multiprocessing."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_REPO = _ROOT / "Ant-Game"
_RULE = _ROOT / "其他版本ai" / "rule_v4"
for p in (_REPO, _ROOT / "code", _RULE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from ai import AI as RuleV4


class Cfg:
    pass


def _worker(ckpt_path: str, seed: int, num_heads: int):
    torch.set_num_threads(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = create_model(num_heads=num_heads)
    model.set_parameters_from_vector(ckpt["top2_params"][0].numpy())
    neural = NeuralAgent(model=model)
    rule = RuleV4()

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        u = neural._choose_operations(state, 0)
        v = rule.choose_operations(state, 1)
        state.resolve_turn(u or [], v or [])
    h0, h1 = state.bases[0].hp, state.bases[1].hp
    return seed, 1 if h0 > h1 else 0, h0, h1


def main():
    ckpt_path = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    num_heads = ckpt.get("num_heads", 3)

    wins = 0
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for seed in range(games):
            futures.append(pool.submit(_worker, ckpt_path, seed, num_heads))
        for f in as_completed(futures):
            seed, result, h0, h1 = f.result()
            wins += result
            print(f"  seed={seed:2d}  {'WIN' if result else 'LOSS'}  HP {h0:2d} vs {h1:2d}")
    print(f"\nResult: {wins}/{games} ({wins/games*100:.0f}%)")


if __name__ == "__main__":
    main()
