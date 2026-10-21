"""补齐耗时测量：原生引擎 advance_round / facade / Python 手写价值 / 我们值网（单条与批量）。"""
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))


def timeit(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    import native_game
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from my_ai.distill_top2.lure_value_eval import evaluate_terminal

    g = native_game.NativeGame()
    g.init_game(seed=732, movement_policy="enhanced", cold=True)
    for _ in range(200):
        g.advance_round()
    print(f"[原生] NativeGame.advance_round      ≈ {timeit(g.advance_round):.1f} µs")
    print(f"[原生] NativeGame.clone              ≈ {timeit(g.clone):.1f} µs")

    facade = GameStateFacade(g)
    print(f"[facade] advance_round（含清缓存）   ≈ {timeit(facade.advance_round):.1f} µs")

    def read_state():
        _ = list(facade.towers)
        _ = list(facade.ants)
        _ = [b.hp for b in facade.bases]
        _ = list(facade.coins)
        return None

    read_state()
    print(f"[facade] 读一次全状态               ≈ {timeit(read_state):.1f} µs")
    print(f"[Python] evaluate_terminal（手写价值）≈ {timeit(lambda: evaluate_terminal(facade, 0)):.1f} µs")

    # 我们的值网
    import torch
    torch_mod = torch
    from my_ai.network import create_model, model_kwargs_from_ckpt
    from SDK.utils.features import FeatureExtractor as _FE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _REPO / "checkpoints/grafted_sdk_A.pt"
    if not ckpt_path.exists():
        print(f"[值网] 缺 {ckpt_path}，跳过")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_model(**model_kwargs_from_ckpt(ckpt)).to(device).eval()
    feat = _FE(max_actions=96)
    enc_us = timeit(lambda: feat.encode_observation(facade, 0, np.zeros(96)), n=100)
    print(f"[值网] encode_observation（每个状态一次）≈ {enc_us:.1f} µs")
    obs = feat.encode_observation(facade, 0, np.zeros(96))
    board = torch.from_numpy(obs["board"]).unsqueeze(0).float().to(device)
    stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float().to(device)
    with torch_mod.no_grad():
        model(board, stats)
        for batch in (1, 64, 256):
            b = board.expand(batch, *board.shape[1:]).contiguous()
            s = stats.expand(batch, *stats.shape[1:]).contiguous()
            for _ in range(5):
                model(b, s)
            if device.type == "cuda":
                torch_mod.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 20
            for _ in range(reps):
                model(b, s)
            if device.type == "cuda":
                torch_mod.cuda.synchronize()
            us = (time.perf_counter() - t0) / reps * 1e6
            print(f"[值网] batch={batch:<4} 前向 {us:8.1f} µs  ⇒ 每条 {us / batch:7.1f} µs")


if __name__ == "__main__":
    main()
