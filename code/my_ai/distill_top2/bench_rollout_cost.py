"""实测三件事的耗时：C++ rollout / pybind 引擎一次推进 / Python 版 evaluate_terminal / 我们的值网。

目的是估算"把亚军的算法原样搬到 Python（引擎走 pybind）+ 换我们的值网"会慢多少。
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "cpp_engine"))
sys.path.insert(0, str(_REPO / "Ant-Game"))
sys.path.insert(0, str(_REPO / "Ant-Game" / "SDK"))

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")
STATES = _REPO / "training_history/distill_top2/abl_linear2_raw/ents_seed00732.jsonl"


def make_env():
    env = os.environ.copy()
    env["PATH"] = r"C:\mingw64\bin;" + env.get("PATH", "")
    return env


def bench_cpp_rollout() -> None:
    """用**亚军自己的合法计划**（mode=rollout, max_plans=0）测单次 rollout：
       两次不同 rollout 数取斜率，抵消启动/建树/JSON 开销。"""
    state = json.loads(STATES.open(encoding="utf-8").readlines()[200])
    base = dict(state, seed=732, player=0, mode="rollout", max_plans=0)
    times, counts = {}, {}
    for n_roll in (4, 32):
        payload = (json.dumps(dict(base, rollouts=n_roll)) + "\n").encode()
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            proc = subprocess.run([str(EXE)], input=payload, capture_output=True, env=make_env())
            samples.append(time.perf_counter() - t0)
            assert proc.returncode == 0, proc.stderr[:300]
        result = json.loads(proc.stdout.decode("utf-8").splitlines()[0])
        valid = [p for p in result["plans"] if p.get("n")]
        counts[n_roll] = len(valid) * n_roll
        times[n_roll] = statistics.median(samples)
    per_us = (times[32] - times[4]) / (counts[32] - counts[4]) * 1e6
    print(f"[C++] {counts[4]} 次 rollout: {times[4]*1000:.0f} ms；{counts[32]} 次: {times[32]*1000:.0f} ms")
    print(f"[C++] **单次 8 回合 rollout ≈ {per_us:.1f} µs**"
          f"  ⇒ 一次决策 14,500 次 ≈ {per_us * 14500 / 1e6:.2f} s")

def bench_pybind() -> None:
    """pybind 引擎：clone / advance_round / 读状态 各自多快。"""
    import native_game
    from my_ai.az_intent.game_state_facade import GameStateFacade

    facade = GameStateFacade.initial(732)
    for _ in range(200):          # 推进到中局
        facade.advance_round()

    rounds = 200
    t0 = time.perf_counter()
    for _ in range(rounds):
        facade.advance_round()
    per_round_us = (time.perf_counter() - t0) / rounds * 1e6
    print(f"[pybind] advance_round ≈ {per_round_us:.1f} µs/次")

    t0 = time.perf_counter()
    for _ in range(rounds):
        _ = list(facade.towers)
        _ = list(facade.ants)
    per_read_us = (time.perf_counter() - t0) / rounds * 1e6
    print(f"[pybind] 读一次 towers+ants ≈ {per_read_us:.1f} µs")

    t0 = time.perf_counter()
    for _ in range(50):
        _ = facade.clone()
    per_clone_us = (time.perf_counter() - t0) / 50 * 1e6
    print(f"[pybind] clone ≈ {per_clone_us:.1f} µs")


def bench_py_eval() -> None:
    """Python 版 evaluate_terminal（亚军手写价值）单次耗时。"""
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from my_ai.distill_top2.lure_value_eval import evaluate_terminal

    facade = GameStateFacade.initial(732)
    for _ in range(200):
        facade.advance_round()
    evaluate_terminal(facade, 0)      # 预热
    rounds = 200
    t0 = time.perf_counter()
    for _ in range(rounds):
        evaluate_terminal(facade, 0)
    per_call_us = (time.perf_counter() - t0) / rounds * 1e6
    print(f"[Python] evaluate_terminal ≈ {per_call_us:.1f} µs/次")


def bench_value_net() -> None:
    """我们的值网前向：单条 vs 批量。"""
    import torch
    from my_ai.network import create_model, model_kwargs_from_ckpt
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from my_ai.az_intent.eval import _FE  # 特征提取器

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _REPO / "checkpoints/grafted_sdk_A.pt"
    if not ckpt_path.exists():
        print(f"[值网] 缺少 {ckpt_path.name}，跳过")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_model(**model_kwargs_from_ckpt(ckpt)).to(device).eval()
    feat = _FE(max_actions=96)
    facade = GameStateFacade.initial(732)
    for _ in range(200):
        facade.advance_round()
    obs = feat.encode_observation(facade, 0, np.zeros(96))
    board = torch.from_numpy(obs["board"]).unsqueeze(0).float().to(device)
    stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float().to(device)
    with torch.no_grad():
        model(board, stats)
        for batch in (1, 64, 256):
            b = board.expand(batch, *board.shape[1:]).contiguous()
            s = stats.expand(batch, *stats.shape[1:]).contiguous()
            for _ in range(5):
                model(b, s)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                model(b, s)
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_batch_us = (time.perf_counter() - t0) / 20 * 1e6
            print(f"[值网] batch={batch:<4} 一次前向 {per_batch_us:8.1f} µs "
                  f"⇒ 每条 {per_batch_us / batch:7.1f} µs")
    # 特征提取（每换一个状态就要一次）
    rounds = 100
    t0 = time.perf_counter()
    for _ in range(rounds):
        feat.encode_observation(facade, 0, np.zeros(96))
    print(f"[值网] encode_observation ≈ {(time.perf_counter()-t0)/rounds*1e6:.1f} µs/次")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "cpp"):
        bench_cpp_rollout()
    if which in ("all", "pybind"):
        bench_pybind()
    if which in ("all", "eval"):
        bench_py_eval()
    if which in ("all", "net"):
        bench_value_net()
