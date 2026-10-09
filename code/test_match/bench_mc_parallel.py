"""测：MC 的候选 rollout（`quiet_value`）能否用**线程**并行（GIL 是否释放）。

背景：子博弈里一条 128 局臂要 ~5 h，**吞吐是整个项目的瓶颈**。MC 每 5 回合仲裁一次、每次评估 5 个候选，
每个候选 = 一次"没人出招"的引擎 rollout（≤256 回合）⇒ 若能并行且**决策逐位不变**，就是纯工程加速（不改算法）。
本脚本只测"并行是否提速"，不改任何仓库代码。

用法: python code/test_match/bench_mc_parallel.py [--rounds 256] [--n 8] [--workers 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from my_ai.az_intent.az_selfplay import make_initial_state  # noqa: E402
from my_ai.az_intent.mc_quiet import quiet_value           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=256)
    ap.add_argument("--n", type=int, default=8, help="要评估的候选数（= clone 数）")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    st0 = make_initial_state(a.seed)
    clones = [st0.clone() for _ in range(a.n)]
    print(f"state 造好: seed={a.seed} player=0 round={getattr(st0, 'round_index', '?')} horizon={a.rounds} n={a.n}")

    t0 = time.perf_counter()
    seq = [quiet_value(c, 0, max_rounds=a.rounds) for c in clones]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        par = list(ex.map(lambda c: quiet_value(c, 0, max_rounds=a.rounds), clones))
    t_par = time.perf_counter() - t0

    print(f"顺序 {a.n} 次: {t_seq:.2f} s   ({t_seq / a.n * 1000:.0f} ms/次)")
    print(f"线程 {a.workers} 并发: {t_par:.2f} s   => 加速 {t_seq / t_par:.2f}×")
    print(f"结果一致: {seq == par}   (若不等于 1 ⇒ 线程化不安全)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
