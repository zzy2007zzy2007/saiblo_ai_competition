"""稳健性检查：把关键 arm 换一组**全新 seed**（90000+）重测，防止"种子过拟合"。

跑的是本次结论依赖的 5 个臂（vs 同一个 ref），N=192/臂，但 seed 从 90000 开始（主表用的是 2000+）。
"""
from __future__ import annotations

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_match.explore_strategy_sweep as sw  # noqa: E402
from test_match.fast_arena import play_game  # noqa: E402

ARMS = [
    ("ref", sw.REF),
    ("weapons=none", sw.spec(weapons=())),
    ("build_reserve=300", sw.spec(build_reserve=300)),
    ("upg=never", sw.spec(upg="never")),
    ("upg=damaged60", sw.spec(upg="damaged", upg_hp_thr=0.6)),
    ("repair_cycle60", sw.spec(upg="damaged", upg_hp_thr=0.6, repair_cycle=True,
                               build_reserve=100)),
]


def _job(a):
    return play_game(*a)


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    jobs, index = [], []
    for name, sp in ARMS:
        for s in range(seeds):
            jobs.append((sp, sw.REF, 90000 + s))
            index.append(name)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(_job, jobs, chunksize=8))
    by = {}
    for nm, o in zip(index, outs):
        by.setdefault(nm, []).append(o.margin_a)
    print(f"{'arm':22s} {'n':>4s} {'margin':>8s} {'se':>6s} {'win%':>6s}")
    for name, _ in ARMS:
        mg = by[name]
        print(f"{name:22s} {len(mg):4d} {statistics.mean(mg):8.2f} "
              f"{statistics.pstdev(mg)/len(mg)**0.5:6.2f} "
              f"{sum(1 for x in mg if x > 0)/len(mg)*100:5.1f}%")


if __name__ == "__main__":
    main()
