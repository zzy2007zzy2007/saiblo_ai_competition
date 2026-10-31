"""座位效应：mirror 自对弈 512 局，按座位拆开看 margin。

用途：判断"镜像座位"到底是不是镜像（P0/P1 的可建塔区域形状不同 ⇒ 可能是两个不同的游戏）。
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


def _job(a):
    return play_game(*a)


def st(v):
    return (f"n={len(v)} mean={statistics.mean(v):+.2f} "
            f"se={statistics.pstdev(v)/len(v)**0.5:.2f} "
            f"win%={sum(1 for x in v if x > 0)/len(v):.1%}")


def main():
    n = 512
    jobs = [(sw.REF, sw.REF, 5000 + i) for i in range(n)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(_job, jobs, chunksize=8))
    m0 = [o.margin_a for o in outs if o.a_seat == 0]
    m1 = [o.margin_a for o in outs if o.a_seat == 1]
    print("ref vs ref, A=P0:", st(m0))
    print("ref vs ref, A=P1:", st(m1))
    print("pooled      :", st(m0 + m1))
    print("rounds mean", round(statistics.mean(o.rounds for o in outs), 1),
          "terminal frac", round(statistics.mean(1 if (o.hp[0] == 0 or o.hp[1] == 0) else 0
                                                  for o in outs), 3))
    # 纯 noop 的座位效应（无塔，只测路径/兵流对称性）
    jobs = [("noop", "noop", 5000 + i) for i in range(n)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(_job, jobs, chunksize=8))
    print("noop vs noop, A=P0:", st([o.margin_a for o in outs if o.a_seat == 0]))
    print("noop vs noop, A=P1:", st([o.margin_a for o in outs if o.a_seat == 1]))


if __name__ == "__main__":
    main()
