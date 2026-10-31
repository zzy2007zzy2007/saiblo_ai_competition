"""策略矩阵：两两对打，检查传递性 + 定位"为什么少造塔反而更强"。

只针对 fast_arena 的 RulePolicy；64 seed/对（座位镜像）。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_match.explore_strategy_sweep as sw  # noqa: E402
from test_match.fast_arena import duel  # noqa: E402

SPECS = {
    "noop": "noop",
    "ref": sw.REF,
    "nt0": sw.spec(n_towers=0),
    "nt0_nowpn": sw.spec(n_towers=0, weapons=()),
    "res200": sw.spec(reserve=200),
    "nt2": sw.spec(n_towers=2),
    "basic": sw.spec(target="basic", upg="never"),
}


def main():
    names = list(SPECS)
    print("pairwise (64 seeds, margin for row policy vs col policy):")
    for a, b in itertools.combinations(names, 2):
        r = duel(SPECS[a], SPECS[b], seeds=64, workers=8, start_seed=3000)
        print("{:10s} vs {:10s} margin={:+7.2f} +-{:.2f} win={}/{}".format(
            a, b, r["margin"], r["se"], r["win"], r["n"]), flush=True)


if __name__ == "__main__":
    main()
