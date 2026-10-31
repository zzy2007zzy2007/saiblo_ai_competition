"""机制实验：把"升级"当成**修理**用（升级回满血 ⇒ 廉价修塔），和我们默认的
"有钱就升级"、"从不升级" 做对照。

背景假设（来自冠军对照：冠军每局 33 次升级 + 32 次降级，我们 0 次）：
  · 升级**回满血** ⇒ 60 币把 Basic 修满；再降级只退 54 ⇒ **净花 6 币修一次**。
  · 200 币的三级升级 = 2.2 次闪电 ⇒ 按 §3 的策略轴看是负收益。
  · 所以"升级"应该只用来**修理**（或修完就降级回去），而不是"有钱就升"。

臂（每臂 192 局 = 96 seed × 2 座位，参照 = `ref`）：
  ref                : build_reserve=100, upg=always
  never              : 从不升级
  damaged40/60/80    : 只在塔血量比 <= 0.4/0.6/0.8 时升级（= 修理）
  damaged60_br300    : 修理 + 攒钱（build_reserve=300）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_match.explore_strategy_sweep as sw  # noqa: E402


def main():
    arms = [
        ("ref", sw.REF, sw.REF),
        ("upg=never", sw.spec(upg="never"), sw.REF),
        ("damaged40", sw.spec(upg="damaged", upg_hp_thr=0.4), sw.REF),
        ("damaged60", sw.spec(upg="damaged", upg_hp_thr=0.6), sw.REF),
        ("damaged80", sw.spec(upg="damaged", upg_hp_thr=0.8), sw.REF),
        ("damaged60_br300", sw.spec(upg="damaged", upg_hp_thr=0.6, build_reserve=300), sw.REF),
        ("damaged60_n4", sw.spec(upg="damaged", upg_hp_thr=0.6, n_towers=4), sw.REF),
        ("always_br300(best)", sw.spec(build_reserve=300, upg="always"), sw.REF),
    ]
    seeds = int(os.environ.get("SEEDS", "96"))
    workers = int(os.environ.get("WORKERS", "6"))
    rows = sw.run_arms(arms, seeds, workers)
    sw.print_rows(rows)
    sw.save_rows(rows, f"repair_n{seeds}.tsv")


if __name__ == "__main__":
    main()
