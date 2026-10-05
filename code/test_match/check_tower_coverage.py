"""覆盖率检查：**这批采集数据里有多少"有塔局面"？**（随机钉类是否真的把经济状态喂进来了）

依据：`docs/az_posnet_random_class_plan.md` §6 判据 1/2（通道覆盖、位置样本量），
以及 2026-10-05 实测的缺口——**纯自对弈（A1 配置）100 局里"有塔"样本数 = 0**。

"有塔"的定义与价值网训练器一致：`board[4]`（己方塔通道）**任意格非 0**
（`train_value_net.py:47,166`；通道含义见 `SDK/utils/features.py:177-180`）。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/check_tower_coverage.py \
      <pkl 目录> [--limit 40]
"""
from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

CH_OWN, CH_ENEMY = 4, 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--limit", type=int, default=0, help="只读前 N 个 pkl（0=全部）")
    a = ap.parse_args()

    d = Path(a.dir)
    files = sorted(d.rglob("az_selfplay_seed*.pkl"))
    if a.limit:
        files = files[:a.limit]
    print(f"目录: {d}  pkl 数={len(files)}")

    n_tot = n_own = n_enemy = 0
    per_game = []
    # 位置目标落在哪个类（用 bundles/visit 还原"搜索选中"的那个 bundle 的第一个 op 的类）
    for f in files:
        with open(f, "rb") as fh:
            samples = pickle.load(fh)["samples"]
        own = en = 0
        for s in samples:
            b = np.asarray(s["board"])
            if (b[CH_OWN] != 0).any():
                own += 1
            if (b[CH_ENEMY] != 0).any():
                en += 1
        n_tot += len(samples)
        n_own += own
        n_enemy += en
        per_game.append((f.name, len(samples), own, en))

    print(f"\n样本总数 = {n_tot}")
    print(f"  己方有塔的样本 = {n_own}  ({n_own/max(n_tot,1):.1%})")
    print(f"  敌方有塔的样本 = {n_enemy}  ({n_enemy/max(n_tot,1):.1%})")
    print("\n逐局（样本数 / 己方有塔 / 敌方有塔）：")
    for name, n, own, en in per_game[:20]:
        print(f"  {name}: {n:5d} / {own:5d} / {en:5d}")
    if len(per_game) > 20:
        print(f"  …（共 {len(per_game)} 局）")
    print("\n判读：`己方有塔` 比例从 0% 变成可见的百分比 ⇒ 随机钉类确实把'经济局面'喂进来了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
