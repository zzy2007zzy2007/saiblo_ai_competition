"""统计 `inject_ex02` 这批注入采集的数据质量：**"有塔局面"的覆盖到底有多少？**

为什么关键：类轴计划 (a) 的全部意义就是"让价值网见过有塔的局面"。所以要先量：
  * 有多少比例的决策点，**该玩家场上有自己的塔**（board 通道 4 ≠ 0）；
  * 有多少比例场上有**敌方**塔（通道 5）；有 ≥2 座自己塔的比例；
  * 决定性比例 / value_target 分布 / 每个玩家的样本数是否均衡。

`board` 通道（`SDK/utils/features.py:177-180`）：4 = 自己的塔，5 = 敌方的塔，6 = 塔等级/2。

用法: python _tmp_inject_data_stats.py [--dir training_history/inject_ex02/data] [--workers 16]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CH_OWN, CH_ENEMY = 4, 5


def _scan(f: str):
    d = pickle.load(open(f, "rb"))
    samples = d["samples"] if isinstance(d, dict) else d
    n = own = enemy = own2 = 0
    vt = []
    vt_by_player = {0: [], 1: []}
    for s in samples:
        b = np.asarray(s["board"])
        n += 1
        if b.ndim == 3:
            mine = np.count_nonzero(b[CH_OWN] != 0)
            theirs = np.count_nonzero(b[CH_ENEMY] != 0)
        else:                                    # 万一被展平
            mine = theirs = 0
        if mine:
            own += 1
        if mine >= 2:
            own2 += 1
        if theirs:
            enemy += 1
        v = s.get("value_target")
        if v is not None:
            vt.append(float(v))
            vt_by_player[int(s["player"])].append(float(v))
    return {"file": Path(f).name, "n": n, "own": own, "own2": own2, "enemy": enemy,
            "vt_mean": float(np.mean(vt)) if vt else float("nan"),
            "vt_std": float(np.std(vt)) if vt else float("nan"),
            "p0": len(vt_by_player[0]), "p1": len(vt_by_player[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="training_history/inject_ex02/data")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.dir).glob("az_selfplay_seed*.pkl"))
    print("[stats] %d 个 pkl" % len(files), flush=True)

    import multiprocessing as mp
    with mp.Pool(a.workers) as pool:
        res = pool.map(_scan, files)

    n = sum(r["n"] for r in res)
    own = sum(r["own"] for r in res)
    own2 = sum(r["own2"] for r in res)
    enemy = sum(r["enemy"] for r in res)
    p0 = sum(r["p0"] for r in res)
    p1 = sum(r["p1"] for r in res)
    print("\n=== %s ===" % a.dir)
    print("  样本总数      %8d  (每局均 %.0f)" % (n, n / max(len(res), 1)))
    print("  我方在场塔    %8d  = %5.1f%%   <- **有塔局面的覆盖**" % (own, 100.0 * own / max(n, 1)))
    print("  我方 >=2 塔   %8d  = %5.1f%%" % (own2, 100.0 * own2 / max(n, 1)))
    print("  敌方在场塔    %8d  = %5.1f%%" % (enemy, 100.0 * enemy / max(n, 1)))
    print("  player0/1     %8d / %d   (均衡度 %.3f)" % (p0, p1, p0 / max(p0 + p1, 1)))
    vt_all = np.concatenate([[r["vt_mean"]] * r["n"] for r in res]) if res else np.zeros(0)
    print("  每局 value_target 均值: 均值 %.4f, 局间标准差 %.4f"
          % (np.mean([r["vt_mean"] for r in res]), np.std([r["vt_mean"] for r in res])))
    print("  每局 value_target 局内标准差: 均值 %.4f" % np.mean([r["vt_std"] for r in res]))


if __name__ == "__main__":
    main()
