"""看一眼 `collect_value_prior.py` 产出的 npz 结构（用于判断"能不能拿它训类头"）。

用法: D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/inspect_vprior_npz.py [npz 路径]

关键结论（`training_history/vprior/vp_A_k5_m32.npz`，2026-10-05 实读）：
  * 350,793 行；每行 = 一个 (决策, head, 类) 的候选格列表：
    `cell[32,2]` + `adv[32]`（对每个候选格做 1-ply 价值评估得到） + `cnt` + `weight`；
  * **行可以按"相邻行 board/player 相同"分组回决策**：相邻行 board 相同的比例 91.7%、
    连续相同段数 28,989 ⇒ 平均 12.1 行/决策（3 个 head × 每 head 若干个被采样的类）；
  * `*_meta` 记录了算 `adv` 用的价值网（`ckpt_meta`）、K/M/λ —— 换价值网必须重采。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "training_history" / "vprior" / "vp_A_k5_m32.npz"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    z = np.load(path, allow_pickle=True)
    print(f"file: {path}")
    print("keys:", list(z.keys()))
    for k in z.keys():
        a = z[k]
        print("  %-12s shape=%-18s dtype=%s" % (k, getattr(a, "shape", None), a.dtype))
    print()
    for k in ("head", "cls", "cnt", "weight", "player"):
        if k in z:
            print("%-8s[:16] = %s" % (k, z[k][:16]))
    for k in z.keys():
        if "meta" in k:
            v = z[k]
            print("meta %-10s = %s" % (k, v if v.size <= 8 else v.ravel()[:8]))
    print()
    n = len(z["adv"])
    board = z["board"]
    same = np.all(board[1:] == board[:-1], axis=tuple(range(1, board.ndim)))
    seg = 1 + int((~same).sum())
    print("rows = %d" % n)
    print("adjacent-same-board ratio = %.3f" % same.mean())
    print("runs of identical board = %d  -> avg rows/decision = %.2f" % (seg, n / seg))
    # 每个决策里 head 的分布（抽样前 5000 行）
    import collections
    hc = collections.Counter(z["head"][:5000].tolist())
    print("head histogram (first 5000 rows) =", dict(sorted(hc.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
