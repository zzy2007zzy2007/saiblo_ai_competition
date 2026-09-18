"""比不同 target_tau 训出来的位置网：**只差一个尺度，还是真的不一样？**

分析预测（docs/az_value_prior_plan.md §0.2）τ 近似重参数化：因为采样侧解码器会重新
z-score，τ 与地图的绝对尺度一起被约掉 ⇒ 三档应当**形状几乎相同、只差尺度**。

这个检验比对局胜率灵敏得多：不掺对局噪声，直接看
  ① 原始地图的尺度 (std)          —— 预期**不同**
  ② 单位方差归一化后的地图 余弦    —— 预期 **≈ 1.000**
  ③ t_pos=1.0 下抽出的分布 余弦    —— 预期 **≈ 1.000**
  ④ 该分布的 top1 落在同一个格的比例 —— 预期 **≈ 100%**

用法: python _tmp_vprior_map_compare.py <ckptA> <ckptB> [<ckptC> ...] [--n 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data", default="training_history/vprior/vp_400.npz")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--t-pos", type=float, default=1.0)
    a = ap.parse_args()
    torch.set_num_threads(4)

    from my_ai.az_intent.az_selfplay import load_three_models

    d = np.load(a.data, allow_pickle=True)
    idx = np.arange(0, len(d["cnt"]), max(len(d["cnt"]) // a.n, 1))[:a.n]
    board = torch.from_numpy(np.asarray(d["board"][idx], dtype=np.float32))
    stats = torch.from_numpy(np.asarray(d["stats"][idx], dtype=np.float32))
    cls = np.asarray(d["cls"][idx], dtype=np.int64)
    cell = np.asarray(d["cell"][idx], dtype=np.int64)
    cnt = np.asarray(d["cnt"][idx], dtype=np.int64)
    MAXC = cell.shape[1]
    valid = np.arange(MAXC)[None, :] < cnt[:, None]

    maps, dists, stds = [], [], []
    for ck in a.ckpts:
        _, pos_model, _ = load_three_models(ck)
        pos_model.eval()
        vals = np.zeros((len(idx), MAXC), dtype=np.float64)
        for i in range(0, len(idx), 64):
            b = board[i:i + 64]
            with torch.no_grad():
                am = pos_model(b, stats[i:i + 64])["action_map"].numpy()
            for j in range(len(b)):
                r = i + j
                for kk in range(cnt[r]):
                    x, y = cell[r, kk]
                    vals[r, kk] = am[j, cls[r], x, y]
        m = np.where(valid, vals, 0.0)
        mu = m.sum(1, keepdims=True) / np.maximum(cnt[:, None], 1)
        sd = np.sqrt(np.where(valid, (vals - mu) ** 2, 0.0).sum(1, keepdims=True)
                     / np.maximum(cnt[:, None], 1)).clip(min=1e-8)
        z = (vals - mu) / sd
        lg = np.where(valid, z / a.t_pos, -1e9)
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p = p / p.sum(1, keepdims=True)
        maps.append(z); dists.append(p); stds.append(float(sd.mean()))
        print("[cmp] %-42s 原始地图 std 均值 = %.4f   分布有效候选 = %.1f"
              % (Path(ck).name, stds[-1], float(np.exp(-(p * np.log(p + 1e-12)).sum(1)).mean())),
              flush=True)

    print("\n%-46s %10s %10s %10s" % ("两两比较", "归一化地图", "分布余弦", "top1同格"))
    for i in range(len(a.ckpts)):
        for j in range(i + 1, len(a.ckpts)):
            def cos(X, Y):
                num = np.where(valid, X * Y, 0.0).sum(1)
                den = np.sqrt(np.where(valid, X * X, 0.0).sum(1)
                              * np.where(valid, Y * Y, 0.0).sum(1)).clip(min=1e-12)
                return float((num / den).mean())
            t1 = float((dists[i].argmax(1) == dists[j].argmax(1)).mean())
            print("%-46s %10.4f %10.4f %9.0f%%"
                  % ("%s vs %s" % (Path(a.ckpts[i]).stem, Path(a.ckpts[j]).stem),
                     cos(maps[i], maps[j]), cos(dists[i], dists[j]), 100 * t1), flush=True)


if __name__ == "__main__":
    main()
