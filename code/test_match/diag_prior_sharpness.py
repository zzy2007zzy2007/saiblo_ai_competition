"""量化先验"锐度"：合法格上的归一化熵 + top-k 概率质量（2026-10-25）。

假说：#9 的收益依赖 f10 的"尖锐"分布（候选按先验概率抽样 ⇒ 尖 ≈ 高效枚举 top-k），
     而离线 recall@k 测不出这一点（f10 94.4% vs bc_prior 93.9%，几乎相同）。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import load_three_models

NETS = [
    ("f10 (#9 用的)", "checkpoints/prior_distill_f10.pt"),
    ("bc_prior", "checkpoints/bc_prior.pt"),
    ("posnet_A (冠军)", "checkpoints/posnet_A_k5_m32.pt"),
]


def main() -> int:
    files = sorted(glob.glob(str(REPO / "training_history/bc_dataset/seed*.npz")))[-4:]
    boards = np.concatenate([np.load(f)["board"] for f in files]).astype(np.float32)
    stats = np.concatenate([np.load(f)["stats"] for f in files]).astype(np.float32)
    pms = np.concatenate([np.load(f)["pmask"] for f in files])
    print(f"  样本 {len(boards)}（来自 {len(files)} 局）", flush=True)
    for tag, ck in NETS:
        _, m, _ = load_three_models(str(REPO / ck))
        m.eval()
        ents, m6, m12 = [], [], []
        with torch.no_grad():
            for i in range(0, len(boards), 64):
                b = torch.from_numpy(boards[i:i + 64])
                s = torch.from_numpy(stats[i:i + 64])
                amap = m(b, s)["action_map"].cpu().numpy()
                for j in range(len(amap)):
                    mask = pms[i + j].astype(bool)
                    if mask.sum() < 4:
                        continue
                    sc = amap[j][0:16].mean(0)
                    v = np.where(mask, sc, -np.inf).reshape(-1)
                    v = v[np.isfinite(v)]
                    if v.size < 4:
                        continue
                    e = np.exp(v - v.max()); p = e / e.sum()
                    ps = np.sort(p)[::-1]
                    ents.append(float(-(p * np.log(np.clip(p, 1e-12, None))).sum()) / np.log(len(p)))
                    m6.append(float(ps[:6].sum())); m12.append(float(ps[:12].sum()))
        print(f"  {tag:<18} 归一化熵 {np.mean(ents):.3f}   top6质量 {np.mean(m6):.3f}   top12质量 {np.mean(m12):.3f}   n={len(ents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
