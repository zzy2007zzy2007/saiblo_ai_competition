"""离线验证"覆盖率优先"候选生成器（2026-10-24）。

生成器：按类先验取 **top-N 类**，每类再取 **top-M 格**，共 N×M 个候选。
本脚本用 32 局 BC 数据（标签 = 搜索选中的动作）算**覆盖率**：
  该候选集里是否包含"搜索真正选中的那个动作（类组 + 格子）"。
目标：≥85%（对应"砍到 24 个候选也不丢最优动作"）。
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

CKPT = REPO / "checkpoints/posnet_A_k5_m32.pt"
GROUP = {11: list(range(16)), 13: [16], 21: [17]}


def main() -> int:
    files = sorted(glob.glob(str(REPO / "training_history/bc_dataset/seed*.npz")))
    hold = files[-4:]
    boards = np.concatenate([np.load(f)["board"] for f in hold])
    stats = np.concatenate([np.load(f)["stats"] for f in hold])
    labels = np.concatenate([np.load(f)["label"] for f in hold])
    cms = np.concatenate([np.load(f)["cmask"] for f in hold])
    pms = np.concatenate([np.load(f)["pmask"] for f in hold])
    _, pos_m, _ = load_three_models(str(CKPT))
    pos_m.eval()
    with torch.no_grad():
        amap = pos_m(torch.from_numpy(boards.astype(np.float32)),
                     torch.from_numpy(stats.astype(np.float32)))["action_map"].cpu().numpy()
    print(f"  样本 {len(labels)}，amap {amap.shape}", flush=True)

    results = {}
    for N, M in ((4, 6), (4, 12), (8, 3), (8, 8), (2, 12)):
        hit = tot = 0
        for i in range(len(labels)):
            t, x, y = int(labels[i][0]), int(labels[i][1]), int(labels[i][2])
            grp = GROUP.get(t)
            if grp is None:
                continue
            cm = cms[i].astype(bool)
            if not (0 <= x < 19 and 0 <= y < 19):
                continue
            cls_score = amap[i].mean(axis=(1, 2))
            cls_score = np.where(cm, cls_score, -1e9)
            top_cls = list(np.argsort(-cls_score)[:N])
            cand = set()
            for c in top_cls:
                m = pms[i] if c in GROUP[11] else None
                sc = amap[i][c]
                if m is not None:
                    sc = np.where(m, sc, -1e9)
                else:
                    # 非建塔类：用该类自身的可选性（降级=所有己方塔格，闪电=全部格）
                    sc = np.where(cm[c], sc, -1e9)
                flat = np.argsort(-sc.reshape(-1))[:M]
                for f in flat:
                    if sc.reshape(-1)[f] <= -1e8:
                        continue
                    cand.add((c, int(f) // 19, int(f) % 19))
            ok = any((c in grp and cx == x and cy == y) for (c, cx, cy) in cand)
            tot += 1
            hit += int(ok)
        results[(N, M)] = (hit, tot)
        print(f"  N={N} 类 × M={M} 格 = {N*M:>2} 候选 ⇒ 覆盖率 {hit}/{tot} = {100*hit/max(1,tot):5.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
