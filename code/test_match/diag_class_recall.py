"""类级 recall：先验把"正确的类"排在第几（2026-10-24）。

背景（上一轮结论）：同类合法格口径下原网 recall@24 已 94%，BC 训练无增益
⇒ 瓶颈不在类内位置，而在**类级候选**。本脚本量：
  按类先验（`head_logits`）给合法类排序，"搜索选中的动作所属的类组"是否在前 k。
类组定义（dump 的 op_type 不含塔型）：
  · build(11)  → 类 0..15 任一（塔型未知）
  · downgrade(13) → 类 16
  · lightning(21) → 类 17
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
KS = (1, 2, 4, 8)


def main() -> int:
    files = sorted(glob.glob(str(REPO / "training_history/bc_dataset/seed*.npz")))
    hold = files[-4:]
    print(f"  留出局 {len(hold)}", flush=True)
    _, pos_m, _ = load_three_models(str(CKPT))
    pos_m.eval()
    boards = np.concatenate([np.load(f)["board"] for f in hold])
    stats = np.concatenate([np.load(f)["stats"] for f in hold])
    labels = np.concatenate([np.load(f)["label"] for f in hold])
    cms = np.concatenate([np.load(f)["cmask"] for f in hold])

    b = torch.from_numpy(boards.astype(np.float32))
    s = torch.from_numpy(stats.astype(np.float32))
    with torch.no_grad():
        out = pos_m(b, s)
    hl = out.get("head_logits")
    print(f"  head_logits 类型: {type(hl)}", flush=True)
    if isinstance(hl, (list, tuple)):
        print(f"    长度 {len(hl)}，元素 shape {[[tuple(t.shape) for t in x] if isinstance(x, (list, tuple)) else tuple(x.shape) for x in hl[:1]]}", flush=True)
        hl = hl[0] if isinstance(hl[0], torch.Tensor) else hl[0][0]
    if isinstance(hl, torch.Tensor):
        print(f"    tensor shape {tuple(hl.shape)}", flush=True)
    else:
        print("  ✗ head_logits 不可用，改用 action_map 的类均值作为类先验", flush=True)
        hl = out["action_map"].mean(dim=(2, 3))

    agg = {k: [0, 0] for k in KS}
    per = {g: {k: [0, 0] for k in KS} for g in ("build", "downgrade", "lightning")}
    names = {11: "build", 13: "downgrade", 21: "lightning"}
    used = 0
    for i in range(len(labels)):
        t = int(labels[i][0])
        cls_group = GROUP.get(t)
        if cls_group is None:
            continue
        cm = cms[i].astype(bool)
        sc = hl[i].detach().cpu().numpy() if isinstance(hl, torch.Tensor) else np.asarray(hl[i])
        sc = np.where(cm, sc, -1e9)
        order = list(np.argsort(-sc))
        best = min((order.index(c) + 1) for c in cls_group if c in order) if any(c in order for c in cls_group) else 10 ** 6
        used += 1
        g = names[t]
        for k in KS:
            agg[k][0] += int(best <= k); agg[k][1] += 1
            per[g][k][0] += int(best <= k); per[g][k][1] += 1
    print(f"\n  类级 recall（{used} 个样本）:")
    for k in KS:
        hit, n = agg[k]
        print(f"    class-recall@{k} = {hit}/{n} = {100*hit/max(1,n):5.1f}%")
    for g, dd in per.items():
        n = dd[1][1]
        if n:
            print(f"    [{g:<10}] n={n:<5} @1={100*dd[1][0]/n:5.1f}%  @4={100*dd[4][0]/n:5.1f}%  @8={100*dd[8][0]/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
