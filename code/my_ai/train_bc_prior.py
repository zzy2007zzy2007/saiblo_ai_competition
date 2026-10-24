"""以"搜索选中动作"为标签训练位置先验 + 离线 recall@k（2026-10-24，v2 小批量）。

v1 的问题：
  · 训练循环按"整局"做梯度步 ⇒ 6 epoch 只有 168 步，几乎没训练；
  · `nb += 0` 让 loss 显示失真。
v2：把所有样本摊平成数组，按小批量（默认 64）训练，正确统计平均 loss。

指标口径（重要）：recall 在**同类合法格**（19×19 内该类 mask）上计算，
  对应"砍 k 时会不会丢掉最优格"，与早期"300 混合候选 vs 值网排序"不是同一个指标。
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn.functional as F

from my_ai.az_intent.az_selfplay import load_three_models

CKPT = REPO / "checkpoints/posnet_A_k5_m32.pt"
GROUP = {11: "build", 13: "downgrade", 21: "lightning"}
KS = (1, 4, 24)


def channel_scores(amap: torch.Tensor, op_type: int) -> torch.Tensor:
    if op_type == 11:
        return amap[0:16].mean(0)
    if op_type == 13:
        return amap[16]
    if op_type == 21:
        return amap[17]
    return amap[23]


def load_split(files):
    boards, stats, labels, pmasks = [], [], [], []
    for f in files:
        d = np.load(f)
        boards.append(d["board"]); stats.append(d["stats"])
        labels.append(d["label"]); pmasks.append(d["pmask"])
    return (np.concatenate(boards), np.concatenate(stats),
            np.concatenate(labels), np.concatenate(pmasks))


def evaluate(model, arrays, tag: str):
    boards, stats, labels, pmasks = arrays
    b = torch.from_numpy(boards.astype(np.float32))
    s = torch.from_numpy(stats.astype(np.float32))
    with torch.no_grad():
        out = model(b, s)["action_map"]
    agg = {k: [0, 0] for k in KS}
    per = {g: {k: [0, 0] for k in KS} for g in GROUP.values()}
    n_used = 0
    for i in range(len(labels)):
        t, x, y = int(labels[i][0]), int(labels[i][1]), int(labels[i][2])
        g = GROUP.get(t)
        if g is None:
            continue
        m = pmasks[i].astype(bool)
        if not (0 <= x < 19 and 0 <= y < 19) or not m[x, y]:
            continue
        sc = channel_scores(out[i], t).cpu().numpy()
        flat = np.where(m, sc, -1e9).reshape(-1)
        target = x * 19 + y
        pos = int(np.where(np.argsort(-flat) == target)[0][0]) + 1
        n_used += 1
        for k in KS:
            agg[k][0] += int(pos <= k); agg[k][1] += 1
            per[g][k][0] += int(pos <= k); per[g][k][1] += 1
    print(f"  --- {tag}（{n_used} 个样本）")
    for k in KS:
        hit, n = agg[k]
        print(f"    recall@{k:<3} = {hit}/{n} = {100*hit/max(1,n):5.1f}%")
    for g, dd in per.items():
        n = dd[1][1]
        if n:
            print(f"    [{g:<10}] n={n:<5} @1={100*dd[1][0]/n:5.1f}%  "
                  f"@4={100*dd[4][0]/n:5.1f}%  @24={100*dd[24][0]/n:5.1f}%")
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/bc_prior.pt")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "training_history/bc_dataset/seed*.npz")))
    if len(files) < 6:
        print(f"  ✗ 数据不足（{len(files)} 局）"); return 1
    train_arr = load_split(files[:-args.holdout])
    hold_arr = load_split(files[-args.holdout:])
    print(f"  训练局 {len(files)-args.holdout} / 留出局 {args.holdout} "
          f"| 训练样本 {len(train_arr[2])}", flush=True)

    _, pos_m, _ = load_three_models(str(CKPT))
    pos_m.eval()
    evaluate(pos_m, hold_arr, "原网 posnet_A（基线）")

    for name, p in pos_m.named_parameters():
        p.requires_grad = "action_map_conv" in name
    n_par = sum(p.numel() for p in pos_m.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in pos_m.parameters() if p.requires_grad], lr=args.lr)
    print(f"  可训练参数 {n_par:,}", flush=True)

    boards, stats, labels, pmasks = train_arr
    # 预筛可用样本（有位置标签且合法）
    usable = []
    for i in range(len(labels)):
        t, x, y = int(labels[i][0]), int(labels[i][1]), int(labels[i][2])
        if t not in GROUP:
            continue
        if not (0 <= x < 19 and 0 <= y < 19) or not pmasks[i][x, y]:
            continue
        usable.append(i)
    usable = np.asarray(usable)
    print(f"  可用训练样本 {len(usable)}", flush=True)

    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(len(usable))
        tot, nb = 0.0, 0
        for start in range(0, len(perm), args.batch):
            idx = usable[perm[start:start + args.batch]]
            b = torch.from_numpy(boards[idx].astype(np.float32))
            s = torch.from_numpy(stats[idx].astype(np.float32))
            out = pos_m(b, s)["action_map"]
            losses = []
            for j, i in enumerate(idx):
                t, x, y = int(labels[i][0]), int(labels[i][1]), int(labels[i][2])
                m = torch.from_numpy(pmasks[i].astype(bool))
                sc = channel_scores(out[j], t)
                logits = sc.masked_fill(~m, -1e9).reshape(-1)
                losses.append(F.cross_entropy(logits.unsqueeze(0),
                                              torch.tensor([int(x) * 19 + int(y)])))
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch {ep}/{args.epochs}  loss={tot/max(1,nb):.4f}", flush=True)

    print("\n  训练后:", flush=True)
    evaluate(pos_m, hold_arr, "BC 训练版")

    ckpt = torch.load(CKPT, map_location="cpu")
    ckpt["pos_state"] = pos_m.state_dict()
    ckpt["bc_note"] = f"action_map_conv BC on search-chosen action, epochs={args.epochs}, batch={args.batch}"
    torch.save(ckpt, REPO / args.out)
    print(f"  已写出 {REPO/args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
