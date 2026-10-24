"""以"搜索选中动作"为标签训练位置先验 + 离线 recall@k 评估（2026-10-24）。

数据：`training_history/bc_dataset/seed*.npz`
  board(S,28,19,19) / stats(S,D) / label(S,3)=(op_type,x,y) / cmask(S,24) / pmask(S,19,19)
  · label 的 op_type：11=建塔(位置=格)、13=降级(位置=塔格)、21=闪电(位置=格)
  · 注意：dump 里 op_type=11 **不含塔型** ⇒ 不做类头训练，只训**位置**。

训练：冻结主干与 value_head，只训 `action_map_conv`（塔 0-15 / 降级 16 / 闪电 17 三组通道）。
损失：每组各自的 **masked cross-entropy**（在合法格上做 softmax，标签 = 选中格）。

评估（离线，不打对局）：
  recall@k = 先验排序的前 k 个格里是否包含"搜索选中的格"（按 op_type 分组统计）。
  同时输出原网（未训练）的同一指标作为基线。
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


def channel_scores(amap: torch.Tensor, op_type: int) -> torch.Tensor:
    """取该 op 类型对应的 (19,19) 分数图。"""
    if op_type == 11:
        return amap[0:16].mean(0)
    if op_type == 13:
        return amap[16]
    if op_type == 21:
        return amap[17]
    return amap[23]


def recall_at(scored: np.ndarray, mask: np.ndarray, lab, ks=(1, 4, 24)):
    """scored/mask: (19,19)；lab=(x,y)。返回各 k 是否命中。"""
    s = np.where(mask, scored, -1e9).reshape(-1)
    order = np.argsort(-s)
    target = int(lab[0]) * s.shape[0] // 19 + int(lab[1])  # 扁平索引（19x19 行优先）
    pos = int(np.where(order == target)[0][0]) + 1 if target in order else 10 ** 6
    return {k: pos <= k for k in ks}, pos


def flatten_index(x: int, y: int, w: int = 19) -> int:
    return int(x) * w + int(y)


def evaluate(model, files, tag: str):
    agg = {k: [0, 0] for k in (1, 4, 24)}
    per_group = {g: {k: [0, 0] for k in (1, 4, 24)} for g in GROUP.values()}
    positions = []
    for f in files:
        d = np.load(f)
        boards = torch.from_numpy(d["board"].astype(np.float32))
        stats = torch.from_numpy(d["stats"].astype(np.float32))
        labels = d["label"]
        pmask = d["pmask"]
        with torch.no_grad():
            out = model(boards, stats)["action_map"]
        for i in range(len(labels)):
            t, x, y = int(labels[i][0]), int(labels[i][1]), int(labels[i][2])
            g = GROUP.get(t)
            if g is None:
                continue
            m = pmask[i]
            if not m[x, y]:
                continue
            sc = channel_scores(out[i], t).cpu().numpy()
            s = np.where(m, sc, -1e9).reshape(-1)
            target = flatten_index(x, y)
            order = np.argsort(-s)
            pos = int(np.where(order == target)[0][0]) + 1
            positions.append(pos)
            for k in (1, 4, 24):
                agg[k][0] += int(pos <= k); agg[k][1] += 1
                per_group[g][k][0] += int(pos <= k); per_group[g][k][1] += 1
    print(f"  --- {tag}（{len(positions)} 个带位置标签的样本）")
    for k in (1, 4, 24):
        hit, n = agg[k]
        print(f"    recall@{k:<3} = {hit}/{n} = {100*hit/max(1,n):5.1f}%")
    for g, dd in per_group.items():
        n = dd[1][1]
        if n:
            print(f"    [{g:<10}] n={n:<5} recall@1={100*dd[1][0]/n:5.1f}%  "
                  f"@4={100*dd[4][0]/n:5.1f}%  @24={100*dd[24][0]/n:5.1f}%")
    return positions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/bc_prior.pt")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "training_history/bc_dataset/seed*.npz")))
    if len(files) < 6:
        print(f"  ✗ 数据不足（{len(files)} 局）")
        return 1
    hold = files[-args.holdout:]
    train_files = files[:-args.holdout]
    print(f"  训练局 {len(train_files)} / 留出局 {len(hold)}", flush=True)

    _, pos_m, _ = load_three_models(str(CKPT))
    pos_m.eval()
    print("  基线（未训练）:", flush=True)
    evaluate(pos_m, hold, "原网 posnet_A")

    for name, p in pos_m.named_parameters():
        p.requires_grad = "action_map_conv" in name
    opt = torch.optim.Adam([p for p in pos_m.parameters() if p.requires_grad], lr=args.lr)

    data = []
    for f in train_files:
        d = np.load(f)
        data.append((torch.from_numpy(d["board"].astype(np.float32)),
                     torch.from_numpy(d["stats"].astype(np.float32)),
                     d["label"], d["pmask"]))
    n_total = sum(len(x[2]) for x in data)
    print(f"  训练样本 {n_total}", flush=True)

    for ep in range(1, args.epochs + 1):
        tot, nb = 0.0, 0
        idx = np.random.permutation(len(data))
        for j in idx:
            boards, stats, labels, pmask = data[j]
            out = pos_m(boards, stats)["action_map"]
            loss = torch.zeros(())
            for i in range(len(labels)):
                t, x, y = int(labels[i][0]), int(labels[i][1]), int(labels[i][2])
                if t not in GROUP:
                    continue
                m = torch.from_numpy(pmask[i].astype(bool))
                if not bool(m[x, y]):
                    continue
                sc = channel_scores(out[i], t)
                logits = sc.masked_fill(~m, -1e9).reshape(-1)
                loss = loss + F.cross_entropy(logits.unsqueeze(0),
                                              torch.tensor([flatten_index(x, y)]))
                nb += 1
            if nb == 0:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 0
        print(f"  epoch {ep}/{args.epochs}  loss={tot/max(1,ep*1):.4f}", flush=True)

    print("\n  训练后:", flush=True)
    evaluate(pos_m, hold, "BC 训练版")

    ckpt = torch.load(CKPT, map_location="cpu")
    ckpt["pos_state"] = pos_m.state_dict()
    ckpt["bc_note"] = f"action_map_conv 由'搜索选中动作'BC 训练，epochs={args.epochs}"
    torch.save(ckpt, REPO / args.out)
    print(f"  已写出 {REPO/args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
