"""v2 训练器：修 op_type→通道 的映射错误 + 建塔通道合并（2026-10-25）。

修的两个问题（见 docs/prior_repair_20261023.md §三十一）：
  ① 旧版把 `op_type` 当成"塔型通道号"：11(BUILD)→通道11、13(DOWNGRADE)→通道13，
     而 21-24（闪电/EMP/护盾/回避）因 `c>15` 被**整类丢弃**。
  ② 建塔的塔型在标签里不存在 ⇒ 正确做法是把建塔标签**广播到 0-15 全部通道**
     （等价于"只保留一个建塔位置图"），损失直接作用在 `amap[0:16].mean(0)` 上。

映射：11→0..15（建塔） 13→16（降级） 21→17（闪电） 22→18 23→19 24→20 31→21 32→22
支持集：用该局面**该动作类出现的候选格**做 softmax 支撑（先验只需在候选集内排序）。
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

OP2CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20], 31: [21], 32: [22]}
NAME = {11: "build", 13: "downgrade", 21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}


def load_games(files):
    """→ list of (feat[28,19,19], {op: {(x,y): value}})"""
    out = []
    for f in files:
        d = np.load(f)
        feats = d["feat"]; stt = d["stats"]
        cls = d["cand_cls"]; xs = d["cand_x"]; ys = d["cand_y"]
        vals = d["cand_val"]; off = d["cand_off"]
        for i in range(len(off) - 1):
            a, b = int(off[i]), int(off[i + 1])
            per = {}
            for t, x, y, v in zip(cls[a:b], xs[a:b], ys[a:b], vals[a:b]):
                t = int(t)
                if t not in OP2CH:
                    continue
                key = (int(x), int(y))
                m = per.setdefault(t, {})
                m[key] = max(m.get(key, -1e9), float(v))
            if per:
                out.append((feats[i].astype(np.float32), stt[i].astype(np.float32), per))
    return out


def chan_pred(amap: torch.Tensor, op: int) -> torch.Tensor:
    if op == 11:
        return amap[0:16].mean(0)          # 建塔：16 通道均值 ⇒ 广播监督
    return amap[OP2CH[op][0]]


def recall_stats(model, games, ks=(1, 4, 24)):
    agg = {k: [0, 0] for k in ks}
    per = {}
    for feat, stt, targets in games:
        with torch.no_grad():
            amap = model(torch.from_numpy(feat).unsqueeze(0),
                         torch.from_numpy(stt).unsqueeze(0))["action_map"][0]
        for op, cells in targets.items():
            sc = chan_pred(amap, op).numpy()
            keys = list(cells.keys()); v = np.array([cells[c] for c in keys])
            s = np.array([sc[x, y] for (x, y) in keys])
            order = np.argsort(-s)
            pos = int(np.where(order == int(np.argmax(v)))[0][0]) + 1
            d = per.setdefault(NAME.get(op, str(op)), {k: [0, 0] for k in ks})
            for k in ks:
                agg[k][0] += int(pos <= k); agg[k][1] += 1
                d[k][0] += int(pos <= k); d[k][1] += 1
    return agg, per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="g93*.npz")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--unfreeze", default="head", choices=["head", "trunk"])
    ap.add_argument("--out", default="checkpoints/prior_v2.pt")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "training_history/prior_labels" / args.pattern)))
    if len(files) < 6:
        print(f"  ✗ 数据不足 {len(files)} 局"); return 1
    train = load_games(files[:-args.holdout]); hold = load_games(files[-args.holdout:])
    print(f"  训练 {len(files)-args.holdout} 局 / {len(train)} 局面；留出 {len(hold)} 局面", flush=True)

    _, m, _ = load_three_models(str(REPO / "checkpoints/posnet_A_k5_m32.pt"))
    m.eval()
    a0, p0 = recall_stats(m, hold)
    print("  训练前:" + "  ".join(f"@{k}={100*a0[k][0]/max(1,a0[k][1]):.1f}%" for k in (1, 4, 24)), flush=True)
    for name, p in m.named_parameters():
        p.requires_grad = ("value_head" not in name) if args.unfreeze == "trunk" else ("action_map_conv" in name)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  可训练参数 {n_par:,}", flush=True)

    T = 0.5
    for ep in range(1, args.epochs + 1):
        idx = np.random.permutation(len(train))
        tot, nb = 0.0, 0
        for s in range(0, len(idx), args.batch):
            chunk = [train[i] for i in idx[s:s + args.batch]]
            b = torch.from_numpy(np.stack([c[0] for c in chunk]))
            sb = torch.from_numpy(np.stack([c[1] for c in chunk]))
            out = m(b, sb)["action_map"]
            losses = []
            for j, (_, _, targets) in enumerate(chunk):
                for op, cells in targets.items():
                    keys = list(cells.keys())
                    if len(keys) < 2:
                        continue
                    v = torch.tensor([cells[c] for c in keys])
                    w = torch.softmax(v / T, dim=0)
                    sc = chan_pred(out[j], op)
                    xi = torch.tensor([k[0] for k in keys]); yi = torch.tensor([k[1] for k in keys])
                    logits = sc[xi, yi]          # 用张量索引，保持梯度
                    losses.append(-(w * torch.log_softmax(logits, dim=0)).sum())
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch {ep}/{args.epochs} loss={tot/max(1,nb):.4f}", flush=True)

    a1, p1 = recall_stats(m, hold)
    print("  训练后:" + "  ".join(f"@{k}={100*a1[k][0]/max(1,a1[k][1]):.1f}%" for k in (1, 4, 24)), flush=True)
    print("  分组（训练后）:")
    for g, d in p1.items():
        n = d[1][1]
        if n:
            print(f"    {g:<11} n={n:<6} @1={100*d[1][0]/n:5.1f}%  @4={100*d[4][0]/n:5.1f}%  @24={100*d[24][0]/n:5.1f}%")
    ck = torch.load(REPO / "checkpoints/posnet_A_k5_m32.pt", map_location="cpu")
    ck["pos_state"] = m.state_dict()
    ck["note_v2"] = "op_type→通道 映射修正 + 建塔0-15广播"
    torch.save(ck, REPO / args.out)
    print(f"  已写出 {REPO/args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
