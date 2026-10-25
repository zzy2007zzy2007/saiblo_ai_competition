"""z-score 相关性损失（2026-10-25）。

用户提议：把网络输出与 rollout 分数**各自在局面内 z-score**，再让网络去学。
数学上等价于 **最大化每局面内的 Pearson 相关**：

    MSE(z_pred, z_tgt) = 2 * (1 - r)

相比旧版 softmax CE(T=0.5)：
  · 不会被指数放大成 one-hot 目标；
  · 预测/目标都标准化 ⇒ 梯度量级正常；
  · 每个动作组各自标准化 ⇒ 各组损失可比。

支持 `--overfit-states N --steps S` 做可学性检验（小样本过拟合）。
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

from my_ai.az_intent.az_selfplay import load_three_models

OP2CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20], 31: [21], 32: [22]}
NAME = {11: "build", 13: "downgrade", 21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}


def load_games(files):
    out = []
    for f in files:
        d = np.load(f)
        feats, stt = d["feat"], d["stats"]
        cls, xs, ys, vals, off = d["cand_cls"], d["cand_x"], d["cand_y"], d["cand_val"], d["cand_off"]
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
    return amap[0:16].mean(0) if op == 11 else amap[OP2CH[op][0]]


def zscore(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (t - t.mean()) / (t.std(unbiased=False) + eps)


def groups_loss(out_row: torch.Tensor, targets: dict, clip: float, min_cells: int) -> list:
    losses = []
    for op, cells in targets.items():
        keys = list(cells.keys())
        if len(keys) < min_cells:
            continue
        v = torch.tensor([cells[k] for k in keys])
        if float(v.std()) < 1e-9:
            continue
        sc = chan_pred(out_row, op)
        xi = torch.tensor([k[0] for k in keys]); yi = torch.tensor([k[1] for k in keys])
        pred = sc[xi, yi]
        zv = zscore(v).clamp(-clip, clip)
        losses.append(((zscore(pred) - zv) ** 2).mean())
    return losses


def recall_eval(model, games, ks=(1, 4, 24)):
    agg = {k: [0, 0] for k in ks}; per = {}
    for feat, stt, targets in games:
        with torch.no_grad():
            amap = model(torch.from_numpy(feat).unsqueeze(0),
                         torch.from_numpy(stt).unsqueeze(0))["action_map"][0]
        for op, cells in targets.items():
            keys = list(cells.keys())
            if len(keys) < 3:
                continue
            sc = chan_pred(amap, op).numpy()
            v = np.array([cells[k] for k in keys])
            s = np.array([sc[x, y] for (x, y) in keys])
            pos = int(np.where(np.argsort(-s) == int(np.argmax(v)))[0][0]) + 1
            d = per.setdefault(NAME.get(op, str(op)), {k: [0, 0] for k in ks})
            for k in ks:
                agg[k][0] += int(pos <= k); agg[k][1] += 1
                d[k][0] += int(pos <= k); d[k][1] += 1
    return agg, per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="g93*.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--unfreeze", default="trunk", choices=["head", "trunk"])
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--overfit-states", type=int, default=0, help=">0 时只取 N 个局面做过拟合检验")
    ap.add_argument("--steps", type=int, default=0, help="过拟合检验的总步数（覆盖 epochs）")
    ap.add_argument("--out", default="checkpoints/prior_z.pt")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "training_history/prior_labels" / args.pattern)))
    train = load_games(files[:-args.holdout]); hold = load_games(files[-args.holdout:])
    print(f"  训练局 {len(files)-args.holdout} / 局面 {len(train)}；留出 {len(hold)}", flush=True)
    if args.overfit_states:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(train), min(args.overfit_states, len(train)), replace=False)
        train = [train[i] for i in idx]
        hold = train
        print(f"  ⚠️ 过拟合检验模式：只用 {len(train)} 个局面（训练=留出）", flush=True)

    _, m, _ = load_three_models(str(REPO / "checkpoints/posnet_A_k5_m32.pt"))
    m.eval()
    a0, _ = recall_eval(m, hold)
    print("  训练前:" + "  ".join(f"@{k}={100*a0[k][0]/max(1,a0[k][1]):.1f}%" for k in (1, 4, 24)), flush=True)
    for name, p in m.named_parameters():
        p.requires_grad = ("value_head" not in name) if args.unfreeze == "trunk" else ("action_map_conv" in name)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    print(f"  可训练参数 {sum(p.numel() for p in m.parameters() if p.requires_grad):,}", flush=True)

    n_ep = args.epochs
    step = 0
    for ep in range(1, n_ep + 1):
        idx = np.random.permutation(len(train))
        tot, nb = 0.0, 0
        for s in range(0, len(idx), args.batch):
            chunk = [train[i] for i in idx[s:s + args.batch]]
            out = m(torch.from_numpy(np.stack([c[0] for c in chunk])),
                    torch.from_numpy(np.stack([c[1] for c in chunk])))["action_map"]
            losses = []
            for j, (_, _, targets) in enumerate(chunk):
                losses += groups_loss(out[j], targets, args.clip, args.min_cells)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            gn = float(sum(float(p.grad.norm()) ** 2 for p in m.parameters()
                           if p.grad is not None) ** 0.5)
            tot += float(loss); nb += 1; step += 1
            if args.overfit_states and step % 25 == 0:
                print(f"    step {step}: loss={float(loss):.4f}  grad_norm={gn:.3e}", flush=True)
            if args.steps and step >= args.steps:
                break
        print(f"  epoch {ep}/{n_ep} loss={tot/max(1,nb):.4f}  grad_norm={gn:.3e}", flush=True)
        if args.steps and step >= args.steps:
            break

    a1, p1 = recall_eval(m, hold)
    print("  训练后:" + "  ".join(f"@{k}={100*a1[k][0]/max(1,a1[k][1]):.1f}%" for k in (1, 4, 24)), flush=True)
    for g, d in p1.items():
        n = d[24][1]
        if n:
            print(f"    {g:<11} n={n:<6} @1={100*d[1][0]/n:5.1f}%  @4={100*d[4][0]/n:5.1f}%  @24={100*d[24][0]/n:5.1f}%")
    if not args.overfit_states:
        ck = torch.load(REPO / "checkpoints/posnet_A_k5_m32.pt", map_location="cpu")
        ck["pos_state"] = m.state_dict()
        ck["note"] = "z-score 相关性损失（Pearson），建塔 0-15 广播"
        torch.save(ck, REPO / args.out)
        print(f"  已写出 {REPO/args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
