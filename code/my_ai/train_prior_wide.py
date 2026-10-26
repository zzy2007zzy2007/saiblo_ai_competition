"""加宽/加深位置网的对照训练（2026-10-26）。

复用 train_prior_z.py 的数据管线与 z-score（Pearson）损失，只把模型换成
`AntWarNetwork(latent_dim=W, num_resblocks=B)`（原版是 64/6），并可选 weight decay。

用法：
  python train_prior_wide.py --width 128 --blocks 6 --epochs 20 --wd 1e-4 --tag A_wide128
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

from my_ai.network import AntWarNetwork
from my_ai.train_prior_z import load_games, groups_loss, recall_eval

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="g93*.npz")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--tag", default="wide")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "training_history/prior_labels" / args.pattern)))
    train = load_games(files[:-args.holdout])
    hold = load_games(files[-args.holdout:])
    print(f"  [{args.tag}] 训练局 {len(files)-args.holdout}/局面 {len(train)}；留出 {len(hold)}", flush=True)

    m = AntWarNetwork(num_resblocks=args.blocks, latent_dim=args.width).to(DEV)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"  [{args.tag}] 参数 {n_par:,}（width={args.width} blocks={args.blocks}）", flush=True)

    a0, _ = recall_eval(m, hold)
    print(f"  [{args.tag}] 训练前: " + "  ".join(f"@{k}={100*a0[k][0]/max(1,a0[k][1]):.1f}%" for k in (1, 4, 24)), flush=True)

    opt = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.wd)
    best = {"r24": -1, "ep": -1}
    for ep in range(1, args.epochs + 1):
        m.train()
        idx = np.random.permutation(len(train))
        tot, nb = 0.0, 0
        for s in range(0, len(idx), args.batch):
            chunk = [train[i] for i in idx[s:s + args.batch]]
            out = m(torch.from_numpy(np.stack([c[0] for c in chunk])).to(DEV),
                    torch.from_numpy(np.stack([c[1] for c in chunk])).to(DEV))["action_map"]
            losses = []
            for j, (_, _, targets) in enumerate(chunk):
                losses += groups_loss(out[j], targets, args.clip, args.min_cells)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        m.eval()
        if ep % args.eval_every == 0 or ep == args.epochs:
            a1, _ = recall_eval(m, hold)
            r = {k: 100 * a1[k][0] / max(1, a1[k][1]) for k in (1, 4, 24)}
            flag = ""
            if r[24] > best["r24"]:
                best = {"r24": r[24], "ep": ep}
                torch.save({"pos_state": {k: v.cpu() for k, v in m.state_dict().items()},
                            "note": f"wide width={args.width} blocks={args.blocks} ep{ep} @24={r[24]:.1f}"},
                           REPO / f"checkpoints/wide_{args.tag}.pt")
                flag = "  ← best"
            print(f"  [{args.tag}] ep{ep:>3} loss={tot/max(1,nb):.4f}  "
                  f"@1={r[1]:5.1f}% @4={r[4]:5.1f}% @24={r[24]:5.1f}%{flag}", flush=True)
    print(f"  [{args.tag}] 最佳 @24 = {best['r24']:.1f}% (ep{best['ep']})，已存 checkpoints/wide_{args.tag}.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
