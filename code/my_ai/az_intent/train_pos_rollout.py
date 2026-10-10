"""路线 A：用**安静 rollout 教师**数据训练 `pos_state`（自写训练循环）。

为什么自写：`train_value_prior.py` 在**全量数据**上出现"train/val CE 逐位冻结、从不保存 ckpt"的现象
（2026-10-11 实测；同一 ckpt + 同一 npz 的手工循环却能学下去），原因未定位。
本脚本用等价但可观测的循环（每 epoch 打印 **参数变化范数**）绕开它，先把候选训出来。

口径（与 `train_value_prior.py` 一致，便于与该线的历史读数对比）：
  逐行软 CE；两侧都在**该行的合法格**上做 z-score：
    目标  p = softmax( z(adv)        / target_tau )
    预测  q = softmax( z(action_map[cls]) / t_pos )
  只训 `pos_state`；`class_state`/`value_state` 逐位不动，用 `az_train.save_three_net` 写成三网 ckpt
  ⇒ 与 incumbent 构成**单变量**候选。
坐标口径：npz 的 `cell` 是 **(row, col) = (y, x)**（与历史 npz / `train_value_prior.py` 的
`am[..., xy[0], xy[1]]` 一致；写成 (x,y) 会转置 ⇒ 目标变噪声）。

用法：
  python -u code/my_ai/az_intent/train_pos_rollout.py \
    --ckpt checkpoints/grafted_sdk_A.pt --data training_history/posroll/pilot30_yx.npz \
    --out checkpoints/grafted_sdk_A_posroll_s1.pt --epochs 12 --batch-size 256 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-tau", type=float, default=1.0)
    ap.add_argument("--t-pos", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.15, help="**按行块**切（行按局顺序排列 ⇒ 近似按局切）")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 行（0=全部；冒烟用）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.az_train import save_three_net

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    rng = np.random.default_rng(args.seed)

    z = np.load(args.data, allow_pickle=True)
    board = np.asarray(z["board"], dtype=np.float16)
    stats = np.asarray(z["stats"], dtype=np.float16)
    cls = np.asarray(z["cls"], dtype=np.int64)
    cell = np.asarray(z["cell"], dtype=np.int64)
    adv = np.asarray(z["adv"], dtype=np.float32)
    cnt = np.asarray(z["cnt"], dtype=np.int64) if "cnt" in z else np.full(len(cls), cell.shape[1], np.int64)
    if args.limit:
        board, stats, cls, cell, adv, cnt = (board[:args.limit], stats[:args.limit], cls[:args.limit],
                                             cell[:args.limit], adv[:args.limit], cnt[:args.limit])
    N, C = adv.shape
    n_val = max(int(N * args.val_frac), 2)
    val_idx = np.arange(N - n_val, N)          # **行块**切（不随机 ⇒ 不泄漏到训练块）
    tr_idx = np.arange(0, N - n_val)
    print(f"[posroll-train] data={args.data} N={N} C={C} 训练块={len(tr_idx)} 验证块={len(val_idx)} "
          f"device={dev} epochs={args.epochs} bs={args.batch_size} lr={args.lr}", flush=True)
    print(f"[posroll-train] cls 分布 = {np.bincount(cls, minlength=24).tolist()[:6]}…；"
          f"平局行占比 = {float((np.abs(adv[:, 0] - adv[:, 1]) < 1e-9).mean()) * 100:.1f}%", flush=True)

    class_model, pos_model, value_model = load_three_models(args.ckpt)
    for m in (class_model, value_model):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        m.to(dev)
    for p in pos_model.parameters():
        p.requires_grad = True
    pos_model.train().to(dev)
    n_par = sum(p.numel() for p in pos_model.parameters())
    print(f"[posroll-train] pos_model 参数 = {n_par}", flush=True)

    opt = torch.optim.Adam([p for p in pos_model.parameters() if p.requires_grad], lr=args.lr)
    init = [p.detach().clone() for p in pos_model.parameters()]

    B = torch.from_numpy(board)
    S = torch.from_numpy(stats)
    CL = torch.from_numpy(cls)
    CE = torch.from_numpy(cell)
    AA = torch.from_numpy(adv)
    CN = torch.from_numpy(cnt)
    CMAX = int(CN.max()) if len(CN) else 1
    VALID = torch.arange(CMAX).unsqueeze(0) < CN.unsqueeze(1)

    def run(idx: np.ndarray, train: bool) -> tuple[float, float]:
        tot_ce = tot_hit = tot_n = 0.0
        order = rng.permutation(len(idx)) if train else np.arange(len(idx))
        for i in range(0, len(idx), args.batch_size):
            b = idx[order[i:i + args.batch_size]]
            bt = torch.from_numpy(np.ascontiguousarray(b))
            bb = B[bt].to(dev).float()
            ss = S[bt].to(dev).float()
            cc = CL[bt].to(dev)
            xy = CE[bt].to(dev)
            aa = AA[bt].to(dev)
            vv = VALID[bt].to(dev)
            with torch.set_grad_enabled(train):
                out = pos_model(bb, ss)
                am = out["action_map"]
                rows = torch.arange(len(bt), device=dev).unsqueeze(1)
                vals = am[rows, cc.unsqueeze(1), xy[..., 0], xy[..., 1]]
                n = vv.sum(dim=1, keepdim=True).clamp(min=2).float()
                zero = torch.zeros_like(vals)
                mv = torch.where(vv, vals, zero).sum(1, keepdim=True) / n
                ma = torch.where(vv, aa, zero).sum(1, keepdim=True) / n
                sv = torch.sqrt(torch.where(vv, (vals - mv) ** 2, zero).sum(1, keepdim=True) / n).clamp(min=1e-8)
                sa = torch.sqrt(torch.where(vv, (aa - ma) ** 2, zero).sum(1, keepdim=True) / n).clamp(min=1e-8)
                NEG = -1e9
                logq = torch.log_softmax(torch.where(vv, (vals - mv) / sv / args.t_pos,
                                                     torch.full_like(vals, NEG)), dim=1)
                p = torch.softmax(torch.where(vv, (aa - ma) / sa / args.target_tau,
                                              torch.full_like(aa, NEG)), dim=1) * vv
                p = p / p.sum(1, keepdim=True).clamp(min=1e-12)
                ce = -(p * logq).sum(1)
                ok = (vv.sum(1) > 1).float()
                loss = (ce * ok).sum() / ok.sum().clamp(min=1e-9)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                tot_ce += float(ce[ok.bool()].sum())
                tot_hit += float(((logq.argmax(1) == p.argmax(1)) & ok.bool()).sum())
                tot_n += float(ok.sum())
        return tot_ce / max(tot_n, 1), tot_hit / max(tot_n, 1)

    base_ce, base_hit = run(val_idx, False)
    print(f"[posroll-train] 训练前 val: CE={base_ce:.4f} top1={base_hit*100:.2f}%", flush=True)
    best = base_ce
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_ce, tr_hit = run(tr_idx, True)
        va_ce, va_hit = run(val_idx, False)
        dnorm = float(sum((a - b).pow(2).sum() for a, b in zip(pos_model.parameters(), init)) ** 0.5)
        flag = ""
        if va_ce < best:
            best = va_ce
            save_three_net(args.out, class_model, pos_model, value_model,
                           {"num_heads": 3, "posroll_epoch": ep, "posroll_data": args.data,
                            "posroll_target_tau": args.target_tau, "posroll_tpos": args.t_pos})
            flag = "  <- saved"
        print(f"[posroll-train] epoch {ep:2d}/{args.epochs}  train CE={tr_ce:.4f} top1={tr_hit*100:.2f}% | "
              f"val CE={va_ce:.4f} top1={va_hit*100:.2f}% | **参数变化范数={dnorm:.4f}**  ({time.time()-t0:.0f}s){flag}",
              flush=True)
    print(f"[posroll-train] 结束：best val CE={best:.4f} ⇒ {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
