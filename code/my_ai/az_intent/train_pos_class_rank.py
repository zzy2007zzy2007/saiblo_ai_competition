"""用**跨类 rollout 教师**训练 `pos_state`（只改位置头），并输出"类排序一致率"闸门。

动机（第 86 轮，读代码得到）：MC 菜单的类候选来自
`_econ_rank`：`v_c = max over c 的合法格 of AM[c]`，**按 v_c 跨类排序**取前 `MC_TOPK` ⇒
**位置头决定菜单组成**（注释自承"跨类 map 最大值不可比" ⇒ 菜单长期被类 0/16 占据 ⇒ 升级进不了菜单）。
⇒ 本脚本把"per-class 值"训成与**安静 rollout 的类分数同序**。

口径：
- 数据来自 `collect_class_rollout_teacher.py`：每行 = 一个决策，`cls/cell/score/cnt`（候选类 = HOLD + 经济类前 k；
  `cell=(row,col)=(y,x)`，HOLD 行为 `(-1,-1)`）；
- 每行的类值 `v_j = AM[cls_j, y_j, x_j]`（HOLD 行取该决策**未出手**时的整体基线不参与 ⇒ HOLD 用 0 占位并**在秩时按同尺度比较**，
  见下：本实现只用**非 HOLD**候选做 z-score，HOLD 的分数单独报告 ⇒ 避免"空操作没有格子"导致的口径混淆）；
- 目标 `p = softmax(z(score))`，预测 `q = softmax(z(v))`（两边都逐行 z-score ⇒ 尺度无关），损失 = 软 CE；
- 全平的行使 z 无定义 ⇒ 跳过（占比单独打印）。

输出闸门（训练前后各测一次）：**类排序一致率** = 模型 argmax 类 == 教师 argmax 类的比例（非平局行）；
以及**成对一致率**（所有候选对里模型与教师同序的比例）。

用法：
  python -u code/my_ai/az_intent/train_pos_class_rank.py \
    --ckpt checkpoints/grafted_sdk_A.pt --data training_history/posroll/class30.npz \
    --out checkpoints/grafted_sdk_A_clsrank.pt --epochs 8 --batch-size 128 --lr 1e-3
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
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
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
    score = np.asarray(z["score"], dtype=np.float32)
    cnt = np.asarray(z["cnt"], dtype=np.int64)
    if args.limit:
        board, stats, cls, cell, score, cnt = (board[:args.limit], stats[:args.limit], cls[:args.limit],
                                               cell[:args.limit], score[:args.limit], cnt[:args.limit])
    N, C = score.shape
    n_val = max(int(N * args.val_frac), 2)
    # ⚠️ 2026-10-11 实测：按"最后 15% 行"切会切到**局面尾部切片**（终局盘满、非平局行只剩 9 条）⇒ val 读数不可用。
    #    改用**随机行切分**（每行是一个决策、只出现一次 ⇒ 无行级泄漏；仅同局相邻决策有相关性，足够做闸门判读）。
    _perm = rng.permutation(N)
    va = np.sort(_perm[:n_val])
    tr = np.sort(_perm[n_val:])
    print(f"[clsrank] data={args.data} N={N} C={C} 训练={len(tr)} 验证={len(va)} "
          f"每行候选数均值={cnt.mean():.2f}（最大 {int(cnt.max())}）device={dev}", flush=True)

    class_model, pos_model, value_model = load_three_models(args.ckpt)
    for m in (class_model, value_model):
        for p in m.parameters():
            p.requires_grad = False
        m.eval(); m.to(dev)
    for p in pos_model.parameters():
        p.requires_grad = True
    pos_model.train().to(dev)
    print(f"[clsrank] pos_model 参数 = {sum(p.numel() for p in pos_model.parameters())}", flush=True)
    opt = torch.optim.Adam([p for p in pos_model.parameters() if p.requires_grad], lr=args.lr)
    init = [p.detach().clone() for p in pos_model.parameters()]

    B = torch.from_numpy(board); S = torch.from_numpy(stats)
    CL = torch.from_numpy(cls); CE = torch.from_numpy(cell)
    SC = torch.from_numpy(score); CN = torch.from_numpy(cnt)
    VALID = torch.arange(C).unsqueeze(0) < CN.unsqueeze(1)

    def forward_values(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (v, target_scores, valid)：v=(n,C) 模型在该行各候选类上的类值（HOLD 记 0）"""
        bt = torch.from_numpy(np.ascontiguousarray(idx))
        bb = B[bt].to(dev).float(); ss = S[bt].to(dev).float()
        cc = CL[bt].to(dev); xy = CE[bt].to(dev); aa = SC[bt].to(dev); vv = VALID[bt].to(dev)
        out = pos_model(bb, ss)
        am = out["action_map"]                                    # (B,24,19,19)
        rows = torch.arange(len(bt), device=dev).unsqueeze(1)
        y = xy[..., 0].clamp(min=0); x = xy[..., 1].clamp(min=0)  # HOLD 行是 (-1,-1) ⇒ clamp 到 (0,0) 后置 0
        v = am[rows, cc, y, x]
        v = torch.where(xy[..., 0] < 0, torch.zeros_like(v), v)   # HOLD（无格子）记 0
        return v, aa, vv

    def metrics(idx: np.ndarray, train: bool = False):
        tot_ce = tot_n = 0.0
        arg_hit = arg_tot = 0
        pair_hit = pair_tot = 0
        skipped = 0
        first_dbg = None
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            with torch.set_grad_enabled(train):
                v, aa, vv = forward_values(b)

                zero = torch.zeros_like(v)
                n = vv.sum(1, keepdim=True).clamp(min=2).float()
                mv = torch.where(vv, v, zero).sum(1, keepdim=True) / n
                ma = torch.where(vv, aa, zero).sum(1, keepdim=True) / n
                sv = torch.sqrt(torch.where(vv, (v - mv) ** 2, zero).sum(1, keepdim=True) / n).clamp(min=1e-8)
                sa = torch.sqrt(torch.where(vv, (aa - ma) ** 2, zero).sum(1, keepdim=True) / n).clamp(min=1e-8)
                NEG = -1e9
                logq = torch.log_softmax(torch.where(vv, (v - mv) / sv, torch.full_like(v, NEG)), dim=1)
                p = torch.softmax(torch.where(vv, (aa - ma) / sa, torch.full_like(aa, NEG)), dim=1) * vv
                p = p / p.sum(1, keepdim=True).clamp(min=1e-12)
                ce = -(p * logq).sum(1)
                ok = (vv.sum(1) > 1) & ((aa.max(1).values - aa.min(1).values) > 1e-9)   # 跳过全平行
                if first_dbg is None:
                    first_dbg = (int(ok.sum()), bool(v.requires_grad), bool(ce.requires_grad))
                loss = ce[ok].mean() if ok.any() else torch.zeros((), device=dev)
                if train:
                    opt.zero_grad()
                    if loss.requires_grad:
                        loss.backward(); opt.step()
                    else:
                        # 该 batch 没有可用行（ok 全假）⇒ 常量 loss 无梯度；跳过并计数（2026-10-11 修）
                        skipped += 1
            with torch.no_grad():
                tot_ce += float(ce[ok].sum()); tot_n += float(ok.sum())
                arg_hit += int(((logq.argmax(1) == p.argmax(1)) & ok).sum()); arg_tot += int(ok.sum())
                # 成对一致率
                n_ok = int(ok.sum())
                if n_ok:
                    vv2 = vv[ok].cpu().numpy(); aa2 = aa[ok].cpu().numpy()
                    for r in range(n_ok):
                        m = vv2[r]
                        vr = v[r][torch.from_numpy(m).to(dev)].cpu().numpy()
                        ar = aa2[r][m]
                        for a_i in range(len(vr)):
                            for b_i in range(a_i + 1, len(vr)):
                                if abs(ar[a_i] - ar[b_i]) < 1e-9:
                                    continue
                                pair_tot += 1
                                if (vr[a_i] - vr[b_i]) * (ar[a_i] - ar[b_i]) > 0:
                                    pair_hit += 1
        if first_dbg is not None and train:
            print(f"      [dbg] 首 batch: ok行={first_dbg[0]} v.requires_grad={first_dbg[1]} ce.requires_grad={first_dbg[2]} 跳过batch={skipped}", flush=True)
        return (tot_ce / max(tot_n, 1), arg_hit / max(arg_tot, 1), pair_hit / max(pair_tot, 1), int(arg_tot))

    ce0, ar0, pr0, n0 = metrics(va, False)
    tr0 = metrics(tr, False)
    print(f"[clsrank] **训练前 train**：CE={tr0[0]:.4f} argmax={tr0[1]*100:.2f}% pair={tr0[2]*100:.2f}% （非平局行 {tr0[3]}）", flush=True)
    print(f"[clsrank] **训练前 val**：CE={ce0:.4f}  argmax 一致率={ar0*100:.2f}%  成对一致率={pr0*100:.2f}%  "
          f"（非平局行 {n0}）", flush=True)
    best = ce0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trm = metrics(tr, True)
        vam = metrics(va, False)
        dnorm = float(sum((a - b).pow(2).sum() for a, b in zip(pos_model.parameters(), init)) ** 0.5)
        flag = ""
        if vam[0] < best:
            best = vam[0]
            save_three_net(args.out, class_model, pos_model, value_model,
                           {"num_heads": 3, "clsrank_epoch": ep, "clsrank_data": args.data})
            flag = "  <- saved"
        print(f"[clsrank] epoch {ep:2d}/{args.epochs} train CE={trm[0]:.4f} argmax={trm[1]*100:.2f}% "
              f"pair={trm[2]*100:.2f}% | val CE={vam[0]:.4f} argmax={vam[1]*100:.2f}% pair={vam[2]*100:.2f}% "
              f"| 参数变化范数={dnorm:.4f} ({time.time()-t0:.0f}s){flag}", flush=True)
    print(f"[clsrank] 结束：best val CE={best:.4f} ⇒ {args.out}", flush=True)
    print("[clsrank] **闸门判读**：若 val 的 argmax/成对一致率相比『训练前』没有明显提升（≥+10pp 才算有信号）"
          " ⇒ 说明跨类教师也不可学 ⇒ 关闭该方向。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
