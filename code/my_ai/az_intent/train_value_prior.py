"""用"价值偏好"目标训位置网（第一轮）。**类网 / 价值网冻结，只训位置网。**

数据来自 `collect_value_prior.py` 的 npz。每行 = 一个 (决策点, 头)：
    board(28,19,19)  stats(42,)  player  head  cls  cell(Nc,2)  adv(Nc,)

损失 = 逐行的位置 CE，两侧都用**同一套 z-score**（在这些 cells 上）：

    目标  p = softmax( z(adv) / target_tau )
    预测  q = softmax( z(action_map[cls]) / t_pos )
    L = -Σ_x p_x · log q_x

（两侧同尺度是 2026-09-17 那个 100× bug 的教训：采样器/解码器/损失必须逐字对齐。）

⚠️ **`--target-tau` 近似是个重参数化、不是"锐度旋钮"**（2026-09-18 分析，用户提问引出）：
令 ∂L/∂m = 0 得 q = p ⇒ `m/t_pos = z(adv)/tau + 常数` ⇒ **m ∝ z(adv)**，与 tau 无关；
而采样时解码器**会重新 z-score**（`(m-mean)/std`），于是 `tau` 与地图的绝对尺度一起被约掉，
抽出来的分布只剩 `softmax(单位方差的 z(adv) / t_pos)`。
⇒ **部署侧的锐度旋钮只有 `t_pos`；tau 只在"网络容量有限时损失如何分配注意力"上起作用。**
默认 `target_tau = t_pos = 1.0`（等价于"用单位方差的优势当目标"）。
命名注意：**这个 tau 与价值标签的时间衰减 tau（`az_train.add_weighted_labels`，γ=exp(-1/tau)）
完全无关**，故此处改名为 `--target-tau`。

**锚定**（可选，`--anchor-lambda`）：参数空间 L2 `λ·‖θ_pos − θ_pos^init‖²`。
因为目标只覆盖类 17（类头塌缩成闪电 ⇒ 三个头的 argmax 都是 17），其余通道拿不到梯度，
而共享骨干会被类 17 的梯度带着走。参数 L2 是最便宜的正则（不需要额外前向）。
⚠️ 这不是原方案（记录 `action_map` 做 MSE）——本数据没记 action_map；λ 未标定。

预注册判据（跑前写死，见 docs/az_value_prior_plan.md §6）：
  1. **CE 明显下降**：均匀基线 = ln(271) = 5.60。
  2. **top1 命中率明显上升**：随机基线 = 1/271 = 0.37%。
  3. （最终）训出的位置网当先验、k=24 打 rule_v4 —— 追 oracle 的 37.5%（现状 21.1%）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-tau", type=float, default=1.0,
                    help="目标 softmax 的温度。⚠️ 近似重参数化、不是锐度旋钮（锐度是 t_pos）；"
                         "见文件头注释的推导。与价值标签的时间衰减 tau 无关（故改名）。")
    ap.add_argument("--t-pos", type=float, default=1.0, help="预测 softmax 的温度（应=采样器的 t_pos）")
    ap.add_argument("--anchor-lambda", type=float, default=0.0,
                    help="参数空间 L2 锚定强度（0=关）。目标只覆盖类 17，其余通道无梯度")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"[vprior] device={dev}  target_tau={args.target_tau}  t_pos={args.t_pos}  "
          f"anchor_lambda={args.anchor_lambda}", flush=True)

    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.az_train import save_three_net

    d = np.load(args.data, allow_pickle=True)
    board = torch.from_numpy(np.asarray(d["board"], dtype=np.float32))
    stats = torch.from_numpy(np.asarray(d["stats"], dtype=np.float32))
    cls = torch.from_numpy(np.asarray(d["cls"], dtype=np.int64))
    cell = torch.from_numpy(np.asarray(d["cell"], dtype=np.int64))
    adv = torch.from_numpy(np.asarray(d["adv"], dtype=np.float32))
    cnt = torch.from_numpy(np.asarray(d["cnt"], dtype=np.int64))
    N, MAXC = adv.shape
    print(f"[vprior] 数据 {args.data}: {N} 行（决策点×头），每行格数 均值 {cnt.float().mean():.1f}"
          f"  最大 {int(cnt.max())}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(int(N * args.val_frac), 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    class_model, pos_model, value_model = load_three_models(args.ckpt)
    for m in (class_model, value_model):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
    class_model.to(dev); value_model.to(dev)
    for p in pos_model.parameters():
        p.requires_grad = True
    pos_model.train().to(dev)

    init_ref = None
    if args.anchor_lambda > 0:
        init_ref = [p.detach().clone() for p in pos_model.parameters()]

    opt = torch.optim.Adam([p for p in pos_model.parameters() if p.requires_grad], lr=args.lr)
    valid = torch.arange(MAXC).unsqueeze(0) < cnt.unsqueeze(1)          # (N, MAXC)
    cy = cell.clamp(min=0)                                              # 填充位 clamp 到 (0,0)

    def run(idx: np.ndarray, train: bool) -> tuple[float, float, float]:
        tot_ce, tot_hit, tot_n = 0.0, 0.0, 0.0
        order = rng.permutation(len(idx)) if train else np.arange(len(idx))
        for i in range(0, len(idx), args.batch_size):
            b = idx[order[i:i + args.batch_size]]
            bt = torch.from_numpy(b)
            bb = board[bt].to(dev); ss = stats[bt].to(dev)
            cc = cls[bt].to(dev); xy = cy[bt].to(dev); aa = adv[bt].to(dev)
            vv = valid[bt].to(dev)
            with torch.set_grad_enabled(train):
                out = pos_model(bb, ss)
                am = out["action_map"]                                # (B,24,19,19)
                rows = torch.arange(len(bt), device=dev).unsqueeze(1)
                vals = am[rows, cc.unsqueeze(1), xy[..., 0], xy[..., 1]]   # (B,MAXC)
                # 掩码统计（填充位用 0 参与）—— 直接对 finfo.min 填充求和会溢出成 -inf
                n = vv.sum(dim=1, keepdim=True).clamp(min=2).float()
                zero = torch.zeros_like(vals)
                v0 = torch.where(vv, vals, zero)
                a0 = torch.where(vv, aa, zero)
                m_v = v0.sum(dim=1, keepdim=True) / n
                m_a = a0.sum(dim=1, keepdim=True) / n
                s_v = torch.sqrt((torch.where(vv, (vals - m_v) ** 2, zero)).sum(dim=1, keepdim=True) / n
                                 ).clamp(min=1e-8)
                s_a = torch.sqrt((torch.where(vv, (aa - m_a) ** 2, zero)).sum(dim=1, keepdim=True) / n
                                 ).clamp(min=1e-8)
                NEG = -1e9
                logq = torch.log_softmax(torch.where(vv, (vals - m_v) / s_v / args.t_pos,
                                                     torch.full_like(vals, NEG)), dim=1)
                p = torch.softmax(torch.where(vv, (aa - m_a) / s_a / args.target_tau,
                                              torch.full_like(aa, NEG)), dim=1) * vv
                p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-12)
                ce = -(p * logq).sum(dim=1)
                loss = (ce * (vv.sum(dim=1) > 1)).sum() / max(int((vv.sum(dim=1) > 1).sum()), 1)
                if train:
                    opt.zero_grad()
                    if init_ref is not None:
                        reg = sum(((q - p0) ** 2).sum() for q, p0 in
                                  zip(pos_model.parameters(), init_ref))
                        loss = loss + args.anchor_lambda * reg
                    loss.backward()
                    opt.step()
            ok = (vv.sum(dim=1) > 1)
            with torch.no_grad():
                tot_ce += float(ce[ok].sum())
                tot_hit += float(((logq.argmax(1) == p.argmax(1)) & ok).sum())
                tot_n += float(ok.sum())
        return tot_ce / max(tot_n, 1), tot_hit / max(tot_n, 1), tot_n

    print("[vprior] 均匀基线 CE = ln(271) = %.3f ；随机 top1 = 1/271 = %.2f%%"
          % (np.log(271), 100 / 271), flush=True)
    base_ce, base_hit, _ = run(val_idx, False)
    print("[vprior] **训练前** val: CE=%.4f  top1命中=%.2f%%" % (base_ce, 100 * base_hit), flush=True)

    best = (base_ce, -1)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_ce, tr_hit, _ = run(tr_idx, True)
        va_ce, va_hit, _ = run(val_idx, False)
        flag = ""
        if va_ce < best[0]:
            best = (va_ce, ep)
            save_three_net(args.out, class_model, pos_model, value_model,
                           {"num_heads": 3, "vprior_target_tau": args.target_tau, "vprior_tpos": args.t_pos,
                            "vprior_anchor": args.anchor_lambda, "vprior_epoch": ep,
                            "vprior_data": args.data})
            flag = "  ← 保存"
        print("[vprior] epoch %2d/%d  train CE=%.4f top1=%.2f%% | val CE=%.4f top1=%.2f%%  "
              "(%.0fs)%s" % (ep, args.epochs, tr_ce, 100 * tr_hit, va_ce, 100 * va_hit,
                             time.time() - t0, flag), flush=True)

    print("[vprior] 最佳 val CE=%.4f @epoch %d ⇒ %s" % (best[0], best[1], args.out), flush=True)
    print("[vprior] 判据1（CE 从 %.4f 降到 %.4f，降 %.1f%%）；判据2（top1 %.2f%% → %.2f%%）"
          % (base_ce, best[0], 100 * (1 - best[0] / max(base_ce, 1e-9)),
             100 * base_hit, 100 * va_hit), flush=True)


if __name__ == "__main__":
    main()
