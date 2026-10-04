"""候选 C 的训练器：**只训类头（class_state），目标是"我们自己的 1-ply 价值先验"**。

计划/依据：`docs/class_head_value_prior_plan.md`（先写的计划）+ `docs/next_step_survey_20261005.md` §3c。
数据：`collect_value_prior.py` 产的 npz（每行 = (决策, head, 类) 的候选格 + 逐格 `adv`）。
⚠️ **不模仿任何对手的动作**：目标来自我们自己的价值网（`adv` 就是 1-ply 价值差）。

核心设计（为什么这么做，见计划 §2）：
  * 每个 (决策, head) 只有 ≤5 个**被评估**的类 ⇒ 目标 = 这些类上 `max(adv)` 的 z-score → softmax；
  * 损失 = **只在被评估类上做条件 CE**：`-Σ p_c log( q_c / Σ_{c'∈C} q_{c'} )`
    —— 若用普通 CE（分母 24 类），等于把**没评估的合法类**强行压下去（注入未知先验）；
  * **单变量**：`pos_model`/`value_model` 冻结且逐位写回，只换 `class_state`。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_class_prior.py \
      --ckpt training_history/vprior/posnet_A_k5_m32.pt \
      --data training_history/vprior/vp_A_k5_m32.npz \
      --epochs 30 --batch-size 256 --lr 1e-3 --target-tau 1.0 --val-frac 0.1 \
      --out training_history/vprior/posnet_C_class.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

K_MAX = 5   # 每个 (决策, head) 最多被评估的类数（collect_value_prior --multi-class-k 5）


def group_rows(stats, player):
    """按"相邻行 (stats, player) 相同"切段 ⇒ 每段 = 一个决策（避免逐行比 3.5GB 的 board）。"""
    key = np.concatenate([stats.astype(np.float32), player[:, None].astype(np.float32)], axis=1)
    same = np.all(key[1:] == key[:-1], axis=1)
    n = len(player)
    return [0] + (np.nonzero(~same)[0] + 1).tolist() + [n]


def build_dataset(npz_path: Path, max_decisions: int | None, k_max: int = K_MAX):
    """返回 (board, stats, dec_ids, cls_pad, tgt_pad, tgt_mask) —— 全部按决策对齐。"""
    z = np.load(npz_path, allow_pickle=True)
    board, stats, player = z["board"], z["stats"], z["player"]
    head, cls, adv = z["head"], z["cls"], z["adv"]
    bounds = group_rows(stats, player)
    n_dec = len(bounds) - 1
    keep = range(n_dec) if not max_decisions else range(0, n_dec, max(1, n_dec // max_decisions))

    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for d in keep:
        a, b = bounds[d], bounds[d + 1]
        cls_pad = np.full((3, k_max), -1, dtype=np.int64)
        tgt_pad = np.zeros((3, k_max), dtype=np.float32)
        mask = np.zeros((3,), dtype=bool)
        for h in range(3):
            sel = [j for j in range(a, b) if int(head[j]) == h]
            if len(sel) < 2:
                continue
            if len(sel) > k_max:
                sel = sel[:k_max]
            cs = np.array([int(cls[j]) for j in sel])
            amax = np.array([float(np.max(adv[j])) for j in sel], dtype=np.float64)
            zz = (amax - amax.mean()) / (amax.std() + 1e-9)
            cls_pad[h, :len(sel)] = cs
            tgt_pad[h, :len(sel)] = zz
            mask[h] = True
        if mask.any():
            rows.append((a, cls_pad, tgt_pad, mask))

    dec_idx = np.array([r[0] for r in rows], dtype=np.int64)
    cls_pad = np.stack([r[1] for r in rows])          # (N,3,K)
    tgt_pad = np.stack([r[2] for r in rows])          # (N,3,K)  z 分数
    mask = np.stack([r[3] for r in rows])             # (N,3)
    print(f"[data] 决策总数={n_dec}  可用于类头训练={len(rows)}（每决策至少 1 个 head 有 ≥2 个被评估类）"
          f"  head 覆盖数={int(mask.sum())}", flush=True)
    return board, stats, dec_idx, cls_pad, tgt_pad, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="collect_value_prior 产的 npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-tau", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="按**决策位置**取尾部当验证集（npz 无 game id ⇒ 只能近似局级切分，勿当判据）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-decisions", type=int, default=None, help="冒烟用：只取这么多决策")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
        ("cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    print(f"[cfg] device={dev} epochs={args.epochs} bs={args.batch_size} lr={args.lr} "
          f"target_tau={args.target_tau} val_frac={args.val_frac}", flush=True)

    t0 = time.time()
    board, stats, dec_idx, cls_pad, tgt_pad, mask = build_dataset(
        Path(args.data), args.max_decisions)
    print(f"[data] 载入用时 {time.time()-t0:.1f}s", flush=True)

    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.az_train import save_three_net

    src = torch.load(args.ckpt, map_location="cpu")
    class_model, pos_model, value_model = load_three_models(args.ckpt)
    for m in (pos_model, value_model):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    class_model.to(dev)
    opt = torch.optim.Adam([p for p in class_model.parameters() if p.requires_grad], lr=args.lr)

    N = len(dec_idx)
    n_val = max(1, int(N * args.val_frac))
    tr = np.arange(0, N - n_val)
    va = np.arange(N - n_val, N)
    print(f"[split] train={len(tr)} val={len(va)}（val 取尾部 ⇒ 近似局级切分）", flush=True)

    cls_t = torch.from_numpy(cls_pad).to(dev)
    tgt_t = torch.from_numpy(tgt_pad).to(dev)
    msk_t = torch.from_numpy(mask).to(dev)

    def batch_loss(ids: np.ndarray, train: bool):
        b = torch.from_numpy(board[dec_idx[ids]].astype(np.float32)).to(dev)
        s = torch.from_numpy(stats[dec_idx[ids]].astype(np.float32)).to(dev)
        c = cls_t[ids]                     # (B,3,K)
        t = tgt_t[ids]                     # (B,3,K)
        m = msk_t[ids]                     # (B,3)
        out = class_model(b, s)
        logit = torch.stack([out[f"head{h+1}_logits"] for h in range(3)], dim=1)   # (B,3,C)
        k = c.shape[-1]
        safe = c.clamp(min=0)
        gathered = torch.gather(logit, 2, safe)                                    # (B,3,K)
        valid = (c >= 0) & m.unsqueeze(-1)
        neg = torch.finfo(gathered.dtype).min
        g = torch.where(valid, gathered, torch.full_like(gathered, neg))
        z = g - g.amax(dim=-1, keepdim=True)        # 条件 softmax（max 减掉；padding 被 mask 掉）
        logq = z - torch.logsumexp(z, dim=-1, keepdim=True)
        p = torch.softmax(t / args.target_tau, dim=-1)
        p = torch.where(valid, p, torch.zeros_like(p))
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        ce = -(p * torch.where(valid, logq, torch.zeros_like(logq))).sum(dim=-1)   # (B,3)
        ce = ce[m]
        # 机制读数：策略(argmax) 与 价值 top-1 的一致率
        with torch.no_grad():
            agree = (g.argmax(dim=-1) == t.masked_fill(~valid, -1e9).argmax(dim=-1))[m]
        return ce.mean(), agree.float().mean().item(), int(m.sum())

    for ep in range(1, args.epochs + 1):
        class_model.train()
        perm = np.random.permutation(tr)
        tot, n, ag, nag = 0.0, 0, 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            ids = perm[i:i + args.batch_size]
            loss, a, nm = batch_loss(ids, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * nm
            n += nm
            ag += a * nm
            nag += nm
        class_model.eval()
        vtot, vn, vag = 0.0, 0, 0.0
        for i in range(0, len(va), args.batch_size):
            ids = va[i:i + args.batch_size]
            with torch.no_grad():
                loss, a, nm = batch_loss(ids, False)
            vtot += float(loss) * nm
            vn += nm
            vag += a * nm
        print(f"[ep {ep:3d}] train_cond_CE={tot/max(n,1):.4f} (n={n})  "
              f"val_cond_CE={vtot/max(vn,1):.4f}  "
              f"train agree(policy vs value)={ag/max(nag,1):.1%}  "
              f"val agree={vag/max(vn,1):.1%}", flush=True)

    meta = dict(src)
    meta.update({"classprior_data": str(args.data), "classprior_epoch": args.epochs,
                 "classprior_target_tau": args.target_tau, "classprior_lr": args.lr})
    save_three_net(args.out, class_model, pos_model, value_model, meta)
    print(f"[done] {time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
