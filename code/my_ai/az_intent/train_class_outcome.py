"""候选 R：**用对局结果训练类头**（AWR —— 结果加权的行为克隆）。

计划：`docs/class_head_outcome_plan.md`。动机：实测类头是**常数函数**
（各类 logit 跨局面标准差中位 0.024、top1−top2 恒为 0.21 ⇒ 它根本不看局面），
而**价值网在类级上没有分辨力** ⇒ 唯一带类级分辨力的信号是**对局结果**。

数据：`ingest_bridge_dump.py` 从 bridge 裸 dump 产出的 pkl（含
`board / stats / player / chosen_cls[3] / class_mask[24] / value_target`）。

目标（先写死）：
    a = 该样本的 value_target（已按 player 取符号，胜为正）
    w = softmax(a / beta) 的逐样本权重（batch 内归一化）
    loss = - Σ_samples w · Σ_{h: chosen_cls[h] != 255} log q_h(chosen_cls[h])
其中 `q_h = softmax(head_h_logits 在 class_mask 为真的类上 / 1.0)`。

⚠️ 只训 `class_state`；`pos_state`/`value_state` 冻结并逐位写回。
⚠️ 这是**弱信号**（~400 步一局、二值结果）⇒ 每个 epoch 都打印
**"各类 logit 的跨局面标准差"**（= 类头有没有变成依赖状态的函数）作为机制读数。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_class_outcome.py \
      --ckpt training_history/vprior/posnet_A_k5_m32.pt \
      --data training_history/vprior/data_rv4_exp \
      --epochs 10 --lr 1e-3 --beta 0.5 --out training_history/vprior/posnet_R_class.pt
"""
from __future__ import annotations

import argparse
import pickle
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


def load_samples(data_dir: Path):
    rows = []
    for f in sorted(data_dir.glob("az_selfplay_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for s in d["samples"]:
            if "chosen_cls" not in s or "class_mask" not in s:
                continue
            rows.append(s)
    if not rows:
        raise SystemExit(f"{data_dir} 下没有带 chosen_cls/class_mask 的样本（先跑 ingest_bridge_dump.py）")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.5, help="AWR 温度：w ∝ exp(a/beta)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
        ("cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    print(f"[cfg] device={dev} epochs={args.epochs} bs={args.batch_size} lr={args.lr} "
          f"beta={args.beta}", flush=True)

    t0 = time.time()
    rows = load_samples(Path(args.data))
    board = np.stack([np.asarray(r["board"], dtype=np.float16) for r in rows])
    stats = np.stack([np.asarray(r["stats"], dtype=np.float16) for r in rows])
    chosen = np.stack([np.asarray(r["chosen_cls"], dtype=np.int64) for r in rows])      # (N,3)
    cmask = np.stack([np.asarray(r["class_mask"], dtype=bool) for r in rows])           # (N,24)
    adv = np.asarray([float(r["value_target"]) for r in rows], dtype=np.float32)        # (N,)
    print(f"[data] {len(rows)} 条；chosen 有效 head 数 = {(chosen != 255).sum()}；"
          f"优势均值 {adv.mean():+.3f}（>0 的比例 {(adv > 0).mean():.1%}）；"
          f"类掩码平均可选类数 {cmask.sum(axis=1).mean():.1f}", flush=True)
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

    N = len(rows)
    n_val = max(1, int(N * args.val_frac))
    tr = np.arange(0, N - n_val)
    va = np.arange(N - n_val, N)
    print(f"[split] train={len(tr)} val={len(va)}（按行块切；npz/pkl 无 game id ⇒ 近似局级）", flush=True)

    chosen_t = torch.from_numpy(chosen).to(dev)
    cmask_t = torch.from_numpy(cmask).to(dev)
    adv_t = torch.from_numpy(adv).to(dev)

    def batch_step(ids: np.ndarray, train: bool):
        b = torch.from_numpy(board[ids].astype(np.float32)).to(dev)
        s = torch.from_numpy(stats[ids].astype(np.float32)).to(dev)
        c = chosen_t[ids]                      # (B,3)
        m = cmask_t[ids]                       # (B,24)
        a = adv_t[ids]                         # (B,)
        out = class_model(b, s)
        logit = torch.stack([out[f"head{h+1}_logits"] for h in range(3)], dim=1)      # (B,3,24)
        neg = torch.finfo(logit.dtype).min
        masked = torch.where(m.unsqueeze(1), logit, torch.full_like(logit, neg))
        logq = torch.log_softmax(masked, dim=-1)                                      # (B,3,24)
        valid = (c != 255) & (c >= 0) & (c < logit.shape[-1])
        # ⚠️ 必须再过滤"**被选中的类却不在合法掩码里**"的样本：`pos_pin=argmax` 钉的是
        # **不套掩码**的 argmax（经典情形：闪电买不起 ⇒ argmax=17 但 17 不可执行）
        # ⇒ 这些决策的 chosen_cls 天然在掩码外，若不过滤会把 log q 拉成 -inf（实测踩到）。
        cm_g = torch.gather(m, 1, c.clamp(0, logit.shape[-1] - 1))
        valid = valid & cm_g
        # 取每个 head 的 chosen 的 log 概率
        idx = c.clamp(0, logit.shape[-1] - 1).unsqueeze(-1)                           # (B,3,1)
        lp = torch.gather(logq, 2, idx).squeeze(-1)                                   # (B,3)
        lp = torch.where(valid, lp, torch.zeros_like(lp))
        per_sample = lp.sum(dim=1)                                                    # (B,)
        w = torch.softmax(a / args.beta, dim=0) * len(a)                              # 归一化到均值 1
        w = torch.where(valid.any(dim=1), w, torch.zeros_like(w))
        loss = -(w * per_sample).sum() / w.sum().clamp(min=1e-9)
        # 机制读数：各类 logit 的**跨局面**标准差（常数函数 ⇒ ≈0.02）
        with torch.no_grad():
            cs = logit.reshape(-1, logit.shape[-1]).std(dim=0).mean()
        return loss, float(cs), int(valid.sum())

    for ep in range(1, args.epochs + 1):
        class_model.train()
        perm = np.random.permutation(tr)
        tot, n, cs_sum, nb = 0.0, 0, 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            ids = perm[i:i + args.batch_size]
            loss, cs, nv = batch_step(ids, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * nv
            n += nv
            cs_sum += cs
            nb += 1
        class_model.eval()
        vtot, vn, vcs, vnb = 0.0, 0, 0.0, 0
        for i in range(0, len(va), args.batch_size):
            ids = va[i:i + args.batch_size]
            with torch.no_grad():
                loss, cs, nv = batch_step(ids, False)
            vtot += float(loss) * nv
            vn += nv
            vcs += cs
            vnb += 1
        print(f"[ep {ep:3d}] train_loss={tot/max(n,1):.4f} val_loss={vtot/max(vn,1):.4f} "
              f"| 跨局面 logit std: train={cs_sum/max(nb,1):.4f} val={vcs/max(vnb,1):.4f} "
              f"(常数类头基准≈0.02)", flush=True)

    meta = dict(src)
    meta.update({"classR_data": str(args.data), "classR_epoch": args.epochs,
                 "classR_beta": args.beta, "classR_lr": args.lr})
    save_three_net(args.out, class_model, pos_model, value_model, meta)
    print(f"[done] {time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
