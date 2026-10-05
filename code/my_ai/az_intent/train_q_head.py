"""训练类条件价值头 Q(s, c)（方法侧 M2）。

数据：`--data` 下的 pkl（`ingest_selfplay_classes.py --label executed` 的产物）：
    每条 = board / stats / player / chosen_cls[3] / class_mask[24] / value_target

做法（写死在预注册里）：
  1. 用**冻结**的 `--ckpt`（三网）的 class 网跑一遍，把 `state_emb` 缓存下来（主干不动）；
  2. 小 MLP：`[state_emb ⊕ 类 one-hot] → 64 → 1`，MSE 拟合 `value_target`；
     * 只取**实际执行**的类（`chosen_cls != 255` 且在掩码内）；HOLD(23) 也照收（它也是"一个选择"）；
     * **按类频率开方反加权**（避免 HOLD 96% 主导）；`--class-weight-power` 可调；
  3. 保存 `q_state` 到 `--out`（`QHead.save`），部署时用 `AZAI_Q_CKPT` + `pos_pin=q` 加载。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_q_head.py \
      --ckpt training_history/vprior/posnet_A_k5_m32.pt \
      --data training_history/vprior/data_sp_reserve60_exec \
      --out training_history/vprior/qhead_M2.pt --epochs 30
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条（调试用）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    rows = []
    for f in sorted(Path(args.data).glob("az_selfplay_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rows.extend(d["samples"])
    if args.limit:
        rows = rows[:args.limit]
    print(f"[data] {len(rows)} 条样本（{args.data}）", flush=True)

    board = np.stack([np.asarray(r["board"], dtype=np.float32) for r in rows])
    stats = np.stack([np.asarray(r["stats"], dtype=np.float32) for r in rows])

    # ① 冻结主干 -> state_emb 缓存
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import load_three_models

    class_model, _pos, _val = load_three_models(args.ckpt)
    class_model.eval()
    embs = np.zeros((len(rows), 128), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            b = torch.from_numpy(board[i:i + 512])
            s = torch.from_numpy(stats[i:i + 512])
            o = class_model(b, s)
            embs[i:i + 512] = o["state_emb"].cpu().numpy()
    print(f"[emb] state_emb 缓存完成 {embs.shape}（{time.time()-t0:.1f}s）", flush=True)

    # ② 只取"实际执行"的 (emb, 类) 对
    X, C, Y = [], [], []
    cnt = np.zeros(24, dtype=np.int64)
    for i, r in enumerate(rows):
        cc = np.asarray(r["chosen_cls"], dtype=np.int64)
        cm = np.asarray(r["class_mask"], dtype=bool)
        v = float(r["value_target"])
        for c in cc:
            c = int(c)
            if c == 255 or not (0 <= c < 24) or not cm[c]:
                continue
            X.append(i)
            C.append(c)
            Y.append(v)
            cnt[c] += 1
    X = np.asarray(X); C = np.asarray(C); Y = np.asarray(Y, dtype=np.float32)
    print(f"[pairs] {len(X)} 个 (局面, 类) 对；类分布前 5: "
          + ", ".join("%d:%d" % (c, cnt[c]) for c in np.argsort(-cnt)[:5]), flush=True)

    # ③ 权重
    nz = cnt[cnt > 0]
    mean_cnt = float(nz.mean()) if len(nz) else 1.0
    w_cls = np.ones(24, dtype=np.float32)
    for c in range(24):
        w_cls[c] = float((mean_cnt / cnt[c]) ** args.class_weight_power) if cnt[c] > 0 else 0.0
    print("[weight] power=%.2f ⇒ 前 5 频繁类权重: " % args.class_weight_power
          + ", ".join("%d:%.2f" % (c, w_cls[c]) for c in np.argsort(-cnt)[:5]), flush=True)

    from my_ai.az_intent.q_head import QHead

    N = len(X)
    n_val = max(1, int(N * args.val_frac))
    perm = np.random.permutation(N)
    tr, va = perm[:-n_val], perm[-n_val:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = QHead().to(dev)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    emb_t = torch.from_numpy(embs)
    oh_t = torch.eye(24)

    def step(ids, train: bool):
        e = emb_t[X[ids]].to(dev)
        o = oh_t[C[ids]].to(dev)
        y = torch.from_numpy(Y[ids]).to(dev)
        w = torch.from_numpy(w_cls[C[ids]]).to(dev)
        p = q(e, o)
        loss = (w * (p - y) ** 2).sum() / w.sum().clamp(min=1e-9)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            err = (p - y).abs()
            # 机制读数：Q(花钱) - Q(HOLD) 在"花钱样本"上应为正
            return float(loss), float(err.mean()), e, o

    for ep in range(1, args.epochs + 1):
        q.train()
        p = np.random.permutation(tr)
        tl, tn, ta, nb = 0.0, 0, 0.0, 0
        for i in range(0, len(p), args.batch_size):
            ids = p[i:i + args.batch_size]
            l, a, _, _ = step(ids, True)
            tl += l * len(ids); ta += a * len(ids); tn += len(ids); nb += 1
        q.eval()
        with torch.no_grad():
            vl, va_, vn = 0.0, 0.0, 0
            for i in range(0, len(va), args.batch_size):
                ids = va[i:i + args.batch_size]
                l, a, _, _ = step(ids, False)
                vl += l * len(ids); va_ += a * len(ids); vn += len(ids)
        print(f"[ep {ep:3d}] train MSE={tl/max(tn,1):.5f} MAE={ta/max(tn,1):.4f} | "
              f"val MSE={vl/max(vn,1):.5f} MAE={va_/max(vn,1):.4f}", flush=True)

    # ④ 机制判据 P-M2a：Q(该样本实际选的类) 与 Q(HOLD) 的差，在"花钱类"样本上应为正
    with torch.no_grad():
        q.eval()
        sample = np.arange(0, len(X), max(1, len(X) // 4000))
        e = emb_t[X[sample]].to(dev)
        allq = q.score_all(e).cpu().numpy()
        c = C[sample]
        delta = allq[np.arange(len(c)), c] - allq[:, 23]
        is_spend = (c >= 0) & (c <= 16)
        print(f"[P-M2a] Q(实际选的类) - Q(HOLD)：全样本均值 {delta.mean():+.4f}；"
              f"花钱类样本 n={int(is_spend.sum())} 均值 {delta[is_spend].mean() if is_spend.any() else float('nan'):+.4f}"
              f"；为正的比例 {float((delta[is_spend] > 0).mean()) if is_spend.any() else float('nan'):.1%}", flush=True)

    q.save(args.out)
    print(f"[done] {time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
