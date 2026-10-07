"""类头"同状态内类偏好"蒸馏训练器（第 14 轮，预注册 `docs/prereg_20261014_cls_visit_distill.md`）。

**学生**：`class_state`（`class_model` 是独立模块 ⇒ 只训它 = 天然单变量）。
**教师**：K48 搜索的 `intent_counts` = 每决策 **3 个 slot** 的 `(类, 格子) -> 访问数`
  ⇒ 按类边缘化得到**同状态内的类偏好**（去掉了"花钱只出现在富状态"那类混淆）。

损失 = 逐 slot 的类交叉熵（软目标，`class_mask` 上）。指标 = **argmax 类分布的前后变化**（决策相关，不看 logit std）。

用法:
  python -u code/my_ai/az_intent/train_cls_visit_distill.py --ckpt checkpoints/posnet_A_k5_m32.pt \
      --data training_history/vprior/data_k48_80 --out training_history/vprior/posnet_dk48cls.pt \
      --epochs 8 --lr 3e-4 --val-frac 0.2
"""
from __future__ import annotations

import argparse
import collections
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from my_ai.az_intent.az_selfplay import load_three_models  # noqa: E402

NC = 24


def rows_from_game(path: Path, max_rows: int) -> list[tuple]:
    """每行 = 一个决策：输入 (board, stats, player, class_mask) + 3 个 slot 的类目标分布。"""
    d = pickle.load(open(path, "rb"))
    out = []
    for rec in d.get("samples", []):
        ic = rec.get("intent_counts")
        if not ic or not isinstance(ic[0], list):
            continue
        heads = ic[0]
        tgt = np.zeros((len(heads), NC), dtype=np.float32)
        any_mass = False
        for si, h in enumerate(heads):
            if not isinstance(h, dict):
                continue
            for k, v in h.items():
                if isinstance(k, tuple) and len(k) == 3:
                    c = int(k[0])
                    if 0 <= c < NC:
                        tgt[si, c] += float(v)
                        any_mass = True
            s = tgt[si].sum()
            if s > 0:
                tgt[si] /= s
        if not any_mass:
            continue
        cm = np.asarray(rec.get("class_mask", np.ones(NC, dtype=bool)), dtype=bool)
        if cm.shape[0] != NC:
            cm = np.ones(NC, dtype=bool)
        out.append((np.asarray(rec["board"], dtype=np.float32),
                    np.asarray(rec["stats"], dtype=np.float32),
                    int(rec["player"]), cm, tgt))
        if max_rows and len(out) >= max_rows:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--t-class", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-rows-per-game", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(args.device if args.device != "auto" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    files = sorted(Path(args.data).rglob("az_selfplay_seed*.pkl"))
    print(f"[cls-distill] 数据 {args.data}: {len(files)} 个 pkl；device={dev}", flush=True)
    if len(files) < 4:
        print("[cls-distill] pkl 太少（<4），先别训"); return 1
    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac))
    val_files, tr_files = files[:n_val], files[n_val:]
    tr = [r for f in tr_files for r in rows_from_game(f, args.max_rows_per_game)]
    va = [r for f in val_files for r in rows_from_game(f, args.max_rows_per_game)]
    print(f"[cls-distill] 训练行={len(tr)}（{len(tr_files)} 局） 验证行={len(va)}（{len(val_files)} 局）", flush=True)

    class_model, _, _ = load_three_models(args.ckpt)
    class_model = class_model.to(dev)
    base = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    opt = torch.optim.Adam(class_model.parameters(), lr=args.lr)

    def run(rows, train: bool):
        class_model.train(train)
        tot, n, nrow = 0.0, 0, 0
        argmax_hist = collections.Counter()
        rng2 = random.Random(1234)
        idx = list(range(len(rows)))
        if train:
            rng2.shuffle(idx)
        for i in range(0, len(idx), 32):
            batch = [rows[j] for j in idx[i:i + 32]]
            b = torch.from_numpy(np.stack([r[0] for r in batch])).to(dev)
            s = torch.from_numpy(np.stack([r[1] for r in batch])).to(dev)
            cm = torch.from_numpy(np.stack([r[3] for r in batch])).to(dev)
            tgt = torch.from_numpy(np.stack([r[4] for r in batch])).to(dev)   # (B, S, 24)
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                out = class_model(b, s)
                hl = torch.stack([out[f"head{h+1}_logits"] for h in range(3)], dim=1)  # (B,3,C)
                C = hl.shape[-1]
                # ⚠️ 头只有 C 类（如 23；HOLD=23 不在头输出里）⇒ 目标与掩码都切到 C 并重新归一
                t = tgt[:, :, :C].clone()
                t[:, :, :] = torch.where(cm[:, None, :C], t[:, :, :], torch.zeros_like(t[:, :, :]))
                t = t / t.sum(-1, keepdim=True).clamp(min=1e-9)
                hl = torch.where(cm[:, None, :C], hl, torch.full_like(hl, -1e9))
                logp = torch.log_softmax(hl / args.t_class, dim=-1)
                w = (t.sum(-1) > 0).float()                                  # (B, S)
                ce = -(t * logp).sum(-1)                                     # (B, S)
                loss = (ce * w).sum() / w.sum().clamp(min=1e-9)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * float(w.sum())
            n += int(w.sum())
            for k in range(len(batch)):
                argmax_hist[int(hl[k, 0].argmax())] += 1                     # slot0 的 argmax 分布
            nrow += 1
        return tot / max(n, 1), argmax_hist, nrow

    base_ce, base_hist, _ = run(va, False)
    top = ", ".join(f"{c}:{v}" for c, v in base_hist.most_common(4))
    print(f"[cls-distill] **训练前基线** val CE={base_ce:.4f}  slot0 argmax 分布前4 = {top}", flush=True)
    best_ce, best_state = 1e9, None
    for ep in range(1, args.epochs + 1):
        tr_ce, _, _ = run(tr, True)
        va_ce, va_hist, _ = run(va, False)
        top = ", ".join(f"{c}:{v}" for c, v in va_hist.most_common(4))
        flag = ""
        if va_ce < best_ce:
            best_ce, flag = va_ce, "  <- best"
            best_state = {k: v.detach().clone() for k, v in class_model.state_dict().items()}
        print(f"[cls-distill] ep {ep:2d}/{args.epochs} train CE={tr_ce:.4f} val CE={va_ce:.4f} "
              f"val argmax 前4=[{top}]{flag}", flush=True)

    if best_state is None:
        print("[cls-distill] 没有更好的轮次 ⇒ 不产出"); return 1
    graft = 0
    for k, v in best_state.items():
        if k in base["class_state"] and base["class_state"][k].shape == v.shape:
            base["class_state"][k] = v.cpu(); graft += 1
    torch.save(base, args.out)
    print(f"[cls-distill] 已把训练后的 class_state 嫁接到 {args.out}（{graft} 个张量；"
          f"pos_state/value_state 一字未动）best val CE={best_ce:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
