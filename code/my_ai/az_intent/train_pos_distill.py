"""位置头蒸馏训练器（第 14 轮，预注册 `docs/prereg_20261013_distill_k48_cells.md` 第 2 步）。

**学生**：`pos_state`（`pos_model` 是独立模块 ⇒ 只训它 = 天然单变量）。
**教师**：K48 搜索的 `intent_counts`（每决策 3 个 slot 的 `(类, 格子) -> 访问数`）。

目标构造：把 3 个 slot 的访问数按 `(类,格子)` 聚合 ⇒ 取**访问质量最大的那个类** `cls*`
⇒ 候选 = 该类 **top-K 格子**（按访问数），目标分布 = 归一化访问数（软目标）。
损失 = 在候选格子上、对 `action_map[cls*]` 做 **z-score + softmax(t_pos)** 后的交叉熵（与 `train_value_prior.py` 同口径）。

用法:
  python -u code/my_ai/az_intent/train_pos_distill.py --ckpt checkpoints/posnet_A_k5_m32.pt \
      --data training_history/vprior/data_k48_80 --out training_history/vprior/posnet_dk48.pt \
      --epochs 8 --lr 3e-4 --topk 8 --val-frac 0.2
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


def rows_from_game(path: Path, topk: int, max_rows: int) -> list[tuple]:
    d = pickle.load(open(path, "rb"))
    out = []
    for rec in d.get("samples", []):
        ic = rec.get("intent_counts")
        if not ic:
            continue
        heads = ic[0]
        mass: dict[tuple[int, int, int], float] = collections.defaultdict(float)
        for h in heads:
            if isinstance(h, dict):
                for k, v in h.items():
                    if isinstance(k, tuple) and len(k) == 3 and k[1] >= 0 and k[2] >= 0:
                        mass[(int(k[0]), int(k[1]), int(k[2]))] += float(v)
        if not mass:
            continue
        by_cls: dict[int, list[tuple[tuple[int, int], float]]] = collections.defaultdict(list)
        for (c, x, y), v in mass.items():
            by_cls[c].append(((x, y), v))
        cls_star, best = None, -1.0
        for c, lst in by_cls.items():
            s = sum(v for _, v in lst)
            if s > best:
                best, cls_star = s, c
        lst = sorted(by_cls[cls_star], key=lambda t: -t[1])[:topk]
        cells = np.asarray([c for c, _ in lst], dtype=np.int64)
        w = np.asarray([v for _, v in lst], dtype=np.float32)
        tgt = w / max(w.sum(), 1e-9)
        out.append((np.asarray(rec["board"], dtype=np.float32),
                    np.asarray(rec["stats"], dtype=np.float32),
                    int(rec["player"]), int(cls_star), cells, tgt, float(w.sum())))
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
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--t-pos", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-rows-per-game", type=int, default=120)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(args.device if args.device != "auto" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    files = sorted(Path(args.data).rglob("az_selfplay_seed*.pkl"))
    print(f"[pos-distill] 数据 {args.data}: {len(files)} 个 pkl；device={dev}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac))
    val_files, tr_files = files[:n_val], files[n_val:]
    tr = [r for f in tr_files for r in rows_from_game(f, args.topk, args.max_rows_per_game)]
    va = [r for f in val_files for r in rows_from_game(f, args.topk, args.max_rows_per_game)]
    print(f"[pos-distill] 训练行={len(tr)}（{len(tr_files)} 局）  验证行={len(va)}（{len(val_files)} 局）", flush=True)

    _, pos_model, _ = load_three_models(args.ckpt)
    pos_model = pos_model.to(dev)
    base = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    opt = torch.optim.Adam(pos_model.parameters(), lr=args.lr)

    def run(rows, train: bool) -> tuple[float, float, int]:
        pos_model.train(train)
        tot = hit = n = 0
        rng2 = random.Random(1234)
        idx = list(range(len(rows)))
        if train:
            rng2.shuffle(idx)
        bs = 32
        for i in range(0, len(idx), bs):
            batch = [rows[j] for j in idx[i:i + bs]]
            b = torch.from_numpy(np.stack([r[0] for r in batch])).to(dev)
            s = torch.from_numpy(np.stack([r[1] for r in batch])).to(dev)
            cls = torch.tensor([r[3] for r in batch], dtype=torch.long, device=dev)
            mx = max(len(r[4]) for r in batch)
            cell = torch.full((len(batch), mx, 2), -1, dtype=torch.long, device=dev)
            tgt = torch.zeros((len(batch), mx), dtype=torch.float32, device=dev)
            msk = torch.zeros((len(batch), mx), dtype=torch.bool, device=dev)
            for k, r in enumerate(batch):
                L = len(r[4])
                cell[k, :L] = torch.from_numpy(r[4]).to(dev)
                tgt[k, :L] = torch.from_numpy(r[5]).to(dev)
                msk[k, :L] = True
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                out = pos_model(b, s)
                am = out["action_map"]                     # (B, 24, 19, 19)
                sel = am.gather(1, cls.view(-1, 1, 1, 1).expand(-1, 1, 19, 19)).squeeze(1)
                sel = sel.reshape(sel.shape[0], -1)        # (B, 361) 展平后才能用一维下标 gather
                cx = cell[..., 0].clamp(min=0)
                cy = cell[..., 1].clamp(min=0)
                logits = sel.gather(1, (cy * 19 + cx))     # (B, mx)
                logits = torch.where(msk, logits, torch.full_like(logits, -1e9))
                z = logits - logits.mean(dim=1, keepdim=True)
                z = z / (z.std(dim=1, keepdim=True) + 1e-6)
                logp = torch.log_softmax(z / args.t_pos, dim=1)
                loss = -(tgt * logp * msk).sum() / msk.sum().clamp(min=1)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            tot += float(loss.detach()) * int(msk.sum())
            hit += int(((logits.masked_fill(~msk, -1e9).argmax(1)) == (tgt.argmax(1))).sum())
            n += int(msk.sum())
        return tot / max(n, 1), hit / max(len(rows), 1), len(rows)

    best_ce, best_state = 1e9, None
    # ---- 训练前基线（原始 ckpt 与教师的 top1 一致率）—— 否则无法判断"学到了多少" ----
    base_va_ce, base_va_hit, _ = run(va, False)
    base_tr_ce, base_tr_hit, _ = run(tr, False)
    print(f"[pos-distill] **训练前基线** train CE={base_tr_ce:.4f} top1命中={base_tr_hit:.3f} | "
          f"val CE={base_va_ce:.4f} top1命中={base_va_hit:.3f}", flush=True)
    for ep in range(1, args.epochs + 1):
        tr_ce, tr_hit, _ = run(tr, True)
        va_ce, va_hit, _ = run(va, False)
        flag = ""
        if va_ce < best_ce:
            best_ce, flag = va_ce, "  <- best"
            best_state = {k: v.detach().clone() for k, v in pos_model.state_dict().items()}
        print(f"[pos-distill] ep {ep:2d}/{args.epochs}  train CE={tr_ce:.4f} top1命中={tr_hit:.3f} | "
              f"val CE={va_ce:.4f} top1命中={va_hit:.3f}{flag}", flush=True)

    if best_state is None:
        print("[pos-distill] 没有更好的轮次 ⇒ 不产出（保持原 ckpt）")
        return 1
    graft = 0
    for k, v in best_state.items():
        if k in base["pos_state"] and base["pos_state"][k].shape == v.shape:
            base["pos_state"][k] = v.cpu()
            graft += 1
    torch.save(base, args.out)
    print(f"[pos-distill] 已把训练后的 pos_state 嫁接到 {args.out}（{graft} 个张量；"
          f"class_state/value_state 一字未动）best val CE={best_ce:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
