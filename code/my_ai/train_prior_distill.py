"""训练位置先验（"往哪建"格子偏好），教师 = 我们自己的搜索估值（2026-10-23 夜）。

数据：`training_history/prior_labels/g*.npz`（`collect_prior_labels.py` 产出）
  feat (S,28,19,19) / stats (S,D) / cand_cls,x,y,val / cand_off / rounds

方法：
  · 对每个局面，取 `cls in 0..15`（建塔）的候选，按 (x,y) 聚合，教师分 = 该格最大估值；
  · 预测 = `action_map[0:16]` 在 (x,y) 的均值；
  · 损失 = listwise 软标签交叉熵（`softmax(teacher/T')` 当目标，`T=1.0`、`T'=0.5`）；
  · 只解冻 `action_map_conv`（冻结主干/类头/价值头，防遗忘闪电通道）；
  · 学习率 1e-3，Adam，按 epoch 数训练；**留出一局做验证**，报告 Spearman(pred, teacher)。

用法：
  python train_prior_distill.py --frac 1.0 --epochs 20 --out checkpoints/prior_distill_full.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import load_three_models

CKPT = REPO / "checkpoints/posnet_A_k5_m32.pt"
T = 1.0     # 预测温度
T_TGT = 0.5 # 目标温度


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def build_state_groups(d, target: str = "value") -> list[dict]:
    """把 npz 拆成"每个局面一个 dict"：{(x,y): 教师分} + 特征。"""
    offs = d["cand_off"]
    out = []
    for i in range(len(offs) - 1):
        lo, hi = int(offs[i]), int(offs[i + 1])
        if hi <= lo:
            continue
        cls = d["cand_cls"][lo:hi]
        xs = d["cand_x"][lo:hi]
        ys = d["cand_y"][lo:hi]
        vals = d["cand_val"][lo:hi]
        cell = {}
        for c, x, y, v in zip(cls, xs, ys, vals):
            if not (0 <= int(c) <= 15):
                continue
            key = (int(x), int(y))
            if target == "visit":
                cell[key] = cell.get(key, 0.0) + max(0.0, float(v))
            else:
                cell[key] = max(cell.get(key, -1e9), float(v))
        if len(cell) < 4:
            continue
        out.append({
            "board": torch.from_numpy(d["feat"][i].astype(np.float32)).unsqueeze(0),
            "stats": torch.from_numpy(d["stats"][i].astype(np.float32)).unsqueeze(0),
            "cell": cell,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0, help="用多少比例的训练局（学习曲线用）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/prior_distill_full.pt")
    ap.add_argument("--holdout", type=int, default=4, help="留出最后 N 个 npz 做验证")
    ap.add_argument("--pattern", default="g*.npz", help="数据文件通配（mcts 教师用 m*.npz）")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--target", default="value", choices=["value", "visit"],
                    help="value=rollout分(旧), visit=MCTS访问概率(按格求和后做软标签)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(REPO / "training_history/prior_labels" / args.pattern)))
    if not files:
        print("  ✗ 没有标签文件（先跑 collect_prior_labels.py）")
        return 1
    n_hold = max(1, args.holdout)
    train_files = files[:-n_hold]
    valid_files = files[-n_hold:]
    if args.frac < 1.0:
        k = max(1, int(len(train_files) * args.frac))
        train_files = train_files[:k]
    print(f"  训练局 {len(train_files)}（frac={args.frac}）  验证局 {len(valid_files)}", flush=True)

    _, pos_m, _ = load_three_models(str(CKPT))
    pos_m.eval()
    for name, p in pos_m.named_parameters():
        p.requires_grad = "action_map_conv" in name
    n_train = sum(p.numel() for p in pos_m.parameters() if p.requires_grad)
    print(f"  可训练参数 {n_train:,}（只有 action_map_conv）", flush=True)
    opt = torch.optim.Adam([p for p in pos_m.parameters() if p.requires_grad], lr=args.lr)

    groups = []
    for f in train_files:
        groups += build_state_groups(np.load(f, allow_pickle=False), args.target)
    print(f"  训练样本 {len(groups)} 个局面", flush=True)

    print("  训练前基线:", flush=True)
    _base = []
    for f in valid_files:
        for g in build_state_groups(np.load(f, allow_pickle=False), args.target):
            with torch.no_grad():
                amap = pos_m(g["board"], g["stats"])["action_map"][0]
            cells = list(g["cell"].keys())
            preds = [float(amap[:16, x, y].mean()) for (x, y) in cells]
            _base.append(spearman(preds, [g["cell"][c] for c in cells]))
    _base = [r for r in _base if not np.isnan(r)]
    print(f"    基线 Spearman = {np.mean(_base) if _base else float('nan'):+.3f}（n={len(_base)}）", flush=True)

    for ep in range(1, args.epochs + 1):
        tot, nb = 0.0, 0
        perm = np.random.permutation(len(groups))
        _buf, _nb = [], 0
        for idx in perm:
            g = groups[idx]
            out = pos_m(g["board"], g["stats"])
            amap = out["action_map"][0]                       # (24,19,19)
            cells = list(g["cell"].keys())
            preds = torch.stack([amap[:16, x, y].mean() for (x, y) in cells])
            tv = torch.tensor([g["cell"][c] for c in cells], dtype=torch.float32)
            if args.target == "visit":
                w = tv / torch.clamp(tv.sum(), min=1e-8)
            else:
                w = torch.softmax(tv / T_TGT, dim=0)
            logp = torch.log_softmax(preds / T, dim=0)
            loss = -(w * logp).sum()
            _buf.append(loss); _nb += 1
            if _nb >= args.batch:
                _l = torch.stack(_buf).mean()
                opt.zero_grad(); _l.backward(); opt.step()
                tot += float(_l) * len(_buf); nb += len(_buf)
                _buf, _nb = [], 0
        if _buf:
            _l = torch.stack(_buf).mean()
            opt.zero_grad(); _l.backward(); opt.step()
            tot += float(_l) * len(_buf); nb += len(_buf)
        print(f"  epoch {ep}/{args.epochs}  loss={tot / max(1, nb):.4f}", flush=True)

    # 验证：留出局上的 Spearman
    rhos = []
    for f in valid_files:
        for g in build_state_groups(np.load(f, allow_pickle=False), args.target):
            with torch.no_grad():
                amap = pos_m(g["board"], g["stats"])["action_map"][0]
            cells = list(g["cell"].keys())
            preds = [float(amap[:16, x, y].mean()) for (x, y) in cells]
            rhos.append(spearman(preds, [g["cell"][c] for c in cells]))
    rhos = [r for r in rhos if not np.isnan(r)]
    print(f"  验证 Spearman 均值 = {np.mean(rhos) if rhos else float('nan'):+.3f}（n={len(rhos)} 局面）", flush=True)

    ckpt = torch.load(CKPT, map_location="cpu")
    ckpt["pos_state"] = pos_m.state_dict()
    ckpt["distill_note"] = f"action_map_conv 由自搜索教师重训，frac={args.frac}, epochs={args.epochs}"
    out_path = REPO / args.out
    torch.save(ckpt, out_path)
    print(f"  已写出 {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
