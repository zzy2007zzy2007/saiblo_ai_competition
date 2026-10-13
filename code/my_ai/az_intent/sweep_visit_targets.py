"""扫描：**位置头重训的"训练方式"**（见 `docs/pos_target_training_sweep_plan.md`）。

纪律（计划 §1）：**判据段（seed 7..70）禁止用来选配方** ✗ ⇒ 本脚本只用 **dump 数据的离线指标**（零对局 ✓）；
只给最终 1–2 个入围配方各跑一次判据臂 ✓。

数据：`--cache` 指向 `train_visit_marginal.py` 产出的 npz
（`board/stats/cls/cell/prob/cnt` ✓；`prob` = 访问分布按 (类,格) **边缘化**后的目标 ✓；`cell` = 该行候选格 (x,y) ✓）。

⚠️ 已延后：`hard_fulllegal`（目标在**该类全部合法格**上归一化）需要**数据格式改动** ✗
（每行得带上该类的全部合法格；现有缓存只有候选格 ✓）⇒ 留到下一版数据构建时做 ✓。

变体（定义见 `VARIANTS` ✓；全部 warm start 自 incumbent、冻结 `class_state`/`value_state` ✓）：
  soft（对照）· hard（现状基线）· hard_smooth010 · rank1_rest（0.5/0.5 ✓）
  · hard_dropclass16 · hard_headonly（只训 `action_map_conv` ✓）
  · hard_lr3e4 / hard_lr3e3 · rank_hinge（pairwise margin ✓）

离线指标（val split ✓）：CE、与目标 argmax 的一致率（= "图的 argmax 是否为目标的 rank-1 格" ✓，**部署真正读的东西** ✓）、
以及 incumbent 的**同指标基线** ✓、参数变化范数、变更张量数（核对单变量 ✓）。

用法：
  python -u code/my_ai/az_intent/sweep_visit_targets.py --cache <npz> \
    --variants soft,hard,rank1_rest,hard_headonly,rank_hinge --epochs 10 [--save-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VARIANTS: dict = {
    "soft":             dict(target="soft",        lr=1e-3, head_only=False, drop16=False, loss="ce"),
    "hard":             dict(target="hard",        lr=1e-3, head_only=False, drop16=False, loss="ce"),
    "hard_smooth005":   dict(target="hard_smooth", lr=1e-3, eps=0.05),
    "hard_smooth010":   dict(target="hard_smooth", lr=1e-3, eps=0.10),
    "hard_smooth020":   dict(target="hard_smooth", lr=1e-3, eps=0.20),
    "rank1_rest":       dict(target="rank1_rest",  lr=1e-3, head_only=False, drop16=False, loss="ce"),
    "hard_dropclass16": dict(target="hard",        lr=1e-3, drop16=True),
    "hard_headonly":    dict(target="hard",        lr=1e-3, head_only=True),
    "hard_lr3e4":       dict(target="hard",        lr=3e-4),
    "hard_lr3e3":       dict(target="hard",        lr=3e-3),
    "rank_hinge":       dict(target="hard",        lr=1e-3, loss="hinge"),
}
for _k, _v in VARIANTS.items():
    _v.setdefault("head_only", False)
    _v.setdefault("drop16", False)
    _v.setdefault("loss", "ce")
    _v.setdefault("eps", 0.10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="training_history/visitmarg/seedstate_rows.npz")
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--variants", default="soft,hard")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--out", default="training_history/visitmarg/sweep.json")
    args = ap.parse_args()

    # ⚠️ 2026-10-12（用户指出）：**必须限制 CPU 线程** —— 训练默认开满 OMP 线程，会与对局采集抢 CPU ✗
    #   （8 路对局本身已吃满 16 核；抢 CPU 会拖慢对局，甚至触发 SLOW_DECISION_SEC 影响**数据质量** ✓）
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "2")
    _os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(int(_os.environ.get("OMP_NUM_THREADS", "2")))

    from my_ai.az_intent.az_selfplay import load_three_models

    z = np.load(args.cache, allow_pickle=True)
    board, stats, cls, cell, prob, cnt = (z["board"], z["stats"], z["cls"], z["cell"], z["prob"], z["cnt"])
    N, C = int(board.shape[0]), int(cell.shape[1])
    print(f"[sweep] 数据 {N} 行 × 最多 {C} 格（每行格数均值 {cnt.mean():.1f}）", flush=True)

    # ⚠️ 2026-10-12：**保持 fp16**（原实现转 fp32 ⇒ 23k 行多占 ~0.9 GB、每次 H2D 搬运翻倍 ✗）；
    #   在 GPU 上按批 `.float()` 转换 ✓（上传统减半 ✓）
    B = torch.from_numpy(np.asarray(board, dtype=np.float16))
    S = torch.from_numpy(np.asarray(stats, dtype=np.float16))
    CL = torch.from_numpy(np.asarray(cls, dtype=np.int64))
    CE_ = torch.from_numpy(np.asarray(cell, dtype=np.int64))
    P = torch.from_numpy(np.asarray(prob, dtype=np.float32))
    CN = torch.from_numpy(np.asarray(cnt, dtype=np.int64))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    print(f"[sweep] train={len(tr)} val={len(va)}（随机切 ✓）；" + (f"存盘目录 {args.save_dir}" if args.save_dir else "不存盘"), flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    def target_of(cfg: dict, idx: torch.Tensor):
        p = P[idx].clone()
        m = torch.arange(C).unsqueeze(0) < CN[idx].unsqueeze(1)     # 有效格掩码
        if cfg["target"] == "soft":
            t = p
        elif cfg["target"] == "hard_smooth":
            arg = p.argmax(1, keepdim=True)
            t = torch.full_like(p, cfg["eps"] / max(C - 1, 1)).scatter_(1, arg, 1.0 - cfg["eps"])
        elif cfg["target"] == "rank1_rest":
            arg = p.argmax(1, keepdim=True)
            n_eff = CN[idx].clamp(min=2).float().unsqueeze(1)
            t = ((1.0 - 0.5) / (n_eff - 1.0)).expand_as(p).clone()
            t = t.scatter_(1, arg, 0.5)
        else:  # hard
            arg = p.argmax(1, keepdim=True)
            t = torch.zeros_like(p).scatter_(1, arg, 1.0)
        t = t * m.float()
        return t, m

    def build(seed: int):
        _cm, pm, _vm = load_three_models(args.ckpt)
        pm.to(dev)
        return _cm, pm, _vm

    def run_variant(name: str):
        cfg = VARIANTS[name]
        base_cm, pm, base_vm = build(args.seed)
        # 梯度开关：默认整网；head_only ⇒ 只开 action_map_conv ✓
        for n_, p_ in pm.named_parameters():
            p_.requires_grad_(bool((not cfg["head_only"]) or n_.startswith("action_map_conv")))
        n_train = int(sum(p_.numel() for p_ in pm.parameters() if p_.requires_grad))
        params = [p_ for p_ in pm.parameters() if p_.requires_grad]
        opt = torch.optim.Adam(params, lr=float(cfg["lr"]))
        base_sd = {k: v.detach().clone() for k, v in pm.state_dict().items()}
        tr_idx = tr if not cfg["drop16"] else tr[CL[tr].numpy() != 16]

        def evaluate(idx_np):
            # ⚠️ 2026-10-12：指标**累积在 GPU 上**，本函数结束时只 `.item()` 一次 ✓
            #   （原来每个 batch 都 `float(...)` ⇒ 强制 CUDA 同步数百次/epoch ✗）
            pm.eval()
            ce_acc = None
            hit_acc = None
            n = 0
            with torch.no_grad():
                for i in range(0, len(idx_np), 256):
                    b = torch.as_tensor(idx_np[i:i + 256])
                    am = pm(B[b].to(dev).float(), S[b].to(dev).float())["action_map"]
                    rows = torch.arange(len(b), device=dev).unsqueeze(1)
                    cc = CL[b].to(dev).unsqueeze(1)
                    y = CE_[b][..., 0].to(dev).clamp(min=0)
                    x = CE_[b][..., 1].to(dev).clamp(min=0)
                    lg = am[rows, cc, x, y]
                    t, m = target_of(cfg, b)
                    t, m = t.to(dev), m.to(dev)
                    lg = torch.where(m, lg, torch.full_like(lg, -1e9))
                    tt = t / t.sum(1, keepdim=True).clamp(min=1e-9)
                    _ce = -(tt * F.log_softmax(lg, dim=1)).sum(1).sum()
                    _hit = (lg.argmax(1) == tt.argmax(1)).sum()   # = 图 argmax 是否为目标 rank-1 格
                    ce_acc = _ce if ce_acc is None else (ce_acc + _ce)
                    hit_acc = _hit if hit_acc is None else (hit_acc + _hit)
                    n += len(b)
            if ce_acc is None:
                return float("nan"), float("nan")
            return float(ce_acc.item()) / max(n, 1), float(hit_acc.item()) / max(n, 1)

        ce0, hit0 = evaluate(va)
        best = (1e9, -1, None)
        for ep in range(1, args.epochs + 1):
            pm.train()
            order = rng.permutation(tr_idx)
            for i in range(0, len(order), args.batch_size):
                b = torch.as_tensor(order[i:i + args.batch_size])
                am = pm(B[b].to(dev).float(), S[b].to(dev).float())["action_map"]
                rows = torch.arange(len(b), device=dev).unsqueeze(1)
                cc = CL[b].to(dev).unsqueeze(1)
                y = CE_[b][..., 0].to(dev).clamp(min=0)
                x = CE_[b][..., 1].to(dev).clamp(min=0)
                lg = am[rows, cc, x, y]
                t, m = target_of(cfg, b)
                t, m = t.to(dev), m.to(dev)
                lg = torch.where(m, lg, torch.full_like(lg, -1e9))
                if cfg["loss"] == "hinge":
                    arg = t.argmax(1, keepdim=True)
                    pos = lg.gather(1, arg)
                    same = torch.arange(C, device=dev).unsqueeze(0) == arg
                    diff = (pos - lg).masked_fill((~m) | same, 0.0)
                    loss = F.relu(args.margin - diff).sum(1).mean()
                else:
                    tt = t / t.sum(1, keepdim=True).clamp(min=1e-9)
                    loss = -(tt * F.log_softmax(lg, dim=1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            ce_v, hit_v = evaluate(va)
            # 逐 epoch 打印（用户 2026-10-12 要求：看训练/验证损失下降是否正常 ✓）
            #   train 用 4,000 行子集估（控成本 ✓）；val 全量 ✓
            ce_t, hit_t = evaluate(tr_idx[: min(len(tr_idx), 4000)])
            print(f"    [{name:17s}] epoch {ep:2d}  train(4k) CE {ce_t:.4f} 命中 {hit_t * 100:5.1f}% "
                  f"| **val CE {ce_v:.4f} 命中 {hit_v * 100:5.1f}%**", flush=True)
            if ce_v <= best[0]:
                best = (ce_v, ep, {k: v.detach().clone() for k, v in pm.state_dict().items()})
        dnorm = float(np.sqrt(sum(float(((best[2][k].cpu() - base_sd[k].cpu()) ** 2).sum()) for k in base_sd)))
        nchg = sum(1 for k in base_sd if not torch.equal(best[2][k], base_sd[k]))
        r = dict(variant=name, n_train=n_train, val_ce0=round(ce0, 4), val_hit0=round(hit0 * 100, 2),
                 val_ce=round(best[0], 4), val_hit=round(hit_v * 100, 2), best_epoch=int(best[1]),
                 dnorm=round(dnorm, 1), n_changed=f"{nchg}/{len(base_sd)}", ckpt=None)
        if args.save_dir:
            out = dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
            out["class_state"] = base_cm.state_dict()
            out["pos_state"] = best[2]
            out["value_state"] = base_vm.state_dict()
            out["sweep_variant"] = name
            fp = Path(args.save_dir) / f"visitsweep_{name}.pt"
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(out, fp)
            r["ckpt"] = str(fp)
            try:                                   # 自检（新纪律 ✓）
                load_three_models(str(fp)); r["roundtrip"] = "ok"
            except Exception as e:  # noqa: BLE001
                r["roundtrip"] = f"FAIL {str(e)[:60]}"
        return r

    rows = []
    for name in [x.strip() for x in args.variants.split(",") if x.strip()]:
        if name not in VARIANTS:
            print(f"  [skip] 未知变体 {name}（可选：{', '.join(VARIANTS)}）")
            continue
        r = run_variant(name)
        rows.append(r)
        print(f"  [{name:17s}] 可训参数 {r['n_train']:>7d} | 训练前 val CE {r['val_ce0']:.4f} / 命中 {r['val_hit0']:5.2f}% "
              f"| **训练后 val CE {r['val_ce']:.4f} / 命中 {r['val_hit']:5.2f}%** | best ep {r['best_epoch']:2d} "
              f"| 参数范数 {r['dnorm']:8.1f} | 变更 {r['n_changed']}"
              + (f" | 往返 {r.get('roundtrip')}" if args.save_dir else ""), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sweep] 表写入 {args.out}")
    print("[sweep] ⚠️ 选择只看这些**离线指标** ✓；判据段（7..70）留给入围者且只跑一次 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
