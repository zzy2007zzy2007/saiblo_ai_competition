"""训练器：**位置头 ← 搜索访问分布（按类边缘化到格）** —— 计划 `docs/pos_seeded_sampling_plan.md` §3 第 2 步。

数据来源（`AZAI_SEED_STATE=1` 采集的真对局 dump，两者逐决策配对）：
  * policy JSONL：`cands` / `labels`（每候选的 op 串 `[类,x,y]`）/ **`visits_raw`（真访问计数）** / `chosen`；
  * obs 二进制：`board(28,19,19) f16 + stats(42,) f16 + player(u8) + chosen(3) + mask(24)`（`REC=20328`，偏移同 `ingest_bridge_dump.py`）。

目标构造：把 `visits_raw` 归一化后，按 **(类, x, y)** 累加（边缘化，`intent_counts` 设计给 AZ 用的同一件事 ✓），
取**质量最大**的那个类（≥2 个格）作为该行的监督：
  `row = (board, stats, 类 c, 格列表 [(x,y)...], 目标分布 p)`，`p` 为该类各格上的边缘质量（归一化 ✓）。
⚠️ 坐标约定：`labels` 里是 `(类, x, y)`，模型 `action_map` 索引是 `[类, x, y]` ✓（本会话已多次验证 ✓）。

损失：**软交叉熵** `CE(softmax(logits), p)`，`logits = action_map[c, x, y]`（该行候选格上）✓
—— **不做逐行 z-score** ✗（z-score 会改掉目标的真实温度：A②/ r1 配方都用的是 z-score 后的"有序软标签"，
而这里我们要**忠实拟合分布本身** ✓）。

只训 `pos_state`（warm start 自 incumbent，`class_state`/`value_state` 冻结 ✓），随机行切 ✓，
打印训练前的 train/val 基线（= incumbent 头对该目标的 CE/一致率 ✓，这就是 §9 闸门的对照 ✓）。

用法：
  python -u code/my_ai/az_intent/train_visit_marginal.py \
    --policy-dir data/seedstate_dump/az --obs-dir data/seedstate_dump/obs \
    --out checkpoints/grafted_sdk_A_visitmarg.pt --epochs 10
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BOARD_SHAPE = (28, 19, 19)
N_CLS = 24
BSZ = int(np.prod(BOARD_SHAPE)) * 2
REC = BSZ + 84 + 1 + 3 + N_CLS


def build_rows(policy_dir: Path, obs_dir: Path, max_games: int, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"[vm] 复用缓存 {cache}（{z['board'].shape[0]} 行）", flush=True)
        return z["board"], z["stats"], z["cls"], z["cell"], z["prob"], z["cnt"]
    boards, stats_l, clss, cells, probs, cnts = [], [], [], [], [], []
    files = sorted(glob.glob(str(policy_dir / "az_*.jsonl")))[:max_games]
    n_skip_single = n_skip_novis = 0
    for jf in files:
        tok = Path(jf).name.split("seed")[1].split(".")[0]
        of = obs_dir / f"obs_seed{tok}.bin"
        recs = []
        for line in open(jf, encoding="utf-8", errors="replace"):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
        if not of.exists():
            continue
        raw = np.fromfile(of, dtype=np.uint8)
        n_obs = raw.size // REC
        buf = raw[: n_obs * REC].reshape(n_obs, REC)
        for i in range(min(len(recs), n_obs)):
            d = recs[i]
            vr, labels = d.get("visits_raw"), d.get("labels")
            if not (isinstance(vr, list) and vr and labels):
                n_skip_novis += 1
                continue
            w = np.asarray([float(x) for x in vr], dtype=np.float64)
            s = w.sum()
            if s <= 0:
                n_skip_novis += 1
                continue
            p = w / s
            by = collections.defaultdict(lambda: collections.defaultdict(float))
            for k, bundle in enumerate(labels):
                if k >= len(p):
                    break
                for op in bundle:
                    c, x, y = int(op[0]), int(op[1]), int(op[2])
                    by[c][(x, y)] += float(p[k])
            if not by:
                continue
            c_best = max(by, key=lambda c: sum(by[c].values()))
            items = sorted(by[c_best].items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            if len(items) < 2:
                n_skip_single += 1
                continue
            cells.append([xy for xy, _ in items])
            probs.append([m for _, m in items])
            clss.append(c_best)
            cnts.append(len(items))
            boards.append(buf[i, :BSZ].copy().view(np.float16).reshape(BOARD_SHAPE))
            stats_l.append(buf[i, BSZ:BSZ + 84].copy().view(np.float16).reshape(42))
    print(f"[vm] 建集：{len(clss)} 行（跳过：无 visits_raw {n_skip_novis}，单格 {n_skip_single}）", flush=True)
    C = max(cnts)
    cell = np.full((len(clss), C, 2), -1, dtype=np.int16)
    prob = np.zeros((len(clss), C), dtype=np.float32)
    for r, (cs, ps) in enumerate(zip(cells, probs)):
        cell[r, :len(cs)] = np.asarray(cs, dtype=np.int16)
        prob[r, :len(ps)] = np.asarray(ps, dtype=np.float32)
    board = np.asarray(boards, dtype=np.float16)
    stats = np.asarray(stats_l, dtype=np.float16)
    cls = np.asarray(clss, dtype=np.int16)
    cnt = np.asarray(cnts, dtype=np.int16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, board=board, stats=stats, cls=cls, cell=cell, prob=prob, cnt=cnt)
    print(f"[vm] 缓存写出 {cache}", flush=True)
    return board, stats, cls, cell, prob, cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", default="data/seedstate_dump/az")
    ap.add_argument("--obs-dir", default="data/seedstate_dump/obs")
    ap.add_argument("--max-games", type=int, default=44)
    ap.add_argument("--cache", default="training_history/visitmarg/seedstate_rows.npz")
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--out", default="checkpoints/grafted_sdk_A_visitmarg.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from my_ai.az_intent.az_selfplay import load_three_models

    board, stats, cls, cell, prob, cnt = build_rows(Path(args.policy_dir), Path(args.obs_dir),
                                                    args.max_games, Path(args.cache))
    N = int(board.shape[0])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    print(f"[vm] N={N}，train={len(tr)}，val={len(va)}（随机切 ✓）；每行格数均值={cnt.mean():.1f}", flush=True)

    torch.manual_seed(args.seed)
    _cm, pm, _vm = load_three_models(args.ckpt)
    for p in pm.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(pm.parameters(), lr=args.lr)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pm.to(dev)
    base = {k: v.detach().clone() for k, v in pm.state_dict().items()}

    B = torch.from_numpy(np.asarray(board, dtype=np.float32))
    S = torch.from_numpy(np.asarray(stats, dtype=np.float32))
    CL = torch.from_numpy(np.asarray(cls, dtype=np.int64))
    CE_ = torch.from_numpy(np.asarray(cell, dtype=np.int64))
    P = torch.from_numpy(np.asarray(prob, dtype=np.float32))
    CN = torch.from_numpy(np.asarray(cnt, dtype=np.int64))

    def evaluate(idx):
        pm.eval()
        tot_ce = tot_n = tot_hit = tot_arg = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                am = pm(B[b].to(dev), S[b].to(dev))["action_map"]
                rows = torch.arange(len(b), device=dev).unsqueeze(1)
                cc = CL[b].to(dev).unsqueeze(1)
                y = CE_[b][..., 0].to(dev).clamp(min=0)
                x = CE_[b][..., 1].to(dev).clamp(min=0)
                lg = am[rows, cc, x, y]
                mask = torch.arange(CE_.shape[1], device=dev).unsqueeze(0) < CN[b].to(dev).unsqueeze(1)
                lg = torch.where(mask, lg, torch.full_like(lg, -1e9))
                tgt = torch.where(mask, P[b].to(dev), torch.zeros_like(lg))
                tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-9)
                tot_ce += float(-(tgt * F.log_softmax(lg, dim=1)).sum(1).sum())
                tot_n += len(b)
                tot_hit += float((lg.argmax(1) == tgt.argmax(1)).sum())
        return tot_ce / max(tot_n, 1), tot_hit / max(tot_n, 1)

    ce_tr, hit_tr = evaluate(tr)
    ce_va, hit_va = evaluate(va)
    print(f"[vm] **训练前 train**：CE={ce_tr:.4f} 与目标 argmax 一致={hit_tr*100:.2f}%", flush=True)
    print(f"[vm] **训练前 val**：  CE={ce_va:.4f} 与目标 argmax 一致={hit_va*100:.2f}%（N={len(va)}）", flush=True)

    best = (ce_va, 0)
    for ep in range(1, args.epochs + 1):
        pm.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            am = pm(B[b].to(dev), S[b].to(dev))["action_map"]
            rows = torch.arange(len(b), device=dev).unsqueeze(1)
            cc = CL[b].to(dev).unsqueeze(1)
            y = CE_[b][..., 0].to(dev).clamp(min=0)
            x = CE_[b][..., 1].to(dev).clamp(min=0)
            lg = am[rows, cc, x, y]
            mask = torch.arange(CE_.shape[1], device=dev).unsqueeze(0) < CN[b].to(dev).unsqueeze(1)
            lg = torch.where(mask, lg, torch.full_like(lg, -1e9))
            tgt = torch.where(mask, P[b].to(dev), torch.zeros_like(lg))
            tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-9)
            loss = -(tgt * F.log_softmax(lg, dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        ce_tr2, hit_tr2 = evaluate(tr)
        ce_va2, hit_va2 = evaluate(va)
        with torch.no_grad():
            dnorm = float(np.sqrt(sum(float(((pm.state_dict()[k2] - base[k2]) ** 2).sum())
                                     for k2 in base)))
        print(f"[vm] epoch {ep:2d}/{args.epochs}  train CE={ce_tr2:.4f} 一致={hit_tr2*100:.2f}% "
              f"| val CE={ce_va2:.4f} 一致={hit_va2*100:.2f}% | 参数变化范数={dnorm:.3f}", flush=True)
        if ce_va2 <= best[0]:
            best = (ce_va2, ep)
            # ⚠️ 2026-10-12 修（冒烟 0 建塔的根因 ✗）：必须**保留原 ckpt 的全部顶层键** ——
            #   `no_bn`/`gn`/`gn_groups`/`latent_dim`/`num_resblocks`/`three_net` … 是**架构元数据**，
            #   丢了就会被 `load_three_models` / `make_net_fn_from_ckpt` 按**默认架构**重建
            #   ⇒ 加载失败或半随机 ⇒ 一局 0 建塔（冒烟守卫拦下 ✓）。
            _out = dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
            _out["class_state"] = _cm.state_dict()
            _out["pos_state"] = pm.state_dict()
            _out["value_state"] = _vm.state_dict()
            _out["visitmarg_note"] = "pos_state ← 搜索访问分布（边缘化到格）软 CE 蒸馏；class/value 逐位不变"
            torch.save(_out, args.out)
    print(f"[vm] 结束：best val CE={best[0]:.4f}（epoch {best[1]}）⇒ {args.out}", flush=True)
    print(f"[vm] **§9 闸门对照**：训练前 val CE={ce_va:.4f} / 一致率={hit_va*100:.2f}%  "
          f"⇒ 需 CE 下降 ≥0.10 且一致率超过 incumbent 头才算过关 ✓", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
