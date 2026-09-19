"""(a) **只训价值网**：三网 ckpt 进，`class_state`/`pos_state` 原样照抄，只有 `value_state` 会变。

为什么另写（计划：`docs/az_class_axis_plan.md` §5(a-1)）：
  * `az_train.py --value-only` 只在 `--split` 分支有效，而 `--split` 的 `load_split_models`
    **明确拒绝三网 ckpt**（`_guard_no_three_net`）；
  * `--pos-only-net --train-value` 会连 `pos_state` 一起改 ⇒ `_tmp_rank_quality.py` 里的
    对局就不是同一批了，A/B 立刻不干净。
照 `train_value_prior.py` 的模子（三网进、只训一个网、另两网照抄），只是把"位置"换成"价值"。

**A/B 干净性**：同一份 `class_state`/`pos_state` ⇒ 判据脚本里价值网看到的对局逐位相同，
唯一变量就是价值网。

标签：默认 `--label-mode terminal` = 采集时已存的 `value_target`（每局终局 HP 差、clip ±1、
按 player 签名），与产出当前 `value_state` 的 value_warmup 配方对齐；可选
`rel/abs/mix` 走 `az_train.add_weighted_labels`（逐局流式调用，不新增实现）。

用法:
    PY=D:/anaconda3/envs/pytorch-gpu/python.exe; S=code/my_ai/az_intent/train_value_net.py
    # 0) 建紧凑缓存（一次；17 GB pkl -> board/stats/player/value_target 的 memmap）
    $PY -u $S --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt \
        --data training_history/inject_ex02/data \
        --cache training_history/inject_ex02/value_cache --build-cache-only
    # 1) 只评估（不训）：按"有塔/无塔"分组看误差 —— 直接量"价值网会不会判有塔的局面"
    $PY -u $S --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt \
        --cache training_history/inject_ex02/value_cache --eval-only
    # 2) 训
    $PY -u $S --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt \
        --cache training_history/inject_ex02/value_cache \
        --epochs 4 --lr 3e-4 --out training_history/inject_ex02/valnet_r1.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CH_OWN, CH_ENEMY = 4, 5          # board 通道：4=己方塔，5=敌方塔（SDK/utils/features.py:177-180）


# ----------------------------------------------------------------------------- cache

def _cache_one(job):
    """一个 pkl -> (board float16, stats float16, player int8, value_target float32) 的 npz。"""
    src, dst = job
    d = pickle.load(open(src, "rb"))
    samples = d["samples"] if isinstance(d, dict) else d
    board = np.stack([np.asarray(s["board"], dtype=np.float16) for s in samples])
    stats = np.stack([np.asarray(s["stats"], dtype=np.float16) for s in samples])
    player = np.asarray([int(s["player"]) for s in samples], dtype=np.int8)
    vt = np.asarray([float(s["value_target"]) for s in samples], dtype=np.float32)
    np.savez(dst, board=board, stats=stats, player=player, value_target=vt)
    return Path(dst).name, len(samples)


def build_cache(data_dir: str, cache: str, workers: int) -> None:
    """把 pkl 目录压成一个大 memmap（board/stats/player/value_target）。

    分两趟：① 并行逐局压成小 npz（每个 worker 只驻留一局）；② 顺序拼进一个 memmap。
    这样内存占用与局数无关（400 局全读进 RAM 要 ~7.5 GB）。
    """
    import multiprocessing as mp
    srcs = sorted(str(p) for p in Path(data_dir).glob("az_selfplay_seed*.pkl"))
    if not srcs:
        raise SystemExit(f"[vcache] {data_dir} 下没有 az_selfplay_seed*.pkl")
    out = Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    print(f"[vcache] ① {len(srcs)} 局 -> 逐局 npz（{workers} workers）", flush=True)
    t0 = time.time()
    jobs = [(s, str(tmp / (Path(s).stem + ".npz"))) for s in srcs]
    with mp.Pool(workers) as pool:
        res = pool.map(_cache_one, jobs)
    print(f"[vcache] ① 完成 {len(res)} 局，用时 {time.time()-t0:.0f}s", flush=True)

    counts = [n for _, n in res]
    order = [d for _, d in jobs]                    # 目标 npz 路径（与 srcs/res 同序）
    total = int(sum(counts))
    print(f"[vcache] 总计 {total} 个决策点", flush=True)

    z0 = np.load(order[0])                          # 形状探测
    b0 = z0["board"]
    C, H, W = b0.shape[1:]
    S = z0["stats"].shape[1]
    print(f"[vcache] board=({total},{C},{H},{W}) stats=({total},{S})", flush=True)

    mm_b = np.lib.format.open_memmap(out / "board.npy", mode="w+", dtype=np.float16,
                                     shape=(total, C, H, W))
    mm_s = np.lib.format.open_memmap(out / "stats.npy", mode="w+", dtype=np.float16,
                                     shape=(total, S))
    mm_p = np.lib.format.open_memmap(out / "player.npy", mode="w+", dtype=np.int8,
                                     shape=(total,))
    mm_v = np.lib.format.open_memmap(out / "value_target.npy", mode="w+", dtype=np.float32,
                                     shape=(total,))
    off = 0
    for i, path in enumerate(order):
        z = np.load(path)
        n = counts[i]
        mm_b[off:off + n] = z["board"]
        mm_s[off:off + n] = z["stats"]
        mm_p[off:off + n] = z["player"]
        mm_v[off:off + n] = z["value_target"]
        off += n
        if (i + 1) % 50 == 0:
            print(f"  [vcache] ② {i+1}/{len(order)}", flush=True)
    mm_b.flush(); mm_s.flush(); mm_p.flush(); mm_v.flush()
    np.savez(out / "meta.npz", counts=np.asarray(counts, dtype=np.int64),
             names=np.asarray([Path(p).stem for p in order]))
    print(f"[vcache] ② 完成：{out}/board.npy 等 4 个 memmap（共 {total} 行）", flush=True)


def load_cache(cache: str):
    out = Path(cache)
    b = np.load(out / "board.npy", mmap_mode="r")
    s = np.load(out / "stats.npy", mmap_mode="r")
    p = np.load(out / "player.npy", mmap_mode="r")
    v = np.load(out / "value_target.npy", mmap_mode="r")
    meta = np.load(out / "meta.npz", allow_pickle=True)
    return b, s, p, v, meta


# ----------------------------------------------------------------------------- train

def _batch(b, s, v, idx, dev):
    bb = torch.from_numpy(np.asarray(b[idx], dtype=np.float32)).to(dev, non_blocking=True)
    ss = torch.from_numpy(np.asarray(s[idx], dtype=np.float32)).to(dev, non_blocking=True)
    vv = torch.from_numpy(np.asarray(v[idx], dtype=np.float32)).to(dev, non_blocking=True)
    return bb, ss, vv


@torch.no_grad()
def evaluate(model, b, s, p, v, idx, dev, batch_size: int) -> dict:
    """返回整体/分组 MSE + 与目标的相关。分组 = 该样本**场上有没有己方塔**。"""
    model.eval()
    own = np.asarray(b[idx, CH_OWN] != 0).any(axis=(1, 2))
    se = np.zeros(len(idx), dtype=np.float64)
    pr = np.zeros(len(idx), dtype=np.float64)
    tg = np.zeros(len(idx), dtype=np.float64)
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        bb, ss, vv = _batch(b, s, v, j, dev)
        out = model(bb, ss)["value"].squeeze(-1)
        pr[i:i + len(j)] = out.detach().cpu().numpy()
        tg[i:i + len(j)] = vv.cpu().numpy()
    se = (pr - tg) ** 2
    res = {"n": len(idx), "mse": float(se.mean()), "var": float(tg.var())}
    res["corr"] = float(np.corrcoef(pr, tg)[0, 1]) if len(idx) > 2 else float("nan")
    for name, m in (("有塔", own), ("无塔", ~own)):
        if m.any():
            res[f"mse_{name}"] = float(se[m].mean())
            res[f"n_{name}"] = int(m.sum())
            res[f"corr_{name}"] = (float(np.corrcoef(pr[m], tg[m])[0, 1])
                                   if m.sum() > 2 else float("nan"))
    return res


def fmt(tag: str, r: dict) -> str:
    return ("%-14s n=%6d  MSE=%.5f (常数基线 %.5f)  r=%+.3f | 有塔 n=%6d MSE=%.5f r=%+.3f"
            " | 无塔 n=%6d MSE=%.5f r=%+.3f"
            % (tag, r["n"], r["mse"], r["var"], r["corr"],
               r.get("n_有塔", 0), r.get("mse_有塔", float("nan")), r.get("corr_有塔", float("nan")),
               r.get("n_无塔", 0), r.get("mse_无塔", float("nan")), r.get("corr_无塔", float("nan"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="源三网 ckpt（class/pos 照抄、value 起点）")
    ap.add_argument("--data", default=None, help="az_selfplay pkl 目录（建缓存时用）")
    ap.add_argument("--cache", required=True, help="紧凑缓存目录（--build-cache-only 时是输出）")
    ap.add_argument("--build-cache-only", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true", help="只评估 --ckpt，不训练")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-games", type=int, default=20, help="留出的验证**局数**（按局切，防泄漏）")
    ap.add_argument("--label-mode", type=str, default="terminal",
                    choices=["terminal", "rel", "abs", "mix"])
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--label-scale", type=float, default=1.0)
    ap.add_argument("--label-mix-alpha", type=float, default=0.5)
    ap.add_argument("--label-weight", type=str, default="geo", choices=["geo", "kgeo"])
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="冻结 initial_conv+resblocks，只训头")
    ap.add_argument("--anchor-lambda", type=float, default=0.0,
                    help="参数空间 L2 锚回 value_state 初值")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", default=None, help="输出的新三网 ckpt")
    args = ap.parse_args()

    if args.build_cache_only:
        if not args.data:
            raise SystemExit("--build-cache-only 需要 --data")
        build_cache(args.data, args.cache, args.workers)
        return

    dev_s = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_s)
    torch.manual_seed(args.seed)
    print(f"[vnet] device={dev}  ckpt={args.ckpt}  cache={args.cache}", flush=True)

    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.az_train import save_three_net, add_weighted_labels

    b, s, p, v, meta = load_cache(args.cache)
    counts = meta["counts"]
    N = int(counts.sum())
    print(f"[vnet] 缓存 {N} 个决策点 / {len(counts)} 局", flush=True)

    # 按局切分（同局样本高度相关，按样本切会泄漏）
    rng = np.random.default_rng(args.seed)
    gi = rng.permutation(len(counts))
    val_g = set(int(x) for x in gi[:args.val_games])
    offs = np.concatenate([[0], np.cumsum(counts)])
    val_idx = np.concatenate([np.arange(offs[g], offs[g + 1]) for g in sorted(val_g)])
    tr_idx = np.concatenate([np.arange(offs[g], offs[g + 1])
                             for g in range(len(counts)) if g not in val_g])
    print(f"[vnet] train {len(tr_idx)} 决策点 / val {len(val_idx)} 决策点（{args.val_games} 局）",
          flush=True)

    v_src = np.asarray(v[:], dtype=np.float32)
    if args.label_mode == "terminal":
        labels = v_src.copy()
    else:
        # 非 terminal：逐局把该局的 stats 序列喂给 az_train.add_weighted_labels（复用现成配方，
        # 不新增实现）。⚠️ 它按**局内时序**前缀和递推，所以必须逐局切、不能跨局拼。
        # 动机：terminal 标签**一局之内所有决策点同值** ⇒ 对"动作之间的差别"是零信号；
        # abs/rel 用局内的未来 HP 轨迹 ⇒ 每个决策点有各自的标签。
        labels = np.zeros(N, dtype=np.float32)
        st_all = np.asarray(s, dtype=np.float32)
        pl_all = np.asarray(p, dtype=np.int64)
        for g in range(len(counts)):
            sl = slice(int(offs[g]), int(offs[g + 1]))
            shim = [{"stats": st_all[i], "player": int(pl_all[i])} for i in range(sl.start, sl.stop)]
            add_weighted_labels(shim, tau=args.tau, label_scale=args.label_scale,
                                label_mode=args.label_mode,
                                mix_alpha=args.label_mix_alpha,
                                label_weight=args.label_weight)
            labels[sl] = np.asarray([x["value_label"] for x in shim], dtype=np.float32)
    print(f"[vnet] 标签 mode={args.label_mode} weight={args.label_weight} tau={args.tau} "
          f"scale={args.label_scale}  mean={labels.mean():+.4f} std={labels.std():.4f} "
          f"range=[{labels.min():+.3f},{labels.max():+.3f}]", flush=True)
    lab = np.ascontiguousarray(labels, dtype=np.float32)   # 训练/评估都用它当目标

    src = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    class_model, pos_model, value_model = load_three_models(args.ckpt)
    class_model.eval(); pos_model.eval()
    for mm in (class_model, pos_model):
        for q in mm.parameters():
            q.requires_grad = False
    value_model.to(dev)
    n_frozen = 0
    if args.freeze_backbone:
        for name, q in value_model.named_parameters():
            if name.startswith("initial_conv") or name.startswith("resblocks"):
                q.requires_grad = False
                n_frozen += q.numel()
    print(f"[vnet] 骨干冻结 {n_frozen:,} 参数；可训 {sum(q.numel() for q in value_model.parameters() if q.requires_grad):,}",
          flush=True)

    base = evaluate(value_model, b, s, p, lab, val_idx, dev, args.batch_size)
    print("[vnet] **训练前** " + fmt("val", base), flush=True)
    if args.eval_only:
        return

    init_ref = None
    if args.anchor_lambda > 0:
        init_ref = [q.detach().clone() for q in value_model.parameters()]
    opt = torch.optim.Adam([q for q in value_model.parameters() if q.requires_grad], lr=args.lr)

    best = (base["mse"], -1)
    if args.out:
        save_three_net(args.out, class_model, pos_model, value_model,
                       {**{k: val for k, val in src.items() if isinstance(val, (int, float, str))},
                        "valnet_src": args.ckpt, "valnet_label": args.label_mode,
                        "valnet_freeze_backbone": int(args.freeze_backbone),
                        "valnet_anchor": args.anchor_lambda, "valnet_epoch": 0,
                        "valnet_cache": args.cache})
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        value_model.train()
        order = np.random.default_rng(args.seed + ep).permutation(len(tr_idx))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            j = tr_idx[order[i:i + args.batch_size]]
            bb, ss, vv = _batch(b, s, lab, j, dev)
            out = value_model(bb, ss)["value"].squeeze(-1)
            loss = torch.nn.functional.mse_loss(out, vv)
            opt.zero_grad()
            if init_ref is not None:
                reg = sum(((q - q0) ** 2).sum() for q, q0 in
                          zip(value_model.parameters(), init_ref))
                loss = loss + args.anchor_lambda * reg
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        r = evaluate(value_model, b, s, p, lab, val_idx, dev, args.batch_size)
        flag = ""
        if r["mse"] < best[0]:
            best = (r["mse"], ep)
            if args.out:
                save_three_net(args.out, class_model, pos_model, value_model,
                               {**{k: val for k, val in src.items()
                                   if isinstance(val, (int, float, str))},
                                "valnet_src": args.ckpt, "valnet_label": args.label_mode,
                                "valnet_freeze_backbone": int(args.freeze_backbone),
                                "valnet_anchor": args.anchor_lambda, "valnet_epoch": ep,
                                "valnet_cache": args.cache})
            flag = "  <- 保存"
        print("[vnet] epoch %2d/%d  train MSE=%.5f | %s  (%.0fs)%s"
              % (ep, args.epochs, tot / max(nb, 1), fmt("val", r), time.time() - t0, flag),
              flush=True)

    print(f"[vnet] 最佳 val MSE=%.5f @epoch %d  ->  %s" % (best[0], best[1], args.out), flush=True)
    print("[vnet] 判据（_tmp_rank_quality.py 默认口径）：价值网分位 > 56.4%、ρ > 0.139、"
          "随机分位仍 ≈49.6%", flush=True)


if __name__ == "__main__":
    main()
