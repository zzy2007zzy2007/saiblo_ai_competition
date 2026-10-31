"""值网「辅助双头」训练（aux2-head，2026-11-01 用户提议）。

## 为什么

现役值网**抄 `hp_delta`**（`stats[1]` 就是当前血差；`docs/az_pool_tau_label_plan.md:89-111` 量化过
tau50-abs 标签有 ~87% 方差可由它解释）。用户的想法：把"未来加权血"拆成**我方/敌方两个标量**分别预测，
逼值网分别建模"我会不会被打崩 / 我能不能清掉对方"——而这两件事都依赖金币/冷却/闪电，
所以**可能**间接逼出"金币×冷却"的定价。与 `rel` 标签（改标签削捷径）是**互补**的两条路。

## 结构

骨干与价值通路**完全不动**（`AntWarNetwork`，`code/my_ai/network.py:236-283`）：
    state_emb(128) → value_head[0] Linear(128,64) → ReLU → h(64)
在 `h` 上分叉出三个头：
    V          = value_head[2](h)      ← 主头，就是部署时用的 `value_state`
    aux_self   = Linear(64,1)(h)       ← 预测"我方未来 kgeo 加权血量 / 50"
    aux_enemy  = Linear(64,1)(h)       ← 预测"敌方未来 kgeo 加权血量 / 50"

损失：`MSE(V, label) + λ·[MSE(aux_self, raw_self) + MSE(aux_enemy, raw_enemy)]`
  · 主标签 `label` 与 `abs+kgeo τ50` 臂**逐位一致**（脚本自带断言：`label == clip(2.5·(aux_self−aux_enemy), ±1)·scale`）
  · `raw_self/raw_enemy` 用 **HP/50** 的量纲（与 `stats[24]/[25]` 同尺度），两者之差 ×2.5 就是 `/20` 的血差标签
  · **辅助头只在训练时存在，保存时丢弃** ⇒ 产出的三网 ckpt 与现役**同构**，可直接部署/被探针消费

## 口径细节（容易踩的）

样本是 P0/P1 **交错**存的（一回合 2 条）⇒ "我方血量"这条序列在样本轴上会左右跳。
所以先把每帧还原成**物理**的 `HP0/HP1`（由 `stats[24]/[25]` 按 `player` 摆正），
再对 `HP0`、`HP1` 各自做 kgeo 加权未来平均，最后按该样本的 `player` 取 `self/enemy`。

用法：
    python -m my_ai.az_intent.train_value_aux2 --ckpt checkpoints/posnet_A_k5_m32.pt \
        --cache training_history/value_retrain_v1/cache_snap1 --epochs 3 --lr 3e-4 \
        --tau 50 --label-scale 1.5 --aux-lambda 0.5 \
        --out training_history/value_retrain_v1/arm_aux2_kgeo50.pt --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HP_SCALE = 20.0
HP_DIV = 50.0          # stats[24]/[25] 是 hp/50


def _kgeo_wf(x: np.ndarray, tau: float) -> np.ndarray:
    """kgeo 加权未来平均：`Σ k·γ^k·x_{t+k} / Σ k·γ^k`（w_0 = 0，末帧退化回 x_t）。"""
    n = len(x)
    g = float(np.exp(-1.0 / tau))
    out = np.empty(n, dtype=np.float64)
    A = B = WA = WB = 0.0
    for t in range(n - 1, -1, -1):
        B = g * (B + A)
        WB = g * (WB + WA)
        A = x[t] + g * A
        WA = 1.0 + g * WA
        out[t] = (B / WB) if WB > 1e-9 else x[t]
    return out


def _labels_one_game(stats: np.ndarray, player: np.ndarray, tau: float, label_scale: float):
    """-> (label, aux_self, aux_enemy)，三者都已按该样本的 player 视角摆好。

    **与用户 2026-11-01 的定义逐字对齐**（`w_k = k·γ^k`，tau 与主标签相同）：

        label     = Σ w_k·(HP_self − HP_enemy)_{t+k} / Σ w_k        （= abs+kgeo 主标签）
        aux_self  = Σ w_k·HP_self_{t+k} / Σ w_k
        aux_enemy = Σ w_k·HP_enemy_{t+k} / Σ w_k

    恒有 `label = aux_self − aux_enemy`（辅助头只是"不取差"，核/窗口/tau 完全相同）。
    ⚠️ 两个实现细节：
      1) **血值取整数精确值**：`stats[24] = hp/50` 是 float16（有 ~5e-3 HP 的量化误差），
         所以先 `round(50·stats[24])` 还原**整数基地血**，再用 `stats[1]`（`hp_delta`，整数、精确）算敌方：
         `HP_enemy = HP_self − hp_delta`。这样主标签与 `az_train`（只用 `stats[1]`）**逐位一致**。
      2) 辅助头的目标量纲取 **HP/50**（与 `stats[24]` 同尺度）；否则 HP∈[0,50] 的 MSE ~O(100)，
         λ=0.5 会让辅助项直接压过主项。折算回来 `label = 2.5·(aux_self − aux_enemy)`（`(·)/20` 与 `/50`）。
    """
    pl = np.asarray(player, dtype=np.int64)
    hp_delta = np.asarray(stats[:, 1], dtype=np.float64)                # player 视角的精确血差
    hp_self = np.round(np.asarray(stats[:, 24], dtype=np.float64) * HP_DIV)   # 还原整数基地血
    hp_enemy = hp_self - hp_delta                                        # 精确（同为整数）
    # 摆正成物理序列 HP0/HP1（样本交错存，直接拿 self 序列会左右跳）
    hp0 = np.where(pl == 0, hp_self, hp_enemy)
    hp1 = np.where(pl == 0, hp_enemy, hp_self)
    wf0 = _kgeo_wf(hp0, tau)
    wf1 = _kgeo_wf(hp1, tau)
    wf_self = np.where(pl == 0, wf0, wf1)
    wf_enemy = np.where(pl == 0, wf1, wf0)
    label = np.clip((wf_self - wf_enemy) / HP_SCALE, -1.0, 1.0) * label_scale
    return (label.astype(np.float32), (wf_self / HP_DIV).astype(np.float32),
            (wf_enemy / HP_DIV).astype(np.float32))


def _split(counts: np.ndarray, val_games: int, seed: int):
    rng = np.random.default_rng(seed)
    gi = rng.permutation(len(counts))
    val_g = set(int(x) for x in gi[:val_games])
    offs = np.concatenate([[0], np.cumsum(counts)])
    val_idx = np.concatenate([np.arange(offs[g], offs[g + 1]) for g in sorted(val_g)])
    tr_idx = np.concatenate([np.arange(offs[g], offs[g + 1])
                             for g in range(len(counts)) if g not in val_g])
    return tr_idx.astype(np.int64), val_idx.astype(np.int64), offs


def _set_bn_eval(model) -> int:
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval(); n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="源三网 ckpt（class/pos 照抄、value 起点）")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=50.0)
    ap.add_argument("--label-scale", type=float, default=1.5)
    ap.add_argument("--aux-lambda", type=float, default=0.5)
    ap.add_argument("--val-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device if args.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    from my_ai.az_intent.train_value_net import load_cache
    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.az_train import save_three_net

    b, s, p, v, meta = load_cache(args.cache)
    counts = meta["counts"]
    N = int(counts.sum())
    tr_idx, val_idx, offs = _split(counts, args.val_games, args.seed)
    print(f"[aux2] cache={args.cache} N={N} games={len(counts)} "
          f"train={len(tr_idx)} val={len(val_idx)}（{args.val_games} 局）", flush=True)

    # ── 标签：主标签（abs+kgeo）+ 两个辅助目标 ────────────────────────────
    lab = np.zeros(N, dtype=np.float32)
    a_self = np.zeros(N, dtype=np.float32)
    a_enemy = np.zeros(N, dtype=np.float32)
    for g in range(len(counts)):
        sl = slice(int(offs[g]), int(offs[g + 1]))
        st = np.asarray(s[sl], dtype=np.float32)
        pl = np.asarray(p[sl], dtype=np.int64)
        y, ys, ye = _labels_one_game(st, pl, args.tau, args.label_scale)
        lab[sl], a_self[sl], a_enemy[sl] = y, ys, ye
    # 自检：主标签必须与训练器 `abs+kgeo`（同一 tau/scale）逐位一致
    try:
        from my_ai.az_intent.az_train import add_weighted_labels
        sl = slice(int(offs[0]), int(offs[1]))
        shim = [{"stats": np.asarray(s[sl], dtype=np.float32)[i],
                 "player": int(np.asarray(p[sl])[i])} for i in range(int(counts[0]))]
        add_weighted_labels(shim, tau=args.tau, label_scale=args.label_scale,
                            label_mode="abs", label_weight="kgeo")
        ref = np.asarray([x["value_label"] for x in shim], dtype=np.float32)
        d = float(np.abs(ref - lab[sl]).max())
        print(f"[aux2] 自检：主标签 vs az_train(abs+kgeo, tau={args.tau:g}, "
              f"scale={args.label_scale}): max|diff| = {d:.2e}" + ("  ✓" if d < 1e-5 else "  ✗"))
        if d >= 1e-5:
            raise SystemExit("主标签与训练器不一致")
    except ImportError:
        print("[aux2] 自检跳过（导入失败）")
    print(f"[aux2] label mean/std={lab.mean():+.4f}/{lab.std():.4f}  "
          f"aux_self std={a_self.std():.4f} aux_enemy std={a_enemy.std():.4f}", flush=True)

    # ── 模型：现役三网 → 只训 value 分支 + 两个辅助头 ──────────────────────
    class_model, pos_model, value_model = load_three_models(args.ckpt)
    class_model.eval(); pos_model.eval()
    for m in (class_model, pos_model):
        for q in m.parameters():
            q.requires_grad = False
    value_model.to(dev)
    # 辅助头挂在 value_head 的隐藏层 h(64) 上；保存时丢弃 ⇒ ckpt 与现役同构
    aux_self_head = torch.nn.Linear(value_model.value_head[0].out_features, 1).to(dev)
    aux_enemy_head = torch.nn.Linear(value_model.value_head[0].out_features, 1).to(dev)
    params = list(value_model.parameters()) + list(aux_self_head.parameters()) \
        + list(aux_enemy_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    print(f"[aux2] 可训参数 {sum(q.numel() for q in params):,}"
          f"（含两个辅助头各 {value_model.value_head[0].out_features + 1}）", flush=True)

    def forward_v(board_t, stats_t):
        out = value_model(board_t, stats_t)
        h = F.relu(value_model.value_head[0](out["state_emb"]))
        return value_model.value_head[2](h).squeeze(-1), \
            aux_self_head(h).squeeze(-1), aux_enemy_head(h).squeeze(-1)

    @torch.no_grad()
    def evaluate(idx):
        value_model.eval()
        pr = np.zeros(len(idx)); tg = np.zeros(len(idx))
        aux_s = aux_e = 0.0
        for i in range(0, len(idx), args.batch_size):
            j = idx[i:i + args.batch_size]
            bb = torch.from_numpy(np.asarray(b[j], dtype=np.float32)).to(dev)
            ss = torch.from_numpy(np.asarray(s[j], dtype=np.float32)).to(dev)
            vv = torch.from_numpy(lab[j]).to(dev)
            t_s = torch.from_numpy(a_self[j]).to(dev)
            t_e = torch.from_numpy(a_enemy[j]).to(dev)
            o, o_s, o_e = forward_v(bb, ss)
            pr[i:i + len(j)] = o.cpu().numpy(); tg[i:i + len(j)] = vv.cpu().numpy()
            # ⚠️ `reduction="sum"` 已经是这个 batch 的总和，**不要再乘 batch 大小**
            aux_s += float(F.mse_loss(o_s, t_s, reduction="sum"))
            aux_e += float(F.mse_loss(o_e, t_e, reduction="sum"))
        n = max(len(idx), 1)
        return {"n": len(idx), "mse": float(((pr - tg) ** 2).mean()),
                "var": float(tg.var()),
                "corr": float(np.corrcoef(pr, tg)[0, 1]) if len(idx) > 2 else float("nan"),
                "aux_self_mse": aux_s / n, "aux_enemy_mse": aux_e / n,
                "aux_self_var": float(np.var(a_self[idx])),
                "aux_enemy_var": float(np.var(a_enemy[idx]))}

    def fmt(tag, r):
        return (f"{tag:<12} n={r['n']:6d} 主头 MSE={r['mse']:.5f} (常数基线 {r['var']:.5f}) "
                f"r={r['corr']:+.3f} | aux_self MSE={r['aux_self_mse']:.5f}"
                f"(var {r['aux_self_var']:.4f}) aux_enemy MSE={r['aux_enemy_mse']:.5f}"
                f"(var {r['aux_enemy_var']:.4f})")

    base = evaluate(val_idx)
    print("[aux2] **训练前** " + fmt("val", base), flush=True)
    best = (base["mse"], -1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        value_model.train()
        _set_bn_eval(value_model)          # 源值网 BN 从未训过 ⇒ 冻结 running 统计
        order = np.random.default_rng(args.seed + ep).permutation(len(tr_idx))
        tot = nll = 0.0; nb = 0
        for i in range(0, len(order), args.batch_size):
            j = tr_idx[order[i:i + args.batch_size]]
            bb = torch.from_numpy(np.asarray(b[j], dtype=np.float32)).to(dev)
            ss = torch.from_numpy(np.asarray(s[j], dtype=np.float32)).to(dev)
            vv = torch.from_numpy(lab[j]).to(dev)
            ae = torch.from_numpy(a_self[j]).to(dev)
            an = torch.from_numpy(a_enemy[j]).to(dev)
            o, o_s, o_e = forward_v(bb, ss)
            l_main = F.mse_loss(o, vv)
            l_aux = F.mse_loss(o_s, ae) + F.mse_loss(o_e, an)
            loss = l_main + args.aux_lambda * l_aux
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(l_main.detach()); nll += float(l_aux.detach()); nb += 1
        r = evaluate(val_idx)
        flag = ""
        if r["mse"] < best[0]:
            best = (r["mse"], ep)
            save_three_net(str(out_path), class_model, pos_model, value_model,
                           {"valnet_src": args.ckpt, "valnet_label": f"aux2_abs_kgeo_tau{args.tau:g}",
                            "valnet_label_scale": args.label_scale, "valnet_aux_lambda": args.aux_lambda,
                            "valnet_epoch": ep, "valnet_cache": args.cache,
                            "valnet_aux_heads": "dropped_on_save"})
            flag = "  <- 保存"
        print(f"[aux2] epoch {ep:2d}/{args.epochs} train 主头={tot / max(nb,1):.5f} "
              f"aux={nll / max(nb,1):.5f} | {fmt('val', r)}  ({time.time()-t0:.0f}s){flag}", flush=True)
    print(f"[aux2] 最佳 val 主头 MSE={best[0]:.5f} @epoch {best[1]} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
