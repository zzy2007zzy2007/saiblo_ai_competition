#!/usr/bin/env python3
"""把 `y_std/y_mean` 折进 `value_head` 最后一层 ⇒ 产出"可直接当现役值网用"的 ckpt。

## 为什么必须折

`ValueNetwork` 是在 **z-score 标签** `(y - y_mean) / y_std` 上训练的（本次 `y_std=0.3861`、
`y_mean=-0.0004`），而部署侧 `AZAI_VALUE_TANH=1` **只做 tanh、不做反归一化**
（`flat_rollout_greedy._leaf_values` / `value_batch.make_batched_value_fn`）。
不折就把输出放大了 `1/0.386 ≈ 2.6×` ⇒ `|v|>0.5` 的叶子全被 tanh 压到 `|tanh|>0.86`、彼此几乎
分不开，还会和"终局叶子（±1）"撞在一起 ⇒ **排序被压平**（= `docs/value_net_pipeline_survey.md` §5.3
那个历史事故）。折法与 `graft_value_ckpt.py:68-76` 完全一致：

    W' = W * y_std        b' = b * y_std + y_mean        （value_head.2）

## 自检（两条，缺一不可）

1. **等价性（强判据）**：`folded(x)` 与 `raw(x) * y_std + y_mean` 必须逐位相等
   （`max|Δ| < 1e-4`，只允许 float32 舍入）。这条与数据无关，是"折对了"的**充分**证据。
2. **与老师比**：抽 `--check-frames` 帧留出集，`Spearman(tanh(folded), tanh(teacher))` 应
   ≈ 0.96（与 `docs/small_value_net_20261027.md` §3 的 0.9603 同口径）。
   同时打印**未折**版本的对照（Spearman 不变、Pearson/R² 会错、tanh 提前饱和）——
   这个坑必须可见。

用法：
  python code/test_match/fold_small_value_ckpt.py \
      --student-ckpt checkpoints/small32_3/lowq_value_seed0.pt \
      --out checkpoints/small32_3/v_32_3_folded.pt --check-frames 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.value_network import ValueNetwork                      # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models         # noqa: E402


def spearman(a, b) -> float:
    """秩相关（照抄 `code/test_match/eval_small_value.py`）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def pearson(a, b) -> float:
    a = np.asarray(a, float) - float(np.mean(a))
    b = np.asarray(b, float) - float(np.mean(b))
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den else float("nan")


def r2(y, p) -> float:
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def build_from_ckpt(path: Path) -> tuple[ValueNetwork, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    net = ValueNetwork(latent_dim=int(cfg.get("latent_dim", 64)),
                       num_resblocks=int(cfg.get("num_resblocks", 6)))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def fold_state_dict(sd: dict, y_std: float, y_mean: float, *, inplace: bool = False) -> dict:
    """`out_new = out_old * y_std + y_mean` 折进 `value_head.2`（照 graft_value_ckpt.py:68-76）。"""
    out = sd if inplace else {k: v.clone() for k, v in sd.items()}
    wk, bk = "value_head.2.weight", "value_head.2.bias"
    if wk not in out or bk not in out:
        raise KeyError(f"ckpt 里没有 {wk}/{bk}，无法折归一化")
    if str(out[wk].dtype) == "torch.float32":
        out[wk] = out[wk] * float(y_std)
        out[bk] = out[bk] * float(y_std) + float(y_mean)
    else:  # 理论上不会遇到
        out[wk] = (out[wk].double() * float(y_std)).to(out[wk].dtype)
        out[bk] = (out[bk].double() * float(y_std) + float(y_mean)).to(out[bk].dtype)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", default="checkpoints/small32_3/lowq_value_seed0.pt")
    ap.add_argument("--out", default="checkpoints/small32_3/v_32_3_folded.pt")
    ap.add_argument("--teacher-ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--data", default="training_history/distill_top2/lowq_f64")
    ap.add_argument("--label-file", default="label.npy")
    ap.add_argument("--val-games", type=int, default=500,
                    help="按局切留出集（最后多少局），与训练/评估同口径")
    ap.add_argument("--check-frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))

    def _abs(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (_REPO / q)

    sp, op, tp, dp = _abs(a.student_ckpt), _abs(a.out), _abs(a.teacher_ckpt), _abs(a.data)

    # ── ① 读学生 ckpt、造折叠后的 state_dict ────────────────────────────────
    student, ck = build_from_ckpt(sp)
    cfg = ck.get("config") or {}
    sd_raw = student.state_dict()
    y_std = float(ck.get("y_std", 1.0))
    y_mean = float(ck.get("y_mean", 0.0))
    print(f"[fold] student={sp}")
    print(f"[fold] ValueNetwork(latent_dim={cfg.get('latent_dim', 64)}, "
          f"num_resblocks={cfg.get('num_resblocks', 6)}) "
          f"params={student.count_parameters():,}")
    print(f"[fold] y_std={y_std:.6f} y_mean={y_mean:.6f} "
          f"| 折前 value_head.2.weight[0,0]={float(sd_raw['value_head.2.weight'].flatten()[0]):.6f} "
          f"bias={float(sd_raw['value_head.2.bias'].flatten()[0]):.6f}")
    print(f"[fold] ckpt 元数据：val_r2={ck.get('val_r2')} seed={ck.get('seed')} "
          f"label={ck.get('label')}")

    sd_folded = fold_state_dict(sd_raw, y_std, y_mean)
    w_new = float(sd_folded["value_head.2.weight"].flatten()[0])
    b_new = float(sd_folded["value_head.2.bias"].flatten()[0])
    print(f"[fold] 折后 value_head.2.weight[0,0]={w_new:.6f} bias={b_new:.6f} "
          f"（bias 应 = 折前 {float(sd_raw['value_head.2.bias'].flatten()[0]):.6f}*{y_std:.4f}"
          f"+{y_mean:.4f} = {float(sd_raw['value_head.2.bias'].flatten()[0]) * y_std + y_mean:.6f}）")

    net_folded = ValueNetwork(latent_dim=int(cfg.get("latent_dim", 64)),
                              num_resblocks=int(cfg.get("num_resblocks", 6)))
    net_folded.load_state_dict(sd_folded)
    net_folded.eval()

    # ── ② 取留出帧 ────────────────────────────────────────────────────────
    if (dp / "game_offsets.npy").exists():
        offs = np.load(dp / "game_offsets.npy")
    else:
        gid = np.load(dp / "game_id.npy")
        offs = np.searchsorted(gid, np.arange(int(gid[-1]) + 2))
    n_games = len(offs) - 1
    lo = int(offs[max(0, n_games - a.val_games)])
    hi = int(offs[-1])
    rows = np.arange(lo, hi)
    rng = np.random.default_rng(a.seed)
    n = min(a.check_frames, len(rows))
    idx = np.sort(rng.choice(rows, size=n, replace=False))
    print(f"[data] {dp.name}: 留出演局 {max(0, n_games - a.val_games)}..{n_games - 1} "
          f"rows=[{lo},{hi})={len(rows)}，抽 {n} 帧（seed={a.seed}，无放回）")

    board = np.load(dp / "board.npy", mmap_mode="r")
    stats = np.load(dp / "stats.npy", mmap_mode="r")
    y = np.load(dp / a.label_file).astype(np.float64)
    y_sel = y[idx]

    # ── ③ 老师（三网 ckpt 的 value_state，输出 dict → 已是标签尺度）──────────
    _c, _p, teacher = load_three_models(str(tp))
    teacher.eval()
    print(f"[teacher] {tp.name} value_state params={sum(p.numel() for p in teacher.parameters()):,}")

    outs = {"t": [], "f": [], "r": []}
    with torch.no_grad():
        for s in range(0, n, a.batch):
            bi = idx[s:s + a.batch]
            b = torch.from_numpy(np.array(board[bi], np.float32))
            st = torch.from_numpy(np.array(stats[bi], np.float32))
            to = teacher(b, st)
            outs["t"].append((to["value"] if isinstance(to, dict) else to)
                             .reshape(-1).float().cpu().numpy())
            outs["f"].append(net_folded(b, st).reshape(-1).float().cpu().numpy())
            outs["r"].append(student(b, st).reshape(-1).float().cpu().numpy())
    raw_t = np.concatenate(outs["t"]).astype(np.float64)
    raw_f = np.concatenate(outs["f"]).astype(np.float64)
    raw_r = np.concatenate(outs["r"]).astype(np.float64)

    # 未折版按定义反归一化后的参照
    raw_r_folded = raw_r * y_std + y_mean
    diff = float(np.max(np.abs(raw_f - raw_r_folded)))
    print("\n===== 自检 ① 等价性（与数据无关的充分判据）=====")
    print(f"  max|folded - (raw*y_std+y_mean)| = {diff:.3e}   "
          f"{'✅ 折叠正确' if diff < 1e-4 else '❌ 折叠错误'}")
    print(f"  folded  raw: min={raw_f.min():.4f} max={raw_f.max():.4f} std={raw_f.std():.4f}")
    print(f"  未折 raw   : min={raw_r.min():.4f} max={raw_r.max():.4f} std={raw_r.std():.4f}"
          f"   ← 未折的 std 是折后的 {raw_r.std() / max(1e-12, raw_f.std()):.2f} 倍")

    tanh_t = np.tanh(raw_t)
    tanh_f = np.tanh(raw_f)
    tanh_r = np.tanh(raw_r)          # 未折（= 部署侧若直接用它）
    print("\n===== 自检 ② 与老师比（同批 %d 帧，部署口径 tanh）=====" % n)
    print("  -- 折后（本脚本产出，v_32_3_folded.pt）--")
    print(f"    Spearman rho(tanh) = {spearman(tanh_f, tanh_t):.4f}   <-- 应 ≈ 0.96")
    print(f"    Pearson  r(tanh)   = {pearson(tanh_f, tanh_t):.4f}")
    print(f"    Pearson  r(raw)    = {pearson(raw_f, raw_t):.4f}")
    print(f"    平均|Δtanh| = {float(np.mean(np.abs(tanh_f - tanh_t))):.4f}  "
          f"符号不一致={100.0 * float(np.mean(np.sign(tanh_f) != np.sign(tanh_t))):.2f}%")
    print(f"    R² vs label: raw={r2(y_sel, raw_f):.4f} tanh={r2(y_sel, tanh_f):.4f}")
    print("  -- 对照：未折（若把它直接接到部署侧就会这样）--")
    print(f"    Spearman rho(tanh) = {spearman(tanh_r, tanh_t):.4f}   （Spearman 对单调缩放不变）")
    print(f"    Pearson  r(tanh)   = {pearson(tanh_r, tanh_t):.4f}   ← 会掉")
    print(f"    tanh 分布 std: 折后={tanh_f.std():.4f} 未折={tanh_r.std():.4f} 老师={tanh_t.std():.4f}")
    print(f"    |tanh|>0.9 的占比: 折后={100.0 * float(np.mean(np.abs(tanh_f) > 0.9)):.2f}% "
          f"未折={100.0 * float(np.mean(np.abs(tanh_r) > 0.9)):.2f}% "
          f"（未折被压到饱和 ⇒ 排序被压平）")
    print(f"  -- 老师（参照）R² vs label: raw={r2(y_sel, raw_t):.4f} tanh={r2(y_sel, tanh_t):.4f}")

    # ── ④ 落盘 ────────────────────────────────────────────────────────────
    out_ck = {
        "state_dict": sd_folded,
        "config": {"latent_dim": int(cfg.get("latent_dim", 64)),
                   "num_resblocks": int(cfg.get("num_resblocks", 6))},
        "y_std": y_std, "y_mean": y_mean,
        # ⚠️ 标记位：`value_override.load_override` 见到 True 就**不再**折（否则会折两次）
        "y_folded": True,
        "fold_info": {"folded_into": "value_head.2",
                      "rule": "W' = W*y_std ; b' = b*y_std + y_mean",
                      "folded_at": "2026-10-27"},
        "source": str(sp),
        "val_r2": ck.get("val_r2"), "seed": ck.get("seed"), "label": ck.get("label"),
    }
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ck, op)
    print(f"\n[fold] -> {op}  "
          f"（{op.stat().st_size:,} B，config={out_ck['config']}，y_folded=True）")
    print("[fold] 这一份**直接**可以当现役值网用："
          "AZAI_VALUE_CKPT=<该文件>，输出已是标签尺度，部署侧 tanh 不用改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
