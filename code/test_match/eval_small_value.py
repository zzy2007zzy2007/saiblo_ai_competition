#!/usr/bin/env python3
"""学生 vs 老师：小值网（width/blocks 可小）与**现役值网**的排序一致性。

为什么需要这个脚本：`train_value_stream.py` 的 val R² 只回答"谁更准"，
不回答"换了叶子值网会不会**改变搜索的排序**"。搜索只消费**相对顺序**
（`flat_rollout_greedy` 取 argmax、M6c-ROOT 比较叶子值），所以替换的安全性
判据是**秩相关**，不是 R²。

做法：
  1. 数据用 `lowq_f64` 的**按局留出集**（最后 `--val-games` 局，与训练/历史读数同口径）；
  2. 从留出行里**随机抽** `--n-frames` 帧（固定 `--seed`，可复现）；
  3. 老师 = `--teacher-ckpt`（默认 `checkpoints/posnet_A_k5_m32.pt`，取 `load_three_models()[2]`，
     即**现役值网**）；学生 = `--student-ckpt`（独立 `ValueNetwork`）；
  4. 两侧都换算到**部署侧真正消费的量** `tanh(v)`，输出 Spearman ρ 与 Pearson r。

⚠️ 三个坑（都踩过）：
  * **量纲**：`ValueNetwork` 是在 **z-score 标签** `(y-y_mean)/y_std` 上训的，裸输出不是标签尺度。
    `graft_value_ckpt.py` 的做法是把 `y_std/y_mean` 折进 `value_head.2`；这里等价地在外面做
    `v = raw * y_std + y_mean`。**漏掉这一步 = 量纲错**（历史事故：terminal 放大 2.6×、rel 13×）。
    （Spearman 对单调变换不变，所以 ρ 不受影响，但 **Pearson 和 R² 会错**——仍然必须做。）
  * **输出形态不同**：老师是 `AntWarNetwork`，`model(b,s)` 返回 **dict**（取 `["value"]`）；
    学生是独立 `ValueNetwork`，返回**裸张量 (B,)**。不能照抄 `model(...)["value"]`。
  * **别按帧切**：留出集必须按局切（`game_id.npy` / `game_offsets.npy`），按帧切会泄漏。

用法：
  python code/test_match/eval_small_value.py \
      --student-ckpt checkpoints/small32_3/lowq_value_seed0.pt \
      --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64
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

from my_ai.value_network import ValueNetwork          # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models  # noqa: E402


def spearman(a, b) -> float:
    """秩相关（照抄 `code/my_ai/train_prior_distill.py:37` 的实现）。"""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", required=True, help="新训的独立 ValueNetwork ckpt")
    ap.add_argument("--teacher-ckpt", default="checkpoints/posnet_A_k5_m32.pt",
                    help="现役三网 ckpt（取 value_state）；默认 = 部署在用的 posnet_A_k5_m32")
    ap.add_argument("--data", default="training_history/distill_top2/lowq_f64")
    ap.add_argument("--label-file", default="label.npy")
    ap.add_argument("--val-games", type=int, default=500, help="按局留出：最后多少局")
    ap.add_argument("--n-frames", type=int, default=20000, help="从留出行里随机抽多少帧")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cpu", help="默认 cpu：不跟正在跑的训练抢显存")
    ap.add_argument("--out-json", default="", help="可选：把指标落成 json")
    a = ap.parse_args()

    d = Path(a.data)
    if not d.is_absolute():
        d = _REPO / d
    sc = Path(a.student_ckpt)
    if not sc.is_absolute():
        sc = _REPO / sc
    tc = Path(a.teacher_ckpt)
    if not tc.is_absolute():
        tc = _REPO / tc

    # ── 按局切出留出集（最后 --val-games 局）───────────────────────────────
    if (d / "game_offsets.npy").exists():
        offs = np.load(d / "game_offsets.npy")
    else:
        gid = np.load(d / "game_id.npy")
        offs = np.searchsorted(gid, np.arange(int(gid[-1]) + 2))
    n_games = len(offs) - 1
    lo = int(offs[max(0, n_games - a.val_games)])
    hi = int(offs[-1])
    rows = np.arange(lo, hi)
    print(f"[data] {d.name}: N={int(offs[-1])} games={n_games} "
          f"holdout=games {max(0, n_games-a.val_games)}..{n_games-1} rows=[{lo},{hi}) = {len(rows)}")

    rng = np.random.default_rng(a.seed)
    n = min(a.n_frames, len(rows))
    idx = np.sort(rng.choice(rows, size=n, replace=False))
    print(f"[data] sampled {n} frames (seed={a.seed}, no replacement)")

    board = np.load(d / "board.npy", mmap_mode="r")
    stats = np.load(d / "stats.npy", mmap_mode="r")
    y = np.load(d / a.label_file).astype(np.float64)
    y_sel = y[idx]
    print(f"[data] label range=[{y_sel.min():.3f},{y_sel.max():.3f}] std={y_sel.std():.4f}")

    dev = torch.device(a.device)

    # ── 老师：三网 ckpt 的 value_state（AntWarNetwork，输出 dict）──────────────
    _cls, _pos, teacher = load_three_models(str(tc))
    teacher.to(dev).eval()
    # 老师输出是**标签尺度**还是 z 尺度？就地用 R² 判（raw 的 R²≈0.45 ⇒ 已经是标签尺度）
    print(f"[teacher] {tc.name}: params={sum(p.numel() for p in teacher.parameters()):,}")

    # ── 学生：独立 ValueNetwork（输出**裸张量**），量纲要自己折回去 ─────────────
    sk = torch.load(sc, map_location="cpu", weights_only=False)
    _cfg = sk.get("config") or {}
    student = ValueNetwork(latent_dim=int(_cfg.get("latent_dim", 64)),
                           num_resblocks=int(_cfg.get("num_resblocks", 6)))
    student.load_state_dict(sk["state_dict"])
    student.to(dev).eval()
    s_std = float(sk.get("y_std", 1.0))
    s_mean = float(sk.get("y_mean", 0.0))
    print(f"[student] {sc.name}: ValueNetwork({_cfg.get('latent_dim', 64)},"
          f"{_cfg.get('num_resblocks', 6)}) params={student.count_parameters():,} "
          f"y_mean={s_mean:.4f} y_std={s_std:.4f} val_r2(ckpt)={sk.get('val_r2')}")

    t_out, s_out = [], []
    with torch.no_grad():
        for s in range(0, n, a.batch):
            bi = idx[s:s + a.batch]
            b = torch.from_numpy(np.array(board[bi], np.float32)).to(dev)
            st = torch.from_numpy(np.array(stats[bi], np.float32)).to(dev)
            to = teacher(b, st)
            if isinstance(to, dict):           # AntWarNetwork → dict
                to = to["value"]
            t_out.append(to.reshape(-1).float().cpu().numpy())
            s_out.append(student(b, st).reshape(-1).float().cpu().numpy())  # 裸张量
    raw_t = np.concatenate(t_out).astype(np.float64)
    raw_s = np.concatenate(s_out).astype(np.float64) * s_std + s_mean   # ← 折回标签尺度
    tanh_t, tanh_s = np.tanh(raw_t), np.tanh(raw_s)

    print("\n[scale]  raw 形状：teacher min=%.3f max=%.3f std=%.3f | student(折回) "
          "min=%.3f max=%.3f std=%.3f"
          % (raw_t.min(), raw_t.max(), raw_t.std(), raw_s.min(), raw_s.max(), raw_s.std()))
    print("[scale]  tanh 形状：teacher std=%.4f | student std=%.4f"
          % (tanh_t.std(), tanh_s.std()))

    rho = spearman(tanh_s, tanh_t)
    r_tanh = pearson(tanh_s, tanh_t)
    r_raw = pearson(raw_s, raw_t)
    print("\n===== 学生 vs 老师（同一批 %d 帧，部署口径 tanh）=====" % n)
    print("  Spearman rho(tanh学生, tanh老师) = %.4f   <-- 主判据（门槛 >= 0.95）" % rho)
    print("  Pearson  r(tanh学生, tanh老师)    = %.4f" % r_tanh)
    print("  Pearson  r(裸值学生, 裸值老师)    = %.4f   （参考，不含 tanh）" % r_raw)
    print("  平均绝对差 |tanh_s - tanh_t|     = %.4f   （tanh 量纲，<=2）"
          % float(np.mean(np.abs(tanh_s - tanh_t))))
    print("  符号不一致占比（谁领先的看法相反） = %.2f%%"
          % (100.0 * float(np.mean(np.sign(tanh_s) != np.sign(tanh_t)))))
    # ⚠️ 符号翻转率看起来吓人，但值网输出是**连续分**、大多数帧都在 0 附近：
    # 分档看才能回答"在老师有把握的局面上，学生会不会反过来看"。
    print("  -- 按老师把握程度分档（|tanh老师|）--")
    for thr in (0.05, 0.2, 0.4):
        m = np.abs(tanh_t) >= thr
        if not m.any():
            continue
        print("     |tanh_t|>=%.2f: %6d 帧 (%5.1f%%)  符号不一致=%6.2f%%  rho=%.4f"
              % (thr, int(m.sum()), 100.0 * m.mean(),
                 100.0 * float(np.mean(np.sign(tanh_s[m]) != np.sign(tanh_t[m]))),
                 spearman(tanh_s[m], tanh_t[m])))

    # ── 附：各自对 label 的 R²（确认两边都是"标签尺度"，否则上面全是错的）──
    print("\n===== 对同一批帧的 label 拟合（量纲自检）=====")
    print("  老师 raw  R2 vs label = %.4f | tanh = %.4f"
          % (r2(y_sel, raw_t), r2(y_sel, tanh_t)))
    print("  学生 raw  R2 vs label = %.4f | tanh = %.4f"
          % (r2(y_sel, raw_s), r2(y_sel, tanh_s)))

    if a.out_json:
        import json
        o = Path(a.out_json)
        if not o.is_absolute():
            o = _REPO / o
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps({
            "student": str(sc), "teacher": str(tc), "data": str(d),
            "n_frames": int(n), "val_games": a.val_games, "sample_seed": a.seed,
            "spearman_tanh": rho, "pearson_tanh": r_tanh, "pearson_raw": r_raw,
            "mean_abs_diff_tanh": float(np.mean(np.abs(tanh_s - tanh_t))),
            "sign_disagree_frac": float(np.mean(np.sign(tanh_s) != np.sign(tanh_t))),
            "teacher_r2_raw": r2(y_sel, raw_t), "teacher_r2_tanh": r2(y_sel, tanh_t),
            "student_r2_raw": r2(y_sel, raw_s), "student_r2_tanh": r2(y_sel, tanh_s),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[out] {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
