"""诊断：**同一批决策上，比较两张位置网的"选格质量"**（用教师自己的 rollout 分数口径）。

动机（2026-10-11）：A② 修正版**确实学到了教师**（val CE −1.01、top1 7.59%→14.51%），判据却 **−15.6pp** ✗。
于是要区分两种可能：
  (a) 学生学得还不够好 —— 那它的"选格后悔值"应比 incumbent **更差**；
  (b) 学生学得够好、但**教师这个指标跟棋力不对齐** —— 那它的后悔值应比 incumbent **更好/持平**，却仍然更弱。

做法（零对局，纯前向）：用 `collect_pos_rollout_teacher.py` 采的 npz（每行 = 一个决策，含该钉类的若干合法格
`cell (x,y)` 与教师给每格的安静 rollout 分 `adv`，`cnt` = 有效格数），对每张网：
  * 取出它在**这些同一批格**上的值 `v = action_map[cls, x, y]`（掩码内）；
  * 选格 = argmax(v)；教师最优 = argmax(adv)；
  * 指标：教师分均值 `mean adv[pick]`、**后悔值** `regret = adv[best] − adv[pick]`（越低越好）、
    与教师 argmax 的一致率、以及 `regret == 0`（命中最优）的比例。

用法：
  python -u code/my_ai/az_intent/eval_map_vs_teacher.py \
    --data training_history/posroll2/cell90_h1.npz \
    --ckpts checkpoints/grafted_sdk_A.pt checkpoints/grafted_sdk_A_posroll_v2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    from my_ai.az_intent.az_selfplay import load_three_models

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    z = np.load(args.data)
    N = int(z["adv"].shape[0]) if not args.limit else min(int(args.limit), int(z["adv"].shape[0]))
    board = torch.from_numpy(np.asarray(z["board"][:N], dtype=np.float32))
    stats = torch.from_numpy(np.asarray(z["stats"][:N], dtype=np.float32))
    cls = torch.from_numpy(np.asarray(z["cls"][:N], dtype=np.int64))
    cell = torch.from_numpy(np.asarray(z["cell"][:N], dtype=np.int64))
    adv = torch.from_numpy(np.asarray(z["adv"][:N], dtype=np.float32))
    cnt = torch.from_numpy(np.asarray(z["cnt"][:N], dtype=np.int64))
    C = adv.shape[1]
    valid = torch.arange(C).unsqueeze(0) < cnt.unsqueeze(1)
    NEG = -1e9
    adv_m = torch.where(valid, adv, torch.full_like(adv, NEG))
    best_idx = adv_m.argmax(1)
    best_val = adv_m.max(1).values
    print(f"[map-vs-teacher] data={args.data} N={N} 每行格数均值={cnt.float().mean():.1f} "
          f"教师分（最优格）均值={float(best_val.mean()):.4f}", flush=True)

    for ck in args.ckpts:
        _cm, _pm, _vm = load_three_models(ck)
        # ⚠️ 位置图是 **pos_model** 的输出（class 网在两张 ckpt 之间是逐位相同的 ⇒ 用错会得到两行完全一样的假读数）
        _pm.eval().to(dev)
        tot = dict(n=0, adv=0.0, regret=0.0, hit=0, tie=0, flat=0)
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                b = slice(i, min(i + args.batch_size, N))
                am = _pm(board[b].to(dev), stats[b].to(dev))["action_map"]
                B = am.shape[0]
                rows = torch.arange(B, device=dev).unsqueeze(1)
                cc = cls[b].to(dev).unsqueeze(1)
                y = cell[b][..., 0].to(dev).clamp(min=0)
                x = cell[b][..., 1].to(dev).clamp(min=0)
                v = am[rows, cc, y, x]                      # (B, C) 各候选格上的网值
                vv = valid[b].to(dev)
                v_m = torch.where(vv, v, torch.full_like(v, NEG))
                pick = v_m.argmax(1)
                a_pick = adv_m[b].to(dev).gather(1, pick.unsqueeze(1)).squeeze(1)
                b_val = best_val[b].to(dev)
                tot["n"] += B
                tot["adv"] += float(a_pick.sum())
                tot["regret"] += float((b_val - a_pick).sum())
                tot["hit"] += int((pick == best_idx[b].to(dev)).sum())
                tot["tie"] += int(((b_val - a_pick).abs() < 1e-9).sum())
        n = max(tot["n"], 1)
        print(f"   {Path(ck).name:34s} 选中格的教师分均值={tot['adv']/n:+.4f}  "
              f"**后悔值均值={tot['regret']/n:.4f}**  命中最优={tot['hit']/n*100:.2f}%  "
              f"（零后悔={tot['tie']/n*100:.2f}%）", flush=True)
    print("[map-vs-teacher] 判读：若候选的后悔值 **不高于** incumbent 却判据更差 ⇒ **教师指标与棋力不对齐**；"
          "若候选后悔值明显更高 ⇒ 是**学生没学好**。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
