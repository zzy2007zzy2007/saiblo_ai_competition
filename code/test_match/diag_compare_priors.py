"""在同一批留出局上比较 f10 与 prior_z_80g.best 的离线指标（2026-10-25）。

口径统一：
  · 数据：training_history/prior_labels/g9376-g9379（80 局版的留出 4 局，训练时未见过）
  · 支撑集：该局面该动作组**实际出现的候选格**
  · 指标：recall@1/@4/@24（该组教师分数最高的那个格，在先验排序里排第几）
  · 建塔：16 个塔型通道取均值（与部署时的取法一致）
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import load_three_models

OP2CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20]}
NAME = {11: "build", 13: "downgrade", 21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}


def load_games(files):
    out = []
    for f in files:
        d = np.load(f)
        feats, stt = d["feat"], d["stats"]
        cls, xs, ys, vals, off = d["cand_cls"], d["cand_x"], d["cand_y"], d["cand_val"], d["cand_off"]
        for i in range(len(off) - 1):
            a, b = int(off[i]), int(off[i + 1])
            per = {}
            for t, x, y, v in zip(cls[a:b], xs[a:b], ys[a:b], vals[a:b]):
                t = int(t)
                if t not in OP2CH:
                    continue
                key = (int(x), int(y))
                m = per.setdefault(t, {})
                m[key] = max(m.get(key, -1e9), float(v))
            if per:
                out.append((feats[i].astype(np.float32), stt[i].astype(np.float32), per))
    return out


def evaluate(ckpt, games, ks=(1, 4, 24)):
    _, m, _ = load_three_models(ckpt)
    m.eval()
    agg = {k: [0, 0] for k in ks}
    per = {}
    for feat, stt, targets in games:
        with torch.no_grad():
            amap = m(torch.from_numpy(feat).unsqueeze(0),
                     torch.from_numpy(stt).unsqueeze(0))["action_map"][0]
        for op, cells in targets.items():
            keys = list(cells.keys())
            if len(keys) < 3:
                continue
            sc = (amap[0:16].mean(0) if op == 11 else amap[OP2CH[op][0]]).numpy()
            v = np.array([cells[k] for k in keys])
            s = np.array([sc[x, y] for (x, y) in keys])
            pos = int(np.where(np.argsort(-s) == int(np.argmax(v)))[0][0]) + 1
            d = per.setdefault(NAME[op], {k: [0, 0] for k in ks})
            for k in ks:
                agg[k][0] += int(pos <= k); agg[k][1] += 1
                d[k][0] += int(pos <= k); d[k][1] += 1
    return agg, per


def main() -> int:
    files = sorted(glob.glob(str(REPO / "training_history/prior_labels/g93*.npz")))[-4:]
    games = load_games(files)
    print(f"  留出局: {[Path(f).stem for f in files]}  局面 {len(games)}", flush=True)
    for tag, ck in (("f10（旧）", "checkpoints/prior_distill_f10.pt"),
                    ("prior_z_80g.best（新）", "checkpoints/prior_z_80g.best.pt"),
                    ("posnet_A（未训练基线）", "checkpoints/posnet_A_k5_m32.pt")):
        agg, per = evaluate(str(REPO / ck), games)
        s = "  ".join(f"@{k}={100*agg[k][0]/max(1,agg[k][1]):5.1f}%" for k in (1, 4, 24))
        print(f"  {tag:<22} {s}  (n={agg[24][1]})", flush=True)
        for g in ("build", "lightning", "evasion", "deflector", "emp", "downgrade"):
            d = per.get(g)
            if d and d[24][1]:
                n = d[24][1]
                print(f"      {g:<10} n={n:<5} @1={100*d[1][0]/n:5.1f}%  @4={100*d[4][0]/n:5.1f}%  @24={100*d[24][0]/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
