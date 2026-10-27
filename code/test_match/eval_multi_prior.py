"""统一评估多个先验 checkpoint：同一批留出局、同一支撑集口径（2026-10-27）。

留出：`g9376-g9379`（80 局版训练时未见过）
支撑集：该局面该动作组实际出现的候选格
指标：recall@1/@4/@24 + 每格支撑集大小（用于给随机基线）
"""
from __future__ import annotations

import glob
import statistics
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import load_three_models
from my_ai.train_prior_z import load_games

OP2CH = {11: list(range(16)), 13: [16], 21: [17], 22: [18], 23: [19], 24: [20]}
NAME = {11: "build", 13: "downgrade", 21: "lightning", 22: "emp", 23: "deflector", 24: "evasion"}

NETS = [
    ("posnet_A(基线)", "checkpoints/posnet_A_k5_m32.pt"),
    ("z_only", "checkpoints/prior_hybrid_z_only.pt"),
    ("z_ce_w1", "checkpoints/prior_hybrid_z_ce_w1.pt"),
    ("z_ce_w3", "checkpoints/prior_hybrid_z_ce_w3.pt"),
]


def evaluate(net_path: str, games, ks=(1, 4, 24)):
    _, m, _ = load_three_models(str(REPO / net_path))
    m.eval()
    agg = {k: [0, 0] for k in ks}
    per_op = {}
    sup_all = []
    for board, stats, targets in games:
        with torch.no_grad():
            amap = m(torch.from_numpy(board).unsqueeze(0),
                     torch.from_numpy(stats).unsqueeze(0))["action_map"][0]
        amap = amap.detach().cpu()
        for op, cells in targets.items():
            keys = list(cells.keys())
            if len(keys) < 3:
                continue
            sc = (amap[0:16].mean(0) if op == 11 else amap[OP2CH[op][0]]).numpy()
            v = np.array([cells[k] for k in keys])
            s = np.array([sc[x, y] for (x, y) in keys])
            pos = int(np.where(np.argsort(-s) == int(np.argmax(v)))[0][0]) + 1
            n = len(keys)
            sup_all.append(n)
            d = per_op.setdefault(NAME.get(op, str(op)), {k: [0, 0] for k in ks} | {"sup": []})
            for k in ks:
                agg[k][0] += int(pos <= k); agg[k][1] += 1
                d[k][0] += int(pos <= k); d[k][1] += 1
            d["sup"].append(n)
    return agg, per_op, sup_all


def main() -> int:
    files = sorted(glob.glob(str(REPO / "training_history/prior_labels/g93*.npz")))[-4:]
    games = load_games(files)
    print(f"  留出局面 {len(games)}（{len(files)} 局：{[Path(f).stem for f in files]}）\n", flush=True)

    rows = {}
    for tag, path in NETS:
        agg, per_op, sup = evaluate(path, games)
        rows[tag] = (agg, per_op)
        s = "  ".join(f"@{k}={100*agg[k][0]/max(1,agg[k][1]):5.1f}%" for k in (1, 4, 24))
        print(f"  {tag:<16} {s}   (n={agg[24][1]})", flush=True)

    med = statistics.median([x for x in rows[NETS[0][0]][1].get("build", {}).get("sup", [0])]) if rows else 0
    print("\n  按动作类型（@1 / @24）：")
    for op in ("build", "lightning", "downgrade"):
        for tag, _ in NETS:
            d = rows[tag][1].get(op)
            if not d or not d[24][1]:
                continue
            n = d[24][1]
            sup = d["sup"]
            b1 = sum(1.0 / max(1, s) for s in sup) / len(sup)
            b24 = sum(min(24, s) / max(1, s) for s in sup) / len(sup)
            print(f"    {op:<10} {tag:<16} n={n:<5} @1={100*d[1][0]/n:5.1f}%(随机{100*b1:5.1f})  "
                  f"@24={100*d[24][0]/n:5.1f}%(随机{100*b24:5.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
