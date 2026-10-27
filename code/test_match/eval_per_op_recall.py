# DEPRECATED: 已被 eval_recall_report.py 取代（2026-10-27）
"""按动作类型统计 recall@1/@4/@24（含降级，支撑集≥2）——2026-10-27。

之前 `eval_multi_prior.py` 用 min-cells=3，导致**降级（支撑集中位数 2）被整类筛掉**，
所以只看到了 build/lightning。本脚本：
  · 每个动作类型单独统计，支撑集门槛可配（默认 2）
  · 同时给随机基线（1/n、min(4,n)/n、min(24,n)/n 的均值）
  · 同时给支撑集中位数
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
NAME = {11: "建塔", 13: "降级/拆塔", 21: "闪电"}

NETS = [
    ("posnet_A(基线)", "checkpoints/posnet_A_k5_m32.pt"),
    ("z_only", "checkpoints/prior_hybrid_z_only.pt"),
    ("hard_ce_w1", "checkpoints/prior_hard_ce_w1.pt"),
    ("hard_ce_w3", "checkpoints/prior_hard_ce_w3.pt"),
]
MIN_CELLS = 2


def evaluate(path, games):
    _, m, _ = load_three_models(str(REPO / path))
    m.eval()
    out = {}
    for board, stats, targets in games:
        with torch.no_grad():
            amap = m(torch.from_numpy(board).unsqueeze(0),
                     torch.from_numpy(stats).unsqueeze(0))["action_map"][0].detach().cpu()
        for op, cells in targets.items():
            if op not in OP2CH:
                continue
            keys = list(cells.keys())
            if len(keys) < MIN_CELLS:
                continue
            sc = (amap[0:16].mean(0) if op == 11 else amap[OP2CH[op][0]]).numpy()
            vals = np.array([cells[k] for k in keys])
            ss = np.array([sc[x, y] for (x, y) in keys])
            pos = int(np.where(np.argsort(-ss) == int(np.argmax(vals)))[0][0]) + 1
            d = out.setdefault(op, {"n": 0, "h1": 0, "h4": 0, "h24": 0, "sup": [],
                                    "b1": 0.0, "b4": 0.0, "b24": 0.0})
            d["n"] += 1
            d["h1"] += int(pos <= 1)
            d["h4"] += int(pos <= 4)
            d["h24"] += int(pos <= 24)
            d["sup"].append(len(keys))
            d["b1"] += 1.0 / len(keys)
            d["b4"] += min(4, len(keys)) / len(keys)
            d["b24"] += min(24, len(keys)) / len(keys)
    return out


def main() -> int:
    files = sorted(glob.glob(str(REPO / "training_history/prior_labels/g93*.npz")))[-4:]
    games = load_games(files)
    print(f"  留出 {len(games)} 局面；支撑集门槛 ≥{MIN_CELLS}（所以降级也进来了）\n", flush=True)
    res = {tag: evaluate(p, games) for tag, p in NETS}

    for op in (11, 13, 21):
        print(f"  === {NAME[op]}（op={op}）===")
        base = res[NETS[0][0]].get(op)
        if base:
            sup = base["sup"]
            print(f"    {'网络':<16} {'n':>5} {'支撑中位':>8} {'@1':>7} {'@4':>7} {'@24':>7}")
        for tag, _ in NETS:
            d = res[tag].get(op)
            if not d or not d["n"]:
                print(f"    {tag:<16}     0（无样本）")
                continue
            n = d["n"]
            print(f"    {tag:<16} {n:>5} {statistics.median(d['sup']):>8.0f} "
                  f"{100*d['h1']/n:>6.1f}% {100*d['h4']/n:>6.1f}% {100*d['h24']/n:>6.1f}%")
        if base:
            n = base["n"]
            print(f"    {'【随机基线】':<14} {n:>5} {statistics.median(base['sup']):>8.0f} "
                  f"{100*base['b1']/n:>6.1f}% {100*base['b4']/n:>6.1f}% {100*base['b24']/n:>6.1f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
