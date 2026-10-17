#!/usr/bin/env python3
"""分析亚军 debug=plans 的 per-plan 日志：每次决策不同候选的**价值差**（= 价值网需要多高的分辨率）。

输入：桥接归档的 `ai*_runnerup_seed*.stderr.log`（亚军 debug=plans 写到 stderr 的
      `{"kind":"plan",...}` 行，每行一个候选 plan）。

输出（按"决策"分组，决策 = (round, player, serial)）：
  - 候选数、`mean_terminal_score`（rollout 终局值）的 max-min / std / top1-top2 gap；
  - 选择用的 `score`（=mean_rollout_score+heuristic-penalty）的同类统计；
  - 各分项（base_hp/money/worker_threat/combat_threat）的 spread。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]


def load_plans(path: Path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith('{"kind":"plan"'):
                continue
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return rows


def _stats(xs):
    xs = np.asarray([x for x in xs if x is not None], dtype=np.float64)
    if len(xs) == 0:
        return None
    xs = np.sort(xs)[::-1]
    out = {"n": int(len(xs)), "max": float(xs[0]), "min": float(xs[-1])}
    out["range"] = out["max"] - out["min"]
    out["std"] = float(xs.std())
    out["gap12"] = float(xs[0] - xs[1]) if len(xs) >= 2 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="match_results", help="含 ai*_runnerup_seed*.stderr.log 的目录")
    ap.add_argument("--glob", default="ai*_runnerup_seed*.stderr.log")
    a = ap.parse_args()

    d = Path(a.dir)
    if not d.is_absolute():
        d = _REPO / d
    files = sorted(d.glob(a.glob))
    print(f"[spread] {len(files)} 个 stderr 文件", flush=True)

    by_dec = defaultdict(list)
    n_plan = 0
    for f in files:
        rows = load_plans(f)
        n_plan += len(rows)
        for r in rows:
            key = (int(r.get("round", -1)), int(r.get("player", -1)), int(r.get("serial", -1)))
            by_dec[key].append(r)
    print(f"[spread] plan 行 {n_plan}，决策 {len(by_dec)}", flush=True)

    fields = ["mean_terminal_score", "score", "score_before_heuristic",
              "mean_base_hp_raw", "mean_money_raw",
              "mean_worker_threat_raw", "mean_combat_threat_raw"]
    per_field = {k: {"range": [], "std": [], "gap12": [], "n": []} for k in fields}
    multi = 0
    for key, cands in by_dec.items():
        if len(cands) < 2:
            continue
        multi += 1
        for k in fields:
            st = _stats([c.get(k) for c in cands])
            if st is None:
                continue
            per_field[k]["range"].append(st["range"])
            per_field[k]["std"].append(st["std"])
            per_field[k]["gap12"].append(st["gap12"])
            per_field[k]["n"].append(st["n"])
    print(f"[spread] 多候选决策 {multi} 个\n", flush=True)

    for k in fields:
        rng = np.asarray(per_field[k]["range"], dtype=np.float64)
        gap = np.asarray(per_field[k]["gap12"], dtype=np.float64)
        nn = np.asarray(per_field[k]["n"], dtype=np.float64)
        if len(rng) == 0:
            continue
        print(f"--- {k} ---")
        print(f"  候选数 mean={nn.mean():.1f} med={np.median(nn):.0f}")
        print(f"  max-min  : mean={rng.mean():.1f} med={np.median(rng):.1f} "
              f"p25={np.percentile(rng,25):.1f} p75={np.percentile(rng,75):.1f} "
              f"p95={np.percentile(rng,95):.1f}")
        print(f"  top1-top2: mean={gap.mean():.1f} med={np.median(gap):.1f} "
              f"p25={np.percentile(gap,25):.1f} p75={np.percentile(gap,75):.1f} "
              f"(此项 ≈ 价值网要分辨的最小差)")
    print("\n[解读] 若价值网 MAE 远小于 top1-top2 中位数 => 排序基本正确；"
          "MAE 接近或超过它 => 会在近并列样本上选错。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
