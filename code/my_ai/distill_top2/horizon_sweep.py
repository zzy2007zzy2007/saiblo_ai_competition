"""前瞻深度扫描：亚军的决策质量随「往后看几步」怎么变？

背景（见 docs/findings_runnerup_prior_structure.md）：
- 亚军默认 `long_eval_horizon = 8`，每臂 ~125 次 rollout。
- 我们的价值网天然是"叶子评估 + 0/1 步"，实验证明它在决策上等于"永远不动"。
- 那么**到底要看多少步**？把 horizon 从 1 扫到 16，看排序/决策与 h=8（亚军设置）的一致率。

输出每个状态在每个 horizon 下的：
  - 与 h=8 的 Spearman 排序相关
  - 决策一致率（含 +80 先验）
  - 是否"该动手时动手"（full 动手的状态里，本 horizon 也动手的比例）

用法：
    python horizon_sweep.py --seeds 732 733 --rollouts 24 --horizons 1,2,3,4,6,8,12,16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

EXE = Path(r"D:\dsh_build_tmp\dump_plan_prior.exe")


def make_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = r"C:\mingw64\bin;" + env.get("PATH", "")
    return env


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    order_by_x = sorted(range(n), key=lambda i: xs[i])
    order_by_y = sorted(range(n), key=lambda i: ys[i])
    rank = [0.0] * n
    for position, index in enumerate(order_by_x):
        rank[index] = float(position)
    rank_y = [0.0] * n
    for position, index in enumerate(order_by_y):
        rank_y[index] = float(position)
    mx = statistics.mean(rank)
    my = statistics.mean(rank_y)
    num = sum((a - mx) * (b - my) for a, b in zip(rank, rank_y))
    den = (sum((a - mx) ** 2 for a in rank) * sum((b - my) ** 2 for b in rank_y)) ** 0.5
    return num / den if den > 1e-12 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training_history/distill_top2/abl_linear2_raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[732])
    parser.add_argument("--rounds", default="100,180,260,340,420")
    parser.add_argument("--players", default="0,1")
    parser.add_argument("--rollouts", type=int, default=24)
    parser.add_argument("--horizons", default="1,2,3,4,6,8,12,16")
    parser.add_argument("--reference", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if args.reference not in horizons:
        horizons.append(args.reference)
    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    players = [int(x) for x in args.players.split(",")]
    env = make_env()

    # 每个 horizon 累积：rho 列表、决策一致、动手召回
    stats: dict[int, dict[str, list]] = {
        h: {"rho": [], "dec_agree": [], "act_recall": [], "hold_when_hold": [],
            "spread": [], "sigma": []} for h in horizons
    }
    n_states = 0
    t_start = time.time()

    for seed in args.seeds:
        paths = sorted(glob.glob(str(Path(args.data_dir) / f"ents_*{seed}.jsonl")))
        if not paths:
            print(f"[skip] seed={seed}")
            continue
        with open(paths[0], encoding="utf-8") as handle:
            all_states = [json.loads(line) for line in handle if line.strip()]
        for round_index in rounds:
            if round_index >= len(all_states):
                continue
            for player in players:
                state = dict(all_states[round_index])
                state.update(seed=seed, player=player, mode="rollout",
                             rollouts=args.rollouts, max_plans=0, horizons=horizons)
                t0 = time.time()
                proc = subprocess.run([str(EXE)], input=(json.dumps(state) + "\n").encode(),
                                      capture_output=True, env=env)
                elapsed = time.time() - t0
                if proc.returncode != 0:
                    print(f"[fail] seed={seed} r={round_index} p{player}")
                    continue
                result = json.loads(proc.stdout.decode("utf-8", "replace").splitlines()[0])
                if result.get("ok") != 1:
                    continue
                plans = result["plans"]
                if len(plans) < 5:
                    continue
                n_states += 1

                by_horizon: dict[int, list[float]] = {}
                sigma: dict[int, float] = {}
                for plan in plans:
                    for entry in plan.get("horizons", []):
                        by_horizon.setdefault(entry["h"], []).append(entry["mean"])
                        sigma.setdefault(entry["h"], entry["std"])
                if args.reference not in by_horizon:
                    continue
                heur = [plan["heuristic"] for plan in plans]
                reference = by_horizon[args.reference]
                ref_decision = max(range(len(plans)), key=lambda i: reference[i] + heur[i])
                ref_acts = plans[ref_decision]["key"] != "hold"

                for horizon in horizons:
                    values = by_horizon.get(horizon)
                    if values is None or len(values) != len(plans):
                        continue
                    bucket = stats[horizon]
                    bucket["rho"].append(spearman(reference, values))
                    decision = max(range(len(plans)), key=lambda i: values[i] + heur[i])
                    bucket["dec_agree"].append(int(decision == ref_decision))
                    acts = plans[decision]["key"] != "hold"
                    if ref_acts:
                        bucket["act_recall"].append(int(acts))
                    else:
                        bucket["hold_when_hold"].append(int(not acts))
                    bucket["spread"].append(max(values) - min(values))
                    bucket["sigma"].append(sigma.get(horizon, 0.0))
                print(f"[ok] seed={seed} r={round_index:3d} p{player} n={len(plans):3d} {elapsed:.2f}s")

    if not n_states:
        print("没有结果")
        return 2

    rows = []
    print(f"\n=== 前瞻深度扫描（{n_states} 个状态，每臂 {args.rollouts} 次采样，参照 h={args.reference}）===")
    print(f"{'h':>3} {'rho中位':>9} {'决策一致':>9} {'该动时动了':>11} {'该hold时hold':>13} "
          f"{'离散度':>8} {'单次σ':>7}")
    for horizon in horizons:
        bucket = stats[horizon]
        if not bucket["rho"]:
            continue
        row = {
            "h": horizon,
            "rho_median": statistics.median(bucket["rho"]),
            "dec_agree": sum(bucket["dec_agree"]) / len(bucket["dec_agree"]),
            "act_recall": (sum(bucket["act_recall"]) / len(bucket["act_recall"])
                           if bucket["act_recall"] else float("nan")),
            "hold_when_hold": (sum(bucket["hold_when_hold"]) / len(bucket["hold_when_hold"])
                               if bucket["hold_when_hold"] else float("nan")),
            "spread": statistics.mean(bucket["spread"]),
            "sigma": statistics.median(bucket["sigma"]),
            "n_decisions": len(bucket["dec_agree"]),
        }
        rows.append(row)
        print(f"{horizon:>3} {row['rho_median']:>9.3f} {100 * row['dec_agree']:>8.1f}% "
              f"{100 * row['act_recall']:>10.1f}% {100 * row['hold_when_hold']:>12.1f}% "
              f"{row['spread']:>8.0f} {row['sigma']:>7.0f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[写] {out_path}")
    print(f"\n总耗时 {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
