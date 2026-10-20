"""量亚军 rollout 搜索的**噪声下限**：σ（同状态同计划、不同 rollout 的分数标准差）与 SE=σ/√N。

为什么：亚军的决策是 `argmax(mean_terminal_score + heuristic)`，而 `mean_terminal_score`
是 N 次随机 rollout 的均值。要判断"我们的网络能不能替代它的搜索"，必须先知道
**这份搜索本身有多吵**。SE 与 top1-top2 分差（日志实测中位 ~71）一比，就知道
"复现亚军决策"到底是一个确定性问题还是概率问题。

用法：
    python measure_rollout_noise.py --seeds 732 733 --rounds 0,40,80,...,400 --rollouts 64 --max-plans 6
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
DEFAULT_DIR = Path("training_history/distill_top2/abl_linear2_raw")


def make_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = r"C:\mingw64\bin;" + env.get("PATH", "")
    return env


def run_one(state: dict, env: dict) -> dict:
    payload = (json.dumps(state) + "\n").encode("utf-8")
    proc = subprocess.run([str(EXE)], input=payload, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:400])
    lines = [ln for ln in proc.stdout.decode("utf-8", "replace").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("exe 无输出")
    return json.loads(lines[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--seeds", nargs="+", type=int, default=[732])
    parser.add_argument("--rounds", default="0,40,80,120,160,200,240,280,320,360,400")
    parser.add_argument("--players", default="0,1")
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--max-plans", type=int, default=6)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    players = [int(x) for x in args.players.split(",")]
    env = make_env()

    rows_all: list[dict] = []
    t_start = time.time()
    for seed in args.seeds:
        paths = sorted(glob.glob(str(Path(args.data_dir) / f"ents_*{seed}.jsonl")))
        if not paths:
            print(f"[skip] seed={seed} 找不到状态文件")
            continue
        with open(paths[0], encoding="utf-8") as handle:
            states_all = [json.loads(line) for line in handle if line.strip()]
        for round_index in rounds:
            if round_index >= len(states_all):
                continue
            for player in players:
                state = dict(states_all[round_index])
                state["seed"] = seed
                state["player"] = player
                state["mode"] = "rollout"
                state["rollouts"] = args.rollouts
                state["max_plans"] = args.max_plans
                t0 = time.time()
                result = run_one(state, env)
                elapsed = time.time() - t0
                if result.get("ok") != 1:
                    print(f"[warn] seed={seed} r={round_index} p={player}: {result.get('error')}")
                    continue
                for plan in result["plans"]:
                    rows_all.append({
                        "seed": seed,
                        "round": round_index,
                        "player": player,
                        "key": plan["key"],
                        "n": plan["n"],
                        "mean": plan["mean"],
                        "std": plan["std"],
                        "heuristic": plan["heuristic"],
                        "min": plan["min"],
                        "max": plan["max"],
                        "elapsed_s": elapsed,
                    })
                print(f"[ok] seed={seed} r={round_index:3d} p={player} "
                      f"plans={len(result['plans'])} {elapsed:.2f}s")

    if not rows_all:
        print("没有结果")
        return 2

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows_all:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[写] {out_path}")

    stds = [row["std"] for row in rows_all]
    per_rollout = stds
    se125 = [s / (125 ** 0.5) for s in per_rollout]
    seN = [s / (args.rollouts ** 0.5) for s in per_rollout]
    print(f"\n=== rollout 噪声（{len(rows_all)} 条 计划×状态）===")
    print(f"每个计划的 σ（单次 rollout 分数标准差）: "
          f"均值 {statistics.mean(per_rollout):.1f} 中位 {statistics.median(per_rollout):.1f} "
          f"p90 {sorted(per_rollout)[int(0.9 * len(per_rollout))]:.1f} max {max(per_rollout):.1f}")
    print(f"SE（N=125，亚军实际每臂量级）           : "
          f"均值 {statistics.mean(se125):.1f} 中位 {statistics.median(se125):.1f}")
    print(f"SE（N={args.rollouts}，本次测量）        : "
          f"均值 {statistics.mean(seN):.1f} 中位 {statistics.median(seN):.1f}")

    # 同一状态内：真实均值之间的差（信号） vs SE（噪声）
    signals = []
    by_state: dict[tuple, list[dict]] = {}
    for row in rows_all:
        by_state.setdefault((row["seed"], row["round"], row["player"]), []).append(row)
    for plans in by_state.values():
        means = sorted((p["mean"] for p in plans), reverse=True)
        if len(means) >= 2:
            signals.append(means[0] - means[1])
    if signals:
        print(f"\n同一状态内 top1-top2 的真实均分之差（信号）: "
              f"均值 {statistics.mean(signals):.1f} 中位 {statistics.median(signals):.1f} "
              f"p10 {sorted(signals)[int(0.1 * len(signals))]:.1f}")
        ratio = [s / max(1e-9, statistics.median(se125)) for s in signals]
        print(f"信号 / SE(125) 比值: 中位 {statistics.median(ratio):.2f}")
        print(f"  比值 <1 的决策占比: {100.0 * sum(1 for r in ratio if r < 1) / len(ratio):.1f}% "
              f"（这些决策本身就是噪声决定的）")

    total_time = time.time() - t_start
    print(f"\n总耗时 {total_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
