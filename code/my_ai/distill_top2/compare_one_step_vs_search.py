"""核心实验：**1 步价值评估** 能不能替代亚军的 **8 步 rollout 搜索**？

动机（主线收尾）：
- 亚军的决策 = `argmax(mean_terminal_score + heuristic)`，其中 `mean_terminal_score`
  是 8 回合随机 rollout 的终止评估均值（= 蒙特卡洛期望）。
- 我们的价值网（entity Transformer，R² 0.9969）预测的是**单步状态**的 `evaluate_terminal`，
  天然对应"计划执行 + 前瞻 1 回合后评估"——即 `one_step`。
- 所以真正的问题是：**`one_step` 的排序 ≈ `full` 的排序吗？**

若接近 ⇒ 我们的架构形态是对的，只差精度；
若差别大 ⇒ 8 步前瞻本身就是信息源，必须实现 rollout（我们 M6c 已经在做）。

用法：
    python compare_one_step_vs_search.py --seeds 732 733 --rollouts 32
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
    """秩相关（无并列修正的简化版；本数据里 full_mean 几乎无并列）。"""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        rank = [0.0] * n
        index = 0
        while index < n:
            end = index
            while end + 1 < n and values[order[end + 1]] == values[order[index]]:
                end += 1
            average = (index + end) / 2.0 + 1.0
            for pos in range(index, end + 1):
                rank[order[pos]] = average
            index = end + 1
        return rank

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 1e-12 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training_history/distill_top2/abl_linear2_raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[732])
    parser.add_argument("--rounds", default="80,140,200,260,320,380,440")
    parser.add_argument("--players", default="0,1")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    players = [int(x) for x in args.players.split(",")]
    env = make_env()

    state_results: list[dict] = []
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
                             rollouts=args.rollouts, max_plans=0, compare=True)
                t0 = time.time()
                proc = subprocess.run([str(EXE)], input=(json.dumps(state) + "\n").encode(),
                                      capture_output=True, env=env)
                elapsed = time.time() - t0
                if proc.returncode != 0:
                    print(f"[fail] seed={seed} r={round_index} p={player}")
                    continue
                result = json.loads(proc.stdout.decode("utf-8", "replace").splitlines()[0])
                if result.get("ok") != 1:
                    print(f"[warn] {result.get('error')}")
                    continue
                plans = result["plans"]
                if len(plans) < 3:
                    continue

                full = [p["mean"] for p in plans]
                one = [p["one_step_mean"] for p in plans]
                heur = [p["heuristic"] for p in plans]
                full_std = [p["std"] for p in plans]
                one_std = [p["one_step_std"] for p in plans]

                # 决策比较（含先验加成）
                full_dec = max(range(len(plans)), key=lambda i: full[i] + heur[i])
                one_dec = max(range(len(plans)), key=lambda i: one[i] + heur[i])
                # 纯评估比较（不含先验）
                full_arg = max(range(len(plans)), key=lambda i: full[i])
                one_arg = max(range(len(plans)), key=lambda i: one[i])

                # 只看 full 排名的 top-k 里的计划，1 步评估能否也排前面的命中率
                order_full = sorted(range(len(plans)), key=lambda i: -full[i])
                order_one = sorted(range(len(plans)), key=lambda i: -one[i])
                top5_full = set(order_full[:5])
                top5_one = set(order_one[:5])

                se = statistics.median(full_std) / (args.rollouts ** 0.5)
                state_results.append({
                    "seed": seed, "round": round_index, "player": player,
                    "n": len(plans),
                    "spearman": spearman(full, one),
                    "full_dec_key": plans[full_dec]["key"],
                    "one_dec_key": plans[one_dec]["key"],
                    "dec_agree": int(full_dec == one_dec),
                    "arg_agree": int(full_arg == one_arg),
                    "top5_overlap": len(top5_full & top5_one) / 5.0,
                    "full_spread": max(full) - min(full),
                    "one_spread": max(one) - min(one),
                    "full_median_std": statistics.median(full_std),
                    "one_median_std": statistics.median(one_std),
                    "se_full": se,
                    "elapsed_s": elapsed,
                })
                print(f"[ok] seed={seed} r={round_index:3d} p{player} n={len(plans):3d} "
                      f"rho={state_results[-1]['spearman']:+.3f} "
                      f"dec_agree={state_results[-1]['dec_agree']} "
                      f"top5={state_results[-1]['top5_overlap']:.2f} {elapsed:.2f}s")

    if not state_results:
        print("没有结果")
        return 2

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in state_results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[写] {out_path}")

    rhos = [r["spearman"] for r in state_results if r["spearman"] == r["spearman"]]
    print(f"\n=== 1 步评估 vs 8 步搜索（{len(state_results)} 个状态）===")
    print(f"Spearman(one_step, full)     : 均值 {statistics.mean(rhos):+.3f} "
          f"中位 {statistics.median(rhos):+.3f} "
          f"p10 {sorted(rhos)[int(0.1 * len(rhos))]:+.3f} max {max(rhos):+.3f}")
    print(f"决策一致率（含 +80 先验）    : "
          f"{sum(r['dec_agree'] for r in state_results)}/{len(state_results)} "
          f"= {100.0 * sum(r['dec_agree'] for r in state_results) / len(state_results):.1f}%")
    print(f"纯评估 argmax 一致率         : "
          f"{100.0 * sum(r['arg_agree'] for r in state_results) / len(state_results):.1f}%")
    print(f"top5 排名重叠率              : "
          f"均值 {statistics.mean([r['top5_overlap'] for r in state_results]):.2f}")
    print(f"\n计划间离散度: full 均值 {statistics.mean([r['full_spread'] for r in state_results]):.0f} "
          f"/ 1step 均值 {statistics.mean([r['one_spread'] for r in state_results]):.0f}")
    print(f"单次 rollout 的 σ（中位）: full {statistics.median([r['full_median_std'] for r in state_results]):.0f} "
          f"/ 1step {statistics.median([r['one_median_std'] for r in state_results]):.0f}")
    print(f"\n总耗时 {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
