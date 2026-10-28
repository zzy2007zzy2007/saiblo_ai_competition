#!/usr/bin/env python
"""**混杂对照**：A 臂比 B 臂多一个 oracle 进程 ⇒ CPU 竞争更重。亚军是**部分墙钟预算**的
（`lure_strategy_v4_evaluation.hpp:378-392` 的 `action_target_time_ms` / `over_budget()`），
所以必须直接核验"**对手在两臂里的搜索量是否相同**"，否则 A−B 里会混进"对手被拖慢"。

读对手自己的对局 stderr（每回合一条 JSON：
`{"kind":"decision","round":N,"total_rollouts":M,...,"elapsed_us":U}`），
按回合分桶比较两臂的 `total_rollouts` 与 `elapsed_us`：
  * `total_rollouts` 相同（同桶中位数差 < 2%）⇒ 对手的**搜索量**没变 ⇒ 无混杂；
  * A 臂 `total_rollouts` 明显更低而 `elapsed_us` 更高 ⇒ **对手被拖慢了** ⇒ A−B 被高估，必须写进报告。

用法：
  python code/test_match/compare_opp_budget.py \
      --arm A=match_results/inject_vs_ru/A/stderr \
      --arm B=match_results/inject_vs_ru/B/stderr
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 384), (384, 513)]


def read_opp(path_dir: Path, seeds: set[int] | None = None) -> dict:
    rows = []
    for f in sorted(path_dir.glob("ai?_runnerup_seed*.stderr.log")):
        m = re.search(r"seed(\d+)\.stderr", f.name)
        if not m:
            continue
        seed = int(m.group(1))
        if seeds is not None and seed not in seeds:
            continue
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"kind":"decision"' not in ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:                                                 # noqa: BLE001
                continue
            rows.append({"seed": seed, "round": int(d.get("round", -1)),
                         "rollouts": int(d.get("total_rollouts", 0)),
                         "us": float(d.get("elapsed_us", 0) or 0),
                         "plans": int(d.get("plans_total", 0)),
                         "acts": int(d.get("action_count", 0) or 0)})
    return {"dir": str(path_dir), "rows": rows}


def bucket_of(r: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= r < hi:
            return f"{lo}-{hi-1}"
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=STDERRDIR")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12, 13, 14],
                    help="只统计这些地图 seed（默认 = 本实验那 8 张；避免把 match_results/ 里\n"
                         "历史别的 seed 的 stderr 混进来）")
    args = ap.parse_args()
    if len(args.arm) < 1:
        print("need --arm LABEL=DIR", file=sys.stderr)
        return 2
    seeds = set(args.seeds)
    arms = {}
    for spec in args.arm:
        lab, d = spec.split("=", 1)
        arms[lab] = read_opp(Path(d), seeds)
    labs = list(arms)

    print("== 每人一行的对手读数（按回合桶；中位数）==")
    hdr = f"{'bucket':<10}" + "".join(f"{l+' rollouts':>18}{l+' ms':>10}" for l in labs)
    print(hdr)
    stats: dict = {}
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi-1}"
        line = f"{b:<10}"
        stats[b] = {}
        for l in labs:
            sel = [r for r in arms[l]["rows"] if lo <= r["round"] < hi]
            if sel:
                rl = statistics.median([r["rollouts"] for r in sel])
                us = statistics.median([r["us"] for r in sel]) / 1000.0
                line += f"{rl:>18.0f}{us:>10.1f}"
                stats[b][l] = {"n": len(sel), "rollouts": rl, "ms": us}
            else:
                line += f"{'-':>18}{'-':>10}"
                stats[b][l] = {"n": 0}
        print(line)

    if len(labs) >= 2:
        a, b_ = labs[0], labs[1]
        print(f"\n== 差值（{a} − {b_}）==")
        print("   ⚠️ 判据：只有同时满足「rollouts 更低」**且**「每-rollout 耗时更高」才是"
              "'被负载拖慢'；\n      只差 rollouts 而每-rollout 耗时相近 ⇒ 是**局面差异**"
              "（rollout 数由行动臂数/局面决定）。")
        dif_hit = 0
        for b in stats:
            sa, sb = stats[b].get(a, {}), stats[b].get(b_, {})
            if sa.get("n") and sb.get("n"):
                dr = 100.0 * (sa["rollouts"] - sb["rollouts"]) / max(1.0, sb["rollouts"])
                dt = sa["ms"] - sb["ms"]
                ra = sa["ms"] * 1000.0 / max(1.0, sa["rollouts"])
                rb = sb["ms"] * 1000.0 / max(1.0, sb["rollouts"])
                drt = 100.0 * (ra - rb) / max(1e-9, rb)
                load = (dr <= -2.0) and (drt >= 3.0)
                print(f"  {b:<10} rollouts {dr:+6.1f}%   elapsed {dt:+7.1f} ms   "
                      f"每-rollout {ra:.1f} vs {rb:.1f} µs ({drt:+.1f}%)"
                      f"{'  ⚠️ 更像被负载拖慢' if load else ''}")
                dif_hit += 1 if load else 0
        print("\n判定：" + (
            f"⚠️ 有 {dif_hit} 个桶同时满足「搜索量更低 + 每-rollout 更慢」⇒ A−B 需要打折"
            if dif_hit else
            "✓ 没有桶同时满足「搜索量更低 + 每-rollout 更慢」⇒ 无「对手被拖慢」的混杂"))
        print("（亚军的硬上限是 6 s / 目标 4.5 s，见 lure_strategy_v4_params.hpp:7-13；"
              "只要 elapsed 没逼近它，墙钟成分就不参与决定搜索量）")

    # 出手率也应一致（若对手行为不同，胜负差异可能来自它）
    print("\n== 对手出手率（action_count>0 的决策占比）==")
    for l in labs:
        rows = arms[l]["rows"]
        if rows:
            print(f"  {l}: {sum(1 for r in rows if r['acts'] > 0)}/{len(rows)} "
                  f"= {100.0*sum(1 for r in rows if r['acts']>0)/len(rows):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
